/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "CNvInferTask.hpp"
#include "cuda_kernels.h"
#include "cvt2d_kernels.h"
#include <chrono>

CNvInferTask::CNvInferTask(NvInferInitParams &init_params, uint32_t id, cudaStream_t stream, bool bUsePva)
    : m_init_params(init_params)
    , m_id(id)
    , m_stream(stream)
    , m_Labels(init_params.labels)
    , m_NetworkScaleFactor(init_params.networkScaleFactor)
    , m_NetworkMode(init_params.networkMode)
    , m_Fp16ModelCacheFilePath(init_params.fp16ModelCacheFilePath)
    , m_scoreThresh(init_params.scoreThresh)
    , m_nmsThresh(init_params.nmsThresh)
    , m_groupIouThresh(init_params.groupIouThresh)
    , m_groupThresh(init_params.groupThresh)
    , m_NetworkInputLayerName(init_params.inputImageLayerName)
    , m_OutputCoverageLayerName(init_params.outputCoverageLayerName)
    , m_OutputBboxLayerName(init_params.outputBboxLayerName)
    , m_bUsePva(bUsePva)
{
}

bool CNvInferTask::Init()
{
    if (m_OutputBboxLayerName.empty()) {
        LOG_ERR("Error: Output BBox layer name should be provided for detectors.\n");
        return false;
    }

    if (m_OutputCoverageLayerName.empty()) {
        LOG_ERR("Error: Output coverage layer name should be provided.\n");
        return false;
    }

    if (!Initialize()) {
        return false;
    }

    return true;
}

CNvInferTask::~CNvInferTask() {}

bool CNvInferTask::GetIOSizesAndDims()
{
    if (!m_NetworkInputLayerName.empty()) {
        if (!trt_ctx->GetTensorSizeAndDim(m_NetworkInputLayerName, m_NetworkInputTensorSize, m_NetworkInputLayerDim)) {
            LOG_ERR("Input layer not found. Please check input layer name in config.\n");
            return false;
        } else {
            LOG_INFO("GetIOSizesAndDims m_NetworkInputLayerName %s!\n", m_NetworkInputLayerName.c_str());
            LOG_INFO("GetIOSizesAndDims m_NetworkInputTensorSize %d!\n", m_NetworkInputTensorSize);
            LOG_INFO("GetIOSizesAndDims m_NetworkInputLayerDim ndim %d!\n", m_NetworkInputLayerDim.nbDims);
            LOG_INFO("GetIOSizesAndDims m_NetworkInputLayerDim n %d!\n", m_NetworkInputLayerDim.d[0]);
            LOG_INFO("GetIOSizesAndDims m_NetworkInputLayerDim c %d!\n", m_NetworkInputLayerDim.d[1]);
            LOG_INFO("GetIOSizesAndDims m_NetworkInputLayerDim h %d!\n", m_NetworkInputLayerDim.d[2]);
            LOG_INFO("GetIOSizesAndDims m_NetworkInputLayerDim w %d!\n", m_NetworkInputLayerDim.d[3]);
        }
        m_NetChannels = m_NetworkInputLayerDim.d[1];
        m_NetHeight = m_NetworkInputLayerDim.d[2];
        m_NetWidth = m_NetworkInputLayerDim.d[3];
        m_NetworkInputTensor = NULL;
    }

    if (!m_OutputBboxLayerName.empty()) {
        if (!trt_ctx->GetTensorSizeAndDim(m_OutputBboxLayerName, m_OutputBboxTensorSize, m_OutputBboxLayerDim)) {
            LOG_ERR("Output BBOX layer not found. Please check output BBOX layer name in config.");
            return false;
        } else {
            LOG_INFO("GetIOSizesAndDims m_OutputBboxLayerName %s!\n", m_OutputBboxLayerName.c_str());
            LOG_INFO("GetIOSizesAndDims m_OutputBboxTensorSize %d!\n", m_OutputBboxTensorSize);
            LOG_INFO("GetIOSizesAndDims m_OutputBboxLayerDim ndim %d!\n", m_OutputBboxLayerDim.nbDims);
            LOG_INFO("GetIOSizesAndDims m_OutputBboxLayerDim n %d!\n", m_OutputBboxLayerDim.d[0]);
            LOG_INFO("GetIOSizesAndDims m_OutputBboxLayerDim c %d!\n", m_OutputBboxLayerDim.d[1]);
            LOG_INFO("GetIOSizesAndDims m_OutputBboxLayerDim h %d!\n", m_OutputBboxLayerDim.d[2]);
            LOG_INFO("GetIOSizesAndDims m_OutputBboxLayerDim w %d!\n", m_OutputBboxLayerDim.d[3]);
        }
        m_OutputBboxTensor = NULL;
    }

    if (!m_OutputCoverageLayerName.empty()) {
        if (!trt_ctx->GetTensorSizeAndDim(m_OutputCoverageLayerName, m_OutputCoverageTensorSize,
                                          m_OutputCoverageLayerDim)) {
            LOG_ERR("Output Coverage layer not found. Please check output Coverage layer name in config.\n");
            return false;
        } else {
            LOG_INFO("GetIOSizesAndDims m_OutputCoverageLayerName %s!\n", m_OutputCoverageLayerName.c_str());
            LOG_INFO("GetIOSizesAndDims m_OutputCoverageTensorSize %d!\n", m_OutputCoverageTensorSize);
            LOG_INFO("GetIOSizesAndDims m_OutputCoverageLayerName ndim %d!\n", m_OutputCoverageLayerDim.nbDims);
            LOG_INFO("GetIOSizesAndDims m_OutputCoverageLayerName n %d!\n", m_OutputCoverageLayerDim.d[0]);
            LOG_INFO("GetIOSizesAndDims m_OutputCoverageLayerName c %d!\n", m_OutputCoverageLayerDim.d[1]);
            LOG_INFO("GetIOSizesAndDims m_OutputCoverageLayerName h %d!\n", m_OutputCoverageLayerDim.d[2]);
            LOG_INFO("GetIOSizesAndDims m_OutputCoverageLayerName w %d!\n", m_OutputCoverageLayerDim.d[3]);
        }
        m_OutputCoverageTensor = NULL;
    }

    return true;
}

bool CNvInferTask::Initialize()
{
    trt_ctx = new CNvInferContext(m_id, m_init_params);
    if (!trt_ctx->build()) {
        LOG_ERR("CNvInferContext build error!\n");
        return false;
    }
    if (!GetIOSizesAndDims()) {
        LOG_ERR("Error: Failed to get buffer size and dimension information for I/O layers.\n");
        return false;
    }

    checkCudaErrors(cudaMallocManaged(&m_NetworkInputTensor, m_NetworkInputTensorSize));
    checkCudaErrors(cudaMallocManaged(&m_OutputBboxTensor, m_OutputBboxTensorSize));
    checkCudaErrors(cudaMallocManaged(&m_OutputCoverageTensor, m_OutputCoverageTensorSize));
    checkCudaErrors(cudaStreamAttachMemAsync(m_stream, m_NetworkInputTensor));
    checkCudaErrors(cudaStreamAttachMemAsync(m_stream, m_OutputBboxTensor));
    checkCudaErrors(cudaStreamAttachMemAsync(m_stream, m_OutputCoverageTensor));

    trt_ctx->BufferRegister(m_NetworkInputTensor, m_OutputBboxTensor, m_OutputCoverageTensor);

    if (m_bUsePva) {
        //In order to keep consistent with the requirements of the map file, PVA requires the input resolution to be 3840x2160.
        unsigned uIpWidth = 3840U;
        unsigned uIpHeight = 2160U;

        m_pPvaUtil = nullptr;
        m_pPvaUtil = new CPvaUtil(uIpWidth, uIpHeight, m_NetWidth, m_NetHeight, m_NetworkScaleFactor);
        if (!m_pPvaUtil->Initialize(m_stream, m_NetworkInputTensor)) {
            LOG_ERR("Failed to initialize m_pPvaUtil\n");
            delete m_pPvaUtil;
            m_pPvaUtil = nullptr;
            return false;
        }
    }

    return true;
}

bool CNvInferTask::Process(const cudaArray_t *inputImageBuffer,
                           int inputImageWidth,
                           int inputImageHeight,
                           std::vector<NvInferObject> &vObjs,
                           bool bDraw)
{
    double scale_ratio_x, scale_ratio_y;
    scale_ratio_x = 1.0 * m_NetWidth / inputImageWidth;
    scale_ratio_y = 1.0 * m_NetHeight / inputImageHeight;

    cudaEvent_t start, pre_event, infer_event, end;
    checkCudaErrors(cudaEventCreate(&end));
    checkCudaErrors(cudaEventCreate(&pre_event));
    checkCudaErrors(cudaEventCreate(&infer_event));
    checkCudaErrors(cudaEventCreate(&start));

    checkCudaErrors(cudaStreamSynchronize(m_stream));
    auto starttime = std::chrono::system_clock::now();
    checkCudaErrors(cudaEventRecord(start, m_stream));
    std::chrono::duration<double> diff_pva;
    std::chrono::duration<double> diff_nopva;

    if (m_bUsePva) {
        if (inputImageWidth != 3840 || inputImageHeight != 2160) {
            LOG_ERR("Error: Pva preprocess could only support 3840x2160 size, if need more size support,\
                        please refer NvSIPL Multi-cast Reference Application Guide.\n");
            return false;
        }
    }

    switch (m_NetworkMode) {
        case NvInferNetworkMode::FP16:
            if (m_bUsePva) {
                int res = m_pPvaUtil->Launch(inputImageBuffer, m_NetworkInputTensor, m_stream, m_ProcessCount);
                m_ProcessCount++;
                diff_pva = std::chrono::system_clock::now() - starttime;
                LOG_INFO("pva preprocess time cost %f ms", diff_pva.count() * 1000);
                if (res != 0) {
                    LOG_ERR("Error: Pva launch error.\n");
                    return false;
                }
            } else {
                CvtNv12blToRgbPlanar(inputImageBuffer, inputImageWidth, inputImageHeight, m_NetworkInputTensor, m_NetWidth,
                                     m_NetHeight, 1, m_NetworkScaleFactor, m_stream);
                diff_nopva = std::chrono::system_clock::now() - starttime;
                LOG_INFO("cuda preprocess time cost %f ms", diff_nopva.count() * 1000);
            }
            break;
        default:
            LOG_ERR("Error: Unsupported network input Mode.\n");
            return false;
    }
    checkCudaErrors(cudaEventRecord(pre_event, m_stream));
    if (!trt_ctx->infer(m_stream)) {
        LOG_ERR("Error: Trt infer error.\n");
        return false;
    }
    checkCudaErrors(cudaEventRecord(infer_event, m_stream));
    checkCudaErrors(cudaStreamSynchronize(m_stream));
    // Post-process detection boxes and types on cpu
    if (!PostProcess(scale_ratio_x, scale_ratio_y, vObjs)) {
        LOG_ERR("PostProcess Error\n");
        return false;
    }

    if (bDraw) {
        if (!nv12blDrawRect(inputImageBuffer, inputImageWidth, inputImageWidth, inputImageHeight, vObjs, m_stream)) {
            LOG_ERR("nv12blDrawRect Error\n");
            return false;
        }
    }

    checkCudaErrors(cudaEventRecord(end, m_stream));
    checkCudaErrors(cudaEventSynchronize(end));
    float gpu_time;
    float cuda_preprocess_time;
    float infer_time;
    float draw_time;
    checkCudaErrors(cudaEventElapsedTime(&cuda_preprocess_time, start, pre_event));
    checkCudaErrors(cudaEventElapsedTime(&infer_time, pre_event, infer_event));
    checkCudaErrors(cudaEventElapsedTime(&draw_time, infer_event, end));
    checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, end));
    checkCudaErrors(cudaStreamSynchronize(m_stream));
    std::chrono::duration<double> diff = std::chrono::system_clock::now() - starttime;
    LOG_INFO("Cuda preprocess cost %f ms, Inference cost %f ms, Draw cost %f ms, All GPU cost %f ms, CPU time cost %f ms",
             cuda_preprocess_time, infer_time, draw_time, gpu_time, diff.count() * 1000);

    checkCudaErrors(cudaEventDestroy(end));
    checkCudaErrors(cudaEventDestroy(pre_event));
    checkCudaErrors(cudaEventDestroy(infer_event));
    checkCudaErrors(cudaEventDestroy(start));

    return true;
}

bool CNvInferTask::NmsCpu(std::vector<Bndbox> bndboxes, std::vector<Bndbox> &nms_pred)
{
    std::sort(bndboxes.begin(), bndboxes.end(),
              [](Bndbox boxes1, Bndbox boxes2) { return boxes1.score > boxes2.score; });
    std::vector<int> suppressed(bndboxes.size(), 0);
    for (size_t i = 0; i < bndboxes.size(); i++) {
        if (suppressed[i] == 1) {
            continue;
        }
        unsigned int times = 0;
        for (size_t j = i + 1; j < bndboxes.size(); j++) {
            if (suppressed[j] == 1) {
                continue;
            }

            int sa = (bndboxes[i].xMax - bndboxes[i].xMin) * (bndboxes[i].yMax - bndboxes[i].yMin);
            int sb = (bndboxes[j].xMax - bndboxes[j].xMin) * (bndboxes[j].yMax - bndboxes[j].yMin);

            int xMin_inter = std::max(bndboxes[i].xMin, bndboxes[j].xMin);
            int yMin_inter = std::max(bndboxes[i].yMin, bndboxes[j].yMin);
            int xMax_inter = std::min(bndboxes[i].xMax, bndboxes[j].xMax);
            int yMax_inter = std::min(bndboxes[i].yMax, bndboxes[j].yMax);

            int s_overlap = 0;
            if (xMin_inter < xMax_inter && yMin_inter < yMax_inter)
                s_overlap = (xMax_inter - xMin_inter) * (yMax_inter - yMin_inter);

            float iou = float(s_overlap) / std::max(float(sa + sb - s_overlap), ThresHold);

            if (s_overlap > 0 && iou >= m_groupIouThresh) {
                times++;
            }

            if (iou >= m_nmsThresh) {
                suppressed[j] = 1;
            }
        }
        if (times >= m_groupThresh) {
            nms_pred.emplace_back(bndboxes[i]);
        }
    }
    return true;
}

bool CNvInferTask::PostProcess(double scale_ratio_x, double scale_ratio_y, std::vector<NvInferObject> &objs)
{
    for (size_t c = 0; c < (size_t)m_OutputCoverageLayerDim.d[1]; c++) {
        std::vector<Bndbox> rect_list, nms_pred;
        switch (m_NetworkMode) {
            case NvInferNetworkMode::FP16:
                ParseBoundingBox((float *)m_OutputBboxTensor, (float *)m_OutputCoverageTensor, rect_list, c);
                break;
            default:
                LOG_ERR("parseBoundingBox Error: Unsupported network mode.\n");
                return false;
        }

        NmsCpu(rect_list, nms_pred);
        rect_list.clear();
        for (auto &rect : nms_pred) {
            NvInferObject object;
            object.left = rect.xMin;
            object.top = rect.yMin;
            object.width = rect.xMax - rect.xMin;
            object.height = rect.yMax - rect.yMin;
            object.classIndex = c;
            if (c < m_Labels.size() && m_Labels[c].size() > 0)
                object.label = m_Labels[c][0];
            objs.push_back(object);
        }
    }

    LOG_INFO("[NvInfer %u  Detection] - %u objects were detected \n", m_id, objs.size());

    for (unsigned int i = 0; i < objs.size(); i++) {
        NvInferObject &obj = objs[i];

        /* Scale the bounding boxes proportionally based on how the object/frame was
         * scaled during input. */
        obj.left /= scale_ratio_x;
        obj.top /= scale_ratio_y;
        obj.width /= scale_ratio_x;
        obj.height /= scale_ratio_y;

        LOG_WARN("[ %u %u %u %u ] object[%u]: %s \n", (unsigned)obj.left, (unsigned)obj.top, (unsigned)obj.width,
                 (unsigned)obj.height, i, obj.label.c_str());
    }
    return true;
}

bool CNvInferTask::Destroy()
{
    if (trt_ctx != NULL) {
        delete trt_ctx;
    }

    if (m_bUsePva) {
        if (m_pPvaUtil) {
            m_pPvaUtil->Deinitialize();
            delete m_pPvaUtil;
            m_pPvaUtil = nullptr;
        }
    }

    checkCudaErrors(cudaFree(m_NetworkInputTensor));
    checkCudaErrors(cudaFree(m_OutputBboxTensor));
    checkCudaErrors(cudaFree(m_OutputCoverageTensor));
    return true;
}

void CNvInferTask::ParseBoundingBox(const float *outputBboxBuffer,
                                    const float *outputCoverageBuffer,
                                    std::vector<Bndbox> &rectList,
                                    unsigned int classIndex)
{
    int gridC = m_OutputCoverageLayerDim.d[1];
    int gridH = m_OutputCoverageLayerDim.d[2];
    int gridW = m_OutputCoverageLayerDim.d[3];
    int gridSize = gridW * gridH;
    int gridOffset = gridC * gridSize * 0;

    int targetShape[2] = { gridW, gridH };
    float bboxNorm[2] = { 35.0, 35.0 };
    float gcCenters0[targetShape[0]];
    float gcCenters1[targetShape[1]];

    for (int i = 0; i < targetShape[0]; i++) {
        gcCenters0[i] = (float)(i * 16 + 0.5);
        gcCenters0[i] /= (float)bboxNorm[0];
    }
    for (int i = 0; i < targetShape[1]; i++) {
        gcCenters1[i] = (float)(i * 16 + 0.5);
        gcCenters1[i] /= (float)bboxNorm[1];
    }

    /* Pointers to memory regions containing the (x1,y1) and (x2,y2) coordinates
     * of rectangles in the output bounding box layer. */
    const float *outputX1 = outputBboxBuffer +
                            classIndex * sizeof(float) * m_OutputBboxLayerDim.d[2] * m_OutputBboxLayerDim.d[3] +
                            m_OutputBboxLayerDim.d[1] * m_OutputBboxLayerDim.d[2] * m_OutputBboxLayerDim.d[3] * 0;

    const float *outputY1 = outputX1 + gridSize;
    const float *outputX2 = outputY1 + gridSize;
    const float *outputY2 = outputX2 + gridSize;

    /* Iterate through each point in the grid and check if the rectangle at that
     * point meets the minimum threshold criteria. */
    for (int h = 0; h < gridH; h++) {
        for (int w = 0; w < gridW; w++) {
            int i = w + h * gridW;
            float score = outputCoverageBuffer[gridOffset + classIndex * gridSize + i];
            if (score < m_scoreThresh)
                continue;

            int rectX1, rectY1, rectX2, rectY2;
            float rectX1Float, rectY1Float, rectX2Float, rectY2Float;

            /* Centering and normalization of the rectangle. */
            rectX1Float = float(outputX1[w + h * gridW] - gcCenters0[w]);
            rectY1Float = float(outputY1[w + h * gridW] - gcCenters1[h]);
            rectX2Float = float(outputX2[w + h * gridW] + gcCenters0[w]);
            rectY2Float = float(outputY2[w + h * gridW] + gcCenters1[h]);

            rectX1Float *= (float)(-bboxNorm[0]);
            rectY1Float *= (float)(-bboxNorm[1]);
            rectX2Float *= (float)(bboxNorm[0]);
            rectY2Float *= (float)(bboxNorm[1]);

            rectX1 = (int)rectX1Float;
            rectY1 = (int)rectY1Float;
            rectX2 = (int)rectX2Float;
            rectY2 = (int)rectY2Float;
            /* Clip parsed rectangles to frame bounds. */
            if (rectX1 >= (int)m_NetWidth)
                rectX1 = m_NetWidth - 1;
            if (rectX2 >= (int)m_NetWidth)
                rectX2 = m_NetWidth - 1;
            if (rectY1 >= (int)m_NetHeight)
                rectY1 = m_NetHeight - 1;
            if (rectY2 >= (int)m_NetHeight)
                rectY2 = m_NetHeight - 1;

            if (rectX1 < 0)
                rectX1 = 0;
            if (rectX2 < 0)
                rectX2 = 0;
            if (rectY1 < 0)
                rectY1 = 0;
            if (rectY2 < 0)
                rectY2 = 0;

            rectList.push_back(Bndbox(rectX1, rectY1, rectX2, rectY2, score));
        }
    }
}
