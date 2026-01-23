/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef CNVINFER_TASK_HPP
#define CNVINFER_TASK_HPP

#include "CNvInferContext.hpp"
#include "CarCommon.hpp"
#include "CUtils.hpp"
#include "CPvaUtil.h"
#include <algorithm>
#include <vector>

class CNvInferTask
{
  public:
    CNvInferTask(NvInferInitParams &init_params, uint32_t id, cudaStream_t m_stream, bool bUsePva = false);
    ~CNvInferTask();
    bool Init();
    bool Process(const cudaArray_t *input_frame,
                 int inputImageWidth,
                 int inputImageHeight,
                 std::vector<NvInferObject> &vObjs,
                 bool bDraw);
    bool Destroy();

  private:
    bool Initialize();

    NvInferInitParams m_init_params;
    int m_ProcessCount = 0;
    CPvaUtil *m_pPvaUtil;
    uint32_t m_id;
    cudaStream_t m_stream;
    /** NvInferContext to be used for inferencing. */
    CNvInferContext *trt_ctx;

    /* Dimensions of network input. */
    unsigned int m_NetWidth;
    unsigned int m_NetHeight;
    unsigned int m_NetChannels;

    uint64_t m_NetworkInputTensorSize;
    void *m_NetworkInputTensor;
    nvinfer1::Dims m_NetworkInputLayerDim;

    uint64_t m_OutputBboxTensorSize;
    void *m_OutputBboxTensor;
    nvinfer1::Dims m_OutputBboxLayerDim;

    uint64_t m_OutputCoverageTensorSize;
    void *m_OutputCoverageTensor;
    nvinfer1::Dims m_OutputCoverageLayerDim;

    std::vector<std::vector<std::string>> m_Labels;

    const float ThresHold = 1e-8;

    float m_NetworkInputTensorScale;
    float m_OutputBboxTensorScale;
    float m_OutputCoverageTensorScale;

    float m_NetworkScaleFactor;

    NvInferNetworkMode m_NetworkMode;
    std::string m_Fp16ModelCacheFilePath;

    float m_scoreThresh;
    float m_nmsThresh;
    float m_groupIouThresh;
    unsigned int m_groupThresh;

    std::string m_NetworkInputLayerName;
    std::string m_OutputCoverageLayerName;
    std::string m_OutputBboxLayerName;

    bool m_bUsePva;

    std::string m_name{ "NvInferTask" };

    bool PostProcess(double scale_ratio_x, double scale_ratio_y, std::vector<NvInferObject> &objs);
    bool GetIOSizesAndDims();
    bool GetIODynamicRange();
    bool ReadPerTensorDynamicRangeValues();
    bool NmsCpu(std::vector<Bndbox> bndboxes, std::vector<Bndbox> &nms_pred);

    void ParseBoundingBox(const float *outputBboxBuffer,
                          const float *outputCoverageBuffer,
                          std::vector<Bndbox> &rectList,
                          unsigned int classIndex);
};

#endif // CNVINFER_TASK_HPP
