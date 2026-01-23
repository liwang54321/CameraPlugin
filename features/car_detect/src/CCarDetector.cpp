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

#include "CCarDetector.hpp"
#include "CNvInferTask.hpp"
#include "Common.hpp"
#include "CUtils.hpp"
#include "cuda_kernels.h"
#include <iostream>

CCarDetector::CCarDetector(bool bUsePva)
    : m_initParams()
    , m_upNvInferTask(nullptr)
    , m_stream(nullptr)
    , m_id(0)
    , m_init_success(false)
    , m_bUsePva(bUsePva)
{
}

CCarDetector::~CCarDetector(void) {}

void CCarDetector::DeInit()
{
    if (m_upNvInferTask != nullptr) {
        m_upNvInferTask->Destroy();
        m_upNvInferTask.reset(nullptr);
    }
}

bool CCarDetector::Init(uint32_t id, cudaStream_t stream)
{
    m_id = id;
    int cudaDeviceId = 0;
    int numOfGPUs = 0;
    checkCudaErrors(cudaGetDeviceCount(&numOfGPUs));
    LOG_INFO("%d GPUs found\n", numOfGPUs);
    if (!numOfGPUs) {
        LOG_ERR("No GPUs found!!\n");
        return false;
    }

    int major = 0, minor = 0;
    checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, cudaDeviceId));
    checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, cudaDeviceId));
    LOG_INFO("GPU Device %d: with compute capability %d.%d\n", cudaDeviceId, major, minor);

    LOG_INFO(">>> Use GPU Device %d\n", cudaDeviceId);
    checkCudaErrors(cudaSetDevice(cudaDeviceId));

    // Implicit create GPU context
    checkCudaErrors(cudaFree(0));
    // stream to run
    m_stream = stream;

    // the parameter of the model
    // User need to modify here
    m_initParams.networkMode = NvInferNetworkMode::FP16; // Or INT8
    // no support INT8 because need calibration file

    m_initParams.fp16ModelCacheFilePath = "./resnet10_fp16_gpu.bin";

    m_initParams.networkScaleFactor = 0.0039215697906911373;

    m_initParams.scoreThresh = 0.8;
    m_initParams.nmsThresh = 0.2;
    m_initParams.groupThresh = 1;
    m_initParams.groupIouThresh = 0.7;

    // onnx gpu
    m_initParams.inputImageLayerName = "data";
    m_initParams.outputBboxLayerName = "Layer7_bbox";
    m_initParams.outputCoverageLayerName = "Layer7_cov";

    std::vector<std::vector<std::string>> tmp_labels{ { "Car" }, { "Bicycle" }, { "Person" }, { "Roadsign" } };
    m_initParams.labels = tmp_labels;

    m_upNvInferTask.reset(new CNvInferTask(m_initParams, m_id, m_stream, m_bUsePva));
    if (m_upNvInferTask == nullptr) {
        LOG_WARN("NvInfer memory allocation failed\n");
        return false;
    } else {
        LOG_INFO("m_upNvInferTask reset!\n");
    }
    // check the nvinfer task init success
    m_init_success = m_upNvInferTask->Init();
    if (!m_init_success) {
        LOG_WARN("WARN: nvinfer task init failed! run without inference!\n");
        return false;
    } else {
        LOG_INFO("m_upNvInferTask m_init_success!\n");
    }

    return true;
}

CCarDetector::DetectResult CCarDetector::Process(const cudaArray_t *inputBuf,
                                                 uint32_t inputImageWidth,
                                                 uint32_t inputImageHeight,
                                                 std::vector<NvInferObject> &vObjs,
                                                 bool draw)
{
    if (m_init_success) {
        return m_upNvInferTask->Process(inputBuf, inputImageWidth, inputImageHeight, vObjs, draw)
                   ? DetectResult::CAR_DETECT_SUCCESS
                   : DetectResult::CAR_DETECT_EXEC_ERROR;
    } else {
        return DetectResult::CAR_DETECT_NOT_INITIALIZED;
    }
}
