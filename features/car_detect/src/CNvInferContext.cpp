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

#include "CNvInferContext.hpp"
#include "NvInfer.h"
#include <memory>
#include <algorithm>
#include <vector>
#include <numeric>
#include "CUtils.hpp"

CNvInferContext::CNvInferContext(uint32_t id, const NvInferInitParams &InitParams)
    : mInitParams(InitParams)
    , mnbBindings(0)
    , mBinding()
    , mRuntime(nullptr)
    , mEngine(nullptr)
    , mContext(nullptr)
{
}

bool CNvInferContext::build()
{
    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_Logger));
    if (!mRuntime) {
        return false;
    }

    // 0. getting the engine file
    std::ifstream engineFile(mInitParams.fp16ModelCacheFilePath.c_str(), std::ios::binary);
    if (!engineFile) {
        LOG_ERR("[NvInfer]Error opening engine file %s\n", mInitParams.fp16ModelCacheFilePath.c_str());
        return false;
    }
    engineFile.seekg(0, std::ifstream::end);
    int64_t fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);
    std::vector<char> engineData(fsize);
    if (!engineFile.read(engineData.data(), fsize)) {
        LOG_ERR("[NvInfer]Error loading engine file: %s\n", mInitParams.fp16ModelCacheFilePath.c_str());
        return false;
    }

    // 2. deser from the model memory, and get a ICudaEngine
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(engineData.data(), engineData.size()), InferDeleter());
    if (!mEngine) {
        return false;
    }

    mContext = std::shared_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!mContext) {
        return false;
    }

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool CNvInferContext::infer(cudaStream_t m_stream)
{
#if NV_BUILD_DOS7
    bool status = mContext->enqueueV3(m_stream);
#else
    bool status = mContext->enqueueV2(&mBinding[0], m_stream, nullptr);
#endif
    return status;
}

inline int64_t volume(const nvinfer1::Dims &d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t) {
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT8:
            return 1;
        default:
            LOG_ERR("getElementSize Unknown Data Type!\n");
            return 0;
    }
}

bool CNvInferContext::GetTensorSizeAndDim(const std::string &tensorName, uint64_t &tensorSize, nvinfer1::Dims &d)
{
    LOG_WARN("GetTensorSizeAndDim %s!\n", tensorName.c_str());
#if NV_BUILD_DOS7
    mnbBindings = mEngine->getNbIOTensors();
    for (int i = 0; i < mnbBindings; i++) {
        if (tensorName != mEngine->getIOTensorName(i)) {
            continue;
        } else {
            nvinfer1::DataType dtype = mEngine->getTensorDataType(tensorName.c_str());
            d = mEngine->getTensorShape(tensorName.c_str());

            tensorSize = volume(d) * getElementSize(dtype);

            auto const &mode = mEngine->getTensorIOMode(tensorName.c_str());
            if (mode == nvinfer1::TensorIOMode::kINPUT && dtype == nvinfer1::DataType::kFLOAT) {
                mContext->setInputShape(tensorName.c_str(), d);
            }
        }
    }
#else
    mnbBindings = mEngine->getNbBindings();
    int32_t index = mEngine->getBindingIndex(tensorName.c_str());
    if (index < 0) {
        LOG_ERR("getBindingIndex Error! %d\n", index);
        return false;
    } else {
        LOG_INFO("%s index is %d\n", tensorName, index);
    }
    d = mEngine->getBindingDimensions(index);

    nvinfer1::DataType dtype = mEngine->getBindingDataType(index); // in/out node dataType
    tensorSize = volume(d) * getElementSize(dtype);                // in/out node size
#endif

    return (tensorSize != 0);
}

bool CNvInferContext::BufferRegister(void *in_buf, void *bbox_buf, void *coverage_buf)
{
    mBinding.clear();
    mBinding.resize(3);
    mBinding[0] = in_buf;
    mBinding[1] = bbox_buf;
    mBinding[2] = coverage_buf;

#if NV_BUILD_DOS7
    mContext->setTensorAddress(mInitParams.inputImageLayerName.c_str(), (void *)mBinding[0]);
    mContext->setTensorAddress(mInitParams.outputBboxLayerName.c_str(), (void *)mBinding[1]);
    mContext->setTensorAddress(mInitParams.outputCoverageLayerName.c_str(), (void *)mBinding[2]);
#endif

    LOG_INFO("ALL GPU MEMORY REGISTERED SUCCESSFULLY \n");
    return true;
}