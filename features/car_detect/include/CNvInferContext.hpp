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

#ifndef CNVINFER_CONTEXT_HPP
#define CNVINFER_CONTEXT_HPP

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <unordered_map>
#include <memory>
#include <vector>

#include "CarCommon.hpp"
#include "CUtils.hpp"
#include "NvInfer.h"
#include "NvOnnxParser.h"

class NvInferMulticastLogger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        /* suppress info-level messages */
        if (severity < Severity::kINFO)
            std::cerr << "Multicast Info TensorRT: " << msg << std::endl;
    }
};

static class NvInferMulticastLogger g_Logger;

struct InferDeleter
{
    template <typename T> void operator()(T *obj) const { delete obj; }
};

// template <typename T>
// using SampleUniquePtr = std::unique_ptr<T, InferDeleter>;

static auto StreamDeleter = [](cudaStream_t *pStream) {
    if (pStream) {
        cudaStreamDestroy(*pStream);
        delete pStream;
    }
};

inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream()
{
    std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
    if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess) {
        pStream.reset(nullptr);
    }

    return pStream;
}

class CNvInferContext
{
  public:
    CNvInferContext(uint32_t id, const NvInferInitParams &InitParams);
    ~CNvInferContext() = default;

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer(cudaStream_t m_stream);
    bool GetTensorSizeAndDim(const std::string &tensorName, uint64_t &tensorSize, nvinfer1::Dims &d);
    bool BufferRegister(void *in_buf, void *bbox_buf, void *coverage_buf);

  private:
    NvInferInitParams mInitParams;

    int mnbBindings;              //in+out node number
    std::vector<void *> mBinding; //in/out node data tensor

    std::shared_ptr<nvinfer1::IRuntime> mRuntime;          //!< The TensorRT runtime used to deserialize the engine
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;        //!< The TensorRT engine used to run the network
    std::shared_ptr<nvinfer1::IExecutionContext> mContext; //!< The TensorRT context used to run the network
};

#endif
