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

#ifndef CCARDETECTOR_HPP
#define CCARDETECTOR_HPP

#include "CNvInferTask.hpp"
#include "CUtils.hpp"
#include "CarCommon.hpp"
#include <atomic>
#include <memory>

class CCarDetector
{
  public:
    // define return value of Process
    enum class DetectResult : uint8_t
    {
        CAR_DETECT_NOT_INITIALIZED,
        CAR_DETECT_EXEC_ERROR,
        CAR_DETECT_SUCCESS
    };

    CCarDetector(bool bUsePva = false);
    virtual ~CCarDetector(void);
    bool Init(uint32_t id, cudaStream_t stream);
    void DeInit();
    DetectResult Process(const cudaArray_t *inputBuf,
                         uint32_t inputImageWidth,
                         uint32_t inputImageHeight,
                         std::vector<NvInferObject> &vObjs,
                         bool bDraw);

  private:
    NvInferInitParams m_initParams;
    std::unique_ptr<CNvInferTask> m_upNvInferTask;
    cudaStream_t m_stream;
    uint32_t m_id;
    bool m_init_success;
    bool m_bUsePva;
};
#endif
