/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef CPVA_UTIL_H
#define CPVA_UTIL_H

#include <dlfcn.h>
#include <string>
#include <dlfcn.h>
#include <cuda_runtime.h>

#include "CUtils.hpp"
#include "pva_pre.h"

class CPvaUtil
{
  public:
    CPvaUtil(unsigned uIpWidth, unsigned uIpHeight, unsigned uOpWidth, unsigned uOpHeight, float fNetworkScaleFactor);

    bool Initialize(cudaStream_t stream, void *dpTsr);
    int Launch(const cudaArray_t *inputImageBuffer, void *dpTsr, cudaStream_t stream, int count);
    void Deinitialize();

  private:
    const std::string m_sPvaPath = "./";
    uint32_t m_uOpRsWidth, m_uOpRsHeight;
    unsigned m_uIpWidth, m_uIpHeight, m_uOpWidth, m_uOpHeight;
    float m_fNetworkScaleFactor;
    PvaPreBase *m_pva = nullptr;
    create_t *m_pCreateFunc;
    destroy_t *m_pDestroyFunc;
};

#endif // CPVA_UTIL_H
