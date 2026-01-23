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

#include "CPvaUtil.h"

CPvaUtil::CPvaUtil(
    uint32_t uIpWidth, uint32_t uIpHeight, uint32_t uOpWidth, uint32_t uOpHeight, float fNetworkScaleFactor)
    : m_uOpRsWidth(720U)
    , m_uOpRsHeight(480U)
    , m_uIpWidth(uIpWidth)
    , m_uIpHeight(uIpHeight)
    , m_uOpWidth(uOpWidth)
    , m_uOpHeight(uOpHeight)
    , m_fNetworkScaleFactor(fNetworkScaleFactor)
{
}

bool CPvaUtil::Initialize(cudaStream_t stream, void *dpTsr)
{
    std::string sPvaLibPath = m_sPvaPath + "libpva_pre_lib.so";
    const char *error = nullptr;
    void *handle = dlopen(sPvaLibPath.c_str(), RTLD_LAZY);
    if (!handle) {
        LOG_ERR("libpva_pre_lib.so dlopen failed!\n");
        return false;
    }

    m_pCreateFunc = (create_t *)dlsym(handle, "create");
    if ((error = dlerror()) != nullptr) {
        LOG_ERR("dlsym error!, error code: %s.\n", dlerror());
        return false;
    }

    m_pDestroyFunc = (destroy_t *)dlsym(handle, "destory");
    if ((error = dlerror()) != nullptr) {
        LOG_ERR("dlsym error!, destory, error: %s.\n", dlerror());
        return false;
    }

    m_pva = m_pCreateFunc();
    if (!m_pva) {
        LOG_ERR("Create PVA algorithms failed!\n");
        return false;
    }
    int status = m_pva->pva_init(stream, m_uIpWidth, m_uIpHeight, m_uOpRsWidth, m_uOpRsHeight, m_uOpWidth, m_uOpHeight,
                                 m_sPvaPath.c_str(), dpTsr, m_fNetworkScaleFactor, 1, 0);
    if (status != 0) {
        LOG_ERR("PVA preprocess init failed, status: %d\n", status);
        return false;
    }

    return true;
}

int CPvaUtil::Launch(const cudaArray_t *inputImageBuffer, void *dpTsr, cudaStream_t stream, int count)
{
    return m_pva->pva_launch(inputImageBuffer, dpTsr, stream, count);
}

void CPvaUtil::Deinitialize()
{
    int statusDeallocate = m_pva->pva_deinit();
    if (statusDeallocate != 0) {
        LOG_ERR("PVA deinit failed\n");
    }
    m_pDestroyFunc(m_pva);
    m_pva = nullptr;
}