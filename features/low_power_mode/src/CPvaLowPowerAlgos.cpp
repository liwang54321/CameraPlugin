/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "CPvaLowPowerAlgos.hpp"

#include <dlfcn.h>
#include "CUtils.hpp"

namespace {
constexpr char kPvaAlgosLib[] = "libpva_low_power.so";
} // namespace

int CPvaLowPowerAlgos::PvaLowPowerAlgosInit()
{
    std::lock_guard<std::mutex> lg(m_mtx);
    if (!m_bInited) {
        if (m_pLibHandle == nullptr) {
            m_pLibHandle = dlopen(kPvaAlgosLib, RTLD_LAZY);
            if (!m_pLibHandle) {
                LOG_ERR("CpvaLowPowerAlgos dlopen failed:%s\n", dlerror());
                return -1;
            }
        }
        if (m_pFuncCreateAlgo == nullptr) {
            m_pFuncCreateAlgo = (create_t *)dlsym(m_pLibHandle, "create");
            if (!m_pFuncCreateAlgo) {
                LOG_ERR("CpvaLowPowerAlgos dlsym error!, create:%s\n", dlerror());
                return -1;
            }
        }
        if (m_pFuncDestroyAlgo == nullptr) {
            m_pFuncDestroyAlgo = (destroy_t *)dlsym(m_pLibHandle, "destory");
            if (!m_pFuncDestroyAlgo) {
                LOG_ERR("CpvaLowPowerAlgos dlsym error!, destroy:%s\n", dlerror());
                return -1;
            }
        }
        if (m_pFuncCuPvaFillNvSciBufAttrList == nullptr) {
            m_pFuncCuPvaFillNvSciBufAttrList = (FuncCuPvaFillNvSciBufAttrList)dlsym(m_pLibHandle, "CuPvaFillNvSciBufAttrList");
            if (!m_pFuncCuPvaFillNvSciBufAttrList) {
                LOG_ERR("CpvaLowPowerAlgos dlsym error!, m_pFuncCuPvaFillNvSciBufAttrList:%s\n", dlerror());
                return -1;
            }
        }
        if (m_pFuncCuPvaNvFillNvSciSyncAttrList == nullptr) {
            m_pFuncCuPvaNvFillNvSciSyncAttrList = (FuncCuPvaFillNvSciSyncBufAttrList)dlsym(m_pLibHandle, "CuPvaNvFillNvSciSyncAttrList");
            if (!m_pFuncCuPvaNvFillNvSciSyncAttrList) {
                LOG_ERR("CpvaLowPowerAlgos dlsym error!, m_pFuncCuPvaFillNvSciSyncAttrList:%s\n", dlerror());
                return -1;
            }
        }
        m_bInited = true;
    }

    return 0;
}

void CPvaLowPowerAlgos::PvaLowPowerAlgosDeinit()
{
    int ret = dlclose(m_pLibHandle);
    if (ret < 0) {
        LOG_ERR("CpvaLowPowerAlgos dlclose failed:%s\n", dlerror());
    }
    m_pLibHandle = nullptr;
    m_pFuncCreateAlgo = nullptr;
    m_pFuncDestroyAlgo = nullptr;
    m_pFuncCuPvaFillNvSciBufAttrList = nullptr;
    m_pFuncCuPvaNvFillNvSciSyncAttrList = nullptr;
    m_bInited = false;
}

PvaAlgos* CPvaLowPowerAlgos::CreateAlgo(const std::string& type)
{
    return m_pFuncCreateAlgo(type.c_str());
}

int CPvaLowPowerAlgos::DestroyAlgo(PvaAlgos* pPvaAlgos)
{
    m_pFuncDestroyAlgo(pPvaAlgos);
    return 0;
}

int CPvaLowPowerAlgos::CuPvaFillNvSciBufAttrList(NvSciBufAttrList attrs)
{
    return m_pFuncCuPvaFillNvSciBufAttrList(attrs);
}

int CPvaLowPowerAlgos::CuPvaFillNvSciSyncAttrList(NvSciSyncAttrList* pAttrs, int syncType)
{
    return m_pFuncCuPvaNvFillNvSciSyncAttrList(pAttrs, syncType);
}