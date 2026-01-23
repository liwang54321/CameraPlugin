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
#ifndef CPVA_LOWPOWER_ALGOS_HPP
#define CPVA_LOWPOWER_ALGOS_HPP

#include <cmath>
#include <memory>
#include <mutex>

#include "nvscibuf.h"
#include "nvscisync.h"

class PvaAlgos {
public:
    PvaAlgos() {}
    virtual ~PvaAlgos() {}

    virtual int PvaAlgosInit(uint32_t uIpWidth, uint32_t uipHeight, uint32_t planePitches[], uint32_t uInputFormat,
                             uint32_t uOpWidth, uint32_t uOpHeight, const char *pAssetPath, uint32_t uVpuId) = 0;
    virtual int PvaAlgosPreprocess(const NvSciBufObj& nvsci_buf, float_t *pRgbBuf) = 0;
    virtual int PvaAlgosInference(const float_t *prRgbBuf) = 0;
    virtual int PvaAlgosPostprocess(int *pBoxNum, int *pPosX, int *pPosY, int *pPosW, int *pPosH) = 0;
    virtual int PvaAlgosDeinit() = 0;
};


class CPvaLowPowerAlgos {
public:
    using FuncCuPvaFillNvSciBufAttrList = int (*)(NvSciBufAttrList);
    using FuncCuPvaFillNvSciSyncBufAttrList = int (*)(NvSciSyncAttrList*, int);

    static CPvaLowPowerAlgos& GetInstance() {
        static CPvaLowPowerAlgos instance;
        return instance;
    }

    int PvaLowPowerAlgosInit();
    PvaAlgos* CreateAlgo(const std::string& type);
    int DestroyAlgo(PvaAlgos* pPvaAlgos);
    int CuPvaFillNvSciBufAttrList (NvSciBufAttrList attrs);
    int CuPvaFillNvSciSyncAttrList(NvSciSyncAttrList* pAttrs, int syncType);

private:
    // the types of the class factories
    typedef PvaAlgos *create_t(const char *);
    typedef void destroy_t(PvaAlgos *);
    std::mutex m_mtx;
    bool m_bInited{false};
    void *m_pLibHandle{ nullptr };
    create_t *m_pFuncCreateAlgo{ nullptr };
    destroy_t *m_pFuncDestroyAlgo{ nullptr };
    FuncCuPvaFillNvSciBufAttrList m_pFuncCuPvaFillNvSciBufAttrList = nullptr;
    FuncCuPvaFillNvSciSyncBufAttrList m_pFuncCuPvaNvFillNvSciSyncAttrList = nullptr;
    void PvaLowPowerAlgosDeinit();

    CPvaLowPowerAlgos() {}
    ~CPvaLowPowerAlgos() { PvaLowPowerAlgosDeinit(); }
    CPvaLowPowerAlgos(const CPvaLowPowerAlgos& other) = delete;
    CPvaLowPowerAlgos(const CPvaLowPowerAlgos&& other) = delete;
    CPvaLowPowerAlgos& operator=(const CPvaLowPowerAlgos& other) = delete;
};
#endif // CPVA_LOWPOWER_ALGOS_HPP