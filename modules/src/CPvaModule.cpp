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

#include <chrono>
#include <iostream>
#include <dlfcn.h>
#include <fstream>

#include "CPvaModule.hpp"
#include "nvscibuf.h"
#include "CElementDescription.hpp"
#include "CUtils.hpp"

namespace {
constexpr uint32_t kTargetFps = 5;
constexpr uint32_t kDlRunFrameNums = 10;
constexpr uint32_t kPVA_SIGNALER = 1;
constexpr uint32_t kPVA_WAITER = 2;
constexpr uint32_t kRGB_WIDTH = 320;
constexpr uint32_t kRGB_HEIGHT = 320;
constexpr uint32_t kRGB_CHANNEL = 3;
constexpr char kPvaAlgosEssentials[] = "./data/";
} // namespace

const std::unordered_map<std::string, Option> PvaOptionTable = {
    { "dl", { "run DL pipleline", offsetof(PvaOption, bRunDLPipeline), OptionType::BOOL } },
    { "vpuId", { "pva vpu id", offsetof(PvaOption, uVpuId), OptionType::UINT32 } },
    { "data", { "data for running", offsetof(PvaOption, sData), OptionType::STRING } }
};

CElementDescription pvaDescription{ "Pva", "Pva module to handle DL and CV pipeline",
                                    &CBaseModule::m_baseModuleOptionTable, &PvaOptionTable };

CPvaModule::CPvaModule(std::shared_ptr<CModuleCfg> spModuleCfg, IEventListener<CBaseModule> *pListener)
    : CBaseModule(spModuleCfg, pListener)
{
    spModuleCfg->m_cpuWaitCfg.bWaitPrefence = true;
}

CPvaModule::~CPvaModule()
{
    PLOG_DBG("release.\n");
}

NvError CPvaModule::Init()
{
    PLOG_DBG("Enter: CPvaModule::Init()\n");

    NvError err = NvError_Success;
    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler = std::make_unique<CProfiler>();
        err = m_upProfiler->Init(m_pAppCfg->m_sciSyncModule, m_pAppCfg->GetPerfDataSaveFolder(), GetName(), true,
                                 m_pAppCfg->GetMaxPerfSampleNum());
        PCHK_ERROR_AND_RETURN(err, "CProfiler::Init()");

        m_upProfiler->RecordInitBeginTime();
    }

    err = CBaseModule::Init();
    PCHK_ERROR_AND_RETURN(err, "CBaseModule::Init()");

    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler->RecordInitEndTime();
    }
    int status = CPvaLowPowerAlgos::GetInstance().PvaLowPowerAlgosInit();
    if (status != 0) {
        PLOG_ERR("PvaLowPowerAlgosInit failed, status = %d\n", status);
        return NvError_InvalidState;
    }

    return NvError_Success;
}

void CPvaModule::DeInit()
{
    CBaseModule::DeInit();
    PvaAlgosDeInit();
}

NvError
CPvaModule::FillDataBufAttrList(CClientCommon *pClient, PacketElementType userType, NvSciBufAttrList *pBufAttrList)
{
    PLOG_DBG("Enter FillDataBufAttrList()\n");

    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValAccessPerm accessPerm = NvSciBufAccessPerm_ReadWrite;
    bool cpuAccess = true;
    bool enableCpuCache = true;
    NvSciBufAttrKeyValuePair attrs[] = { { NvSciBufGeneralAttrKey_RequiredPerm, &accessPerm, sizeof(accessPerm) },
                                         { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
                                         { NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuAccess, sizeof(cpuAccess) },
                                         { NvSciBufGeneralAttrKey_EnableCpuCache, &enableCpuCache,
                                           sizeof(enableCpuCache) } };

    auto sciError = NvSciBufAttrListSetAttrs(*pBufAttrList, attrs, ARRAY_SIZE(attrs));
    PCHK_NVSCISTATUS_AND_RETURN(sciError, "NvSciBufAttrListSetAttrs");

    int status = CPvaLowPowerAlgos::GetInstance().CuPvaFillNvSciBufAttrList(*pBufAttrList);
    if (status != 0) {
        PLOG_ERR("m_pFuncCuPvaFillNvSciBufAttrList failed, status = %d\n", status);
        return NvError_BadParameter;
    }
    return NvError_Success;
}

NvError CPvaModule::FillSyncSignalerAttrList(CClientCommon *pClient,
                                                PacketElementType userType,
                                                NvSciSyncAttrList *pSignalerAttrList)
{
    PLOG_DBG("Enter FillSyncSignalerAttrList()\n");
    int syncType = kPVA_SIGNALER;
    int status = CPvaLowPowerAlgos::GetInstance().CuPvaFillNvSciSyncAttrList(pSignalerAttrList, syncType);
    if (status != 0) {
        PLOG_ERR("FillNvSciSyncAttrList (signaler) failed, status = %d\n", status);
        return NvError_BadParameter;
    }

    PLOG_DBG("Exit FillSyncSignalerAttrList()\n");

    return NvError_Success;
}

NvError CPvaModule::FillSyncWaiterAttrList(CClientCommon *pClient,
                                              PacketElementType userType,
                                              NvSciSyncAttrList *pWaiterAttrList)
{
    PLOG_DBG("Enter FillSyncWaiterAttrList()\n");
    int syncType = kPVA_WAITER;
    int status = CPvaLowPowerAlgos::GetInstance().CuPvaFillNvSciSyncAttrList(pWaiterAttrList, syncType);
    if (status != 0) {
        PLOG_ERR("FillNvSciSyncAttrList (waiter) failed, status = %d\n", status);
        return NvError_BadParameter;
    }

    PLOG_DBG("Exit FillSyncSignalerAttrList()\n");

    return NvError_Success;
}

NvError
CPvaModule::RegisterSignalSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj signalSyncObj)
{
    auto status = CBaseModule::RegisterSignalSyncObj(pClient, userType, signalSyncObj);
    PCHK_ERROR_AND_RETURN(status, "CBaseModule::RegisterSignalSyncObj");

    return NvError_Success;
}

NvError
CPvaModule::RegisterWaiterSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj waiterSyncObj)
{
    auto status = CBaseModule::RegisterWaiterSyncObj(pClient, userType, waiterSyncObj);
    PCHK_ERROR_AND_RETURN(status, "CBaseModule::RegisterWaiterSyncObj");

    return NvError_Success;
}

NvError CPvaModule::RegisterBufObj(CClientCommon *pClient,
                                      PacketElementType userType,
                                      uint32_t uPacketIndex,
                                      NvSciBufObj bufObj)
{
    auto status = CBaseModule::RegisterBufObj(pClient, userType, uPacketIndex, bufObj);
    PCHK_ERROR_AND_RETURN(status, "CBaseModule::RegisterBufObj");

    status = PopulateBufAttr(bufObj, m_bufAttrs[uPacketIndex]);
    PCHK_ERROR_AND_RETURN(status, "PopulateBufAttr");
    m_sciBufObjs[uPacketIndex] = bufObj;

    return NvError_Success;
}

NvError CPvaModule::OnDataBufAttrListRecvd(CClientCommon *pClient, NvSciBufAttrList bufAttrList)
{
    uint16_t uWidth = 0U;
    uint16_t uHeight = 0U;
    uint32_t planePitches[MAX_NUM_SURFACES]{0};
    NvError status = GetWidthAndHeight(bufAttrList, uWidth, uHeight, planePitches, sizeof(planePitches));
    PCHK_ERROR_AND_RETURN(status, "GetWidthAndHeight");
    if (m_pvaOption.sData.empty()) {
        status = PvaAlgosInit(kPvaAlgosEssentials, uWidth, uHeight, planePitches);
    } else {
        status = PvaAlgosInit(m_pvaOption.sData + "/", uWidth, uHeight, planePitches);
    }
    PCHK_ERROR_AND_RETURN(status, "PvaAlgoInit");

    return NvError_Success;
}

NvError CPvaModule::InsertPrefence(CClientCommon *pClient,
                                      PacketElementType userType,
                                      uint32_t uPacketIndex,
                                      NvSciSyncFence *pPrefence)
{
    return NvError_Success;
}

NvError CPvaModule::ProcessPayload(CClientCommon *pClient, uint32_t uPacketIndex)
{
    m_pMetaData = reinterpret_cast<MetaData *>(pClient->GetMetaPtr(uPacketIndex));
    m_pMetaData->bTriggerEncodingValid = true;
    m_pMetaData->bTriggerEncoding = false;

    // skip the frames
    {
        std::lock_guard<std::mutex> lk(m_FrameMutex);
        if (m_uFrameNum % (30 / kTargetFps) != 0) {
            return NvError_Success;
        }
    }

    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler->RecordExecutionBeginTime();
    }
    NvError status = PvaAlgosPreprocess(m_sciBufObjs[uPacketIndex], m_upRgbBuf.get());
    PCHK_ERROR_AND_RETURN(status, "PvaAlgosInference");
    if (m_pvaOption.bRunDLPipeline) {
        BBox result;
        NvError status = PvaAlgosInference(m_upRgbBuf.get(), result);
        PCHK_ERROR_AND_RETURN(status, "PvaAlgosInference");
        if (result.iNumSelectedDetections != 0) {
            PLOG_DBG("[CPvaModule][PVA] detect Num = %d\n", result.iNumSelectedDetections);
            for (int idx = 0; idx < result.iNumSelectedDetections; idx++) {
                PLOG_DBG("(%d, %d, %d, %d) ", result.iArrPosX[idx], result.iArrPosY[idx], result.iArrPosW[idx],
                        result.iArrPosH[idx]);
            }
            m_pMetaData->bTriggerEncoding = true;
        }
    }
    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler->RecordExecutionEndTime();
    }

    return NvError_Success;
}

NvError CPvaModule::OnProcessPayloadDone(CClientCommon *pClient, uint32_t uPacketIndex)
{
    return NvError_Success;
}

const OptionTable *CPvaModule::GetOptionTable() const
{
    return &PvaOptionTable;
}

const void *CPvaModule::GetOptionBaseAddress() const
{
    return &m_pvaOption;
}

NvError CPvaModule::PvaAlgosPreprocess(const NvSciBufObj& nvsciBuf, float_t *rgb_buf) {
    auto status = m_pPvaAlgos->PvaAlgosPreprocess(nvsciBuf, rgb_buf);
    if (status != 0) {
        PLOG_ERR("pva_algos_preprocess failed, error = %d\n", status);
        return NvError_InvalidState;
    }
    return NvError_Success;

}

NvError CPvaModule::PvaAlgosInference(const float_t *rgb_buf, BBox &result)
{
    auto status = m_pPvaAlgos->PvaAlgosInference(rgb_buf);
    if (status != 0) {
        PLOG_ERR("pva_algos_inference failed, error = %d\n", status);
        return NvError_InvalidState;
    }
    status = m_pPvaAlgos->PvaAlgosPostprocess(&result.iNumSelectedDetections, result.iArrPosX, result.iArrPosY,
                                              result.iArrPosW, result.iArrPosH);
    if (status != 0) {
        PLOG_ERR("pva_algos_postprocess failed, error = %d\n", status);
        return NvError_InvalidState;
    }
    return NvError_Success;
}

NvError CPvaModule::PvaAlgosInit(const std::string &assetPath, uint32_t width, uint32_t height, uint32_t *planePitches)
{
    m_pPvaAlgos = CPvaLowPowerAlgos::GetInstance().CreateAlgo("PVA");
    if (!m_pPvaAlgos) {
        PLOG_ERR("Create PVA algorithms error!\n");
        return NvError_InvalidState;
    }

    auto status = m_pPvaAlgos->PvaAlgosInit(width, height, planePitches, 1,
                                            kRGB_WIDTH, kRGB_HEIGHT, assetPath.c_str(), m_pvaOption.uVpuId);
    if (status != 0) {
        PLOG_ERR("PVA algo init failed, error = %d\n", status);
        return NvError_NotInitialized;
    }

    m_upRgbBuf = std::make_unique<float_t[]>(kRGB_WIDTH * kRGB_HEIGHT * kRGB_CHANNEL);

    return NvError_Success;
}

NvError CPvaModule::PvaAlgosDeInit()
{
    printf("pva algos deinit...\n");
    if (m_pPvaAlgos != nullptr) {
        m_pPvaAlgos->PvaAlgosDeinit();
        CPvaLowPowerAlgos::GetInstance().DestroyAlgo(m_pPvaAlgos);
        m_pPvaAlgos = nullptr;
    }
    return NvError_Success;
}
