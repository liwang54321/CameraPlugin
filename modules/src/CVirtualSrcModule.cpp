/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "CVirtualSrcModule.hpp"
#include "CElementDescription.hpp"

static constexpr uint64_t NANO_PER_SECOND = 1000000000U;
static constexpr uint64_t SCHEDULE_LATENCY_IN_NANOS = 50000U;

const std::unordered_map<std::string, Option> VirtualSrcOptionTable = {
    { "width",
      { "set the image buffer width (default 1920)", offsetof(VirtualSrcOption, uWidth), OptionType::UINT32 } },
    { "height",
      { "set the image buffer height (default 1080)", offsetof(VirtualSrcOption, uHeight), OptionType::UINT32 } },
    { "layout",
      { "set the image buffer layout (default BlockLinear).Valid options are b|bl|blocklinear|p|pl|pitchlinear",
        offsetof(VirtualSrcOption, sLayout), OptionType::STRING } },
    { "framerate",
      { "set the video framerate (default 30)", offsetof(VirtualSrcOption, fFramerate), OptionType::FLOAT } }
};

CElementDescription virtualSrcDescription{ "VirtualSrc",
                                           "Virtual source module for sending the buffer to the downstream.(Test Only)",
                                           &CBaseModule::m_baseModuleOptionTable, &VirtualSrcOptionTable };

CVirtualSrcModule::CVirtualSrcModule(std::shared_ptr<CModuleCfg> spModuleCfg, IEventListener<CBaseModule> *pListener)
    : CBaseModule(std::move(spModuleCfg), pListener)
{
}

CVirtualSrcModule::~CVirtualSrcModule()
{
    PLOG_DBG("release.\n");
}

NvError CVirtualSrcModule::Init()
{
    PLOG_DBG("Enter: CVirtualSrcModule::Init()\n");

    NvError error = NvError_Success;

    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler = std::make_unique<CProfiler>();
        error = m_upProfiler->Init(m_pAppCfg->m_sciSyncModule, m_pAppCfg->GetPerfDataSaveFolder(), GetName(), true,
                                   m_pAppCfg->GetMaxPerfSampleNum());
        PCHK_ERROR_AND_RETURN(error, "CProfiler::Init()");

        m_upProfiler->RecordInitBeginTime();
    }

    error = CBaseModule::Init();
    PCHK_ERROR_AND_RETURN(error, "CBaseModule::Init()");

    m_FrameTime = std::chrono::nanoseconds(static_cast<uint64_t>(NANO_PER_SECOND / m_virtualSrcOption.fFramerate));
    if (!m_virtualSrcOption.sLayout.empty()) {
        const char *pLayout = m_virtualSrcOption.sLayout.c_str();
        if (!strcasecmp(pLayout, "b") || !strcasecmp(pLayout, "bl") || !strcasecmp(pLayout, "blocklinear")) {
            m_LayoutType = NvSciBufImage_BlockLinearType;
        } else if (!strcasecmp(pLayout, "p") || !strcasecmp(pLayout, "pl") || !strcasecmp(pLayout, "pitchlinear")) {
            m_LayoutType = NvSciBufImage_PitchLinearType;
        } else {
            PLOG_ERR("Unknown layout type %s", pLayout);
            return NvError_BadValue;
        }
    }

    m_upEventHandler = std::make_unique<CEventHandler<CVirtualSrcModule>>();
    error = m_upEventHandler->RegisterHandler(&CVirtualSrcModule::Generator, this, false);
    PCHK_ERROR_AND_RETURN(error, "RegisterHandler");

    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler->RecordInitEndTime();
    }

    PLOG_DBG("Exit: CVirtualSrcModule::Init()\n");

    return NvError_Success;
}

void CVirtualSrcModule::DeInit()
{
    PLOG_DBG("Enter: CVirtualSrcModule::DeInit()\n");

    if (m_upEventHandler) {
        m_upEventHandler->QuitThread();
    }

    CBaseModule::DeInit();

    PLOG_DBG("Exit: CVirtualSrcModule::DeInit()\n");
}

NvError CVirtualSrcModule::Start()
{
    PLOG_DBG("Enter: CVirtualSrcModule::Start()\n");

    auto error = CBaseModule::Start();
    PCHK_ERROR_AND_RETURN(error, " Start");

    for (auto &bufObj : m_vBufObjs) {
        m_bufObjQueue.push_back(bufObj);
    }

    error = m_upEventHandler->StartThread();
    PCHK_ERROR_AND_RETURN(error, " StartThread");

    PLOG_DBG("Exit: CVirtualSrcModule::Start()\n");
    return NvError_Success;
}

NvError CVirtualSrcModule::Stop()
{
    PLOG_DBG("Enter: CVirtualSrcModule::Stop()\n");

    CBaseModule::Stop();
    if (m_upEventHandler) {
        m_upEventHandler->StopThread();
    }

    PLOG_DBG("Exit: CVirtualSrcModule::Stop()\n");
    return NvError_Success;
}

EventStatus CVirtualSrcModule::Generator()
{
    uint64_t uFrameCaptureTSC{ 0 };
    uint64_t uFrameCaptureStartTSC{ 0 };
    if (m_pAppCfg->IsProfilingEnabled()) {
        uFrameCaptureStartTSC = CProfiler::GetCurrentTSC();
    }
    std::this_thread::sleep_for(m_FrameTime - std::chrono::nanoseconds(m_uWorkTime));

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    if (m_bufObjQueue.empty()) {
        PLOG_WARN("BufObj queue empty, busy!\n");
        return EventStatus::ERROR;
    }

    auto bufObj = m_bufObjQueue.front();
    m_bufObjQueue.pop_front();

    UpdateFrameStatistics();

    if (m_pAppCfg->IsProfilingEnabled()) {
        uFrameCaptureTSC = CProfiler::GetCurrentTSC();
        if (!m_bHasDownstream) {
            auto error = m_upProfiler->RecordExecutionAndPipelineTime(uFrameCaptureStartTSC, uFrameCaptureTSC);
            if (error != NvError_Success) {
                PLOG_ERR("RecordExecutionAndPipelineTime failed: %d\n", error);
                return EventStatus::ERROR;
            }
        } else {
            auto error = m_upProfiler->RecordExecutionTime(uFrameCaptureStartTSC, uFrameCaptureTSC);
            if (error != NvError_Success) {
                PLOG_ERR("RecordExecutionTime failed: %d\n", error);
                return EventStatus::ERROR;
            }
        }

        MetaData *pMetaData = m_spProducer->GetMetaPtr(bufObj);
        if (pMetaData != nullptr) {
            pMetaData->Set(uFrameCaptureTSC, uFrameCaptureStartTSC, true, ++m_uFrameSequenceNumber);
        }
    }

    m_spProducer->Post(&bufObj, nullptr);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    m_uWorkTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() + SCHEDULE_LATENCY_IN_NANOS;
    return EventStatus::OK;
}

NvError CVirtualSrcModule::FillDataBufAttrList(CClientCommon *pClient,
                                               PacketElementType userType,
                                               NvSciBufAttrList *pBufAttrList)
{
    bool bImgCpuAccess = false;
    NvSciBufAttrValAccessPerm accessPerm = NvSciBufAccessPerm_ReadWrite;
    uint32_t uPlaneWidth = m_virtualSrcOption.uWidth;
    uint32_t uPlaneHeight = m_virtualSrcOption.uHeight;
    NvSciBufSurfSampleType surfSampleType = NvSciSurfSampleType_420;
    NvSciBufSurfBPC surfBPC = NvSciSurfBPC_8;
    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufSurfType surfType = NvSciSurfType_YUV;
    NvSciBufSurfMemLayout surfMemLayout = NvSciSurfMemLayout_SemiPlanar;
    NvSciBufSurfComponentOrder surfCompOrder = NvSciSurfComponentOrder_YUV;
    NvSciBufAttrValColorStd surfColorStd[] = { NvSciColorStd_REC709_ER };
    NvSciBufAttrValImageLayoutType layout = m_LayoutType;
    NvSciBufAttrValImageScanType scanType = NvSciBufScan_ProgressiveType;
    NvSciBufAttrKeyValuePair keyVals[] = {
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { NvSciBufImageAttrKey_SurfType, &surfType, sizeof(surfType) },
        { NvSciBufImageAttrKey_SurfBPC, &surfBPC, sizeof(surfBPC) },
        { NvSciBufImageAttrKey_SurfMemLayout, &surfMemLayout, sizeof(surfMemLayout) },
        { NvSciBufImageAttrKey_SurfSampleType, &surfSampleType, sizeof(surfSampleType) },
        { NvSciBufImageAttrKey_SurfComponentOrder, &surfCompOrder, sizeof(surfCompOrder) },
        { NvSciBufImageAttrKey_SurfColorStd, &surfColorStd, sizeof(surfColorStd) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &bImgCpuAccess, sizeof(bool) },
        { NvSciBufGeneralAttrKey_RequiredPerm, &accessPerm, sizeof(accessPerm) },
        { NvSciBufImageAttrKey_Layout, &layout, sizeof(layout) },
        { NvSciBufImageAttrKey_SurfWidthBase, &uPlaneWidth, sizeof(uint32_t) },
        { NvSciBufImageAttrKey_SurfHeightBase, &uPlaneHeight, sizeof(uint32_t) },
        { NvSciBufImageAttrKey_PlaneScanType, &scanType, sizeof(scanType) },
    };

    auto err = NvSciBufAttrListSetAttrs(*pBufAttrList, keyVals, ARRAY_SIZE(keyVals));
    CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufAttrListSetAttrs");

    auto error = CBaseModule::FillDataBufAttrList(pClient, userType, pBufAttrList);
    CHK_ERROR_AND_RETURN(error, "CBaseModule::FillDataBufAttrList");

    return NvError_Success;
}

NvError CVirtualSrcModule::FillSyncSignalerAttrList(CClientCommon *pClient,
                                                    PacketElementType userType,
                                                    NvSciSyncAttrList *pSignalerAttrList)
{
    bool bNeedCpuAccess = true;
    NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_SignalOnly;
    NvSciSyncAttrKeyValuePair keyValues[] = { { NvSciSyncAttrKey_NeedCpuAccess, &bNeedCpuAccess,
                                                sizeof(bNeedCpuAccess) },
                                              { NvSciSyncAttrKey_RequiredPerm, &cpuPerm, sizeof(cpuPerm) } };

    auto sciErr = NvSciSyncAttrListSetAttrs(*pSignalerAttrList, keyValues, ARRAY_SIZE(keyValues));
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListSetAttrs signal");

    return NvError_Success;
}

NvError CVirtualSrcModule::FillSyncWaiterAttrList(CClientCommon *pClient,
                                                  PacketElementType userType,
                                                  NvSciSyncAttrList *pWaiterAttrList)
{
    bool bNeedCpuAccess = true;
    NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_WaitOnly;
    NvSciSyncAttrKeyValuePair setAttrs[] = { { NvSciSyncAttrKey_NeedCpuAccess, &bNeedCpuAccess,
                                               sizeof(bNeedCpuAccess) },
                                             { NvSciSyncAttrKey_RequiredPerm, &cpuPerm, sizeof(cpuPerm) } };
    NvSciError sciErr = NvSciSyncAttrListSetAttrs(*pWaiterAttrList, setAttrs, ARRAY_SIZE(setAttrs));
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListSetAttrs signal");

    auto error = CBaseModule::FillSyncWaiterAttrList(pClient, userType, pWaiterAttrList);
    CHK_ERROR_AND_RETURN(error, "CBaseModule::FillSyncWaiterAttrList");

    return NvError_Success;
}

NvError
CVirtualSrcModule::RegisterSignalSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj signalSyncObj)
{
    return NvError_Success;
}

NvError
CVirtualSrcModule::RegisterWaiterSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj waiterSyncObj)
{
    return NvError_Success;
}

//Before calling PreSync, m_nvmBuffers[uPacketIndex] should already be filled.
NvError CVirtualSrcModule::InsertPrefence(CClientCommon *pClient,
                                          PacketElementType userType,
                                          uint32_t uPacketIndex,
                                          NvSciSyncFence *pPrefence)
{
    return NvError_Success;
}

NvError CVirtualSrcModule::ProcessPayload(CClientCommon *pClient, uint32_t uPacketIndex)
{
    return NvError_Success;
}

NvError CVirtualSrcModule::OnPacketGotten(CClientCommon *pClient, uint32_t uPacketIndex, bool *pHandled)
{
    m_bufObjQueue.push_back(*pClient->GetBufObj(uPacketIndex));
    return NvError_Success;
}

NvError CVirtualSrcModule::RegisterBufObj(CClientCommon *pClient,
                                          PacketElementType userType,
                                          uint32_t uPacketIndex,
                                          NvSciBufObj bufObj)
{
    m_vBufObjs.push_back(bufObj);
    return NvError_Success;
}

const OptionTable *CVirtualSrcModule::GetOptionTable() const
{
    return &VirtualSrcOptionTable;
}

const void *CVirtualSrcModule::GetOptionBaseAddress() const
{
    return &m_virtualSrcOption;
}
