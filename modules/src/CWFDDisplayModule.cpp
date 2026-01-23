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

#include "CWFDDisplayModule.hpp"
#include "CElementDescription.hpp"

CElementDescription wfdDescription{ "Display", "Display through openWFD", &CBaseModule::m_baseModuleOptionTable,
                                    nullptr };

constexpr static uint32_t MAX_WFD_DEVICE_NUM = 1U;
constexpr static uint32_t WFD_PORT_MODES_COUNT = 1U;
constexpr static uint32_t MAX_FRAME_NUM_FOR_DISPLAY = 3; //This value should be less than the packet number.

const std::unordered_map<std::string, Option> CWFDDisplayModule::m_wfdDisplayOptionTable = {
    { "portid", { "displayPortId", offsetof(WFDResInputInfo, uPortIdx), OptionType::UINT32 } },
    { "width", { "the width of the input buffer", offsetof(WFDResInputInfo, uWidth), OptionType::UINT32 } },
    { "height", { "the height of the input buffer", offsetof(WFDResInputInfo, uHeight), OptionType::UINT32 } },
    { "colortype", { "displaycolortype", offsetof(WFDResInputInfo, sColorType), OptionType::STRING } },
    { "imagelayout", { "displayimagelayout", offsetof(WFDResInputInfo, sImageLayout), OptionType::STRING } },
};

const std::unordered_set<std::string> CWFDDisplayModule::m_wfdSupportedColor = { "NV12", "ARGB", "ABGR" };

CWFDDisplayModule::CWFDDisplayModule(std::shared_ptr<CModuleCfg> spModuleCfg, IEventListener<CBaseModule> *pListener)
    : CBaseModule(std::move(spModuleCfg), pListener)
{
    m_postFences.reserve(MAX_NUM_PACKETS);
}

CWFDDisplayModule::~CWFDDisplayModule()
{
    PLOG_DBG("CWFDDisplayModule release.\n");
    for (auto &fence : m_postFences) {
        NvSciSyncFenceClear(&fence);
    }
}

NvError CWFDDisplayModule::Init()
{
    PLOG_DBG("Enter Init\n");

    NvError error = NvError_Success;

    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler = std::make_unique<CProfiler>();
        error = m_upProfiler->Init(m_pAppCfg->m_sciSyncModule, m_pAppCfg->GetPerfDataSaveFolder(), GetName(), true,
                                   m_pAppCfg->GetMaxPerfSampleNum());
        PCHK_ERROR_AND_RETURN(error, "CProfiler::Init()");

        m_upProfiler->RecordInitBeginTime();
    }

    //For display module, we need to get the same sync aggertator before CBaseModule::Init(),
    //because wfd shares the only one signaler object.
    m_spSyncAggregator = CWFDDisplayModule::GetSyncAggregator();
    if (!m_spSyncAggregator) {
        PLOG_ERR("CWFDNvSciSyncAggregator get failed! \n");
        return NvError_ResourceError;
    }

    error = CBaseModule::Init();
    PCHK_ERROR_AND_RETURN(error, "CBaseModule::Init()");

    m_spWFDResInst = CWFDResManager::GetInstance();
    if (!m_spWFDResInst) {
        PLOG_ERR("CWFDResManager get failed! \n");
        return NvError_ResourceError;
    }

    LOG_MSG("CWFDDisplayModule::Init, device id = %d, port id = %d, pipeline id = %d \n",
            static_cast<int>(m_wfdResInputInfo.uDeviceIdx), static_cast<int>(m_wfdResInputInfo.uPortIdx),
            static_cast<int>(m_wfdResInputInfo.uPipelineIdx));
    error = m_spWFDResInst->CreateResouce(&m_wfdResInputInfo, &m_wfdRes);
    PCHK_ERROR_AND_RETURN(error, "CreateResouce");

    // set port mode
    WFDPortMode wfdPortMode = WFD_INVALID_HANDLE;
    auto wfdNumModes = wfdGetPortModes(m_wfdRes.wfdDevice, m_wfdRes.wfdPort, &wfdPortMode, WFD_PORT_MODES_COUNT);
    if (!wfdNumModes) {
        PGET_WFDERROR_AND_RETURN(m_wfdRes.wfdDevice);
    }

    if (0 == m_dstWidth || (0 == m_dstHeight)) {
        m_dstWidth = wfdGetPortModeAttribi(m_wfdRes.wfdDevice, m_wfdRes.wfdPort, wfdPortMode, WFD_PORT_MODE_WIDTH);
        m_dstHeight = wfdGetPortModeAttribi(m_wfdRes.wfdDevice, m_wfdRes.wfdPort, wfdPortMode, WFD_PORT_MODE_HEIGHT);
        LOG_MSG("window width = %d, window height = %d \n", m_dstWidth, m_dstHeight);
    }

    wfdSetPortMode(m_wfdRes.wfdDevice, m_wfdRes.wfdPort, wfdPortMode);
    PGET_WFDERROR_AND_RETURN(m_wfdRes.wfdDevice);

    wfdDeviceCommit(m_wfdRes.wfdDevice, WFD_COMMIT_ENTIRE_PORT, m_wfdRes.wfdPort);
    PGET_WFDERROR_AND_RETURN(m_wfdRes.wfdDevice);

    // set pipeline attribute
    WFDint ret = wfdGetPipelineAttribi(m_wfdRes.wfdDevice, m_wfdRes.wfdPipeline,
                                       static_cast<WFDPipelineConfigAttrib>(WFD_PIPELINE_COMMIT_NON_BLOCKING_NVX));
    PGET_WFDERROR_AND_RETURN(m_wfdRes.wfdDevice);

    if (WFD_FALSE == ret) {
        PLOG_INFO("openwfd non-blocking commit disabled, to enable it\n");
        wfdSetPipelineAttribi(m_wfdRes.wfdDevice, m_wfdRes.wfdPipeline,
                              static_cast<WFDPipelineConfigAttrib>(WFD_PIPELINE_COMMIT_NON_BLOCKING_NVX), WFD_TRUE);
        PGET_WFDERROR_AND_RETURN(m_wfdRes.wfdDevice);
    }

    // Note: On DOS7.0, the WFD_PIPELINE_POSTFENCE_SCANOUT_BEGIN_NVX is deprecated, and do not need to set it explicitly.
#ifdef WFD_PIPELINE_POSTFENCE_SCANOUT_BEGIN_NVX
    ret = wfdGetPipelineAttribi(m_wfdRes.wfdDevice, m_wfdRes.wfdPipeline,
                                static_cast<WFDPipelineConfigAttrib>(WFD_PIPELINE_POSTFENCE_SCANOUT_BEGIN_NVX));
    PGET_WFDERROR_AND_RETURN(m_wfdRes.wfdDevice);

    if (WFD_TRUE == ret) {
        // set it to false means post fence to be signaled after scannout end.
        wfdSetPipelineAttribi(m_wfdRes.wfdDevice, m_wfdRes.wfdPipeline,
                              static_cast<WFDPipelineConfigAttrib>(WFD_PIPELINE_POSTFENCE_SCANOUT_BEGIN_NVX),
                              WFD_FALSE);
        PGET_WFDERROR_AND_RETURN(m_wfdRes.wfdDevice);
    }
#endif

    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler->RecordInitEndTime();
    }

    PLOG_DBG("Exit Init\n");
    return NvError_Success;
}

NvError CWFDDisplayModule::Start()
{
    PLOG_DBG("Enter: CWFDDisplayModule::Start()\n");

    // set port mode
    WFDPortMode wfdPortMode = WFD_INVALID_HANDLE;
    auto wfdNumModes = wfdGetPortModes(m_wfdRes.wfdDevice, m_wfdRes.wfdPort, &wfdPortMode, WFD_PORT_MODES_COUNT);
    if (!wfdNumModes) {
        PGET_WFDERROR_AND_RETURN(m_wfdRes.wfdDevice);
    }

    wfdSetPortMode(m_wfdRes.wfdDevice, m_wfdRes.wfdPort, wfdPortMode);
    PGET_WFDERROR_AND_RETURN(m_wfdRes.wfdDevice);

    wfdDeviceCommit(m_wfdRes.wfdDevice, WFD_COMMIT_ENTIRE_PORT, m_wfdRes.wfdPort);
    PGET_WFDERROR_AND_RETURN(m_wfdRes.wfdDevice);

    wfdSetPipelineAttribi(m_wfdRes.wfdDevice, m_wfdRes.wfdPipeline,
                          static_cast<WFDPipelineConfigAttrib>(WFD_PIPELINE_COMMIT_NON_BLOCKING_NVX), WFD_TRUE);
    PGET_WFDERROR_AND_RETURN(m_wfdRes.wfdDevice);

    auto error = CBaseModule::Start();
    PCHK_ERROR_AND_RETURN(error, "CBaseModule::Start");

    PLOG_DBG("Exit: CWFDDisplayModule::Start()\n");

    return NvError_Success;
}

NvError CWFDDisplayModule::Stop()
{
    PLOG_DBG("Enter: CWFDDisplayModule::Stop()\n");

    auto error = CBaseModule::Stop();
    PCHK_ERROR_AND_RETURN(error, "CBaseModule::Stop");

    if (m_wfdRes.wfdDevice && m_wfdRes.wfdPipeline) {
        // Perform a null flip
        wfdBindSourceToPipeline(m_wfdRes.wfdDevice, m_wfdRes.wfdPipeline, (WFDSource)0, WFD_TRANSITION_AT_VSYNC, NULL);
        CHECK_WFD_ERROR(m_wfdRes.wfdDevice);

        wfdSetPipelineAttribi(m_wfdRes.wfdDevice, m_wfdRes.wfdPipeline,
                              static_cast<WFDPipelineConfigAttrib>(WFD_PIPELINE_COMMIT_NON_BLOCKING_NVX), WFD_FALSE);
        CHECK_WFD_ERROR(m_wfdRes.wfdDevice);

        wfdDeviceCommit(m_wfdRes.wfdDevice, WFD_COMMIT_PIPELINE, m_wfdRes.wfdPipeline);
        CHECK_WFD_ERROR(m_wfdRes.wfdDevice);
    }

    for (auto &fence : m_postFences) {
        NvSciSyncFenceClear(&fence);
    }
    m_postFences.clear();

    PLOG_DBG("Exit: CWFDDisplayModule::Stop()\n");

    return NvError_Success;
}

void CWFDDisplayModule::DeInit()
{
    PLOG_DBG("Enter: CWFDDisplayModule::DeInit()\n");

    for (uint32_t i = 0U; i < MAX_NUM_PACKETS; ++i) {
        if (m_wfdSources[i] != WFD_INVALID_HANDLE) {
            wfdDestroySource(m_wfdRes.wfdDevice, m_wfdSources[i]);
            CHECK_WFD_ERROR(m_wfdRes.wfdDevice);
        }
    }

    CBaseModule::DeInit();

    PLOG_DBG("Exit: CWFDDisplayModule::DeInit()\n");
}

NvError CWFDDisplayModule::FillDataBufAttrList(CClientCommon *pClient,
                                               PacketElementType userType,
                                               NvSciBufAttrList *pBufAttrList)
{
    PLOG_DBG("Enter: CWFDDisplayModule::FillDataBufAttrList()\n");

    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValImageScanType bufScanType = NvSciBufScan_ProgressiveType;
    NvSciBufAttrValAccessPerm bufPerm = NvSciBufAccessPerm_Readonly;

    bool needCpuAccessFlag = false;
    NvSciBufAttrKeyValuePair bufAttrs[] = {
        { NvSciBufGeneralAttrKey_RequiredPerm, &bufPerm, sizeof(bufPerm) },
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &needCpuAccessFlag, sizeof(needCpuAccessFlag) },
        { NvSciBufImageAttrKey_ScanType, &bufScanType, sizeof(bufScanType) },
    };

    // Default buffer attributes
    WFDErrorCode wfdErr = wfdNvSciBufSetDisplayAttributesNVX(pBufAttrList);
    PCHK_WFDSTATUS_AND_RETURN(wfdErr, "wfdNvSciBufSetDisplayAttributesNVX");

    auto sciErr = NvSciBufAttrListSetAttrs(*pBufAttrList, bufAttrs, ARRAY_SIZE(bufAttrs));
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

    // For opaque element type, color type and memory layout are decided by options.
    if (userType == PacketElementType::OPAQUE) {
        std::string &sImageLayout = m_wfdResInputInfo.sImageLayout;
        std::string &sColorType = m_wfdResInputInfo.sColorType;

        // Check supported imagelayout.
        if (!sImageLayout.empty() && sImageLayout != "BL" && sImageLayout != "PL") {
            PLOG_ERR("Image layout type specified not support by display.\n"
                     "Supported Image layout type option: bl(default), pl.\n");
            return NvError_BadValue;
        }

        // Check supported color type attributes.
        if (!sColorType.empty() && m_wfdSupportedColor.find(sColorType) == m_wfdSupportedColor.end()) {
            PLOG_ERR("Color type specified not support by display.\n"
                     "Supported color type: nv12(default), argb, abgr.\n");
            return NvError_BadValue;
        }

        auto error =
            SetBufAttr(pBufAttrList, sColorType, sImageLayout, m_wfdResInputInfo.uWidth, m_wfdResInputInfo.uHeight);
        PCHK_ERROR_AND_RETURN(error, "SetBufAttr");
    }

    PLOG_DBG("Exit: CWFDDisplayModule::FillDataBufAttrList()\n");
    return NvError_Success;
}

NvError CWFDDisplayModule::FillSyncSignalerAttrList(CClientCommon *pClient,
                                                    PacketElementType userType,
                                                    NvSciSyncAttrList *pSignalerAttrList)
{
    PLOG_DBG("Enter: CWFDDisplayModule::FillSyncSignalerAttrList()\n");

    WFDErrorCode wfdErr = wfdNvSciSyncSetSignalerAttributesNVX(pSignalerAttrList);
    PCHK_WFDSTATUS_AND_RETURN(wfdErr, "wfdNvSciSyncSetSignalerAttributesNVX");

    bool bNeedCpuAccess = true;
    NvSciSyncAccessPerm accessPerm = NvSciSyncAccessPerm_SignalOnly;
    NvSciSyncAttrValPrimitiveType primitiveInfo[1] = { NvSciSyncAttrValPrimitiveType_Syncpoint };
    NvSciSyncAttrKeyValuePair syncSignalAttrs[] = {
        { NvSciSyncAttrKey_NeedCpuAccess, &bNeedCpuAccess, sizeof(bNeedCpuAccess) },
        { NvSciSyncAttrKey_RequiredPerm, (void *)&accessPerm, sizeof(accessPerm) },
        { NvSciSyncAttrKey_PrimitiveInfo, &primitiveInfo[0], sizeof(primitiveInfo) }
    };

    auto sciErr = NvSciSyncAttrListSetAttrs(*pSignalerAttrList, syncSignalAttrs, ARRAY_SIZE(syncSignalAttrs));
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListSetAttrs");

    PLOG_DBG("Exit: CWFDDisplayModule::FillSyncSignalerAttrList()\n");
    return NvError_Success;
}

NvError CWFDDisplayModule::FillSyncWaiterAttrList(CClientCommon *pClient,
                                                  PacketElementType userType,
                                                  NvSciSyncAttrList *pWaiterAttrList)
{
    PLOG_DBG("Enter: CWFDDisplayModule::FillSyncWaiterAttrList()\n");

    WFDErrorCode wfdErr = wfdNvSciSyncSetWaiterAttributesNVX(pWaiterAttrList);
    PCHK_WFDSTATUS_AND_RETURN(wfdErr, "wfdNvSciSyncSetWaiterAttributesNVX");

    NvSciSyncAccessPerm accessPerm = NvSciSyncAccessPerm_WaitOnly;
    NvSciSyncAttrValPrimitiveType primitiveInfo[1] = { NvSciSyncAttrValPrimitiveType_Syncpoint };
    NvSciSyncAttrKeyValuePair syncWaiterAttrs[] = {
        { NvSciSyncAttrKey_RequiredPerm, (void *)&accessPerm, sizeof(accessPerm) },
        { NvSciSyncAttrKey_PrimitiveInfo, &primitiveInfo[0], sizeof(primitiveInfo) }
    };

    auto sciErr = NvSciSyncAttrListSetAttrs(*pWaiterAttrList, syncWaiterAttrs, ARRAY_SIZE(syncWaiterAttrs));
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListSetAttrs");

    PLOG_DBG("Exit: CWFDDisplayModule::FillSyncWaiterAttrList()\n");

    return NvError_Success;
}

NvError
CWFDDisplayModule::RegisterSignalSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj signalSyncObj)
{
    PLOG_DBG("Enter: CWFDDisplayModule::RegisterSignalSyncObj()\n");

    // only one signal object can be registered.
    WFDErrorCode wfdErr = wfdRegisterPostFlipNvSciSyncObjNVX(m_wfdRes.wfdDevice, &signalSyncObj);
    PCHK_WFDSTATUS_AND_RETURN(wfdErr, "wfdRegisterPostFlipNvSciSyncObjNVX");

    PLOG_DBG("Exit: CWFDDisplayModule::RegisterSignalSyncObj()\n");

    return NvError_Success;
}

NvError
CWFDDisplayModule::RegisterWaiterSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj waiterSyncObj)
{
    PLOG_DBG("Enter: CWFDDisplayModule::RegisterWaiterSyncObj()\n");

    PLOG_DBG("Exit: CWFDDisplayModule::RegisterWaiterSyncObj()\n");
    return NvError_Success;
}

//Before calling PreSync, m_nvmBuffers[uPacketIndex] should already be filled.
NvError CWFDDisplayModule::InsertPrefence(CClientCommon *pClient,
                                          PacketElementType userType,
                                          uint32_t uPacketIndex,
                                          NvSciSyncFence *pPrefence)
{
    PLOG_DBG("Enter: CWFDDisplayModule::InsertPrefence()\n");

    WFDErrorCode wfdErr = wfdBindNvSciSyncFenceToSourceNVX(m_wfdRes.wfdDevice, m_wfdSources[uPacketIndex], pPrefence);
    PCHK_WFDSTATUS_AND_RETURN(wfdErr, "wfdBindNvSciSyncFenceToSourceNVX");

    PLOG_DBG("Exit: CWFDDisplayModule::InsertPrefence()\n");
    return NvError_Success;
}

NvError CWFDDisplayModule::GetEofSyncFence(CClientCommon *pClient, NvSciSyncFence *pPostfence)
{
    PLOG_DBG("Enter: CWFDDisplayModule::GetEofSyncFence()\n");

    if (m_isFrameDrop || m_postFences.empty()) {
        return NvError_Success;
    }

    auto sciErr = NvSciSyncFenceDup(&m_postFences.back(), pPostfence);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncFenceDup");
    PLOG_DBG("Exit: CWFDDisplayModule::GetEofSyncFence()\n");

    if (m_pAppCfg->IsProfilingEnabled()) {
        if (!m_bHasDownstream && m_pMetaData) {
            m_upProfiler->RecordExecutionAndPipelineTime(m_pMetaData->uFrameCaptureStartTSC, pPostfence);
            m_pMetaData = nullptr;
        } else {
            m_upProfiler->RecordExecutionEndTime(pPostfence);
        }
    }

    return NvError_Success;
}

bool CWFDDisplayModule::IsFull(CClientCommon *pClient)
{
    const uint32_t &fenceNum = m_postFences.size();
    if (fenceNum == MAX_FRAME_NUM_FOR_DISPLAY ||
        fenceNum == MAX_NUM_PACKETS) { // we should care about when the MAX_FRAME_NUM_FOR_DISPLAY > MAX_NUM_PACKETS.
        // check if the oldest flip is complete.
        auto sciErr = pClient->IsPostFenceExpired(&m_postFences[0]);
        if (NvSciError_Success == sciErr) {
            // the oldest flip is complete, need to clear the fence, erase it from queue to make room.
            NvSciSyncFenceClear(&m_postFences[0]);
            m_postFences.erase(m_postFences.begin(), m_postFences.begin() + 1U);
            return false;
        } else {
            PLOG_DBG("nvsci status: %u(0x%x)\n", sciErr, sciErr);
            return true;
        }
    }

    return false;
}

NvError CWFDDisplayModule::ProcessPayload(CClientCommon *pClient, uint32_t uPacketIndex)
{
    PLOG_DBG("Enter: CWFDDisplayModule::ProcessPayload()\n");

    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler->RecordSubmissionBeginTime();
    }

    m_isFrameDrop = IsFull(pClient);
    if (m_isFrameDrop) {
        PLOG_WARN("Frame drop, packet id = %u \n", uPacketIndex);
        return NvError_Success;
    }

    wfdBindSourceToPipeline(m_wfdRes.wfdDevice, m_wfdRes.wfdPipeline, m_wfdSources[uPacketIndex],
                            WFD_TRANSITION_AT_VSYNC, NULL);
    PGET_WFDERROR_AND_RETURN(m_wfdRes.wfdDevice);

    NvSciSyncFence postFence = NvSciSyncFenceInitializer;
    wfdDeviceCommitWithNvSciSyncFenceNVX(m_wfdRes.wfdDevice, WFD_COMMIT_PIPELINE, m_wfdRes.wfdPipeline, &postFence);
    PGET_WFDERROR_AND_RETURN(m_wfdRes.wfdDevice);

    m_postFences.push_back(postFence);

    if (m_pAppCfg->IsProfilingEnabled()) {
        NvError error = m_upProfiler->RecordSubmissionEndTime();
        PCHK_ERROR_AND_RETURN(error, "RecordSubmissionEndTime");

        error = m_upProfiler->RecordExecutionBeginTime();
        PCHK_ERROR_AND_RETURN(error, "RecordExecutionBeginTime");

        m_pMetaData = reinterpret_cast<MetaData *>(pClient->GetMetaPtr(uPacketIndex));
    }

    PLOG_DBG("Exit: CWFDDisplayModule::ProcessPayload()\n");
    return NvError_Success;
}

NvError CWFDDisplayModule::RegisterBufObj(CClientCommon *pClient,
                                          PacketElementType userType,
                                          uint32_t uPacketIndex,
                                          NvSciBufObj bufObj)
{
    PLOG_DBG("Enter: CWFDDisplayModule::RegisterBufObj()\n");

    m_wfdSources[uPacketIndex] = wfdCreateSourceFromNvSciBufNVX(m_wfdRes.wfdDevice, m_wfdRes.wfdPipeline, &bufObj);
    if (!m_wfdSources[uPacketIndex]) {
        PGET_WFDERROR_AND_RETURN(m_wfdRes.wfdDevice);
    }

    PLOG_DBG("Exit: CWFDDisplayModule::RegisterBufObj()\n");
    return NvError_Success;
}

NvError CWFDDisplayModule::OnDataBufAttrListRecvd(CClientCommon *pClient, NvSciBufAttrList bufAttrList)
{
    uint16_t srcWidth = 0U;
    uint16_t srcHeight = 0U;

    auto error = GetWidthAndHeight(bufAttrList, srcWidth, srcHeight);
    PCHK_ERROR_AND_RETURN(error, "GetWidthAndHeight");

    LOG_MSG("Display::SetRect, expected src width = %u, src height = %u, dst width = %u, dst height = %u \n", srcWidth,
            srcHeight, m_dstWidth, m_dstHeight);

#ifndef DISIPLAY_SCALING_USED
    if (srcWidth > m_dstWidth && srcHeight > m_dstHeight) {
        LOG_MSG("CWFDDisplayModule, Warning:the source resolution is different from the display resolution, forced "
                "adjustment !!!\n");
        // Currently, we do NOT support display downscaling officially.
        // So, if you only want to verify this use case and are not expected to the correct display content,
        // you can try to adjust the srcWidth = m_dstWidth, srcHeight = m_dstHeight.
        srcWidth = m_dstWidth;
        srcHeight = m_dstHeight;

        LOG_MSG(
            "CWFDDisplayModule, Warning: There is no guarantee that the displayed content will be as expected !!!\n");
    } else if (srcWidth < m_dstWidth || srcHeight < m_dstHeight) {
        LOG_ERR("Not supported! \n");
        return NvError_NotSupported;
    }
#endif

    LOG_MSG("Display::SetRect, actual src width = %u, src height = %u, dst width = %u, dst height = %u \n", srcWidth,
            srcHeight, m_dstWidth, m_dstHeight);
    WFDint wfdSrcRect[4]{ 0, 0, static_cast<WFDint>(srcWidth), static_cast<WFDint>(srcHeight) };
    WFDint wfdDstRect[4]{ 0, 0, static_cast<WFDint>(m_dstWidth), static_cast<WFDint>(m_dstHeight) };

    wfdSetPipelineAttribiv(m_wfdRes.wfdDevice, m_wfdRes.wfdPipeline, WFD_PIPELINE_SOURCE_RECTANGLE, 4, wfdSrcRect);
    PGET_WFDERROR_AND_RETURN(m_wfdRes.wfdDevice);

    wfdSetPipelineAttribiv(m_wfdRes.wfdDevice, m_wfdRes.wfdPipeline, WFD_PIPELINE_DESTINATION_RECTANGLE, 4, wfdDstRect);
    PGET_WFDERROR_AND_RETURN(m_wfdRes.wfdDevice);

    return NvError_Success;
}

const OptionTable *CWFDDisplayModule::GetOptionTable() const
{
    return &m_wfdDisplayOptionTable;
}

const void *CWFDDisplayModule::GetOptionBaseAddress() const
{
    return &m_wfdResInputInfo;
}

std::shared_ptr<CSyncAggregator> CWFDDisplayModule::GetSyncAggregator()
{
    static std::weak_ptr<CSyncAggregator> s_wpInstance;
    static std::mutex s_instanceMutex;
    std::shared_ptr<CSyncAggregator> wfdSyncAggr = nullptr;

    {
        std::unique_lock<std::mutex> lock{ s_instanceMutex };
        wfdSyncAggr = s_wpInstance.lock();
        if (!wfdSyncAggr) {
            LOG_INFO("CSyncAggregator instance created \n");
            wfdSyncAggr = std::shared_ptr<CSyncAggregator>(new CSyncAggregator());
            s_wpInstance = wfdSyncAggr;
        } else {
            LOG_INFO("CSyncAggregator instance gotten \n");
        }
    }

    return wfdSyncAggr;
}

std::shared_ptr<CWFDResManager> CWFDResManager::GetInstance()
{
    static std::weak_ptr<CWFDResManager> s_wpInstance;
    static std::mutex s_instanceMutex;
    std::shared_ptr<CWFDResManager> wfdRes = nullptr;

    {
        std::unique_lock<std::mutex> lock{ s_instanceMutex };
        wfdRes = s_wpInstance.lock();
        if (!wfdRes) {
            LOG_INFO("CWFDResManager instance created \n");
            wfdRes = std::shared_ptr<CWFDResManager>(new CWFDResManager());
            s_wpInstance = wfdRes;
        } else {
            LOG_INFO("CWFDResManager instance gotten \n");
        }
    }

    return wfdRes;
}

CWFDResManager::CWFDResManager()
{
    m_vWfdResInfos.resize(MAX_WFD_DEVICE_NUM);
    for (uint32_t i = 0U; i < m_vWfdResInfos.size(); ++i) {
        m_vWfdResInfos[i].first = WFD_INVALID_HANDLE;
        std::vector<WFDPortWithPipelines> &wfdPortInfo = m_vWfdResInfos[i].second;
        wfdPortInfo.resize(MAX_NUM_WFD_PORTS);
        for (uint32_t j = 0U; j < wfdPortInfo.size(); ++j) {
            wfdPortInfo[j].first = WFD_INVALID_HANDLE;
            wfdPortInfo[j].second.resize(MAX_NUM_WFD_PIPELINES, WFD_INVALID_HANDLE);
        }
    }
}

CWFDResManager::~CWFDResManager()
{
    LOG_DBG("Enter CWFDResManager::CWFDResManager release\n");

    for (uint32_t i = 0U; i < m_vWfdResInfos.size(); ++i) {
        std::vector<WFDPortWithPipelines> &wfdPortInfo = m_vWfdResInfos[i].second;
        for (uint32_t j = 0U; j < wfdPortInfo.size(); ++j) {
            std::vector<WFDPipeline> &wfdPipelines = wfdPortInfo[j].second;
            for (uint32_t k = 0U; k < wfdPipelines.size(); ++k) {
                if (wfdPipelines[k]) {
                    LOG_INFO("CWFDResManager, wfdDestroyPipeline index %d \n", k);
                    wfdDestroyPipeline(m_vWfdResInfos[i].first, wfdPipelines[k]);
                    CHECK_WFD_ERROR(m_vWfdResInfos[i].first);
                }
            }

            if (wfdPortInfo[j].first) {
                LOG_INFO("CWFDResManager, wfdDestroyPort index %d \n", j);
                wfdDestroyPort(m_vWfdResInfos[i].first, wfdPortInfo[j].first);
                CHECK_WFD_ERROR(m_vWfdResInfos[i].first);
            }
        }

        if (m_vWfdResInfos[i].first) {
            LOG_INFO("CWFDResManager, wfdDestroyDevice index %d \n", i);
            wfdDestroyDevice(m_vWfdResInfos[i].first);
        }
    }

    LOG_DBG("End CWFDResManager::CWFDResManager release\n");
}

WFDDevice CWFDResManager::GetDisplayDevice(uint32_t uDeviceIdx)
{
    if (uDeviceIdx >= MAX_WFD_DEVICE_NUM) {
        LOG_ERR("CWFDResManager, the maxinum number of device is %d \n", MAX_WFD_DEVICE_NUM);
        return WFD_INVALID_HANDLE;
    }

    WFDDevice wfdDevice = WFD_INVALID_HANDLE;
    {
        std::unique_lock<std::mutex> lock{ m_mutex };
        if (WFD_INVALID_HANDLE == m_vWfdResInfos[uDeviceIdx].first) {
            m_vWfdResInfos[uDeviceIdx].first = wfdCreateDevice(uDeviceIdx, NULL);
        }

        wfdDevice = m_vWfdResInfos[uDeviceIdx].first;
    }

    return wfdDevice;
}

WFDPort CWFDResManager::GetWfdPort(uint32_t uDeviceIdx, uint32_t uPortIdx)
{
    if (uPortIdx >= MAX_NUM_WFD_PORTS) {
        LOG_ERR("CWFDResManager, the maxinum number of ports is %d \n", MAX_NUM_WFD_PORTS);
        return WFD_INVALID_HANDLE;
    }

    WFDPort wfdPort = WFD_INVALID_HANDLE;
    {
        std::unique_lock<std::mutex> lock{ m_mutex };
        WFDDeviceWithPortsInfo &wfdDevWithPortsInfo = m_vWfdResInfos[uDeviceIdx];
        WFDPortWithPipelines &wfdPortWithPipelines = wfdDevWithPortsInfo.second[uPortIdx];
        if (WFD_INVALID_HANDLE == wfdPortWithPipelines.first) {
            WFDint wfdPortIds[MAX_NUM_WFD_PORTS] = { 0 };
            WFDint wfdNumPorts = wfdEnumeratePorts(wfdDevWithPortsInfo.first, wfdPortIds, MAX_NUM_WFD_PORTS, NULL);
            if (static_cast<int32_t>(uPortIdx) >= wfdNumPorts) {
                LOG_ERR("wfdEnumeratePorts wfdNumPorts wrong error! \n");
                return WFD_INVALID_HANDLE;
            }

            LOG_MSG("CWFDResManager::GetWfdPort, wfd number of ports = %d, get the wfd port id %d \n", wfdNumPorts,
                    wfdPortIds[uPortIdx]);

            wfdPortWithPipelines.first = wfdCreatePort(wfdDevWithPortsInfo.first, wfdPortIds[uPortIdx], NULL);
        }

        wfdPort = wfdPortWithPipelines.first;
    }

    return wfdPort;
}

WFDPipeline CWFDResManager::GetPipeline(uint32_t uDeviceIdx, uint32_t uPortIdx, uint32_t uPipelineIdx)
{
    if (uPipelineIdx >= MAX_NUM_WFD_PIPELINES) {
        LOG_ERR("CWFDResManager, can not create more pipelines, maybe you can increase the max number of pipelines"
                " MAX_NUM_WFD_PIPELINES(%u)\n",
                MAX_NUM_WFD_PIPELINES);
        return WFD_INVALID_HANDLE;
    }

    WFDPipeline wfdPipeline = WFD_INVALID_HANDLE;
    {
        std::unique_lock<std::mutex> lock{ m_mutex };
        WFDDeviceWithPortsInfo &wfdDevWithPortsInfo = m_vWfdResInfos[uDeviceIdx];
        WFDPortWithPipelines &wfdPortWithPipelines = wfdDevWithPortsInfo.second[uPortIdx];
        if (WFD_INVALID_HANDLE == wfdPortWithPipelines.second[uPipelineIdx]) {
            if (CreatePipeline(wfdDevWithPortsInfo.first, wfdPortWithPipelines.first, uPipelineIdx,
                               wfdPortWithPipelines.second[uPipelineIdx]) != NvError_Success) {
                return WFD_INVALID_HANDLE;
            }
        }

        wfdPipeline = wfdPortWithPipelines.second[uPipelineIdx];
    }

    return wfdPipeline;
}

NvError
CWFDResManager::CreatePipeline(WFDDevice wfdDev, WFDPort wfdPort, uint32_t uPipelineIdx, WFDPipeline &wfdPipeline)
{
    //Get the number of bindable pipeline IDs for a port
    auto wfdNumPipelines = wfdGetPortAttribi(wfdDev, wfdPort, WFD_PORT_PIPELINE_ID_COUNT);
    PGET_WFDERROR_AND_RETURN(wfdDev);

    //Populate pipeline IDs into m_wfdPipeline
    std::vector<WFDint> wfdBindablePipeIds(wfdNumPipelines);
    wfdGetPortAttribiv(wfdDev, wfdPort, WFD_PORT_BINDABLE_PIPELINE_IDS, wfdNumPipelines, wfdBindablePipeIds.data());
    PGET_WFDERROR_AND_RETURN(wfdDev);
    if (static_cast<int32_t>(uPipelineIdx) >= wfdNumPipelines) {
        LOG_ERR("CWFDResManager, no pipeline is found, wfdNumPipelines = %d, uPipelineIdx = %d \n", wfdNumPipelines,
                uPipelineIdx);
        return NvError_BadValue;
    }

    LOG_MSG("CWFDResManager, wfd number of pipelines = %d, wfdBindablePipeIds = %d\n", wfdNumPipelines,
            wfdBindablePipeIds[uPipelineIdx]);

    wfdPipeline = wfdCreatePipeline(wfdDev, wfdBindablePipeIds[uPipelineIdx], NULL);
    if (!wfdPipeline) {
        PGET_WFDERROR_AND_RETURN(wfdDev);
    }

    wfdBindPipelineToPort(wfdDev, wfdPort, wfdPipeline);
    PGET_WFDERROR_AND_RETURN(wfdDev);

    LOG_DBG("%s: pipeline is created and bound successfully\n", __func__);

    wfdDeviceCommit(wfdDev, WFD_COMMIT_ENTIRE_PORT, wfdPort);
    PGET_WFDERROR_AND_RETURN(wfdDev);

    LOG_DBG("%s: wfdBindPipelineToPort success\n", __func__);

    return NvError_Success;
}

NvError CWFDResManager::CreateResouce(const WFDResInputInfo *const pInputInfo, WFDResource *pWfdRes)
{
    pWfdRes->wfdDevice = GetDisplayDevice(pInputInfo->uDeviceIdx);
    if (!pWfdRes->wfdDevice) {
        LOG_ERR("CWFDResManager::CreateResouce, get wfd device failed\n");
        return NvError_ResourceError;
    }

    pWfdRes->wfdPort = GetWfdPort(pInputInfo->uDeviceIdx, pInputInfo->uPortIdx);
    if (!pWfdRes->wfdPort) {
        LOG_ERR("CWFDResManager::CreateResouce, get wfd port failed\n");
        PGET_WFDERROR_AND_RETURN(pWfdRes->wfdDevice);
    }

    pWfdRes->wfdPipeline = GetPipeline(pInputInfo->uDeviceIdx, pInputInfo->uPortIdx, pInputInfo->uPipelineIdx);
    if (!pWfdRes->wfdPipeline) {
        LOG_ERR("CWFDResManager::CreateResouce, get wfd pipeline failed\n");
        PGET_WFDERROR_AND_RETURN(pWfdRes->wfdDevice);
    }

    return NvError_Success;
}
