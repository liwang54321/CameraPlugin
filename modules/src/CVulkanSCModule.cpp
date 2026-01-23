/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "CVulkanSCModule.hpp"

const std::unordered_map<std::string, Option> CVulkanSCModule::m_vulkanSCOptionTable = {
    { "width", { "the width of the input buffer from Nvm2d", offsetof(VulkanSCOption, uWidth), OptionType::UINT32 } },
    { "height",
      { "the height of the input buffer from Nvm2d", offsetof(VulkanSCOption, uHeight), OptionType::UINT32 } },
    { "useVkSema",
      { "specify whether vkSemaphore is used or not", offsetof(VulkanSCOption, bUseVkSemaphore), OptionType::BOOL } },
    { "colortype", { "inputColortype", offsetof(VulkanSCOption, sColorType), OptionType::STRING } },
};

const std::unordered_set<std::string> CVulkanSCModule::m_vulkanSCSupportedColor = { "ARGB", "ABGR" };

CElementDescription vulkanSCDescription{ "VulkanSC", "VulkanSC module for demostrate rendering to NvSciBuf",
                                         &CBaseModule::m_baseModuleOptionTable,
                                         &CVulkanSCModule::m_vulkanSCOptionTable };

CVulkanSCModule::CVulkanSCModule(std::shared_ptr<CModuleCfg> spModuleCfg, IEventListener<CBaseModule> *pListener)
    : CBaseModule(std::move(spModuleCfg), pListener)
{
}

NvError CVulkanSCModule::Init()
{
    PLOG_DBG("Enter: CVulkanSCModule::Init()\n");

    NvError error = NvError_Success;

    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler = std::make_unique<CProfiler>();
        error = m_upProfiler->Init(m_pAppCfg->m_sciSyncModule, m_pAppCfg->GetPerfDataSaveFolder(), GetName(), false,
                                   m_pAppCfg->GetMaxPerfSampleNum());
        PCHK_ERROR_AND_RETURN(error, "CProfiler::Init()");

        m_upProfiler->RecordInitBeginTime();
    }

    error = CBaseModule::Init();
    PCHK_ERROR_AND_RETURN(error, "CBaseModule::Init()");

    m_spVKSCEngine = std::make_shared<CVulkanSCEngine>();
    error = m_spVKSCEngine->Init(m_VulkanSCOption.uWidth, m_VulkanSCOption.uHeight, m_VulkanSCOption.bUseVkSemaphore,
                                 m_VulkanSCOption.sColorType);
    PCHK_ERROR_AND_RETURN(error, "CVulkanSCEngine::Init()");

    if (m_upOutputBuf == nullptr) {
        m_uOutputBufValidLen = m_spVKSCEngine->GetOutputImageSize();
        m_upOutputBuf.reset(new (std::nothrow) uint8_t[m_uOutputBufValidLen]);
    }

    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler->RecordInitEndTime();
    }

    PLOG_DBG("Exit: CVulkanSCModule::Init()\n");
    return NvError_Success;
}

void CVulkanSCModule::DeInit()
{
    CBaseModule::DeInit();

    if (m_spVKSCEngine != nullptr) {
        m_spVKSCEngine->DeInit();
    }
}

NvError
CVulkanSCModule::FillDataBufAttrList(CClientCommon *pClient, PacketElementType userType, NvSciBufAttrList *pBufAttrList)
{
    PLOG_DBG("Enter FillDataBufAttrList()\n");

    std::string &sColorType = m_VulkanSCOption.sColorType;
    // Check supported color type attributes.
    if (!sColorType.empty() && m_vulkanSCSupportedColor.find(sColorType) == m_vulkanSCSupportedColor.end()) {
        PLOG_ERR("Color type specified not support by VulkanSC.\n"
                 "Supported color type: abgr(default), argb.\n");
        return NvError_NotSupported;
    }

    NvError error = NvError_Success;

    if (m_spVKSCEngine != nullptr) {
        error = m_spVKSCEngine->GetSciBufAttributesNV(pBufAttrList);
        PCHK_ERROR_AND_RETURN(error, "m_spVKSCEngine->GetSciBufAttributesNV");
    }

    //Block linear for best performance.
    error = SetBufAttr(pBufAttrList, sColorType, "BL", m_VulkanSCOption.uWidth, m_VulkanSCOption.uHeight);
    PCHK_ERROR_AND_RETURN(error, "SetBufAttr");

    return NvError_Success;
}

NvError CVulkanSCModule::FillSyncSignalerAttrList(CClientCommon *pClient,
                                                  PacketElementType userType,
                                                  NvSciSyncAttrList *pSignalerAttrList)
{
    PLOG_DBG("Enter FillSyncSignalerAttrList()\n");

    if (m_spVKSCEngine != nullptr) {
        auto error = m_spVKSCEngine->GetSciSyncAttributesNV(VKSCSyncType::VulkanSCSignaler, pSignalerAttrList);
        PCHK_ERROR_AND_RETURN(error, "m_spVKSCEngine->GetSciSyncAttributesNV");
    }

    PLOG_DBG("Exit FillSyncSignalerAttrList()\n");

    return NvError_Success;
}

NvError CVulkanSCModule::FillSyncWaiterAttrList(CClientCommon *pClient,
                                                PacketElementType userType,
                                                NvSciSyncAttrList *pWaiterAttrList)
{
    PLOG_DBG("Enter FillSyncWaiterAttrList()\n");

    if (m_spVKSCEngine != nullptr) {
        auto error = m_spVKSCEngine->GetSciSyncAttributesNV(VKSCSyncType::VulkanSCWaiter, pWaiterAttrList);
        PCHK_ERROR_AND_RETURN(error, "m_spVKSCEngine->GetSciSyncAttributesNV");
    }

    PLOG_DBG("Exit FillSyncWaiterAttrList()\n");

    return NvError_Success;
}

NvError
CVulkanSCModule::RegisterSignalSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj signalSyncObj)
{
    m_signalSyncObj = signalSyncObj;

    if (m_spVKSCEngine != nullptr) {
        auto error = m_spVKSCEngine->RegisterSyncObj(VKSCSyncType::VulkanSCSignaler, signalSyncObj);
        PCHK_ERROR_AND_RETURN(error, "m_spVKSCEngine->RegisterSyncObj");
    }

    return NvError_Success;
}

NvError
CVulkanSCModule::RegisterWaiterSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj waiterSyncObj)
{
    auto error = CBaseModule::RegisterWaiterSyncObj(pClient, userType, waiterSyncObj);
    PCHK_ERROR_AND_RETURN(error, "CBaseModule::RegisterWaiterSyncObj");

    if (m_spVKSCEngine != nullptr) {
        error = m_spVKSCEngine->RegisterSyncObj(VKSCSyncType::VulkanSCWaiter, waiterSyncObj);
        PCHK_ERROR_AND_RETURN(error, "m_spVKSCEngine->RegisterSyncObj");
    }

    return NvError_Success;
}

NvError CVulkanSCModule::RegisterBufObj(CClientCommon *pClient,
                                        PacketElementType userType,
                                        uint32_t uPacketIndex,
                                        NvSciBufObj bufObj)
{
    auto error = CBaseModule::RegisterBufObj(pClient, userType, uPacketIndex, bufObj);
    PCHK_ERROR_AND_RETURN(error, "CBaseModule::RegisterBufObj");

    if (m_spVKSCEngine != nullptr) {
        error = m_spVKSCEngine->RegisterBufObj(uPacketIndex, bufObj);
        PCHK_ERROR_AND_RETURN(error, "m_spVKSCEngine->RegisterBufObj");
    }

    return NvError_Success;
}

NvError CVulkanSCModule::InsertPrefence(CClientCommon *pClient,
                                        PacketElementType userType,
                                        uint32_t uPacketIndex,
                                        NvSciSyncFence *pPrefence)
{
    NvError error = NvError_Success;

    if (m_spVKSCEngine != nullptr) {
        error = m_spVKSCEngine->InsertFenceSciSyncPrefence(pPrefence);
        PCHK_ERROR_AND_RETURN(error, "m_spVKSCEngine->InsertFenceSciSyncPrefence");
    }

    return error;
}

NvError CVulkanSCModule::ProcessPayload(CClientCommon *pClient, uint32_t uPacketIndex)
{
    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler->RecordExecutionBeginTime();
    }

    if (m_spVKSCEngine == nullptr) {
        PLOG_ERR("VulkanSC engine is nullptr.");
        return NvError_InvalidState;
    }

    // get current time
    uint64_t frameCaptureTSC = static_cast<const MetaData *>(pClient->GetMetaPtr(uPacketIndex))->uFrameCaptureTSC;
    std::string frameCaptureTSCString = std::to_string(frameCaptureTSC);

    auto error = m_spVKSCEngine->Draw(uPacketIndex, frameCaptureTSCString);
    PCHK_ERROR_AND_RETURN(error, "m_spVKSCEngine->Draw");

    if (DumpEnabled()) {
        error = m_spVKSCEngine->DumpImage(uPacketIndex, m_upOutputBuf.get());
        PCHK_ERROR_AND_RETURN(error, "m_spVKSCEngine->DumpImage");
    }

    if (m_pAppCfg->IsProfilingEnabled()) {
        m_pMetaData = reinterpret_cast<MetaData *>(pClient->GetMetaPtr(uPacketIndex));
    }

    return NvError_Success;
}

NvError CVulkanSCModule::GetEofSyncFence(CClientCommon *pClient, NvSciSyncFence *pPostfence)
{
    if (m_spVKSCEngine != nullptr) {
        auto error = m_spVKSCEngine->GetEofFenceSciSyncFence(pPostfence);
        PCHK_ERROR_AND_RETURN(error, "m_spVKSCEngine->GetEofFenceSciSyncFence");
    }

    return NvError_Success;
}

NvError CVulkanSCModule::OnProcessPayloadDone(CClientCommon *pClient, uint32_t uPacketIndex)
{
    NvError error = NvError_Success;

    error = CBaseModule::OnProcessPayloadDone(pClient, uPacketIndex);
    PCHK_ERROR_AND_RETURN(error, "CBaseModule::OnProcessPayloadDone");

    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler->RecordExecutionEndTime();
        if (!m_bHasDownstream && m_pMetaData) {
            m_upProfiler->RecordPipelineTime(m_pMetaData->uFrameCaptureStartTSC);
            m_pMetaData = nullptr;
        }
    }

    return error;
}

const OptionTable *CVulkanSCModule::GetOptionTable() const
{
    return &m_vulkanSCOptionTable;
}

const void *CVulkanSCModule::GetOptionBaseAddress() const
{
    return &m_VulkanSCOption;
}

const std::string &CVulkanSCModule::GetOutputFileName()
{
    if (m_sOutputFileName != "") {
        PLOG_DBG("This CVulkanSCModule's OutputFileName already exists: %s\n", m_sOutputFileName.c_str());
        return m_sOutputFileName;
    }
    std::string suffix = m_spModuleCfg->m_sensorId != INVALID_ID ? std::to_string(m_spModuleCfg->m_sensorId)
                                                                 : std::to_string(m_spModuleCfg->m_moduleId);
    m_sOutputFileName = "multicast_vksc" + suffix + ".argb";
    PLOG_DBG("This CVulkanSCModule's OutputName: %s\n", m_sOutputFileName.c_str());

    return m_sOutputFileName;
}
