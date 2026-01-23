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

#include "CVirtualDstModule.hpp"
#include "CElementDescription.hpp"

CElementDescription virtualDstDescription{ "VirtualDst",
                                           "Virtual destination module for receiving the input buffers.(Test Only)",
                                           &CBaseModule::m_baseModuleOptionTable, nullptr };

const std::unordered_map<std::string, Option> CVirtualDstModule::m_virtualDstOptionTable = {
    { "width", { "the width of the input buffer", offsetof(VirtualDstInputInfo, uWidth), OptionType::UINT32 } },
    { "height", { "the height of the input buffer", offsetof(VirtualDstInputInfo, uHeight), OptionType::UINT32 } },
    { "colortype", { "colortype", offsetof(VirtualDstInputInfo, sColorType), OptionType::STRING } },
    { "imagelayout", { "imagelayout", offsetof(VirtualDstInputInfo, sImageLayout), OptionType::STRING } },
};

const std::unordered_set<std::string> CVirtualDstModule::m_virtualDstSupportedColor = { "NV12", "ARGB", "ABGR" };

CVirtualDstModule::CVirtualDstModule(std::shared_ptr<CModuleCfg> spModuleCfg, IEventListener<CBaseModule> *pListener)
    : CBaseModule(spModuleCfg, pListener)
{
    spModuleCfg->m_cpuWaitCfg = { true, false };
}

CVirtualDstModule::~CVirtualDstModule()
{
    PLOG_DBG("release.\n");
}

NvError CVirtualDstModule::Init()
{
    PLOG_DBG("Enter: CVirtualDstModule::Init()\n");

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

    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler->RecordInitEndTime();
    }

    PLOG_DBG("Exit: CVirtualDstModule::Init()\n");

    return NvError_Success;
}

NvError CVirtualDstModule::FillDataBufAttrList(CClientCommon *pClient,
                                               PacketElementType userType,
                                               NvSciBufAttrList *pBufAttrList)
{
    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_Readonly;
    bool bCpuAccessFlag = true;

    NvSciBufAttrKeyValuePair bufAttrs[] = {
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &bCpuAccessFlag, sizeof(bCpuAccessFlag) },
    };

    auto sciErr = NvSciBufAttrListSetAttrs(*pBufAttrList, bufAttrs, ARRAY_SIZE(bufAttrs));
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

    // For opaque element type, color type and memory layout are decided by options.
    if (userType == PacketElementType::OPAQUE) {
        std::string &sImageLayout = m_virtualDstInputInfo.sImageLayout;
        std::string &sColorType = m_virtualDstInputInfo.sColorType;

        // Check supported imagelayout.
        if (!sImageLayout.empty() && sImageLayout != "BL" && sImageLayout != "PL") {
            PLOG_ERR("Image layout type specified not support by VirtualDst.\n"
                     "Supported Image layout type option: bl(default), pl.\n");
            return NvError_BadValue;
        }

        // Check supported color type attributes.
        if (!sColorType.empty() && m_virtualDstSupportedColor.find(sColorType) == m_virtualDstSupportedColor.end()) {
            PLOG_ERR("Color type specified not support by VirtualDst.\n"
                     "Supported color type: nv12(default), argb, abgr.\n");
            return NvError_BadValue;
        }

        auto error = SetBufAttr(pBufAttrList, sColorType, sImageLayout, m_virtualDstInputInfo.uWidth,
                                m_virtualDstInputInfo.uHeight);
        PCHK_ERROR_AND_RETURN(error, "SetBufAttr");
    }

    return NvError_Success;
}

NvError CVirtualDstModule::FillSyncSignalerAttrList(CClientCommon *pClient,
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

NvError CVirtualDstModule::FillSyncWaiterAttrList(CClientCommon *pClient,
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

    return NvError_Success;
}

NvError
CVirtualDstModule::RegisterSignalSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj signalSyncObj)
{
    return NvError_Success;
}

NvError
CVirtualDstModule::RegisterWaiterSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj waiterSyncObj)
{
    return NvError_Success;
}

//Before calling PreSync, m_nvmBuffers[uPacketIndex] should already be filled.
NvError CVirtualDstModule::InsertPrefence(CClientCommon *pClient,
                                          PacketElementType userType,
                                          uint32_t uPacketIndex,
                                          NvSciSyncFence *pPrefence)
{
    return NvError_Success;
}

NvError CVirtualDstModule::ProcessPayload(CClientCommon *pClient, uint32_t uPacketIndex)
{
    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler->RecordExecutionTime();
        m_pMetaData = reinterpret_cast<MetaData *>(pClient->GetMetaPtr(uPacketIndex));
        if (!m_bHasDownstream && m_pMetaData) {
            m_upProfiler->RecordPipelineTime(m_pMetaData->uFrameCaptureStartTSC);
            m_pMetaData = nullptr;
        }
    }
    return NvError_Success;
}

NvError CVirtualDstModule::RegisterBufObj(CClientCommon *pClient,
                                          PacketElementType userType,
                                          uint32_t uPacketIndex,
                                          NvSciBufObj bufObj)
{
    return NvError_Success;
}

const OptionTable *CVirtualDstModule::GetOptionTable() const
{
    return &m_virtualDstOptionTable;
}

const void *CVirtualDstModule::GetOptionBaseAddress() const
{
    return &m_virtualDstInputInfo;
}
