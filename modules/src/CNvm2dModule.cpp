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

#include <inttypes.h>
#include "CNvm2dModule.hpp"
#include "CElementDescription.hpp"

const std::unordered_map<std::string, Option> Nvm2DOptionTable = {
    { "type", { "Vic operation type", offsetof(Nvm2DOption, type), OptionType::INT } }
};

CElementDescription vicDescription{
    "Nvm2d", "Vic module for color format conversion/image buffer copy/image up(down) scale and composition",
    &CBaseModule::m_baseModuleOptionTable, &Nvm2DOptionTable
};

CNvm2dModule::CNvm2dModule(std::shared_ptr<CModuleCfg> spModuleCfg, IEventListener<CBaseModule> *pListener)
    : CBaseModule(spModuleCfg, pListener)
{
    //Use CPU wait for waiting prefence here as WAR for issue in Bug 4364352.
    //FIXME: will remove it once there is a final fix.
    spModuleCfg->m_cpuWaitCfg.bWaitPrefence = true;
}

CNvm2dModule::~CNvm2dModule()
{
    PLOG_DBG("release.\n");
}

NvError CNvm2dModule::Init()
{
    PLOG_DBG("Enter: CNvm2dModule::Init()\n");
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

    NvMedia2D *p2dHandle = nullptr;
    NvMediaStatus nvmStatus = NvMedia2DCreate(&p2dHandle, nullptr);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DCreate");

    m_up2DDevice.reset(p2dHandle);

    m_upVicConfigurator = std::make_unique<CRegionDivision>();

    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler->RecordInitEndTime();
    }

    PLOG_DBG("Exit: CNvm2dModule::Init()\n");
    return NvError_Success;
}

void CNvm2dModule::DeInit()
{
    CBaseModule::DeInit();

    m_up2DDevice.reset();
    m_upVicConfigurator.reset();
}

NvError
CNvm2dModule::FillDataBufAttrList(CClientCommon *pClient, PacketElementType userType, NvSciBufAttrList *pBufAttrList)
{
    PLOG_DBG("Enter FillDataBufAttrList()\n");

    if (pClient->IsConsumer()) {
        bool bImgCpuAccess = false;
        NvSciBufAttrValAccessPerm accessPerm = NvSciBufAccessPerm_ReadWrite;
        NvSciBufType bufType = NvSciBufType_Image;
        bool bIsEnableCpuCache = true;

        /* Set all key-value pairs */
        NvSciBufAttrKeyValuePair attributes[] = {
            { NvSciBufGeneralAttrKey_RequiredPerm, &accessPerm, sizeof(accessPerm) },
            { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
            { NvSciBufGeneralAttrKey_NeedCpuAccess, &bImgCpuAccess, sizeof(bImgCpuAccess) },
            { NvSciBufGeneralAttrKey_EnableCpuCache, &bIsEnableCpuCache, sizeof(bIsEnableCpuCache) }
        };

        auto sciErr = NvSciBufAttrListSetAttrs(*pBufAttrList, attributes, ARRAY_SIZE(attributes));
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

        std::unique_lock<std::mutex> lock(m_2dMutex);
        auto error = NvMedia2DFillNvSciBufAttrList(m_up2DDevice.get(), *pBufAttrList);
        PCHK_NVMSTATUS_AND_RETURN(error, "NvMedia2DFillNvSciBufAttrList");
    } else {
        bool bImgCpuAccess = false;
        NvSciBufType bufType = NvSciBufType_Image;
        NvSciBufAttrValAccessPerm accessPerm = NvSciBufAccessPerm_ReadWrite;
        NvSciBufAttrValImageScanType bufScanType = NvSciBufScan_ProgressiveType;

        NvSciBufAttrKeyValuePair keyVals[] = {
            { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
            { NvSciBufGeneralAttrKey_NeedCpuAccess, &bImgCpuAccess, sizeof(bool) },
            { NvSciBufGeneralAttrKey_RequiredPerm, &accessPerm, sizeof(accessPerm) },
            { NvSciBufImageAttrKey_ScanType, &bufScanType, sizeof(bufScanType) },
        };

        auto err = NvSciBufAttrListSetAttrs(*pBufAttrList, keyVals, ARRAY_SIZE(keyVals));
        CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufAttrListSetAttrs");

        std::unique_lock<std::mutex> lock(m_2dMutex);
        auto nvmStatus = NvMedia2DFillNvSciBufAttrList(m_up2DDevice.get(), *pBufAttrList);
        PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DFillNvSciBufAttrList");
    }

    return NvError_Success;
}

NvError CNvm2dModule::FillSyncSignalerAttrList(CClientCommon *pClient,
                                               PacketElementType userType,
                                               NvSciSyncAttrList *pSignalerAttrList)
{
    PLOG_DBG("Enter FillSyncSignalerAttrList()\n");

    std::unique_lock<std::mutex> lck(m_2dMutex);
    NvMediaStatus nvmStatus = NvMedia2DFillNvSciSyncAttrList(m_up2DDevice.get(), *pSignalerAttrList, NVMEDIA_SIGNALER);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DFillNvSciSyncAttrList signalerAttrList");

    PLOG_DBG("Exit FillSyncSignalerAttrList()\n");

    return NvError_Success;
}

NvError CNvm2dModule::FillSyncWaiterAttrList(CClientCommon *pClient,
                                             PacketElementType userType,
                                             NvSciSyncAttrList *pWaiterAttrList)
{
    PLOG_DBG("Enter FillSyncWaiterAttrList()\n");

    std::unique_lock<std::mutex> lck(m_2dMutex);
    auto nvmStatus = NvMedia2DFillNvSciSyncAttrList(m_up2DDevice.get(), *pWaiterAttrList, NVMEDIA_WAITER);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DFillNvSciSyncAttrList waiterAttrList");

    PLOG_DBG("Exit FillSyncWaiterAttrList()\n");

    return NvError_Success;
}

NvError
CNvm2dModule::RegisterSignalSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj signalSyncObj)
{
    std::unique_lock<std::mutex> lck(m_2dMutex);
    if (m_2DSignalSyncObj == nullptr) {
        auto error = CBaseModule::RegisterSignalSyncObj(pClient, userType, signalSyncObj);
        PCHK_ERROR_AND_RETURN(error, "CBaseModule::RegisterSignalSyncObj");

        m_2DSignalSyncObj = signalSyncObj;
        auto nvmStatus = NvMedia2DRegisterNvSciSyncObj(m_up2DDevice.get(), NVMEDIA_EOFSYNCOBJ, signalSyncObj);
        PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DRegisterNvSciSyncObj for EOF");
    }

    return NvError_Success;
}

NvError
CNvm2dModule::RegisterWaiterSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj waiterSyncObj)
{
    std::unique_lock<std::mutex> lck(m_2dMutex);

    auto error = CBaseModule::RegisterWaiterSyncObj(pClient, userType, waiterSyncObj);
    PCHK_ERROR_AND_RETURN(error, "CBaseModule::RegisterWaiterSyncObj");

    auto nvmStatus = NvMedia2DRegisterNvSciSyncObj(m_up2DDevice.get(), NVMEDIA_PRESYNCOBJ, waiterSyncObj);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DRegisterNvSciSyncObj for PRE");

    return NvError_Success;
}

NvError CNvm2dModule::RegisterBufObj(CClientCommon *pClient,
                                     PacketElementType userType,
                                     uint32_t uPacketIndex,
                                     NvSciBufObj bufObj)
{
    auto error = CBaseModule::RegisterBufObj(pClient, userType, uPacketIndex, bufObj);
    PCHK_ERROR_AND_RETURN(error, "CBaseModule::RegisterBufObj");

    std::unique_lock<std::mutex> lck(m_2dMutex);
    NvMediaStatus nvmStatus = NvMedia2DRegisterNvSciBufObj(m_up2DDevice.get(), bufObj);
    CHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DRegisterNvSciBufObj");

    return NvError_Success;
}

NvError CNvm2dModule::OnDataBufAttrListRecvd(CClientCommon *pClient, NvSciBufAttrList bufAttrList)
{
    uint16_t uWidth = 0U;
    uint16_t uHeight = 0U;

    if (pClient->IsProducer()) {
        auto error = GetWidthAndHeight(bufAttrList, uWidth, uHeight);
        PCHK_ERROR_AND_RETURN(error, "GetWidthAndHeight");

        LOG_MSG("VIC::ComputeInputRects, sensor count %zu, expected dst width = %" PRIu16 " dst height = %" PRIu16 "\n",
                m_vspConsumers.size(), uWidth, uHeight);

        m_upVicConfigurator->ComputeInputRects(m_vspConsumers.size(), uWidth, uHeight);
    }
    return NvError_Success;
}

//Before calling PreSync, m_nvmBuffers[uPacketIndex] should already be filled.
NvError CNvm2dModule::InsertPrefence(CClientCommon *pClient,
                                     PacketElementType userType,
                                     uint32_t uPacketIndex,
                                     NvSciSyncFence *pPrefence)
{
    std::unique_lock<std::mutex> lck(m_2dMutex);
    if (m_2DParams == 0U) {
        auto nvmStatus = NvMedia2DGetComposeParameters(m_up2DDevice.get(), &m_2DParams);
        PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DGetComposeParameters");
    }

    auto nvmStatus = NvMedia2DInsertPreNvSciSyncFence(m_up2DDevice.get(), m_2DParams, pPrefence);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DInsertPreNvSciSyncFence");

    return NvError_Success;
}

NvError CNvm2dModule::SetEofSyncObj(CClientCommon *pClient)
{
    std::unique_lock<std::mutex> lck(m_2dMutex);
    if (m_2DParams == 0U) {
        auto nvmStatus = NvMedia2DGetComposeParameters(m_up2DDevice.get(), &m_2DParams);
        PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DGetComposeParameters");
    }

    auto nvmStatus = NvMedia2DSetNvSciSyncObjforEOF(m_up2DDevice.get(), m_2DParams, m_2DSignalSyncObj);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DSetNvSciSyncObjforEOF");

    return NvError_Success;
}

NvError CNvm2dModule::UnregisterSyncObj(NvSciSyncObj syncObj)
{
    auto nvmStatus = NvMedia2DUnregisterNvSciSyncObj(m_up2DDevice.get(), syncObj);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DUnregisterNvSciSyncObj");

    return NvError_Success;
}

NvError CNvm2dModule::UnregisterBufObj(NvSciBufObj bufObj)
{
    auto nvmStatus = NvMedia2DUnregisterNvSciBufObj(m_up2DDevice.get(), bufObj);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DUnregisterNvSciBufObj");

    return NvError_Success;
}

NvError CNvm2dModule::ProcessPayload(CClientCommon *pClient, uint32_t uPacketIndex)
{
    return NvError_Success;
}

NvError CNvm2dModule::ProcessPayload(std::vector<NvSciBufObj> &vSrcBufObjs, NvSciBufObj dstBufObj, MetaData *pMetaData)
{
    PLOG_DBG("Enter ProcessPayload()\n");

    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler->RecordSubmissionBeginTime();
    }

    std::unique_lock<std::mutex> lck(m_2dMutex);
    if (m_2DParams == 0U) {
        PLOG_ERR("ProcessPayload, m_2DParams is uninitialized.\n");
        return NvError_InvalidState;
    }

    for (auto i = 0U; i < vSrcBufObjs.size(); ++i) {
        NvMediaRect dstRect;
        auto nvmStatus = NvMedia2DSetSrcNvSciBufObj(m_up2DDevice.get(), m_2DParams, i, vSrcBufObjs[i]);
        PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DSetSrcNvSciBufObj");
        auto error = m_upVicConfigurator->GetDstRect(i, &dstRect);
        PCHK_ERROR_AND_RETURN(error, "GetDstRect");
        nvmStatus =
            NvMedia2DSetSrcGeometry(m_up2DDevice.get(), m_2DParams, i, nullptr, &dstRect, NVMEDIA_2D_TRANSFORM_NONE);
        PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DSetSrcGeometry");
    }
    auto nvmStatus = NvMedia2DSetDstNvSciBufObj(m_up2DDevice.get(), m_2DParams, dstBufObj);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DSetDstNvSciBufObj");

    nvmStatus = NvMedia2DCompose(m_up2DDevice.get(), m_2DParams, &m_composeResult);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DCompose");

    m_2DParams = 0U;
    /* Get the end-of-frame fence for the compose operation */

    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler->RecordSubmissionEndTime();
        m_upProfiler->RecordExecutionBeginTime();
        m_pMetaData = pMetaData;
    }

    PLOG_DBG("Exit ProcessPayload()\n");
    return NvError_Success;
}

const OptionTable *CNvm2dModule::GetOptionTable() const
{
    return &Nvm2DOptionTable;
}

const void *CNvm2dModule::GetOptionBaseAddress() const
{
    return &m_nvm2DOption;
}

NvError CNvm2dModule::GetEofSyncFence(CClientCommon *pClient, NvSciSyncFence *pPostfence)
{
    auto nvmStatus = NvMedia2DGetEOFNvSciSyncFence(m_up2DDevice.get(), &m_composeResult, pPostfence);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMedia2DGetEOFNvSciSyncFence");

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