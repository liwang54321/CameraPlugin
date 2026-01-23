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

#include <algorithm>
#include <iomanip>
#include <inttypes.h>
#include "CBaseModule.hpp"
#include "CPassthroughConsumer.hpp"
#include "CPassthroughProducer.hpp"
#include "CFactory.hpp"

const std::unordered_map<std::string, Option> CBaseModule::m_baseModuleOptionTable = {
    { "latemods", { "late attach modules", offsetof(BaseModuleOption, sLateMods), OptionType::STRING } },
    { "elems", { "elements", offsetof(BaseModuleOption, sElems), OptionType::STRING } },
    { "passthrough", { "isPassthrough", offsetof(BaseModuleOption, bPassthrough), OptionType::BOOL } },
    { "filesink", { "isFileSink", offsetof(BaseModuleOption, bFileSink), OptionType::BOOL } },
    { "limit", { "limiter block capacity", offsetof(BaseModuleOption, uLimitNum), OptionType::UINT32 } },
    { "queueType",
      { "specify the queue block type.Acceptable values are f|fifo|m|mailbox.", offsetof(BaseModuleOption, sQueueType),
        OptionType::STRING } },
    { "dumpStartFrame",
      { "the start frame number for frame dumping", offsetof(BaseModuleOption, uDumpStartFrame), OptionType::UINT32 } },
    { "dumpEndFrame",
      { "the end frame number for frame dumping", offsetof(BaseModuleOption, uDumpEndFrame), OptionType::UINT32 } },
};

CBaseModule::CBaseModule(std::shared_ptr<CModuleCfg> spModuleCfg, IEventListener<CBaseModule> *pListener)
    : m_pAppCfg(spModuleCfg->m_pAppCfg)
    , m_spModuleCfg(spModuleCfg)
    , m_pEventListener(pListener)
{
}
CBaseModule::~CBaseModule() = default;

NvError CBaseModule::Init()
{
    PLOG_DBG("Enter: CBaseModule::Init(), isPassthrough: %u\n", m_baseModuleOption.bPassthrough);

    for (const auto &consumer : m_vspConsumers) {
        m_vspClients.push_back(consumer);
    }

    if (m_spProducer) {
        m_vspClients.push_back(m_spProducer);
    }

    if (m_baseModuleOption.bPassthrough) {
        if (!m_spProducer || m_vspConsumers.empty()) {
            PLOG_ERR("PassThrough, producer or consumer is null\n");
            return NvError_InvalidState;
        }
        CPassthroughProducer *pPassthroughProducer = static_cast<CPassthroughProducer *>(m_spProducer.get());
        CPassthroughConsumer *pPassthroughConsumer = static_cast<CPassthroughConsumer *>(m_vspConsumers[0].get());
        pPassthroughProducer->SetCallback(pPassthroughConsumer);
        pPassthroughConsumer->SetCallback(pPassthroughProducer);
    } else {
        // So far,there are two usages with CSyncAggregator,
        // one is a module has more clients, then need to aggregate
        // the sync attributes from those clients, like stiching
        // multiple camera videos in Nvm2dModule.
        // Another is multiple modules must share the same one sync object, then need to
        // create the shared CSyncAggregator in the specific module, like CWFDDisplayModule.
        if (m_spSyncAggregator) {
            for (uint32_t i = 0U; i < m_vspClients.size(); ++i) {
                m_spSyncAggregator->AddClient(m_vspClients[i].get());
            }
        } else if (m_vspClients.size() > 1) {
            m_spSyncAggregator = std::make_shared<CSyncAggregator>(m_vspClients);
        } else {
            // Do nothing for sync aggregator.
        }

        if (m_spProducer && (!m_vspConsumers.empty())) {
            m_upBufAggregator = std::make_unique<CBufAggregator>(this);
        }
    }

    for (auto &client : m_vspClients) {
        auto error = client->Init();
        PCHK_ERROR_AND_RETURN(error, "client->Init()");
    }

    if (m_baseModuleOption.bFileSink) {
        PLOG_DBG("FileSink create and Init()\n");
        m_upFileSink = std::make_unique<CDefaultFileSink>();
        CHK_PTR_AND_RETURN(m_upFileSink, "Create CDefaultFileSink")
        auto error = m_upFileSink->Init(GetOutputFileName());
        PCHK_ERROR_AND_RETURN(error, "m_upFileSink->Init()");
    }

    PLOG_DBG("Exit: CBaseModule::Init()\n");
    return NvError_Success;
}

void CBaseModule::DeInit()
{
    PLOG_DBG("Enter: CBaseModule::DeInit()\n");

    for (NvSciBufObj bufObj : m_vBufObjs) {
        UnregisterBufObj(bufObj);
    }

    for (NvSciSyncObj syncObj : m_vSyncObjs) {
        UnregisterSyncObj(syncObj);
    }

    for (auto &client : m_vspClients) {
        client->DeInit();
    }

    m_uOutputBufValidLen = 0;
    m_uOutputBufCapacity = 0;
    m_sOutputFileName.clear();

    m_pEventListener->OnEvent(this, EventStatus::QUITTED);

    PLOG_DBG("Exit: CBaseModule::DeInit()\n");
}

NvError CBaseModule::OnWaiterAttrEventRecvd(CClientCommon *pClient, bool &bHandled)
{
    if (m_spSyncAggregator != nullptr) {
        bHandled = true;
        return m_spSyncAggregator->OnWaiterAttrEventRecvd(pClient);
    }

    bHandled = false;
    return NvError_Success;
}

NvError CBaseModule::RegisterBufObj(CClientCommon *pClient,
                                    PacketElementType userType,
                                    uint32_t uPacketIndex,
                                    NvSciBufObj bufObj)
{
    m_vBufObjs.push_back(bufObj);

    if (m_upBufAggregator != nullptr) {
        auto error = m_upBufAggregator->RegisterBufObj(pClient, uPacketIndex, bufObj);
        PCHK_ERROR_AND_RETURN(error, "m_upBufAggregator->RegisterBufObj");
    }

    return NvError_Success;
}

NvError
CBaseModule::RegisterSignalSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj signalSyncObj)
{
    if (m_signalSyncObj == nullptr) {
        m_signalSyncObj = signalSyncObj;
        m_vSyncObjs.push_back(signalSyncObj);
    }

    return NvError_Success;
}

NvError
CBaseModule::RegisterWaiterSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj waiterSyncObj)
{
    m_vSyncObjs.push_back(waiterSyncObj);
    return NvError_Success;
}

NvError CBaseModule::Reconcile()
{
    PLOG_DBG("Enter: CBaseModule::Reconcile()\n");

    for (auto &client : m_vspClients) {
        auto error = client->Reconcile();
        PCHK_ERROR_AND_RETURN(error, (client->GetName() + " Reconcile.").c_str());
    }

    PLOG_DBG("Exit: CBaseModule::Reconcile()\n");
    return NvError_Success;
}

NvError CBaseModule::Start()
{
    PLOG_DBG("Enter: CBaseModule::Start()\n");

    for (auto it = m_vspClients.rbegin(); it != m_vspClients.rend(); ++it) {
        auto error = (*it)->Start();
        PCHK_ERROR_AND_RETURN(error, ((*it)->GetName() + " Start.").c_str());
    }

    m_pEventListener->OnEvent(this, EventStatus::STARTED);

    PLOG_DBG("Exit: CBaseModule::Start()\n");
    return NvError_Success;
}

NvError CBaseModule::PreStop()
{
    PLOG_DBG("Enter: CBaseModule::PreStop()\n");

    for (auto &client : m_vspClients) {
        auto error = client->PreStop();
        PCHK_ERROR_AND_RETURN(error, (client->GetName() + " PreStop.").c_str());
    }

    PLOG_DBG("Exit: CBaseModule::PreStop()\n");
    return NvError_Success;
}

NvError CBaseModule::Stop()
{
    PLOG_DBG("Enter: CBaseModule::Stop()\n");

    if (m_upBufAggregator != nullptr) {
        auto error = m_upBufAggregator->Stop();
        PCHK_ERROR_AND_RETURN(error, "m_upBufAggregator->Stop");
    }

    for (auto &client : m_vspClients) {
        auto error = client->Stop();
        PCHK_ERROR_AND_RETURN(error, (client->GetName() + " Stop.").c_str());
    }

    PLOG_DBG("Exit: CBaseModule::Stop()\n");
    return NvError_Success;
}

NvError CBaseModule::PostStop()
{
    PLOG_DBG("Enter: CBaseModule::PostStop()\n");

    for (auto &client : m_vspClients) {
        auto error = client->PostStop();
        PCHK_ERROR_AND_RETURN(error, (client->GetName() + " PostStop.").c_str());
    }

    m_pEventListener->OnEvent(this, EventStatus::STOPPED);

    PLOG_DBG("Exit: CBaseModule::PostStop()\n");
    return NvError_Success;
}

NvError CBaseModule::GetPerf(std::vector<Perf> &vPerf)
{
    if (m_pAppCfg->IsProfilingEnabled()) {
        /*
         * Get the perf from CProfiler instance
         */
        return m_upProfiler ? m_upProfiler->GetPerf(vPerf) : NvError_InvalidState;
    }
    return NvError_NotInitialized;
}

const std::vector<std::shared_ptr<CBaseModule>> &CBaseModule::GetDownstreamModules() const
{
    return m_vspDownstreamModules;
}

void CBaseModule::PrintFps()
{
    uint64_t uTimeElapsedMs = 0U;
    uint64_t uFrameCountDelta = 0U;
    {
        std::lock_guard<std::mutex> lk(m_FrameMutex);
        uFrameCountDelta = m_uFrameNum - m_uPrevFrameNum;
        if (uFrameCountDelta > 0) {
            /*
            * The time of (m_uPrevFrameNum, m_uFrameNum] range is used to calculate the frame rate.
            */
            --uFrameCountDelta;
        }
        m_uPrevFrameNum = m_uFrameNum;
        uTimeElapsedMs = std::chrono::duration<double, std::milli>(m_uEndTimePoint - m_uStarTimePoint).count();
        if (0U == uTimeElapsedMs) {
            uTimeElapsedMs = 1U;
        }
    }

    auto fps = uFrameCountDelta / std::max<size_t>(m_vspConsumers.size(), 1) / (uTimeElapsedMs / 1000.0);
    auto flags = std::cout.flags();
    std::cout << std::left << std::setw(32) << GetName() << " Frame rate (fps):\t" << std::fixed << std::setprecision(1)
              << fps << std::endl;
    std::cout.flags(flags);
}

std::shared_ptr<CProducer> CBaseModule::GetProducer()
{
    if (!m_spProducer) {
        std::string sProdName = m_baseModuleOption.bPassthrough ? "_PassthroughProducer" : "_Producer";
        std::string sClientName = GetName() + sProdName;
        const std::vector<ElementInfo> *pElementInfos = m_spModuleCfg->GetElementInfos(GetProdElemsName());
        auto clientCfg = m_spModuleCfg->CreateClientCfg(sClientName, m_numConsumers, m_baseModuleOption.uLimitNum,
                                                        pElementInfos, m_queueType);
        m_spProducer = m_baseModuleOption.bPassthrough ? std::make_shared<CPassthroughProducer>(clientCfg, this)
                                                       : std::make_shared<CProducer>(clientCfg, this);
    }

    return m_spProducer;
}

std::shared_ptr<CConsumer> CBaseModule::GetConsumer()
{
    std::string sConsName = m_baseModuleOption.bPassthrough ? "_PassthroughConsumer" : "_Consumer";
    std::string sClientName = GetName() + sConsName + std::to_string(m_vspConsumers.size());
    const std::vector<ElementInfo> *pElementInfos = m_spModuleCfg->GetElementInfos(m_baseModuleOption.sElems);
    auto clientCfg = m_spModuleCfg->CreateClientCfg(sClientName, m_numConsumers, m_baseModuleOption.uLimitNum,
                                                    pElementInfos, m_queueType);
    auto consumer = m_baseModuleOption.bPassthrough ? std::make_shared<CPassthroughConsumer>(clientCfg, this)
                                                    : std::make_shared<CConsumer>(clientCfg, this);
    m_vspConsumers.push_back(consumer);

    return consumer;
}

NvError CBaseModule::ConnectIpcDst(const std::shared_ptr<CIpcEndpointElem> &spIpcDst)
{
    NvError error = NvError_Success;
    //For stitching case, connect ipc dst for each sensor
    if (m_spModuleCfg->m_sensorId == INVALID_ID) {
        if (!spIpcDst->m_isStitching) {
            for (auto sensorId : spIpcDst->GetSensorIds()) {
                error = GetConsumer()->ConnectIpc(spIpcDst, sensorId);
                CHK_ERROR_AND_RETURN(error, "GetConsumer()->ConnectIpc");
            }
        } else {
            error = GetConsumer()->ConnectIpc(spIpcDst, m_spModuleCfg->m_sensorId);
            CHK_ERROR_AND_RETURN(error, "GetConsumer()->ConnectIpc");
        }
    } else {
        if (spIpcDst->m_isStitching) {
            LOG_ERR("Invalid pipeline.\n");
            return NvError_NotSupported;
        }
        error = GetConsumer()->ConnectIpc(spIpcDst, m_spModuleCfg->m_sensorId);
        CHK_ERROR_AND_RETURN(error, "GetConsumer()->ConnectIpc");
    }

    if (m_pAppCfg->IsProfilingEnabled()) {
        /*
         * Current module has upstream becuase it connects to a ipc dst.
         */
        m_bHasUpstream = true;
    }
    return NvError_Success;
}

NvError CBaseModule::ConnectIpcSrc(const std::shared_ptr<CIpcEndpointElem> &spIpcSrc)
{
    if ((m_spModuleCfg->m_sensorId == INVALID_ID && !spIpcSrc->m_isStitching) ||
        (m_spModuleCfg->m_sensorId != INVALID_ID && spIpcSrc->m_isStitching)) {
        LOG_ERR("Invalid pipeline.\n");
        return NvError_NotSupported;
    }

    auto error = GetProducer()->ConnectIpc(spIpcSrc, m_spModuleCfg->m_sensorId);
    CHK_ERROR_AND_RETURN(error, "GetProducer()->ConnectIpc");

    if (m_pAppCfg->IsProfilingEnabled()) {
        m_bHasDownstream = true;
    }

    return NvError_Success;
}

NvError CBaseModule::ConnectModule(const std::shared_ptr<CBaseModule> &spDownsteamModule)
{
    // Set nvm2dopaque element for downstream modules in pipeline
    if (GetProdElemsName() == "nvm2dopaque") {
        spDownsteamModule->m_baseModuleOption.sElems = "nvm2dopaque";
    }

    auto error = ClientConnect(GetProducer(), spDownsteamModule->GetConsumer());
    CHK_ERROR_AND_RETURN(error, "ClientConnect");
    if (m_pAppCfg->IsProfilingEnabled()) {
        /*
         * Remember the downstream module for module chain iteration.
         */
        m_vspDownstreamModules.emplace_back(spDownsteamModule);
        m_bHasDownstream = true;
        spDownsteamModule->m_bHasUpstream = true;
    }
    return NvError_Success;
}

void CBaseModule::OnEvent(CClientCommon *pClient, EventStatus event)
{
    PLOG_DBG("Received event: %s %s\n", pClient->GetName().c_str(), EventStatusToString(event));

    switch (event) {
        case EventStatus::CONNECTED:
            if (++m_uClientConnectCount == m_vspClients.size()) {
                m_pEventListener->OnEvent(this, EventStatus::CONNECTED);
            }
            break;
        case EventStatus::RECONCILED:
            if (++m_uClientSetupCompleteCount == m_vspClients.size()) {
                m_pEventListener->OnEvent(this, EventStatus::RECONCILED);
            }
            break;
        default:
            m_pEventListener->OnEvent(this, event);
    }
}

NvError CBaseModule::FillMetaBufAttrList(CClientCommon *pClient, NvSciBufAttrList *pBufAttrList)
{
    /* Meta buffer requires write access by CPU. */
    NvSciBufAttrValAccessPerm metaPerm = NvSciBufAccessPerm_ReadWrite;
    bool bMetaCpu = true;
    NvSciBufType metaBufType = NvSciBufType_RawBuffer;
    uint64_t uMetaSize = sizeof(MetaData);
    uint64_t uMetaAlign = 1U;
    NvSciBufAttrKeyValuePair metaKeyVals[] = { { NvSciBufGeneralAttrKey_Types, &metaBufType, sizeof(metaBufType) },
                                               { NvSciBufRawBufferAttrKey_Size, &uMetaSize, sizeof(uMetaSize) },
                                               { NvSciBufRawBufferAttrKey_Align, &uMetaAlign, sizeof(uMetaAlign) },
                                               { NvSciBufGeneralAttrKey_RequiredPerm, &metaPerm, sizeof(metaPerm) },
                                               { NvSciBufGeneralAttrKey_NeedCpuAccess, &bMetaCpu, sizeof(bMetaCpu) } };

    auto sciErr = NvSciBufAttrListSetAttrs(*pBufAttrList, metaKeyVals, ARRAY_SIZE(metaKeyVals));
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs(meta)");

    return NvError_Success;
}

void CBaseModule::CreateLateDstModules(std::vector<std::shared_ptr<CBaseModule>> &vspModules)
{
    for (auto modType : m_lateDstModuleTypeSet) {
        std::shared_ptr<CModuleCfg> spCfg = std::make_shared<CModuleCfg>();
        spCfg->m_moduleType = modType;
        auto spMod = CFactory::CreateModule(std::move(spCfg), nullptr);
        vspModules.push_back(spMod);
    }
}

NvError
CBaseModule::FillDataBufAttrList(CClientCommon *pClient, PacketElementType userType, NvSciBufAttrList *pBufAttrList)
{
    if (!m_lateDstModuleTypeSet.empty()) {
        std::vector<std::shared_ptr<CBaseModule>> vspModules;
        CreateLateDstModules(vspModules);

        std::vector<NvSciBufAttrList> attrs;
        std::vector<std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList>> attrPtrs;
        for (auto &spMod : vspModules) {
            std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> bufAttrPtr(new NvSciBufAttrList(nullptr));
            auto sciErr = NvSciBufAttrListCreate(m_pAppCfg->m_sciBufModule, bufAttrPtr.get());
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListCreate.");
            auto error = spMod->FillDataBufAttrList(pClient, userType, bufAttrPtr.get());
            PCHK_ERROR_AND_RETURN(error, "FillDataBufAttrList");
            attrs.push_back(*bufAttrPtr.get());
            attrPtrs.push_back(std::move(bufAttrPtr));
        }
        attrs.push_back(*pBufAttrList);
        std::unique_ptr<NvSciBufAttrList, CloseNvSciBufAttrList> bufAttrPtr(new NvSciBufAttrList(*pBufAttrList));
        attrPtrs.push_back(std::move(bufAttrPtr));
        auto sciErr = NvSciBufAttrListAppendUnreconciled(attrs.data(), attrs.size(), pBufAttrList);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListAppendUnreconciled.");
    }

    return NvError_Success;
}

NvError CBaseModule::FillSyncWaiterAttrList(CClientCommon *pClient,
                                            PacketElementType userType,
                                            NvSciSyncAttrList *pWaiterAttrList)
{
    if (!m_lateDstModuleTypeSet.empty()) {
        std::vector<std::shared_ptr<CBaseModule>> vspModules;
        CreateLateDstModules(vspModules);

        std::vector<NvSciSyncAttrList> attrs;
        std::vector<std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList>> attrPtrs;
        for (auto &spMod : vspModules) {
            std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> syncAttrPtr(new NvSciSyncAttrList(nullptr));
            auto sciErr = NvSciSyncAttrListCreate(m_pAppCfg->m_sciSyncModule, syncAttrPtr.get());
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListCreate.");
            auto error = spMod->FillSyncWaiterAttrList(pClient, userType, syncAttrPtr.get());
            PCHK_ERROR_AND_RETURN(error, "FillSyncWaiterAttrList");
            attrs.push_back(*syncAttrPtr.get());
            attrPtrs.push_back(std::move(syncAttrPtr));
        }
        attrs.push_back(*pWaiterAttrList);
        std::unique_ptr<NvSciSyncAttrList, CloseNvSciSyncAttrList> syncAttrPtr(new NvSciSyncAttrList(*pWaiterAttrList));
        attrPtrs.push_back(std::move(syncAttrPtr));
        auto sciErr = NvSciSyncAttrListAppendUnreconciled(attrs.data(), attrs.size(), pWaiterAttrList);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListAppendUnreconciled.");
    }

    return NvError_Success;
}

bool CBaseModule::DumpEnabled()
{
    bool bNeedToDump = false;

    std::lock_guard<std::mutex> lk(m_FrameMutex);
    bNeedToDump = m_baseModuleOption.bFileSink && (m_uFrameNum >= m_baseModuleOption.uDumpStartFrame &&
                                                   m_uFrameNum <= m_baseModuleOption.uDumpEndFrame);

    return bNeedToDump;
}

NvError CBaseModule::OnProcessPayloadDone(CClientCommon *pClient, uint32_t uPacketIndex)
{
    NvError error = NvError_Success;

    if (DumpEnabled()) {
        uint64_t uFrameNum = 0U;
        {
            std::lock_guard<std::mutex> lk(m_FrameMutex);
            uFrameNum = m_uFrameNum;
        }
        if (m_upOutputBuf && m_uOutputBufValidLen > 0) {
            PLOG_DBG("writing %" PRIu32 " bytes, uFrameNum %" PRIu64 "\n", m_uOutputBufValidLen, uFrameNum);
            error = m_upFileSink->WriteBufToFile(m_upOutputBuf.get(), m_uOutputBufValidLen);
        }
    }

    return error;
}

NvError CBaseModule::GetMinPacketCount(CClientCommon *pClient, uint32_t *pPacketCount)
{
    CHK_PTR_AND_RETURN_BADARG(pPacketCount, "GetMinPacketCount");
    *pPacketCount = 0U;
    return NvError_Success;
}

NvError CBaseModule::OnPacketGotten(CClientCommon *pClient, uint32_t uPacketIndex, bool *pHandled)
{
    if (m_pAppCfg->IsProfilingEnabled()) {
        /*
         * The transmission time of returning the packet is not accounted here.
         * The module without upstream doesn't have transmission time.
         */
        if (pClient->IsConsumer() && m_bHasUpstream) {
            MetaData *pMetaData = reinterpret_cast<MetaData *>(pClient->GetMetaPtr(uPacketIndex));
            if (pMetaData && m_upProfiler) {
                m_upProfiler->RecordTransmissionTime(pMetaData->uSendTSC, pMetaData->uReceiveTSC);
            }
        }
    }

    if (pClient->IsConsumer()) {
        UpdateFrameStatistics();
    }

    if (m_upBufAggregator != nullptr) {
        auto error = m_upBufAggregator->OnPacketGotten(pClient, uPacketIndex);
        PCHK_ERROR_AND_RETURN(error, "m_upBufAggregator->OnPacketGotten");
        if (pHandled) {
            *pHandled = true;
        }
    }

    return NvError_Success;
}

void CBaseModule::OnError(int moduleId, uint32_t uErrorId)
{
    // The errorId indicates which module reported an error
    PLOG_ERR("OnError moduleId %d, error %d\n", moduleId, uErrorId);
    m_pEventListener->OnError(this, moduleId, uErrorId);
}

void CBaseModule::OnCommand(uint32_t uCmdId, void *pParam) {}

NvError CBaseModule::ParseOptions()
{
    std::vector<OptionParserInfo> vOptionParserInfo;
    vOptionParserInfo.push_back({ &m_baseModuleOptionTable, &m_baseModuleOption });
    if (GetOptionTable() != nullptr && GetOptionBaseAddress() != nullptr) {
        vOptionParserInfo.push_back({ GetOptionTable(), GetOptionBaseAddress() });
    }

    auto error = COptionParser::ParseOptions(m_spModuleCfg->m_options, vOptionParserInfo);
    CHK_ERROR_AND_RETURN(error, "COptionParser::ParseOptions()");

    if (!m_baseModuleOption.sLateMods.empty()) {
        std::vector<std::string> vLateModStrs = splitString(std::string(m_baseModuleOption.sLateMods), '&');
        for (std::string &lateModStr : vLateModStrs) {
            auto it = pipelineElemName2TypeMap.find(lateModStr);
            if (it == pipelineElemName2TypeMap.end()) {
                LOG_MSG("CIpcEndpointElem::ParseOptions, element: %s is not supported!\n", lateModStr.c_str());
                return NvError_NotSupported;
            }
            m_lateDstModuleTypeSet.insert(static_cast<ModuleType>(it->second));
        }
    }

    if (m_baseModuleOption.bPassthrough) {
        m_spModuleCfg->m_cpuWaitCfg.bWaitPrefence = false;
        m_spModuleCfg->m_cpuWaitCfg.bWaitPostfence = false;
    }

    if (!m_baseModuleOption.sQueueType.empty()) {
        const char *pQueueType = m_baseModuleOption.sQueueType.c_str();
        if (!strcasecmp(pQueueType, "f") || !strcasecmp(pQueueType, "fifo")) {
            m_queueType = QueueType::Fifo;
        } else if (!strcasecmp(pQueueType, "m") || !strcasecmp(pQueueType, "mailbox")) {
            m_queueType = QueueType::Mailbox;
        } else {
            PLOG_ERR("Unknown queue type %s", pQueueType);
            return NvError_BadValue;
        }
    }
    return NvError_Success;
}

void CBaseModule::UpdateFrameStatistics()
{
    std::lock_guard<std::mutex> lk(m_FrameMutex);
    m_uEndTimePoint = std::chrono::steady_clock::now();
    if (m_uFrameNum++ == m_uPrevFrameNum) {
        m_uStarTimePoint = m_uEndTimePoint;
    }
}
