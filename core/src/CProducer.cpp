/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "CProducer.hpp"

CProducer::CProducer(std::shared_ptr<CClientCfg> spClientCfg, IModuleCallback *pCallback)
    : CClientCommon(spClientCfg, pCallback)
    , m_uNumConsumers(spClientCfg->m_numConsumers)
{
}

CProducer::~CProducer() = default;

bool CProducer::HasLateAttach()
{
    for (const auto &channelEntity : m_channelEntityMap) {
        if (channelEntity.second.bLateAttach) {
            return true;
        }
    }
    return false;
}

NvError CProducer::HandleSetupComplete()
{
    CClientCommon::HandleSetupComplete();

    NvSciStreamEventType eventType = NvSciStreamEventType_Error;
    NvSciStreamCookie cookie = NvSciStreamCookie_Invalid;
    uint32_t uPacketIndex = 0;

    // Producer receives notification and takes initial ownership of packets
    for (uint32_t i = 0U; i < m_uNumPacket; i++) {
        NvSciError sciErr = NvSciStreamBlockEventQuery(m_handle, QUERY_TIMEOUT, &eventType);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Get initial ownership of packet");

        if (eventType != NvSciStreamEventType_PacketReady) {
            PLOG_ERR("Didn't receive expected PacketReady event.\n");
            return NvError_BadParameter;
        }
        sciErr = NvSciStreamProducerPacketGet(m_handle, &cookie);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamProducerPacketGet");

        auto error = GetIndexFromCookie(cookie, uPacketIndex);
        PCHK_ERROR_AND_RETURN(error, "GetIndexFromCookie");

        std::lock_guard<std::mutex> lk(m_mutex);
        m_packetQueue.push_back(&m_packets[uPacketIndex]);
    }

    if (m_pModuleCallback) {
        auto error = m_pModuleCallback->GetMinPacketCount(this, &m_uMinPacket);
        PCHK_ERROR_AND_RETURN(error, "GetMinPacketCount");
    }

    if (HasLateAttach()) {
        NvSciError sciErr = NvSciStreamBlockEventQuery(m_multicastHandle, QUERY_TIMEOUT, &eventType);
        if (NvSciError_Success != sciErr || eventType != NvSciStreamEventType_SetupComplete) {
            PLOG_ERR("Query SetupComplete error: 0x%.8X, event: 0x%.8X\n", sciErr, eventType);
            return NvError_InvalidState;
        } else {
            PLOG_DBG("%s: Query SetupComplete succ.\n", __func__);
        }
    }

    std::lock_guard<std::mutex> lk(m_setupMutex);
    m_bSetupComplete = true;
    m_cvSetupComplete.notify_one();

    return NvError_Success;
}

NvError CProducer::HandlePayload()
{
    NvSciStreamCookie cookie = NvSciStreamCookie_Invalid;
    uint32_t uPacketIndex = 0;

    auto sciErr = NvSciStreamProducerPacketGet(m_handle, &cookie);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Obtain packet for payload");

    auto error = GetIndexFromCookie(cookie, uPacketIndex);
    PCHK_ERROR_AND_RETURN(error, "GetIndexFromCookie");

    ClientPacket *pPacket = GetPacketByCookie(cookie);
    PCHK_PTR_AND_RETURN(pPacket, "Get packet by cookie\n");

    /* Query fences for this element from each consumer */
    for (uint32_t i = 0U; i < m_uNumConsumers; ++i) {
        for (uint32_t j = 0U; j < m_vElemsInfos.size(); ++j) {
            /* If the received waiter obj if NULL,
             * the consumer is done using this element,
             * skip waiting on pre-fence.
             */
            if (nullptr == m_waiterSyncObjs[i][j]) {
                continue;
            }

            PLOG_DBG("Query fence from consumer: %d, m_interested element id = %d\n", i, j);
            std::unique_ptr<NvSciSyncFence, std::function<void(NvSciSyncFence *)>> spPrefence(
                new NvSciSyncFence(NvSciSyncFenceInitializer), [](NvSciSyncFence *pFence) {
                    NvSciSyncFenceClear(pFence);
                    delete pFence;
                });

            sciErr =
                NvSciStreamBlockPacketFenceGet(m_handle, pPacket->handle, i, m_vElemsInfos[j].uIndex, spPrefence.get());
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockPacketFenceGet");

            if (IsClearedFence(spPrefence.get())) {
                PLOG_DBG("Empty fence supplied as prefence.Skipping prefence insertion \n");
                OnClearedFenceReceived(uPacketIndex);
                continue;
            }

            if (m_cpuWaitCfg.bWaitPrefence || m_bStop) {
                sciErr = NvSciSyncFenceWait(spPrefence.get(), m_cpuWaitPreContext, FENCE_FRAME_TIMEOUT_US);
                PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncFenceWait prefence");
            } else {
                error = InsertPrefence(m_vElemsInfos[j].userType, uPacketIndex, spPrefence.get());
                PCHK_ERROR_AND_RETURN(error, "Insert prefence");
            }
        }
    }

    {
        std::lock_guard<std::mutex> lk(m_mutex);
        m_packetQueue.push_back(&m_packets[uPacketIndex]);
        if (m_bStop && m_packetQueue.size() == m_uNumPacket) {
            m_stopConditionVar.notify_all();
        }
    }

    if (!m_bStop) {
        if (m_pAppCfg->IsProfilingEnabled()) {
            /*
             * Record the receive TSC.
             */
            MetaData *pMetaData = reinterpret_cast<MetaData *>(GetMetaPtr(uPacketIndex));
            if (pMetaData != nullptr) {
                pMetaData->uReceiveTSC = CProfiler::GetCurrentTSC();
            }
        }
        m_pModuleCallback->OnPacketGotten(this, uPacketIndex);
    }

    return NvError_Success;
}

NvError CProducer::InsertPrefence(PacketElementType userType, uint32_t uPacketIndex, NvSciSyncFence *pPrefence)
{
    return m_pModuleCallback->InsertPrefence(this, userType, uPacketIndex, pPrefence);
}

NvError CProducer::Post(void *pBuffer, NvSciSyncFence *pPostFence)
{
    NvSciError sciErr = NvSciError_Success;
    uint32_t uPacketIndex = 0;
    uint32_t uElementId = 0U;

    auto error = MapPayload(pBuffer, uPacketIndex, uElementId);
    PCHK_ERROR_AND_RETURN(error, "MapPayload");

    if (pPostFence) {
        if (m_cpuWaitCfg.bWaitPostfence || m_bStop) {
            sciErr = NvSciSyncFenceWait(pPostFence, m_cpuWaitPreContext, FENCE_FRAME_TIMEOUT_US);
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncFenceWait postfence");
        } else {
            sciErr = NvSciStreamBlockPacketFenceSet(m_handle, m_packets[uPacketIndex].handle,
                                                    m_vElemsInfos[uElementId].uIndex, pPostFence);
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Post, NvSciStreamBlockPacketFenceSet");
        }
    }

    if (m_bStop) {
        m_pModuleCallback->OnPacketGotten(this, uPacketIndex);
        return NvError_Success;
    }

    if (m_pAppCfg->IsProfilingEnabled()) {
        /*
         * Record the send TSC.
         */
        MetaData *pMetaData = reinterpret_cast<MetaData *>(GetMetaPtr(uPacketIndex));
        if (pMetaData != nullptr) {
            pMetaData->uSendTSC = CProfiler::GetCurrentTSC();
        }
    }

    sciErr = NvSciStreamProducerPacketPresent(m_handle, m_packets[uPacketIndex].handle);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamProducerPacketPresent");

    std::lock_guard<std::mutex> lk(m_mutex);
    auto it = std::find(m_packetQueue.begin(), m_packetQueue.end(), &m_packets[uPacketIndex]);
    if (it != m_packetQueue.end()) {
        m_packetQueue.erase(it);
    }

    return NvError_Success;
}

NvError CProducer::MultiPost(MultiPostInfo *pPostInfo)
{
    if (pPostInfo == nullptr) {
        return NvError_BadParameter;
    }

    uint32_t uPacketIndex = pPostInfo->uPacketIndex;
    if (uPacketIndex >= MAX_NUM_PACKETS) {
        return NvError_BadParameter;
    }

    for (size_t i = 0; i < pPostInfo->vPostFences.size(); i++) {
        uint32_t uElementIndex = pPostInfo->vElementIndexs[i];
        NvSciSyncFence postFence = pPostInfo->vPostFences[i];
        if (m_cpuWaitCfg.bWaitPostfence || m_bStop) {
            auto sciErr = NvSciSyncFenceWait(&postFence, m_cpuWaitPostContext, FENCE_FRAME_TIMEOUT_US);
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncFenceWait postfence");
        } else {
            /* Update postfence for this element */
            auto sciErr =
                NvSciStreamBlockPacketFenceSet(m_handle, m_packets[uPacketIndex].handle, uElementIndex, &postFence);
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, "MultiPost, NvSciStreamBlockPacketFenceSet");
        }
        NvSciSyncFenceClear(&postFence);
    }

    if (m_bStop) {
        m_pModuleCallback->OnPacketGotten(this, uPacketIndex);
        return NvError_Success;
    }

    if (m_pAppCfg->IsProfilingEnabled()) {
        /*
         * Record the send TSC.
         */
        MetaData *pMetaData = reinterpret_cast<MetaData *>(GetMetaPtr(uPacketIndex));
        if (pMetaData != nullptr) {
            pMetaData->uSendTSC = CProfiler::GetCurrentTSC();
        }
    }

    NvSciError sciErr = NvSciStreamProducerPacketPresent(m_handle, m_packets[uPacketIndex].handle);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "MultiPost, NvSciStreamProducerPacketPresent");

    std::lock_guard<std::mutex> lk(m_mutex);
    auto it = std::find(m_packetQueue.begin(), m_packetQueue.end(), &m_packets[uPacketIndex]);
    if (it != m_packetQueue.end()) {
        m_packetQueue.erase(it);
    }

    return NvError_Success;
}

NvError CProducer::FillBufAttrList(PacketElementType userType, NvSciBufAttrList *pBufAttrList)
{
    bool isC2cLate = false;
    for (const auto &channelEntity : m_channelEntityMap) {
        if (channelEntity.second.bLateAttach && channelEntity.second.bC2C) {
            isC2cLate = true;
            break;
        }
    }
    if (isC2cLate) {
        NvSciBufPeerLocationInfo peerInfo = { 0 };
        peerInfo.vmID = 0;
        peerInfo.reserved = 0;
        //Hard-coded temporarily, the value should come from /proc/device-tree/soc_id
        peerInfo.socID = 2;
        NvSciBufAttrKeyValuePair bufAttrs[] = {
            { NvSciBufGeneralAttrKey_PeerLocationInfo, &peerInfo, sizeof(peerInfo) },
        };
        NvSciError sciErr =
            NvSciBufAttrListSetAttrs(*pBufAttrList, bufAttrs, sizeof(bufAttrs) / sizeof(NvSciBufAttrKeyValuePair));
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");
    }

    auto error = CClientCommon::FillBufAttrList(userType, pBufAttrList);
    PCHK_ERROR_AND_RETURN(error, "Producer CClientCommon::FillBufAttrList");
    return NvError_Success;
}

NvError CProducer::ConnectIpc(const std::string &sChannelName, IpcEntity &ipcEntity)
{
    NvSciError sciErr = NvSciError_Success;
    NvError error = NvError_Success;

    NvSciStreamBlock connectBlock;

    m_eCommType = CommType::InterProcess;
    error = GetConnectBlock(&connectBlock);
    CHK_ERROR_AND_RETURN(error, "srcClient->GetConnectBlock");

    if (ipcEntity.uLimitNum != 0) {
        sciErr = NvSciStreamLimiterCreate(ipcEntity.uLimitNum, &ipcEntity.limiterBlock);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamLimiterCreate");
        sciErr = NvSciStreamBlockConnect(connectBlock, ipcEntity.limiterBlock);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockConnect");
        connectBlock = ipcEntity.limiterBlock;
    }

    LOG_MSG(GetName() + " NvSciIpcOpenEndpoint: " + sChannelName + "\n");
    sciErr = NvSciIpcOpenEndpoint(sChannelName.c_str(), &ipcEntity.endPoint);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciIpcOpenEndpoint ");

    NvSciIpcResetEndpointSafe(ipcEntity.endPoint);

    sciErr = NvSciStreamIpcSrcCreate2(ipcEntity.endPoint, m_pAppCfg->m_sciSyncModule, m_pAppCfg->m_sciBufModule,
                                      static_cast<NvSciStreamBlock>(0), &ipcEntity.ipcBlock);
    if (sciErr != NvSciError_Success) {
        NvSciIpcCloseEndpointSafe(ipcEntity.endPoint, false);
        LOG_ERR("Create ipc block failed, error: 0x%x\n", sciErr);
        return NvError_InvalidState;
    }

    sciErr = NvSciStreamBlockConnect(connectBlock, ipcEntity.ipcBlock);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockConnect");

    return NvError_Success;
}

NvError
CProducer::ConnectC2c(const std::string &sChannelName, IpcEntity &ipcEntity, std::vector<NvSciStreamBlock> &vBlocks)
{
    NvSciStreamBlock queueBlock = 0U;
    NvSciStreamBlock presentSyncBlock = 0U;
    NvSciStreamBlock connectBlock = 0U;

    m_eCommType = CommType::InterChip;
    auto sciErr = m_queueType == QueueType::Mailbox ? NvSciStreamMailboxQueueCreate(&queueBlock)
                                                    : NvSciStreamFifoQueueCreate(&queueBlock);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamQueueCreate");

    auto error = GetConnectBlock(&connectBlock);
    CHK_ERROR_AND_RETURN(error, "srcClient->GetConnectBlock");

    if (ipcEntity.uLimitNum != 0) {
        sciErr = NvSciStreamLimiterCreate(ipcEntity.uLimitNum, &ipcEntity.limiterBlock);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamLimiterCreate");
        sciErr = NvSciStreamBlockConnect(connectBlock, ipcEntity.limiterBlock);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockConnect");
        connectBlock = ipcEntity.limiterBlock;
    }

    LOG_MSG(GetName() + " NvSciIpcOpenEndpoint: " + sChannelName + "\n");
    sciErr = NvSciIpcOpenEndpoint(sChannelName.c_str(), &ipcEntity.endPoint);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciIpcOpenEndpoint");

    NvSciIpcResetEndpointSafe(ipcEntity.endPoint);

    sciErr = NvSciStreamIpcSrcCreate2(ipcEntity.endPoint, m_pAppCfg->m_sciSyncModule, m_pAppCfg->m_sciBufModule,
                                      queueBlock, &ipcEntity.ipcBlock);
    if (sciErr != NvSciError_Success) {
        NvSciIpcCloseEndpointSafe(ipcEntity.endPoint, false);
        LOG_ERR("Create ipc block failed, error: 0x%x\n", sciErr);
        return NvError_BadParameter;
    }

    sciErr = NvSciStreamPresentSyncCreate(m_pAppCfg->m_sciSyncModule, &presentSyncBlock);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamPresentSyncCreate");

    sciErr = NvSciStreamBlockConnect(connectBlock, presentSyncBlock);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockConnect presentSyncBlock");

    sciErr = NvSciStreamBlockConnect(presentSyncBlock, ipcEntity.ipcBlock);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockConnect ipcSrcBlock");

    vBlocks.push_back(queueBlock);
    vBlocks.push_back(presentSyncBlock);

    return NvError_Success;
}

NvError CProducer::AttachLateIpc(const std::string &sChannelName)
{
    std::unique_lock<std::mutex> setupLock(m_setupMutex);
    m_cvSetupComplete.wait(setupLock, [this] { return m_bSetupComplete.load(); });

    LOG_MSG("Begin to attach late consumer: %s\n", sChannelName.c_str());
    ChannelEntity &channelEntity = m_channelEntityMap[sChannelName];
    std::lock_guard<std::mutex> lk(channelEntity.entityMutex);
    bool bC2C = channelEntity.bC2C;
    NvSciStreamEventType event = NvSciStreamEventType_Error;
    if (!bC2C) {
        auto error = ConnectIpc(sChannelName, channelEntity.ipcEntity);
        PCHK_ERROR_AND_RETURN(error, "ConnectIpc");
    } else {
        auto error = ConnectC2c(sChannelName, channelEntity.ipcEntity, channelEntity.vC2CBlockHandles);
        CHK_ERROR_AND_RETURN(error, "ConnectC2C");
    }
    auto sciErr = NvSciStreamBlockSetupStatusSet(m_multicastHandle, NvSciStreamSetup_Connect, true);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Set NvSciStreamSetup_Connect!");

    sciErr = NvSciStreamBlockEventQuery(channelEntity.ipcEntity.ipcBlock, QUERY_TIMEOUT, &event);
    PCHK_NVSCICONNECT_AND_RETURN(sciErr, event, "query block");

    // make sure late consumers attach success
    sciErr = NvSciStreamBlockEventQuery(m_multicastHandle, QUERY_TIMEOUT_FOREVER, &event);
    if (NvSciError_Success != sciErr || event != NvSciStreamEventType_SetupComplete) {
        PLOG_ERR("Multicast block setup error: 0x%.8X .\n", sciErr);

        // release relevant resource if attach fail
        return NvError_InvalidState;
    }

    // late consumer connected, set connected flag and begin to monitor heartbeat
    channelEntity.bConnected = true;
    channelEntity.lastHeartBeatTime = std::chrono::steady_clock::now();
    return NvError_Success;
}

void CProducer::DetachIpc(const std::string &sChannelName)
{
    LOG_MSG("Begin to detach consumer: %s\n", sChannelName.c_str());
    ChannelEntity &channelEntity = m_channelEntityMap[sChannelName];
    std::lock_guard<std::mutex> lk(channelEntity.entityMutex);
    NvSciStreamBlockDisconnect(channelEntity.ipcEntity.ipcBlock);
    NvSciStreamBlockDelete(channelEntity.ipcEntity.ipcBlock);
    NvSciIpcCloseEndpointSafe(channelEntity.ipcEntity.endPoint, false);
    channelEntity.ipcEntity.ipcBlock = 0U;
    channelEntity.ipcEntity.endPoint = 0U;
    if (channelEntity.ipcEntity.limiterBlock) {
        NvSciStreamBlockDelete(channelEntity.ipcEntity.limiterBlock);
        channelEntity.ipcEntity.limiterBlock = 0U;
    }
    if (channelEntity.bC2C) {
        for (auto &handle : channelEntity.vC2CBlockHandles) {
            NvSciStreamBlockDelete(handle);
        }
    }
    channelEntity.vC2CBlockHandles.clear();
    channelEntity.bIpcSetup = false;
    channelEntity.bLateAttach = true;
    channelEntity.bConnected = false;
}

NvError CProducer::ConnectIpc(const std::shared_ptr<CIpcEndpointElem> &spIpcSrc, int sensorId)
{
    std::string sChannelName = spIpcSrc->GetChannelStr(sensorId);
    if (sChannelName.empty()) {
        LOG_ERR("GetChannelStr failed, sensorId: %d\n", sensorId);
        return NvError_BadValue;
    }

    bool bLateAttach = spIpcSrc->IsLateAttach();
    ChannelEntity &channelEntity = m_channelEntityMap[sChannelName];
    std::unique_lock<std::mutex> lk(channelEntity.entityMutex);
    channelEntity.bLateAttach = bLateAttach;
    channelEntity.bC2C = spIpcSrc->IsC2C();
    channelEntity.ipcEntity.uLimitNum = spIpcSrc->GetLimitNum();

    if (!bLateAttach) {
        if (!spIpcSrc->IsC2C()) {
            auto error = ConnectIpc(sChannelName, channelEntity.ipcEntity);
            CHK_ERROR_AND_RETURN(error, "ConnectIpc");
        } else {
            auto error = ConnectC2c(sChannelName, channelEntity.ipcEntity, channelEntity.vC2CBlockHandles);
            CHK_ERROR_AND_RETURN(error, "ConnectC2C");
        }

        m_vBlockPairs.emplace_back(false, channelEntity.ipcEntity.ipcBlock);
        for (auto block : channelEntity.vC2CBlockHandles) {
            m_vBlockPairs.emplace_back(false, block);
        }
        channelEntity.bIpcSetup = true;
    } else {
        channelEntity.bIpcSetup = false;
    }
    lk.unlock();

    if (m_spControlChannel) {
        std::shared_ptr<CMsgReader> spMsgReader =
            m_spControlChannel->CreateReader(sChannelName, [this, sChannelName](void *pContent, uint32_t size) {
                this->HandleConsumerMsg(sChannelName, pContent, size);
            });
    }
    return NvError_Success;
}

void CProducer::HandleConsumerMsg(const std::string &sChannelName, void *pContent, uint32_t size)
{
    if (size != sizeof(HeartBeatMsgType)) {
        PLOG_ERR("Unexpected msg size (%" PRIu32 "). Expected size is: %zu", size, sizeof(HeartBeatMsgType));
        return;
    }
    HeartBeatMsgType msgType = *reinterpret_cast<HeartBeatMsgType *>(pContent);
    switch (msgType) {
        case HeartBeatMsgType::CONNECT_REQUEST:
            if (m_channelEntityMap[sChannelName].bLateAttach) {
                PLOG_INFO("Received late consumer connect request, begin to attach channel: %s\n",
                          sChannelName.c_str());
                auto error = AttachLateIpc(sChannelName);
                if (NvError_Success != error) {
                    PLOG_WARN("Failed to attach late consumer. Ipc endpoint: %s\n", sChannelName.c_str());
                    return;
                }
            }
            break;
        case HeartBeatMsgType::HEARTBEAT:
            m_channelEntityMap[sChannelName].lastHeartBeatTime = std::chrono::steady_clock::now();
            break;
        case HeartBeatMsgType::DISCONNECT_REQUEST:
            PLOG_INFO("Received disconnect request, begin to detach channel: %s\n", sChannelName.c_str());
            DetachIpc(sChannelName);
            break;
        default:
            break;
    }
}

void CProducer::MonitorHeartBeat()
{
    auto HeartBeatCheck = [this](const std::string &sChannelName, const ChannelEntity &channelEntity) {
        auto curTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            curTime.time_since_epoch() - channelEntity.lastHeartBeatTime.time_since_epoch());
        if (duration.count() > kHeartBeatTimeoutMs) {
            LOG_MSG("channel: %s lost heartbeat for %ldms, begin to detach!\n", sChannelName.c_str(), duration.count());
            this->DetachIpc(sChannelName);
        }
    };

    for (const auto &channelEntity : m_channelEntityMap) {
        // skip unconnected ipc channels
        if (channelEntity.second.bConnected) {
            HeartBeatCheck(channelEntity.first, channelEntity.second);
        }
    }
}

std::unique_ptr<CPoolManager> CProducer::CreatePoolManager()
{
    return std::make_unique<CPoolManager>(m_poolHandle, GetName(), MAX_NUM_PACKETS, false);
}

NvError CProducer::GetConnectBlock(NvSciStreamBlock *pBlock)
{
    PLOG_DBG("Enter: CProducer::GetConnectBlock\n");

    if (!m_handle) {
        auto sciErr = NvSciStreamStaticPoolCreate(MAX_NUM_PACKETS, &m_poolHandle);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamStaticPoolCreate");
        m_upPoolManger = CreatePoolManager();

        sciErr = NvSciStreamProducerCreate2(m_poolHandle, false, &m_handle);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamProducerCreate2");
        LOG_INFO("%s: m_uNumConsumers: %u \n", __func__, m_uNumConsumers);

        if (m_eCommType != CommType::IntraProcess || HasLateAttach()) {
            auto error = m_pAppCfg->GetPeerValidator()->SendValidationInfo(m_handle);
            LOG_DBG(GetName() + " SendValidationInfo, error = %d.\n", error);
        }

        sciErr = NvSciStreamMulticastCreate(m_uNumConsumers, &m_multicastHandle);

        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamMulticastCreate");

        sciErr = NvSciStreamBlockConnect(m_handle, m_multicastHandle);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Connect producer to multicast");

        m_vBlockPairs.emplace_back(false, m_poolHandle);
        m_vBlockPairs.emplace_back(false, m_handle);
        m_vBlockPairs.emplace_back(false, m_multicastHandle);
    }

    *pBlock = m_multicastHandle;

    PLOG_DBG("Exit: CProducer::GetConnectBlock\n");

    return NvError_Success;
}

EventStatus CProducer::HandlePoolEvent()
{
    EventStatus status = m_upPoolManger->HandleEvent();
    if (status == EventStatus::ERROR) {
        m_pModuleCallback->OnEvent(this, EventStatus::ERROR);
    }
    return status;
}

NvError CProducer::Init()
{
    if (HasLateAttach()) {
        auto sciErr = NvSciStreamBlockSetupStatusSet(m_multicastHandle, NvSciStreamSetup_Connect, true);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Set NvSciStreamSetup_Connect!");
    }
    return CClientCommon::Init();
}

void CProducer::DeInit()
{
    CClientCommon::DeInit();

    if (m_upPoolEventHandler.get()) {
        m_upPoolEventHandler->QuitThread();
    }

    for (auto &channelEntity : m_channelEntityMap) {
        std::lock_guard<std::mutex> lk(channelEntity.second.entityMutex);
        if (channelEntity.second.bIpcSetup) {
            IpcEntity &ipcEntity = channelEntity.second.ipcEntity;
            NvSciStreamBlockDisconnect(ipcEntity.ipcBlock);
            NvSciStreamBlockDelete(ipcEntity.ipcBlock);
            NvSciIpcCloseEndpointSafe(ipcEntity.endPoint, false);
            if (ipcEntity.limiterBlock) {
                NvSciStreamBlockDelete(ipcEntity.limiterBlock);
            }
            for (auto &handle : channelEntity.second.vC2CBlockHandles) {
                NvSciStreamBlockDelete(handle);
            }
            channelEntity.second.vC2CBlockHandles.clear();
        }
    }

    if (m_multicastHandle) {
        NvSciStreamBlockDelete(m_multicastHandle);
        m_multicastHandle = 0;
    }

    if (m_poolHandle) {
        NvSciStreamBlockDelete(m_poolHandle);
        m_poolHandle = 0;
    }

    if (m_handle) {
        NvSciStreamBlockDelete(m_handle);
        m_handle = 0;
    }
}

NvError CProducer::Reconcile()
{
    auto error = CClientCommon::Reconcile();
    PCHK_ERROR_AND_RETURN(error, "CClientCommon::Reconcile");

    m_upPoolEventHandler = std::make_unique<CEventHandler<CProducer>>();
    error = m_upPoolEventHandler->RegisterHandler(&CProducer::HandlePoolEvent, this);
    PCHK_ERROR_AND_RETURN(error, "RegisterHandler");
    error = m_upPoolEventHandler->StartThread();
    PCHK_ERROR_AND_RETURN(error, "StartThread");

    return NvError_Success;
}

NvError CProducer::Start()
{
    if (m_upPoolEventHandler.get()) {
        m_upPoolEventHandler->QuitThread();
    }

    return CClientCommon::Start();
}

NvError CProducer::Stop()
{
    PLOG_DBG("Enter: CProducer::Stop()\n");

    {
        std::unique_lock<std::mutex> lk(m_mutex);
        if (m_packetQueue.size() != m_uNumPacket) {
            if (m_stopConditionVar.wait_for(lk, std::chrono::seconds(MAX_NUM_PACKETS)) == std::cv_status::timeout) {
                PLOG_ERR("CProducer::Stop wait packet timeout!\n");
            };
        }
    }

    PLOG_DBG("Exit: CProducer::Stop()\n");
    return CClientCommon::Stop();
}

NvError CProducer::OnConnected()
{
    if (m_eCommType != CommType::IntraProcess || HasLateAttach()) {
        // begin to monitor consumers' heartbeat
        for (auto &channelEnity : m_channelEntityMap) {
            if (!channelEnity.second.bLateAttach) {
                channelEnity.second.bConnected = true;
                channelEnity.second.lastHeartBeatTime = std::chrono::steady_clock::now();
            }
        }
        if (m_spTimer != nullptr) {
            m_spTimer->AddTask([this]() { this->MonitorHeartBeat(); }, kHeartBeatIntervalMs);
        }
    }
    return NvError_Success;
}
