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

#include "CConsumer.hpp"

CConsumer::CConsumer(std::shared_ptr<CClientCfg> spClientCfg, IModuleCallback *pCallback)
    : CClientCommon(std::move(spClientCfg), pCallback)
{
}

NvError CConsumer::AcquirePacket(uint32_t &uPacketIndex)
{
    NvSciStreamCookie cookie = NvSciStreamCookie_Invalid;

    /* Obtain packet with the new payload */
    auto sciErr = NvSciStreamConsumerPacketAcquire(m_handle, &cookie);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamConsumerPacketAcquire");
    PLOG_DBG("Acquired a packet (cookie = %lu).\n", cookie);

    auto error = GetIndexFromCookie(cookie, uPacketIndex);
    PCHK_ERROR_AND_RETURN(error, "GetIndexFromCookie");

    ClientPacket *packet = GetPacketByCookie(cookie);
    PCHK_PTR_AND_RETURN(packet, "GetPacketByCookie");

    return NvError_Success;
}

NvError CConsumer::InsertPrefence(uint32_t uPacketIndex)
{
    uint32_t uElemIndex = m_vElemsInfos[m_uDataElemId].uIndex;
    /* If the received waiter obj is NULL,
     * the producer is done writing data into this element, skip waiting on pre-fence.
     * For consumer, there is only one waiter, using index 0 as default.
     */
    if (m_waiterSyncObjs[0U][m_uDataElemId] != nullptr) {
        PLOG_DBG("Get prefence and insert it, waiter sync objects = %p\n", m_waiterSyncObjs[0U][m_uDataElemId]);
        std::unique_ptr<NvSciSyncFence, std::function<void(NvSciSyncFence *)>> upPrefence(
            new NvSciSyncFence(NvSciSyncFenceInitializer), [](NvSciSyncFence *fence) {
                NvSciSyncFenceClear(fence);
                delete fence;
            });
        auto sciErr =
            NvSciStreamBlockPacketFenceGet(m_handle, m_packets[uPacketIndex].handle, 0U, uElemIndex, upPrefence.get());
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockPacketFenceGet");

        if (IsClearedFence(upPrefence.get())) {
            PLOG_DBG("Empty fence supplied as prefence.Skipping prefence insertion \n");
        } else {
            if (m_cpuWaitCfg.bWaitPrefence) {
                sciErr = NvSciSyncFenceWait(upPrefence.get(), m_cpuWaitPreContext, FENCE_FRAME_TIMEOUT_US);
                PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncFenceWait");
            } else {
                auto error = m_pModuleCallback->InsertPrefence(this, m_vElemsInfos[m_uDataElemId].userType,
                                                               uPacketIndex, upPrefence.get());
                PCHK_ERROR_AND_RETURN(error, "InsertPrefence");
            }
        }
    }

    return NvError_Success;
}

NvError CConsumer::ProcessPacket(uint32_t uPacketIndex)
{
    auto error = m_pModuleCallback->ProcessPayload(this, uPacketIndex);
    PCHK_ERROR_AND_RETURN(error, "m_pModuleCallback->ProcessPayload");

    return NvError_Success;
}

NvError CConsumer::PostProcessPacket(uint32_t uPacketIndex, NvSciSyncFence *pPostfence)
{
    if (m_cpuWaitCfg.bWaitPostfence && m_cpuWaitPostContext != nullptr) {
        auto sciErr = NvSciSyncFenceWait(pPostfence, m_cpuWaitPostContext, FENCE_FRAME_TIMEOUT_US);
        NvSciSyncFenceClear(pPostfence);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncFenceWait");
    }

    auto error = m_pModuleCallback->OnProcessPayloadDone(this, uPacketIndex);
    PCHK_ERROR_AND_RETURN(error, "OnProcessPayloadDone");

    return NvError_Success;
}

NvError CConsumer::ReleasePacket(uint32_t uPacketIndex)
{
    /* Release the packet back to the producer */
    auto sciErr = NvSciStreamConsumerPacketRelease(m_handle, m_packets[uPacketIndex].handle);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamConsumerPacketRelease");

    return NvError_Success;
}

NvError CConsumer::ReleasePacket(uint32_t uPacketIndex, NvSciSyncFence *pPostfence, bool bClearFence)
{
    if ((pPostfence) && (!IsClearedFence(pPostfence))) {
        auto sciErr = NvSciStreamBlockPacketFenceSet(m_handle, m_packets[uPacketIndex].handle,
                                                     m_vElemsInfos[m_uDataElemId].uIndex, pPostfence);
        if (bClearFence) {
            NvSciSyncFenceClear(pPostfence);
        }

        if (sciErr != NvSciError_Success) {
            PLOG_WARN("NvSciStreamBlockPacketFenceSet failed, error: %u(0x%x)\n", sciErr, sciErr);
        }
    }

    return ReleasePacket(uPacketIndex);
}

NvError CConsumer::OnPacketProcessed(uint32_t uPacketIndex, NvSciSyncFence *pPostfence, bool bClearFence)
{
    ReleasePacket(uPacketIndex, pPostfence, bClearFence);
    return NvError_Success;
}

NvError CConsumer::HandlePayload()
{
    uint32_t uPacketIndex = MAX_NUM_PACKETS;
    NvSciSyncFence postfence = NvSciSyncFenceInitializer;
    bool bHandled = false;

    auto error = AcquirePacket(uPacketIndex);
    PCHK_ERROR_AND_RETURN(error, "AcquirePacket");

    m_uFrameNum++;
    if (m_uFrameNum % m_pAppCfg->GetFrameFilter() != 0) {
        /* Release the packet back to the producer */
        ReleasePacket(uPacketIndex);
        return NvError_Success;
    }

    /* Lock m_bStop to avoid sending null frames and image frames simultaneously. */
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_bStop) {
        /* Release the packet back to the producer if module stopped*/
        ReleasePacket(uPacketIndex);
        return NvError_Success;
    }

    if (m_pAppCfg->IsProfilingEnabled()) {
        /*
             * Recorde the receive TSC.
             */
        MetaData *pMetaData = reinterpret_cast<MetaData *>(GetMetaPtr(uPacketIndex));
        if (pMetaData != nullptr) {
            pMetaData->uReceiveTSC = CProfiler::GetCurrentTSC();
        }
    }

    error = m_pModuleCallback->OnPacketGotten(this, uPacketIndex, &bHandled);
    if (error == NvError_EndOfFile) {
        ReleasePacket(uPacketIndex);
        return NvError_Success;
    } else {
        PCHK_ERROR_AND_RETURN(error, "m_pModuleCallback->OnPacketGotten");
    }

    if (bHandled) {
        return NvError_Success;
    }

    error = InsertPrefence(uPacketIndex);
    PCHK_ERROR_AND_RETURN(error, "InsertPrefence");

    error = m_pModuleCallback->SetEofSyncObj(this);
    PCHK_ERROR_AND_RETURN(error, "m_pModuleCallback->SetEofSyncObj");

    error = ProcessPacket(uPacketIndex);
    PCHK_ERROR_AND_RETURN(error, "ProcessPacket");

    error = m_pModuleCallback->GetEofSyncFence(this, &postfence);
    PCHK_ERROR_AND_RETURN(error, "m_pModuleCallback->SetEofSyncObj");

    error = PostProcessPacket(uPacketIndex, &postfence);
    PCHK_ERROR_AND_RETURN(error, "PostProcessPacket");

    error = OnPacketProcessed(uPacketIndex, &postfence);
    PCHK_ERROR_AND_RETURN(error, "OnPacketProcessed");

    return NvError_Success;
}

NvError CConsumer::SetUnusedElement(uint32_t uElemIndex)
{
    auto err = NvSciStreamBlockElementUsageSet(m_handle, uElemIndex, false);
    PCHK_NVSCISTATUS_AND_RETURN(err, "NvSciStreamBlockElementUsageSet");

    return NvError_Success;
}

NvError CConsumer::GetConnectBlock(NvSciStreamBlock *pBlock)
{
    if (!m_handle) {
        auto sciErr = m_queueType == QueueType::Mailbox ? NvSciStreamMailboxQueueCreate(&m_queueHandle)
                                                        : NvSciStreamFifoQueueCreate(&m_queueHandle);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamQueueCreate");

        sciErr = NvSciStreamConsumerCreate2(m_queueHandle, false, &m_handle);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamConsumerCreate2");

        if (m_eCommType == CommType::IntraProcess && m_spClientCfg->m_uLimitNum != 0) {
            sciErr = NvSciStreamLimiterCreate(m_spClientCfg->m_uLimitNum, &m_limiterHandle);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamLimiterCreate");
            sciErr = NvSciStreamBlockConnect(m_limiterHandle, m_handle);
            CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockConnect");
            m_vBlockPairs.emplace_back(false, m_limiterHandle);
        }
        m_vBlockPairs.emplace_back(false, m_handle);
        m_vBlockPairs.emplace_back(false, m_queueHandle);
    }

    if (m_eCommType == CommType::IntraProcess && m_spClientCfg->m_uLimitNum != 0) {
        *pBlock = m_limiterHandle;
    } else {
        *pBlock = m_handle;
    }

    return NvError_Success;
}

NvError CConsumer::ConnectIpc(const std::shared_ptr<CIpcEndpointElem> &spIpcDst, int sensorId)
{
    std::string sChannelName = spIpcDst->GetChannelStr(sensorId);
    if (sChannelName.empty()) {
        LOG_ERR("GetChannelStr failed, sensorId: %d\n", sensorId);
        return NvError_BadValue;
    }
    std::vector<NvSciStreamBlock> vC2cHandlesTmp;

    if (!spIpcDst->IsC2C()) {
        auto error = ConnectIpc(sChannelName, m_ipcEntity);
        CHK_ERROR_AND_RETURN(error, "ConnectIpc");
    } else {
        auto error = ConnectC2c(sChannelName, m_ipcEntity, vC2cHandlesTmp);
        CHK_ERROR_AND_RETURN(error, "ConnectC2C");
    }

    m_vBlockPairs.emplace_back(false, m_ipcEntity.ipcBlock);
    for (auto block : vC2cHandlesTmp) {
        m_vBlockPairs.emplace_back(false, block);
        m_vC2cHandles.emplace_back(block);
    }

    // create control channel writer
    if (m_spControlChannel) {
        std::string sSrcChannelName = spIpcDst->GetOppositeChannelStr(sensorId);
        m_spMsgWriter = m_spControlChannel->CreateWriter(sSrcChannelName.c_str());
        PCHK_PTR_AND_RETURN(m_spMsgWriter, "Create control channel writer");
        // send connect request
        HeartBeatMsgType msgType = HeartBeatMsgType::CONNECT_REQUEST;
        auto error = m_spMsgWriter->Write(&msgType, sizeof(HeartBeatMsgType));
        PCHK_ERROR_AND_RETURN(error, "Send connect request");
    }

    return NvError_Success;
}

NvError CConsumer::ConnectIpc(const std::string &sChannelName, IpcEntity &ipcEntity)
{
    NvSciStreamBlock connectBlock;

    m_eCommType = CommType::InterProcess;
    auto error = GetConnectBlock(&connectBlock);
    CHK_ERROR_AND_RETURN(error, "GetConnectBlock");

    LOG_MSG(GetName() + " NvSciIpcOpenEndpoint: " + sChannelName + "\n");
    auto sciErr = NvSciIpcOpenEndpoint(sChannelName.c_str(), &ipcEntity.endPoint);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciIpcOpenEndpoint ");
    NvSciIpcResetEndpointSafe(ipcEntity.endPoint);
    sciErr = NvSciStreamIpcDstCreate2(ipcEntity.endPoint, m_pAppCfg->m_sciSyncModule, m_pAppCfg->m_sciBufModule,
                                      static_cast<NvSciStreamBlock>(0), &ipcEntity.ipcBlock);
    if (sciErr != NvSciError_Success) {
        NvSciIpcCloseEndpointSafe(ipcEntity.endPoint, false);
        LOG_ERR("Create ipc block failed, error: 0x%x\n", sciErr);
        return NvError_BadValue;
    }

    sciErr = NvSciStreamBlockConnect(ipcEntity.ipcBlock, connectBlock);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Connect blocks: dstIpc - consumer");

    return NvError_Success;
}

NvError
CConsumer::ConnectC2c(const std::string &sChannelName, IpcEntity &ipcEntity, std::vector<NvSciStreamBlock> &vBlocks)
{
    NvSciStreamBlock poolHandle = 0U;

    m_eCommType = CommType::InterChip;
    auto sciErr = NvSciStreamStaticPoolCreate(MAX_NUM_PACKETS, &poolHandle);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamStaticPoolCreate");
    m_upPoolManager = CreatePoolManager(poolHandle);

    std::vector<PacketElementType> vElemTypesToSkip{};
    for (ElementInfo elemInfo : m_vElemsInfos) {
        if (!elemInfo.bIsUsed) {
            vElemTypesToSkip.push_back(elemInfo.userType);
        }
    }
    m_upPoolManager->SetElemTypesToSkip(vElemTypesToSkip);

    NvSciStreamBlock connectBlock;
    auto error = GetConnectBlock(&connectBlock);
    CHK_ERROR_AND_RETURN(error, "GetConnectBlock");

    LOG_MSG(GetName() + " NvSciIpcOpenEndpoint: " + sChannelName + "\n");
    sciErr = NvSciIpcOpenEndpoint(sChannelName.c_str(), &ipcEntity.endPoint);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciIpcOpenEndpoint ");
    NvSciIpcResetEndpointSafe(ipcEntity.endPoint);

    sciErr = NvSciStreamIpcDstCreate2(ipcEntity.endPoint, m_pAppCfg->m_sciSyncModule, m_pAppCfg->m_sciBufModule,
                                      poolHandle, &ipcEntity.ipcBlock);
    if (sciErr != NvSciError_Success) {
        NvSciIpcCloseEndpointSafe(ipcEntity.endPoint, false);
        LOG_ERR("Create ipc block failed, error: 0x%x\n", sciErr);
        return NvError_BadParameter;
    }

    sciErr = NvSciStreamBlockConnect(ipcEntity.ipcBlock, connectBlock);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Connect blocks: dstIpc - consumer");

    vBlocks.push_back(poolHandle);

    return NvError_Success;
}

std::unique_ptr<CPoolManager> CConsumer::CreatePoolManager(NvSciStreamBlock poolHandle)
{
    return std::make_unique<CPoolManager>(poolHandle, GetName(), MAX_NUM_PACKETS, true);
}

void CConsumer::DeInit()
{
    CClientCommon::DeInit();

    if (m_upPoolEventHandler.get()) {
        m_upPoolEventHandler->QuitThread();
    }

    if (m_ipcEntity.ipcBlock) {
        NvSciStreamBlockDelete(m_ipcEntity.ipcBlock);
        m_ipcEntity.ipcBlock = 0U;
    }

    if (m_ipcEntity.endPoint) {
        NvSciIpcCloseEndpointSafe(m_ipcEntity.endPoint, false);
        m_ipcEntity.endPoint = 0U;
    }

    if (m_queueHandle) {
        NvSciStreamBlockDelete(m_queueHandle);
        m_queueHandle = 0;
    }
    if (m_handle) {
        NvSciStreamBlockDelete(m_handle);
        m_handle = 0;
    }
    if (m_limiterHandle) {
        NvSciStreamBlockDelete(m_limiterHandle);
        m_limiterHandle = 0;
    }
    for (auto &handle : m_vC2cHandles) {
        NvSciStreamBlockDelete(handle);
        handle = 0;
    }
}

NvError CConsumer::Reconcile()
{
    auto error = CClientCommon::Reconcile();
    PCHK_ERROR_AND_RETURN(error, "CConsumer CClientCommon::Reconcile");

    if (m_upPoolManager != nullptr) {
        m_upPoolEventHandler = std::make_unique<CEventHandler<CConsumer>>();
        error = m_upPoolEventHandler->RegisterHandler(&CConsumer::HandlePoolEvent, this);
        PCHK_ERROR_AND_RETURN(error, "RegisterHandler");
        error = m_upPoolEventHandler->StartThread();
        PCHK_ERROR_AND_RETURN(error, "StartThread");
    }

    return NvError_Success;
}

EventStatus CConsumer::HandlePoolEvent()
{
    EventStatus status = m_upPoolManager->HandleEvent();
    if (status == EventStatus::ERROR) {
        m_pModuleCallback->OnEvent(this, EventStatus::ERROR);
    }
    return status;
}

void CConsumer::ReportHeartBeat()
{
    HeartBeatMsgType msgType = HeartBeatMsgType::HEARTBEAT;
    auto error = m_spMsgWriter->Write(&msgType, sizeof(HeartBeatMsgType));
    if (NvError_Success != error) {
        PLOG_ERR("failed (0x%x) to report heartbeat!\n", error);
    }
}

NvError CConsumer::OnConnected()
{
    if (m_eCommType != CommType::IntraProcess) {
        auto error = m_pAppCfg->GetPeerValidator()->Validate(m_handle);
        if (NvError_Success != error) {
            LOG_WARN(GetName() + "Peer Validate Error, error=%d.\n", error);
            return NvError_BadParameter;
        }
        // when ipc connected, begin to report heartbeat
        if (m_spTimer != nullptr) {
            m_spTimer->AddTask([this] { this->ReportHeartBeat(); }, kHeartBeatIntervalMs);
        }
    }
    return NvError_Success;
}

NvError CConsumer::Stop()
{
    PLOG_DBG("Enter: CConsumer::Stop()\n");
    std::lock_guard<std::mutex> lock(m_mutex);

    PLOG_DBG("Exit: CConsumer::Stop()\n");
    return CClientCommon::Stop();
}
