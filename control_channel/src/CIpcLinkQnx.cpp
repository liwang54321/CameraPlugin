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

#ifdef NVMEDIA_QNX

#include "CIpcLinkQnx.hpp"
#include <sys/neutrino.h>
#include <unistd.h>

constexpr int16_t kEventCode = 0x30;

CIpcLink::CIpcLink(IMsgReadCallback *pCallback, bool isServer)
    : m_pReadCallBack(pCallback)
    , m_bServer(isServer)
{
}

NvError CIpcLink::Init(const std::string &sEndpointName)
{
    LOG_DBG("Enter: CIpcLink::Init()\n");

    m_sEndpointName = sEndpointName;
    NvSciError sciErr = NvSciError_Success;

    LOG_INFO(" NvSciIpcOpenEndpoint: " + m_sEndpointName + "\n");
    // Open named endpoint
    sciErr = NvSciIpcOpenEndpoint(m_sEndpointName.c_str(), &m_ipcHandle);
    if (NvSciError_Success != sciErr) {
        LOG_WARN("NvSciIpcOpenEndpoint failed (0x%x)\n", sciErr);
        return NvError_ResourceError;
    }

    // Create QNX channel for monitoring IPC
    m_chid = ChannelCreate_r(_NTO_CHF_UNBLOCK);
    if (0 > m_chid) {
        LOG_ERR("ChannelCreate_r failed (%d:%x)\n", m_chid, NvSciError_ResourceError);
        return NvError_ResourceError;
    }

    // Connect QNX channel for monitoring IPC
    m_connId = ConnectAttach_r(0, 0, m_chid, _NTO_SIDE_CHANNEL, 0);
    if (0 > m_connId) {
        LOG_ERR("ConnectAttach_r failed (%d:%x)\n", m_connId, NvSciError_ResourceError);
        return NvError_ResourceError;
    }

    // Bind IPC events to QNX connection
    sciErr = NvSciIpcSetQnxPulseParamSafe(m_ipcHandle, m_connId, SIGEV_PULSE_PRIO_INHERIT, kEventCode);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciIpcSetQnxPulseParamSafe");

    sciErr = NvSciIpcGetEndpointInfo(m_ipcHandle, &m_ipcInfo);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciIpcGetEndpointInfo");
    LOG_DBG("Endpoint info: endpoint name = %s, nframes = %d, frame_size = %d\n", m_sEndpointName.c_str(),
            m_ipcInfo.nframes, m_ipcInfo.frame_size);

    sciErr = NvSciIpcResetEndpointSafe(m_ipcHandle);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciIpcResetEndpointSafe");

    m_upWriteBuf.reset(new uint8_t[m_ipcInfo.frame_size]);
    CHK_PTR_AND_RETURN(m_upWriteBuf, "Create write buffer");

    m_upReadBuf.reset(new uint8_t[m_ipcInfo.frame_size]);
    CHK_PTR_AND_RETURN(m_upReadBuf, "Create read buffer");

    m_bInit = true;
    LOG_DBG("Exit: CIpcLink::Init()\n");
    return NvError_Success;
}

NvError CIpcLink::Start()
{
    LOG_DBG("Enter: CIpcLink::Start()\n");
    m_upReadThread = std::make_unique<std::thread>(&CIpcLink::ReadThreadFunc, this);
    CHK_PTR_AND_RETURN(m_upReadThread, "Create read thread");

    if (!m_bServer) {
        m_upConnectThread = std::make_unique<std::thread>(&CIpcLink::SendConnectMsg, this);
        CHK_PTR_AND_RETURN(m_upConnectThread, "Create read thread");
    }
    LOG_DBG("Exit: CIpcLink::Start()\n");
    return NvError_Success;
}

void CIpcLink::Stop()
{
    LOG_DBG("Enter: CIpcLink::Stop()\n");
    if (m_bConnected) {
        // inform opposite ipc that this side is about to quit
        MessageHeader header("", sizeof(MessageHeader), 0, MsgType::DISCONNECT_MSG);
        Write(&header, nullptr);
        m_bConnected = false;
    }

    m_bStop = true;

    NvSciError sciErr = NvSciIpcCloseEndpointSafe(m_ipcHandle, false);
    if (NvSciError_Success != sciErr) {
        LOG_ERR("NvSciIpcCloseEndpointSafe failed (%x)\n", sciErr);
    }
    if (m_connId != 0) {
        (void)ConnectDetach_r(m_connId);
        m_connId = 0;
    }
    if (m_chid != 0) {
        (void)ChannelDestroy_r(m_chid);
        m_chid = 0;
    }

    // wait thread join
    if (m_upReadThread != nullptr && m_upReadThread->joinable()) {
        m_upReadThread->join();
    }
    if (m_upConnectThread != nullptr && m_upConnectThread->joinable()) {
        m_upConnectThread->join();
    }
    LOG_DBG("Exit: CIpcLink::Stop()\n");
}

void CIpcLink::Deinit()
{
    LOG_DBG("Enter: CIpcLink::Deinit()\n");

    if (!m_bInit) {
        return;
    }
    m_upConnectThread.reset(nullptr);
    m_upReadThread.reset(nullptr);
    m_upReadBuf.reset(nullptr);
    m_upWriteBuf.reset(nullptr);
    LOG_DBG("Exit: CIpcLink::Deinit()\n");
}

NvError CIpcLink::WaitEvent(uint32_t value, uint32_t uTimeoutMs)
{
    struct _pulse pulse;
    uint32_t event = 0;
    NvSciError sciErr = NvSciError_Success;

    while (!m_bStop) {

        // Get pending IPC events
        sciErr = NvSciIpcGetEventSafe(m_ipcHandle, &event);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciIpcGetEventSafe");

        // Return if event is the kind we're looking for
        if (0U != (event & value)) {
            break;
        }
        // Wait for pulse indicating new event
        int64_t timeoutUs =
            uTimeoutMs == kEventWaitInfinite ? NVSCIIPC_INFINITE_WAIT : static_cast<int64_t>(uTimeoutMs) * 1000;
        int32_t ret = NvSciIpcWaitEventQnx(m_chid, timeoutUs, sizeof(pulse), &pulse);
        if (m_bStop) {
            return NvError_InvalidState;
        }
        if (-ETIMEDOUT == ret) {
            return NvError_Timeout;
        } else if (0 > ret) {
            LOG_ERR("NvSciIpcWaitEventQnx failed (%d)\n", ret);
            return NvError_FileOperationFailed;
        } else if (kEventCode != pulse.code) {
            LOG_ERR("Invalid pulse %d\n", pulse.code);
            return NvError_FileOperationFailed;
        }
    }
    return NvError_Success;
}

void CIpcLink::SendConnectMsg()
{
    MessageHeader header("", sizeof(MessageHeader), 0, MsgType::CONNECT_MSG);
    auto error = Write(&header, nullptr);
    if (NvError_Success != error) {
        LOG_ERR("CIpcLink failed (0x%x) to send connect msg\n", error);
    }
}

NvError CIpcLink::Write(const MessageHeader *pHeaderBuf, const void *pContentBuf)
{
    if (m_bStop) {
        LOG_ERR("Ipc link %s has been stopped, failed to write message\n", m_sEndpointName.c_str());
        return NvError_InvalidState;
    }
    NvError error = NvError_Success;

    std::unique_lock<std::mutex> lk(m_writeMutex);
    while (!m_bStop) {
        error = WaitEvent(NV_SCI_IPC_EVENT_WRITE, kWriteDefaultTimeoutMs);
        if (NvError_Timeout == error) {
            LOG_WARN("Wait write event timeout\n");
            continue;
        } else {
            break;
        }
    }
    if (m_bStop) {
        return NvError_Success;
    }
    CHK_ERROR_AND_RETURN(error, "waitEvent");

    memcpy(m_upWriteBuf.get(), pHeaderBuf, sizeof(MessageHeader));
    uint32_t uMsgSize = pHeaderBuf->uContentLength + pHeaderBuf->uContentOffset;
    if (uMsgSize > m_ipcInfo.frame_size) {
        LOG_ERR("Msg length (%d) exceeds ipc frame size (%d)\n", uMsgSize, m_ipcInfo.frame_size);
        return NvError_BadValue;
    }

    if (pContentBuf && pHeaderBuf->uContentLength != 0) {
        memcpy(m_upWriteBuf.get() + pHeaderBuf->uContentOffset, pContentBuf, pHeaderBuf->uContentLength);
    }

    uint32_t size = pHeaderBuf->uContentOffset + pHeaderBuf->uContentLength;
    uint32_t bytes;
    auto sciErr = NvSciIpcWriteSafe(m_ipcHandle, m_upWriteBuf.get(), size, &bytes);
    if (NvSciError_Success != sciErr || bytes != size) {
        LOG_ERR("Failed (0x%x) to NvSciIpcWriteSafe, message type: %d\n", sciErr, pHeaderBuf->msgType);
        return NvError_FileWriteFailed;
    }
    return NvError_Success;
}

void CIpcLink::ReadThreadFunc()
{
    pthread_setname_np(pthread_self(), "IpcLinkReadThrd");
    NvSciError sciErr = NvSciError_Success;
    while (!m_bStop) {
        auto error = WaitEvent(NV_SCI_IPC_EVENT_READ, kEventWaitInfinite);
        if (m_bStop) {
            return;
        }
        if (NvError_Success != error) {
            LOG_ERR("Failed (0x%x) to wait event\n", error);
            return;
        }
        uint32_t readBytes = 0;
        sciErr =
            NvSciIpcReadSafe(m_ipcHandle, static_cast<void *>(m_upReadBuf.get()), m_ipcInfo.frame_size, &readBytes);
        if (NvSciError_Success != sciErr || readBytes != m_ipcInfo.frame_size) {
            LOG_ERR("Failed (0x%x) to NvSciIpcReadSafe\n", sciErr);
            return;
        }

        MessageHeader *pHeaderBuf = reinterpret_cast<MessageHeader *>(m_upReadBuf.get());
        void *pContentBuf = static_cast<void *>(m_upReadBuf.get() + pHeaderBuf->uContentOffset);
        error = HandleEvent(pHeaderBuf, pContentBuf);
        if (NvError_Success != error) {
            LOG_ERR("CIpcLink failed to process msg\n");
        }
    }
}

NvError CIpcLink::HandleEvent(MessageHeader *pHeaderBuf, void *pContentBuf)
{
    if (pHeaderBuf->msgType == MsgType::CONNECT_MSG) {
        m_bConnected = true;
    } else if (pHeaderBuf->msgType == MsgType::DISCONNECT_MSG) {
        m_bConnected = false;
    }
    return m_pReadCallBack->HandleEvent(shared_from_this(), pHeaderBuf, pContentBuf);
}

#endif
