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

#ifndef NVMEDIA_QNX

#include "CIpcLinkLinux.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <inttypes.h>

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

    sciErr = NvSciIpcGetLinuxEventFd(m_ipcHandle, &m_ipcEventFd);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciIpcGetLinuxEventFd");

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

    if (pipe(m_pipeFd) == -1) {
        LOG_ERR("Failed to setup pipe\n");
        return NvError_FileOperationFailed;
    }
    // Set write end non-blocking
    int ret = fcntl(m_pipeFd[1], F_SETFL, O_NONBLOCK);
    if (ret < 0) {
        LOG_ERR("Failed to set the write end non-blocking");
        Deinit();
        return NvError_FileOperationFailed;
    }
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
        NvError error = Write(&header, nullptr);
        if (NvError_Success != error) {
            LOG_ERR("Failed to write disconnect msg.");
        }
        m_bConnected = false;
    }

    // wait thread join
    m_bStop = true;
    if (write(m_pipeFd[1], "q", 1) != 1) {
        LOG_ERR("Pipe write quit command failed");
    }
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
    NvSciError sciErr = NvSciIpcCloseEndpointSafe(m_ipcHandle, false);
    if (NvSciError_Success != sciErr) {
        LOG_ERR("NvSciIpcCloseEndpointSafe failed (%x)\n", sciErr);
    }
    m_upConnectThread.reset(nullptr);
    m_upReadThread.reset(nullptr);
    m_upReadBuf.reset(nullptr);
    m_upWriteBuf.reset(nullptr);
    close(m_pipeFd[0]);
    close(m_pipeFd[1]);
    LOG_DBG("Exit: CIpcLink::Deinit()\n");
}

NvError CIpcLink::WaitEvent(uint32_t value, uint64_t uTimeoutMs)
{
    fd_set rfds;
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

        FD_ZERO(&rfds);
        FD_SET(m_ipcEventFd, &rfds);
        FD_SET(m_pipeFd[0], &rfds);
        int maxfd = std::max(m_ipcEventFd, m_pipeFd[0]);
        struct timeval tv;
        tv.tv_sec = uTimeoutMs / 1000;
        tv.tv_usec = (uTimeoutMs % 1000) * 1000;
        // Wait for signalling indicating new event
        int ret = select(maxfd + 1, &rfds, NULL, NULL, &tv);
        if (ret < 0) {
            // select failed
            LOG_ERR("Select error\n");
            return NvError_FileOperationFailed;
        } else if (ret == 0) {
            LOG_WARN("Select timeout(%" PRIu64 "ms)\n", uTimeoutMs);
            return NvError_Timeout;
        }

        if (FD_ISSET(m_pipeFd[0], &rfds)) {
            LOG_INFO("CIpcLink::WaitEvent received quit command\n");
            break;
        } else if (!FD_ISSET(m_ipcEventFd, &rfds)) {
            LOG_ERR("FD_ISSET error\n");
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
    uint32_t bytes = 0U;
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
        if (NvError_Success != error) {
            LOG_ERR("Failed (0x%x) to wait event\n", error);
            return;
        }
        if (m_bStop) {
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