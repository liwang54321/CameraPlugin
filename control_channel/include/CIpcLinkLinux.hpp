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

#ifndef CIPCLINK_HPP
#define CIPCLINK_HPP

#include "nvsciipc.h"
#include "nvscievent.h"
#include "CUtils.hpp"
#include "CMsgCommon.hpp"
#include <thread>
#include <memory>

constexpr uint64_t kWriteDefaultTimeoutMs = 1000UL;
constexpr uint64_t kEventWaitInfinite = UINT64_MAX;

class CIpcLink : public std::enable_shared_from_this<CIpcLink>
{
  public:
    class IMsgReadCallback
    {
      public:
        virtual NvError
        HandleEvent(std::shared_ptr<CIpcLink> spIpcLink, MessageHeader *pMsgHeader, void *pContentBuf) = 0;

      protected:
        IMsgReadCallback() = default;
        virtual ~IMsgReadCallback() = default;
    };

    CIpcLink(IMsgReadCallback *pCallback, bool isServer);
    ~CIpcLink() = default;

    NvError Init(const std::string &sEndpointName);
    NvError Start();
    void Stop();
    void Deinit();

    NvError Write(const MessageHeader *pHeaderBuf, const void *pContentBuf);
    NvError HandleEvent(MessageHeader *pHeaderBuf, void *pContentBuf);
    inline std::string GetName() { return m_sEndpointName; }
    inline bool IsConnected() { return m_bConnected; }

  private:
    void SendConnectMsg();
    NvError WaitEvent(uint32_t value, uint64_t uTimeoutMs);
    void ReadThreadFunc();

    std::string m_sEndpointName;
    IMsgReadCallback *m_pReadCallBack;
    bool m_bServer;
    std::atomic<bool> m_bInit{ false };
    std::atomic<bool> m_bStop{ false };
    std::atomic<bool> m_bConnected{ false };
    NvSciIpcEndpoint m_ipcHandle = 0U;
    struct NvSciIpcEndpointInfo m_ipcInfo{};
    int32_t m_ipcEventFd = -1;
    int32_t m_pipeFd[2];
    std::mutex m_writeMutex;
    std::unique_ptr<uint8_t[]> m_upWriteBuf = { nullptr };
    std::unique_ptr<uint8_t[]> m_upReadBuf = { nullptr };
    std::unique_ptr<std::thread> m_upReadThread = { nullptr };
    std::unique_ptr<std::thread> m_upConnectThread = { nullptr };
};

#endif

#endif //NVMEDIA_QNX