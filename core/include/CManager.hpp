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

#ifndef CMANAGER_HPP
#define CMANAGER_HPP

/* STL Headers */
#include <cstring>
#include <iostream>
#include <functional>
#include <memory>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>

#include "CConfig.hpp"
#include "CChannel.hpp"
#include "CUtils.hpp"
#include "NvSIPLCamera.hpp"
#include "NvSIPLPipelineMgr.hpp"

#include "nvscibuf.h"
#include "nvscistream.h"
#include "nvscisync.h"

using EventCallback = std::function<void(CChannel *pChannel, EventStatus event)>;

/** CManager class */
class CManager : public IEventListener<CChannel>
{
  public:
    CManager(std::shared_ptr<CAppCfg> spAppConfig, EventCallback pEventCallback = nullptr);
    virtual ~CManager();

    virtual NvError Init();
    virtual void DeInit();
    virtual void Stop();
    virtual void Start();
    virtual void Run();
    virtual void Quit();
    void SwitchPowerMode(bool bLowPowerState);

    void MonitorThreadFunc();
    virtual void OnError(CChannel *pChannel, int moduleId, uint32_t errorId) override;

  protected:
    virtual void OnEvent(CChannel *pChannel, EventStatus event) override;
    virtual bool IsControlChannelNeeded() { return true; }

    std::shared_ptr<CAppCfg> m_spAppCfg;
    std::unique_ptr<std::thread> m_upMonitorThread;
    std::atomic<bool> m_bMonitorThreadRunning{ false };
    std::atomic<bool> m_bMonitorThreadPause{ false };

  private:
    NvError InitStream();
    NvError CreateChannels();
    NvError CreateControlChannel();
    void InputEventLoop();
    void SetLogLevel(uint32_t uVerbosity);

    std::atomic<bool> m_bQuit{ false };
    std::unique_ptr<std::thread> m_upEventThread;
    std::atomic<bool> m_bEventThreadRunning{ false };
    std::mutex m_monitorThreadMutex;
    std::condition_variable m_monitorThreadConditionVar;
    std::mutex m_cmdMutex;
    std::condition_variable m_cmdConditionVar;
    std::vector<std::unique_ptr<CChannel>> m_vupChannels;
    int m_channelQuitCount{ 0 };
    EventCallback m_pEventCallback;
};

#endif //CMANAGER_HPP
