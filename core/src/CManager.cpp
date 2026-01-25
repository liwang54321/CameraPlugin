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

#include <sys/time.h>
#include <algorithm>
#include "CManager.hpp"

using namespace nvsipl;

constexpr int SECONDS_PER_ITERATION = 2;
constexpr int STOP_TIMEOUT_SECONDS = 5;
constexpr int MESSAGE_BUFFER_LENGTH = 256;
constexpr uint32_t THREADPOOL_SIZE = 2U;
constexpr uint64_t CONTROL_CHANNEL_SETUP_TIMEOUT_MS = 1000U;

CManager::CManager(std::shared_ptr<CAppCfg> spAppConfig, EventCallback pEventCallback)
    : m_spAppCfg(std::move(spAppConfig))
    , m_pEventCallback(std::move(pEventCallback))
{
}

CManager::~CManager() = default;

NvError CManager::Init()
{
    NvError error = NvError_Success;
    LOG_DBG("Enter: CManager::Init()\n");

    SetLogLevel(m_spAppCfg->GetVerbosity());

    error = InitStream();
    CHK_ERROR_AND_RETURN(error, "CManager::InitStream");

    // Disable control channel for status manager.
    if (m_spAppCfg->GetCommType() != CommType::IntraProcess && IsControlChannelNeeded()) {
        error = CreateControlChannel();
        CHK_ERROR_AND_RETURN(error, "CManager::CreateControlChannel");
        error = m_spAppCfg->m_spControlChannel->Init();
        CHK_ERROR_AND_RETURN(error, "CControlChannelManager::Init");

        m_spAppCfg->m_spTimer = std::make_shared<CTimer>(THREADPOOL_SIZE);
        CHK_PTR_AND_RETURN(m_spAppCfg->m_spTimer, "Timer create");
        error = m_spAppCfg->m_spTimer->Init();
        CHK_ERROR_AND_RETURN(error, "Timer init");

        error = m_spAppCfg->m_spControlChannel->WaitSetupComplete(CONTROL_CHANNEL_SETUP_TIMEOUT_MS);
        CHK_ERROR_AND_RETURN(error, "CControlChannel WaitSetupComplete");
    }

    m_spAppCfg->Init();
    error = CreateChannels();
    CHK_ERROR_AND_RETURN(error, "CManager: CreateChannels");

    for (auto &channel : m_vupChannels) {
        channel->Init();
    }

    LOG_DBG("Exit: CManager::Init()\n");

    return NvError_Success;
}

void CManager::DeInit()
{
    LOG_DBG("Enter: CManager::DeInit()\n");

    if (m_spAppCfg->m_spTimer != nullptr) {
        m_spAppCfg->m_spTimer->Deinit();
    }

    if (m_spAppCfg->m_spControlChannel != nullptr) {
        m_spAppCfg->m_spControlChannel->Deinit();
    }

    if (m_spAppCfg->GetCommType() != CommType::IntraProcess) {
        NvSciIpcDeinit();
    }

    if (m_spAppCfg->m_sciBufModule != nullptr) {
        NvSciBufModuleClose(m_spAppCfg->m_sciBufModule);
    }

    if (m_spAppCfg->m_sciSyncModule != nullptr) {
        NvSciSyncModuleClose(m_spAppCfg->m_sciSyncModule);
    }

    LOG_DBG("Exit: CManager::DeInit()\n");
}

void CManager::Run()
{
    LOG_DBG("Enter: CManager::Run()\n");

    m_bMonitorThreadRunning = true;
    m_bMonitorThreadPause = true;
    m_upMonitorThread.reset(new std::thread(&CManager::MonitorThreadFunc, this));

    m_bEventThreadRunning = true;
    InputEventLoop();
    m_upMonitorThread->join();

    LOG_DBG("Exit: CManager::Run()\n");
}

void CManager::Quit()
{
    LOG_DBG("Enter: CManager::Quit()\n");

    bool bExpected = false;
    if (m_bQuit.compare_exchange_strong(bExpected, true)) {

        for (auto &channel : m_vupChannels) {
            channel->Quit();
        }
    }

    LOG_DBG("Exit: CManager::Quit()\n");
}

void CManager::Start()
{
    LOG_DBG("Enter: CManager::Start()\n");

    for (auto &channel : m_vupChannels) {
        channel->Start();
    }

    LOG_DBG("Exit: CManager::Start()\n");
}

void CManager::Stop()
{
    LOG_DBG("Enter: CManager::Stop()\n");

    auto areAllChannelsStopped = [this]() {
        return std::all_of(m_vupChannels.begin(), m_vupChannels.end(),
                           [](const auto &channel) { return channel->IsStopped(); });
    };

    m_bMonitorThreadPause = true;

    for (auto &channel : m_vupChannels) {
        channel->Stop();
    }

    std::unique_lock<std::mutex> lock(m_cmdMutex);
    if (!m_cmdConditionVar.wait_for(lock, std::chrono::seconds(STOP_TIMEOUT_SECONDS), areAllChannelsStopped)) {
        LOG_ERR("CManager::Stop() timeout, channels not stopped within %d seconds\n", STOP_TIMEOUT_SECONDS);
        Quit();
    }

    LOG_DBG("Exit: CManager::Stop()\n");
}

void CManager::SwitchPowerMode(bool bLowPowerState)
{
    LOG_DBG("Enter: CManager::SwitchPowerMode()\n");

    for (auto &channel : m_vupChannels) {
        channel->HandleCmdAsync(bLowPowerState ? CMDType::ENTER_LOW_POWER_MODE : CMDType::ENTER_FULL_POWER_MODE);
    }

    LOG_DBG("Exit: CManager::SwitchPowerMode()\n");
}

void CManager::SetLogLevel(uint32_t uVerbosity)
{
    LOG_INFO("Setting verbosity level: %u\n", uVerbosity);

    CLogger::GetInstance().SetLogLevel((CLogger::LogLevel)uVerbosity);
}

NvError CManager::InitStream()
{
    LOG_DBG("Enter: CManager::InitStream()\n");

    auto sciErr = NvSciBufModuleOpen(&m_spAppCfg->m_sciBufModule);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufModuleOpen");

    sciErr = NvSciSyncModuleOpen(&m_spAppCfg->m_sciSyncModule);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncModuleOpen");

    if (m_spAppCfg->GetCommType() != CommType::IntraProcess) {
        sciErr = NvSciIpcInit();
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciIpcInit");
    }

    LOG_DBG("Exit: CManager::InitStream()\n");

    return NvError_Success;
}

void CManager::MonitorThreadFunc()
{
    uint64_t uTimeElapsedSum = 0u;
    uint32_t uRunDurationSec = m_spAppCfg->GetRunDurationSec();
    const auto sleepTime = std::chrono::seconds(SECONDS_PER_ITERATION);

    LOG_DBG("Enter: MonitorThreadFunc()\n");
    pthread_setname_np(pthread_self(), "MonitorThrd");
    bool debug = getenv("SIPL_DEBUG") == nullptr? false : true;
    while (m_bMonitorThreadRunning) {
        if (m_bMonitorThreadPause) {
            std::unique_lock<std::mutex> lock(m_monitorThreadMutex);
            m_monitorThreadConditionVar.wait(lock);
        }
        auto oStartTime = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(sleepTime);
        auto uTimeElapsedMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - oStartTime).count();
        uTimeElapsedSum += uTimeElapsedMs;
        if (debug) {
            std::cout << std::endl;
            for (const auto &channel : m_vupChannels) {
                channel->PrintFps();
                std::cout << std::endl;
            }
        }

        if (uRunDurationSec && (uTimeElapsedSum / 1000 >= uRunDurationSec)) {
            Quit();
        }
    }

    LOG_DBG("Exit: MonitorThreadFunc()\n");
}

NvError CManager::CreateChannels()
{
    LOG_DBG("Enter: CManager::CreateChannels()\n");

    std::vector<std::shared_ptr<CChannelCfg>> vspChannelCfgs;
    m_spAppCfg->CreateChannelCfgs(vspChannelCfgs);

    for (const auto &channelCfg : vspChannelCfgs) {
        m_vupChannels.push_back(std::make_unique<CChannel>(channelCfg, this));
    }

    LOG_DBG("Exit: CManager::CreateChannels()\n");
    return NvError_Success;
}

NvError CManager::CreateControlChannel()
{
    LOG_DBG("Enter: CManager::CreateControlChannel()\n");

    ControlChannelConfig cccfg = { .isCentralNode = m_spAppCfg->IsCentralNode(),
                                   .isP2P = (m_spAppCfg->GetCommType() != CommType::IntraProcess),
                                   .isC2C = (m_spAppCfg->GetCommType() == CommType::InterChip),
                                   .isMasterSoc = m_spAppCfg->IsC2CMasterSoc() };
    LOG_DBG("isCentralNode: %s\n", cccfg.isCentralNode ? "true" : "false");
    LOG_DBG("isP2P: %s\n", cccfg.isP2P ? "true" : "false");
    LOG_DBG("isC2C: %s\n", cccfg.isC2C ? "true" : "false");
    LOG_DBG("isMasterSoc: %s\n", cccfg.isMasterSoc ? "true" : "false");
    m_spAppCfg->m_spControlChannel = std::make_shared<CControlChannelManager>(cccfg);
    CHK_PTR_AND_RETURN(m_spAppCfg->m_spControlChannel, "Control channel create");

    LOG_DBG("Exit: CManager::CreateControlChannel()\n");
    return NvError_Success;
}

void CManager::OnEvent(CChannel *pChannel, EventStatus event)
{
    switch (event) {
        case EventStatus::ERROR:
            LOG_ERR("Received fatal error, quit!\n");
            Quit();
            break;
        case EventStatus::DISCONNECT:
            LOG_MSG("Received disconnect event, quit!\n");
            Quit();
            break;
        case EventStatus::STARTED:
            m_bMonitorThreadPause = false;
            m_monitorThreadConditionVar.notify_all();
            break;
        case EventStatus::STOPPED:
            LOG_MSG("Received a stopped event!\n");
            m_cmdConditionVar.notify_all();
            break;
        case EventStatus::QUITTED:
            if (++m_channelQuitCount == static_cast<int>(m_vupChannels.size())) {
                m_bEventThreadRunning = false;

                m_bMonitorThreadRunning = false;
                if (m_bMonitorThreadPause) {
                    m_bMonitorThreadPause = false;
                    m_monitorThreadConditionVar.notify_all();
                }
            }
            break;
        default:
            break;
    }
    if (nullptr != m_pEventCallback) {
        m_pEventCallback(pChannel, event);
    }
}

void CManager::OnError(CChannel *channel, int moduleId, uint32_t errorId)
{
    LOG_ERR("CManager::OnError moduleId %d errorId %d\n", moduleId, errorId);
    Quit();
}

void CManager::InputEventLoop()
{
    LOG_DBG("Enter: InputEventLoop()\n");
#if 0
    std::cout << "Enter 'q' to quit the application" << std::endl;
    std::cout << "Enter 's to suspend application" << std::endl;
    std::cout << "Enter 'r to resume application" << std::endl;
    std::cout << "-" << std::endl;
#endif
    timeval timeout;

    while (m_bEventThreadRunning) {
        fd_set read_set;
        FD_ZERO(&read_set);
        FD_SET(STDIN_FILENO, &read_set);
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;

        int ret = select(STDIN_FILENO + 1, &read_set, nullptr, nullptr, &timeout);
        if (ret == -1) {
            std::cerr << "Error selecting cin.\n";
            Quit();
        } else if (ret == 0) {
            continue;
        }
        if (FD_ISSET(STDIN_FILENO, &read_set)) {
            char line[MESSAGE_BUFFER_LENGTH];
            std::cin.getline(line, MESSAGE_BUFFER_LENGTH);
            if (std::cin.eof()) {
                std::cout << "Stdin redirecting and reaching EOF." << std::endl;
                break;
            }
            if (line[0] == 'q') {
                std::cout << "Application exiting\n";
                Quit();
            } else if (line[0] == 's') {
                std::cout << "Application suspending\n";
                Stop();
            } else if (line[0] == 'r') {
                std::cout << "Application resuming\n";
                Start();
            } else if (line[0] == 'l') {
                std::cout << "Application enter low power mode\n";
                SwitchPowerMode(true);
            } else if (line[0] == 'f') {
                std::cout << "Application enter full power mode\n";
                SwitchPowerMode(false);
            }
        }
    }
    LOG_DBG("Exit: InputEventLoop()\n");
}
