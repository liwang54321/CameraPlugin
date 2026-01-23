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

#include "CStatusManager.hpp"
#include "CLogger.hpp"
#include "CUtils.hpp"

CStatusManager::CStatusManager(std::shared_ptr<CAppCfg> spAppConfig, EventCallback pEventCallback)
    : CManager(spAppConfig, pEventCallback)
{
}

CStatusManager::~CStatusManager() = default;

NvError CStatusManager::Init()
{
    NvError error = NvError_Success;

    LOG_DBG("Enter: CStatusManager::Init()\n");

    m_spStatusManagerClient = std::make_shared<CStatusManagerClient>();

    std::string sClientName("nvsipl_multicast");
    error = m_spStatusManagerClient->StatusManagerRegister(sClientName, this);
    if (error != NvError_Success) {
        LOG_ERR("CStatusManagerClient::StatusManagerRegister failed!\n");
        return error;
    }

    m_spAppCfg->SetCudaRunningFlag(false);

    error = CManager::Init();
    if (error != NvError_Success) {
        LOG_ERR("CManager init failed!\n");
        return error;
    }

    LOG_DBG("Exit: CStatusManager::Init()\n");

    return NvError_Success;
}

void CStatusManager::Quit()
{
    LOG_DBG("Enter: CStatusManager::Quit()\n");

    return;

    LOG_DBG("Exit: CStatusManager::Quit()\n");
}

void CStatusManager::Run()
{
    LOG_DBG("Enter: CStatusManager::Run()\n");

    m_spStatusManagerClient->StatusManagerEventListener();

    LOG_DBG("Exit: CStatusManager::Run()\n");
}

void CStatusManager::DeInit()
{
    LOG_DBG("Enter: CStatusManager::DeInit()\n");

    m_spStatusManagerClient.reset();

    LOG_DBG("Exit: CStatusManager::DeInit()\n");
}

NvError CStatusManager::ClientInit()
{
    NvError error = NvError_Success;
    LOG_DBG("Enter: CStatusManager::ClientInit()\n");

    m_bMonitorThreadRunning = true;
    m_bMonitorThreadPause = false;
    if (m_upMonitorThread == nullptr) {
        m_upMonitorThread.reset(new std::thread(&CManager::MonitorThreadFunc, this));
    }

    std::unique_lock<std::mutex> lock(m_StatusManagerMutex);
    while (!m_bStatusManagerInitDone) {
        // Make sure currently multicast has been started,
        // So that the application status could be set to InitDone.
        m_StatusManagerConditionVar.wait(lock);
    }
    lock.unlock();

    LOG_DBG("Exit: CStatusManager::ClientInit()\n");
    return error;
}

void CStatusManager::ClientRun()
{
    LOG_DBG("Enter: CStatusManager::ClientRun()\n");

    // When StatusManager is enabled, the monitor is already running at Init;
    // And all event will be passed from status_manger.
    // For fast camera streaming, 1st time, Init/Runtime will run together.
    // What we need to do at run time is to turn on the GPU tasks after switching to operational state.
    m_spAppCfg->SetCudaRunningFlag(true);

    LOG_DBG("Exit: CStatusManager::ClientRun()\n");
}

void CStatusManager::ClientSuspend()
{
    LOG_DBG("Enter: CStatusManager::ClientSuspend()\n");

    CManager::Stop();

    LOG_DBG("Exit: CStatusManager::ClientSuspend()\n");
}

void CStatusManager::ClientResume()
{
    LOG_DBG("Enter: CStatusManager::ClientResume()\n");

    CManager::Start();

    LOG_DBG("Exit: CStatusManager::ClientResume()\n");
}

void CStatusManager::ClientReInit()
{
    LOG_DBG("Enter: CStatusManager::ClientReInit()\n");

    m_spAppCfg->SetCudaRunningFlag(false);
    CManager::Start();

    std::unique_lock<std::mutex> lock(m_StatusManagerMutex);
    m_bStatusManagerInitDone = false;
    while (!m_bStatusManagerInitDone) {
        // Make sure currently multicast has been started after reinit.
        m_StatusManagerConditionVar.wait(lock);
    }
    lock.unlock();

    LOG_DBG("Exit: CStatusManager::ClientReInit()\n");
}

void CStatusManager::ClientDeInitPrepare()
{
    LOG_DBG("Enter: CStatusManager::ClientDeInitPrepare()\n");

    Stop();

    LOG_DBG("Exit: CStatusManager::ClientDeInitPrepare()\n");
}

void CStatusManager::ClientDeInit()
{
    LOG_DBG("Enter: CStatusManager::ClientDeInit()\n");

    CManager::Quit();
    // Make sure the quit done and monitor thread is joined.
    m_upMonitorThread->join();

    CManager::DeInit();

    LOG_DBG("Exit: CStatusManager::ClientDeInit()\n");
}

void CStatusManager::ClientEnterLowPowerMode()
{
    LOG_DBG("Enter: CStatusManager::ClientEnterLowPowerMode()\n");

    LOG_MSG("Application enter low power mode\n");
    CManager::SwitchPowerMode(true);

    LOG_DBG("Exit: CStatusManager::ClientEnterLowPowerMode()\n");
}

void CStatusManager::ClientEnterFullPowerMode()
{
    LOG_DBG("Enter: CStatusManager::ClientEnterFullPowerMode()\n");

    LOG_MSG("Application enter full power mode\n");
    CManager::SwitchPowerMode(false);

    LOG_DBG("Exit: CStatusManager::ClientEnterFullPowerMode()\n");
}

void CStatusManager::OnEvent(CChannel *pChannel, EventStatus event)
{
    LOG_DBG("Enter: CStatusManager::OnEvent()\n");
    CManager::OnEvent(pChannel, event);

    // Make sure the multicast has been started.
    if (event == EventStatus::STARTED) {
        m_bStatusManagerInitDone = true;
        m_StatusManagerConditionVar.notify_all();
    }

    LOG_DBG("Exit: CStatusManager::OnEvent()\n");
}
