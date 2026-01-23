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

#ifndef CSTATUSMANAGER_HPP
#define CSTATUSMANAGER_HPP

#include "CManager.hpp"
#include "CStatusManagerClient.hpp"

class CStatusManager : public CManager, public CStatusManagerClientInterface
{
  public:
    CStatusManager(std::shared_ptr<CAppCfg> spAppConfig, EventCallback pEventCallback = nullptr);
    ~CStatusManager();

    NvError Init() override;
    void Run() override;
    void Quit() override;
    void DeInit() override;

  private:
    NvError ClientInit() override;
    void ClientSuspend() override;
    void ClientResume() override;
    void ClientReInit() override;
    void ClientRun() override;
    void ClientDeInitPrepare() override;
    void ClientDeInit() override;
    void ClientEnterLowPowerMode() override;
    void ClientEnterFullPowerMode() override;
    void OnEvent(CChannel *pChannel, EventStatus event) override;
    bool IsControlChannelNeeded() override { return false; }

    std::mutex m_StatusManagerMutex;
    std::condition_variable m_StatusManagerConditionVar;
    std::atomic<bool> m_bStatusManagerInitDone{ false };
    std::shared_ptr<CStatusManagerClient> m_spStatusManagerClient{ nullptr };
};
#endif // CSTATUSMANAGER_HPP
