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

#ifndef CSTATUS_MANAGER_CLIENT_HPP
#define CSTATUS_MANAGER_CLIENT_HPP

#include <mutex>
#include <list>
#include <atomic>
#include <condition_variable>

#include "CStatusManagerCommon.hpp"

class CStatusManagerClientInterface
{
  public:
    virtual NvError ClientInit() = 0;
    virtual void ClientSuspend() = 0;
    virtual void ClientResume() = 0;
    virtual void ClientReInit() = 0;
    virtual void ClientRun() = 0;
    virtual void ClientDeInitPrepare() = 0;
    virtual void ClientDeInit() = 0;
    virtual void ClientEnterLowPowerMode() = 0;
    virtual void ClientEnterFullPowerMode() = 0;
};

class CStatusManagerClient
{
  public:
    NvError StatusManagerRegister(std::string &sName, CStatusManagerClientInterface *pSMClientInterface);
    void StatusManagerEventListener();

  private:
    NvError SocketConnect();
    NvError ClientEventHandle(const MsgHeader *pMsgHeader);
    int m_fd{ -1 };
    std::string m_sName;
    bool m_bRegistered{ false };
    bool m_bEventListenerRunning{ false };
    bool m_bDeinitEventReceived{ false };
    CStatusManagerClientInterface *m_pSMClientInterface{ nullptr };
    MessageType m_msgTypeNeedACK{ MessageType::MSG_TYPE_END };
    StatusMangerState m_statusManagerState{ StatusMangerState::STATUSMANAGER_STATE_INIT };
};

#endif // CSTATUS_MANAGER_CLIENT_HPP
