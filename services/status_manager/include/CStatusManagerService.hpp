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

#ifndef CSTATUS_MANAGER_SERVICE_HPP
#define CSTATUS_MANAGER_SERVICE_HPP

#include <getopt.h>
#include "CStatusManagerCommon.hpp"

void ShowUsage();
class CStatusManagerOsHelper
{
  public:
    virtual NvError Init() { return NvError_Success; }
    virtual NvError DeInit() { return NvError_Success; }
    virtual NvError CheckOsInitDoneStatus() { return NvError_Success; }
    virtual NvError SetOsDvmsState(StatusMangerState statusManagerState) = 0;
    virtual NvError WaitForResume() = 0;
};

class CStatusManagerService
{
  public:
    CStatusManagerService(uint32_t uNumClients)
        : m_uNumClients(uNumClients)
    {
    }

    struct AppClient
    {
        AppClient(int fd)
            : m_fd(fd) {};
        int m_fd;
        StatusMangerState m_state = StatusMangerState::STATUSMANAGER_STATE_INIT;
        std::string m_sName;
    };

  public:
    NvError Init();
    NvError Start();
    NvError DeInit();

  private:
    void HandleClientConnection();
    void HandleClientEvent(int fd);
    void NotifyClientStateFromInput();

    NvError InitSocket();
    NvError NotifyClients(StatusMangerState statusManagerState);

    int m_serverFd;
    std::vector<pollfd> m_vPollFds;
    std::vector<AppClient> m_vAppClients;
    std::shared_ptr<CStatusManagerOsHelper> m_spStatusManagerOsHelper;

    uint32_t m_uNumClients{ 1 };
    uint32_t m_clientsInitDoneCount{ 0 };
    uint32_t m_clientsOperationalCount{ 0 };
    uint32_t m_clientsReInitCount{ 0 };
    uint32_t m_clientsSuspendCount{ 0 };
    uint32_t m_clientsResumeCount{ 0 };
    uint32_t m_clientsLowPowerCount{ 0 };
    uint32_t m_clientsFullPowerCount{ 0 };
    uint32_t m_clientsDeinitPrepareCount{ 0 };
    uint32_t m_clientsDeinitCount{ 0 };
    bool m_bEventThreadRunning{ false };
    StatusMangerState m_statusManagerDvmsState{ StatusMangerState::STATUSMANAGER_STATE_INIT };
};
#endif // #define CSTATUS_MANAGER_SERVICE_HPP
