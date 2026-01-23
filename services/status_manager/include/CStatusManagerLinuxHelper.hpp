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

#ifndef CSTATUS_MANAGER_LINUX_HELPER_HPP
#define CSTATUS_MANAGER_LINUX_HELPER_HPP

#include <condition_variable>
#include <atomic>
#include <linux/netlink.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "CStatusManagerCommon.hpp"
#include "CStatusManagerService.hpp"

#define NETLINK_USERSPACE_PM 30
#define MAX_PAYLOAD 1024

char *sock_recv(int sock_fd, struct nlmsghdr *nlh);
void sock_send(int sock_fd, struct nlmsghdr *nlh, const char *string);

class CStatusManagerLinuxHelper : public CStatusManagerOsHelper
{
  public:
    virtual NvError Init() override;
    virtual NvError DeInit() override;
    virtual NvError SetOsDvmsState(StatusMangerState statusManagerState) override;
    virtual NvError WaitForResume() override;

  private:
    void KernelEventHandleFunc();
    void HandleKernelEvent(int fd, struct nlmsghdr *nlh);
    NvError TriggerSuspend();
    NvError SetPowerMode(StatusMangerState statusManagerState);

    int m_termiatefd[2] = { -1, -1 };
    int m_socketFd = -1;
    std::vector<pollfd> m_vPollFds;
    std::mutex m_statusManagerOsResumeMutex;
    std::atomic<bool> m_bKernelEventThreadRunning{ false };
    std::atomic<bool> m_bStatusManagerOsResumeDone{ false };
    std::condition_variable m_statusManagerOsResumeConditionVar;
    std::unique_ptr<std::thread> m_linuxHelperThread = nullptr;
};
#endif //CSTATUS_MANAGER_LINUX_HELPER_HPP
