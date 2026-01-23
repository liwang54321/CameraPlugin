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

#include "CStatusManagerLinuxHelper.hpp"
static constexpr const char *registerReq = "PM Register";
static constexpr const char *deRegisterReq = "PM Deregister";
#ifdef NV_BUILD_DOS7
static constexpr const char *suspendReq = "Suspend Reques";
static constexpr const char *resumeReq = "Resume Reques";
#else
static constexpr const char *suspendReq = "Suspend Request";
static constexpr const char *resumeReq = "Resume Request";
#endif
static constexpr const char *suspendResp = "Suspend Response";
static constexpr const char *resumeResp = "Resume Response";

NvError CStatusManagerLinuxHelper::Init()
{
    int ret = pipe(m_termiatefd);
    if (ret < 0) {
        LOG_ERR("pipe failed: %d.\n", errno);
        return NvError_ResourceError;
    }
    m_vPollFds.push_back({ m_termiatefd[0], POLLIN, 0 });

    m_bKernelEventThreadRunning = true;
    m_linuxHelperThread.reset(new std::thread(&CStatusManagerLinuxHelper::KernelEventHandleFunc, this));

    return NvError_Success;
}

NvError CStatusManagerLinuxHelper::DeInit()
{
    struct nlmsghdr *nlh;
    nlh = (struct nlmsghdr *)malloc(NLMSG_SPACE(MAX_PAYLOAD));
    sock_send(m_socketFd, nlh, deRegisterReq);

    m_bKernelEventThreadRunning = false;
    int iTerminateValue = 1;
    int iBytes = write(m_termiatefd[1], &iTerminateValue, sizeof(iTerminateValue));
    if (iBytes < 0) {
        LOG_ERR("Terminate pipe write failure.");
        return NvError_ResourceError;
    }

    if (m_linuxHelperThread->joinable()) {
        m_linuxHelperThread->join();
        m_linuxHelperThread.reset();
    }

    return NvError_Success;
}

NvError CStatusManagerLinuxHelper::SetOsDvmsState(StatusMangerState statusManagerState)
{
    NvError err = NvError_Success;
    // Linux only need to handle suspend and power mode transition.
    switch (statusManagerState) {
        case StatusMangerState::STATUSMANAGER_STATE_SUSPEND:
            err = TriggerSuspend();
            break;
        case StatusMangerState::STATUSMANAGER_STATE_LOW_POWER:
        case StatusMangerState::STATUSMANAGER_STATE_FULL_POWER:
            err = SetPowerMode(statusManagerState);
            break;
        default:
            break;
    }

    return err;
}

NvError CStatusManagerLinuxHelper::WaitForResume()
{
    LOG_DBG("Enter: CStatusManagerLinuxHelper::WaitForResume()");

    std::unique_lock<std::mutex> lock(m_statusManagerOsResumeMutex);
    while (!m_bStatusManagerOsResumeDone) {
        // Wait for Os Resume.
        m_statusManagerOsResumeConditionVar.wait(lock);
    }

    m_bStatusManagerOsResumeDone = false;
    LOG_DBG("Exit: CStatusManagerLinuxHelper::WaitForResume()");
    return NvError_Success;
}

void CStatusManagerLinuxHelper::HandleKernelEvent(int fd, struct nlmsghdr *nlh)
{
    char *pRecvStr = sock_recv(fd, nlh);
    if (pRecvStr) {
        if (strcmp(pRecvStr, suspendReq) == 0) {
            sock_send(fd, nlh, suspendResp);
        } else if (strcmp(pRecvStr, resumeReq) == 0) {
            sock_send(fd, nlh, resumeResp);
            m_bStatusManagerOsResumeDone = true;
            m_statusManagerOsResumeConditionVar.notify_all();
        } else {
            LOG_ERR("Unknown message %s", pRecvStr);
        }
    } else {
        LOG_ERR("Null string received");
    }
}

void CStatusManagerLinuxHelper::KernelEventHandleFunc()
{
    pthread_setname_np(pthread_self(), "KernelEventHandleFunc");

    m_socketFd = socket(PF_NETLINK, SOCK_RAW, NETLINK_USERSPACE_PM);
    if (m_socketFd < 0) {
        std::cerr << "Socket API failed with errno :" << errno << std::endl;
        return;
    }

    struct sockaddr_nl src_addr;
    src_addr.nl_family = AF_NETLINK;
    src_addr.nl_pid = getpid(); /* self pid */
    src_addr.nl_groups = 0;     /* unicast */

    if (bind(m_socketFd, (struct sockaddr *)&src_addr, sizeof(src_addr)) < 0) {
        std::cerr << "Socket bind failed with errno :" << errno << std::endl;
        close(m_socketFd);
        return;
    }

    struct nlmsghdr *nlh;
    nlh = (struct nlmsghdr *)malloc(NLMSG_SPACE(MAX_PAYLOAD));
    sock_send(m_socketFd, nlh, registerReq);

    m_vPollFds.push_back({ m_socketFd, POLLIN, 0 });

    while (m_bKernelEventThreadRunning) {
        std::vector<pollfd> vPollFds = m_vPollFds;
        int ret = poll(vPollFds.data(), vPollFds.size(), -1);
        if (ret == -1) {
            LOG_ERR("Error selecting sockets.");
            break;
        }

        for (auto pollFd : vPollFds) {
            if ((pollFd.revents & POLLIN) == 0) {
                continue;
            }

            if (pollFd.fd == m_termiatefd[0]) {
                int iTerminateValue;
                int iBytes = read(m_termiatefd[0], &iTerminateValue, sizeof(iTerminateValue));
                if (iBytes < 0) {
                    LOG_ERR("Terminal pipe read failure.");
                }
                m_bKernelEventThreadRunning = false;
                break;
            } else {
                HandleKernelEvent(pollFd.fd, nlh);
            }
        }
    }
}

NvError CStatusManagerLinuxHelper::TriggerSuspend()
{
    struct FileOperation
    {
        const char *path;
        const char *value;
    };

    const FileOperation operations[] = {
#ifdef NV_BUILD_DOS7
        { "/sys/module/printk/parameters/console_suspend", "N" },
        { "/proc/sys/kernel/printk", "8" },
#endif
        { "/sys/class/tegra_hv_pm_ctl/tegra_hv_pm_ctl/device/trigger_sys_suspend", "1" }
    };
    for (const auto &op : operations) {
        int fd = -1;
        fd = open(op.path, O_WRONLY);
        if (fd != -1) {
            ssize_t valueLen = strlen(op.value);
            if (write(fd, op.value, valueLen) != valueLen) {
                LOG_ERR("Error while writing to file: %s\n", op.path);
                close(fd);
                return NvError_ResourceError;
            }
            close(fd);
            sleep(1);
        } else {
            LOG_ERR("Error while open %s\n", op.path);
            return NvError_ResourceError;
        }
    }

    return NvError_Success;
}

NvError CStatusManagerLinuxHelper::SetPowerMode(StatusMangerState statusManagerState)
{
    if (statusManagerState == StatusMangerState::STATUSMANAGER_STATE_LOW_POWER) {
        int ret = system("/usr/bin/nvpmodel -m 1 --verbose");
        if (ret != 0) {
            LOG_ERR("Failed to set low power mode\n");
            return NvError_InvalidState;
        }
    } else if (statusManagerState == StatusMangerState::STATUSMANAGER_STATE_FULL_POWER) {
        int ret = system("/usr/bin/nvpmodel -m 0 --verbose");
        if (ret != 0) {
            LOG_ERR("Failed to set full power mode\n");
            return NvError_InvalidState;
        }
    } else {
        return NvError_BadParameter;
    }
    return NvError_Success;
}

static void prepare_msg(
    struct msghdr *msg, struct nlmsghdr *nlh, struct sockaddr_nl *dest_addr, struct iovec *iov, const char *string)
{
    memset(nlh, 0, NLMSG_SPACE(MAX_PAYLOAD));
    memset(dest_addr, 0, sizeof(*dest_addr));
    memset(iov, 0, sizeof(*iov));
    memset(msg, 0, sizeof(*msg));

    if (string != NULL)
        strncpy((char *)NLMSG_DATA(nlh), string, MAX_PAYLOAD);

    nlh->nlmsg_len = NLMSG_SPACE(MAX_PAYLOAD);
    nlh->nlmsg_pid = getpid();

    dest_addr->nl_family = AF_NETLINK;
    dest_addr->nl_pid = 0;    /* self pid */
    dest_addr->nl_groups = 0; /* unicast */

    iov->iov_base = (void *)nlh;
    iov->iov_len = nlh->nlmsg_len;
    msg->msg_name = (void *)dest_addr;
    msg->msg_namelen = sizeof(*dest_addr);
    msg->msg_iov = iov;
    msg->msg_iovlen = 1;
}

char *sock_recv(int sock_fd, struct nlmsghdr *nlh)
{
    struct sockaddr_nl dest_addr;
    struct iovec iov;
    struct msghdr msg;

    prepare_msg(&msg, nlh, &dest_addr, &iov, NULL);
    int ret = recvmsg(sock_fd, &msg, 0);
    if (ret < 0) {
        LOG_ERR("Failed to recvmsg from %d\n", sock_fd);
        return nullptr;
    }

    LOG_INFO("pm_service: from kernel: %s\n", (char *)NLMSG_DATA(nlh));
    return (char *)NLMSG_DATA(nlh);
}

void sock_send(int sock_fd, struct nlmsghdr *nlh, const char *string)
{
    struct sockaddr_nl dest_addr;
    struct iovec iov;
    struct msghdr msg;

    prepare_msg(&msg, nlh, &dest_addr, &iov, string);
    int ret = sendmsg(sock_fd, &msg, 0);
    if (ret < 0) {
        LOG_ERR("Failed to sendmsg to %d\n", sock_fd);
    } else {
        LOG_INFO("pm_service: to kernel: %s\n", string);
    }
}
