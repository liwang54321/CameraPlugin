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
#ifndef CSTATUS_MANAGER_COMMON_HPP
#define CSTATUS_MANAGER_COMMON_HPP

#include <iostream>
#include <string>
#include <set>
#include <unistd.h>
#include <stdarg.h>
#include <poll.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/select.h>
#include <fcntl.h>
#include <cstring>
#include <vector>
#include <algorithm>
#include <memory>
#include <thread>
#include <mutex>

#include "nverror.h"
#include "CLogger.hpp"

// For qnx safety, we must use the path that is allow_attached by io_pkt_v6_hc_eqos_0_t
// This config is located in file policy_debug_orin_gos_vm_safety.txt on qnx safety.
#define SOCKET_PATH "/tmp/nvsocket_routed"

#define MESSAGE_BUFFER_SIZE 256
#define INVALID_FD -1

#define CHK_ERROR_AND_RETURN(error, api)                   \
    if ((error) != NvError_Success) {                      \
        LOG_ERR("%s failed, error: %u\n", (api), (error)); \
        return (error);                                    \
    }

#define CHK_DVMSSTATUS_AND_RETURN(nvDvmsStatus, api)               \
    if ((nvDvmsStatus) != NvDvmsSuccess) {                         \
        LOG_ERR("%s failed, status: %x\n", (api), (nvDvmsStatus)); \
        return NvError_ResourceError;                              \
    }

enum class StatusMangerState : uint8_t
{
    STATUSMANAGER_STATE_INIT = 0,
    STATUSMANAGER_STATE_INIT_DONE,
    STATUSMANAGER_STATE_REINIT,
    STATUSMANAGER_STATE_SUSPEND,
    STATUSMANAGER_STATE_RESUME,
    STATUSMANAGER_STATE_OPERATIONAL,
    STATUSMANAGER_STATE_LOW_POWER,
    STATUSMANAGER_STATE_FULL_POWER,
    STATUSMANAGER_STATE_DEINIT_PREPARE,
    STATUSMANAGER_STATE_DEINIT
};

enum class MessageType : uint8_t
{
    MSG_TYPE_REGISTER = 0,
    MSG_TYPE_REGISTER_ACK,
    MSG_TYPE_STATE_SET,
    MSG_TYPE_STATE_SET_ACK,
    MSG_TYPE_STATE_GET,
    MSG_TYPE_STATE_GET_RETURN,
    MSG_TYPE_END = 0xFF,
};

typedef struct
{
    MessageType type;
    uint32_t uMessageSize;
} MsgHeader;

typedef struct
{
    MsgHeader header;
    StatusMangerState stateType;
} MsgState;

typedef struct
{
    MsgHeader header;
    char clientName[0];
} MsgRegister;

NvError SendMessageBase(int fd, MessageType msgType);
NvError SendMessageRegister(int fd, std::string &sClientName);
NvError SendMessageState(int fd, MessageType msgType, StatusMangerState stateType);
NvError ValidateStateSwitch(StatusMangerState preStateType, StatusMangerState nextStateType);

#endif // CSTATUS_MANAGER_COMMON_HPP
