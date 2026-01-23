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

#include "CStatusManagerCommon.hpp"

NvError SendMessageBase(int fd, MessageType msgType)
{
    MsgHeader msgBase;
    msgBase.type = msgType;
    msgBase.uMessageSize = sizeof(MsgHeader);

    int bytesSent = 0;
    do {
        bytesSent = send(fd, &msgBase, sizeof(MsgHeader), 0);
    } while (bytesSent == -1 && errno == EINTR);

    if (bytesSent != sizeof(MsgHeader)) {
        LOG_ERR("Failed to send header msg to client, errno %d.", errno);
        return NvError_ResourceError;
    }

    LOG_INFO("MsgHeader was sent and the msgType is %d.", msgType);
    return NvError_Success;
}

NvError SendMessageState(int fd, MessageType msgType, StatusMangerState stateType)
{
    MsgState msgState;
    msgState.header.type = msgType;
    msgState.header.uMessageSize = sizeof(MsgState);
    msgState.stateType = stateType;

    int bytesSent = 0;
    do {
        bytesSent = send(fd, &msgState, sizeof(MsgState), 0);
    } while (bytesSent == -1 && errno == EINTR);

    if (bytesSent != sizeof(MsgState)) {
        LOG_ERR("Failed to send state msg, errno %d.", errno);
        return NvError_ResourceError;
    }

    LOG_INFO("Sent MessageType %d, MsgState is %d.", msgType, stateType);
    return NvError_Success;
}

NvError SendMessageRegister(int fd, std::string &sClientName)
{
    uint32_t uMsgRegisterSize = sizeof(MsgState) + sClientName.length();
    MsgRegister *msgRegister = (MsgRegister *)calloc(1, uMsgRegisterSize);
    msgRegister->header.type = MessageType::MSG_TYPE_REGISTER;
    msgRegister->header.uMessageSize = uMsgRegisterSize;
    memcpy(msgRegister->clientName, sClientName.c_str(), sClientName.length());

    int bytesSent = 0;
    do {
        bytesSent = send(fd, msgRegister, uMsgRegisterSize, 0);
    } while (bytesSent == -1 && errno == EINTR);

    if (bytesSent != (int)uMsgRegisterSize) {
        LOG_ERR("Failed to send register to server, bytes sent %d,\
            bytes %d need to be sent, errno %u.",
                bytesSent, uMsgRegisterSize, errno);
        free(msgRegister);
        return NvError_InvalidSize;
    }

    free(msgRegister);
    return NvError_Success;
}

NvError ValidateStateSwitch(StatusMangerState curStateType, StatusMangerState nextStateType)
{
    NvError nvErr = NvError_Success;
    switch (nextStateType) {
        case StatusMangerState::STATUSMANAGER_STATE_INIT_DONE:
            if (curStateType != StatusMangerState::STATUSMANAGER_STATE_INIT &&
                curStateType != StatusMangerState::STATUSMANAGER_STATE_RESUME &&
                curStateType != StatusMangerState::STATUSMANAGER_STATE_REINIT &&
                curStateType != StatusMangerState::STATUSMANAGER_STATE_FULL_POWER) {
                nvErr = NvError_InvalidState;
            }
            break;
        case StatusMangerState::STATUSMANAGER_STATE_OPERATIONAL:
            if (curStateType != StatusMangerState::STATUSMANAGER_STATE_INIT_DONE) {
                nvErr = NvError_InvalidState;
            }
            break;
        case StatusMangerState::STATUSMANAGER_STATE_REINIT:
            if (curStateType != StatusMangerState::STATUSMANAGER_STATE_DEINIT_PREPARE) {
                nvErr = NvError_InvalidState;
            }
            break;
        case StatusMangerState::STATUSMANAGER_STATE_SUSPEND:
            if (curStateType != StatusMangerState::STATUSMANAGER_STATE_INIT_DONE) {
                nvErr = NvError_InvalidState;
            }
            break;
        case StatusMangerState::STATUSMANAGER_STATE_RESUME:
            if (curStateType != StatusMangerState::STATUSMANAGER_STATE_SUSPEND) {
                nvErr = NvError_InvalidState;
            }
            break;
        case StatusMangerState::STATUSMANAGER_STATE_DEINIT_PREPARE:
            if (curStateType != StatusMangerState::STATUSMANAGER_STATE_INIT_DONE &&
                curStateType != StatusMangerState::STATUSMANAGER_STATE_OPERATIONAL &&
                curStateType != StatusMangerState::STATUSMANAGER_STATE_INIT) {
                nvErr = NvError_InvalidState;
            }
            break;
        case StatusMangerState::STATUSMANAGER_STATE_DEINIT:
            if (curStateType != StatusMangerState::STATUSMANAGER_STATE_DEINIT_PREPARE) {
                nvErr = NvError_InvalidState;
            }
            break;
        case StatusMangerState::STATUSMANAGER_STATE_LOW_POWER:
            if (curStateType != StatusMangerState::STATUSMANAGER_STATE_INIT_DONE) {
                nvErr = NvError_InvalidState;
            }
            break;
        case StatusMangerState::STATUSMANAGER_STATE_FULL_POWER:
            if (curStateType != StatusMangerState::STATUSMANAGER_STATE_LOW_POWER) {
                nvErr = NvError_InvalidState;
            }
            break;
        default:
            nvErr = NvError_InvalidState;
            break;
    }

    if (nvErr != NvError_Success) {
        LOG_ERR("Next StatusMangerState state %d is not reasonable to switch from previous StatusMangerState state %d.",
                nextStateType, curStateType);
    }

    return nvErr;
}
