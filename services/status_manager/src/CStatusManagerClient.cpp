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
#include "CStatusManagerClient.hpp"

NvError CStatusManagerClient::SocketConnect()
{
    m_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (m_fd < 0) {
        LOG_ERR("Socket API failed with errno : %d\n", errno);
        return NvError_ResourceError;
    }

    sockaddr_un addr;
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (connect(m_fd, (sockaddr *)&addr, sizeof(addr)) == -1) {
        LOG_ERR("Socket connect failed with errno : %d\n", errno);
        close(m_fd);
        return NvError_ResourceError;
    }

    return NvError_Success;
}

NvError CStatusManagerClient::StatusManagerRegister(std::string &sName,
                                                    CStatusManagerClientInterface *pSMClientInterface)
{
    if (pSMClientInterface != nullptr) {
        m_pSMClientInterface = pSMClientInterface;
    } else {
        LOG_ERR("pClientInterface is nullptr!");
        return NvError_BadParameter;
    }

    m_sName = sName;

    NvError nvErr = SocketConnect();
    CHK_ERROR_AND_RETURN(nvErr, "SocketConnect");

    nvErr = SendMessageRegister(m_fd, m_sName);
    CHK_ERROR_AND_RETURN(nvErr, "SendMessageRegister");

    m_msgTypeNeedACK = MessageType::MSG_TYPE_REGISTER_ACK;

    return NvError_Success;
}

void CStatusManagerClient::StatusManagerEventListener()
{
    timeval timeout;
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;
    m_bEventListenerRunning = true;
    while (m_bEventListenerRunning) {
        fd_set read_set;
        FD_ZERO(&read_set);
        FD_SET(m_fd, &read_set);

        int ret = select(m_fd + 1, &read_set, nullptr, nullptr, &timeout);
        if (ret == -1) {
            LOG_ERR("Error selecting sockets.");
            m_bEventListenerRunning = false;
            break;
        } else if (ret == 0)
            continue;

        if (FD_ISSET(m_fd, &read_set)) {
            char msg[MESSAGE_BUFFER_SIZE];
            int iLen = recv(m_fd, msg, MESSAGE_BUFFER_SIZE - 1, 0);

            if (iLen == -1) {
                LOG_ERR("Socket read failed with errno : %d\n", errno);
                close(m_fd);
                return;
            } else if (iLen >= 0) {
                unsigned int uOffset = 0;
                while (iLen > 0 && iLen >= (int)sizeof(MsgHeader)) {
                    const MsgHeader *pMsgBase = (const MsgHeader *)(&msg[uOffset]);
                    if (iLen >= (int)pMsgBase->uMessageSize) {
                        NvError nvErr = ClientEventHandle(pMsgBase);
                        if (nvErr != NvError_Success) {
                            LOG_ERR("ClientEventHandle failed.");
                        }
                    } else {
                        LOG_ERR("iLen %d, should be larger than MessageSize %u.", iLen, pMsgBase->uMessageSize);
                    }

                    iLen -= pMsgBase->uMessageSize;
                    uOffset += pMsgBase->uMessageSize;
                }
            }
        }
    }

    close(m_fd);
}

NvError CStatusManagerClient::ClientEventHandle(const MsgHeader *pMsgBase)
{
    if (m_msgTypeNeedACK != MessageType::MSG_TYPE_END && pMsgBase->type != m_msgTypeNeedACK) {
        LOG_ERR("Need to recive ACK %d for previous set!", m_msgTypeNeedACK);
        return NvError_ResourceError;
    }

    NvError nvErr = NvError_Success;

    switch (pMsgBase->type) {
        case MessageType::MSG_TYPE_REGISTER_ACK: {
            LOG_MSG("Register to server success!");
            m_bRegistered = true;

            m_pSMClientInterface->ClientInit();
            // Once app registered, change to init done state.
            nvErr = SendMessageState(m_fd, MessageType::MSG_TYPE_STATE_SET,
                                     StatusMangerState::STATUSMANAGER_STATE_INIT_DONE);
            if (nvErr != NvError_Success) {
                LOG_ERR("SendMessageState fail!");
                break;
            }
            m_statusManagerState = StatusMangerState::STATUSMANAGER_STATE_INIT_DONE;

            m_msgTypeNeedACK = MessageType::MSG_TYPE_STATE_SET_ACK;
            break;
        }
        case MessageType::MSG_TYPE_STATE_SET_ACK:
            LOG_INFO("ACK for status_manager received!\n");
            m_msgTypeNeedACK = MessageType::MSG_TYPE_END;
            if (m_bDeinitEventReceived) {
                m_bEventListenerRunning = false;
            }
            break;
        case MessageType::MSG_TYPE_STATE_SET: {
            if (m_bRegistered != true || m_pSMClientInterface == nullptr) {
                LOG_ERR("Need to register client first!");
                nvErr = NvError_ResourceError;
                break;
            }

            const MsgState *pMsgState = reinterpret_cast<const MsgState *>(pMsgBase);
            switch (pMsgState->stateType) {
                case StatusMangerState::STATUSMANAGER_STATE_INIT_DONE: {
                    nvErr = SendMessageState(m_fd, MessageType::MSG_TYPE_STATE_SET,
                                             StatusMangerState::STATUSMANAGER_STATE_INIT_DONE);
                    if (nvErr != NvError_Success) {
                        LOG_ERR("SendStateSetMsgAndGetACK fail %d.", nvErr);
                        break;
                    }
                    m_statusManagerState = StatusMangerState::STATUSMANAGER_STATE_INIT_DONE;
                    break;
                }
                case StatusMangerState::STATUSMANAGER_STATE_OPERATIONAL: {
                    m_pSMClientInterface->ClientRun();

                    nvErr = SendMessageState(m_fd, MessageType::MSG_TYPE_STATE_SET,
                                             StatusMangerState::STATUSMANAGER_STATE_OPERATIONAL);
                    if (nvErr != NvError_Success) {
                        LOG_ERR("SendStateSetMsgAndGetACK fail %d.", nvErr);
                        break;
                    }

                    m_statusManagerState = StatusMangerState::STATUSMANAGER_STATE_OPERATIONAL;
                    break;
                }
                case StatusMangerState::STATUSMANAGER_STATE_SUSPEND: {
                    m_pSMClientInterface->ClientSuspend();

                    nvErr = SendMessageState(m_fd, MessageType::MSG_TYPE_STATE_SET,
                                             StatusMangerState::STATUSMANAGER_STATE_SUSPEND);
                    if (nvErr != NvError_Success) {
                        LOG_ERR("SendStateSetMsgAndGetACK fail %d.", nvErr);
                        break;
                    }

                    m_statusManagerState = StatusMangerState::STATUSMANAGER_STATE_SUSPEND;
                    break;
                }
                case StatusMangerState::STATUSMANAGER_STATE_RESUME: {
                    m_pSMClientInterface->ClientResume();

                    nvErr = SendMessageState(m_fd, MessageType::MSG_TYPE_STATE_SET,
                                             StatusMangerState::STATUSMANAGER_STATE_RESUME);
                    if (nvErr != NvError_Success) {
                        LOG_ERR("SendStateSetMsgAndGetACK fail %d.", nvErr);
                        break;
                    }

                    m_statusManagerState = StatusMangerState::STATUSMANAGER_STATE_RESUME;
                    break;
                }
                case StatusMangerState::STATUSMANAGER_STATE_LOW_POWER: {
                    m_pSMClientInterface->ClientEnterLowPowerMode();

                    nvErr = SendMessageState(m_fd, MessageType::MSG_TYPE_STATE_SET,
                                             StatusMangerState::STATUSMANAGER_STATE_LOW_POWER);
                    if (nvErr != NvError_Success) {
                        LOG_ERR("SendStateSetMsgAndGetACK fail %d.", nvErr);
                        break;
                    }

                    m_statusManagerState = StatusMangerState::STATUSMANAGER_STATE_LOW_POWER;
                    break;
                }
                case StatusMangerState::STATUSMANAGER_STATE_FULL_POWER: {
                    m_pSMClientInterface->ClientEnterFullPowerMode();

                    nvErr = SendMessageState(m_fd, MessageType::MSG_TYPE_STATE_SET,
                                             StatusMangerState::STATUSMANAGER_STATE_FULL_POWER);
                    if (nvErr != NvError_Success) {
                        LOG_ERR("SendStateSetMsgAndGetACK fail %d.", nvErr);
                        break;
                    }

                    m_statusManagerState = StatusMangerState::STATUSMANAGER_STATE_FULL_POWER;
                    break;
                }
                case StatusMangerState::STATUSMANAGER_STATE_DEINIT_PREPARE: {
                    m_pSMClientInterface->ClientDeInitPrepare();

                    nvErr = SendMessageState(m_fd, MessageType::MSG_TYPE_STATE_SET,
                                             StatusMangerState::STATUSMANAGER_STATE_DEINIT_PREPARE);
                    if (nvErr != NvError_Success) {
                        LOG_ERR("SendStateSetMsgAndGetACK fail %d.", nvErr);
                        break;
                    }

                    m_statusManagerState = StatusMangerState::STATUSMANAGER_STATE_DEINIT_PREPARE;
                    break;
                }
                case StatusMangerState::STATUSMANAGER_STATE_REINIT: {
                    m_pSMClientInterface->ClientReInit();

                    nvErr = SendMessageState(m_fd, MessageType::MSG_TYPE_STATE_SET,
                                             StatusMangerState::STATUSMANAGER_STATE_REINIT);
                    if (nvErr != NvError_Success) {
                        LOG_ERR("SendStateSetMsgAndGetACK fail %d.", nvErr);
                        break;
                    }

                    m_statusManagerState = StatusMangerState::STATUSMANAGER_STATE_REINIT;
                    break;
                }
                case StatusMangerState::STATUSMANAGER_STATE_DEINIT: {
                    m_pSMClientInterface->ClientDeInit();

                    nvErr = SendMessageState(m_fd, MessageType::MSG_TYPE_STATE_SET,
                                             StatusMangerState::STATUSMANAGER_STATE_DEINIT);
                    if (nvErr != NvError_Success) {
                        LOG_ERR("SendStateSetMsgAndGetACK fail %d.", nvErr);
                        break;
                    }

                    m_bDeinitEventReceived = true;
                    m_statusManagerState = StatusMangerState::STATUSMANAGER_STATE_DEINIT;
                    break;
                }
                default:
                    LOG_ERR("Wrong msg state type %d received.", pMsgState->stateType);
                    nvErr = NvError_BadParameter;
                    break;
            }

            if (nvErr == NvError_Success) {
                m_msgTypeNeedACK = MessageType::MSG_TYPE_STATE_SET_ACK;
            }
        }
        default:
            break;
    }

    return nvErr;
}
