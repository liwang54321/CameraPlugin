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

#include "CStatusManagerService.hpp"

#ifdef NVMEDIA_QNX
#include "CStatusManagerQnxHelper.hpp"
#else
#include "CStatusManagerLinuxHelper.hpp"
#endif // NVMEDIA_QNX

constexpr int MAX_CLIENTS = 64;
constexpr int MILLISECONDS_TIMEOUT = 2000;

static const std::set<std::string> sClientNamesAllowList = { "nvsipl_multicast" };

NvError CStatusManagerService::InitSocket()
{
    unlink(SOCKET_PATH);
    m_serverFd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (m_serverFd == -1) {
        LOG_ERR("Error creating server socket.");
        return NvError_ResourceError;
    }

    sockaddr_un address;

    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    strncpy(address.sun_path, SOCKET_PATH, sizeof(address.sun_path) - 1);

    if (bind(m_serverFd, (sockaddr *)&address, sizeof(address)) == -1) {
        LOG_ERR("Error binding server socket, errno %d.", errno);
        close(m_serverFd);
        return NvError_ResourceError;
    }

    LOG_INFO("Listening on " SOCKET_PATH);
    if (listen(m_serverFd, MAX_CLIENTS) == -1) {
        LOG_ERR("Error listening for connections, errno %d.", errno);
        close(m_serverFd);
        return NvError_ResourceError;
    }

    m_vPollFds.push_back({ m_serverFd, POLLIN, 0 });

    return NvError_Success;
}

NvError CStatusManagerService::Init()
{

#ifdef NVMEDIA_QNX
    m_spStatusManagerOsHelper = std::make_shared<CStatusManagerQnxHelper>();
#else
    m_spStatusManagerOsHelper = std::make_shared<CStatusManagerLinuxHelper>();
#endif

    NvError nvErr = m_spStatusManagerOsHelper->Init();
    CHK_ERROR_AND_RETURN(nvErr, "m_spStatusManagerOsHelper->Init");

    nvErr = m_spStatusManagerOsHelper->CheckOsInitDoneStatus();
    CHK_ERROR_AND_RETURN(nvErr, "m_spStatusManagerOsHelper->CheckOsInitDoneStatus");

    nvErr = InitSocket();
    CHK_ERROR_AND_RETURN(nvErr, "CheckInitDoneStatus");

    return nvErr;
}

NvError CStatusManagerService::DeInit()
{
    for (auto &pollFd : m_vPollFds) {
        close(pollFd.fd);
    }
    unlink(SOCKET_PATH);

    if (m_statusManagerDvmsState == StatusMangerState::STATUSMANAGER_STATE_DEINIT) {
        m_spStatusManagerOsHelper->SetOsDvmsState(StatusMangerState::STATUSMANAGER_STATE_DEINIT); //shut down system.
    }

    return NvError_Success;
}

void CStatusManagerService::HandleClientConnection()
{
    int newFd = accept(m_serverFd, nullptr, nullptr);
    if (newFd == -1) {
        LOG_ERR("Error accepting new connection.");
    } else {
        LOG_INFO("New client fd %d connected.", newFd);
        if (m_vPollFds.size() < MAX_CLIENTS) {
            m_vPollFds.push_back({ newFd, POLLIN, 0 });
            m_vAppClients.push_back({ newFd });
        } else {
            LOG_ERR("Max client limit reached. Connection rejected.");
            close(newFd);
        }
    }
}

void CStatusManagerService::HandleClientEvent(int fd)
{
    LOG_INFO("Enter: HandleClientEvent()\n");
    char buffer[MESSAGE_BUFFER_SIZE];
    int iLen = recv(fd, buffer, MESSAGE_BUFFER_SIZE, 0);
    if (iLen <= 0) {
        LOG_ERR(": Client fd %d disconnected.", fd);
        close(fd);
        m_vPollFds.erase(
            std::remove_if(m_vPollFds.begin(), m_vPollFds.end(), [&fd](pollfd pollFd) { return fd == pollFd.fd; }),
            m_vPollFds.end());

        m_vAppClients.erase(std::remove_if(m_vAppClients.begin(), m_vAppClients.end(),
                                           [&fd](AppClient client) { return fd == client.m_fd; }),
                            m_vAppClients.end());
    } else {
        MsgHeader *pHeader = reinterpret_cast<MsgHeader *>(&buffer[0]);
        LOG_INFO("Receive fd: %d and message type %d", fd, pHeader->type);

        switch (pHeader->type) {
            case MessageType::MSG_TYPE_REGISTER: {
                MsgRegister *pMsgStateReg = reinterpret_cast<MsgRegister *>(pHeader);

                char sName[256] = { '\0' };
                strncpy(sName, pMsgStateReg->clientName, pHeader->uMessageSize - sizeof(MsgHeader));
                LOG_INFO("Receving register client name: %s\n", sName);

                std::string sClientName = std::string(sName);
                if (sClientNamesAllowList.find(sClientName) == sClientNamesAllowList.end()) {
                    LOG_ERR("Registered client name %s not in the StatusMangerState server allow list.", sClientName);
                    close(fd);

                    m_vAppClients.erase(std::remove_if(m_vAppClients.begin(), m_vAppClients.end(),
                                                       [&fd](AppClient client) { return fd == client.m_fd; }),
                                        m_vAppClients.end());
                    break;
                }

                for (AppClient &client : m_vAppClients) {
                    if (client.m_fd == fd) {
                        client.m_sName = sClientName;
                        break;
                    }
                }

                LOG_MSG("Client %s(fd: %d) registered\n", sName, fd);
                SendMessageBase(fd, MessageType::MSG_TYPE_REGISTER_ACK);
                break;
            }
            case MessageType::MSG_TYPE_STATE_SET: {
                MsgState *pMsgStateSet = reinterpret_cast<MsgState *>(pHeader);

                bool bClientValid = false;
                for (AppClient &client : m_vAppClients) {
                    if (client.m_fd == fd) {
                        NvError nvErr = ValidateStateSwitch(client.m_state, pMsgStateSet->stateType);
                        if (nvErr != NvError_Success) {
                            LOG_ERR("ValidateStateSwitch fail for client %s_%d.", client.m_sName, client.m_fd);
                        } else {
                            //ACK the message to client.
                            nvErr = SendMessageBase(fd, MessageType::MSG_TYPE_STATE_SET_ACK);
                            if (nvErr != NvError_Success) {
                                LOG_ERR("SendMessageState MessageType::MSG_TYPE_STATE_SET_ACK fail.");
                                break;
                            }
                            client.m_state = pMsgStateSet->stateType;
                            bClientValid = true;

                            LOG_MSG("From client:%s: Set state %d, ACK done!\n", client.m_sName.c_str(),
                                    client.m_state);
                        }
                        break;
                    }
                }

                if (bClientValid == false) {
                    break;
                }

                NvError nvErr = ValidateStateSwitch(m_statusManagerDvmsState, pMsgStateSet->stateType);
                if (nvErr != NvError_Success) {
                    LOG_ERR("ValidateStateSwitch fail");
                    break;
                }

                switch (pMsgStateSet->stateType) {
                    case StatusMangerState::STATUSMANAGER_STATE_INIT_DONE:
                        m_clientsInitDoneCount++;
                        if (m_clientsInitDoneCount == m_uNumClients) {
                            m_statusManagerDvmsState = StatusMangerState::STATUSMANAGER_STATE_INIT_DONE;
                            m_clientsInitDoneCount = 0;
                            LOG_MSG("All registered clients in InitDone state,\n"
                                    "Input for next state:\n"
                                    "'dp': switch to deinit prepare state.\n"
                                    "'s': switch to suspend state.\n"
                                    "'r': switch to operational state.\n"
                                    "'lp': switch to low power state.\n");
                            NotifyClientStateFromInput();
                        }
                        break;
                    case StatusMangerState::STATUSMANAGER_STATE_REINIT:
                        m_clientsReInitCount++;
                        if (m_clientsReInitCount == m_uNumClients) {
                            //After setting the NVDVMS_REINIT to DriveOS,
                            //the system state will be set to NVDVMS_INIT_DONE by server io-nvdvms.
                            nvErr = m_spStatusManagerOsHelper->CheckOsInitDoneStatus();
                            if (nvErr != NvError_Success) {
                                LOG_ERR("CheckOsInitDoneStatus in Reinit fail.");
                                break;
                            }

                            m_clientsReInitCount = 0;
                            m_statusManagerDvmsState = StatusMangerState::STATUSMANAGER_STATE_REINIT;
                            nvErr = NotifyClients(StatusMangerState::STATUSMANAGER_STATE_INIT_DONE);
                        }
                        break;
                    case StatusMangerState::STATUSMANAGER_STATE_OPERATIONAL:
                        m_clientsOperationalCount++;
                        if (m_clientsOperationalCount == m_uNumClients) {
                            m_statusManagerDvmsState = StatusMangerState::STATUSMANAGER_STATE_OPERATIONAL;
                            m_clientsOperationalCount = 0;

                            LOG_MSG("All registered clients in Operational state, "
                                    "Input for next state:\n"
                                    "'dp': switch to deinit prepare state.\n");
                            NotifyClientStateFromInput();
                        }
                        break;
                    case StatusMangerState::STATUSMANAGER_STATE_SUSPEND:
                        m_clientsSuspendCount++;
                        if (m_clientsSuspendCount == m_uNumClients) {
                            nvErr = m_spStatusManagerOsHelper->SetOsDvmsState(
                                StatusMangerState::STATUSMANAGER_STATE_SUSPEND);
                            if (nvErr != NvError_Success) {
                                LOG_ERR("SetOsDvmsState fail");
                                break;
                            }
                            m_clientsSuspendCount = 0;
                            m_statusManagerDvmsState = StatusMangerState::STATUSMANAGER_STATE_SUSPEND;
                            nvErr = m_spStatusManagerOsHelper->WaitForResume();
                            if (nvErr != NvError_Success) {
                                LOG_ERR("CheckInitDoneStatus when resuming from suspend fail");
                                break;
                            }

                            //Notify clients to resume.
                            nvErr = NotifyClients(StatusMangerState::STATUSMANAGER_STATE_RESUME);
                            if (nvErr != NvError_Success) {
                                LOG_ERR("Resume from suspend fail");
                                break;
                            }
                        }
                        break;
                    case StatusMangerState::STATUSMANAGER_STATE_RESUME:
                        m_clientsResumeCount++;
                        if (m_clientsResumeCount == m_uNumClients) {
                            m_clientsResumeCount = 0;
                            m_statusManagerDvmsState = StatusMangerState::STATUSMANAGER_STATE_RESUME;
                            //Notify clients to InitDone.
                            nvErr = NotifyClients(StatusMangerState::STATUSMANAGER_STATE_INIT_DONE);
                            if (nvErr != NvError_Success) {
                                LOG_ERR("Resume from suspend fail");
                                break;
                            }
                        }
                        break;
                    case StatusMangerState::STATUSMANAGER_STATE_LOW_POWER:
                        m_clientsLowPowerCount++;
                        if (m_clientsLowPowerCount == m_uNumClients) {
                            m_clientsLowPowerCount = 0;
                            nvErr = m_spStatusManagerOsHelper->SetOsDvmsState(
                                StatusMangerState::STATUSMANAGER_STATE_LOW_POWER);
                            if (nvErr != NvError_Success) {
                                LOG_ERR("SetOsDvmsState fail for low power state.\n");
                                break;
                            }
                            m_statusManagerDvmsState = StatusMangerState::STATUSMANAGER_STATE_LOW_POWER;
                            LOG_MSG("All registered clients at low power state,\n"
                                    "Input for next state:\n"
                                    "'fp' : switch to full power state.\n");
                            NotifyClientStateFromInput();
                        }
                        break;
                    case StatusMangerState::STATUSMANAGER_STATE_FULL_POWER:
                        m_clientsFullPowerCount++;
                        if (m_clientsFullPowerCount == m_uNumClients) {
                            m_clientsFullPowerCount = 0;
                            m_statusManagerDvmsState = StatusMangerState::STATUSMANAGER_STATE_FULL_POWER;
                            //Notify clients to InitDone.
                            nvErr = NotifyClients(StatusMangerState::STATUSMANAGER_STATE_INIT_DONE);
                            if (nvErr != NvError_Success) {
                                LOG_ERR("Enter full power mode fail.\n");
                                break;
                            }
                        }
                        break;
                    case StatusMangerState::STATUSMANAGER_STATE_DEINIT_PREPARE:
                        m_clientsDeinitPrepareCount++;
                        if (m_clientsDeinitPrepareCount == m_uNumClients) {
                            m_statusManagerDvmsState = StatusMangerState::STATUSMANAGER_STATE_DEINIT_PREPARE;
                            m_clientsDeinitPrepareCount = 0;

                            LOG_MSG("All registered client in Deinit Prepare state, "
                                    "Input for next state:\n"
                                    "'re' : switch to init done state by reinit.\n"
                                    "'di': switch to deinit state.\n");
                            NotifyClientStateFromInput();
                        }
                        break;
                    case StatusMangerState::STATUSMANAGER_STATE_DEINIT:
                        m_clientsDeinitCount++;
                        if (m_clientsDeinitCount == m_uNumClients) {
                            LOG_MSG("All registered clients in Deinit state.\n");
#ifdef NVMEDIA_QNX
                            LOG_MSG("System will be shutdown!\n");
#else
                            LOG_MSG("All clients registered will be closed!\n");
#endif
                            m_statusManagerDvmsState = StatusMangerState::STATUSMANAGER_STATE_DEINIT;
                            m_bEventThreadRunning = false; // shut down.
                            m_clientsDeinitCount = 0;

                            nvErr = m_spStatusManagerOsHelper->DeInit();
                            if (nvErr != NvError_Success) {
                                LOG_ERR("m_spStatusManagerOsHelper->DeInit fail");
                                break;
                            }
                        }
                        break;
                    default:
                        LOG_ERR("Received wrong state type %d.", pMsgStateSet->stateType);
                        break;
                }
                break;
            }
            default:
                LOG_ERR("Received wrong message type %d.", pHeader->type);
        }
    }
    LOG_INFO("Exit: HandleClientEvent()\n");
}

void CStatusManagerService::NotifyClientStateFromInput()
{
    LOG_INFO("Enter: NotifyClientStateFromInput()\n");

    timeval timeout;
    bool bNotifySent = false;
    while (!bNotifySent) {
        fd_set read_set;
        FD_ZERO(&read_set);
        FD_SET(STDIN_FILENO, &read_set);
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;

        int ret = select(STDIN_FILENO + 1, &read_set, nullptr, nullptr, &timeout);
        if (ret == -1) {
            LOG_ERR("Error selecting cin.");
            return;
        } else if (ret == 0) {
            continue;
        }
        if (FD_ISSET(STDIN_FILENO, &read_set)) {
            char line[MESSAGE_BUFFER_SIZE];
            std::cin.getline(line, MESSAGE_BUFFER_SIZE);
            if (std::cin.eof()) {
                LOG_ERR("Stdin redirecting and reaching EOF.");
                break;
            }
            if (!strcmp(line, "dp")) {
                LOG_MSG("Trying to switch to deinit prepare state...\n");
                m_spStatusManagerOsHelper->SetOsDvmsState(StatusMangerState::STATUSMANAGER_STATE_DEINIT_PREPARE);
                NvError nvErr = NotifyClients(StatusMangerState::STATUSMANAGER_STATE_DEINIT_PREPARE);
                if (nvErr != NvError_Success) {
                    LOG_ERR("Switch to prepare state fail, please follow switch rules and retry.");
                } else {
                    bNotifySent = true;
                }
            } else if (!strcmp(line, "s")) {
                LOG_MSG("Trying to switch to suspend state...\n");
                NvError nvErr = NotifyClients(StatusMangerState::STATUSMANAGER_STATE_SUSPEND);
                if (nvErr != NvError_Success) {
                    LOG_ERR("Switch to suspend state fail, please follow switch rules and retry.");
                } else {
                    bNotifySent = true;
                }
            } else if (!strcmp(line, "re")) {
                LOG_MSG("Trying to switch to init done state by reinit...\n");
                m_spStatusManagerOsHelper->SetOsDvmsState(StatusMangerState::STATUSMANAGER_STATE_REINIT);
                NvError nvErr = NotifyClients(StatusMangerState::STATUSMANAGER_STATE_REINIT);
                if (nvErr != NvError_Success) {
                    LOG_ERR("Switch to reinit state fail, please follow switch rules and retry.");
                } else {
                    bNotifySent = true;
                }
            } else if (!strcmp(line, "r")) {
                LOG_MSG("Trying to switch to operational state...\n");
                m_spStatusManagerOsHelper->SetOsDvmsState(StatusMangerState::STATUSMANAGER_STATE_OPERATIONAL);
                NvError nvErr = NotifyClients(StatusMangerState::STATUSMANAGER_STATE_OPERATIONAL);
                if (nvErr != NvError_Success) {
                    LOG_ERR("Switch to operational state fail, please follow switch rules and retry.");
                } else {
                    bNotifySent = true;
                }
            } else if (!strcmp(line, "lp")) {
                LOG_MSG("Trying to switch to low power state...\n");
                NvError nvErr = NotifyClients(StatusMangerState::STATUSMANAGER_STATE_LOW_POWER);
                if (nvErr != NvError_Success) {
                    LOG_ERR("Switch to low power state fail, please follow switch rules and retry.");
                } else {
                    bNotifySent = true;
                }
            } else if (!strcmp(line, "fp")) {
                LOG_MSG("Trying to switch to full power state...\n");
                NvError nvErr =
                    m_spStatusManagerOsHelper->SetOsDvmsState(StatusMangerState::STATUSMANAGER_STATE_FULL_POWER);
                if (nvErr != NvError_Success) {
                    LOG_ERR("SetOsDvmsState fail for full power state.\n");
                }
                nvErr = NotifyClients(StatusMangerState::STATUSMANAGER_STATE_FULL_POWER);
                if (nvErr != NvError_Success) {
                    LOG_ERR("Switch to full power state fail, please follow switch rules and retry.");
                } else {
                    bNotifySent = true;
                }
            } else if (!strcmp(line, "di")) {
                LOG_MSG("Trying to switch to deinit state...\n");
                NvError nvErr = NotifyClients(StatusMangerState::STATUSMANAGER_STATE_DEINIT);
                if (nvErr != NvError_Success) {
                    LOG_ERR("Switch to deinit state fail, please follow switch rules and retry.");
                } else {
                    bNotifySent = true;
                }
            } else {
                LOG_ERR("Please input valid switch state.!");
                continue;
            }
        }
    }
    LOG_INFO("Exit: NotifyClientStateFromInput().");
}

NvError CStatusManagerService::Start()
{
    m_bEventThreadRunning = true;

    while (m_bEventThreadRunning) {
        std::vector<pollfd> pollFds = m_vPollFds;

        //Add timeout, once timeout happen StatusManager server will check all clients state.
        int ret = poll(pollFds.data(), pollFds.size(), MILLISECONDS_TIMEOUT);

        if (ret == -1) {
            LOG_ERR("Error selecting sockets.");
            break;
        }

        for (auto pollFd : pollFds) {
            if ((pollFd.revents & POLLIN) == 0) {
                continue;
            }
            if (pollFd.fd == m_serverFd) {
                HandleClientConnection();
            } else {
                HandleClientEvent(pollFd.fd);
            }
        }
    }

    return NvError_Success;
}

NvError CStatusManagerService::NotifyClients(StatusMangerState statusManagerState)
{
    NvError nvErr = ValidateStateSwitch(m_statusManagerDvmsState, statusManagerState);
    CHK_ERROR_AND_RETURN(nvErr, "ValidateStateSwitch");

    bool bClientStateValid = true;
    for (AppClient &client : m_vAppClients) {
        if (client.m_state != m_statusManagerDvmsState) {
            LOG_ERR("client.m_state %d not match m_statusManagerDvmsState %d", client.m_state,
                    m_statusManagerDvmsState);
            bClientStateValid = false;
            break;
        }
    }

    if (bClientStateValid == false) {
        return NvError_ResourceError;
    }

    for (AppClient &client : m_vAppClients) {
        nvErr = SendMessageState(client.m_fd, MessageType::MSG_TYPE_STATE_SET, statusManagerState);
        CHK_ERROR_AND_RETURN(nvErr, "SendMessageState");
        LOG_MSG("To client:%s(%d): Set state %d.\n", client.m_sName.c_str(), client.m_fd, statusManagerState);
    }

    return nvErr;
}

void ShowUsage()
{
    // clang-format off
    std::cout << "Usage:\n";
    std::cout << "-h or --help                               :Prints this help\n";
    std::cout << "-n or --numConsumers                       :Set clients number\n";
    std::cout << "-v or --verbosity <level>                  :Set verbosity\n";

    return;
}

int main(int argc, char *argv[])
{
    const char *const short_options = "hn:v:";
    const struct option long_options[] = {
        // clang-format off
        { "help",                 no_argument,       0, 'h' },
        { "numConsumers",         no_argument,       0, 'n' },
        { "verbosity",            required_argument, 0, 'v' },
    };

    int index = 0;
    bool bShowHelp = false;
    uint32_t uVerbosity = 1;
    uint32_t uNumClients = 1;

    while (1) {
        const auto getopt_ret = getopt_long(argc, argv, short_options, &long_options[0], &index);
        if (getopt_ret == -1) {
            // Done parsing all arguments.
            break;
        }

        switch (getopt_ret) {
            default:  /* Unrecognized option */
            case 'h': /* -h or --help */
                bShowHelp = true;
                break;
            case 'v':
                uVerbosity = std::atoi(optarg);
                break;
            case 'n':
                uNumClients = std::atoi(optarg);
                if (uNumClients > MAX_CLIENTS) {
                     LOG_ERR("uNumClients can not exceed %d.", MAX_CLIENTS);
                     return -1;
                }
                break;
        }
    }

    if (bShowHelp) {
        ShowUsage();
        return 0;
    }

    CLogger::GetInstance().SetLogLevel((CLogger::LogLevel)uVerbosity);

    std::unique_ptr<CStatusManagerService> upStatusManagerSocketServer = std::make_unique<CStatusManagerService>(uNumClients);

    NvError nvErr = upStatusManagerSocketServer->Init();
    CHK_ERROR_AND_RETURN(nvErr, "upStatusManagerSocketServer->Init()");

    nvErr = upStatusManagerSocketServer->Start();
    CHK_ERROR_AND_RETURN(nvErr, "upStatusManagerSocketServer->Start()");

    nvErr = upStatusManagerSocketServer->DeInit();
    CHK_ERROR_AND_RETURN(nvErr, "upStatusManagerSocketServer->DeInit()");

    return 0;
}
