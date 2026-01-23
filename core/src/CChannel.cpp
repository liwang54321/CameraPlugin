/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <algorithm>
#include <array>

#include "CChannel.hpp"
#include "CFactory.hpp"

using namespace nvsipl;

CChannel::CChannel(std::shared_ptr<CChannelCfg> spChannelCfg, IEventListener<CChannel> *pListener)
    : m_spChannelCfg(std::move(spChannelCfg))
    , m_pEventListener(pListener)
    , m_channelState(ChannelState::CREATED)
{
}

void CChannel::Init()
{
    if (!m_upEventThread) {
        m_bEventThreadQuit = false;
        m_upEventThread.reset(new std::thread(&CChannel::ThreadFunc, this));
        m_upEventThread->detach();
    }
};

void CChannel::Quit()
{
    PushCmd(CMDType::QUIT);
}

void CChannel::Start()
{
    PushCmd(CMDType::START);
}

void CChannel::Stop()
{
    PushCmd(CMDType::STOP);
}

void CChannel::PrintFps()
{
    std::cout << "Channel " + GetName() + ":" << std::endl;

    for (const auto &mod : m_vspModules) {
        mod->PrintFps();
    }
}

void CChannel::BuildPipeline()
{
    PLOG_DBG("Enter: CChannel::BuildPipeline()\n");
    std::vector<std::shared_ptr<CModuleElem>> vModuleElems;

    auto error = ParsePipelineDescriptor(m_spChannelCfg->m_vPipelineDescriptor, vModuleElems);
    if (error != NvError_Success) {
        LOG_ERR("ParsePipelineDescriptor failed!\n");
        PushEvent(EventStatus::ERROR);
        return;
    }

    error = CreateModules(vModuleElems);
    if (error != NvError_Success) {
        LOG_ERR("CreateModules failed!\n");
        PushEvent(EventStatus::ERROR);
        return;
    }

    error = ParsePipelineOptions(vModuleElems);
    if (error != NvError_Success) {
        LOG_ERR("ParsePipelineOptions failed!\n");
        PushEvent(EventStatus::ERROR);
        return;
    }

    error = ConnectModules(vModuleElems);
    if (error != NvError_Success) {
        LOG_ERR("ConnectModules failed!\n");
        PushEvent(EventStatus::ERROR);
        return;
    }

    PLOG_DBG("Exit: CChannel::BuildPipeline()\n");
};

void CChannel::InitModules()
{
    PLOG_DBG("Enter: CChannel::InitModules()\n");

    for (auto &spMod : m_vspModules) {
        auto error = spMod->Init();
        if (error != NvError_Success) {
            LOG_ERR("Module Init failed!\n");
            PushEvent(EventStatus::ERROR);
            return;
        }
    }

    PLOG_DBG("Exit: CChannel::InitModules()\n");
};

void CChannel::ReconcileModules()
{
    PLOG_DBG("Enter: CChannel::ReconcileModules()\n");

    for (auto it = m_vspModules.rbegin(); it != m_vspModules.rend(); ++it) {
        auto error = (*it)->Reconcile();
        if (error != NvError_Success) {
            LOG_ERR("Module Reconcile failed!\n");
            PushEvent(EventStatus::ERROR);
            break;
        }
    }

    PLOG_DBG("Exit: CChannel::ReconcileModules()\n");
}

void CChannel::StartModules()
{
    PLOG_DBG("Enter: CChannel::StartModules()\n");

    for (auto it = m_vspModules.rbegin(); it != m_vspModules.rend(); ++it) {
        auto error = (*it)->Start();
        if (error != NvError_Success) {
            LOG_ERR("Module Start failed!\n");
            PushEvent(EventStatus::ERROR);
            break;
        }
    }

    PLOG_DBG("Exit: CChannel::StartModules()\n");
}

void CChannel::StopModules()
{
    PLOG_DBG("Enter: CChannel::StopModules()\n");

    /*PreStop is to notify module cosume all packets before stoped.*/
    for (auto &mod : m_vspModules) {
        mod->PreStop();
    }

    /*Stop modules*/
    for (auto it = m_vspModules.rbegin(); it != m_vspModules.rend(); ++it) {
        (*it)->Stop();
    }

    /*PostStop is used to wait all fence timeout
      after all modules are stoped to meet SC7 requirement*/
    for (auto &mod : m_vspModules) {
        mod->PostStop();
    }

    PLOG_DBG("Exit: CChannel::StopModules()\n");
}

void CChannel::CommandModules(CMDType cmd)
{
    PLOG_DBG("Enter: CChannel::CommandModules(CMDType cmd)\n");

    for (auto it = m_vspModules.begin(); it != m_vspModules.end(); ++it) {
        (*it)->OnCommand(static_cast<uint32_t>(cmd), nullptr);
    }

    PLOG_DBG("Exit: CChannel::CommandModules(CMDType cmd)\n");
}

/*
 * DFS modules blong to the data pipeline
 */
NvError CChannel::PrintPerf(const std::vector<std::shared_ptr<CBaseModule>> &vspModules,
                            std::vector<std::shared_ptr<CBaseModule>> &vspPipelineModules)
{
    NvError error = NvError_Success;
    for (const auto &module : vspModules) {
        const std::vector<std::shared_ptr<CBaseModule>> &vspDownstreamModules = module->GetDownstreamModules();
        if (!vspDownstreamModules.empty()) {
            vspPipelineModules.emplace_back(module);
            PrintPerf(vspDownstreamModules, vspPipelineModules);
            vspPipelineModules.pop_back();
        } else {
            std::vector<std::shared_ptr<CBaseModule>> vspCurrentPipelineModules = vspPipelineModules;
            vspCurrentPipelineModules.emplace_back(module);
            auto size = vspCurrentPipelineModules.size();
            /*
            * Print the module specific perf
            */
            for (uint32_t uIndex = 0; uIndex < size; ++uIndex) {
                std::shared_ptr<CBaseModule> spCurrentModule = vspCurrentPipelineModules[uIndex];
                std::vector<Perf> vPerfData;
                error = spCurrentModule->GetPerf(vPerfData);
                CHK_ERROR_AND_RETURN(error, "GetPerf");

                for (const Perf &perf : vPerfData) {
                    if (m_spChannelCfg->m_pAppCfg->GetProfilingMode() == ProfilingMode::FULL ||
                        PerfType::PIPELINE == perf.m_type) {
                        std::string sTitle = spCurrentModule->GetName() + " " + ToString(perf.m_type) + " latency";
                        CProfiler::ShowAndSavePerfData(perf.m_pPerfData, sTitle,
                                                       m_spChannelCfg->m_pAppCfg->NeedSavePerfData());
                    }
                }
            }
        }
    }
    return error;
}

void CChannel::DeInitModules()
{
    PLOG_DBG("Enter: CChannel::DeInit()\n");

    if (m_spChannelCfg->m_pAppCfg->IsProfilingEnabled()) {
        /*
         * Print the perf data.
         */
        std::vector<std::shared_ptr<CBaseModule>> vspModuleChain;
        PrintPerf(m_vspHeadModules, vspModuleChain);
    }
    for (auto &mod : m_vspModules) {
        mod->DeInit();
    }

    PLOG_DBG("Exit: CChannel::DeInit()\n");
}

NvError CChannel::ParseModuleConnection(std::unordered_map<std::string, std::shared_ptr<CModuleElem>> &moduleElemMap,
                                        std::shared_ptr<CPipelineElem> spPreElem,
                                        std::shared_ptr<CPipelineElem> spElem)
{
    if (spPreElem->IsModule()) {
        if (spElem->IsModule()) {
            moduleElemMap[spPreElem->m_sName]->AddDownstreamModule(moduleElemMap[spElem->m_sName]);
        } else if (spElem->m_type == PipelineElemType::IpcSrc) {
            moduleElemMap[spPreElem->m_sName]->AddIpcSrc(std::dynamic_pointer_cast<CIpcEndpointElem>(spElem));
        } else {
            LOG_ERR("Please check pipeline config! Only Module or IpcSrc can be defined after a Module.\n");
            return NvError_BadParameter;
        }
    } else if (spPreElem->m_type == PipelineElemType::IpcDst) {
        if (spElem->IsModule()) {
            moduleElemMap[spElem->m_sName]->AddIpcDst(std::dynamic_pointer_cast<CIpcEndpointElem>(spPreElem));
        } else {
            LOG_ERR("Please check pipeline config! IpcDst must be defined beforea a Module.\n");
            return NvError_BadParameter;
        }
    } else {
        LOG_ERR("Please check pipeline config! not supported connection currently.\n");
        return NvError_NotSupported;
    }

    return NvError_Success;
}

NvError CChannel::ParsePipelineDescriptor(const PipelineDescriptor &pipelineDescriptor,
                                          std::vector<std::shared_ptr<CModuleElem>> &vModuleElems)
{
    std::unordered_map<std::string, std::shared_ptr<CModuleElem>> moduleElemMap{};

    for (const auto &row : pipelineDescriptor) {
        std::array<std::shared_ptr<CPipelineElem>, MAX_NUM_CASCADED_MODULES> spPipelineElems{};
        bool bIsFirstModule = true;
        for (size_t i = 0; i < row.size(); i++) {
            spPipelineElems[i] = m_spChannelCfg->ParsePipelineStr(row[i]);
            PCHK_PTR_AND_RETURN(spPipelineElems[i], "ParsePipelineStr");

            if (spPipelineElems[i]->IsModule()) {
                auto it = moduleElemMap.find(spPipelineElems[i]->m_sName);
                if (it == moduleElemMap.end()) {
                    std::shared_ptr<CModuleElem> spModuleElem =
                        std::dynamic_pointer_cast<CModuleElem>(spPipelineElems[i]);
                    moduleElemMap.emplace(spPipelineElems[i]->m_sName, spModuleElem);
                    vModuleElems.push_back(spModuleElem);
                    if (m_spChannelCfg->m_pAppCfg->IsProfilingEnabled()) {
                        /*
                         * Collect the head module elements.
                         */
                        if (bIsFirstModule) {
                            m_vspHeadModuleElems.emplace_back(spModuleElem);
                        }
                    }
                }
                bIsFirstModule = false;
            }

            if (i > 0 && (spPipelineElems[i]->IsModule() || (spPipelineElems[i]->m_type == PipelineElemType::IpcSrc))) {
                auto error = ParseModuleConnection(moduleElemMap, spPipelineElems[i - 1], spPipelineElems[i]);
                PCHK_ERROR_AND_RETURN(error, "ParseModuleConnection");
            }
        }
    }

    return NvError_Success;
}

NvError CChannel::CreateModules(std::vector<std::shared_ptr<CModuleElem>> &vModuleElems)
{
    for (auto &spModElem : vModuleElems) {
        if (spModElem->m_isStitching) {
            auto spModuleCfg = m_spChannelCfg->CreateModuleCfg(INVALID_ID, spModElem);
            auto spMod = CFactory::CreateModule(std::move(spModuleCfg), this);
            PCHK_PTR_AND_RETURN(spMod, "factory.CreateModule");
            spModElem->m_vspMods.emplace_back(spMod);
            m_vspModules.push_back(spMod);
        } else {
            for (auto sensorId : spModElem->m_vSensorIds) {
                auto spModuleCfg = m_spChannelCfg->CreateModuleCfg(sensorId, spModElem);
                auto spMod = CFactory::CreateModule(std::move(spModuleCfg), this);
                PCHK_PTR_AND_RETURN(spMod, "factory.CreateModule");
                spModElem->m_vspMods.emplace_back(spMod);
                m_vspModules.push_back(spMod);
            }
        }
    }

    if (0 == m_vspModules.size()) {
        LOG_ERR("No module was created!\n");
        return NvError_BadParameter;
    }

    if (m_spChannelCfg->m_pAppCfg->IsProfilingEnabled()) {
        /*
         * Collect the head modules
         */
        for (const auto &spModuleElem : m_vspHeadModuleElems) {
            for (const auto &spMod : spModuleElem->m_vspMods) {
                m_vspHeadModules.emplace_back(spMod);
            }
        }
    }

    return NvError_Success;
}

NvError CChannel::ParsePipelineOptions(std::vector<std::shared_ptr<CModuleElem>> &vModuleElems)
{
    for (auto &spModElem : vModuleElems) {
        //Parse options
        if (!spModElem->m_options.empty()) {
            for (auto &spMod : spModElem->m_vspMods) {
                auto error = spMod->ParseOptions();
                PCHK_ERROR_AND_RETURN(error, "spMod->ParseOptions");
            }
        }
        for (const auto &spIpcSrc : spModElem->m_vspIpcSrcs) {
            if (!spIpcSrc->m_options.empty()) {
                auto error = spIpcSrc->ParseOptions();
                PCHK_ERROR_AND_RETURN(error, "spIpcSrc->ParseOptions");
            }
        }
        for (const auto &spIpcDst : spModElem->m_vspIpcDsts) {
            if (!spIpcDst->m_options.empty()) {
                auto error = spIpcDst->ParseOptions();
                PCHK_ERROR_AND_RETURN(error, "spIpcDst->ParseOptions");
            }
        }
    }

    return NvError_Success;
}

NvError CChannel::ConnectModules(std::vector<std::shared_ptr<CModuleElem>> &vModuleElems)
{
    NvError error = NvError_Success;
    for (auto &spModElem : vModuleElems) {
        if (spModElem->m_isStitching) {
            auto &spMod = spModElem->m_vspMods[0];
            for (const auto &spIpcDst : spModElem->m_vspIpcDsts) {
                error = spMod->ConnectIpcDst(spIpcDst);
                CHK_ERROR_AND_RETURN(error, "CBaseModule->ConnectIpcDst");
            }
            for (const auto &spIpcSrc : spModElem->m_vspIpcSrcs) {
                error = spMod->ConnectIpcSrc(spIpcSrc);
                CHK_ERROR_AND_RETURN(error, "CBaseModule->ConnectIpcSrc");
            }
            for (const auto &spDownstreamMod : spModElem->m_vspDownstreamModules) {
                // the downstream module should also be a stitching module
                error = spMod->ConnectModule(spDownstreamMod->m_vspMods[0]);
                CHK_ERROR_AND_RETURN(error, "CBaseModule->ConnectModule");
            }
        } else {
            for (auto i = 0U; i < spModElem->m_vSensorIds.size(); ++i) {
                std::vector<std::shared_ptr<CIpcEndpointElem>> vspIpcSrcs;
                std::vector<std::shared_ptr<CBaseModule>> vspDownstreamMods;
                for (const auto &spIpcDst : spModElem->m_vspIpcDsts) {
                    auto it = std::find(spIpcDst->m_vSensorIds.begin(), spIpcDst->m_vSensorIds.end(),
                                        spModElem->m_vSensorIds[i]);
                    if (it != spIpcDst->m_vSensorIds.end()) {
                        error = spModElem->m_vspMods[i]->ConnectIpcDst(spIpcDst);
                        CHK_ERROR_AND_RETURN(error, "CBaseModule->ConnectIpcDst");
                    }
                }
                for (const auto &spIpcSrc : spModElem->m_vspIpcSrcs) {
                    auto it = std::find(spIpcSrc->m_vSensorIds.begin(), spIpcSrc->m_vSensorIds.end(),
                                        spModElem->m_vSensorIds[i]);
                    if (it != spIpcSrc->m_vSensorIds.end()) {
                        vspIpcSrcs.emplace_back(spIpcSrc);
                    }
                }
                for (const auto &spDownstreamMod : spModElem->m_vspDownstreamModules) {
                    auto it = std::find(spDownstreamMod->m_vSensorIds.begin(), spDownstreamMod->m_vSensorIds.end(),
                                        spModElem->m_vSensorIds[i]);
                    if (it != spDownstreamMod->m_vSensorIds.end()) {
                        int index = spDownstreamMod->m_isStitching
                                        ? 0
                                        : std::distance(spDownstreamMod->m_vSensorIds.begin(), it);
                        vspDownstreamMods.emplace_back(spDownstreamMod->m_vspMods[index]);
                    }
                }
                spModElem->m_vspMods[i]->SetConsumerNumber((int)(vspIpcSrcs.size() + vspDownstreamMods.size()));
                for (const auto &spIpcSrc : vspIpcSrcs) {
                    error = spModElem->m_vspMods[i]->ConnectIpcSrc(spIpcSrc);
                    CHK_ERROR_AND_RETURN(error, "CBaseModule->ConnectIpcDst");
                }
                for (const auto &spMod : vspDownstreamMods) {
                    error = spModElem->m_vspMods[i]->ConnectModule(spMod);
                    CHK_ERROR_AND_RETURN(error, "CBaseModule->ConnectModule");
                }
            }
        }
    }
    return NvError_Success;
}

void CChannel::OnError(CBaseModule *pModule, int moduleId, uint32_t errorId)
{
    PLOG_ERR("Received error %s ModuleId %d OnError %d\n", pModule->GetName().c_str(), moduleId, errorId);
    if (m_pEventListener) {
        m_pEventListener->OnError(this, moduleId, errorId);
    }
}

void CChannel::HandleCmdAsync(CMDType cmd)
{
    PushCmd(cmd);
}

void CChannel::PushEvent(EventStatus event)
{
    PLOG_DBG("Push event %d\n", event);

    std::unique_lock<std::mutex> lock(m_eventQueueMutex);
    m_eventQueue.push(event);
    lock.unlock();

    m_semaphore.Signal();
}

void CChannel::PushCmd(CMDType cmd)
{
    PLOG_DBG("Received cmd %d\n", cmd);

    std::unique_lock<std::mutex> lock(m_cmdQueueMutex);
    m_cmdQueue.push(cmd);
    lock.unlock();

    m_semaphore.Signal();
}

void CChannel::ThreadFunc()
{
    PLOG_DBG("Enter: CChannel::ThreadFunc()\n");

    pthread_setname_np(pthread_self(), "ChannelThrd");

    BuildPipeline();
    InitModules();

    // Wait for quit
    while (!m_bEventThreadQuit) {
        m_semaphore.Wait();
        HandleCmd();
        HandleEvent();
    }

    PLOG_DBG("Exit: CChannel::ThreadFunc()\n");
}

void CChannel::HandleCmd()
{
    while (!m_cmdQueue.empty()) {
        CMDType cmd = m_cmdQueue.front();
        m_cmdQueue.pop();
        ChannelState state = m_channelState;

        switch (cmd) {
            case CMDType::START:
                switch (state) {
                    case ChannelState::STOPPED:
                        if (m_spChannelCfg->m_pAppCfg->GetPipelineType() != PipelineType::SentryPipelineConsumer) {
                            StartModules();
                        } else {
                            // For the sentry pipeline consumer channel, no need
                            // to start the modules, just set the state to STOPPED
                            // and notify the STARTED event to the status manager
                            // to complete the re-initialization.
                            m_channelState = ChannelState::STOPPED;
                            m_pEventListener->OnEvent(this, EventStatus::STARTED);
                        }
                        break;
                    default:
                        LOG_WARN("invalid state, do nothing! state %d, cmd %d\n", state, cmd);
                }
                break;
            case CMDType::STOP:
                switch (state) {
                    case ChannelState::RUNNING:
                        StopModules();
                        break;
                    case ChannelState::STOPPED:
                        // For the sentry pipeline consumer channel, no need
                        // to stop the modules, just set the state to STOPPED
                        // and notify the STOPPED event to the status manager
                        // to complete the deinit-prepare.
                        if (m_spChannelCfg->m_pAppCfg->GetPipelineType() == PipelineType::SentryPipelineConsumer) {
                            m_channelState = ChannelState::STOPPED;
                            m_pEventListener->OnEvent(this, EventStatus::STOPPED);
                        } else {
                            LOG_WARN("invalid state, do nothing! state %d, cmd %d\n", state, cmd);
                        }
                        break;
                    default:
                        LOG_WARN("invalid state, do nothing! state %d, cmd %d\n", state, cmd);
                }
                break;
            case CMDType::QUIT:
                switch (state) {
                    case ChannelState::RUNNING:
                    case ChannelState::STOPPED:
                    case ChannelState::CREATED:
                    case ChannelState::CONNECTED:
                    case ChannelState::RECONCILED:
                        if (m_vspModules.empty()) {
                            // If no modules were created, send QUITTED event directly
                            m_pEventListener->OnEvent(this, EventStatus::QUITTED);
                        } else {
                            if (state == ChannelState::RUNNING) {
                                StopModules();
                            }
                            DeInitModules();
                        }
                        m_bEventThreadQuit = true;
                        break;
                    default:
                        LOG_WARN("invalid state, do nothing! state %d, cmd %d\n", state, cmd);
                }
                break;
            case CMDType::ENTER_LOW_POWER_MODE:
            case CMDType::ENTER_FULL_POWER_MODE:
                switch (state) {
                    case ChannelState::RUNNING:
                    case ChannelState::STOPPED:
                        if (ValidatePowerModeSwitch(cmd, state) != NvError_Success) {
                            LOG_WARN("Cmd %d is not reasonable to perform in current state %d.", cmd, state);
                        } else {
                            SwitchPowerMode(cmd);
                        }
                        break;
                    default:
                        LOG_WARN("invalid state, do nothing! state %d, cmd %d\n", state, cmd);
                }
                break;
            default:
                LOG_WARN("Invalid cmd: %d\n", cmd);
                break;
        }
    }
}

void CChannel::SwitchPowerMode(CMDType cmd)
{
    auto pipelineType = m_spChannelCfg->m_pAppCfg->GetPipelineType();
    if (pipelineType == PipelineType::SentryPipelineProducer) {
        CommandModules(cmd);
    } else if (pipelineType == PipelineType::SentryPipelineConsumer) {
        cmd == CMDType::ENTER_LOW_POWER_MODE ? StartModules() : StopModules();
    } else {
        cmd == CMDType::ENTER_LOW_POWER_MODE ? StopModules() : StartModules();
    }
}

NvError CChannel::ValidatePowerModeSwitch(CMDType cmd, ChannelState state)
{
    NvError error = NvError_Success;
    auto pipelineType = m_spChannelCfg->m_pAppCfg->GetPipelineType();
    switch (cmd) {
        case CMDType::ENTER_LOW_POWER_MODE:
            if ((pipelineType == PipelineType::NormalPipeline && state == ChannelState::STOPPED) ||
                (pipelineType == PipelineType::SentryPipelineConsumer && state == ChannelState::RUNNING)) {
                error = NvError_InvalidState;
            }
            break;
        case CMDType::ENTER_FULL_POWER_MODE:
            if ((pipelineType == PipelineType::NormalPipeline && state == ChannelState::RUNNING) ||
                (pipelineType == PipelineType::SentryPipelineConsumer && state == ChannelState::STOPPED)) {
                error = NvError_InvalidState;
            }
            break;
        default:
            LOG_ERR("Invalid cmd: %d\n", cmd);
            break;
    }

    return error;
}

void CChannel::HandleEvent()
{
    auto AreAllModulesReported = [this](EventStatus event) {
        if (++m_eventCountMap[event] == static_cast<int>(m_vspModules.size())) {
            m_eventCountMap.erase(event);
            return true;
        }
        return false;
    };

    auto CheckAndTransitState = [this, AreAllModulesReported](EventStatus event, ChannelState newState) {
        if (AreAllModulesReported(event)) {
            m_channelState = newState;
            m_pEventListener->OnEvent(this, event);
            PLOG_DBG("Transfer Channel State to %d\n", newState);
            return true;
        }
        return false;
    };
    while (!m_eventQueue.empty()) {
        EventStatus event = m_eventQueue.front();
        m_eventQueue.pop();
        ChannelState state = m_channelState;

        switch (event) {
            case EventStatus::CONNECTED:
                switch (state) {
                    case ChannelState::CREATED:
                        if (CheckAndTransitState(event, ChannelState::CONNECTED)) {
                            ReconcileModules();
                        }
                        break;
                    default:
                        LOG_WARN("invalid state, do nothing! state %d, event %d\n", state, event);
                }
                break;
            case EventStatus::RECONCILED:
                switch (state) {
                    case ChannelState::CONNECTED:
                        if (CheckAndTransitState(event, ChannelState::RECONCILED)) {
                            if (m_spChannelCfg->m_pAppCfg->GetPipelineType() != PipelineType::SentryPipelineConsumer) {
                                StartModules();
                            } else {
                                // For the sentry pipeline consumer channel, no need
                                // to start the modules, just set the state to STOPPED
                                // and notify the STARTED event to the status manager
                                // to complete the initialization.
                                m_channelState = ChannelState::STOPPED;
                                m_pEventListener->OnEvent(this, EventStatus::STARTED);
                            }
                        }
                        break;
                    default:
                        LOG_WARN("invalid state, do nothing! state %d, event %d\n", state, event);
                }
                break;
            case EventStatus::STARTED:
                switch (state) {
                    case ChannelState::RECONCILED:
                    case ChannelState::STOPPED:
                        CheckAndTransitState(event, ChannelState::RUNNING);
                        break;
                    default:
                        LOG_WARN("invalid state, do nothing! state %d, event %d\n", state, event);
                }
                break;
            case EventStatus::STOPPED:
                switch (state) {
                    case ChannelState::RUNNING:
                        CheckAndTransitState(event, ChannelState::STOPPED);
                        break;
                    default:
                        LOG_WARN("invalid state, do nothing! state %d, event %d\n", state, event);
                }
                break;
            case EventStatus::QUITTED:
                switch (state) {
                    case ChannelState::STOPPED:
                    case ChannelState::CREATED:
                    case ChannelState::CONNECTED:
                    case ChannelState::RECONCILED:
                        CheckAndTransitState(event, ChannelState::DEINIT);
                        break;
                    default:
                        LOG_WARN("invalid state, do nothing! state %d, event %d\n", state, event);
                }
                break;
            case EventStatus::ERROR:
            case EventStatus::DISCONNECT:
                m_pEventListener->OnEvent(this, event);
                break;
            default:
                LOG_WARN("Unhandled event %s\n", EventStatusToString(event));
        }
    }
}

void CChannel::OnEvent(CBaseModule *pModule, EventStatus event)
{
    PLOG_DBG("Received event: %s %s\n", pModule->GetName().c_str(), EventStatusToString(event));

    PushEvent(event);
}
