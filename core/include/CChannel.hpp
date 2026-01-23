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

#ifndef CCHANNEL_H
#define CCHANNEL_H

/* STL Headers */
#include <cstring>
#include <unordered_map>
#include <queue>
#include <map>
#include "CConfig.hpp"
#include "CClientCommon.hpp"
#include "CUtils.hpp"
#include "CBaseModule.hpp"

using namespace nvsipl;

class CChannel : public IEventListener<CBaseModule>
{
  public:
    enum class ChannelState : uint8_t
    {
        CREATED = 0,
        CONNECTED,
        RECONCILED,
        RUNNING,
        STOPPED,
        DEINIT,
        ERROR,
    };

    CChannel(std::shared_ptr<CChannelCfg> spChannelCfg, IEventListener<CChannel> *pListener);
    virtual ~CChannel() {}

    void Init();
    void Quit();
    void Start();
    void Stop();
    void PrintFps();
    inline const std::string &GetName() { return m_spChannelCfg->m_sName; }

    virtual void OnError(CBaseModule *pModule, int moduleId, uint32_t uErrorId) override;
    void HandleCmdAsync(CMDType cmd);
    bool IsStopped() const { return m_channelState == ChannelState::STOPPED; }

  private:
    void BuildPipeline();
    void InitModules();
    void ReconcileModules();
    void StartModules();
    void StopModules();
    void DeInitModules();
    void CommandModules(CMDType cmd);

    NvError ParseModuleConnection(std::unordered_map<std::string, std::shared_ptr<CModuleElem>> &moduleElemMap,
                                  std::shared_ptr<CPipelineElem> spPreElem,
                                  std::shared_ptr<CPipelineElem> spElem);
    NvError ParsePipelineDescriptor(const PipelineDescriptor &pipelineDescriptor,
                                    std::vector<std::shared_ptr<CModuleElem>> &vModuleElems);
    NvError CreateModules(std::vector<std::shared_ptr<CModuleElem>> &vModuleElems);
    NvError ParsePipelineOptions(std::vector<std::shared_ptr<CModuleElem>> &vModuleElems);
    NvError ConnectModules(std::vector<std::shared_ptr<CModuleElem>> &vModuleElems);

    void PushEvent(EventStatus event);
    void PushCmd(CMDType cmd);
    void ThreadFunc();
    void HandleCmd();
    void HandleEvent();
    void SwitchPowerMode(CMDType cmd);
    NvError ValidatePowerModeSwitch(CMDType cmd, ChannelState state);
    virtual void OnEvent(CBaseModule *pModule, EventStatus event) override;
    NvError PrintPerf(const std::vector<std::shared_ptr<CBaseModule>> &vspModules,
                      std::vector<std::shared_ptr<CBaseModule>> &vspPipelineModules);
    std::shared_ptr<CChannelCfg> m_spChannelCfg;

    std::vector<std::shared_ptr<CBaseModule>> m_vspModules;
    IEventListener<CChannel> *m_pEventListener = nullptr;

    CSemaphore m_semaphore;
    ChannelState m_channelState;
    std::map<EventStatus, int> m_eventCountMap;
    std::atomic<bool> m_bEventThreadQuit{ false };
    std::unique_ptr<std::thread> m_upEventThread;
    std::queue<EventStatus> m_eventQueue;
    std::queue<CMDType> m_cmdQueue;
    std::mutex m_eventQueueMutex;
    std::mutex m_cmdQueueMutex;
    /*
     * The header elements which contain the head modules.
     */
    std::vector<std::shared_ptr<CModuleElem>> m_vspHeadModuleElems;
    /*
     * The head modules.
     */
    std::vector<std::shared_ptr<CBaseModule>> m_vspHeadModules;
};
#endif
