/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef CPRODUCER_HPP
#define CPRODUCER_HPP

#include <atomic>
#include <deque>

#include "CClientCommon.hpp"
#include "CPoolManager.hpp"
#include "nvscibuf.h"

constexpr int64_t kHeartBeatTimeoutMs = 2000;

struct ChannelEntity
{
    std::mutex entityMutex;
    bool bC2C = false;
    std::atomic<bool> bLateAttach{ false };
    std::atomic<bool> bConnected{ false };
    std::atomic<bool> bIpcSetup{ false };
    std::chrono::time_point<std::chrono::steady_clock> lastHeartBeatTime;
    IpcEntity ipcEntity;
    std::vector<NvSciStreamBlock> vC2CBlockHandles;
};

class CProducer : public CClientCommon
{
  public:
    /** @brief Default constructor. */
    /** @brief Default destructor. */
    CProducer(std::shared_ptr<CClientCfg> spClientCfg, IModuleCallback *pCallback);
    virtual ~CProducer();
    virtual NvError Post(void *pBuffer, NvSciSyncFence *pPostFence);
    virtual NvError MultiPost(MultiPostInfo *postInfo);
    NvError ConnectIpc(const std::shared_ptr<CIpcEndpointElem> &spIpcSrc, int sensorId);
    virtual NvError GetConnectBlock(NvSciStreamBlock *pBlock) override;
    virtual bool IsProducer() { return true; }
    EventStatus HandlePoolEvent();

    virtual NvError Init() override;
    virtual void DeInit() override;
    virtual NvError Reconcile() override;
    virtual NvError Start() override;
    virtual NvError Stop() override;
    NvError AttachLateIpc(const std::string &sChannelName);
    void DetachIpc(const std::string &sChannelName);
    bool HasLateAttach();

  protected:
    virtual NvError FillBufAttrList(PacketElementType userType, NvSciBufAttrList *pBufAttrList) override;
    virtual NvError HandlePayload() override;
    NvError HandleStreamInit();
    virtual NvError HandleSetupComplete() override;
    virtual std::unique_ptr<CPoolManager> CreatePoolManager();
    virtual void OnClearedFenceReceived(uint32_t uPacketIndex) {}
    virtual NvError InsertPrefence(PacketElementType userType, uint32_t uPacketIndex, NvSciSyncFence *pPrefence);
    virtual NvError OnConnected() override;

    NvSciStreamBlock m_poolHandle = 0U;
    std::unique_ptr<CPoolManager> m_upPoolManger;

  private:
    NvError ConnectIpc(const std::string &sChannelName, IpcEntity &ipcEntity);
    NvError ConnectC2c(const std::string &sChannelName, IpcEntity &ipcEntity, std::vector<NvSciStreamBlock> &vBlocks);
    void HandleConsumerMsg(const std::string &sChannelName, void *pContent, uint32_t size);
    void MonitorHeartBeat();

    std::deque<ClientPacket *> m_packetQueue;
    std::unique_ptr<CEventHandler<CProducer>> m_upPoolEventHandler;
    NvSciStreamBlock m_multicastHandle = 0U;

    uint32_t m_uNumConsumers;
    std::mutex m_mutex;
    std::condition_variable m_stopConditionVar;
    std::atomic<bool> m_bSetupComplete{ false };
    std::mutex m_setupMutex;
    std::condition_variable m_cvSetupComplete;
    std::unordered_map<std::string, ChannelEntity> m_channelEntityMap;
};
#endif
