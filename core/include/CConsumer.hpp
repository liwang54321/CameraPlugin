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

#ifndef CCONSUMER_HPP
#define CCONSUMER_HPP

#include <atomic>

#include "nvscibuf.h"
#include "CClientCommon.hpp"
#include "CPoolManager.hpp"

class CConsumer : public CClientCommon
{
  public:
    /** @brief Default constructor. */
    CConsumer(std::shared_ptr<CClientCfg> spClientCfg, IModuleCallback *pCallback);
    /** @brief Default destructor. */
    virtual ~CConsumer() = default;

    virtual NvError GetConnectBlock(NvSciStreamBlock *pBlock) override;
    virtual void DeInit() override;
    virtual NvError Reconcile() override;
    virtual bool IsConsumer() { return true; }
    NvError ConnectIpc(const std::shared_ptr<CIpcEndpointElem> &spIpcDst, int sensorId);
    NvError InsertPrefence(uint32_t uPacketIndex);
    virtual NvError OnPacketProcessed(uint32_t uPacketIndex, NvSciSyncFence *pPostfence, bool bClearFence = true);
    NvError ReleasePacket(uint32_t packetIndex);

    uint64_t m_uFrameNum = 0U;

  protected:
    NvError HandlePayload() override;
    virtual NvError SetUnusedElement(uint32_t uElemIndex) override;
    NvError AcquirePacket(uint32_t &uPacketIndex);
    NvError ProcessPacket(uint32_t uPacketIndex);
    NvError PostProcessPacket(uint32_t uPacketIndex, NvSciSyncFence *pPostfence);
    NvError ReleasePacket(uint32_t uPacketIndex, NvSciSyncFence *pPostfence, bool bClearFence = true);
    NvError OnConnected() override;
    virtual NvError Stop() override;

  private:
    NvError ConnectIpc(const std::string &sChannelName, IpcEntity &ipcEntity);
    NvError ConnectC2c(const std::string &sChannelName, IpcEntity &ipcEntity, std::vector<NvSciStreamBlock> &vBlocks);
    std::unique_ptr<CPoolManager> CreatePoolManager(NvSciStreamBlock poolHandle);
    EventStatus HandlePoolEvent();
    void ReportHeartBeat();

    NvSciBufObj m_sciBufObjs[MAX_NUM_PACKETS]{ nullptr };
    IpcEntity m_ipcEntity;
    std::vector<NvSciStreamBlock> m_vC2cHandles;
    std::unique_ptr<CPoolManager> m_upPoolManager = { nullptr };
    std::unique_ptr<CEventHandler<CConsumer>> m_upPoolEventHandler = { nullptr };
    NvSciStreamBlock m_queueHandle = 0U;
    NvSciStreamBlock m_limiterHandle = 0U;
    std::shared_ptr<CMsgWriter> m_spMsgWriter = { nullptr };
    std::mutex m_mutex;
};

#endif
