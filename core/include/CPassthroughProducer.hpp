/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef CPASSTHROUGHPRODUCER_HPP
#define CPASSTHROUGHPRODUCER_HPP

#include "CProducer.hpp"
#include "CPassthroughPoolManager.hpp"
#include "CPassthroughConsumer.hpp"

#include <mutex>

class CPassthroughProducer : public CProducer,
                             public IDownstreamCallback,
                             public CPassthroughPoolManager::IPassthroughPoolCallback
{
  public:
    CPassthroughProducer(std::shared_ptr<CClientCfg> spClientCfg, IModuleCallback *pCallback);
    /** @brief Default destructor. */
    virtual ~CPassthroughProducer() = default;

    // void SetCallback(ICallback *pCallback);
    void SetCallback(IUpstreamCallback *pCallback);
    virtual NvError OnUpstreamBufAttrRecvd(const ElemBufAttr (&elemBufAttrs)[MAX_NUM_ELEMENTS],
                                           uint32_t uNumElems) override;
    virtual NvError OnUpstreamWaiterAttrRecvd(const ElemSyncAttr &elemSyncAttr) override;
    virtual NvError OnUpstreamPacketCreated(const std::vector<ElemBufObj> &vElemBufObjs) override;
    virtual NvError OnUpstreamPacketProcessed(NvSciBufObj *pBufObj, NvSciSyncFence *pPostfence) override;

    virtual NvError OnDownstreamBufAttrRecvd(const ElemBufAttr (&elemBufAttrs)[MAX_NUM_ELEMENTS],
                                             uint32_t uNumElems) override;
    virtual NvError Init() override;
    virtual NvError PreStop() override;
    virtual NvError Stop() override;
    virtual NvError PostStop() override;

  protected:
    virtual std::unique_ptr<CPoolManager> CreatePoolManager() override;
    virtual NvError HandleSyncSupport() override;
    virtual NvError FillSyncWaiterAttrList(PacketElementType userType, NvSciSyncAttrList *pWaiterAttrList) override;
    // MapDataBuffer is already called in CPassthroughConsumer, no need to call it again in CPassthroughProducer
    virtual void MapDataBuffer(PacketElementType userType, uint32_t uPacketIndex, NvSciBufObj bufObj) {};
    virtual NvError RegisterSignalSyncObj(uint32_t uElemId, NvSciSyncObj signalSyncObj) override;
    virtual NvError RegisterWaiterSyncObj(uint32_t uElemId, NvSciSyncObj waiterSyncObj) override;
    virtual NvError
    InsertPrefence(PacketElementType userType, uint32_t uPacketIndex, NvSciSyncFence *pPrefence) override;
    virtual void OnClearedFenceReceived(uint32_t uPacketIndex) override;

  private:
    void RemoveUnusedElements();
    IUpstreamCallback *m_pUpstreamCallback = nullptr;
    std::mutex m_mutex;
    std::atomic<bool> m_bPendingSyncSupport{ false };
    ElemSyncAttr m_upstreamSyncAttr;
};

#endif
