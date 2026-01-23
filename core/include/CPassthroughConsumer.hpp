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

#ifndef CPASSTHROUGHCONSUMER_HPP
#define CPASSTHROUGHCONSUMER_HPP

#include "CConsumer.hpp"

#include <mutex>

class IDownstreamCallback
{
  public:
    virtual NvError OnUpstreamBufAttrRecvd(const ElemBufAttr (&elemBufAttrs)[MAX_NUM_ELEMENTS], uint32_t uNumElems) = 0;
    virtual NvError OnUpstreamWaiterAttrRecvd(const ElemSyncAttr &elemSyncAttr) = 0;
    virtual NvError OnUpstreamPacketCreated(const std::vector<ElemBufObj> &vElemBufObjs) = 0;
    virtual NvError OnUpstreamPacketProcessed(NvSciBufObj *pBufObj, NvSciSyncFence *pPostfence) = 0;

  protected:
    IDownstreamCallback() = default;
    virtual ~IDownstreamCallback() = default;
};

class IUpstreamCallback
{
  public:
    virtual NvError OnDownstreamBufAttrRecvd(const ElemBufAttr (&elemBufAttrs)[MAX_NUM_ELEMENTS],
                                             uint32_t uNumElems) = 0;
    virtual NvError OnSyncObjCreated(NvSciSyncObj signalSyncObj) = 0;
    virtual NvError OnDownstreamSyncObjRecvd(NvSciSyncObj waiterSyncObj) = 0;
    virtual void OnDownstreamPacketProcessed(NvSciBufObj *pBufObj, NvSciSyncFence *pPrefence) = 0;

  protected:
    IUpstreamCallback() = default;
    virtual ~IUpstreamCallback() = default;
};

class CPassthroughConsumer : public CConsumer, public IUpstreamCallback
{
  public:
    CPassthroughConsumer(std::shared_ptr<CClientCfg> spClientCfg, IModuleCallback *pCallback);
    /** @brief Default destructor. */
    virtual ~CPassthroughConsumer() = default;

    // void SetCallback(ICallback *pCallback);
    void SetCallback(IDownstreamCallback *pCallback);

    virtual NvError Reconcile() override;
    virtual NvError FillBufAttrList(PacketElementType userType, NvSciBufAttrList *pBufAttrList) override;
    virtual NvError OnDownstreamBufAttrRecvd(const ElemBufAttr (&elemBufAttrs)[MAX_NUM_ELEMENTS], uint32_t uNumElems);
    virtual NvError OnDownstreamSyncObjRecvd(NvSciSyncObj syncObj) override;
    virtual NvError OnSyncObjCreated(NvSciSyncObj signalSyncObj) override;
    virtual void OnDownstreamPacketProcessed(NvSciBufObj *pBufObj, NvSciSyncFence *pPrefence) override;

  protected:
    virtual NvError OnBufAttrListRecvd(const ElemBufAttr (&elemBufAttrs)[MAX_NUM_ELEMENTS],
                                       uint32_t uNumElems) override;
    virtual NvError HandleSyncExport() override;
    virtual NvError
    OnPacketProcessed(uint32_t uPacketIndex, NvSciSyncFence *pPostfence, bool bClearFence = true) override;
    virtual NvError OnPacketCreated(const std::vector<ElemBufObj> &vElemBufObjs) override;

  private:
    ElemBufAttr *GetDownstreamBufAttr(PacketElementType userType);
    IDownstreamCallback *m_pDownstreamCallback = nullptr;
    std::mutex m_mutex;
    std::atomic<bool> m_bPendingReconcile{ false };
    ElemBufAttr m_downstreamBufAttrs[MAX_NUM_ELEMENTS];
    uint32_t m_uNumDownstreamElems = 0U;
};

#endif
