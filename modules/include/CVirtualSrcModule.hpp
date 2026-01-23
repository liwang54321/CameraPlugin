/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef CVIRTUALSRCMODULE_H
#define CVIRTUALSRCMODULE_H

#include <deque>

#include "CBaseModule.hpp"
#include "CEventHandler.hpp"

struct VirtualSrcOption
{
    uint32_t uWidth = 1920U;
    uint32_t uHeight = 1080U;
    std::string sLayout = "BlockLinear";
    float fFramerate = 30;
};

class CVirtualSrcModule : public CBaseModule
{
  public:
    CVirtualSrcModule(std::shared_ptr<CModuleCfg> spModuleCfg, IEventListener<CBaseModule> *pListener);
    virtual ~CVirtualSrcModule();

    virtual NvError Init() override;
    virtual void DeInit() override;
    virtual NvError Start() override;
    virtual NvError Stop() override;

    virtual NvError
    FillDataBufAttrList(CClientCommon *pClient, PacketElementType userType, NvSciBufAttrList *pBufAttrList) override;
    virtual NvError FillSyncSignalerAttrList(CClientCommon *pClient,
                                             PacketElementType userType,
                                             NvSciSyncAttrList *pSignalerAttrList) override;
    virtual NvError FillSyncWaiterAttrList(CClientCommon *pClient,
                                           PacketElementType userType,
                                           NvSciSyncAttrList *pWaiterAttrList) override;
    virtual NvError RegisterBufObj(CClientCommon *pClient,
                                   PacketElementType userType,
                                   uint32_t uPacketIndex,
                                   NvSciBufObj bufObj) override;
    virtual NvError
    RegisterSignalSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj signalSyncObj) override;
    virtual NvError
    RegisterWaiterSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj waiterSyncObj) override;
    virtual NvError InsertPrefence(CClientCommon *pClient,
                                   PacketElementType userType,
                                   uint32_t uPacketIndex,
                                   NvSciSyncFence *pPrefence) override;
    virtual NvError ProcessPayload(CClientCommon *pClient, uint32_t uPacketIndex) override;
    virtual NvError OnPacketGotten(CClientCommon *pClient, uint32_t uPacketIndex, bool *pHandled = nullptr) override;
    virtual const OptionTable *GetOptionTable() const override;
    virtual const void *GetOptionBaseAddress() const override;

  protected:
    EventStatus Generator();
    std::unique_ptr<CEventHandler<CVirtualSrcModule>> m_upEventHandler;
    std::deque<NvSciBufObj> m_bufObjQueue;
    std::vector<NvSciBufObj> m_vBufObjs;
    uint64_t m_uFrameSequenceNumber{ 0 };
    VirtualSrcOption m_virtualSrcOption;
    NvSciBufAttrValImageLayoutType m_LayoutType;
    std::chrono::nanoseconds m_FrameTime;
    uint64_t m_uWorkTime{};
};
#endif
