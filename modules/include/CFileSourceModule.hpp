/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef CFILESOURCEMODULE_H
#define CFILESOURCEMODULE_H

#include <deque>

#include "CBaseModule.hpp"
#include "CEventHandler.hpp"
#include "CFrameHandler.hpp"
#include "Common.hpp"
#include "CUtils.hpp"

struct FileSourceOption
{
    // required options
    FileSourceType type = FileSourceType::UNDEFINED;
    std::string sPath = "";
    uint32_t uWidth = 0U;
    uint32_t uHeight = 0U;
    // optional options
    uint32_t uInstanceId = 0U;
    std::string sROIFilePath;
    std::string sROIParams;
};

class CFileSourceModule : public CBaseModule
{
  public:
    CFileSourceModule(std::shared_ptr<CModuleCfg> spModuleCfg, IEventListener<CBaseModule> *pListener);
    virtual ~CFileSourceModule();

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

  protected:
    std::unique_ptr<CEventHandler<CFileSourceModule>> m_upEventHandler;
    uint64_t m_uFrameSequenceNumber{ 0 };

  private:
    FileSourceOption m_fileSrcOption;
    std::unique_ptr<CFrameHandler> m_upFrameHandler;
    std::vector<std::vector<NvMediaRect>> m_rois;
};
#endif
