/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef CVIRTUALDSTMODULE_H
#define CVIRTUALDSTMODULE_H

#include "CBaseModule.hpp"

typedef struct
{
    uint32_t uWidth = 3840U;
    uint32_t uHeight = 2160U;
    std::string sColorType = "";
    std::string sImageLayout = "";
} VirtualDstInputInfo;

class CVirtualDstModule : public CBaseModule
{
  public:
    CVirtualDstModule(std::shared_ptr<CModuleCfg> spModuleCfg, IEventListener<CBaseModule> *pListener);
    virtual ~CVirtualDstModule();
    virtual NvError Init() override;
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

    virtual const OptionTable *GetOptionTable() const override;
    virtual const void *GetOptionBaseAddress() const override;

  private:
    static const std::unordered_map<std::string, Option> m_virtualDstOptionTable;
    static const std::unordered_set<std::string> m_virtualDstSupportedColor;
    VirtualDstInputInfo m_virtualDstInputInfo;
};
#endif
