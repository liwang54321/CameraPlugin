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

#ifndef CVULKANSCMODULE_HPP
#define CVULKANSCMODULE_HPP

#include<array>
#include "CBaseModule.hpp"
#include "CVulkanSCEngine.hpp"

struct VulkanSCOption
{
    uint32_t uWidth  = 3840U;
    uint32_t uHeight = 2160U;
    bool bUseVkSemaphore = false;
    std::string sColorType = "ABGR";
};

class CVulkanSCModule : public CBaseModule
{
 public:
    CVulkanSCModule(std::shared_ptr<CModuleCfg> spModuleCfg, IEventListener<CBaseModule> *pListener);
    virtual ~CVulkanSCModule() = default;

    virtual NvError Init() override;
    virtual void DeInit() override;
    virtual NvError FillDataBufAttrList(CClientCommon *pClient,
                                           PacketElementType userType,
                                           NvSciBufAttrList *pBufAttrList) override;
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
    virtual NvError RegisterSignalSyncObj(CClientCommon *pClient,
                                             PacketElementType userType,
                                             NvSciSyncObj signalSyncObj) override;
    virtual NvError RegisterWaiterSyncObj(CClientCommon *pClient,
                                             PacketElementType userType,
                                             NvSciSyncObj waiterSyncObj) override;
    virtual NvError InsertPrefence(CClientCommon *pClient,
                                      PacketElementType userType,
                                      uint32_t uPacketIndex,
                                      NvSciSyncFence *pPrefence) override;
    virtual NvError GetEofSyncFence(CClientCommon *pClient, NvSciSyncFence *pPostfence) override;
    virtual NvError ProcessPayload(CClientCommon *pClient, uint32_t uPacketIndex) override;
    virtual NvError OnProcessPayloadDone(CClientCommon *pClient, uint32_t uPacketIndex) override;

    virtual const OptionTable *GetOptionTable() const override;
    virtual const void *GetOptionBaseAddress() const override;

    static const std::unordered_map<std::string, Option> m_vulkanSCOptionTable;

protected:
    virtual const std::string &GetOutputFileName() override;

private:
    std::shared_ptr<CVulkanSCEngine> m_spVKSCEngine;
    VulkanSCOption m_VulkanSCOption;
    static const std::unordered_set<std::string> m_vulkanSCSupportedColor;
};

#endif
