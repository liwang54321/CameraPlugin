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

#ifndef CNVM2DMODULE_HPP
#define CNVM2DMODULE_HPP

#include <mutex>

#include "CBaseModule.hpp"
#include "CVicConfigurator.hpp"
#include "nvmedia_2d_sci.h"

enum class VicOperationType : uint8_t
{
    Stitch,
    Convert
};

struct Nvm2DOption
{
    VicOperationType type = VicOperationType::Stitch;
};

class CNvm2dModule : public CBaseModule
{
  public:
    CNvm2dModule(std::shared_ptr<CModuleCfg> spModuleCfg, IEventListener<CBaseModule> *pListener);
    virtual ~CNvm2dModule();

    virtual NvError Init() override;
    virtual void DeInit() override;
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
    virtual NvError SetEofSyncObj(CClientCommon *pClient) override;
    virtual NvError GetEofSyncFence(CClientCommon *pClient, NvSciSyncFence *pPostfence) override;
    virtual NvError ProcessPayload(CClientCommon *pClient, uint32_t uPacketIndex) override;
    virtual NvError ProcessPayload(std::vector<NvSciBufObj> &vSrcBufObjs,
                                   NvSciBufObj dstBufObj,
                                   MetaData *pMetaData = nullptr) override;
    virtual NvError OnDataBufAttrListRecvd(CClientCommon *pClient, NvSciBufAttrList bufAttrList) override;

    virtual const OptionTable *GetOptionTable() const override;
    virtual const void *GetOptionBaseAddress() const override;
    virtual const std::string &GetProdElemsName() const override { return nvm2dopaque; };

  protected:
    virtual NvError UnregisterBufObj(NvSciBufObj bufObj) override;
    virtual NvError UnregisterSyncObj(NvSciSyncObj syncObj) override;
    struct DestroyNvMedia2DDevice
    {
        void operator()(NvMedia2D *p) const { NvMedia2DDestroy(p); }
    };

  private:
    std::unique_ptr<NvMedia2D, DestroyNvMedia2DDevice> m_up2DDevice;
    NvMedia2DComposeParameters m_2DParams = 0U;
    NvMedia2DComposeResult m_composeResult{};
    std::mutex m_2dMutex;
    NvSciSyncObj m_2DSignalSyncObj = nullptr;
    //temporarily defined here
    std::unique_ptr<CVicConfigurator> m_upVicConfigurator = nullptr;

  private:
    Nvm2DOption m_nvm2DOption;
    const std::string nvm2dopaque = "nvm2dopaque";
};
#endif
