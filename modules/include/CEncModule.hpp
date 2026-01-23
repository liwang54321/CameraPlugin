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

#ifndef CENCMODULE_HPP
#define CENCMODULE_HPP

#include "CBaseModule.hpp"
#include "nvmedia_iep.h"

struct EncOption
{
  #if !NV_IS_SAFETY
    EncoderType enctype = EncoderType::H265;
  #else
    EncoderType enctype = EncoderType::H264;
  #endif
    uint32_t uWidth = 3848U;
    uint32_t uHeight = 2168U;
    uint32_t uMaxOutputBuffer = 1U;
    uint32_t uInstanceId = 0U;
    uint32_t uAverageBitrate = 20000000U;
    bool bDesensitized = false;
};

class CEncModule : public CBaseModule
{
  public:
    CEncModule(std::shared_ptr<CModuleCfg> spModuleCfg, IEventListener<CBaseModule> *pListener);
    virtual ~CEncModule() {};

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
    virtual NvError OnPacketGotten(CClientCommon *pClient, uint32_t uPacketIndex, bool *pHandled = nullptr) override;
    virtual NvError SetEofSyncObj(CClientCommon *pClient) override;
    virtual NvError GetEofSyncFence(CClientCommon *pClient, NvSciSyncFence *pPostfence) override;
    virtual NvError OnDataBufAttrListRecvd(CClientCommon *pClient, NvSciBufAttrList bufAttrList) override;
    virtual NvError ProcessPayload(CClientCommon *pClient, uint32_t uPacketIndex) override;
    virtual NvError OnProcessPayloadDone(CClientCommon *pClient, uint32_t uPacketIndex) override;

    virtual const OptionTable *GetOptionTable() const override;
    virtual const void *GetOptionBaseAddress() const override;

    virtual bool DumpEnabled() override;

  protected:
    virtual NvError UnregisterBufObj(NvSciBufObj bufObj) override;
    virtual NvError UnregisterSyncObj(NvSciSyncObj syncObj) override;
    virtual const std::string &GetOutputFileName() override;

  private:
    struct DestroyNvMediaIEP
    {
        void operator()(NvMediaIEP *p) const { NvMediaIEPDestroy(p); }
    };
    NvError InitEncoder(NvSciBufAttrList bufAttrList);
    NvError EncodeOneFrame(NvSciBufObj sciBufObj);
    NvError SetupIEPH264(NvSciBufAttrList bufAttrList, uint32_t uWidth, uint32_t uHeight);
#if !NV_IS_SAFETY
    NvError SetupIEPH265(NvSciBufAttrList bufAttrList, uint32_t uWidth, uint32_t uHeight);
#endif // !NV_IS_SAFETY

    std::unique_ptr<NvMediaIEP, DestroyNvMediaIEP> m_upNvMIEP;

    NvMediaEncodeConfigH264 m_stEncodeConfigH264Params{};
    EncOption m_encOption;
    uint16_t m_uWidth = 3848U;
    uint16_t m_uHeight = 2168U;
};
#endif
