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

#ifndef CPVAMODULE_HPP
#define CPVAMODULE_HPP

#include <atomic>

#include "CBaseModule.hpp"
#include "CPvaLowPowerAlgos.hpp"

struct PvaOption
{
    bool bRunDLPipeline{ true };
    std::string sData;
    uint32_t uVpuId{ 0 };
};

class CPvaModule : public CBaseModule
{
  public:
    CPvaModule(std::shared_ptr<CModuleCfg> spModuleCfg, IEventListener<CBaseModule> *pListener);
    virtual ~CPvaModule();

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
    virtual NvError
    RegisterBufObj(CClientCommon *pClient, PacketElementType userType, uint32_t uPacketIndex, NvSciBufObj) override;
    virtual NvError
    RegisterSignalSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj signalSyncObj) override;
    virtual NvError
    RegisterWaiterSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj waiterSyncObj) override;
    virtual NvError InsertPrefence(CClientCommon *pClient,
                                   PacketElementType userType,
                                   uint32_t uPacketIndex,
                                   NvSciSyncFence *pPrefence) override;
    virtual NvError OnDataBufAttrListRecvd(CClientCommon *pClient, NvSciBufAttrList bufAttrList) override;
    virtual NvError ProcessPayload(CClientCommon *pClient, uint32_t uPacketIndex) override;
    virtual NvError OnProcessPayloadDone(CClientCommon *pClient, uint32_t uPacketIndex) override;

    virtual const OptionTable *GetOptionTable() const override;
    virtual const void *GetOptionBaseAddress() const override;

  private:
    static constexpr int kMAX_NUM_PEDESTRIAN = 128;
    struct BBox
    {
        int iCVNumSelectedDetections = 0;
        int iCVArrPosX[kMAX_NUM_PEDESTRIAN] = { 0 };
        int iCVArrPosY[kMAX_NUM_PEDESTRIAN] = { 0 };
        int iCVArrPosW[kMAX_NUM_PEDESTRIAN] = { 0 };
        int iCVArrPosH[kMAX_NUM_PEDESTRIAN] = { 0 };

        int iNumSelectedDetections = 0;
        int iArrPosX[kMAX_NUM_PEDESTRIAN] = { 0 };
        int iArrPosY[kMAX_NUM_PEDESTRIAN] = { 0 };
        int iArrPosW[kMAX_NUM_PEDESTRIAN] = { 0 };
        int iArrPosH[kMAX_NUM_PEDESTRIAN] = { 0 };
    };

    NvError PvaAlgosInit(const std::string &assetPath, uint32_t width, uint32_t height, uint32_t *planePitches);
    NvError PvaAlgosDeInit();
    NvError PvaAlgosPreprocess(const NvSciBufObj& nvsciBuf, float_t *rgb_buf);
    NvError PvaAlgosInference(const float_t *rgb_buf, BBox &result);

    std::unique_ptr<float_t[]> m_upRgbBuf{ nullptr };
    PvaAlgos *m_pPvaAlgos{ nullptr };
    PvaOption m_pvaOption;
    NvSciBufObj m_sciBufObjs[MAX_NUM_PACKETS]{ nullptr };
    BufferAttrs m_bufAttrs[MAX_NUM_PACKETS];
};
#endif
