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

#ifndef CWFDDISPLAYMODULE_HPP
#define CWFDDISPLAYMODULE_HPP

#define WFD_NVX_create_source_from_nvscibuf
#define WFD_WFDEXT_PROTOTYPES

#include <WF/wfd.h>
#include <WF/wfdext.h>

#include "CBaseModule.hpp"
#include "CUtils.hpp"

typedef struct
{
    uint32_t uDeviceIdx = 0U;   // only one device
    uint32_t uPortIdx = 0U;     // range: 0-1 on Orin DOS6.0, 0-3 on Thor DOS7.0
    uint32_t uPipelineIdx = 0U; // range: 0-1
    uint32_t uWidth = 3840U;
    uint32_t uHeight = 2160U;
    std::string sColorType = "";
    std::string sImageLayout = "";
} WFDResInputInfo;

typedef struct
{
    WFDDevice wfdDevice = WFD_INVALID_HANDLE;
    WFDPort wfdPort = WFD_INVALID_HANDLE;
    WFDPipeline wfdPipeline = WFD_INVALID_HANDLE;
} WFDResource;

class CWFDResManager
{
  public:
    ~CWFDResManager();

    static std::shared_ptr<CWFDResManager> GetInstance();
    NvError CreateResouce(const WFDResInputInfo *const pInputInfo, WFDResource *pWfdRes);

  private:
    CWFDResManager();
    CWFDResManager(const CWFDResManager &obj);
    CWFDResManager &operator=(const CWFDResManager &obj);

    WFDDevice GetDisplayDevice(uint32_t uDeviceIdx);
    WFDPort GetWfdPort(uint32_t uDeviceIdx, uint32_t uPortIdx);
    WFDPipeline GetPipeline(uint32_t uDeviceIdx, uint32_t uPortIdx, uint32_t uPipelineIdx);
    NvError CreatePipeline(WFDDevice wfdDev, WFDPort wfdPort, uint32_t uPipelineIdx, WFDPipeline &wfdPipeline);

    typedef std::pair<WFDPort, std::vector<WFDPipeline>> WFDPortWithPipelines;
    typedef std::pair<WFDDevice, std::vector<WFDPortWithPipelines>> WFDDeviceWithPortsInfo;

    std::vector<WFDDeviceWithPortsInfo> m_vWfdResInfos;
    std::mutex m_mutex;
};

class CWFDDisplayModule : public CBaseModule
{
  public:
    CWFDDisplayModule(std::shared_ptr<CModuleCfg> spModuleCfg, IEventListener<CBaseModule> *pListener);
    virtual ~CWFDDisplayModule();
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

    virtual NvError GetEofSyncFence(CClientCommon *pClient, NvSciSyncFence *pPostfence) override;
    virtual NvError ProcessPayload(CClientCommon *pClient, uint32_t uPacketIndex) override;
    virtual NvError OnDataBufAttrListRecvd(CClientCommon *pClient, NvSciBufAttrList bufAttrList) override;

    virtual const OptionTable *GetOptionTable() const override;
    virtual const void *GetOptionBaseAddress() const override;

    static std::shared_ptr<CSyncAggregator> GetSyncAggregator();

  private:
    bool IsFull(CClientCommon *pClient);
    std::shared_ptr<CWFDResManager> m_spWFDResInst = nullptr;
    WFDResource m_wfdRes;
    WFDSource m_wfdSources[MAX_NUM_PACKETS];
    int32_t m_dstWidth = 0;
    int32_t m_dstHeight = 0;

    static const std::unordered_map<std::string, Option> m_wfdDisplayOptionTable;
    static const std::unordered_set<std::string> m_wfdSupportedColor;
    WFDResInputInfo m_wfdResInputInfo;
    std::vector<NvSciSyncFence> m_postFences;
    bool m_isFrameDrop = false;
};
#endif
