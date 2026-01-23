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

#ifndef CBASEMODULE_HPP
#define CBASEMODULE_HPP

#include <unordered_map>
#include <chrono>
#include <mutex>

#include "nvscibuf.h"
#include "CConsumer.hpp"
#include "CProducer.hpp"
#include "CSyncAggregator.hpp"
#include "CBufAggregator.hpp"
#include "CUtils.hpp"
#include "COptionParser.hpp"
#include "CElementDescription.hpp"
#include "CProfiler.hpp"
#include "CFileSink.hpp"

class CBaseModule : public CClientCommon::IModuleCallback
{
  public:
    CBaseModule(std::shared_ptr<CModuleCfg> spModuleCfg, IEventListener<CBaseModule> *pListener);
    /** @brief Default destructor. */
    virtual ~CBaseModule();

    virtual NvError Init();
    virtual void DeInit();
    virtual NvError Reconcile();
    virtual NvError Start();
    virtual NvError PreStop();
    virtual NvError Stop();
    virtual NvError PostStop();
    /*
     * Get the module specific perfiling data which may include the
     * initialization latency, the submission latency and the
     * execution latency. The returned perfiling data will be
     * available before DeInit.
     */
    virtual NvError GetPerf(std::vector<Perf> &vPerf);
    const std::vector<std::shared_ptr<CBaseModule>> &GetDownstreamModules() const;

    NvError ConnectIpcDst(const std::shared_ptr<CIpcEndpointElem> &spIpcDst);
    NvError ConnectIpcSrc(const std::shared_ptr<CIpcEndpointElem> &spIpcSrc);
    NvError ConnectModule(const std::shared_ptr<CBaseModule> &spDownsteamModule);
    void PrintFps();
    virtual void OnError(int moduleId, uint32_t errorId) override;
    virtual void OnCommand(uint32_t uCmdId, void *pParam);

    inline const std::string &GetName() { return m_spModuleCfg->m_sName; }
    inline int GetSensorId() { return m_spModuleCfg->m_sensorId; }
    inline ModuleType GetModuleType() const { return m_spModuleCfg->m_moduleType; }
    inline void SetConsumerNumber(int consumerNumber) { m_numConsumers = consumerNumber; }

    NvError
    FillDataBufAttrList(CClientCommon *pClient, PacketElementType userType, NvSciBufAttrList *pBufAttrList) override;
    NvError FillSyncWaiterAttrList(CClientCommon *pClient,
                                   PacketElementType userType,
                                   NvSciSyncAttrList *pWaiterAttrList) override;
    NvError ParseOptions();
    virtual const OptionTable *GetOptionTable() const { return nullptr; };
    virtual const void *GetOptionBaseAddress() const { return nullptr; };

    virtual bool DumpEnabled();

    virtual const std::string &GetProdElemsName() const { return m_baseModuleOption.sElems; };

  protected:
    virtual NvError OnProcessPayloadDone(CClientCommon *pClient, uint32_t uPacketIndex) override;
    virtual NvError ProcessPayload(std::vector<NvSciBufObj> &vSrcBufObjs, NvSciBufObj dstBufObj, MetaData *pMetaData)
    {
        return NvError_Success;
    };
    virtual NvError OnPacketGotten(CClientCommon *pClient, uint32_t uPacketIndex, bool *pHandled = nullptr) override;
    virtual NvError FillMetaBufAttrList(CClientCommon *pClient, NvSciBufAttrList *pBufAttrList) override;
    virtual NvError SetEofSyncObj(CClientCommon *pClient) { return NvError_Success; }
    virtual NvError GetEofSyncFence(CClientCommon *pClient, NvSciSyncFence *pPostfence) { return NvError_Success; }
    virtual NvError OnDataBufAttrListRecvd(CClientCommon *pClient, NvSciBufAttrList bufAttrList)
    {
        return NvError_Success;
    }
    virtual NvError OnWaiterAttrEventRecvd(CClientCommon *pClient, bool &bHandled) override;
    virtual NvError RegisterBufObj(CClientCommon *pClient,
                                   PacketElementType userType,
                                   uint32_t uPacketIndex,
                                   NvSciBufObj bufObj) override;
    virtual NvError
    RegisterSignalSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj signalSyncObj) override;
    virtual NvError
    RegisterWaiterSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj waiterSyncObj) override;
    virtual NvError UnregisterBufObj(NvSciBufObj bufObj) { return NvError_Success; }
    virtual NvError UnregisterSyncObj(NvSciSyncObj syncObj) { return NvError_Success; }
    virtual NvError GetMinPacketCount(CClientCommon *pClient, uint32_t *pPacketCount) override;

    virtual void OnEvent(CClientCommon *pClient, EventStatus event) override;

    virtual const std::string &GetOutputFileName() { return m_sOutputFileName; }

    void UpdateFrameStatistics();

    CAppCfg *m_pAppCfg = nullptr;
    std::shared_ptr<CModuleCfg> m_spModuleCfg;
    std::shared_ptr<CProducer> m_spProducer;
    std::vector<std::shared_ptr<CConsumer>> m_vspConsumers;
    NvSciSyncObj m_signalSyncObj = nullptr;
    std::mutex m_FrameMutex;
    std::chrono::time_point<std::chrono::steady_clock> m_uStarTimePoint{};
    std::chrono::time_point<std::chrono::steady_clock> m_uEndTimePoint{};
    uint64_t m_uFrameNum{ 0U };
    uint64_t m_uPrevFrameNum{ 0U };
    /*
     * The profiler instance.
     */
    std::unique_ptr<CProfiler> m_upProfiler;
    /*
     * The downstream modules.
     */
    std::vector<std::shared_ptr<CBaseModule>> m_vspDownstreamModules;

    /*
     * Whether the module has upstream modules or IpcDst.
     */
    bool m_bHasUpstream = false;

    /*
     * Whether the module has downstream modules or IpcDst.
     */
    bool m_bHasDownstream = false;

    /*
     * Current packet related meta data.
     */
    MetaData *m_pMetaData = nullptr;

  private:
    struct BaseModuleOption
    {
        std::string sLateMods = "";
        std::string sElems = "";
        bool bPassthrough = false;
        bool bFileSink = false;
        uint32_t uLimitNum = 0U;
        std::string sQueueType = "Fifo";
        uint32_t uDumpStartFrame = DUMP_START_FRAME;
        uint32_t uDumpEndFrame = DUMP_START_FRAME;
    };
    std::shared_ptr<CProducer> GetProducer();
    std::shared_ptr<CConsumer> GetConsumer();
    void CreateLateDstModules(std::vector<std::shared_ptr<CBaseModule>> &vspModules);

    std::vector<std::shared_ptr<CClientCommon>> m_vspClients;
    IEventListener<CBaseModule> *m_pEventListener = nullptr;
    uint32_t m_uClientSetupCompleteCount{ 0 };
    uint32_t m_uClientConnectCount{ 0 };
    std::vector<NvSciBufObj> m_vBufObjs;
    std::vector<NvSciSyncObj> m_vSyncObjs;
    std::unique_ptr<CBufAggregator> m_upBufAggregator = nullptr;

    std::unordered_set<ModuleType> m_lateDstModuleTypeSet;
    int m_numConsumers = 1;
    QueueType m_queueType = QueueType::Fifo;

  protected:
    std::shared_ptr<CSyncAggregator> m_spSyncAggregator = nullptr;

    std::unique_ptr<uint8_t[]> m_upOutputBuf = nullptr;
    uint32_t m_uOutputBufValidLen = 0;
    uint32_t m_uOutputBufCapacity = 0;
    std::string m_sOutputFileName = "";
    std::unique_ptr<CAbstractFileSink> m_upFileSink = nullptr;

    BaseModuleOption m_baseModuleOption;

  public:
    static const std::unordered_map<std::string, Option> m_baseModuleOptionTable;
};
#endif
