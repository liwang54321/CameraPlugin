/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef CCLIENTCOMMON_H
#define CCLIENTCOMMON_H

#include <atomic>
#include <condition_variable>
#include <cstdarg>
#include <cstring>
#include <iostream>
#include <string.h>
#include <thread>
#include <unistd.h>

#include "CConfig.hpp"
#include "CEventHandler.hpp"
#include "CProfiler.hpp"
#include "CUtils.hpp"
#include "Common.hpp"
#include "nvscistream.h"

constexpr NvSciStreamCookie cookieBase = 0xC00C1E4U;
constexpr uint32_t kHeartBeatIntervalMs = 100U;

// Define Packet struct which is used by the client
typedef struct {
    /* The client's handle for the packet */
    NvSciStreamCookie cookie;
    /* The NvSciStream's Handle for the packet */
    NvSciStreamPacket handle;
    /* NvSci buffer objects for the packet's buffer */
    NvSciBufObj bufObjs[MAX_NUM_ELEMENTS];
    NvSciSyncFence preFence[MAX_NUM_CONSUMERS][MAX_NUM_ELEMENTS];
} ClientPacket;

// Define Packet struct which is used by the client
typedef struct {
    /* Packet index */
    uint32_t uPacketIndex;
    /* Multi element post fence */
    std::vector<NvSciSyncFence> vPostFences;
    /* Multi element id */
    std::vector<uint32_t> vElementIndexs;
} MultiPostInfo;

struct MetaData {
    static constexpr uint32_t kMaxROIRegions = 64U;

public:
    MetaData(uint64_t uFrameCaptureTSC = 0, uint64_t uFrameCaptureStartTSC = 0,
             bool bFrameSeqNumValid = false, uint64_t uFrameSequenceNumber = 0,
             uint64_t uSendTSC = 0, uint64_t uReceiveTSC = 0, bool bTriggerEncodingValid = false,
             bool bTriggerEncoding = false) {
        Set(uFrameCaptureTSC, uFrameCaptureStartTSC, bFrameSeqNumValid, uFrameSequenceNumber,
            uSendTSC, uReceiveTSC, bTriggerEncodingValid, bTriggerEncoding);
    }

    void Set(uint64_t uFrameCaptureTSC = 0, uint64_t uFrameCaptureStartTSC = 0,
             bool bFrameSeqNumValid = false, uint64_t uFrameSequenceNumber = 0,
             uint64_t uSendTSC = 0, uint64_t uReceiveTSC = 0, bool bTriggerEncodingValid = false,
             bool bTriggerEncoding = false)

    {
        this->uFrameCaptureTSC = uFrameCaptureTSC;
        this->uFrameCaptureStartTSC = uFrameCaptureStartTSC;

        this->bFrameSeqNumValid = bFrameSeqNumValid;
        this->uFrameSequenceNumber = uFrameSequenceNumber;
        this->uSendTSC = uSendTSC;
        this->uReceiveTSC = uReceiveTSC;
        this->uNumROIRegions = 0U;
        memset(this->ROIRect, 0, sizeof(NvMediaRect) * kMaxROIRegions);
        this->bTriggerEncodingValid = bTriggerEncodingValid;
        this->bTriggerEncoding = bTriggerEncoding;
    }

    std::string ToString() {
        std::string sMetaDataString =
            "FrameCaptureTSC=" + std::to_string(uFrameCaptureTSC) +
            " FrameCaptureStartTSC=" + std::to_string(uFrameCaptureStartTSC) +
            " FrameSendTSC=" + std::to_string(uSendTSC) +
            " FrameReceiveTSC=" + std::to_string(uReceiveTSC) +
            " FrameSeqNumValid=" + std::to_string(bFrameSeqNumValid) +
            " FrameSequenceNumber=" + std::to_string(uFrameSequenceNumber) +
            " TriggerEncodingValid=" + std::to_string(bTriggerEncodingValid) +
            " TriggerEncoding=" + std::to_string(bTriggerEncoding);

        return sMetaDataString;
    }

    ~MetaData() { Set(0U, 0U, false, 0U); }

    /** Holds the TSC timestamp of the frame capture */
    uint64_t uFrameCaptureTSC;
    /** Holds the TSC timestamp of the start of frame for capture. */
    uint64_t uFrameCaptureStartTSC;
    /** Holds a flag which enables OR DISABLES the frame sequence number block.*/
    bool bFrameSeqNumValid;
    /** Holds the sensor frame sequence number value.*/
    uint64_t uFrameSequenceNumber;
    /** Holds the TSC timestamp of sending the frame.*/
    uint64_t uSendTSC;
    /** Holds the TSC timestamp of receiving the frame.*/
    uint64_t uReceiveTSC;
    /** Num of Region of interest (ROI) */
    uint32_t uNumROIRegions;
    /** Region of interest (ROI) */
    NvMediaRect ROIRect[kMaxROIRegions];
    /** flag to trigger encoding valid in low power mode */
    bool bTriggerEncodingValid;
    /** flag to trigger encoding in low power mode */
    bool bTriggerEncoding;
};

struct IpcEntity {
    NvSciIpcEndpoint endPoint = 0U;
    NvSciStreamBlock ipcBlock = 0U;
    uint32_t uLimitNum = 0U;
    NvSciStreamBlock limiterBlock = 0U;
};

enum class HeartBeatMsgType : uint8_t { CONNECT_REQUEST, HEARTBEAT, DISCONNECT_REQUEST };

class CClientCommon {
public:
    class IModuleCallback {
    public:
        virtual NvError ProcessPayload(CClientCommon *pClient, uint32_t uPacketIndex) = 0;
        virtual NvError ProcessPayload(std::vector<NvSciBufObj> &vSrcBufObjs, NvSciBufObj dstBufObj,
                                       MetaData *pMetaData = nullptr) = 0;
        virtual NvError OnProcessPayloadDone(CClientCommon *pClient, uint32_t uPacketIndex) = 0;
        virtual NvError OnPacketGotten(CClientCommon *pClient, uint32_t uPacketIndex,
                                       bool *pHandled = nullptr) = 0;

        virtual NvError InsertPrefence(CClientCommon *pClient, PacketElementType userType,
                                       uint32_t uPacketIndex, NvSciSyncFence *pPrefence) = 0;
        virtual NvError SetEofSyncObj(CClientCommon *pClient) = 0;
        virtual NvError GetEofSyncFence(CClientCommon *pClient, NvSciSyncFence *pPostfence) = 0;
        virtual NvError OnDataBufAttrListRecvd(CClientCommon *pClient,
                                               NvSciBufAttrList bufAttrList) = 0;
        virtual NvError OnWaiterAttrEventRecvd(CClientCommon *pClient, bool &bHandled) = 0;
        virtual NvError FillDataBufAttrList(CClientCommon *pClient, PacketElementType userType,
                                            NvSciBufAttrList *pBufAttrList) = 0;
        virtual NvError FillMetaBufAttrList(CClientCommon *pClient,
                                            NvSciBufAttrList *pBufAttrList) = 0;
        virtual NvError FillSyncSignalerAttrList(CClientCommon *pClient, PacketElementType userType,
                                                 NvSciSyncAttrList *pSignalerAttrList) = 0;
        virtual NvError FillSyncWaiterAttrList(CClientCommon *pClient, PacketElementType userType,
                                               NvSciSyncAttrList *pWaiterAttrList) = 0;
        virtual NvError RegisterSignalSyncObj(CClientCommon *pClient, PacketElementType userType,
                                              NvSciSyncObj signalSyncObj) = 0;
        virtual NvError RegisterWaiterSyncObj(CClientCommon *pClient, PacketElementType userType,
                                              NvSciSyncObj waiterSyncObj) = 0;
        virtual NvError RegisterBufObj(CClientCommon *pClient, PacketElementType userType,
                                       uint32_t uPacketIndex, NvSciBufObj bufObj) = 0;
        virtual NvError GetMinPacketCount(CClientCommon *pClient, uint32_t *uPacketCount) = 0;
        virtual void OnEvent(CClientCommon *pClient, EventStatus event) = 0;
        virtual void OnError(int moduleId, uint32_t errorId) = 0;

    protected:
        IModuleCallback() = default;
        virtual ~IModuleCallback() = default;
    };

    CClientCommon(std::shared_ptr<CClientCfg> spClientCfg,
                  CClientCommon::IModuleCallback *pCallback);
    virtual ~CClientCommon();

    virtual NvError Init();
    virtual void DeInit() = 0;
    virtual NvError Reconcile();
    virtual NvError Start();
    virtual NvError PreStop();
    virtual NvError Stop();
    virtual NvError PostStop();

    virtual bool IsConsumer() { return false; }
    virtual bool IsProducer() { return false; }

    inline const std::string &GetName() { return m_spClientCfg->m_sName; }
    inline void *GetMetaPtr(uint32_t uPacketIndex) { return m_metaPtrs[uPacketIndex]; }
    friend NvError ClientConnect(std::shared_ptr<CClientCommon> spSrcClient,
                                 std::shared_ptr<CClientCommon> spDstClient);
    NvSciBufObj *GetBufObj(uint32_t uPacketIndex);
    MetaData *GetMetaPtr(NvSciBufObj bufObj);
    NvError AllocSignalSyncObj(std::vector<NvSciSyncAttrList> &vUnreconciledAttrLists,
                               NvSciSyncObj *pSignalSyncObj);
    NvError RecvWaiterAttr(ElemSyncAttr &elemSyncAttr);
    NvError RegisterSignalSyncObj(NvSciSyncObj signalSyncObj);
    NvError ExportSignalSyncObj(NvSciSyncObj syncObj);
    NvError GetElemIndexByUserType(PacketElementType userType, uint32_t &uElementIndex);
    NvSciError IsPostFenceExpired(NvSciSyncFence *pFence);

protected:
    virtual NvError FillBufAttrList(PacketElementType userType, NvSciBufAttrList *pBufAttrList);
    virtual NvError FillSyncWaiterAttrList(PacketElementType userType,
                                           NvSciSyncAttrList *pWaiterAttrList);
    virtual NvError RegisterSignalSyncObj(uint32_t uElemId, NvSciSyncObj signalSyncObj);
    virtual NvError RegisterWaiterSyncObj(uint32_t uElemId, NvSciSyncObj waiterSyncObj);
    virtual NvError OnPacketCreated(const std::vector<ElemBufObj> &vElemBufObjs) {
        return NvError_Success;
    }
    virtual NvError GetConnectBlock(NvSciStreamBlock *pBlock) = 0;
    virtual void MapDataBuffer(PacketElementType userType, uint32_t uPacketIndex,
                               NvSciBufObj bufObj);
    virtual NvError MapMetaBuffer(uint32_t uPacketIndex, NvSciBufObj bufObj);
    virtual NvError MapPayload(void *pBuffer, uint32_t &uPacketIndex, uint32_t &uElementId);

    virtual NvError HandleSetupComplete() { return NvError_Success; }
    virtual NvError HandleSyncSupport();
    virtual NvError HandleSyncExport();
    virtual NvError HandlePayload() = 0;

    virtual NvError OnBufAttrListRecvd(const ElemBufAttr (&elemBufAttrs)[MAX_NUM_ELEMENTS],
                                       const uint32_t uNumElems);
    virtual NvError SetUnusedElement(uint32_t uElemIndex) { return NvError_Success; }
    virtual NvError CollectWaiterAttrList(uint32_t uElementId,
                                          std::vector<NvSciSyncAttrList> &vUnreconciledAttrList);
    virtual uint32_t GetWaitSyncObjCount() { return 1; }
    virtual NvError OnConnected() { return NvError_Success; };

    NvError SetMetaBufAttrList(NvSciBufAttrList &bufAttrList);
    NvError GetElemIdByUserType(PacketElementType userType, uint32_t &uElementId);

    inline NvError GetIndexFromCookie(NvSciStreamCookie cookie, uint32_t &uIndex) {
        if (cookie <= cookieBase) {
            PLOG_ERR("invalid cookie assignment\n");
            return NvError_BadParameter;
        }
        uIndex = static_cast<uint32_t>(cookie - cookieBase) - 1U;
        return NvError_Success;
    }
    inline ClientPacket *GetPacketByCookie(const NvSciStreamCookie &cookie) {
        uint32_t uId = 0U;
        auto error = GetIndexFromCookie(cookie, uId);
        PLOG_DBG("GetPacketByCookie: packetId: %u\n", uId);
        if (error != NvError_Success) {
            return nullptr;
        }
        return &(m_packets[uId]);
    }
    // Decide the cookie for the new packet
    inline NvSciStreamCookie AssignPacketCookie() {
        NvSciStreamCookie cookie = cookieBase + static_cast<NvSciStreamCookie>(m_uNumPacket);
        return cookie;
    }

    inline bool IsClearedFence(NvSciSyncFence *pFence) {
        uint64_t id;
        uint64_t value;

        auto sciErr = NvSciSyncFenceExtractFence(pFence, &id, &value);
        if (NvSciError_ClearedFence == sciErr) {
            return true;
        }

        return false;
    }

    std::vector<ElementInfo> m_vElemsInfos;
    std::unordered_map<NvSciBufObj, uint32_t> m_bufObjPacketIndexMap;
    NvSciSyncObj m_waiterSyncObjs[MAX_WAIT_SYNCOBJ][MAX_NUM_ELEMENTS];
    NvSciSyncCpuWaitContext m_cpuWaitPreContext = nullptr;
    NvSciSyncCpuWaitContext m_cpuWaitPostContext = nullptr;
    void *m_metaPtrs[MAX_NUM_PACKETS] = {nullptr};

    CAppCfg *m_pAppCfg = nullptr;
    std::shared_ptr<CClientCfg> m_spClientCfg;
    NvSciSyncAttrList m_signalerAttrLists[MAX_NUM_ELEMENTS];
    NvSciSyncAttrList m_waiterAttrLists[MAX_NUM_ELEMENTS];
    /* Sync attributes for CPU waiting */
    CpuWaitCfg m_cpuWaitCfg;
    NvSciSyncAttrList m_cpuWaitAttr = nullptr;

    NvSciSyncObj m_signalSyncObjs[MAX_NUM_ELEMENTS];

    uint32_t m_uNumReconciledElem = 0U;
    uint32_t m_uNumReconciledElemRecvd = 0U;

    uint32_t m_uNumPacket = 0U;
    uint32_t m_uMinPacket = 0U;
    ClientPacket m_packets[MAX_NUM_PACKETS];

    bool m_bIsSyncAttrListSet = false;
    NvSciStreamBlock m_handle = 0U;
    std::vector<std::pair<bool, NvSciStreamBlock>> m_vBlockPairs;

    NvSciSyncAttrList m_signalerAttrList = nullptr;
    NvSciSyncAttrList m_waiterAttrList = nullptr;
    QueueType m_queueType = QueueType::Fifo;
    IModuleCallback *m_pModuleCallback = nullptr;

    // the index of the first data element
    uint32_t m_uDataElemId = MAX_NUM_ELEMENTS;
    CommType m_eCommType = CommType::IntraProcess;

    std::atomic<bool> m_bStop{true};
    std::shared_ptr<CControlChannelManager> m_spControlChannel = {nullptr};
    std::shared_ptr<CTimer> m_spTimer = {nullptr};

private:
    EventStatus HandleConnectEvent();
    EventStatus HandleClientEvent();
    EventStatus HandleEvent();
    NvError HandleElemSupport();
    NvError HandleElemSetting();
    NvError HandlePacketCreate();
    NvError HandleSyncImport();

    NvError AllocSignalSyncObj(std::vector<NvSciSyncAttrList> &vUnreconciledAttrLists,
                               uint32_t uElemIndex, NvSciSyncObj *pSignalSyncObj);
    NvError SetSignalSyncObj(uint32_t uElementId, NvSciSyncObj signalSyncObj);
    EventStatus CheckConnection();
    NvError SetCpuSyncAttrList(NvSciSyncAttrList attrList, NvSciSyncAccessPerm cpuPerm,
                               bool bCpuSync);

    std::unique_ptr<CEventHandler<CClientCommon>> m_upEventHandler;
};

#endif
