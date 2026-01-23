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

#include "CSiplModule.hpp"
#include "CSiplCamera.hpp"
#include "CElementDescription.hpp"
#include "NvCamFsync.h"
#ifdef NVMEDIA_QNX
#include <sys/neutrino.h>
#endif

static constexpr uint64_t MIN_MULTIPLIER = 1U;
static constexpr uint64_t MAX_MULTIPLIER = 32U;

CElementDescription siplDescription{ "SIPL", "SIPL module for sensor input processing",
                                     &CBaseModule::m_baseModuleOptionTable, nullptr };

// device block notification queue timeout US
constexpr unsigned long kEventQueueTimeoutUs = 1000000U;

// frame queue timeout US
constexpr unsigned long kImageQueueTimeoutUs = 1000000U;

CSiplModule::CSiplModule(std::shared_ptr<CModuleCfg> spModuleCfg, IEventListener<CBaseModule> *pListener)
    : CBaseModule(spModuleCfg, pListener)
    , m_bStarted(false)
{
    m_pAppCfg = spModuleCfg->m_pAppCfg;
    LOG_INFO("CSiplModule GetSensorId() %d\n", GetSensorId());

    if (m_pAppCfg->IsProfilingEnabled()) {
        /*
         * We do need cpu wait context for postfence
         */
        spModuleCfg->m_cpuWaitCfg = { true, true };
    } else {
        // We can not insert a vaild pre-ference into sipl because of the Bug 4364352.
        // Using cpu wait context as a temp solution until bug fixed.
        spModuleCfg->m_cpuWaitCfg.bWaitPrefence = true;
    }
}

CSiplModule::~CSiplModule() {}

NvError CSiplModule::Init()
{
    PLOG_DBG("Enter: CSiplModule::Init\n");

    NvError error = NvError_Success;

    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler = std::make_unique<CProfiler>();
        error = m_upProfiler->Init(m_pAppCfg->m_sciSyncModule, m_pAppCfg->GetPerfDataSaveFolder(), GetName(), true,
                                   m_pAppCfg->GetMaxPerfSampleNum());
        PCHK_ERROR_AND_RETURN(error, "CProfiler::Init()");

        m_upProfiler->RecordInitBeginTime();
    }

    error = CBaseModule::Init();
    PCHK_ERROR_AND_RETURN(error, "CBaseModule::Init()");

    error = InitSiplCamera(m_spCamera);
    PCHK_ERROR_AND_RETURN(error, "InitSiplCamera");

    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler->RecordInitEndTime();
    }

    return NvError_Success;
}

void CSiplModule::DeInit()
{
    PLOG_DBG("Enter: CSiplModule::DeInit\n");

    FreeResources();

    CBaseModule::DeInit();

    PLOG_DBG("Exit: CSiplModule::DeInit\n");
}

void CSiplModule::OnError(uint32_t uCameraId, uint32_t uErrorId)
{
    CBaseModule::OnError(uCameraId, uErrorId);
    PLOG_ERR("SIPLModule sensorId %d uErrorId %d\n", uCameraId, uErrorId);
}

NvError CSiplModule::MapElemTypeToOutputType(PacketElementType userType,
                                             INvSIPLClient::ConsumerDesc::OutputType &outputType)
{
    static std::map<PacketElementType, INvSIPLClient::ConsumerDesc::OutputType> outTypeMap = {
        { PacketElementType::NV12_BL, INvSIPLClient::ConsumerDesc::OutputType::ISP0 },
        { PacketElementType::NV12_PL, INvSIPLClient::ConsumerDesc::OutputType::ISP1 },
        { PacketElementType::ICP_RAW, INvSIPLClient::ConsumerDesc::OutputType::ICP },
    };

    auto iterator = outTypeMap.find(userType);
    if (iterator != outTypeMap.end()) {
        outputType = iterator->second;
        return NvError_Success;
    }
    return NvError_BadParameter;
}

NvError CSiplModule::OverrideIspAttributes(PacketElementType userType, NvSciBufAttrList &bufAttrList)
{
    NvSciError sciErr;
    NvSciBufAttrValImageLayoutType layout = NvSciBufImage_PitchLinearType;
    if (PacketElementType::NV12_BL == userType) {
        layout = NvSciBufImage_BlockLinearType;
    } else if (PacketElementType::NV12_PL == userType) {
        layout = NvSciBufImage_PitchLinearType;
    } else {
        LOG_ERR("SetDataBufAttrList: Unsuported ISP output type. \n");
        return NvError_BadParameter;
    }

    bool bNeedCpuAccess = true;
    bool bEnableCpuCache = true;

    NvSciBufAttrKeyValuePair attrs[] = {
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &bNeedCpuAccess, sizeof(bNeedCpuAccess) },
        { NvSciBufGeneralAttrKey_EnableCpuCache, &bEnableCpuCache, sizeof(bEnableCpuCache) }
    };
    sciErr = NvSciBufAttrListSetAttrs(bufAttrList, attrs, ARRAY_SIZE(attrs));
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

    NvSciBufSurfSampleType surfSampleType = NvSciSurfSampleType_420;
    NvSciBufSurfBPC surfBPC = NvSciSurfBPC_8;
    NvSciBufSurfType surfType = NvSciSurfType_YUV;
    NvSciBufSurfMemLayout surfMemLayout = NvSciSurfMemLayout_SemiPlanar;
    NvSciBufSurfComponentOrder surfCompOrder = NvSciSurfComponentOrder_YUV;
    NvSciBufAttrValColorStd surfColorStds[] = { NvSciColorStd_REC709_ER };
    NvSciBufAttrKeyValuePair surfaceAttrs[] = {
        { NvSciBufImageAttrKey_SurfType, &surfType, sizeof(surfType) },
        { NvSciBufImageAttrKey_SurfBPC, &surfBPC, sizeof(surfBPC) },
        { NvSciBufImageAttrKey_SurfMemLayout, &surfMemLayout, sizeof(surfMemLayout) },
        { NvSciBufImageAttrKey_SurfSampleType, &surfSampleType, sizeof(surfSampleType) },
        { NvSciBufImageAttrKey_SurfComponentOrder, &surfCompOrder, sizeof(surfCompOrder) },
        { NvSciBufImageAttrKey_SurfColorStd, &surfColorStds, sizeof(surfColorStds) },
        { NvSciBufImageAttrKey_Layout, &layout, sizeof(layout) },
    };

    sciErr = NvSciBufAttrListSetAttrs(bufAttrList, surfaceAttrs, ARRAY_SIZE(surfaceAttrs));
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

    return NvError_Success;
}

NvError
CSiplModule::FillDataBufAttrList(CClientCommon *pClient, PacketElementType userType, NvSciBufAttrList *pBufAttrList)
{
    INvSIPLClient::ConsumerDesc::OutputType outputType;
    if (MapElemTypeToOutputType(userType, outputType)) {
        return NvError_BadParameter;
    }

    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_ReadWrite;

    NvSciBufAttrKeyValuePair attrs[] = {
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm) },
    };
    NvSciError sciErr = NvSciBufAttrListSetAttrs(*pBufAttrList, attrs, ARRAY_SIZE(attrs));
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

    LOG_DBG("[[FillDataBufAttrList]] FillDataBufAttrList GetSensorId() %d outputType %d, userType = %d\n",
            GetSensorId(), (int)outputType, (int)userType);
    if (INvSIPLClient::ConsumerDesc::OutputType::ICP != outputType) {
        if (OverrideIspAttributes(userType, *pBufAttrList)) {
            PLOG_ERR("Failed to override isp attributes");
            return NvError_BadParameter;
        }
    }

    NvError siplStatus = m_spCamera->GetImageAttributes(GetSensorId(), outputType, *pBufAttrList);
    PCHK_ERROR_AND_RETURN(siplStatus, "GetImageAttributes");

    siplStatus = CBaseModule::FillDataBufAttrList(pClient, userType, pBufAttrList);
    CHK_ERROR_AND_RETURN(siplStatus, "CBaseModule::FillDataBufAttrList");

    return siplStatus;
}

NvError CSiplModule::FillSyncSignalerAttrList(CClientCommon *pClient,
                                              PacketElementType userType,
                                              NvSciSyncAttrList *pSignalelAttrList)
{
    INvSIPLClient::ConsumerDesc::OutputType outputType;
    if (MapElemTypeToOutputType(userType, outputType)) {
        return NvError_BadParameter;
    }

    NvError error = m_spCamera->FillSyncSignalerAttrList(GetSensorId(), outputType, *pSignalelAttrList, SIPL_SIGNALER);
    PCHK_ERROR_AND_RETURN(error, "FillSyncSignalerAttrList");

    return NvError_Success;
}

NvError CSiplModule::FillSyncWaiterAttrList(CClientCommon *pClient,
                                            PacketElementType userType,
                                            NvSciSyncAttrList *pWaiterAttrList)
{
    INvSIPLClient::ConsumerDesc::OutputType outputType;
    if (MapElemTypeToOutputType(userType, outputType)) {
        return NvError_BadParameter;
    }

    NvError error = m_spCamera->FillSyncWaiterAttrList(GetSensorId(), outputType, *pWaiterAttrList, SIPL_WAITER);
    PCHK_ERROR_AND_RETURN(error, "FillSyncWaiterAttrList");

    error = CBaseModule::FillSyncWaiterAttrList(pClient, userType, pWaiterAttrList);
    CHK_ERROR_AND_RETURN(error, "CBaseModule::FillSyncWaiterAttrList");

    return NvError_Success;
}

NvError CSiplModule::RegisterBufObj(CClientCommon *pClient,
                                    PacketElementType userType,
                                    uint32_t uPacketIndex,
                                    NvSciBufObj bufObj)
{
    INvSIPLClient::ConsumerDesc::OutputType outputType;
    if (MapElemTypeToOutputType(userType, outputType)) {
        return NvError_BadParameter;
    }

    NvError siplStatus = NvError_Success;
    uint32_t uElementIndex = 0U;
    auto error = pClient->GetElemIndexByUserType(userType, uElementIndex);
    PCHK_ERROR_AND_RETURN(error, "GetElemIndexByUserType");

    m_siplBufObjMap[outputType].push_back(bufObj);
    m_elementInfoMaps[bufObj].uPacketIndex = uPacketIndex;
    m_elementInfoMaps[bufObj].uElementIndex = uElementIndex;

    return siplStatus;
}

NvError
CSiplModule::RegisterSignalSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj signalSyncObj)
{
    INvSIPLClient::ConsumerDesc::OutputType outputType;
    if (MapElemTypeToOutputType(userType, outputType)) {
        return NvError_BadParameter;
    }

    // For ISP sync, only one signalSyncObj.
    m_siplSignalSyncObjMap[outputType] = signalSyncObj;

    return NvError_Success;
}

NvError
CSiplModule::RegisterWaiterSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj waiterSyncObj)
{
    if (EventStatus::RECONCILED == m_EventStatus) {
        PLOG_INFO("Camera only support registering sync obj in initialization phase. \n");
        return NvError_Success;
    }

    INvSIPLClient::ConsumerDesc::OutputType outputType;
    if (MapElemTypeToOutputType(userType, outputType)) {
        return NvError_BadParameter;
    }

    m_siplWaiterSyncObjMap[outputType] = waiterSyncObj;

    return NvError_Success;
}

NvError CSiplModule::InsertPrefence(CClientCommon *pClient,
                                    PacketElementType userType,
                                    uint32_t uPacketIndex,
                                    NvSciSyncFence *pPrefence)
{
    INvSIPLClient::ConsumerDesc::OutputType outputType;
    if (MapElemTypeToOutputType(userType, outputType)) {
        return NvError_BadParameter;
    }
    if (m_packetBufMap.find(uPacketIndex) != m_packetBufMap.end()) {
        for (auto it = m_packetBufMap[uPacketIndex].begin(); it != m_packetBufMap[uPacketIndex].end(); it++) {
            if (it->first == outputType) {
                NvError error = toNvError(it->second->AddNvSciSyncPrefence(*pPrefence));
                PCHK_ERROR_AND_RETURN(error, "AddNvSciSyncPrefence");
            }
        }
    }
    return NvError_Success;
}

NvError CSiplModule::Start()
{
    PLOG_DBG("Enter: CSiplModule::Start()\n");

    /* SIPLCamera may exit due to SC7 in postStop, and thus it needs to be reinitialized in Start*/
    if (m_spCamera == nullptr) {
        auto error = InitSiplCamera(m_spCamera);
        PCHK_ERROR_AND_RETURN(error, "InitSiplCamera");
    }

    /* Start up frame queue thread and pipeline notification thread for each module */
    m_upFrameThread.reset(new std::thread(&CSiplModule::FrameQueueThread, this, &m_pipelineQueues));

    m_upNotificationThread.reset(
        new std::thread(&CSiplModule::PipelineQueueThread, this, m_pipelineQueues.notificationQueue));

    NvError error = m_spCamera->Init(GetSensorId());
    PCHK_ERROR_AND_RETURN(error, "CSiplCamera::Init");

    for (auto &pair : m_siplBufObjMap) {
        auto outputType = pair.first;
        const auto &vBufObjs = pair.second;
        if (vBufObjs.size() < MAX_NUM_PACKETS) {
            return NvError_BadValue;
        }
        auto error = m_spCamera->RegisterImages(GetSensorId(), outputType, vBufObjs);
        PCHK_ERROR_AND_RETURN(error, "RegisterImages");
    }

    for (auto &pair : m_siplSignalSyncObjMap) {
        auto outputType = pair.first;
        auto signalSyncObj = pair.second;
        auto error = m_spCamera->RegisterSignalSyncObj(GetSensorId(), outputType, NVSIPL_EOFSYNCOBJ, signalSyncObj);
        PCHK_ERROR_AND_RETURN(error, "RegisterSignalSyncObj");
    }

    for (auto &pair : m_siplWaiterSyncObjMap) {
        auto outputType = pair.first;
        auto waiterSyncObj = pair.second;
        auto error = m_spCamera->RegisterWaiterSyncObj(GetSensorId(), outputType, NVSIPL_PRESYNCOBJ, waiterSyncObj);
        PCHK_ERROR_AND_RETURN(error, "RegisterWaiterSyncObj");
    }

    error = m_spCamera->Start(GetSensorId());
    PCHK_ERROR_AND_RETURN(error, "CSiplCamera::Start");

    error = CBaseModule::Start();
    PCHK_ERROR_AND_RETURN(error, "CBaseModule::Start");

    m_bStarted = true;
    PLOG_DBG("Exit: CSiplModule::Start()\n");
    return NvError_Success;
}

NvError CSiplModule::Stop()
{
    CBaseModule::Stop();

    PLOG_DBG("Enter: CSiplModule::Stop()\n");
    NvError error = FreeResources();
    PLOG_DBG("Exit: CSiplModule::Stop()\n");
    return error;
}

NvError CSiplModule::PostStop()
{
    /*ReInit the camera before suspend, then start can get better KPI*/
#ifndef NVMEDIA_QNX
    if (m_pAppCfg->IsStatusManagerEnabled()) {
        auto error = InitSiplCamera(m_spCamera);
        PCHK_ERROR_AND_RETURN(error, "InitSiplCamera");
    }
#endif

    auto error = CBaseModule::PostStop();
    PCHK_ERROR_AND_RETURN(error, "CBaseModule::PostStop");

    return NvError_Success;
}

void CSiplModule::OnCommand(uint32_t uCmdId, void *pParam)
{
    PLOG_DBG("Enter: CSiplModule::OnCommand()\n");
    CMDType cmd = static_cast<CMDType>(uCmdId);
    switch (cmd) {
        case CMDType::ENTER_LOW_POWER_MODE:
        case CMDType::ENTER_FULL_POWER_MODE: {
            bool bLowPowerMode = cmd == CMDType::ENTER_LOW_POWER_MODE ? true : false;
            auto error = ConfigureCameraFrameRate(bLowPowerMode);
            if (error != NvError_Success) {
                PLOG_ERR("ConfigureCameraFrameRate failed, error %d\n", error);
            }
            break;
        }
        default:
            PLOG_WARN("invalid cmd, do nothing! cmd %d\n", uCmdId);
            break;
    }

    PLOG_DBG("Exit: CSiplModule::OnCommand()\n");
}

NvError CSiplModule::ConfigureCameraFrameRate(bool bLowPowerMode)
{
    PLOG_DBG("Enter: CSiplModule::ConfigureCameraFrameRate() %s \n",
             bLowPowerMode ? "SetLowPowerMode" : "UnSetLowPowerMode");

    uint64_t uProgramTscTicks{ 0 };
    CAM_FSYNC_STATUS status;

    status = cam_fsync_stop_group(0);
    if (status != CAM_FSYNC_OK) {
        PLOG_ERR("ConfigureCameraFrameRate: cam_fsync_stop_group API failed (status=%d)\n", status);
        return NvError_InvalidState;
    }

    /* Reduce the frame-rate to 15 for Surround view sensors in Sentry mode */
    uint32_t frameRate = bLowPowerMode ? 15 : 30;
    status = cam_fsync_reconfigure_generator(0, 0, frameRate, 50, 0);
    if (status != CAM_FSYNC_OK) {
        PLOG_ERR("ConfigureCameraFrameRate: Error while reconfiguring fsync generator to %d FPS (status=%d)\n",
                 frameRate, status);
        return NvError_InvalidState;
    }

    uProgramTscTicks = GetCurrentTSCTicks();

    /* Add 10ms to the current time to ensure the program time is in the future */
    uProgramTscTicks += (10000 * 1000) / 32;
    /* Program and start fsync signal for group 0 with start_time */
    if (!ProgramFsync(0, uProgramTscTicks)) {
        PLOG_ERR("ConfigureCameraFrameRate: Error while programming Fsync\n");
        return NvError_InvalidState;
    }

    PLOG_DBG("Exit: CSiplModule::ConfigureCameraFrameRate() %s \n",
             bLowPowerMode ? "SetLowPowerMode" : "UnSetLowPowerMode");

    return NvError_Success;
}

NvError CSiplModule::ProcessPayload(CClientCommon *pClient, uint32_t uPacketIndex)
{
    return NvError_Success;
}

NvError CSiplModule::OnPacketGotten(CClientCommon *pClient, uint32_t uPacketIndex, bool *pHandled)
{
    if (m_packetBufMap.find(uPacketIndex) != m_packetBufMap.end()) {
        for (auto it = m_packetBufMap[uPacketIndex].begin(); it != m_packetBufMap[uPacketIndex].end(); it++) {
            it->second->Release();
        }
    }
    return NvError_Success;
}

NvError CSiplModule::GetMinPacketCount(CClientCommon *pClient, uint32_t *pPacketCount)
{
    CHK_PTR_AND_RETURN_BADARG(pPacketCount, "GetMinPacketCount");
    if (m_pipelineQueues.isp0CompletionQueue != nullptr || m_pipelineQueues.isp1CompletionQueue != nullptr ||
        m_pipelineQueues.isp1CompletionQueue != nullptr) {
        *pPacketCount = 0U;
    } else {
        *pPacketCount = 1U;
    }
    return NvError_Success;
}

NvError CSiplModule::OnFrameAvailable(
    std::vector<std::pair<INvSIPLClient::ConsumerDesc::OutputType, INvSIPLClient::INvSIPLBuffer *>> siplBuffers)
{
    PLOG_DBG("Enter: CSiplModule::OnFrameAvailable\n");
    std::map<INvSIPLClient::ConsumerDesc::OutputType, INvSIPLClient::INvSIPLBuffer *> bufMap;
    MultiPostInfo postInfo;
    uint32_t uPacketIndex = 0;

    if (siplBuffers.empty()) {
        PLOG_ERR("empty siplBuffers\n");
        return NvError_BadParameter;
    }

    UpdateFrameStatistics();

    uint64_t uFrameCaptureTSC = 0;
    uint64_t uFrameCaptureStartTSC = 0;
    bool bFrameSeqNumValid = false;
    uint64_t uFrameSequenceNumber = 0;
    for (auto siplBuf : siplBuffers) {
        NvSciSyncFence postFence = NvSciSyncFenceInitializer;
        auto error = toNvError(siplBuf.second->GetEOFNvSciSyncFence(&postFence));
        PCHK_ERROR_AND_RETURN(error, "GetEOFNvSciSyncFence");

        /* get bufObj from siplBuf */
        INvSIPLClient::INvSIPLNvMBuffer *nvmBuf = reinterpret_cast<INvSIPLClient::INvSIPLNvMBuffer *>(siplBuf.second);
        NvSciBufObj sciBufObj = nvmBuf->GetNvSciBufImage();

        const INvSIPLClient::ImageMetaData &md = nvmBuf->GetImageData();
        if (0U == m_uMultiplier) {
            uint64_t uCurrentTSC = CProfiler::GetCurrentTSC();
            m_uMultiplier = uCurrentTSC / md.frameCaptureTSC;
            if (m_uMultiplier <= 0) {
                m_uMultiplier = MIN_MULTIPLIER;
            } else if (m_uMultiplier > MAX_MULTIPLIER) {
                m_uMultiplier = MAX_MULTIPLIER;
            }
        }
        uFrameCaptureTSC = md.frameCaptureTSC * m_uMultiplier;
        uFrameCaptureStartTSC = md.frameCaptureStartTSC * m_uMultiplier;
        bFrameSeqNumValid = md.frameSeqNumInfo.frameSeqNumValid;
        uFrameSequenceNumber = md.frameSeqNumInfo.frameSequenceNumber;

        /* extract element id & packet index */
        postInfo.uPacketIndex = m_elementInfoMaps[sciBufObj].uPacketIndex;
        postInfo.vPostFences.push_back(postFence);
        postInfo.vElementIndexs.push_back(m_elementInfoMaps[sciBufObj].uElementIndex);

        bufMap[siplBuf.first] = siplBuf.second;
        uPacketIndex = postInfo.uPacketIndex;

        if (m_pAppCfg->IsProfilingEnabled()) {
            if (siplBuf.first != INvSIPLClient::ConsumerDesc::OutputType::ICP) {
                NvSciSyncFence fence = NvSciSyncFenceInitializer;
                NvError error = toNvError(nvmBuf->GetEOFNvSciSyncFence(&fence));
                PCHK_ERROR_AND_RETURN(error, "INvSIPLClient::INvSIPLNvMBuffer::GetEOFNvSciSyncFence");

                error = m_bHasDownstream ? m_upProfiler->RecordExecutionTime(uFrameCaptureStartTSC, &fence)
                                         : m_upProfiler->RecordExecutionAndPipelineTime(uFrameCaptureStartTSC,
                                                                                        uFrameCaptureStartTSC, &fence);
                PCHK_ERROR_AND_RETURN(error,
                                      m_bHasDownstream ? "RecordExecutionTime" : "RecordExecutionAndPipelineTime");
                NvSciSyncFenceClear(&fence);
            } else {
                /*
                 * VI only
                 */
                error = m_bHasDownstream
                            ? m_upProfiler->RecordExecutionTime(uFrameCaptureStartTSC, uFrameCaptureTSC)
                            : m_upProfiler->RecordExecutionAndPipelineTime(uFrameCaptureStartTSC, uFrameCaptureTSC);
                PCHK_ERROR_AND_RETURN(error,
                                      m_bHasDownstream ? "RecordExecutionTime" : "RecordExecutionAndPipelineTime");
            }
        }
    }
    m_packetBufMap[uPacketIndex] = std::move(bufMap);

    /* Set frameCaptureTSC to producer.*/
    MetaData *pMetaData = reinterpret_cast<MetaData *>(m_spProducer->GetMetaPtr(uPacketIndex));
    if (pMetaData != nullptr) {
        pMetaData->Set(uFrameCaptureTSC, uFrameCaptureStartTSC, bFrameSeqNumValid, uFrameSequenceNumber);
    }

    /* post multi element */
    m_spProducer->MultiPost(&postInfo);
    PLOG_DBG("Exit: CSiplModule::OnFrameAvailable\n");

    return NvError_Success;
}

void CSiplModule::FrameQueueThread(NvSIPLPipelineQueues *pipelineQueues)
{
    NvError error = NvError_Success;

    LOG_INFO("Enter: FrameQueueThread\n");

    std::string threadName("FrameQueue_" + std::to_string(GetSensorId()));
    pthread_setname_np(pthread_self(), threadName.c_str());

    std::vector<std::pair<INvSIPLClient::ConsumerDesc::OutputType, INvSIPLFrameCompletionQueue *>> frameQueues;
    if (pipelineQueues->captureCompletionQueue != nullptr) {
        frameQueues.emplace_back(INvSIPLClient::ConsumerDesc::OutputType::ICP, pipelineQueues->captureCompletionQueue);
    }
    if (pipelineQueues->isp0CompletionQueue != nullptr) {
        frameQueues.emplace_back(INvSIPLClient::ConsumerDesc::OutputType::ISP0, pipelineQueues->isp0CompletionQueue);
    }
    if (pipelineQueues->isp1CompletionQueue != nullptr) {
        frameQueues.emplace_back(INvSIPLClient::ConsumerDesc::OutputType::ISP1, pipelineQueues->isp1CompletionQueue);
    }
    if (pipelineQueues->isp2CompletionQueue != nullptr) {
        frameQueues.emplace_back(INvSIPLClient::ConsumerDesc::OutputType::ISP2, pipelineQueues->isp2CompletionQueue);
    }

    if (frameQueues.empty()) {
        LOG_ERR("empty frame queue\n");
        return;
    }

    m_bQuit.store(false);

    while (!m_bQuit.load()) {
        std::vector<std::pair<INvSIPLClient::ConsumerDesc::OutputType, INvSIPLClient::INvSIPLBuffer *>> vpBufpairs;
        INvSIPLClient::INvSIPLBuffer *pbuf = nullptr;
        for (auto &queue : frameQueues) {
            if (queue.second == nullptr) {
                LOG_ERR("queue nullptr\n");
                return;
            }
            LOG_DBG("FrameQueueThread queue %p\n", queue.second);
            error = toNvError(queue.second->Get(pbuf, kImageQueueTimeoutUs));
            if (error == NvError_Success) {
                vpBufpairs.push_back({ queue.first, pbuf });
            } else {
                break;
            }
        }

        if (NvError_Success == error) {
            error = OnFrameAvailable(std::move(vpBufpairs));
            if (error != NvError_Success) {
                LOG_ERR("OnFrameAvailable failed. (error:%u)\n", error);
                // sensorId as moduleId
                OnError(GetSensorId(), (uint32_t)error);
                m_bQuit.store(true);
            }
        }

        if (NvError_Success == error) {
            LOG_DBG("CPipelineFrameQueueHandler Queue goes to next frame\n");
        } else if (NvError_Timeout == error) {
            LOG_DBG("CPipelineFrameQueueHandler Queue timeout\n");
        } else if (NvError_EndOfFile == error) {
            LOG_DBG("CPipelineFrameQueueHandler Queue shutdown\n");
            m_bQuit.store(true);
            return;
        } else {
            LOG_ERR("Unexpected queue return error: %u\n", error);
            m_bQuit.store(true);
            return;
        }
    }
}

void CSiplModule::OnPipelineEvent(NvSIPLPipelineNotifier::NotificationData &oNotificationData)
{
    switch (oNotificationData.eNotifType) {
        case NvSIPLPipelineNotifier::NOTIF_INFO_ICP_PROCESSING_DONE:
            LOG_INFO("Pipeline: %u, NOTIF_INFO_ICP_PROCESSING_DONE\n", oNotificationData.uIndex);
            break;
        case NvSIPLPipelineNotifier::NOTIF_INFO_ISP_PROCESSING_DONE:
            LOG_INFO("Pipeline: %u, NOTIF_INFO_ISP_PROCESSING_DONE\n", oNotificationData.uIndex);
            break;
        case NvSIPLPipelineNotifier::NOTIF_INFO_ACP_PROCESSING_DONE:
            LOG_INFO("Pipeline: %u, NOTIF_INFO_ACP_PROCESSING_DONE\n", oNotificationData.uIndex);
            break;
        case NvSIPLPipelineNotifier::NOTIF_INFO_ICP_AUTH_SUCCESS:
            LOG_DBG("Pipeline: %u, ICP_AUTH_SUCCESS frame=%lu\n", oNotificationData.uIndex,
                    oNotificationData.frameSeqNumber);
            break;
        case NvSIPLPipelineNotifier::NOTIF_INFO_CDI_PROCESSING_DONE:
            LOG_INFO("Pipeline: %u, NOTIF_INFO_CDI_PROCESSING_DONE\n", oNotificationData.uIndex);
            break;
        case NvSIPLPipelineNotifier::NOTIF_WARN_ICP_FRAME_DROP:
            LOG_WARN("Pipeline: %u, NOTIF_WARN_ICP_FRAME_DROP\n", oNotificationData.uIndex);
            // m_uNumFrameDrops++;
            break;
        case NvSIPLPipelineNotifier::NOTIF_WARN_ICP_FRAME_DISCONTINUITY:
            LOG_WARN("Pipeline: %u, NOTIF_WARN_ICP_FRAME_DISCONTINUITY\n", oNotificationData.uIndex);
            break;
        case NvSIPLPipelineNotifier::NOTIF_WARN_ICP_CAPTURE_TIMEOUT:
            LOG_WARN("Pipeline: %u, NOTIF_WARN_ICP_CAPTURE_TIMEOUT\n", oNotificationData.uIndex);
            break;
        case NvSIPLPipelineNotifier::NOTIF_ERROR_ICP_BAD_INPUT_STREAM:
            LOG_ERR("Pipeline: %u, NOTIF_ERROR_ICP_BAD_INPUT_STREAM\n", oNotificationData.uIndex);
            break;
        case NvSIPLPipelineNotifier::NOTIF_ERROR_ICP_CAPTURE_FAILURE:
            LOG_ERR("Pipeline: %u, NOTIF_ERROR_ICP_CAPTURE_FAILURE\n", oNotificationData.uIndex);
            break;
        case NvSIPLPipelineNotifier::NOTIF_ERROR_ICP_EMB_DATA_PARSE_FAILURE:
            LOG_ERR("Pipeline: %u, NOTIF_ERROR_ICP_EMB_DATA_PARSE_FAILURE\n", oNotificationData.uIndex);
            break;
        case NvSIPLPipelineNotifier::NOTIF_ERROR_ISP_PROCESSING_FAILURE:
            LOG_ERR("Pipeline: %u, NOTIF_ERROR_ISP_PROCESSING_FAILURE\n", oNotificationData.uIndex);
            break;
        case NvSIPLPipelineNotifier::NOTIF_ERROR_ACP_PROCESSING_FAILURE:
            LOG_ERR("Pipeline: %u, NOTIF_ERROR_ACP_PROCESSING_FAILURE\n", oNotificationData.uIndex);
            break;
        case NvSIPLPipelineNotifier::NOTIF_ERROR_CDI_SET_SENSOR_CTRL_FAILURE:
            LOG_ERR("Pipeline: %u, NOTIF_ERROR_CDI_SET_SENSOR_CTRL_FAILURE\n", oNotificationData.uIndex);
            break;
        case NvSIPLPipelineNotifier::NOTIF_ERROR_INTERNAL_FAILURE:
            LOG_ERR("Pipeline: %u, NOTIF_ERROR_INTERNAL_FAILURE\n", oNotificationData.uIndex);
            break;
        case NvSIPLPipelineNotifier::NOTIF_ERROR_ICP_AUTH_FAILURE:
            LOG_ERR("Pipeline: %u, ICP_AUTH_FAILURE frame=%lu\n", oNotificationData.uIndex,
                    oNotificationData.frameSeqNumber);
            break;
        default:
            LOG_WARN("Pipeline: %u, Unknown/Invalid notification\n", oNotificationData.uIndex);
            break;
    }
    return;
}

void CSiplModule::PipelineQueueThread(INvSIPLNotificationQueue *notificationQueue)
{
    NvError error = NvError_Success;
    NvSIPLPipelineNotifier::NotificationData notificationData;

    if (notificationQueue == nullptr) {
        LOG_ERR("empty nofitication queue\n");
        return;
    }

    pthread_setname_np(pthread_self(), "PipelineNotifier");

    m_bQuit.store(false);

    while (!m_bQuit.load()) {
        error = toNvError(notificationQueue->Get(notificationData, kEventQueueTimeoutUs));
        if (error == NvError_Success) {
            OnPipelineEvent(notificationData);
        } else if (error == NvError_Timeout) {
            LOG_DBG("Queue timeout\n");
        } else if (error == NvError_EndOfFile) {
            LOG_DBG("Queue shutdown\n");
            m_bQuit.store(true);
        } else {
            LOG_ERR("Unexpected queue return error\n");
            m_bQuit.store(true);
        }
    }
}

NvError CSiplModule::InitSiplCamera(std::shared_ptr<CSiplCamera> &spCamera)
{

    /* CSiplCamera will call Init camera in its construct function */
    spCamera = CSiplCamera::GetInstance(m_pAppCfg);
    if (spCamera == nullptr) {
        LOG_ERR("CSiplCamera GetInstance Failed\n");
        return NvError_ResourceError;
    }

    spCamera->RegisterCallback(GetSensorId(), this);

    /* get pipeline queue for this sensor id */
    spCamera->GetPipelineQueues(GetSensorId(), m_pipelineQueues);
    if (m_pipelineQueues.notificationQueue == nullptr) {
        LOG_ERR("GetPipeineQueues empty\n");
        return NvError_BadParameter;
    }

    return NvError_Success;
}

NvError CSiplModule::FreeResources()
{
    NvError error = NvError_Success;
    if (m_bStarted) {
        m_bStarted = false;
        error = m_spCamera->Stop(GetSensorId());
        if (NvError_Success != error) {
            PLOG_ERR("CSiplModule::Stop failed, error: %u\n", error);
        }
    }

    m_bQuit.store(true);
    if (m_upFrameThread != nullptr && m_upFrameThread->joinable()) {
        m_upFrameThread->join();
        m_upFrameThread.reset(nullptr);
    }

    if (m_upNotificationThread != nullptr && m_upNotificationThread->joinable()) {
        m_upNotificationThread->join();
        m_upNotificationThread.reset(nullptr);
    }

    m_spCamera.reset();

    return error;
}

void CSiplModule::OnEvent(CClientCommon *pClient, EventStatus event)
{
    CBaseModule::OnEvent(pClient, event);
    PLOG_INFO("SIPL MODULE Received event: %s %s\n", pClient->GetName().c_str(), EventStatusToString(event));
    m_EventStatus = event;
}

uint64_t CSiplModule::GetCurrentTSCTicks()
{
    uint64_t tscTicks{ 0U };
#ifdef NVMEDIA_QNX
    tscTicks = ClockCycles();
#else
    __asm__ __volatile__("ISB;                                                     \
                           mrs %[result], cntvct_el0;                               \
                           ISB"
                         : [result] "=r"(tscTicks)
                         :
                         : "memory");
#endif
    return tscTicks;
}

bool CSiplModule::ProgramFsync(uint32_t const fsyncGroupId, uint64_t const startTimeTSCTicks)
{
    LOG_INFO("Program and start fsync signal for group: %u with start_time %llu\n", fsyncGroupId, startTimeTSCTicks);
    CAM_FSYNC_STATUS fsync_status{ cam_fsync_program_abs_start_value(fsyncGroupId, startTimeTSCTicks) };
    if (fsync_status != CAM_FSYNC_OK) {
        LOG_ERR("Failed to program and start fsync signal for group: %u with status: %d\n", fsyncGroupId, fsync_status);
        return false;
    }
    return true;
}

constexpr uint64_t CSiplModule::UsToTicks(uint64_t const microseconds)
{
    uint32_t NS_PER_TICK{ 32U };
    return (microseconds * 1000) / NS_PER_TICK;
}
