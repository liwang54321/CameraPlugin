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

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <inttypes.h>
#include "CFileSourceModule.hpp"
#include "CElementDescription.hpp"

const std::unordered_map<std::string, Option> fileSrcOptionTable = {
    { "type", { "the type of source file", offsetof(FileSourceOption, type), OptionType::INT } },
    { "path", { "the path of source file located", offsetof(FileSourceOption, sPath), OptionType::STRING } },
    { "width", { "the width of the image", offsetof(FileSourceOption, uWidth), OptionType::UINT32 } },
    { "height", { "the height of the image", offsetof(FileSourceOption, uHeight), OptionType::UINT32 } },
    { "instanceId", { "the instanceId of the decoder", offsetof(FileSourceOption, uInstanceId), OptionType::UINT32 } },
    { "ROIFilePath", { "the ROI parameter File Path", offsetof(FileSourceOption, sROIFilePath), OptionType::STRING } },
    { "ROIParams",
      { "the ROI parameters of Image buffer", offsetof(FileSourceOption, sROIParams), OptionType::STRING } }
};

CElementDescription fileSrcDescription{ "FileSrc",
                                        "File source module which read data from local YUV files or H264/H265 files "
                                        "and send the buffer to the downstream.(Test Only)",
                                        &CBaseModule::m_baseModuleOptionTable, &fileSrcOptionTable };

CFileSourceModule::CFileSourceModule(std::shared_ptr<CModuleCfg> spModuleCfg, IEventListener<CBaseModule> *pListener)
    : CBaseModule(spModuleCfg, pListener)
{
    spModuleCfg->m_cpuWaitCfg.bWaitPrefence = true;
    spModuleCfg->m_cpuWaitCfg.bWaitPostfence = true;
}

CFileSourceModule::~CFileSourceModule()
{
    PLOG_DBG("CFileSourceModule::release.\n");
}

NvError CFileSourceModule::Init()
{

    NvError error = NvError_Success;

    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler = std::make_unique<CProfiler>();
        error = m_upProfiler->Init(m_pAppCfg->m_sciSyncModule, m_pAppCfg->GetPerfDataSaveFolder(), GetName(), true,
                                   m_pAppCfg->GetMaxPerfSampleNum());
        PCHK_ERROR_AND_RETURN(error, "CProfiler::Init()");

        m_upProfiler->RecordInitBeginTime();
    }

    if (m_fileSrcOption.type == FileSourceType::UNDEFINED || m_fileSrcOption.sPath == "" ||
        m_fileSrcOption.uWidth == 0u || m_fileSrcOption.uHeight == 0u) {
        PLOG_ERR("Init: All required module options must be set correctly!\n");
        return NvError_BadValue;
    }

    /** Check ROI file first */
    if (!m_fileSrcOption.sROIFilePath.empty()) {
        std::ifstream sRoiFile(m_fileSrcOption.sROIFilePath, std::ios::binary);
        if (!sRoiFile.is_open()) {
            PLOG_ERR("can not open ROI File %s\n", m_fileSrcOption.sROIFilePath.c_str());
            return NvError_FileOperationFailed;
        }
        std::string sLine;
        while (std::getline(sRoiFile, sLine)) {
            std::vector<NvMediaRect> rois;
            GetRoi(sLine, rois);
            m_rois.emplace_back(std::move(rois));
        }
        sRoiFile.close();
    } else if (!m_fileSrcOption.sROIParams.empty()) {
        /** one line parameter */
        std::vector<NvMediaRect> rois;
        GetRoi(m_fileSrcOption.sROIParams, rois);
        m_rois.emplace_back(std::move(rois));
    }

    m_upFrameHandler =
        CreateFrameHandler(m_fileSrcOption.type, m_fileSrcOption.sPath, m_fileSrcOption.uWidth, m_fileSrcOption.uHeight,
                           m_spModuleCfg->m_sensorId, m_fileSrcOption.uInstanceId);
    if (!m_upFrameHandler) {
        PLOG_ERR("Init: CreateFrameHandler CFrameHandler failed.\n");
        return NvError_InsufficientMemory;
    }
    error = m_upFrameHandler->Init();
    PCHK_ERROR_AND_RETURN(error, "CFrameHandler::Init()");

    error = CBaseModule::Init();
    PCHK_ERROR_AND_RETURN(error, "CBaseModule::Init()");

    m_upEventHandler = std::make_unique<CEventHandler<CFileSourceModule>>();
    error = m_upEventHandler->RegisterHandler(&CFileSourceModule::Generator, this);
    PCHK_ERROR_AND_RETURN(error, "RegisterHandler");

    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler->RecordInitEndTime();
    }
    return NvError_Success;
}

void CFileSourceModule::DeInit()
{
    CBaseModule::DeInit();

    if (m_upFrameHandler.get()) {
        m_upFrameHandler->DeInit();
    }

    if (m_upEventHandler.get()) {
        m_upEventHandler->QuitThread();
    }
}

NvError CFileSourceModule::Start()
{
    auto error = m_upFrameHandler->Start();
    PCHK_ERROR_AND_RETURN(error, "CFrameHandler::Start");

    error = m_upEventHandler->StartThread();
    PCHK_ERROR_AND_RETURN(error, " StartThread");

    error = CBaseModule::Start();
    PCHK_ERROR_AND_RETURN(error, "CBaseModule::Start");

    return NvError_Success;
}

NvError CFileSourceModule::Stop()
{
    m_upFrameHandler->Stop();

    CBaseModule::Stop();
    m_upEventHandler->StopThread();

    return NvError_Success;
}

EventStatus CFileSourceModule::Generator()
{
    uint64_t uFrameCaptureTSC{ 0 };
    uint64_t uFrameCaptureStartTSC{ 0 };
    if (m_pAppCfg->IsProfilingEnabled()) {
        uFrameCaptureStartTSC = CProfiler::GetCurrentTSC();
    }

    std::chrono::system_clock::time_point timeStart = std::chrono::system_clock::now();

    EventStatus eventStatus = EventStatus::OK;
    NvSciBufObj bufObj;
    NvSciSyncFence *pPostfence = nullptr;
    eventStatus = m_upFrameHandler->LoadFrameData(bufObj, pPostfence);
    if (eventStatus != EventStatus::OK) {
        return eventStatus;
    }

    std::chrono::system_clock::time_point timeEnd = std::chrono::system_clock::now();
    auto duration = timeEnd - timeStart;
    auto durationMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    if (durationMilliseconds > 33) {
        PLOG_WARN("Generator: The processing time of this frame exceeds 33ms.");
    } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(33 - durationMilliseconds));
    }

    UpdateFrameStatistics();

    if (m_pAppCfg->IsProfilingEnabled()) {
        uFrameCaptureTSC = CProfiler::GetCurrentTSC();
        if (!m_bHasDownstream) {
            auto error = m_upProfiler->RecordExecutionAndPipelineTime(uFrameCaptureStartTSC, uFrameCaptureTSC);
            if (error != NvError_Success) {
                PLOG_ERR("RecordExecutionAndPipelineTime failed: %d\n", error);
                return EventStatus::ERROR;
            }
        } else {
            auto error = m_upProfiler->RecordExecutionTime(uFrameCaptureStartTSC, uFrameCaptureTSC);
            if (error != NvError_Success) {
                PLOG_ERR("RecordExecutionTime failed: %d\n", error);
                return EventStatus::ERROR;
            }
        }

        MetaData *pMetaData = m_spProducer->GetMetaPtr(bufObj);
        if (pMetaData != nullptr) {
            pMetaData->Set(uFrameCaptureTSC, uFrameCaptureStartTSC, true, m_uFrameSequenceNumber + 1);
        }
    }

    if (!m_rois.empty()) {
        MetaData *pMetaData = m_spProducer->GetMetaPtr(bufObj);
        if (pMetaData != nullptr) {
            std::vector<NvMediaRect> rois =
                m_rois[m_uFrameSequenceNumber < m_rois.size() ? m_uFrameSequenceNumber : m_rois.size() - 1];
            pMetaData->uNumROIRegions = rois.size();
            for (size_t i = 0; i < rois.size() && i < static_cast<int>(MetaData::kMaxROIRegions); ++i) {
                pMetaData->ROIRect[i].x0 = rois[i].x0;
                pMetaData->ROIRect[i].y0 = rois[i].y0;
                pMetaData->ROIRect[i].x1 = rois[i].x1;
                pMetaData->ROIRect[i].y1 = rois[i].y1;
            }
        }
    }
    ++m_uFrameSequenceNumber;

    m_spProducer->Post(&bufObj, pPostfence);

    return eventStatus;
}

NvError CFileSourceModule::FillDataBufAttrList(CClientCommon *pClient,
                                               PacketElementType userType,
                                               NvSciBufAttrList *pBufAttrList)
{
    NvError error = m_upFrameHandler->FillNvSciBufAttrList(*pBufAttrList);
    PCHK_ERROR_AND_RETURN(error, "CFrameHandler::FillNvSciBufAttrList");

    error = CBaseModule::FillDataBufAttrList(pClient, userType, pBufAttrList);
    CHK_ERROR_AND_RETURN(error, "CBaseModule::FillDataBufAttrList");

    return NvError_Success;
}

NvError CFileSourceModule::FillSyncSignalerAttrList(CClientCommon *pClient,
                                                    PacketElementType userType,
                                                    NvSciSyncAttrList *pSignalerAttrList)
{
    NvError error = m_upFrameHandler->FillSyncSignalerAttrList(*pSignalerAttrList);
    PCHK_ERROR_AND_RETURN(error, "CFrameHandler::FillSyncSignalerAttrList");

    return NvError_Success;
}

NvError CFileSourceModule::FillSyncWaiterAttrList(CClientCommon *pClient,
                                                  PacketElementType userType,
                                                  NvSciSyncAttrList *pWaiterAttrList)
{
    NvError error = m_upFrameHandler->FillSyncWaiterAttrList(*pWaiterAttrList);
    PCHK_ERROR_AND_RETURN(error, "CFrameHandler::FillSyncWaiterAttrList");

    error = CBaseModule::FillSyncWaiterAttrList(pClient, userType, pWaiterAttrList);
    CHK_ERROR_AND_RETURN(error, "CBaseModule::FillSyncWaiterAttrList");

    return NvError_Success;
}

NvError
CFileSourceModule::RegisterSignalSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj signalSyncObj)
{
    CBaseModule::RegisterSignalSyncObj(pClient, userType, signalSyncObj);

    NvError error = m_upFrameHandler->RegisterSignalSyncObj(signalSyncObj);
    CHK_ERROR_AND_RETURN(error, "CFrameHandler::RegisterSignalSyncObj");

    return NvError_Success;
}

NvError
CFileSourceModule::RegisterWaiterSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj waiterSyncObj)
{
    CBaseModule::RegisterWaiterSyncObj(pClient, userType, waiterSyncObj);

    return NvError_Success;
}

//Before calling PreSync, m_nvmBuffers[uPacketIndex] should already be filled.
NvError CFileSourceModule::InsertPrefence(CClientCommon *pClient,
                                          PacketElementType userType,
                                          uint32_t uPacketIndex,
                                          NvSciSyncFence *pPrefence)
{
    return NvError_Success;
}

NvError CFileSourceModule::ProcessPayload(CClientCommon *pClient, uint32_t uPacketIndex)
{
    return NvError_Success;
}

NvError CFileSourceModule::OnPacketGotten(CClientCommon *pClient, uint32_t uPacketIndex, bool *pHandled)
{
    m_upFrameHandler->ReturnBuffer(*pClient->GetBufObj(uPacketIndex));
    return CBaseModule::OnPacketGotten(pClient, uPacketIndex);
}

NvError CFileSourceModule::RegisterBufObj(CClientCommon *pClient,
                                          PacketElementType userType,
                                          uint32_t uPacketIndex,
                                          NvSciBufObj bufObj)
{
    auto error = CBaseModule::RegisterBufObj(pClient, userType, uPacketIndex, bufObj);
    PCHK_ERROR_AND_RETURN(error, "CBaseModule::RegisterBufObj");

    error = m_upFrameHandler->RegisterNvSciBuf(bufObj);
    PCHK_ERROR_AND_RETURN(error, "RegisterNvSciBuf");

    return NvError_Success;
}

const OptionTable *CFileSourceModule::GetOptionTable() const
{
    return &fileSrcOptionTable;
}

const void *CFileSourceModule::GetOptionBaseAddress() const
{
    return &m_fileSrcOption;
}
