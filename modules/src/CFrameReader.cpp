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

#include <malloc.h>
#include <inttypes.h>
#include "CFrameReader.hpp"
#include "CUtils.hpp"

CFrameReader::CFrameReader(
    FileSourceType type, const std::string &sFilePath, uint32_t uWidth, uint32_t uHeight, int sensorId)
    : CFrameHandler(type, sFilePath, uWidth, uHeight, sensorId)
{
}

CFrameReader::~CFrameReader() {}

NvError CFrameReader::Init()
{
    if (GetSourceFileType() == FileSourceType::YUV420P_SEQUENCE) {
        std::string sYuvSequenceFilePath =
            GetCameraDirName() + "/" + YUV_SEQUENCE_FILE_PREFIX + IntToStringWithLeadingZero(GetSensorId()) + ".yuv";
        m_yuvSequenceFile.open(sYuvSequenceFilePath, std::ios::in | std::ios::binary);
        if (!m_yuvSequenceFile.is_open()) {
            PLOG_ERR("Init: Open yuv sequence file: %s failed! \n", sYuvSequenceFilePath.c_str());
            return NvError_FileOperationFailed;
        }
    }

    m_upReadHandler = std::make_unique<CEventHandler<CFrameReader>>();
    auto error = m_upReadHandler->RegisterHandler(&CFrameReader::ProcessFrames, this);
    PCHK_ERROR_AND_RETURN(error, "RegisterHandler");

    error = m_upReadHandler->StartThread();
    PCHK_ERROR_AND_RETURN(error, " StartThread");

    return NvError_Success;
}

void CFrameReader::DeInit()
{
    if (m_upReadHandler) {
        m_upReadHandler->QuitThread();
    }

    if (m_yuvSequenceFile.is_open()) {
        m_yuvSequenceFile.close();
    }
}

NvError CFrameReader::Start()
{
    m_bStopReading.store(false);
    auto error = m_upReadHandler->StartThread();
    PCHK_ERROR_AND_RETURN(error, "StartThread");

    return NvError_Success;
}

void CFrameReader::Stop()
{
    m_bStopReading.store(true);
    m_freeBufDataCond.notify_all();
    m_validBufDataCond.notify_all();
    if (m_upReadHandler) {
        m_upReadHandler->StopThread();
    }
}

EventStatus CFrameReader::ProcessFrames()
{
    if (GetSourceFileType() == FileSourceType::YUV420P_SINGLE_FRAME) {
        return ReadYuvFrame();
    } else if (GetSourceFileType() == FileSourceType::YUV420P_SEQUENCE) {
        return ReadYuvSequence();
    } else {
        PLOG_ERR("ProcessFrames: Unsupported type of source file.\n");
        return EventStatus::ERROR;
    }
    return EventStatus::OK;
}

EventStatus CFrameReader::ReadYuvFrame()
{
    while (m_uMissingFileCount < 3) {
        std::string sYUVFilePath = GetCameraDirName() + "/" + YUV_FILE_PREFIX +
                                   IntToStringWithLeadingZero(GetSensorId()) + "_" + std::to_string(m_uCurrentFrameId) +
                                   ".yuv";
        std::ifstream yuvFile(sYUVFilePath, std::ios::in | std::ios::binary);
        if (!yuvFile.is_open()) {
            PLOG_WARN("ReadYuvFrame: Open yuv file %s failed, try to get next yuv frame.\n", sYUVFilePath.c_str());
            m_uMissingFileCount++;
            m_uCurrentFrameId++;
            continue;
        }

        auto error = ReadYuvData(yuvFile);
        if (yuvFile.is_open()) {
            yuvFile.close();
        }
        return error;
    }
    PLOG_ERR("ReadYuvFrame: Open yuv file failed, app exited. \n");
    return EventStatus::ERROR;
}

EventStatus CFrameReader::ReadYuvSequence()
{
    if (!m_yuvSequenceFile.is_open()) {
        PLOG_ERR("ReadYuvSequence: Read yuv sequence file failed! \n");
        return EventStatus::ERROR;
    }

    if (m_yuvSequenceFile.eof()) {
        LOG_MSG("CFrameReader ReadYuvSequence: Read yuv sequence completed, app exited. \n");
        return EventStatus::QUITTED;
    }

    return ReadYuvData(m_yuvSequenceFile);
}

EventStatus CFrameReader::ReadYuvData(std::ifstream &yuvFile)
{
    NvSciBufObj bufObj = nullptr;
    {
        std::unique_lock<std::mutex> lock(m_freeBufMutex);
        while (!m_bStopReading.load() && m_freeBufferList.empty()) {
            m_freeBufDataCond.wait(lock);
        }
        if (m_bStopReading.load()) {
            PLOG_WARN("ReadYuvData: Stop reading, app stopped!\n");
            return EventStatus::QUITTED;
        }
        bufObj = m_freeBufferList.front();
        m_freeBufferList.pop_front();
    }

    if (!bufObj) {
        PLOG_ERR("ReadYuvData: No free buffer to store frame data.\n");
        return EventStatus::ERROR;
    }

    void *pSciBufPtr = nullptr;
    NvSciError sciErr;
    sciErr = NvSciBufObjGetCpuPtr(bufObj, &pSciBufPtr);
    if (sciErr != NvSciError_Success) {
        PLOG_ERR("ReadYuvData: NvSciBufObjGetConstCpuPtr failed.\n");
        return EventStatus::ERROR;
    }
    BufferAttrs bufAttrs = m_bufToAttrsMap[bufObj];
    for (uint32_t plane = 0; plane < bufAttrs.planeCount; ++plane) {
        uint64_t uSciBufPlaneOffset = bufAttrs.planeOffsets[plane];
        uint32_t uSiBufPlanePitch = bufAttrs.planePitches[plane];
        uint32_t uPlaneWidth = bufAttrs.planeWidths[plane];
        uint32_t uPlaneHeight = bufAttrs.planeHeights[plane];
        PLOG_DBG("ReadYuvData plane: %" PRIu32 " planeOffsets: %" PRIu64 " planePitches: %" PRIu32
                 " uPlaneWidth: %" PRIu32 " uPlaneHeight: %" PRIu32,
                 plane, uSciBufPlaneOffset, uSiBufPlanePitch, uPlaneWidth, uPlaneHeight);
        for (uint32_t line = 0; line < uPlaneHeight; ++line) {
            char *pSciBufLine = (char *)pSciBufPtr + uSciBufPlaneOffset + line * uSiBufPlanePitch;
            yuvFile.read(pSciBufLine, uPlaneWidth);
            if ((uint64_t)yuvFile.gcount() != uPlaneWidth) {
                PLOG_ERR("ReadYuvData: Read yuv frame failed! \n");
                return EventStatus::ERROR;
            }
        }
    }

    std::lock_guard<std::mutex> lock(m_validBufMutex);
    m_validBufferQueue.emplace(bufObj);
    m_validBufDataCond.notify_all();

    m_uCurrentFrameId++;
    return EventStatus::OK;
}

NvError CFrameReader::FillNvSciBufAttrList(NvSciBufAttrList &bufAttrList)
{
    bool bImgCpuAccess = true;
    uint32_t uWidth = GetFrameWidth();
    uint32_t uHeight = GetFrameHeight();
    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufSurfType surfType = NvSciSurfType_YUV;
    NvSciBufSurfBPC surfBPC = NvSciSurfBPC_8;
    NvSciBufAttrValAccessPerm accessPerm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufAttrValImageScanType scanType = NvSciBufScan_ProgressiveType;
    NvSciBufSurfMemLayout surfMemLayout = NvSciSurfMemLayout_Planar;
    NvSciBufAttrValImageLayoutType layout = NvSciBufImage_PitchLinearType;
    NvSciBufSurfSampleType surfSampleType = NvSciSurfSampleType_420;
    NvSciBufSurfComponentOrder surfCompOrder = NvSciSurfComponentOrder_YUV;
    NvSciBufAttrValColorStd surfColorStd[] = { NvSciColorStd_REC709_ER };

    NvSciBufAttrKeyValuePair keyVals[] = {
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { NvSciBufImageAttrKey_SurfType, &surfType, sizeof(surfType) },
        { NvSciBufImageAttrKey_SurfBPC, &surfBPC, sizeof(surfBPC) },
        { NvSciBufGeneralAttrKey_RequiredPerm, &accessPerm, sizeof(accessPerm) },
        { NvSciBufImageAttrKey_PlaneScanType, &scanType, sizeof(scanType) },
        { NvSciBufImageAttrKey_SurfMemLayout, &surfMemLayout, sizeof(surfMemLayout) },
        { NvSciBufImageAttrKey_Layout, &layout, sizeof(layout) },
        { NvSciBufImageAttrKey_SurfSampleType, &surfSampleType, sizeof(surfSampleType) },
        { NvSciBufImageAttrKey_SurfComponentOrder, &surfCompOrder, sizeof(surfCompOrder) },
        { NvSciBufImageAttrKey_SurfColorStd, &surfColorStd, sizeof(surfColorStd) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &bImgCpuAccess, sizeof(bool) },
        { NvSciBufImageAttrKey_SurfWidthBase, &uWidth, sizeof(uint32_t) },
        { NvSciBufImageAttrKey_SurfHeightBase, &uHeight, sizeof(uint32_t) },
    };

    auto err = NvSciBufAttrListSetAttrs(bufAttrList, keyVals, ARRAY_SIZE(keyVals));
    CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufAttrListSetAttrs");

    return NvError_Success;
}

NvError CFrameReader::FillSyncSignalerAttrList(NvSciSyncAttrList &signalerAttrList)
{
    bool bNeedCpuAccess = true;
    NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_SignalOnly;
    NvSciSyncAttrKeyValuePair keyValues[] = { { NvSciSyncAttrKey_NeedCpuAccess, &bNeedCpuAccess,
                                                sizeof(bNeedCpuAccess) },
                                              { NvSciSyncAttrKey_RequiredPerm, &cpuPerm, sizeof(cpuPerm) } };

    auto sciErr = NvSciSyncAttrListSetAttrs(signalerAttrList, keyValues, ARRAY_SIZE(keyValues));
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListSetAttrs signal");

    return NvError_Success;
}

NvError CFrameReader::FillSyncWaiterAttrList(NvSciSyncAttrList &waiterAttrList)
{
    bool bNeedCpuAccess = true;
    NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_WaitOnly;
    NvSciSyncAttrKeyValuePair setAttrs[] = { { NvSciSyncAttrKey_NeedCpuAccess, &bNeedCpuAccess,
                                               sizeof(bNeedCpuAccess) },
                                             { NvSciSyncAttrKey_RequiredPerm, &cpuPerm, sizeof(cpuPerm) } };
    NvSciError sciErr = NvSciSyncAttrListSetAttrs(waiterAttrList, setAttrs, ARRAY_SIZE(setAttrs));
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListSetAttrs signal");
    return NvError_Success;
}

NvError CFrameReader::RegisterNvSciBuf(NvSciBufObj &bufObj)
{
    BufferAttrs bufAttrs;
    NvError error = PopulateBufAttr(bufObj, bufAttrs);
    PCHK_ERROR_AND_RETURN(error, "PopulateBufAttr");
    m_bufToAttrsMap[bufObj] = bufAttrs;

    std::lock_guard<std::mutex> lock(m_freeBufMutex);
    m_freeBufferList.emplace_back(bufObj);
    m_freeBufDataCond.notify_one();

    return NvError_Success;
}

EventStatus CFrameReader::LoadFrameData(NvSciBufObj &bufObj, NvSciSyncFence *&pPostFence)
{
    {
        static_cast<void>(pPostFence);
        std::unique_lock<std::mutex> lock(m_validBufMutex);
        while (!m_bStopReading.load() && m_validBufferQueue.empty()) {
            m_validBufDataCond.wait(lock);
        }
        if (m_bStopReading.load()) {
            PLOG_WARN("LoadFrameData: Stop reading, app stopped!\n");
            return EventStatus::QUITTED;
        }
        bufObj = m_validBufferQueue.front();
        m_validBufferQueue.pop();
    }
    return EventStatus::OK;
}

void CFrameReader::ReturnBuffer(NvSciBufObj &bufObj)
{
    {
        std::lock_guard<std::mutex> lock(m_freeBufMutex);
        m_freeBufferList.emplace_back(bufObj);
        m_freeBufDataCond.notify_one();
    }
}
