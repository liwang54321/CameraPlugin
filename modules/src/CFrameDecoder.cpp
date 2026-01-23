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

#if !NV_IS_SAFETY

#include <chrono>
#include <thread>
#include <malloc.h>
#include "CFrameDecoder.hpp"
#include "CUtils.hpp"
#include "CEventHandler.hpp"

CFrameDecoder::CFrameDecoder(FileSourceType type,
                             const std::string &sFilePath,
                             uint32_t uWidth,
                             uint32_t uHeight,
                             int sensorId,
                             uint32_t uInstanceId)
    : CFrameHandler(type, sFilePath, uWidth, uHeight, sensorId)
    , m_uInstanceId(uInstanceId)
{
    m_clientCb = { &cbBeginSequence,
                   &cbDecodePicture,
                   &cbDisplayPicture,
                   nullptr,
                   &cbAllocPictureBuffer,
                   &cbRelease,
                   &cbAddRef,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr,
                   nullptr };
}

CFrameDecoder::~CFrameDecoder() {}

NvError CFrameDecoder::Init()
{
    std::string sInputCodecFileSuffix;
    if (GetSourceFileType() == FileSourceType::H264) {
        m_clientCtx.eCodec = NVMEDIA_VIDEO_CODEC_H264;
        sInputCodecFileSuffix = ".h264";
    } else if (GetSourceFileType() == FileSourceType::H265) {
        m_clientCtx.eCodec = NVMEDIA_VIDEO_CODEC_HEVC;
        sInputCodecFileSuffix = ".h265";
    } else {
        PLOG_ERR("Init: Invalid input codec type.\n");
        return NvError_BadParameter;
    }
    m_clientCtx.pParser = nullptr;
    m_clientCtx.decodeWidth = GetFrameWidth();
    m_clientCtx.decodeHeight = GetFrameHeight();
    m_clientCtx.displayWidth = GetFrameWidth();
    m_clientCtx.displayHeight = GetFrameHeight();
    m_clientCtx.pDecoder = nullptr;
    m_clientCtx.decodeCount = 0;

    m_clientCtx.uBuffers = 0;
    m_clientCtx.picNum = 0;
    memset(&m_clientCtx.frameBufs[0], 0, sizeof(FrameBuffer) * MAX_DEC_BUFFERS);

    m_clientCtx.instanceId = static_cast<NvMediaDecoderInstanceId>(m_uInstanceId);
    m_clientCtx.frameCount = 0;

    m_clientCtx.eofSyncObj = nullptr;

    memset(&m_clientCtx.nvmParserParams, 0, sizeof(NvMediaParserParams));
    m_clientCtx.nvmParserParams.pClient = &m_clientCb;
    m_clientCtx.nvmParserParams.pClientCtx = &m_clientCtx;
    m_clientCtx.nvmParserParams.uErrorThreshold = 50;
    m_clientCtx.nvmParserParams.uReferenceClockRate = 0;
    m_clientCtx.nvmParserParams.eCodec = m_clientCtx.eCodec;

    m_clientCtx.pParser = NvMediaParserCreate(&m_clientCtx.nvmParserParams);
    if (!m_clientCtx.pParser) {
        PLOG_ERR("Init: NvMediaParserCreate failed\n");
        return NvError_InsufficientMemory;
    }

    float defaultDecFrameRate = 30.0f;
    NvMediaParserSetAttribute(m_clientCtx.pParser, NvMParseAttr_SetDefaultFramerate, sizeof(float),
                              &defaultDecFrameRate);

    std::string sInputCodecFilePath = GetCameraDirName() + "/" + CODEC_FILE_PREFIX +
                                      IntToStringWithLeadingZero(GetSensorId()) + sInputCodecFileSuffix;
    m_inputCodecFile = fopen(sInputCodecFilePath.c_str(), "rb");
    if (!m_inputCodecFile) {
        PLOG_ERR("Init: Failed open %s file for reading.\n", sInputCodecFilePath.c_str());
        return NvError_FileOperationFailed;
    }

    m_codecStreamBuf = new (std::nothrow) uint8_t[IDE_CODEC_STREAM_READ_SIZE];
    if (!m_codecStreamBuf) {
        PLOG_ERR("Init: Failed allocating memory for code stream buffer.\n");
        return NvError_InsufficientMemory;
    }

    m_upDecodeHandler = std::make_unique<CEventHandler<CFrameDecoder>>();
    auto error = m_upDecodeHandler->RegisterHandler(&CFrameDecoder::Decode, this);
    PCHK_ERROR_AND_RETURN(error, "RegisterHandler");

    error = m_upDecodeHandler->StartThread();
    PCHK_ERROR_AND_RETURN(error, " StartThread");

    {
        std::unique_lock<std::mutex> lock(m_clientCtx.updatedMutex);
        while (!m_clientCtx.bParamsHasUpdated.load()) {
            m_clientCtx.updatedCond.wait(lock);
        }
    }

    error = InitDecoder();
    PCHK_ERROR_AND_RETURN(error, " InitDecoder");

    return NvError_Success;
}

NvError CFrameDecoder::InitDecoder()
{
    uint32_t maxReferences = MAX_NUM_PACKETS - 1;

    LOG_DBG(" Size: %dx%d maxReferences: %d\n", m_clientCtx.decodeWidth, m_clientCtx.decodeHeight, maxReferences);

#if NV_BUILD_DOS7
    NvMediaDeviceList deviceList{};

    auto nvmediaStatus = NvMediaIDEQueryHWDevices(&deviceList);
    PCHK_NVMSTATUS_AND_RETURN(nvmediaStatus, "NvMediaIDEQueryHWDevices");
    LOG_DBG("Decoder device list size: %d \n", deviceList.deviceListSize);
    if (deviceList.deviceListSize == 0U) {
        PLOG_ERR("No support for Decoder device.\n");
        return NvError_NotSupported;
    }
    if (m_clientCtx.instanceId >= deviceList.deviceListSize) {
        PLOG_ERR("No support for requested decoder instance id.\n");
        return NvError_BadParameter;
    }
    for (auto uIndex = 0U; uIndex < deviceList.deviceListSize; ++uIndex) {
        PLOG_INFO("device num %d numaDomainId %d \n", uIndex, deviceList.deviceList[uIndex].numaDomainId);
        PrintDeviceGid("Decoder", "Primary GPU UUID:", deviceList.deviceList[uIndex].uuid.id);
        PrintDeviceGid("Decoder", "Hardware GPU UUID:", deviceList.deviceList[uIndex].hwGid.id);
    }
#endif

    m_clientCtx.pDecoder = NvMediaIDECreateCtx();

    if (!m_clientCtx.pDecoder) {
        PLOG_ERR("InitDecoder: Unable to create decoder.\n");
        return NvError_InsufficientMemory;
    }

    NvMediaStatus status = NvMediaIDEInit(m_clientCtx.pDecoder, m_clientCtx.eCodec,
                                          m_clientCtx.decodeWidth,       // width
                                          m_clientCtx.decodeHeight,      // height
                                          maxReferences,                 // maxReferences
                                          m_clientCtx.uMaxBitstreamSize, //maxBitstreamSize
                                          1,                             // inputBuffering
                                          0,                             // decoder flags
                                          m_clientCtx.instanceId         // instance ID
    );
    PCHK_NVMSTATUS_AND_RETURN(status, "NvMediaIDEInit");

    return NvError_Success;
}

void CFrameDecoder::DeInit()
{
    m_clientCtx.bQuitDecoding.store(true);
    if (m_upDecodeHandler) {
        m_upDecodeHandler->QuitThread();
    }

    NvMediaParserDestroy(m_clientCtx.pParser);
    NvMediaStatus status = NVMEDIA_STATUS_OK;

    if (m_clientCtx.pDecoder) {
        for (uint32_t i = 0; i < MAX_DEC_BUFFERS; i++) {
            if (m_clientCtx.frameBufs[i].videoSurface) {
                NvSciSyncFenceClear(&m_clientCtx.frameBufs[i].preFence);
                status = NvMediaIDEUnregisterNvSciBufObj(m_clientCtx.pDecoder, m_clientCtx.frameBufs[i].videoSurface);
                if (status != NVMEDIA_STATUS_OK) {
                    LOG_ERR("Failed to uregister NvSciBufObj.\n");
                }
                NvSciBufObjFree(m_clientCtx.frameBufs[i].videoSurface);
                m_clientCtx.frameBufs[i].videoSurface = nullptr;
            }
        }
        if (m_clientCtx.eofSyncObj != nullptr) {
            NvMediaIDEUnregisterNvSciSyncObj(m_clientCtx.pDecoder, m_clientCtx.eofSyncObj);
            m_clientCtx.eofSyncObj = nullptr;
        }
        NvMediaIDEDestroy(m_clientCtx.pDecoder);
        m_clientCtx.pDecoder = nullptr;
    }

    if (m_inputCodecFile) {
        fclose(m_inputCodecFile);
        m_inputCodecFile = nullptr;
    }

    if (m_codecStreamBuf) {
        delete[] m_codecStreamBuf;
        m_codecStreamBuf = nullptr;
    }
}

NvError CFrameDecoder::Start()
{
    AcquireFreeBuffers();

    {
        std::lock_guard<std::mutex> lock(m_clientCtx.decodingReadyMutex);
        m_clientCtx.bDecodingReady.store(true);
        m_clientCtx.decodingReadyCond.notify_all();
    }

    m_clientCtx.bStopDecoding.store(false);
    auto error = m_upDecodeHandler->StartThread();
    PCHK_ERROR_AND_RETURN(error, "StartThread");

    return NvError_Success;
}

void CFrameDecoder::Stop()
{
    m_clientCtx.bStopDecoding.store(true);
    m_clientCtx.validBufCond.notify_all();
}

void CFrameDecoder::AcquireFreeBuffers()
{
    {
        std::lock_guard<std::mutex> lock(m_clientCtx.freeBufMutex);
        m_clientCtx.freeBufferList.clear();
    }

    /*
    Scenario 1: When starting for the first time, inserte all registered NvSciBufs into freeBufferList;
    Scenario 2 : When the command line inputs 's' to stop, some NvSciBufs will not be inserted into
    freeBufferList through ReturnBuffer, so after restarting, clear freeBufferList and insert all
    unfilled NvSciBufs into freeBufferList.
    */
    for (auto it = m_clientCtx.matchBufMap.begin(); it != m_clientCtx.matchBufMap.end(); ++it) {
        std::lock_guard<std::mutex> lock(m_clientCtx.validBufMutex);
        auto pos = std::find(m_clientCtx.validBufferQueue.begin(), m_clientCtx.validBufferQueue.end(), it->second);
        if (pos == m_clientCtx.validBufferQueue.end()) {
            std::lock_guard<std::mutex> lock(m_clientCtx.freeBufMutex);
            m_clientCtx.freeBufferList.emplace_back(it->second);
        }
    }
}

NvError CFrameDecoder::FillNvSciBufAttrList(NvSciBufAttrList &bufAttrList)
{
    auto nvmStatus = NvMediaIDEFillNvSciBufAttrList(m_clientCtx.instanceId, bufAttrList);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMediaIDEFillNvSciBufAttrList");

    NvSciBufAttrValColorStd colorFormat = NvSciColorStd_REC601_SR;
    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValAccessPerm accessPerm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufAttrValColorFmt planerColorFormats[NV_SCI_BUF_IMAGE_MAX_PLANES];
    NvSciBufAttrValColorStd colorStd[NV_SCI_BUF_IMAGE_MAX_PLANES];
    NvSciBufAttrValImageLayoutType layout = NvSciBufImage_BlockLinearType;
    NvSciBufAttrValImageScanType scanType = NvSciBufScan_ProgressiveType;
    uint32_t planeCount = IDE_APP_MAX_INPUT_PLANE_COUNT;
    uint32_t planeWidth[NV_SCI_BUF_IMAGE_MAX_PLANES] = { 0 };
    uint32_t planeHeight[NV_SCI_BUF_IMAGE_MAX_PLANES] = { 0 };
    uint32_t baseAddrAlign[NV_SCI_BUF_IMAGE_MAX_PLANES] = { 0 };
    uint64_t padding[NV_SCI_BUF_IMAGE_MAX_PLANES] = { 0 };
    bool bNeedCpuAccess = true;
    bool vprFlag = false;

    planerColorFormats[0] = NvSciColor_Y8;
    planerColorFormats[1] = NvSciColor_V8U8;

    /* Set image dimensions */
    planeWidth[0] = m_clientCtx.displayWidth;
    planeHeight[0] = m_clientCtx.displayHeight;
    planeWidth[1] = planeWidth[0] >> 1;
    planeHeight[1] = planeHeight[0] >> 1;

    m_clientCtx.decodeWidth = planeWidth[0];
    m_clientCtx.decodeHeight = planeHeight[0];

    colorStd[0] = colorFormat;
    colorStd[1] = colorFormat;
    baseAddrAlign[0] = IDE_APP_BASE_ADDR_ALIGN;
    baseAddrAlign[1] = IDE_APP_BASE_ADDR_ALIGN;

    LOG_DBG("planeWidth: %d planeHeight: %d decodeWidth: %d decodeHeight: %d\n", planeWidth[0], planeHeight[0],
            m_clientCtx.decodeWidth, m_clientCtx.decodeHeight);

    /* Set all key-value pairs */
    NvSciBufAttrKeyValuePair attributes[] = {
        { NvSciBufGeneralAttrKey_RequiredPerm, &accessPerm, sizeof(accessPerm) },
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &bNeedCpuAccess, sizeof(bNeedCpuAccess) },
        { NvSciBufGeneralAttrKey_EnableCpuCache, &bNeedCpuAccess, sizeof(bNeedCpuAccess) },
        { NvSciBufImageAttrKey_TopPadding, &padding, planeCount * sizeof(padding[0]) },
        { NvSciBufImageAttrKey_BottomPadding, &padding, planeCount * sizeof(padding[0]) },
        { NvSciBufImageAttrKey_LeftPadding, &padding, planeCount * sizeof(padding[0]) },
        { NvSciBufImageAttrKey_RightPadding, &padding, planeCount * sizeof(padding[0]) },
        { NvSciBufImageAttrKey_Layout, &layout, sizeof(layout) },
        { NvSciBufImageAttrKey_PlaneCount, &planeCount, sizeof(planeCount) },
        { NvSciBufImageAttrKey_PlaneColorFormat, &planerColorFormats, planeCount * sizeof(NvSciBufAttrValColorFmt) },
        { NvSciBufImageAttrKey_PlaneColorStd, &colorStd, planeCount * sizeof(NvSciBufAttrValColorStd) },
        { NvSciBufImageAttrKey_PlaneBaseAddrAlign, &baseAddrAlign, planeCount * sizeof(uint32_t) },
        { NvSciBufImageAttrKey_PlaneWidth, &planeWidth, planeCount * sizeof(uint32_t) },
        { NvSciBufImageAttrKey_PlaneHeight, &planeHeight, planeCount * sizeof(uint32_t) },
        { NvSciBufImageAttrKey_VprFlag, &vprFlag, sizeof(vprFlag) },
        { NvSciBufImageAttrKey_ScanType, &scanType, sizeof(NvSciBufAttrValImageScanType) }
    };

    auto sciErr = NvSciBufAttrListSetAttrs(bufAttrList, attributes, ARRAY_SIZE(attributes));
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

    return NvError_Success;
}

NvError CFrameDecoder::FillSyncSignalerAttrList(NvSciSyncAttrList &signalerAttrList)
{
    PCHK_PTR_AND_RETURN_ERR(m_clientCtx.pDecoder, "m_clientCtx.pDecoder");

    auto nvmStatus = NvMediaIDEFillNvSciSyncAttrList(m_clientCtx.pDecoder, signalerAttrList, NVMEDIA_SIGNALER);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "Signaler NvMediaIDEFillNvSciSyncAttrList");

    return NvError_Success;
}

NvError CFrameDecoder::FillSyncWaiterAttrList(NvSciSyncAttrList &waiterAttrList)
{
    PCHK_PTR_AND_RETURN_ERR(m_clientCtx.pDecoder, "m_clientCtx.pDecoder");

    auto nvmStatus = NvMediaIDEFillNvSciSyncAttrList(m_clientCtx.pDecoder, waiterAttrList, NVMEDIA_WAITER);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "Waiter NvMediaIDEFillNvSciSyncAttrList");

    return NvError_Success;
}

NvError CFrameDecoder::RegisterNvSciBuf(NvSciBufObj &bufObj)
{
    PCHK_PTR_AND_RETURN_ERR(m_clientCtx.pDecoder, "m_clientCtx.pDecoder");

    auto sciErr = NvSciBufObjDup(bufObj, &m_clientCtx.frameBufs[m_clientCtx.uBuffers].videoSurface);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjDup");

    auto nvmStatus =
        NvMediaIDERegisterNvSciBufObj(m_clientCtx.pDecoder, m_clientCtx.frameBufs[m_clientCtx.uBuffers].videoSurface);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMediaIDERegisterNvSciBufObj");

    m_clientCtx.matchBufMap.insert({ m_clientCtx.frameBufs[m_clientCtx.uBuffers].videoSurface, bufObj });
    m_clientCtx.uBuffers++;

    return NvError_Success;
}

NvError CFrameDecoder::RegisterSignalSyncObj(NvSciSyncObj &signalSyncObj)
{
    PCHK_PTR_AND_RETURN_ERR(m_clientCtx.pDecoder, "m_clientCtx.pDecoder");

    if (!m_clientCtx.eofSyncObj) {
        m_clientCtx.eofSyncObj = signalSyncObj;
    }
    auto nvmStatus = NvMediaIDERegisterNvSciSyncObj(m_clientCtx.pDecoder, NVMEDIA_EOFSYNCOBJ, signalSyncObj);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMediaIDERegisterNvSciSyncObj");

    nvmStatus = NvMediaIDESetNvSciSyncObjforEOF(m_clientCtx.pDecoder, signalSyncObj);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMediaIDESetNvSciSyncObjforEOF");

    return NvError_Success;
}

EventStatus CFrameDecoder::LoadFrameData(NvSciBufObj &bufObj, NvSciSyncFence *&pPostFence)
{
    std::unique_lock<std::mutex> lock(m_clientCtx.validBufMutex);
    while (!m_clientCtx.bStopDecoding.load() && !m_clientCtx.bQuitDecoding.load() &&
           m_clientCtx.validBufferQueue.empty()) {
        m_clientCtx.validBufCond.wait(lock);
    }
    if (m_clientCtx.bStopDecoding.load() || m_clientCtx.bQuitDecoding.load()) {
        PLOG_WARN("LoadFrameData: Stop decoding, app quitted!\n");
        return EventStatus::QUITTED;
    }
    bufObj = m_clientCtx.validBufferQueue.front();
    m_clientCtx.validBufferQueue.pop_front();
    pPostFence = m_clientCtx.buf2FenceMap[bufObj];

    return EventStatus::OK;
}

void CFrameDecoder::ReturnBuffer(NvSciBufObj &bufObj)
{
    std::lock_guard<std::mutex> lock(m_clientCtx.freeBufMutex);
    m_clientCtx.freeBufferList.emplace_back(bufObj);
}

int32_t CFrameDecoder::cbBeginSequence(void *ptr, const NvMediaParserSeqInfo *pnvsi)
{
    ParserClientCtx *m_clientCtx = (ParserClientCtx *)ptr;
    if (!pnvsi || !m_clientCtx) {
        LOG_ERR("cbBeginSequence: Invalid NvMediaParserSeqInfo or ParserClientCtx\n");
        return -1;
    }

    if (pnvsi->eCodec < 0) {
        LOG_ERR("cbBeginSequence: Invalid codec type: %d\n", pnvsi->eCodec);
        return 0;
    }

    if (pnvsi->eCodec == NVMEDIA_VIDEO_CODEC_HEVC) {
        if ((pnvsi->uCodedWidth < 144) || (pnvsi->uCodedHeight < 144)) {
            LOG_ERR("cbBeginSequence: (Width=%d, Height=%d) < (144, 144) NOT SUPPORTED for HEVC\n", pnvsi->uCodedWidth,
                    pnvsi->uCodedHeight);
            return -1;
        }
    }

    uint32_t decodeBuffers = pnvsi->uDecodeBuffers;

    LOG_DBG("cbBeginSequence: %dx%d (disp: %dx%d) decode buffers: %d aspect: %d:%d fps: %f \n", pnvsi->uCodedWidth,
            pnvsi->uCodedHeight, pnvsi->uDisplayWidth, pnvsi->uDisplayHeight, pnvsi->uDecodeBuffers, pnvsi->uDARWidth,
            pnvsi->uDARHeight, pnvsi->fFrameRate);

    m_clientCtx->decodeWidth = pnvsi->uCodedWidth;
    m_clientCtx->decodeHeight = pnvsi->uCodedHeight;

    m_clientCtx->displayWidth = pnvsi->uDisplayWidth;
    m_clientCtx->displayHeight = pnvsi->uDisplayHeight;

    LOG_DBG("cbBeginSequence: pnvsi->uCodedWidth: %d m_clientCtx->decodeWidth: %d pnvsi->uCodedHeight: %d "
            "m_clientCtx->decodeHeight: %d",
            pnvsi->uCodedWidth, m_clientCtx->decodeWidth, pnvsi->uCodedHeight, m_clientCtx->decodeHeight);

    switch (pnvsi->eCodec) {
        case NVMEDIA_VIDEO_CODEC_H264:
            m_clientCtx->eCodec = NVMEDIA_VIDEO_CODEC_H264;
            LOG_INFO("NVMEDIA_VIDEO_CODEC_H264");
            break;
        case NVMEDIA_VIDEO_CODEC_HEVC:
            m_clientCtx->eCodec = NVMEDIA_VIDEO_CODEC_HEVC;
            LOG_INFO("NVMEDIA_VIDEO_CODEC_HEVC");
            break;
        default:
            LOG_ERR("cbBeginSequence: Invalid decoder type\n");
            return 0;
    }
    m_clientCtx->uMaxBitstreamSize = pnvsi->uMaxBitstreamSize;

    {
        std::lock_guard<std::mutex> lock(m_clientCtx->updatedMutex);
        m_clientCtx->bParamsHasUpdated.store(true);
        m_clientCtx->updatedCond.notify_all();
    }

    return decodeBuffers;
}

NvMediaStatus CFrameDecoder::cbDecodePicture(void *ptr, NvMediaParserPictureData *pd)
{
    ParserClientCtx *m_clientCtx = (ParserClientCtx *)ptr;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    FrameBuffer *targetBuffer = nullptr;
    NvMediaBitstreamBuffer bitStreamBuffer;
    NvMediaIDEFrameStatus frameStatus = { 0 };
    NvMediaIDEFrameStats frameStatsDump = { 0 };

    if (!pd || !m_clientCtx) {
        LOG_ERR("cbDecodePicture: Invalid NvMediaParserPictureData or ParserClientCtx.\n");
        return NVMEDIA_STATUS_ERROR;
    }

    if (pd->pCurrPic) {
        targetBuffer = (FrameBuffer *)pd->pCurrPic;

        /*
         * Update NvMedia reference pointers from parser with corresponding
         * NvSciBufObj pointers in NvMediaPictureInfo structure
         * for each codec type.
         */
        switch (m_clientCtx->eCodec) {
            case NVMEDIA_VIDEO_CODEC_H264:
                status = UpdateNvMediaSurfacePictureInfoH264(m_clientCtx,
                                                             (NvMediaPictureInfoH264 *)&pd->CodecSpecificInfo.h264);
                break;
            case NVMEDIA_VIDEO_CODEC_HEVC:
                status = UpdateNvMediaSurfacePictureInfoH265(m_clientCtx,
                                                             (NvMediaPictureInfoH265 *)&pd->CodecSpecificInfo.hevc);
                break;
            default:
                LOG_ERR("cbDecodePicture: Invalid decoder type.\n");
                return NVMEDIA_STATUS_ERROR;
        }
        if (status != NVMEDIA_STATUS_OK) {
            LOG_ERR("cbDecodePicture: Decode failed in UpdateNvMediaSurfacePictureInfo. Status: %d\n", status);
            return NVMEDIA_STATUS_ERROR;
        }
        targetBuffer->frameNum = m_clientCtx->picNum;
        targetBuffer->topFieldFirstFlag = !!pd->top_field_first;      // Frame pictures only
        targetBuffer->progressiveFrameFlag = !!pd->progressive_frame; // Frame is progressive
        bitStreamBuffer.bitstream = (uint8_t *)pd->pBitstreamData;
        bitStreamBuffer.bitstreamBytes = pd->uBitstreamDataLen;

        targetBuffer->lDARWidth = pd->uDARWidth;
        targetBuffer->lDARHeight = pd->uDARHeight;
        targetBuffer->displayLeftOffset = pd->uDisplayLeftOffset;
        targetBuffer->displayTopOffset = pd->uDisplayTopOffset;
        targetBuffer->displayWidth = pd->uDisplayWidth;
        targetBuffer->displayHeight = pd->uDisplayHeight;

        LOG_DBG("cbDecodePicture: %d Ptr: %p Surface: %p (stream ptr: %p size: %d)\n", m_clientCtx->picNum,
                targetBuffer, targetBuffer->videoSurface, pd->pBitstreamData, pd->uBitstreamDataLen);
        m_clientCtx->picNum++;

        if (targetBuffer->videoSurface) {
            status = NvMediaIDEDecoderRender(m_clientCtx->pDecoder,                        // decoder
                                             targetBuffer->videoSurface,                   // target
                                             (NvMediaPictureInfo *)&pd->CodecSpecificInfo, // pictureInfo
                                             nullptr,                                      // encryptParams
                                             1,                                            // numBitstreamBuffers
                                             &bitStreamBuffer,                             // bitstreams
                                             &frameStatsDump,                              // FrameStatsDump
                                             m_clientCtx->instanceId);                     // instance ID
            if (status != NVMEDIA_STATUS_OK) {
                LOG_ERR("cbDecodePicture: Decode failed. Status: %d\n", status);
                return NVMEDIA_STATUS_ERROR;
            }
            LOG_DBG("cbDecodePicture: Frame decode done\n");

            status =
                NvMediaIDEGetEOFNvSciSyncFence(m_clientCtx->pDecoder, m_clientCtx->eofSyncObj, &targetBuffer->preFence);
            if (status != NVMEDIA_STATUS_OK) {
                LOG_ERR("cbDecodePicture: NvMediaIDEGetEOFNvSciSyncFence failed. Status: %x\n", status);
                return status;
            }
            m_clientCtx->buf2FenceMap[m_clientCtx->matchBufMap[targetBuffer->videoSurface]] = &targetBuffer->preFence;

            status = NvMediaIDEGetFrameDecodeStatus(m_clientCtx->pDecoder, frameStatsDump.uRingEntryIdx, &frameStatus);
            if (status == NVMEDIA_STATUS_OK) {
                LOG_DBG("cbDecodePicture: Frame Status decode_error %u\n", frameStatus.decode_error);
            }
        } else {
            LOG_ERR("cbDecodePicture: Invalid target surface.\n");
        }

        m_clientCtx->decodeCount++;
    } else {
        LOG_ERR("cbDecodePicture: No valid frame.\n");
        return NVMEDIA_STATUS_ERROR;
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus CFrameDecoder::cbDisplayPicture(void *ptr, NvMediaRefSurface *p, int64_t llPts)
{
    ParserClientCtx *m_clientCtx = (ParserClientCtx *)ptr;
    FrameBuffer *buffer = (FrameBuffer *)p;

    if (!m_clientCtx) {
        LOG_ERR("cbDisplayPicture: Invalid ParserClientCtx.\n");
        return NVMEDIA_STATUS_ERROR;
    }

    if (m_clientCtx->bQuitDecoding.load()) {
        LOG_WARN("cbDisplayPicture: Stop decoding, app quitted!\n");
        return NVMEDIA_STATUS_OK;
    }

    if (buffer) {
        LOG_DBG(" cbDisplayPicture: Display buffer for picture %d index: %d Ptr:%p Surface:%p\n",
                m_clientCtx->frameCount, buffer->index, buffer, buffer->videoSurface);
        {
            std::unique_lock<std::mutex> lock(m_clientCtx->freeBufMutex);
            for (auto it = m_clientCtx->freeBufferList.begin(); it != m_clientCtx->freeBufferList.end(); ++it) {
                if (*it == m_clientCtx->matchBufMap[buffer->videoSurface]) {
                    m_clientCtx->freeBufferList.erase(it);
                    break;
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(m_clientCtx->validBufMutex);
            m_clientCtx->validBufferQueue.emplace_back(m_clientCtx->matchBufMap[buffer->videoSurface]);
            m_clientCtx->validBufCond.notify_all();
        }

        m_clientCtx->frameCount++;
    } else {
        LOG_ERR("cbDisplayPicture: Invalid buffer.\n");
        return NVMEDIA_STATUS_ERROR;
    }

    return NVMEDIA_STATUS_OK;
}

NvMediaStatus CFrameDecoder::cbAllocPictureBuffer(void *ptr, NvMediaRefSurface **p)
{
    ParserClientCtx *m_clientCtx = (ParserClientCtx *)ptr;

    if (!m_clientCtx) {
        LOG_ERR("cbAllocPictureBuffer: Invalid ParserClientCtx.\n");
        return NVMEDIA_STATUS_ERROR;
    }

    *p = (NvMediaRefSurface *)nullptr;

    {
        std::unique_lock<std::mutex> lock(m_clientCtx->decodingReadyMutex);
        while (!m_clientCtx->bQuitDecoding.load() && !m_clientCtx->bDecodingReady.load()) {
            m_clientCtx->decodingReadyCond.wait(lock);
        }
        if (m_clientCtx->bQuitDecoding.load()) {
            LOG_WARN("cbAllocPictureBuffer: Stop decoding, app quitted!\n");
            return NVMEDIA_STATUS_OK;
        }
    }

    bool bBufferAssigned = false;
    int CountForTimeout = ITERATIONS_TILL_TIMEOUT;
    while (!bBufferAssigned && !m_clientCtx->bQuitDecoding.load()) {
        // to support r/s feature
        while (m_clientCtx->bStopDecoding.load() && !m_clientCtx->bQuitDecoding.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_FOR_PACKET_MS));
        }
        for (uint32_t i = 0; i < m_clientCtx->uBuffers; i++) {
            if (!m_clientCtx->frameBufs[i].refCount) {
                std::lock_guard<std::mutex> lock(m_clientCtx->freeBufMutex);
                for (auto it = m_clientCtx->freeBufferList.begin(); it != m_clientCtx->freeBufferList.end(); ++it) {
                    if (*it == m_clientCtx->matchBufMap[m_clientCtx->frameBufs[i].videoSurface]) {
                        *p = (NvMediaRefSurface *)&m_clientCtx->frameBufs[i];
                        m_clientCtx->frameBufs[i].refCount++;
                        m_clientCtx->frameBufs[i].index = i;
                        bBufferAssigned = true;
                        LOG_DBG(" cbAllocPictureBuffer: Allocated buffer for picture %d index: %d Ptr:%p "
                                "Surface:%p\n",
                                m_clientCtx->picNum, i, *p, m_clientCtx->frameBufs[i].videoSurface);
                        return NVMEDIA_STATUS_OK;
                    }
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_FOR_PACKET_MS));
        if (CountForTimeout == 0) {
            LOG_ERR("Alloc picture buffer failed: no free buffer. \n");
            return NVMEDIA_STATUS_ERROR;
        }
        CountForTimeout--;
    }

    if (m_clientCtx->bQuitDecoding.load()) {
        LOG_WARN("cbAllocPictureBuffer: Stop decoding, app quitted!\n");
        return NVMEDIA_STATUS_OK;
    }

    LOG_ERR("Alloc picture buffer failed.\n");
    return NVMEDIA_STATUS_ERROR;
}

void CFrameDecoder::cbRelease(void *ptr, NvMediaRefSurface *p)
{
    FrameBuffer *buffer = (FrameBuffer *)p;

    if (!buffer) {
        LOG_ERR("cbRelease: Invalid FrameBuffer.\n");
        return;
    }

    LOG_DBG("cbRelease: Releasing picture: %d index: %d refCount: %d\n", buffer->frameNum, buffer->index,
            buffer->refCount);
    if (buffer->refCount > 0)
        buffer->refCount--;
}

void CFrameDecoder::cbAddRef(void *ptr, NvMediaRefSurface *p)
{
    FrameBuffer *buffer = (FrameBuffer *)p;

    if (!buffer) {
        LOG_ERR("cbAddRef: Invalid FrameBuffer.\n");
        return;
    }

    LOG_DBG("Adding reference to picture: %d\n", buffer->frameNum);
    buffer->refCount++;
}

NvMediaStatus CFrameDecoder::UpdateNvMediaSurfacePictureInfoH264(ParserClientCtx *m_clientCtx,
                                                                 NvMediaPictureInfoH264 *pictureInfo)
{
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    for (uint32_t i = 0; i < UPDATE_NVM_SURFACE; i++) {
        NvMediaReferenceFrameH264 *pRefFrameOut = &pictureInfo->referenceFrames[i];
        FrameBuffer *pFrameBuf = (FrameBuffer *)pRefFrameOut->surface;
        pRefFrameOut->surface = pFrameBuf ? (NvMediaRefSurface *)(pFrameBuf->videoSurface) : nullptr;
    }
    return status;
}

NvMediaStatus CFrameDecoder::UpdateNvMediaSurfacePictureInfoH265(ParserClientCtx *m_clientCtx,
                                                                 NvMediaPictureInfoH265 *pictureInfo)
{
    FrameBuffer *pFrameBuf;
    NvMediaStatus status = NVMEDIA_STATUS_OK;
    for (uint32_t i = 0; i < UPDATE_NVM_SURFACE; i++) {
        pFrameBuf = (FrameBuffer *)pictureInfo->RefPics[i];
        pictureInfo->RefPics[i] = pFrameBuf ? (NvMediaRefSurface *)(pFrameBuf->videoSurface) : nullptr;
    }
    return status;
}

EventStatus CFrameDecoder::Decode()
{
    uint32_t uReadSize = IDE_CODEC_STREAM_READ_SIZE;
    uint32_t bitstream_error = 0;

    if (!m_codecStreamBuf) {
        LOG_ERR("CFrameDecoder::Decode: No valid code stream buffer.\n");
        return EventStatus::ERROR;
    }

    while (!feof(m_inputCodecFile) && !m_clientCtx.bQuitDecoding.load()) {
        // to support r/s feature
        while (m_clientCtx.bStopDecoding.load() && !m_clientCtx.bQuitDecoding.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(SLEEP_FOR_PACKET_MS));
        }
        NvMediaBitStreamPkt packet;
        memset(&packet, 0, sizeof(NvMediaBitStreamPkt));
        size_t len = fread(m_codecStreamBuf, 1, uReadSize, m_inputCodecFile);
        packet.uDataLength = (uint32_t)len;
        packet.pByteStream = m_codecStreamBuf;
        packet.bEOS = feof(m_inputCodecFile) ? true : false;
        LOG_DBG("CFrameDecoder Decode: EOS %d is sent...\n", packet.bEOS);
        packet.bPTSValid = 0; // (pts != (uint32_t)-1);
        packet.llPts = 0;     // packet.bPTSValid ? (1000 * pts / 9)  : 0;    // 100 ns scale
        if (NvMediaParserParse(m_clientCtx.pParser, &packet) != NVMEDIA_STATUS_OK) {
            LOG_ERR("CFrameDecoder Decode: NvMediaParserParse returned with failure.\n");
            return EventStatus::ERROR;
        }
    }
    NvMediaParserFlush(m_clientCtx.pParser);
    LOG_DBG("CFrameDecoder Decode: Finished decoding. Flushing parser and display.\n");

    rewind(m_inputCodecFile);
    if (NvMediaParserGetAttribute(m_clientCtx.pParser, NvMParseAttr_GetBitstreamError, sizeof(uint32_t),
                                  &bitstream_error) == NVMEDIA_STATUS_OK) {
        LOG_DBG("CFrameDecoder Decode: Bitstream error after parsing %u.\n", bitstream_error);
    }

    if (feof(m_inputCodecFile)) {
        LOG_MSG("CFrameDecoder Decode: Decoding is completed, app exited!\n");
        return EventStatus::QUITTED;
    }

    if (m_clientCtx.bQuitDecoding.load()) {
        LOG_WARN("CFrameDecoder Decode: Stop decoding, app quitted!\n");
        return EventStatus::QUITTED;
    }

    return EventStatus::OK;
}

#endif // !NV_IS_SAFETY
