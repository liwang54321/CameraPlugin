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

#include "CEncModule.hpp"
#include "nvscibuf.h"
#include "CElementDescription.hpp"

const std::unordered_map<std::string, Option> EncOptionTable = {
    { "enctype", { "Encode type", offsetof(EncOption, enctype), OptionType::INT } },
    { "width", { "the width of buffer", offsetof(EncOption, uWidth), OptionType::UINT32 } },
    { "height", { "the height of buffer", offsetof(EncOption, uHeight), OptionType::UINT32 } },
    { "maxOutput", { "max output buffer count", offsetof(EncOption, uMaxOutputBuffer), OptionType::UINT32 } },
    { "instanceId", { "instance id", offsetof(EncOption, uInstanceId), OptionType::UINT32 } },
#ifdef NVMEDIA_ENCODE_MAX_PII_REGIONS
    { "desensitized", { "desensitized the ROI region", offsetof(EncOption, bDesensitized), OptionType::BOOL } },
#endif
    { "averageBitrate", { "cbr average bitrate", offsetof(EncOption, uAverageBitrate), OptionType::UINT32 } },
};

#if !NV_IS_SAFETY
CElementDescription encDescription{ "Enc", "Encoder module for encoding YUV into H.264/H.265",
                                    &CBaseModule::m_baseModuleOptionTable, &EncOptionTable };
#else
CElementDescription encDescription{ "Enc", "Encoder module for encoding YUV into H.264",
                                    &CBaseModule::m_baseModuleOptionTable, &EncOptionTable };
#endif

CEncModule::CEncModule(std::shared_ptr<CModuleCfg> spModuleCfg, IEventListener<CBaseModule> *pListener)
    : CBaseModule(spModuleCfg, pListener)
{
    spModuleCfg->m_cpuWaitCfg = { true, true };
}

NvError CEncModule::Init()
{
    PLOG_DBG("Enter Init\n");

    NvError error = NvError_Success;
    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler = std::make_unique<CProfiler>();
        error = m_upProfiler->Init(m_pAppCfg->m_sciSyncModule, m_pAppCfg->GetPerfDataSaveFolder(), GetName(), false,
                                   m_pAppCfg->GetMaxPerfSampleNum());
        PCHK_ERROR_AND_RETURN(error, "CProfiler::Init()");
    }

    error = CBaseModule::Init();
    PCHK_ERROR_AND_RETURN(error, "CBaseModule::Init()");

    PLOG_DBG("Exit Init\n");
    return NvError_Success;
}

void CEncModule::DeInit()
{
    PLOG_DBG("Enter DeInit\n");

    CBaseModule::DeInit();

    m_upNvMIEP.reset();

    PLOG_DBG("Exit DeInit\n");
}

NvError CEncModule::SetupIEPH264(NvSciBufAttrList bufAttrList, uint32_t uWidth, uint32_t uHeight)
{
    NvMediaEncodeInitializeParamsH264 initParam = { 0 };
    initParam.profile = NVMEDIA_ENCODE_PROFILE_AUTOSELECT;
    initParam.level = NVMEDIA_ENCODE_LEVEL_AUTOSELECT;
    initParam.encodeHeight = uHeight;
    initParam.encodeWidth = uWidth;
    initParam.useBFramesAsRef = 0;
    initParam.frameRateDen = 1;
    initParam.frameRateNum = 30;
    initParam.maxNumRefFrames = 1;
    initParam.enableExternalMEHints = NVMEDIA_FALSE;
    initParam.enableAllIFrames = NVMEDIA_FALSE;

#ifdef NVMEDIA_ENCODE_MAX_PII_REGIONS
    if (m_encOption.bDesensitized) {
        initParam.enableAnonEncode = NVMEDIA_TRUE;
    }
#endif

    NvMediaEncodeConfigH264 configParam = { 0 };
    NvMediaEncodeConfigH264VUIParams vuiParams = { 0 };
    vuiParams.timingInfoPresentFlag = 1;
    configParam.h264VUIParameters = &vuiParams;

    // Setting Up Config Params
    configParam.gopLength = 30;
    configParam.idrPeriod = 300;
    configParam.repeatSPSPPS = NVMEDIA_ENCODE_SPSPPS_REPEAT_INTRA_FRAMES;
    configParam.adaptiveTransformMode = NVMEDIA_ENCODE_H264_ADAPTIVE_TRANSFORM_AUTOSELECT;
    configParam.bdirectMode = NVMEDIA_ENCODE_H264_BDIRECT_MODE_DISABLE;
    configParam.entropyCodingMode = NVMEDIA_ENCODE_H264_ENTROPY_CODING_MODE_CAVLC;

    // Ultra fast Encoder Quality preset
    configParam.encPreset = NVMEDIA_ENC_PRESET_UHP;

    // CBR, I P only
    configParam.rcParams.rateControlMode = NVMEDIA_ENCODE_PARAMS_RC_CBR;
    configParam.rcParams.params.cbr_minqp.averageBitRate = m_encOption.uAverageBitrate;
    // Set to 0 to automatically determine the right VBV buffer size and init delay
    configParam.rcParams.params.cbr_minqp.vbvBufferSize = 0;
    configParam.rcParams.params.cbr_minqp.vbvInitialDelay = 0;
    configParam.rcParams.numBFrames = 0;

    m_upNvMIEP.reset(
        NvMediaIEPCreate(NVMEDIA_IMAGE_ENCODE_H264,                                        // codec
                         &initParam,                                                       // init params
                         bufAttrList,                                                      // reconciled attr list
                         m_encOption.uMaxOutputBuffer,                                     // maxOutputBuffering
                         static_cast<NvMediaEncoderInstanceId>(m_encOption.uInstanceId))); // encoder instance

    PCHK_PTR_AND_RETURN(m_upNvMIEP, "NvMediaIEPCreate");

    auto nvmediaStatus = NvMediaIEPSetConfiguration(m_upNvMIEP.get(), &configParam);
    PCHK_NVMSTATUS_AND_RETURN(nvmediaStatus, "NvMediaIEPSetConfiguration failed");

    return NvError_Success;
}

#if !NV_IS_SAFETY
NvError CEncModule::SetupIEPH265(NvSciBufAttrList bufAttrList, uint32_t uWidth, uint32_t uHeight)
{
    NvMediaEncodeInitializeParamsH265 initParam = { 0 };
    initParam.profile = NVMEDIA_ENCODE_PROFILE_AUTOSELECT;
    initParam.level = NVMEDIA_ENCODE_LEVEL_AUTOSELECT;
    initParam.encodeHeight = uHeight;
    initParam.encodeWidth = uWidth;
    initParam.useBFramesAsRef = 0;
    initParam.frameRateDen = 1;
    initParam.frameRateNum = 30;
    initParam.maxNumRefFrames = 1;
    initParam.enableExternalMEHints = NVMEDIA_FALSE;
    initParam.enableAllIFrames = NVMEDIA_FALSE;

    NvMediaEncodeConfigH265 configParam = { 0 };
    configParam.gopLength = 30;
    configParam.idrPeriod = 300;
    configParam.repeatSPSPPS = NVMEDIA_ENCODE_SPSPPS_REPEAT_INTRA_FRAMES;

    // Ultra fast Encoder Quality preset
    configParam.encPreset = NVMEDIA_ENC_PRESET_UHP;

    // CBR, I P only
    configParam.rcParams.rateControlMode = NVMEDIA_ENCODE_PARAMS_RC_CBR;
    configParam.rcParams.params.cbr_minqp.averageBitRate = m_encOption.uAverageBitrate;
    // Set to 0 to automatically determine the right VBV buffer size and init delay
    configParam.rcParams.params.cbr_minqp.vbvBufferSize = 0;
    configParam.rcParams.params.cbr_minqp.vbvInitialDelay = 0;
    configParam.rcParams.numBFrames = 0;

    m_upNvMIEP.reset(
        NvMediaIEPCreate(NVMEDIA_IMAGE_ENCODE_HEVC,                                        // codec
                         &initParam,                                                       // init params
                         bufAttrList,                                                      // reconciled attr list
                         m_encOption.uMaxOutputBuffer,                                     // maxOutputBuffering
                         static_cast<NvMediaEncoderInstanceId>(m_encOption.uInstanceId))); // encoder instance

    PCHK_PTR_AND_RETURN(m_upNvMIEP, "NvMediaIEPCreate");

    auto nvmediaStatus = NvMediaIEPSetConfiguration(m_upNvMIEP.get(), &configParam);
    PCHK_NVMSTATUS_AND_RETURN(nvmediaStatus, "NvMediaIEPSetConfiguration failed");

    return NvError_Success;
}
#endif // !NV_IS_SAFETY

NvError CEncModule::InitEncoder(NvSciBufAttrList bufAttrList)
{
    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler->RecordInitBeginTime();
    }
#if NV_BUILD_DOS7
    NvMediaDeviceList deviceList{};

    auto nvmediaStatus = NvMediaIEPQueryDevices(&deviceList);
    PCHK_NVMSTATUS_AND_RETURN(nvmediaStatus, "NvMediaIEPQueryDevices");
    LOG_DBG("Enc device list size %d \n", deviceList.deviceListSize);
    if (deviceList.deviceListSize == 0U) {
        PLOG_ERR("No support for Encoder device \n");
        return NvError_NotSupported;
    }
    if (m_encOption.uInstanceId >= deviceList.deviceListSize) {
        PLOG_ERR("No support for requested encoder device index\n");
        return NvError_NotSupported;
    }
    for (auto uIndex = 0U; uIndex < deviceList.deviceListSize; ++uIndex) {
        PLOG_INFO("device num %d numaDomainId %d \n", uIndex, deviceList.deviceList[uIndex].numaDomainId);
        PrintDeviceGid("Encoder", "Primary GPU UUID:", deviceList.deviceList[uIndex].uuid.id);
        PrintDeviceGid("Encoder", "Hardware GPU UUID:", deviceList.deviceList[uIndex].hwGid.id);
    }
#endif

    auto error = GetWidthAndHeight(bufAttrList, m_uWidth, m_uHeight);
    PCHK_ERROR_AND_RETURN(error, "GetWidthAndHeight");
    if (m_encOption.enctype == EncoderType::H264) {
        error = SetupIEPH264(bufAttrList, m_uWidth, m_uHeight);
        PCHK_ERROR_AND_RETURN(error, "IEPConfigH264");
    }
#if !NV_IS_SAFETY
    else if (m_encOption.enctype == EncoderType::H265) {
        error = SetupIEPH265(bufAttrList, m_uWidth, m_uHeight);
        PCHK_ERROR_AND_RETURN(error, "IEPConfigH265");
    }
#endif // !NV_IS_SAFETY
    else {
        PLOG_ERR("Unsupported encoder type\n");
        return NvError_BadParameter;
    }

    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler->RecordInitEndTime();
    }

    return error;
}

// Buffer setup functions
NvError
CEncModule::FillDataBufAttrList(CClientCommon *pClient, PacketElementType userType, NvSciBufAttrList *pBufAttrList)
{
    auto nvmStatus =
        NvMediaIEPFillNvSciBufAttrList(static_cast<NvMediaEncoderInstanceId>(m_encOption.uInstanceId), *pBufAttrList);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMediaIEPFillNvSciBufAttrList");

    NvSciBufAttrValAccessPerm accessPerm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufType bufType = NvSciBufType_Image;
    bool needCpuAccess = true;
    bool isEnableCpuCache = true;

    /* Set all key-value pairs */
    NvSciBufAttrKeyValuePair attributes[] = {
        { NvSciBufGeneralAttrKey_RequiredPerm, &accessPerm, sizeof(accessPerm) },
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &needCpuAccess, sizeof(needCpuAccess) },
        { NvSciBufGeneralAttrKey_EnableCpuCache, &isEnableCpuCache, sizeof(isEnableCpuCache) }
    };

    auto sciErr = NvSciBufAttrListSetAttrs(*pBufAttrList, attributes, ARRAY_SIZE(attributes));
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

    // For opaque element type, color type and memory layout are set explicitly.
    if (userType == PacketElementType::OPAQUE) {
        // Enc currently only support nv12 with bl imagelayout.
        auto error = SetBufAttr(pBufAttrList, "NV12", "BL", m_encOption.uWidth, m_encOption.uHeight);
        PCHK_ERROR_AND_RETURN(error, "SetBufAttr");
    }

    return NvError_Success;
}

NvError CEncModule::FillSyncSignalerAttrList(CClientCommon *pClient,
                                             PacketElementType userType,
                                             NvSciSyncAttrList *pSignalerAttrList)
{
    PCHK_PTR_AND_RETURN_ERR(m_upNvMIEP, "m_upNvMIEP");

    auto nvmStatus = NvMediaIEPFillNvSciSyncAttrList(m_upNvMIEP.get(), *pSignalerAttrList, NVMEDIA_SIGNALER);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "Signaler NvMediaIEPFillNvSciSyncAttrList");

    return NvError_Success;
}

NvError CEncModule::FillSyncWaiterAttrList(CClientCommon *pClient,
                                           PacketElementType userType,
                                           NvSciSyncAttrList *pWaiterAttrList)
{
    PCHK_PTR_AND_RETURN_ERR(m_upNvMIEP, "m_upNvMIEP");

    auto nvmStatus = NvMediaIEPFillNvSciSyncAttrList(m_upNvMIEP.get(), *pWaiterAttrList, NVMEDIA_WAITER);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "Waiter NvMediaIEPFillNvSciSyncAttrList");

    return NvError_Success;
}

NvError CEncModule::RegisterBufObj(CClientCommon *pClient,
                                   PacketElementType userType,
                                   uint32_t uPacketIndex,
                                   NvSciBufObj bufObj)
{
    PCHK_PTR_AND_RETURN_ERR(m_upNvMIEP, "m_upNvMIEP");

    auto error = CBaseModule::RegisterBufObj(pClient, userType, uPacketIndex, bufObj);
    PCHK_ERROR_AND_RETURN(error, "CBaseModule::RegisterBufObj");

    NvMediaStatus nvmStatus = NvMediaIEPRegisterNvSciBufObj(m_upNvMIEP.get(), bufObj);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMediaIEPRegisterNvSciBufObj");

    return NvError_Success;
}

NvError CEncModule::UnregisterBufObj(NvSciBufObj bufObj)
{
    PCHK_PTR_AND_RETURN_ERR(m_upNvMIEP, "m_upNvMIEP");

    auto nvmStatus = NvMediaIEPUnregisterNvSciBufObj(m_upNvMIEP.get(), bufObj);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMediaIEPUnregisterNvSciBufObj");

    return NvError_Success;
}

NvError
CEncModule::RegisterSignalSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj signalSyncObj)
{
    PCHK_PTR_AND_RETURN_ERR(m_upNvMIEP, "m_upNvMIEP");

    CBaseModule::RegisterSignalSyncObj(pClient, userType, signalSyncObj);

    auto nvmStatus = NvMediaIEPRegisterNvSciSyncObj(m_upNvMIEP.get(), NVMEDIA_EOFSYNCOBJ, signalSyncObj);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMediaIEPRegisterNvSciSyncObj for EOF");

    return NvError_Success;
}

NvError
CEncModule::RegisterWaiterSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj waiterSyncObj)
{
    PCHK_PTR_AND_RETURN_ERR(m_upNvMIEP, "m_upNvMIEP");

    CBaseModule::RegisterWaiterSyncObj(pClient, userType, waiterSyncObj);

    auto nvmStatus = NvMediaIEPRegisterNvSciSyncObj(m_upNvMIEP.get(), NVMEDIA_PRESYNCOBJ, waiterSyncObj);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMediaIEPRegisterNvSciSyncObj for PRE");

    return NvError_Success;
}

NvError CEncModule::UnregisterSyncObj(NvSciSyncObj syncObj)
{
    PCHK_PTR_AND_RETURN_ERR(m_upNvMIEP, "m_upNvMIEP");

    auto nvmStatus = NvMediaIEPUnregisterNvSciSyncObj(m_upNvMIEP.get(), syncObj);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMediaIEPUnregisterNvSciSyncObj");

    return NvError_Success;
}

const std::string &CEncModule::GetOutputFileName()
{
    if (m_sOutputFileName != "") {
        PLOG_DBG("This EncMoudule's OutputFileName already exist: %s\n", m_sOutputFileName.c_str());
        return m_sOutputFileName;
    }

    std::string suffix = m_spModuleCfg->m_sensorId != INVALID_ID ? std::to_string(m_spModuleCfg->m_sensorId)
                                                                 : std::to_string(m_spModuleCfg->m_moduleId);
    m_sOutputFileName = "multicast_enc" + suffix;
    if (m_encOption.enctype == EncoderType::H264) {
        m_sOutputFileName += ".h264";
    } else if (m_encOption.enctype == EncoderType::H265) {
        m_sOutputFileName += ".h265";
    } else {
        m_sOutputFileName += ".unknown";
    }
    PLOG_DBG("This EncMoudule's OutputName: %s\n", m_sOutputFileName.c_str());

    return m_sOutputFileName;
}

NvError CEncModule::InsertPrefence(CClientCommon *pClient,
                                   PacketElementType userType,
                                   uint32_t uPacketIndex,
                                   NvSciSyncFence *pPrefence)
{
    PCHK_PTR_AND_RETURN_ERR(m_upNvMIEP, "m_upNvMIEP");

    auto nvmStatus = NvMediaIEPInsertPreNvSciSyncFence(m_upNvMIEP.get(), pPrefence);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMediaIEPInsertPreNvSciSyncFence");

    return NvError_Success;
}

NvError CEncModule::SetEofSyncObj(CClientCommon *pClient)
{
    PCHK_PTR_AND_RETURN_ERR(m_upNvMIEP, "m_upNvMIEP");

    auto nvmStatus = NvMediaIEPSetNvSciSyncObjforEOF(m_upNvMIEP.get(), m_signalSyncObj);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMediaIEPSetNvSciSyncObjforEOF");

    return NvError_Success;
}

NvError CEncModule::GetEofSyncFence(CClientCommon *pClient, NvSciSyncFence *pPostfence)
{
    auto nvmStatus = NvMediaIEPGetEOFNvSciSyncFence(m_upNvMIEP.get(), m_signalSyncObj, pPostfence);
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, ": NvMediaIEPGetEOFNvSciSyncFence");

    return NvError_Success;
}

NvError CEncModule::EncodeOneFrame(NvSciBufObj sciBufObj)
{
    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler->RecordSubmissionBeginTime();
    }

    NvMediaEncodePicParamsH264 encodePicParams;
    uint32_t uNumBytes = 0U;
    uint32_t uNumBytesAvailable = 0U;

    //set one frame params, default = 0
    memset(&encodePicParams, 0, sizeof(NvMediaEncodePicParamsH264));

    {
        std::lock_guard<std::mutex> lk(m_FrameMutex);
        encodePicParams.pictureType =
            (m_uFrameNum == DUMP_START_FRAME || m_pMetaData->bTriggerEncodingValid == true)
            ? NVMEDIA_ENCODE_PIC_TYPE_IDR : NVMEDIA_ENCODE_PIC_TYPE_AUTOSELECT;
    }
    encodePicParams.encodePicFlags = NVMEDIA_ENCODE_PIC_FLAG_OUTPUT_SPSPPS;
    encodePicParams.nextBFrames = 0;

#ifdef NVMEDIA_ENCODE_MAX_PII_REGIONS
    if (m_encOption.bDesensitized && m_pMetaData) {
        if (m_pMetaData->uNumROIRegions > NVMEDIA_ENCODE_MAX_PII_REGIONS) {
            PLOG_ERR("Error: ROI num %d exceeds %d\n", m_pMetaData->uNumROIRegions, NVMEDIA_ENCODE_MAX_PII_REGIONS);
            return NvError_BadValue;
        }
        encodePicParams.numPIIRegions = m_pMetaData->uNumROIRegions;
        for (uint32_t i = 0; i < m_pMetaData->uNumROIRegions; i++) {
            if (m_pMetaData->ROIRect[i].x0 > m_pMetaData->ROIRect[i].x1 ||
                m_pMetaData->ROIRect[i].y0 > m_pMetaData->ROIRect[i].y1 || m_pMetaData->ROIRect[i].x1 >= m_uWidth ||
                m_pMetaData->ROIRect[i].y1 >= m_uHeight) {
                PLOG_ERR("Error PII Region [%d %d %d %d], Resolution [%d %d]\n", m_pMetaData->ROIRect[i].x0,
                         m_pMetaData->ROIRect[i].y0, m_pMetaData->ROIRect[i].x1, m_pMetaData->ROIRect[i].y1, m_uWidth,
                         m_uHeight);
                return NvError_BadValue;
            }
            encodePicParams.PIIparams[i].piiRect.x0 = m_pMetaData->ROIRect[i].x0;
            encodePicParams.PIIparams[i].piiRect.y0 = m_pMetaData->ROIRect[i].y0;
            encodePicParams.PIIparams[i].piiRect.x1 = m_pMetaData->ROIRect[i].x1;
            encodePicParams.PIIparams[i].piiRect.y1 = m_pMetaData->ROIRect[i].y1;
            PLOG_DBG("PII Region [%d %d %d %d], Resolution [%d %d]\n", m_pMetaData->ROIRect[i].x0,
                     m_pMetaData->ROIRect[i].y0, m_pMetaData->ROIRect[i].x1, m_pMetaData->ROIRect[i].y1, m_uWidth,
                     m_uHeight);
        }
    }
#endif

    auto nvmStatus = NvMediaIEPFeedFrame(m_upNvMIEP.get(),                                                // *encoder
                                         sciBufObj,                                                       // *frame
                                         &encodePicParams,                                                // parameter
                                         static_cast<NvMediaEncoderInstanceId>(m_encOption.uInstanceId)); // instance id
    PCHK_NVMSTATUS_AND_RETURN(nvmStatus, "NvMediaIEPFeedFrame");

    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler->RecordSubmissionEndTime();
        m_upProfiler->RecordExecutionBeginTime();
    }

    bool bEncodeFrameDone = false;
    while (!bEncodeFrameDone) {
        NvMediaBitstreamBuffer bitstreams = { 0 };
        uNumBytesAvailable = 0U;
        uNumBytes = 0U;

        nvmStatus = NvMediaIEPBitsAvailable(m_upNvMIEP.get(), &uNumBytesAvailable,
                                            NVMEDIA_ENCODE_BLOCKING_TYPE_IF_PENDING, NVMEDIA_ENCODE_TIMEOUT_INFINITE);
        switch (nvmStatus) {
            case NVMEDIA_STATUS_OK:
                if (uNumBytesAvailable > m_uOutputBufCapacity) {
                    m_upOutputBuf.reset(new (std::nothrow) uint8_t[uNumBytesAvailable]);
                    if (!m_upOutputBuf) {
                        PLOG_ERR("Out of memory, uNumBytesAvailable: %u\n", uNumBytesAvailable);
                        return NvError_InsufficientMemory;
                    }
                    m_uOutputBufCapacity = uNumBytesAvailable;
                }
                bitstreams = { m_upOutputBuf.get(), 0, uNumBytesAvailable };
                nvmStatus = NvMediaIEPGetBits(m_upNvMIEP.get(), &uNumBytes, 1U, &bitstreams, nullptr);
                if (nvmStatus != NVMEDIA_STATUS_OK && nvmStatus != NVMEDIA_STATUS_NONE_PENDING) {
                    PLOG_ERR("Error getting encoded bits\n");
                    return NvError_ResourceError;
                }

                if (uNumBytes != uNumBytesAvailable) {
                    PLOG_ERR("Error-byte counts do not match %d vs. %d\n", uNumBytesAvailable, uNumBytes);
                    return NvError_CountMismatch;
                }
                m_uOutputBufValidLen = uNumBytesAvailable;
                bEncodeFrameDone = true;
                break;

            case NVMEDIA_STATUS_PENDING:
                PLOG_ERR("Error - encoded data is pending\n");
                return NvError_ResourceError;

            case NVMEDIA_STATUS_NONE_PENDING:
                PLOG_ERR("Error - no encoded data is pending\n");
                return NvError_ResourceError;

            default:
                PLOG_ERR("Error occured\n");
                return NvError_ResourceError;
        }
    }
    return NvError_Success;
}

// Streaming functions
NvError CEncModule::ProcessPayload(CClientCommon *pClient, uint32_t uPacketIndex)
{
    PCHK_PTR_AND_RETURN_ERR(pClient, "pClient");

    NvSciBufObj *pBufObj = pClient->GetBufObj(uPacketIndex);
    PCHK_PTR_AND_RETURN_ERR(pBufObj, "pBufObj");

    auto error = EncodeOneFrame(*pBufObj);
    PCHK_ERROR_AND_RETURN(error, "EncodeOneFrame");

    return NvError_Success;
}

NvError CEncModule::OnProcessPayloadDone(CClientCommon *pClient, uint32_t uPacketIndex)
{
    NvError error = NvError_Success;

    error = CBaseModule::OnProcessPayloadDone(pClient, uPacketIndex);
    PCHK_ERROR_AND_RETURN(error, "CBaseModule::OnProcessPayloadDone");

    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler->RecordExecutionEndTime();
        m_pMetaData = reinterpret_cast<MetaData *>(pClient->GetMetaPtr(uPacketIndex));
        if (!m_bHasDownstream && m_pMetaData) {
            m_upProfiler->RecordPipelineTime(m_pMetaData->uFrameCaptureStartTSC);
            m_pMetaData = nullptr;
        }
    }

    return error;
}

NvError CEncModule::OnDataBufAttrListRecvd(CClientCommon *pClient, NvSciBufAttrList bufAttrList)
{
    if (m_upNvMIEP == nullptr) {
        auto error = InitEncoder(bufAttrList);
        PCHK_ERROR_AND_RETURN(error, "InitEncoder");
    }

    return NvError_Success;
}

const OptionTable *CEncModule::GetOptionTable() const
{
    return &EncOptionTable;
}

const void *CEncModule::GetOptionBaseAddress() const
{
    return &m_encOption;
}

NvError CEncModule::OnPacketGotten(CClientCommon *pClient, uint32_t uPacketIndex, bool *pHandled)
{
    auto status = CBaseModule::OnPacketGotten(pClient, uPacketIndex, pHandled);
    PCHK_ERROR_AND_RETURN(status, "CBaseModule::OnProcessPayloadDone");
    m_pMetaData = reinterpret_cast<MetaData *>(pClient->GetMetaPtr(uPacketIndex));
    bool bDiscard = (m_pMetaData->bTriggerEncodingValid == true && m_pMetaData->bTriggerEncoding == false);
    if (bDiscard) {
        return NvError_EndOfFile;
    }
    return NvError_Success;
}

bool CEncModule::DumpEnabled()
{
    bool bDumpTrigger = m_pMetaData && m_pMetaData->bTriggerEncodingValid && m_pMetaData->bTriggerEncoding;
    return CBaseModule::DumpEnabled() || bDumpTrigger;
}
