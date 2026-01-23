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

// STL Headers
#include <unistd.h>
#include <cstring>
#include <vector>
#include <thread>
#include <inttypes.h>

// NvSIPL Headers
#include "NvSIPLVersion.hpp" // Version
#include "NvSIPLTrace.hpp"   // Trace
#if !NV_IS_SAFETY
#include "NvSIPLQuery.hpp"      // Query
#include "NvSIPLQueryTrace.hpp" // Query Trace
#endif
#include "NvSIPLCommon.hpp"      // Common
#include "NvSIPLCamera.hpp"      // Camera
#include "NvSIPLPipelineMgr.hpp" // Pipeline manager
#include "NvSIPLPlatformCfg.hpp" // Platform Cfg
#include "NvSIPLClient.hpp"      // Client

// Local Headers
#include "CUtils.hpp"
#include "CSiplCamera.hpp"

// device block notification queue timeout US
constexpr unsigned long kEventQueueTimeoutUs = 1000000U;

/**
* @brief singleton GetInstance
* @return CSIPLCamera instance
*/
std::shared_ptr<CSiplCamera> CSiplCamera::GetInstance(CAppCfg *pAppCfg)
{
    static std::weak_ptr<CSiplCamera> s_wpCameraInstance;
    static std::mutex s_instanceMutex;

    /* Start multicast several times with different appcfg (integration test)*/
    std::shared_ptr<CSiplCamera> siplCamera = nullptr;
    std::lock_guard<std::mutex> lock{ s_instanceMutex };
    siplCamera = s_wpCameraInstance.lock();
    if (!siplCamera) {
        siplCamera = std::make_shared<CSiplCamera>(pAppCfg);
        s_wpCameraInstance = siplCamera;
        if (siplCamera->PreInit() != NvError_Success) {
            LOG_ERR("PreInit Failed\n");
            return nullptr;
        }
    }
    return siplCamera;
}

NvError CSiplCamera::GetPipelineCfg(SensorInfo &sensorInfo, NvSIPLPipelineConfiguration &pipeCfg)
{
#if !NV_IS_SAFETY
    if (sensorInfo.vcInfo.inputFormat == NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV422 || sensorInfo.isTPGEnabled == true) {
#else
    if (sensorInfo.vcInfo.inputFormat == NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV422) {
#endif
        pipeCfg.captureOutputRequested = true;
        pipeCfg.isp0OutputRequested = false;
        pipeCfg.isp1OutputRequested = false;
        pipeCfg.disableSubframe = true;
    } else if (m_bIsMultiElem) {
        pipeCfg.captureOutputRequested = false;
        pipeCfg.isp0OutputRequested = true;
        pipeCfg.isp1OutputRequested = true;
        LOG_MSG("Enable ISP1 output for multiple elements\n");
    } else {
        pipeCfg.captureOutputRequested = false;
        pipeCfg.isp0OutputRequested = true;
        pipeCfg.isp1OutputRequested = false;
    }
    pipeCfg.isp2OutputRequested = false;
    // pipeCfg.disableSubframe = true;
    return NvError_Success;
}

CSiplCamera::CSiplCamera(CAppCfg *pAppCfg)
{
    if (pAppCfg == nullptr) {
        return;
    }
    m_vPlatformCfgs = pAppCfg->GetPlatformCfgs();
    m_bIsMultiElem = pAppCfg->IsMultiElementsEnabled();
    m_vPipelineQueues.resize(MAX_SENSORS_PER_PLATFORM);
    m_vDevblkQueues.resize(m_vPlatformCfgs.size());
    m_sNitoPath = pAppCfg->GetNitoFolderPath();
}

CSiplCamera::~CSiplCamera()
{
    LOG_INFO("~CSiplCamera\n");

    for (auto &camera : m_vupCameras) {
        camera->Deinit();
        camera.reset(nullptr);
    }
    m_vupCameras.clear();
}

NvError CSiplCamera::RegisterCallback(uint32_t uSensorId, ISiplModuleCallback *pSiplModuleCallback)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (pSiplModuleCallback == nullptr) {
        return NvError_BadParameter;
    }

    m_siplModuleCallbackMap[uSensorId] = pSiplModuleCallback;
    return NvError_Success;
}

bool CSiplCamera::CheckSIPLVersion()
{
    NvSIPLVersion oVer{};
    NvSIPLGetVersion(oVer);

    LOG_INFO("NvSIPL library version: %u.%u.%u\n", oVer.uMajor, oVer.uMinor, oVer.uPatch);
    LOG_INFO("NVSIPL header version: %u %u %u\n", NVSIPL_MAJOR_VER, NVSIPL_MINOR_VER, NVSIPL_PATCH_VER);
    if (oVer.uMajor != NVSIPL_MAJOR_VER || oVer.uMinor != NVSIPL_MINOR_VER || oVer.uPatch != NVSIPL_PATCH_VER) {
        LOG_ERR("NvSIPL library and header version mismatch\n");
        return false;
    }
    return true;
}

NvError CSiplCamera::GetPipelineQueues(uint32_t uSensorId, NvSIPLPipelineQueues &pipelineQueues)
{
    pipelineQueues = m_vPipelineQueues[uSensorId];
    LOG_INFO("uSensorId %" PRIu32 ", pipelineQueues %p, this %p\n", uSensorId, &pipelineQueues, this);
    return NvError_Success;
}

NvError CSiplCamera::PreInit()
{
    /* Setup once */
    if (CheckSIPLVersion() == false) {
        return NvError_BadParameter;
    }

    int cameraIdx = 0;
    for (auto &platcfg : m_vPlatformCfgs) {
        NvError error = UpdatePlatformCfgPerBoardModel(&platcfg);
        CHK_ERROR_AND_RETURN(error, "UpdatePlatformCfgPerBoardModel");

        /* Get NvSIPLCamera instance for each platform config */
        auto camera = INvSIPLCamera::GetInstance();

        /* camera <--> platform config 1:1 */
        error = toNvError(camera->SetPlatformCfg(&platcfg, m_vDevblkQueues[cameraIdx++]));
        CHK_ERROR_AND_RETURN(error, "INvSIPLCamera::SetPlatformCfg");

        /* create notifier thread for each devblk in platofmr config*/
        for (uint32_t i = 0; i < platcfg.numDeviceBlocks; i++) {
            auto &devblk = platcfg.deviceBlockList[i];
            for (uint32_t m = 0; m < devblk.numCameraModules; m++) {
                auto &cameraModule = devblk.cameraModuleInfoList[m];
                NvSIPLPipelineConfiguration pipelineCfg;
                GetPipelineCfg(cameraModule.sensorInfo, pipelineCfg);
                LOG_INFO("[[SIPLCamera]] SetPipelineCfg sensor %d format %d\n", cameraModule.sensorInfo.id,
                         (int)cameraModule.sensorInfo.vcInfo.inputFormat);
                error = toNvError(camera->SetPipelineCfg(cameraModule.sensorInfo.id, pipelineCfg,
                                                         m_vPipelineQueues[cameraModule.sensorInfo.id]));
                CHK_ERROR_AND_RETURN(error, "INvSIPLCamera::SetPlatformCfg");
                m_uCameraIdxMap[cameraModule.sensorInfo.id] = m_vupCameras.size();
            }
        }
        m_vupCameras.emplace_back(std::move(camera));
    }
    return NvError_Success;
}

NvError CSiplCamera::Init(uint32_t uSensorId)
{
    uint32_t uCameraIdx = m_uCameraIdxMap[uSensorId];

    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_uCameraInitRefCountMap[uCameraIdx]++ == 0) {
        auto &upCamera = GetNvSIPLCamera(uSensorId);
        CHK_PTR_AND_RETURN_BADARG(upCamera, "GetNvSIPLCamera");

        NvError error = toNvError(upCamera->Init());
        CHK_ERROR_AND_RETURN(error, "INvSIPLCamera::Init");
    }

    return NvError_Success;
}

NvError CSiplCamera::CreateDevblkNotifierThread(INvSIPLCamera *pCamera,
                                                const PlatformCfg &platcfg,
                                                NvSIPLDeviceBlockQueues &devblkQueue,
                                                ISiplModuleCallback *pModuleCallback)
{
    for (uint32_t i = 0; i < platcfg.numDeviceBlocks; i++) {
        auto upDevblkNotifierHandler = std::unique_ptr<CDevblkNotifierHandler>(new CDevblkNotifierHandler());
        NvError error = upDevblkNotifierHandler->Init(pCamera, i, pModuleCallback, platcfg.deviceBlockList[i],
                                                      devblkQueue.notificationQueue[i]);
        CHK_ERROR_AND_RETURN(error, "Device Block Notification Handler Init");
        m_vupDevblkNotifierHandlers.push_back(std::move(upDevblkNotifierHandler));
    }
    return NvError_Success;
}

NvError CSiplCamera::RegisterAutoControlPlugin(INvSIPLCamera *camera, const PlatformCfg &platcfg)
{
    for (uint32_t i = 0; i < platcfg.numDeviceBlocks; i++) {
        /* Register Nito file for each camera module in device block */
        auto &devblk = platcfg.deviceBlockList[i];
        for (uint32_t m = 0; m < devblk.numCameraModules; m++) {
            auto &cameraModule = devblk.cameraModuleInfoList[m];
#if !NV_IS_SAFETY
            if (cameraModule.sensorInfo.vcInfo.inputFormat != NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV422 &&
                cameraModule.sensorInfo.isTPGEnabled == false) {
#else
            if (cameraModule.sensorInfo.vcInfo.inputFormat != NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV422) {
#endif
                std::vector<uint8_t> blob;
                NvError error = LoadNITOFile(m_sNitoPath, cameraModule.name, blob);
                CHK_ERROR_AND_RETURN(error, "LoadNITOFile");
                error =
                    toNvError(camera->RegisterAutoControlPlugin(cameraModule.sensorInfo.id, NV_PLUGIN, nullptr, blob));
                CHK_ERROR_AND_RETURN(error, "INvSIPLCamera::RegisterAutoControlPlugin");
            }
        }
    }
    return NvError_Success;
}

NvError CSiplCamera::Start(uint32_t uSensorId)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    auto &upCamera = GetNvSIPLCamera(uSensorId);
    CHK_PTR_AND_RETURN_BADARG(upCamera, "GetNvSIPLCamera");
    if (IsReadyToStart(uSensorId)) {
        int32_t cameraIndex = GetCameraIndex(uSensorId);
        if (cameraIndex >= 0L && static_cast<std::size_t>(cameraIndex) < m_vupCameras.size()) {
            NvError error =
                CreateDevblkNotifierThread(m_vupCameras[cameraIndex].get(), m_vPlatformCfgs[cameraIndex],
                                           m_vDevblkQueues[cameraIndex], m_siplModuleCallbackMap[uSensorId]);
            CHK_ERROR_AND_RETURN(error, "CreateDevblkNotifierThread");
            error = RegisterAutoControlPlugin(m_vupCameras[cameraIndex].get(), m_vPlatformCfgs[cameraIndex]);
            CHK_ERROR_AND_RETURN(error, "RegisterAutoControlPlugin");

            LOG_INFO("[[SIPLCamera]] sensor %d Start this %p\n", uSensorId, this);
            error = toNvError(upCamera->Start());
            CHK_ERROR_AND_RETURN(error, "INvSIPLCamera::Start");
        } else {
            LOG_ERR("Invalid camera index %d", cameraIndex);
            return NvError_BadValue;
        }
    }
    return NvError_Success;
}

NvError CSiplCamera::Stop(uint32_t uSensorId)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    auto &upCamera = GetNvSIPLCamera(uSensorId);
    CHK_PTR_AND_RETURN_BADARG(upCamera, "GetNvSIPLCamera");
    if (IsReadyToStop(uSensorId)) {
        NvError error = toNvError(upCamera->Stop());
        CHK_ERROR_AND_RETURN(error, "INvSIPLCamera::Stop");
        m_vupDevblkNotifierHandlers.clear();
    }
    return NvError_Success;
}

bool CSiplCamera::IsReadyToStart(uint32_t uSensorId)
{
    uint32_t uCameraIdx = m_uCameraIdxMap[uSensorId];
    m_uCameraRefCountMap[uCameraIdx]++;

    uint32_t uMaxCount = 0;
    for (auto it = m_uCameraIdxMap.begin(); it != m_uCameraIdxMap.end(); it++) {
        if (it->second == uCameraIdx) {
            uMaxCount++;
        }
    }

    return (uMaxCount == m_uCameraRefCountMap[uCameraIdx]);
}

bool CSiplCamera::IsReadyToStop(uint32_t uSensorId)
{
    uint32_t uCameraIdx = m_uCameraIdxMap[uSensorId];
    if (m_uCameraRefCountMap.find(uCameraIdx) != m_uCameraRefCountMap.end()) {
        m_uCameraRefCountMap[uCameraIdx]--;
    }
    return (m_uCameraRefCountMap[uCameraIdx] == 0);
}

int32_t CSiplCamera::GetCameraIndex(uint32_t uSensorId)
{
    if (m_uCameraIdxMap.find(uSensorId) == m_uCameraIdxMap.end()) {
        return INVALID_ID;
    }
    return m_uCameraIdxMap[uSensorId];
}

std::unique_ptr<INvSIPLCamera> &CSiplCamera::GetNvSIPLCamera(uint32_t uSensorId)
{
    static std::unique_ptr<INvSIPLCamera> notFound;
    int32_t cameraIndex = GetCameraIndex(uSensorId);
    if (cameraIndex >= 0L && static_cast<std::size_t>(cameraIndex) < m_vupCameras.size()) {
        return m_vupCameras[cameraIndex];
    }
    return notFound;
}

NvError CSiplCamera::GetImageAttributes(uint32_t uSensorId,
                                        INvSIPLClient::ConsumerDesc::OutputType outType,
                                        NvSciBufAttrList &bufAttrList)
{
    // api thread safe : no
    std::lock_guard<std::mutex> lock(m_mutex);
    auto &upCamera = GetNvSIPLCamera(uSensorId);
    CHK_PTR_AND_RETURN_BADARG(upCamera, "GetNvSIPLCamera");
    NvError error = toNvError(upCamera->GetImageAttributes(uSensorId, outType, bufAttrList));
    CHK_ERROR_AND_RETURN(error, "GetImageAttributes");
    return NvError_Success;
}

NvError CSiplCamera::RegisterImages(uint32_t uSensorId,
                                    INvSIPLClient::ConsumerDesc::OutputType outType,
                                    const std::vector<NvSciBufObj> &vBufObjs)
{
    // api thread safe : no
    std::lock_guard<std::mutex> lock(m_mutex);
    auto &upCamera = GetNvSIPLCamera(uSensorId);
    CHK_PTR_AND_RETURN_BADARG(upCamera, "GetNvSIPLCamera");
    NvError error = toNvError(upCamera->RegisterImages(uSensorId, outType, vBufObjs));
    CHK_ERROR_AND_RETURN(error, "RegisterImages");
    return NvError_Success;
}

NvError CSiplCamera::RegisterSignalSyncObj(uint32_t uSensorId,
                                           INvSIPLClient::ConsumerDesc::OutputType outType,
                                           NvSiplNvSciSyncObjType syncType,
                                           NvSciSyncObj signalSyncObj)
{
    // api thread safe : no
    std::lock_guard<std::mutex> lock(m_mutex);
    auto &upCamera = GetNvSIPLCamera(uSensorId);
    CHK_PTR_AND_RETURN_BADARG(upCamera, "GetNvSIPLCamera");
    NvError error = toNvError(upCamera->RegisterNvSciSyncObj(uSensorId, outType, syncType, signalSyncObj));
    CHK_ERROR_AND_RETURN(error, "RegisterNvSciSyncObj");
    return NvError_Success;
}

NvError CSiplCamera::RegisterWaiterSyncObj(uint32_t uSensorId,
                                           INvSIPLClient::ConsumerDesc::OutputType outType,
                                           NvSiplNvSciSyncObjType syncType,
                                           NvSciSyncObj waiterSyncObj)
{
    // api thread safe : no
    std::lock_guard<std::mutex> lock(m_mutex);
    auto &upCamera = GetNvSIPLCamera(uSensorId);
    CHK_PTR_AND_RETURN_BADARG(upCamera, "GetNvSIPLCamera");
    NvError error = toNvError(upCamera->RegisterNvSciSyncObj(uSensorId, outType, syncType, waiterSyncObj));
    CHK_ERROR_AND_RETURN(error, "RegisterNvSciSyncObj");
    return NvError_Success;
}

NvError CSiplCamera::FillSyncSignalerAttrList(uint32_t uSensorId,
                                              INvSIPLClient::ConsumerDesc::OutputType outType,
                                              NvSciSyncAttrList &signalerAttrList,
                                              NvSiplNvSciSyncClientType syncType)
{
    // api thread safe : no
    std::lock_guard<std::mutex> lock(m_mutex);
    auto &upCamera = GetNvSIPLCamera(uSensorId);
    CHK_PTR_AND_RETURN_BADARG(upCamera, "GetNvSIPLCamera");
    NvError error = toNvError(upCamera->FillNvSciSyncAttrList(uSensorId, outType, signalerAttrList, syncType));
    CHK_ERROR_AND_RETURN(error, "FillNvSciSyncAttrList(signal)");
    return NvError_Success;
}

NvError CSiplCamera::FillSyncWaiterAttrList(uint32_t uSensorId,
                                            INvSIPLClient::ConsumerDesc::OutputType outType,
                                            NvSciSyncAttrList &waiterAttrList,
                                            NvSiplNvSciSyncClientType syncType)
{
    // api thread safe : no
    std::lock_guard<std::mutex> lock(m_mutex);
    auto &upCamera = GetNvSIPLCamera(uSensorId);
    CHK_PTR_AND_RETURN_BADARG(upCamera, "GetNvSIPLCamera");
    NvError error = toNvError(upCamera->FillNvSciSyncAttrList(uSensorId, outType, waiterAttrList, syncType));
    CHK_ERROR_AND_RETURN(error, "FillNvSciSyncAttrList(wait)");
    return NvError_Success;
}

NvError CSiplCamera::UpdatePlatformCfgPerBoardModel(PlatformCfg *platformCfg)
{
    CHK_PTR_AND_RETURN_BADARG(platformCfg, "platformCfg");

    /**
     * GPIO power control (GPIO7) is required for Drive Orin (P3663) but not
     * Firespray (P3710). GPIO0 is used for checking Error error.
     * If using another platform (something customer-specific, for example)
     * the GPIO field may need to be modified
     * */
    bool bIsP3663 = false;
    std::vector<uint32_t> gpios;
    NvError error = CheckSKU("3663", bIsP3663);
    CHK_ERROR_AND_RETURN(error, "CheckSKU");
    if (bIsP3663) {
        gpios = { 7 };
        CHK_PTR_AND_RETURN_BADARG(platformCfg->deviceBlockList, "deviceBlockList");
        platformCfg->deviceBlockList[0].gpios = std::move(gpios);
    }

#if NVMEDIA_QNX
    if (bIsP3663) {
        /**
     * Bug 3951727: GPIO needs to be updated. Following values are taken from
     * Device tree files for QNX for P3663 and P3710.
     **/
        gpios = { 0, 1, 7 }; // For QNX in P3663
    } else {
        gpios = { 0, 1 }; // For QNX in P3710 and other boards
    }
    CHK_PTR_AND_RETURN_BADARG(platformCfg->deviceBlockList, "deviceBlockList");
    platformCfg->deviceBlockList[0].gpios = std::move(gpios);
#endif

    return error;
}

NvError CSiplCamera::CDevblkNotifierHandler::Init(INvSIPLCamera *pCamera,
                                                  uint32_t uDevblkIdx,
                                                  ISiplModuleCallback *pModuleCallback,
                                                  const DeviceBlockInfo &devblkInfo,
                                                  INvSIPLNotificationQueue *pNotificationQueue)
{
    if (pNotificationQueue == nullptr) {
        LOG_ERR("Invalid Notification Queue\n");
        return NvError_BadParameter;
    }
    if (pModuleCallback == nullptr) {
        LOG_ERR("No callback registered, exit devblk notifier handler thread\n");
        return NvError_BadParameter;
    }
    m_uDevBlkIndex = uDevblkIdx;
    m_deviceBlockInfo = devblkInfo;
    m_pNotificationQueue = pNotificationQueue;
    m_pCamera = pCamera;
    m_pSiplModuleCallback = pModuleCallback;

    NvError error = toNvError(m_pCamera->GetMaxErrorSize(m_uDevBlkIndex, m_errorSize));
    if (error != NvError_Success) {
        LOG_ERR("DeviceBlock: %u, GetMaxErrorSize failed\n", m_uDevBlkIndex);
        return error;
    }

    if (m_errorSize > 0) {
        m_deserializerErrorInfo.upErrorBuffer.reset(new uint8_t[m_errorSize]);
        m_deserializerErrorInfo.bufferSize = m_errorSize;

        m_serializerErrorInfo.upErrorBuffer.reset(new uint8_t[m_errorSize]);
        m_serializerErrorInfo.bufferSize = m_errorSize;

        m_sensorErrorInfo.upErrorBuffer.reset(new uint8_t[m_errorSize]);
        m_sensorErrorInfo.bufferSize = m_errorSize;
    }

    m_upDevblkThread.reset(
        new std::thread(&CSiplCamera::CDevblkNotifierHandler::DeviveBlockQueueThread, this, m_pNotificationQueue));
    return NvError_Success;
}

void CSiplCamera::CDevblkNotifierHandler::Deinit()
{
    m_bQuit.store(true);
    if (m_upDevblkThread != nullptr && m_upDevblkThread->joinable()) {
        m_upDevblkThread->join();
        m_upDevblkThread.reset();
    }
}

CSiplCamera::CDevblkNotifierHandler::~CDevblkNotifierHandler()
{
    Deinit();
}

void CSiplCamera::CDevblkNotifierHandler::DeviveBlockQueueThread(INvSIPLNotificationQueue *pNotificationQueue)
{
    NvError error = NvError_Success;
    NvSIPLPipelineNotifier::NotificationData notificationData;

    if (pNotificationQueue == nullptr) {
        LOG_ERR("Invalid thread data\n");
        return;
    }

    pthread_setname_np(pthread_self(), "DevblkNotifier");
    m_bQuit.store(false);

    // OnDevblkEvent(notificationData); // test onError in init stage
    while (!m_bQuit.load()) {
        error = toNvError(pNotificationQueue->Get(notificationData, kEventQueueTimeoutUs));
        if (error == NvError_Success) {
            OnDevblkEvent(notificationData);
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

void CSiplCamera::CDevblkNotifierHandler::HandleDeserializerError()
{
    bool bIsRemoteError = false;
    uint8_t uLinkErrorMask = 0;

    /* Get detailed error information (if error size is non-zero) and
        * information about remote error and link error. */
    NvError error = toNvError(m_pCamera->GetDeserializerErrorInfo(
        m_uDevBlkIndex, (m_errorSize > 0) ? &m_deserializerErrorInfo : nullptr, bIsRemoteError, uLinkErrorMask));
    if (error != NvError_Success) {
        LOG_ERR("DeviceBlock: %u, GetDeserializerErrorInfo failed\n", m_uDevBlkIndex);
        m_bInError = true;
        return;
    }

    if ((m_errorSize > 0) && (m_deserializerErrorInfo.sizeWritten != 0)) {
        LOG_ERR("DeviceBlock[%d] Deserializer Error Buffer: ", m_uDevBlkIndex);
        for (uint32_t i = 0; i < m_deserializerErrorInfo.sizeWritten; i++) {
            printf("0x%x ", m_deserializerErrorInfo.upErrorBuffer[i]);
        }
        printf("\n");
        m_bInError = true;
    }

    if (bIsRemoteError) {
        LOG_ERR("DeviceBlock[%d] Deserializer Remote Error: ", m_uDevBlkIndex);
        for (uint32_t i = 0; i < m_deviceBlockInfo.numCameraModules; i++) {
            HandleCameraModuleError(m_deviceBlockInfo.cameraModuleInfoList[i].sensorInfo.id);
        }
    }

    if (uLinkErrorMask != 0U) {
        LOG_ERR("DeviceBlock: %u, Deserializer link error. mask: %u\n", m_uDevBlkIndex, uLinkErrorMask);
        m_bInError = true;
    }
}

void CSiplCamera::CDevblkNotifierHandler::HandleCameraModuleError(uint32_t uSensorId)
{
    if (m_errorSize <= 0) {
        return;
    }

    /* Get detailed error information. */
    NvError error = toNvError(m_pCamera->GetModuleErrorInfo(uSensorId, &m_serializerErrorInfo, &m_sensorErrorInfo));
    if (error != NvError_Success) {
        LOG_ERR("uSensorId: %u, GetModuleErrorInfo failed\n", uSensorId);
        m_bInError = true;
        m_pSiplModuleCallback->OnError(uSensorId, error);
    }

    if (m_serializerErrorInfo.sizeWritten != 0) {
        LOG_ERR("Pipeline[%d] Serializer Error Buffer: ", uSensorId);
        for (uint32_t i = 0; i < m_serializerErrorInfo.sizeWritten; i++) {
            printf("0x%x ", m_serializerErrorInfo.upErrorBuffer[i]);
        }
        printf("\n");

        m_bInError = true;
        m_pSiplModuleCallback->OnError(uSensorId, error);
    }

    if (m_sensorErrorInfo.sizeWritten != 0) {
        LOG_ERR("Pipeline[%d] Sensor Error Buffer: ", uSensorId);
        for (uint32_t i = 0; i < m_sensorErrorInfo.sizeWritten; i++) {
            printf("0x%x ", m_sensorErrorInfo.upErrorBuffer[i]);
        }
        printf("\n");

        m_bInError = true;
        m_pSiplModuleCallback->OnError(uSensorId, error);
    }
}

void CSiplCamera::CDevblkNotifierHandler::OnDevblkEvent(NvSIPLPipelineNotifier::NotificationData &notificationData)
{
    if (m_bIgnoreError) {
        return;
    }

    switch (notificationData.eNotifType) {
        case NvSIPLPipelineNotifier::NOTIF_ERROR_DESERIALIZER_FAILURE:
            LOG_ERR("DeviceBlock: %u, NOTIF_ERROR_DESERIALIZER_FAILURE\n", m_uDevBlkIndex);
            HandleDeserializerError();
            if (m_pSiplModuleCallback) {
                m_pSiplModuleCallback->OnError(m_uDevBlkIndex, NvError_ResourceError);
            }
            break;
        case NvSIPLPipelineNotifier::NOTIF_ERROR_SERIALIZER_FAILURE:
            LOG_ERR("DeviceBlock: %u, NOTIF_ERROR_SERIALIZER_FAILURE\n", m_uDevBlkIndex);
            for (uint32_t i = 0; i < m_deviceBlockInfo.numCameraModules; i++) {
                if ((notificationData.uLinkMask & (1 << (m_deviceBlockInfo.cameraModuleInfoList[i].linkIndex))) != 0) {
                    HandleCameraModuleError(m_deviceBlockInfo.cameraModuleInfoList[i].sensorInfo.id);
                    if (m_pSiplModuleCallback) {
                        m_pSiplModuleCallback->OnError(m_uDevBlkIndex, NvError_ResourceError);
                    }
                }
            }
            break;
        case NvSIPLPipelineNotifier::NOTIF_ERROR_SENSOR_FAILURE:
            LOG_ERR("DeviceBlock: %u, NOTIF_ERROR_SENSOR_FAILURE\n", m_uDevBlkIndex);
            for (uint32_t i = 0; i < m_deviceBlockInfo.numCameraModules; i++) {
                if ((notificationData.uLinkMask & (1 << (m_deviceBlockInfo.cameraModuleInfoList[i].linkIndex))) != 0) {
                    HandleCameraModuleError(m_deviceBlockInfo.cameraModuleInfoList[i].sensorInfo.id);
                    if (m_pSiplModuleCallback) {
                        m_pSiplModuleCallback->OnError(m_uDevBlkIndex, NvError_ResourceError);
                    }
                }
            }
            break;
        case NvSIPLPipelineNotifier::NOTIF_ERROR_INTERNAL_FAILURE:
            LOG_ERR("DeviceBlock: %u, NOTIF_ERROR_INTERNAL_FAILURE\n", m_uDevBlkIndex);
            if (notificationData.uLinkMask != 0U) {
                for (uint32_t i = 0; i < m_deviceBlockInfo.numCameraModules; i++) {
                    if ((notificationData.uLinkMask & (1 << (m_deviceBlockInfo.cameraModuleInfoList[i].linkIndex))) !=
                        0) {
                        HandleCameraModuleError(m_deviceBlockInfo.cameraModuleInfoList[i].sensorInfo.id);
                        if (m_pSiplModuleCallback) {
                            m_pSiplModuleCallback->OnError(m_uDevBlkIndex, NvError_ResourceError);
                        }
                    }
                }
            } else {
                m_bInError = true;
            }
            break;
        default:
            LOG_WARN("DeviceBlock: %u, Unknown/Invalid notification\n", m_uDevBlkIndex);
            if (m_pSiplModuleCallback) {
                m_pSiplModuleCallback->OnError(m_uDevBlkIndex, NvError_ResourceError);
            }
            break;
    }
    return;
}
