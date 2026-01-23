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

#ifndef CSIPLCAMERA_HPP
#define CSIPLCAMERA_HPP

// STL Headers
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <map>
#include <unordered_map>

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
#include "NvSIPLClient.hpp"      // Client

// Local Headers
#include "CUtils.hpp"
#include "CConfig.hpp"

#include "CSiplModuleCallback.hpp"

class CSiplCamera
{
  public:
    explicit CSiplCamera(CAppCfg *pAppCfg);

    ~CSiplCamera();

    /**
    * @brief GetInstance from NVSIPLCamera
    * @param pAppCfg global appConfig
    * @return
    */
    static std::shared_ptr<CSiplCamera> GetInstance(CAppCfg *pAppCfg);

    /**
    * @brief get instance from INvSIPLCamera, SetPlatformCfg, SetPipelineCfg, Init
    * @return NvError_Success if no error, otherwise return error from NvSIPLCamera
    */
    NvError PreInit();
    NvError Init(uint32_t uSensorId);

    /**
    * @brief start NVSIPLCamera
    * @return OK if start successfully
    */
    NvError Start(uint32_t uSensorId);

    /**
    * @brief stop NVSIPLCamera
    * @return OK if stop successfully
    */
    NvError Stop(uint32_t uSensorId);

    /**
    * @brief get autocontrol plugin from appConfig and register it into NVSIPLCamera
    * @param camera camera instance for this platform config
    * @param platformCfg single platform config
    * @return NvError OK if successfully
    *          if Nito Path is invalid
    *          SIPL error if RegisterAutoControlPlugin fail
    */
    NvError RegisterAutoControlPlugin(INvSIPLCamera *camera, const PlatformCfg &platformCfg);

    /**
    * @brief create device block notifier thread
    * @param camera camera instance for this platform config
    * @param platformCfg single platform config
    * @param devblkQueue device block queue
    * @param pModuleCallback module callback to SIPLModule
    * @return NvError OK if successfully
    */
    NvError CreateDevblkNotifierThread(INvSIPLCamera *camera,
                                       const PlatformCfg &platcfg,
                                       NvSIPLDeviceBlockQueues &devblkQueue,
                                       ISiplModuleCallback *pModuleCallback);

    /**
    * @brief get image attribute from NvSIPLCamera
    * @param uSensorId sensor id
    * @param outType output type (ICP/ISP0/ISP1/ISP2)
    * @param bufAttrList buffer attribute list(output)
    * @return NvError OK if successfully
    */
    NvError GetImageAttributes(uint32_t uSensorId,
                               INvSIPLClient::ConsumerDesc::OutputType outType,
                               NvSciBufAttrList &bufAttrList);

    /**
    * @brief register image to NvSIPLCamera
    * @param uSensorId sensor id
    * @param outType output type (ICP/ISP0/ISP1/ISP2)
    * @param bufObjs group of buffer obj (typically 6 packets)
    * @return NvError OK if successfully
    */
    NvError RegisterImages(uint32_t uSensorId,
                           INvSIPLClient::ConsumerDesc::OutputType outType,
                           const std::vector<NvSciBufObj> &bufObjs);

    /**
    * @brief register signal sync obj to NvSIPLCamera
    * @param uSensorId sensor id
    * @param outType output type (ICP/ISP0/ISP1/ISP2)
    * @param syncType sync type (NVSIPL_EOFSYNCOBJ)
    * @param signalSyncObj signal sync obj
    * @return NvError OK if successfully
    */
    NvError RegisterSignalSyncObj(uint32_t uSensorId,
                                  INvSIPLClient::ConsumerDesc::OutputType outType,
                                  NvSiplNvSciSyncObjType syncType,
                                  NvSciSyncObj signalSyncObj);

    /**
    * @brief register signal sync obj to NvSIPLCamera
    * @param uSensorId sensor id
    * @param outType output type (ICP/ISP0/ISP1/ISP2)
    * @param syncType sync type (NVSIPL_PRESYNCOBJ)
    * @param waiterSyncObj wait sync obj
    * @return NvError OK if successfully
    */
    NvError RegisterWaiterSyncObj(uint32_t uSensorId,
                                  INvSIPLClient::ConsumerDesc::OutputType outType,
                                  NvSiplNvSciSyncObjType syncType,
                                  NvSciSyncObj waiterSyncObj);

    /**
    * @brief fill signal attribute list to NvSIPLCamera
    * @param uSensorId sensor id
    * @param outType output type (ICP/ISP0/ISP1/ISP2)
    * @param signalerAttrList signal attribute list
    * @param syncType sync type (SIPL_SIGNALER)
    * @return NvError OK if successfully
    */
    NvError FillSyncSignalerAttrList(uint32_t uSensorId,
                                     INvSIPLClient::ConsumerDesc::OutputType outType,
                                     NvSciSyncAttrList &signalerAttrList,
                                     NvSiplNvSciSyncClientType syncType);

    /**
    * @brief fill signal attribute list to NvSIPLCamera
    * @param uSensorId sensor id
    * @param outType output type (ICP/ISP0/ISP1/ISP2)
    * @param waiterAttrList wait attribute list
    * @param syncType wait type (SIPL_WAITER)
    * @return NvError OK if successfully
    */
    NvError FillSyncWaiterAttrList(uint32_t uSensorId,
                                   INvSIPLClient::ConsumerDesc::OutputType outType,
                                   NvSciSyncAttrList &waiterAttrList,
                                   NvSiplNvSciSyncClientType syncType);

    class CDevblkNotifierHandler : public NvSIPLPipelineNotifier
    {
      public:
        virtual ~CDevblkNotifierHandler();
        uint32_t m_uDevBlkIndex = 0U;

        /**
        * @brief Initializes the Pipeline Notification Handler
        *        GetMaxErrorSize from NvSIPLCamera, start device block thread with notification queue
        * @param camera NvSIPLCamera ptr
        * @param devblkIdx device block index
        * @param pModuleCallback sipl module callback
        * @param devblkInfo device block info
        * @param notificationQueue notification queue
        * @return NvError OK if successfully
        */
        NvError Init(INvSIPLCamera *camera,
                     uint32_t devblkIdx,
                     ISiplModuleCallback *pModuleCallback,
                     const DeviceBlockInfo &devblkInfo,
                     INvSIPLNotificationQueue *notificationQueue);

        /**
        * @brief exit device block thread
        */
        void Deinit();

        /**
        * @brief handle device block event, distinguish error in deserializer, serializer, sensor, internal
        *        call siplmodule callback OnError to notify error to the upper level
        * @param notificationData notification data from notification queue
        */
        void OnDevblkEvent(NotificationData &oNotificationData);

        /**
        * @brief get device block error from NvSIPLCamera
        * @param pipelineQueues ICP/ISP0/ISP1/ISP2 pipeline queue
        */
        void DeviveBlockQueueThread(INvSIPLNotificationQueue *pNotificationQueue);

      private:
        /**
        * @brief handle deserializer error, call by OnDevblkEvent
        */
        void HandleDeserializerError();
        /**
        * @brief handle sensor error, call by OnDevblkEvent
        * @param uSensorId sensor id
        */
        void HandleCameraModuleError(uint32_t uSensorId);

        bool m_bInError = false;
        std::unique_ptr<std::thread> m_upDevblkThread = nullptr;
        INvSIPLNotificationQueue *m_pNotificationQueue = nullptr;
        DeviceBlockInfo m_deviceBlockInfo;
        INvSIPLCamera *m_pCamera = nullptr;
        bool m_bIgnoreError = false;

        size_t m_errorSize = 0;
        SIPLErrorDetails m_deserializerErrorInfo{};
        SIPLErrorDetails m_serializerErrorInfo{};
        SIPLErrorDetails m_sensorErrorInfo{};
        ISiplModuleCallback *m_pSiplModuleCallback;

      public:
        std::atomic<bool> m_bQuit;
    };

    /**
    * @brief provide registered pipeline queue to SIPLModule according to sensor id
    * @param uSensorId sensor id
    * @param pipelineQueues pipeline queue for this sensor id
    */
    NvError GetPipelineQueues(uint32_t uSensorId, NvSIPLPipelineQueues &pipelineQueues);

    /**
    * @brief call by CSIPLModule to register callback for error handling
    * @param uSensorId sensor id
    * @param pSiplModuleCallback sipl module callback
    * @return NvError OK if successfully, BAD_ARGUMENT if callback nullptr
    */
    NvError RegisterCallback(uint32_t uSensorId, ISiplModuleCallback *pSiplModuleCallback);

  private:
    /**
    * @brief call CheckSKU to check if board name is 3663, if so change gpios
    * @return OK if stop successfully
    */
    NvError UpdatePlatformCfgPerBoardModel(PlatformCfg *pPlatformCfg);

    /**
    * @brief get pipeline cfg according to sensor type or multi-element
    * @param sensorInfo input sensor info
    * @param pipeCfg output pipelineCfg
    */
    NvError GetPipelineCfg(SensorInfo &sensorInfo, NvSIPLPipelineConfiguration &pipeCfg);

    /**
    * @brief get SIPLClient consumer OutputType list according to pipelineCfg
    * @param pipeCfg input pipelineCfg
    * @param outputList output outputtype list
    * @return OK if successfully
    */
    NvError GetOutputTypeList(NvSIPLPipelineConfiguration &pipeCfg,
                              std::vector<INvSIPLClient::ConsumerDesc::OutputType> &outputList);

    /**
    * @brief check if SIPL version get from NvSIPLGetVersion same as marco
    * @return OK if successfully
    */
    bool CheckSIPLVersion();

    /**
    * @brief get NvSIPLCamera according to sensor id
    * @param uSensorId sensor id
    * @return NvSIPLCamera ptr, null INvSIPLCamera if not found
    */
    std::unique_ptr<INvSIPLCamera> &GetNvSIPLCamera(uint32_t uSensorId);

    /**
    * @brief get camera index according to sensor id
    * @param uSensorId sensor id
    * @return camera index, INVALID_ID if not found
    */
    int32_t GetCameraIndex(uint32_t uSensorId);

    /**
    * @brief check if all sensor in the same device block called start
    *        check valid camera idx before call it
    * @param uSensorId sensor id
    * @return true to start, false not to start
    */
    bool IsReadyToStart(uint32_t uSensorId);

    /**
    * @brief check if all sensor in the same device block called stop
    *        if yes, stop camera
    *        check valid camera idx before call it
    * @param uSensorId sensor id
    * @return true to stop, false not to stop
    */
    bool IsReadyToStop(uint32_t uSensorId);

  public:
    // multi-instance NVSIPLCamera
    std::vector<std::unique_ptr<INvSIPLCamera>> m_vupCameras;
    std::atomic<bool> m_bQuit{false};

  private:
    bool m_bIgnoreError = false;
    std::string m_sNitoPath;
    bool m_bIsErrorIgnored = false;
    bool m_bIsMultiElem = false;
    // INvSIPLCamera api lock
    std::mutex m_mutex;
    // multi-platformcfg
    std::vector<PlatformCfg> m_vPlatformCfgs;
    std::vector<std::unique_ptr<CDevblkNotifierHandler>> m_vupDevblkNotifierHandlers;
    std::vector<NvSIPLDeviceBlockQueues> m_vDevblkQueues;
    /* sensor id -> camera instance index */
    std::map<uint32_t, uint32_t> m_uCameraIdxMap;
    /* camera instance index -> reference count */
    std::map<uint32_t, uint32_t> m_uCameraRefCountMap;
    std::map<uint32_t, uint32_t> m_uCameraInitRefCountMap;

    std::vector<NvSIPLPipelineQueues> m_vPipelineQueues;
    std::map<uint32_t, ISiplModuleCallback *> m_siplModuleCallbackMap;
};

#endif //CSiplCamera_HPP
