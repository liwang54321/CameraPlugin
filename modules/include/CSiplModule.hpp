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

#ifndef CSiplModule_H
#define CSiplModule_H

#include <vector>
#include <map>

#include "CBaseModule.hpp"
#include "NvSIPLDeviceBlockInfo.hpp"
#include "NvSIPLPipelineMgr.hpp"
#include "CSiplCamera.hpp"

class CSiplModule : public CBaseModule, public ISiplModuleCallback
{
  public:
    /**
    * @brief get frame from SIPLCamera
    * @param pipelineQueues ICP/ISP0/ISP1/ISP2 pipeline queue
    */
    void FrameQueueThread(NvSIPLPipelineQueues *pipelineQueues);

    /**
    * @brief get notification from SIPLCamera
    * @param notificationQueue pipeline notification
    */
    void PipelineQueueThread(INvSIPLNotificationQueue *notificationQueue);

    CSiplModule(std::shared_ptr<CModuleCfg> moduleCfg, IEventListener<CBaseModule> *pListener);
    virtual ~CSiplModule();

    /**
    * @brief call CSIPLCamera to do init
    * @return NvError_Success if successfully, error if SIPL return error
    */
    virtual NvError Init() override;

    /**
    * @brief call CSIPLCamera to do Deinit & release buffer
    * @return NvError_Success if successfully, error if SIPL return error
    */
    virtual void DeInit() override;

    /**
    * @brief call CSIPLCamera to do start
    * @return NvError_Success if successfully, error if SIPL return error
    */
    virtual NvError Start() override;

    /**
    * @brief call CSIPLCamera to stop init
    * @return NvError_Success if successfully, error if SIPL return error
    */
    virtual NvError Stop() override;

    virtual NvError PostStop() override;

    /**
    * @brief handle command
    * @param uCmdId command id
    * @param pParam command parameter
    */
    virtual void OnCommand(uint32_t uCmdId, void *pParam) override;

    /**
    * @brief fill databuf attribute list
    * @param pClient clientCommon, to check is producer or consumer
    * @param userType multi element type
    * @param pBufAttrList output buffer attribute list
    * @return NvError_Success if successfully, error if GetImageAttributes return error
    */
    virtual NvError
    FillDataBufAttrList(CClientCommon *pClient, PacketElementType userType, NvSciBufAttrList *pBufAttrList) override;

    /**
    * @brief fill signal attribute list
    * @param pClient clientCommon, to check is producer or consumer
    * @param userType  no use (need to delete)
    * @param pSignalerAttrList output signal attribute list
    * @return NvError_Success if successfully, error if SIPL fill list return error
    */
    virtual NvError FillSyncSignalerAttrList(CClientCommon *pClient,
                                             PacketElementType userType,
                                             NvSciSyncAttrList *pSignalerAttrList) override;
    /**
    * @brief fill waiter attribute list
    * @param pClient clientCommon, to check is producer or consumer
    * @param userType  no use (need to delete)
    * @param pWaiterAttrList output waiter attribute list
    * @return NvError_Success if successfully, error if SIPL fill list return error
    */
    virtual NvError FillSyncWaiterAttrList(CClientCommon *pClient,
                                           PacketElementType userType,
                                           NvSciSyncAttrList *pWaiterAttrList) override;

    /**
    * @brief call NVSIPLCamera->RegisterImages to register data buffer into NVSIPL
    * @param pClient clientCommon, no use
    * @param userType user type for this buffer
    * @param uPacketIndex packet index
    * @param bufObj buffer obj
    * @return NvError_Success if successfully, error if SIPL return error
    */
    virtual NvError RegisterBufObj(CClientCommon *pClient,
                                   PacketElementType userType,
                                   uint32_t uPacketIndex,
                                   NvSciBufObj bufObj) override;

    /**
    * @brief call NVSIPLCamera->RegisterNvSciSyncObj to register signal into NVSIPL
    * @param pClient clientCommon, no use
    * @param signalSyncObj signal obj
    * @return NvError_Success if successfully, error if SIPL return error
    */
    virtual NvError
    RegisterSignalSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj signalSyncObj) override;

    /**
    * @brief call NVSIPLCamera->RegisterNvSciSyncObj to register waiter into NVSIPL
    * @param pClient clientCommon, no use
    * @param waiterSyncObj waiter obj
    * @return NvError_Success if successfully, error if SIPL return error
    */
    virtual NvError
    RegisterWaiterSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj waiterSyncObj) override;

    /**
    * @brief call NVSIPLCamera->AddNvSciSyncPrefence to add packet's prefence into NVSIPL
    * @param pClient clientCommon, no use
    * @param uPacketIndex uPacketIndex
    * @param prefence prefence
    * @return NvError_Success if successfully, error if SIPL return error
    */
    virtual NvError InsertPrefence(CClientCommon *pClient,
                                   PacketElementType userType,
                                   uint32_t uPacketIndex,
                                   NvSciSyncFence *prefence) override;

    /**
    * @brief call NVSIPLCamera->GetEOFNvSciSyncFence to get packet's postfence from NVSIPL
    * @param pClient clientCommon, no use
    * @param uPacketIndex uPacketIndex
    * @param prefence prefence
    * @return NvError_Success if successfully, error if SIPL return error
    */
    virtual NvError ProcessPayload(CClientCommon *pClient, uint32_t uPacketIndex) override;

    /**
    * @brief call NVSIPLCamera->Release to release packet to NVSIPL
    * @param pClient clientCommon, no use
    * @param uPacketIndex uPacketIndex
    * @param pHandled, whether or not the event has been handled.
    * @return NvError_Success if successfully, error if SIPL return error
    */
    virtual NvError OnPacketGotten(CClientCommon *pClient, uint32_t uPacketIndex, bool *pHandled = nullptr) override;

    /**
    * @brief get mininum sipl packet count
    * @param pClient clientCommon, no use
    * @param pPacketCount packet count, if ISP0, return 0, if ICP, return 1
    * @return NvError_Success if successfully, error if SIPL return error
    */
    virtual NvError GetMinPacketCount(CClientCommon *pClient, uint32_t *pPacketCount) override;

    /**
    * @brief handle pipeline event
    * @param oNotificationData notification
    */
    void OnPipelineEvent(NvSIPLPipelineNotifier::NotificationData &oNotificationData);

    /**
    * @brief handle buffers from CSIPLCamera
    * @param siplBuffers pair to store output type and INvSIPLBuffer
    * @return NvError_Success if successfully, error if SIPL return error
    */
    NvError OnFrameAvailable(
        std::vector<std::pair<INvSIPLClient::ConsumerDesc::OutputType, INvSIPLClient::INvSIPLBuffer *>> siplBuffers);

    /**
    * @brief received event error from client
    * @param pClient client object
    * @param event event error of client
    */
    void OnEvent(CClientCommon *pClient, EventStatus event) override;

    /**
    * @brief report error to upper layer
    * @param uCameraId cameraId or devick block id
    * @param uErrorId errorId
    */
    void OnError(uint32_t uCameraId, uint32_t uErrorId);

    std::atomic<bool> m_bQuit{ false };

  private:
    struct elementInfo
    {
        /* element index */
        uint32_t uElementIndex = 0U;
        /* packet index */
        uint32_t uPacketIndex = 0U;
    };

    /* For multi element post */
    std::map<NvSciBufObj, elementInfo> m_elementInfoMaps;

    /* store buffer from sipl (AddRef & Release)*/
    std::map<uint32_t, std::map<INvSIPLClient::ConsumerDesc::OutputType, INvSIPLClient::INvSIPLBuffer *>>
        m_packetBufMap;
    /* store buffer obj */
    std::map<INvSIPLClient::ConsumerDesc::OutputType, std::vector<NvSciBufObj>> m_siplBufObjMap;
    std::map<INvSIPLClient::ConsumerDesc::OutputType, NvSciSyncObj> m_siplSignalSyncObjMap;
    std::map<INvSIPLClient::ConsumerDesc::OutputType, NvSciSyncObj> m_siplWaiterSyncObjMap;

    NvError InitSiplCamera(std::shared_ptr<CSiplCamera> &spCamera);
    NvError FreeResources();

    /**
    * @brief override ISP attribute
    * @param userType input user type
    * @param bufAttrList output attribute list
    * @return MULTICAST_ERROR_GENERAL_OK if successfully,
    *         MULTICAST_ERROR_SIPLMODULE_BAD_ARGUMENT if can not find a match output type
    */
    NvError OverrideIspAttributes(PacketElementType userType, NvSciBufAttrList &bufAttrList);

    /**
    * @brief get output type from user type
    * @param userType input user type
    * @param outputType output SIPL output type(ICP/ISP0/ISP1/ISP2)
    * @return MULTICAST_ERROR_GENERAL_OK if successfully,
    *         MULTICAST_ERROR_SIPLMODULE_BAD_ARGUMENT if can not find a match output type
    */
    NvError MapElemTypeToOutputType(PacketElementType userType, INvSIPLClient::ConsumerDesc::OutputType &outputType);

    /**
     * @brief Configure the camera frame rate
     *
     * @param bLowPowerMode true if low power mode, false otherwise
     *
     * @return NvError_Success if successfully, error if SIPL return error
     */
    NvError ConfigureCameraFrameRate(bool bLowPowerMode);

    /**
     * @brief Get the Current TSC Ticks counter
     *
     * @return (uint64_t) TSC Ticks counter
     */
    uint64_t GetCurrentTSCTicks();

    /**
     * @brief Program the Fsync group for given start time
     *
     * @param[in] fsyncGroupId          Fsync group ID
     * @param[in] startTimeTSCTicks     Start time in TSC ticks
     *
     * @retval true                     If the Fsync group is successfully programmed
     * @retval false                    If the Fsync group is not programmed
     */
    bool ProgramFsync(uint32_t const fsyncGroupId, uint64_t const startTimeTSCTicks);

    /**
     * @brief Convert microseconds to TSC Ticks
     *
     * @param microseconds            Time in microseconds
     *
     * @retval (uint64_t)             Ticks in TSC counter
     */
    constexpr uint64_t UsToTicks(uint64_t const microseconds);

    /* Frame completion Queue (ICP/ISP0/ISP1/ISP2)*/
    std::unique_ptr<std::thread> m_upFrameThread;
    /* Pipeline notification Queue*/
    NvSIPLPipelineQueues m_pipelineQueues{};
    /* Pipeline notification thread */
    std::unique_ptr<std::thread> m_upNotificationThread;
    std::shared_ptr<CSiplCamera> m_spCamera;
    CAppCfg *m_pAppCfg;
    EventStatus m_EventStatus{ EventStatus::ERROR };
    bool m_bStarted;
    uint64_t m_uMultiplier{ 0U };
};

#endif
