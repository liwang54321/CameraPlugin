/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef CCONFIG_HPP
#define CCONFIG_HPP

#include <unordered_map>
#include <unordered_set>

#include "CControlChannelManager.hpp"
#include "COptionParser.hpp"
#include "CPeerValidator.hpp"
#include "CTimer.hpp"
#include "CUtils.hpp"
#include "Common.hpp"
#include "nverror.h"
#include <atomic>

struct CameraConfig {
    uint32_t id;
    std::string name;
    std::string type;
    std::string rtp_ip;
    uint16_t port;
    std::string dev_device;
    std::string test_file;
    uint32_t width;
    uint32_t height;
    uint8_t fps;
};

struct CameraConfigs {
    std::vector<CameraConfig> config;
};
struct CameraStatus {
    CameraConfig config;
    bool is_open = false;
};

using namespace nvsipl;
using PipelineDescriptor = std::vector<std::vector<std::string>>;
// <pipeline-parameter, pipeline descriptor in string>
using PipelineCfgTable = std::unordered_map<std::string, std::string>;
typedef const char *const (*PipelineTable)[2];

class CChannelCfg;
class CModuleCfg;
class CClientCfg;
class CPipelineElem;
class CIpcEndpointElem;
class CModuleElem;
class CBaseModule;
// clang-format off
static const std::unordered_map<std::string, PipelineElemType> pipelineElemName2TypeMap {
    { "Enc", PipelineElemType::Enc },
    { "Cuda", PipelineElemType::Cuda },
    { "Stitch", PipelineElemType::Stitch },
    { "Display", PipelineElemType::Display },
    { "Nvm2d", PipelineElemType::Nvm2d },
    { "SIPL", PipelineElemType::SIPL },
    { "VirtualSrc", PipelineElemType::VirtualSrc },
    { "VirtualDst", PipelineElemType::VirtualDst },
    { "FileSrc", PipelineElemType::FileSource },
    { "Pva", PipelineElemType::Pva },
    { "IpcSrc", PipelineElemType::IpcSrc },
    { "IpcDst", PipelineElemType::IpcDst },
    { "VulkanSC", PipelineElemType::VulkanSC },
};

static const std::unordered_map<std::string, std::string> ElemNameToDisplayName {
    { "VirtualSrc", "VirtSrc" },
    { "VirtualDst", "VirtDst" },
    { "VulkanSC", "Vulkan" },
    { "FrameDecoder", "FrmDec" },
    { "FrameReader", "FrmRdr" },
    { "PassthroughProducer", "P" },
    { "Producer", "P" },
    { "PassthroughConsumer", "C" },
    { "Consumer", "C" },
};

std::string ShortenName(const std::string &name);

enum class ProfilingMode : uint8_t
{
    PIPELINE,
    FULL
};

struct PipelineGroup {
    std::string sPipelineDescriptor;
    std::string sStaticConfigName;
#if !NV_IS_SAFETY
    std::string sDynamicConfigName; // one channel only have one dynamic config name
    std::vector<uint32_t> vuMasks;
#endif
};

constexpr uint8_t kMaxVideoChannelCount = 16;
class CAppCfg
{
  public:
    CAppCfg();
    ~CAppCfg();

    void Init();
    uint32_t GetVerbosity() const { return m_uVerbosity; }
#if !NV_IS_SAFETY
    std::string &GetDynamicConfigName(uint32_t uChannelId = 0) { return m_vPipelineGroups.at(uChannelId).sDynamicConfigName; }
    const std::vector<uint32_t> &GetMasks(uint32_t uChannelId = 0) const { return m_vPipelineGroups.at(uChannelId).vuMasks; }
#endif
    const std::string &GetStaticConfigName(uint32_t uChannelId = 0) const { return m_vPipelineGroups.at(uChannelId).sStaticConfigName; }
    const std::string &GetNitoFolderPath() const { return m_sNitoFolderPath; }
    void UpdateByPipelineDescStr(const char *pipelineDescriptionString);
    const std::string &GetPipelineDescriptorString(uint32_t uChannelId = 0) const { return m_vPipelineGroups.at(uChannelId).sPipelineDescriptor; }
    CommType GetCommType() const { return m_eCommType; }
    bool IsErrorIgnored() const { return m_bIgnoreError; }
    bool IsVersionShown() const { return m_bShowVersion; }
    bool IsMultiElementsEnabled() const { return m_bEnableMultiElements; }
    bool IsStatusManagerEnabled() const { return m_bEnableStatusManager; }
    bool IsCentralNode() const { return m_bPureSource || m_bCentralNode; }
    bool IsC2CMasterSoc() const { return m_bC2CMasterSoc; }
    uint8_t GetFrameFilter() const { return m_uFrameFilter; }
    uint32_t GetRunDurationSec() const { return m_uRunDurationSec; }
    EncoderType GetEncoderType() const { return m_eEncType; }
    bool IsProfilingEnabled() const { return m_bEnableProfiling; }
    ProfilingMode GetProfilingMode() const { return m_uProfilingMode; }
    uint32_t GetMaxPerfSampleNum() const { return m_uMaxPerfSampleNum; }
    bool NeedSavePerfData() const { return m_bSavePerfData; }
    const std::string &GetPerfDataSaveFolder() const { return m_sPerfDataSaveFolder; };
    std::vector<PlatformCfg> GetPlatformCfgs();
    void ExtractSensorIds(const std::string &platformName, std::vector<int>& outputSensorIds);
    void GetPipelineTable(PipelineTable &pipelineTable, uint32_t &uSize) const;
    PipelineType GetPipelineType() { return m_ePipelineType; }
    NvError CheckPipelineGroups();
    void AddPipelineGroup(PipelineGroup &pipelineGroup);

    bool IsYuvOrTpgSensor(uint32_t uSensorId);
    void PopulateCameraModuleInfo();
    std::vector<CameraModuleInfo> GetCameraModuleInfo();
    PipelineDescriptor ConvertToPipelineDescriptor(const char *pipelineDescriptorString);
    void CreateChannelCfgs(std::vector<std::shared_ptr<CChannelCfg>> &vspChannelCfgs);
    std::shared_ptr<CPeerValidator> GetPeerValidator();
    NvSciSyncModule m_sciSyncModule{ nullptr };
    NvSciBufModule m_sciBufModule{ nullptr };
    std::shared_ptr<CControlChannelManager> m_spControlChannel = { nullptr };
    std::shared_ptr<CTimer> m_spTimer = { nullptr };
    void SetCudaRunningFlag(bool bRunning) { m_bCudaRunning = bRunning; };
    bool IsCudaRunningEnabled() const { return m_bCudaRunning; };

    using CameraCallback =
    std::function<void(uint32_t cameraId, uint64_t timestamp, const uint8_t *payload, size_t size)>;
    void RegisterCameraPlugin(uint32_t cameraId,  CameraCallback cb) {
        cb_[cameraId] = std::move(cb);
    }

    void CallCameraPlugin(uint32_t cameraId, uint64_t timestamp, const uint8_t *payload, size_t size) {
        if(cb_[cameraId]) cb_[cameraId](cameraId, timestamp, payload, size);
    }

    void SetCpuOutPut(bool enable) {
        is_cpu_sink_.store(enable);
    }

    bool IsCputOutput(void) {
        return is_cpu_sink_;
    }

    void SetMask(uint16_t mask) {
        mask_.store(mask);
    }

    uint16_t GetMask() {
        return mask_;
    }

    void RegisterModuleInfo(const std::map<uint8_t, CameraStatus>& camera_configs)
     {
        camera_configs_ = camera_configs;
     }

     std::optional<CameraConfig> GetCameraConfig(uint8_t index) {
        if (camera_configs_.find(index) == camera_configs_.end()) {
            return std::nullopt;
        }
        return camera_configs_[index].config;
     }

    friend class CCmdLineParser;

  private:
    void PopulatePipelineCfgTable();
    bool GetPipelineStrFromTable(PipelineCfgTable &pipelineCfgTable,
                                 const char *pipelineDescriptionString,
                                 std::string &pipelineStr);
    void PopulatePlatformCfgs();

    uint32_t m_uVerbosity = 1u;
    std::string m_sNitoFolderPath;
    EncoderType m_eEncType = EncoderType::H264;
    CommType m_eCommType = CommType::IntraProcess;
    PipelineType m_ePipelineType = PipelineType::NormalPipeline;
    uint8_t m_uFrameFilter = 1U;
    uint32_t m_uRunDurationSec = 0U;
    std::vector<PlatformCfg> m_platformCfgs;
    std::vector<CameraModuleInfo> m_vCameraModules;
    PipelineCfgTable m_pipelineCfgTable;
    std::vector<PipelineGroup> m_vPipelineGroups;
    CameraCallback cb_[kMaxVideoChannelCount];
    std::map<uint8_t, CameraStatus> camera_configs_;
    std::atomic_bool is_cpu_sink_;
    std::atomic_uint16_t mask_;
    // switch flag
    bool m_bIgnoreError = false;
    bool m_bShowVersion = false;
    bool m_bEnableMultiElements = false;
    bool m_bCudaRunning = true;
    bool m_bEnableStatusManager = false;
    bool m_bEnableProfiling = false;
    ProfilingMode m_uProfilingMode = ProfilingMode::PIPELINE;
    bool m_bIsMaxPerfSampleNumSpecified = false;
    uint32_t m_uMaxPerfSampleNum = MAX_NUM_PERF_SAMPLES;
    bool m_bSavePerfData = false;
    bool m_bCentralNode = false;
    bool m_bPureSource = false;
    bool m_bC2CMasterSoc = false;
    std::string m_sPerfDataSaveFolder = "./nvsipl_multicast_perfs";
    std::shared_ptr<CPeerValidator> m_pPeerValidator{ nullptr };
};

class CChannelCfg
{
  public:
    CChannelCfg(CAppCfg *pAppCfg, PipelineDescriptor &&vPipelineDescriptor, const char *pName);
    NvError ExtractOptions(PipelineElemType type, const char *parameters, Options &options);
    std::shared_ptr<CPipelineElem> ParsePipelineStr(const std::string &pipelineStr);
    std::shared_ptr<CModuleCfg> CreateModuleCfg(int sensorId, std::shared_ptr<CModuleElem> &spModuleElem);

    CAppCfg *m_pAppCfg;
    PipelineGroup m_channelPipelineGroup;
    std::string m_sPlatformConfigName;
    PipelineDescriptor m_vPipelineDescriptor;
    std::string m_sName;
    uint32_t m_uChannelId;
    bool m_bIsYuvSensor = false;
};

/* Names for the packet elements, should be 0~N */
enum class PacketElementType : uint32_t
{
    NV12_BL = 0,
    NV12_PL = 1,
    METADATA = 2,
    ABGR8888_PL = 3,
    ICP_RAW = 4,
    OPAQUE = 5,
    DEFAULT = 6
};

typedef struct
{
    PacketElementType userType;
    bool bIsUsed = false;
    bool bHasSibling = false;
    uint32_t uIndex = MAX_NUM_ELEMENTS;
} ElementInfo;

class CPipelineElem
{
  public:
    CPipelineElem(const std::string &name,
                  PipelineElemType type,
                  std::vector<int> &vSensorIds,
                  int elemId,
                  int elemNum,
                  bool isStitching,
                  Options &options)
        : m_sName(name)
        , m_type(type)
        , m_vSensorIds(vSensorIds)
        , m_elemId(elemId)
        , m_elemNum(elemNum)
        , m_isStitching(isStitching)
        , m_options(std::move(options)) {};
    virtual ~CPipelineElem() {};

    bool IsModule() { return static_cast<uint8_t>(m_type) < static_cast<uint8_t>(PipelineElemType::IpcSrc); };
    std::vector<int> &GetSensorIds() { return m_vSensorIds; };

    std::string m_sName;
    PipelineElemType m_type;
    std::vector<int> m_vSensorIds;
    int m_elemId;
    int m_elemNum;
    bool m_isStitching;
    Options m_options;
};

class CIpcEndpointElem : public CPipelineElem
{
  public:
    CIpcEndpointElem(const std::string &name,
                     PipelineElemType type,
                     std::vector<int> &vSensorIds,
                     int elemId,
                     int elemNum,
                     bool isStitching,
                     Options &options)
        : CPipelineElem(name, type, vSensorIds, elemId, elemNum, isStitching, options) {};
    virtual ~CIpcEndpointElem() {};
    const std::string GetChannelStr(int sensorId);
    const std::string GetOppositeChannelStr(int sensorId);
    NvError ParseOptions();
    bool IsLateAttach() { return m_ipcEndPointOption.bIsLate; };
    bool IsC2C() { return m_ipcEndPointOption.bIsC2C; };
    uint32_t GetLimitNum() { return m_ipcEndPointOption.uLimitNum; };

  private:
    struct IpcEndPointOption
    {
        bool bIsLate = false;
        bool bIsC2C = false;
        uint32_t uLimitNum = 0U;
    };
    IpcEndPointOption m_ipcEndPointOption;

  public:
    static const OptionTable m_ipcEndpointOptionTable;
};

class CModuleElem : public CPipelineElem
{
  public:
    CModuleElem(const std::string &name,
                PipelineElemType type,
                std::vector<int> &vSensorIds,
                int elemId,
                int elemNum,
                bool isStitching,
                Options &options)
        : CPipelineElem(name, type, vSensorIds, elemId, elemNum, isStitching, options) {};
    virtual ~CModuleElem() {};
    void AddIpcDst(std::shared_ptr<CIpcEndpointElem> spIpcDst);
    void AddIpcSrc(std::shared_ptr<CIpcEndpointElem> spIpcSrc);
    void AddDownstreamModule(std::shared_ptr<CModuleElem> spMod);
    std::vector<std::shared_ptr<CIpcEndpointElem>> m_vspIpcDsts;
    std::vector<std::shared_ptr<CIpcEndpointElem>> m_vspIpcSrcs;
    std::vector<std::shared_ptr<CModuleElem>> m_vspDownstreamModules;
    std::vector<std::shared_ptr<CBaseModule>> m_vspMods;
};

typedef struct
{
    bool bWaitPrefence = false;
    bool bWaitPostfence = false;
} CpuWaitCfg;

class CModuleCfg
{
  public:
    CModuleCfg() {};
    CModuleCfg(CAppCfg *pAppCfg, const std::string &sName, int sensorId, std::shared_ptr<CModuleElem> &spElem)
        : m_pAppCfg(pAppCfg)
        , m_sName(std::move(sName))
        , m_sensorId(sensorId)
        , m_moduleId(spElem->m_elemId)
        , m_moduleType(static_cast<ModuleType>(spElem->m_type))
        , m_options(spElem->m_options) {};

    std::shared_ptr<CClientCfg> CreateClientCfg(const std::string &sName,
                                                int numConsumers,
                                                uint32_t uLimitNum,
                                                const std::vector<ElementInfo> *pElementInfos,
                                                QueueType queueType);
    const std::vector<ElementInfo> *GetElementInfos(const std::string &sElems);

    CAppCfg *m_pAppCfg = nullptr;
    const std::string m_sName;
    int m_sensorId = INVALID_ID;
    int m_moduleId = INVALID_ID;
    ModuleType m_moduleType = ModuleType::Unknown;
    CpuWaitCfg m_cpuWaitCfg;
    Options m_options;
};

class CClientCfg
{
  public:
    CClientCfg(CAppCfg *pAppCfg,
               const std::string &sName,
               const std::vector<ElementInfo> *pElementInfos,
               int numConsumers,
               uint32_t uLimitNum,
               CpuWaitCfg cpuWaitCfg,
               QueueType queueType)
        : m_pAppCfg(pAppCfg)
        , m_sName(std::move(sName))
        , m_pElementInfos(pElementInfos)
        , m_numConsumers(numConsumers)
        , m_uLimitNum(uLimitNum)
        , m_cpuWaitCfg(cpuWaitCfg)
        , m_queueType(queueType)
    {
    }

    CAppCfg *m_pAppCfg;
    const std::string m_sName;
    const std::vector<ElementInfo> *m_pElementInfos;
    int m_numConsumers = 1;
    uint32_t m_uLimitNum = 0U;
    CpuWaitCfg m_cpuWaitCfg;
    QueueType m_queueType{ QueueType::Fifo };
};

struct ElemBufAttr
{
    PacketElementType userType;
    NvSciBufAttrList bufAttrList = nullptr;
    ~ElemBufAttr()
    {
        if (bufAttrList != nullptr) {
            NvSciBufAttrListFree(bufAttrList);
            bufAttrList = nullptr;
        }
    }
};

struct ElemSyncAttr
{
    PacketElementType userType;
    NvSciSyncAttrList syncAttrList = nullptr;
    ~ElemSyncAttr()
    {
        if (syncAttrList != nullptr) {
            NvSciSyncAttrListFree(syncAttrList);
            syncAttrList = nullptr;
        }
    }
};

typedef struct
{
    PacketElementType userType;
    NvSciBufObj bufObj;
} ElemBufObj;

#endif //CCONFIG_HPP
