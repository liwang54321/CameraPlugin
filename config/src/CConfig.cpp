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

#include <algorithm>
#include <array>
#include <cctype>
#include <sstream>
#include <regex>

#include "CConfig.hpp"
#if !NV_IS_SAFETY
#include "NvSIPLQuery.hpp"      // Query
#include "NvSIPLQueryTrace.hpp" // Query Trace
#endif
#if NV_BUILD_DOS7
#include "platform/thor/ar0820.hpp"
#include "platform/thor/imx623vb2.hpp"
#include "platform/thor/imx728vb2.hpp"
#include "platform/thor/isx031.hpp"
#include "platform/thor/max96712_tpg_yuv.hpp"
#else
#include "platform/orin/ar0820.hpp"
#include "platform/orin/imx623vb2.hpp"
#include "platform/orin/imx728vb2.hpp"
#include "platform/orin/isx031.hpp"
#include "platform/orin/max96712_tpg_yuv.hpp"
#endif
#include "CElementDescription.hpp"
#include "CPipelineConfig.hpp"

static constexpr const char *kWhiteSpaces = " \n\t\r";
constexpr int MAX_NUM_MODULE_INSTANCES = 10U;

static const std::vector<ElementInfo> vSingleElemInfos_Sipl = {
    /*userType, bIsUsed, bHasSibling*/
    { PacketElementType::ICP_RAW, true, false },
    { PacketElementType::METADATA, true, false },
    { PacketElementType::NV12_BL, true, false },
};

static const std::vector<ElementInfo> vSingleElemInfos_Bl = {
    /*userType, bIsUsed, bHasSibling*/
    { PacketElementType::ICP_RAW, false, false },
    { PacketElementType::METADATA, true, false },
    { PacketElementType::NV12_BL, true, false },
};

static const std::vector<ElementInfo> vSingleElemInfos_Pl = {
    /*userType, bIsUsed, bHasSibling*/
    { PacketElementType::ICP_RAW, false, false },
    { PacketElementType::METADATA, true, false },
    { PacketElementType::NV12_PL, true, false },
};

static const std::vector<ElementInfo> vICPSingleElemInfos = {
    /*userType, isUsed, hasSibling*/
    { PacketElementType::ICP_RAW, true, false },
    { PacketElementType::METADATA, true, false },
};

static const std::vector<ElementInfo> vMultiElemInfos_Sipl = {
    /*userType, bIsUsed, bHasSibling*/
    { PacketElementType::ICP_RAW, true, false },
    { PacketElementType::METADATA, true, false },
    { PacketElementType::NV12_BL, true, true },
    { PacketElementType::NV12_PL, true, true },
};

static const std::vector<ElementInfo> vMultiElemInfos_Pl = {
    /*userType, bIsUsed, bHasSibling*/
    { PacketElementType::ICP_RAW, false, false },
    { PacketElementType::METADATA, true, false },
    { PacketElementType::NV12_BL, false, false },
    { PacketElementType::NV12_PL, true, false },
};

static const std::vector<ElementInfo> vMultiElemInfos_Bl = {
    /*userType, bIsUsed, bHasSibling*/
    { PacketElementType::ICP_RAW, false, false },
    { PacketElementType::METADATA, true, false },
    { PacketElementType::NV12_BL, true, false },
    { PacketElementType::NV12_PL, false, false },
};

static const std::vector<ElementInfo> vElemInfosNvm2dOpaque = {
    /*userType, bIsUsed, bHasSibling*/
    { PacketElementType::METADATA, true, false },
    { PacketElementType::OPAQUE, true, false },
};

static const std::unordered_map<std::string, const std::vector<ElementInfo> *> name2ElementInfoMap{
    { "pl", &vSingleElemInfos_Pl },           { "plMulti", &vMultiElemInfos_Pl },
    { "bl", &vSingleElemInfos_Bl },           { "blMulti", &vMultiElemInfos_Bl },
    { "siplSingle", &vSingleElemInfos_Sipl }, { "siplMulti", &vMultiElemInfos_Sipl },
    { "icp", &vICPSingleElemInfos },          { "nvm2dopaque", &vElemInfosNvm2dOpaque },
};

const OptionTable CIpcEndpointElem::m_ipcEndpointOptionTable = {
    { "late", { "late attach", offsetof(IpcEndPointOption, bIsLate), OptionType::BOOL } },
    { "c2c", { "C2C", offsetof(IpcEndPointOption, bIsC2C), OptionType::BOOL } },
    { "limit", { "limit block capacity", offsetof(IpcEndPointOption, uLimitNum), OptionType::UINT32 } },
};

std::string ShortenName(const std::string &name)
{
    std::vector<std::string> nameParts = splitString(name, '_');
    size_t namePartsLen = nameParts.size();
    std::string channel, elemPart, sensorId, clientPart;

    if (namePartsLen == 3 || namePartsLen == 4) {
        channel = nameParts[0];
        elemPart = nameParts[1];
        sensorId = nameParts[2];

        elemPart = ReplaceName(elemPart, ElemNameToDisplayName);
        if (std::all_of(sensorId.begin(), sensorId.end(), ::isdigit)) {
            std::stringstream sensorIndex;
            sensorIndex << std::hex << std::stoi(sensorId);
            sensorId = sensorIndex.str();
        }

        // Module name
        if (namePartsLen == 3) {
            return channel + "_" + elemPart + "_" + sensorId;
        // Client name
        } else if (namePartsLen == 4) {
            clientPart = nameParts[3];
            clientPart = ReplaceName(clientPart, ElemNameToDisplayName);

            if (sensorId == "*") {
                return channel + "_" + elemPart + "_" + clientPart;
            } else {
                std::regex digitPattern("\\d+");
                return channel + "_" + elemPart + "_" + sensorId + "_" + std::regex_replace(clientPart, digitPattern, "");
            }
        }
    }
    return name;
};


CAppCfg::CAppCfg()
{
    PopulatePipelineCfgTable();
}

CAppCfg::~CAppCfg() = default;

void CAppCfg::Init()
{
    PopulatePlatformCfgs();
    PopulateCameraModuleInfo();
}

void CAppCfg::PopulatePipelineCfgTable()
{
    for (const auto &p : sPipelineTable) {
        m_pipelineCfgTable.emplace(p[0], p[1]);
    }
}

void CAppCfg::PopulatePlatformCfgs()
{
    if (m_platformCfgs.empty()) {
        for (auto &inputGroup : m_vPipelineGroups) {
            if (!inputGroup.sStaticConfigName.empty()) {
                if (inputGroup.sStaticConfigName == "F008A120RM0AV2_CPHY_x4") {
                    m_platformCfgs.push_back(platformCfgAr0820);
                } else if (inputGroup.sStaticConfigName == "V1SIM623S4RU5195NB3_CPHY_x4") {
                    m_platformCfgs.push_back(platformCfgIMX623VB2);
                } else if (inputGroup.sStaticConfigName == "V1SIM728S1RU3120NB20_CPHY_x4") {
                    m_platformCfgs.push_back(platformCfgIMX728VB2);
                } else if (inputGroup.sStaticConfigName == "MAX96712_YUV_8_TPG_CPHY_x4") {
                    m_platformCfgs.push_back(platformCfgMax96712TPGYUV);
                } else if (inputGroup.sStaticConfigName == "MAX96712_2880x1860_YUV_8_TPG_DPHY_x4") {
                    m_platformCfgs.push_back(platformCfgMax96712TPGYUV_5m);
                } else if (inputGroup.sStaticConfigName == "SG3_ISX031_H190X_YUV_8_CPHY_x2") {
                    m_platformCfgs.push_back(platformCfgIsx031x2);
                } else if (inputGroup.sStaticConfigName == "SG3_ISX031_H190X_YUV_8_CPHY_x4") {
                    m_platformCfgs.push_back(platformCfgIsx031x4);
                } else {
                    LOG_ERR("Unexpected platform configuration\n");
                }
            } else {
#if !NV_IS_SAFETY
                PlatformCfg platcfg;
                // INvSIPLQuery
                auto pQuery = INvSIPLQuery::GetInstance();
                if (pQuery == nullptr) {
                    LOG_ERR("INvSIPLQuery::GetInstance() return null.\n");
                }
                auto error = toNvError(pQuery->ParseDatabase());
                if (error != NvError_Success) {
                    LOG_ERR("INvSIPLQuery::ParseDatabase() failed.\n");
                }
                error = toNvError(pQuery->GetPlatformCfg(inputGroup.sDynamicConfigName, platcfg));
                if (error != NvError_Success) {
                    LOG_ERR("INvSIPLQuery::GetPlatformCfg failed, error: %u\n", error);
                }
                // Apply mask
                LOG_INFO("Setting link masks\n");
                error = toNvError(pQuery->ApplyMask(platcfg, inputGroup.vuMasks));
                if (error != NvError_Success) {
                    LOG_ERR("INvSIPLQuery::ApplyMask failed, error: %u\n", error);
                }
                m_platformCfgs.push_back(platcfg);
#endif // !NV_IS_SAFETY
            }
        }
    }
}

std::vector<PlatformCfg> CAppCfg::GetPlatformCfgs()
{
    return m_platformCfgs;
}

void CAppCfg::ExtractSensorIds(const std::string &platformName, std::vector<int> &outputSensorIds)
{
    for (auto &platcfg : GetPlatformCfgs()) {
        if (platcfg.platformConfig == platformName) {
            for (auto d = 0u; d != platcfg.numDeviceBlocks; d++) {
                const auto &db = platcfg.deviceBlockList[d];
                for (auto m = 0u; m != db.numCameraModules; m++) {
                    outputSensorIds.push_back(db.cameraModuleInfoList[m].sensorInfo.id);
                }
            }
        }
    }
}

void CAppCfg::GetPipelineTable(PipelineTable &pipelineTable, uint32_t &uSize) const
{
    pipelineTable = sPipelineTable;
    uSize = ARRAY_SIZE(sPipelineTable);
}

NvError CAppCfg::CheckPipelineGroups()
{
    std::unordered_set<std::string> seenNames;
    uint32_t ConfigCount = 0;
#if !NV_IS_SAFETY
    bool bHasStaticConfig = false;
#endif

    for (auto &inputGroup : m_vPipelineGroups) {
        ConfigCount++;
        if (ConfigCount > MAX_DEVICEBLOCKS_PER_PLATFORM) {
            LOG_ERR("Config count exceed max camera instance.");
            return NvError_BadParameter;
        }

        if (!inputGroup.sStaticConfigName.empty()) {
#if !NV_IS_SAFETY
            bHasStaticConfig = true;
            // can't set in same input pipeline group
            if (!inputGroup.sDynamicConfigName.empty()) {
                LOG_ERR("Dynamic config cannot be set together with static config.");
                return NvError_BadParameter;
            }
#endif
            if (seenNames.find(inputGroup.sStaticConfigName) != seenNames.end()) {
                LOG_ERR("Duplicate static config name found: %s", inputGroup.sStaticConfigName.c_str());
                return NvError_BadParameter;
            }
            seenNames.insert(inputGroup.sStaticConfigName);
        } else {
#if !NV_IS_SAFETY
            // can't set in different input pipeline group
            if (bHasStaticConfig) {
                LOG_ERR("Dynamic config cannot be set together with static config.");
                return NvError_BadParameter;
            }
            // Same name shall not exist in sDynamicConfigName
            if (seenNames.find(inputGroup.sDynamicConfigName) != seenNames.end()) {
                LOG_ERR("Duplicate DynamicConfigName found: %s", inputGroup.sDynamicConfigName.c_str());
                return NvError_BadParameter;
            }
            seenNames.insert(inputGroup.sDynamicConfigName);

            if (inputGroup.vuMasks.empty()) {
                LOG_ERR("Dynamic config name %s has no masks.", inputGroup.sDynamicConfigName.c_str());
                return NvError_BadParameter;
            }
#endif
        }
    }
    return NvError_Success;
}

void CAppCfg::AddPipelineGroup(PipelineGroup &pipelineGroup)
{
    if (pipelineGroup.sStaticConfigName.empty()) {
#if !NV_IS_SAFETY
        if (pipelineGroup.sDynamicConfigName.empty()) {
#endif
        pipelineGroup.sStaticConfigName = "V1SIM728S1RU3120NB20_CPHY_x4";
#if !NV_IS_SAFETY
        }
#endif
    }
    if (pipelineGroup.sPipelineDescriptor.empty()) {
        pipelineGroup.sPipelineDescriptor = "multicast";
    }
    m_vPipelineGroups.push_back(pipelineGroup);
}

void CAppCfg::PopulateCameraModuleInfo()
{
    if (m_vCameraModules.empty()) {
        for (auto &platcfg : m_platformCfgs) {
            for (auto d = 0u; d != platcfg.numDeviceBlocks; d++) {
                const auto &db = platcfg.deviceBlockList[d];
                for (auto m = 0u; m != db.numCameraModules; m++) {
                    m_vCameraModules.push_back(db.cameraModuleInfoList[m]);
                }
            }
        }
    }
}

std::vector<CameraModuleInfo> CAppCfg::GetCameraModuleInfo()
{
    return m_vCameraModules;
}

bool CAppCfg::GetPipelineStrFromTable(PipelineCfgTable &pipelineCfgTable,
                                      const char *pipelineDescriptionString,
                                      std::string &pipelineStr)
{
    auto it = pipelineCfgTable.find(pipelineDescriptionString);
    if (it != pipelineCfgTable.end()) {
        pipelineStr = it->second;
        return true;
    }
    return false;
}

void CAppCfg::UpdateByPipelineDescStr(const char *pipelineDescriptionString)
{
    if (nullptr == pipelineDescriptionString) {
        return;
    }
    std::string pipelineStr = "";
    bool isFound = GetPipelineStrFromTable(m_pipelineCfgTable, pipelineDescriptionString, pipelineStr);
    const char *actualPipelineString = isFound ? pipelineStr.c_str() : pipelineDescriptionString;

    std::array<std::string, 2> vIpcLabels = { { "IpcSrc", "IpcDst" } };
    std::array<std::string, 1> vC2CLabels = { { "c2c" } };
    if (ContainAnySubStr(vIpcLabels, actualPipelineString)) {
        if (ContainAnySubStr(vC2CLabels, actualPipelineString)) {
            m_eCommType = CommType::InterChip;
        } else {
            m_eCommType = CommType::InterProcess;
        }
    }
    std::array<std::string, 1> vIpcDstLabel = { { "IpcDst" } };
    if (!ContainAnySubStr(vIpcDstLabel, actualPipelineString)) {
        m_bPureSource = true;
    }
    std::array<std::string, 1> vMultiElemsLabels = { { "Multi" } };
    if (ContainAnySubStr(vMultiElemsLabels, actualPipelineString)) {
        m_bEnableMultiElements = true;
    }

    std::array<std::string, 2> vMasterSocLabel = { { "IpcSrc", "c2c" } };
    PipelineDescriptor vPipelines = ConvertToPipelineDescriptor(actualPipelineString);
    for (const auto &pipeline : vPipelines) {
        for (const auto &pipelineElem : pipeline) {
            if (ContainAllSubStr(vMasterSocLabel, pipelineElem.c_str())) {
                m_bC2CMasterSoc = true;
            }
        }
    }
}

bool CAppCfg::IsYuvOrTpgSensor(uint32_t uSensorId)
{
    for (auto &cameraModule : GetCameraModuleInfo()) {
        SensorInfo *pSensorInfo = &cameraModule.sensorInfo;
        if (pSensorInfo->id == uSensorId) {
#if !NV_IS_SAFETY
            return (pSensorInfo->vcInfo.inputFormat == NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV422 ||
                    pSensorInfo->isTPGEnabled == true);
#else
            return pSensorInfo->vcInfo.inputFormat == NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV422;
#endif
        }
    }

    LOG_WARN("CAppCfg::IsYuvOrTpgSensor(), invalid uSensorId: %u\n", uSensorId);
    return false;
}

PipelineDescriptor CAppCfg::ConvertToPipelineDescriptor(const char *pipelineDescriptorString)
{
    PipelineDescriptor pipelineDescriptor;

    pipelineDescriptorString += std::strspn(pipelineDescriptorString, kWhiteSpaces);
    std::vector<std::string> vPipelineStrs = splitString(std::string(pipelineDescriptorString), ';');
    for (const std::string &vPipelineStr : vPipelineStrs) {
        pipelineDescriptor.emplace_back(std::move(splitString(vPipelineStr, ',')));
    }

    return pipelineDescriptor;
}

void CAppCfg::CreateChannelCfgs(std::vector<std::shared_ptr<CChannelCfg>> &vspChannelCfgs)
{
    uint32_t uChannelIndex = 0;
    if (m_vPipelineGroups.empty()) {
        PipelineGroup pipelineGroup;
        AddPipelineGroup(pipelineGroup);
    }
    for (const auto &inputGroup : m_vPipelineGroups) {
        PipelineDescriptor pipelines;
        std::string sChannelName = "c" + std::to_string(uChannelIndex);
        std::string pipelineStr = "";

        bool isFound = GetPipelineStrFromTable(m_pipelineCfgTable, inputGroup.sPipelineDescriptor.c_str(), pipelineStr);
        if (isFound) {
            pipelines = ConvertToPipelineDescriptor(pipelineStr.c_str());
        } else {
            pipelines = ConvertToPipelineDescriptor(inputGroup.sPipelineDescriptor.c_str());
        }

        auto channelCfg = std::make_shared<CChannelCfg>(this, std::move(pipelines), sChannelName.c_str());
        channelCfg->m_uChannelId = uChannelIndex++;
        if (!inputGroup.sStaticConfigName.empty()) {
            channelCfg->m_sPlatformConfigName = inputGroup.sStaticConfigName;
        } else {
#if !NV_IS_SAFETY
            channelCfg->m_sPlatformConfigName = inputGroup.sDynamicConfigName;
#endif
        }
        vspChannelCfgs.emplace_back(channelCfg);
    }
}

std::shared_ptr<CPeerValidator> CAppCfg::GetPeerValidator()
{
    if (!m_pPeerValidator) {
        LOG_DBG("CAppCfg::SetPeerValidator\n");
        m_pPeerValidator.reset(new CPeerValidator(this));
    }

    return m_pPeerValidator;
}

CChannelCfg::CChannelCfg(CAppCfg *pAppCfg, PipelineDescriptor &&vPipelineDescriptor, const char *pName)
    : m_pAppCfg(pAppCfg)
    , m_vPipelineDescriptor(std::move(vPipelineDescriptor))
    , m_sName(pName)
{
}

NvError CChannelCfg::ExtractOptions(PipelineElemType type, const char *parameters, Options &options)
{
    NvError error = NvError_Success;
    while (parameters && *parameters) {
        std::string sKey;
        error = GetToken(&parameters, "=", sKey);
        CHK_ERROR_AND_RETURN(error, "Get the key");
        if (sKey.empty()) {
            error = NvError_BadParameter;
            LOG_ERR("Key is empty\n");
            break;
        }
        ++parameters;

        std::string sValue;
        error = GetToken(&parameters, ":", sValue);
        CHK_ERROR_AND_RETURN(error, "Get the value");
        if (sValue.empty()) {
            error = NvError_BadParameter;
            LOG_ERR("The value of %s is empty\n", sKey.c_str());
            break;
        }
        if (*parameters) {
            ++parameters;
        }
        options.emplace(std::move(sKey), std::move(sValue));
    }
    return error;
}

std::shared_ptr<CPipelineElem> CChannelCfg::ParsePipelineStr(const std::string &pipelineStr)
{
    /*
     * pipelineStr is not nullable
     */
    if (!pipelineStr.empty()) {
        int elemId = 0;
        int elemNum = 1;
        std::vector<int> vSensorIds;
        std::string sType;
        bool isStitching = false;

        /*
         * Get the module string.
         */
        const char *pPipelineString = pipelineStr.c_str();
        std::string sModuleString;
        NvError error = GetToken(&pPipelineString, "=", sModuleString);
        if (NvError_Success != error) {
            LOG_ERR("Failed to get the `=` token.\n");
            return nullptr;
        }

        /*
         * Get the module string without the sensor id.
         */
        const char *pModuleString = sModuleString.c_str();
        std::string sModuleStringWithoutSensorId;
        error = GetToken(&pModuleString, "_", sModuleStringWithoutSensorId);
        if (NvError_Success != error) {
            LOG_ERR("Failed to get the `_` token.\n");
            return nullptr;
        }
        if ('_' == *pModuleString) {
            for (auto i = 1U; i <= sModuleString.size() - sModuleStringWithoutSensorId.size() - 1; ++i) {
                char ch = std::toupper(static_cast<unsigned char>(pModuleString[i]));
                int idx = (ch <= '9') ? (ch - '0') : (ch - 'A' + 10);
                if (!(idx >= 0 && idx <= static_cast<int>(MAX_NUM_SENSORS))) {
                    LOG_ERR("Invalid sensor id: %c\n", pModuleString[i]);
                    return nullptr;
                }
                vSensorIds.push_back(idx);
            }
        } else {
            //extract SensorIds from PlatformCfgs
            if (!m_sPlatformConfigName.empty()) {
                m_pAppCfg->ExtractSensorIds(m_sPlatformConfigName, vSensorIds);
            } else {
                LOG_ERR("Platform config name for %s is empty\n", m_sName.c_str());
                return nullptr;
            }
        }

        if ('*' == sModuleStringWithoutSensorId[0]) {
            isStitching = true;
            sModuleStringWithoutSensorId = sModuleStringWithoutSensorId.substr(1);
        }

        if (std::isdigit(sModuleStringWithoutSensorId.back())) {
            elemId = sModuleStringWithoutSensorId.back() - '0';
            std::string::reverse_iterator iter = sModuleStringWithoutSensorId.rbegin();
            iter++;
            if (*iter++ == '/') {
                if (!std::isdigit(*iter)) {
                    LOG_ERR("Failed to get elem index, pipelineStr: %s\n", pipelineStr.c_str());
                    return nullptr;
                }
                elemNum = elemId;
                elemId = *iter - '0';
                sModuleStringWithoutSensorId.pop_back();
                sModuleStringWithoutSensorId.pop_back();
            }
            if (elemId >= MAX_NUM_MODULE_INSTANCES || elemNum > MAX_NUM_MODULE_INSTANCES) {
                LOG_ERR("Invalid setting, elemNum: %d, elemId: %d\n", elemNum, elemId);
                return nullptr;
            }
            sType = sModuleStringWithoutSensorId.substr(0, sModuleStringWithoutSensorId.size() - 1);
        } else {
            elemId = 0;
            sType = sModuleStringWithoutSensorId;
        }

        auto it = pipelineElemName2TypeMap.find(sType);
        if (it == pipelineElemName2TypeMap.end()) {
            LOG_MSG("element: %s is not supported!\n", sModuleStringWithoutSensorId.c_str());
            return nullptr;
        }

        auto type = it->second;
        std::string sName = std::move(sModuleStringWithoutSensorId);

        /*
         * Parse the parameters
         */
        Options options;
        if ('=' == *pPipelineString) {
            NvError error = ExtractOptions(type, pPipelineString + 1, options);
            if (NvError_Success != error) {
                LOG_ERR("Failed to parse parameters.\n");
                return nullptr;
            }
        }

        if (vSensorIds.empty()) {
            LOG_ERR("Channel_%d: Sensor ids are empty.\n", m_uChannelId);
            return nullptr;
        }
        if (type >= PipelineElemType::IpcSrc) {
            return std::make_shared<CIpcEndpointElem>(sName, type, vSensorIds, elemId, elemNum, isStitching, options);
        }
        return std::make_shared<CModuleElem>(sName, type, vSensorIds, elemId, elemNum, isStitching, options);
    }

    return nullptr;
}

std::shared_ptr<CModuleCfg> CChannelCfg::CreateModuleCfg(int sensorId, std::shared_ptr<CModuleElem> &spModuleElem)
{
    std::string sensorStr = sensorId != INVALID_ID ? std::to_string(sensorId) : "*";
    std::string sName = m_sName + "_" + spModuleElem->m_sName + "_" + sensorStr;
    std::shared_ptr<CModuleCfg> spModuleCfg = std::make_shared<CModuleCfg>(m_pAppCfg, sName, sensorId, spModuleElem);

    return spModuleCfg;
}

std::shared_ptr<CClientCfg> CModuleCfg::CreateClientCfg(const std::string &sName,
                                                        int numConsumers,
                                                        uint32_t uLimitNum,
                                                        const std::vector<ElementInfo> *pElementInfos,
                                                        QueueType queueType)
{
    std::shared_ptr<CClientCfg> spClientCfg =
        std::make_shared<CClientCfg>(m_pAppCfg, sName, pElementInfos, numConsumers, uLimitNum, m_cpuWaitCfg, queueType);
    return spClientCfg;
}

const std::vector<ElementInfo> *CModuleCfg::GetElementInfos(const std::string &sElems)
{
    const std::vector<ElementInfo> *pElementInfos = nullptr;

    if (!sElems.empty()) {
        if (name2ElementInfoMap.find(sElems) != name2ElementInfoMap.end()) {
            pElementInfos = name2ElementInfoMap.at(sElems);
        }
    } else {
        // No sElems set for producer, using default element Infos by sensor type and module type.
        if (m_sensorId != INVALID_ID && m_pAppCfg->IsYuvOrTpgSensor(m_sensorId)) {
            pElementInfos = &vICPSingleElemInfos;
        } else if (m_moduleType == ModuleType::SIPL) {
            pElementInfos = &vSingleElemInfos_Sipl;
        }
    }

    if (pElementInfos == nullptr) {
        LOG_MSG("Using default elemInfo.\n");
        pElementInfos = &vSingleElemInfos_Bl;
    }

    return pElementInfos;
}

CElementDescription ipcSrcDescription{ "IpcSrc", "Ipc source endpoint", nullptr,
                                       &CIpcEndpointElem::m_ipcEndpointOptionTable };

CElementDescription ipcDstDescription{ "IpcDst", "Ipc dest endpoint", nullptr,
                                       &CIpcEndpointElem::m_ipcEndpointOptionTable };

NvError CIpcEndpointElem::ParseOptions()
{
    std::vector<OptionParserInfo> vOptionParserInfo;

    vOptionParserInfo.push_back({ &m_ipcEndpointOptionTable, &m_ipcEndPointOption });

    auto error = COptionParser::ParseOptions(m_options, vOptionParserInfo);
    CHK_ERROR_AND_RETURN(error, "COptionParser::ParseOptions()");

    return NvError_Success;
}

const std::string CIpcEndpointElem::GetChannelStr(int sensorId)
{
    int elemSensorIndex = 0;
    //For stitching case, the sensor index is 0
    if (sensorId != INVALID_ID) {
        auto it = std::find(m_vSensorIds.begin(), m_vSensorIds.end(), sensorId);
        if (it != m_vSensorIds.end()) {
            elemSensorIndex = distance(m_vSensorIds.begin(), it);
        } else {
            LOG_ERR("GetChannelStr(), sensorId: %d is not found.\n", sensorId);
            return "";
        }
    }
    if (!IsC2C()) {
        if (m_type == PipelineElemType::IpcSrc) {
            return IPC_CHANNEL_PREFIX + std::to_string(elemSensorIndex * m_elemNum * 2 + 2 * m_elemId);
        } else {
            return IPC_CHANNEL_PREFIX + std::to_string(elemSensorIndex * m_elemNum * 2 + 2 * m_elemId + 1);
        }
    } else {
        if (m_type == PipelineElemType::IpcSrc) {
            return C2C_SRC_CHANNEL_PREFIX + std::to_string(elemSensorIndex * m_elemNum + m_elemId + 1);
        } else {
            return C2C_DST_CHANNEL_PREFIX + std::to_string(elemSensorIndex * m_elemNum + m_elemId + 1);
        }
    }
}

const std::string CIpcEndpointElem::GetOppositeChannelStr(int sensorId)
{
    int elemSensorIndex = 0;
    //For stitching case, the sensor index is 0
    if (sensorId != INVALID_ID) {
        auto it = std::find(m_vSensorIds.begin(), m_vSensorIds.end(), sensorId);
        if (it != m_vSensorIds.end()) {
            elemSensorIndex = distance(m_vSensorIds.begin(), it);
        } else {
            LOG_ERR("GetChannelStr(), sensorId: %d is not found.\n", sensorId);
            return "";
        }
    }
    if (!IsC2C()) {
        if (m_type == PipelineElemType::IpcSrc) {
            return IPC_CHANNEL_PREFIX + std::to_string(elemSensorIndex * m_elemNum * 2 + 2 * m_elemId + 1);
        } else {
            return IPC_CHANNEL_PREFIX + std::to_string(elemSensorIndex * m_elemNum * 2 + 2 * m_elemId);
        }
    } else {
        if (m_type == PipelineElemType::IpcSrc) {
            return C2C_DST_CHANNEL_PREFIX + std::to_string(elemSensorIndex * m_elemNum + m_elemId + 1);
        } else {
            return C2C_SRC_CHANNEL_PREFIX + std::to_string(elemSensorIndex * m_elemNum + m_elemId + 1);
        }
    }
}

void CModuleElem::AddIpcDst(std::shared_ptr<CIpcEndpointElem> spIpcDst)
{
    auto it = std::find_if(m_vspIpcDsts.begin(), m_vspIpcDsts.end(),
                           [&spIpcDst](const std::shared_ptr<CIpcEndpointElem> &m_vspIpcDst) {
                               return m_vspIpcDst->m_sName == spIpcDst->m_sName;
                           });
    if (it == m_vspIpcDsts.end()) {
        m_vspIpcDsts.push_back(spIpcDst);
    }
}

void CModuleElem::AddIpcSrc(std::shared_ptr<CIpcEndpointElem> spIpcSrc)
{
    auto it = std::find(m_vspIpcSrcs.begin(), m_vspIpcSrcs.end(), spIpcSrc);
    if (it == m_vspIpcSrcs.end()) {
        m_vspIpcSrcs.push_back(spIpcSrc);
    }
}

void CModuleElem::AddDownstreamModule(std::shared_ptr<CModuleElem> spMod)
{
    auto it = std::find(m_vspDownstreamModules.begin(), m_vspDownstreamModules.end(), spMod);
    if (it == m_vspDownstreamModules.end()) {
        m_vspDownstreamModules.push_back(spMod);
    }
}
