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

#include <string>
#include "CConfig.hpp"
#include "CPeerValidator.hpp"

/* Type of info */
#define VALIDATION_INFO_TYPE 0xabcd
/* Size of info */
#define VALIDATION_INFO_SIZE 256

NvError CPeerValidator::SendValidationInfo(const NvSciStreamBlock handle)
{
    LOG_DBG("CPeerValidator::SendValidationInfo.\n");

    std::string message;
    ComposeValidationInfo(message);
    NvSciError err = NvSciStreamBlockUserInfoSet(handle, VALIDATION_INFO_TYPE, message.size(), message.c_str());
    CHK_NVSCISTATUS_AND_RETURN(err, "NvSciStreamBlockUserInfoSet");
    return NvError_Success;
}

NvError CPeerValidator::Validate(const NvSciStreamBlock handle)
{
    LOG_DBG("CPeerValidator::Validate.\n");

    uint32_t size = VALIDATION_INFO_SIZE;
    char infoProd[VALIDATION_INFO_SIZE] = "";
    NvError error = NvError_Success;

    NvSciError err =
        NvSciStreamBlockUserInfoGet(handle, NvSciStreamBlockType_Producer, 0U, VALIDATION_INFO_TYPE, &size, &infoProd);
    if (NvSciError_Success == err) {
        std::string message;
        ComposeValidationInfo(message);
        LOG_DBG("the info from producer: %s.\n", infoProd);
        LOG_DBG("the info in this consumer: %s.\n", message.c_str());
        if (0 == std::strcmp(message.c_str(), infoProd)) {
            LOG_INFO("Peer validation succeeded.\n");
            error = NvError_Success;
        } else {
            LOG_WARN("Peer validation failed. Prod info: %s, Cons info: %s.\n", infoProd, message.c_str());
            error = NvError_BadParameter;
        }
    } else if (NvSciError_StreamInfoNotProvided == err) {
        LOG_ERR("validation info not provided by the producer.\n");
        error = NvError_BadParameter;
    } else {
        LOG_ERR("failed to query the producer info.\n");
        error = NvError_BadParameter;
    }

    return error;
}

void CPeerValidator::ComposeValidationInfo(std::string &outputInfo)
{
    LOG_DBG("CPeerValidator::GetStringInfo\n");

    int isStaticConfig;
    std::string platformName;
    int isMultiElementsEnabled = (int)m_pAppCfg->IsMultiElementsEnabled();

#if !NV_IS_SAFETY
    if (!m_pAppCfg->GetDynamicConfigName().empty()) {
        isStaticConfig = 0;
        auto name = m_pAppCfg->GetDynamicConfigName();
        platformName = platformName.append(name);
        platformName = platformName.append(";");
    } else {
#endif
        isStaticConfig = 1;
        platformName =
            m_pAppCfg->GetStaticConfigName().empty() ? "F008A120RM0AV2_CPHY_x4" : m_pAppCfg->GetStaticConfigName();
#if !NV_IS_SAFETY
    }
    std::string masks;
    auto mask = m_pAppCfg->GetMasks();
    for (auto m : mask) {
        if (m == 0)
            continue;
        masks.append(std::to_string(m) + ",");
    }
    masks.append(";");
#endif
    outputInfo = "{isStaticConfig=" + std::to_string(isStaticConfig) + "}{platformName=" + platformName
#if !NV_IS_SAFETY
                 + "}{masks=" + masks
#endif
                 + "}{isMultiElementsEnabled=" + std::to_string(isMultiElementsEnabled) + "}";
}