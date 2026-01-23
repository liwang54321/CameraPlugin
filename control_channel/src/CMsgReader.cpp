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

#include "CMsgReader.hpp"

CMsgReader::CMsgReader(const std::string &sZoneName, MsgHandler msgHandler)
    : m_msgHandler(std::move(msgHandler))
{
    if (sZoneName.size() > kMaxZoneNameLength) {
        m_sZoneName = sZoneName.substr(0, kMaxZoneNameLength);
        LOG_WARN("Control Zone name exceeds max Zone length: %d, Will be truncated to %s", kMaxZoneNameLength,
                 m_sZoneName.c_str());
    } else {
        m_sZoneName = sZoneName;
    }
    LOG_DBG("A reader is created in Zone: %s\n", m_sZoneName.c_str());
}

NvError CMsgReader::ProcessMsg(MessageHeader *pHeaderBuf, void *pContentBuf)
{
    CHK_PTR_AND_RETURN(pHeaderBuf, "CMsgReader::onMsgAvailable");
    CHK_PTR_AND_RETURN(pContentBuf, "CMsgReader::onMsgAvailable");
    if (NULL == m_msgHandler) {
        LOG_ERR("Msg handler is NULL\n");
        return NvError_InvalidState;
    }
    m_msgHandler(pContentBuf, pHeaderBuf->uContentLength);
    return NvError_Success;
}