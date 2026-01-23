/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "CMsgWriter.hpp"

CMsgWriter::CMsgWriter(const std::string &sZoneName, IMsgWriteCallback *pCallback)
{
    if (sZoneName.size() > kMaxZoneNameLength) {
        m_sZoneName = sZoneName.substr(0, kMaxZoneNameLength);
        LOG_WARN("Control Zone name exceeds max Zone length: %d, Will be truncated to %s", kMaxZoneNameLength,
                 m_sZoneName.c_str());
    } else {
        m_sZoneName = sZoneName;
    }
    m_pWriteCallback = pCallback;
    LOG_DBG("A writer is created in Zone: %s\n", m_sZoneName.c_str());
}

NvError CMsgWriter::Write(void *pContentBuf, uint32_t size)
{
    // generate header
    MessageHeader messageHeader(m_sZoneName, sizeof(MessageHeader), size, MsgType::USER_MSG);
    return m_pWriteCallback->PostEvent(&messageHeader, pContentBuf);
}
