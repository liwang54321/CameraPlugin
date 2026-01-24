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

#include "CFrameHandler.hpp"
#include "CFrameReader.hpp"
#if !NV_IS_SAFETY
#include "CFrameDecoder.hpp"
#endif // !NV_IS_SAFETY

CFrameHandler::CFrameHandler(
    FileSourceType type, const std::string &sFilePath, uint32_t uWidth, uint32_t uHeight, int sensorId)
    : m_type(type)
    , m_sFilePath(sFilePath)
    , m_uWidth(uWidth)
    , m_uHeight(uHeight)
    , m_sensorId(sensorId)
{
}

CFrameHandler::~CFrameHandler() {}

NvError CFrameHandler::RegisterSignalSyncObj(NvSciSyncObj &signalSyncObj)
{
    return NvError_Success;
}

std::string CFrameHandler::GetCameraDirName()
{
    return m_sFilePath + "/" + CAMERA_DIR_PREFIX + IntToStringWithLeadingZero(m_sensorId);
}

std::unique_ptr<CFrameHandler> CreateFrameHandler(FileSourceType type,
                                                  const std::string &sFilePath,
                                                  const std::string &src_ip,
                                                  uint16_t src_port,
                                                  const std::string &dst_ip,
                                                  uint32_t uWidth,
                                                  uint32_t uHeight,
                                                  int sensorId,
                                                  uint32_t uInstanceId)
{
    if (type == FileSourceType::YUV420P_SINGLE_FRAME || type == FileSourceType::YUV420P_SEQUENCE) {
        return std::make_unique<CFrameReader>(type, sFilePath, uWidth, uHeight, sensorId);
    } else if (type == FileSourceType::H264 || type == FileSourceType::H265) {
#if !NV_IS_SAFETY
        return std::make_unique<CFrameDecoder>(type, src_ip, src_port, dst_ip, uWidth, uHeight, sensorId, uInstanceId);
#else
        LOG_ERR("CreateFrameHandler: Decoding is not supported on safety platform!\n");
        return nullptr;
#endif // !NV_IS_SAFETY
    } else {
        LOG_ERR("Unsupported type of source file.\n");
        return nullptr;
    }
}
