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

#ifndef CFRAME_HANDLER_H
#define CFRAME_HANDLER_H

#include <memory>
#include <string>
#include <stdint.h>
#include "Common.hpp"
#include "CUtils.hpp"
#include "NvSIPLCommon.hpp"

class CFrameHandler
{
  public:
    CFrameHandler(FileSourceType type, const std::string &sFilePath, uint32_t uWidth, uint32_t uHeight, int sensorId);

    virtual ~CFrameHandler();

    virtual NvError Init() = 0;

    virtual void DeInit() = 0;

    virtual NvError Start() = 0;

    virtual void Stop() = 0;

    virtual NvError FillNvSciBufAttrList(NvSciBufAttrList &bufAttrList) = 0;

    virtual NvError FillSyncSignalerAttrList(NvSciSyncAttrList &signalerAttrList) = 0;

    virtual NvError FillSyncWaiterAttrList(NvSciSyncAttrList &waiterAttrList) = 0;

    virtual NvError RegisterNvSciBuf(NvSciBufObj &bufObj) = 0;

    virtual NvError RegisterSignalSyncObj(NvSciSyncObj &signalSyncObj);

    virtual EventStatus LoadFrameData(NvSciBufObj &bufObj, NvSciSyncFence *&pPostFence) = 0;

    virtual void ReturnBuffer(NvSciBufObj &bufObj) = 0;

  protected:
    FileSourceType GetSourceFileType() { return m_type; }

    uint32_t GetFrameWidth() { return m_uWidth; }

    uint32_t GetFrameHeight() { return m_uHeight; }

    int GetSensorId() { return m_sensorId; }

    std::string GetCameraDirName();

  private:
    FileSourceType m_type = FileSourceType::UNDEFINED;
    std::string m_sFilePath = "";
    uint32_t m_uWidth = 0U;
    uint32_t m_uHeight = 0U;
    int m_sensorId = INVALID_ID;
};

std::unique_ptr<CFrameHandler> CreateFrameHandler(FileSourceType type,
                                                  const std::string &sFilePath,
                                                  const std::string &src_ip,
                                                  uint16_t src_port,
                                                  const std::string &dst_ip,
                                                  uint32_t uWidth,
                                                  uint32_t uHeight,
                                                  int sensorId,
                                                  uint32_t uInstanceId);

#endif
