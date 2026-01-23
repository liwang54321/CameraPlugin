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

#ifndef CFRAME_READER_H
#define CFRAME_READER_H

#include <list>
#include <mutex>
#include <queue>
#include <atomic>
#include <string>
#include <memory>
#include <fstream>
#include <unordered_map>
#include <condition_variable>
#include <stdint.h>
#include "Common.hpp"
#include "CUtils.hpp"
#include "CEventHandler.hpp"
#include "CFrameHandler.hpp"

class CFrameReader : public CFrameHandler
{
  public:
    CFrameReader(FileSourceType type, const std::string &sFilePath, uint32_t uWidth, uint32_t uHeight, int sensorId);

    virtual ~CFrameReader() override;

    virtual NvError Init() override;

    virtual void DeInit() override;

    virtual NvError Start() override;

    virtual void Stop() override;

    virtual NvError FillNvSciBufAttrList(NvSciBufAttrList &bufAttrList) override;

    virtual NvError FillSyncSignalerAttrList(NvSciSyncAttrList &signalerAttrList) override;

    virtual NvError FillSyncWaiterAttrList(NvSciSyncAttrList &waiterAttrList) override;

    virtual NvError RegisterNvSciBuf(NvSciBufObj &bufObj) override;

    virtual EventStatus LoadFrameData(NvSciBufObj &bufObj, NvSciSyncFence *&pPostFence) override;

    virtual void ReturnBuffer(NvSciBufObj &bufObj) override;

    inline const std::string &GetName() { return m_name; }

  private:
    EventStatus ProcessFrames();

    EventStatus ReadYuvFrame();

    EventStatus ReadYuvSequence();

    EventStatus ReadYuvData(std::ifstream &yuvFile);

  private:
    std::unique_ptr<CEventHandler<CFrameReader>> m_upReadHandler;
    std::atomic<bool> m_bStopReading{ false };
    uint32_t m_uCurrentFrameId{ 0 };
    std::ifstream m_yuvSequenceFile;
    std::mutex m_freeBufMutex;
    std::mutex m_validBufMutex;
    std::condition_variable m_freeBufDataCond;
    std::condition_variable m_validBufDataCond;
    std::list<NvSciBufObj> m_freeBufferList;
    std::queue<NvSciBufObj> m_validBufferQueue;
    std::unordered_map<NvSciBufObj, BufferAttrs> m_bufToAttrsMap;
    uint32_t m_uMissingFileCount{ 0 };
    std::string m_name{ "FrameReader" };
};

#endif
