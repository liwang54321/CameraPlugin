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

#ifndef CBUFAGGREGATOR_HPP
#define CBUFAGGREGATOR_HPP

#include <mutex>
#include <map>
#include <thread>
#include <atomic>
#include <queue>

#include "CClientCommon.hpp"
#include "CProducer.hpp"
#include "CConsumer.hpp"

class CBufAggregator
{
  public:
    CBufAggregator(CClientCommon::IModuleCallback *pCallback);
    /** @brief Default destructor. */
    virtual ~CBufAggregator(){};

    NvError Stop();
    NvError Start();
    inline const std::string GetName() { return "CBufAggregator"; }
    NvError RegisterBufObj(CClientCommon *pClient, uint32_t uPacketIndex, NvSciBufObj bufObj);
    NvError OnPacketGotten(CClientCommon *pClient, uint32_t uPacketIndex);

  private:
    typedef struct
    {
        NvSciBufObj bufObjs[MAX_NUM_PACKETS] = { nullptr };
        /*
         * The buffer object corresponding meta data.
         */
        MetaData *pMetaDataList[MAX_NUM_PACKETS] = { nullptr };
        uint32_t currentIndex = MAX_NUM_PACKETS;
    } BufObjsInfo;

    bool CollectedAllBufs();
    NvError DoWork();

    CProducer *m_pProducer = nullptr;
    CClientCommon::IModuleCallback *m_pModuleCallback = nullptr;
    std::mutex m_packetMutex;
    std::map<CClientCommon *, BufObjsInfo> m_srcBufObjsMap;
    BufObjsInfo m_dstBufObjsInfo;
    std::queue<uint32_t> m_dstAvailableIndexQ;
};

#endif
