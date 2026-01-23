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

#include <limits>
#include "CBufAggregator.hpp"

//Ensure pCallback isn't nullptr on calling the constructor
CBufAggregator::CBufAggregator(CClientCommon::IModuleCallback *pCallback)
    : m_pModuleCallback(pCallback)
{
}

NvError CBufAggregator::Stop()
{
    PLOG_DBG("Enter: CBufAggregator::Stop()\n");

    std::unique_lock<std::mutex> lck(m_packetMutex);

    for (auto iter = m_srcBufObjsMap.begin(); iter != m_srcBufObjsMap.end(); ++iter) {
        if (iter->second.currentIndex == MAX_NUM_PACKETS) {
            continue;
        }

        CConsumer *pConsumer = dynamic_cast<CConsumer *>(iter->first);
        if (pConsumer) {
            auto error = pConsumer->ReleasePacket(iter->second.currentIndex);
            PCHK_ERROR_AND_RETURN(error, "pConsumer->ReleasePacket");
            iter->second.currentIndex = MAX_NUM_PACKETS;
        } else {
            PLOG_ERR("Null consumer");
            return NvError_InvalidState;
        }
    }

    while (!m_dstAvailableIndexQ.empty()) {
        m_dstAvailableIndexQ.pop();
    }

    PLOG_DBG("Exit: CBufAggregator::Stop()\n");
    return NvError_Success;
}

NvError CBufAggregator::Start()
{
    PLOG_DBG("Enter: CBufAggregator::Start()\n");

    std::unique_lock<std::mutex> lck(m_packetMutex);

    for (uint32_t uPacketIndex = 0U; uPacketIndex < MAX_NUM_PACKETS; ++uPacketIndex) {
        m_dstAvailableIndexQ.push(uPacketIndex);
    }

    PLOG_DBG("Exit: CBufAggregator::Start()\n");
    return NvError_Success;
}

NvError CBufAggregator::RegisterBufObj(CClientCommon *pClient, uint32_t uPacketIndex, NvSciBufObj bufObj)
{
    if (pClient->IsConsumer()) {
        BufObjsInfo bufObjsInfo;
        if (m_srcBufObjsMap.find(pClient) == m_srcBufObjsMap.end()) {
            m_srcBufObjsMap[pClient] = bufObjsInfo;
        }
        m_srcBufObjsMap[pClient].bufObjs[uPacketIndex] = bufObj;
    } else {
        if (!m_pProducer) {
            m_pProducer = dynamic_cast<CProducer *>(pClient);
        }
        m_dstBufObjsInfo.bufObjs[uPacketIndex] = bufObj;
        m_dstAvailableIndexQ.push(uPacketIndex);
    }

    return NvError_Success;
}

bool CBufAggregator::CollectedAllBufs()
{
    for (auto iter = m_srcBufObjsMap.begin(); iter != m_srcBufObjsMap.end(); ++iter) {
        if (iter->second.currentIndex == MAX_NUM_PACKETS) {
            return false;
        }
    }
    if (m_dstAvailableIndexQ.empty()) {
        return false;
    }

    return true;
}

NvError CBufAggregator::OnPacketGotten(CClientCommon *pClient, uint32_t uPacketIndex)
{
    std::unique_lock<std::mutex> lck(m_packetMutex);

    if (pClient->IsConsumer()) {
        BufObjsInfo &bufObjInfo = m_srcBufObjsMap[pClient];
        if (bufObjInfo.currentIndex != MAX_NUM_PACKETS) {
            LOG_DBG("Overwriting previous packet %u.\n", bufObjInfo.currentIndex);
            CConsumer *pConsumer = dynamic_cast<CConsumer *>(pClient);
            if (pConsumer) {
                NvSciSyncFence postfence = NvSciSyncFenceInitializer;
                auto error = pConsumer->OnPacketProcessed(bufObjInfo.currentIndex, &postfence);
                PCHK_ERROR_AND_RETURN(error, "pConsumer->OnPacketProcessed");
            } else {
                PLOG_ERR("Null consumer");
                return NvError_InvalidState;
            }
        }

        bufObjInfo.currentIndex = uPacketIndex;
        MetaData *pMetaData = reinterpret_cast<MetaData *>(pClient->GetMetaPtr(uPacketIndex));
        bufObjInfo.pMetaDataList[uPacketIndex] = pMetaData;
    } else {
        m_dstAvailableIndexQ.push(uPacketIndex);
    }

    if (CollectedAllBufs()) {
        uint64_t frameCaptureTSC =
            static_cast<const MetaData *>(m_srcBufObjsMap.begin()->first->GetMetaPtr(uPacketIndex))->uFrameCaptureTSC;
        static_cast<MetaData *>(m_pProducer->GetMetaPtr(m_dstAvailableIndexQ.front()))->uFrameCaptureTSC =
            frameCaptureTSC;
        auto error = DoWork();
        PCHK_ERROR_AND_RETURN(error, "DoWork");
    }

    return NvError_Success;
}

NvError CBufAggregator::DoWork()
{
    std::vector<NvSciBufObj> vSrcBufObjs;
    std::unique_ptr<NvSciSyncFence, std::function<void(NvSciSyncFence *)>> postfence(
        new NvSciSyncFence(NvSciSyncFenceInitializer), [](NvSciSyncFence *fence) {
            NvSciSyncFenceClear(fence);
            delete fence;
        });

    PCHK_PTR_AND_RETURN(m_pProducer, "m_pProducer");
    MetaData *pMinMetaData = nullptr;
    m_dstBufObjsInfo.currentIndex = m_dstAvailableIndexQ.front();
    m_dstAvailableIndexQ.pop();
    NvSciBufObj &dstBufObj = m_dstBufObjsInfo.bufObjs[m_dstBufObjsInfo.currentIndex];
    for (auto iter = m_srcBufObjsMap.begin(); iter != m_srcBufObjsMap.end(); ++iter) {
        CConsumer *pConsumer = dynamic_cast<CConsumer *>(iter->first);
        if (pConsumer) {
            auto error = pConsumer->InsertPrefence(iter->second.currentIndex);
            PCHK_ERROR_AND_RETURN(error, "pConsumer->InsertPrefence");
            vSrcBufObjs.push_back(iter->second.bufObjs[iter->second.currentIndex]);
            MetaData *pCurrentMetaData = iter->second.pMetaDataList[iter->second.currentIndex];
            /*
             * Find the minimal
             */
            if (!pMinMetaData || (pCurrentMetaData->uFrameCaptureStartTSC < pMinMetaData->uFrameCaptureStartTSC)) {
                pMinMetaData = pCurrentMetaData;
            }
        } else {
            PLOG_ERR("Null consumer");
            return NvError_InvalidState;
        }
    }

    auto error = m_pModuleCallback->SetEofSyncObj(nullptr);
    PCHK_ERROR_AND_RETURN(error, "m_pModuleCallback->SetEofSyncObj");

    error = m_pModuleCallback->ProcessPayload(vSrcBufObjs, dstBufObj, pMinMetaData);
    PCHK_ERROR_AND_RETURN(error, "m_pModuleCallback->AggregateBufs");

    error = m_pModuleCallback->GetEofSyncFence(nullptr, postfence.get());
    PCHK_ERROR_AND_RETURN(error, "m_pModuleCallback->SetEofSyncObj");

    for (auto iter = m_srcBufObjsMap.begin(); iter != m_srcBufObjsMap.end(); ++iter) {
        CConsumer *pConsumer = dynamic_cast<CConsumer *>(iter->first);
        if (pConsumer) {
            error = pConsumer->OnPacketProcessed(iter->second.currentIndex, postfence.get(), false);
            PCHK_ERROR_AND_RETURN(error, "pConsumer->OnPacketProcessed");
        } else {
            PLOG_ERR("Null consumer");
            return NvError_InvalidState;
        }
    }

    /*
     * Copy the meta data of the consumer into the producer's
     */
    MetaData *pMetaData = reinterpret_cast<MetaData *>(m_pProducer->GetMetaPtr(m_dstBufObjsInfo.currentIndex));
    if (pMinMetaData && pMetaData) {
        *pMetaData = *pMinMetaData;
    }
    error = m_pProducer->Post(&dstBufObj, postfence.get());
    PCHK_ERROR_AND_RETURN(error, "m_producer->Post");

    for (auto iter = m_srcBufObjsMap.begin(); iter != m_srcBufObjsMap.end(); ++iter) {
        iter->second.currentIndex = MAX_NUM_PACKETS;
    }

    return NvError_Success;
}
