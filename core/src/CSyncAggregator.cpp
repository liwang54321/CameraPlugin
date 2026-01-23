/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "CSyncAggregator.hpp"

CSyncAggregator::CSyncAggregator(std::vector<std::shared_ptr<CClientCommon>> &vspClients)
{
    for (uint32_t i = 0U; i < vspClients.size(); ++i) {
        m_map[vspClients[i].get()] = nullptr;
    }
}

void CSyncAggregator::AddClient(CClientCommon *pClient)
{
    std::unique_lock<std::mutex> lck(m_syncMutex);

    if (m_map.find(pClient) == m_map.end()) {
        m_map[pClient] = nullptr;
    }
}

bool CSyncAggregator::AllReceived()
{
    for (auto iter = m_map.begin(); iter != m_map.end(); ++iter) {
        if (iter->second == nullptr) {
            return false;
        }
    }

    return true;
}

NvError CSyncAggregator::OnWaiterAttrEventRecvd(CClientCommon *pClient)
{
    ElemSyncAttr elemSyncAttr;
    std::vector<NvSciSyncAttrList> vUnreconciledList;
    NvSciSyncObj syncObj = nullptr;

    auto error = pClient->RecvWaiterAttr(elemSyncAttr);
    PCHK_ERROR_AND_RETURN(error, "pClient->RecvWaiterAttr");

    std::unique_lock<std::mutex> lck(m_syncMutex);

    auto sciErr = NvSciSyncAttrListClone(elemSyncAttr.syncAttrList, &m_map[pClient]);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListClone");

    if (AllReceived()) {
        for (auto iter = m_map.begin(); iter != m_map.end(); ++iter) {
            vUnreconciledList.push_back(iter->second);
        }

        auto iter = m_map.begin();
        auto error = iter->first->AllocSignalSyncObj(vUnreconciledList, &syncObj);
        PCHK_ERROR_AND_RETURN(error, "AllocSignalSyncObj");

        error = iter->first->RegisterSignalSyncObj(syncObj);
        PCHK_ERROR_AND_RETURN(error, "RegisterSignalSyncObj");

        for (; iter != m_map.end(); ++iter) {
            error = iter->first->ExportSignalSyncObj(syncObj);
            PCHK_ERROR_AND_RETURN(error, "ExportSignalSyncObj");
        }
    }

    return NvError_Success;
}
