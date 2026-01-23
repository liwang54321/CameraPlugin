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

#ifndef CSYNCAGGREGATOR_HPP
#define CSYNCAGGREGATOR_HPP

#include "CClientCommon.hpp"

#include <mutex>
#include <map>

class CSyncAggregator
{
  public:
    CSyncAggregator() {}
    CSyncAggregator(std::vector<std::shared_ptr<CClientCommon>> &vspClients);
    /** @brief Default destructor. */
    virtual ~CSyncAggregator() {}

    void AddClient(CClientCommon *pClient);
    inline const std::string GetName() { return "CSyncAggregator"; }
    NvError OnWaiterAttrEventRecvd(CClientCommon *pClient);

  private:
    bool AllReceived();

    std::map<CClientCommon *, NvSciSyncAttrList> m_map;
    std::mutex m_syncMutex;
};

#endif
