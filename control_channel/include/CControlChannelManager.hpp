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

#ifndef CCONTROLCHANNELMANAGER_HPP
#define CCONTROLCHANNELMANAGER_HPP

#include <condition_variable>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <set>
#include "CMsgReader.hpp"
#include "CMsgWriter.hpp"
#ifdef NVMEDIA_QNX
#include "CIpcLinkQnx.hpp"
#else
#include "CIpcLinkLinux.hpp"
#endif
#include "CThreadPool.hpp"

/* <write endpoint name, read endpoint name> */
using EndpointPair = std::pair<std::string, std::string>;

typedef struct
{
    bool isCentralNode;
    bool isP2P;
    bool isC2C;
    bool isMasterSoc;
} ControlChannelConfig;

enum class NodeType : uint8_t
{
    MASTER_SOC_CENTRAL_NODE = 0,
    MASTER_SOC_NON_CENTRAL_NODE = 1,
    SLAVE_SOC_CENTRAL_NODE = 2,
    SLAVE_SOC_NON_CENTRAL_NODE = 3,
    UNKNOW_NODE_TYPE = 4
};

template <typename T> class CRouteTable
{
  public:
    void Insert(const std::string &sZoneName, std::shared_ptr<T> spZoneMember)
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        m_routeTable[sZoneName].insert(spZoneMember);
    };

    void clear() { m_routeTable.clear(); };

    void DeleteItem(std::shared_ptr<T> deleteItem)
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        for (auto &zone : m_routeTable) {
            auto it = std::find_if(zone.second.begin(), zone.second.end(),
                                   [deleteItem](const std::shared_ptr<T> &spMember) { return deleteItem == spMember; });
            if (it != zone.second.end()) {
                zone.second.erase(it);
            }
        }
    };

    bool empty() const { return m_routeTable.empty(); };

    std::set<std::shared_ptr<T>> &operator[](const std::string &sZoneName)
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        return m_routeTable[sZoneName];
    };

    void GetAllZones(std::set<std::string> &vZones)
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        for (const auto &entity : m_routeTable) {
            vZones.insert(entity.first);
        }
    }

  private:
    std::mutex m_mutex;
    // <Zone name, Zone members>
    std::unordered_map<std::string, std::set<std::shared_ptr<T>>> m_routeTable;
};

class CControlChannelManager : public CIpcLink::IMsgReadCallback, public CMsgWriter::IMsgWriteCallback
{
  public:
    CControlChannelManager(const ControlChannelConfig &cccfg);
    virtual ~CControlChannelManager() = default;

    NvError Init();
    void Deinit();
    NvError WaitSetupComplete(uint64_t uTimeoutMs);
    std::shared_ptr<CMsgReader> CreateReader(const std::string &sZoneName, MsgHandler msgHandler);
    std::shared_ptr<CMsgWriter> CreateWriter(const std::string &sZoneName);

  private:
    virtual NvError
    HandleEvent(std::shared_ptr<CIpcLink> spIpcLink, MessageHeader *pMsgHeader, void *pContentBuf) override;
    virtual NvError PostEvent(MessageHeader *pMsgHeader, void *pContentBuf) override;

    NvError HandleConnect(std::shared_ptr<CIpcLink> &spIpcLink, MessageHeader *pMsgHeader, void *pContentBuf);
    NvError HandleDisconnect(std::shared_ptr<CIpcLink> &spIpcLink);

    NvError HandleTopologyExport(std::shared_ptr<CIpcLink> &spIpcLink);
    NvError HandleTopologyImport(std::shared_ptr<CIpcLink> &spIpcLink, MessageHeader *pMsgHeader, void *pContentBuf);
    NvError HandleReaderImport(std::shared_ptr<CIpcLink> &spIpcLink, MessageHeader *pMsgHeader, void *pContentBuf);
    NvError
    HandleReaderExport(const std::shared_ptr<CIpcLink> &spExcludeIpcLink, MessageHeader *pMsgHeader, void *pContentBuf);
    NvError PostEventIntra(MessageHeader *pMsgHeader, void *pContentBuf);
    NvError PostEventIpc(MessageHeader *pMsgHeader, void *pContentBuf);
    NvError
    PostEventIpc(const std::shared_ptr<CIpcLink> &spExcludeIpcLink, MessageHeader *pMsgHeader, void *pContentBuf);

  private:
    ControlChannelConfig m_cccfg;
    NodeType m_nodeType;
    std::atomic<bool> m_bStop{ false };
    std::atomic<bool> m_bSetupDone{ false };
    std::mutex m_mutex;
    std::condition_variable m_cvSetupDone;
    std::shared_ptr<CIpcLink> m_spC2CLink{ nullptr };
    std::mutex m_ipcGroupMutex;
    std::vector<std::shared_ptr<CIpcLink>> m_vIpcGroup;
    CRouteTable<CIpcLink> m_ipcRouteTable;
    CRouteTable<CMsgReader> m_intraRouteTable;
    std::unique_ptr<CThreadPool> m_upThreadPool = { nullptr };
};

#endif
