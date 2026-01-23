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

#include "CControlChannelManager.hpp"

#include <array>
#include <cstddef>
#include <future>
#include <string>

static constexpr uint32_t kCentralNodeThreadNum = 4U;
static constexpr uint32_t kNonCentralNodeThreadNum = 2U;

static constexpr std::size_t kNumEndpointPairs = 4;
#if NV_BUILD_DOS7
#define MULTICAST_IPC_PREFIX "nvsipl_multicast_ipc_"
std::array<std::array<std::string, 2>, kNumEndpointPairs> makeEndpointTable();
std::array<std::array<std::string, 2>, kNumEndpointPairs> makeEndpointTable() {
    std::array<std::array<std::string, 2>, kNumEndpointPairs> table{};
    for (size_t i = 0; i < kNumEndpointPairs; ++i) {
        table[i][0] = MULTICAST_IPC_PREFIX + std::to_string(i * 2);
        table[i][1] = MULTICAST_IPC_PREFIX + std::to_string(i * 2 + 1);
    }
    return table;
}
const auto sEndpointTable = makeEndpointTable();
#else
static constexpr std::array<std::array<const char*, 2>, kNumEndpointPairs> sEndpointTable{{
    {{"ipc_test_0", "ipc_test_1"}},
    {{"ipc_test_a_0", "ipc_test_a_1"}},
    {{"ipc_test_b_0", "ipc_test_b_1"}},
    {{"ipc_test_c_0", "ipc_test_c_1"}}
}};
#endif

#if NV_BUILD_DOS7
static constexpr const char *sC2CEP[2] = { "nvscic2c_pcie_s0_c5_11", "nvscic2c_pcie_s1_c5_11" };
#else
static constexpr const char *sC2CEP[2] = { "nvscic2c_pcie_s0_c5_11", "nvscic2c_pcie_s0_c6_11" };
#endif


CControlChannelManager::CControlChannelManager(const ControlChannelConfig &cccfg)
    : m_cccfg(cccfg)
{
    m_nodeType = (!cccfg.isC2C || cccfg.isMasterSoc) && cccfg.isCentralNode    ? NodeType::MASTER_SOC_CENTRAL_NODE
                 : (!cccfg.isC2C || cccfg.isMasterSoc) && !cccfg.isCentralNode ? NodeType::MASTER_SOC_NON_CENTRAL_NODE
                 : cccfg.isC2C && !cccfg.isMasterSoc && cccfg.isCentralNode    ? NodeType::SLAVE_SOC_CENTRAL_NODE
                 : cccfg.isC2C && !cccfg.isMasterSoc && !cccfg.isCentralNode   ? NodeType::SLAVE_SOC_NON_CENTRAL_NODE
                                                                               : NodeType::UNKNOW_NODE_TYPE;
}

NvError CControlChannelManager::Init()
{
    LOG_DBG("Enter: CControlChannelManager::Init()\n");

    m_upThreadPool =
        std::make_unique<CThreadPool>(m_cccfg.isCentralNode ? kCentralNodeThreadNum : kNonCentralNodeThreadNum);
    CHK_PTR_AND_RETURN(m_upThreadPool, "Create threadpool");
    auto error = m_upThreadPool->Init();
    CHK_ERROR_AND_RETURN(error, "CThreadPool::Init");

    if (!m_cccfg.isP2P) {
        m_bSetupDone = true;
        return NvError_Success;
    }

    auto CentralNodeOpenAllEndpoint = [this]() -> NvError {
        for (size_t i = 0; i < sEndpointTable.size(); i++) {
            auto spIpcLink = std::make_shared<CIpcLink>(this, true);
            CHK_PTR_AND_RETURN(spIpcLink, "Create Ipc link");
            auto error = spIpcLink->Init(sEndpointTable[i][0]);
            CHK_ERROR_AND_RETURN(error, "IpcLink::Init");
            error = spIpcLink->Start();
            CHK_ERROR_AND_RETURN(error, "IpcLink::Start");
            m_vIpcGroup.push_back(spIpcLink);
        }
        return NvError_Success;
    };

    auto NonCentralNodeOpenEndpoint = [this]() -> NvError {
        auto error = NvError_Success;
        auto spIpcLink = std::make_shared<CIpcLink>(this, false);
        CHK_PTR_AND_RETURN(spIpcLink, "Create Ipc link");
        for (size_t i = 0; i < sEndpointTable.size(); i++) {
            auto error = spIpcLink->Init(sEndpointTable[i][1]);
            if (NvError_ResourceError != error) {
                break;
            }
        }
        CHK_ERROR_AND_RETURN(error, "IpcLink::Init");
        error = spIpcLink->Start();
        CHK_ERROR_AND_RETURN(error, "IpcLink::Start");
        m_vIpcGroup.push_back(spIpcLink);
        return NvError_Success;
    };

    auto CreateC2CLink = [this](bool bMasterSoc) -> NvError {
        m_spC2CLink = std::make_shared<CIpcLink>(this, bMasterSoc ? true : false);
        CHK_PTR_AND_RETURN(m_spC2CLink, "Create C2C Ipc link");
        auto error = m_spC2CLink->Init(bMasterSoc ? sC2CEP[0] : sC2CEP[1]);
        CHK_ERROR_AND_RETURN(error, "C2C IpcLink::Init");
        error = m_spC2CLink->Start();
        CHK_ERROR_AND_RETURN(error, "C2C IpcLink::Start");
        m_vIpcGroup.push_back(m_spC2CLink);
        return NvError_Success;
    };

    switch (m_nodeType) {
        case NodeType::MASTER_SOC_CENTRAL_NODE:
            if (m_cccfg.isC2C) {
                error = CreateC2CLink(true);
                CHK_ERROR_AND_RETURN(error, "Create C2C link");
            }
            if (m_cccfg.isP2P) {
                error = CentralNodeOpenAllEndpoint();
                CHK_ERROR_AND_RETURN(error, "Create  link");
            }
            break;
        case NodeType::SLAVE_SOC_CENTRAL_NODE:
            error = CreateC2CLink(false);
            CHK_ERROR_AND_RETURN(error, "Create C2C link");
            error = CentralNodeOpenAllEndpoint();
            CHK_ERROR_AND_RETURN(error, "Create default link");
            break;
        case NodeType::MASTER_SOC_NON_CENTRAL_NODE:
        case NodeType::SLAVE_SOC_NON_CENTRAL_NODE:
            error = NonCentralNodeOpenEndpoint();
            CHK_ERROR_AND_RETURN(error, "Create default link");
            break;
        default:
            LOG_ERR("unknow node type: %d", m_nodeType);
            break;
    }

    // master soc central node connect done, other nodes need to wait connect to central node
    if (m_cccfg.isCentralNode && (!m_cccfg.isC2C || m_cccfg.isMasterSoc)) {
        m_bSetupDone = true;
    }
    LOG_DBG("Exit: CControlChannelManager::Init()\n");
    return NvError_Success;
}

NvError CControlChannelManager::WaitSetupComplete(uint64_t uTimeoutMs)
{
    std::unique_lock<std::mutex> lk(m_mutex);
    if (!m_bSetupDone) {
        m_cvSetupDone.wait_for(lk, std::chrono::milliseconds(uTimeoutMs));
        if (!m_bSetupDone) {
            return NvError_Timeout;
        }
    }
    return NvError_Success;
}

std::shared_ptr<CMsgReader> CControlChannelManager::CreateReader(const std::string &sZoneName, MsgHandler msgHandler)
{
    if (!m_bSetupDone) {
        LOG_ERR("Failed to create reader. Connect not done!\n");
        return nullptr;
    }

    std::shared_ptr<CMsgReader> spMsgReader(new CMsgReader(sZoneName, std::move(msgHandler)));
    std::string sZoneNameAfterCut =
        sZoneName.length() > kMaxZoneNameLength ? sZoneName.substr(0, kMaxZoneNameLength) : sZoneName;

    // add into route table
    m_intraRouteTable.Insert(sZoneNameAfterCut, spMsgReader);

    // export reader
    MessageHeader header(sZoneNameAfterCut, sizeof(MessageHeader), 0, MsgType::READER_CREATE_MSG);
    auto error = HandleReaderExport(nullptr, &header, nullptr);
    if (NvError_Success != error) {
        LOG_ERR("CControlChannelManager failed (0x%x) to HandleReaderExport\n", error);
        return nullptr;
    }
    return spMsgReader;
}

std::shared_ptr<CMsgWriter> CControlChannelManager::CreateWriter(const std::string &sZoneName)
{
    if (!m_bSetupDone) {
        LOG_ERR("Failed to create writer. Connect not done!\n");
        return nullptr;
    }
    std::shared_ptr<CMsgWriter> spMsgWriter(new CMsgWriter(sZoneName, this));
    return spMsgWriter;
}

NvError CControlChannelManager::HandleConnect(std::shared_ptr<CIpcLink> &spIpcLink,
                                              MessageHeader *pMsgHeader,
                                              void *pContentBuf)
{
    auto error = NvError_Success;
    switch (m_nodeType) {
        case NodeType::MASTER_SOC_CENTRAL_NODE:
            error = HandleTopologyExport(spIpcLink);
            CHK_ERROR_AND_RETURN(error, "HandleTopologyExport");
            break;
        case NodeType::SLAVE_SOC_CENTRAL_NODE:
            if (spIpcLink == m_spC2CLink) {
                error = HandleTopologyImport(spIpcLink, pMsgHeader, pContentBuf);
                CHK_ERROR_AND_RETURN(error, "HandleTopologyImport");
                m_bSetupDone = true;
                m_cvSetupDone.notify_all();
            } else {
                error = HandleTopologyExport(spIpcLink);
                CHK_ERROR_AND_RETURN(error, "HandleTopologyExport");
            }
            break;
        case NodeType::MASTER_SOC_NON_CENTRAL_NODE:
        case NodeType::SLAVE_SOC_NON_CENTRAL_NODE:
            error = HandleTopologyImport(spIpcLink, pMsgHeader, pContentBuf);
            CHK_ERROR_AND_RETURN(error, "HandleTopologyImport");
            m_bSetupDone = true;
            m_cvSetupDone.notify_all();
            break;
        default:
            LOG_ERR("unknow node type: %d", m_nodeType);
            break;
    }
    return NvError_Success;
}

NvError CControlChannelManager::HandleDisconnect(std::shared_ptr<CIpcLink> &spIpcLink)
{
    // remove from route table
    m_ipcRouteTable.DeleteItem(spIpcLink);
    return NvError_Success;
}

NvError CControlChannelManager::HandleTopologyExport(std::shared_ptr<CIpcLink> &spIpcLink)
{
    std::string sAllZone = "";
    if (!m_intraRouteTable.empty() || !m_ipcRouteTable.empty()) {
        // exchange topology info
        std::set<std::string> vZones;
        m_intraRouteTable.GetAllZones(vZones);
        m_ipcRouteTable.GetAllZones(vZones);
        for (const auto &sZone : vZones) {
            sAllZone += sZone + ",";
        }
        // trim end ','
        sAllZone = sAllZone.substr(0, sAllZone.length() - 1);
    }
    MessageHeader header("", sizeof(MessageHeader), sAllZone.length(), MsgType::CONNECT_MSG);
    auto error =
        spIpcLink->Write(&header, sAllZone.length() == 0 ? nullptr : reinterpret_cast<const void *>(sAllZone.c_str()));
    CHK_ERROR_AND_RETURN(error, "Write topology msg");
    return NvError_Success;
}

NvError CControlChannelManager::HandleTopologyImport(std::shared_ptr<CIpcLink> &spIpcLink,
                                                     MessageHeader *pMsgHeader,
                                                     void *pContentBuf)
{
    if (pMsgHeader->uContentLength == 0) {
        return NvError_Success;
    }
    std::string sAllZone(reinterpret_cast<const char *>(pContentBuf), pMsgHeader->uContentLength);
    std::vector<std::string> vZones(splitString(sAllZone, ','));
    for (const auto &sZone : vZones) {
        LOG_DBG("Import Zone name: %s\n", sZone.c_str());
        // add into route table
        m_ipcRouteTable.Insert(sZone, spIpcLink);
    }
    return NvError_Success;
}

NvError CControlChannelManager::HandleReaderImport(std::shared_ptr<CIpcLink> &spIpcLink,
                                                   MessageHeader *pMsgHeader,
                                                   void *pContentBuf)
{
    std::string sMsgZone = pMsgHeader->sZoneName;
    m_ipcRouteTable.Insert(sMsgZone, spIpcLink);
    return NvError_Success;
}

NvError CControlChannelManager::HandleReaderExport(const std::shared_ptr<CIpcLink> &spExcludeIpcLink,
                                                   MessageHeader *pMsgHeader,
                                                   void *pContentBuf)
{
    std::lock_guard<std::mutex> lk(m_ipcGroupMutex);
    for (auto &spIpcLink : m_vIpcGroup) {
        if (spIpcLink != spExcludeIpcLink && spIpcLink->IsConnected()) {
            auto error = spIpcLink->Write(pMsgHeader, pContentBuf);
            CHK_ERROR_AND_RETURN(error, "CIpcLink::Write");
        }
    }
    return NvError_Success;
}

NvError CControlChannelManager::PostEventIntra(MessageHeader *pMsgHeader, void *pContentBuf)
{
    std::string sMsgZone = pMsgHeader->sZoneName;
    using result_type = std::shared_ptr<std::future<NvError>>;
    std::vector<result_type> vResults;
    for (auto &spMsgReader : m_intraRouteTable[sMsgZone]) {
        auto result = m_upThreadPool->SubmitTask(
            [spMsgReader, pMsgHeader, pContentBuf] { return spMsgReader->ProcessMsg(pMsgHeader, pContentBuf); });
        CHK_PTR_AND_RETURN(result, "CThreadPool::SubmitTask");
        vResults.push_back(result);
    }

    return NvError_Success;
}

NvError CControlChannelManager::PostEventIpc(MessageHeader *pMsgHeader, void *pContentBuf)
{
    std::string sMsgZone = pMsgHeader->sZoneName;
    for (auto &ipcLink : m_ipcRouteTable[sMsgZone]) {
        if (ipcLink->IsConnected()) {
            auto error = ipcLink->Write(pMsgHeader, pContentBuf);
            CHK_ERROR_AND_RETURN(error, "CIpcLink::Write");
        }
    }
    return NvError_Success;
}

NvError CControlChannelManager::PostEventIpc(const std::shared_ptr<CIpcLink> &spExcludeIpcLink,
                                             MessageHeader *pMsgHeader,
                                             void *pContentBuf)
{
    std::string sMsgZone = pMsgHeader->sZoneName;
    for (auto &spIpcLink : m_ipcRouteTable[sMsgZone]) {
        if (spIpcLink != spExcludeIpcLink && spIpcLink->IsConnected()) {
            auto error = spIpcLink->Write(pMsgHeader, pContentBuf);
            CHK_ERROR_AND_RETURN(error, "CIpcLink::Write");
        }
    }
    return NvError_Success;
}

NvError
CControlChannelManager::HandleEvent(std::shared_ptr<CIpcLink> spIpcLink, MessageHeader *pMsgHeader, void *pContentBuf)
{
    switch (pMsgHeader->msgType) {
        case MsgType::CONNECT_MSG: {
            LOG_INFO("%s has connected!\n", spIpcLink->GetName().c_str());
            auto error = HandleConnect(spIpcLink, pMsgHeader, pContentBuf);
            CHK_ERROR_AND_RETURN(error, "HandleConnect");
            break;
        }
        case MsgType::DISCONNECT_MSG: {
            LOG_INFO("%s has disconnected!\n", spIpcLink->GetName().c_str());
            auto error = HandleDisconnect(spIpcLink);
            CHK_ERROR_AND_RETURN(error, "HandleDisconnect");
            break;
        }
        case MsgType::READER_CREATE_MSG: {
            LOG_INFO("%s received reader create msg!\n", spIpcLink->GetName().c_str());
            // insert into route table
            auto error = HandleReaderImport(spIpcLink, pMsgHeader, pContentBuf);
            CHK_ERROR_AND_RETURN(error, "HandleReaderImport");

            if (m_cccfg.isCentralNode) {
                // broadcast to other processes
                auto error = HandleReaderExport(spIpcLink, pMsgHeader, pContentBuf);
                CHK_ERROR_AND_RETURN(error, "HandleReaderImport");
            }
            break;
        }
        case MsgType::USER_MSG: {
            LOG_INFO("%s received user msg!\n", spIpcLink->GetName().c_str());
            // route to intra readers
            auto error = PostEventIntra(pMsgHeader, pContentBuf);
            CHK_ERROR_AND_RETURN(error, "PostEventIntra");

            if (m_cccfg.isCentralNode) {
                // route to other processes
                error = PostEventIpc(spIpcLink, pMsgHeader, pContentBuf);
                CHK_ERROR_AND_RETURN(error, "PostEventIntra");
            }
            break;
        }
        default:
            break;
    }
    return NvError_Success;
}

NvError CControlChannelManager::PostEvent(MessageHeader *pMsgHeader, void *pContentBuf)
{
    if (m_bStop) {
        LOG_ERR("Failed to post message. Control channel stopped.\n");
        return NvError_InvalidState;
    }
    switch (pMsgHeader->msgType) {
        case MsgType::USER_MSG: {
            // route to intra readers
            auto error = PostEventIntra(pMsgHeader, pContentBuf);
            CHK_ERROR_AND_RETURN(error, "PostEventIntra");
            // route to ipc readers
            error = PostEventIpc(pMsgHeader, pContentBuf);
            CHK_ERROR_AND_RETURN(error, "PostEventIpc");
            break;
        }
        default:
            break;
    }
    return NvError_Success;
}

void CControlChannelManager::Deinit()
{
    LOG_DBG("Enter: CControlChannelManager::Deinit()\n");
    m_bStop = true;

    for (auto &spIpcLink : m_vIpcGroup) {
        spIpcLink->Stop();
        spIpcLink->Deinit();
    }
    m_vIpcGroup.clear();

    m_upThreadPool->Deinit();
    m_upThreadPool.reset(nullptr);
    m_ipcRouteTable.clear();
    m_intraRouteTable.clear();

    LOG_DBG("Exit: CControlChannelManager::Deinit()\n");
}
