/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <algorithm>
#include <inttypes.h>

#include "CClientCommon.hpp"

CClientCommon::CClientCommon(std::shared_ptr<CClientCfg> spClientCfg, CClientCommon::IModuleCallback *pCallback)
    : m_vElemsInfos(*spClientCfg->m_pElementInfos)
    , m_pAppCfg(spClientCfg->m_pAppCfg)
    , m_spClientCfg(spClientCfg)
    , m_cpuWaitCfg(spClientCfg->m_cpuWaitCfg)
    , m_queueType(spClientCfg->m_queueType)
    , m_pModuleCallback(pCallback)
    , m_spControlChannel(spClientCfg->m_pAppCfg->m_spControlChannel)
    , m_spTimer(spClientCfg->m_pAppCfg->m_spTimer)
{
    for (uint32_t i = 0U; i < MAX_WAIT_SYNCOBJ; ++i) {
        for (uint32_t j = 0U; j < MAX_NUM_ELEMENTS; ++j) {
            m_waiterSyncObjs[i][j] = nullptr;
        }
    }

    for (uint32_t i = 0U; i < MAX_NUM_ELEMENTS; ++i) {
        m_signalerAttrLists[i] = nullptr;
        m_waiterAttrLists[i] = nullptr;
        m_signalSyncObjs[i] = nullptr;
    }

    for (uint32_t i = 0U; i < MAX_NUM_PACKETS; ++i) {
        m_packets[i].cookie = NvSciStreamCookie_Invalid;
        m_packets[i].handle = NvSciStreamPacket_Invalid;
        for (uint32_t j = 0U; j < MAX_NUM_ELEMENTS; ++j) {
            m_packets[i].bufObjs[j] = nullptr;
        }
        for (uint32_t j = 0U; j < MAX_NUM_CONSUMERS; ++j) {
            for (uint32_t k = 0U; k < MAX_NUM_ELEMENTS; ++k) {
                m_packets[i].preFence[j][k] = NvSciSyncFenceInitializer;
            }
        }
    }
}

CClientCommon::~CClientCommon()
{
    PLOG_DBG("ClientCommon release.\n");

    for (uint32_t i = 0U; i < MAX_WAIT_SYNCOBJ; ++i) {
        for (uint32_t j = 0U; j < MAX_NUM_ELEMENTS; ++j) {
            if (m_waiterSyncObjs[i][j] != nullptr) {
                NvSciSyncObjFree(m_waiterSyncObjs[i][j]);
                m_waiterSyncObjs[i][j] = nullptr;
            }
        }
    }

    for (uint32_t i = 0U; i < MAX_NUM_ELEMENTS; ++i) {
        if (m_signalerAttrLists[i] != nullptr) {
            NvSciSyncAttrListFree(m_signalerAttrLists[i]);
            m_signalerAttrLists[i] = nullptr;
        }

        if (m_waiterAttrLists[i] != nullptr) {
            NvSciSyncAttrListFree(m_waiterAttrLists[i]);
            m_waiterAttrLists[i] = nullptr;
        }

        if (m_signalSyncObjs[i] != nullptr) {
            NvSciSyncObjFree(m_signalSyncObjs[i]);
            m_signalSyncObjs[i] = nullptr;
        }
    }

    for (uint32_t i = 0U; i < MAX_NUM_PACKETS; ++i) {
        m_packets[i].cookie = NvSciStreamCookie_Invalid;
        m_packets[i].handle = NvSciStreamPacket_Invalid;
        for (uint32_t j = 0U; j < MAX_NUM_ELEMENTS; ++j) {
            if (m_packets[i].bufObjs[j] != nullptr) {
                NvSciBufObjFree(m_packets[i].bufObjs[j]);
                m_packets[i].bufObjs[j] = nullptr;
            }
        }
        for (uint32_t j = 0U; j < MAX_NUM_CONSUMERS; ++j) {
            for (uint32_t k = 0U; k < MAX_NUM_ELEMENTS; ++k) {
                NvSciSyncFenceClear(&m_packets[i].preFence[j][k]);
            }
        }
    }

    if (m_cpuWaitAttr != nullptr) {
        NvSciSyncAttrListFree(m_cpuWaitAttr);
        m_cpuWaitAttr = nullptr;
    }

    if (m_cpuWaitPreContext != nullptr) {
        NvSciSyncCpuWaitContextFree(m_cpuWaitPreContext);
        m_cpuWaitPreContext = nullptr;
    }

    if (m_cpuWaitPostContext != nullptr) {
        NvSciSyncCpuWaitContextFree(m_cpuWaitPostContext);
        m_cpuWaitPostContext = nullptr;
    }
}

NvError CClientCommon::Init()
{
    PLOG_DBG("Enter: CClientCommon::Init()\n");

    m_upEventHandler = std::make_unique<CEventHandler<CClientCommon>>();
    auto error = m_upEventHandler->RegisterHandler(&CClientCommon::HandleConnectEvent, this);
    PCHK_ERROR_AND_RETURN(error, "RegisterHandler");
    error = m_upEventHandler->StartThread();
    PCHK_ERROR_AND_RETURN(error, "StartThread");

    PLOG_DBG("Exit: CClientCommon::Init()\n");
    return NvError_Success;
}

void CClientCommon::DeInit()
{
    PLOG_DBG("Enter: CClientCommon::DeInit()\n");

    if (m_upEventHandler.get()) {
        m_upEventHandler->QuitThread();
    }

    PLOG_DBG("Exit: CClientCommon::DeInit()\n");
}

NvError CClientCommon::Reconcile()
{
    PLOG_DBG("Enter: CClientCommon::Reconcile()\n");

    auto error = HandleElemSupport();
    PCHK_ERROR_AND_RETURN(error, "HandleElemSupport");

    error = m_upEventHandler->RegisterHandler(&CClientCommon::HandleClientEvent, this);
    PCHK_ERROR_AND_RETURN(error, "RegisterHandler");
    error = m_upEventHandler->StartThread();
    PCHK_ERROR_AND_RETURN(error, "StartThread");

    PLOG_DBG("Exit: CClientCommon::Reconcile()\n");
    return NvError_Success;
}

NvError CClientCommon::Start()
{
    PLOG_DBG("Enter: CClientCommon::Start()\n");

    if (m_bStop) {
        m_bStop = false;
    }

    PLOG_DBG("Exit: CClientCommon::Start()\n");
    return NvError_Success;
}

NvError CClientCommon::PreStop()
{
    PLOG_DBG("Enter: CClientCommon::PreStop()\n");

    m_bStop = true;

    PLOG_DBG("Exit: CClientCommon::PreStop()\n");
    return NvError_Success;
}

NvError CClientCommon::Stop()
{
    PLOG_DBG("Enter: CClientCommon::Stop()\n");

    PLOG_DBG("Exit: CClientCommon::Stop()\n");
    return NvError_Success;
}

NvError CClientCommon::PostStop()
{
    PLOG_DBG("Enter: CClientCommon::PostStop()\n");

    PLOG_DBG("Exit: CClientCommon::PostStop()\n");
    return NvError_Success;
}

EventStatus CClientCommon::HandleEvent()
{
    NvSciStreamEventType event = NvSciStreamEventType_Error;
    int error = NvError_Success;
    NvSciError sciStatus = NvSciError_Unknown;
    EventStatus eventStatus = EventStatus::OK;

    auto sciErr = NvSciStreamBlockEventQuery(m_handle, QUERY_TIMEOUT, &event);

    if (NvSciError_Timeout == sciErr) {
        PLOG_WARN("Event query, timed out.\n");
        return EventStatus::TIMED_OUT;
    } else if (sciErr != NvSciError_Success) {
        PLOG_ERR("Event query, failed with error 0x%x\n", sciErr);
        return EventStatus::ERROR;
    }

    switch (event) {
        /* Process all element support from producer and consumer(s) */
        case NvSciStreamEventType_Elements:
            PLOG_DBG("HandleEvent, received NvSciStreamEventType_Elements.\n");
            error = HandleElemSetting();
            if (error == NvError_Success)
                error = HandleSyncSupport();
            eventStatus = (error == NvError_Success) ? EventStatus::OK : EventStatus::ERROR;
            break;
        case NvSciStreamEventType_PacketCreate:
            /* Handle creation of a new packet */
            PLOG_DBG("HandleEvent, received NvSciStreamEventType_PacketCreate.\n");
            error = HandlePacketCreate();
            eventStatus = (error == NvError_Success) ? EventStatus::OK : EventStatus::ERROR;
            break;
        case NvSciStreamEventType_PacketsComplete:
            /* Handle packet complete*/
            PLOG_DBG("HandleEvent, received NvSciStreamEventType_PacketsComplete.\n");
            sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_PacketImport, true);
            if (NvSciError_Success != sciErr) {
                error = NvError_InvalidState;
            }
            eventStatus = (error == NvError_Success) ? EventStatus::OK : EventStatus::ERROR;
            break;
        case NvSciStreamEventType_PacketDelete:
            PLOG_WARN("HandleEvent, received NvSciStreamEventType_PacketDelete.\n");
            break;
            /* Set up signaling sync object from consumer's wait attributes */
        case NvSciStreamEventType_WaiterAttr:
            PLOG_DBG("HandleEvent, received NvSciStreamEventType_WaiterAttr.\n");
            error = HandleSyncExport();
            eventStatus = (error == NvError_Success) ? EventStatus::OK : EventStatus::ERROR;
            break;
            /* Import consumer sync objects for all elements */
        case NvSciStreamEventType_SignalObj:
            PLOG_DBG("HandleEvent, received NvSciStreamEventType_SignalObj.\n");
            error = HandleSyncImport();
            eventStatus = (error == NvError_Success) ? EventStatus::OK : EventStatus::ERROR;
            break;
            /* All setup complete. Transition to runtime phase */
        case NvSciStreamEventType_SetupComplete:
            PLOG_DBG("HandleEvent, received NvSciStreamEventType_SetupComplete.\n");
            error = HandleSetupComplete();
            PLOG_DBG("HandleEvent, Setup completed.\n");
            if (error == NvError_Success)
                return EventStatus::RECONCILED;
            break;
        /* Processs payloads when packets arrive */
        case NvSciStreamEventType_PacketReady:
            PLOG_DBG("HandleEvent, received NvSciStreamEventType_PacketReady.\n");
            error = HandlePayload();
            eventStatus = (error == NvError_Success) ? EventStatus::OK : EventStatus::ERROR;
            break;

        case NvSciStreamEventType_Error:
            PLOG_ERR("HandleEvent, received NvSciStreamEventType_Error.\n");
            sciErr = NvSciStreamBlockErrorGet(m_handle, &sciStatus);
            if (NvSciError_Success != sciErr) {
                PLOG_ERR("Failed to query the error event code 0x%x\n", sciErr);
            } else {
                PLOG_ERR("Received error event: 0x%x\n", sciStatus);
            }
            eventStatus = EventStatus::ERROR;
            break;
        case NvSciStreamEventType_Disconnected:
            LOG_MSG("HandleEvent, received NvSciStreamEventType_Disconnected.\n");
            return EventStatus::DISCONNECT;
        default:
            PLOG_ERR("Received unknown event 0x%x\n", event);
            eventStatus = EventStatus::ERROR;
            break;
    }

    PLOG_DBG("HandleEvent() return error %u\n", error);
    return eventStatus;
}

NvError CClientCommon::GetElemIdByUserType(PacketElementType userType, uint32_t &uElementId)
{
    auto lambda = [userType](const ElementInfo &info) {
        return (userType == PacketElementType::DEFAULT) ? (info.userType != PacketElementType::METADATA && info.bIsUsed)
                                                        : (info.userType == userType);
    };

    auto it = std::find_if(m_vElemsInfos.begin(), m_vElemsInfos.end(), lambda);

    if (m_vElemsInfos.end() == it) {
        PLOG_ERR("Can't find the element! userType: %u\n", userType);
        return NvError_BadParameter;
    }

    uElementId = std::distance(m_vElemsInfos.begin(), it);

    return NvError_Success;
}

NvError CClientCommon::GetElemIndexByUserType(PacketElementType userType, uint32_t &uElementIndex)
{
    uint32_t uElemId = MAX_NUM_ELEMENTS;

    auto error = GetElemIdByUserType(userType, uElemId);
    PCHK_ERROR_AND_RETURN(error, "GetElemIdByUserType");

    uElementIndex = m_vElemsInfos[uElemId].uIndex;
    return NvError_Success;
}

NvError CClientCommon::FillBufAttrList(PacketElementType userType, NvSciBufAttrList *pBufAttrList)
{
    if (userType != PacketElementType::METADATA) {
        auto error = m_pModuleCallback->FillDataBufAttrList(this, userType, pBufAttrList);
        PCHK_ERROR_AND_RETURN(error, "FillDataBufAttrList");
    } else {
        auto error = m_pModuleCallback->FillMetaBufAttrList(this, pBufAttrList);
        PCHK_ERROR_AND_RETURN(error, "FillMetaBufAttrList");
    }

    return NvError_Success;
}

NvError CClientCommon::HandleElemSupport()
{
    NvSciBufAttrList bufAttrList;

    // Set the packet element attributes one by one.
    for (uint32_t i = 0U; i < m_vElemsInfos.size(); ++i) {
        auto sciErr = NvSciBufAttrListCreate(m_pAppCfg->m_sciBufModule, &bufAttrList);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListCreate.");

        auto error = FillBufAttrList(m_vElemsInfos[i].userType, &bufAttrList);
        PCHK_ERROR_AND_RETURN(error, "FillBufAttrList");

        sciErr = NvSciStreamBlockElementAttrSet(m_handle, static_cast<uint32_t>(m_vElemsInfos[i].userType), bufAttrList);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockElementAttrSet");
        PLOG_DBG("Set element: %u attributes.\n", i);

        NvSciBufAttrListFree(bufAttrList);
    }

    // Indicate that all element information has been exported
    auto sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_ElementExport, true);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockSetupStatusSet");

    return NvError_Success;
}

NvError CClientCommon::SetCpuSyncAttrList(NvSciSyncAttrList attrList, NvSciSyncAccessPerm cpuPerm, bool bCpuSync)
{
    /* Fill attribute list for CPU waiting */
    NvSciSyncAttrKeyValuePair cpuKeyVals[] = { { NvSciSyncAttrKey_NeedCpuAccess, &bCpuSync, sizeof(bCpuSync) },
                                               { NvSciSyncAttrKey_RequiredPerm, &cpuPerm, sizeof(cpuPerm) } };

    auto sciErr = NvSciSyncAttrListSetAttrs(attrList, cpuKeyVals, ARRAY_SIZE(cpuKeyVals));
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListSetAttrs");

    return NvError_Success;
}

NvError CClientCommon::FillSyncWaiterAttrList(PacketElementType userType, NvSciSyncAttrList *pWaiterAttrList)
{
    auto error = m_pModuleCallback->FillSyncWaiterAttrList(this, userType, pWaiterAttrList);
    return error;
}

// Create and set CPU signaler and waiter attribute lists.
NvError CClientCommon::HandleSyncSupport()
{
    /* To support SC7,  m_cpuWaitPreContext must be used.*/
    auto sciErr = NvSciSyncAttrListCreate(m_pAppCfg->m_sciSyncModule, &m_cpuWaitAttr);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListCreate");

    auto error = SetCpuSyncAttrList(m_cpuWaitAttr, NvSciSyncAccessPerm_WaitOnly, true);
    PCHK_ERROR_AND_RETURN(error, "SetCpuSyncAttrList");

    /* Create a context for CPU waiting */
    sciErr = NvSciSyncCpuWaitContextAlloc(m_pAppCfg->m_sciSyncModule, &m_cpuWaitPreContext);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncCpuWaitContextAlloc");

    sciErr = NvSciSyncCpuWaitContextAlloc(m_pAppCfg->m_sciSyncModule, &m_cpuWaitPostContext);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncCpuWaitContextAlloc");

    for (uint32_t i = 0U; i < m_vElemsInfos.size(); ++i) {
        if (m_vElemsInfos[i].bIsUsed && m_vElemsInfos[i].userType != PacketElementType::METADATA) {
            std::unique_ptr<NvSciSyncAttrList, std::function<void(NvSciSyncAttrList *)>> upWaiterAttrList(
                new NvSciSyncAttrList(nullptr), [](NvSciSyncAttrList *attrList) {
                    if (attrList)
                        NvSciSyncAttrListFree(*attrList);
                    delete attrList;
                });
            auto sciErr = NvSciSyncAttrListCreate(m_pAppCfg->m_sciSyncModule, upWaiterAttrList.get());
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListCreate");

            if (m_cpuWaitCfg.bWaitPrefence) {
                auto error = SetCpuSyncAttrList(*upWaiterAttrList.get(), NvSciSyncAccessPerm_WaitOnly, true);
                PCHK_ERROR_AND_RETURN(error, "SetCpuSyncAttrList");
            } else {
                auto error = FillSyncWaiterAttrList(m_vElemsInfos[i].userType, upWaiterAttrList.get());
                PCHK_ERROR_AND_RETURN(error, "FillSyncWaiterAttrList");
            }

            sciErr = NvSciStreamBlockElementWaiterAttrSet(m_handle, m_vElemsInfos[i].uIndex, *upWaiterAttrList);
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockElementWaiterAttrSet");
        }
    }

    /* Indicate that waiter attribute export is done. */
    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_WaiterAttrExport, true);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Complete waiter attr export");
    return NvError_Success;
}

NvSciError CClientCommon::IsPostFenceExpired(NvSciSyncFence *pFence)
{
    return NvSciSyncFenceWait(pFence, m_cpuWaitPostContext, 0U);
}

NvError CClientCommon::OnBufAttrListRecvd(const ElemBufAttr (&elemBufAttrs)[MAX_NUM_ELEMENTS], const uint32_t uNumElems)
{
    for (auto i = 0U; i < uNumElems; ++i) {
        if (elemBufAttrs[i].userType != PacketElementType::METADATA) {
            auto error = m_pModuleCallback->OnDataBufAttrListRecvd(this, elemBufAttrs[i].bufAttrList);
            PCHK_ERROR_AND_RETURN(error, "m_pModuleCallback->OnDataBufAttrListRecvd");
        }
    }

    return NvError_Success;
}

NvError CClientCommon::HandleElemSetting()
{
    ElemBufAttr elemBufAttrs[MAX_NUM_ELEMENTS];
    uint32_t uNumElems = 0U;
    uint32_t uRecvdElemCount = 0U;
    uint32_t uUserType;

    auto sciErr = NvSciStreamBlockElementCountGet(m_handle, NvSciStreamBlockType_Pool, &uRecvdElemCount);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockElementCountGet failed");

    for (uint32_t i = 0U; i < uRecvdElemCount; ++i) {
        auto sciErr = NvSciStreamBlockElementAttrGet(m_handle, NvSciStreamBlockType_Pool, i, &uUserType,
                                                     &elemBufAttrs[uNumElems].bufAttrList);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockElementAttrGet");

        uint32_t j = 0U;
        for (; j < m_vElemsInfos.size(); ++j) {
            if (uUserType == static_cast<uint32_t>(m_vElemsInfos[j].userType)) {
                m_vElemsInfos[j].uIndex = i;
                break;
            }
        }

        if (j == m_vElemsInfos.size() || !m_vElemsInfos[j].bIsUsed) {
            if (elemBufAttrs[uNumElems].bufAttrList) {
                NvSciBufAttrListFree(elemBufAttrs[uNumElems].bufAttrList);
                elemBufAttrs[uNumElems].bufAttrList = nullptr;
            }
            auto error = SetUnusedElement(i);
            PCHK_ERROR_AND_RETURN(error, "SetUnusedElement");
            continue;
        }

        elemBufAttrs[uNumElems].userType = static_cast<PacketElementType>(uUserType);
        uNumElems++;
    }

    auto error = GetElemIdByUserType(PacketElementType::DEFAULT, m_uDataElemId);
    PCHK_ERROR_AND_RETURN(error, "GetElemIdByUserType");

    error = OnBufAttrListRecvd(elemBufAttrs, uNumElems);
    PCHK_ERROR_AND_RETURN(error, "OnBufAttrListRecvd");

    /* Indicate that element import is complete */
    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_ElementImport, true);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Complete element import");

    return NvError_Success;
}

NvError CClientCommon::HandlePacketCreate()
{
    /* Retrieve handle for packet pending creation */
    NvSciStreamPacket packetHandle = NvSciStreamPacket_Invalid;
    uint32_t uPacketIndex = 0;
    std::vector<ElemBufObj> vElemBufObjs{};

    auto sciErr = NvSciStreamBlockPacketNewHandleGet(m_handle, &packetHandle);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Retrieve handle for the new packet");

    /* Make sure there is room for more packets */
    if (MAX_NUM_PACKETS <= m_uNumPacket) {
        PLOG_ERR("Exceeded max packets\n");
        sciErr =
            NvSciStreamBlockPacketStatusSet(m_handle, packetHandle, NvSciStreamCookie_Invalid, NvSciError_Overflow);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Inform pool of packet error");
    }

    m_uNumPacket++;
    PLOG_DBG("Received PacketCreate from pool, m_numPackets: %" PRIu32, m_uNumPacket);

    NvSciStreamCookie cookie = AssignPacketCookie();
    ClientPacket *pPacket = GetPacketByCookie(cookie);
    PCHK_PTR_AND_RETURN(pPacket, "Get packet by cookie");
    pPacket->cookie = cookie;
    pPacket->handle = packetHandle;

    auto error = GetIndexFromCookie(cookie, uPacketIndex);
    PCHK_ERROR_AND_RETURN(error, "GetIndexFromCookie");

    for (uint32_t i = 0U; i < m_vElemsInfos.size(); ++i) {
        if (!m_vElemsInfos[i].bIsUsed) {
            continue;
        }
        /* Retrieve all buffers and map into application */
        NvSciBufObj bufObj = nullptr;
        sciErr = NvSciStreamBlockPacketBufferGet(m_handle, packetHandle, m_vElemsInfos[i].uIndex, &bufObj);
        if (NvSciError_Success != sciErr) {
            PLOG_ERR("Failed (0x%x) to retrieve buffer (0x%lx)\n", sciErr, packetHandle);
            return NvError_BadParameter;
        }

        pPacket->bufObjs[i] = bufObj;

        vElemBufObjs.push_back({ m_vElemsInfos[i].userType, bufObj });

        if (m_vElemsInfos[i].userType != PacketElementType::METADATA) {
            MapDataBuffer(m_vElemsInfos[i].userType, uPacketIndex, bufObj);
            m_bufObjPacketIndexMap.emplace(bufObj, uPacketIndex);
        } else {
            error = MapMetaBuffer(uPacketIndex, bufObj);
            PCHK_ERROR_AND_RETURN(error, "MapMetaBuffer");
        }
    }

    error = OnPacketCreated(vElemBufObjs);
    PCHK_ERROR_AND_RETURN(error, "OnPacketCreated");

    sciErr = NvSciStreamBlockPacketStatusSet(m_handle, pPacket->handle, cookie, NvSciError_Success);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Inform pool of packet error");
    PLOG_DBG("Set packet status success, cookie: %lu.\n", cookie);

    return NvError_Success;
}

NvError CClientCommon::AllocSignalSyncObj(std::vector<NvSciSyncAttrList> &vUnreconciledAttrLists,
                                          NvSciSyncObj *pSignalSyncObj)
{
    return AllocSignalSyncObj(vUnreconciledAttrLists, m_uDataElemId, pSignalSyncObj);
}

NvError CClientCommon::AllocSignalSyncObj(std::vector<NvSciSyncAttrList> &vUnreconciledAttrLists,
                                          uint32_t uElemId,
                                          NvSciSyncObj *pSignalSyncObj)
{
    NvSciSyncAttrList signalerAttrList;

    if (m_cpuWaitAttr != nullptr) {
        vUnreconciledAttrLists.push_back(m_cpuWaitAttr);
    }

    auto sciErr = NvSciSyncAttrListCreate(m_pAppCfg->m_sciSyncModule, &signalerAttrList);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListCreate");

    auto error = m_pModuleCallback->FillSyncSignalerAttrList(this, m_vElemsInfos[uElemId].userType, &signalerAttrList);
    PCHK_ERROR_AND_RETURN(error, "FillSyncSignalerAttrList");

    vUnreconciledAttrLists.push_back(signalerAttrList);

    NvSciSyncAttrList reconciled = nullptr;
    NvSciSyncAttrList conflicts = nullptr;

    sciErr = NvSciSyncAttrListReconcile(vUnreconciledAttrLists.data(), vUnreconciledAttrLists.size(), &reconciled,
                                        &conflicts);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListReconcile");

    for (NvSciSyncAttrList syncAttrList : vUnreconciledAttrLists) {
        //m_cpuWaitAttr will be freed later
        if (syncAttrList != m_cpuWaitAttr) {
            NvSciSyncAttrListFree(syncAttrList);
        }
    }

    /* Allocate sync object */
    sciErr = NvSciSyncObjAlloc(reconciled, pSignalSyncObj);
    NvSciSyncAttrListFree(reconciled);
    NvSciSyncAttrListFree(conflicts);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncObjAlloc");

    m_signalSyncObjs[uElemId] = *pSignalSyncObj;

    return NvError_Success;
}

NvError CClientCommon::RegisterSignalSyncObj(NvSciSyncObj signalSyncObj)
{
    return RegisterSignalSyncObj(m_uDataElemId, signalSyncObj);
}

NvError CClientCommon::RegisterSignalSyncObj(uint32_t uElemId, NvSciSyncObj signalSyncObj)
{
    auto error = m_pModuleCallback->RegisterSignalSyncObj(this, m_vElemsInfos[uElemId].userType, signalSyncObj);
    PCHK_ERROR_AND_RETURN(error, "RegisterSignalSyncObj");

    return NvError_Success;
}

NvError CClientCommon::RecvWaiterAttr(ElemSyncAttr &elemSyncAttr)
{
    auto sciErr =
        NvSciStreamBlockElementWaiterAttrGet(m_handle, m_vElemsInfos[m_uDataElemId].uIndex, &elemSyncAttr.syncAttrList);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockElementWaiterAttrGet");
    elemSyncAttr.userType = m_vElemsInfos[m_uDataElemId].userType;

    /* Indicate that waiter attribute import is done. */
    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_WaiterAttrImport, true);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Complete waiter attr import");

    return NvError_Success;
}

NvError CClientCommon::ExportSignalSyncObj(NvSciSyncObj syncObj)
{
    //Export the sync obj from downstream to upstream
    auto error = SetSignalSyncObj(m_uDataElemId, syncObj);
    PCHK_ERROR_AND_RETURN(error, "SetSignalObject");

    auto sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_SignalObjExport, true);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Complete signal obj export");

    return NvError_Success;
}

NvError CClientCommon::HandleSyncExport()
{
    auto sciErr = NvSciError_Success;

    std::vector<uint32_t> vuProcessedIds;
    bool bHandled = false;

    auto error = m_pModuleCallback->OnWaiterAttrEventRecvd(this, bHandled);
    PCHK_ERROR_AND_RETURN(error, "m_pModuleCallback->OnWaiterAttrEventRecvd");
    if (bHandled) {
        return NvError_Success;
    }

    for (uint32_t i = 0U; i < m_vElemsInfos.size(); ++i) {
        if (m_vElemsInfos[i].userType == PacketElementType::METADATA || !m_vElemsInfos[i].bIsUsed ||
            vuProcessedIds.end() != std::find(vuProcessedIds.begin(), vuProcessedIds.end(), i)) {
            continue;
        }

        // Merge and reconcile sync attrs.
        std::vector<NvSciSyncAttrList> vUnreconciledAttrLists;
        error = CollectWaiterAttrList(i, vUnreconciledAttrLists);
        PCHK_ERROR_AND_RETURN(error, "CollectWaiterAttrList");

        // If it has slbling, collect the waiter attribute list one by one.
        // For example, there is only one shared sync object for ISP0&ISP1 buffer.
        if (m_vElemsInfos[i].bHasSibling) {
            for (uint32_t j = i + 1; j < m_vElemsInfos.size(); ++j) {
                if (!m_vElemsInfos[j].bHasSibling) {
                    continue;
                }

                error = CollectWaiterAttrList(j, vUnreconciledAttrLists);
                PCHK_ERROR_AND_RETURN(error, "CollectWaiterAttrList");
                vuProcessedIds.push_back(j);
            }
        }

        if (vUnreconciledAttrLists.empty()) {
            continue;
        }
        NvSciSyncObj &signalSyncObj = m_signalSyncObjs[i];
        error = AllocSignalSyncObj(vUnreconciledAttrLists, i, &signalSyncObj);
        PCHK_ERROR_AND_RETURN(error, "AllocSignalSyncObj");

        error = RegisterSignalSyncObj(i, signalSyncObj);
        PCHK_ERROR_AND_RETURN(error, "RegisterSignalObject");

        error = SetSignalSyncObj(i, signalSyncObj);
        PCHK_ERROR_AND_RETURN(error, "SetSignalSyncObj");

        // If it has sibling, set the same sync object.
        if (m_vElemsInfos[i].bHasSibling) {
            for (uint32_t k = i + 1; k < m_vElemsInfos.size(); ++k) {
                if (!m_vElemsInfos[k].bHasSibling) {
                    continue;
                }

                error = SetSignalSyncObj(k, signalSyncObj);
                PCHK_ERROR_AND_RETURN(error, "SetSignalSyncObj");
            }
        }
    }

    /* Indicate that waiter attribute import is done. */
    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_WaiterAttrImport, true);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Complete waiter attr import");

    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_SignalObjExport, true);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Complete signal obj export");

    return NvError_Success;
}

NvError CClientCommon::RegisterWaiterSyncObj(uint32_t uElemId, NvSciSyncObj waiterSyncObj)
{
    PLOG_DBG("Enter RegisterWaiterSyncObj\n");

    auto error = m_pModuleCallback->RegisterWaiterSyncObj(this, m_vElemsInfos[uElemId].userType, waiterSyncObj);
    PCHK_ERROR_AND_RETURN(error, "RegisterWaiterSyncObj");

    PLOG_DBG("Exit RegisterWaiterSyncObj\n");
    return NvError_Success;
}

NvError CClientCommon::HandleSyncImport()
{
    NvSciError sciErr = NvSciError_Success;

    /* Query sync objects for each element from the other endpoint */
    for (uint32_t i = 0U; i < GetWaitSyncObjCount(); ++i) {
        for (uint32_t j = 0U; j < m_vElemsInfos.size(); ++j) {
            if (PacketElementType::METADATA == m_vElemsInfos[j].userType || !m_vElemsInfos[j].bIsUsed) {
                continue;
            }

            NvSciSyncObj waiterObj = nullptr;
            sciErr = NvSciStreamBlockElementSignalObjGet(m_handle, i, m_vElemsInfos[j].uIndex, &waiterObj);
            if (NvSciError_Success != sciErr) {
                PLOG_ERR("Failed (0x%x) to query sync obj from index %u, element id %u\n", sciErr, i, j);
                return NvError_BadParameter;
            }

            // If producer has the elements that the customer dose not need to sync,
            // the waiter obj will be null, then we shouldn't register it.
            if (waiterObj) {
                if (m_waiterSyncObjs[i][j]) {
                    NvSciSyncObjFree(m_waiterSyncObjs[i][j]);
                }
                m_waiterSyncObjs[i][j] = waiterObj;
                if (!m_cpuWaitCfg.bWaitPrefence) {
                    auto error = RegisterWaiterSyncObj(j, waiterObj);
                    PCHK_ERROR_AND_RETURN(error, "RegisterWaiterSyncObj");
                }
            } else {
                PLOG_DBG("Null sync obj for element type %u\n", m_vElemsInfos[j].userType);
            }
        }
    }

    /* Indicate that element import is complete */
    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_SignalObjImport, true);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Complete signal obj import");

    return NvError_Success;
}

NvError CClientCommon::CollectWaiterAttrList(uint32_t uElementId,
                                             std::vector<NvSciSyncAttrList> &vUnreconciledAttrLists)
{
    NvSciSyncAttrList attrList = nullptr;
    auto sciErr = NvSciStreamBlockElementWaiterAttrGet(m_handle, m_vElemsInfos[uElementId].uIndex, &attrList);
    if (NvSciError_Success != sciErr) {
        PLOG_ERR("Failed (0x%x) to get waiter attr, element id %u\n", sciErr, uElementId);
        return NvError_BadParameter;
    }

    if (attrList != nullptr) {
        vUnreconciledAttrLists.push_back(attrList);
    } else {
        // If both producer and consumer have not set sync attribute list for this element,
        // we will get null attribute list, and then skipping the sync object creation of this element.
        PLOG_DBG("Get null waiter attr, element id %u\n", uElementId);
    }

    return NvError_Success;
}

NvError CClientCommon::SetSignalSyncObj(uint32_t uElementId, NvSciSyncObj signalSyncObj)
{
    auto sciErr = NvSciStreamBlockElementSignalObjSet(m_handle, m_vElemsInfos[uElementId].uIndex, signalSyncObj);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Send sync object");

    return NvError_Success;
}

// Create client buffer objects from NvSciBufObj
NvError CClientCommon::MapMetaBuffer(uint32_t uPacketIndex, NvSciBufObj bufObj)
{
    PLOG_DBG("Mapping meta buffer, uPacketIndex: %u.\n", uPacketIndex);

    if (m_pAppCfg->IsProfilingEnabled()) {
        /*
         * The modification of meta data is needed for recording the sending and receiving
         * time marks.
         */
        auto sciErr = NvSciBufObjGetCpuPtr(bufObj, (void **)&m_metaPtrs[uPacketIndex]);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjGetCpuPtr");
    } else {
        auto sciErr = NvSciBufObjGetConstCpuPtr(bufObj, (void const **)&m_metaPtrs[uPacketIndex]);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufObjGetConstCpuPtr");
    }

    return NvError_Success;
}

void CClientCommon::MapDataBuffer(PacketElementType userType, uint32_t uPacketIndex, NvSciBufObj bufObj)
{
    m_pModuleCallback->RegisterBufObj(this, userType, uPacketIndex, bufObj);
}

NvSciBufObj *CClientCommon::GetBufObj(uint32_t uPacketIndex)
{
    if (uPacketIndex >= MAX_NUM_PACKETS || m_uDataElemId >= MAX_NUM_ELEMENTS) {
        PLOG_ERR("GetBufObj, uPacketIndex: %u, m_uDataElemId: %u\n", uPacketIndex, m_uDataElemId);
        return nullptr;
    }

    return &m_packets[uPacketIndex].bufObjs[m_uDataElemId];
}

MetaData *CClientCommon::GetMetaPtr(NvSciBufObj bufObj)
{
    auto it = m_bufObjPacketIndexMap.find(bufObj);
    return it != m_bufObjPacketIndexMap.end() ? reinterpret_cast<MetaData *>(GetMetaPtr(it->second)) : nullptr;
}

NvError CClientCommon::MapPayload(void *pBuffer, uint32_t &uPacketIndex, uint32_t &uElementId)
{
    NvSciBufObj bufobj = *(static_cast<NvSciBufObj *>(pBuffer));
    for (uint32_t i = 0U; i < MAX_NUM_PACKETS; ++i) {
        for (uint32_t j = 0U; j < MAX_NUM_ELEMENTS; ++j) {
            if (m_packets[i].bufObjs[j] == bufobj) {
                uPacketIndex = i;
                uElementId = j;
                return NvError_Success;
            };
        }
    }
    return NvError_BadParameter;
}

NvError ClientConnect(std::shared_ptr<CClientCommon> spSrcClient, std::shared_ptr<CClientCommon> spDstClient)
{
    NvError error = NvError_Success;
    NvSciStreamBlock srcBlock;
    NvSciStreamBlock dstBlock;

    LOG_DBG("Enter: ClientConnect()\n");

    error = spSrcClient->GetConnectBlock(&srcBlock);
    CHK_ERROR_AND_RETURN(error, "spSrcClient->GetConnectBlock");

    error = spDstClient->GetConnectBlock(&dstBlock);
    CHK_ERROR_AND_RETURN(error, "spDstClient->GetConnectBlock");

    auto sciErr = NvSciStreamBlockConnect(srcBlock, dstBlock);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Connect blocks: srcBlock - dstBlock");

    LOG_DBG("Exit: ClientConnect()\n");

    return NvError_Success;
}

EventStatus CClientCommon::CheckConnection()
{
    for (auto &blockPair : m_vBlockPairs) {
        if (!blockPair.first) {
            NvSciStreamEventType event = NvSciStreamEventType_Error;
            NvSciError sciErr = NvSciStreamBlockEventQuery(blockPair.second, QUERY_TIMEOUT, &event);
            if (sciErr == NvSciError_Success) {
                blockPair.first = (event == NvSciStreamEventType_Connected);
                PLOG_DBG("%s: connected checked for %lu.\n", __func__, blockPair.second);
            } else if (sciErr == NvSciError_Timeout) {
                return EventStatus::TIMED_OUT;
            } else {
                PLOG_ERR("NvSciStreamBlockEventQuery Query connetion failed! sciErr=%d\n", sciErr);
                return EventStatus::ERROR;
            }
        }
    }

    // LOG_MSG(GetName() + " blocks are connected to the stream!\n");
    return EventStatus::CONNECTED;
}

EventStatus CClientCommon::HandleConnectEvent()
{
    auto eventStatus = CheckConnection();

    switch (eventStatus) {
        case EventStatus::CONNECTED:
            if (NvError_Success != OnConnected()) {
                m_pModuleCallback->OnEvent(this, EventStatus::ERROR);
                eventStatus = EventStatus::ERROR;
            } else {
                m_pModuleCallback->OnEvent(this, EventStatus::CONNECTED);
                eventStatus = EventStatus::QUITTED;
            }
            break;
        case EventStatus::TIMED_OUT:
        case EventStatus::OK:
            break;
        default:
            PLOG_ERR("CheckConnection return error! error=%d\n", eventStatus);
            m_pModuleCallback->OnEvent(this, EventStatus::ERROR);
            eventStatus = EventStatus::ERROR;
            break;
    }

    PLOG_DBG("HandleConnectEvent return %d\n", static_cast<int>(eventStatus));
    return eventStatus;
}

EventStatus CClientCommon::HandleClientEvent()
{
    auto eventStatus = HandleEvent();

    switch (eventStatus) {
        case EventStatus::RECONCILED:
            m_pModuleCallback->OnEvent(this, eventStatus);
            break;
        case EventStatus::DISCONNECT:
            m_pModuleCallback->OnEvent(this, eventStatus);
            eventStatus = EventStatus::QUITTED;
            break;
        case EventStatus::TIMED_OUT:
        case EventStatus::OK:
            break;
        default:
            m_pModuleCallback->OnEvent(this, EventStatus::ERROR);
            eventStatus = EventStatus::ERROR;
            break;
    }

    return eventStatus;
}
