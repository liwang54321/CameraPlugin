/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "CPoolManager.hpp"

CPoolManager::CPoolManager(NvSciStreamBlock handle, const std::string &sName, uint32_t uNumPackets, bool bIsC2C)
    : m_handle(handle)
    , m_sName(sName + "_Pool")
    , m_uNumPackets(uNumPackets)
    , m_bIsC2C(bIsC2C)
{
}

CPoolManager::~CPoolManager()
{
    PLOG_DBG("Pool release.\n");
}

NvError CPoolManager::Init()
{
    PLOG_DBG("Pool Init.\n");
    /* Query number of consumers */
    auto sciErr = NvSciStreamBlockConsumerCountGet(m_handle, &m_uNumConsumers);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Query number of consumers");
    LOG_MSG("Pool: Consumer count is %u\n", m_uNumConsumers);

    return NvError_Success;
}

EventStatus CPoolManager::HandleEvent()
{
    NvSciStreamEventType event = NvSciStreamEventType_Error;
    EventStatus status = EventStatus::OK;
    NvError error = NvError_Success;
    NvSciError sciStatus = NvSciError_Unknown;

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
            error = HandlePoolBufferSetup();
            status = (error == NvError_Success) ? EventStatus::OK : EventStatus::ERROR;
            break;
        case NvSciStreamEventType_PacketStatus:
            if (++m_uNumPacketReady < m_uNumPackets) {
                break;
            }

            PLOG_DBG("Received all the PacketStatus events.\n");
            error = HandlePacketsStatus();
            status = (error == NvError_Success) ? EventStatus::OK : EventStatus::ERROR;
            break;
        case NvSciStreamEventType_Error:
            sciErr = NvSciStreamBlockErrorGet(m_handle, &sciStatus);
            if (NvSciError_Success != sciErr) {
                PLOG_DBG("Failed to query the error event code 0x%x\n", sciErr);
            } else {
                PLOG_DBG("Received error event: 0x%x\n", sciStatus);
            }
            status = EventStatus::ERROR;
            break;
        case NvSciStreamEventType_Disconnected:
            if (!m_bElementsDone) {
                PLOG_WARN("Disconnect before element support\n");
            } else if (!m_bPacketsDone) {
                PLOG_WARN("Disconnect before packet setup\n");
            }
            status = EventStatus::QUITTED;
            break;
        /* All setup complete. Transition to runtime phase */
        case NvSciStreamEventType_SetupComplete:
            PLOG_DBG("Setup completed\n");
            status = EventStatus::QUITTED;
            break;
        default:
            PLOG_ERR("Received unknown event 0x%x\n", event);
            status = EventStatus::ERROR;
            break;
    }

    if (status != EventStatus::OK && status != EventStatus::QUITTED) {
        PLOG_ERR("HandleEvent enconter error 0x%x\n", status);
    }

    return status;
}

NvError CPoolManager::HandlePoolBufferSetup()
{
    NvError error = NvError_Success;

    PLOG_DBG("HandlePoolBufferSetup, m_bIsC2C: %u\n", m_bIsC2C);
    if (!m_bIsC2C) {
        error = HandleElements();
        CHK_ERROR_AND_RETURN(error, "HandleElements");
    } else {
        error = HandleC2CElements();
        CHK_ERROR_AND_RETURN(error, "Pool: HandleC2CElements");
    }

    error = HandleBuffers();
    CHK_ERROR_AND_RETURN(error, "Pool: HandleBuffers");

    return NvError_Success;
}

NvError CPoolManager::HandleElements()
{
    uint32_t uNumProdElem = 0U, uNumConsElem = 0U;
    ElemBufAttr prodElems[MAX_NUM_ELEMENTS]{};
    ElemBufAttr consElems[MAX_NUM_ELEMENTS]{};

    /* Query producer element count */
    auto sciErr = NvSciStreamBlockElementCountGet(m_handle, NvSciStreamBlockType_Producer, &uNumProdElem);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Query producer element count");

    /* Query consumer element count */
    sciErr = NvSciStreamBlockElementCountGet(m_handle, NvSciStreamBlockType_Consumer, &uNumConsElem);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Query consumer element count");

    if (uNumProdElem > MAX_NUM_ELEMENTS || uNumConsElem > MAX_NUM_ELEMENTS) {
        PLOG_ERR("ReconcileElements, invalid element count, prod elem count: %u, cons elem count: %u\n", uNumProdElem,
                 uNumConsElem);
        return NvError_BadValue;
    }

    /* Query all producer elements */
    for (auto i = 0U; i < uNumProdElem; ++i) {
        sciErr = NvSciStreamBlockElementAttrGet(m_handle, NvSciStreamBlockType_Producer, i,
                                                reinterpret_cast<uint32_t *>(&prodElems[i].userType),
                                                &prodElems[i].bufAttrList);
        if (NvSciError_Success != sciErr) {
            PLOG_ERR("Failed (0x%x) to query producer element %u\n", sciErr, i);
            return NvError_BadParameter;
        }
    }

    /* Query all consumer elements */
    for (auto i = 0U; i < uNumConsElem; ++i) {
        sciErr = NvSciStreamBlockElementAttrGet(m_handle, NvSciStreamBlockType_Consumer, i,
                                                reinterpret_cast<uint32_t *>(&consElems[i].userType),
                                                &consElems[i].bufAttrList);
        if (NvSciError_Success != sciErr) {
            PLOG_ERR("Failed (0x%x) to query consumer element %d\n", sciErr, i);
            return NvError_BadParameter;
        }
    }

    /* Indicate that all element information has been imported */
    m_bElementsDone = true;
    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_ElementImport, true);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Complete element import");

    m_uNumElem = 0U;
    std::vector<NvSciBufAttrList> vUnreconciledAttrLists;
    for (auto p = 0U; p < uNumProdElem; ++p) {
        ElemBufAttr *pProdElem = &prodElems[p];
        vUnreconciledAttrLists.push_back(pProdElem->bufAttrList);
        for (auto c = 0U; c < uNumConsElem; ++c) {
            ElemBufAttr *pConsElem = &consElems[c];
            /* If requested element types match, combine the entries */
            if (pProdElem->userType == pConsElem->userType) {
                vUnreconciledAttrLists.push_back(pConsElem->bufAttrList);
                /* Found a match for this producer element so move on */
                break;
            } /* if match */
        } /* for all requested consumer elements */

        ElemBufAttr *pPoolElem = &m_elems[m_uNumElem++];
        pPoolElem->userType = pProdElem->userType;
        pPoolElem->bufAttrList = nullptr;

        /* Combine and reconcile the attribute lists */
        NvSciBufAttrList conflicts = nullptr;
        sciErr = NvSciBufAttrListReconcile(vUnreconciledAttrLists.data(), vUnreconciledAttrLists.size(),
                                           &pPoolElem->bufAttrList, &conflicts);
        if (nullptr != conflicts) {
            NvSciBufAttrListFree(conflicts);
        }

        vUnreconciledAttrLists.clear();

        /* Abort on error */
        if (NvSciError_Success != sciErr) {
            PLOG_ERR("Failed to reconcile element 0x%x attrs (0x%x)\n", pPoolElem->userType, sciErr);
            return NvError_BadParameter;
        }
    } /* for all requested producer elements */

    /* Should be at least one element */
    if (0U == m_uNumElem) {
        PLOG_ERR("Didn't find any common elements\n");
        return NvError_BadValue;
    }

    /* Inform the stream of the chosen elements */
    for (auto e = 0U; e < m_uNumElem; ++e) {
        ElemBufAttr *pPoolElem = &m_elems[e];
        sciErr = NvSciStreamBlockElementAttrSet(m_handle, static_cast<uint32_t>(pPoolElem->userType),
                                                pPoolElem->bufAttrList);
        if (NvSciError_Success != sciErr) {
            PLOG_ERR("Failed (0x%x) to send element %u info\n", sciErr, e);
            return NvError_BadParameter;
        }
    }

    /* Indicate that all element information has been exported */
    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_ElementExport, true);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Complete element export");
    PLOG_DBG("HandleElements, NvSciStreamSetup_ElementExport\n");

    return NvError_Success;
}

NvError CPoolManager::HandleC2CElements()
{
    /* Query producer element count */
    auto sciErr = NvSciStreamBlockElementCountGet(m_handle, NvSciStreamBlockType_Producer, &m_uNumElem);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Query producer element count");

    if (m_uNumElem > MAX_NUM_ELEMENTS) {
        PLOG_ERR("GetC2CElements, invalid elem count: %u\n", m_uNumElem);
        return NvError_BadValue;
    }

    /* Query all producer elements */
    for (uint32_t i = 0U; i < m_uNumElem; ++i) {
        sciErr =
            NvSciStreamBlockElementAttrGet(m_handle, NvSciStreamBlockType_Producer, i,
                                           reinterpret_cast<uint32_t *>(&m_elems[i].userType), &m_elems[i].bufAttrList);
        if (NvSciError_Success != sciErr) {
            PLOG_ERR("Failed (0x%x) to query producer element %u\n", sciErr, i);
            return NvError_BadParameter;
        }
    }

    /* Indicate that all element information has been imported */
    m_bElementsDone = true;
    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_ElementImport, true);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Complete element import");
    PLOG_DBG("HandleC2CElements, NvSciStreamSetup_ElementImport\n");

    return NvError_Success;
}

NvError CPoolManager::HandleBuffers()
{
    NvSciError sciErr = NvSciError_Success;
    /*
     * Create and send all the packets and their buffers
     * Note: Packets and buffers are not guaranteed to be received by
     *       producer and consumer in the same order sent, nor are the
     *       error messages sent back guaranteed to preserve ordering.
     *       This is one reason why an event driven model is more robust.
     */
    for (auto i = 0U; i < m_uNumPackets; ++i) {
        /*Our pool implementation doesn't need to save any packet-specific
         *   data, but we do need to provide unique cookies, so we just
         *   use the pointer to the location we save the handle.
         */
        NvSciStreamCookie cookie = (NvSciStreamCookie)&m_packetHandles[i];
        sciErr = NvSciStreamPoolPacketCreate(m_handle, cookie, &m_packetHandles[i]);
        if (NvSciError_Success != sciErr) {
            PLOG_ERR("Failed (0x%x) to create packet %d\n", sciErr, i);
            return NvError_BadParameter;
        }

        /* Create buffers for the packet */
        for (auto e = 0U; e < m_uNumElem; ++e) {
            // Skip the elements which are not used
            if (std::find(m_vuElemTypesToSkip.begin(), m_vuElemTypesToSkip.end(), m_elems[e].userType) !=
                m_vuElemTypesToSkip.end()) {
                continue;
            }
            /* Allocate a buffer object */
            NvSciBufObj obj = nullptr;
            sciErr = NvSciBufObjAlloc(m_elems[e].bufAttrList, &obj);
            if (NvSciError_Success != sciErr) {
                PLOG_ERR("Failed (0x%x) to allocate buffer %u of packet %u\n", sciErr, e, i);
                return NvError_BadParameter;
            }
            /* Insert the buffer in the packet */
            sciErr = NvSciStreamPoolPacketInsertBuffer(m_handle, m_packetHandles[i], e, obj);
            /* The pool doesn't need to keep a copy of the object handle */
            NvSciBufObjFree(obj);
            obj = nullptr;
            if (NvSciError_Success != sciErr) {
                PLOG_ERR("Failed (0x%x) to insert buffer %u of packet %u\n", sciErr, e, i);
                return NvError_BadParameter;
            }
        }
        /* Indicate packet setup is complete */
        sciErr = NvSciStreamPoolPacketComplete(m_handle, m_packetHandles[i]);
        if (NvSciError_Success != sciErr) {
            PLOG_ERR("Failed (0x%x) to complete packet %u setup\n", sciErr, i);
            return NvError_BadParameter;
        }
    }

    /*
     * Indicate that all packets have been sent.
     * Note: An application could choose to wait to send this until
     *  the error has been received, in order to try to make any
     *  corrections for rejected packets.
     */
    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_PacketExport, true);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Complete packet export");
    PLOG_DBG("NvSciStreamSetup_PacketExport\n");

    return NvError_Success;
}

/* Check packet error */
NvError CPoolManager::HandlePacketsStatus()
{
    bool bPacketFailure = false;
    NvSciError sciErr;

    /* Check each packet */
    for (uint32_t p = 0U; p < m_uNumPackets; ++p) {
        /* Check packet acceptance */
        bool bAccept = false;
        sciErr = NvSciStreamPoolPacketStatusAcceptGet(m_handle, m_packetHandles[p], &bAccept);
        if (NvSciError_Success != sciErr) {
            PLOG_ERR("Failed (0x%x) to retrieve packet %u's acceptance-statue\n", sciErr, p);
            return NvError_BadParameter;
        }
        if (bAccept) {
            continue;
        }

        /* On rejection, query and report details */
        bPacketFailure = true;
        NvSciError error = NvSciError_Unknown;

        /* Check packet error from producer */
        sciErr = NvSciStreamPoolPacketStatusValueGet(m_handle, m_packetHandles[p], NvSciStreamBlockType_Producer, 0U,
                                                     &error);
        if (NvSciError_Success != sciErr) {
            PLOG_ERR("Failed (0x%x) to retrieve packet %u's statue from producer\n", sciErr, p);
            return NvError_BadParameter;
        }
        if (error != NvSciError_Success) {
            PLOG_ERR("Producer rejected packet %u with error 0x%x\n", p, error);
        }

        /* Check packet error from consumers */

        for (uint32_t c = 0U; c < m_uNumConsumers; ++c) {
            sciErr = NvSciStreamPoolPacketStatusValueGet(m_handle, m_packetHandles[p], NvSciStreamBlockType_Consumer, c,
                                                         &error);
            if (NvSciError_Success != sciErr) {
                PLOG_ERR("Failed (0x%x) to retrieve packet %u's statue from consumer %u\n", sciErr, p, c);
                return NvError_BadParameter;
            }
            if (error != NvSciError_Success) {
                PLOG_ERR("Consumer %u rejected packet %d with error 0x%x\n", c, p, error);
            }
        }
    }

    /* Indicate that error for all packets has been received. */
    m_bPacketsDone = true;
    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_PacketImport, true);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Complete packet import");
    PLOG_DBG("NvSciStreamSetup_PacketImport, bPacketFailure: %u\n", bPacketFailure);

    return bPacketFailure ? NvError_BadParameter : NvError_Success;
}

void CPoolManager::SetElemTypesToSkip(const std::vector<PacketElementType> &vuElemTypesToSkip)
{
    m_vuElemTypesToSkip = vuElemTypesToSkip;
}
