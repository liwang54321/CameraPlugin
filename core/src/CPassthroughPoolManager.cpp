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

#include "CPassthroughPoolManager.hpp"

CPassthroughPoolManager::CPassthroughPoolManager(NvSciStreamBlock handle,
                                                 const std::string &sName,
                                                 uint32_t numPackets,
                                                 IPassthroughPoolCallback *pCallback)
    : CPoolManager(handle, sName, numPackets, false)
    , m_pCallback(pCallback)
{
    m_sName = sName + "_PassthroughPool";
}

NvError CPassthroughPoolManager::HandlePoolBufferSetup(void)
{
    return HandleElements();
}

NvError CPassthroughPoolManager::HandleElements(void)
{
    uint32_t uNumConsElem = 0U;
    ElemBufAttr consElems[MAX_NUM_ELEMENTS]{};

    /* Query consumer element count */
    auto sciErr = NvSciStreamBlockElementCountGet(m_handle, NvSciStreamBlockType_Consumer, &uNumConsElem);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Query consumer element count");
    PLOG_DBG("Receive consumer elem count: %u\n", uNumConsElem);

    if (uNumConsElem > MAX_NUM_ELEMENTS) {
        PLOG_ERR("Invalid element count, cons elem count: %u\n", uNumConsElem);
        return NvError_BadValue;
    }

    /* Query all consumer elements */
    for (auto i = 0U; i < uNumConsElem; ++i) {
        sciErr = NvSciStreamBlockElementAttrGet(m_handle, NvSciStreamBlockType_Consumer, i,
                                                reinterpret_cast<uint32_t *>(&consElems[i].userType),
                                                &consElems[i].bufAttrList);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamBlockElementAttrGet");
        PLOG_DBG("HandleElements: consumer, elemId: %u, elemType: %u\n", i, consElems[i].userType);
    }

    auto error = m_pCallback->OnDownstreamBufAttrRecvd(consElems, uNumConsElem);
    CHK_ERROR_AND_RETURN(error, "m_pCallback->OnDownstreamBufAttrRecvd");

    /* Indicate that all element information has been imported */
    m_bElementsDone = true;
    sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_ElementImport, true);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Complete element import");

    return NvError_Success;
}

bool CPassthroughPoolManager::GetElemId(PacketElementType userType, uint32_t &uElemId)
{
    for (auto i = 0U; i < m_uRecvdElemCount; ++i) {
        if (m_recvdElems[i].userType == userType && m_recvdElems[i].bufAttrList != nullptr) {
            uElemId = i;
            return true;
        }
    }
    return false;
}

NvError CPassthroughPoolManager::ExportBufAttr(const ElemBufAttr (&elemBufAttrs)[MAX_NUM_ELEMENTS], uint32_t uNumElems)
{
    m_uRecvdElemCount = uNumElems;
    for (auto e = 0U; e < uNumElems; ++e) {
        auto sciErr = NvSciBufAttrListClone(elemBufAttrs[e].bufAttrList, &m_recvdElems[e].bufAttrList);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListClone");
        m_recvdElems[e].userType = elemBufAttrs[e].userType;

        const ElemBufAttr *elem = &elemBufAttrs[e];
        sciErr = NvSciStreamBlockElementAttrSet(m_handle, static_cast<uint32_t>(elem->userType), elem->bufAttrList);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: NvSciStreamBlockElementAttrSet");
    }

    /* Indicate that all element information has been exported */
    auto sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_ElementExport, true);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "Pool: Complete element export");

    return NvError_Success;
}

/* Check packet error */
NvError CPassthroughPoolManager::ExportPacket(const std::vector<ElemBufObj> &vElemBufObjs)
{
    NvSciStreamCookie cookie = PoolCookieBase + static_cast<NvSciStreamCookie>(m_uRecvdPackets);
    auto sciErr = NvSciStreamPoolPacketCreate(m_handle, cookie, &m_packetHandles[m_uRecvdPackets]);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamPoolPacketCreate");

    for (auto elemBufObj : vElemBufObjs) {
        uint32_t uElemId = 0U;
        if (!GetElemId(elemBufObj.userType, uElemId)) {
            PLOG_ERR("ExportPacket, GetElemId failed, userType: %u\n", elemBufObj.userType);
            return NvError_BadParameter;
        }
        sciErr =
            NvSciStreamPoolPacketInsertBuffer(m_handle, m_packetHandles[m_uRecvdPackets], uElemId, elemBufObj.bufObj);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamPoolPacketInsertBuffer");
    }

    /* Indicate packet setup is complete */
    sciErr = NvSciStreamPoolPacketComplete(m_handle, m_packetHandles[m_uRecvdPackets]);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciStreamPoolPacketComplete");

    m_uRecvdPackets++;

    if (m_uRecvdPackets == m_uNumPackets) {
        sciErr = NvSciStreamBlockSetupStatusSet(m_handle, NvSciStreamSetup_PacketExport, true);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "Complete packet export");
    }

    return NvError_Success;
}