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

#include "CPassthroughConsumer.hpp"
#include <array>

CPassthroughConsumer::CPassthroughConsumer(std::shared_ptr<CClientCfg> spClientCfg, IModuleCallback *pCallback)
    : CConsumer(std::move(spClientCfg), pCallback)
{
}

void CPassthroughConsumer::SetCallback(IDownstreamCallback *pCallback)
{
    m_pDownstreamCallback = pCallback;
}

NvError CPassthroughConsumer::Reconcile()
{
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        if (m_uNumDownstreamElems == 0U) {
            PLOG_DBG("Reconcile(), hasn't received downstream buffer attr, exit!\n");
            m_bPendingReconcile = true;
            return NvError_Success;
        }
    }

    auto error = CConsumer::Reconcile();
    PCHK_ERROR_AND_RETURN(error, "CConsumer::Reconcile");

    return NvError_Success;
}

ElemBufAttr *CPassthroughConsumer::GetDownstreamBufAttr(PacketElementType userType)
{
    for (auto i = 0U; i < m_uNumDownstreamElems; ++i) {
        if (m_downstreamBufAttrs[i].userType == userType && m_downstreamBufAttrs[i].bufAttrList != nullptr) {
            return &m_downstreamBufAttrs[i];
        }
    }

    PLOG_DBG("GetDownstreamBufAttr return null, userType: %u\n", userType);
    return nullptr;
}

//On receiving unreconciled downstream buf attr
NvError CPassthroughConsumer::OnDownstreamBufAttrRecvd(const ElemBufAttr (&elemBufAttrs)[MAX_NUM_ELEMENTS],
                                                       uint32_t uNumElems)
{
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        for (auto i = 0U; i < uNumElems; ++i) {
            auto sciErr = NvSciBufAttrListClone(elemBufAttrs[i].bufAttrList, &m_downstreamBufAttrs[i].bufAttrList);
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListClone");
            m_downstreamBufAttrs[i].userType = elemBufAttrs[i].userType;
        }
        m_uNumDownstreamElems = uNumElems;
    }

    if (m_bPendingReconcile) {
        auto error = CConsumer::Reconcile();
        PCHK_ERROR_AND_RETURN(error, "CConsumer::Reconcile()");
    }

    return NvError_Success;
}

NvError CPassthroughConsumer::FillBufAttrList(PacketElementType userType, NvSciBufAttrList *pBufAttrList)
{
    std::array<NvSciBufAttrList, 2U> attrs;
    ElemBufAttr upstreamBufAttr;
    ElemBufAttr *pDownstreamBufAttr = nullptr;

    auto sciErr = NvSciBufAttrListCreate(m_pAppCfg->m_sciBufModule, &upstreamBufAttr.bufAttrList);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListCreate.");

    auto error = CConsumer::FillBufAttrList(userType, &upstreamBufAttr.bufAttrList);
    PCHK_ERROR_AND_RETURN(error, "CConsumer::FillBufAttrList");
    upstreamBufAttr.userType = userType;

    {
        std::unique_lock<std::mutex> lk(m_mutex);
        pDownstreamBufAttr = GetDownstreamBufAttr(upstreamBufAttr.userType);
        if (pDownstreamBufAttr == nullptr) {
            //If lack of corresponding downstream buf attr, populate the upstream buf attr only
            if (*pBufAttrList) {
                NvSciBufAttrListFree(*pBufAttrList);
                *pBufAttrList = nullptr;
            }
            auto sciErr = NvSciBufAttrListClone(upstreamBufAttr.bufAttrList, pBufAttrList);
            PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListClone");
            return NvError_Success;
        }

        //Collect both downstream and upstream buf attr lists and append together
        attrs[0] = upstreamBufAttr.bufAttrList;
        attrs[1] = pDownstreamBufAttr->bufAttrList;

        if (*pBufAttrList) {
            NvSciBufAttrListFree(*pBufAttrList);
            *pBufAttrList = nullptr;
        }
        auto sciErr = NvSciBufAttrListAppendUnreconciled(attrs.data(), attrs.size(), pBufAttrList);
        PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListAppendUnreconciled.");
    }

    return NvError_Success;
}

//On receiving reconciled bufAttrList from upstream, export it to downstream
NvError CPassthroughConsumer::OnBufAttrListRecvd(const ElemBufAttr (&elemBufAttrs)[MAX_NUM_ELEMENTS],
                                                 uint32_t uNumElems)
{
    auto error = m_pDownstreamCallback->OnUpstreamBufAttrRecvd(elemBufAttrs, uNumElems);
    PCHK_ERROR_AND_RETURN(error, "m_pDownstreamCallback->OnUpstreamBufAttrRecvd");

    return NvError_Success;
}

//On receiving signal sync obj from passthrough producer, register it with module
NvError CPassthroughConsumer::OnSyncObjCreated(NvSciSyncObj signalSyncObj)
{
    auto error = RegisterSignalSyncObj(signalSyncObj);
    PCHK_ERROR_AND_RETURN(error, "RegisterSignalSyncObj");

    return NvError_Success;
}

// On receiving waiter sync obj from downstream consumer, export it to upstream
NvError CPassthroughConsumer::OnDownstreamSyncObjRecvd(NvSciSyncObj syncObj)
{
    auto error = ExportSignalSyncObj(syncObj);
    PCHK_ERROR_AND_RETURN(error, "ExportSignalSyncObj");

    return NvError_Success;
}

NvError CPassthroughConsumer::HandleSyncExport()
{
    ElemSyncAttr elemSyncAttr;

    auto error = RecvWaiterAttr(elemSyncAttr);
    PCHK_ERROR_AND_RETURN(error, "RecvWaiterAttr");

    error = m_pDownstreamCallback->OnUpstreamWaiterAttrRecvd(elemSyncAttr);
    PCHK_ERROR_AND_RETURN(error, "m_pDownstreamCallback->OnUpstreamWaiterAttrRecvd");

    return NvError_Success;
}

NvError CPassthroughConsumer::OnPacketCreated(const std::vector<ElemBufObj> &vElemBufObjs)
{
    auto error = m_pDownstreamCallback->OnUpstreamPacketCreated(vElemBufObjs);
    PCHK_ERROR_AND_RETURN(error, "m_pDownstreamCallback->OnUpstreamPacketCreated");

    return NvError_Success;
}

NvError CPassthroughConsumer::OnPacketProcessed(uint32_t uPacketIndex, NvSciSyncFence *pPostfence, bool bClearFence)
{
    NvSciBufObj *pBufObj = GetBufObj(uPacketIndex);
    PCHK_PTR_AND_RETURN_ERR(pBufObj, "GetBufObj");

    auto error = m_pDownstreamCallback->OnUpstreamPacketProcessed(pBufObj, pPostfence);
    PCHK_ERROR_AND_RETURN(error, "m_pPassThrough->PostPacket");

    return NvError_Success;
}

void CPassthroughConsumer::OnDownstreamPacketProcessed(NvSciBufObj *pBufObj, NvSciSyncFence *pPrefence)
{
    uint32_t uPacketIndex = 0;
    uint32_t uElementId = 0U;

    auto error = MapPayload(pBufObj, uPacketIndex, uElementId);
    if (error != NvError_Success) {
        PLOG_ERR("MapPayload failed with error %x.\n", error);
        return;
    }

    pPrefence ? ReleasePacket(uPacketIndex, pPrefence) : ReleasePacket(uPacketIndex);
}
