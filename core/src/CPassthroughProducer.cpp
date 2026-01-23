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

#include "CPassthroughProducer.hpp"

CPassthroughProducer::CPassthroughProducer(std::shared_ptr<CClientCfg> spClientCfg, IModuleCallback *pCallback)
    : CProducer(std::move(spClientCfg), pCallback)
{
}

void CPassthroughProducer::SetCallback(IUpstreamCallback *pCallback)
{
    m_pUpstreamCallback = pCallback;
}

std::unique_ptr<CPoolManager> CPassthroughProducer::CreatePoolManager()
{
    return std::make_unique<CPassthroughPoolManager>(m_poolHandle, GetName(), MAX_NUM_PACKETS, this);
}

// Received reconciled upstream buf attr
NvError CPassthroughProducer::OnUpstreamBufAttrRecvd(const ElemBufAttr (&elemBufAttrs)[MAX_NUM_ELEMENTS],
                                                     uint32_t uNumElems)
{
    CPassthroughPoolManager *pPassthroughPoolManger = static_cast<CPassthroughPoolManager *>(m_upPoolManger.get());
    auto error = pPassthroughPoolManger->ExportBufAttr(elemBufAttrs, uNumElems);
    PCHK_ERROR_AND_RETURN(error, "pPassthroughPoolManger->ExportBufAttr");

    return NvError_Success;
}

// Received unreconciled downstream buf attr
NvError CPassthroughProducer::OnDownstreamBufAttrRecvd(const ElemBufAttr (&elemBufAttrs)[MAX_NUM_ELEMENTS],
                                                       uint32_t uNumElems)
{
    auto error = m_pUpstreamCallback->OnDownstreamBufAttrRecvd(elemBufAttrs, uNumElems);
    PCHK_ERROR_AND_RETURN(error, "m_pUpstreamCallback->OnDownstreamBufAttrRecvd");

    return NvError_Success;
}

NvError CPassthroughProducer::Init()
{
    auto error = CProducer::Init();
    PCHK_ERROR_AND_RETURN(error, "CProducer::Init()");

    RemoveUnusedElements();

    return NvError_Success;
}

/*For passthrough producer, It should transfer the packets to upstream.*/
NvError CPassthroughProducer::PreStop()
{
    return NvError_Success;
}

NvError CPassthroughProducer::Stop()
{
    return CClientCommon::Stop();
}

NvError CPassthroughProducer::PostStop()
{
    return CClientCommon::PostStop();
}

void CPassthroughProducer::RemoveUnusedElements()
{
    for (auto iter = m_vElemsInfos.begin(); iter != m_vElemsInfos.end();) {
        if (!iter->bIsUsed) {
            iter = m_vElemsInfos.erase(iter);
        } else {
            ++iter;
        }
    }
}

NvError CPassthroughProducer::HandleSyncSupport()
{
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        if (m_upstreamSyncAttr.syncAttrList == nullptr) {
            PLOG_DBG("HandleSyncSupport, hasn't received upstream sync attr, exit!\n");
            m_bPendingSyncSupport = true;
            return NvError_Success;
        }
    }

    auto error = CProducer::HandleSyncSupport();
    PCHK_ERROR_AND_RETURN(error, "CProducer::HandleSyncSupport()");

    return NvError_Success;
}

NvError CPassthroughProducer::RegisterSignalSyncObj(uint32_t uElemId, NvSciSyncObj signalSyncObj)
{
    auto error = m_pUpstreamCallback->OnSyncObjCreated(signalSyncObj);
    PCHK_ERROR_AND_RETURN(error, "m_pUpstreamCallback->OnSyncObjCreated");

    return NvError_Success;
}

NvError CPassthroughProducer::OnUpstreamWaiterAttrRecvd(const ElemSyncAttr &elemSyncAttr)
{
    std::unique_lock<std::mutex> lk(m_mutex);
    auto sciErr = NvSciSyncAttrListClone(elemSyncAttr.syncAttrList, &m_upstreamSyncAttr.syncAttrList);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListClone");
    m_upstreamSyncAttr.userType = elemSyncAttr.userType;

    if (m_bPendingSyncSupport) {
        auto error = CProducer::HandleSyncSupport();
        PCHK_ERROR_AND_RETURN(error, "CProducer::HandleSyncSupport()");
    }

    return NvError_Success;
}

NvError CPassthroughProducer::FillSyncWaiterAttrList(PacketElementType userType, NvSciSyncAttrList *pWaiterAttrList)
{
    CHK_PTR_AND_RETURN_BADARG(m_upstreamSyncAttr.syncAttrList, "upstream sync attr");

    if (m_upstreamSyncAttr.userType != userType) {
        PLOG_ERR("FillSyncWaiterAttrList, invalid userType, upstream: %u, downstream: %u\n",
                 m_upstreamSyncAttr.userType, userType);
        return NvError_BadParameter;
    }

    //Fill the upstream sync attr as the waiter attr
    if (*pWaiterAttrList) {
        NvSciSyncAttrListFree(*pWaiterAttrList);
        *pWaiterAttrList = nullptr;
    }
    auto sciErr = NvSciSyncAttrListClone(m_upstreamSyncAttr.syncAttrList, pWaiterAttrList);
    PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListClone");

    return NvError_Success;
}

//On receiving the created packet from upstream, export it via passthrough poor manager
NvError CPassthroughProducer::OnUpstreamPacketCreated(const std::vector<ElemBufObj> &vElemBufObjs)
{
    CPassthroughPoolManager *pPassthroughPoolManger = static_cast<CPassthroughPoolManager *>(m_upPoolManger.get());
    auto error = pPassthroughPoolManger->ExportPacket(vElemBufObjs);
    PCHK_ERROR_AND_RETURN(error, "pPassthroughPoolManger->ExportPacket");

    return NvError_Success;
}

// Propagate the waiter sync obj to upstream
NvError CPassthroughProducer::RegisterWaiterSyncObj(uint32_t uElemId, NvSciSyncObj waiterSyncObj)
{
    auto error = m_pUpstreamCallback->OnDownstreamSyncObjRecvd(waiterSyncObj);
    PCHK_ERROR_AND_RETURN(error, "pPassthroughPoolManger->ExportPacket");

    return NvError_Success;
}

//After a packet is processed by upstream, it need to be sent to downstream.
NvError CPassthroughProducer::OnUpstreamPacketProcessed(NvSciBufObj *pBufObj, NvSciSyncFence *pPostfence)
{
    auto error = Post(pBufObj, pPostfence);
    PCHK_ERROR_AND_RETURN(error, "Post");

    return NvError_Success;
}

//The method is called when there is a downstream packet is released,
//insert prefence and release the packet to upstream
NvError
CPassthroughProducer::InsertPrefence(PacketElementType userType, uint32_t uPacketIndex, NvSciSyncFence *pPrefence)
{
    NvSciBufObj *pBufObj = GetBufObj(uPacketIndex);
    m_pUpstreamCallback->OnDownstreamPacketProcessed(pBufObj, pPrefence);
    return NvError_Success;
}

//When cleared fence is received, release the packet to upstream
void CPassthroughProducer::OnClearedFenceReceived(uint32_t uPacketIndex)
{
    NvSciBufObj *pBufObj = GetBufObj(uPacketIndex);
    m_pUpstreamCallback->OnDownstreamPacketProcessed(pBufObj, nullptr);
}