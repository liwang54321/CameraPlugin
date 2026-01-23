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

#ifndef CPASSTHROUGHPOOLMANAGER_HPP
#define CPASSTHROUGHPOOLMANAGER_HPP

#include "CPoolManager.hpp"

constexpr NvSciStreamCookie PoolCookieBase = 0xD00C1E4U;

class CPassthroughPoolManager : public CPoolManager
{
  public:
    class IPassthroughPoolCallback
    {
      public:
        virtual NvError OnDownstreamBufAttrRecvd(const ElemBufAttr (&elemBufAttrs)[MAX_NUM_ELEMENTS],
                                                 uint32_t uNumElems) = 0;
    };

    CPassthroughPoolManager(NvSciStreamBlock handle,
                            const std::string &sName,
                            uint32_t numPackets,
                            IPassthroughPoolCallback *pCallback);
    virtual ~CPassthroughPoolManager() {}
    NvError ExportBufAttr(const ElemBufAttr (&elemBufAttrs)[MAX_NUM_ELEMENTS], uint32_t uNumElems);
    NvError ExportPacket(const std::vector<ElemBufObj> &vElemBufObjs);

  protected:
    virtual NvError HandlePoolBufferSetup(void) override;
    virtual NvError HandleElements(void) override;

  private:
    bool GetElemId(PacketElementType userType, uint32_t &uElemId);

    IPassthroughPoolCallback *m_pCallback = nullptr;
    ElemBufAttr m_recvdElems[MAX_NUM_ELEMENTS]{};
    uint32_t m_uRecvdElemCount = 0U;
    uint32_t m_uRecvdPackets = 0U;
};

#endif
