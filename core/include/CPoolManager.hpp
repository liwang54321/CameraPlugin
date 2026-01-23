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

#ifndef CPOOLMANAGER_HPP
#define CPOOLMANAGER_HPP

#include "CConfig.hpp"
#include "CUtils.hpp"
#include "Common.hpp"
#include "nvscistream.h"
#include "CEventHandler.hpp"

class CPoolManager
{
  public:
    CPoolManager(NvSciStreamBlock handle, const std::string &sName, uint32_t uNumPackets, bool bIsC2C);
    virtual ~CPoolManager();

    virtual NvError Init();
    EventStatus HandleEvent();
    void SetElemTypesToSkip(const std::vector<PacketElementType> &vuElemTypesToSkip);
    static NvError GetBufAttrList(NvSciBufAttrList outBufAttrList);

    inline const std::string &GetName() { return m_sName; }
    NvError HandleBuffers();

  protected:
    virtual NvError HandlePoolBufferSetup(void);
    virtual NvError HandleElements(void);
    virtual NvError HandlePacketsStatus(void);

    NvSciStreamBlock m_handle;
    std::string m_sName;
    bool m_bElementsDone = false;
    uint32_t m_uNumPackets = 0U;

    // Packet element descriptions
    NvSciStreamPacket m_packetHandles[MAX_NUM_PACKETS];

  private:
    NvError HandleC2CElements();
    void FreeElements();

    uint32_t m_uNumConsumers = 0U;

    // Reconciled packet element atrribute
    uint32_t m_uNumElem = 0U;
    ElemBufAttr m_elems[MAX_NUM_ELEMENTS];

    uint32_t m_uNumPacketReady = 0U;
    bool m_bPacketsDone = false;
    bool m_bIsC2C = false;
    std::vector<PacketElementType> m_vuElemTypesToSkip{};
};

#endif
