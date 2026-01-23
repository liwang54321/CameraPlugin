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

#ifndef CMSGEADER_HPP
#define CMSGEADER_HPP

#include "CUtils.hpp"
#include "CMsgCommon.hpp"

class CControlChannelManager;

class CMsgReader
{
  public:
    inline std::string GetZoneName() { return m_sZoneName; };

  private:
    CMsgReader(const std::string &sZoneName, MsgHandler msgHandler);
    NvError ProcessMsg(MessageHeader *pHeaderBuf, void *pContentBuf);

    std::string m_sZoneName;
    MsgHandler m_msgHandler;
    friend class CControlChannelManager;
};

#endif