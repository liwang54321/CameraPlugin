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

#ifndef CMSGWRITER_HPP
#define CMSGWRITER_HPP

#include <memory>
#include "CMsgCommon.hpp"
#include "CUtils.hpp"

class CControlChannelManager;

class CMsgWriter
{
  public:
    class IMsgWriteCallback
    {
      public:
        virtual NvError PostEvent(MessageHeader *pMsgHeader, void *pContentBuf) = 0;

      protected:
        IMsgWriteCallback() = default;
        virtual ~IMsgWriteCallback() = default;
    };
    NvError Write(void *pContentBuf, uint32_t size);
    inline std::string GetZoneName() { return m_sZoneName; };

  private:
    CMsgWriter(const std::string &sZoneName, IMsgWriteCallback *pCallback);

    std::string m_sZoneName;
    IMsgWriteCallback *m_pWriteCallback;
    friend class CControlChannelManager;
};

#endif