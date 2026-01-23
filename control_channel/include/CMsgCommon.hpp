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

#ifndef CMESSAGECOMMON_HPP
#define CMESSAGECOMMON_HPP

#include "CUtils.hpp"
#include <functional>

constexpr uint32_t kMaxZoneNameLength = 64U;
constexpr uint32_t kMaxEndpointLength = 64U;

enum class MsgType : uint8_t
{
    CONNECT_MSG = 0,
    DISCONNECT_MSG = 1,
    READER_CREATE_MSG = 2,
    USER_MSG = 3
};

typedef struct __attribute__((packed)) MessageHeader
{
    char sZoneName[kMaxZoneNameLength];
    uint32_t uContentOffset;
    uint32_t uContentLength;
    MsgType msgType;

    MessageHeader(const std::string &zone, uint32_t offset, uint32_t length, MsgType type)
        : uContentOffset(offset)
        , uContentLength(length)
        , msgType(type)
    {
        strncpy(sZoneName, zone.c_str(), sizeof(sZoneName) - 1);
    };
} MessageHeader;

typedef struct __attribute__((packed))
{
    char sEndpointName[kMaxEndpointLength];
} MessageEndpoint;

using MsgHandler = std::function<void(void *, uint32_t)>;
using MsgHandlerWithHeader = std::function<NvError(MessageHeader *, void *)>;

#endif
