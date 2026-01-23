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

#ifndef COPTIONPARSER_HPP
#define COPTIONPARSER_HPP

#include <string>
#include <cstdint>
#include <utility>
#include <unordered_map>
#include "CUtils.hpp"

enum class OptionType : uint8_t
{
    INT,
    INT64,
    DOUBLE,
    FLOAT,
    STRING,
    UINT32,
    UINT64,
    BOOL,
};

struct Option
{
    std::string sHelp;
    int offset;
    OptionType type;
    double min; ///< minimum valid value for the option
    double max; ///< maximum valid value for the option
    int flags;
};

using Options = std::unordered_map<std::string, std::string>;
using OptionTable = std::unordered_map<std::string, Option>;

struct OptionParserInfo
{
    const OptionTable *pOptionTable;
    const void *pBaseAddr;
};

class COptionParser
{
  public:
    static NvError ParseOptions(const Options &options, const std::vector<OptionParserInfo> &vOptionParserInfo);
    static const char *ToString(OptionType type);

  private:
    static NvError
    SetOption(const Option &option, const std::pair<std::string, std::string> &pair, const void *pBaseAddress);
};

#endif