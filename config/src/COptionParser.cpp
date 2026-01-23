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
//#include "CUtils.hpp"
#include "COptionParser.hpp"

NvError COptionParser::SetOption(const Option &option,
                                 const std::pair<std::string, std::string> &pair,
                                 const void *pBaseAddress)
{
    NvError error = NvError_Success;
    const uint8_t *base = static_cast<const uint8_t *>(pBaseAddress);
    if (base != nullptr) {
        if (!pair.second.empty()) {
            try {
                switch (option.type) {
                    case OptionType::INT: {
                        *((int *)(base + option.offset)) = std::stoi(pair.second);
                    } break;

                    case OptionType::INT64: {
                        *((int64_t *)(base + option.offset)) = std::stol(pair.second);
                    } break;

                    case OptionType::DOUBLE: {
                        *((double *)(base + option.offset)) = std::stod(pair.second);
                    } break;

                    case OptionType::FLOAT: {
                        *((float *)(base + option.offset)) = std::stof(pair.second);
                    } break;

                    case OptionType::STRING: {
                        *((std::string *)(base + option.offset)) = pair.second;
                    } break;

                    case OptionType::UINT32: {
                        *((uint32_t *)(base + option.offset)) = std::stoul(pair.second);
                    } break;

                    case OptionType::UINT64: {
                        *((uint64_t *)(base + option.offset)) = std::stoull(pair.second);
                    } break;

                    case OptionType::BOOL: {
                        *((bool *)(base + option.offset)) = !!std::stoi(pair.second);
                    } break;

                    default: {
                        error = NvError_NotSupported;
                    } break;
                }
            }
            catch (...) {
                LOG_ERR("Failed to parse %s with %s\n", pair.first.c_str(), pair.second.c_str());
                error = NvError_BadValue;
            }
        } else {
            error = NvError_BadParameter;
        }
    } else {
        error = NvError_InvalidState;
        LOG_ERR("base address is nullptr\n");
    }

    return error;
}

NvError COptionParser::ParseOptions(const Options &options, const std::vector<OptionParserInfo> &vOptionParserInfo)
{
    for (const auto &parserInfo : vOptionParserInfo) {
        if (!parserInfo.pOptionTable || parserInfo.pOptionTable->empty()) {
            LOG_ERR("Option table is null\n");
            return NvError_InvalidState;
        }
        if (!parserInfo.pBaseAddr) {
            LOG_ERR("Base address is null\n");
            return NvError_BadParameter;
        }
    }
    for (const auto &option : options) {
        bool isFound = false;
        for (const auto &parserInfo : vOptionParserInfo) {
            auto it = parserInfo.pOptionTable->find(option.first);
            if (it != parserInfo.pOptionTable->end()) {
                NvError error = SetOption(it->second, option, parserInfo.pBaseAddr);
                CHK_ERROR_AND_RETURN(error, "SetOption");
                isFound = true;
                break;
            }
        }
        if (!isFound) {
            LOG_ERR("Unknown option %s\n", option.first.c_str());
            return NvError_BadParameter;
        }
    }

    return NvError_Success;
}

const char *COptionParser::ToString(OptionType type)
{
    const char *pTypeString = nullptr;
    switch (type) {
        case OptionType::INT: {
            pTypeString = "int";
        } break;

        case OptionType::INT64: {
            pTypeString = "int64";
        } break;

        case OptionType::DOUBLE: {
            pTypeString = "double";
        } break;

        case OptionType::FLOAT: {
            pTypeString = "float";
        } break;

        case OptionType::STRING: {
            pTypeString = "string";
        } break;

        case OptionType::UINT32: {
            pTypeString = "uint32";
        } break;

        case OptionType::UINT64: {
            pTypeString = "uint64";
        } break;

        case OptionType::BOOL: {
            pTypeString = "bool";
        } break;

        default: {
            pTypeString = "Unknown";
        } break;
    }
    return pTypeString;
}