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

/* STL Headers */
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <getopt.h>
#include <iomanip>

#include "nverror.h"
#include "CUtils.hpp"
#include "NvSIPLTrace.hpp" // NvSIPLTrace to set library trace level
#if !NV_IS_SAFETY
#include "NvSIPLQuery.hpp" // NvSIPLQuery to display platform config
#endif
#include "CConfig.hpp"

#ifndef CCMDLINEPARSER_HPP
#define CCMDLINEPARSER_HPP

using namespace nvsipl;

class CCmdLineParser
{
  public:
    NvError Parse(int argc, char *argv[], std::shared_ptr<CAppCfg> &spAppConfig);

  private:
    void ShowUsage();
    void ShowConfigs();
    static void ShowElements();
    static void ShowListHelp(const char *pListOption);
    static void ShowElementOptions(const char *pElementName, bool bIgnoreCase = true);
    static void ShowPipelines(const std::shared_ptr<CAppCfg> &spAppConfig);
    NvError ParseEncoderTypeStr(const std::string &sEncoderType, EncoderType &eEncoderType);
    NvError ParsePipelineTypeStr(const std::string &sPipelineType, PipelineType &ePipelineType);
};

#endif
