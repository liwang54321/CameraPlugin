/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <string.h>
#include "CCmdLineParser.hpp"
#include "CElementDescription.hpp"
#include "CElementDescriptionCollector.hpp"
#if NV_BUILD_DOS7
#include "platform/thor/ar0820.hpp"
#include "platform/thor/imx623vb2.hpp"
#include "platform/thor/imx728vb2.hpp"
#include "platform/thor/isx031.hpp"
#include "platform/thor/max96712_tpg_yuv.hpp"
#else
#include "platform/orin/ar0820.hpp"
#include "platform/orin/imx623vb2.hpp"
#include "platform/orin/imx728vb2.hpp"
#include "platform/orin/isx031.hpp"
#include "platform/orin/max96712_tpg_yuv.hpp"
#endif

static constexpr int kMaxMaskSize = 128;
NvError CCmdLineParser::Parse(int argc, char *argv[], std::shared_ptr<CAppCfg> &spAppConfig)
{
#if !NV_IS_SAFETY
    const char *const short_options = "hCc:m:v:t:l::N:7Ip:r:k:VE:K::M:P:S::";
#else
    const char *const short_options = "hv:t:l::N:7Ip:r:k:q:VE:K::M:P:S::";
#endif
    const struct option long_options[] = {
        // clang-format off
        { "help",                 no_argument,       0, 'h' },
        { "centralNode",          no_argument,       0, 'C' },
#if !NV_IS_SAFETY
        { "platform-config",      required_argument, 0, 'c' },
        { "link-enable-masks",    required_argument, 0, 'm' },
#endif
        { "verbosity",            required_argument, 0, 'v' },
        { "frameFilter",          required_argument, 0, 'k' },
        { "version",              no_argument,       0, 'V' },
        { "runfor",               required_argument, 0, 'r' },
        { "statusManagerEnable",  no_argument,       0, '7' },
#if !NV_IS_SAFETY
        { "encoderType",          required_argument, 0, 'E' },
#endif
        { "pipelineType",         required_argument, 0, 'P' },
        { "pipeline",             required_argument, 0, 'p' },
        { "profile",              optional_argument, 0, 'K' },
        { "maxPerSampleNum",      required_argument, 0, 'M' },
        { "savePerfData",         optional_argument, 0, 'S' },
        { 0,                      0,                 0,  0  } // clang-format on
    };

    int index = 0;
    bool bShowHelp = false;
    std::string sEncoderType;
    std::string sPipelineType;

    PipelineGroup currentPipelineGroup;
    while (1) {
        const auto getopt_ret = getopt_long(argc, argv, short_options, &long_options[0], &index);
        if (getopt_ret == -1) {
            // Done parsing all arguments.
            break;
        }

        switch (getopt_ret) {
            default:  /* Unrecognized option */
            case '?': /* Unrecognized option */
                std::cout << "Invalid or Unrecognized command line option. Specify -h or --help for options\n";
                bShowHelp = true;
                break;
            case 'h': /* -h or --help */
                bShowHelp = true;
                break;
            case 'C':
                /* this option is intended to specify a central node of control channel in slave soc.
                 * If your process runs in master soc, please do not set this option cause central node
                 * in master soc is choosed automatically */
                spAppConfig->m_bCentralNode = true;
                break;
#if !NV_IS_SAFETY
            case 'c':
                if (optarg) {
                    currentPipelineGroup.sDynamicConfigName = optarg;
                } else {
                    LOG_ERR("dynamic config name must be specified with option `c`.");
                    return NvError_BadParameter;
                }
                break;
            case 'm': {
                std::vector<uint32_t> masks;
                char mask[kMaxMaskSize] = { '\0' };
                memcpy(mask, optarg, std::min((int)strlen(optarg), (int)kMaxMaskSize));
                char *token = std::strtok(mask, " ");
                while (token != NULL) {
                    masks.push_back(std::stoi(token, nullptr, 16));
                    token = std::strtok(NULL, " ");
                }
                currentPipelineGroup.vuMasks = masks;
            } break;
#endif
            case 'v':
                spAppConfig->m_uVerbosity = std::atoi(optarg);
                break;
            case 't':
                if (optarg) {
                    currentPipelineGroup.sStaticConfigName = optarg;
                } else {
                    LOG_ERR("static config name must be specified with option `t`.");
                    return NvError_BadParameter;
                }
                break;
            case 'l': {
                if (optarg == NULL && optind < argc && argv[optind][0] != '-') {
                    optarg = argv[optind++];
                }
                if (optarg) {
                    if (!strcasecmp(optarg, "configs") || !strcasecmp(optarg, "c")) {
                        ShowConfigs();
                    } else if (!strcasecmp(optarg, "elements") || !strcasecmp(optarg, "e")) {
                        ShowElements();
                    } else if (strchr(optarg, '=')) {
                        const char *pArg = optarg;
                        std::string sKey;
                        std::string sValue;
                        NvError error = GetToken(&pArg, "=", sKey);
                        if (NvError_Success == error && sKey == "element") {
                            ++pArg;
                            error = GetToken(&pArg, ":", sValue);
                            if (NvError_Success == error && !sValue.empty()) {
                                ShowElementOptions(sValue.c_str(), true);
                            } else {
                                ShowListHelp(optarg);
                            }
                        } else {
                            ShowListHelp(optarg);
                        }
                    } else if (!strcasecmp(optarg, "pipelines") || !strcasecmp(optarg, "p")) {
                        ShowPipelines(spAppConfig);
                    } else {
                        ShowListHelp(optarg);
                    }
                } else {
                    ShowConfigs();
                }
                return NvError_EndOfFile;
            } break;
            case 'N':
                if (optarg) {
                    spAppConfig->m_sNitoFolderPath = optarg;
                } else {
                    LOG_ERR("Nito folder path must be specified with option `N`.");
                    return NvError_BadParameter;
                }
                break;
            case 'I':
                spAppConfig->m_bIgnoreError = true;
                break;
            case 'k':
                spAppConfig->m_uFrameFilter = atoi(optarg);
                break;
            case 'r':
                spAppConfig->m_uRunDurationSec = atoi(optarg);
                break;
            case 'V':
                spAppConfig->m_bShowVersion = true;
                break;
            case '7':
                spAppConfig->m_bEnableStatusManager = true;
                break;
#if !NV_IS_SAFETY
            case 'E':
                if (optarg) {
                    sEncoderType = optarg;
                } else {
                    LOG_ERR("Encoder type must be specified with option `E`.");
                    return NvError_BadParameter;
                }
                break;
#endif
            case 'P':
                sPipelineType = optarg;
                break;
            case 'p':
                currentPipelineGroup.sPipelineDescriptor = optarg;
                spAppConfig->AddPipelineGroup(currentPipelineGroup);
                currentPipelineGroup = PipelineGroup(); // Reset for next pipeline
                spAppConfig->UpdateByPipelineDescStr(optarg);
                break;

            case 'K':
                spAppConfig->m_bEnableProfiling = true;
                if (optarg == nullptr && optind < argc && argv[optind][0] != '-') {
                    optarg = argv[optind++];
                }
                if (optarg) {
                    if (!strcasecmp(optarg, "pipeline") || !strcasecmp(optarg, "p")) {
                        spAppConfig->m_uProfilingMode = ProfilingMode::PIPELINE;
                    } else if (!strcasecmp(optarg, "full") || !strcasecmp(optarg, "f")) {
                        spAppConfig->m_uProfilingMode = ProfilingMode::FULL;
                    } else {
                        std::cout << "Unknown profiling mode \'" << optarg
                                  << "\'. Available modes are pipeline|p|full|f." << std::endl;
                        return NvError_BadParameter;
                    }
                }
                break;

            case 'M':
                spAppConfig->m_bIsMaxPerfSampleNumSpecified = true;
                spAppConfig->m_uMaxPerfSampleNum = atoi(optarg);
                break;

            case 'S':
                spAppConfig->m_bSavePerfData = true;
                if (optarg == nullptr && optind < argc && argv[optind][0] != '-') {
                    spAppConfig->m_sPerfDataSaveFolder = argv[optind++];
                }
                break;
        }
    }

    if (bShowHelp) {
        ShowUsage();
        return NvError_EndOfFile;
    }

    // Display is currently not supported for NvSciBufPath
    if ((spAppConfig->m_uFrameFilter < 1) || (spAppConfig->m_uFrameFilter > 5)) {
        std::cout << "Invalid value of frame filter, the range is 1-5\n";
        return NvError_BadParameter;
    }

#if !NV_IS_SAFETY
    if (!sEncoderType.empty() && ParseEncoderTypeStr(sEncoderType, spAppConfig->m_eEncType) != NvError_Success) {
        std::cout << "Unsupported Encoder Type, h264, h265\n";
        return NvError_BadParameter;
    }
#endif

    if (!sPipelineType.empty() &&
        ParsePipelineTypeStr(sPipelineType, spAppConfig->m_ePipelineType) != NvError_Success) {
        std::cout << "Unsupported pipeline Type, supported: n, sp, sc \n";
        return NvError_BadParameter;
    }

    if (spAppConfig->m_vPipelineGroups.empty()) {
        // ./multicast
        spAppConfig->AddPipelineGroup(currentPipelineGroup);
    }
    if (NvError_Success != spAppConfig->CheckPipelineGroups()) {
        std::cout << "Input pipeline groups are not valid.\n";
        return NvError_BadParameter;
    }
#if NV_BUILD_DOS7
    if (spAppConfig->m_ePipelineType != PipelineType::NormalPipeline) {
        std::cout << "TBD: Sentry Mode Transition is not supported in current version.\n";
        return NvError_BadParameter;
    }
#endif

    if (!spAppConfig->m_bEnableProfiling && spAppConfig->m_bIsMaxPerfSampleNumSpecified) {
        std::cout << "Max perf sample number specified but profiling not enabled.\n";
        return NvError_EndOfFile;
    }
    if (!spAppConfig->m_bEnableProfiling && spAppConfig->m_bSavePerfData) {
        std::cout << "Perf data folder specified but profiling not enabled.\n";
        return NvError_EndOfFile;
    }
    if (spAppConfig->m_bSavePerfData) {
        if (spAppConfig->m_sPerfDataSaveFolder.back() != '/') {
            spAppConfig->m_sPerfDataSaveFolder += '/';
        }
        DIR *dir = opendir(spAppConfig->m_sPerfDataSaveFolder.c_str());
        if (!dir) {
            if (ENOENT == errno) {
                LOG_INFO("Profilng data save folder doesn't exist. Create %s",
                         spAppConfig->m_sPerfDataSaveFolder.c_str());
                int ret = mkdir(spAppConfig->m_sPerfDataSaveFolder.c_str(), 0777);
                if (ret != 0) {
                    LOG_ERR("Failed to create %s. Error %s", spAppConfig->m_sPerfDataSaveFolder.c_str(),
                            strerror(errno));
                    return NvError_DirOperationFailed;
                }
            } else {
                LOG_ERR("Failed to open perf data folder. Error %s.", strerror(errno));
                return NvError_DirOperationFailed;
            }
        } else {
            closedir(dir);
        }
    }

    return NvError_Success;
}

NvError CCmdLineParser::ParseEncoderTypeStr(const std::string &sEncoderType, EncoderType &eEncoderType)
{
    if (sEncoderType == "h264") {
        eEncoderType = EncoderType::H264;
    } else if (sEncoderType == "h265") {
        eEncoderType = EncoderType::H265;
    } else {
        std::cout << "Unsupported encoder type! supported: h264, h265\n";
        return NvError_BadParameter;
    }
    return NvError_Success;
}

NvError CCmdLineParser::ParsePipelineTypeStr(const std::string &sPipelineType, PipelineType &ePipelineType)
{
    if (sPipelineType == "n") {
        ePipelineType = PipelineType::NormalPipeline;
    } else if (sPipelineType == "sp") {
        ePipelineType = PipelineType::SentryPipelineProducer;
    } else if (sPipelineType == "sc") {
        ePipelineType = PipelineType::SentryPipelineConsumer;
    } else {
        std::cout << "Unsupported pipeline type! supported: n, sp, sc \n";
        return NvError_BadParameter;
    }
    return NvError_Success;
}

void CCmdLineParser::ShowUsage()
{
    // clang-format off
    std::cout << "Usage:\n";
    std::cout << "-h or --help                               :Prints this help\n";
#if !NV_IS_SAFETY
    std::cout << "-c or --platform-config 'name'             :Specify dynamic platform configuration, which is fetched via SIPL Query.\n";
    std::cout << "--link-enable-masks 'masks'                :Enable masks for links on each deserializer connected to CSI\n";
    std::cout << "                                           :masks is a list of masks for each deserializer.\n";
    std::cout << "                                           :Eg: '0x0000 0x1101 0x0000 0x0000' disables all but links 0, 2 and 3 on CSI-CD interface\n";
#endif // !NV_IS_SAFETY
    std::cout << "-v or --verbosity <level>                  :Set verbosity\n";
#if !NV_IS_SAFETY
    std::cout << "                                           :Supported values (default: 1)\n";
    std::cout << "                                           : " << INvSIPLTrace::LevelNone << " (None)\n";
    std::cout << "                                           : " << INvSIPLTrace::LevelError << " (Errors)\n";
    std::cout << "                                           : " << INvSIPLTrace::LevelWarning << " (Warnings and above)\n";
    std::cout << "                                           : " << INvSIPLTrace::LevelInfo << " (Infos and above)\n";
    std::cout << "                                           : " << INvSIPLTrace::LevelDebug << " (Debug and above)\n";
#endif // !NV_IS_SAFETY
    std::cout << "-t <platformCfgName>                       :Specify static platform configuration, which is defined in header files, default is F008A120RM0AV2_CPHY_x4\n";
    std::cout << "-l                                         :List all the supported configs, elements, element options or pipelines.\n";
    std::cout << "                                           :Supported values (default: configs)\n";
    std::cout << "                                           : c or configs (lists all the supported configs)\n";
    std::cout << "                                           : e or elements (lists all the supported elements)\n";
    std::cout << "                                           : element=`element name` (lists all the element specific options)\n";
    std::cout << "                                           : pipelines (lists all the supported pipelines)\n";
    std::cout << "-N <folder>                                :Path to folder containing NITO files\n";
    std::cout << "-I                                         :Ignore the fatal error\n";
    std::cout << "-p                                         :Pipeline description string.\n";
    std::cout << "                                           :Bultin supported pipeline\n";
    std::cout << "                                           : 'enc' is the abbrivation of one to many pipeline.\n";
    std::cout << "                                           : 'cuda' is the abbrivation of many to one pipeline.\n";
    std::cout << "                                           : 'sipl' is the abbrivation of sipl pipeline.\n";
    std::cout << "                                           : 'virtual' is the abbrivation of virtual pipeline.\n";
    std::cout << "                                           : 'display' is the abbrivation of display pipeline.\n";
    std::cout << "                                           : 'passthrough' is the abbrivation of passthrough pipeline.\n";
    std::cout << "                                           : 'stitch' is the abbrivation of stitch pipeline.\n";
    std::cout << "-k or --frameFilter <n>                    :Process every Nth frame, range:[1, 5], (default: 1)\n";
    std::cout << "-V or --version                            :Show version\n";
    std::cout << "-r or --runfor <seconds>                   :Exit application after n seconds\n";
    std::cout << "-C or --centralNode                        :Specify current process is a central node of contrl channel. Only intended for slave soc.\n";
    std::cout << "                                           :Processes in master soc don't need to specify.\n";
#if !NV_IS_SAFETY
    std::cout << "-E 'type'                                  :EncoderType: h264(default), h265\n";
#endif // !NV_IS_SAFETY
#if !NV_BUILD_DOS7
    std::cout << "-P 'pipelineType'                          :PipelineType: n(default), sp, sc\n";
#endif
    std::cout << "-K or --profile <mode>                     :Enable profiling.\n";
    std::cout << "                                           :Available mode (default: pipeline)\n";
    std::cout << "                                           : p or pipeline show the pipeline perf data only.\n";
    std::cout << "                                           : f or full show both the module and pipeline perf data.\n";
    std::cout << "-M or --maxSampleNum <number>              :Specify the maximum perf sample number(default: 10000).\n";
    std::cout << "-S or --saveProfilingData <folder>         :Save the perf data to the specified folder(default:nvsipl_multicast_perfs).\n";
    std::cout << "-7 or --statusManagerEnable                :Enable status manager, Once this flag been set, this application will be managed by status_manage.\n";
    // clang-format on
    return;
}

void CCmdLineParser::ShowConfigs()
{
#if !NV_IS_SAFETY
    std::cout << "Dynamic platform configurations:\n";
    auto pQuery = INvSIPLQuery::GetInstance();
    if (pQuery == nullptr) {
        std::cout << "INvSIPLQuery::GetInstance() failed\n";
    } else {
        auto error = toNvError(pQuery->ParseDatabase());
        if (error != NvError_Success) {
            std::cout << "INvSIPLQuery::ParseDatabase failed\n";
        }

        for (const auto &cfg : pQuery->GetPlatformCfgList()) {
            std::cout << "\t" << std::setw(35) << std::left << cfg->platformConfig << ":" << cfg->description
                      << std::endl;
        }
    }
    std::cout << "Static platform configurations:\n";
#endif
    std::cout << "\t" << std::setw(35) << std::left << platformCfgAr0820.platformConfig << ":"
              << platformCfgAr0820.description << std::endl;
    std::cout << "\t" << std::setw(35) << std::left << platformCfgIMX623VB2.platformConfig << ":"
              << platformCfgIMX623VB2.description << std::endl;
    std::cout << "\t" << std::setw(35) << std::left << platformCfgIMX728VB2.platformConfig << ":"
              << platformCfgIMX728VB2.description << std::endl;
    std::cout << "\t" << std::setw(35) << std::left << platformCfgMax96712TPGYUV.platformConfig << ":"
              << platformCfgMax96712TPGYUV.description << std::endl;
    std::cout << "\t" << std::setw(35) << std::left << platformCfgMax96712TPGYUV_5m.platformConfig << ":"
              << platformCfgMax96712TPGYUV_5m.description << std::endl;
    std::cout << "\t" << std::setw(35) << std::left << platformCfgIsx031x2.platformConfig << ":"
              << platformCfgIsx031x2.description << std::endl;
    std::cout << "\t" << std::setw(35) << std::left << platformCfgIsx031x4.platformConfig << ":"
              << platformCfgIsx031x4.description << std::endl;
}

void CCmdLineParser::ShowElements()
{
    CElementDescriptionCollector &collector = CElementDescriptionCollector::GetInstance();
    std::vector<std::pair<std::string, std::string>> elementDescriptions = collector.GetElementDescriptions();
    if (!elementDescriptions.empty()) {
        const auto flags = std::cout.flags();
        std::cout << "nvsipl_multicast elements:" << std::endl;
        size_t maxSize = 0;
        for (const auto &elementDescription : elementDescriptions) {
            maxSize = std::max(maxSize, elementDescription.first.size());
        }
        ++maxSize;
        for (const auto &elementDescription : elementDescriptions) {
            std::cout << "\t" << std::setw(maxSize) << std::left << elementDescription.first << " "
                      << elementDescription.second << std::endl;
        }
        std::cout.flags(flags);
    }
}

void CCmdLineParser::ShowListHelp(const char *pListOption)
{
    std::cout << "Invliad option `" << (pListOption ? pListOption : "null")
              << "'.Valid options are empty string and [c|configs|e|elements|element=`element name`|p|pipelines]."
              << std::endl;
}

void CCmdLineParser::ShowElementOptions(const char *pElementName, bool bIgnoreCase)
{
    if (pElementName && *pElementName != '\0') {
        CElementDescriptionCollector &collector = CElementDescriptionCollector::GetInstance();
        const CElementDescription *pElementDescription = collector.FindElementDescription(pElementName, bIgnoreCase);
        if (pElementDescription) {
            size_t maxSize = strlen(pElementName);
            if (pElementDescription->pParentOptionsTable) {
                for (auto &option : *pElementDescription->pParentOptionsTable) {
                    maxSize = std::max(maxSize, option.first.size());
                }
            }

            if (pElementDescription->pOptionsTable) {
                for (auto &option : *pElementDescription->pOptionsTable) {
                    maxSize = std::max(maxSize, option.first.size());
                }
            }

            ++maxSize;

            const auto flags = std::cout.flags();
            std::cout << pElementName << ":" << std::endl;
            std::cout << "\t" << std::setw(maxSize) << std::left << pElementDescription->pDescription << std::endl;

            if (pElementDescription->pParentOptionsTable) {
                std::cout << "BaseModule options:" << std::endl;
                for (auto &option : *pElementDescription->pParentOptionsTable) {
                    std::string sTypeString = COptionParser::ToString(option.second.type);
                    sTypeString = "<" + sTypeString + ">";
                    std::cout << "\t" << std::setw(maxSize) << std::left << option.first << std::setw(10) << sTypeString
                              << option.second.sHelp << std::endl;
                }
            }

            std::cout << pElementName << " options:" << std::endl;
            if (pElementDescription->pOptionsTable) {
                for (auto &option : *pElementDescription->pOptionsTable) {
                    std::string sTypeString = COptionParser::ToString(option.second.type);
                    sTypeString = "<" + sTypeString + ">";
                    std::cout << "\t" << std::setw(maxSize) << std::left << option.first << std::setw(10) << sTypeString
                              << option.second.sHelp << std::endl;
                }
            }
            std::cout.flags(flags);
        } else {
            std::cerr << "Uknown element " << pElementName << std::endl;
        }
    } else {
        std::cerr << "Invalid element " << std::endl;
    }
}

void CCmdLineParser::ShowPipelines(const std::shared_ptr<CAppCfg> &spAppConfig)
{
    if (spAppConfig) {
        PipelineTable pipelineTable = nullptr;
        uint32_t uPipelineTableSize = 0;
        spAppConfig->GetPipelineTable(pipelineTable, uPipelineTableSize);

        std::cout << "nvsipl_multicast pipelines:" << std::endl;
        size_t maxSize = 0;
        if (pipelineTable && uPipelineTableSize > 0) {
            for (uint32_t uIndex = 0; uIndex < uPipelineTableSize; ++uIndex) {
                maxSize = std::max(maxSize, std::strlen(pipelineTable[uIndex][0]));
            }
            ++maxSize;
            for (uint32_t uIndex = 0; uIndex < uPipelineTableSize; ++uIndex) {
                std::cout << "\t" << std::setw(maxSize) << std::left << pipelineTable[uIndex][0] << " "
                          << pipelineTable[uIndex][1] << std::endl;
            }
        }
    } else {
        std::cerr << "CAppCfg is null" << std::endl;
    }
}