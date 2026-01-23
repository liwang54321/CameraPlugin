/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <iostream>
#include <pthread.h>
#include <sys/select.h>

#include <thread>
#include <unistd.h>

#include "CManager.hpp"
#include "CStatusManager.hpp"
#include "CCmdLineParser.hpp"

#if NV_BUILD_DOS7
constexpr uint32_t MAJOR_VER = 4U; /**< Indicates the major revision. */
constexpr uint32_t MINOR_VER = 5U; /**< Indicates the minor revision. */
constexpr uint32_t PATCH_VER = 0U; /**< Indicates the patch revision. */
#else
constexpr uint32_t MAJOR_VER = 3U; /**< Indicates the major revision. */
constexpr uint32_t MINOR_VER = 5U; /**< Indicates the minor revision. */
constexpr uint32_t PATCH_VER = 0U; /**< Indicates the patch revision. */
#endif

std::unique_ptr<CManager> upManager;

/** Signal handler.*/
static void SigHandler(int signum)
{
    LOG_WARN("Received signal: %u. Quitting\n", signum);
    signal(SIGINT, SIG_IGN);
    signal(SIGTERM, SIG_IGN);
    signal(SIGQUIT, SIG_IGN);
    signal(SIGHUP, SIG_IGN);

    if (upManager)
        upManager->Quit();

    signal(SIGINT, SIG_DFL);
    signal(SIGTERM, SIG_DFL);
    signal(SIGQUIT, SIG_DFL);
    signal(SIGHUP, SIG_DFL);
}

/** Sets up signal handler.*/
static void SigSetup()
{
    struct sigaction action{};
    action.sa_handler = SigHandler;

    sigaction(SIGINT, &action, nullptr);
    sigaction(SIGTERM, &action, nullptr);
    sigaction(SIGQUIT, &action, nullptr);
    sigaction(SIGHUP, &action, nullptr);

    signal(SIGILL, SIG_DFL);
    signal(SIGFPE, SIG_DFL);
    signal(SIGSEGV, SIG_DFL);
    signal(SIGABRT, SIG_DFL);
}

int main(int argc, char *argv[])
{
    NvError error = NvError_Success;
    std::shared_ptr<CAppCfg> appCfg = std::make_shared<CAppCfg>();

    if (argc > 1) {
        CCmdLineParser cmdline;
        error = cmdline.Parse(argc, argv, appCfg);
        /*
         * NvError_EndOfFile is not an error.
         */
        if (NvError_EndOfFile == error) {
            return 0;
        }
        CHK_ERROR_AND_RETURN(error, "CCmdLineParser::Parse");

        if (appCfg->IsVersionShown()) {
            std::cout << "nvsipl_multicast " << MAJOR_VER << "." << MINOR_VER << "." << PATCH_VER << std::endl;
            return 0;
        }
    }

    LOG_INFO("Setting up signal handler\n");
    SigSetup();

    auto quitStatus = EventStatus::OK;
    auto quitStatusSet = [&quitStatus](CChannel *pChannel, EventStatus event) {
        if (EventStatus::ERROR != quitStatus) {
            // skip to update following event error if errors happened
            quitStatus = event;
        }
    };

    if (appCfg->IsStatusManagerEnabled()) {
        upManager = std::make_unique<CStatusManager>(appCfg, quitStatusSet);
    } else {
        upManager = std::make_unique<CManager>(appCfg, quitStatusSet);
    }

    error = upManager->Init();
    if (error != NvError_Success) {
        LOG_ERR("CManager init failed!\n");
        upManager->Quit();
    } else {
        upManager->Run();
    }
    upManager->DeInit();
    upManager.reset();

    if (EventStatus::QUITTED == quitStatus) {
        std::cout << "nvsipl_multicast properly quitted" << std::endl;
    }

    return 0;
}
