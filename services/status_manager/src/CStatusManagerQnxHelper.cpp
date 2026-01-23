/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "CStatusManagerQnxHelper.hpp"

constexpr int SECONDS_TIMEOUT_INITDONE = 4;
constexpr int MILLISECONDS_PER_CHECK = 200;

NvError CStatusManagerQnxHelper::SetOsDvmsState(StatusMangerState statusManagerState)
{
    NvDvmsState dvmsState;
    switch (statusManagerState) {
        case StatusMangerState::STATUSMANAGER_STATE_OPERATIONAL:
            dvmsState = NVDVMS_OPERATIONAL;
            break;
        case StatusMangerState::STATUSMANAGER_STATE_SUSPEND:
            dvmsState = NVDVMS_SUSPEND;
            break;
        case StatusMangerState::STATUSMANAGER_STATE_DEINIT_PREPARE:
            dvmsState = NVDVMS_DEINIT_PREPARE;
            break;
        case StatusMangerState::STATUSMANAGER_STATE_REINIT:
            // Reinit support from 6.0.9.3/6.5.0.0
            dvmsState = NVDVMS_REINIT;
            break;
        case StatusMangerState::STATUSMANAGER_STATE_DEINIT:
            dvmsState = NVDVMS_DEINIT;
            break;
        case StatusMangerState::STATUSMANAGER_STATE_LOW_POWER: {
            NvDvmsStatus nvDvmsStatus = nvdvms_set_power_profile(NVDVMS_SOC_OP_1);
            CHK_DVMSSTATUS_AND_RETURN(nvDvmsStatus, "nvdvms_set_power_profile");
            dvmsState = NVDVMS_INIT_DONE;
            break;
        }
        case StatusMangerState::STATUSMANAGER_STATE_FULL_POWER: {
            NvDvmsStatus nvDvmsStatus = nvdvms_set_power_profile(NVDVMS_SOC_OP_0);
            CHK_DVMSSTATUS_AND_RETURN(nvDvmsStatus, "nvdvms_set_power_profile");
            dvmsState = NVDVMS_INIT_DONE;
            break;
        }
        default:
            LOG_ERR("Invalid StatusMangerState state %d set to OS.", statusManagerState);
            return NvError_InvalidState;
    }

    NvDvmsStatus nvDvmsStatus;
    NvDvmsState currentDvmsState;
    bool bSetSuccess = false;
    do {
        if (dvmsState != NVDVMS_INIT_DONE) {
            nvDvmsStatus = nvdvms_set_vm_state(dvmsState);
            CHK_DVMSSTATUS_AND_RETURN(nvDvmsStatus, "nvdvms_set_vm_state");
        }
        nvDvmsStatus = nvdvms_get_vm_state(&currentDvmsState);
        CHK_DVMSSTATUS_AND_RETURN(nvDvmsStatus, "nvdvms_get_vm_state");
        bSetSuccess = currentDvmsState == dvmsState || (currentDvmsState == NVDVMS_INIT_DONE &&
                                                        (dvmsState == NVDVMS_SUSPEND || dvmsState == NVDVMS_REINIT));
    } while (!bSetSuccess);

    return NvError_Success;
}

NvError CStatusManagerQnxHelper::CheckOsInitDoneStatus()
{
    uint64_t uTimeElapsedSum = 0u;
    const auto sleepTime = std::chrono::milliseconds(MILLISECONDS_PER_CHECK);

    while (uTimeElapsedSum / 1000 < SECONDS_TIMEOUT_INITDONE) {
        NvDvmsState dvmsState;
        NvDvmsStatus nvDvmsStatus = nvdvms_get_vm_state(&dvmsState);
        if (nvDvmsStatus != NvDvmsSuccess) {
            LOG_ERR("nvdvms_get_vm_state fail with %x", nvDvmsStatus);
        }

        if (dvmsState != NVDVMS_INIT_DONE) {
            LOG_INFO("dvmsState not at NVDVMS_INIT_DONE, current state %d", dvmsState);
            auto oStartTime = std::chrono::steady_clock::now();
            std::this_thread::sleep_for(sleepTime);
            auto uTimeElapsedMs =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - oStartTime).count();
            uTimeElapsedSum += uTimeElapsedMs;
        } else {
            LOG_INFO("Drive OS Init Done!");
            return NvError_Success;
        }
    }

    LOG_ERR("Drive OS Init Fail!");
    return NvError_ResourceError;
}

//After setting the NVDVMS_SUSPEND to DriveOS,
//the system state will be set to NVDVMS_INIT_DONE by server io-nvdvms.
NvError CStatusManagerQnxHelper::WaitForResume()
{
    return CheckOsInitDoneStatus();
}
