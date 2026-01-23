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

#ifndef CSTATUS_MANAGER_QNX_HELPER_HPP
#define CSTATUS_MANAGER_QNX_HELPER_HPP

#include "CStatusManagerService.hpp"
#include "nvdvms_client.h"

class CStatusManagerQnxHelper : public CStatusManagerOsHelper
{
  public:
    virtual NvError SetOsDvmsState(StatusMangerState statusManagerState) override;
    virtual NvError CheckOsInitDoneStatus() override;
    virtual NvError WaitForResume() override;
};

#endif //CSTATUS_MANAGER_QNX_HELPER_HPP