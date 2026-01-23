/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef CSIPLMODULECALLBACK_H
#define CSIPLMODULECALLBACK_H

#include <stdint.h>

class ISiplModuleCallback
{
  public:
    virtual ~ISiplModuleCallback() {}

    /**
    * @brief call SIPLMdule callback
    * @param uCameraIdx camera id for this sipl module
    * @param uErrorId error id for this sipl module
    */
    virtual void OnError(uint32_t uCameraIdx, uint32_t uErrorId) = 0;
};

#endif // CSIPLMODULECALLBACK_H
