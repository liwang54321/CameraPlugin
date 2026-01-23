/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef CPEERVALIDATIOR_H
#define CPEERVALIDATIOR_H

#include "CUtils.hpp"

class CAppCfg;
class CPeerValidator
{
  public:
    CPeerValidator(CAppCfg *pAppConfig)
        : m_pAppCfg(pAppConfig) {};
    CPeerValidator() = default;

    NvError SendValidationInfo(const NvSciStreamBlock handle);
    NvError Validate(const NvSciStreamBlock handle);

  private:
    void ComposeValidationInfo(std::string &outputInfo);
    CAppCfg *m_pAppCfg;
};

#endif