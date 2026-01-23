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

#ifndef CFACTORY_H
#define CFACTORY_H

#include "CBaseModule.hpp"
#include "CConfig.hpp"

using namespace nvsipl;

class CFactory
{
  public:
    static CFactory &GetInstance()
    {
        static CFactory instance;
        return instance;
    }

    static std::shared_ptr<CBaseModule> CreateModule(std::shared_ptr<CModuleCfg> spModuleCfg,
                                                     IEventListener<CBaseModule> *pListener);

  private:
    CFactory(){};
    CFactory(const CFactory &obj) = delete;
    CFactory &operator=(const CFactory &obj) = delete;
};

#endif
