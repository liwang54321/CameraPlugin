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

#include "CFactory.hpp"

#include "CNvm2dModule.hpp"
#include "CCudaModule.hpp"
#include "CEncModule.hpp"
#include "CVirtualDstModule.hpp"
#include "CVirtualSrcModule.hpp"
#include "CSiplModule.hpp"
#include "CWFDDisplayModule.hpp"
#include "CFileSourceModule.hpp"
#include "CPvaModule.hpp"

#ifdef BUILD_VULKANSC
#include "CVulkanSCModule.hpp"
#endif

std::shared_ptr<CBaseModule> CFactory::CreateModule(std::shared_ptr<CModuleCfg> spModuleCfg,
                                                    IEventListener<CBaseModule> *pListener)
{
    LOG_DBG("CreateModule %d %s\n", spModuleCfg->m_moduleType, spModuleCfg->m_sName.c_str());

    std::shared_ptr<CBaseModule> spMod;

    switch (spModuleCfg->m_moduleType) {
        case ModuleType::VirtualDst:
            spMod = std::make_shared<CVirtualDstModule>(spModuleCfg, pListener);
            break;
        case ModuleType::Cuda:
            spMod = std::make_shared<CCudaModule>(spModuleCfg, pListener);
            break;
        case ModuleType::Nvm2d:
            spMod = std::make_shared<CNvm2dModule>(spModuleCfg, pListener);
            break;
        case ModuleType::Enc:
            spMod = std::make_shared<CEncModule>(spModuleCfg, pListener);
            break;
        case ModuleType::VirtualSrc:
            spMod = std::make_shared<CVirtualSrcModule>(spModuleCfg, pListener);
            break;
        case ModuleType::SIPL:
            spMod = std::make_shared<CSiplModule>(spModuleCfg, pListener);
            break;
        case ModuleType::Display:
            spMod = std::make_shared<CWFDDisplayModule>(spModuleCfg, pListener);
            break;
        case ModuleType::FileSource:
            spMod = std::make_shared<CFileSourceModule>(spModuleCfg, pListener);
            break;
        case ModuleType::Pva:
            spMod = std::make_shared<CPvaModule>(spModuleCfg, pListener);
            break;
#ifdef BUILD_VULKANSC
        case ModuleType::VulkanSC:
            spMod = std::make_shared<CVulkanSCModule>(spModuleCfg, pListener);
            break;
#endif
        default:
            LOG_ERR("Module Type %d is not supported!\n", spModuleCfg->m_moduleType);
            break;
    }

    return spMod;
}
