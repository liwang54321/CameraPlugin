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

#include "CElementDescriptionCollector.hpp"

extern const CElementDescription ipcSrcDescription;
extern const CElementDescription ipcDstDescription;
extern const CElementDescription cudaDescription;
extern const CElementDescription encDescription;
extern const CElementDescription virtualDstDescription;
extern const CElementDescription virtualSrcDescription;
extern const CElementDescription vicDescription;
extern const CElementDescription siplDescription;
extern const CElementDescription wfdDescription;
extern const CElementDescription fileSrcDescription;
extern const CElementDescription pvaDescription;

CElementDescriptionCollector &CElementDescriptionCollector::GetInstance()
{
    static CElementDescriptionCollector instance;
    return instance;
}

const CElementDescription *CElementDescriptionCollector::FindElementDescription(const char *pElementName,
                                                                                bool bIgnoreCase)
{
    const CElementDescription *pElementDescription = nullptr;
    if (pElementName) {
        int result = 0;
        for (auto it : m_vElementDescriptions) {
            result = bIgnoreCase ? strcasecmp(pElementName, it->pName) : strcmp(pElementName, it->pName);
            if (!result) {
                pElementDescription = it;
                break;
            }
        }
    }
    return pElementDescription;
}

std::vector<std::pair<std::string, std::string>> CElementDescriptionCollector::GetElementDescriptions()
{
    std::vector<std::pair<std::string, std::string>> vElementDescriptions;
    for (const CElementDescription *pElementDescription : m_vElementDescriptions) {
        vElementDescriptions.emplace_back(pElementDescription->pName, pElementDescription->pDescription);
    }
    return vElementDescriptions;
}

const OptionTable *CElementDescriptionCollector::GetElementOption(const char *pElementName)
{
    const OptionTable *pOptionsTable = nullptr;
    const CElementDescription *pElementDescription = FindElementDescription(pElementName);
    if (pElementDescription) {
        pOptionsTable = pElementDescription->pOptionsTable;
    }
    return pOptionsTable;
}

const OptionTable *CElementDescriptionCollector::GetElementOption(const std::string &sModuleName)
{
    return GetElementOption(sModuleName.c_str());
}

CElementDescriptionCollector::CElementDescriptionCollector()
{
    m_vElementDescriptions.emplace_back(&ipcSrcDescription);
    m_vElementDescriptions.emplace_back(&ipcDstDescription);
    m_vElementDescriptions.emplace_back(&cudaDescription);
    m_vElementDescriptions.emplace_back(&encDescription);
    m_vElementDescriptions.emplace_back(&virtualDstDescription);
    m_vElementDescriptions.emplace_back(&virtualSrcDescription);
    m_vElementDescriptions.emplace_back(&vicDescription);
    m_vElementDescriptions.emplace_back(&siplDescription);
    m_vElementDescriptions.emplace_back(&wfdDescription);
    m_vElementDescriptions.emplace_back(&fileSrcDescription);
    m_vElementDescriptions.emplace_back(&pvaDescription);
}

CElementDescriptionCollector::~CElementDescriptionCollector() {}