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

#ifndef CELEMENTDESCRIPTIONCOLLECTOR_HPP
#define CELEMENTDESCRIPTIONCOLLECTOR_HPP

#include <unordered_map>
#include <vector>
#include "COptionParser.hpp"
#include "CElementDescription.hpp"

class CElementDescriptionCollector final
{
  public:
    static CElementDescriptionCollector &GetInstance();
    const CElementDescription *FindElementDescription(const char *pElementName, bool bIgnoreCase = false);
    std::vector<std::pair<std::string, std::string>> GetElementDescriptions();
    const OptionTable *GetElementOption(const char *pElementName);
    const OptionTable *GetElementOption(const std::string &pElementName);

  private:
    std::vector<const CElementDescription *> m_vElementDescriptions;

  private:
    CElementDescriptionCollector();
    ~CElementDescriptionCollector();
    CElementDescriptionCollector(const CElementDescriptionCollector &other) = delete;
    CElementDescriptionCollector &operator=(const CElementDescriptionCollector &other) = delete;
};
#endif