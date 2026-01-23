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

#ifndef CVICCONFIGURATOR_HPP
#define CVICCONFIGURATOR_HPP

#include "CUtils.hpp"

class CVicConfigurator
{
  public:
    virtual ~CVicConfigurator() = default;
    virtual NvError GetDstRect(uint32_t uSensor, NvMediaRect *pDstRect) { return NvError_Success; }
    virtual void ComputeInputRects(uint32_t uSensorCount, uint32_t uWidth, uint32_t uHeight) {}
};

class CRegionDivision : public CVicConfigurator
{
  public:
    virtual NvError GetDstRect(uint32_t uSensor, NvMediaRect *pDstRect)
    {
        if (uSensor >= m_uSensorCount) {
            LOG_ERR("GetDstRect(), invalid uSensor: %u\n", uSensor);
            return NvError_BadParameter;
        }
        *pDstRect = m_rects[uSensor];

        return NvError_Success;
    }

    void ComputeInputRects(uint32_t uSensorCount, uint32_t uWidth, uint32_t uHeight)
    {
        m_uSensorCount = uSensorCount;

        // Set up the destination rectangles
        uint16_t countPerLine = ceil(sqrt(uSensorCount));
        uint16_t xStep = uWidth / countPerLine;
        uint16_t yStep = uHeight / countPerLine;
        for (auto i = 0U; i < uSensorCount; ++i) {
            auto rowIndex = i / countPerLine;
            auto colIndex = i % countPerLine;
            uint16_t startx = colIndex * xStep;
            uint16_t starty = rowIndex * yStep;
            uint16_t endx = startx + xStep;
            uint16_t endy = starty + yStep;
            m_rects[i] = { startx, starty, endx, endy };
        }
    }

    uint32_t m_uSensorCount = 0U;
    NvMediaRect m_rects[MAX_NUM_SENSORS] = {};
};

#endif // CVICCONFIGURATOR_HPP