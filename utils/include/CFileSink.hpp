/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef FILE_SINK_HPP
#define FILE_SINK_HPP

#include <fstream>
#include "CUtils.hpp"

class CAbstractFileSink
{
  public:
    CAbstractFileSink() = default;
    virtual ~CAbstractFileSink() = default;
    virtual NvError Init(const std::string &fileName) = 0;
    virtual void Deinit() = 0;
    virtual NvError WriteBufToFile(const uint8_t *buf, const uint32_t bufSize) = 0;
};

class CDefaultFileSink : public CAbstractFileSink
{

  public:
    CDefaultFileSink() = default;
    virtual ~CDefaultFileSink();
    NvError Init(const std::string &fileName) override;
    void Deinit() override;
    NvError WriteBufToFile(const uint8_t *buf, const uint32_t bufSize) override;

  private:
    std::string m_sFileName;
    std::ofstream m_outFile;
};

#endif // FILE_SINK_H