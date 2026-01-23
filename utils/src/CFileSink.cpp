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

#include "CFileSink.hpp"

CDefaultFileSink::~CDefaultFileSink()
{
    Deinit();
}

NvError CDefaultFileSink::Init(const std::string &fileName)
{
    if (fileName.empty()) {
        LOG_ERR("CDefaultFileSink::Init: filename is empty()\n");
        return NvError_BadParameter;
    }

    m_sFileName = fileName;
    m_outFile.open(m_sFileName, std::ios::binary);
    if (!m_outFile.is_open()) {
        LOG_ERR("CDefaultFileSink::Init: open file failed!\n");
        return NvError_FileOperationFailed;
    }
    return NvError_Success;
}

void CDefaultFileSink::Deinit()
{
    LOG_DBG("CDefaultFileSink::Deinit()\n");

    m_sFileName.clear();
    if (m_outFile.is_open()) {
        m_outFile.flush();
        m_outFile.close();
    }
}

NvError CDefaultFileSink::WriteBufToFile(const uint8_t *buf, const uint32_t bufSize)
{
    const char *dataAddr = reinterpret_cast<const char *>(buf);
    m_outFile.write(dataAddr, bufSize);
    if (!m_outFile.good()) {
        LOG_ERR("CDefaultFileSink::WriteBufToFile failed\n");
        return NvError_FileWriteFailed;
    }

    return NvError_Success;
}
