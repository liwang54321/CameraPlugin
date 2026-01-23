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

#include "CThreadPool.hpp"

CThreadPool::CThreadPool(uint32_t uThreadNum)
    : m_uThreadNum(uThreadNum)
{
}

NvError CThreadPool::Init()
{
    for (uint32_t i = 0; i < m_uThreadNum; i++) {
        m_vWorkers.emplace_back([this] { this->WorkerThreadFunc(); });
    }
    return NvError_Success;
}

void CThreadPool::WorkerThreadFunc()
{
    pthread_setname_np(pthread_self(), "WorkerThrd");
    while (!m_bStop || !m_taskQueue.empty()) {
        std::unique_lock<std::mutex> lk(m_queueMutex);
        while (m_taskQueue.empty() && !m_bStop) {
            m_cvQueueNotEmpty.wait(lk);
        }
        if (!m_taskQueue.empty()) {
            auto task = std::move(m_taskQueue.front());
            m_taskQueue.pop();
            lk.unlock();
            task();
        }
    }
}

void CThreadPool::Deinit()
{
    LOG_DBG("Enter: CThreadPool::Deinit()\n");
    m_bStop = true;
    m_cvQueueNotEmpty.notify_all();
    for (auto &worker : m_vWorkers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    LOG_DBG("Exit: CThreadPool::Deinit()\n");
}