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

#include "CTimer.hpp"

CTimer::CTimer(uint32_t uThreadPoolSize)
    : m_uThreadNum(uThreadPoolSize)
{
}

NvError CTimer::Init()
{
    m_upThreadPool = std::make_unique<CThreadPool>(m_uThreadNum);
    CHK_PTR_AND_RETURN(m_upThreadPool, "Create thread pool");
    auto error = m_upThreadPool->Init();
    CHK_ERROR_AND_RETURN(error, "Threadpool init");

    m_upTimerThread = std::make_unique<std::thread>(&CTimer::TimerThreadFunc, this);
    CHK_PTR_AND_RETURN(m_upTimerThread, "Create timer pool");
    return NvError_Success;
}

void CTimer::Deinit()
{
    LOG_DBG("Enter: CTimer::Deinit()\n");
    m_bStop = true;
    m_cvTask.notify_all();
    if (m_upTimerThread && m_upTimerThread->joinable()) {
        m_upTimerThread->join();
        m_upTimerThread.reset(nullptr);
    }
    m_upThreadPool->Deinit();
    m_upThreadPool.reset(nullptr);
    LOG_DBG("Exit: CTimer::Deinit()\n");
}

void CTimer::TimerThreadFunc()
{
    pthread_setname_np(pthread_self(), "TimerThrd");
    while (!m_bStop) {
        std::unique_lock<std::mutex> lk(m_queueMutex);
        m_cvTask.wait(lk, [this] { return (m_bStop || !m_taskQueue.empty()); });
        if (m_bStop) {
            return;
        }
        Task curTask = m_taskQueue.top();
        auto now = std::chrono::steady_clock::now();
        if (now >= curTask.nextRunTime) {
            m_taskQueue.pop();
            m_upThreadPool->SubmitTask(curTask.func);
            curTask.nextRunTime += curTask.interval;
            m_taskQueue.push(curTask);
        } else {
            m_cvTask.wait_for(lk, std::chrono::duration_cast<std::chrono::nanoseconds>(curTask.nextRunTime - now));
        }
    }
}

void CTimer::AddTask(std::function<void()> taskFunc, uint64_t uIntervalMs)
{
    Task task = { .func = taskFunc,
                  .interval = std::chrono::milliseconds(uIntervalMs),
                  .nextRunTime = std::chrono::steady_clock::now() + std::chrono::milliseconds(uIntervalMs) };
    std::lock_guard<std::mutex> lk(m_queueMutex);
    m_taskQueue.push(std::move(task));
    m_cvTask.notify_one();
}