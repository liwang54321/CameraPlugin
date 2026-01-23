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

#ifndef CTIMER_HPP
#define CTIMER_HPP

#include <chrono>
#include "CThreadPool.hpp"

class CTimer
{
  public:
    CTimer(uint32_t uThreadPoolSize);
    ~CTimer() = default;
    NvError Init();
    void Deinit();
    void AddTask(std::function<void()> taskFunc, uint64_t uIntervalMs);

  private:
    struct Task
    {
        std::function<void()> func;
        std::chrono::milliseconds interval;
        std::chrono::time_point<std::chrono::steady_clock> nextRunTime;
    };
    struct CompareTaskPriority
    {
        bool operator()(const Task &t1, const Task &t2) { return t1.nextRunTime > t2.nextRunTime; }
    };

    void TimerThreadFunc();

  private:
    std::atomic<bool> m_bStop{ false };
    uint32_t m_uThreadNum;
    std::priority_queue<Task, std::vector<Task>, CompareTaskPriority> m_taskQueue;
    std::mutex m_queueMutex;
    std::condition_variable m_cvTask;
    std::unique_ptr<std::thread> m_upTimerThread = { nullptr };
    std::unique_ptr<CThreadPool> m_upThreadPool = { nullptr };
};

#endif