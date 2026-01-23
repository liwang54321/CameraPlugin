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

#ifndef CTHREADPOOL_HPP
#define CTHREADPOOL_HPP

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <queue>
#include <thread>
#include <vector>

#include "Common.hpp"
#include "CUtils.hpp"

class CThreadPool
{
  public:
    CThreadPool(uint32_t uThreadNum);
    ~CThreadPool() = default;
    NvError Init();
    void Deinit();

    template <class F, class... Args>
    auto SubmitTask(F &&f, Args &&...args) -> std::shared_ptr<std::future<typename std::result_of<F(Args...)>::type>>
    {
        if (m_bStop) {
            LOG_ERR("Failed to submit task. Thread pool is stopped");
            return nullptr;
        }
        using return_type = typename std::result_of<F(Args...)>::type;
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        auto result = std::make_shared<std::future<return_type>>(task->get_future());
        {
            std::lock_guard<std::mutex> lk(m_queueMutex);
            // insert into task queue
            m_taskQueue.emplace([task]() { (*task)(); });
        }
        m_cvQueueNotEmpty.notify_one();
        return result;
    };

  private:
    void WorkerThreadFunc();

  private:
    uint32_t m_uThreadNum;
    std::atomic<bool> m_bStop{ false };
    std::queue<std::function<void()>> m_taskQueue;
    std::vector<std::thread> m_vWorkers;
    std::mutex m_queueMutex;
    std::condition_variable m_cvQueueNotEmpty;
};

#endif