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

#ifndef CEVENTHANDLER_H
#define CEVENTHANDLER_H

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <thread>
#include <unistd.h>
#include <functional>
#include "Common.hpp"
#include "CUtils.hpp"
#include "CConfig.hpp"

template <typename T> class CEventHandler
{
  public:
    CEventHandler() {}
    ~CEventHandler() {};

    NvError StartThread()
    {
        if (!m_handler) {
            LOG_ERR("Please call RegisterHandler firstly!\n");
            return NvError_InvalidState;
        }

        if (!m_bRunning) {
            m_bRunning = true;
            m_bStop = false;
            m_upThread = std::make_unique<std::thread>(&CEventHandler::ThreadFunc, this);
        } else {
            if (m_bStop) {
                m_bStop = false;
                m_conditionVar.notify_all();
            }
        }

        return NvError_Success;
    }

    void QuitThread()
    {
        if (m_upThread.get() && m_upThread->joinable()) {
            {
                std::unique_lock<std::mutex> lk(m_mutex);
                m_bRunning = false;
                m_bStop = false;
                m_conditionVar.notify_all();
            }

            m_upThread->join();
            m_upThread.reset();
        }
    }

    void StopThread()
    {
        if (!m_bStop) {
            std::unique_lock<std::mutex> lk(m_mutex);
            m_bStop = true;
            m_stopConditionVar.wait(lk);
        }
    }

    NvError RegisterHandler(EventStatus (T::*handle)(), T *object, bool bContinue = true)
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        while (m_bRunning && !m_bStop) {
            m_stopConditionVar.wait(lk);
        }
        lk.unlock();
        m_handler = std::bind(handle, object);
        m_object = object;
        m_bContinue = bContinue;
        return NvError_Success;
    }

  private:
    void ThreadFunc()
    {
        PLOG_DBG("Enter: CEventHandler::ThreadFunc()\n");

        std::string threadName = ShortenName(m_object->GetName());
        pthread_setname_np(pthread_self(), threadName.c_str());

        while (m_bRunning) {
            EventStatus status = m_handler();

            switch (status) {
                case EventStatus::OK:
                    if (m_bContinue) {
                        /*
                         * Continue to invoke the handler function until not EventStatus::OK.
                         */
                        continue;
                    } else {
                        break;
                    }
                case EventStatus::ERROR:
                case EventStatus::QUITTED:
                    m_bStop = true;
                    break;
                default:
                    break;
            }

            std::unique_lock<std::mutex> lk(m_mutex);
            if (m_bStop && m_bRunning) {
                m_stopConditionVar.notify_all();
                m_conditionVar.wait(lk);
            }
        }

        PLOG_DBG("Exit: CEventHandler::ThreadFunc()\n");
    }
    inline const std::string &GetName() { return m_object->GetName(); }

    std::atomic<bool> m_bRunning{ false };
    std::unique_ptr<std::thread> m_upThread;
    std::atomic<bool> m_bStop{ false };
    std::condition_variable m_conditionVar;
    std::condition_variable m_stopConditionVar;
    std::mutex m_mutex;
    std::function<EventStatus()> m_handler;
    T *m_object = nullptr;
    bool m_bContinue = true;
};

#endif
