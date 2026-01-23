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
#ifndef CPROFILER_HPP
#define CPROFILER_HPP

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <dlfcn.h>
#include "NvSIPLCommon.hpp"
#include "CNvPlayfairWrapper.hpp"
#include "nvscisync.h"
#include "CUtils.hpp"

using namespace nvsipl;

enum class PerfType : uint8_t
{
    INIT = 0,
    TRANSMISSION,
    SUBMISSION,
    EXECUTION,
    PIPELINE,
    TOTAL
};

const char *ToString(PerfType type);

struct Perf
{
    Perf(PerfType type, PerfData *pPerfData)
        : m_type(type)
        , m_pPerfData(pPerfData)
    {
    }
    PerfType m_type;
    PerfData *m_pPerfData;
};

class CProfiler
{
  public:
    static uint64_t GetCurrentTSC();
    static NvError ShowAndSavePerfData(PerfData *pPerfData, const std::string &sTitle, bool bSave);

  public:
    CProfiler();
    ~CProfiler();

    NvError Init(NvSciSyncModule sciSyncModule,
                 const std::string &sProfilingSaveFolder,
                 const std::string &sName,
                 bool bFenceWaiterNeeded,
                 uint32_t uMaxSampleNumber);
    void DeInit();

    NvError RecordInitBeginTime(uint64_t uBeginTime = 0U);
    NvError RecordInitEndTime(uint64_t uEndTime = 0U);

    NvError RecordTransmissionBeginTime(uint64_t uBeginTime = 0U);
    NvError RecordTransmissionEndTime(uint64_t uEndTime = 0U);
    NvError RecordTransmissionTime(uint64_t uBeginTime = 0U, uint64_t uEndTime = 0U);

    NvError RecordSubmissionBeginTime(uint64_t uBeginTime = 0U);
    NvError RecordSubmissionEndTime(uint64_t uEndTime = 0U);

    NvError RecordExecutionBeginTime(uint64_t uBeginTime = 0U);
    NvError RecordExecutionEndTime(uint64_t uEndTime = 0U);
    NvError RecordExecutionEndTime(NvSciSyncFence *pFence);
    NvError RecordExecutionTime(uint64_t uBeginTime = 0U, uint64_t uEndTime = 0U);
    NvError RecordExecutionTime(uint64_t uBeginTime, NvSciSyncFence *pFence);

    NvError RecordPipelineBeginTime(uint64_t uBeginTime = 0U);
    NvError RecordPipelineEndTime(uint64_t uEndTime = 0U);
    NvError RecordPipelineTime(uint64_t uBeginTime = 0U, uint64_t uEndTime = 0U);

    NvError RecordExecutionAndPipelineTime(uint64_t uPipelineBeginTime, NvSciSyncFence *pFence);
    NvError RecordExecutionAndPipelineTime(uint64_t uBeginTime, uint64_t uPipelineBeginTime, NvSciSyncFence *pFence);
    NvError RecordExecutionAndPipelineTime(uint64_t uBeginTime = 0U, uint64_t uEndTime = 0U);

    NvError RecordBeginTime(PerfType type, uint64_t uBeginTime = 0U);
    NvError RecordEndTime(PerfType type, uint64_t uEndTime = 0U);

    NvError GetPerf(std::vector<Perf> &vPerf);

  private:
    NvError FenceWaitThreadFunc();
    NvError WaitForFence(NvSciSyncFence *pFence);
    CProfiler(const CProfiler &) = delete;
    CProfiler(CProfiler &&) = delete;
    CProfiler &operator=(const CProfiler &) = delete;
    CProfiler &operator=(CProfiler &&) = delete;

  private:
    NvSciSyncCpuWaitContext m_cpuWaitContext = nullptr;
    uint64_t m_uInitBeginTimeMark{ 0 };
    uint64_t m_uTransmissionBeginTimeMark{ 0 };
    uint64_t m_uSubmissionBeginTimeMark{ 0 };
    uint64_t m_uExecutionBeginTimeMark{ 0 };
    uint64_t m_uPipelineBeginTimeMark{ 0 };
    PerfData m_latencies[static_cast<uint8_t>(PerfType::TOTAL)]{};
    std::thread m_FenceWaitThread;
    std::atomic<bool> m_bQuit{ false };
    std::mutex m_oFenceWaitRequestQueueMutex;
    std::condition_variable m_oFenceWaitRequestQueueCond;

    struct FenceWaitRequest
    {
      public:
        FenceWaitRequest(uint64_t uBeginTime, uint64_t uPipelineBeginTime)
            : m_uBeginTime(uBeginTime)
            , m_uPipelineBeginTime(uPipelineBeginTime)
            , m_fence(NvSciSyncFenceInitializer)
        {
        }

        NvError Init(NvSciSyncFence *pFence)
        {
            NvSciError err = NvSciSyncFenceDup(pFence, &m_fence);
            CHK_NVSCISTATUS_AND_RETURN(err, "NvSciSyncFenceDup");
            return NvError_Success;
        }

        void DeInit()
        {
            m_uBeginTime = 0U;
            m_uPipelineBeginTime = 0U;
            NvSciSyncFenceClear(&m_fence);
        }

        ~FenceWaitRequest() { DeInit(); }
        uint64_t m_uBeginTime;
        uint64_t m_uPipelineBeginTime;
        NvSciSyncFence m_fence;
    };
    std::queue<FenceWaitRequest *> m_oFenceWaitRequestQueue;
    std::unique_ptr<CNvPlayfairWrapper> m_upNvPlayfairWrapper;
};
#endif // CPROFILER_HPP