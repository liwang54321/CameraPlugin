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

#include <unordered_map>
#include "CProfiler.hpp"

static constexpr const uint32_t LATENCY_NAME_SIZE = 1024;
const char *ToString(PerfType type)
{
    const char *pTypeString = "unknown";
    static std::unordered_map<PerfType, const char *> perfType2StringMap = {
        { PerfType::INIT, "init" },
        { PerfType::TRANSMISSION, "transmission" },
        { PerfType::SUBMISSION, "submission" },
        { PerfType::EXECUTION, "execution" },
        { PerfType::PIPELINE, "pipeline" },
    };

    auto result = perfType2StringMap.find(type);
    if (result != perfType2StringMap.end()) {
        pTypeString = result->second;
    }

    return pTypeString;
}

uint64_t CProfiler::GetCurrentTSC()
{
    return CNvPlayfairWrapper::GetTimeMark();
}

NvError CProfiler::ShowAndSavePerfData(PerfData *pPerfData, const std::string &sTitle, bool bSave)
{
    if (!pPerfData) {
        return NvError_BadParameter;
    }

    CNvPlayfairWrapper wrapper;
    NvError error = wrapper.Init();
    CHK_ERROR_AND_RETURN(error, "CNvPlayfairWrapper::Init");

    PerfStats perfStats;
    PerfStatus perfStatus = wrapper.CalcStats(pPerfData, &perfStats, TimeUnit::MILLISECONDS);
    CHK_PERFSTATUS_AND_RETURN(perfStatus, "CalcStats");

    perfStatus = wrapper.PrintStats(pPerfData, &perfStats, TimeUnit::MILLISECONDS, sTitle.c_str(), false);
    CHK_PERFSTATUS_AND_RETURN(perfStatus, "PrintStats");

    if (bSave) {
        PerfStatus perfStatus = wrapper.DumpData(pPerfData);
        CHK_PERFSTATUS_AND_RETURN(perfStatus, "DumpData");
    }
    return NvError_Success;
}

CProfiler::CProfiler() {}

CProfiler::~CProfiler()
{
    DeInit();
}

NvError CProfiler::Init(NvSciSyncModule sciSyncModule,
                        const std::string &sFolder,
                        const std::string &sName,
                        bool bFenceWaiterNeeded,
                        uint32_t uMaxSampleNumber)
{
    NvSciError sciErr = NvSciSyncCpuWaitContextAlloc(sciSyncModule, &m_cpuWaitContext);
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncCpuWaitContextAlloc");

    m_upNvPlayfairWrapper = std::make_unique<CNvPlayfairWrapper>();
    CHK_PTR_AND_RETURN(m_upNvPlayfairWrapper, "Create CNvPlayfairWrapper");

    NvError error = m_upNvPlayfairWrapper->Init();
    CHK_ERROR_AND_RETURN(error, "CNvPlayfairWrapper::Init");

    std::unique_ptr<char[]> oLatencyName(new char[LATENCY_NAME_SIZE]);
    for (PerfType uType = PerfType::INIT; uType < PerfType::TOTAL; uType = static_cast<PerfType>((uint8_t)uType + 1)) {
        snprintf(oLatencyName.get(), LATENCY_NAME_SIZE, "%s%s_%s.csv", sFolder.c_str(), sName.c_str(), ToString(uType));
        PerfStatus perfStatus = m_upNvPlayfairWrapper->ConstructPerfData(&m_latencies[static_cast<uint8_t>(uType)],
                                                                         uMaxSampleNumber, oLatencyName.get());
        CHK_PERFSTATUS_AND_RETURN(perfStatus, "ConstructPerfData");
    }

    if (bFenceWaiterNeeded) {
        m_FenceWaitThread = std::thread(&CProfiler::FenceWaitThreadFunc, this);
    }
    return NvError_Success;
}

void CProfiler::DeInit()
{
    m_bQuit = true;
    m_oFenceWaitRequestQueueCond.notify_one();
    if (m_FenceWaitThread.joinable()) {
        m_FenceWaitThread.join();
    }

    {
        /*
         * Delete all the remaining FenceWaitRequest
         */
        std::unique_lock<std::mutex> queueLock(m_oFenceWaitRequestQueueMutex);
        while (!m_oFenceWaitRequestQueue.empty()) {
            FenceWaitRequest *pFenceWaitRequest(m_oFenceWaitRequestQueue.front());
            m_oFenceWaitRequestQueue.pop();
            if (pFenceWaitRequest) {
                delete pFenceWaitRequest;
                pFenceWaitRequest = nullptr;
            }
        }
    }

    if (m_cpuWaitContext != nullptr) {
        NvSciSyncCpuWaitContextFree(m_cpuWaitContext);
        m_cpuWaitContext = nullptr;
    }

    for (PerfType uType = PerfType::INIT; uType < PerfType::TOTAL; uType = static_cast<PerfType>((uint8_t)uType + 1)) {
        if (m_latencies[static_cast<uint8_t>(uType)].bInitialized) {
            m_upNvPlayfairWrapper->DestroyPerfData(&m_latencies[static_cast<uint8_t>(uType)]);
        }
    }
}

NvError CProfiler::RecordInitBeginTime(uint64_t uBeginTime)
{
    return RecordBeginTime(PerfType::INIT, uBeginTime);
}

NvError CProfiler::RecordInitEndTime(uint64_t uEndTime)
{
    return RecordEndTime(PerfType::INIT, uEndTime);
}

NvError CProfiler::RecordTransmissionBeginTime(uint64_t uBeginTime)
{
    return RecordBeginTime(PerfType::TRANSMISSION, uBeginTime);
}

NvError CProfiler::RecordTransmissionEndTime(uint64_t uEndTime)
{
    return RecordEndTime(PerfType::TRANSMISSION, uEndTime);
}

NvError CProfiler::RecordTransmissionTime(uint64_t uBeginTime, uint64_t uEndTime)
{
    NvError error = RecordTransmissionBeginTime(uBeginTime);
    CHK_ERROR_AND_RETURN(error, "RecordTransmissionBeginTime");

    error = RecordTransmissionEndTime(uEndTime);
    CHK_ERROR_AND_RETURN(error, "RecordTransmissionEndTime");

    return NvError_Success;
}

NvError CProfiler::RecordSubmissionBeginTime(uint64_t uBeginTime)
{
    return RecordBeginTime(PerfType::SUBMISSION, uBeginTime);
}

NvError CProfiler::RecordSubmissionEndTime(uint64_t uEndTime)
{
    return RecordEndTime(PerfType::SUBMISSION, uEndTime);
}

NvError CProfiler::RecordExecutionBeginTime(uint64_t uBeginTime)
{
    return RecordBeginTime(PerfType::EXECUTION, uBeginTime);
}

NvError CProfiler::RecordExecutionEndTime(uint64_t uEndTime)
{
    return RecordEndTime(PerfType::EXECUTION, uEndTime);
}

NvError CProfiler::RecordExecutionEndTime(NvSciSyncFence *pFence)
{
    return RecordExecutionTime(m_uExecutionBeginTimeMark, pFence);
}

NvError CProfiler::RecordExecutionTime(uint64_t uBeginTime, uint64_t uEndTime)
{
    NvError error = RecordExecutionBeginTime(uBeginTime);
    CHK_ERROR_AND_RETURN(error, "RecordExecutionBeginTime");

    error = RecordExecutionEndTime(uEndTime);
    CHK_ERROR_AND_RETURN(error, "RecordExecutionEndTime");

    return NvError_Success;
}

NvError CProfiler::RecordExecutionTime(uint64_t uBeginTime, NvSciSyncFence *pFence)
{
    return RecordExecutionAndPipelineTime(uBeginTime, 0U, pFence);
}

NvError CProfiler::RecordPipelineBeginTime(uint64_t uBeginTime)
{
    return RecordBeginTime(PerfType::PIPELINE, uBeginTime);
}

NvError CProfiler::RecordPipelineEndTime(uint64_t uEndTime)
{
    return RecordEndTime(PerfType::PIPELINE, uEndTime);
}

NvError CProfiler::RecordPipelineTime(uint64_t uBeginTime, uint64_t uEndTime)
{
    NvError error = RecordPipelineBeginTime(uBeginTime);
    CHK_ERROR_AND_RETURN(error, "RecordPipelineBeginTime");

    error = RecordPipelineEndTime(uEndTime);
    CHK_ERROR_AND_RETURN(error, "RecordPipelineEndTime");

    return NvError_Success;
}

NvError CProfiler::RecordExecutionAndPipelineTime(uint64_t uPipelineBeginTime, NvSciSyncFence *pFence)
{
    return RecordExecutionAndPipelineTime(m_uExecutionBeginTimeMark, uPipelineBeginTime, pFence);
}

NvError CProfiler::RecordExecutionAndPipelineTime(uint64_t uBeginTime, uint64_t uPipelineBeginTime, NvSciSyncFence *pFence)
{
    NvError error = NvError_Success;
    if (!pFence) {
        LOG_ERR("NULL NvSciSyncFence");
        return NvError_BadParameter;
    }

    if (!m_FenceWaitThread.joinable()) {
        LOG_ERR("Fence waiter thread doesn't exist.");
        return NvError_InvalidState;
    }

    std::lock_guard<std::mutex> lk(m_oFenceWaitRequestQueueMutex);
    FenceWaitRequest *pFenceWaitRequest = new FenceWaitRequest(uBeginTime, uPipelineBeginTime);
    if (pFenceWaitRequest) {
        error = pFenceWaitRequest->Init(pFence);
        if (error == NvError_Success) {
            m_oFenceWaitRequestQueue.push(pFenceWaitRequest);
            m_oFenceWaitRequestQueueCond.notify_one();
        } else {
            LOG_ERR("Failed to initialize FenceWaitRequest object");
            delete pFenceWaitRequest;
        }
    } else {
        error = NvError_InsufficientMemory;
    }

    return error;
}

NvError CProfiler::RecordExecutionAndPipelineTime(uint64_t uBeginTime, uint64_t uEndTime)
{
    NvError error = RecordExecutionBeginTime(uBeginTime);
    CHK_ERROR_AND_RETURN(error, "RecordExecutionBeginTime");

    error = RecordExecutionEndTime(uEndTime);
    CHK_ERROR_AND_RETURN(error, "RecordExecutionEndTime");

    error = RecordPipelineBeginTime(uBeginTime);
    CHK_ERROR_AND_RETURN(error, "RecordPipelineBeginTime");

    error = RecordPipelineEndTime(uEndTime);
    CHK_ERROR_AND_RETURN(error, "RecordPipelineEndTime");

    return NvError_Success;
}

NvError CProfiler::RecordBeginTime(PerfType type, uint64_t uBeginTime)
{
    NvError error = NvError_Success;
    switch (type) {
        case PerfType::INIT: {
            m_uInitBeginTimeMark = uBeginTime != 0 ? uBeginTime : CNvPlayfairWrapper::GetTimeMark();
        } break;

        case PerfType::TRANSMISSION: {
            m_uTransmissionBeginTimeMark = uBeginTime != 0 ? uBeginTime : CNvPlayfairWrapper::GetTimeMark();
        } break;

        case PerfType::SUBMISSION: {
            m_uSubmissionBeginTimeMark = uBeginTime != 0 ? uBeginTime : CNvPlayfairWrapper::GetTimeMark();
        } break;

        case PerfType::EXECUTION: {
            m_uExecutionBeginTimeMark = uBeginTime != 0 ? uBeginTime : CNvPlayfairWrapper::GetTimeMark();
        } break;

        case PerfType::PIPELINE: {
            m_uPipelineBeginTimeMark = uBeginTime != 0 ? uBeginTime : CNvPlayfairWrapper::GetTimeMark();
        } break;

        default: {
            LOG_ERR("Invalid perf data type %u", type);
            error = NvError_BadParameter;
        } break;
    }

    return error;
}

NvError CProfiler::RecordEndTime(PerfType type, uint64_t uEndTime)
{
    NvError error = NvError_Success;
    switch (type) {
        case PerfType::INIT: {
            PerfStatus perfStatus =
                m_upNvPlayfairWrapper->RecordSample(&m_latencies[static_cast<uint8_t>(type)], m_uInitBeginTimeMark,
                                                    uEndTime != 0 ? uEndTime : CNvPlayfairWrapper::GetTimeMark());
            CHK_PERFSTATUS_AND_RETURN(perfStatus, "NvpRecordSample init");
        } break;

        case PerfType::TRANSMISSION: {
            PerfStatus perfStatus = m_upNvPlayfairWrapper->RecordSample(
                &m_latencies[static_cast<uint8_t>(type)], m_uTransmissionBeginTimeMark,
                uEndTime != 0 ? uEndTime : CNvPlayfairWrapper::GetTimeMark());
            CHK_PERFSTATUS_AND_RETURN(perfStatus, "NvpRecordSample transmission");
        } break;

        case PerfType::SUBMISSION: {
            PerfStatus perfStatus = m_upNvPlayfairWrapper->RecordSample(
                &m_latencies[static_cast<uint8_t>(type)], m_uSubmissionBeginTimeMark,
                uEndTime != 0 ? uEndTime : CNvPlayfairWrapper::GetTimeMark());
            CHK_PERFSTATUS_AND_RETURN(perfStatus, "NvpRecordSample submission");
        } break;

        case PerfType::EXECUTION: {
            PerfStatus perfStatus =
                m_upNvPlayfairWrapper->RecordSample(&m_latencies[static_cast<uint8_t>(type)], m_uExecutionBeginTimeMark,
                                                    uEndTime != 0 ? uEndTime : CNvPlayfairWrapper::GetTimeMark());
            CHK_PERFSTATUS_AND_RETURN(perfStatus, "NvpRecordSample execution");
        } break;

        case PerfType::PIPELINE: {
            PerfStatus perfStatus =
                m_upNvPlayfairWrapper->RecordSample(&m_latencies[static_cast<uint8_t>(type)], m_uPipelineBeginTimeMark,
                                                    uEndTime != 0 ? uEndTime : CNvPlayfairWrapper::GetTimeMark());
            CHK_PERFSTATUS_AND_RETURN(perfStatus, "NvpRecordSample pipeline");
        } break;

        default: {
            LOG_ERR("Invalid perf data type %u", static_cast<uint8_t>(type));
            error = NvError_BadParameter;
        } break;
    }
    return error;
}

NvError CProfiler::GetPerf(std::vector<Perf> &vPerf)
{
    vPerf.clear();
    for (PerfType uType = PerfType::INIT; uType < PerfType::TOTAL; uType = static_cast<PerfType>((uint8_t)uType + 1)) {
        if (m_latencies[static_cast<uint8_t>(uType)].uSampleNumber > 0) {
            vPerf.emplace_back(uType, &m_latencies[static_cast<uint8_t>(uType)]);
        }
    }

    return NvError_Success;
}

NvError CProfiler::FenceWaitThreadFunc()
{
    NvError error = NvError_Success;
    pthread_setname_np(pthread_self(), "FenceWaiter");

    while (!m_bQuit) {
        // Wait for buffer to be delivered to thread
        std::unique_lock<std::mutex> queueLock(m_oFenceWaitRequestQueueMutex);
        while (m_oFenceWaitRequestQueue.empty()) {
            m_oFenceWaitRequestQueueCond.wait(queueLock);
            if (m_bQuit) {
                return NvError_Success;
            }
        }

        // Handle the incomming fence wait request
        std::unique_ptr<FenceWaitRequest> upFenceWaitRequest(m_oFenceWaitRequestQueue.front());
        if (upFenceWaitRequest) {
            m_oFenceWaitRequestQueue.pop();
            queueLock.unlock();

            error = WaitForFence(&upFenceWaitRequest->m_fence);
            CHK_ERROR_AND_RETURN(error, "WaitForFence");

            uint64_t uEndTime = CNvPlayfairWrapper::GetTimeMark();
            RecordExecutionTime(upFenceWaitRequest->m_uBeginTime, uEndTime);
            if (upFenceWaitRequest->m_uPipelineBeginTime != 0U) {
                RecordPipelineTime(upFenceWaitRequest->m_uPipelineBeginTime, uEndTime);
            }
        }
    }
    return error;
}

NvError CProfiler::WaitForFence(NvSciSyncFence *pFence)
{
    if (pFence) {
        NvSciError sciErr = NvSciSyncFenceWait(pFence, m_cpuWaitContext, FENCE_FRAME_TIMEOUT_US);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncFenceWait");

        NvSciSyncTaskStatus taskStatus;
        taskStatus.status = NvSciSyncTaskStatus_Invalid;
        sciErr = NvSciSyncFenceGetTaskStatus(pFence, &taskStatus);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncFenceGetTaskStatus");

        LOG_INFO("Task PerfStatus was %d\n", taskStatus.status);

        if (taskStatus.status != NvSciSyncTaskStatus_Success) {
            if (taskStatus.status == NvSciSyncTaskStatus_Invalid) {
                LOG_WARN("TaskStatus was not populated by engine\n");
            } else {
                LOG_ERR("TaskStatus was not a success\n");
                return NvError_InvalidState;
            }
        }

        // uint64_t timestampUS = 0;
        // sciErr = NvSciSyncFenceGetTimestamp(pFence, &timestampUS);
        // PCHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncFenceGetTimestamp failed");

        // LOG_INFO("Timestamp in us was %ld\n", timestampUS);

        NvSciSyncFenceClear(pFence);
        return NvError_Success;
    } else {
        LOG_ERR("Invaid fence");
    }
    return NvError_BadParameter;
}