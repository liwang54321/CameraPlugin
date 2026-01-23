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

#ifndef CNVPLAYFAIRWRAPPER_HPP
#define CNVPLAYFAIRWRAPPER_HPP

#include <stdlib.h>
#include <stdint.h>
#include <dlfcn.h>
#include <mutex>
#include "nverror.h"

/*
 * In order to remove the dependency of nvplayfair header files, we have to declare
 * the similiar enumerations, structures and function prototypes.
 */
enum class TimeUnit : uint8_t
{
    SECONDS,
    MILLISECONDS,
    MICROSECONDS,
    NANOSECONDS
};

enum class LogBackend : uint8_t
{
    CONSOLE,
    NVOS
};

enum class PerfStatus : uint8_t
{
    PASS,
    FAIL_ALLOC,
    FAIL_NOINIT,
    FAIL_FILEOP,
    FAIL_NULLPTR,
    FAIL_NO_SAMPLES,
    FAIL_VERSION_MISMATCH,
    FAIL_INVALID_TIME_UNIT,
    FAIL_INVALID_LOG_BACKEND,
    FAIL_SAMPLE_COUNT_MISMATCH
};

struct PerfStats
{
    double dMin;
    double dMax;
    double dMean;
    double dPct99;
    double dStdev;
    uint32_t uCount;
};

struct RateLimitInfo
{
    uint32_t uPeriodUs;
    uint32_t uPeriodNumber;
    uint64_t uPeriodicExecStartTimeUs;
};

struct PerfData
{
    uint64_t uSampleNumber;
    uint64_t *pTimestamps;
    uint64_t *pLatencies;
    uint32_t uMaxSamples;
    bool bInitialized;
    char *pFilename;
};

typedef PerfStatus (*ConstructPerfDataFunc)(PerfData *pPerfData, uint32_t uNumOfSamples, char const *pFilename);
typedef PerfStatus (*DestroyPerfDataFunc)(PerfData *pPerfData);
typedef PerfStatus (*RecordSampleFunc)(PerfData *pPerfData, uint64_t uSampleStartTime, uint64_t uSampleEndTime);
typedef PerfStatus (*DumpDataFunc)(PerfData *pPerfData);
typedef PerfStatus (*CalcPercentilesFunc)(PerfData *pPerfData,
                                          uint32_t uPumOfPercentilePoints,
                                          double *pPercentilePointsArray,
                                          double **pPercentileValues,
                                          TimeUnit unit);
typedef PerfStatus (*CalcStatsFunc)(PerfData *pPerfData, PerfStats *pStats, TimeUnit unit);
typedef PerfStatus (*PrintStatsExtFunc)(PerfData *pPerfData,
                                        PerfStats *pStats,
                                        TimeUnit unit,
                                        char const *pMsg,
                                        bool bCsv,
                                        LogBackend logBackend,
                                        char const *pReportFilename);
typedef PerfStatus (*PrintStatsFunc)(
    PerfData *pPerfData, PerfStats *pStats, TimeUnit unit, char const *pMsg, bool bCsv);
typedef PerfStatus (*RateLimitInitFunc)(RateLimitInfo *pRtInfo, uint32_t uFps);
typedef PerfStatus (*MarkPeriodicExecStartFunc)(RateLimitInfo *pRtInfo);
typedef PerfStatus (*RateLimitWaitFunc)(RateLimitInfo *pRtInfo);
typedef PerfStatus (*AggregatePerfDataFunc)(PerfData *pNetPerfData,
                                            PerfData **pInputPerfDataArray,
                                            uint32_t uNumOfPerfDataObjs);

class CNvPlayfairWrapper
{
  public:
    static uint64_t GetTimeMark()
    {
        uint64_t uMark;

#ifdef NVPLAYFAIR_ARCH_IS_X86
        int ret;
        struct timespec tp;

        ret = clock_gettime(CLOCK_MONOTONIC, &tp);
        if (ret != 0) {
            fprintf(stderr, "%s, %s:%d, NvPlayfair Error clock_gettime %d\n", __FILE__, __func__, __LINE__, ret);
            exit(EXIT_FAILURE);
        }

        uMark = tp.tv_sec * 1000000000UL + tp.tv_nsec;

#else /* NVPLAYFAIR_ARCH_IS_X86 */
        __asm__ __volatile__("ISB;                                                     \
                              mrs %[result], cntvct_el0;                               \
                              ISB"
                             : [result] "=r"(uMark)
                             :
                             : "memory");

#endif /* NVPLAYFAIR_ARCH_IS_X86 */

        return uMark;
    }

    CNvPlayfairWrapper();
    ~CNvPlayfairWrapper();

    NvError Init();
    void DeInit();

    PerfStatus ConstructPerfData(PerfData *pPerfData, uint32_t uNumOfSamples, char const *pFilename);
    PerfStatus DestroyPerfData(PerfData *pPerfData);
    PerfStatus RecordSample(PerfData *pPerfData, uint64_t uSampleStartTime, uint64_t uSampleEndTime);
    PerfStatus DumpData(PerfData *pPerfData);
    PerfStatus CalcPercentiles(PerfData *pPerfData,
                               uint32_t uNumOfPercentilePoints,
                               double *pPercentilePointsArray,
                               double **pPercentileValues,
                               TimeUnit unit);
    PerfStatus CalcStats(PerfData *pPerfData, PerfStats *pStats, TimeUnit unit);
    PerfStatus PrintStatsExt(PerfData *pPerfData,
                             PerfStats *pStats,
                             TimeUnit unit,
                             char const *pMsg,
                             bool bCsv,
                             LogBackend logBackend,
                             char const *pReportFilename);
    PerfStatus PrintStats(PerfData *pPerfData, PerfStats *pStats, TimeUnit unit, char const *pMsg, bool bCsv);
    PerfStatus RateLimitInit(RateLimitInfo *pRtInfo, uint32_t uFps);
    PerfStatus MarkPeriodicExecStart(RateLimitInfo *pRtInfo);
    PerfStatus RateLimitWait(RateLimitInfo *pRtInfo);
    PerfStatus AggregatePerfData(PerfData *pNetPerfData, PerfData **pInputPerfDataArray, uint32_t uNumOfPerfDataObjs);

  private:
    void Unload();
    CNvPlayfairWrapper(const CNvPlayfairWrapper &) = delete;
    CNvPlayfairWrapper(CNvPlayfairWrapper &&) = delete;
    CNvPlayfairWrapper &operator=(const CNvPlayfairWrapper &) = delete;
    CNvPlayfairWrapper &operator=(CNvPlayfairWrapper &&) = delete;

  private:
    static uint32_t m_uCounter;
    static std::mutex m_oMutex;
    static void *m_pHandle;
    static ConstructPerfDataFunc m_pConstructPerfData;
    static DestroyPerfDataFunc m_pDestroyPerfData;
    static RecordSampleFunc m_pRecordSample;
    static DumpDataFunc m_pDumpData;
    static CalcPercentilesFunc m_pCalcPercentiles;
    static CalcStatsFunc m_pCalcStats;
    static PrintStatsExtFunc m_pPrintStatsExt;
    static PrintStatsFunc m_pPrintStats;
    static RateLimitInitFunc m_pRateLimitInit;
    static MarkPeriodicExecStartFunc m_pMarkPeriodicExecStart;
    static RateLimitWaitFunc m_pRateLimitWait;
    static AggregatePerfDataFunc m_pAggregatePerfData;

  private:
    bool m_bInited = false;
};
#endif // CNVPLAYFAIRWRAPPER_HPP