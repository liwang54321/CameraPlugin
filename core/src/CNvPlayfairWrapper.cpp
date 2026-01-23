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

#include "CNvPlayfairWrapper.hpp"
#include "CUtils.hpp"

/*
 * The library name of nvplayfair
 */
static constexpr const char *NVPLAYFAIR_LIBRARY_NAME = "libnvplayfair.so";
uint32_t CNvPlayfairWrapper::m_uCounter{ 0 };
std::mutex CNvPlayfairWrapper::m_oMutex{};
void *CNvPlayfairWrapper::m_pHandle = nullptr;
ConstructPerfDataFunc CNvPlayfairWrapper::m_pConstructPerfData = nullptr;
DestroyPerfDataFunc CNvPlayfairWrapper::m_pDestroyPerfData = nullptr;
RecordSampleFunc CNvPlayfairWrapper::m_pRecordSample = nullptr;
DumpDataFunc CNvPlayfairWrapper::m_pDumpData = nullptr;
CalcPercentilesFunc CNvPlayfairWrapper::m_pCalcPercentiles = nullptr;
CalcStatsFunc CNvPlayfairWrapper::m_pCalcStats = nullptr;
PrintStatsExtFunc CNvPlayfairWrapper::m_pPrintStatsExt = nullptr;
PrintStatsFunc CNvPlayfairWrapper::m_pPrintStats = nullptr;
RateLimitInitFunc CNvPlayfairWrapper::m_pRateLimitInit = nullptr;
MarkPeriodicExecStartFunc CNvPlayfairWrapper::m_pMarkPeriodicExecStart = nullptr;
RateLimitWaitFunc CNvPlayfairWrapper::m_pRateLimitWait = nullptr;
AggregatePerfDataFunc CNvPlayfairWrapper::m_pAggregatePerfData = nullptr;

CNvPlayfairWrapper::CNvPlayfairWrapper() {}

CNvPlayfairWrapper::~CNvPlayfairWrapper()
{
    DeInit();
}

NvError CNvPlayfairWrapper::Init()
{
    if (!m_bInited) {
        std::lock_guard<std::mutex> lk(m_oMutex);
        /*
         * In order to remove the dependency of libnvplayfair.so, we must load
         * libnvplayfair.so with dlopen and resolve the needed symbols with
         * dlsym explicitly.
         */
        if (0 == m_uCounter++) {
            m_pHandle = dlopen(NVPLAYFAIR_LIBRARY_NAME, RTLD_LOCAL | RTLD_LAZY);
            if (!m_pHandle) {
                LOG_ERR("Failed to dlopen %s", NVPLAYFAIR_LIBRARY_NAME);
                return NvError_LibraryNotFound;
            }

#define LOAD_SYMBOL(SYMBOL)                                               \
    m_p##SYMBOL = (SYMBOL##Func)dlsym(m_pHandle, "Nvp" #SYMBOL);          \
    if (!m_p##SYMBOL) {                                                   \
        LOG_ERR("Failed to load %s due to %s", "Nvp" #SYMBOL, dlerror()); \
        Unload();                                                         \
        return NvError_SymbolNotFound;                                    \
    }
            if (m_pHandle) {
                LOAD_SYMBOL(ConstructPerfData)
                LOAD_SYMBOL(DestroyPerfData)
                LOAD_SYMBOL(RecordSample)
                LOAD_SYMBOL(DumpData)
                LOAD_SYMBOL(CalcPercentiles)
                LOAD_SYMBOL(CalcStats)
                LOAD_SYMBOL(PrintStatsExt)
                LOAD_SYMBOL(PrintStats)
                LOAD_SYMBOL(RateLimitInit)
                LOAD_SYMBOL(MarkPeriodicExecStart)
                LOAD_SYMBOL(RateLimitWait)
                LOAD_SYMBOL(AggregatePerfData)
            }
        }
        m_bInited = true;
    }
    return NvError_Success;
}

void CNvPlayfairWrapper::DeInit()
{
    if (m_bInited) {
        m_bInited = false;
        std::lock_guard<std::mutex> lk(m_oMutex);
        if (--m_uCounter == 0) {
            Unload();
        }
    }
}

PerfStatus CNvPlayfairWrapper::ConstructPerfData(PerfData *pPerfData, uint32_t uNumOfSamples, char const *pFilename)
{
    return m_pConstructPerfData ? m_pConstructPerfData(pPerfData, uNumOfSamples, pFilename) : PerfStatus::FAIL_NOINIT;
}

PerfStatus CNvPlayfairWrapper::DestroyPerfData(PerfData *pPerfData)
{
    return m_pDestroyPerfData ? m_pDestroyPerfData(pPerfData) : PerfStatus::FAIL_NOINIT;
}

PerfStatus CNvPlayfairWrapper::RecordSample(PerfData *pPerfData, uint64_t uSampleStartTime, uint64_t uSampleEndTime)
{
    return m_pRecordSample ? m_pRecordSample(pPerfData, uSampleStartTime, uSampleEndTime) : PerfStatus::FAIL_NOINIT;
}

PerfStatus CNvPlayfairWrapper::DumpData(PerfData *pPerfData)
{
    return m_pDumpData ? m_pDumpData(pPerfData) : PerfStatus::FAIL_NOINIT;
}

PerfStatus CNvPlayfairWrapper::CalcPercentiles(PerfData *pPerfData,
                                               uint32_t uNumOfPercentilePoints,
                                               double *pPercentilePointsArray,
                                               double **pPercentileValues,
                                               TimeUnit unit)
{
    return m_pCalcPercentiles
               ? m_pCalcPercentiles(pPerfData, uNumOfPercentilePoints, pPercentilePointsArray, pPercentileValues, unit)
               : PerfStatus::FAIL_NOINIT;
}

PerfStatus CNvPlayfairWrapper::CalcStats(PerfData *pPerfData, PerfStats *pStats, TimeUnit unit)
{
    return m_pCalcStats ? m_pCalcStats(pPerfData, pStats, unit) : PerfStatus::FAIL_NOINIT;
}

PerfStatus CNvPlayfairWrapper::PrintStatsExt(PerfData *pPerfData,
                                             PerfStats *pStats,
                                             TimeUnit unit,
                                             char const *pMsg,
                                             bool bCsv,
                                             LogBackend logBackend,
                                             char const *pReportFilename)
{
    return m_pPrintStatsExt ? m_pPrintStatsExt(pPerfData, pStats, unit, pMsg, bCsv, logBackend, pReportFilename)
                            : PerfStatus::FAIL_NOINIT;
}

PerfStatus
CNvPlayfairWrapper::PrintStats(PerfData *pPerfData, PerfStats *pStats, TimeUnit unit, char const *pMsg, bool bCsv)
{
    return m_pPrintStats ? m_pPrintStats(pPerfData, pStats, unit, pMsg, bCsv) : PerfStatus::FAIL_NOINIT;
}

PerfStatus CNvPlayfairWrapper::RateLimitInit(RateLimitInfo *pRtInfo, uint32_t uFps)
{
    return m_pRateLimitInit ? m_pRateLimitInit(pRtInfo, uFps) : PerfStatus::FAIL_NOINIT;
}

PerfStatus CNvPlayfairWrapper::MarkPeriodicExecStart(RateLimitInfo *pRtInfo)
{
    return m_pMarkPeriodicExecStart ? m_pMarkPeriodicExecStart(pRtInfo) : PerfStatus::FAIL_NOINIT;
}

PerfStatus CNvPlayfairWrapper::RateLimitWait(RateLimitInfo *pRtInfo)
{
    return m_pRateLimitWait ? m_pRateLimitWait(pRtInfo) : PerfStatus::FAIL_NOINIT;
}

PerfStatus CNvPlayfairWrapper::AggregatePerfData(PerfData *pNetPerfData,
                                                 PerfData **pInputPerfDataArray,
                                                 uint32_t uNumOfPerfDataObjs)
{
    return m_pAggregatePerfData ? m_pAggregatePerfData(pNetPerfData, pInputPerfDataArray, uNumOfPerfDataObjs)
                                : PerfStatus::FAIL_NOINIT;
}

void CNvPlayfairWrapper::Unload()
{
    m_pConstructPerfData = nullptr;
    m_pDestroyPerfData = nullptr;
    m_pRecordSample = nullptr;
    m_pDumpData = nullptr;
    m_pCalcPercentiles = nullptr;
    m_pCalcStats = nullptr;
    m_pPrintStatsExt = nullptr;
    m_pPrintStats = nullptr;
    m_pRateLimitInit = nullptr;
    m_pMarkPeriodicExecStart = nullptr;
    m_pRateLimitWait = nullptr;
    m_pAggregatePerfData = nullptr;

    dlclose(m_pHandle);
    m_pHandle = nullptr;
}