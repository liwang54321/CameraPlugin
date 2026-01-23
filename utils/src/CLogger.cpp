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
#include <chrono>
#include <fstream>
#include <cstring>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <limits.h>
#include <libgen.h>
#include "CLogger.hpp"

using system_clock = std::chrono::system_clock;

/*
 * Seconds per hour
 */
static constexpr uint32_t EPOC_YEAR = 1900U;

/*
 * Seconds per hour
 */
static constexpr uint32_t SECONDS_PER_HOUR = 3600U;

/*
 * Log buffer size
 */
static constexpr uint32_t LOG_BUF_SIZE = 1024U;

/*
 * Default executable file name
 */
static constexpr const char *DEFAULT_EXECUTABLE_NAME = "nvsipl_multicast";

/*
 * Log header format: Year-Month-Day Hour:Minute:Second Process ID-Thread ID/Eexecutable FileName level prefix
 */
static constexpr const char *LOG_BUF_HEADER = "%04d-%02d-%02d %02d:%02d:%09.6f %d-%02d/%s %-7s %s";

// Log utils
CLogger &CLogger::GetInstance()
{
    static CLogger instance;
    return instance;
}

void CLogger::SetLogLevel(LogLevel level)
{
    m_level = (level > LEVEL_DBG) ? LEVEL_DBG : level;
}

CLogger::LogLevel CLogger::GetLogLevel()
{
    return m_level;
}

void CLogger::SetLogStyle(LogStyle style)
{
    m_style = (style > LogStyle::FUNCTION_LINE) ? LogStyle::FUNCTION_LINE : style;
}

CLogger::CLogger()
{
    system_clock::time_point tp = system_clock::now();
    std::time_t time = system_clock::to_time_t(tp);
    std::tm localtime;
    m_timeZoneOffset = (timegm(localtime_r(&time, &localtime)) - time) / SECONDS_PER_HOUR;

#ifdef NVMEDIA_QNX
    std::ifstream file("/proc/self/exefile");
    std::string sPath;
    std::getline(file, sPath);
    size_t ret = sPath.length();
    char result[PATH_MAX];
    strncpy(result, sPath.c_str(), PATH_MAX);
#else
    char result[PATH_MAX] = { 0 };
    ssize_t ret = readlink("/proc/self/exe", result, PATH_MAX);
#endif
    m_sExecutableFileName = ret > 0 ? basename(result) : DEFAULT_EXECUTABLE_NAME;
}

const char *CLogger::GetLevelString(LogLevel level) const
{
    static const std::unordered_map<LogLevel, const char *> logLevel2StringMap = { { LEVEL_NONE, "NONE" },
                                                                                   { LEVEL_ERR, "ERROR" },
                                                                                   { LEVEL_WARN, "WARNING" },
                                                                                   { LEVEL_INFO, "INFO" },
                                                                                   { LEVEL_DBG, "DEBUG" } };
    auto result = logLevel2StringMap.find(level);
    return result != logLevel2StringMap.end() ? result->second : "UNKNOWN";
}

void CLogger::LogLevelMessageVa(
    LogLevel level, const char *functionName, uint32_t lineNumber, const char *prefix, const char *format, va_list ap)
{
    if (level > m_level) {
        return;
    }

    char str[LOG_BUF_SIZE] = {
        '\0',
    };

    /*
     * Get current system time
     */
    system_clock::time_point tp = system_clock::now();
    std::time_t time = system_clock::to_time_t(tp);
    std::tm gmt{};
    gmtime_r(&time, &gmt);

    std::chrono::duration<double> fractionalSeconds =
        (tp - system_clock::from_time_t(time)) + std::chrono::seconds(gmt.tm_sec);

    /*
     * Log header
     */
    snprintf(str, sizeof(str), LOG_BUF_HEADER, gmt.tm_year + EPOC_YEAR, gmt.tm_mon + 1, gmt.tm_mday,
             gmt.tm_hour + m_timeZoneOffset, gmt.tm_min, fractionalSeconds.count(), getpid(), gettid(),
             m_sExecutableFileName.c_str(), GetLevelString(level), prefix);

    size_t length = strlen(str);
    /*
     * Log body
     */
    vsnprintf(str + length, sizeof(str) - length, format, ap);

    /*
     * Append `\n`
     */
    length = strlen(str);
    if (m_style == LogStyle::NORMAL) {
        if (length != 0 && str[length - 1] != '\n') {
            if (length < sizeof(str) - 1) {
                strcat(str, "\n");
            } else {
                str[strlen(str) - 1] = '\n';
            }
        }
    } else if (m_style == LogStyle::FUNCTION_LINE) {
        if (length != 0 && str[length - 1] == '\n') {
            str[length - 1] = 0;
        }
        snprintf(str + length, sizeof(str) - length, " at %s():%d\n", functionName, lineNumber);
    }

    std::cout << str;
}

void CLogger::LogLevelMessage(LogLevel level, const char *functionName, uint32_t lineNumber, const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    LogLevelMessageVa(level, functionName, lineNumber, "", format, ap);
    va_end(ap);
}

void CLogger::LogLevelMessage(LogLevel level, std::string functionName, uint32_t lineNumber, std::string format, ...)
{
    va_list ap;
    va_start(ap, format);
    LogLevelMessageVa(level, functionName.c_str(), lineNumber, "", format.c_str(), ap);
    va_end(ap);
}

void CLogger::PLogLevelMessage(
    LogLevel level, const char *functionName, uint32_t lineNumber, std::string prefix, const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    LogLevelMessageVa(level, functionName, lineNumber, prefix.c_str(), format, ap);
    va_end(ap);
}

void CLogger::PLogLevelMessage(
    LogLevel level, std::string functionName, uint32_t lineNumber, std::string prefix, std::string format, ...)
{
    va_list ap;
    va_start(ap, format);
    LogLevelMessageVa(level, functionName.c_str(), lineNumber, prefix.c_str(), format.c_str(), ap);
    va_end(ap);
}

void CLogger::LogMessageVa(const char *format, va_list ap)
{
    char str[256] = {
        '\0',
    };
    vsnprintf(str, sizeof(str), format, ap);
    std::cout << str;
}

void CLogger::LogMessage(const char *format, ...)
{
    va_list ap;
    va_start(ap, format);
    LogMessageVa(format, ap);
    va_end(ap);
}

void CLogger::LogMessage(std::string format, ...)
{
    va_list ap;
    va_start(ap, format);
    LogMessageVa(format.c_str(), ap);
    va_end(ap);
}
