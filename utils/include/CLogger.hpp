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

#ifndef CLOGGER_HPP
#define CLOGGER_HPP

#include <iostream>
#include <cstdarg>
#include <string>

#define LINE_INFO __FUNCTION__, __LINE__

//! Quick-log a message at debugging level
#define LOG_DBG(...)                                       \
    if (CLogger::GetInstance().GetLogLevel() >= CLogger::LogLevel::LEVEL_DEBUG) \
    CLogger::GetInstance().LogLevelMessage(CLogger::LogLevel::LEVEL_DEBUG, LINE_INFO, __VA_ARGS__)

#define PLOG_DBG(...)                                      \
    if (CLogger::GetInstance().GetLogLevel() >= CLogger::LogLevel::LEVEL_DEBUG) \
    CLogger::GetInstance().PLogLevelMessage(CLogger::LogLevel::LEVEL_DEBUG, LINE_INFO, GetName() + ": ", __VA_ARGS__)

//! Quick-log a message at info level
#define LOG_INFO(...)                                       \
    if (CLogger::GetInstance().GetLogLevel() >= CLogger::LogLevel::LEVEL_INFORMATION) \
    CLogger::GetInstance().LogLevelMessage(CLogger::LogLevel::LEVEL_INFORMATION, LINE_INFO, __VA_ARGS__)

#define PLOG_INFO(...)                                      \
    if (CLogger::GetInstance().GetLogLevel() >= CLogger::LogLevel::LEVEL_INFORMATION) \
    CLogger::GetInstance().PLogLevelMessage(CLogger::LogLevel::LEVEL_INFORMATION, LINE_INFO, GetName() + ": ", __VA_ARGS__)

//! Quick-log a message at warning level
#define LOG_WARN(...)                                       \
    if (CLogger::GetInstance().GetLogLevel() >= CLogger::LogLevel::LEVEL_WARNING) \
    CLogger::GetInstance().LogLevelMessage(CLogger::LogLevel::LEVEL_WARNING, LINE_INFO, __VA_ARGS__)

#define PLOG_WARN(...)                                      \
    if (CLogger::GetInstance().GetLogLevel() >= CLogger::LogLevel::LEVEL_WARNING) \
    CLogger::GetInstance().PLogLevelMessage(CLogger::LogLevel::LEVEL_WARNING, LINE_INFO, GetName() + ": ", __VA_ARGS__)

//! Quick-log a message at error level
#define LOG_ERR(...)                                       \
    if (CLogger::GetInstance().GetLogLevel() >= CLogger::LogLevel::LEVEL_ERROR) \
    CLogger::GetInstance().LogLevelMessage(CLogger::LogLevel::LEVEL_ERROR, LINE_INFO, __VA_ARGS__)

#define PLOG_ERR(...)                                      \
    if (CLogger::GetInstance().GetLogLevel() >= CLogger::LogLevel::LEVEL_ERROR) \
    CLogger::GetInstance().PLogLevelMessage(CLogger::LogLevel::LEVEL_ERROR, LINE_INFO, GetName() + ": ", __VA_ARGS__)

//! Quick-log a message at preset level
#define LOG_MSG(...) CLogger::GetInstance().LogMessage(__VA_ARGS__)

#define LEVEL_NONE CLogger::LogLevel::LEVEL_NO_LOG

#define LEVEL_ERR CLogger::LogLevel::LEVEL_ERROR

#define LEVEL_WARN CLogger::LogLevel::LEVEL_WARNING

#define LEVEL_INFO CLogger::LogLevel::LEVEL_INFORMATION

#define LEVEL_DBG CLogger::LogLevel::LEVEL_DEBUG

//! \brief Logger utility class
//! This is a singleton class - at most one instance can exist at all times.
class CLogger
{
  public:
    //! enum describing the different levels for logging
    enum class LogLevel : uint8_t
    {
        /** no log */
        LEVEL_NO_LOG = 0,
        /** error level */
        LEVEL_ERROR,
        /** warning level */
        LEVEL_WARNING,
        /** info level */
        LEVEL_INFORMATION,
        /** debug level */
        LEVEL_DEBUG
    };

    //! enum describing the different styles for logging
    enum class LogStyle : uint8_t
    {
        NORMAL = 0,
        FUNCTION_LINE = 1
    };

    //! Get the logging instance.
    //! \return Reference to the Logger object.
    static CLogger &GetInstance();

    //! Set the level for logging.
    //! \param[in] eLevel The logging level.
    void SetLogLevel(LogLevel eLevel);

    //! Get the level for logging.
    LogLevel GetLogLevel();

    //! Set the style for logging.
    //! \param[in] eStyle The logging style.
    void SetLogStyle(LogStyle eStyle);

    //! Log a message (cstring).
    //! \param[in] eLevel The logging level,
    //! \param[in] pszunctionName Name of the function as a cstring.
    //! \param[in] sLineNumber Line number,
    //! \param[in] pszFormat Format string as a cstring.
    void
    LogLevelMessage(LogLevel eLevel, const char *pszFunctionName, uint32_t sLineNumber, const char *pszFormat, ...);

    //! Log a message (C++ string).
    //! \param[in] eLevel The logging level,
    //! \param[in] sFunctionName Name of the function as a C++ string.
    //! \param[in] sLineNumber Line number,
    //! \param[in] sFormat Format string as a C++ string.
    void LogLevelMessage(LogLevel eLevel, std::string sFunctionName, uint32_t sLineNumber, std::string sFormat, ...);

    //! Log a message (cstring).
    //! \param[in] eLevel The logging level,
    //! \param[in] pszunctionName Name of the function as a cstring.
    //! \param[in] sLineNumber Line number,
    //! \param[in] prefix Prefix string.
    //! \param[in] pszFormat Format string as a cstring.
    void PLogLevelMessage(LogLevel eLevel,
                          const char *pszFunctionName,
                          uint32_t sLineNumber,
                          std::string prefix,
                          const char *pszFormat,
                          ...);

    //! Log a message (C++ string).
    //! \param[in] eLevel The logging level,
    //! \param[in] sFunctionName Name of the function as a C++ string.
    //! \param[in] sLineNumber Line number,
    //! \param[in] prefix Prefix string.
    //! \param[in] sFormat Format string as a C++ string.
    void PLogLevelMessage(
        LogLevel eLevel, std::string sFunctionName, uint32_t sLineNumber, std::string prefix, std::string sFormat, ...);

    //! Log a message (cstring) at preset level.
    //! \param[in] pszFormat Format string as a cstring.
    void LogMessage(const char *pszFormat, ...);

    //! Log a message (C++ string) at preset level.
    //! \param[in] sFormat Format string as a C++ string.
    void LogMessage(std::string sFormat, ...);

  private:
    //! Need private constructor because this is a singleton.
    CLogger();
    int32_t m_timeZoneOffset = 0;
    std::string m_sExecutableFileName;
    LogLevel m_level = LEVEL_ERR;
    LogStyle m_style = LogStyle::NORMAL;

    const char *GetLevelString(LogLevel level) const;

    void LogLevelMessageVa(LogLevel eLevel,
                           const char *pszFunctionName,
                           uint32_t sLineNumber,
                           const char *prefix,
                           const char *pszFormat,
                           va_list ap);
    void LogMessageVa(const char *pszFormat, va_list ap);
};
// CLogger class

#endif // CLOGGER_HPP
