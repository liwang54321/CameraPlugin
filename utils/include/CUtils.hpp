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

#ifndef CUTILS_HPP
#define CUTILS_HPP

#include <algorithm>
#include <iostream>
#include <memory>
#include <cstdarg>
#include <vector>
#include <condition_variable>
#include <mutex>
#include <atomic>
#include <regex>
#include <unordered_map>
#include <WF/wfd.h>
#ifdef BUILD_VULKANSC
#ifdef VULKAN
#include <vulkan/vulkan.h>
#else
#include <vulkan/vulkan_sc.h>
#endif
#endif

#include "nvmedia_core.h"
#include "nvscierror.h"
#include "Common.hpp"
#include "NvSIPLCommon.hpp"
#include "NvSIPLCamera.hpp"
#include "nvscibuf.h"
#include "nvscisync.h"
#include "CNvPlayfairWrapper.hpp"
#include "nverror.h"
#include "CLogger.hpp"

using namespace nvsipl;

NvError toNvError(PerfStatus status);
NvError toNvError(SIPLStatus status);
NvError toNvError(NvMediaStatus status);
NvError toNvError(WFDErrorCode code);
#ifdef BUILD_VULKANSC
NvError toNvError(VkResult result);
#endif

struct CloseNvSciBufAttrList
{
    void operator()(NvSciBufAttrList *attrList) const
    {
        if (attrList != nullptr) {
            if ((*attrList) != nullptr) {
                NvSciBufAttrListFree(*attrList);
            }
            delete attrList;
        }
    }
};

struct CloseNvSciBufObj
{
    void operator()(NvSciBufObj *bufObj) const
    {
        if (bufObj != nullptr) {
            if ((*bufObj) != nullptr) {
                NvSciBufObjFree(*bufObj);
            }
            delete bufObj;
        }
    }
};

struct CloseNvSciSyncAttrList
{
    void operator()(NvSciSyncAttrList *attrList) const
    {
        if (attrList != nullptr) {
            if ((*attrList) != nullptr) {
                NvSciSyncAttrListFree(*attrList);
            }
            delete attrList;
        }
    }
};

struct CloseNvSciSyncObj
{
    void operator()(NvSciSyncObj *syncObj) const
    {
        if (syncObj != nullptr) {
            if ((*syncObj) != nullptr) {
                NvSciSyncObjFree(*syncObj);
            }
            delete syncObj;
        }
    }
};

struct CloseFile
{
    void operator()(FILE *pFile) const
    {
        if (pFile) {
            fclose(pFile);
            pFile = nullptr;
        }
    }
};

/** Helper MACROS */
#define CHK_PTR_AND_RETURN(ptr, api)       \
    if ((ptr) == nullptr) {                \
        LOG_ERR("%s failed\n", (api));     \
        return NvError_InsufficientMemory; \
    }

#define CHK_PTR_AND_RETURN_BADARG(ptr, name) \
    if ((ptr) == nullptr) {                  \
        LOG_ERR("%s is null\n", (name));     \
        return NvError_BadParameter;         \
    }

#define CHK_ERROR_AND_RETURN(error, api)                   \
    if ((error) != NvError_Success) {                      \
        LOG_ERR("%s failed, error: %u\n", (api), (error)); \
        return (error);                                    \
    }

#define CHK_ERROR_AND_RETURN(error, api)                   \
    if ((error) != NvError_Success) {                      \
        LOG_ERR("%s failed, error: %u\n", (api), (error)); \
        return (error);                                    \
    }

#define CHK_NVMSTATUS_AND_RETURN(nvmStatus, api)               \
    if ((nvmStatus) != NVMEDIA_STATUS_OK) {                    \
        LOG_ERR("%s failed, error: %u\n", (api), (nvmStatus)); \
        return toNvError(nvmStatus);                           \
    }

#define CHK_NVSCISTATUS_AND_RETURN(nvSciStatus, api)             \
    if ((nvSciStatus) != NvSciError_Success) {                   \
        LOG_ERR("%s failed, error: %u\n", (api), (nvSciStatus)); \
        return NvError_ResourceError;                            \
    }

#define CHK_PERFSTATUS_AND_RETURN(perfStatus, api)              \
    if ((perfStatus) != PerfStatus::PASS) {                     \
        LOG_ERR("%s failed, error: %u\n", (api), (perfStatus)); \
        return toNvError(perfStatus);                           \
    }

/* prefix help MACROS */
#define PCHK_PTR_AND_RETURN(ptr, api)      \
    if ((ptr) == nullptr) {                \
        PLOG_ERR("%s failed\n", (api));    \
        return NvError_InsufficientMemory; \
    }

#define PCHK_PTR_AND_RETURN_ERR(ptr, api)  \
    if ((ptr) == nullptr) {                \
        PLOG_ERR("%s is null\n", (api));   \
        return NvError_InsufficientMemory; \
    }

#define PCHK_ERROR_AND_RETURN(error, api)                   \
    if ((error) != NvError_Success) {                       \
        PLOG_ERR("%s failed, error: %u\n", (api), (error)); \
        return (error);                                     \
    }

#define PCHK_NVMSTATUS_AND_RETURN(nvmStatus, api)               \
    if ((nvmStatus) != NVMEDIA_STATUS_OK) {                     \
        PLOG_ERR("%s failed, error: %u\n", (api), (nvmStatus)); \
        return toNvError(nvmStatus);                            \
    }

#define PCHK_NVSCISTATUS_AND_RETURN(nvSciStatus, api)                                  \
    if ((nvSciStatus) != NvSciError_Success) {                                         \
        PLOG_ERR("%s failed, error: %u(0x%x)\n", (api), (nvSciStatus), (nvSciStatus)); \
        return NvError_ResourceError;                                                  \
    }

#define PCHK_CUDASTATUS_AND_RETURN(cudaStatus, api)                                                  \
    if ((cudaStatus) != cudaSuccess) {                                                               \
        PLOG_ERR("%s failed, error: %u(%s)\n", (api), (cudaStatus), (cudaGetErrorName(cudaStatus))); \
        return NvError_ResourceError;                                                                \
    }

#define CHK_CUDASTATUS_AND_RETURN(cudaStatus, api)                                                  \
    if ((cudaStatus) != cudaSuccess) {                                                              \
        LOG_ERR("%s failed, error: %u(%s)\n", (api), (cudaStatus), (cudaGetErrorName(cudaStatus))); \
        return NvError_ResourceError;                                                               \
    }

#define CHK_CUDAERR_AND_RETURN(e, api)                                           \
    {                                                                            \
        auto ret = (e);                                                          \
        if (ret != CUDA_SUCCESS) {                                               \
            const auto flags = std::cout.flags();                                \
            std::cout << api << " CUDA error: " << std::hex << ret << std::endl; \
            std::cout.flags(flags);                                              \
            return NvError_ResourceError;                                        \
        }                                                                        \
    }

#define PCHK_NVSCICONNECT_AND_RETURN(nvSciStatus, event, api)                                      \
    if (NvSciError_Success != nvSciStatus) {                                                       \
        std::cout << GetName() << ": " << api << " connect failed. " << nvSciStatus << std::endl;  \
        return NvError_ResourceError;                                                              \
    }                                                                                              \
    if (event != NvSciStreamEventType_Connected) {                                                 \
        std::cout << GetName() << ": " << api << " didn't receive connected event. " << std::endl; \
        return NvError_ResourceError;                                                              \
    }

#define CHECK_WFD_ERROR(device)                                                                 \
    {                                                                                           \
        WFDErrorCode err = wfdGetError(device);                                                 \
        if (err) {                                                                              \
            std::cerr << "WFD Error 0x" << std::hex << err << " at: " << std::dec << std::endl; \
            std::cerr << __FILE__ << ":" << __LINE__ << std::endl;                              \
        };                                                                                      \
    }

#define PCHK_WFDSTATUS_AND_RETURN(wfdStatus, api)              \
    if (wfdStatus) {                                           \
        LOG_ERR("%s failed, error: %u\n", (api), (wfdStatus)); \
        return toNvError(wfdStatus);                           \
    }

#define PGET_WFDERROR_AND_RETURN(device)                           \
    {                                                              \
        WFDErrorCode wfdErr = wfdGetError(device);                 \
        if (wfdErr) {                                              \
            LOG_ERR("WFD error %x, line: %u\n", wfdErr, __LINE__); \
            return toNvError(wfdErr);                              \
        }                                                          \
    }

#define CHK_VKSCSTATUS_AND_RETURN(vkscStatus, api)                   \
    {                                                                \
        if (vkscStatus != VK_SUCCESS) {                              \
            LOG_ERR("%s failed, status: %d\n", (api), (vkscStatus)); \
            return toNvError(vkscStatus);                            \
        }                                                            \
    }

#define PCHK_VKSCSTATUS_AND_RETURN(vkscStatus, api)                   \
    {                                                                 \
        if (vkscStatus != VK_SUCCESS) {                               \
            PLOG_ERR("%s failed, status: %d\n", (api), (vkscStatus)); \
            return toNvError(vkscStatus);                             \
        }                                                             \
    }

NvError LoadNITOFile(const std::string &folderPath, const std::string &moduleName, std::vector<uint8_t> &nito);

typedef std::vector<std::pair<INvSIPLClient::ConsumerDesc::OutputType, INvSIPLClient::INvSIPLBuffer *>> NvSIPLBuffers;

const char *NvSciBufAttrKeyToString(NvSciBufAttrKey key);

#define MAX_NUM_SURFACES (3U)

#define ARRAY_SIZE(x) sizeof(x) / sizeof(x[0])

typedef struct
{
    NvSciBufType bufType = NvSciBufType_General;
    uint64_t size = {};
    uint32_t planeCount = {};
    NvSciBufAttrValImageLayoutType layout = NvSciBufImage_BlockLinearType;
    uint32_t planeWidths[MAX_NUM_SURFACES] = {};
    uint32_t planeHeights[MAX_NUM_SURFACES] = {};
    uint32_t planePitches[MAX_NUM_SURFACES] = {};
    uint32_t planeBitsPerPixels[MAX_NUM_SURFACES] = {};
    uint32_t planeAlignedHeights[MAX_NUM_SURFACES] = {};
    uint64_t planeAlignedSizes[MAX_NUM_SURFACES] = {};
    uint8_t planeChannelCounts[MAX_NUM_SURFACES] = {};
    uint64_t planeOffsets[MAX_NUM_SURFACES] = {};
    uint64_t topPadding[MAX_NUM_SURFACES] = {};
    uint64_t bottomPadding[MAX_NUM_SURFACES] = {};
    bool needSwCacheCoherency = false;
    NvSciBufAttrValColorFmt planeColorFormats[MAX_NUM_SURFACES] = {};
} BufferAttrs;

NvError SetBufAttr(NvSciBufAttrList *pBufAttrList,
                   const std::string &sColorType,
                   const std::string &sImageLayout,
                   uint32_t &uWidth,
                   uint32_t &uHeight);
NvError PopulateBufAttr(const NvSciBufObj &sciBufObj, BufferAttrs &bufAttrs);
NvError GetWidthAndHeight(const NvSciBufAttrList bufAttrList,
                          uint16_t &uWidth,
                          uint16_t &uHeight,
                          uint32_t *pPlanePitches = nullptr,
                          uint32_t size = 0);
std::vector<std::string> splitString(const std::string &inputString, char delimiter);
NvError DumpBufAttr(const NvSciBufObj &sciBufObj);
const char *EventStatusToString(EventStatus event);

NvError GetDTPropAsString(const void *node, const char *const name, char val[], const uint32_t size);
NvError CheckSKU(const std::string &findStr, bool &bFound);

class CSemaphore
{
  public:
    CSemaphore() {}
    void Wait()
    {
        std::unique_lock<std::mutex> lock{ m_mutex };
        m_conditionVar.wait(lock, [&]() -> bool { return m_count > 0; });
        --m_count;
    }
    void Signal()
    {
        std::lock_guard<std::mutex> lock{ m_mutex };
        if (++m_count == 1) {
            m_conditionVar.notify_one();
        }
    }

  private:
    std::atomic<int> m_count{ 0 };
    std::mutex m_mutex;
    std::condition_variable m_conditionVar;
};

template <class T> class IEventListener
{
  public:
    virtual void OnEvent(T *object, EventStatus event) = 0;
    virtual void OnError(T *object, int moduleId, uint32_t errorId) = 0;
};

NvError GetToken(const char **pBuf, const char *pTerm, std::string &token);
std::string TrimBlank(std::string str);
std::string ToLower(std::string str);

std::string IntToStringWithLeadingZero(int num);

void PrintDeviceGid(const char *pDeviceStr, const char *pTypeStr, const uint8_t *pId);
bool GetRoi(const std::string &sLine, std::vector<NvMediaRect> &rois);
void recordTimestampInCarveout(const std::string &carveoutMsg, const std::string &logFileName = "");

template <typename StringContainer,
          typename std::enable_if_t<std::is_same<typename StringContainer::value_type, std::string>::value> * = nullptr>
inline bool ContainAnySubStr(const StringContainer &container, const std::string &str)
{
    return std::find_if(std::begin(container), std::end(container), [&str](const std::string &s) {
               return std::string::npos != str.rfind(s);
           }) != std::end(container);
}

template <typename StringContainer,
          typename std::enable_if_t<std::is_same<typename StringContainer::value_type, std::string>::value> * = nullptr>
inline bool ContainAllSubStr(const StringContainer &container, const std::string &str)
{
    return std::all_of(std::begin(container), std::end(container),
                       [&str](const std::string &s) { return std::string::npos != str.find(s); });
}

long Random() noexcept;

inline std::string ReplaceName(const std::string& name, const std::unordered_map<std::string, std::string>& nameMap) {
    std::regex pattern(R"((\D+)(\d*))");
    std::smatch match;

    if (std::regex_search(name, match, pattern) && match.size() > 1) {
        auto it = nameMap.find(match[1].str());
        if (it != nameMap.end()) {
            return it->second + match[2].str();
        }
    }
    return name;
};
#endif