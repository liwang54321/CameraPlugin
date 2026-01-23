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

#include <sys/stat.h>
#include <fcntl.h>
#include <cstring>
#include <sstream>
#include <fstream>
#include <sys/types.h>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <unistd.h>
#include <unordered_map>
#include <inttypes.h>
#include "CClientCommon.hpp"
#include "CUtils.hpp"
#if NVMEDIA_QNX
#include "nvdtcommon.h"
#include <devctl.h>
#endif

NvError toNvError(PerfStatus error)
{
    static const std::unordered_map<PerfStatus, NvError> perfStatus2NvErrorMap = {
        { PerfStatus::PASS, NvError_Success },
        { PerfStatus::FAIL_ALLOC, NvError_InsufficientMemory },
        { PerfStatus::FAIL_NOINIT, NvError_NotInitialized },
        { PerfStatus::FAIL_FILEOP, NvError_BadParameter },
        { PerfStatus::FAIL_NULLPTR, NvError_BadValue },
        { PerfStatus::FAIL_NO_SAMPLES, NvError_InvalidState },
        { PerfStatus::FAIL_VERSION_MISMATCH, NvError_InvalidState },
        { PerfStatus::FAIL_INVALID_TIME_UNIT, NvError_NotSupported },
        { PerfStatus::FAIL_INVALID_LOG_BACKEND, NvError_NotSupported },
        { PerfStatus::FAIL_SAMPLE_COUNT_MISMATCH, NvError_CountMismatch }
    };

    auto it = perfStatus2NvErrorMap.find(error);
    return it != perfStatus2NvErrorMap.end() ? it->second : NvError_BadValue;
}

NvError toNvError(SIPLStatus status)
{
    static const std::unordered_map<SIPLStatus, NvError> siplStatus2NvErrorMap = {
        { NVSIPL_STATUS_OK, NvError_Success },
        { NVSIPL_STATUS_BAD_ARGUMENT, NvError_BadParameter },
        { NVSIPL_STATUS_NOT_SUPPORTED, NvError_NotSupported },
        { NVSIPL_STATUS_OUT_OF_MEMORY, NvError_InsufficientMemory },
        { NVSIPL_STATUS_RESOURCE_ERROR, NvError_ResourceError },
        { NVSIPL_STATUS_TIMED_OUT, NvError_Timeout },
        { NVSIPL_STATUS_INVALID_STATE, NvError_InvalidState },
        { NVSIPL_STATUS_EOF, NvError_EndOfFile },
        { NVSIPL_STATUS_NOT_INITIALIZED, NvError_NotInitialized },
        { NVSIPL_STATUS_FAULT_STATE, NvError_InvalidState },
        { NVSIPL_STATUS_ERROR, NvError_InvalidState }
    };

    auto it = siplStatus2NvErrorMap.find(status);
    return it != siplStatus2NvErrorMap.end() ? it->second : NvError_BadValue;
}

NvError toNvError(NvMediaStatus status)
{
    static const std::unordered_map<NvMediaStatus, NvError> mediaStatus2NvErrorMap = {
        { NVMEDIA_STATUS_OK, NvError_Success },
        { NVMEDIA_STATUS_BAD_PARAMETER, NvError_BadParameter },
        { NVMEDIA_STATUS_PENDING, NvError_ResourceError },
        { NVMEDIA_STATUS_TIMED_OUT, NvError_Timeout },
        { NVMEDIA_STATUS_OUT_OF_MEMORY, NvError_InsufficientMemory },
        { NVMEDIA_STATUS_NOT_INITIALIZED, NvError_NotInitialized },
        { NVMEDIA_STATUS_NOT_SUPPORTED, NvError_NotSupported },
        { NVMEDIA_STATUS_ERROR, NvError_ResourceError },
        { NVMEDIA_STATUS_NONE_PENDING, NvError_ResourceError },
        { NVMEDIA_STATUS_INSUFFICIENT_BUFFERING, NvError_ResourceError },
        { NVMEDIA_STATUS_INVALID_SIZE, NvError_InvalidSize },
        { NVMEDIA_STATUS_INCOMPATIBLE_VERSION, NvError_ResourceError },
        { NVMEDIA_STATUS_UNDEFINED_STATE, NvError_InvalidState },
        { NVMEDIA_STATUS_PFSD_ERROR, NvError_ResourceError },
        { NVMEDIA_STATUS_INVALID_STATE, NvError_InvalidState }
    };

    auto it = mediaStatus2NvErrorMap.find(status);
    return it != mediaStatus2NvErrorMap.end() ? it->second : NvError_BadValue;
}

NvError toNvError(WFDErrorCode code)
{
    static const std::unordered_map<WFDErrorCode, NvError> wfdErrorCode2NvErrorMap = {
        { WFD_ERROR_NONE, NvError_Success },
        { WFD_ERROR_OUT_OF_MEMORY, NvError_InsufficientMemory },
        { WFD_ERROR_ILLEGAL_ARGUMENT, NvError_BadParameter },
        { WFD_ERROR_NOT_SUPPORTED, NvError_NotSupported },
        { WFD_ERROR_BAD_ATTRIBUTE, NvError_BadParameter },
        { WFD_ERROR_IN_USE, NvError_ResourceAlreadyInUse },
        { WFD_ERROR_BUSY, NvError_Busy },
        { WFD_ERROR_BAD_DEVICE, NvError_ResourceError },
        { WFD_ERROR_BAD_HANDLE, NvError_ResourceError },
        { WFD_ERROR_INCONSISTENCY, NvError_ResourceError },
        { WFD_ERROR_FORCE_32BIT, NvError_Force32 }
    };

    auto it = wfdErrorCode2NvErrorMap.find(code);
    return it != wfdErrorCode2NvErrorMap.end() ? it->second : NvError_BadValue;
}

#ifdef BUILD_VULKANSC
NvError toNvError(VkResult result)
{
    static const std::unordered_map<VkResult, NvError> vkResultCode2NvErrorMap = {
        { VK_SUCCESS, NvError_Success },
        { VK_NOT_READY, NvError_NotInitialized },
        { VK_TIMEOUT, NvError_Timeout },
        { VK_EVENT_SET, NvError_NotInitialized },
        { VK_EVENT_RESET, NvError_NotInitialized },
        { VK_INCOMPLETE, NvError_NotInitialized },
        { VK_ERROR_OUT_OF_HOST_MEMORY, NvError_InsufficientMemory },
        { VK_ERROR_OUT_OF_DEVICE_MEMORY, NvError_InsufficientMemory },
        { VK_ERROR_INITIALIZATION_FAILED, NvError_NotInitialized },
        { VK_ERROR_DEVICE_LOST, NvError_DeviceNotFound },
        { VK_ERROR_MEMORY_MAP_FAILED, NvError_MemoryMapFailed },
        { VK_ERROR_LAYER_NOT_PRESENT, NvError_ModuleNotPresent },
        { VK_ERROR_EXTENSION_NOT_PRESENT, NvError_ModuleNotPresent },
        { VK_ERROR_FEATURE_NOT_PRESENT, NvError_ModuleNotPresent },
        { VK_ERROR_INCOMPATIBLE_DRIVER, NvError_ResourceError },
        { VK_ERROR_TOO_MANY_OBJECTS, NvError_OverFlow },
        { VK_ERROR_FORMAT_NOT_SUPPORTED, NvError_NotSupported },
        { VK_ERROR_FRAGMENTED_POOL, NvError_ResourceError },
        { VK_ERROR_SURFACE_LOST_KHR, NvError_InvalidState },
        { VK_ERROR_NATIVE_WINDOW_IN_USE_KHR, NvError_Busy },
        { VK_SUBOPTIMAL_KHR, NvError_NotInitialized },
        { VK_ERROR_OUT_OF_DATE_KHR, NvError_NotInitialized },
        { VK_ERROR_INCOMPATIBLE_DISPLAY_KHR, NvError_NotInitialized },
        // { VK_ERROR_VALIDATION_FAILED_EXT, NvError_NotInitialized },
        // { VK_ERROR_INVALID_SHADER_NV, NvError_InvalidState },
        // { VK_ERROR_OUT_OF_POOL_MEMORY_KHR, NvError_NotInitialized },
        // { VK_ERROR_INVALID_EXTERNAL_HANDLE_KHR, NvError_InvalidState }
    };

    auto it = vkResultCode2NvErrorMap.find(result);
    return it != vkResultCode2NvErrorMap.end() ? it->second : NvError_BadValue;
}
#endif

/* Loads NITO file for given camera module.
 The function assumes the .nito files to be named same as camera module name.
 */
NvError LoadNITOFile(const std::string &folderPath, const std::string &moduleName, std::vector<uint8_t> &nito)
{
    // Set up blob file
    std::string nitoFilePath = (folderPath != "") ? folderPath : "/usr/share/camera/";
    std::string nitoFile = nitoFilePath + moduleName + ".nito";

    std::string moduleNameLower{};
    for (const auto &c : moduleName) {
        moduleNameLower.push_back(std::tolower(c));
    }
    std::string nitoFileLower = nitoFilePath + moduleNameLower + ".nito";
    std::string nitoFileDefault = nitoFilePath + "default.nito";

    // Open NITO file
    auto fp = fopen(nitoFile.c_str(), "rb");
    if (fp == nullptr) {
        LOG_INFO("File \"%s\" not found\n", nitoFile.c_str());
        // Try lower case name
        fp = fopen(nitoFileLower.c_str(), "rb");
        if (fp == nullptr) {
            LOG_INFO("File \"%s\" not found\n", nitoFileLower.c_str());
            LOG_ERR("Unable to open NITO file for module \"%s\", image quality is not supported!\n",
                    moduleName.c_str());
            return NvError_BadParameter;
        } else {
            LOG_MSG("nvsipl_multicast: Opened NITO file for module \"%s\"\n", moduleName.c_str());
        }
    } else {
        LOG_MSG("nvsipl_multicast: Opened NITO file for module \"%s\"\n", moduleName.c_str());
    }

    // Check file size
    fseek(fp, 0, SEEK_END);
    auto fsize = ftell(fp);
    rewind(fp);

    if (fsize <= 0U) {
        LOG_ERR("NITO file for module \"%s\" is of invalid size\n", moduleName.c_str());
        fclose(fp);
        return NvError_BadParameter;
    }

    /* allocate blob memory */
    nito.resize(fsize);

    /* load nito */
    auto result = (long int)fread(nito.data(), 1, fsize, fp);
    if (result != fsize) {
        LOG_ERR("Fail to read data from NITO file for module \"%s\", image quality is not supported!\n",
                moduleName.c_str());
        nito.resize(0);
        fclose(fp);
        return NvError_BadParameter;
    }
    /* close file */
    fclose(fp);

    LOG_INFO("data from NITO file loaded for module \"%s\"\n", moduleName.c_str());

    return NvError_Success;
}

const char *NvSciBufAttrKeyToString(NvSciBufAttrKey key)
{
    switch (key) {
        case NvSciBufGeneralAttrKey_Types:
            return "NvSciBufGeneralAttrKey_Types";
        case NvSciBufGeneralAttrKey_NeedCpuAccess:
            return "NvSciBufGeneralAttrKey_NeedCpuAccess";
        case NvSciBufGeneralAttrKey_RequiredPerm:
            return "NvSciBufGeneralAttrKey_RequiredPerm";
        case NvSciBufGeneralAttrKey_EnableCpuCache:
            return "NvSciBufGeneralAttrKey_EnableCpuCache";
        case NvSciBufGeneralAttrKey_GpuId:
            return "NvSciBufGeneralAttrKey_GpuId";
        case NvSciBufGeneralAttrKey_CpuNeedSwCacheCoherency:
            return "NvSciBufGeneralAttrKey_CpuNeedSwCacheCoherency";
        case NvSciBufGeneralAttrKey_ActualPerm:
            return "NvSciBufGeneralAttrKey_ActualPerm";
        case NvSciBufGeneralAttrKey_VidMem_GpuId:
            return "NvSciBufGeneralAttrKey_VidMem_GpuId";
        case NvSciBufGeneralAttrKey_EnableGpuCache:
            return "NvSciBufGeneralAttrKey_EnableGpuCache";
        case NvSciBufGeneralAttrKey_GpuSwNeedCacheCoherency:
            return "NvSciBufGeneralAttrKey_GpuSwNeedCacheCoherency";
        case NvSciBufGeneralAttrKey_EnableGpuCompression:
            return "NvSciBufGeneralAttrKey_EnableGpuCompression";
        case NvSciBufRawBufferAttrKey_Size:
            return "NvSciBufRawBufferAttrKey_Size";
        case NvSciBufRawBufferAttrKey_Align:
            return "NvSciBufRawBufferAttrKey_Align";
        case NvSciBufImageAttrKey_Layout:
            return "NvSciBufImageAttrKey_Layout";
        case NvSciBufImageAttrKey_TopPadding:
            return "NvSciBufImageAttrKey_TopPadding";
        case NvSciBufImageAttrKey_BottomPadding:
            return "NvSciBufImageAttrKey_BottomPadding";
        case NvSciBufImageAttrKey_LeftPadding:
            return "NvSciBufImageAttrKey_LeftPadding";
        case NvSciBufImageAttrKey_RightPadding:
            return "NvSciBufImageAttrKey_RightPadding";
        case NvSciBufImageAttrKey_VprFlag:
            return "NvSciBufImageAttrKey_VprFlag";
        case NvSciBufImageAttrKey_Size:
            return "NvSciBufImageAttrKey_Size";
        case NvSciBufImageAttrKey_Alignment:
            return "NvSciBufImageAttrKey_Alignment";
        case NvSciBufImageAttrKey_PlaneCount:
            return "NvSciBufImageAttrKey_PlaneCount";
        case NvSciBufImageAttrKey_PlaneColorFormat:
            return "NvSciBufImageAttrKey_PlaneColorFormat";
        case NvSciBufImageAttrKey_PlaneColorStd:
            return "NvSciBufImageAttrKey_PlaneColorStd";
        case NvSciBufImageAttrKey_PlaneBaseAddrAlign:
            return "NvSciBufImageAttrKey_PlaneBaseAddrAlign";
        case NvSciBufImageAttrKey_PlaneWidth:
            return "NvSciBufImageAttrKey_PlaneWidth";
        case NvSciBufImageAttrKey_PlaneHeight:
            return "NvSciBufImageAttrKey_PlaneHeight";
        case NvSciBufImageAttrKey_ScanType:
            return "NvSciBufImageAttrKey_ScanType";
        case NvSciBufImageAttrKey_PlaneBitsPerPixel:
            return "NvSciBufImageAttrKey_PlaneBitsPerPixel";
        case NvSciBufImageAttrKey_PlaneOffset:
            return "NvSciBufImageAttrKey_PlaneOffset";
        case NvSciBufImageAttrKey_PlaneDatatype:
            return "NvSciBufImageAttrKey_PlaneDatatype";
        case NvSciBufImageAttrKey_PlaneChannelCount:
            return "NvSciBufImageAttrKey_PlaneChannelCount";
        case NvSciBufImageAttrKey_PlaneSecondFieldOffset:
            return "NvSciBufImageAttrKey_PlaneSecondFieldOffset";
        case NvSciBufImageAttrKey_PlanePitch:
            return "NvSciBufImageAttrKey_PlanePitch";
        case NvSciBufImageAttrKey_PlaneAlignedHeight:
            return "NvSciBufImageAttrKey_PlaneAlignedHeight";
        case NvSciBufImageAttrKey_PlaneAlignedSize:
            return "NvSciBufImageAttrKey_PlaneAlignedSize";
        case NvSciBufImageAttrKey_ImageCount:
            return "NvSciBufImageAttrKey_ImageCount";
        case NvSciBufImageAttrKey_SurfType:
            return "NvSciBufImageAttrKey_SurfType";
        case NvSciBufImageAttrKey_SurfMemLayout:
            return "NvSciBufImageAttrKey_SurfMemLayout";
        case NvSciBufImageAttrKey_SurfSampleType:
            return "NvSciBufImageAttrKey_SurfSampleType";
        case NvSciBufImageAttrKey_SurfBPC:
            return "NvSciBufImageAttrKey_SurfBPC";
        case NvSciBufImageAttrKey_SurfComponentOrder:
            return "NvSciBufImageAttrKey_SurfComponentOrder";
        case NvSciBufImageAttrKey_SurfWidthBase:
            return "NvSciBufImageAttrKey_SurfWidthBase";
        case NvSciBufImageAttrKey_SurfHeightBase:
            return "NvSciBufImageAttrKey_SurfHeightBase";
        case NvSciBufTensorAttrKey_DataType:
            return "NvSciBufTensorAttrKey_DataType";
        case NvSciBufTensorAttrKey_NumDims:
            return "NvSciBufTensorAttrKey_NumDims";
        case NvSciBufTensorAttrKey_SizePerDim:
            return "NvSciBufTensorAttrKey_SizePerDim";
        case NvSciBufTensorAttrKey_AlignmentPerDim:
            return "NvSciBufTensorAttrKey_AlignmentPerDim";
        case NvSciBufTensorAttrKey_StridesPerDim:
            return "NvSciBufTensorAttrKey_StridesPerDim";
        case NvSciBufTensorAttrKey_PixelFormat:
            return "NvSciBufTensorAttrKey_PixelFormat";
        case NvSciBufTensorAttrKey_BaseAddrAlign:
            return "NvSciBufTensorAttrKey_BaseAddrAlign";
        case NvSciBufTensorAttrKey_Size:
            return "NvSciBufTensorAttrKey_Size";
        case NvSciBufArrayAttrKey_DataType:
            return "NvSciBufArrayAttrKey_DataType";
        case NvSciBufArrayAttrKey_Stride:
            return "NvSciBufArrayAttrKey_Stride";
        case NvSciBufArrayAttrKey_Capacity:
            return "NvSciBufArrayAttrKey_Capacity";
        case NvSciBufArrayAttrKey_Size:
            return "NvSciBufArrayAttrKey_Size";
        case NvSciBufArrayAttrKey_Alignment:
            return "NvSciBufArrayAttrKey_Alignment";
        case NvSciBufPyramidAttrKey_NumLevels:
            return "NvSciBufPyramidAttrKey_NumLevels";
        case NvSciBufPyramidAttrKey_Scale:
            return "NvSciBufPyramidAttrKey_Scale";
        case NvSciBufPyramidAttrKey_LevelOffset:
            return "NvSciBufPyramidAttrKey_LevelOffset";
        case NvSciBufPyramidAttrKey_LevelSize:
            return "NvSciBufPyramidAttrKey_LevelSize";
        case NvSciBufPyramidAttrKey_Alignment:
            return "NvSciBufPyramidAttrKey_Alignment";
        default:
            return "Unknown Attribute";
    }
}

/*
 * Copy if the source buffer is valid and the size of the dest buffer greater or equal to the size
 * of the source buffer.
 */
#define NVSCIBUFATTR_COPY(attrKey, nvsciDst, nvsciSrc, length)                                                   \
    {                                                                                                            \
        if (nvsciSrc) {                                                                                          \
            if (sizeof(nvsciDst) >= length) {                                                                    \
                const void *src = (const void *)nvsciSrc;                                                        \
                void *dst = (void *)&(nvsciDst);                                                                 \
                memcpy(dst, src, length);                                                                        \
            } else {                                                                                             \
                LOG_ERR("Retrieved attribute(%s) length is out of range Length:%zu, Expected range: 0 to %zu\n", \
                        NvSciBufAttrKeyToString(attrKey), length, sizeof(nvsciDst));                             \
                return NvError_InsufficientMemory;                                                               \
            }                                                                                                    \
        } else {                                                                                                 \
            LOG_WARN("Retrieved attribute(%s) doesn't exist.\n", NvSciBufAttrKeyToString(attrKey));              \
        }                                                                                                        \
    }

NvError SetBufAttr(NvSciBufAttrList *pBufAttrList,
                   const std::string &sColorType,
                   const std::string &sImageLayout,
                   uint32_t &uWidth,
                   uint32_t &uHeight)
{
    NvSciError sciErr = NvSciError_Success;

    // Default layout type attribute(BL)
    NvSciBufAttrValImageLayoutType layout;
    if (sImageLayout.empty() || sImageLayout == "BL") {
        layout = NvSciBufImage_BlockLinearType;
    } else if (sImageLayout == "PL") {
        layout = NvSciBufImage_PitchLinearType;
    } else {
        LOG_ERR("Image layout type specified not support.\n");
        return NvError_NotSupported;
    }

    // Default color type attributes(nv12)
    if (sColorType.empty() || sColorType == "NV12") {
        NvSciBufSurfSampleType surfSampleType = NvSciSurfSampleType_420;
        NvSciBufSurfBPC surfBPC = NvSciSurfBPC_8;
        NvSciBufSurfType surfType = NvSciSurfType_YUV;
        NvSciBufSurfMemLayout surfMemLayout = NvSciSurfMemLayout_SemiPlanar;
        NvSciBufSurfComponentOrder surfCompOrder = NvSciSurfComponentOrder_YUV;
        NvSciBufAttrValColorStd surfColorStd[] = { NvSciColorStd_REC709_ER };

        NvSciBufAttrKeyValuePair bufAttrs[] = {
            { NvSciBufImageAttrKey_SurfType, &surfType, sizeof(surfType) },
            { NvSciBufImageAttrKey_SurfBPC, &surfBPC, sizeof(surfBPC) },
            { NvSciBufImageAttrKey_SurfWidthBase, &uWidth, sizeof(uint32_t) },
            { NvSciBufImageAttrKey_SurfHeightBase, &uHeight, sizeof(uint32_t) },
            { NvSciBufImageAttrKey_SurfMemLayout, &surfMemLayout, sizeof(surfMemLayout) },
            { NvSciBufImageAttrKey_SurfSampleType, &surfSampleType, sizeof(surfSampleType) },
            { NvSciBufImageAttrKey_SurfComponentOrder, &surfCompOrder, sizeof(surfCompOrder) },
            { NvSciBufImageAttrKey_SurfColorStd, &surfColorStd, sizeof(surfColorStd) },
            { NvSciBufImageAttrKey_Layout, (void *)&layout, sizeof(layout) },
        };

        sciErr = NvSciBufAttrListSetAttrs(*pBufAttrList, bufAttrs, ARRAY_SIZE(bufAttrs));
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");
    } else if (sColorType == "ABGR" || sColorType == "ARGB") {
        NvSciBufAttrValColorFmt colorFormat = sColorType == "ABGR" ? NvSciColor_A8B8G8R8 : NvSciColor_A8R8G8B8;
        uint32_t uPlaneCount = 1U;

        NvSciBufAttrKeyValuePair bufAttrs[] = {
            { NvSciBufImageAttrKey_PlaneCount, (void *)&uPlaneCount, sizeof(uPlaneCount) },
            { NvSciBufImageAttrKey_PlaneColorFormat, (void *)&colorFormat, sizeof(colorFormat) },
            { NvSciBufImageAttrKey_PlaneWidth, &uWidth, sizeof(uint32_t) },
            { NvSciBufImageAttrKey_PlaneHeight, &uHeight, sizeof(uint32_t) },
            { NvSciBufImageAttrKey_Layout, (void *)&layout, sizeof(layout) },
        };
        sciErr = NvSciBufAttrListSetAttrs(*pBufAttrList, bufAttrs, ARRAY_SIZE(bufAttrs));
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");
    } else {
        LOG_ERR("Color type specified not support.\n");
        return NvError_NotSupported;
    }

    return NvError_Success;
}

NvError PopulateBufAttr(const NvSciBufObj &sciBufObj, BufferAttrs &bufAttrs)
{
    NvSciError err = NvSciError_Success;
    NvSciBufAttrList bufAttrList;

    NvSciBufAttrKeyValuePair imgAttrs[] = {
        { NvSciBufImageAttrKey_Size, NULL, 0 },                     //0
        { NvSciBufImageAttrKey_Layout, NULL, 0 },                   //1
        { NvSciBufImageAttrKey_PlaneCount, NULL, 0 },               //2
        { NvSciBufImageAttrKey_PlaneWidth, NULL, 0 },               //3
        { NvSciBufImageAttrKey_PlaneHeight, NULL, 0 },              //4
        { NvSciBufImageAttrKey_PlanePitch, NULL, 0 },               //5
        { NvSciBufImageAttrKey_PlaneBitsPerPixel, NULL, 0 },        //6
        { NvSciBufImageAttrKey_PlaneAlignedHeight, NULL, 0 },       //7
        { NvSciBufImageAttrKey_PlaneAlignedSize, NULL, 0 },         //8
        { NvSciBufImageAttrKey_PlaneChannelCount, NULL, 0 },        //9
        { NvSciBufImageAttrKey_PlaneOffset, NULL, 0 },              //10
        { NvSciBufImageAttrKey_PlaneColorFormat, NULL, 0 },         //11
        { NvSciBufImageAttrKey_TopPadding, NULL, 0 },               //12
        { NvSciBufImageAttrKey_BottomPadding, NULL, 0 },            //13
        { NvSciBufGeneralAttrKey_CpuNeedSwCacheCoherency, NULL, 0 } //14
    };

    err = NvSciBufObjGetAttrList(sciBufObj, &bufAttrList);
    CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufObjGetAttrList");
    err = NvSciBufAttrListGetAttrs(bufAttrList, imgAttrs, ARRAY_SIZE(imgAttrs));
    CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufAttrListGetAttrs");

    bufAttrs.size = *(static_cast<const uint64_t *>(imgAttrs[0].value));
    bufAttrs.layout = *(static_cast<const NvSciBufAttrValImageLayoutType *>(imgAttrs[1].value));
    bufAttrs.planeCount = *(static_cast<const uint32_t *>(imgAttrs[2].value));
    bufAttrs.needSwCacheCoherency = *(static_cast<const bool *>(imgAttrs[14].value));

    NVSCIBUFATTR_COPY(NvSciBufImageAttrKey_PlaneWidth, bufAttrs.planeWidths, imgAttrs[3].value, imgAttrs[3].len);

    NVSCIBUFATTR_COPY(NvSciBufImageAttrKey_PlaneHeight, bufAttrs.planeHeights, imgAttrs[4].value, imgAttrs[4].len);

    NVSCIBUFATTR_COPY(NvSciBufImageAttrKey_PlanePitch, bufAttrs.planePitches, imgAttrs[5].value, imgAttrs[5].len);

    NVSCIBUFATTR_COPY(NvSciBufImageAttrKey_PlaneBitsPerPixel, bufAttrs.planeBitsPerPixels, imgAttrs[6].value,
                      imgAttrs[6].len);

    NVSCIBUFATTR_COPY(NvSciBufImageAttrKey_PlaneAlignedHeight, bufAttrs.planeAlignedHeights, imgAttrs[7].value,
                      imgAttrs[7].len);

    NVSCIBUFATTR_COPY(NvSciBufImageAttrKey_PlaneAlignedSize, bufAttrs.planeAlignedSizes, imgAttrs[8].value,
                      imgAttrs[8].len);

    NVSCIBUFATTR_COPY(NvSciBufImageAttrKey_PlaneChannelCount, bufAttrs.planeChannelCounts, imgAttrs[9].value,
                      imgAttrs[9].len);

    NVSCIBUFATTR_COPY(NvSciBufImageAttrKey_PlaneOffset, bufAttrs.planeOffsets, imgAttrs[10].value, imgAttrs[10].len);

    NVSCIBUFATTR_COPY(NvSciBufImageAttrKey_PlaneColorFormat, bufAttrs.planeColorFormats, imgAttrs[11].value,
                      imgAttrs[11].len);

    NVSCIBUFATTR_COPY(NvSciBufImageAttrKey_TopPadding, bufAttrs.topPadding, imgAttrs[12].value, imgAttrs[12].len);

    NVSCIBUFATTR_COPY(NvSciBufImageAttrKey_BottomPadding, bufAttrs.bottomPadding, imgAttrs[13].value, imgAttrs[13].len);

    //Print sciBuf attributes
    LOG_DBG("========PopulateBufAttr========\n");
    LOG_DBG("size=%lu, layout=%u, planeCount=%u\n", bufAttrs.size, bufAttrs.layout, bufAttrs.planeCount);
    for (auto i = 0U; i < bufAttrs.planeCount; i++) {
        LOG_DBG(
            "plane %u: planeWidth=%u, planeHeight=%u, planePitch=%u, planeBitsPerPixels=%u, planeAlignedHeight=%u\n", i,
            bufAttrs.planeWidths[i], bufAttrs.planeHeights[i], bufAttrs.planePitches[i], bufAttrs.planeBitsPerPixels[i],
            bufAttrs.planeAlignedHeights[i]);
        LOG_DBG("plane %u: planeAlignedSize=%lu, planeOffset=%lu, planeColorFormat=%u, planeChannelCount=%u\n", i,
                bufAttrs.planeAlignedSizes[i], bufAttrs.planeOffsets[i], bufAttrs.planeColorFormats[i],
                bufAttrs.planeChannelCounts[i]);
    }

    return NvError_Success;
}

NvError GetWidthAndHeight(
    const NvSciBufAttrList bufAttrList, uint16_t &uWidth, uint16_t &uHeight, uint32_t *pPlanePitches, uint32_t size)
{
    uint32_t planeWidths[MAX_NUM_SURFACES] = { 0 };
    uint32_t planeHeights[MAX_NUM_SURFACES] = { 0 };

    NvSciBufAttrKeyValuePair imgAttrs[] = {
        { NvSciBufImageAttrKey_PlaneWidth, NULL, 0 },
        { NvSciBufImageAttrKey_PlaneHeight, NULL, 0 },
        { NvSciBufImageAttrKey_PlanePitch, NULL, 0 },
    };

    auto err = NvSciBufAttrListGetAttrs(bufAttrList, imgAttrs, ARRAY_SIZE(imgAttrs));
    CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufAttrListGetAttrs");

    NVSCIBUFATTR_COPY(NvSciBufImageAttrKey_PlaneWidth, planeWidths, imgAttrs[0].value, imgAttrs[0].len);
    NVSCIBUFATTR_COPY(NvSciBufImageAttrKey_PlaneHeight, planeHeights, imgAttrs[1].value, imgAttrs[1].len);

    uWidth = (uint16_t)planeWidths[0];
    uHeight = (uint16_t)planeHeights[0];
    if (pPlanePitches && size > 0) {
        int size_min = std::min<uint32_t>(size, imgAttrs[2].len);
        memcpy((void *)pPlanePitches, (const void *)imgAttrs[2].value, size_min);
    }

    return NvError_Success;
}

std::vector<std::string> splitString(const std::string &inputString, char delimiter)
{
    std::vector<std::string> result;
    std::istringstream iss(inputString);
    std::string token;

    while (std::getline(iss, token, delimiter)) {
        result.push_back(token);
    }

    return result;
}

NvError DumpBufAttr(const NvSciBufObj &sciBufObj)
{
    NvSciError err = NvSciError_Success;
    NvSciBufAttrList bufAttrList;

    NvSciBufAttrKeyValuePair imgAttrs[] = {
        { NvSciBufGeneralAttrKey_Types, NULL, 0 },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, NULL, 0 },
        { NvSciBufGeneralAttrKey_ActualPerm, NULL, 0 },
        { NvSciBufImageAttrKey_Layout, NULL, 0 },
        { NvSciBufImageAttrKey_PlaneCount, NULL, 0 },
        { NvSciBufImageAttrKey_PlaneWidth, NULL, 0 },
        { NvSciBufImageAttrKey_PlaneHeight, NULL, 0 },
        { NvSciBufImageAttrKey_PlanePitch, NULL, 0 },
        { NvSciBufImageAttrKey_PlaneBitsPerPixel, NULL, 0 },
        { NvSciBufImageAttrKey_PlaneAlignedHeight, NULL, 0 },
        { NvSciBufImageAttrKey_PlaneAlignedSize, NULL, 0 },
        { NvSciBufImageAttrKey_PlaneChannelCount, NULL, 0 },
        { NvSciBufImageAttrKey_PlaneOffset, NULL, 0 },
        { NvSciBufImageAttrKey_PlaneColorFormat, NULL, 0 },
        { NvSciBufImageAttrKey_TopPadding, NULL, 0 },
        { NvSciBufImageAttrKey_BottomPadding, NULL, 0 },
        { NvSciBufGeneralAttrKey_CpuNeedSwCacheCoherency, NULL, 0 },
    };

    err = NvSciBufObjGetAttrList(sciBufObj, &bufAttrList);
    CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufObjGetAttrList");
    err = NvSciBufAttrListGetAttrs(bufAttrList, imgAttrs, ARRAY_SIZE(imgAttrs));
    CHK_NVSCISTATUS_AND_RETURN(err, "NvSciBufAttrListGetAttrs");

    LOG_MSG("Dump scibuf attribute:\n");
    for (size_t i = 0; i < ARRAY_SIZE(imgAttrs); i++) {
        if ((imgAttrs[i].len > 0) && imgAttrs[i].value != nullptr)
            LOG_MSG("    %s 0x%x %ld\n", NvSciBufAttrKeyToString(imgAttrs[i].key), *(uint32_t *)imgAttrs[i].value,
                    imgAttrs[i].len);
    }

    return NvError_Success;
}

const char *EventStatusToString(EventStatus event)
{
    switch (event) {
        case EventStatus::OK:
            return "EventStatus::OK";
        case EventStatus::RECONCILED:
            return "EventStatus::RECONCILED";
        case EventStatus::DISCONNECT:
            return "EventStatus::DISCONNECT";
        case EventStatus::CONNECTED:
            return "EventStatus::CONNECTED";
        case EventStatus::STARTED:
            return "EventStatus::STARTED";
        case EventStatus::STOPPED:
            return "EventStatus::STOPPED";
        case EventStatus::QUITTED:
            return "EventStatus::QUITTED";
        case EventStatus::TIMED_OUT:
            return "EventStatus::TIMED_OUT";
        case EventStatus::ERROR:
            return "EventStatus::ERROR";
        default:
            return "Unknown EventStatus";
    }
}

NvError CheckSKU(const std::string &findStr, bool &bFound)
{
#if !NVMEDIA_QNX
    std::string sTargetModelNode = "/proc/device-tree/model";
    std::ifstream fs;
    fs.open(sTargetModelNode, std::ifstream::in);
    if (!fs.is_open()) {
        LOG_ERR("%s: cannot open board node %s\n", __func__, sTargetModelNode.c_str());
        return NvError_FileOperationFailed;
    }

    // Read the file in to the string.
    std::string nodeString;
    fs >> nodeString;

    if (strstr(nodeString.c_str(), findStr.c_str())) {
        bFound = true;
    }

    if (fs.is_open()) {
        fs.close();
    }
#else  // NVMEDIA_QNX
    /* Get handle for DTB */
    if (NVDT_SUCCESS != nvdt_open()) {
        LOG_ERR("nvdt_open failed\n");
        return NvError_InsufficientMemory;
    }

    /* Check the Model */
    const void *modelNode;
    modelNode = nvdt_get_node_by_path("/");
    if (modelNode == NULL) {
        LOG_ERR("No node for model\n");
        (void)nvdt_close();
        return NvError_InsufficientMemory;
    }

    char name[20];
    NvError error =
        GetDTPropAsString(modelNode, "model", &name[0], static_cast<uint32_t>(sizeof(name) / sizeof(name[0])));
    if (error != NvError_Success) {
        (void)nvdt_close();
        return error;
    }

    if (strstr(name, findStr.c_str())) {
        bFound = true;
    }
    /* close nvdt once done */
    (void)nvdt_close();
#endif // !NVMEDIA_QNX

    return NvError_Success;
}

#if NVMEDIA_QNX
NvError GetDTPropAsString(const void *node, const char *const name, char val[], const uint32_t size)
{
    CHK_PTR_AND_RETURN_BADARG(node, "node");
    CHK_PTR_AND_RETURN_BADARG(name, "name");
    CHK_PTR_AND_RETURN_BADARG(val, "val");

    if (size == 0U) {
        LOG_ERR("size cannot be zero\n");
        return NvError_BadValue;
    }

    uint32_t propLengthBytes = 0U;
    void const *const val_str = nvdt_node_get_prop(node, name, &propLengthBytes);
    CHK_PTR_AND_RETURN_BADARG(val_str, "val_str");
    if (propLengthBytes == 0U) {
        LOG_ERR("Property string cannot be zero-length\n");
        return NvError_BadValue;
    }

    if (propLengthBytes > size) {
        LOG_ERR("Property string exceeds maximum length\n");
        return NvError_BadValue;
    }

    memcpy(&val[0], val_str, static_cast<std::size_t>(propLengthBytes));

    if (val[(propLengthBytes - 1U)] != '\0') {
        LOG_ERR("Failed to parse property string\n");
        return NvError_BadValue;
    }

    return NvError_Success;
}
#endif

static constexpr const char *kWhiteSpaces = " \n\t\r";
NvError GetToken(const char **pBuf, const char *pTerm, std::string &token)
{
    NvError error = NvError_Success;
    if (pBuf && pTerm) {
        const char *pCurrent = *pBuf;
        pCurrent += std::strspn(pCurrent, kWhiteSpaces);
        while (*pCurrent && !std::strspn(pCurrent, pTerm)) {
            char c = *pCurrent++;
            if (c == '\\' && *pCurrent) {
                token.push_back(*pCurrent++);
            } else if (c == '\'') {
                while (*pCurrent && *pCurrent != '\'') {
                    token.push_back(*pCurrent++);
                }
                if (*pCurrent) {
                    pCurrent++;
                }
            } else {
                token.push_back(c);
            }
        }
        *pBuf = pCurrent;
    } else {
        LOG_ERR("Invalid buffer string or terminate string.\n");
        error = NvError_BadParameter;
    }
    return error;
}

std::string TrimBlank(std::string str)
{
    str.erase(remove(str.begin(), str.end(), ' '), str.end());
    return str;
}

std::string ToLower(std::string str)
{
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    return str;
}

std::string IntToStringWithLeadingZero(int num)
{
    std::ostringstream oss;
    oss << std::setw(2) << std::setfill('0') << num;
    return oss.str();
}

void PrintDeviceGid(const char *pDeviceStr, const char *pTypeStr, const uint8_t *pId)
{
    LOG_INFO(" %s - %-25s %02hhx%02hhx%02hhx%02hhx-%02hhx%02hhx-%02hhx%02hhx-"
             "%02hhx%02hhx%02hhx%02hhx%02hhx%02hhx%02hhx%02hhx \n",
             pDeviceStr, pTypeStr, pId[0U], pId[1U], pId[2U], pId[3U], pId[4U], pId[5U], pId[6U], pId[7U], pId[8U],
             pId[9U], pId[10U], pId[11U], pId[12U], pId[13U], pId[14U], pId[15U]);
}

bool GetRoi(const std::string &sLine, std::vector<NvMediaRect> &rois)
{
    std::istringstream iss(sLine);
    int numROIs;
    iss >> numROIs;

    if (numROIs >= 0 && static_cast<uint32_t>(numROIs) <= MetaData::kMaxROIRegions) {
        for (int i = 0; i < numROIs; ++i) {
            NvMediaRect rect;
            iss >> rect.x0 >> rect.y0 >> rect.x1 >> rect.y1;
            rois.push_back(rect);
        }
    } else {
        LOG_ERR("ROI num(%d) out of range.Expected range[%" PRIu32 ", %" PRIu32 "]", numROIs, 0U,
                MetaData::kMaxROIRegions);
    }

    return !rois.empty();
}

#if !NVMEDIA_QNX
#define SYSFS_PROFILING_POINT "/sys/kernel/tegra_bootloader/add_profiler_record" //Linux only
#else
#define QNX_PROFILING_POINT "/dev/bootprofiler" //QNX only
#endif

#if !NVMEDIA_QNX
void recordTimestampInCarveout(const std::string &carveoutMsg, const std::string &logFileName)
{
    std::ofstream carveoutProfilingPoint;
    const std::string &carveoutProfilingName = logFileName.empty() ? SYSFS_PROFILING_POINT : logFileName;

    carveoutProfilingPoint.open(carveoutProfilingName);
    if (!carveoutProfilingPoint.is_open()) {
        LOG_ERR("Failed to open profiling carveout file\n");
        return;
    }
    carveoutProfilingPoint << carveoutMsg;
    carveoutProfilingPoint.close();
}
#else
void recordTimestampInCarveout(const std::string &carveoutMsg, const std::string &logFileName)
{
    const std::string &carveoutProfilingName = logFileName.empty() ? QNX_PROFILING_POINT : logFileName;
    int profilerFd = open(carveoutProfilingName.c_str(), O_RDWR);
    if (profilerFd < 0) {
        LOG_ERR("Failed to open profiling carveout file: " + carveoutProfilingName);
        return;
    }
    devctl(profilerFd, __DIOTF(_DCMD_MISC, 0x01, char), const_cast<char *>(carveoutMsg.c_str()),
           carveoutMsg.length() + 1, NULL);
    close(profilerFd);
}
#endif

long Random() noexcept
{
    long lResult = -1;
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd >= 0) {
        ssize_t size = read(fd, &lResult, sizeof(lResult));
        if (size < 0) {
            LOG_ERR("Failed to read data from /dev/urandom");
        }
        close(fd);
        fd = -1;
    } else {
        LOG_ERR("Failed to open /dev/urandom due to %s", strerror(errno));
    }
    return lResult;
}