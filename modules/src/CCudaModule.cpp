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

#include "CCudaModule.hpp"
#include "CElementDescription.hpp"
// cuda cvt2d kernels includes
#ifndef NVMEDIA_QNX
#include "cvt2d_kernels.h"
#include "cudaYUV.h"
#endif

#if BUILD_CARDETECT
CElementDescription cudaDescription{ "Cuda", "Cuda module for car detection", &CBaseModule::m_baseModuleOptionTable,
                                     nullptr };
#else
CElementDescription cudaDescription{ "Cuda", "Cuda module for API demostration", &CBaseModule::m_baseModuleOptionTable,
                                     nullptr };
#endif

const std::unordered_map<std::string, Option> CCudaModule::m_cudaOptionTable = {
    { "width", { "the width of the input buffer", offsetof(CUDAInputInfo, uWidth), OptionType::UINT32 } },
    { "height", { "the height of the input buffer", offsetof(CUDAInputInfo, uHeight), OptionType::UINT32 } },
    { "imagelayout", { "display imagelayout", offsetof(CUDAInputInfo, sImageLayout), OptionType::STRING } },
    { "usePva", { "use pva for preprocessing", offsetof(CUDAInputInfo, bUsePva), OptionType::BOOL } },
    { "draw", { "draw the bounding box on the image", offsetof(CUDAInputInfo, bDraw), OptionType::BOOL } }
};

int CCudaModule::m_cudaDeviceId = 0;

CCudaModule::CCudaModule(std::shared_ptr<CModuleCfg> spModuleCfg, IEventListener<CBaseModule> *pListener)
    : CBaseModule(spModuleCfg, pListener)
    , m_streamWaiter(nullptr)
    , m_signalerSem(0U)
    , m_waiterSem(0U)
    , m_extMem{ 0U }
    , m_pDevPtrs{ nullptr }
    , m_bFirstCall(true)
    , m_pCvtDevPtrs{ nullptr }
{
    spModuleCfg->m_cpuWaitCfg.bWaitPrefence = true;
    spModuleCfg->m_cpuWaitCfg.bWaitPostfence = true;
}

NvError CCudaModule::InitCuda()
{
    size_t unused;
    auto cudaStatus = cudaDeviceGetLimit(&unused, cudaLimitStackSize);
    PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaDeviceGetLimit");

    int numOfGPUs = 0;
    std::vector<int> deviceIds;
    cudaStatus = cudaGetDeviceCount(&numOfGPUs);
    PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaGetDeviceCount");
    if (numOfGPUs <= 0) {
        PLOG_ERR("Created consumer's CUDA stream.\n");
        return NvError_BadValue;
    }

    deviceIds.resize(numOfGPUs);
    cudaStatus = cudaGetDevice(deviceIds.data());
    PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaGetDevice");
    m_cudaDeviceId = deviceIds[0]; // use the first device.

    cudaStatus = cudaSetDevice(m_cudaDeviceId);
    PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaSetDevice");

    cudaStatus = cudaStreamCreateWithFlags(&m_streamWaiter, cudaStreamNonBlocking);
    PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaStreamCreateWithFlags");

    auto sensor_id = GetSensorId();
    auto config = m_pAppCfg->GetCameraConfig(sensor_id);
    if(config) {
        return NvError_BadParameter;
    }
    m_cudaInputInfo.uCvtWidth = config->width;
    m_cudaInputInfo.uCvtHeight = config->height;
    m_cudaInputInfo.uHeight = config->height;
    m_cudaInputInfo.uWidth = config->width;
    m_cudaInputInfo.sImageLayout = "BL";

    image_size_ = m_cudaInputInfo.uCvtWidth * m_cudaInputInfo.uCvtHeight *
                            3 * sizeof(uint8_t);
    cudaStatus = cudaMalloc(&m_pCvtDevPtrs,
                            image_size_);
    PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaMalloc m_pCvtDevPtrs");
    PLOG_DBG("Created consumer's CUDA stream.\n");

#if BUILD_CARDETECT
    // car detection
    if (m_cudaInputInfo.bUsePva)
        m_upCarDetect.reset(new CCarDetector(true));
    else
        m_upCarDetect.reset(new CCarDetector());

    PCHK_PTR_AND_RETURN(m_upCarDetect, "CCarDetector memory allocation")
    bool ret = m_upCarDetect->Init(GetSensorId(), m_streamWaiter);

    if (!ret) {
        PLOG_WARN("Car detection won't be enable because of init failed, Please check the model exist.\n");
        LOG_MSG("Running the cuda consumer without inference now ...\n");
    }
#else
    LOG_MSG("Warning: Running the cuda consumer without inference now ...\n");
    LOG_MSG("Warning: If you want inference, please enable the compile switch NV_BUILD_CARDETECT\n");
#endif

    return NvError_Success;
}

NvError CCudaModule::Init()
{
    PLOG_DBG("Enter: CCudaModule::Init()\n");

    NvError error = NvError_Success;
    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler = std::make_unique<CProfiler>();
        error = m_upProfiler->Init(m_pAppCfg->m_sciSyncModule, m_pAppCfg->GetPerfDataSaveFolder(), GetName(), false,
                                   m_pAppCfg->GetMaxPerfSampleNum());
        PCHK_ERROR_AND_RETURN(error, "CProfiler::Init()");

        m_upProfiler->RecordInitBeginTime();
    }

    error = CBaseModule::Init();
    PCHK_ERROR_AND_RETURN(error, "CBaseModule::Init()");

    error = InitCuda();
    PCHK_ERROR_AND_RETURN(error, "InitCuda");

    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler->RecordInitEndTime();
    }

    PLOG_DBG("Exit: CCudaModule::Init()\n");

    return NvError_Success;
}

void CCudaModule::DeInit()
{
    PLOG_DBG("Enter: CCudaModule::DeInit()\n");

#if BUILD_CARDETECT
    // car detection
    if (m_upCarDetect != nullptr) {
        m_upCarDetect->DeInit();
        m_upCarDetect.reset(nullptr);
    }
#endif
    if (m_waiterSem != nullptr) {
        cudaDestroyExternalSemaphore(m_waiterSem);
        m_waiterSem = nullptr;
    }
    if (m_signalerSem != nullptr) {
        cudaDestroyExternalSemaphore(m_signalerSem);
        m_signalerSem = nullptr;
    }

    for (uint32_t i = 0U; i < MAX_NUM_PACKETS; i++) {
        if (m_pDevPtrs[i] != nullptr) {
            cudaFree(m_pDevPtrs[i]);
            m_pDevPtrs[i] = nullptr;
        }
        if (m_pCvtDevPtrs != nullptr) {
            cudaFree(m_pCvtDevPtrs);
            m_pCvtDevPtrs = nullptr;
        }

        if(output_buffer_ != nullptr) {
            cudaFree(output_buffer_);
            output_buffer_ = nullptr;
        }

        if (m_extMem[i] != 0) {
            cudaDestroyExternalMemory(m_extMem[i]);
            m_extMem[i] = nullptr;
        }
    }
    for (uint32_t i = 0U; i < MAX_NUM_PACKETS; i++) {
        for (uint32_t j = 0U; j < MAX_NUM_SURFACES; j++) {
            if (m_mipmapArray[i][j] != nullptr) {
                cudaFreeMipmappedArray(m_mipmapArray[i][j]);
                m_mipmapArray[i][j] = nullptr;
            }
        }
    }

    cudaStreamDestroy(m_streamWaiter);

    CBaseModule::DeInit();

    PLOG_DBG("Exit: CCudaModule::DeInit()\n");
}

NvError CCudaModule::GetSyncWaiterAttrList(NvSciSyncAttrList outWaiterAttrList)
{
    if (!outWaiterAttrList) {
        return NvError_BadParameter;
    }

    auto cudaStatus = cudaDeviceGetNvSciSyncAttributes(outWaiterAttrList, m_cudaDeviceId, cudaNvSciSyncAttrWait);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaDeviceGetNvSciSyncAttributes");

    return NvError_Success;
}

NvError CCudaModule::GetBufAttrList(NvSciBufAttrList bufAttrList)
{
    size_t unused;
    auto cudaStatus = cudaDeviceGetLimit(&unused, cudaLimitStackSize);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaDeviceGetLimit");

    cudaStatus = cudaSetDevice(m_cudaDeviceId);
    CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaSetDevice");

    NvSciRmGpuId gpuId;
    CUuuid uuid{};
    auto cudaErr = cuDeviceGetUuid(&uuid, m_cudaDeviceId);
    CHK_CUDAERR_AND_RETURN(cudaErr, "cuDeviceGetUuid");
    memcpy(&gpuId.bytes, &uuid.bytes, sizeof(uuid.bytes));
    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_ReadWrite;
    bool bCpuAccess = true;

    NvSciBufAttrKeyValuePair bufAttrs[] = {
        { NvSciBufGeneralAttrKey_GpuId, &gpuId, sizeof(gpuId) },
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &bCpuAccess, sizeof(bCpuAccess) },
    };

    NvSciError sciErr = NvSciBufAttrListSetAttrs(bufAttrList, bufAttrs, ARRAY_SIZE(bufAttrs));
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");
    return NvError_Success;
}

NvError
CCudaModule::FillDataBufAttrList(CClientCommon *pClient, PacketElementType userType, NvSciBufAttrList *pBufAttrList)
{
    size_t unused;
    auto cudaStatus = cudaDeviceGetLimit(&unused, cudaLimitStackSize);
    PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaDeviceGetLimit");

    cudaStatus = cudaSetDevice(m_cudaDeviceId);
    PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaSetDevice");

    NvSciRmGpuId gpuId;
    CUuuid uuid{};
    auto cudaErr = cuDeviceGetUuid(&uuid, m_cudaDeviceId);
    CHK_CUDAERR_AND_RETURN(cudaErr, "cuDeviceGetUuid");
    memcpy(&gpuId.bytes, &uuid.bytes, sizeof(uuid.bytes));
    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_ReadWrite;
    bool bCpuAccess = true;

    NvSciBufAttrKeyValuePair bufAttrs[] = {
        { NvSciBufGeneralAttrKey_GpuId, &gpuId, sizeof(gpuId) },
        { NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType) },
        { NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, &bCpuAccess, sizeof(bCpuAccess) },
    };

    NvSciError sciErr = NvSciBufAttrListSetAttrs(*pBufAttrList, bufAttrs, ARRAY_SIZE(bufAttrs));
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

    // For opaque element type, color type and memory layout are decided by options.
    if (userType == PacketElementType::OPAQUE) {
        std::string &sImageLayout = m_cudaInputInfo.sImageLayout;

        // Check supported imagelayout.
        if (!sImageLayout.empty() && sImageLayout != "BL" && sImageLayout != "PL") {
            PLOG_ERR("Image layout type specified not support by cuda.\n"
                     "Supported Image layout type option: bl(default), pl.\n");
            return NvError_NotSupported;
        }

        // CUDA currently only support NV12 mapping.
        auto error = SetBufAttr(pBufAttrList, "NV12", sImageLayout, m_cudaInputInfo.uWidth, m_cudaInputInfo.uHeight);
        PCHK_ERROR_AND_RETURN(error, "SetBufAttr");
    }

    return NvError_Success;
}

NvError CCudaModule::FillSyncSignalerAttrList(CClientCommon *pClient,
                                              PacketElementType userType,
                                              NvSciSyncAttrList *pSignalerAttrList)
{
    auto cudaStatus = cudaDeviceGetNvSciSyncAttributes(*pSignalerAttrList, m_cudaDeviceId, cudaNvSciSyncAttrSignal);
    PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaDeviceGetNvSciSyncAttributes");

    return NvError_Success;
}

NvError CCudaModule::FillSyncWaiterAttrList(CClientCommon *pClient,
                                            PacketElementType userType,
                                            NvSciSyncAttrList *pWaiterAttrList)
{
    auto cudaStatus = cudaDeviceGetNvSciSyncAttributes(*pWaiterAttrList, m_cudaDeviceId, cudaNvSciSyncAttrWait);
    PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaDeviceGetNvSciSyncAttributes");
    return NvError_Success;
}

NvError
CCudaModule::RegisterSignalSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj signalSyncObj)
{
    cudaExternalSemaphoreHandleDesc extSemDescSig;
    memset(&extSemDescSig, 0, sizeof(extSemDescSig));
    extSemDescSig.type = cudaExternalSemaphoreHandleTypeNvSciSync;
    extSemDescSig.handle.nvSciSyncObj = signalSyncObj;
    auto cudaStatus = cudaImportExternalSemaphore(&m_signalerSem, &extSemDescSig);
    PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaImportExternalSemaphore");

    return NvError_Success;
}

NvError
CCudaModule::RegisterWaiterSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj waiterSyncObj)
{
    cudaExternalSemaphoreHandleDesc extSemDesc;
    memset(&extSemDesc, 0, sizeof(extSemDesc));
    extSemDesc.type = cudaExternalSemaphoreHandleTypeNvSciSync;
    extSemDesc.handle.nvSciSyncObj = waiterSyncObj;
    auto cudaStatus = cudaImportExternalSemaphore(&m_waiterSem, &extSemDesc);
    PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaImportExternalSemaphore");

    return NvError_Success;
}

//Before calling PreSync, m_nvmBuffers[uPacketIndex] should already be filled.
NvError CCudaModule::InsertPrefence(CClientCommon *pClient,
                                    PacketElementType userType,
                                    uint32_t uPacketIndex,
                                    NvSciSyncFence *pPrefence)
{
    if (m_bFirstCall) {
        size_t unused;
        auto cudaStatus = cudaDeviceGetLimit(&unused, cudaLimitStackSize);
        PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaDeviceGetLimit");

        cudaStatus = cudaSetDevice(m_cudaDeviceId);
        PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaSetDevice");
        m_bFirstCall = false;
    }

    cudaExternalSemaphoreWaitParams waitParams;
    memset(&waitParams, 0, sizeof(waitParams));
    waitParams.params.nvSciSync.fence = pPrefence;
    waitParams.flags = 0;
    auto cudaStatus = cudaWaitExternalSemaphoresAsync(&m_waiterSem, &waitParams, 1, m_streamWaiter);
    PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cuWaitExternalSemaphoresAsync");

    return NvError_Success;
}

NvError CCudaModule::BlToPlConvertWithGPU(uint32_t uPacketIndex, void *pDstPtr)
{
    uint8_t *pPlaneAddrs[2] = { nullptr };

    auto cudaStatus = cudaGetMipmappedArrayLevel(&m_mipLevelArray[uPacketIndex][0U], m_mipmapArray[uPacketIndex][0], 0U);
    PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaGetMipmappedArrayLevel");
    cudaStatus = cudaGetMipmappedArrayLevel(&m_mipLevelArray[uPacketIndex][1U], m_mipmapArray[uPacketIndex][1], 0U);
    PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaGetMipmappedArrayLevel");

    pPlaneAddrs[0] = (uint8_t *)&m_mipLevelArray[uPacketIndex][0U];
    pPlaneAddrs[1] = (uint8_t *)&m_mipLevelArray[uPacketIndex][1U];
    cudaStatus = cudaMemcpy2DFromArrayAsync(
        pDstPtr, (size_t)m_bufAttrs[uPacketIndex].planeWidths[0U], *((cudaArray_const_t *)pPlaneAddrs[0]), 0, 0,
        (size_t)m_bufAttrs[uPacketIndex].planeWidths[0U], (size_t)m_bufAttrs[uPacketIndex].planeHeights[0U],
        cudaMemcpyDeviceToDevice, m_streamWaiter);
    PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaMemcpy2DFromArrayAsync for plane 0");

    uint8_t *pSecondDst = (uint8_t *)pDstPtr + (size_t)(m_bufAttrs[uPacketIndex].planeWidths[0U] *
                                                        m_bufAttrs[uPacketIndex].planeHeights[0U]);
    cudaStatus = cudaMemcpy2DFromArrayAsync(
        (void *)(pSecondDst), (size_t)m_bufAttrs[uPacketIndex].planeWidths[0U], *((cudaArray_const_t *)pPlaneAddrs[1]),
        0, 0, (size_t)m_bufAttrs[uPacketIndex].planeWidths[0U], (size_t)(m_bufAttrs[uPacketIndex].planeHeights[0U] / 2),
        cudaMemcpyDeviceToDevice, m_streamWaiter);
    PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaMemcpy2DFromArrayAsync for plane 1");

    return NvError_Success;
}

NvError CCudaModule::BlToPlConvert(uint32_t uPacketIndex, void *pDstPtr)
{
    uint8_t *pPlaneAddrs[2] = { nullptr };

    auto cudaStatus =
        cudaGetMipmappedArrayLevel(&m_mipLevelArray[uPacketIndex][0U], m_mipmapArray[uPacketIndex][0], 0U);
    PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaGetMipmappedArrayLevel");
    cudaStatus = cudaGetMipmappedArrayLevel(&m_mipLevelArray[uPacketIndex][1U], m_mipmapArray[uPacketIndex][1], 0U);
    PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaGetMipmappedArrayLevel");

    pPlaneAddrs[0] = (uint8_t *)&m_mipLevelArray[uPacketIndex][0U];
    pPlaneAddrs[1] = (uint8_t *)&m_mipLevelArray[uPacketIndex][1U];
    cudaStatus = cudaMemcpy2DFromArrayAsync(
        pDstPtr, (size_t)m_bufAttrs[uPacketIndex].planeWidths[0U], *((cudaArray_const_t *)pPlaneAddrs[0]), 0, 0,
        (size_t)m_bufAttrs[uPacketIndex].planeWidths[0U], (size_t)m_bufAttrs[uPacketIndex].planeHeights[0U],
        cudaMemcpyDeviceToHost, m_streamWaiter);
    PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaMemcpy2DFromArrayAsync for plane 0");

    uint8_t *pSecondDst = (uint8_t *)pDstPtr + (size_t)(m_bufAttrs[uPacketIndex].planeWidths[0U] *
                                                        m_bufAttrs[uPacketIndex].planeHeights[0U]);
    cudaStatus = cudaMemcpy2DFromArrayAsync(
        (void *)(pSecondDst), (size_t)m_bufAttrs[uPacketIndex].planeWidths[0U], *((cudaArray_const_t *)pPlaneAddrs[1]),
        0, 0, (size_t)m_bufAttrs[uPacketIndex].planeWidths[0U], (size_t)(m_bufAttrs[uPacketIndex].planeHeights[0U] / 2),
        cudaMemcpyDeviceToHost, m_streamWaiter);
    PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaMemcpy2DFromArrayAsync for plane 1");

    return NvError_Success;
}

NvError CCudaModule::GetEofSyncFence(CClientCommon *pClient, NvSciSyncFence *pPostfence)
{
    cudaExternalSemaphoreSignalParams signalParams;
    memset(&signalParams, 0, sizeof(signalParams));
    signalParams.params.nvSciSync.fence = pPostfence;
    signalParams.flags = 0;
    auto cudaStatus = cudaSignalExternalSemaphoresAsync(&m_signalerSem, &signalParams, 1, m_streamWaiter);
    PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaSignalExternalSemaphoresAsync");

    return NvError_Success;
}

NvError CCudaModule::ProcessPayload(CClientCommon *pClient, uint32_t uPacketIndex)
{
    NvError error = NvError_Success;

    // For GPU, there is limitation that we need to make sure there no GPU tasks when switching to Operational state.
    // Add a flag for cuda to notify running until operational state.
    if (m_pAppCfg->IsStatusManagerEnabled() && !m_pAppCfg->IsCudaRunningEnabled()) {
        return error;
    }

    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler->RecordExecutionBeginTime();
    }

    const BufferAttrs &bufAttr = m_bufAttrs[uPacketIndex];
    if (bufAttr.layout == NvSciBufImage_BlockLinearType) {
// Linux and QNX standard
#if BUILD_CARDETECT
        m_pMetaData = reinterpret_cast<MetaData *>(pClient->GetMetaPtr(uPacketIndex));
        error = DoInference(uPacketIndex);
        // inference task error check.
        if ((error != NvError_Success) && (error != NvError_NotInitialized)) {
            return error;
        }
        if (error == NvError_NotInitialized)
#endif

// comment out in QNX temporarily
#ifndef NVMEDIA_QNX
        {
            error = NvError_Success;
            cudaArray_t vaddr_plane[2] = { nullptr };
            auto cudaStatus = cudaGetMipmappedArrayLevel(&vaddr_plane[0U], m_mipmapArray[uPacketIndex][0], 0U);
            PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaGetMipmappedArrayLevel");
            cudaStatus = cudaGetMipmappedArrayLevel(&vaddr_plane[1U], m_mipmapArray[uPacketIndex][1], 0U);
            PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaGetMipmappedArrayLevel");
            bool result = CvtNv12blToRgbPlanar(vaddr_plane, m_cudaInputInfo.uWidth, m_cudaInputInfo.uHeight,
                                               m_pCvtDevPtrs, m_cudaInputInfo.uCvtWidth,
                                               m_cudaInputInfo.uCvtHeight, 1, 1.0f, m_streamWaiter);
            if (!result) {
                PLOG_ERR("CvtNv12blToRgbPlanar failed");
                return NvError_ResourceError;
            }
        }
#endif
        error = BlToPlConvertWithGPU(uPacketIndex, output_buffer_);
        if (error != NvError_Success) {
            PLOG_ERR("BlToPlConvert failed: %u\n", error);
            m_uOutputBufValidLen = 0;
        } else {
            PLOG_DBG("ProcessPayload succeed.\n");
        }

        auto cudaStatus = cudaNV12ToRGB(output_buffer_, (uchar3*)m_pCvtDevPtrs, m_cudaInputInfo.uWidth, m_cudaInputInfo.uHeight, m_streamWaiter);
        PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaNV12ToRGB output_buffer_");

        cudaStreamSynchronize(m_streamWaiter);
        m_pAppCfg->CallCameraPlugin(GetSensorId(), time(NULL), (uint8_t *)m_pCvtDevPtrs, image_size_);

        if (m_upFileSink) {
            error = BlToPlConvert(uPacketIndex, (void *)m_upOutputBuf.get());
            if (error != NvError_Success) {
                PLOG_ERR("BlToPlConvert failed: %u\n", error);
                m_uOutputBufValidLen = 0;
            } else {
                PLOG_DBG("ProcessPayload succeed.\n");
                cudaStreamSynchronize(m_streamWaiter);
            }
        }
    } else if (bufAttr.layout == NvSciBufImage_PitchLinearType) {
        uint32_t uOffsetHost = 0U;
        uint32_t uOffsetDevice = 0U;
        for (uint32_t planeId = 0U; planeId < bufAttr.planeCount; ++planeId) {
            if (planeId > 0U) {
                uOffsetHost += bufAttr.planeHeights[planeId - 1U] * bufAttr.planeWidths[planeId - 1U];
                uOffsetDevice = bufAttr.planeOffsets[planeId];
            }

            auto cudaStatus = cudaMemcpy2DAsync(
                m_upOutputBuf.get() + uOffsetHost,
                bufAttr.planeWidths[planeId] * (bufAttr.planeBitsPerPixels[planeId] / 8U),
                (void *)((uint8_t *)m_pDevPtrs[uPacketIndex] + uOffsetDevice), bufAttr.planePitches[planeId],
                bufAttr.planeWidths[planeId] * (bufAttr.planeBitsPerPixels[planeId] / 8U),
                bufAttr.planeHeights[planeId], cudaMemcpyDeviceToHost, m_streamWaiter);
            if (cudaStatus != cudaSuccess) {
                PLOG_ERR("cudaMemcpy2DAsync failed: %u\n", cudaStatus);
                error = NvError_ResourceError;
                m_uOutputBufValidLen = 0;
                break;
            }
        }

    } else {
        PLOG_ERR("Unsupported layout\n");
        error = NvError_NotSupported;
    }

    if (NvError_Success == error) {
        cudaStreamSynchronize(m_streamWaiter);
    }

    if (m_pAppCfg->IsProfilingEnabled()) {
        m_pMetaData = reinterpret_cast<MetaData *>(pClient->GetMetaPtr(uPacketIndex));
    }
    return error;
}

NvError CCudaModule::OnProcessPayloadDone(CClientCommon *pClient, uint32_t uPacketIndex)
{
    NvError error = NvError_Success;

    error = CBaseModule::OnProcessPayloadDone(pClient, uPacketIndex);
    PCHK_ERROR_AND_RETURN(error, "CBaseModule::OnProcessPayloadDone");

    if (m_pAppCfg->IsProfilingEnabled()) {
        m_upProfiler->RecordExecutionEndTime();
        if (!m_bHasDownstream && m_pMetaData) {
            m_upProfiler->RecordPipelineTime(m_pMetaData->uFrameCaptureStartTSC);
            m_pMetaData = nullptr;
        }
    }
    return error;
}

NvError CCudaModule::RegisterBufObj(CClientCommon *pClient,
                                    PacketElementType userType,
                                    uint32_t uPacketIndex,
                                    NvSciBufObj bufObj)
{
    BufferAttrs bufAttrs;

    auto error = CBaseModule::RegisterBufObj(pClient, userType, uPacketIndex, bufObj);
    PCHK_ERROR_AND_RETURN(error, "CBaseModule::RegisterBufObj");

    error = PopulateBufAttr(bufObj, bufAttrs);
    PCHK_ERROR_AND_RETURN(error, "PopulateBufAttr");
    m_bufAttrs[uPacketIndex] = bufAttrs;

    cudaExternalMemoryHandleDesc memHandleDesc;
    memset(&memHandleDesc, 0, sizeof(memHandleDesc));
    memHandleDesc.type = cudaExternalMemoryHandleTypeNvSciBuf;
    memHandleDesc.handle.nvSciBufObject = bufObj;
    memHandleDesc.size = bufAttrs.size;
    auto cudaStatus = cudaImportExternalMemory(&m_extMem[uPacketIndex], &memHandleDesc);
    PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaImportExternalMemory");

    if (nullptr == m_upOutputBuf) {
        for (uint32_t i = 0U; i < bufAttrs.planeCount; ++i) {
            PLOG_DBG("plane id = %u, planeWidth = %d, planeHeight = %d, planePitch = %d, planeBitsPerPixels = %d, "
                     "planeOffsets = %lu\n",
                     i, bufAttrs.planeWidths[i], bufAttrs.planeHeights[i], bufAttrs.planePitches[i],
                     bufAttrs.planeBitsPerPixels[i], bufAttrs.planeOffsets[i]);
            m_uOutputBufValidLen += (size_t)bufAttrs.planeWidths[i] * (size_t)bufAttrs.planeHeights[i] *
                                    (bufAttrs.planeBitsPerPixels[i] / 8U);
        }
        m_uOutputBufCapacity = m_uOutputBufValidLen;
        m_upOutputBuf.reset(new (std::nothrow) uint8_t[m_uOutputBufCapacity]);
        if (nullptr == m_upOutputBuf) {
            PLOG_ERR("Out of memory\n");
            return NvError_InsufficientMemory;
        }
        cudaStatus = cudaMalloc(&output_buffer_, m_uOutputBufCapacity);
        PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaMalloc output_buffer_");
    }

    if (bufAttrs.layout == NvSciBufImage_BlockLinearType) {
        PLOG_DBG("RegisterBufObj, layout is blockLinear.\n");
        struct cudaExtent extent[MAX_NUM_SURFACES];
        struct cudaChannelFormatDesc desc[MAX_NUM_SURFACES];
        struct cudaExternalMemoryMipmappedArrayDesc mipmapDesc[MAX_NUM_SURFACES];
        (void *)memset(extent, 0, MAX_NUM_SURFACES * sizeof(struct cudaExtent));
        (void *)memset(desc, 0, MAX_NUM_SURFACES * sizeof(struct cudaChannelFormatDesc));
        (void *)memset(mipmapDesc, 0, MAX_NUM_SURFACES * sizeof(struct cudaExternalMemoryMipmappedArrayDesc));
        for (uint32_t planeId = 0; planeId < bufAttrs.planeCount; planeId++) {
            /* Setting for each plane buffer
             * SP format has 2 planes
             * Planar format has 3 planes  */
            if ((bufAttrs.planeColorFormats[planeId] == NvSciColor_Y8) ||
                (bufAttrs.planeColorFormats[planeId] == NvSciColor_U8V8) ||
                (bufAttrs.planeColorFormats[planeId] == NvSciColor_U8_V8) ||
                (bufAttrs.planeColorFormats[planeId] == NvSciColor_V8U8) ||
                (bufAttrs.planeColorFormats[planeId] == NvSciColor_V8_U8)) {
                // only support NV12 now
                extent[planeId].width = bufAttrs.planePitches[planeId];
                extent[planeId].height = bufAttrs.planeAlignedHeights[planeId];
                extent[planeId].depth = 0;
                desc[planeId] = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
                mipmapDesc[planeId].offset = bufAttrs.planeOffsets[planeId];
                mipmapDesc[planeId].formatDesc = desc[planeId];
                mipmapDesc[planeId].extent = extent[planeId];
                mipmapDesc[planeId].flags = 0;
                mipmapDesc[planeId].numLevels = 1;
                cudaStatus = cudaExternalMemoryGetMappedMipmappedArray(&m_mipmapArray[uPacketIndex][planeId],
                                                                       m_extMem[uPacketIndex], &mipmapDesc[planeId]);
                PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaExternalMemoryGetMappedMipmappedArray");
            } else {
                PLOG_ERR("Unsupported blockLinear format\n");
                return NvError_NotSupported;
            }
        } /* end for */
    } else if (bufAttrs.layout == NvSciBufImage_PitchLinearType) {
        PLOG_INFO("Expected layout\n");
        // Map in the buffer as CUDA device memory
        struct cudaExternalMemoryBufferDesc memBufferDesc;
        memset(&memBufferDesc, 0, sizeof(memBufferDesc));
        memBufferDesc.size = bufAttrs.size;
        memBufferDesc.offset = 0;
        auto cudaStatus =
            cudaExternalMemoryGetMappedBuffer(&m_pDevPtrs[uPacketIndex], m_extMem[uPacketIndex], &memBufferDesc);
        PCHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaImportExternalMemory");

    } else {
        PLOG_ERR("Unsupported layout\n");
        return NvError_NotSupported;
    }

    return NvError_Success;
}

const OptionTable *CCudaModule::GetOptionTable() const
{
    return &m_cudaOptionTable;
}

const void *CCudaModule::GetOptionBaseAddress() const
{
    return &m_cudaInputInfo;
}

const std::string &CCudaModule::GetOutputFileName()
{
    if (m_sOutputFileName != "") {
        PLOG_DBG("This CCudaModule's OutputFileName already exists: %s\n", m_sOutputFileName.c_str());
        return m_sOutputFileName;
    }
    std::string suffix = m_spModuleCfg->m_sensorId != INVALID_ID ? std::to_string(m_spModuleCfg->m_sensorId)
                                                                 : std::to_string(m_spModuleCfg->m_moduleId);
    m_sOutputFileName = "multicast_cuda" + suffix + ".yuv";
    PLOG_DBG("This CCudaModule's OutputName: %s\n", m_sOutputFileName.c_str());

    return m_sOutputFileName;
}

#if BUILD_CARDETECT
// Linux and QNX standard
NvError CCudaModule::DoInference(uint32_t uPacketIndex)
{
    NvError error = NvError_Success;

    if (m_bufAttrs[uPacketIndex].layout == NvSciBufImage_BlockLinearType) {
        cudaArray_t vaddr_plane[2] = { nullptr };
        auto cudaStatus = cudaGetMipmappedArrayLevel(&vaddr_plane[0U], m_mipmapArray[uPacketIndex][0], 0U);
        CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaGetMipmappedArrayLevel");
        cudaStatus = cudaGetMipmappedArrayLevel(&vaddr_plane[1U], m_mipmapArray[uPacketIndex][1], 0U);
        CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaGetMipmappedArrayLevel");

        PLOG_INFO("m_upCarDetect->Process, width: %u, height: %u\n", m_bufAttrs[uPacketIndex].planeWidths[0],
                  m_bufAttrs[uPacketIndex].planeHeights[0]);
        // when inference task is not initial successfully, it will return not initialed
        std::vector<NvInferObject> vBoundingBox;
        auto detectResult =
            m_upCarDetect->Process(vaddr_plane, m_bufAttrs[uPacketIndex].planeWidths[0],
                                   m_bufAttrs[uPacketIndex].planeHeights[0], vBoundingBox, m_cudaInputInfo.bDraw);

        if (m_pMetaData) {
            if (vBoundingBox.size() > MetaData::kMaxROIRegions) {
                PLOG_ERR("Error: ROI num %d exceeds %d\n", vBoundingBox.size(), MetaData::kMaxROIRegions);
                return NvError_BadParameter;
            }
            m_pMetaData->uNumROIRegions = vBoundingBox.size();
            int i = 0;
            for (const auto &box : vBoundingBox) {
                m_pMetaData->ROIRect[i].x0 = box.left;
                m_pMetaData->ROIRect[i].y0 = box.top;
                m_pMetaData->ROIRect[i].x1 = box.left + box.width;
                m_pMetaData->ROIRect[i].y1 = box.top + box.height;
                i++;
            }
        }
        PLOG_DBG("DoInference end.\n");
        if (detectResult == CCarDetector::DetectResult::CAR_DETECT_NOT_INITIALIZED) {
            return NvError_NotInitialized;
        }
        error =
            (detectResult == CCarDetector::DetectResult::CAR_DETECT_SUCCESS) ? NvError_Success : NvError_ResourceError;
    } else {
        PLOG_ERR("Unsupported Inference layout\n");
        return NvError_NotSupported;
    }

    return error;
}
#endif
