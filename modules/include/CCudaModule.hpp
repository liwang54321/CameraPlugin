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

#ifndef CCUDAMODULE_H
#define CCUDAMODULE_H

#include "CBaseModule.hpp"
#include "NvSIPLDeviceBlockInfo.hpp"

#if BUILD_CARDETECT
// Linux
#include "CCarDetector.hpp"
#endif

// cuda includes
#include "cuda.h"
#include "cuda_runtime_api.h"

typedef struct
{
    uint32_t uWidth = 3840U;
    uint32_t uHeight = 2160U;
    std::string sImageLayout = "";
    bool bUsePva = false;
    bool bDraw = true;
    uint32_t uCvtWidth  = 640U;
    uint32_t uCvtHeight = 360U;
} CUDAInputInfo;

class CCudaModule : public CBaseModule
{
  public:
    CCudaModule(std::shared_ptr<CModuleCfg> spModuleCfg, IEventListener<CBaseModule> *pListener);
    virtual ~CCudaModule() {};

    virtual NvError Init() override;
    virtual void DeInit() override;
    virtual NvError
    FillDataBufAttrList(CClientCommon *pClient, PacketElementType userType, NvSciBufAttrList *pBufAttrList) override;
    virtual NvError FillSyncSignalerAttrList(CClientCommon *pClient,
                                             PacketElementType userType,
                                             NvSciSyncAttrList *pSignalerAttrList) override;
    virtual NvError FillSyncWaiterAttrList(CClientCommon *pClient,
                                           PacketElementType userType,
                                           NvSciSyncAttrList *pWaiterAttrList) override;
    virtual NvError RegisterBufObj(CClientCommon *pClient,
                                   PacketElementType userType,
                                   uint32_t uPacketIndex,
                                   NvSciBufObj bufObj) override;
    virtual NvError
    RegisterSignalSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj signalSyncObj) override;
    virtual NvError
    RegisterWaiterSyncObj(CClientCommon *pClient, PacketElementType userType, NvSciSyncObj waiterSyncObj) override;
    virtual NvError InsertPrefence(CClientCommon *pClient,
                                   PacketElementType userType,
                                   uint32_t uPacketIndex,
                                   NvSciSyncFence *pPrefence) override;
    virtual NvError GetEofSyncFence(CClientCommon *pClient, NvSciSyncFence *pPostfence) override;
    virtual NvError ProcessPayload(CClientCommon *pClient, uint32_t uPacketIndex) override;
    virtual NvError OnProcessPayloadDone(CClientCommon *pClient, uint32_t uPacketIndex) override;
    static NvError GetBufAttrList(NvSciBufAttrList bufAttrList);
    static NvError GetSyncWaiterAttrList(NvSciSyncAttrList outWaiterAttrList);

    virtual const OptionTable *GetOptionTable() const override;
    virtual const void *GetOptionBaseAddress() const override;

  protected:
    virtual const std::string &GetOutputFileName() override;

  private:
    NvError InitCuda();
    NvError BlToPlConvert(uint32_t uPacketIndex, void *pDstptr);
    NvError BlToPlConvertWithGPU(uint32_t uPacketIndex, void *pDstPtr);

    static int m_cudaDeviceId;

    cudaStream_t m_streamWaiter = nullptr;
    cudaExternalSemaphore_t m_signalerSem;
    cudaExternalSemaphore_t m_waiterSem;
    BufferAttrs m_bufAttrs[MAX_NUM_PACKETS] = {};
    cudaExternalMemory_t m_extMem[MAX_NUM_PACKETS];
    cudaMipmappedArray_t m_mipmapArray[MAX_NUM_PACKETS][MAX_NUM_SURFACES] = {};
    cudaArray_t m_mipLevelArray[MAX_NUM_PACKETS][MAX_NUM_SURFACES] = {};
    void *m_pDevPtrs[MAX_NUM_PACKETS];

    bool m_bFirstCall;

    static const std::unordered_map<std::string, Option> m_cudaOptionTable;
    CUDAInputInfo m_cudaInputInfo;
    void *m_pCvtDevPtrs;
    void *output_buffer_;
    uint32_t image_size_ = 0;
#if BUILD_CARDETECT
    // Only support Linux and QNX standard
    NvError DoInference(uint32_t uPacketIndex); // use for block linear
    std::unique_ptr<CCarDetector> m_upCarDetect = nullptr;
#endif
};
#endif
