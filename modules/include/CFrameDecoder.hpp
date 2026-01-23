/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef CFRAME_DECODER_H
#define CFRAME_DECODER_H

#if !NV_IS_SAFETY

#include <list>
#include <mutex>
#include <deque>
#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>
#include <condition_variable>
#include <stdio.h>
#include "nvmedia_parser.h"
#include "nvmedia_ide.h"
#include "Common.hpp"
#include "CEventHandler.hpp"
#include "CFrameHandler.hpp"

typedef struct _FrameBuffer
{
    int refCount;
    int width;
    int height;
    int frameNum;
    int index;
    NvSciBufObj videoSurface;
    bool progressiveFrameFlag;
    bool topFieldFirstFlag;
    // used for specifying crop rectange information
    int lDARWidth;
    int lDARHeight;
    int displayLeftOffset;
    int displayTopOffset;
    int displayWidth;
    int displayHeight;
    NvSciSyncFence preFence;
} FrameBuffer;

typedef struct _ParserClientCtx
{
    NvMediaParser *pParser;
    NvMediaParserParams nvmParserParams;
    NvMediaVideoCodec eCodec;

    // Decoder params
    int decodeWidth;
    int decodeHeight;
    int displayWidth;
    int displayHeight;
    uint32_t uMaxBitstreamSize;
    std::atomic<bool> bParamsHasUpdated{ false };
    std::mutex updatedMutex;
    std::condition_variable updatedCond;
    NvMediaIDE *pDecoder;
    int decodeCount;

    // Picture buffer params
    uint32_t uBuffers;
    int picNum;
    FrameBuffer frameBufs[MAX_DEC_BUFFERS];

    NvMediaDecoderInstanceId instanceId;
    int frameCount;

    NvSciSyncObj eofSyncObj;

    std::mutex freeBufMutex;
    std::mutex validBufMutex;
    std::condition_variable validBufCond;
    std::list<NvSciBufObj> freeBufferList;
    std::deque<NvSciBufObj> validBufferQueue;
    std::unordered_map<NvSciBufObj, NvSciBufObj> matchBufMap;
    std::unordered_map<NvSciBufObj, NvSciSyncFence *> buf2FenceMap;

    std::atomic<bool> bStopDecoding{ false };
    std::atomic<bool> bQuitDecoding{ false };

    std::mutex decodingReadyMutex;
    std::condition_variable decodingReadyCond;
    std::atomic<bool> bDecodingReady{ false };
} ParserClientCtx;

class CFrameDecoder : public CFrameHandler
{
  public:
    CFrameDecoder(FileSourceType type,
                  const std::string &sFilePath,
                  uint32_t uWidth,
                  uint32_t uHeight,
                  int sensorId,
                  uint32_t uInstanceId);

    virtual ~CFrameDecoder();

    virtual NvError Init() override;

    virtual void DeInit() override;

    virtual NvError Start() override;

    virtual void Stop() override;

    virtual NvError FillNvSciBufAttrList(NvSciBufAttrList &bufAttrList) override;

    virtual NvError FillSyncSignalerAttrList(NvSciSyncAttrList &signalerAttrList) override;

    virtual NvError FillSyncWaiterAttrList(NvSciSyncAttrList &waiterAttrList) override;

    virtual NvError RegisterNvSciBuf(NvSciBufObj &bufObj) override;

    virtual NvError RegisterSignalSyncObj(NvSciSyncObj &signalSyncObj) override;

    virtual EventStatus LoadFrameData(NvSciBufObj &bufObj, NvSciSyncFence *&pPostFence) override;

    virtual void ReturnBuffer(NvSciBufObj &bufObj) override;

    inline const std::string &GetName() { return m_name; }

  private:
    NvError InitDecoder();

    void AcquireFreeBuffers();

    /*Parser Parse Related Callbacks*/
    static int32_t cbBeginSequence(void *ptr, const NvMediaParserSeqInfo *pnvsi);

    static NvMediaStatus cbDecodePicture(void *ptr, NvMediaParserPictureData *pd);

    static NvMediaStatus cbDisplayPicture(void *ptr, NvMediaRefSurface *p, int64_t llPts);

    static NvMediaStatus cbAllocPictureBuffer(void *ptr, NvMediaRefSurface **p);

    static void cbRelease(void *ptr, NvMediaRefSurface *p);

    static void cbAddRef(void *ptr, NvMediaRefSurface *p);

    static NvMediaStatus UpdateNvMediaSurfacePictureInfoH264(ParserClientCtx *mCtx,
                                                             NvMediaPictureInfoH264 *pictureInfo);

    static NvMediaStatus UpdateNvMediaSurfacePictureInfoH265(ParserClientCtx *mCtx,
                                                             NvMediaPictureInfoH265 *pictureInfo);

    EventStatus Decode();

  private:
    uint32_t m_uInstanceId;
    NvMediaParserClientCb m_clientCb;
    ParserClientCtx m_clientCtx;
    FILE *m_inputCodecFile{ nullptr };
    std::unique_ptr<CEventHandler<CFrameDecoder>> m_upDecodeHandler;
    uint8_t *m_codecStreamBuf{ nullptr };
    std::string m_name{ "FrameDecoder" };
};

#endif // !NV_IS_SAFETY

#endif // CFRAME_DECODER_H
