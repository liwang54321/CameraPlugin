/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef COMMON_HPP
#define COMMON_HPP

#include <cstdint>
#include <string>

constexpr uint32_t MAX_NUM_SENSORS = 20U;
constexpr uint32_t MAX_OUTPUTS_PER_SENSOR = 4U;
constexpr uint32_t MAX_NUM_PACKETS = 6U;
constexpr uint32_t MAX_NUM_ELEMENTS = 8U;
constexpr uint32_t NUM_IPC_CONSUMERS = 2U;
constexpr uint32_t MAX_NUM_CONSUMERS = 8U;
constexpr uint32_t MAX_WAIT_SYNCOBJ = MAX_NUM_CONSUMERS;
constexpr uint32_t MAX_NUM_CASCADED_MODULES = 4U;
constexpr uint32_t MAX_NUM_SYNCS = 8U;
constexpr uint32_t MAX_QUERY_TIMEOUTS = 10U;
constexpr int QUERY_TIMEOUT = 1000000; // usecs
constexpr int QUERY_TIMEOUT_FOREVER = -1;
constexpr int INVALID_ID = -1;
constexpr uint32_t NVMEDIA_IMAGE_STATUS_TIMEOUT_MS = 100U;
constexpr uint32_t DUMP_START_FRAME = 30U;
constexpr uint32_t DUMP_END_FRAME = 59U;
constexpr int64_t FENCE_FRAME_TIMEOUT_US = 100000U;
#if NV_BUILD_DOS7
constexpr uint32_t MAX_NUM_WFD_PORTS = 4U;
#else
constexpr uint32_t MAX_NUM_WFD_PORTS = 2U;
#endif
constexpr uint32_t MAX_NUM_WFD_PIPELINES = 2U;
constexpr uint32_t MAX_NUM_PERF_SAMPLES = 10000U;
/* Maximum number of frame_data_buffer */
constexpr uint32_t MAX_FRAME_DATA_BUF_NUM = 6u;
/* Maximum number of planes in a surface */
constexpr uint32_t MAX_PLANE_COUNT = 3U;
constexpr uint32_t PLANE_BITS_PER_PIXEL = 8u;
constexpr uint32_t BITS_PER_BYTE = 8u;
/* Max number of decoder reference buffers */
constexpr uint32_t MAX_DEC_REF_BUFFERS = 16u;
/* Total number of buffers for decoder to operate.*/
constexpr uint32_t MAX_DEC_BUFFERS = (MAX_DEC_REF_BUFFERS + 1);
constexpr uint32_t IDE_APP_BASE_ADDR_ALIGN = 256u;
/* NvMediaIDE only supports input surface formats which have 2 planes */
constexpr uint32_t IDE_APP_MAX_INPUT_PLANE_COUNT = 2u;
constexpr uint32_t UPDATE_NVM_SURFACE = 16u;
constexpr uint32_t IDE_CODEC_STREAM_READ_SIZE = (32 * 1024);
constexpr uint32_t ITERATIONS_TILL_TIMEOUT = 150u;
constexpr uint32_t SLEEP_FOR_PACKET_MS = 10u;

constexpr const char *IPC_CHANNEL_PREFIX = "nvscistream_";
constexpr const char *C2C_SRC_CHANNEL_PREFIX = "nvscic2c_pcie_s0_c5_";

#if NV_BUILD_DOS7
constexpr const char *C2C_DST_CHANNEL_PREFIX = "nvscic2c_pcie_s1_c5_";
#else
constexpr const char *C2C_DST_CHANNEL_PREFIX = "nvscic2c_pcie_s0_c6_";
#endif

constexpr const char *CAMERA_DIR_PREFIX = "cam";
constexpr const char *YUV_FILE_PREFIX = "frame_cam";
constexpr const char *YUV_SEQUENCE_FILE_PREFIX = "frames_cam";
constexpr const char *CODEC_FILE_PREFIX = "frames_cam";

enum class CommType : uint8_t
{
    IntraProcess = 0,
    InterProcess,
    InterChip
};

enum class ModuleType : int8_t
{
    Unknown = -1,
    Enc = 0,
    Cuda,
    Stitch,
    Display,
    Nvm2d,
    SIPL,
    VulkanSC,
    VirtualSrc,
    VirtualDst,
    FileSource,
    Pva,
};

enum class PipelineType : uint8_t
{
    NormalPipeline = 0,
    SentryPipelineProducer,
    SentryPipelineConsumer
};

enum class PipelineElemType : uint8_t
{
    Enc = 0,
    Cuda,
    Stitch,
    Display,
    Nvm2d,
    SIPL,
    VulkanSC,
    VirtualSrc,
    VirtualDst,
    FileSource,
    Pva,

    IpcSrc,
    IpcDst,
};

enum class QueueType : uint8_t
{
    Mailbox = 0,
    Fifo
};

enum class EventType : uint8_t
{
    NVSCISTREAM_EVENT = 0,
    NVSIPL_STATUS,
    CHANNEL_EVENT,
};

enum class EventStatus : uint8_t
{
    OK = 0,
    RECONCILED,
    DISCONNECT,
    CONNECTED,
    STARTED,
    STOPPED,
    QUITTED,
    TIMED_OUT,
    ERROR,
};

enum class CMDType : uint8_t
{
    /* channel */
    START = 0,
    STOP,
    QUIT,
    ATTACH,
    DETACH,
    ENTER_LOW_POWER_MODE,
    ENTER_FULL_POWER_MODE,
    MAX,
};

enum class EncoderType : uint8_t
{
    H264 = 0,
    H265,
};

/* The type of source file from local filesystem */
enum class FileSourceType : uint8_t
{
    UNDEFINED = 0,
    YUV420P_SINGLE_FRAME,
    YUV420P_SEQUENCE,
    H264,
    H265,
};

/* The buffer definition of a image frame data */
typedef struct
{
    void *pBuffer;
    uint64_t uSize;
    uint32_t uPlaneCount;
    void *pPlanePtrs[MAX_PLANE_COUNT];
    uint32_t uPlaneSizes[MAX_PLANE_COUNT];
    uint32_t uPlanePitches[MAX_PLANE_COUNT];
} FrameDataBuffer;

#endif
