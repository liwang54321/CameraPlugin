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

#ifndef CVULKANSCENGINE_HPP
#define CVULKANSCENGINE_HPP

#ifdef VULKAN
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>
#include <fstream>
#else
#define VK_USE_PLATFORM_SCI
#include <vulkan/vulkan_sc.h>
#include <vulkan/vulkan_sci.h>
#endif // #ifdef VULKAN

#include "CFontVulkanSC.hpp"
#include "Common.hpp"
#include "CClientCommon.hpp"
#include "CUtils.hpp"

#ifndef VULKAN
#define GET_DEVICE_PROC_ADDR(entrypoint)                                                                   \
    {                                                                                                      \
        if (!m_gdpa)                                                                                       \
            m_gdpa = (PFN_vkGetDeviceProcAddr)vkGetInstanceProcAddr(m_instance, "vkGetDeviceProcAddr");    \
        m_vkscSciInterface.fp##entrypoint = (PFN_vk##entrypoint)m_gdpa(m_dev, "vk" #entrypoint);           \
        if (m_vkscSciInterface.fp##entrypoint == NULL) {                                                   \
            LOG_ERR("vkGetDeviceProcAddr failed to find vk" #entrypoint, "vkGetDeviceProcAddr Failure\n"); \
        }                                                                                                  \
    }

#define GET_INSTANCE_DEVICE_PROC_ADDR(entrypoint)                                                                    \
    {                                                                                                                \
        m_vkscSciInterface.fp##entrypoint = (PFN_vk##entrypoint)vkGetInstanceProcAddr(m_instance, "vk" #entrypoint); \
        if (m_vkscSciInterface.fp##entrypoint == NULL) {                                                             \
            LOG_ERR("vkGetInstanceProcAddr failed to find vk" #entrypoint, "vkGetInstanceProcAddr Failure\n");       \
        }                                                                                                            \
    }

typedef struct
{
    PFN_vkGetPhysicalDeviceSciBufAttributesNV fpGetPhysicalDeviceSciBufAttributesNV;
    PFN_vkGetPhysicalDeviceExternalMemorySciBufPropertiesNV fpGetPhysicalDeviceExternalMemorySciBufPropertiesNV;
    PFN_vkGetPhysicalDeviceSciSyncAttributesNV fpGetPhysicalDeviceSciSyncAttributesNV;
    PFN_vkGetSemaphoreSciSyncObjNV fpGetSemaphoreSciSyncObjNV;
    PFN_vkImportSemaphoreSciSyncObjNV fpImportSemaphoreSciSyncObjNV;
    PFN_vkCreateSemaphoreSciSyncPoolNV fpCreateSemaphoreSciSyncPoolNV;
    PFN_vkGetFenceSciSyncFenceNV fpGetFenceSciSyncFenceNV;
    PFN_vkGetFenceSciSyncObjNV fpGetFenceSciSyncObjNV;
    PFN_vkImportFenceSciSyncFenceNV fpImportFenceSciSyncFenceNV;
    PFN_vkImportFenceSciSyncObjNV fpImportFenceSciSyncObjNV;
} VkSCSciInterface;
#endif // #ifndef VULKAN

enum class VKSCSyncType : uint8_t
{
    VulkanSCSignaler,
    VulkanSCWaiter,
};

class CVulkanSCEngine
{
  public:
    NvError Init(uint32_t uWidth, uint32_t uHeight, bool bUseVkSemaphore, std::string &sColorType);
    NvError Draw(uint32_t uPacketIndex, std::string &sTimeStampToDraw);
    void DeInit();
    NvError GetSciBufAttributesNV(NvSciBufAttrList *pBufAttrList);
    NvError GetSciSyncAttributesNV(VKSCSyncType vkscSyncType, NvSciSyncAttrList *pSignalerAttrList);
    NvError RegisterBufObj(uint32_t uPacketIndex, NvSciBufObj bufObj);
    NvError RegisterSyncObj(VKSCSyncType vkscSyncType, NvSciSyncObj bufObj);
    NvError InsertFenceSciSyncPrefence(NvSciSyncFence *pPrefence);
    NvError GetEofFenceSciSyncFence(NvSciSyncFence *pPostfence);

    //for image dump
    uint32_t GetOutputImageSize() { return m_dstMemRequirements.size; }
    NvError DumpImage(uint32_t uPacketIndex, uint8_t *pDst);

  private:
    NvError CreateInstanceAndDevice();
    NvError SetupImage();
    NvError CreateCommandBuffers();
    NvError FindQueueFamilies(VkPhysicalDevice &device);
    VkFormat ColorToVkFormat(std::string &sColorType);

#ifndef VULKAN
    NvError LoadPipelineCaches();
    NvError CreatePipelineCaches();
    NvError FillDeviceObjResInfo();
    NvError GetSemaphoreFromVkSciSyncObj(
    NvSciSyncFence* pSciSyncFence,
    VkSemaphore& semaphore,
    VkSemaphoreSciSyncPoolNV& semaphorePool);

    VkSCSciInterface m_vkscSciInterface = {};

    VkPipelineCacheCreateInfo          m_pipelineCacheCreateInfo = {};
    VkDeviceObjectReservationCreateInfo m_devObjectResCreateInfo = {};

    VkDeviceSemaphoreSciSyncPoolReservationCreateInfoNV m_devSemaphoreSciSyncPoolResCreateInfo = {};

    VkSemaphoreSciSyncPoolNV m_semaphoreSciSyncSignalerPool = VK_NULL_HANDLE;
    VkSemaphoreSciSyncPoolNV m_semaphoreSciSyncWaiterPool = VK_NULL_HANDLE;

    VkPipelineCache m_pipelineCache = VK_NULL_HANDLE;
    PFN_vkGetDeviceProcAddr m_gdpa = nullptr;
    uint64_t m_uThresholdVal = 0;

    VkPipelinePoolSize m_pipelinePoolSizes = {VK_STRUCTURE_TYPE_PIPELINE_POOL_SIZE, nullptr, 1024U * 1024U, 32};
#else
    NvError CheckJsonLayerSupport();
#endif // #ifdef VULKAN

    VkInstance       m_instance   = VK_NULL_HANDLE;
    VkPhysicalDevice m_physDev    = VK_NULL_HANDLE;
    VkDevice         m_dev        = VK_NULL_HANDLE;
    VkQueue          m_queue      = VK_NULL_HANDLE;
    VkCommandPool    m_cmdPool    = VK_NULL_HANDLE;
    VkImage          m_dstImage   = VK_NULL_HANDLE;
    VkDeviceMemory   m_dstDevMem  = VK_NULL_HANDLE;
    VkFormat m_inputImageFormat = VK_FORMAT_UNDEFINED;

    VkDeviceMemory  m_vkImportImageMem[MAX_NUM_PACKETS]  = {VK_NULL_HANDLE};
    VkImage         m_vkImportImage[MAX_NUM_PACKETS]     = {VK_NULL_HANDLE};
    VkImageView     m_vkImportImageView[MAX_NUM_PACKETS] = {VK_NULL_HANDLE};
    VkImage         m_vkDepthImage[MAX_NUM_PACKETS]      = {VK_NULL_HANDLE};
    VkDeviceMemory  m_vkDepthDevMem[MAX_NUM_PACKETS]     = {VK_NULL_HANDLE};
    VkImageView     m_vkDepthImageView[MAX_NUM_PACKETS]  = {VK_NULL_HANDLE};

    VkCommandBuffer m_vkCopyCmdBuffer = VK_NULL_HANDLE;
    VkCommandBuffer m_vkCmdBuffer = VK_NULL_HANDLE;

    VkSemaphore m_vkSignalerSem = VK_NULL_HANDLE;
    VkSemaphore m_vkWaiterSem = VK_NULL_HANDLE;
    VkFence m_vkSignalFence = VK_NULL_HANDLE;
    VkFence m_vkWaiterFence = VK_NULL_HANDLE;

    VkPhysicalDeviceMemoryProperties m_memProperties      = {};
    VkPhysicalDeviceProperties       m_deviceProperties   = {};
    VkMemoryRequirements             m_memRequirements    = {};
    VkMemoryRequirements             m_dstMemRequirements = {};

    VkTimelineSemaphoreSubmitInfo m_semaphoreInfo = {};

    uint32_t m_uWidth = 3840;
    uint32_t m_uHeight = 2160;
    uint64_t m_uWaitVal = 0;
    long m_lStartCacheSize = 0;
    char *m_pStartCacheData = nullptr;
    uint32_t m_uQueueFamilyIndex = 0;
    bool m_bUseVkSemaphore = false;
    std::string m_sOutputFileName = "";

    NvSciSyncObj m_signalSyncObj = nullptr;
    NvSciSyncFence m_postfence = NvSciSyncFenceInitializer;

    std::vector<std::shared_ptr<CVulkanSCSceneBase>> m_vVulkanSCSences;
};

#endif // CVULKANSCENGINE_HPP
