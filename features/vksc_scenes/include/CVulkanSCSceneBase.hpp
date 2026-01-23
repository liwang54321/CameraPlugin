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

#ifndef CVULKANSC_SCENE_BASE_HPP
#define CVULKANSC_SCENE_BASE_HPP

#include "CUtils.hpp"
#ifdef VULKAN
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>
#else
#define VK_USE_PLATFORM_SCI
#include <vulkan/vulkan_sc.h>
#endif // #ifdef VULKAN

#define DEFAULT_FENCE_TIMEOUT 1000000000U
class CVulkanSCSceneBase
{
 public:
    CVulkanSCSceneBase(VkDevice& dev, VkPhysicalDevice &physDev, VkQueue &queue,
        VkCommandPool& cmdPool, VkFormat &inputImageFormat, uint32_t uWidth, uint32_t uHeight
#ifndef VULKAN
        , VkPipelineCache &pipelineCache
#endif
        ):
        m_dev(dev), m_physDev(physDev), m_queue(queue), m_cmdPool(cmdPool),
        m_inputImageFormat(inputImageFormat), m_uWidth(uWidth), m_uHeight(uHeight)
#ifndef VULKAN
        , m_pipelineCache(pipelineCache)
#endif
        {}

    virtual ~CVulkanSCSceneBase();

    virtual NvError Init();
    virtual NvError RecordSceneDrawCommand(VkCommandBuffer &vkCmdBuffer, uint32_t uPacketIndex, void *pSceneData) = 0;
    NvError CreateFramebuffer(VkImageView *attachments,uint32_t uNumAttachment, uint32_t uPacketIndex);

 protected:
    virtual NvError CreateVertexBuffer() = 0;
    virtual NvError SetupPipeline() = 0;
    virtual NvError LoadTextures()  = 0;
    virtual NvError SetupRenderPass() = 0;
    virtual NvError CreateDescriptorSets() = 0;
    virtual std::string GetVertexShaderPath() = 0;
    virtual std::string GetFragmentShaderPath() = 0;
    virtual NvError CreateUniformBuffer() { return NvError_Success; }
    virtual NvError CreateIndexBuffer() { return NvError_Success; }
    virtual NvError BeginSingleTimeCommands(VkCommandBuffer &cmdBuffer);
    virtual NvError EndSingleTimeCommands(VkCommandBuffer &cmdBuffer);
    virtual NvError CreateShaderStageInfo();

    NvError CreateVkImage(uint32_t uWidth, uint32_t uHeight, VkFormat format, VkImageTiling tiling,
        VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
    NvError CreateVkBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
    VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    NvError RecordImageLayoutTransitionCmd(VkCommandBuffer &cmdBuffer, VkImage &image,
    VkImageLayout oldLayout, VkImageLayout newLayout);

    VkDevice         m_dev     = VK_NULL_HANDLE;
    VkPhysicalDevice m_physDev = VK_NULL_HANDLE;
    VkQueue          m_queue   = VK_NULL_HANDLE;
    VkCommandPool m_cmdPool    = VK_NULL_HANDLE;
    VkRenderPass  m_renderPass = VK_NULL_HANDLE;
    VkPipeline    m_pipeline   = VK_NULL_HANDLE;
    VkFormat m_inputImageFormat = VK_FORMAT_UNDEFINED;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkFramebuffer m_vkFrameBuffer[MAX_NUM_PACKETS] = {VK_NULL_HANDLE};

    VkPhysicalDeviceMemoryProperties m_memProperties = {};
    VkPipelineShaderStageCreateInfo m_stageInfo[2] = { { }, { } };

    uint32_t m_uWidth  = 3840;
    uint32_t m_uHeight = 2160;

#ifndef VULKAN
    NvError ReadPipelineUUID(const std::string& sFileName, char cUUID[VK_UUID_SIZE]);
    VkPipelineCache m_pipelineCache = VK_NULL_HANDLE;
#else
    VkShaderModule loadSPIRVShader(std::string &sFilename);
#endif

};

VkFormat FindDepthFormat(VkPhysicalDevice physDev, VkImageTiling tiling);
uint32_t GetMemoryType(VkPhysicalDeviceMemoryProperties &memProperties,
    uint32_t uTypeBits, const VkFlags &properties);

std::vector<std::vector<float>> GetPerspectiveMat(float fovy, float fAspect, float fzNear, float fzFar);
std::vector<std::vector<float>> MatMul(std::vector<std::vector<float>> &vfLMat, std::vector<std::vector<float>> &vfRMat);
std::vector<std::vector<float>> Rotate(std::vector<std::vector<float>> &vfMat, float fAngle, std::vector<float> &v);

#endif // CVULKANSC_SCENE_BASE_HPP
