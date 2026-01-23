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

#ifndef CFONT_VULKANSC_HPP
#define CFONT_VULKANSC_HPP

#include <unordered_map>

#include "CUtils.hpp"
#ifdef VULKAN
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>
#else
#define VK_USE_PLATFORM_SCI
#include <vulkan/vulkan_sc.h>
#endif // #ifdef VULKAN

#include "CVulkanSCSceneBase.hpp"

// Default fence timeout in nanoseconds
#define TSC_NUMS 12U
constexpr uint32_t FONT_MAX_BUFFER_SIZE = 8U * 1024U * 1024U; // 8M

typedef struct
{
    uint32_t uBufferSize;
    unsigned char *pBuffer;
    int ixOffset;
    int iyOffset;
    unsigned int uWidth;
    unsigned int uHeight;
    long lAdvance;
} SFont;

struct Vertex
{
    float fPos[2];
    float fTexCoord[2];
};

struct Character {
    ~Character() {
        vkDestroyImageView(dev, descriptorImageInfo.imageView, nullptr);
        vkDestroyImage(dev, image, nullptr);
        vkDestroySampler(dev, descriptorImageInfo.sampler, nullptr);
#ifdef VULKAN
        if(devMem != VK_NULL_HANDLE) {
            vkFreeMemory(dev, devMem, nullptr);
        }
#endif
    };

    VkDevice dev;
    VkDescriptorImageInfo descriptorImageInfo;
    VkDeviceMemory devMem;
    VkImage image;
    int ixOffset;
    int iyOffset;
    uint32_t uWidth;
    uint32_t uHeight;
    long lAdvance;
};

class CFontVulkanSC : public CVulkanSCSceneBase
{
public:
    CFontVulkanSC(VkDevice& dev, VkPhysicalDevice &physDev, VkQueue &queue,
        VkCommandPool& cmdPool, VkFormat &inputImageFormat, uint32_t uWidth, uint32_t uHeight
#ifndef VULKAN
        , VkPipelineCache &pipelineCache
#endif
        );
    ~CFontVulkanSC();

    NvError Init() override;
    NvError RecordSceneDrawCommand(VkCommandBuffer &vkCmdBuffer, uint32_t uPacketIndex, void *pSceneData = nullptr) override;

protected:
    NvError CreateVertexBuffer() override;
    NvError SetupPipeline() override;
    NvError LoadTextures() override;
    NvError CreateDescriptorSets() override;
    NvError SetupRenderPass() override;

    std::string GetVertexShaderPath() override;
    std::string GetFragmentShaderPath() override;
private:
    NvError CreateDescriptorSetLayout();
    NvError CreateDescriptorPool();
    const VkDescriptorSet& GetDescriptorSet(uint32_t index) const;
    const Character* GetCharacter(uint32_t index);
    NvError FindFontIndex(char c, int32_t &iIndex);

    // Font for 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ':', '-'.
    SFont m_fontForTSC[TSC_NUMS]{};
    std::unordered_map<uint32_t, std::unique_ptr<Character>> m_characters;

    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      m_descriptorPool      = VK_NULL_HANDLE;
    VkDescriptorSet m_descriptorSets[TSC_NUMS]  = {VK_NULL_HANDLE};
    VkBuffer m_vkVertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_vkVertexMemory = VK_NULL_HANDLE;
    void *m_pvkVertexMemoryMapped  = nullptr;
};

#endif // CFONTVULKANSC_HPP
