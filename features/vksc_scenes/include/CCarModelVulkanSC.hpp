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

#ifndef CCARMODEL_VULKANSC_HPP
#define CCARMODEL_VULKANSC_HPP

#include "CUtils.hpp"
#ifdef VULKAN
#include <vulkan/vulkan.h>
#include <vulkan/vk_layer.h>
#else
#define VK_USE_PLATFORM_SCI
#include <vulkan/vulkan_sc.h>
#endif // #ifdef VULKAN

#include <vector>
#include <string>
#include <memory>

#include "CVulkanSCSceneBase.hpp"

struct UniformBufferObject
{
    alignas(16) float model[4][4];
    alignas(16) float view[4][4];
    alignas(16) float proj[4][4];
};

struct CarVertex
{
    // position
    float position[3];
    // texCoords
    float texCoords[2];
};

class Texture
{
  public:
    Texture(VkDevice &dev)
        : m_dev(dev)
    {
    }
    ~Texture()
    {
        if (m_texureImageView != VK_NULL_HANDLE) {
            vkDestroyImageView(m_dev, m_texureImageView, nullptr);
        }
        if (m_texureImage != VK_NULL_HANDLE) {
            vkDestroyImage(m_dev, m_texureImage, nullptr);
        }
        if (m_texureImageSampler != VK_NULL_HANDLE) {
            vkDestroySampler(m_dev, m_texureImageSampler, nullptr);
        }
#ifdef VULKAN
        if (m_texureMem != VK_NULL_HANDLE) {
            vkFreeMemory(m_dev, m_texureMem, nullptr);
        }
#endif
        if (m_spPixels != nullptr) {
            delete[] m_spPixels;
        }
    };

    std::string sPath = "";
    char *m_spPixels = nullptr;
    uint32_t uWidth = 0;
    uint32_t uHeight = 0;
    VkDevice m_dev = VK_NULL_HANDLE;
    VkImage m_texureImage = VK_NULL_HANDLE;
    VkDeviceMemory m_texureMem = VK_NULL_HANDLE;
    VkImageView m_texureImageView = VK_NULL_HANDLE;
    VkSampler m_texureImageSampler = VK_NULL_HANDLE;
};

class Mesh
{
  public:
    Mesh(VkDevice &dev)
        : m_dev(dev)
    {
    }
    // mesh Data
    bool m_bHaveTexture = false;
    float m_fAngle = 0.0;
    float m_vDisplace[3];
    std::vector<float> m_fAxis = { 1.0, 0.0, 0.0 };
    std::vector<CarVertex> m_vertices;
    std::vector<uint32_t> m_indices;
    std::shared_ptr<Texture> m_spTexture;

    VkDevice m_dev = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkBuffer m_vkVertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_vkVertexMemory = VK_NULL_HANDLE;
    VkBuffer m_vkIndexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_vkIndexBufferMemory = VK_NULL_HANDLE;

    VkBuffer m_vkUniformBuffer = VK_NULL_HANDLE;
    VkDeviceMemory m_vkUniformMemory = VK_NULL_HANDLE;
    void *m_pvkUniformMemoryMapped = nullptr;
    VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet m_descriptorSets[MAX_NUM_PACKETS] = { VK_NULL_HANDLE };
    VkPipelineShaderStageCreateInfo m_stageInfo[2] = { {}, {} };

    ~Mesh()
    {
        if (m_vkVertexBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(m_dev, m_vkVertexBuffer, nullptr);
        }
        if (m_vkIndexBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(m_dev, m_vkIndexBuffer, nullptr);
        }
        if (m_vkUniformBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(m_dev, m_vkUniformBuffer, nullptr);
        }

#ifdef VULKAN
        if (m_vkVertexMemory != VK_NULL_HANDLE) {
            vkFreeMemory(m_dev, m_vkVertexMemory, nullptr);
        }
        if (m_vkIndexBufferMemory != VK_NULL_HANDLE) {
            vkFreeMemory(m_dev, m_vkIndexBufferMemory, nullptr);
        }
        if (m_vkUniformMemory != VK_NULL_HANDLE) {
            vkFreeMemory(m_dev, m_vkUniformMemory, nullptr);
        }

        if (m_stageInfo[0].module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(m_dev, m_stageInfo[0].module, nullptr);
        }

        if (m_stageInfo[1].module != VK_NULL_HANDLE) {
            vkDestroyShaderModule(m_dev, m_stageInfo[1].module, nullptr);
        }
#endif

        if (m_descriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_dev, m_descriptorSetLayout, nullptr);
        }

        for (uint32_t i = 0; i < MAX_NUM_PACKETS; ++i) {
            if (m_descriptorSets[i] != VK_NULL_HANDLE) {
                vkFreeDescriptorSets(m_dev, m_descriptorPool, 1, &m_descriptorSets[i]);
            }
        }

        if (m_descriptorPool != VK_NULL_HANDLE) {
            vkResetDescriptorPool(m_dev, m_descriptorPool, 0);
        }
    }
};

class CCarModelVulkanSC : public CVulkanSCSceneBase
{
  public:
    CCarModelVulkanSC(VkDevice &dev,
                      VkPhysicalDevice &physDev,
                      VkQueue &queue,
                      VkCommandPool &cmdPool,
                      VkFormat &inputImageFormat,
                      uint32_t uWidth,
                      uint32_t uHeight
#ifndef VULKAN
                      ,
                      VkPipelineCache &pipelineCache
#endif
    );
    ~CCarModelVulkanSC();

    NvError Init() override;
    NvError
    RecordSceneDrawCommand(VkCommandBuffer &vkCmdBuffer, uint32_t uPacketIndex, void *pSceneData = nullptr) override;

  protected:
    NvError CreateVertexBuffer() override;
    NvError CreateIndexBuffer() override;
    NvError SetupPipeline() override;
    NvError LoadTextures() override;
    NvError CreateUniformBuffer() override;
    NvError CreateDescriptorSets() override;
    NvError SetupRenderPass() override;
    NvError CreateShaderStageInfo() override;
    std::string GetVertexShaderPath() override;
    std::string GetFragmentShaderPath() override;

  private:
    static constexpr uint32_t TEX_MAX_BUFFER_SIZE = 64U * 1024U * 1024U; // 64M
    NvError CreateDescriptorSetLayout();
    NvError CreateDescriptorPool();
    NvError LoadModel();
    void UpdateUniformBuffer(uint32_t uPacketIndex, std::shared_ptr<Mesh> spTexture);

    std::vector<std::shared_ptr<Mesh>> m_Meshes;
};

#endif
