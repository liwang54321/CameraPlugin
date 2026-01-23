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

#include <inttypes.h>
#include <chrono>

#include "CCarModelVulkanSC.hpp"

constexpr float ANGLE_TO_RADIANS = 0.0174532925199432;
static constexpr uint32_t MAX_MESH_NUM = 1024U;

CCarModelVulkanSC::CCarModelVulkanSC(VkDevice &dev,
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
                                     )
    : CVulkanSCSceneBase(dev,
                         physDev,
                         queue,
                         cmdPool,
                         inputImageFormat,
                         uWidth,
                         uHeight
#ifndef VULKAN
                         ,
                         pipelineCache
#endif
      )
{
}

CCarModelVulkanSC::~CCarModelVulkanSC() {}

NvError CCarModelVulkanSC::Init()
{
    NvError error = LoadModel();
    CHK_ERROR_AND_RETURN(error, "LoadModel");

    error = CVulkanSCSceneBase::Init();
    CHK_ERROR_AND_RETURN(error, "CVulkanSCSceneBase::Init");

    return error;
}

NvError CCarModelVulkanSC::LoadModel()
{
    std::unique_ptr<FILE, CloseFile> upFile(fopen("carRes.bin", "rb"));
    CHK_PTR_AND_RETURN(upFile.get(), "Load car resource");

    uint32_t uMeshNum = 0;
    size_t uLength = fread(&uMeshNum, sizeof(uint32_t), 1U, upFile.get());
    if (uLength != 1U) {
        LOG_ERR("Read mesh number failed. Expected %" PRIu32 " actual %zu", 1U, uLength);
        return NvError_FileReadFailed;
    }

    if (uMeshNum > MAX_MESH_NUM) {
        LOG_ERR("The mesh number is too large %" PRIu32 ".Expected range[%" PRIu32 ", %" PRIu32 "]", uMeshNum, 0U,
                MAX_MESH_NUM);
        return NvError_BadValue;
    }

    for (uint32_t i = 0; i < uMeshNum; ++i) {
        std::shared_ptr<Mesh> spMesh = std::make_shared<Mesh>(m_dev);
        std::vector<CarVertex> &vertices = spMesh->m_vertices;
        std::vector<uint32_t> &indices = spMesh->m_indices;
        spMesh->m_spTexture = std::make_shared<Texture>(m_dev);

        uLength = fread(&spMesh->m_fAngle, sizeof(float), 1U, upFile.get());
        if (uLength != 1U) {
            LOG_ERR("Read angle failed. Expected %" PRIu32 " actual %zu", 1U, uLength);
            return NvError_FileReadFailed;
        }

        uLength = fread(spMesh->m_vDisplace, sizeof(float) * 3, 1U, upFile.get());
        if (uLength != 1U) {
            LOG_ERR("Read displace failed. Expected %" PRIu32 " actual %zu", 1U, uLength);
            return NvError_FileReadFailed;
        }

        uint32_t uVerticeSize = 0;
        uLength = fread(&uVerticeSize, sizeof(uint32_t), 1U, upFile.get());
        if (uLength != 1U) {
            LOG_ERR("Read vertice size failed. Expected %" PRIu32 " actual %zu", 1U, uLength);
            return NvError_FileReadFailed;
        }

        vertices.resize(uVerticeSize);
        uLength = fread(vertices.data(), sizeof(CarVertex), uVerticeSize, upFile.get());
        if (uLength != uVerticeSize) {
            LOG_ERR("Read car vertex failed. Expected %" PRIu32 " actual %zu", uVerticeSize, uLength);
            return NvError_FileReadFailed;
        }

        uint32_t uIndiceSize = 0;
        uLength = fread(&uIndiceSize, sizeof(uint32_t), 1U, upFile.get());
        if (uLength != 1U) {
            LOG_ERR("Read indice size failed. Expected %" PRIu32 " actual %zu", 1U, uLength);
            return NvError_FileReadFailed;
        }

        indices.resize(uIndiceSize);
        uLength = fread(indices.data(), sizeof(uint32_t), uIndiceSize, upFile.get());
        if (uLength != uIndiceSize) {
            LOG_ERR("Read indices failed. Expected %" PRIu32 " actual %zu", uIndiceSize, uLength);
            return NvError_FileReadFailed;
        }

        uint32_t uHasTexture = 0;
        uLength = fread(&uHasTexture, sizeof(uint32_t), 1U, upFile.get());
        if (uLength != 1U) {
            LOG_ERR("Read has texture failed. Expected %" PRIu32 " actual %zu", 1U, uLength);
            return NvError_FileReadFailed;
        }

        if (uHasTexture) {
            spMesh->m_bHaveTexture = true;
            std::shared_ptr<Texture> spTexture = spMesh->m_spTexture;

            uint32_t uTexWidth = 0, uTexHeight = 0;
            uLength = fread(&uTexWidth, sizeof(uint32_t), 1U, upFile.get());
            if (uLength != 1U) {
                LOG_ERR("Read tex width failed. Expected %" PRIu32 " actual %zu", 1U, uLength);
                return NvError_FileReadFailed;
            }
            spTexture->uWidth = uTexWidth;

            uLength = fread(&uTexHeight, sizeof(uint32_t), 1U, upFile.get());
            if (uLength != 1U) {
                LOG_ERR("Read tex height failed. Expected %" PRIu32 " actual %zu", 1U, uLength);
                return NvError_FileReadFailed;
            }
            spTexture->uHeight = uTexHeight;

            uint32_t uTexSize = uTexWidth * uTexHeight * 4;
            if (uTexSize > TEX_MAX_BUFFER_SIZE) {
                LOG_ERR("The tex buffer size is too large %" PRIu32 ".Expected range[%" PRIu32 ", %" PRIu32 "]",
                        uTexSize, 0UL, TEX_MAX_BUFFER_SIZE);
                return NvError_BadValue;
            }

            std::unique_ptr<char[]> upPixels(new char[uTexSize]);
            uLength = fread(upPixels.get(), uTexSize, 1U, upFile.get());
            if (uLength != 1U) {
                LOG_ERR("Read pixels failed. Expected %" PRIu32 " actual %zu", 1U, uLength);
                return NvError_FileReadFailed;
            }
            spTexture->m_spPixels = upPixels.release();
        }

        m_Meshes.push_back(spMesh);
    }

    return NvError_Success;
}

NvError CCarModelVulkanSC::LoadTextures()
{
    VkResult res = VK_SUCCESS;
    NvError error = NvError_Success;

    for (uint32_t i = 0; i < m_Meshes.size(); ++i) {
        if (!m_Meshes[i]->m_bHaveTexture) {
            continue;
        }

        std::shared_ptr<Texture> spTexture = m_Meshes[i]->m_spTexture;
        // load texture resources.
        uint32_t uTexWidth = spTexture->uWidth;
        uint32_t uTexHeight = spTexture->uHeight;
        VkDeviceSize imageSize = static_cast<VkDeviceSize>(uTexWidth) * static_cast<VkDeviceSize>(uTexHeight) * 4;

        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
        // Create staging buffer
        error = CreateVkBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               stagingBuffer, stagingMemory);
        CHK_ERROR_AND_RETURN(error, "CreateVkBuffer");

        // Copy texture data into staging buffer
        uint8_t *pData = nullptr;
        res = vkMapMemory(m_dev, stagingMemory, 0, static_cast<size_t>(imageSize), 0, (void **)&pData);
        CHK_VKSCSTATUS_AND_RETURN(res, "vkMapMemory");
        memcpy(pData, spTexture->m_spPixels, static_cast<size_t>(imageSize));
        vkUnmapMemory(m_dev, stagingMemory);

        error = CreateVkImage(uTexWidth, uTexHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
                              VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, spTexture->m_texureImage, spTexture->m_texureMem);
        CHK_ERROR_AND_RETURN(error, "CreateVkImage");

        VkCommandBuffer cmdBuffer = VK_NULL_HANDLE;
        error = BeginSingleTimeCommands(cmdBuffer);
        CHK_VKSCSTATUS_AND_RETURN(res, "BeginSingleTimeCommands");

        error = RecordImageLayoutTransitionCmd(cmdBuffer, spTexture->m_texureImage, VK_IMAGE_LAYOUT_UNDEFINED,
                                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        CHK_ERROR_AND_RETURN(error, "RecordImageLayoutTransitionCmd");

        // Copy mip levels from staging buffer
        VkBufferImageCopy bufferCopyRegion = {};
        bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        bufferCopyRegion.imageSubresource.mipLevel = 0;
        bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
        bufferCopyRegion.imageSubresource.layerCount = 1;
        bufferCopyRegion.imageExtent.width = static_cast<uint32_t>(uTexWidth);
        bufferCopyRegion.imageExtent.height = static_cast<uint32_t>(uTexHeight);
        bufferCopyRegion.imageExtent.depth = 1;
        bufferCopyRegion.bufferOffset = 0;
        bufferCopyRegion.imageOffset = { 0, 0, 0 };

        vkCmdCopyBufferToImage(cmdBuffer, stagingBuffer, spTexture->m_texureImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               1, &bufferCopyRegion);

        // Change texture image layout to shader read after copy
        error =
            RecordImageLayoutTransitionCmd(cmdBuffer, spTexture->m_texureImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        CHK_ERROR_AND_RETURN(error, "RecordImageLayoutTransitionCmd");

        error = EndSingleTimeCommands(cmdBuffer);
        CHK_VKSCSTATUS_AND_RETURN(res, "EndSingleTimeCommands")

        // Create descriptor sampler
        VkSamplerCreateInfo samplerCreateInfo = {};
        samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
        samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
        samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerCreateInfo.mipLodBias = 0.0f;
        samplerCreateInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerCreateInfo.minLod = 0.0f;
        samplerCreateInfo.maxLod = 1.0f;
        samplerCreateInfo.maxAnisotropy = 1.0f;
        samplerCreateInfo.anisotropyEnable = VK_FALSE;
        samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        res = vkCreateSampler(m_dev, &samplerCreateInfo, nullptr, &spTexture->m_texureImageSampler);
        CHK_VKSCSTATUS_AND_RETURN(res, "vkCreateSampler");

        // Create descriptor image view
        VkImageViewCreateInfo viewCreateInfo = {};
        viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewCreateInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
        viewCreateInfo.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B,
                                      VK_COMPONENT_SWIZZLE_A };
        viewCreateInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        viewCreateInfo.subresourceRange.levelCount = 1;
        viewCreateInfo.image = spTexture->m_texureImage;
        res = vkCreateImageView(m_dev, &viewCreateInfo, nullptr, &spTexture->m_texureImageView);
        CHK_VKSCSTATUS_AND_RETURN(res, "vkCreateImageView");

        vkDestroyBuffer(m_dev, stagingBuffer, nullptr);
#ifdef VULKAN
        vkFreeMemory(m_dev, stagingMemory, nullptr);
#endif // #ifdef VULKAN
    }

    return error;
}

NvError CCarModelVulkanSC::CreateUniformBuffer()
{
    NvError error = NvError_Success;

    VkDeviceSize bufferSize = sizeof(UniformBufferObject) * MAX_NUM_PACKETS;
    for (uint32_t i = 0; i < m_Meshes.size(); ++i) {
        std::shared_ptr<Mesh> spMesh = m_Meshes[i];
        error = CreateVkBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               spMesh->m_vkUniformBuffer, spMesh->m_vkUniformMemory);
        CHK_ERROR_AND_RETURN(error, "CreateVkBuffer");

        vkMapMemory(m_dev, spMesh->m_vkUniformMemory, 0, bufferSize, 0, &spMesh->m_pvkUniformMemoryMapped);
    }

    return error;
}

NvError CCarModelVulkanSC::CreateVertexBuffer()
{
    NvError error = NvError_Success;

    for (uint32_t i = 0; i < m_Meshes.size(); ++i) {
        std::shared_ptr<Mesh> spMesh = m_Meshes[i];
        VkDeviceSize bufferSize = sizeof(CarVertex) * spMesh->m_vertices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        error = CreateVkBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               stagingBuffer, stagingBufferMemory);
        CHK_ERROR_AND_RETURN(error, "CreateVkBuffer");

        void *pData = nullptr;
        vkMapMemory(m_dev, stagingBufferMemory, 0, bufferSize, 0, &pData);
        memcpy(pData, spMesh->m_vertices.data(), (size_t)bufferSize);
        vkUnmapMemory(m_dev, stagingBufferMemory);

        error = CreateVkBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, spMesh->m_vkVertexBuffer, spMesh->m_vkVertexMemory);
        CHK_ERROR_AND_RETURN(error, "CreateVkBuffer");

        VkCommandBuffer cmdBuffer = VK_NULL_HANDLE;
        error = BeginSingleTimeCommands(cmdBuffer);
        CHK_ERROR_AND_RETURN(error, "BeginSingleTimeCommands");

        VkBufferCopy copyRegion{};
        copyRegion.size = bufferSize;
        vkCmdCopyBuffer(cmdBuffer, stagingBuffer, spMesh->m_vkVertexBuffer, 1, &copyRegion);

        error = EndSingleTimeCommands(cmdBuffer);
        CHK_ERROR_AND_RETURN(error, "EndSingleTimeCommands");

        vkDestroyBuffer(m_dev, stagingBuffer, nullptr);
#ifdef VULKAN
        vkFreeMemory(m_dev, stagingBufferMemory, nullptr);
#endif // #ifdef VULKAN
    }

    return NvError_Success;
}

NvError CCarModelVulkanSC::CreateIndexBuffer()
{
    NvError error = NvError_Success;

    for (uint32_t i = 0; i < m_Meshes.size(); ++i) {
        std::shared_ptr<Mesh> spMesh = m_Meshes[i];
        VkDeviceSize bufferSize = sizeof(uint32_t) * spMesh->m_indices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        error = CreateVkBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               stagingBuffer, stagingBufferMemory);
        CHK_ERROR_AND_RETURN(error, "CreateVkBuffer");

        void *pData = nullptr;
        vkMapMemory(m_dev, stagingBufferMemory, 0, bufferSize, 0, &pData);
        memcpy(pData, spMesh->m_indices.data(), (size_t)bufferSize);
        vkUnmapMemory(m_dev, stagingBufferMemory);

        error =
            CreateVkBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, spMesh->m_vkIndexBuffer, spMesh->m_vkIndexBufferMemory);

        VkCommandBuffer cmdBuffer = VK_NULL_HANDLE;
        error = BeginSingleTimeCommands(cmdBuffer);
        CHK_ERROR_AND_RETURN(error, "BeginSingleTimeCommands");

        VkBufferCopy copyRegion{};
        copyRegion.size = bufferSize;
        vkCmdCopyBuffer(cmdBuffer, stagingBuffer, spMesh->m_vkIndexBuffer, 1, &copyRegion);

        error = EndSingleTimeCommands(cmdBuffer);
        CHK_ERROR_AND_RETURN(error, "EndSingleTimeCommands");

        vkDestroyBuffer(m_dev, stagingBuffer, nullptr);
#ifdef VULKAN
        vkFreeMemory(m_dev, stagingBufferMemory, nullptr);
#endif // #ifdef VULKAN
    }

    return NvError_Success;
}

NvError CCarModelVulkanSC::CreateDescriptorSetLayout()
{
    for (uint32_t i = 0; i < m_Meshes.size(); ++i) {
        std::shared_ptr<Mesh> spMesh = m_Meshes[i];
        std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings;
        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.pImmutableSamplers = nullptr;
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        setLayoutBindings.push_back(std::move(uboLayoutBinding));

        if (spMesh->m_bHaveTexture) {
            VkDescriptorSetLayoutBinding samplerLayoutBinding{};
            samplerLayoutBinding.binding = 1;
            samplerLayoutBinding.descriptorCount = 1;
            samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            samplerLayoutBinding.pImmutableSamplers = nullptr;
            samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            setLayoutBindings.push_back(std::move(samplerLayoutBinding));
        }
        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo{};
        descriptorSetLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutInfo.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
        descriptorSetLayoutInfo.pBindings = setLayoutBindings.data();
        VkResult res =
            vkCreateDescriptorSetLayout(m_dev, &descriptorSetLayoutInfo, nullptr, &spMesh->m_descriptorSetLayout);
        CHK_VKSCSTATUS_AND_RETURN(res, "vkCreateDescriptorSetLayout");
    }

    return NvError_Success;
}

NvError CCarModelVulkanSC::CreateDescriptorPool()
{
    for (uint32_t i = 0; i < m_Meshes.size(); ++i) {
        std::shared_ptr<Mesh> spMesh = m_Meshes[i];
        std::vector<VkDescriptorPoolSize> vPoolSizes;
        VkDescriptorPoolSize poolSizes;
        poolSizes.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes.descriptorCount = static_cast<uint32_t>(MAX_NUM_PACKETS);
        vPoolSizes.push_back(poolSizes);

        if (spMesh->m_bHaveTexture) {
            poolSizes.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            poolSizes.descriptorCount = static_cast<uint32_t>(MAX_NUM_PACKETS);
            vPoolSizes.push_back(poolSizes);
        }

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(vPoolSizes.size());
        poolInfo.pPoolSizes = vPoolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(MAX_NUM_PACKETS);

        VkResult res = vkCreateDescriptorPool(m_dev, &poolInfo, nullptr, &spMesh->m_descriptorPool);
        CHK_VKSCSTATUS_AND_RETURN(res, "vkCreateDescriptorPool");
    }

    return NvError_Success;
}

NvError CCarModelVulkanSC::CreateDescriptorSets()
{
    NvError error = CreateDescriptorSetLayout();
    CHK_ERROR_AND_RETURN(error, "CreateDescriptorSetLayout");
    error = CreateDescriptorPool();
    CHK_ERROR_AND_RETURN(error, "CreateDescriptorPool");

    for (uint32_t i = 0; i < m_Meshes.size(); ++i) {
        std::shared_ptr<Mesh> spMesh = m_Meshes[i];
        std::vector<VkDescriptorSetLayout> layouts(MAX_NUM_PACKETS, spMesh->m_descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = spMesh->m_descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_NUM_PACKETS);
        allocInfo.pSetLayouts = layouts.data();

        // Allocate descriptor set for graphics pipeline
        VkResult res = vkAllocateDescriptorSets(m_dev, &allocInfo, &spMesh->m_descriptorSets[0]);
        CHK_VKSCSTATUS_AND_RETURN(res, "vkAllocateDescriptorSets");

        for (uint32_t j = 0; j < MAX_NUM_PACKETS; ++j) {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = spMesh->m_vkUniformBuffer;
            bufferInfo.offset = j * sizeof(UniformBufferObject);
            bufferInfo.range = sizeof(UniformBufferObject);

            std::vector<VkWriteDescriptorSet> descriptorWrites;
            VkWriteDescriptorSet descriptorWrite = {};
            descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite.dstSet = spMesh->m_descriptorSets[j];
            descriptorWrite.dstBinding = 0;
            descriptorWrite.dstArrayElement = 0;
            descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrite.descriptorCount = 1;
            descriptorWrite.pBufferInfo = &bufferInfo;

            descriptorWrites.push_back(descriptorWrite);
            VkDescriptorImageInfo imageInfo{};
            if (spMesh->m_bHaveTexture) {
                memset(&descriptorWrite, 0, sizeof(VkWriteDescriptorSet));

                imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                imageInfo.imageView = spMesh->m_spTexture->m_texureImageView;
                imageInfo.sampler = spMesh->m_spTexture->m_texureImageSampler;

                descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrite.dstSet = spMesh->m_descriptorSets[j];
                descriptorWrite.dstBinding = 1;
                descriptorWrite.dstArrayElement = 0;
                descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                descriptorWrite.descriptorCount = 1;
                descriptorWrite.pImageInfo = &imageInfo;
                descriptorWrites.push_back(descriptorWrite);
            }

            vkUpdateDescriptorSets(m_dev, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0,
                                   nullptr);
        }
    }

    return NvError_Success;
}
NvError CCarModelVulkanSC::SetupRenderPass()
{
    VkAttachmentDescription attachments[2] = {};

    // Color attachment
    attachments[0].format = m_inputImageFormat;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // Depth attachment
    attachments[1].format = FindDepthFormat(m_physDev, VK_IMAGE_TILING_OPTIMAL);
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorReference = {};
    colorReference.attachment = 0;
    colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.flags = 0;
    subpass.inputAttachmentCount = 0;
    subpass.pInputAttachments = NULL;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorReference;
    subpass.pResolveAttachments = NULL;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;
    subpass.preserveAttachmentCount = 0;
    subpass.pPreserveAttachments = NULL;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.pNext = NULL;
    renderPassInfo.attachmentCount = 2;
    renderPassInfo.pAttachments = &attachments[0];
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    VkResult res = vkCreateRenderPass(m_dev, &renderPassInfo, NULL, &m_renderPass);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkCreateRenderPass");

    return NvError_Success;
}

NvError CCarModelVulkanSC::SetupPipeline()
{
    VkResult res;
    for (uint32_t i = 0; i < m_Meshes.size(); ++i) {
        std::shared_ptr<Mesh> spMesh = m_Meshes[i];
        VkPipelineLayoutCreateInfo layoutCreateInfo = {};
        layoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutCreateInfo.setLayoutCount = 1;
        layoutCreateInfo.pSetLayouts = &spMesh->m_descriptorSetLayout;

        res = vkCreatePipelineLayout(m_dev, &layoutCreateInfo, nullptr, &spMesh->m_pipelineLayout);
        CHK_VKSCSTATUS_AND_RETURN(res, "vkCreatePipelineLayout");

        VkGraphicsPipelineCreateInfo pipelineCreateInfo = {};
        pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineCreateInfo.renderPass = m_renderPass;
        pipelineCreateInfo.layout = spMesh->m_pipelineLayout;

        VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VkVertexInputBindingDescription bindingDescription = {};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(CarVertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription attributeDescriptions[2];

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(CarVertex, position);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(CarVertex, texCoords);

        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount =
            sizeof(attributeDescriptions) / sizeof(attributeDescriptions[0]);
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions;
        pipelineCreateInfo.pVertexInputState = &vertexInputInfo;

        VkPipelineInputAssemblyStateCreateInfo inputAssemblyState = {};
        inputAssemblyState.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;

        // Viewport state
        VkPipelineViewportStateCreateInfo viewportState = {};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;
        pipelineCreateInfo.pViewportState = &viewportState;

        VkPipelineDynamicStateCreateInfo dynamicState = {};
        std::vector<VkDynamicState> dynamicStateEnables;
        dynamicStateEnables.push_back(VK_DYNAMIC_STATE_VIEWPORT);
        dynamicStateEnables.push_back(VK_DYNAMIC_STATE_SCISSOR);
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.pDynamicStates = dynamicStateEnables.data();
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());
        pipelineCreateInfo.pDynamicState = &dynamicState;

        VkPipelineMultisampleStateCreateInfo multisampleState = {};
        multisampleState.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampleState.pSampleMask = NULL;
        multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        pipelineCreateInfo.pMultisampleState = &multisampleState;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        if (spMesh->m_bHaveTexture == false)
            depthStencil.depthWriteEnable = VK_FALSE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;
        pipelineCreateInfo.pDepthStencilState = &depthStencil;

        // Rasterization state
        VkPipelineRasterizationStateCreateInfo rasterizationState = {};
        rasterizationState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizationState.cullMode = VK_CULL_MODE_NONE;
        rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizationState.depthClampEnable = VK_FALSE;
        rasterizationState.rasterizerDiscardEnable = VK_FALSE;
        rasterizationState.depthBiasEnable = VK_FALSE;
        rasterizationState.lineWidth = 1.0f;
        pipelineCreateInfo.pRasterizationState = &rasterizationState;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo colorBlendState = {};
        colorBlendState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlendState.attachmentCount = 1;
        colorBlendState.pAttachments = &colorBlendAttachment;
        pipelineCreateInfo.pColorBlendState = &colorBlendState;

        pipelineCreateInfo.stageCount = 2;
        pipelineCreateInfo.pStages = spMesh->m_stageInfo;

#ifndef VULKAN
        char graphicsUUID[VK_UUID_SIZE];
        std::string sJsonName;
        if (spMesh->m_bHaveTexture) {
            sJsonName = "car_model_render_pipeline.json";
        } else {
            sJsonName = "car_model_notexture_render_pipeline.json";
        }
        NvError status = ReadPipelineUUID(sJsonName, graphicsUUID);
        CHK_ERROR_AND_RETURN(status, "ReadPipelineUUID");
        VkPipelineOfflineCreateInfo pipelineOfflineCreateInfo = {};
        pipelineOfflineCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_OFFLINE_CREATE_INFO;
        memcpy(pipelineOfflineCreateInfo.pipelineIdentifier, graphicsUUID, VK_UUID_SIZE); // Obtained in json.
        pipelineOfflineCreateInfo.poolEntrySize = 1024U * 1024U;
        ;
        pipelineCreateInfo.pNext = &pipelineOfflineCreateInfo;
        res = vkCreateGraphicsPipelines(m_dev, m_pipelineCache, 1, &pipelineCreateInfo, NULL, &spMesh->m_pipeline);
#else
        pipelineCreateInfo.pNext = nullptr;
        res = vkCreateGraphicsPipelines(m_dev, VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, &spMesh->m_pipeline);
#endif // #ifndef VULKAN
        CHK_VKSCSTATUS_AND_RETURN(res, "vkCreateGraphicsPipelines");
    }

    return NvError_Success;
}

NvError CCarModelVulkanSC::RecordSceneDrawCommand(VkCommandBuffer &vkCmdBuffer, uint32_t uPacketIndex, void *pSceneData)
{
    for (uint32_t i = 0; i < m_Meshes.size(); ++i) {
        std::shared_ptr<Mesh> spMesh = m_Meshes[i];
        UpdateUniformBuffer(uPacketIndex, spMesh);

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = m_renderPass;
        renderPassInfo.framebuffer = m_vkFrameBuffer[uPacketIndex];
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent.width = m_uWidth;
        renderPassInfo.renderArea.extent.height = m_uHeight;

        vkCmdBeginRenderPass(vkCmdBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(vkCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, spMesh->m_pipeline);

        // Update dynamic viewport state
        VkViewport viewport{};
        viewport.height = (float)m_uHeight;
        viewport.width = (float)m_uWidth;
        viewport.minDepth = (float)0.0f;
        viewport.maxDepth = (float)1.0f;
        vkCmdSetViewport(vkCmdBuffer, 0, 1, &viewport);

        // Update dynamic scissor state
        VkRect2D scissor{};
        scissor.extent.width = m_uWidth;
        scissor.extent.height = m_uHeight;
        scissor.offset.x = 0;
        scissor.offset.y = 0;
        vkCmdSetScissor(vkCmdBuffer, 0, 1, &scissor);

        VkDeviceSize offsets[1] = { 0 };
        vkCmdBindVertexBuffers(vkCmdBuffer, 0, 1, &spMesh->m_vkVertexBuffer, offsets);
        vkCmdBindIndexBuffer(vkCmdBuffer, spMesh->m_vkIndexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(vkCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, spMesh->m_pipelineLayout, 0, 1,
                                &spMesh->m_descriptorSets[uPacketIndex], 0, NULL);
        vkCmdDrawIndexed(vkCmdBuffer, static_cast<uint32_t>(spMesh->m_indices.size()), 1, 0, 0, 0);
        vkCmdEndRenderPass(vkCmdBuffer);
    }

    return NvError_Success;
}

void CCarModelVulkanSC::UpdateUniformBuffer(uint32_t uPacketIndex, std::shared_ptr<Mesh> spMesh)
{
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    UniformBufferObject ubo{};
    float angle = spMesh->m_fAngle;
    std::vector<std::vector<float>> model = { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 } };

    if (angle > 0) {
        std::vector<float> axis = { 1.0, 0.0, 0.0 };
        model = Rotate(model, time * 3 * angle * ANGLE_TO_RADIANS, spMesh->m_fAxis);
    }

    std::vector<float> vfTrans = { spMesh->m_vDisplace[0], spMesh->m_vDisplace[1] + 0.4f,
                                   spMesh->m_vDisplace[2] + 1.2f };
    std::vector<std::vector<float>> vTransMat = {
        { 0.35, 0, 0, 0 }, { 0, 0.35, 0, 0 }, { 0, 0, 0.35, 0 }, { 0, 0, 0, 0.35 }
    };
    vTransMat[3][0] =
        vTransMat[0][0] * vfTrans[0] + vTransMat[1][0] * vfTrans[1] + vTransMat[2][0] * vfTrans[2] + vTransMat[3][0];
    vTransMat[3][1] =
        vTransMat[0][1] * vfTrans[0] + vTransMat[1][1] * vfTrans[1] + vTransMat[2][1] * vfTrans[2] + vTransMat[3][1];
    vTransMat[3][2] =
        vTransMat[0][2] * vfTrans[0] + vTransMat[1][2] * vfTrans[1] + vTransMat[2][2] * vfTrans[2] + vTransMat[3][2];
    vTransMat[3][3] =
        vTransMat[0][3] * vfTrans[0] + vTransMat[1][3] * vfTrans[1] + vTransMat[2][3] * vfTrans[2] + vTransMat[3][3];
    model = MatMul(vTransMat, model);

    std::vector<std::vector<float>> vfRotateMat = { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 } };
    std::vector<float> vfRotateAxis = { 1.0, 0.0, 0.0 };
    vfRotateMat = Rotate(vfRotateMat, -2 * 45.0f * ANGLE_TO_RADIANS, vfRotateAxis);
    model = MatMul(vfRotateMat, model);
    vfRotateAxis = { 0.0, 0.0, 1.0 };
    vfRotateMat = { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 } };
    vfRotateMat = Rotate(vfRotateMat, 45.0f * ANGLE_TO_RADIANS, vfRotateAxis);
    model = MatMul(vfRotateMat, model);
    model[3][3] = 1.0f;

    std::vector<std::vector<float>> view = { { -1, 0, -0, 0 }, { 0, 0, -1, 0 }, { 0, -1, -0, 0 }, { -0, -0, -3, 1 } };
    std::vector<std::vector<float>> proj =
        GetPerspectiveMat(45.0f * ANGLE_TO_RADIANS, m_uWidth / (float)m_uHeight, 0.1f, 10.0f);

    for (uint32_t i = 0; i < 4; ++i) {
        for (uint32_t j = 0; j < 4; ++j) {
            ubo.model[i][j] = model[i][j];
            ubo.proj[i][j] = proj[i][j];
            ubo.view[i][j] = view[i][j];
        }
    }

    memcpy((char *)spMesh->m_pvkUniformMemoryMapped + uPacketIndex * sizeof(UniformBufferObject), &ubo, sizeof(ubo));
}

NvError CCarModelVulkanSC::CreateShaderStageInfo()
{
    for (uint32_t i = 0; i < m_Meshes.size(); ++i) {
        std::shared_ptr<Mesh> spMesh = m_Meshes[i];
        // In VKSC, the shader compilation is offline.
        spMesh->m_stageInfo[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        spMesh->m_stageInfo[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        spMesh->m_stageInfo[0].pName = "main";
        spMesh->m_stageInfo[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        spMesh->m_stageInfo[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        spMesh->m_stageInfo[1].pName = "main";

#ifndef VULKAN
        spMesh->m_stageInfo[0].module = VK_NULL_HANDLE;
        spMesh->m_stageInfo[1].module = VK_NULL_HANDLE;
#else
        std::string sVertexShaderPath = GetVertexShaderPath();
        std::string sFragmentShaderPath = GetFragmentShaderPath();
        if (spMesh->m_bHaveTexture) {
            sFragmentShaderPath = "carmodel.shader.frag.spv";
        }
        spMesh->m_stageInfo[0].module = loadSPIRVShader(sVertexShaderPath);
        spMesh->m_stageInfo[1].module = loadSPIRVShader(sFragmentShaderPath);
#endif // #ifndef VULKAN
    }

    return NvError_Success;
}

std::string CCarModelVulkanSC::GetVertexShaderPath()
{
    return "carmodel.shader.vert.spv";
}

std::string CCarModelVulkanSC::GetFragmentShaderPath()
{
    return "carmodel_notexture.shader.frag.spv";
}
