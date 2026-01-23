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

#include <array>
#include <inttypes.h>
#include "CFontVulkanSC.hpp"

constexpr uint32_t MAX_TIMESTAMP_LENGTH = 32;

CFontVulkanSC::CFontVulkanSC(VkDevice &dev,
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

CFontVulkanSC::~CFontVulkanSC()
{
    if (m_vkVertexBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(m_dev, m_vkVertexBuffer, nullptr);
    }
#ifdef VULKAN
    if (m_vkVertexMemory != VK_NULL_HANDLE) {
        vkFreeMemory(m_dev, m_vkVertexMemory, nullptr);
    }
#endif

    if (m_descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_dev, m_descriptorSetLayout, nullptr);
    }
    if (m_descriptorSets[0] != VK_NULL_HANDLE) {
        vkFreeDescriptorSets(m_dev, m_descriptorPool, TSC_NUMS, m_descriptorSets);
    }
    if (m_descriptorPool != VK_NULL_HANDLE) {
        vkResetDescriptorPool(m_dev, m_descriptorPool, 0);
    }

    for (uint32_t i = 0; i < TSC_NUMS; ++i) {
        if (m_fontForTSC[i].pBuffer != nullptr) {
            delete[] m_fontForTSC[i].pBuffer;
        }
    }
}

NvError CFontVulkanSC::Init()
{
    NvError error = CVulkanSCSceneBase::Init();
    CHK_ERROR_AND_RETURN(error, "CVulkanSCSceneBase::Init");

    return error;
}

NvError CFontVulkanSC::CreateVertexBuffer()
{
    VkResult res;
    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = sizeof(Vertex) * MAX_TIMESTAMP_LENGTH * MAX_NUM_PACKETS;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    res = vkCreateBuffer(m_dev, &bufferCreateInfo, nullptr, &m_vkVertexBuffer);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkCreateBuffer");

    VkMemoryRequirements memReqs{};
    vkGetBufferMemoryRequirements(m_dev, m_vkVertexBuffer, &memReqs);

    VkMemoryAllocateInfo memAllocInfo{};
    memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memAllocInfo.allocationSize = memReqs.size;
    memAllocInfo.memoryTypeIndex =
        GetMemoryType(m_memProperties, memReqs.memoryTypeBits,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    res = vkAllocateMemory(m_dev, &memAllocInfo, nullptr, &m_vkVertexMemory);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkAllocateMemory");

    res = vkBindBufferMemory(m_dev, m_vkVertexBuffer, m_vkVertexMemory, 0);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkBindBufferMemory");

    res = vkMapMemory(m_dev, m_vkVertexMemory, 0, sizeof(Vertex) * MAX_TIMESTAMP_LENGTH * MAX_NUM_PACKETS, 0,
                      &m_pvkVertexMemoryMapped);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkMapMemory");

    return NvError_Success;
}

NvError CFontVulkanSC::CreateDescriptorSetLayout()
{
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings;
    VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 0;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    setLayoutBindings.push_back(std::move(samplerLayoutBinding));

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo{};
    descriptorSetLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutInfo.bindingCount = static_cast<uint32_t>(setLayoutBindings.size());
    descriptorSetLayoutInfo.pBindings = setLayoutBindings.data();
    VkResult res = vkCreateDescriptorSetLayout(m_dev, &descriptorSetLayoutInfo, nullptr, &m_descriptorSetLayout);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkCreateDescriptorSetLayout");

    return NvError_Success;
}

NvError CFontVulkanSC::SetupPipeline()
{
    VkResult res;

    VkPipelineLayoutCreateInfo layoutCreateInfo = {};
    layoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutCreateInfo.setLayoutCount = 1;
    layoutCreateInfo.pSetLayouts = &m_descriptorSetLayout;

    res = vkCreatePipelineLayout(m_dev, &layoutCreateInfo, nullptr, &m_pipelineLayout);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkCreatePipelineLayout");

    VkGraphicsPipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.renderPass = m_renderPass;
    pipelineCreateInfo.layout = m_pipelineLayout;

    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkVertexInputBindingDescription bindingDescription = {};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attributeDescriptions[2];

    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, fPos);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, fTexCoord);

    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.vertexAttributeDescriptionCount = sizeof(attributeDescriptions) / sizeof(attributeDescriptions[0]);
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
    pipelineCreateInfo.pStages = m_stageInfo;

#ifndef VULKAN
    char graphicsUUID[VK_UUID_SIZE];
    NvError status = ReadPipelineUUID("timestamp_render_pipeline.json", graphicsUUID);
    CHK_ERROR_AND_RETURN(status, "ReadPipelineUUID");
    VkPipelineOfflineCreateInfo pipelineOfflineCreateInfo = {};
    pipelineOfflineCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_OFFLINE_CREATE_INFO;
    memcpy(pipelineOfflineCreateInfo.pipelineIdentifier, graphicsUUID, VK_UUID_SIZE); // Obtained in json.
    pipelineOfflineCreateInfo.poolEntrySize = 1024U * 1024U;
    ;
    pipelineCreateInfo.pNext = &pipelineOfflineCreateInfo;
    res = vkCreateGraphicsPipelines(m_dev, m_pipelineCache, 1, &pipelineCreateInfo, NULL, &m_pipeline);
#else
    pipelineCreateInfo.pNext = nullptr;
    res = vkCreateGraphicsPipelines(m_dev, VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, &m_pipeline);
#endif // #ifndef VULKAN
    CHK_VKSCSTATUS_AND_RETURN(res, "vkCreateGraphicsPipelines");

    return NvError_Success;
}

const VkDescriptorSet &CFontVulkanSC::GetDescriptorSet(uint32_t index) const
{
    return m_descriptorSets[index];
}

const Character *CFontVulkanSC::GetCharacter(uint32_t index)
{
    if (m_characters.find(index) != m_characters.end()) {
        return m_characters[index].get();
    }

    return nullptr;
}

NvError CFontVulkanSC::CreateDescriptorPool()
{
    VkDescriptorPoolSize poolSizes;
    poolSizes.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes.descriptorCount = TSC_NUMS;

    VkDescriptorPoolCreateInfo descriptorPoolInfo{};
    descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolInfo.poolSizeCount = 1U;
    descriptorPoolInfo.pPoolSizes = &poolSizes;
    descriptorPoolInfo.maxSets = TSC_NUMS;
    descriptorPoolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    VkResult res = vkCreateDescriptorPool(m_dev, &descriptorPoolInfo, nullptr, &m_descriptorPool);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkCreateDescriptorPool");

    return NvError_Success;
}

NvError CFontVulkanSC::CreateDescriptorSets()
{
    NvError error = CreateDescriptorSetLayout();
    CHK_ERROR_AND_RETURN(error, "CreateDescriptorSetLayout");
    error = CreateDescriptorPool();
    CHK_ERROR_AND_RETURN(error, "CreateDescriptorPool");

    std::vector<VkDescriptorSetLayout> layouts(TSC_NUMS, m_descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(TSC_NUMS);
    allocInfo.pSetLayouts = layouts.data();

    // Allocate descriptor set for graphics pipeline
    VkResult res = vkAllocateDescriptorSets(m_dev, &allocInfo, m_descriptorSets);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkAllocateDescriptorSets");

    for (uint32_t i = 0; i < TSC_NUMS; ++i) {
        const VkDescriptorImageInfo &imageInfo = m_characters[i]->descriptorImageInfo;
        VkWriteDescriptorSet descriptorWrites = {};
        descriptorWrites.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites.dstSet = m_descriptorSets[i];
        descriptorWrites.dstBinding = 0;
        descriptorWrites.dstArrayElement = 0;
        descriptorWrites.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites.descriptorCount = 1;
        descriptorWrites.pImageInfo = &imageInfo;
        vkUpdateDescriptorSets(m_dev, 1, &descriptorWrites, 0, nullptr);
    }

    return NvError_Success;
}

#define CHK_VKSCSTATUS_AND_DESTROY_RETURN(vkscStatus, api)             \
    {                                                                  \
        if (vkscStatus != VK_SUCCESS) {                                \
            LOG_ERR("%s failed, status: %d\n", (api), (vkscStatus));   \
            if (fence != VK_NULL_HANDLE) {                             \
                vkDestroyFence(m_dev, fence, nullptr);                 \
            }                                                          \
            if (stagingBuffer != VK_NULL_HANDLE) {                     \
                vkDestroyBuffer(m_dev, stagingBuffer, nullptr);        \
            }                                                          \
            if (cmdBuffer != VK_NULL_HANDLE) {                         \
                vkFreeCommandBuffers(m_dev, m_cmdPool, 1, &cmdBuffer); \
            }                                                          \
            return NvError_ResourceError;                              \
        }                                                              \
    }

#define CHK_STATUS_AND_DESTROY_RETURN(status, api)                     \
    {                                                                  \
        if (status != NvError_Success) {                               \
            LOG_ERR("%s failed, status: %d\n", (api), (status));       \
            if (fence != VK_NULL_HANDLE) {                             \
                vkDestroyFence(m_dev, fence, nullptr);                 \
            }                                                          \
            if (stagingBuffer != VK_NULL_HANDLE) {                     \
                vkDestroyBuffer(m_dev, stagingBuffer, nullptr);        \
            }                                                          \
            if (cmdBuffer != VK_NULL_HANDLE) {                         \
                vkFreeCommandBuffers(m_dev, m_cmdPool, 1, &cmdBuffer); \
            }                                                          \
            return NvError_ResourceError;                              \
        }                                                              \
    }

NvError CFontVulkanSC::LoadTextures()
{
    VkResult res = VK_SUCCESS;
    NvError error = NvError_Success;
    vkGetPhysicalDeviceMemoryProperties(m_physDev, &m_memProperties);

    VkCommandBuffer cmdBuffer = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo cmdBufAllocateInfo{};
    cmdBufAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdBufAllocateInfo.pNext = NULL;
    cmdBufAllocateInfo.commandPool = m_cmdPool;
    cmdBufAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdBufAllocateInfo.commandBufferCount = 1;
    res = vkAllocateCommandBuffers(m_dev, &cmdBufAllocateInfo, &cmdBuffer);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkAllocateCommandBuffers");

    // load font resources.
    std::unique_ptr<FILE, CloseFile> upFile(fopen("fontRes.bin", "rb"));
    if (upFile == nullptr) {
        LOG_ERR("Load Font Resources fail.");
        return NvError_ResourceError;
    }
    for (uint32_t c = 0; c < TSC_NUMS; ++c) {
        SFont &font = m_fontForTSC[c];
        size_t uLength = fread(&font.uBufferSize, sizeof(font.uBufferSize), 1U, upFile.get());
        if (uLength != 1U) {
            LOG_ERR("Read font buffer size failed. Expected size %" PRIu32 " actual size %zu", 1U, uLength);
            return NvError_FileReadFailed;
        }
        if (font.uBufferSize > FONT_MAX_BUFFER_SIZE) {
            LOG_ERR("The buffer size is too large %" PRIu32 ".Expected range[%" PRIu32 ", %" PRIu32 "]",
                    font.uBufferSize, 0UL, FONT_MAX_BUFFER_SIZE);
            return NvError_BadValue;
        }
        std::unique_ptr<uint8_t[]> upBuffer(new (std::nothrow) uint8_t[font.uBufferSize]);
        CHK_PTR_AND_RETURN(upBuffer, "Allocate font buffer");
        uLength = fread(upBuffer.get(), font.uBufferSize, 1U, upFile.get());
        if (uLength != 1U) {
            LOG_ERR("Read font buffer failed. Expected size %" PRIu32 " actual size %zu", 1U, uLength);
            return NvError_FileReadFailed;
        }
        uLength = fread((char *)&font + offsetof(SFont, ixOffset), sizeof(font) - offsetof(SFont, ixOffset), 1U,
                        upFile.get());
        if (uLength != 1U) {
            LOG_ERR("Read font buffer content failed. Expected size %" PRIu32 " actual size %zu", 1U, uLength);
            return NvError_FileReadFailed;
        }
        // Transfer the ownership to font.pBuffer
        font.pBuffer = upBuffer.release();
    }

    for (uint32_t c = 0; c < TSC_NUMS; ++c) {
        const SFont &font = m_fontForTSC[c];
        uint32_t uWidth = font.uWidth;
        uint32_t uHeight = font.uHeight;

        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;

        // Create staging buffer
        error = CreateVkBuffer(uWidth * uHeight, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               stagingBuffer, stagingMemory);
        CHK_ERROR_AND_RETURN(error, "CreateVkBuffer");

        // Copy texture data into staging buffer
        uint8_t *pData = nullptr;
        res = vkMapMemory(m_dev, stagingMemory, 0, uWidth * uHeight, 0, (void **)&pData);
        CHK_VKSCSTATUS_AND_DESTROY_RETURN(res, "vkMapMemory");
        memcpy(pData, font.pBuffer, uWidth * uHeight);
        vkUnmapMemory(m_dev, stagingMemory);

        std::unique_ptr<Character> character = std::make_unique<Character>();
        character->dev = m_dev;

        // Create optimal tiled target image
        error = CreateVkImage(uWidth, uHeight, VK_FORMAT_R8_UNORM, VK_IMAGE_TILING_OPTIMAL,
                              VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, character->image, character->devMem);
        CHK_ERROR_AND_RETURN(error, "CreateVkImage");

        // Allocate copy command buffer
        res = vkResetCommandPool(m_dev, m_cmdPool, 0);
        CHK_VKSCSTATUS_AND_DESTROY_RETURN(res, "vkResetCommandPool");

        // Build copy command buffer
        VkCommandBufferBeginInfo cmdBufInfo{};
        cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        cmdBufInfo.pNext = NULL;
        res = vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo);
        CHK_VKSCSTATUS_AND_DESTROY_RETURN(res, "vkBeginCommandBuffer");

        error = RecordImageLayoutTransitionCmd(cmdBuffer, character->image, VK_IMAGE_LAYOUT_UNDEFINED,
                                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        CHK_ERROR_AND_RETURN(error, "RecordImageLayoutTransitionCmd");

        // Copy mip levels from staging buffer
        VkBufferImageCopy bufferCopyRegion = {};
        bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        bufferCopyRegion.imageSubresource.mipLevel = 0;
        bufferCopyRegion.imageSubresource.baseArrayLayer = 0;
        bufferCopyRegion.imageSubresource.layerCount = 1;
        bufferCopyRegion.imageExtent.width = uWidth;
        bufferCopyRegion.imageExtent.height = uHeight;
        bufferCopyRegion.imageExtent.depth = 1;
        bufferCopyRegion.bufferOffset = 0;
        bufferCopyRegion.imageOffset = { 0, 0, 0 };

        vkCmdCopyBufferToImage(cmdBuffer, stagingBuffer, character->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                               &bufferCopyRegion);

        // Change texture image layout to shader read after copy
        error = RecordImageLayoutTransitionCmd(cmdBuffer, character->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        CHK_ERROR_AND_RETURN(error, "RecordImageLayoutTransitionCmd");

        vkEndCommandBuffer(cmdBuffer);
        // Submit copy command buffer to graphics queue
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmdBuffer;

        VkFenceCreateInfo fenceCreateInfo{};
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.flags = 0;
        res = vkCreateFence(m_dev, &fenceCreateInfo, nullptr, &fence);
        CHK_VKSCSTATUS_AND_DESTROY_RETURN(res, "vkCreateFence");

        res = vkQueueSubmit(m_queue, 1, &submitInfo, fence);
        CHK_VKSCSTATUS_AND_DESTROY_RETURN(res, "vkQueueSubmit");
        res = vkWaitForFences(m_dev, 1, &fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT);
        CHK_VKSCSTATUS_AND_DESTROY_RETURN(res, "vkWaitForFences");

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
        samplerCreateInfo.compareOp = VK_COMPARE_OP_NEVER;
        samplerCreateInfo.minLod = 0.0f;
        samplerCreateInfo.maxLod = 1.0f;
        samplerCreateInfo.maxAnisotropy = 1.0f;
        samplerCreateInfo.anisotropyEnable = VK_FALSE;
        samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        res = vkCreateSampler(m_dev, &samplerCreateInfo, nullptr, &character->descriptorImageInfo.sampler);
        CHK_VKSCSTATUS_AND_DESTROY_RETURN(res, "vkCreateSampler");

        // Create descriptor image view
        VkImageViewCreateInfo viewCreateInfo = {};
        viewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewCreateInfo.format = VK_FORMAT_R8_UNORM;
        viewCreateInfo.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B,
                                      VK_COMPONENT_SWIZZLE_A };
        viewCreateInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        viewCreateInfo.subresourceRange.levelCount = 1;
        viewCreateInfo.image = character->image;
        res = vkCreateImageView(m_dev, &viewCreateInfo, nullptr, &character->descriptorImageInfo.imageView);
        CHK_VKSCSTATUS_AND_DESTROY_RETURN(res, "vkCreateImageView");

        // Set descriptor image layout
        character->descriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        character->uWidth = uWidth;
        character->uHeight = uHeight;
        character->lAdvance = font.lAdvance;
        character->ixOffset = font.ixOffset;
        character->iyOffset = font.iyOffset;
        m_characters.emplace(std::make_pair(c, std::move(character)));

        vkDestroyFence(m_dev, fence, nullptr);
        vkDestroyBuffer(m_dev, stagingBuffer, nullptr);

#ifdef VULKAN
        vkFreeMemory(m_dev, stagingMemory, nullptr);
#endif // #ifdef VULKAN
    }

    vkFreeCommandBuffers(m_dev, m_cmdPool, 1, &cmdBuffer);
    return NvError_Success;
}

NvError CFontVulkanSC::SetupRenderPass()
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
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
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

NvError CFontVulkanSC::RecordSceneDrawCommand(VkCommandBuffer &vkCmdBuffer, uint32_t uPacketIndex, void *pSceneData)
{
    NvError status;
    std::string sTimeStampToDraw = *static_cast<std::string *>(pSceneData);
    // timestamp added to the top right of output image.
    float fx = m_uWidth - 64.0f, fy = m_uHeight - 32.0f, fScale = 1.5f;
    for (int32_t i = sTimeStampToDraw.size() - 1; i >= 0; --i) {
        int32_t iIndex = -1;
        status = FindFontIndex(sTimeStampToDraw[i], iIndex);
        CHK_ERROR_AND_RETURN(status, "FindFontIndex");

        if (iIndex == -1) {
            fx -= 10 * fScale;
            continue;
        }

        const Character *pCharacter = GetCharacter(iIndex);
        if (pCharacter == nullptr) {
            continue;
        }

        uint32_t uWidth = m_uWidth;
        uint32_t uHeight = m_uHeight;
        float fh = pCharacter->uHeight * fScale / uHeight;
        float fw = pCharacter->uWidth * fScale / uWidth;
        // orthogonal projection to get timestamp position.
        float xpos = fx / uWidth * 2 - 1.0 + pCharacter->ixOffset * fScale / uWidth;
        float ypos = fy / uHeight * 2 - 1.0 - (pCharacter->iyOffset) * fScale / uHeight;

        float fVertices[6][4] = {
            { xpos, ypos, 0.0, 0.0 }, { xpos + fw, ypos, 1.0, 0.0 },      { xpos + fw, ypos + fh, 1.0, 1.0 },

            { xpos, ypos, 0.0, 0.0 }, { xpos + fw, ypos + fh, 1.0, 1.0 }, { xpos, ypos + fh, 0.0, 1.0 },
        };

        memcpy((void *)((char *)m_pvkVertexMemoryMapped + i * sizeof(Vertex) * 6), fVertices, sizeof(Vertex) * 6);
        fx -= (pCharacter->lAdvance >> 7) * fScale;
    }

    // Render timestamp.
    for (int32_t i = sTimeStampToDraw.size() - 1; i >= 0; --i) {
        int32_t iIndex = -1;
        status = FindFontIndex(sTimeStampToDraw[i], iIndex);
        CHK_ERROR_AND_RETURN(status, "FindFontIndex");

        if (iIndex == -1) {
            continue;
        }

        const VkDescriptorSet &descriptorSet = GetDescriptorSet(iIndex);

        uint32_t uWidth = m_uWidth;
        uint32_t uHeight = m_uHeight;

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = m_renderPass;
        renderPassInfo.framebuffer = m_vkFrameBuffer[uPacketIndex];
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent.width = uWidth;
        renderPassInfo.renderArea.extent.height = uHeight;

        VkClearValue clearValues[2] = {};
        clearValues[1].depthStencil = { 1.0f, 0 };

        renderPassInfo.clearValueCount = 2;
        renderPassInfo.pClearValues = &clearValues[0];

        vkCmdBeginRenderPass(vkCmdBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(vkCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);

        // Update dynamic viewport state
        VkViewport viewport{};
        viewport.height = (float)uHeight;
        viewport.width = (float)uWidth;
        viewport.minDepth = (float)0.0f;
        viewport.maxDepth = (float)1.0f;
        vkCmdSetViewport(vkCmdBuffer, 0, 1, &viewport);

        // Update dynamic scissor state
        VkRect2D scissor{};
        scissor.extent.width = uWidth;
        scissor.extent.height = uHeight;
        scissor.offset.x = 0;
        scissor.offset.y = 0;
        vkCmdSetScissor(vkCmdBuffer, 0, 1, &scissor);

        VkDeviceSize offsets[1] = { i * sizeof(Vertex) * 6 };
        vkCmdBindVertexBuffers(vkCmdBuffer, 0, 1, &m_vkVertexBuffer, offsets);
        vkCmdBindDescriptorSets(vkCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &descriptorSet, 0,
                                NULL);
        vkCmdDraw(vkCmdBuffer, 6, 1, 0, 0);
        vkCmdEndRenderPass(vkCmdBuffer);
    }

    return NvError_Success;
}

NvError CFontVulkanSC::FindFontIndex(char c, int32_t &iIndex)
{
    if (c >= '0' && c <= '9') {
        iIndex = c - '0';
        return NvError_Success;
    }

    switch (c) {
        case ':':
            iIndex = 10;
            break;
        case '-':
            iIndex = 11;
            break;
        default:
            iIndex = -1;
            break;
    }

    return NvError_Success;
}

std::string CFontVulkanSC::GetVertexShaderPath()
{
    return "timestamp.shader.vert.spv";
}

std::string CFontVulkanSC::GetFragmentShaderPath()
{
    return "timestamp.shader.frag.spv";
}
