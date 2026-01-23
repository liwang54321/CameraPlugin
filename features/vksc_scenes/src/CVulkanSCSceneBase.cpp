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

#include "CVulkanSCSceneBase.hpp"
#include<fstream>

CVulkanSCSceneBase::~CVulkanSCSceneBase()
{
    for(uint32_t i = 0; i < MAX_NUM_PACKETS; ++i) {
        if(m_vkFrameBuffer[i] != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(m_dev, m_vkFrameBuffer[i], nullptr);
        }
    }

    if(m_renderPass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(m_dev, m_renderPass, nullptr);
    }

#ifdef VULKAN
    if(m_stageInfo[0].module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(m_dev, m_stageInfo[0].module, nullptr);
    }

    if(m_stageInfo[1].module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(m_dev, m_stageInfo[1].module, nullptr);
    }
#endif

    if(m_pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_dev, m_pipelineLayout, nullptr);
    }

    if(m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_dev, m_pipeline, nullptr);
    }
}

NvError CVulkanSCSceneBase::Init()
{
    NvError error = NvError_Success;

    vkGetPhysicalDeviceMemoryProperties(m_physDev, &m_memProperties);

    error = CreateVertexBuffer();
    CHK_ERROR_AND_RETURN(error, "CreateVertexBuffer");
    error = CreateIndexBuffer();
    CHK_ERROR_AND_RETURN(error, "CreateIndexBuffer");
    error = SetupRenderPass();
    CHK_ERROR_AND_RETURN(error, "SetupRenderPass");
    error = CreateShaderStageInfo();
    CHK_ERROR_AND_RETURN(error, "CreateShaderStageInfo");
    error = LoadTextures();
    CHK_ERROR_AND_RETURN(error, "LoadTextures");
    error = CreateUniformBuffer();
    CHK_ERROR_AND_RETURN(error, "CreateUniformBuffer");
    error = CreateDescriptorSets();
    CHK_ERROR_AND_RETURN(error, "CreateDescriptorPool");
    error = SetupPipeline();
    CHK_ERROR_AND_RETURN(error, "SetupPipeline");

    return error;
}

NvError CVulkanSCSceneBase::CreateFramebuffer(VkImageView *attachments,uint32_t uNumAttachment, uint32_t uPacketIndex)
{
    VkFramebufferCreateInfo frameBufferCreateInfo = {};
    frameBufferCreateInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    frameBufferCreateInfo.pNext           = NULL;
    frameBufferCreateInfo.renderPass      = m_renderPass;
    frameBufferCreateInfo.attachmentCount = uNumAttachment;
    frameBufferCreateInfo.pAttachments    = attachments;
    frameBufferCreateInfo.width           = m_uWidth;
    frameBufferCreateInfo.height          = m_uHeight;
    frameBufferCreateInfo.layers          = 1;
    VkResult res = vkCreateFramebuffer(m_dev, &frameBufferCreateInfo, NULL, &m_vkFrameBuffer[uPacketIndex]);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkCreateImageView");

    return NvError_Success;
}

NvError CVulkanSCSceneBase::CreateShaderStageInfo()
{
    // In VKSC, the shader compilation is offline.
    m_stageInfo[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    m_stageInfo[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    m_stageInfo[0].pName  = "main";
    m_stageInfo[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    m_stageInfo[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    m_stageInfo[1].pName  = "main";
    m_stageInfo[0].module = VK_NULL_HANDLE;
    m_stageInfo[1].module = VK_NULL_HANDLE;
#ifndef VULKAN
    m_stageInfo[0].module = VK_NULL_HANDLE;
    m_stageInfo[1].module = VK_NULL_HANDLE;
#else
    std::string sVertexShaderPath   = GetVertexShaderPath();
    std::string sFragmentShaderPath = GetFragmentShaderPath();
    m_stageInfo[0].module = loadSPIRVShader(sVertexShaderPath);
    m_stageInfo[1].module = loadSPIRVShader(sFragmentShaderPath);
#endif // #ifndef VULKAN

    return NvError_Success;
}

NvError CVulkanSCSceneBase::CreateVkImage(uint32_t uWidth, uint32_t uHeight, VkFormat format, VkImageTiling tiling,
    VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width  = uWidth;
    imageInfo.extent.height = uHeight;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels    = 1;
    imageInfo.arrayLayers  = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult res = vkCreateImage(m_dev, &imageInfo, nullptr, &image);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkCreateImage");

    VkMemoryRequirements memRequirements{};
    vkGetImageMemoryRequirements(m_dev, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize  = memRequirements.size;
    allocInfo.memoryTypeIndex = GetMemoryType(m_memProperties, memRequirements.memoryTypeBits, properties);
    res = vkAllocateMemory(m_dev, &allocInfo, nullptr, &imageMemory);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkAllocateMemory");

    res = vkBindImageMemory(m_dev, image, imageMemory, 0);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkBindImageMemory");

    return NvError_Success;
}

NvError CVulkanSCSceneBase::CreateVkBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
    VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size  = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkResult res = vkCreateBuffer(m_dev, &bufferInfo, nullptr, &buffer);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkCreateBuffer");

    VkMemoryRequirements memRequirements{};
    vkGetBufferMemoryRequirements(m_dev, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize  = memRequirements.size;
    allocInfo.memoryTypeIndex = GetMemoryType(m_memProperties, memRequirements.memoryTypeBits, properties);
    res = vkAllocateMemory(m_dev, &allocInfo, nullptr, &bufferMemory);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkAllocateMemory");

    res = vkBindBufferMemory(m_dev, buffer, bufferMemory, 0);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkBindBufferMemory");

    return NvError_Success;
}

NvError CVulkanSCSceneBase::RecordImageLayoutTransitionCmd(VkCommandBuffer &cmdBuffer, VkImage &image,
    VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
        LOG_ERR("unsupported layout transition!");
        return NvError_ResourceError;
    }

    vkCmdPipelineBarrier(
        cmdBuffer,
        sourceStage, destinationStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );

    return NvError_Success;
}

NvError CVulkanSCSceneBase::BeginSingleTimeCommands(VkCommandBuffer &cmdBuffer)
{
    VkResult res = VK_SUCCESS;

    VkCommandBufferAllocateInfo cmdBufAllocateInfo{};
    cmdBufAllocateInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdBufAllocateInfo.pNext              = NULL;
    cmdBufAllocateInfo.commandPool        = m_cmdPool;
    cmdBufAllocateInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdBufAllocateInfo.commandBufferCount = 1;
    res = vkAllocateCommandBuffers(m_dev, &cmdBufAllocateInfo, &cmdBuffer);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkAllocateCommandBuffers");

    res = vkResetCommandPool(m_dev, m_cmdPool, 0);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkResetCommandPool");

    VkCommandBufferBeginInfo cmdBufInfo{};
    cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmdBufInfo.pNext = NULL;
    res = vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkBeginCommandBuffer");

    return NvError_Success;
}

NvError CVulkanSCSceneBase::EndSingleTimeCommands(VkCommandBuffer &cmdBuffer)
{
    VkResult res = VK_SUCCESS;
    VkFence fence = VK_NULL_HANDLE;

    vkEndCommandBuffer(cmdBuffer);
    // Submit copy command buffer to graphics queue
    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmdBuffer;

    VkFenceCreateInfo fenceCreateInfo{};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = 0;
    res = vkCreateFence(m_dev, &fenceCreateInfo, nullptr, &fence);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkCreateFence");

    res = vkQueueSubmit(m_queue, 1, &submitInfo, fence);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkQueueSubmit");
    res = vkWaitForFences(m_dev, 1, &fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkWaitForFences");

    vkDestroyFence(m_dev, fence, nullptr);

    return NvError_Success;
}

#ifndef VULKAN
NvError CVulkanSCSceneBase::ReadPipelineUUID(const std::string& sFileName, char cUUID[VK_UUID_SIZE])
{
    LOG_INFO("Trying to read pipeline UUID from: %s.", sFileName.c_str());
    FILE* pReadFile = fopen(sFileName.c_str(), "rb");

    if (pReadFile) {
        fseek(pReadFile, 0, SEEK_END);
        long lFileSize = ftell(pReadFile);
        if (lFileSize < 0) {
            LOG_ERR("Failed to get the file size");
            fclose(pReadFile);
            return NvError_FileOperationFailed;
        }
        rewind(pReadFile);

        char* pBuffer = (char*)malloc(sizeof(char) * (lFileSize + 1));
        if (pBuffer == nullptr) {
            LOG_ERR("Memory error.");
            fclose(pReadFile);
            return NvError_ResourceError;
        }
        memset(pBuffer, '\0', lFileSize + 1);

        size_t result = fread(pBuffer, sizeof(char), lFileSize, pReadFile);
        if (result != static_cast<size_t>(lFileSize)) {
            LOG_ERR("Reading error.");
            free(pBuffer);
            pBuffer = nullptr;
            fclose(pReadFile);
            return NvError_FileReadFailed;
        }

        size_t index = 0;
        char* p = strstr(pBuffer, "PipelineUUID");
        if (p != NULL) {
            while (*p && index < VK_UUID_SIZE) {
                if (isdigit(*p)) {
                    cUUID[index++] = (char)strtol(p, &p, 10);
                } else {
                    p++;
                }
            }
        }

        free(pBuffer);
        fclose(pReadFile);
    } else {
        LOG_INFO("ReadPipelineUUID fail.");
        return NvError_ResourceError;
    }

    return NvError_Success;
}

#else
VkShaderModule CVulkanSCSceneBase::loadSPIRVShader(std::string &sFilename)
{
    size_t uShaderSize;
    char* pShaderCode = NULL;

    std::ifstream is(sFilename, std::ios::binary | std::ios::in | std::ios::ate);

    if (is.is_open()) {
        uShaderSize = is.tellg();
        is.seekg(0, std::ios::beg);
        // Copy file contents into a buffer
        pShaderCode = new char[uShaderSize];
        is.read(pShaderCode, uShaderSize);
        is.close();
        if(uShaderSize <= 0) {
            LOG_ERR("shaderSize is no more than zero.");
        }
    }

    if (pShaderCode) {
        // Create a new shader module that will be used for pipeline creation
        VkShaderModuleCreateInfo moduleCreateInfo{};
        moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        moduleCreateInfo.codeSize = uShaderSize;
        moduleCreateInfo.pCode = (uint32_t*)pShaderCode;

        VkShaderModule shaderModule;
        VkResult res = vkCreateShaderModule(m_dev, &moduleCreateInfo, NULL, &shaderModule);
        if(res != VK_SUCCESS) {
            LOG_ERR("vkCreateShaderModule fail.");
            return VK_NULL_HANDLE;
        }

        delete[] pShaderCode;
        return shaderModule;
    } else {
        LOG_ERR("Error: Could not open shader file %s.\n", sFilename.c_str());
        return VK_NULL_HANDLE;
    }
}
#endif // #ifndef VULKAN

VkFormat FindDepthFormat(VkPhysicalDevice physDev, VkImageTiling tiling)
{
    const std::vector<VkFormat> candidates =
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};
    VkFormatFeatureFlags features = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;

    for (auto &format : candidates) {
        VkFormatProperties props{};
        vkGetPhysicalDeviceFormatProperties(physDev, format, &props);

        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
            return format;
        } else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }

    LOG_ERR("Find Depth Format fail.");
    return VK_FORMAT_UNDEFINED;
}

uint32_t GetMemoryType(VkPhysicalDeviceMemoryProperties &memProperties,
                        uint32_t uTypeBits, const VkFlags &properties)
{
    for (uint32_t i = 0; i < 32; ++i)
    {
        if ((uTypeBits & 1) == 1)
        {
            if ((memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            {
                return i;
            }
        }
        uTypeBits >>= 1;
    }

    LOG_ERR("No qualify memory type found!\n");
    return UINT32_MAX;
}

std::vector<std::vector<float>> GetPerspectiveMat(float fovy, float fAspect, float fzNear, float fzFar)
{
    float const fTanHalfFovy = tan(fovy / static_cast<float>(2));

    std::vector<std::vector<float>> vfResult(4, std::vector<float>(4, 0));
    vfResult[0][0] = static_cast<float>(1) / (fAspect * fTanHalfFovy);
    vfResult[1][1] = static_cast<float>(1) / (fTanHalfFovy);
    vfResult[2][2] = - (fzFar + fzNear) / (fzFar - fzNear);
    vfResult[2][3] = - static_cast<float>(1);
    vfResult[3][2] = - (static_cast<float>(2) * fzFar * fzNear) / (fzFar - fzNear);
    return vfResult;
}

std::vector<std::vector<float>> MatMul(std::vector<std::vector<float>> &vfLMat, std::vector<std::vector<float>> &vfRMat)
{
    std::vector<std::vector<float>> vfResult(4, std::vector<float>(4, 0));

    vfResult[0][0] = vfLMat[0][0] * vfRMat[0][0] + vfLMat[1][0] * vfRMat[0][1] + vfLMat[2][0] * vfRMat[0][2] + vfLMat[3][0] * vfRMat[0][3];
    vfResult[0][1] = vfLMat[0][1] * vfRMat[0][0] + vfLMat[1][1] * vfRMat[0][1] + vfLMat[2][1] * vfRMat[0][2] + vfLMat[3][1] * vfRMat[0][3];
    vfResult[0][2] = vfLMat[0][2] * vfRMat[0][0] + vfLMat[1][2] * vfRMat[0][1] + vfLMat[2][2] * vfRMat[0][2] + vfLMat[3][2] * vfRMat[0][3];
    vfResult[0][3] = vfLMat[0][3] * vfRMat[0][0] + vfLMat[1][3] * vfRMat[0][1] + vfLMat[2][3] * vfRMat[0][2] + vfLMat[3][3] * vfRMat[0][3];
    vfResult[1][0] = vfLMat[0][0] * vfRMat[1][0] + vfLMat[1][0] * vfRMat[1][1] + vfLMat[2][0] * vfRMat[1][2] + vfLMat[3][0] * vfRMat[1][3];
    vfResult[1][1] = vfLMat[0][1] * vfRMat[1][0] + vfLMat[1][1] * vfRMat[1][1] + vfLMat[2][1] * vfRMat[1][2] + vfLMat[3][1] * vfRMat[1][3];
    vfResult[1][2] = vfLMat[0][2] * vfRMat[1][0] + vfLMat[1][2] * vfRMat[1][1] + vfLMat[2][2] * vfRMat[1][2] + vfLMat[3][2] * vfRMat[1][3];
    vfResult[1][3] = vfLMat[0][3] * vfRMat[1][0] + vfLMat[1][3] * vfRMat[1][1] + vfLMat[2][3] * vfRMat[1][2] + vfLMat[3][3] * vfRMat[1][3];
    vfResult[2][0] = vfLMat[0][0] * vfRMat[2][0] + vfLMat[1][0] * vfRMat[2][1] + vfLMat[2][0] * vfRMat[2][2] + vfLMat[3][0] * vfRMat[2][3];
    vfResult[2][1] = vfLMat[0][1] * vfRMat[2][0] + vfLMat[1][1] * vfRMat[2][1] + vfLMat[2][1] * vfRMat[2][2] + vfLMat[3][1] * vfRMat[2][3];
    vfResult[2][2] = vfLMat[0][2] * vfRMat[2][0] + vfLMat[1][2] * vfRMat[2][1] + vfLMat[2][2] * vfRMat[2][2] + vfLMat[3][2] * vfRMat[2][3];
    vfResult[2][3] = vfLMat[0][3] * vfRMat[2][0] + vfLMat[1][3] * vfRMat[2][1] + vfLMat[2][3] * vfRMat[2][2] + vfLMat[3][3] * vfRMat[2][3];
    vfResult[3][0] = vfLMat[0][0] * vfRMat[3][0] + vfLMat[1][0] * vfRMat[3][1] + vfLMat[2][0] * vfRMat[3][2] + vfLMat[3][0] * vfRMat[3][3];
    vfResult[3][1] = vfLMat[0][1] * vfRMat[3][0] + vfLMat[1][1] * vfRMat[3][1] + vfLMat[2][1] * vfRMat[3][2] + vfLMat[3][1] * vfRMat[3][3];
    vfResult[3][2] = vfLMat[0][2] * vfRMat[3][0] + vfLMat[1][2] * vfRMat[3][1] + vfLMat[2][2] * vfRMat[3][2] + vfLMat[3][2] * vfRMat[3][3];
    vfResult[3][3] = vfLMat[0][3] * vfRMat[3][0] + vfLMat[1][3] * vfRMat[3][1] + vfLMat[2][3] * vfRMat[3][2] + vfLMat[3][3] * vfRMat[3][3];

    return vfResult;
}

std::vector<std::vector<float>> Rotate(std::vector<std::vector<float>> &vfMat, float fAngle, std::vector<float> &v)
{
    float fcos = cos(fAngle);
    float fsin = sin(fAngle);

    float fInvert = 1.0f/(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    std::vector<float> axis = {v[0]*fInvert, v[1] *fInvert, v[2] * fInvert};
    std::vector<float> temp = {(1.0f - fcos) * axis[0], (1.0f - fcos) * axis[1], (1.0f - fcos) * axis[2] };

    std::vector<std::vector<float>> vfRotate(4, std::vector<float>(4, 0));
    vfRotate[0][0] = fcos + temp[0] * axis[0];
    vfRotate[0][1] = temp[0] * axis[1] + fsin * axis[2];
    vfRotate[0][2] = temp[0] * axis[2] - fsin * axis[1];

    vfRotate[1][0] = temp[1] * axis[0] - fsin * axis[2];
    vfRotate[1][1] = fcos + temp[1] * axis[1];
    vfRotate[1][2] = temp[1] * axis[2] + fsin * axis[0];

    vfRotate[2][0] = temp[2] * axis[0] + fsin * axis[1];
    vfRotate[2][1] = temp[2] * axis[1] - fsin * axis[0];
    vfRotate[2][2] = fcos + temp[2] * axis[2];

    std::vector<std::vector<float>> vfResult(4, std::vector<float>(4, 0));
    vfResult[0][0] = vfMat[0][0] * vfRotate[0][0] + vfMat[1][0] * vfRotate[0][1] + vfMat[2][0] * vfRotate[0][2];
    vfResult[0][1] = vfMat[0][1] * vfRotate[0][0] + vfMat[1][1] * vfRotate[0][1] + vfMat[2][1] * vfRotate[0][2];
    vfResult[0][2] = vfMat[0][2] * vfRotate[0][0] + vfMat[1][2] * vfRotate[0][1] + vfMat[2][2] * vfRotate[0][2];
    vfResult[1][0] = vfMat[0][0] * vfRotate[1][0] + vfMat[1][0] * vfRotate[1][1] + vfMat[2][0] * vfRotate[1][2];
    vfResult[1][1] = vfMat[0][1] * vfRotate[1][0] + vfMat[1][1] * vfRotate[1][1] + vfMat[2][1] * vfRotate[1][2];
    vfResult[1][2] = vfMat[0][2] * vfRotate[1][0] + vfMat[1][2] * vfRotate[1][1] + vfMat[2][2] * vfRotate[1][2];
    vfResult[2][0] = vfMat[0][0] * vfRotate[2][0] + vfMat[1][0] * vfRotate[2][1] + vfMat[2][0] * vfRotate[2][2];
    vfResult[2][1] = vfMat[0][1] * vfRotate[2][0] + vfMat[1][1] * vfRotate[2][1] + vfMat[2][1] * vfRotate[2][2];
    vfResult[2][2] = vfMat[0][2] * vfRotate[2][0] + vfMat[1][2] * vfRotate[2][1] + vfMat[2][2] * vfRotate[2][2];
    vfResult[3] = vfMat[3];

    return vfResult;
}
