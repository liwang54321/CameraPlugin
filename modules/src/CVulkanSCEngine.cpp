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

#include <inttypes.h>
#include "CVulkanSCEngine.hpp"
#include "CFontVulkanSC.hpp"
#include "CCarModelVulkanSC.hpp"

#ifndef VULKAN
const std::vector<const char *> vDeviceExtensions = { VK_NV_EXTERNAL_MEMORY_SCI_BUF_EXTENSION_NAME,
                                                      VK_NV_EXTERNAL_SCI_SYNC_2_EXTENSION_NAME };
constexpr uint32_t uMaxCommandBuffersFromPool = 64U;
constexpr uint32_t uResourceLayerCount = 1U;
constexpr uint32_t uResourceLevelCount = 1U;

#else
static const char *jsonGenLayerName = "VK_LAYER_KHRONOS_json_gen";
#endif // #ifndef VULKAN

NvError CVulkanSCEngine::Init(uint32_t uWidth, uint32_t uHeight, bool bUseVkSemaphore, std::string &sColorType)
{
    LOG_DBG("Enter: CVulkanSCEngine::Init()\n");

    NvError error = NvError_Success;

    m_inputImageFormat = ColorToVkFormat(sColorType);
    if (m_inputImageFormat == VK_FORMAT_UNDEFINED) {
        LOG_ERR("sColorToVkFormat failed.");
        return NvError_BadValue;
    }

    m_uWidth  = uWidth;
    m_uHeight = uHeight;
    m_bUseVkSemaphore = bUseVkSemaphore;

    //Create a Vulkan instance
    error = CreateInstanceAndDevice();
    CHK_ERROR_AND_RETURN(error, "CreateInstanceAndDevice");

    error = SetupImage();
    CHK_ERROR_AND_RETURN(error, "SetupImage");

#ifndef VULKAN
    error = CreatePipelineCaches();
    CHK_ERROR_AND_RETURN(error, "CreatePipelineCaches");
#endif

    error = CreateCommandBuffers();
    CHK_ERROR_AND_RETURN(error, "CreateCommandBuffers");

    std::shared_ptr<CVulkanSCSceneBase> spFontVKSC =
        std::make_shared<CFontVulkanSC>(m_dev, m_physDev,
            m_queue, m_cmdPool, m_inputImageFormat, m_uWidth, m_uHeight
#ifndef VULKAN
        , m_pipelineCache
#endif
        );

    if(spFontVKSC == nullptr) {
        LOG_ERR("CFontVulkanSC creation failed.");
        return NvError_InsufficientMemory;
    }
    m_vVulkanSCSences.push_back(spFontVKSC);

    std::shared_ptr<CVulkanSCSceneBase> spCarVKSC =
        std::make_shared<CCarModelVulkanSC>(m_dev, m_physDev,
            m_queue, m_cmdPool, m_inputImageFormat, m_uWidth, m_uHeight
#ifndef VULKAN
        , m_pipelineCache
#endif
        );

    if(spCarVKSC == nullptr) {
        LOG_ERR("CCarModelVulkanSC creation failed.");
        return NvError_InsufficientMemory;
    }
    m_vVulkanSCSences.push_back(spCarVKSC);

    for(auto &scene : m_vVulkanSCSences) {
        error = scene->Init();
        CHK_ERROR_AND_RETURN(error, "scene->Init");
    }

    if (!m_bUseVkSemaphore) {
        VkFenceCreateInfo fenceCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0, // initial state
        };

        VkResult res = VK_SUCCESS;
        res = vkCreateFence(m_dev, &fenceCreateInfo, nullptr, &m_vkSignalFence);
        CHK_VKSCSTATUS_AND_RETURN(res, "vkCreateFence for m_vkSignalFence");

        res = vkCreateFence(m_dev, &fenceCreateInfo, nullptr, &m_vkWaiterFence);
        CHK_VKSCSTATUS_AND_RETURN(res, "vkCreateFence for m_vkWaiterFence");
    }

#ifndef VULKAN
    // Get NvSciSync related APIs
    GET_INSTANCE_DEVICE_PROC_ADDR(GetPhysicalDeviceSciSyncAttributesNV);
    GET_DEVICE_PROC_ADDR(CreateSemaphoreSciSyncPoolNV);
    GET_DEVICE_PROC_ADDR(GetFenceSciSyncFenceNV);
    GET_DEVICE_PROC_ADDR(GetFenceSciSyncObjNV);
    GET_DEVICE_PROC_ADDR(ImportFenceSciSyncFenceNV);
    GET_DEVICE_PROC_ADDR(ImportFenceSciSyncObjNV);

    // Get NvSciBuf related APIs
    GET_INSTANCE_DEVICE_PROC_ADDR(GetPhysicalDeviceExternalMemorySciBufPropertiesNV);
    GET_INSTANCE_DEVICE_PROC_ADDR(GetPhysicalDeviceSciBufAttributesNV);
#endif

    LOG_DBG("Exit: CVulkanSCEngine::Init()\n");
    return NvError_Success;
}

NvError CVulkanSCEngine::GetSciBufAttributesNV(NvSciBufAttrList *pBufAttrList)
{
    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_ReadWrite;
    NvSciBufAttrValImageScanType planescantype = NvSciBufScan_ProgressiveType;
    bool bNeedCpuAccess = (m_memRequirements.memoryTypeBits & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    bool bNeedCpuCached = (m_memRequirements.memoryTypeBits & VK_MEMORY_PROPERTY_HOST_CACHED_BIT);

    NvSciBufAttrKeyValuePair attributes[] = {
        { NvSciBufImageAttrKey_PlaneScanType, (void *)&planescantype, sizeof(planescantype) },
        { NvSciBufGeneralAttrKey_Types, (void *)&bufType, sizeof(bufType) },
        { NvSciBufGeneralAttrKey_NeedCpuAccess, (void *)&bNeedCpuAccess, sizeof(bNeedCpuAccess) },
        { NvSciBufGeneralAttrKey_EnableCpuCache, (void *)&bNeedCpuCached, sizeof(bNeedCpuCached) },
        { NvSciBufGeneralAttrKey_RequiredPerm, (void *)&perm, sizeof(perm) },
    };

    NvSciError sciErr = NvSciBufAttrListSetAttrs(*pBufAttrList, attributes, ARRAY_SIZE(attributes));
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciBufAttrListSetAttrs");

#ifndef VULKAN
    VkResult res = m_vkscSciInterface.fpGetPhysicalDeviceSciBufAttributesNV(m_physDev, *pBufAttrList);
    CHK_VKSCSTATUS_AND_RETURN(res, "fpGetPhysicalDeviceSciBufAttributesNV");
#endif
    return NvError_Success;
}

NvError CVulkanSCEngine::GetSciSyncAttributesNV(VKSCSyncType vkscSyncType, NvSciSyncAttrList *pAttrList)
{
#ifndef VULKAN
    VkSciSyncPrimitiveTypeNV primitiveType = VK_SCI_SYNC_PRIMITIVE_TYPE_FENCE_NV;
    if (m_bUseVkSemaphore) {
        primitiveType = VK_SCI_SYNC_PRIMITIVE_TYPE_SEMAPHORE_NV;
    }

    VkSciSyncClientTypeNV clientType = VK_SCI_SYNC_CLIENT_TYPE_SIGNALER_NV;
    if (vkscSyncType == VKSCSyncType::VulkanSCWaiter) {
        clientType = VK_SCI_SYNC_CLIENT_TYPE_WAITER_NV;
    }

    VkSciSyncAttributesInfoNV vkSciSyncAttributesInfo = { .sType = VK_STRUCTURE_TYPE_SCI_SYNC_ATTRIBUTES_INFO_NV,
                                                          .pNext = nullptr,
                                                          .clientType = clientType,
                                                          .primitiveType = primitiveType };

    VkResult res =
        m_vkscSciInterface.fpGetPhysicalDeviceSciSyncAttributesNV(m_physDev, &vkSciSyncAttributesInfo, *pAttrList);
    CHK_VKSCSTATUS_AND_RETURN(res, "fpGetPhysicalDeviceSciSyncAttributesNV signalerAttrList");
#else
    NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_SignalOnly;
    if (vkscSyncType == VKSCSyncType::VulkanSCWaiter) {
        cpuPerm = NvSciSyncAccessPerm_WaitOnly;
    }

    bool bCpuSync = true;
    /* Fill attribute list for CPU signaling*/
    NvSciSyncAttrKeyValuePair cpuKeyVals[] = { { NvSciSyncAttrKey_NeedCpuAccess, &bCpuSync, sizeof(bCpuSync) },
                                               { NvSciSyncAttrKey_RequiredPerm, &cpuPerm, sizeof(cpuPerm) } };

    auto sciErr = NvSciSyncAttrListSetAttrs(*pAttrList, cpuKeyVals, ARRAY_SIZE(cpuKeyVals));
    CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncAttrListSetAttrs");
#endif

    return NvError_Success;
}

NvError CVulkanSCEngine::RegisterBufObj(uint32_t uPacketIndex, NvSciBufObj bufObj)
{
    BufferAttrs bufAttrs;
    NvError error = PopulateBufAttr(bufObj, bufAttrs);
    CHK_ERROR_AND_RETURN(error, "PopulateBufAttr");

    VkMemoryAllocateInfo memAlloc{};
    memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memAlloc.pNext = nullptr;
    // Block linear
    memAlloc.memoryTypeIndex = GetMemoryType(m_memProperties, m_memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    memAlloc.allocationSize  = m_memRequirements.size;
    if(m_memRequirements.size > bufAttrs.size) {
        LOG_ERR("Buffer size %" PRIu64 " is less than VkImage required memory size %" PRIu64, bufAttrs.size,
                m_memRequirements.size);
        return NvError_BadValue;
    }

#ifdef VULKAN
    memAlloc.pNext = nullptr;
#else
    VkImportMemorySciBufInfoNV importSciBufInfo{};
    importSciBufInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_SCI_BUF_INFO_NV;
    importSciBufInfo.pNext = nullptr;
    importSciBufInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_SCI_BUF_BIT_NV;
    importSciBufInfo.handle = bufObj;
    memAlloc.pNext = &importSciBufInfo;

    // Ignore the allocate size when importing NvSciBufObj
    memAlloc.allocationSize = 0;
#endif

    VkResult res = vkAllocateMemory(m_dev, &memAlloc, nullptr, &m_vkImportImageMem[uPacketIndex]);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkAllocateMemory");

    res = vkBindImageMemory(m_dev, m_vkImportImage[uPacketIndex], m_vkImportImageMem[uPacketIndex], 0);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkBindImageMemory");

    VkImageViewCreateInfo colorAttachmentView = {};
    colorAttachmentView.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    colorAttachmentView.pNext = nullptr;
    colorAttachmentView.format = m_inputImageFormat;
    colorAttachmentView.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B,
                                       VK_COMPONENT_SWIZZLE_A };
    colorAttachmentView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    colorAttachmentView.subresourceRange.baseMipLevel = 0;
    colorAttachmentView.subresourceRange.levelCount   = uResourceLevelCount;
    colorAttachmentView.subresourceRange.baseArrayLayer = 0;
    colorAttachmentView.subresourceRange.layerCount     = uResourceLayerCount;
    colorAttachmentView.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    colorAttachmentView.flags                           = 0;
    colorAttachmentView.image                           = m_vkImportImage[uPacketIndex];
    res = vkCreateImageView(m_dev, &colorAttachmentView, nullptr, &m_vkImportImageView[uPacketIndex]);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkCreateImageView");

    VkImageView attachments[2] = { m_vkImportImageView[uPacketIndex], m_vkDepthImageView[uPacketIndex] };
    for(auto &scene : m_vVulkanSCSences) {
        error = scene->CreateFramebuffer(attachments, sizeof(attachments)/sizeof(attachments[0]), uPacketIndex);
        CHK_ERROR_AND_RETURN(error, "PopulateBufAttr");
    }

    return NvError_Success;
}

NvError CVulkanSCEngine::RegisterSyncObj(VKSCSyncType vkscSyncType, NvSciSyncObj bufObj)
{
#ifndef VULKAN
    if (m_bUseVkSemaphore) {
        VkSemaphoreSciSyncPoolCreateInfoNV createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SCI_SYNC_POOL_CREATE_INFO_NV;
        createInfo.pNext = nullptr;
        createInfo.handle = bufObj;

        VkResult res = m_vkscSciInterface.fpCreateSemaphoreSciSyncPoolNV(m_dev, &createInfo, nullptr,
                                                                         &m_semaphoreSciSyncWaiterPool);
        CHK_VKSCSTATUS_AND_RETURN(res, "fpCreateSemaphoreSciSyncPoolNV");

        if (vkscSyncType == VKSCSyncType::VulkanSCSignaler) {
            m_signalSyncObj = bufObj;
        }
    } else {
        VkFence vkFence = m_vkWaiterFence;
        if (vkscSyncType == VKSCSyncType::VulkanSCSignaler) {
            vkFence = m_vkSignalFence;
        }
        VkImportFenceSciSyncInfoNV importSciSyncInfo{};
        importSciSyncInfo.sType = VK_STRUCTURE_TYPE_IMPORT_FENCE_SCI_SYNC_INFO_NV;
        importSciSyncInfo.pNext = nullptr;
        importSciSyncInfo.fence = vkFence;
        importSciSyncInfo.handle = bufObj;
        importSciSyncInfo.handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_SCI_SYNC_OBJ_BIT_NV;

        VkResult res = m_vkscSciInterface.fpImportFenceSciSyncObjNV(m_dev, &importSciSyncInfo);
        CHK_VKSCSTATUS_AND_RETURN(res, "fpImportFenceSciSyncObjNV");
    }
#endif

    return NvError_Success;
}

NvError CVulkanSCEngine::InsertFenceSciSyncPrefence(NvSciSyncFence *pPrefence)
{
#ifndef VULKAN
    if (m_bUseVkSemaphore) {
        auto error = GetSemaphoreFromVkSciSyncObj(pPrefence, m_vkWaiterSem, m_semaphoreSciSyncWaiterPool);
        CHK_ERROR_AND_RETURN(error, "GetSemaphoreFromVkSciSyncObj");

        if (m_vkWaiterSem == VK_NULL_HANDLE) {
            LOG_ERR("Fail to Get Semaphore from VkSciSyncObj.");
            return NvError_BadValue;
        }

        uint64_t fenceId = 0;
        NvSciError sciErr = NvSciSyncFenceExtractFence(pPrefence, &fenceId, &m_uWaitVal);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncFenceExtractFence");

        m_semaphoreInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        m_semaphoreInfo.pNext = nullptr;
        m_semaphoreInfo.waitSemaphoreValueCount = 1;
        m_semaphoreInfo.pWaitSemaphoreValues = &m_uWaitVal;
    } else {
        VkImportFenceSciSyncInfoNV importSciSyncInfo{};
        importSciSyncInfo.sType = VK_STRUCTURE_TYPE_IMPORT_FENCE_SCI_SYNC_INFO_NV;
        importSciSyncInfo.pNext = nullptr;
        importSciSyncInfo.fence = m_vkWaiterFence;
        importSciSyncInfo.handle = pPrefence;
        importSciSyncInfo.handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_SCI_SYNC_FENCE_BIT_NV;
        VkResult res = m_vkscSciInterface.fpImportFenceSciSyncFenceNV(m_dev, &importSciSyncInfo);
        CHK_VKSCSTATUS_AND_RETURN(res, "fpImportFenceSciSyncFenceNV");
        res = vkWaitForFences(m_dev, 1, &m_vkWaiterFence, true, DEFAULT_FENCE_TIMEOUT);
        CHK_VKSCSTATUS_AND_RETURN(res, "vkWaitForFences");
        res = vkResetFences(m_dev, 1, &m_vkWaiterFence);
        CHK_VKSCSTATUS_AND_RETURN(res, "vkWaitForFences");
    }
#endif

    return NvError_Success;
}

NvError CVulkanSCEngine::GetEofFenceSciSyncFence(NvSciSyncFence *pPostfence)
{
#ifndef VULKAN
    if (m_bUseVkSemaphore) {
        *pPostfence = m_postfence;
        m_postfence = NvSciSyncFenceInitializer;
    } else {
        VkFenceGetSciSyncInfoNV getSciSyncInfo{};
        getSciSyncInfo.sType = VK_STRUCTURE_TYPE_FENCE_GET_SCI_SYNC_INFO_NV;
        getSciSyncInfo.pNext = nullptr;
        getSciSyncInfo.fence = m_vkSignalFence;
        getSciSyncInfo.handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_SCI_SYNC_FENCE_BIT_NV;

        VkResult res = m_vkscSciInterface.fpGetFenceSciSyncFenceNV(m_dev, &getSciSyncInfo, pPostfence);
        CHK_VKSCSTATUS_AND_RETURN(res, "fpGetFenceSciSyncFenceNV");
    }
#endif

    return NvError_Success;
}

void CVulkanSCEngine::DeInit()
{
    for(auto &scene : m_vVulkanSCSences) {
        scene.reset();
    }

    if (m_dstImage != VK_NULL_HANDLE) {
        vkDestroyImage(m_dev, m_dstImage, nullptr);
#ifdef VULKAN
        vkFreeMemory(m_dev, m_dstDevMem, nullptr);
#endif
    }

    for(uint32_t i = 0; i < MAX_NUM_PACKETS; ++i) {
        if(m_vkImportImage[i] != VK_NULL_HANDLE) {
            vkDestroyImageView(m_dev, m_vkImportImageView[i], nullptr);
            vkDestroyImage(m_dev, m_vkImportImage[i], nullptr);
#ifdef VULKAN
            vkFreeMemory(m_dev, m_vkImportImageMem[i], nullptr);
#endif
        }

        if(m_vkDepthImage[i] != VK_NULL_HANDLE) {
            vkDestroyImageView(m_dev, m_vkDepthImageView[i], nullptr);
            vkDestroyImage(m_dev, m_vkDepthImage[i], nullptr);
#ifdef VULKAN
            vkFreeMemory(m_dev, m_vkDepthDevMem[i], nullptr);
#endif
        }
    }

    if (m_vkCmdBuffer != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(m_dev, m_cmdPool, 1, &m_vkCmdBuffer);
    }

    if (m_vkCopyCmdBuffer != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(m_dev, m_cmdPool, 1, &m_vkCopyCmdBuffer);
    }

#ifdef VULKAN
    if(m_cmdPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_dev, m_cmdPool, nullptr);
    }
#endif

#ifndef VULKAN
    if (m_pipelineCache != VK_NULL_HANDLE) {
        vkDestroyPipelineCache(m_dev, m_pipelineCache, nullptr);
    }
#endif

    if(m_dev != VK_NULL_HANDLE) {
        vkDestroyDevice(m_dev, nullptr);
    }

    if (m_instance != VK_NULL_HANDLE) {
        vkDestroyInstance(m_instance, nullptr);
    }
}

NvError CVulkanSCEngine::CreateInstanceAndDevice()
{
    VkResult res = VK_SUCCESS;
    NvError error = NvError_Success;

#ifdef VULKAN
    error = CheckJsonLayerSupport();
    CHK_ERROR_AND_RETURN(error, "CheckJsonLayerSupport");
#endif // #endif VULKAN

    //Create a Vulkan instance
    VkInstanceCreateInfo instanceCreateInfo = {};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

#ifdef VULKAN
    VkLayerInstanceLink layerLink = { .pNext = nullptr,
                                      .pfnNextGetInstanceProcAddr = nullptr, // this has to be really vulkan entry point
                                      .pfnNextGetPhysicalDeviceProcAddr = nullptr };

    // enable json_gen layer for instance
    VkLayerInstanceCreateInfo layerInfo = {};
    layerInfo.sType = VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO, layerInfo.function = VK_LAYER_LINK_INFO,
    layerInfo.u.pLayerInfo = &layerLink;

    instanceCreateInfo.pNext = &layerInfo;
    instanceCreateInfo.enabledLayerCount = 1;
    instanceCreateInfo.ppEnabledLayerNames = &jsonGenLayerName;
#endif // ifdef VULKAN

    res = vkCreateInstance(&instanceCreateInfo, nullptr, &m_instance);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkCreateInstance");

    uint32_t uDeviceCount = 0;
    res = vkEnumeratePhysicalDevices(m_instance, &uDeviceCount, nullptr);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkEnumeratePhysicalDevices");
    if (uDeviceCount == 0) {
#ifdef VULKAN
        LOG_ERR("No GPU device found with Vulkan support.");
#else
        LOG_ERR("No GPU device found with VulkanSC support.");
#endif
        return NvError_BadValue;
    }

    std::vector<VkPhysicalDevice> vDevices(uDeviceCount);
    res = vkEnumeratePhysicalDevices(m_instance, &uDeviceCount, vDevices.data());
    CHK_VKSCSTATUS_AND_RETURN(res, "vkEnumeratePhysicalDevices");
    m_physDev = vDevices[0];

    vkGetPhysicalDeviceProperties(m_physDev, &m_deviceProperties);
    vkGetPhysicalDeviceMemoryProperties(m_physDev, &m_memProperties);

    // Create device
    error = FindQueueFamilies(m_physDev);
    CHK_ERROR_AND_RETURN(error, "FindQueueFamilies");

    float fQueuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                          nullptr,
                                          0,                   // flags
                                          m_uQueueFamilyIndex, // queueFamilyIndex
                                          1,                   // queueCount
                                          &fQueuePriority };

    VkDeviceCreateInfo deviceInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queueInfo,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
#ifndef VULKAN
        .enabledExtensionCount = (uint32_t)vDeviceExtensions.size(),
        .ppEnabledExtensionNames = vDeviceExtensions.data(),
#else
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = nullptr,
#endif
        .pEnabledFeatures = nullptr,
    };

#ifndef VULKAN
    error = LoadPipelineCaches();
    CHK_ERROR_AND_RETURN(error, "LoadPipelineCaches");
    error = FillDeviceObjResInfo();
    CHK_ERROR_AND_RETURN(error, "FillDeviceObjResInfo");
    deviceInfo.pNext = &m_devObjectResCreateInfo;
#else
    // enable json_gen layer for device
    VkLayerDeviceLink layerDevInfo = {};

    VkLayerDeviceCreateInfo devLayerCreateInfo = {};
    devLayerCreateInfo.sType = VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO;
    devLayerCreateInfo.pNext = &layerDevInfo;
    devLayerCreateInfo.function = VK_LAYER_LINK_INFO;
    devLayerCreateInfo.u.pLayerInfo = &layerDevInfo;

    deviceInfo.pNext = &devLayerCreateInfo;
    deviceInfo.enabledLayerCount = 1;
    deviceInfo.ppEnabledLayerNames = &jsonGenLayerName;
#endif

    res = vkCreateDevice(m_physDev, &deviceInfo, nullptr, &m_dev);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkCreateDevice");

    vkGetDeviceQueue(m_dev, m_uQueueFamilyIndex, 0, &m_queue);

    return NvError_Success;
}

NvError CVulkanSCEngine::CreateCommandBuffers()
{
    VkResult res;
#ifndef VULKAN
    VkCommandPoolMemoryReservationCreateInfo memReserveInfo{};
    memReserveInfo.sType                    = VK_STRUCTURE_TYPE_COMMAND_POOL_MEMORY_RESERVATION_CREATE_INFO;
    memReserveInfo.pNext                    = nullptr;
    memReserveInfo.commandPoolReservedSize  = 1536ULL * 1024ULL*4ULL; // A 6 MB placeholder (default)

    // This value cannot exceed the VkDeviceObjectReservationCreateInfo::commandBufferRequestCount
    memReserveInfo.commandPoolMaxCommandBuffers = uMaxCommandBuffersFromPool;
#endif

    // Command pool
    VkCommandPoolCreateInfo cmdPoolCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
#ifndef VULKAN
        .pNext = &memReserveInfo,
#endif
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = m_uQueueFamilyIndex,
    };
    res = vkCreateCommandPool(m_dev, &cmdPoolCreateInfo, nullptr, &m_cmdPool);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkCreateCommandPool");

    VkCommandBufferAllocateInfo cmdBufAllocateInfo = {};
    cmdBufAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdBufAllocateInfo.pNext = nullptr;
    cmdBufAllocateInfo.commandPool = m_cmdPool;
    cmdBufAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdBufAllocateInfo.commandBufferCount = 1;

    res = vkAllocateCommandBuffers(m_dev, &cmdBufAllocateInfo, &m_vkCmdBuffer);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkCreateCommandPool");

    res = vkAllocateCommandBuffers(m_dev, &cmdBufAllocateInfo, &m_vkCopyCmdBuffer);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkCreateCommandPool");

    return NvError_Success;
}

NvError CVulkanSCEngine::SetupImage()
{
    VkResult res = VK_SUCCESS;
    // Create VkImage Array
#ifndef VULKAN
    VkExternalMemoryImageCreateInfo externalMemInfo = { .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
                                                        .pNext = nullptr,
                                                        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_SCI_BUF_BIT_NV };
#endif

    VkImageCreateInfo imageInfo = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
#ifdef VULKAN
                                    .pNext = nullptr,
#else
                                    .pNext = &externalMemInfo,
#endif
                                    .flags = 0,
                                    .imageType = VK_IMAGE_TYPE_2D,
                                    .format = m_inputImageFormat,
                                    .extent = { m_uWidth, m_uHeight, 1 },
                                    .mipLevels = 1,
                                    .arrayLayers = 1,
                                    .samples = VK_SAMPLE_COUNT_1_BIT,
                                    .tiling = VK_IMAGE_TILING_OPTIMAL,
                                    .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                                    .queueFamilyIndexCount = 0,
                                    .pQueueFamilyIndices = nullptr,
                                    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED };

    for (uint32_t i = 0; i < MAX_NUM_PACKETS; ++i) {
        res = vkCreateImage(m_dev, &imageInfo, nullptr, &m_vkImportImage[i]);
        CHK_VKSCSTATUS_AND_RETURN(res, "vkCreateImage");
    }

    // Get memory requirements for VkImage
    vkGetImageMemoryRequirements(m_dev, m_vkImportImage[0], &m_memRequirements);
    if(__builtin_popcountl(m_memRequirements.alignment) != 1) {
        LOG_ERR("Alignment has to be power of 2!");
        return NvError_InvalidSize;
    }

    // Create dst copy VkImage and bind deviceMemory
    imageInfo.pNext = nullptr;
    imageInfo.tiling = VK_IMAGE_TILING_LINEAR;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    res = vkCreateImage(m_dev, &imageInfo, nullptr, &m_dstImage);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkCreateImage");

    vkGetImageMemoryRequirements(m_dev, m_dstImage, &m_dstMemRequirements);

    VkMemoryAllocateInfo memAlloc{};
    memAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memAlloc.pNext = nullptr;
    memAlloc.allocationSize  = m_dstMemRequirements.size;
    memAlloc.memoryTypeIndex = GetMemoryType(m_memProperties, m_dstMemRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    res = vkAllocateMemory(m_dev, &memAlloc, nullptr, &m_dstDevMem);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkAllocateMemory");

    res = vkBindImageMemory(m_dev, m_dstImage, m_dstDevMem, 0);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkBindImageMemory");

    // Create Depth Image and bind deviceMemory
    VkFormat depthFormat = FindDepthFormat(m_physDev, VK_IMAGE_TILING_OPTIMAL);
    imageInfo.pNext  = nullptr;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage  = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    imageInfo.format = depthFormat;
    for(uint32_t i = 0; i < MAX_NUM_PACKETS; ++i) {
        res = vkCreateImage(m_dev, &imageInfo, nullptr, &m_vkDepthImage[i]);
        CHK_VKSCSTATUS_AND_RETURN(res, "vkCreateImage");

        VkMemoryRequirements depthMemRequirements = {};
        vkGetImageMemoryRequirements(m_dev, m_vkDepthImage[i], &depthMemRequirements);
        memAlloc.allocationSize  = depthMemRequirements.size;
        memAlloc.memoryTypeIndex = GetMemoryType(m_memProperties, depthMemRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        res = vkAllocateMemory(m_dev, &memAlloc, nullptr, &m_vkDepthDevMem[i]);
        CHK_VKSCSTATUS_AND_RETURN(res, "vkAllocateMemory");

        res = vkBindImageMemory(m_dev, m_vkDepthImage[i], m_vkDepthDevMem[i], 0);
        CHK_VKSCSTATUS_AND_RETURN(res, "vkBindImageMemory");

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = m_vkDepthImage[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format   = depthFormat;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        viewInfo.subresourceRange.baseMipLevel   = 0;
        viewInfo.subresourceRange.levelCount     = uResourceLevelCount;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount     = uResourceLayerCount;

        res = vkCreateImageView(m_dev, &viewInfo, nullptr, &m_vkDepthImageView[i]);
        CHK_VKSCSTATUS_AND_RETURN(res, "vkCreateImageView");
    }
    return NvError_Success;
}

NvError CVulkanSCEngine::FindQueueFamilies(VkPhysicalDevice &device)
{
    uint32_t uQueueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &uQueueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> vQueueFamilies(uQueueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &uQueueFamilyCount, vQueueFamilies.data());

    uint32_t uQueueFamilyIndex = 0;
    for (const VkQueueFamilyProperties &queueFamily : vQueueFamilies) {
        if (queueFamily.queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT)) {
            m_uQueueFamilyIndex = uQueueFamilyIndex;
            return NvError_Success;
        }
        ++uQueueFamilyIndex;
    }

    LOG_ERR("Fail to find queueFamily index.");
    return NvError_BadValue;
}

NvError CVulkanSCEngine::Draw(uint32_t uPacketIndex, std::string &sTimeStampToDraw)
{
    NvError error = NvError_Success;
#ifndef VULKAN
    m_uThresholdVal = 0;
    if (m_bUseVkSemaphore) {
        // Instruct VulkanSC to signal fence
        NvSciError sciErr = NvSciSyncObjGenerateFence(m_signalSyncObj, &m_postfence);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncObjGenerateFence");

        uint64_t fenceId = 0;
        sciErr = NvSciSyncFenceExtractFence(&m_postfence, &fenceId, &m_uThresholdVal);
        CHK_NVSCISTATUS_AND_RETURN(sciErr, "NvSciSyncFenceExtractFence");

        error = GetSemaphoreFromVkSciSyncObj(&m_postfence, m_vkSignalerSem, m_semaphoreSciSyncSignalerPool);
        CHK_ERROR_AND_RETURN(error, "GetSemaphoreFromVkSciSyncObj");

        m_semaphoreInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        m_semaphoreInfo.signalSemaphoreValueCount = 1;
        m_semaphoreInfo.pSignalSemaphoreValues = &m_uThresholdVal;
    } else {
        VkResult res = vkResetFences(m_dev, 1, &m_vkSignalFence);
        CHK_VKSCSTATUS_AND_RETURN(res, "vkWaitForFences");
    }
#endif

    // VulkanSC only support reset cmd pool.
    VkResult res = vkResetCommandPool(m_dev, m_cmdPool, 0);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkResetCommandPool");

    VkCommandBufferBeginInfo cmdBufInfo = {};
    cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmdBufInfo.pNext = nullptr;

    res = vkBeginCommandBuffer(m_vkCmdBuffer, &cmdBufInfo);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkBeginCommandBuffer");

    for(auto &scene : m_vVulkanSCSences) {
        error = scene->RecordSceneDrawCommand(m_vkCmdBuffer, uPacketIndex, &sTimeStampToDraw);
        CHK_ERROR_AND_RETURN(error, "RecordSceneDrawCommand");
    }

    res = vkEndCommandBuffer(m_vkCmdBuffer);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkEndCommandBuffer");

    // submit
    VkSubmitInfo submitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 0,
        .pWaitSemaphores = nullptr,
        .pWaitDstStageMask = nullptr,
        .commandBufferCount = 1,
        .pCommandBuffers = &m_vkCmdBuffer,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = nullptr,
    };
#ifndef VULKAN
    VkPipelineStageFlags waitDstStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    if (m_bUseVkSemaphore) {
        submitInfo.pNext = &m_semaphoreInfo;

        if (m_vkWaiterSem != VK_NULL_HANDLE) {
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = &m_vkWaiterSem;
            submitInfo.pWaitDstStageMask = &waitDstStageMask;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &m_vkSignalerSem;
        }
        res = vkQueueSubmit(m_queue, 1, &submitInfo, nullptr);
    } else {
        res = vkQueueSubmit(m_queue, 1, &submitInfo, m_vkSignalFence);
    }
    CHK_VKSCSTATUS_AND_RETURN(res, "vkQueueSubmit");
#else
    res = vkQueueSubmit(m_queue, 1, &submitInfo, nullptr);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkQueueSubmit");
#endif

    res = vkDeviceWaitIdle(m_dev);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkDeviceWaitIdle");

#ifndef VULKAN
    // In VK_NV_VULKANexternal_sci_sync2, we must destroy the VkSemaphore
    // after the submission of the job
    if (m_vkWaiterSem != VK_NULL_HANDLE) {
        vkDestroySemaphore(m_dev, m_vkWaiterSem, nullptr);
        m_vkWaiterSem = VK_NULL_HANDLE;
    }

    if (m_vkSignalerSem != VK_NULL_HANDLE) {
        vkDestroySemaphore(m_dev, m_vkSignalerSem, nullptr);
        m_vkSignalerSem = VK_NULL_HANDLE;
    }
#endif

    return NvError_Success;
}

#ifndef VULKAN
NvError CVulkanSCEngine::FillDeviceObjResInfo()
{
    m_devObjectResCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_OBJECT_RESERVATION_CREATE_INFO,
    m_devObjectResCreateInfo.pNext = nullptr;
    m_devObjectResCreateInfo.pipelineCacheCreateInfoCount = 1;
    m_devObjectResCreateInfo.pPipelineCacheCreateInfos = &m_pipelineCacheCreateInfo;
    m_devObjectResCreateInfo.pipelinePoolSizeCount = 1;
    m_devObjectResCreateInfo.pPipelinePoolSizes = &m_pipelinePoolSizes;
    m_devObjectResCreateInfo.semaphoreRequestCount = 2;
    m_devObjectResCreateInfo.commandBufferRequestCount = 64;
    m_devObjectResCreateInfo.fenceRequestCount = 64;
    m_devObjectResCreateInfo.deviceMemoryRequestCount = 180;
    m_devObjectResCreateInfo.bufferRequestCount = 180;
    m_devObjectResCreateInfo.imageRequestCount = 128;
    m_devObjectResCreateInfo.eventRequestCount = 0;
    m_devObjectResCreateInfo.queryPoolRequestCount = 0;
    m_devObjectResCreateInfo.bufferViewRequestCount = 0;
    m_devObjectResCreateInfo.imageViewRequestCount = 128;
    m_devObjectResCreateInfo.layeredImageViewRequestCount = 1;
    m_devObjectResCreateInfo.pipelineCacheRequestCount = 32;
    m_devObjectResCreateInfo.pipelineLayoutRequestCount = 32;
    m_devObjectResCreateInfo.renderPassRequestCount = 32;
    m_devObjectResCreateInfo.graphicsPipelineRequestCount = 32;
    m_devObjectResCreateInfo.computePipelineRequestCount = 0;
    m_devObjectResCreateInfo.descriptorSetLayoutRequestCount = 32;
    m_devObjectResCreateInfo.samplerRequestCount = 32;
    m_devObjectResCreateInfo.descriptorPoolRequestCount = 32;
    m_devObjectResCreateInfo.descriptorSetRequestCount = 128;
    m_devObjectResCreateInfo.framebufferRequestCount = 128;
    m_devObjectResCreateInfo.commandPoolRequestCount = 1;
    m_devObjectResCreateInfo.samplerYcbcrConversionRequestCount = 0;
    m_devObjectResCreateInfo.surfaceRequestCount = 0;
    m_devObjectResCreateInfo.swapchainRequestCount = 0;
    m_devObjectResCreateInfo.displayModeRequestCount = 0;
    m_devObjectResCreateInfo.subpassDescriptionRequestCount = 0;
    m_devObjectResCreateInfo.descriptorSetLayoutBindingRequestCount = 64;
    m_devObjectResCreateInfo.attachmentDescriptionRequestCount = 0;
    m_devObjectResCreateInfo.descriptorSetLayoutBindingLimit = 64;
    m_devObjectResCreateInfo.maxImageViewMipLevels = 1;
    m_devObjectResCreateInfo.maxImageViewArrayLayers = 0;
    m_devObjectResCreateInfo.maxLayeredImageViewMipLevels = 0;
    m_devObjectResCreateInfo.maxOcclusionQueriesPerPool = 0;
    m_devObjectResCreateInfo.maxPipelineStatisticsQueriesPerPool = 0;
    m_devObjectResCreateInfo.maxTimestampQueriesPerPool = 0;

    m_devSemaphoreSciSyncPoolResCreateInfo.sType =
        VK_STRUCTURE_TYPE_DEVICE_SEMAPHORE_SCI_SYNC_POOL_RESERVATION_CREATE_INFO_NV;
    m_devSemaphoreSciSyncPoolResCreateInfo.pNext = nullptr;
    m_devSemaphoreSciSyncPoolResCreateInfo.semaphoreSciSyncPoolRequestCount = 512;

    m_devObjectResCreateInfo.pNext = &m_devSemaphoreSciSyncPoolResCreateInfo;

    return NvError_Success;
}

NvError CVulkanSCEngine::CreatePipelineCaches()
{
    VkResult res = vkCreatePipelineCache(m_dev, &m_pipelineCacheCreateInfo, nullptr, &m_pipelineCache);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkCreatePipelineCache");

    return NvError_Success;
}

NvError CVulkanSCEngine::LoadPipelineCaches()
{
    std::string sReadFileName = "pipeline_cache.bin";

    LOG_INFO("Trying to read cache from: %s.", sReadFileName.c_str());
    FILE *pReadFile = fopen(sReadFileName.c_str(), "rb");

    if (pReadFile) {
        fseek(pReadFile, 0, SEEK_END);
        m_lStartCacheSize = ftell(pReadFile);
        if (m_lStartCacheSize < 0) {
            LOG_ERR("Failed to get file size");
            fclose(pReadFile);
            return NvError_FileOperationFailed;
        }
        rewind(pReadFile);

        m_pStartCacheData = (char *)malloc(sizeof(char) * m_lStartCacheSize);
        if (m_pStartCacheData == nullptr) {
            LOG_ERR("Memory error.");
            fclose(pReadFile);
            return NvError_InsufficientMemory;
        }

        size_t uResult = fread(m_pStartCacheData, 1, m_lStartCacheSize, pReadFile);
        if (uResult != static_cast<size_t>(m_lStartCacheSize)) {
            LOG_ERR("Reading error.");
            free(m_pStartCacheData);
            fclose(pReadFile);
            return NvError_FileReadFailed;
        }

        fclose(pReadFile);
        LOG_INFO("Pipeline cache HIT!");
        LOG_INFO("CacheData loaded from %s.", sReadFileName.c_str());
    } else {
        LOG_ERR("Error opening cache file %s.", sReadFileName.c_str());
        return NvError_FileOperationFailed;
    }

    m_pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    m_pipelineCacheCreateInfo.pNext = nullptr;
    m_pipelineCacheCreateInfo.initialDataSize = m_lStartCacheSize;
    m_pipelineCacheCreateInfo.pInitialData = m_pStartCacheData;
    m_pipelineCacheCreateInfo.flags =
        VK_PIPELINE_CACHE_CREATE_READ_ONLY_BIT | VK_PIPELINE_CACHE_CREATE_USE_APPLICATION_STORAGE_BIT;

    return NvError_Success;
}

NvError CVulkanSCEngine::GetSemaphoreFromVkSciSyncObj(
    NvSciSyncFence* sciSyncFence,
    VkSemaphore& semaphore,
    VkSemaphoreSciSyncPoolNV& semaphorePool)
{
    if (semaphore != VK_NULL_HANDLE) {
        return NvError_Success;
    }

    VkSemaphoreSciSyncCreateInfoNV semaphoreSciSyncInfo = { .sType =
                                                                VK_STRUCTURE_TYPE_SEMAPHORE_SCI_SYNC_CREATE_INFO_NV,
                                                            .pNext = nullptr,
                                                            .semaphorePool = semaphorePool,
                                                            .pFence = sciSyncFence };

    VkSemaphoreTypeCreateInfo sempahoreTypeInfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .pNext = &semaphoreSciSyncInfo,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = 0 // initialValue must be 0 with NvSciSyncObj
    };

    VkSemaphoreCreateInfo semaphoreCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &sempahoreTypeInfo,
        .flags = 0 // reserved bit, must be zero
    };

    VkResult res = vkCreateSemaphore(m_dev, &semaphoreCreateInfo, nullptr, &semaphore);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkCreateSemaphore");

    return NvError_Success;
}

#else
NvError CVulkanSCEngine::CheckJsonLayerSupport()
{
    uint32_t uLayerCount;
    vkEnumerateInstanceLayerProperties(&uLayerCount, nullptr);

    std::vector<VkLayerProperties> vAvailableLayers(uLayerCount);
    vkEnumerateInstanceLayerProperties(&uLayerCount, vAvailableLayers.data());

    for (uint32_t i = 0; i < uLayerCount; i++) {
        if (!strncmp(vAvailableLayers[i].layerName, jsonGenLayerName, strlen(jsonGenLayerName))) {
            return NvError_Success;
        }
    }

    return NvError_BadValue;
}
#endif

NvError CVulkanSCEngine::DumpImage(uint32_t uPacketIndex, uint8_t *pDst)
{
    VkResult res;
    VkCommandBufferBeginInfo cmdBufInfo{};
    cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cmdBufInfo.pNext = nullptr;

    res = vkResetCommandPool(m_dev, m_cmdPool, 0);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkResetCommandPool");

    res = vkBeginCommandBuffer(m_vkCopyCmdBuffer, &cmdBufInfo);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkBeginCommandBuffer");

    VkImageCopy imageCopyRegion{};
    imageCopyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageCopyRegion.srcSubresource.layerCount = uResourceLayerCount;
    imageCopyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageCopyRegion.dstSubresource.layerCount = uResourceLayerCount;
    imageCopyRegion.extent = { m_uWidth, m_uHeight, 1 };

    // Issue the copy command
    vkCmdCopyImage(
        m_vkCopyCmdBuffer,
        m_vkImportImage[uPacketIndex],
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        m_dstImage,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &imageCopyRegion);

    res =  vkEndCommandBuffer(m_vkCopyCmdBuffer);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkEndCommandBuffer");
    // submit
    VkSubmitInfo submitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .commandBufferCount = 1,
        .pCommandBuffers    = &m_vkCopyCmdBuffer,
    };

    res = vkWaitForFences(m_dev, 1, &m_vkSignalFence, true, DEFAULT_FENCE_TIMEOUT);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkWaitForFences");

    res = vkQueueSubmit(m_queue, 1, &submitInfo, nullptr);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkQueueSubmit");

    res = vkDeviceWaitIdle(m_dev);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkQueueSubmit");

    uint8_t *pCpuPtr = nullptr;
    uint32_t uOffset = 0;
    uint32_t uImageSize = m_dstMemRequirements.size;
    res = vkMapMemory(m_dev, m_dstDevMem, uOffset, uImageSize, 0, (void **)&pCpuPtr);
    CHK_VKSCSTATUS_AND_RETURN(res, "vkMapMemory");

    if (nullptr != pCpuPtr && uImageSize > 0) {
        memcpy(pDst, pCpuPtr, uImageSize);
    }

    vkUnmapMemory(m_dev, m_dstDevMem);

    return NvError_Success;
}

VkFormat CVulkanSCEngine::ColorToVkFormat(std::string &sColorType)
{
    VkFormat vkFormat = VK_FORMAT_UNDEFINED;

    if (sColorType == "ARGB") {
        vkFormat = VK_FORMAT_B8G8R8A8_UNORM;
    } else if (sColorType == "ABGR") {
        vkFormat = VK_FORMAT_A8B8G8R8_UNORM_PACK32;
    } else {
        LOG_ERR("Unsupported sColor type!\n");
    }

    return vkFormat;
}
