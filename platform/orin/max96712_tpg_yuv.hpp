/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef MAX96712_TPG_YUV_HPP
#define MAX96712_TPG_YUV_HPP

static PlatformCfg platformCfgMax96712TPGYUV = {
    .platform = "MAX96712_YUV_8_TPG_CPHY_x4",
    .platformConfig = "MAX96712_YUV_8_TPG_CPHY_x4",
    .description = "MAX96712 YUV 8-bit TPG in 4 lane CPHY mode",
    .numDeviceBlocks = 1U,
    .deviceBlockList = { { .csiPort = NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_AB,
                           .phyMode = NVSIPL_CAP_CSI_CPHY_MODE,
                           .i2cDevice = 0U,
                           .deserInfo = { .name = "MAX96712",
#if !NV_IS_SAFETY
                                          .description = "Maxim 96712 Aggregator",
#endif // !NV_IS_SAFETY
                                          .i2cAddress = 0x29,
                                          .errGpios = {},
                                          .useCDIv2API = true },
                           .numCameraModules = 2U,
                           .cameraModuleInfoList = { { .name = "MAX96712TPG_YUV_8",
#if !NV_IS_SAFETY
                                                       .description = "MAX96712 YUV 8-bit TPG in 4 lane CPHY mode",
#endif // !NV_IS_SAFETY
                                                       .linkIndex = 0U,
#if !NV_IS_SAFETY
                                                       .isSimulatorModeEnabled = false,
#endif // !NV_IS_SAFETY
                                                       .serInfo = { .name = "MAX96712_TPG_SERIALIZER",
#if !NV_IS_SAFETY
                                                                    .description = "Dummy serializer to allow MAXIM "
                                                                                   "96712 Deserializer TPG",
#endif // !NV_IS_SAFETY
                                                                    .i2cAddress = 0x0,
#if !NV_IS_SAFETY
                                                                    .longCable = false,
#endif // !NV_IS_SAFETY
                                                                    .errGpios = {},
                                                                    .useCDIv2API = true,
                                                                    .serdesGPIOPinMappings = {} },
                                                       .sensorInfo = { .id = 0U,
                                                                       .name = "MAX96712_TPG_SENSOR",
#if !NV_IS_SAFETY
                                                                       .description = "MAX96712_TPG_SENSOR",
#endif // !NV_IS_SAFETY
                                                                       .i2cAddress = 0x00,
                                                                       .vcInfo = { .cfa = NVSIPL_PIXEL_ORDER_YUYV,
                                                                                   .embeddedTopLines = 0U,
                                                                                   .embeddedBottomLines = 0U,
                                                                                   .inputFormat =
                                                                                       NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV422,
                                                                                   .resolution = { .width = 1920U,
                                                                                                   .height = 1236U },
                                                                                   .fps = 30.0,
                                                                                   .isEmbeddedDataTypeEnabled = false },
                                                                       .isTriggerModeEnabled = false,
                                                                       .errGpios = {},
                                                                       .useCDIv2API = true,
#if !NV_IS_SAFETY
                                                                       .isTPGEnabled = true
#endif // !NV_IS_SAFETY
                                                       } },
                                                     { .name = "MAX96712TPG_YUV_8",
#if !NV_IS_SAFETY
                                                       .description = "MAX96712 YUV 8-bit TPG in 4 lane CPHY mode",
#endif // !NV_IS_SAFETY
                                                       .linkIndex = 1U,
#if !NV_IS_SAFETY
                                                       .isSimulatorModeEnabled = false,
#endif // !NV_IS_SAFETY
                                                       .serInfo = { .name = "MAX96712_TPG_SERIALIZER",
#if !NV_IS_SAFETY
                                                                    .description = "Dummy serializer to allow MAXIM "
                                                                                   "96712 Deserializer TPG",
#endif // !NV_IS_SAFETY
                                                                    .i2cAddress = 0x0,
#if !NV_IS_SAFETY
                                                                    .longCable = false,
#endif // !NV_IS_SAFETY
                                                                    .errGpios = {},
                                                                    .useCDIv2API = true,
                                                                    .serdesGPIOPinMappings = {} },
                                                       .sensorInfo = { .id = 1U,
                                                                       .name = "MAX96712_TPG_SENSOR",
#if !NV_IS_SAFETY
                                                                       .description = "MAX96712_TPG_SENSOR",
#endif // !NV_IS_SAFETY
                                                                       .i2cAddress = 0x00,
                                                                       .vcInfo = { .cfa = NVSIPL_PIXEL_ORDER_YUYV,
                                                                                   .embeddedTopLines = 0U,
                                                                                   .embeddedBottomLines = 0U,
                                                                                   .inputFormat =
                                                                                       NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV422,
                                                                                   .resolution = { .width = 1920U,
                                                                                                   .height = 1236U },
                                                                                   .fps = 30.0,
                                                                                   .isEmbeddedDataTypeEnabled = false },
                                                                       .isTriggerModeEnabled = false,
                                                                       .errGpios = {},
                                                                       .useCDIv2API = true,
#if !NV_IS_SAFETY
                                                                       .isTPGEnabled = true
#endif // !NV_IS_SAFETY
                                                       } } },
                           .desI2CPort = 0U,
                           .desTxPort = UINT32_MAX,
                           .pwrPort = 0U,
                           .dphyRate = { 2500000U, 2500000U },
                           .cphyRate = { 2000000U, 2000000U },
#if !NV_IS_SAFETY
                           .isPassiveModeEnabled = false,
#endif // !NV_IS_SAFETY
                           .isGroupInitProg = true,
                           .gpios = {},
#if !NV_IS_SAFETY
                           .isPwrCtrlDisabled = false,
                           .longCables = { false, false, false, false },
#endif // !NV_IS_SAFETY
                           .resetAll = false } }
};

static PlatformCfg platformCfgMax96712TPGYUV_5m = {
    .platform = "MAX96712_2880x1860_YUV_8_TPG_CPHY_x4",
    .platformConfig = "MAX96712_2880x1860_YUV_8_TPG_CPHY_x4",
    .description = "MAX96712 2880x1860 YUV 8-bit TPG in 4 lane CPHY mode",
    .numDeviceBlocks = 1U,
    .deviceBlockList = { { .csiPort = NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_AB,
                           .phyMode = NVSIPL_CAP_CSI_CPHY_MODE,
                           .i2cDevice = 0U,
                           .deserInfo = { .name = "MAX96712",
#if !NV_IS_SAFETY
                                          .description = "Maxim 96712 Aggregator",
#endif // !NV_IS_SAFETY
                                          .i2cAddress = 0x29,
                                          .errGpios = {},
                                          .useCDIv2API = true },
                           .numCameraModules = 2U,
                           .cameraModuleInfoList = { { .name = "MAX96712TPG_YUV_8",
#if !NV_IS_SAFETY
                                                       .description = "MAX96712 YUV 8-bit TPG in 4 lane CPHY mode",
#endif // !NV_IS_SAFETY
                                                       .linkIndex = 0U,
#if !NV_IS_SAFETY
                                                       .isSimulatorModeEnabled = false,
#endif // !NV_IS_SAFETY
                                                       .serInfo = { .name = "MAX96712_TPG_SERIALIZER",
#if !NV_IS_SAFETY
                                                                    .description = "Dummy serializer to allow MAXIM "
                                                                                   "96712 Deserializer TPG",
#endif // !NV_IS_SAFETY
                                                                    .i2cAddress = 0x0,
#if !NV_IS_SAFETY
                                                                    .longCable = false,
#endif // !NV_IS_SAFETY
                                                                    .errGpios = {},
                                                                    .useCDIv2API = true,
                                                                    .serdesGPIOPinMappings = {} },
                                                       .sensorInfo = { .id = 0U,
                                                                       .name = "MAX96712_TPG_SENSOR",
#if !NV_IS_SAFETY
                                                                       .description = "MAX96712_TPG_SENSOR",
#endif // !NV_IS_SAFETY
                                                                       .i2cAddress = 0x00,
                                                                       .vcInfo = { .cfa = NVSIPL_PIXEL_ORDER_YUYV,
                                                                                   .embeddedTopLines = 0U,
                                                                                   .embeddedBottomLines = 0U,
                                                                                   .inputFormat =
                                                                                       NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV422,
                                                                                   .resolution = { .width = 2880U,
                                                                                                   .height = 1860U },
                                                                                   .fps = 30.0,
                                                                                   .isEmbeddedDataTypeEnabled = false },
                                                                       .isTriggerModeEnabled = false,
                                                                       .errGpios = {},
                                                                       .useCDIv2API = true,
#if !NV_IS_SAFETY
                                                                       .isTPGEnabled = true
#endif // !NV_IS_SAFETY
                                                       } },
                                                     { .name = "MAX96712TPG_YUV_8",
#if !NV_IS_SAFETY
                                                       .description = "MAX96712 YUV 8-bit TPG in 4 lane CPHY mode",
#endif // !NV_IS_SAFETY
                                                       .linkIndex = 1U,
#if !NV_IS_SAFETY
                                                       .isSimulatorModeEnabled = false,
#endif // !NV_IS_SAFETY
                                                       .serInfo = { .name = "MAX96712_TPG_SERIALIZER",
#if !NV_IS_SAFETY
                                                                    .description = "Dummy serializer to allow MAXIM "
                                                                                   "96712 Deserializer TPG",
#endif // !NV_IS_SAFETY
                                                                    .i2cAddress = 0x0,
#if !NV_IS_SAFETY
                                                                    .longCable = false,
#endif // !NV_IS_SAFETY
                                                                    .errGpios = {},
                                                                    .useCDIv2API = true,
                                                                    .serdesGPIOPinMappings = {} },
                                                       .sensorInfo = { .id = 1U,
                                                                       .name = "MAX96712_TPG_SENSOR",
#if !NV_IS_SAFETY
                                                                       .description = "MAX96712_TPG_SENSOR",
#endif // !NV_IS_SAFETY
                                                                       .i2cAddress = 0x00,
                                                                       .vcInfo = { .cfa = NVSIPL_PIXEL_ORDER_YUYV,
                                                                                   .embeddedTopLines = 0U,
                                                                                   .embeddedBottomLines = 0U,
                                                                                   .inputFormat =
                                                                                       NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV422,
                                                                                   .resolution = { .width = 2880U,
                                                                                                   .height = 1860U },
                                                                                   .fps = 30.0,
                                                                                   .isEmbeddedDataTypeEnabled = false },
                                                                       .isTriggerModeEnabled = false,
                                                                       .errGpios = {},
                                                                       .useCDIv2API = true,
#if !NV_IS_SAFETY
                                                                       .isTPGEnabled = true
#endif // !NV_IS_SAFETY
                                                       } } },
                           .desI2CPort = 0U,
                           .desTxPort = UINT32_MAX,
                           .pwrPort = 0U,
                           .dphyRate = { 2500000U, 2500000U },
                           .cphyRate = { 2000000U, 2000000U },
#if !NV_IS_SAFETY
                           .isPassiveModeEnabled = false,
#endif // !NV_IS_SAFETY
                           .isGroupInitProg = true,
                           .gpios = {},
#if !NV_IS_SAFETY
                           .isPwrCtrlDisabled = false,
                           .longCables = { false, false, false, false },
#endif // !NV_IS_SAFETY
                           .resetAll = false } }
};

#endif // ISX031_HPP
