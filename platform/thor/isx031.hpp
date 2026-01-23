/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef ISX031_HPP
#define ISX031_HPP

static PlatformCfg platformCfgIsx031x2 = {
    .platform = "SG3_ISX031_H190X_YUV_8_CPHY_x2",
    .platformConfig = "SG3_ISX031_H190X_YUV_8_CPHY_x2",
    .description = "ISX031 YUYV module in 2 lane CPHY mode",
    .numDeviceBlocks = 1U,
    .deviceBlockList = {
        {
            .csiPort = NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_E,
            .phyMode = NVSIPL_CAP_CSI_CPHY_MODE,
            .i2cDevice = 12U,
            .deserInfo = {
                .name = "MAX96712",
#if !NV_IS_SAFETY
                .description = "Maxim 96712 Aggregator",
#endif // !NV_IS_SAFETY
                .i2cAddress = 0x29,
                .errGpios = {},
#ifdef NVMEDIA_QNX
                .useCDIv2API = true
#else // Linux
                .useCDIv2API = false
#endif //NVMEDIA_QNX
            },
            .numCameraModules = 2U,
            .cameraModuleInfoList = {
                {
                    .name = "SG3_ISX031_H190X",
#if !NV_IS_SAFETY
                    .description = "Sony ISX031 YUYV module - 120-deg FOV, MIPI-ISX031, MAX9295",
#endif // !NV_IS_SAFETY
                    .linkIndex = 0U,
#if !NV_IS_SAFETY
                    .isSimulatorModeEnabled = false,
#endif // !NV_IS_SAFETY
                    .serInfo = {
                        .name = "MAX9295",
#if !NV_IS_SAFETY
                        .description = "Maxim 9295 Serializer",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x40,
#if !NV_IS_SAFETY
                        .longCable = false,
#endif // !NV_IS_SAFETY
                        .errGpios = {},
#ifdef NVMEDIA_QNX
                        .useCDIv2API = true,
#else // Linux
                        .useCDIv2API = false,
#endif //NVMEDIA_QNX,
                        .serdesGPIOPinMappings = {}
                    },
                    .isEEPROMSupported = false,
                    .eepromInfo = {
                        .name = "",
#if !NV_IS_SAFETY
                        .description = "",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0xFF,
#ifdef NVMEDIA_QNX
                        .useCDIv2API = true
#else // Linux
                        .useCDIv2API = false
#endif //NVMEDIA_QNX
                    },
                    .sensorInfo = {
                            .id = 8U,
                            .name = "SG3_ISX031_H190X",
#if !NV_IS_SAFETY
                            .description = "Sensing ISX031 Sensor",
#endif // !NV_IS_SAFETY
                            .i2cAddress = 0x1a,
                            .vcInfo = {
                                    .cfa = NVSIPL_PIXEL_ORDER_YUYV,
                                    .embeddedTopLines = 1U,
                                    .embeddedBottomLines = 0U,
                                    .inputFormat = NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV422,
                                    .resolution = {
                                        .width = 1920U,
                                        .height = 1536U
                                    },
                                    .fps = 30.0,
                                    .isEmbeddedDataTypeEnabled = false
                            },
                            .isTriggerModeEnabled = true,
                            .errGpios = {},
#ifdef NVMEDIA_QNX
                            .useCDIv2API = true
#else // Linux
                            .useCDIv2API = false
#endif //NVMEDIA_QNX
                    }
                },
                {
                    .name = "SG3_ISX031_H190X",
#if !NV_IS_SAFETY
                    .description = "Sensing ISX031 Sensor",
#endif // !NV_IS_SAFETY
                    .linkIndex = 1U,
#if !NV_IS_SAFETY
                    .isSimulatorModeEnabled = false,
#endif // !NV_IS_SAFETY
                    .serInfo = {
                        .name = "MAX9295",
#if !NV_IS_SAFETY
                        .description = "Maxim 9295 Serializer",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x40,
#if !NV_IS_SAFETY
                        .longCable = false,
#endif // !NV_IS_SAFETY
                        .errGpios = {},
#ifdef NVMEDIA_QNX
                        .useCDIv2API = true,
#else // Linux
                        .useCDIv2API = false,
#endif //NVMEDIA_QNX,
                        .serdesGPIOPinMappings = {}
                    },
                    .isEEPROMSupported = false,
                    .eepromInfo = {
                        .name = "",
#if !NV_IS_SAFETY
                        .description = "",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0xFF,
#ifdef NVMEDIA_QNX
                        .useCDIv2API = true
#else // Linux
                        .useCDIv2API = false
#endif //NVMEDIA_QNX
                    },
                    .sensorInfo = {
                            .id = 9U,
                            .name = "SG3_ISX031_H190X",
#if !NV_IS_SAFETY
                            .description = "Sensing ISX031 Sensor",
#endif // !NV_IS_SAFETY
                            .i2cAddress = 0x1a,
                            .vcInfo = {
                                    .cfa = NVSIPL_PIXEL_ORDER_YUYV,
                                    .embeddedTopLines = 1U,
                                    .embeddedBottomLines = 0U,
                                    .inputFormat = NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV422,
                                    .resolution = {
                                        .width = 1920U,
                                        .height = 1536U
                                    },
                                    .fps = 30.0,
                                    .isEmbeddedDataTypeEnabled = false
                            },
                            .isTriggerModeEnabled = true,
                            .errGpios = {},
#ifdef NVMEDIA_QNX
                            .useCDIv2API = true
#else // Linux
                            .useCDIv2API = false
#endif //NVMEDIA_QNX
                    }
                }
            },
            .desI2CPort = 0U,
            .desTxPort = UINT32_MAX,
            .pwrPort = 4U,
            .dphyRate = {2500000U, 2500000U},
            .cphyRate = {2000000U, 2000000U},
#if !NV_IS_SAFETY
            .isPassiveModeEnabled = false,
#endif // !NV_IS_SAFETY
            .isGroupInitProg = true,
            .gpios = { 7 },
#if !NV_IS_SAFETY
            .isPwrCtrlDisabled = false,
            .longCables = {false, false, false, false},
#endif // !NV_IS_SAFETY
            .resetAll = false
        }
    }
};

static PlatformCfg platformCfgIsx031x4 = {
    .platform = "SG3_ISX031_H190X_YUV_8_CPHY_x4",
    .platformConfig = "SG3_ISX031_H190X_YUV_8_CPHY_x4",
    .description = "ISX031 YUYV module in 2 lane CPHY mode",
    .numDeviceBlocks = 1U,
    .deviceBlockList = {
        {
            .csiPort = NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_EF,
            .phyMode = NVSIPL_CAP_CSI_CPHY_MODE,
            .i2cDevice = 12U,
            .deserInfo = {
                .name = "MAX96712",
#if !NV_IS_SAFETY
                .description = "Maxim 96712 Aggregator",
#endif // !NV_IS_SAFETY
                .i2cAddress = 0x29,
                .errGpios = {},
#ifdef NVMEDIA_QNX
                .useCDIv2API = true
#else // Linux
                .useCDIv2API = false
#endif //NVMEDIA_QNX
            },
            .numCameraModules = 2U,
            .cameraModuleInfoList = {
                {
                    .name = "SG3_ISX031_H190X",
#if !NV_IS_SAFETY
                    .description = "Sony ISX031 YUYV module - 120-deg FOV, MIPI-ISX031, MAX9295",
#endif // !NV_IS_SAFETY
                    .linkIndex = 0U,
#if !NV_IS_SAFETY
                    .isSimulatorModeEnabled = false,
#endif // !NV_IS_SAFETY
                    .serInfo = {
                        .name = "MAX9295",
#if !NV_IS_SAFETY
                        .description = "Maxim 9295 Serializer",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x40,
#if !NV_IS_SAFETY
                        .longCable = false,
#endif // !NV_IS_SAFETY
                        .errGpios = {},
#ifdef NVMEDIA_QNX
                        .useCDIv2API = true,
#else // Linux
                        .useCDIv2API = false,
#endif //NVMEDIA_QNX,
                        .serdesGPIOPinMappings = {}
                    },
                    .isEEPROMSupported = false,
                    .eepromInfo = {
                        .name = "",
#if !NV_IS_SAFETY
                        .description = "",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0xFF,
#ifdef NVMEDIA_QNX
                        .useCDIv2API = true
#else // Linux
                        .useCDIv2API = false
#endif //NVMEDIA_QNX
                    },
                    .sensorInfo = {
                            .id = 8U,
                            .name = "SG3_ISX031_H190X",
#if !NV_IS_SAFETY
                            .description = "Sensing ISX031 Sensor",
#endif // !NV_IS_SAFETY
                            .i2cAddress = 0x1a,
                            .vcInfo = {
                                    .cfa = NVSIPL_PIXEL_ORDER_YUYV,
                                    .embeddedTopLines = 1U,
                                    .embeddedBottomLines = 0U,
                                    .inputFormat = NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV422,
                                    .resolution = {
                                        .width = 1920U,
                                        .height = 1536U
                                    },
                                    .fps = 30.0,
                                    .isEmbeddedDataTypeEnabled = false
                            },
                            .isTriggerModeEnabled = true,
                            .errGpios = {},
#ifdef NVMEDIA_QNX
                            .useCDIv2API = true
#else // Linux
                            .useCDIv2API = false
#endif //NVMEDIA_QNX
                    }
                },
                {
                    .name = "SG3_ISX031_H190X",
#if !NV_IS_SAFETY
                    .description = "Sensing ISX031 Sensor",
#endif // !NV_IS_SAFETY
                    .linkIndex = 1U,
#if !NV_IS_SAFETY
                    .isSimulatorModeEnabled = false,
#endif // !NV_IS_SAFETY
                    .serInfo = {
                        .name = "MAX9295",
#if !NV_IS_SAFETY
                        .description = "Maxim 9295 Serializer",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x40,
#if !NV_IS_SAFETY
                        .longCable = false,
#endif // !NV_IS_SAFETY
                        .errGpios = {},
#ifdef NVMEDIA_QNX
                        .useCDIv2API = true,
#else // Linux
                        .useCDIv2API = false,
#endif //NVMEDIA_QNX,
                        .serdesGPIOPinMappings = {}
                    },
                    .isEEPROMSupported = false,
                    .eepromInfo = {
                        .name = "",
#if !NV_IS_SAFETY
                        .description = "",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0xFF,
#ifdef NVMEDIA_QNX
                        .useCDIv2API = true
#else // Linux
                        .useCDIv2API = false
#endif //NVMEDIA_QNX
                    },
                    .sensorInfo = {
                            .id = 9U,
                            .name = "SG3_ISX031_H190X",
#if !NV_IS_SAFETY
                            .description = "Sensing ISX031 Sensor",
#endif // !NV_IS_SAFETY
                            .i2cAddress = 0x1a,
                            .vcInfo = {
                                    .cfa = NVSIPL_PIXEL_ORDER_YUYV,
                                    .embeddedTopLines = 1U,
                                    .embeddedBottomLines = 0U,
                                    .inputFormat = NVSIPL_CAP_INPUT_FORMAT_TYPE_YUV422,
                                    .resolution = {
                                        .width = 1920U,
                                        .height = 1536U
                                    },
                                    .fps = 30.0,
                                    .isEmbeddedDataTypeEnabled = false
                            },
                            .isTriggerModeEnabled = true,
                            .errGpios = {},
#ifdef NVMEDIA_QNX
                            .useCDIv2API = true
#else // Linux
                            .useCDIv2API = false
#endif //NVMEDIA_QNX
                    }
                }
            },
            .desI2CPort = 0U,
            .desTxPort = UINT32_MAX,
            .pwrPort = 4U,
            .dphyRate = {2500000U, 2500000U},
            .cphyRate = {2000000U, 2000000U},
#if !NV_IS_SAFETY
            .isPassiveModeEnabled = false,
#endif // !NV_IS_SAFETY
            .isGroupInitProg = true,
            .gpios = { 7 },
#if !NV_IS_SAFETY
            .isPwrCtrlDisabled = false,
            .longCables = {false, false, false, false},
#endif // !NV_IS_SAFETY
            .resetAll = false
        }
    }
};
#endif // ISX031_HPP
