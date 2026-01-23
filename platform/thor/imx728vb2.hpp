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

#ifndef IMX728VB2_HPP
#define IMX728VB2_HPP

static PlatformCfg platformCfgIMX728VB2 = {
    // TODO(@pengq): To align with nvsipl_sample to change to the correct platform name
    .platform = "V1SIM728S1RU3120NB20_CPHY_x4",
    .platformConfig = "V1SIM728S1RU3120NB20_CPHY_x4",
    .description = "IMX728+MAX96724 RGGB module in 4 lane CPHY mode",
    .numDeviceBlocks = 1U,
    .deviceBlockList = {
        {
            .csiPort = NVSIPL_CAP_CSI_INTERFACE_TYPE_CSI_AB,
            .phyMode = NVSIPL_CAP_CSI_CPHY_MODE,
            .i2cDevice = 7U,
            .deserInfo = {
                .name = "MAX96724_Fusa_nv",
#if !NV_IS_SAFETY
                .description = "Maxim 96724 Aggregator Fusa",
#endif // !NV_IS_SAFETY
                .i2cAddress = 0x27,
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
                    .name = "V1SIM728MPRU4120ND1",
#if !NV_IS_SAFETY
                    .description = "IMX728 RGGB module - 120FOV - MIPI-IMX728, MAX96717",
#endif // !NV_IS_SAFETY
                    .linkIndex = 0U,
                    .serInfo = {
                        .name = "MAX96717F",
#if !NV_IS_SAFETY
                        .description = "Maxim 96717F Serializer",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x40,
#ifdef NVMEDIA_QNX
                            .useCDIv2API = true
#else // Linux
                            .useCDIv2API = false
#endif //NVMEDIA_QNX
                    },
                    .isEEPROMSupported = true,
                    .eepromInfo = {
                        .name = "M24C04",
#if !NV_IS_SAFETY
                        .description = "M24C04 EEPROM",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x54,
#ifdef NVMEDIA_QNX
                            .useCDIv2API = true
#else // Linux
                            .useCDIv2API = false
#endif //NVMEDIA_QNX
                    },
                    .sensorInfo = {
                            .id = 0U,
                            .name = "IMX728",
#if !NV_IS_SAFETY
                            .description = "SONY IMX728 Sensor",
#endif // !NV_IS_SAFETY
                            .i2cAddress = 0x1C,
                            .vcInfo = {
                                    .cfa = NVSIPL_PIXEL_ORDER_RGGB,
                                    .embeddedTopLines = 5U,
                                    .embeddedBottomLines = 20U,
                                    .inputFormat = NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12RJ,
                                    .resolution = {
                                        .width = 3840U,
                                        .height = 2160U
                                    },
                                    .fps = 30.0,
                                    .isEmbeddedDataTypeEnabled = false
                            },
                            .isTriggerModeEnabled = true,
    #ifdef NVMEDIA_QNX
                            .useCDIv2API = true,
    #else // Linux
                            .useCDIv2API = false,
                            .isAuthEnabled = false,
    #endif //NVMEDIA_QNX
                    },
#ifdef NVMEDIA_QNX
                    .cryptoKeysList = cryptoKeysIMX728
#endif //NVMEDIA_QNX
                },
                {
                    .name = "V1SIM728MPRU4120ND1",
#if !NV_IS_SAFETY
                    .description = "IMX728 RGGB module - 120FOV - MIPI-IMX728, MAX96717",
#endif // !NV_IS_SAFETY
                    .linkIndex = 1U,
                    .serInfo = {
                        .name = "MAX96717F",
#if !NV_IS_SAFETY
                        .description = "Maxim 96717F Serializer",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x40,
#ifdef NVMEDIA_QNX
                            .useCDIv2API = true
#else // Linux
                            .useCDIv2API = false
#endif //NVMEDIA_QNX
                    },
                    .isEEPROMSupported = true,
                    .eepromInfo = {
                        .name = "M24C04",
#if !NV_IS_SAFETY
                        .description = "M24C04 EEPROM",
#endif // !NV_IS_SAFETY
                        .i2cAddress = 0x54,
#ifdef NVMEDIA_QNX
                            .useCDIv2API = true
#else // Linux
                            .useCDIv2API = false
#endif //NVMEDIA_QNX
                    },
                    .sensorInfo = {
                            .id = 1U,
                            .name = "IMX728",
#if !NV_IS_SAFETY
                            .description = "SONY IMX728 Sensor",
#endif // !NV_IS_SAFETY
                            .i2cAddress = 0x1C,
                            .vcInfo = {
                                    .cfa = NVSIPL_PIXEL_ORDER_RGGB,
                                    .embeddedTopLines = 5U,
                                    .embeddedBottomLines = 20U,
                                    .inputFormat = NVSIPL_CAP_INPUT_FORMAT_TYPE_RAW12RJ,
                                    .resolution = {
                                        .width = 3840U,
                                        .height = 2160U
                                    },
                                    .fps = 30.0,
                                    .isEmbeddedDataTypeEnabled = false
                            },
                            .isTriggerModeEnabled = true,
    #ifdef NVMEDIA_QNX
                                .useCDIv2API = true,
    #else // Linux
                                .useCDIv2API = false,
                                .isAuthEnabled = false,
    #endif //NVMEDIA_QNX
                    },
#ifdef NVMEDIA_QNX
                    .cryptoKeysList = cryptoKeysIMX728
#endif //NVMEDIA_QNX
                }
            },
            .desI2CPort = 0U,
            .desTxPort = UINT32_MAX,
            .pwrPort = 1U,
            .dphyRate = {2500000U, 2500000U},
            .cphyRate = {2000000U, 2000000U}
        }
    }
};

#endif // IMX728VB2_HPP
