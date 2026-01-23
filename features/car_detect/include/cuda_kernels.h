/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef KERNELS_H
#define KERNELS_H

#include "CarCommon.hpp"
#include "cvt2d_kernels.h"

bool nv12blDrawRect(const cudaArray_t *yuvInPlanes,
                    int nSrcPitch,
                    int nSrcWidth,
                    int nSrcHeight,
                    std::vector<NvInferObject> bbox,
                    cudaStream_t stream);

bool nv12plDrawRect(const cudaArray_t *yuvInPlanes,
                    int nSrcPitch,
                    int nSrcWidth,
                    int nSrcHeight,
                    std::vector<NvInferObject> bbox,
                    cudaStream_t stream);

#endif