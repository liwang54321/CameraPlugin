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

#ifndef CVT2D_KERNELS_H
#define CVT2D_KERNELS_H

#include <iostream>
#include <cstdint>

#define checkCudaErrors(status)                                                                                   \
    {                                                                                                             \
        if (status != 0) {                                                                                        \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) << " at line " << __LINE__ << " in file " \
                      << __FILE__ << " error status: " << status << std::endl;                                    \
            return false;                                                                                         \
        }                                                                                                         \
    }

bool CvtNv12blToRgbPlanar(const cudaArray_t *yuvInPlanes,
                          int nSrcWidth,
                          int nSrcHeight,
                          void *dpTsr,
                          int nTsrWidth,
                          int nTsrHeight,
                          int nBatchSize,
                          float scaleFactor,
                          cudaStream_t stream);

#endif
