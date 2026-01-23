/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef PVA_PRE_BASE_H
#define PVA_PRE_BASE_H

#include <fstream>
#include <cstring>
#include <string>
#include <cuda_runtime_api.h>

#if defined(__cplusplus)
extern "C"
{
#endif

    class PvaPreBase
    {
      public:
        PvaPreBase() {}
        virtual ~PvaPreBase() {}

        virtual int pva_init(cudaStream_t m_stream,
                             unsigned uIpWidth,
                             unsigned uIpHeight,
                             unsigned uOpRsWidth,
                             unsigned uOpRsHeight,
                             unsigned uOpWidth,
                             unsigned uOpHeight,
                             const char *FilePath,
                             void *dpTsr,
                             float fScaleFactor,
                             unsigned uOutput_format,
                             unsigned uInput_format) = 0;

        virtual int pva_launch(const cudaArray_t *pYUV, void *dpTsr, cudaStream_t m_stream, int count) = 0;

        virtual int pva_deinit() = 0;
    };

    // the types of the class factories
    typedef PvaPreBase *create_t();
    typedef void destroy_t(PvaPreBase *);

#if defined(__cplusplus)
}
#endif

#endif // PVA_PRE_BASE_H