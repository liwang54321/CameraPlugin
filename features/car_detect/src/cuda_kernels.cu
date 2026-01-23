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

#include "cuda_kernels.h"

#define ROUND_DOWN_2(num) ((num) & (~1))

__forceinline__ __device__ static float clampF(float x, float lower, float upper)
{
    return x < lower ? lower : (x > upper ? upper : x);
}

static __global__ void DrawRectKernel(cudaSurfaceObject_t surfLuma,
                                      cudaSurfaceObject_t surfChroma,
                                      int nSrcWidth,
                                      int nSrcHeight,
                                      unsigned int o_x_4,
                                      unsigned int o_y_2,
                                      unsigned int o_w_4,
                                      unsigned int o_h_2,
                                      int class_index)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int px = x * 4, py = y * 2;
    if ((px + 3) >= nSrcWidth || (py + 1) >= nSrcHeight)
        return;
    unsigned int l_w = 1; // 1 * 4
    unsigned int l_h = 2; // 2 * 2
    o_x_4 = (int)(o_x_4 - l_w) < 0 ? l_w : o_x_4;
    o_y_2 = (int)(o_y_2 - l_h) < 0 ? l_h : o_y_2;

    uint8_t d_y, d_u, d_v;
    // Red yuv
    d_y = 65;
    d_u = 100;
    d_v = 212;

    if ((class_index == 0) &&
        (((x <= o_x_4 + l_w) && (x >= o_x_4 - l_w) && (y >= o_y_2 - l_h) && (y <= o_y_2 + o_h_2 + l_h)) ||
         ((x <= o_x_4 + o_w_4 + l_w) && (x >= o_x_4 + o_w_4 - l_w) && (y >= o_y_2 - l_h) &&
          (y <= o_y_2 + o_h_2 + l_h)) ||
         ((y <= o_y_2 + l_h) && (y >= o_y_2 - l_h) && (x >= o_x_4 - l_w) && (x <= o_x_4 + o_w_4 + l_w)) ||
         ((y <= o_y_2 + o_h_2 + l_h) && (y >= o_y_2 + o_h_2 - l_h) && (x >= o_x_4 - l_w) &&
          (x <= o_x_4 + o_w_4 + l_w)))) {
        for (int i = blockIdx.z; i < 1; i += gridDim.z) {
            surf2Dwrite<uint8_t>(uint8_t(d_y), surfLuma, px * 1, (py + i * nSrcHeight) * 1);
            surf2Dwrite<uint8_t>(uint8_t(d_y), surfLuma, (px + 1) * 1, (py + i * nSrcHeight) * 1);
            surf2Dwrite<uint8_t>(uint8_t(d_y), surfLuma, (px + 2) * 1, (py + i * nSrcHeight) * 1);
            surf2Dwrite<uint8_t>(uint8_t(d_y), surfLuma, (px + 3) * 1, (py + i * nSrcHeight) * 1);

            surf2Dwrite<uint8_t>(uint8_t(d_y), surfLuma, px * 1, ((py + i * nSrcHeight) + 1) * 1);
            surf2Dwrite<uint8_t>(uint8_t(d_y), surfLuma, (px + 1) * 1, ((py + i * nSrcHeight) + 1) * 1);
            surf2Dwrite<uint8_t>(uint8_t(d_y), surfLuma, (px + 2) * 1, ((py + i * nSrcHeight) + 1) * 1);
            surf2Dwrite<uint8_t>(uint8_t(d_y), surfLuma, (px + 3) * 1, ((py + i * nSrcHeight) + 1) * 1);

            surf2Dwrite<uint8_t>(uint8_t(d_u), surfChroma, ROUND_DOWN_2(int(px * 1)), (y + i * nSrcHeight / 2) * 1);
            surf2Dwrite<uint8_t>(uint8_t(d_v), surfChroma, ROUND_DOWN_2(int(px * 1)) + 1, (y + i * nSrcHeight / 2) * 1);
            surf2Dwrite<uint8_t>(uint8_t(d_u), surfChroma, ROUND_DOWN_2(int((px + 2) * 1)),
                                 (y + i * nSrcHeight / 2) * 1);
            surf2Dwrite<uint8_t>(uint8_t(d_v), surfChroma, ROUND_DOWN_2(int((px + 2) * 1)) + 1,
                                 (y + i * nSrcHeight / 2) * 1);
        }
    }
}

/* Input format is NV12 block-linear, output is scaled RGB */
bool nv12blDrawRect(const cudaArray_t *yuvInPlanes,
                    int nSrcPitch,
                    int nSrcWidth,
                    int nSrcHeight,
                    std::vector<NvInferObject> bbox,
                    cudaStream_t stream)
{

    cudaResourceDesc resDesc = {};
    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = yuvInPlanes[0];

    cudaSurfaceObject_t surfLuma = 0;
    checkCudaErrors(cudaCreateSurfaceObject(&surfLuma, &resDesc));

    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = yuvInPlanes[1];

    cudaSurfaceObject_t surfChroma = 0;
    checkCudaErrors(cudaCreateSurfaceObject(&surfChroma, &resDesc));

    dim3 dimBlock(32, 32, 1);

    unsigned int blockDimZ = 1;

    dim3 dimGrid((nSrcWidth / 4 + dimBlock.x - 1) / dimBlock.x, (nSrcHeight / 2 + dimBlock.y - 1) / dimBlock.y,
                 blockDimZ);

    for (std::size_t s = 0; s < bbox.size(); s++) {
        DrawRectKernel<<<dimGrid, dimBlock, 0, stream>>>(surfLuma, surfChroma, nSrcWidth, nSrcHeight, bbox[s].left / 4,
                                                         bbox[s].top / 2, bbox[s].width / 4, bbox[s].height / 2,
                                                         bbox[s].classIndex);
    }

    auto err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        return false;
    }

    checkCudaErrors(cudaDestroySurfaceObject(surfLuma));
    checkCudaErrors(cudaDestroySurfaceObject(surfChroma));

    return true;
}

static __global__ void DrawRectKernelPl(cudaSurfaceObject_t surfLuma,
                                        cudaSurfaceObject_t surfChroma,
                                        int nSrcWidth,
                                        int nSrcHeight,
                                        unsigned int o_x,
                                        unsigned int o_y,
                                        unsigned int o_w,
                                        unsigned int o_h,
                                        int class_index)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nSrcWidth || y >= nSrcHeight)
        return;

    uint8_t d_y, d_u, d_v;
    // Red YUV
    d_y = 65;
    d_u = 100;
    d_v = 212;

    if ((class_index == 0) &&
        (((x <= o_x + 1) && (x >= o_x - 1) && (y >= o_y - 1) && (y <= o_y + o_h + 1)) ||
         ((x <= o_x + o_w + 1) && (x >= o_x + o_w - 1) && (y >= o_y - 1) && (y <= o_y + o_h + 1)) ||
         ((y <= o_y + 1) && (y >= o_y - 1) && (x >= o_x - 1) && (x <= o_x + o_w + 1)) ||
         ((y <= o_y + o_h + 1) && (y >= o_y + o_h - 1) && (x >= o_x - 1) && (x <= o_x + o_w + 1)))) {
        surf2Dwrite<uint8_t>(d_y, surfLuma, x, y);
        if (x % 2 == 0 && y % 2 == 0) {
            surf2Dwrite<uint8_t>(d_u, surfChroma, x, y / 2);
            surf2Dwrite<uint8_t>(d_v, surfChroma, x + 1, y / 2);
        }
    }
}

/* Input format is NV12 pitch-linear, output is scaled RGB */
bool nv12plDrawRect(const cudaArray_t *yuvInPlanes,
                    int nSrcPitch,
                    int nSrcWidth,
                    int nSrcHeight,
                    std::vector<NvInferObject> bbox,
                    cudaStream_t stream)
{
    cudaResourceDesc resDesc = {};
    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = yuvInPlanes[0];

    cudaSurfaceObject_t surfLuma = 0;
    checkCudaErrors(cudaCreateSurfaceObject(&surfLuma, &resDesc));

    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = yuvInPlanes[1];

    cudaSurfaceObject_t surfChroma = 0;
    checkCudaErrors(cudaCreateSurfaceObject(&surfChroma, &resDesc));

    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid((nSrcWidth + dimBlock.x - 1) / dimBlock.x, (nSrcHeight + dimBlock.y - 1) / dimBlock.y, 1);

    for (std::size_t s = 0; s < bbox.size(); s++) {
        DrawRectKernelPl<<<dimGrid, dimBlock, 0, stream>>>(surfLuma, surfChroma, nSrcWidth, nSrcHeight, bbox[s].left,
                                                           bbox[s].top, bbox[s].width, bbox[s].height,
                                                           bbox[s].classIndex);
    }

    auto err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        return false;
    }

    checkCudaErrors(cudaDestroySurfaceObject(surfLuma));
    checkCudaErrors(cudaDestroySurfaceObject(surfChroma));

    return true;
}
