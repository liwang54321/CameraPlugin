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

#include "cvt2d_kernels.h"

#define CV_ROUND_DOWN_2(num) ((num) & (~1))

__forceinline__ __device__ static float clampCvF(float x, float lower, float upper)
{
    return x < lower ? lower : (x > upper ? upper : x);
}


static __global__ void CvImageTransformKernel2RGBPlanar(cudaTextureObject_t texLuma,
                                                      cudaTextureObject_t texChroma,
                                                      // YUVImage in,
                                                      float *pDst,
                                                      int nDstWidth,
                                                      int nDstHeight,
                                                      float fxScale,
                                                      float fyScale,
                                                      bool isBGR,
                                                      bool isPlanar,
                                                      float Gamma,
                                                      float Beta,
                                                      int nBatchSize)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int px = x * 4, py = y * 2;
    if ((px + 3) >= nDstWidth || (py + 1) >= nDstHeight)
        return;

    uchar4 luma2x01, luma2x23, uv2;

    for (int i = blockIdx.z; i < nBatchSize; i += gridDim.z) {
        *(uchar4 *)&luma2x01 =
            make_uchar4(tex2D<uint8_t>(texLuma, px * fxScale, (py + i * nDstHeight) * fyScale),
                        tex2D<uint8_t>(texLuma, (px + 1) * fxScale, (py + i * nDstHeight) * fyScale),
                        tex2D<uint8_t>(texLuma, (px + 2) * fxScale, (py + i * nDstHeight) * fyScale),
                        tex2D<uint8_t>(texLuma, (px + 3) * fxScale, (py + i * nDstHeight) * fyScale));

        *(uchar4 *)&luma2x23 =
            make_uchar4(tex2D<uint8_t>(texLuma, px * fxScale, ((py + i * nDstHeight) + 1) * fyScale),
                        tex2D<uint8_t>(texLuma, (px + 1) * fxScale, ((py + i * nDstHeight) + 1) * fyScale),
                        tex2D<uint8_t>(texLuma, (px + 2) * fxScale, ((py + i * nDstHeight) + 1) * fyScale),
                        tex2D<uint8_t>(texLuma, (px + 3) * fxScale, ((py + i * nDstHeight) + 1) * fyScale));

        *(uchar4 *)&uv2 = make_uchar4(
            tex2D<uint8_t>(texChroma, CV_ROUND_DOWN_2(int(px * fxScale)), (y + i * nDstHeight / 2) * fyScale),
            tex2D<uint8_t>(texChroma, CV_ROUND_DOWN_2(int(px * fxScale)) + 1, (y + i * nDstHeight / 2) * fyScale),
            tex2D<uint8_t>(texChroma, CV_ROUND_DOWN_2(int((px + 2) * fxScale)), (y + i * nDstHeight / 2) * fyScale),
            tex2D<uint8_t>(texChroma, CV_ROUND_DOWN_2(int((px + 2) * fxScale)) + 1, (y + i * nDstHeight / 2) * fyScale));

        float2 add00, add01, add02, add03;

        add00.x = 1.1644f * luma2x01.x;
        add01.x = 1.1644f * luma2x01.y;
        add00.y = 1.1644f * luma2x01.z;
        add01.y = 1.1644f * luma2x01.w;

        add02.x = 1.1644f * luma2x23.x;
        add03.x = 1.1644f * luma2x23.y;
        add02.y = 1.1644f * luma2x23.z;
        add03.y = 1.1644f * luma2x23.w;

        float2 add1, add2, add3;

        add1.x = 2.0172f * (uv2.x - 128.0f);
        add1.y = 2.0172f * (uv2.z - 128.0f);

        add2.x = (-0.3918f) * (uv2.x - 128.0f) + (-0.8130f) * (uv2.y - 128.0f);
        add2.y = (-0.3918f) * (uv2.z - 128.0f) + (-0.8130f) * (uv2.w - 128.0f);

        add3.x = 1.5960f * (uv2.y - 128.0f);
        add3.y = 1.5960f * (uv2.w - 128.0f);

        float r00 = Gamma * clampCvF(add00.x + add3.x, 0.0f, 255.0f) + Beta;
        float r01 = Gamma * clampCvF(add01.x + add3.x, 0.0f, 255.0f) + Beta;
        float r02 = Gamma * clampCvF(add00.y + add3.y, 0.0f, 255.0f) + Beta;
        float r03 = Gamma * clampCvF(add01.y + add3.y, 0.0f, 255.0f) + Beta;
        float r10 = Gamma * clampCvF(add02.x + add3.x, 0.0f, 255.0f) + Beta;
        float r11 = Gamma * clampCvF(add03.x + add3.x, 0.0f, 255.0f) + Beta;
        float r12 = Gamma * clampCvF(add02.y + add3.y, 0.0f, 255.0f) + Beta;
        float r13 = Gamma * clampCvF(add03.y + add3.y, 0.0f, 255.0f) + Beta;

        float g00 = Gamma * clampCvF(add00.x + add2.x, 0.0f, 255.0f) + Beta;
        float g01 = Gamma * clampCvF(add01.x + add2.x, 0.0f, 255.0f) + Beta;
        float g02 = Gamma * clampCvF(add00.y + add2.y, 0.0f, 255.0f) + Beta;
        float g03 = Gamma * clampCvF(add01.y + add2.y, 0.0f, 255.0f) + Beta;
        float g10 = Gamma * clampCvF(add02.x + add2.x, 0.0f, 255.0f) + Beta;
        float g11 = Gamma * clampCvF(add03.x + add2.x, 0.0f, 255.0f) + Beta;
        float g12 = Gamma * clampCvF(add02.y + add2.y, 0.0f, 255.0f) + Beta;
        float g13 = Gamma * clampCvF(add03.y + add2.y, 0.0f, 255.0f) + Beta;

        float b00 = Gamma * clampCvF(add00.x + add1.x, 0.0f, 255.0f) + Beta;
        float b01 = Gamma * clampCvF(add01.x + add1.x, 0.0f, 255.0f) + Beta;
        float b02 = Gamma * clampCvF(add00.y + add1.y, 0.0f, 255.0f) + Beta;
        float b03 = Gamma * clampCvF(add01.y + add1.y, 0.0f, 255.0f) + Beta;
        float b10 = Gamma * clampCvF(add02.x + add1.x, 0.0f, 255.0f) + Beta;
        float b11 = Gamma * clampCvF(add03.x + add1.x, 0.0f, 255.0f) + Beta;
        float b12 = Gamma * clampCvF(add02.y + add1.y, 0.0f, 255.0f) + Beta;
        float b13 = Gamma * clampCvF(add03.y + add1.y, 0.0f, 255.0f) + Beta;

        if (isPlanar) {
            float *p_0 = pDst + i * 3 * nDstWidth * nDstHeight + px + py * nDstWidth;
            float *p_1 = pDst + i * 3 * nDstWidth * nDstHeight + px + (py + 1) * nDstWidth;

            int plane_offset = nDstWidth * nDstHeight;

            *(float *)(p_0 + 0) = float(isBGR ? b00 : r00);
            *(float *)(p_0 + 1) = float(isBGR ? b01 : r01);
            *(float *)(p_0 + 2) = float(isBGR ? b02 : r02);
            *(float *)(p_0 + 3) = float(isBGR ? b03 : r03);
            *(float *)(p_1 + 0) = float(isBGR ? b10 : r10);
            *(float *)(p_1 + 1) = float(isBGR ? b11 : r11);
            *(float *)(p_1 + 2) = float(isBGR ? b12 : r12);
            *(float *)(p_1 + 3) = float(isBGR ? b13 : r13);

            *(float *)(p_0 + plane_offset + 0) = float(g00);
            *(float *)(p_0 + plane_offset + 1) = float(g01);
            *(float *)(p_0 + plane_offset + 2) = float(g02);
            *(float *)(p_0 + plane_offset + 3) = float(g03);
            *(float *)(p_1 + plane_offset + 0) = float(g10);
            *(float *)(p_1 + plane_offset + 1) = float(g11);
            *(float *)(p_1 + plane_offset + 2) = float(g12);
            *(float *)(p_1 + plane_offset + 3) = float(g13);

            *(float *)(p_0 + 2 * plane_offset + 0) = float(isBGR ? r00 : b00);
            *(float *)(p_0 + 2 * plane_offset + 1) = float(isBGR ? r01 : b01);
            *(float *)(p_0 + 2 * plane_offset + 2) = float(isBGR ? r02 : b02);
            *(float *)(p_0 + 2 * plane_offset + 3) = float(isBGR ? r03 : b03);
            *(float *)(p_1 + 2 * plane_offset + 0) = float(isBGR ? r10 : b10);
            *(float *)(p_1 + 2 * plane_offset + 1) = float(isBGR ? r11 : b11);
            *(float *)(p_1 + 2 * plane_offset + 2) = float(isBGR ? r12 : b12);
            *(float *)(p_1 + 2 * plane_offset + 3) = float(isBGR ? r13 : b13);
        }
    }
}

/* Input format is NV12 block-linear, output is scaled RGB */
bool CvtNv12blToRgbPlanar(const cudaArray_t *yuvInPlanes,
                          int nSrcWidth,
                          int nSrcHeight,
                          void *dpTsr,
                          int nTsrWidth,
                          int nTsrHeight,
                          int nBatchSize,
                          float scaleFactor,
                          cudaStream_t stream)
{

    /* Calculate the scaling ratio of the frame / object crop. This will be
   * required later for rescaling the detector output boxes to input resolution.
   */

    cudaResourceDesc resDesc = {};
    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = yuvInPlanes[0];

    cudaTextureDesc texDesc = {};
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t texLuma = 0;
    checkCudaErrors(cudaCreateTextureObject(&texLuma, &resDesc, &texDesc, NULL));

    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = yuvInPlanes[1];

    cudaTextureObject_t texChroma = 0;
    checkCudaErrors(cudaCreateTextureObject(&texChroma, &resDesc, &texDesc, NULL));

    dim3 dimBlock(32, 32, 1);
    unsigned int blockDimZ = nBatchSize;

    // Restricting blocks in Z-dim till 32 to not launch too many blocks
    blockDimZ = (blockDimZ > 32) ? 32 : blockDimZ;

    dim3 dimGrid((nTsrWidth / 4 + dimBlock.x - 1) / dimBlock.x, (nTsrHeight / 2 + dimBlock.y - 1) / dimBlock.y,
                 blockDimZ);

    float fxScale = 1.0f * nSrcWidth / nTsrWidth;
    float fyScale = 1.0f * nSrcHeight / nTsrHeight;

    CvImageTransformKernel2RGBPlanar<<<dimGrid, dimBlock, 0, stream>>>(texLuma, texChroma, (float *)dpTsr, nTsrWidth,
                                                                     nTsrHeight, fxScale, fyScale, false, true,
                                                                     scaleFactor, 0.0, nBatchSize);

    auto err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        return false;
    }

    checkCudaErrors(cudaDestroyTextureObject(texLuma));
    checkCudaErrors(cudaDestroyTextureObject(texChroma));

    return true;
}