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

#ifndef CARCOMMON_HPP
#define CARCOMMON_HPP

#include <iostream>
#include <stdint.h>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#if !NV_BUILD_DOS7
#include <cudla.h>
#endif

#define EXIT_SUCCESS 0 /* Successful exit status. */
#define EXIT_FAILURE 1 /* Failing exit status.    */
#define EXIT_WAIVED 2  /* WAIVED exit status.     */

enum class NvInferNetworkMode
{
    FP16,
    INT8
};

struct NvInferInitParams
{
    NvInferNetworkMode networkMode;

    std::string int8CalibrationFilePath;
    std::string int8ModelCacheFilePath;
    std::string fp16ModelCacheFilePath;
    std::string int8TrtEngineFilePath;

    std::vector<std::vector<std::string>> labels;
    std::string testImageFilePath;

    float networkScaleFactor; /* Normalization factor to scale the input pixels. */

    std::string inputImageLayerName;
    std::string outputCoverageLayerName;
    std::string outputBboxLayerName;

    double scoreThresh;
    double nmsThresh;
    double groupIouThresh;
    unsigned int groupThresh;
};

struct NvInferObject
{
    unsigned int left;
    unsigned int top;
    unsigned int width;
    unsigned int height;
    int classIndex;
    std::string label;
};

struct NvInferFrameOutput
{
    std::vector<NvInferObject> objects;
};

struct Bndbox
{
    int xMin;
    int yMin;
    int xMax;
    int yMax;
    float score;
    Bndbox() {};
    Bndbox(int xMin_, int yMin_, int xMax_, int yMax_, float score_)
        : xMin(xMin_)
        , yMin(yMin_)
        , xMax(xMax_)
        , yMax(yMax_)
        , score(score_)
    {
    }
};

#endif