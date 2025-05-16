// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <3dgut/kernels/cuda/common/rayPayload.cuh>
#include <3dgut/renderer/gutRendererParameters.h>

__global__ void projectOnTiles(tcnn::uvec2 tileGrid,
                               uint32_t numParticles,
                               tcnn::ivec2 resolution,
                               threedgut::TSensorModel sensorModel,
                               tcnn::vec3 sensorWorldPosition,
                               tcnn::mat4x3 sensorViewMatrix,
                               threedgut::TSensorState sensorShutterState,
                               uint32_t* __restrict__ particlesTilesOffsetPtr,
                               tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                               tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                               tcnn::vec2* __restrict__ particlesProjectedExtentPtr,
                               float* __restrict__ particlesGlobalDepthPtr,
                               float* __restrict__ particlesPrecomputedFeaturesPtr,
                               int* __restrict__ particlesVisibilityCudaPtr,
                               const uint64_t* __restrict__ parameterMemoryHandles) {

    TGUTProjector::eval(tileGrid,
                        numParticles,
                        resolution,
                        sensorModel,
                        sensorWorldPosition,
                        sensorViewMatrix,
                        sensorShutterState,
                        particlesTilesOffsetPtr,
                        particlesProjectedPositionPtr,
                        particlesProjectedConicOpacityPtr,
                        particlesProjectedExtentPtr,
                        particlesGlobalDepthPtr,
                        particlesPrecomputedFeaturesPtr,
                        particlesVisibilityCudaPtr,
                        {parameterMemoryHandles});
}

__global__ void expandTileProjections(tcnn::uvec2 tileGrid,
                                      uint32_t numParticles,
                                      tcnn::ivec2 resolution,
                                      threedgut::TSensorModel sensorModel,
                                      threedgut::TSensorState sensorState,
                                      const uint32_t* __restrict__ particlesTilesOffsetPtr,
                                      const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                                      const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                                      const tcnn::vec2* __restrict__ particlesProjectedExtentPtr,
                                      const float* __restrict__ particlesGlobalDepthPtr,
                                      const uint64_t* __restrict__ parameterMemoryHandles,
                                      uint64_t* __restrict__ unsortedTileDepthKeysPtr,
                                      uint32_t* __restrict__ unsortedTileParticleIdxPtr) {

    TGUTProjector::expand(tileGrid,
                          numParticles,
                          resolution,
                          sensorModel,
                          sensorState,
                          particlesTilesOffsetPtr,
                          particlesProjectedPositionPtr,
                          particlesProjectedConicOpacityPtr,
                          particlesProjectedExtentPtr,
                          particlesGlobalDepthPtr,
                          {parameterMemoryHandles},
                          unsortedTileDepthKeysPtr,
                          unsortedTileParticleIdxPtr);
}

__global__ void render(threedgut::RenderParameters params,
                       const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
                       const uint32_t* __restrict__ sortedTileDataPtr,
                       const tcnn::vec3* __restrict__ sensorRayOriginPtr,
                       const tcnn::vec3* __restrict__ sensorRayDirectionPtr,
                       tcnn::mat4x3 sensorToWorldTransform,
                       float* __restrict__ worldHitCountPtr,
                       float* __restrict__ worldHitDistancePtr,
                       tcnn::vec4* __restrict__ radianceDensityPtr,
                       const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                       const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                       const float* __restrict__ particlesGlobalDepthPtr,
                       const float* __restrict__ particlesPrecomputedFeaturesPtr,
                       const uint64_t* __restrict__ parameterMemoryHandles,
                       // New multi-sampling parameters
                       const int* __restrict__ sampleCounts,
                       const float* __restrict__ sampleOffsets,
                       const float* __restrict__ sampleWeights) {

    auto ray = initializeRay<TGUTRenderer::TRayPayload>(
        params, sensorRayOriginPtr, sensorRayDirectionPtr, sensorToWorldTransform);

    // Call updated eval with multi-sampling support
    TGUTRenderer::eval(params,
                        ray,
                        sortedTileRangeIndicesPtr,
                        sortedTileDataPtr,
                        particlesProjectedPositionPtr,
                        particlesProjectedConicOpacityPtr,
                        particlesGlobalDepthPtr,
                        particlesPrecomputedFeaturesPtr,
                        {parameterMemoryHandles},
                        sampleCounts,
                        sampleOffsets,
                        sampleWeights);

    finalizeRay(ray, params, sensorRayOriginPtr, worldHitCountPtr, worldHitDistancePtr, radianceDensityPtr, sensorToWorldTransform);
}

__global__ void renderBackward(threedgut::RenderParameters params,
                               const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
                               const uint32_t* __restrict__ sortedTileDataPtr,
                               const tcnn::vec3* __restrict__ sensorRayOriginPtr,
                               const tcnn::vec3* __restrict__ sensorRayDirectionPtr,
                               tcnn::mat4x3 sensorToWorldTransform,
                               const float* __restrict__ worldHitDistancePtr,
                               const float* __restrict__ worldHitDistanceGradientPtr,
                               const tcnn::vec4* __restrict__ radianceDensityPtr,
                               const tcnn::vec4* __restrict__ radianceDensityGradientPtr,
                               tcnn::vec3* __restrict__ /*worldRayOriginGradientPtr*/,
                               tcnn::vec3* __restrict__ /*worldRayDirectionGradientPtr*/,
                               const tcnn::vec2* __restrict__ particlesProjectedPositionPtr,
                               const tcnn::vec4* __restrict__ particlesProjectedConicOpacityPtr,
                               const float* __restrict__ particlesGlobalDepthPtr,
                               const float* __restrict__ particlesPrecomputedFeaturesPtr,
                               const uint64_t* __restrict__ parameterMemoryHandles,
                               tcnn::vec2* __restrict__ particlesProjectedPositionGradPtr,
                               tcnn::vec4* __restrict__ particlesProjectedConicOpacityGradPtr,
                               float* __restrict__ particlesGlobalDepthGradPtr,
                               float* __restrict__ particlesPrecomputedFeaturesGradPtr,
                               const uint64_t* __restrict__ parameterGradientMemoryHandles
                               // Multi-sampling parameters
                               const int* __restrict__ sampleCounts,
                               const float* __restrict__ sampleOffsets,
                               const float* __restrict__ sampleWeights) {

    auto ray = initializeBackwardRay<TGUTRenderer::TRayPayloadBackward>(params,
                                                                        sensorRayOriginPtr,
                                                                        sensorRayDirectionPtr,
                                                                        worldHitDistancePtr,
                                                                        worldHitDistanceGradientPtr,
                                                                        radianceDensityPtr,
                                                                        radianceDensityGradientPtr,
                                                                        sensorToWorldTransform);

    // TGUTModel::evalBackward(params, ray, {parameterMemoryHandles}, {parameterGradientMemoryHandles});

    TGUTBackwardRenderer::eval(params,
                               ray,
                               sortedTileRangeIndicesPtr,
                               sortedTileDataPtr,
                               particlesProjectedPositionPtr,
                               particlesProjectedConicOpacityPtr,
                               particlesGlobalDepthPtr,
                               particlesPrecomputedFeaturesPtr,
                               {parameterMemoryHandles},
                               // Multi-sampling parameters
                               sampleCounts,
                               sampleOffsets,
                               sampleWeights,
                               particlesProjectedPositionGradPtr,
                               particlesProjectedConicOpacityGradPtr,
                               particlesGlobalDepthGradPtr,
                               particlesPrecomputedFeaturesGradPtr,
                               {parameterGradientMemoryHandles});
}

__global__ void projectBackward(tcnn::uvec2 tileGrid,
                                uint32_t numParticles,
                                tcnn::ivec2 resolution,
                                threedgut::TSensorModel sensorModel,
                                tcnn::vec3 sensorWorldPosition,
                                tcnn::mat4x3 sensorViewMatrix,
                                const uint32_t* __restrict__ particlesTilesCountPtr,
                                const uint64_t* __restrict__ parameterMemoryHandles,
                                const tcnn::vec2* __restrict__ particlesProjectedPositionGradPtr,
                                const tcnn::vec4* __restrict__ particlesProjectedConicOpacityGradPtr,
                                const float* __restrict__ particlesGlobalDepthGradPtr,
                                const float* __restrict__ particlesPrecomputedFeaturesPtr,
                                const float* __restrict__ particlesPrecomputedFeaturesGradPtr,
                                const uint64_t* __restrict__ parameterGradientMemoryHandles) {

    TGUTProjector::evalBackward(tileGrid,
                                numParticles,
                                resolution,
                                sensorModel,
                                sensorWorldPosition,
                                sensorViewMatrix,
                                particlesTilesCountPtr,
                                {parameterMemoryHandles},
                                particlesProjectedPositionGradPtr,
                                particlesProjectedConicOpacityGradPtr,
                                particlesGlobalDepthGradPtr,
                                particlesPrecomputedFeaturesPtr,
                                particlesPrecomputedFeaturesGradPtr,
                                {parameterGradientMemoryHandles});
}
'''
projects each particles gradient to the screen space heatmap and accumulates it for each pixel
'''
__global__ void accumulateGradientHeatmap(
    const uint32_t numParticles,
    const vec2* __restrict__ projectedPositions,
    const float* __restrict__ gradientMagnitudes,
    const uvec2 heatmapSize,
    const int downscale,
    float* __restrict__ gradientHeatmap
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    // Get projected position and gradient magnitude
    vec2 pos = projectedPositions[idx];
    float magnitude = gradientMagnitudes[idx];
    
    // Convert to heatmap coordinates
    int hx = (int)(pos.x / downscale);
    int hy = (int)(pos.y / downscale);
    
    // Bounds check
    if (hx >= 0 && hx < heatmapSize.x && hy >= 0 && hy < heatmapSize.y) {
        int heatmapIdx = hy * heatmapSize.x + hx;
        atomicAdd(&gradientHeatmap[heatmapIdx], magnitude);
    }
}

'''
find  overlapping particles based on the depth and spatial distance, for now 8 other overlapping particles
'''
__global__ void detectOverlappingParticles(
    const uint32_t numParticles,
    const vec2* __restrict__ projectedPositions,
    const float* __restrict__ depths,
    const float spatialRadius,
    const float depthThreshold,
    int* __restrict__ overlappingIndices,
    int* __restrict__ overlappingCounts
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    vec2 myPos = projectedPositions[idx];
    float myDepth = depths[idx];
    int overlapCount = 0;
    
    // Check all other particles (simplified - in practice use spatial grid)
    for (uint32_t j = 0; j < numParticles; j++) {
        if (j == idx) continue;
        
        vec2 otherPos = projectedPositions[j];
        float otherDepth = depths[j];
        
        float distance = length(myPos - otherPos);
        float depthDiff = fabsf(myDepth - otherDepth);
        
        if (distance < spatialRadius && depthDiff < depthThreshold) {
            if (overlapCount < 8) {  // Max 8 overlaps tracked
                overlappingIndices[idx * 8 + overlapCount] = j;
                overlapCount++;
            }
        }
    }
    
    overlappingCounts[idx] = overlapCount;
}

'''
calculate the number of samples from gradient and number of overlapping particles. Range is set to 10 percent of the depth range.
Offset is evenly distributed. weights are normally distributed, that is highest weight is at the depth of the gaussian.
'''

__global__ void computeAdaptiveSampleCounts(
    const uint32_t numParticles,
    const vec2* __restrict__ projectedPositions,
    const float* __restrict__ gradientHeatmap,
    const uvec2 heatmapSize,
    const int downscale,
    const int* __restrict__ overlappingCounts,
    int* __restrict__ sampleCounts,
    float* __restrict__ sampleOffsets,
    float* __restrict__ sampleWeights
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    // Get position in heatmap
    vec2 pos = projectedPositions[idx];
    int hx = (int)(pos.x / downscale);
    int hy = (int)(pos.y / downscale);
    
    // Get gradient value at this position
    float gradientValue = 0.0f;
    if (hx >= 0 && hx < heatmapSize.x && hy >= 0 && hy < heatmapSize.y) {
        gradientValue = gradientHeatmap[hy * heatmapSize.x + hx];
    }
    
    // Normalize gradient (simplified - in practice track max)
    gradientValue = fminf(gradientValue / MultiSampleParameters::GradientThreshold, 1.0f);
    
    // Determine sample count based on gradient and overlaps
    int baseSamples = MultiSampleParameters::BaseSamples;
    int maxSamples = MultiSampleParameters::MaxSamplesPerGaussian;
    int overlapCount = overlappingCounts[idx];
    
    // More samples for high gradient or overlapping regions
    int samples = baseSamples;
    if (gradientValue > 0.5f || overlapCount > 0) {
        samples = baseSamples + (int)((maxSamples - baseSamples) * gradientValue);
        samples = max(samples, baseSamples + overlapCount);
        samples = min(samples, maxSamples);
    }
    
    sampleCounts[idx] = samples;
    
    // Generate sample offsets and weights
    if (samples > 1) {
        float range = MultiSampleParameters::SampleRange;
        for (int s = 0; s < samples; s++) {
            float t = (float)s / (float)(samples - 1);  // 0 to 1
            float offset = (t - 0.5f) * 2.0f * range;   // -range to +range
            sampleOffsets[idx * maxSamples + s] = offset;
            
            // Gaussian weights centered at 0
            float sigma = range / 3.0f;
            float weight = expf(-0.5f * (offset * offset) / (sigma * sigma));
            sampleWeights[idx * maxSamples + s] = weight;
        }
        
        // Normalize weights
        float weightSum = 0.0f;
        for (int s = 0; s < samples; s++) {
            weightSum += sampleWeights[idx * maxSamples + s];
        }
        for (int s = 0; s < samples; s++) {
            sampleWeights[idx * maxSamples + s] /= weightSum;
        }
    } else {
        sampleOffsets[idx * maxSamples] = 0.0f;
        sampleWeights[idx * maxSamples] = 1.0f;
    }
}

'''
for each particle computes the L2 norm of the gradient vector
'''
__global__ void computeGradientMagnitudes(
    const uint32_t numParticles,
    const uint32_t featureDim,
    const float* __restrict__ gradients,
    float* __restrict__ magnitudes
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;
    
    float magnitude = 0.0f;
    for (uint32_t i = 0; i < featureDim; i++) {
        float grad = gradients[idx * featureDim + i];
        magnitude += grad * grad;
    }
    magnitudes[idx] = sqrtf(magnitude);
}
