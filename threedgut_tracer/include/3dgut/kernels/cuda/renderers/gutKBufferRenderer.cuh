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
#include <stdio.h>
#include <3dgut/kernels/cuda/common/rayPayloadBackward.cuh>
#include <3dgut/renderer/gutRendererParameters.h>
#include <3dgut/kernels/cuda/common/cudaMath.cuh>


struct HitParticle {
    static constexpr float InvalidHitT = -1.0f;
    int idx                            = -1;
    float hitT                         = InvalidHitT;
    float alpha                        = 0.0f;
};

template <int K>
struct HitParticleKBuffer {
    __device__ HitParticleKBuffer() {
        m_numHits = 0;
#pragma unroll
        for (int i = 0; i < K; ++i) {
            m_kbuffer[i] = HitParticle();
        }
    }

    // insert a new hit into the kbuffer.
    // if the buffer is full overwrite the closest entry
    inline __device__ void insert(HitParticle& hitParticle) {
        const bool isFull = full();
        if (isFull) {
            m_kbuffer[0].hitT = HitParticle::InvalidHitT;
        } else {
            m_numHits++;
        }
#pragma unroll
        for (int i = K - 1; i >= 0; --i) {
            if (hitParticle.hitT > m_kbuffer[i].hitT) {
                const HitParticle tmp = m_kbuffer[i];
                m_kbuffer[i]          = hitParticle;
                hitParticle           = tmp;
            }
        }
    }

    inline __device__ const HitParticle& operator[](int i) const {
        return m_kbuffer[i];
    }

    inline __device__ uint32_t numHits() const {
        return m_numHits;
    }

    inline __device__ bool full() const {
        return m_numHits == K;
    }

    inline __device__ const HitParticle& closestHit(const HitParticle&) const {
        return m_kbuffer[0];
    }

private:
    HitParticle m_kbuffer[K];
    uint32_t m_numHits;
};

template <>
struct HitParticleKBuffer<0> {
    constexpr inline __device__ void insert(HitParticle& hitParticle) const {}
    constexpr inline __device__ HitParticle operator[](int) const { return HitParticle(); }
    constexpr inline __device__ uint32_t numHits() const { return 0; }
    constexpr inline __device__ bool full() const { return true; }
    constexpr inline __device__ const HitParticle& closestHit(const HitParticle& hitParticle) const { return hitParticle; }
};

template <typename Particles, typename Params, bool Backward = false>
struct GUTKBufferRenderer : Params {

    using DensityParameters    = typename Particles::DensityParameters;
    using DensityRawParameters = typename Particles::DensityRawParameters;
    using TFeaturesVec         = typename Particles::TFeaturesVec;

    using TRayPayload         = RayPayload<Particles::FeaturesDim>;
    using TRayPayloadBackward = RayPayloadBackward<Particles::FeaturesDim>;

    struct PrefetchedParticleData {
        uint32_t idx;
        DensityParameters densityParameters;
    };

    struct PrefetchedRawParticleData {
        uint32_t idx;
        TFeaturesVec features;
        DensityRawParameters densityParameters;
    };

    template <typename TRayPayload>
    static inline __device__ void processHitParticle(
        TRayPayload& ray,
        const HitParticle& hitParticle,
        const Particles& particles,
        const TFeaturesVec* __restrict__ particleFeatures,
        TFeaturesVec* __restrict__ particleFeaturesGradient) {

        if constexpr (Backward) {
            float hitAlphaGrad = 0.f;
            if constexpr (Params::PerRayParticleFeatures) {
                particles.featuresIntegrateBwdToBuffer<false>(ray.direction,
                                                              hitParticle.alpha,
                                                              hitAlphaGrad,
                                                              hitParticle.idx,
                                                              particles.featuresFromBuffer(hitParticle.idx, ray.direction),
                                                              ray.featuresBackward,
                                                              ray.featuresGradient);
            } else {
                TFeaturesVec particleFeaturesGradientVec = TFeaturesVec::zero();
                particles.featuresIntegrateBwd(hitParticle.alpha,
                                               hitAlphaGrad,
                                               particleFeatures[hitParticle.idx],
                                               particleFeaturesGradientVec,
                                               ray.featuresBackward,
                                               ray.featuresGradient);
#pragma unroll
                for (int i = 0; i < Particles::FeaturesDim; ++i) {
                    atomicAdd(&(particleFeaturesGradient[hitParticle.idx][i]), particleFeaturesGradientVec[i]);
                }
            }

            particles.densityProcessHitBwdToBuffer<false>(ray.origin,
                                                          ray.direction,
                                                          hitParticle.idx,
                                                          hitParticle.alpha,
                                                          hitAlphaGrad,
                                                          ray.transmittanceBackward,
                                                          ray.transmittanceGradient,
                                                          hitParticle.hitT,
                                                          ray.hitTBackward,
                                                          ray.hitTGradient);

            ray.transmittance *= (1.0 - hitParticle.alpha);

        } else {
            const float hitWeight =
                particles.densityIntegrateHit(hitParticle.alpha,
                                              ray.transmittance,
                                              hitParticle.hitT,
                                              ray.hitT);

            particles.featureIntegrateFwd(hitWeight,
                                          Params::PerRayParticleFeatures ? particles.featuresFromBuffer(hitParticle.idx, ray.direction) : tcnn::max(particleFeatures[hitParticle.idx], 0.f),
                                          ray.features);

            if (hitWeight > 0.0f) ray.countHit();
        }

        if (ray.transmittance < Particles::MinTransmittanceThreshold) {
            ray.kill();
        }
    }

    template <typename TRay>
    static inline __device__ void eval(const threedgut::RenderParameters& params,
                                  TRay& ray,
                                  const tcnn::uvec2* __restrict__ sortedTileRangeIndicesPtr,
                                  const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                  const tcnn::vec2* __restrict__ /*particlesProjectedPositionPtr*/,
                                  const tcnn::vec4* __restrict__ /*particlesProjectedConicOpacityPtr*/,
                                  const float* __restrict__ /*particlesGlobalDepthPtr*/,
                                  const float* __restrict__ particlesPrecomputedFeaturesPtr,
                                  threedgut::MemoryHandles parameters,
                                  // Multi-sampling parameters
                                  const int* __restrict__ sampleCountsPtr = nullptr,
                                  const float* __restrict__ sampleOffsetsPtr = nullptr,
                                  const float* __restrict__ sampleWeightsPtr = nullptr,
                                  // Gradient parameters
                                  tcnn::vec2* __restrict__ /*particlesProjectedPositionGradPtr*/     = nullptr,
                                  tcnn::vec4* __restrict__ /*particlesProjectedConicOpacityGradPtr*/ = nullptr,
                                  float* __restrict__ /*particlesGlobalDepthGradPtr*/                = nullptr,
                                  float* __restrict__ particlesPrecomputedFeaturesGradPtr            = nullptr,
                                  threedgut::MemoryHandles parametersGradient                        = {}) {

        using namespace threedgut;

        const uint32_t tileIdx                       = blockIdx.y * gridDim.x + blockIdx.x;
        const uint32_t tileThreadIdx                 = threadIdx.y * blockDim.x + threadIdx.x;
        const tcnn::uvec2 tileParticleRangeIndices   = sortedTileRangeIndicesPtr[tileIdx];
        uint32_t tileNumParticlesToProcess           = tileParticleRangeIndices.y - tileParticleRangeIndices.x;
        const uint32_t tileNumBlocksToProcess        = tcnn::div_round_up(tileNumParticlesToProcess, GUTParameters::Tiling::BlockSize);
        const TFeaturesVec* particleFeaturesBuffer   = Params::PerRayParticleFeatures ? nullptr : reinterpret_cast<const TFeaturesVec*>(particlesPrecomputedFeaturesPtr);
        TFeaturesVec* particleFeaturesGradientBuffer = (Params::PerRayParticleFeatures || !Backward) ? nullptr : reinterpret_cast<TFeaturesVec*>(particlesPrecomputedFeaturesGradPtr);

        // Check if multi-sampling is enabled
        const bool multiSamplingEnabled = (sampleCountsPtr != nullptr && sampleOffsetsPtr != nullptr && sampleWeightsPtr != nullptr);

        Particles particles;
        particles.initializeDensity(parameters);
        if constexpr (Backward) {
            particles.initializeDensityGradient(parametersGradient);
        }
        particles.initializeFeatures(parameters);
        if constexpr (Backward && Params::PerRayParticleFeatures) {
            particles.initializeFeaturesGradient(parametersGradient);
        }
        // printf("Buffer Size: %d ", Params::KHitBufferSize);
        // if constexpr (Backward) {
        //     printf("Running in backward mode\n");
        // }
        // New eval functions with multi-sampling
        if constexpr (Backward && (Params::KHitBufferSize == 0)) {
            evalBackwardNoKBuffer(ray, particles, tileParticleRangeIndices, tileNumBlocksToProcess, tileNumParticlesToProcess, tileThreadIdx,
                                                        sortedTileParticleIdxPtr, particleFeaturesBuffer, particleFeaturesGradientBuffer
                                                    );
        } else {
            evalKBufferWithMultiSamplingNoClass(ray, particles, tileParticleRangeIndices, tileNumBlocksToProcess, tileNumParticlesToProcess, tileThreadIdx,
                                            sortedTileParticleIdxPtr, particleFeaturesBuffer, particleFeaturesGradientBuffer,
                                            multiSamplingEnabled, sampleCountsPtr, sampleOffsetsPtr, sampleWeightsPtr);
        }
    }

    // Modified version of evalKBuffer with multi-sampling support
    template <typename TRay>
    static inline __device__ void evalKBufferWithMultiSamplingNoClass(
        TRay& ray,
        Particles& particles,
        const tcnn::uvec2& tileParticleRangeIndices,
        uint32_t tileNumBlocksToProcess,
        uint32_t tileNumParticlesToProcess,
        const uint32_t tileThreadIdx,
        const uint32_t* __restrict__ sortedTileParticleIdxPtr,
        const TFeaturesVec* __restrict__ particleFeaturesBuffer,
        TFeaturesVec* __restrict__ particleFeaturesGradientBuffer,
        bool multiSamplingEnabled,
        const int* __restrict__ sampleCountsPtr,
        const float* __restrict__ sampleOffsetsPtr,
        const float* __restrict__ sampleWeightsPtr) {
        using namespace threedgut;
        __shared__ PrefetchedParticleData prefetchedParticlesData[GUTParameters::Tiling::BlockSize];

        HitParticleKBuffer<Params::KHitBufferSize> hitParticleKBuffer;
        // printf("Buffer Size: %d ", Params::KHitBufferSize);

        for (uint32_t i = 0; i < tileNumBlocksToProcess; i++, tileNumParticlesToProcess -= GUTParameters::Tiling::BlockSize) {

            if (__syncthreads_and(!ray.isAlive())) {
                break;
            }

            // Collectively fetch particle data
            const uint32_t toProcessSortedIndex = tileParticleRangeIndices.x + i * GUTParameters::Tiling::BlockSize + tileThreadIdx;
            if (toProcessSortedIndex < tileParticleRangeIndices.y) {
                const uint32_t particleIdx = sortedTileParticleIdxPtr[toProcessSortedIndex];
                if (particleIdx != GUTParameters::InvalidParticleIdx) {
                    prefetchedParticlesData[tileThreadIdx] = {particleIdx, particles.fetchDensityParameters(particleIdx)};
                } else {
                    prefetchedParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
                }
            } else {
                prefetchedParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
            }
            __syncthreads();

            // Process fetched particles
            // printf("ray_x=%.4f, ray_y=%.4f\n", ray.tMinMax.x, ray.tMinMax.y);
            for (int j = 0; ray.isAlive() && j < min(GUTParameters::Tiling::BlockSize, tileNumParticlesToProcess); j++) {

                const PrefetchedParticleData particleData = prefetchedParticlesData[j];
                if (particleData.idx == GUTParameters::InvalidParticleIdx) {
                    i = tileNumBlocksToProcess;
                    break;
                }
                if (multiSamplingEnabled) {
                    // HitParticle hitParticle;
                    // hitParticle.idx = particleData.idx;
                    // if (particles.densityHit(ray.origin,
                    //                         ray.direction,
                    //                         particleData.densityParameters,
                    //                         hitParticle.alpha,
                    //                         hitParticle.hitT)){
                        
                    //     if (particleData.idx == 10) {
                    //         printf("In multisampling Particle %d, hitT=%.4f, alpha=%.4f, ray_x=%.4f, ray_y=%.4f\n", 
                    //             particleData.idx, hitParticle.hitT, hitParticle.alpha, ray.tMinMax.x, ray.tMinMax.y);
                    //     }       
                    //     if ((hitParticle.hitT > ray.tMinMax.x) &&
                    //     (hitParticle.hitT < ray.tMinMax.y)) {
                        
                    //         if (hitParticleKBuffer.full()) {
                    //             processHitParticle(ray,
                    //                             hitParticleKBuffer.closestHit(hitParticle),
                    //                             particles,
                    //                             particleFeaturesBuffer,
                    //                             particleFeaturesGradientBuffer);
                    //         }
                    //         hitParticleKBuffer.insert(hitParticle);
                    //     }
                    // }
                    // Multi-sampling using existing Particles methods
                    const int numSamples = sampleCountsPtr[particleData.idx];
                    const int maxSamples = MultiSampleParameters::MaxSamplesPerGaussian;
                    // printf("in multisampling: samples %d\n", numSamples);
                    
                    // Accumulate contributions from all samples
                    float totalAlpha = 0.0f;
                    float avgHitT = 0.0f;
                    float totalWeight = 0.0f;
                    
                    // //moves the original ray by different offset and check whether they are hitting the gaussian
                    float bestAlpha = 0.0f;
                    float bestHitT = ray.tMinMax.y + 1e10f; // Large number ensures any valid hit is closer
                    tcnn::vec3 ray_origin = ray.origin;
                    // // float bestHitTNorm = 0.0f; // Large number ensures any valid hit is closer
                    // // bool validHit = false;
                    for (int s = 0; s < numSamples; s++) {
                        const int sampleIdx = particleData.idx * maxSamples + s;
                        const float depthOffset = sampleOffsetsPtr[sampleIdx];
                        const float sampleWeight = sampleWeightsPtr[sampleIdx];
                        
                        // Create offset ray for this sample
                        tcnn::vec3 offsetRayOrigin = ray.origin;
                        tcnn::vec3 offsetRayDirection = ray.direction;
                        
                        // Adjust ray origin based on depth offset
                        // This effectively evaluates the Gaussian at different depths
                        // float offsetDistance = depthOffset * length(offsetRayDirection);
                        // offsetRayOrigin += offsetDistance * normalize(offsetRayDirection);
                        // printf("Depth offset: %.4f", depthOffset);
                        tcnn::vec3 samplePosition = ray.origin + depthOffset * ray.direction;

                        // printf("Sample Position: (%.6f, %.6f, %.6f), Ray Origin: (%.6f, %.6f, %.6f)\n",
                        // samplePosition.x, samplePosition.y, samplePosition.z,
                        // ray.origin.x, ray.origin.y, ray.origin.z);

                        // const int sampleIdx = particleData.idx * maxSamples + s;
                        // const float depthOffset = sampleOffsetsPtr[sampleIdx];
                        // const float sampleWeight = sampleWeightsPtr[sampleIdx];
                        
                        // Use const to match the signature cross(const float3&, const float3&)
                        // const float3 rayDir = make_float3(ray.direction.x, ray.direction.y, ray.direction.z);
                        // float3 worldUp = make_float3(0.0f, 1.0f, 0.0f);
                        // worldUp = fabs(rayDir.y) < 0.99f ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f);
                        
                        // // Now this should work without ambiguity
                        // const float3 rayRight_f3 = safe_normalize(make_float3(
                        //     rayDir.y * worldUp.z - rayDir.z * worldUp.y,  // rayDir.y * 0 - rayDir.z * 1 = -rayDir.z
                        //     rayDir.z * worldUp.x - rayDir.x * worldUp.z,  // rayDir.z * 0 - rayDir.x * 0 = 0
                        //     rayDir.x * worldUp.y - rayDir.y * worldUp.x   // rayDir.x * 1 - rayDir.y * 0 = rayDir.x
                        // ));
                        

                        // const float3 rayUp_f3 = make_float3(
                        //     rayDir.y * rayRight_f3.z - rayDir.z * rayRight_f3.y,
                        //     rayDir.z * rayRight_f3.x - rayDir.x * rayRight_f3.z,
                        //     rayDir.x * rayRight_f3.y - rayDir.y * rayRight_f3.x
                        // );

                        // // Convert back to tcnn::vec3
                        // tcnn::vec3 rayRight = tcnn::vec3(rayRight_f3.x, rayRight_f3.y, rayRight_f3.z);
                        // tcnn::vec3 rayUp = tcnn::vec3(rayUp_f3.x, rayUp_f3.y, rayUp_f3.z);
                        
                        // // Sample in circle
                        // float angle = (float)s / (float)numSamples * 2.0f * M_PI;
                        // float radius = abs(depthOffset);
                        // tcnn::vec3 perpOffset = radius * (cos(angle) * rayRight + sin(angle) * rayUp);
                        // // tcnn::vec3 samplePosition = ray.origin + perpOffset;
                        // samplePosition = ray.origin + perpOffset;
                        // tcnn::vec3 newDir = normalize(ray.origin - samplePosition);
                        // // Use existing densityHit with offset ray
                        // // HitParticle sampleHitParticle;
                        // // sampleHitParticle.idx = sampleIdx;

                        float sampleAlpha;
                        float sampleHitT;
                        
                        if (particles.densityHit(samplePosition,
                                            ray.direction,
                                            particleData.densityParameters,
                                            sampleAlpha,
                                            sampleHitT)) {
                            
                            // Adjust hit distance back to original ray space
                            // float globalHitT = depthOffset + sampleHitT;

                            // // Calculate the actual hit point in 3D space
                            // tcnn::vec3 hitPoint = samplePosition + sampleHitT * ray.direction;
                            
                            // // Project back onto the original ray to get the correct depth
                            // tcnn::vec3 rayToHit = hitPoint - ray.origin;
                            
                            // Manual dot product: rayToHit · ray.direction
                            // float globalHitT = rayToHit.x * ray.direction.x + 
                            //                 rayToHit.y * ray.direction.y + 
                            //                 rayToHit.z * ray.direction.z;

                            // if (particleData.idx == 0) {
                            //     printf("in multisampling: Particle %d, Sample %d: hitT=%.4f, globalhitT=%.4f, alpha=%.4f, ray_x=%.4f, ray_y=%.4f\n", 
                            //         particleData.idx, s, sampleHitT, globalHitT, sampleAlpha, ray.tMinMax.x, ray.tMinMax.y);
                            // }
                            // if ((sampleHitParticle.hitT > ray.tMinMax.x) &&
                            //     (sampleHitParticle.hitT < ray.tMinMax.y)) {
                                
                            //     if (hitParticleKBuffer.full()) {
                            //         processHitParticle(ray,
                            //                         hitParticleKBuffer.closestHit(sampleHitParticle),
                            //                         particles,
                            //                         particleFeaturesBuffer,
                            //                         particleFeaturesGradientBuffer);
                            //     }
                            //     hitParticleKBuffer.insert(sampleHitParticle);
                            // }

                            // if (globalHitT > ray.tMinMax.x && globalHitT < ray.tMinMax.y) {
                            totalAlpha += sampleAlpha * sampleWeight;
                            avgHitT += sampleHitT * sampleWeight;
                            totalWeight += sampleWeight;
                            

                            float weightedAlpha = sampleAlpha;

                            // Use hitT from sample with max weighted alpha
                            if (weightedAlpha > bestAlpha) {
                                bestAlpha = weightedAlpha;
                                bestHitT = sampleHitT;
                                ray_origin = samplePosition;
                                // printf("New best sample: s=%d, offset=%.6f, alpha=%.6f, hitT=%.6f, samplePos=(%.4f, %.4f, %.4f)\n",
                                //         s, depthOffset, bestAlpha, bestHitT,
                                //         samplePosition.x, samplePosition.y, samplePosition.z);
                            }
                            // }
                        }
                    }
                    
                    // Create hit particle with combined alpha
                    HitParticle hitParticle;
                    hitParticle.idx = particleData.idx;
                    // hitParticle.alpha = totalAlpha;
                    // hitParticle.hitT = avgHitT;
                    hitParticle.alpha = bestAlpha;
                    hitParticle.hitT = bestHitT;
                    ray.origin = ray_origin;
                    if (hitParticle.hitT > ray.tMinMax.x && hitParticle.hitT < ray.tMinMax.y) {
                        if (hitParticleKBuffer.full()) {
                            processHitParticle(ray,
                                            hitParticleKBuffer.closestHit(hitParticle),
                                            particles,
                                            particleFeaturesBuffer,
                                            particleFeaturesGradientBuffer);
                        }
                        hitParticleKBuffer.insert(hitParticle);
                    }
                    
                } 
                else {
                    // Original single-sample logic
                    HitParticle hitParticle;
                    hitParticle.idx = particleData.idx;
                    if (particles.densityHit(ray.origin,
                                            ray.direction,
                                            particleData.densityParameters,
                                            hitParticle.alpha,
                                            hitParticle.hitT)){
                        
                        // if (particleData.idx == 10) {
                        //     printf("Particle %d, hitT=%.4f, alpha=%.4f, ray_x=%.4f, ray_y=%.4f\n", 
                        //         particleData.idx, hitParticle.hitT, hitParticle.alpha, ray.tMinMax.x, ray.tMinMax.y);
                        // }       
                        if ((hitParticle.hitT > ray.tMinMax.x) &&
                        (hitParticle.hitT < ray.tMinMax.y)) {
                        
                        if (hitParticleKBuffer.full()) {
                                processHitParticle(ray,
                                                hitParticleKBuffer.closestHit(hitParticle),
                                                particles,
                                                particleFeaturesBuffer,
                                                particleFeaturesGradientBuffer);
                            }
                        hitParticleKBuffer.insert(hitParticle);
                        }
                    }   
                }
            }
        }

        if constexpr (Params::KHitBufferSize > 0) {
            // printf("Buffer Size: %d ", Params::KHitBufferSize);
            for (int i = 0; ray.isAlive() && (i < hitParticleKBuffer.numHits()); ++i) {
                processHitParticle(ray,
                                hitParticleKBuffer[Params::KHitBufferSize - hitParticleKBuffer.numHits() + i],
                                particles,
                                particleFeaturesBuffer,
                                particleFeaturesGradientBuffer);
            }
        }
    }

    template <typename TRay>
    static inline __device__ void evalBackwardNoKBuffer(TRay& ray,
                                                        Particles& particles,
                                                        const tcnn::uvec2& tileParticleRangeIndices,
                                                        uint32_t tileNumBlocksToProcess,
                                                        uint32_t tileNumParticlesToProcess,
                                                        const uint32_t tileThreadIdx,
                                                        const uint32_t* __restrict__ sortedTileParticleIdxPtr,
                                                        const TFeaturesVec* __restrict__ particleFeaturesBuffer,
                                                        TFeaturesVec* __restrict__ particleFeaturesGradientBuffer) {
        static_assert(Backward && (Params::KHitBufferSize == 0), "Optimized path for backward pass with no KBuffer");

        using namespace threedgut;
        __shared__ PrefetchedRawParticleData prefetchedRawParticlesData[GUTParameters::Tiling::BlockSize];

        for (uint32_t i = 0; i < tileNumBlocksToProcess; i++, tileNumParticlesToProcess -= GUTParameters::Tiling::BlockSize) {

            if (__syncthreads_and(!ray.isAlive())) {
                break;
            }

            // Collectively fetch particle data
            const uint32_t toProcessSortedIndex = tileParticleRangeIndices.x + i * GUTParameters::Tiling::BlockSize + tileThreadIdx;
            if (toProcessSortedIndex < tileParticleRangeIndices.y) {
                const uint32_t particleIdx = sortedTileParticleIdxPtr[toProcessSortedIndex];
                if (particleIdx != GUTParameters::InvalidParticleIdx) {
                    prefetchedRawParticlesData[tileThreadIdx].densityParameters = particles.fetchDensityRawParameters(particleIdx);
                    if constexpr (Params::PerRayParticleFeatures) {
                        prefetchedRawParticlesData[tileThreadIdx].features = TFeaturesVec::zero();
                    } else {
                        prefetchedRawParticlesData[tileThreadIdx].features = tcnn::max(particleFeaturesBuffer[particleIdx], 0.f);
                    }
                    prefetchedRawParticlesData[tileThreadIdx].idx = particleIdx;
                } else {
                    prefetchedRawParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
                }
            } else {
                prefetchedRawParticlesData[tileThreadIdx].idx = GUTParameters::InvalidParticleIdx;
            }
            __syncthreads();

            // Process fetched particles
            for (int j = 0; j < min(GUTParameters::Tiling::BlockSize, tileNumParticlesToProcess); j++) {

                if (__all_sync(GUTParameters::Tiling::WarpMask, !ray.isAlive())) {
                    break;
                }

                const PrefetchedRawParticleData particleData = prefetchedRawParticlesData[j];
                if (particleData.idx == GUTParameters::InvalidParticleIdx) {
                    ray.kill();
                    break;
                }

                DensityRawParameters densityRawParametersGrad;
                densityRawParametersGrad.density    = 0.0f;
                densityRawParametersGrad.position   = make_float3(0.0f);
                densityRawParametersGrad.quaternion = make_float4(0.0f);
                densityRawParametersGrad.scale      = make_float3(0.0f);

                TFeaturesVec featuresGrad = TFeaturesVec::zero();

                if (ray.isAlive()) {
                    particles.processHitBwd<Params::PerRayParticleFeatures>(
                        ray.origin,
                        ray.direction,
                        particleData.idx,
                        particleData.densityParameters,
                        &densityRawParametersGrad,
                        particleData.features,
                        &featuresGrad,
                        ray.transmittance,
                        ray.transmittanceBackward,
                        ray.transmittanceGradient,
                        ray.features,
                        ray.featuresBackward,
                        ray.featuresGradient,
                        ray.hitT,
                        ray.hitTBackward,
                        ray.hitTGradient);
                    if (ray.transmittance < Particles::MinTransmittanceThreshold) {
                        ray.kill();
                    }
                }

                if constexpr (!Params::PerRayParticleFeatures) {
                    particles.processHitBwdUpdateFeaturesGradient(particleData.idx, featuresGrad,
                                                                  particleFeaturesGradientBuffer, tileThreadIdx);
                }
                particles.processHitBwdUpdateDensityGradient(particleData.idx, densityRawParametersGrad, tileThreadIdx);
            }
        }
    }
};