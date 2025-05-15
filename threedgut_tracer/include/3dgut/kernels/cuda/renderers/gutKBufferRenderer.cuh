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

#include <3dgut/kernels/cuda/common/rayPayloadBackward.cuh>
#include <3dgut/renderer/gutRendererParameters.h>

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

        // Use modified versions without changing the Particles class
        if constexpr (Backward && (Params::KHitBufferSize == 0)) {
            evalBackwardNoKBufferWithMultiSamplingNoClass(ray, particles, tileParticleRangeIndices, tileNumBlocksToProcess, tileNumParticlesToProcess, tileThreadIdx,
                                                        sortedTileParticleIdxPtr, particleFeaturesBuffer, particleFeaturesGradientBuffer,
                                                        multiSamplingEnabled, sampleCountsPtr, sampleOffsetsPtr, sampleWeightsPtr);
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
            for (int j = 0; ray.isAlive() && j < min(GUTParameters::Tiling::BlockSize, tileNumParticlesToProcess); j++) {

                const PrefetchedParticleData particleData = prefetchedParticlesData[j];
                if (particleData.idx == GUTParameters::InvalidParticleIdx) {
                    i = tileNumBlocksToProcess;
                    break;
                }

                if (multiSamplingEnabled) {
                    // Multi-sampling using existing Particles methods
                    const int numSamples = sampleCountsPtr[particleData.idx];
                    const int maxSamples = MultiSampleParameters::MaxSamplesPerGaussian;
                    
                    // Accumulate contributions from all samples
                    float totalAlpha = 0.0f;
                    float avgHitT = 0.0f;
                    float totalWeight = 0.0f;
                    
                    for (int s = 0; s < numSamples; s++) {
                        const int sampleIdx = particleData.idx * maxSamples + s;
                        const float depthOffset = sampleOffsetsPtr[sampleIdx];
                        const float sampleWeight = sampleWeightsPtr[sampleIdx];
                        
                        // Create offset ray for this sample
                        tcnn::vec3 offsetRayOrigin = ray.origin;
                        tcnn::vec3 offsetRayDirection = ray.direction;
                        
                        // Adjust ray origin based on depth offset
                        // This effectively evaluates the Gaussian at different depths
                        float offsetDistance = depthOffset * length(offsetRayDirection);
                        offsetRayOrigin += offsetDistance * normalize(offsetRayDirection);
                        
                        // Use existing densityHit with offset ray
                        float sampleAlpha;
                        float sampleHitT;
                        if (particles.densityHit(offsetRayOrigin,
                                            offsetRayDirection,
                                            particleData.densityParameters,
                                            sampleAlpha,
                                            sampleHitT)) {
                            
                            // Adjust hit distance back to original ray space
                            sampleHitT += offsetDistance;
                            
                            if (sampleHitT > ray.tMinMax.x && sampleHitT < ray.tMinMax.y) {
                                totalAlpha += sampleAlpha * sampleWeight;
                                avgHitT += sampleHitT * sampleWeight;
                                totalWeight += sampleWeight;
                            }
                        }
                    }
                    
                    // Create hit particle with combined alpha
                    if (totalWeight > 0) {
                        HitParticle hitParticle;
                        hitParticle.idx = particleData.idx;
                        hitParticle.alpha = totalAlpha / totalWeight;
                        hitParticle.hitT = avgHitT / totalWeight;
                        
                        if (hitParticleKBuffer.full()) {
                            processHitParticle(ray,
                                            hitParticleKBuffer.closestHit(hitParticle),
                                            particles,
                                            particleFeaturesBuffer,
                                            particleFeaturesGradientBuffer);
                        }
                        hitParticleKBuffer.insert(hitParticle);
                    }
                } else {
                    // Original single-sample logic
                    HitParticle hitParticle;
                    hitParticle.idx = particleData.idx;
                    if (particles.densityHit(ray.origin,
                                            ray.direction,
                                            particleData.densityParameters,
                                            hitParticle.alpha,
                                            hitParticle.hitT) &&
                        (hitParticle.hitT > ray.tMinMax.x) &&
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

        if constexpr (Params::KHitBufferSize > 0) {
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
    static inline __device__ void evalBackwardNoKBufferWithMultiSamplingNoClass(
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
                    if (multiSamplingEnabled) {
                        // Multi-sampling backward pass without modifying Particles class
                        const int numSamples = sampleCountsPtr[particleData.idx];
                        const int maxSamples = MultiSampleParameters::MaxSamplesPerGaussian;
                        
                        // Process each sample independently and accumulate gradients
                        for (int s = 0; s < numSamples; s++) {
                            const int sampleIdx = particleData.idx * maxSamples + s;
                            const float depthOffset = sampleOffsetsPtr[sampleIdx];
                            const float sampleWeight = sampleWeightsPtr[sampleIdx];
                            
                            // Create per-sample gradient accumulators
                            DensityRawParameters sampleDensityGrad = densityRawParametersGrad;
                            TFeaturesVec sampleFeaturesGrad = TFeaturesVec::zero();
                            
                            // Create offset ray for this sample
                            tcnn::vec3 offsetRayOrigin = ray.origin;
                            tcnn::vec3 offsetRayDirection = ray.direction;
                            float offsetDistance = depthOffset * length(offsetRayDirection);
                            offsetRayOrigin += offsetDistance * normalize(offsetRayDirection);
                            
                            // Temporarily modify ray state for this sample
                            float originalHitT = ray.hitT;
                            ray.hitT += offsetDistance;  // Adjust hit distance for offset
                            
                            // Use existing processHitBwd with modified ray
                            particles.processHitBwd<Params::PerRayParticleFeatures>(
                                offsetRayOrigin,
                                offsetRayDirection,
                                particleData.idx,
                                particleData.densityParameters,
                                &sampleDensityGrad,
                                particleData.features,
                                &sampleFeaturesGrad,
                                ray.transmittance,
                                ray.transmittanceBackward,
                                ray.transmittanceGradient,
                                ray.features,
                                ray.featuresBackward,
                                ray.featuresGradient,
                                ray.hitT,
                                ray.hitTBackward,
                                ray.hitTGradient);
                            
                            // Restore original hit distance
                            ray.hitT = originalHitT;
                            
                            // Accumulate weighted gradients
                            densityRawParametersGrad.density += sampleDensityGrad.density * sampleWeight;
                            densityRawParametersGrad.position += sampleDensityGrad.position * sampleWeight;
                            densityRawParametersGrad.quaternion += sampleDensityGrad.quaternion * sampleWeight;
                            densityRawParametersGrad.scale += sampleDensityGrad.scale * sampleWeight;
                            featuresGrad += sampleFeaturesGrad * sampleWeight;
                        }
                        
                    } else {
                        // Original single-sample backward pass
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
                    }
                    
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