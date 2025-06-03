#include <3dgut/screenSpaceHeatmap.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernels
__global__ void accumulate_heatmap_kernel(
    const float* __restrict__ uvCoords,
    const float* __restrict__ values,
    float* __restrict__ heatmap,
    const int N, const int H, const int W,
    const float downscale
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        const float u_float = uvCoords[idx * 2] / downscale;
        const float v_float = uvCoords[idx * 2 + 1] / downscale;
        
        const int u = max(0, min(W - 1, static_cast<int>(roundf(u_float))));
        const int v = max(0, min(H - 1, static_cast<int>(roundf(v_float))));
        
        atomicAdd(&heatmap[v * W + u], values[idx]);
    }
}

__global__ void clear_heatmap_kernel(
    float* __restrict__ heatmap,
    const int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        heatmap[idx] = 0.0f;
    }
}

__global__ void normalize_minmax_kernel(
    float* __restrict__ heatmap,
    const int size,
    const float minVal,
    const float maxVal
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        const float range = maxVal - minVal;
        if (range > 0.0f) {
            heatmap[idx] = (heatmap[idx] - minVal) / range;
        } else {
            heatmap[idx] = 0.0f;
        }
    }
}


void launch_accumulate_heatmap_kernel(
    const float* uvCoords, 
    const float* values, 
    float* heatmap,
    int N, int H, int W, 
    float downscale, 
    cudaStream_t stream
) {
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    accumulate_heatmap_kernel<<<blocks, threadsPerBlock, 0, stream>>>(
        uvCoords, values, heatmap, N, H, W, downscale
    );
    
    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(error));
    }
}

void launch_clear_heatmap_kernel(
    float* heatmap, 
    int size, 
    cudaStream_t stream
) {
    const int threadsPerBlock = 256;
    const int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    clear_heatmap_kernel<<<blocks, threadsPerBlock, 0, stream>>>(
        heatmap, size
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(error));
    }
}

void launch_normalize_kernel(
    float* heatmap, 
    int size, 
    float minVal, 
    float maxVal, 
    cudaStream_t stream
) {
    const int threadsPerBlock = 256;
    const int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    normalize_minmax_kernel<<<blocks, threadsPerBlock, 0, stream>>>(
        heatmap, size, minVal, maxVal
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(error));
    }
}