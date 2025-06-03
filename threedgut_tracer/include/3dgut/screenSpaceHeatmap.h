#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

class ScreenSpaceHeatmap {
    public:
        // Constructor
        ScreenSpaceHeatmap(int imageHeight, int imageWidth, int downscale = 4, bool useCuda = true);
        
        // Destructor
        ~ScreenSpaceHeatmap();
        
        // Core functionality
        void clear();
        void accumulate(const torch::Tensor& uvCoords, const torch::Tensor& values);
        void accumulateBatch(const std::vector<torch::Tensor>& uvCoordsList, 
                            const std::vector<torch::Tensor>& valuesList);
        
        // Processing functions
        void normalize(const std::string& method = "minmax");
        
        // Getters
        torch::Tensor getHeatmap() const;
        torch::Tensor getHeatmapCPU() const;
    
    private:
        // Dimensions
        int m_imageHeight, m_imageWidth;
        int m_height, m_width;
        int m_downscale;
        
        // CUDA state
        bool m_useCuda;
        int m_deviceId;
        cudaStream_t m_stream;
        
        // Heatmap data
        torch::Tensor m_heatmap;
        
        // Statistics tracking
        long long m_totalAccumulated;
        int m_accumulationCount;
        
        // Private helper functions
        void initializeCuda();
        void cleanupCuda();
        void validateInputs(const torch::Tensor& uvCoords, const torch::Tensor& values) const;
        torch::Tensor createHeatmapTensor() const;
};

void launch_accumulate_heatmap_kernel(
    const float* uvCoords, 
    const float* values, 
    float* heatmap,
    int N, int H, int W, 
    float downscale, 
    cudaStream_t stream
);

void launch_clear_heatmap_kernel(
    float* heatmap, 
    int size, 
    cudaStream_t stream
);

void launch_normalize_kernel(
    float* heatmap, 
    int size, 
    float minVal, 
    float maxVal, 
    cudaStream_t stream
);