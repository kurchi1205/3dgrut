#include <3dgut/screenSpaceHeatmap.h>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>


ScreenSpaceHeatmap::ScreenSpaceHeatmap(int imageHeight, int imageWidth, int downscale, bool useCuda)
    : m_imageHeight(imageHeight)
    , m_imageWidth(imageWidth)
    , m_downscale(downscale)
    , m_useCuda(useCuda && torch::cuda::is_available())
    , m_deviceId(0)
    , m_stream(nullptr)
    , m_totalAccumulated(0)
    , m_accumulationCount(0)
{
    // Calculate heatmap dimensions
    m_height = m_imageHeight / m_downscale;
    m_width = m_imageWidth / m_downscale;
    
    if (m_height <= 0 || m_width <= 0) {
        throw std::invalid_argument("Invalid heatmap dimensions after downscaling");
    }
    
    // Initialize CUDA if requested
    if (m_useCuda) {
        initializeCuda();
    }
    
    // Create heatmap tensor
    m_heatmap = createHeatmapTensor();
    
    std::cout << "Created ScreenSpaceHeatmap: " << m_height << "x" << m_width 
              << " (downscale=" << m_downscale << ") on " 
              << (m_useCuda ? "CUDA" : "CPU") << std::endl;
}

ScreenSpaceHeatmap::~ScreenSpaceHeatmap() {
    if (m_useCuda) {
        cleanupCuda();
    }
}

void ScreenSpaceHeatmap::initializeCuda() {
    try {
        CUDA_CHECK(cudaGetDevice(&m_deviceId));
        CUDA_CHECK(cudaStreamCreate(&m_stream));
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize CUDA: " << e.what() << std::endl;
        m_useCuda = false;
        m_stream = nullptr;
    }
}

void ScreenSpaceHeatmap::cleanupCuda() {
    if (m_stream) {
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
    }
}

torch::Tensor ScreenSpaceHeatmap::createHeatmapTensor() const {
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32);
    
    if (m_useCuda) {
        options = options.device(torch::kCUDA, m_deviceId);
    } else {
        options = options.device(torch::kCPU);
    }
    
    return torch::zeros({m_height, m_width}, options);
}

void ScreenSpaceHeatmap::clear() {
    if (m_useCuda) {
        // Use CUDA kernel for clearing
        launch_clear_heatmap_kernel(
            m_heatmap.data_ptr<float>(), 
            m_heatmap.numel(), 
            m_stream
        );
        CUDA_CHECK(cudaStreamSynchronize(m_stream));
    } else {
        // CPU version
        m_heatmap.zero_();
    }
    
    // Reset statistics
    m_totalAccumulated = 0;
    m_accumulationCount = 0;
}

void ScreenSpaceHeatmap::validateInputs(const torch::Tensor& uvCoords, const torch::Tensor& values) const {
    if (!uvCoords.defined() || !values.defined()) {
        throw std::invalid_argument("Input tensors must be defined");
    }
    
    if (uvCoords.dim() != 2 || uvCoords.size(1) != 2) {
        throw std::invalid_argument("uvCoords must have shape [N, 2]");
    }
    
    if (values.dim() != 1) {
        throw std::invalid_argument("values must have shape [N]");
    }
    
    if (uvCoords.size(0) != values.size(0)) {
        throw std::invalid_argument("uvCoords and values must have same batch size");
    }
    
    if (uvCoords.dtype() != torch::kFloat32 || values.dtype() != torch::kFloat32) {
        throw std::invalid_argument("Input tensors must be float32");
    }
    
    if (m_useCuda) {
        if (!uvCoords.is_cuda() || !values.is_cuda()) {
            throw std::invalid_argument("Input tensors must be on CUDA when using CUDA heatmap");
        }
        if (uvCoords.device().index() != m_deviceId || values.device().index() != m_deviceId) {
            throw std::invalid_argument("Input tensors must be on the same CUDA device as heatmap");
        }
    }
}

void ScreenSpaceHeatmap::accumulate(const torch::Tensor& uvCoords, const torch::Tensor& values) {
    // Handle empty input
    if (uvCoords.numel() == 0 || values.numel() == 0) {
        return;
    }
    
    // Validate inputs
    validateInputs(uvCoords, values);
    
    // Ensure tensors are contiguous
    torch::Tensor uvContiguous = uvCoords.contiguous();
    torch::Tensor valuesContiguous = values.contiguous();
    
    const int N = uvContiguous.size(0);
    
    if (m_useCuda) {
        // Use CUDA kernel
        launch_accumulate_heatmap_kernel(
            uvContiguous.data_ptr<float>(),
            valuesContiguous.data_ptr<float>(),
            m_heatmap.data_ptr<float>(),
            N, m_height, m_width, 
            static_cast<float>(m_downscale),
            m_stream
        );
        CUDA_CHECK(cudaStreamSynchronize(m_stream));
    } else {
        // CPU fallback using PyTorch operations
        torch::Tensor u = (uvContiguous.select(1, 0) / m_downscale).to(torch::kLong).clamp(0, m_width - 1);
        torch::Tensor v = (uvContiguous.select(1, 1) / m_downscale).to(torch::kLong).clamp(0, m_height - 1);
        
        torch::Tensor indices = v * m_width + u;
        torch::Tensor accumulated = torch::bincount(indices, valuesContiguous, m_height * m_width);
        m_heatmap += accumulated.view({m_height, m_width});
    }
    
    // Update statistics
    m_totalAccumulated += values.sum().item<float>();
    m_accumulationCount += N;
}

void ScreenSpaceHeatmap::accumulateBatch(const std::vector<torch::Tensor>& uvCoordsList, 
                                        const std::vector<torch::Tensor>& valuesList) {
    if (uvCoordsList.empty() || valuesList.empty()) {
        return;
    }
    
    if (uvCoordsList.size() != valuesList.size()) {
        throw std::invalid_argument("uvCoordsList and valuesList must have same size");
    }
    
    // Concatenate all batches
    torch::Tensor allUvCoords = torch::cat(uvCoordsList, 0);
    torch::Tensor allValues = torch::cat(valuesList, 0);
    
    accumulate(allUvCoords, allValues);
}

void ScreenSpaceHeatmap::normalize(const std::string& method) {
    if (method == "minmax") {
        if (m_useCuda) {
            // Get min/max values
            float minVal = m_heatmap.min().item<float>();
            float maxVal = m_heatmap.max().item<float>();
            
            if (maxVal > minVal) {
                launch_normalize_kernel(
                    m_heatmap.data_ptr<float>(),
                    m_heatmap.numel(),
                    minVal, maxVal,
                    m_stream
                );
                CUDA_CHECK(cudaStreamSynchronize(m_stream));
            } else {
                clear();
            }
        } else {
            // CPU version
            float minVal = m_heatmap.min().item<float>();
            float maxVal = m_heatmap.max().item<float>();
            
            if (maxVal > minVal) {
                m_heatmap = (m_heatmap - minVal) / (maxVal - minVal);
            } else {
                m_heatmap.zero_();
            }
        }
    } else if (method == "zscore") {
        // Z-score normalization
        float mean = m_heatmap.mean().item<float>();
        float std = m_heatmap.std().item<float>();
        
        if (std > 0) {
            m_heatmap = (m_heatmap - mean) / std;
        } else {
            m_heatmap.zero_();
        }
    } else {
        throw std::invalid_argument("Unknown normalization method: " + method);
    }
}
