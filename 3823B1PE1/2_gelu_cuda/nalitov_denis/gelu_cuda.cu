#include "gelu_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace {

constexpr float kScale = 1.5957691216057308f;
constexpr float kCubic = kScale * 0.044715f;
constexpr int kThreads = 256;

void throw_if_cuda_error(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

__global__ void gelu_kernel(float* data, std::size_t n) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float x = data[i];
    const float scaled = x * (kScale + kCubic * x * x);
    data[i] = x / (1.0f + __expf(-scaled));
}

struct GpuBuffer {
    float* data = nullptr;
    std::size_t capacity = 0;
    cudaStream_t stream = nullptr;

    GpuBuffer() {
        throw_if_cuda_error(cudaStreamCreate(&stream));
    }

    ~GpuBuffer() {
        cudaFree(data);
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }

    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;
    GpuBuffer(GpuBuffer&&) = delete;
    GpuBuffer& operator=(GpuBuffer&&) = delete;

    void grow(std::size_t n) {
        if (n <= capacity) {
            return;
        }
        throw_if_cuda_error(cudaFree(data));
        throw_if_cuda_error(cudaMalloc(&data, n * sizeof(float)));
        capacity = n;
    }
};

GpuBuffer& gpu_buffer() {
    static GpuBuffer buffer;
    return buffer;
}

}  // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    if (input.empty()) {
        return {};
    }

    const std::size_t n = input.size();
    const std::size_t bytes = n * sizeof(float);
    GpuBuffer& buffer = gpu_buffer();
    buffer.grow(n);

    throw_if_cuda_error(cudaMemcpyAsync(buffer.data, input.data(), bytes,
                                        cudaMemcpyHostToDevice, buffer.stream));

    const int blocks = static_cast<int>((n + static_cast<std::size_t>(kThreads) - 1) /
                                        static_cast<std::size_t>(kThreads));
    gelu_kernel<<<blocks, kThreads, 0, buffer.stream>>>(buffer.data, n);
    throw_if_cuda_error(cudaGetLastError());

    std::vector<float> output(n);

    throw_if_cuda_error(cudaMemcpyAsync(output.data(), buffer.data, bytes,
                                        cudaMemcpyDeviceToHost, buffer.stream));
    throw_if_cuda_error(cudaStreamSynchronize(buffer.stream));
    return output;
}
