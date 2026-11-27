#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace {
void Check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
__global__ void GeluKernel(const float* input, float* output, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = input[i];
        float z = 0.7978845608f * (x + 0.044715f * x * x * x);
        output[i] = 0.5f * x * (1.0f + tanhf(z));
    }
}
}
std::vector<float> GeluCUDA(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    if (input.empty()) return output;
    float *device_input = nullptr, *device_output = nullptr;
    size_t bytes = input.size() * sizeof(float);
    try {
        Check(cudaMalloc(&device_input, bytes));
        Check(cudaMalloc(&device_output, bytes));
        Check(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice));
        GeluKernel<<<(input.size() + 255) / 256, 256>>>(device_input, device_output, input.size());
        Check(cudaGetLastError());
        Check(cudaMemcpy(output.data(), device_output, bytes, cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(device_input);
        cudaFree(device_output);
        throw;
    }
    cudaFree(device_input);
    cudaFree(device_output);
    return output;
}
