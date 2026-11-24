#include "gelu_cuda.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <stdexcept>

namespace {
__global__ void gelu(float* data, size_t count) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        float x = data[i];
        data[i] = 0.5f * x * (1.0f + tanhf(0.7978845608f * x * (1.0f + 0.044715f * x * x)));
    }
}
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    if (input.empty()) return {};
    thrust::device_vector<float> output(input.begin(), input.end());
    gelu<<<(input.size()+255)/256, 256>>>(output.data().get(), input.size());
    auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    std::vector<float> result(output.size());
    thrust::copy(output.begin(), output.end(), result.begin());
    return result;
}
