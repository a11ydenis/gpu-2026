#include "gelu_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kBlockSize = 256;
constexpr float kLinear = 1.5957691216f;
constexpr float kCubic = 0.0713548163f;

void CheckCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(error));
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t bytes) {
        CheckCuda(cudaMalloc(reinterpret_cast<void**>(&data_), bytes),
                  "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    float* get() const {
        return data_;
    }

private:
    float* data_ = nullptr;
};

__device__ __forceinline__ float GeluApprox(float x) {
    const float argument = x * (kLinear + kCubic * x * x);
    return x / (1.0f + __expf(-argument));
}

__global__ void GeluVectorKernel(const float4* __restrict__ input,
                                 float4* __restrict__ output,
                                 std::size_t vectorCount) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < vectorCount) {
        float4 values = input[index];
        values.x = GeluApprox(values.x);
        values.y = GeluApprox(values.y);
        values.z = GeluApprox(values.z);
        values.w = GeluApprox(values.w);
        output[index] = values;
    }
}

__global__ void GeluTailKernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               std::size_t tailSize) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < tailSize) {
        output[index] = GeluApprox(input[index]);
    }
}

} // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    if (input.empty()) {
        return {};
    }

    const std::size_t size = input.size();
    const std::size_t bytes = size * sizeof(float);
    const std::size_t vectorCount = size / 4;
    const std::size_t tailSize = size % 4;
    std::vector<float> output(size);
    DeviceBuffer deviceInput(bytes);
    DeviceBuffer deviceOutput(bytes);

    CheckCuda(cudaMemcpy(deviceInput.get(), input.data(), bytes,
                         cudaMemcpyHostToDevice),
              "cudaMemcpy host to device");

    if (vectorCount != 0) {
        const unsigned int gridSize = static_cast<unsigned int>(
            (vectorCount + kBlockSize - 1) / kBlockSize);
        GeluVectorKernel<<<gridSize, kBlockSize>>>(
            reinterpret_cast<const float4*>(deviceInput.get()),
            reinterpret_cast<float4*>(deviceOutput.get()), vectorCount);
        CheckCuda(cudaGetLastError(), "GeluVectorKernel launch");
    }

    if (tailSize != 0) {
        float* inputTail = deviceInput.get() + vectorCount * 4;
        float* outputTail = deviceOutput.get() + vectorCount * 4;
        GeluTailKernel<<<1, kBlockSize>>>(inputTail, outputTail, tailSize);
        CheckCuda(cudaGetLastError(), "GeluTailKernel launch");
    }

    CheckCuda(cudaMemcpy(output.data(), deviceOutput.get(), bytes,
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy device to host");

    return output;
}
