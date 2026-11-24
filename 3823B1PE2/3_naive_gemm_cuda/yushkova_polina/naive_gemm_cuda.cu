#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kColumnsPerBlock = 32;
constexpr int kRowsPerBlock = 32;
constexpr int kRowsPerThread = 4;
constexpr int kThreadsY = kRowsPerBlock / kRowsPerThread;

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

__global__ void NaiveGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int firstRow =
        blockIdx.y * kRowsPerBlock + threadIdx.y * kRowsPerThread;
    if (column >= n || firstRow >= n) {
        return;
    }

    float sums[kRowsPerThread] = {};
    for (int k = 0; k < n; ++k) {
        const float bValue =
            b[static_cast<std::size_t>(k) * n + column];
        #pragma unroll
        for (int r = 0; r < kRowsPerThread; ++r) {
            const int row = firstRow + r;
            if (row < n) {
                sums[r] += a[static_cast<std::size_t>(row) * n + k] * bValue;
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
        const int row = firstRow + r;
        if (row < n) {
            c[static_cast<std::size_t>(row) * n + column] = sums[r];
        }
    }
}

} // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n < 0) {
        throw std::invalid_argument("Matrix size must not be negative");
    }
    const std::size_t elementCount = static_cast<std::size_t>(n) * n;
    if (a.size() != elementCount || b.size() != elementCount) {
        throw std::invalid_argument("Input matrices must have size n * n");
    }
    if (n == 0) {
        return {};
    }

    std::vector<float> c(elementCount);
    const std::size_t bytes = elementCount * sizeof(float);
    DeviceBuffer deviceA(bytes);
    DeviceBuffer deviceB(bytes);
    DeviceBuffer deviceC(bytes);

    CheckCuda(cudaMemcpy(deviceA.get(), a.data(), bytes,
                         cudaMemcpyHostToDevice),
              "cudaMemcpy A host to device");
    CheckCuda(cudaMemcpy(deviceB.get(), b.data(), bytes,
                         cudaMemcpyHostToDevice),
              "cudaMemcpy B host to device");

    const dim3 block(kColumnsPerBlock, kThreadsY);
    const dim3 grid((n - 1) / kColumnsPerBlock + 1,
                    (n - 1) / kRowsPerBlock + 1);
    NaiveGemmKernel<<<grid, block>>>(deviceA.get(), deviceB.get(),
                                     deviceC.get(), n);
    CheckCuda(cudaGetLastError(), "NaiveGemmKernel launch");
    CheckCuda(cudaMemcpy(c.data(), deviceC.get(), bytes,
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy C device to host");

    return c;
}
