#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kTileSize = 32;
constexpr int kRowsPerThread = 4;
constexpr int kThreadsY = kTileSize / kRowsPerThread;

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

__global__ void BlockGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    __shared__ float tileA[kTileSize][kTileSize];
    __shared__ float tileB[kTileSize][kTileSize];

    const int column = blockIdx.x * kTileSize + threadIdx.x;
    const int firstRow =
        blockIdx.y * kTileSize + threadIdx.y * kRowsPerThread;
    float sums[kRowsPerThread] = {};

    for (int tileStart = 0; tileStart < n; tileStart += kTileSize) {
        #pragma unroll
        for (int r = 0; r < kRowsPerThread; ++r) {
            const int localRow = threadIdx.y * kRowsPerThread + r;
            const int row = firstRow + r;
            const int aColumn = tileStart + threadIdx.x;
            const int bRow = tileStart + localRow;

            tileA[localRow][threadIdx.x] =
                (row < n && aColumn < n)
                ? a[static_cast<std::size_t>(row) * n + aColumn]
                : 0.0f;
            tileB[localRow][threadIdx.x] =
                (bRow < n && column < n)
                ? b[static_cast<std::size_t>(bRow) * n + column]
                : 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < kTileSize; ++k) {
            const float bValue = tileB[k][threadIdx.x];
            #pragma unroll
            for (int r = 0; r < kRowsPerThread; ++r) {
                const int localRow = threadIdx.y * kRowsPerThread + r;
                sums[r] += tileA[localRow][k] * bValue;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
        const int row = firstRow + r;
        if (row < n && column < n) {
            c[static_cast<std::size_t>(row) * n + column] = sums[r];
        }
    }
}

} // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
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

    const dim3 block(kTileSize, kThreadsY);
    const dim3 grid((n - 1) / kTileSize + 1,
                    (n - 1) / kTileSize + 1);
    BlockGemmKernel<<<grid, block>>>(deviceA.get(), deviceB.get(),
                                     deviceC.get(), n);
    CheckCuda(cudaGetLastError(), "BlockGemmKernel launch");
    CheckCuda(cudaMemcpy(c.data(), deviceC.get(), bytes,
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy C device to host");

    return c;
}
