#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

namespace {

constexpr int TILE = 32;

__global__ void BlockGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE + 1];  // +1 убирает конфликты банков

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;  // накопление блока C в регистре, в конце пишем в глобальную память

    for (int t = 0; t < n; t += TILE) {
        As[ty][tx] = (row < n && t + tx < n) ? a[(size_t)row * n + t + tx] : 0.0f;
        Bs[ty][tx] = (t + ty < n && col < n) ? b[(size_t)(t + ty) * n + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        c[(size_t)row * n + col] = sum;
    }
}

float* d_a = nullptr;
float* d_b = nullptr;
float* d_c = nullptr;
size_t d_capacity = 0;

void EnsureCapacity(size_t count) {
    if (count <= d_capacity) return;
    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    if (d_c) cudaFree(d_c);
    cudaMalloc(&d_a, count * sizeof(float));
    cudaMalloc(&d_b, count * sizeof(float));
    cudaMalloc(&d_c, count * sizeof(float));
    d_capacity = count;
}

} // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const size_t count = (size_t)n * n;
    std::vector<float> c(count);
    if (count == 0) return c;

    EnsureCapacity(count);
    cudaMemcpy(d_a, a.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), count * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    BlockGemmKernel<<<grid, block>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c.data(), d_c, count * sizeof(float), cudaMemcpyDeviceToHost);
    return c;
}
