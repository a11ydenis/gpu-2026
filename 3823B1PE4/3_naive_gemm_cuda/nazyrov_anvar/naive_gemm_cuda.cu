#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

namespace {

constexpr int BX = 32;  // потоки по столбцам: соседние потоки читают соседние элементы B и пишут соседние элементы C
constexpr int BY = 8;

__global__ void NaiveGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c, int n) {
    const int col = blockIdx.x * BX + threadIdx.x;
    const int row = blockIdx.y * BY + threadIdx.y;
    if (row >= n || col >= n) return;

    const float* a_row = a + (size_t)row * n;
    float sum = 0.0f;
    #pragma unroll 8
    for (int k = 0; k < n; ++k) {
        sum += a_row[k] * b[(size_t)k * n + col];
    }
    c[(size_t)row * n + col] = sum;
}

// Память на устройстве выделяется один раз и переиспользуется
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

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    const size_t count = (size_t)n * n;
    std::vector<float> c(count);
    if (count == 0) return c;

    EnsureCapacity(count);
    cudaMemcpy(d_a, a.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), count * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BX, BY);
    dim3 grid((n + BX - 1) / BX, (n + BY - 1) / BY);
    NaiveGemmKernel<<<grid, block>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c.data(), d_c, count * sizeof(float), cudaMemcpyDeviceToHost);
    return c;
}
