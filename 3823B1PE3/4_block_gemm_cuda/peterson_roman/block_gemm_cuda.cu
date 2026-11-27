#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>
namespace {
void Check(cudaError_t error) { if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); }
constexpr int tile = 16;
__global__ void Multiply(const float* a, const float* b, float* c, int n) {
    __shared__ float left[tile][tile], right[tile][tile];
    int row = blockIdx.y * tile + threadIdx.y;
    int col = blockIdx.x * tile + threadIdx.x;
    float sum = 0;
    for (int base = 0; base < n; base += tile) {
        left[threadIdx.y][threadIdx.x] = row < n && base + threadIdx.x < n ? a[row * n + base + threadIdx.x] : 0;
        right[threadIdx.y][threadIdx.x] = col < n && base + threadIdx.y < n ? b[(base + threadIdx.y) * n + col] : 0;
        __syncthreads();
        for (int k = 0; k < tile; ++k) sum += left[threadIdx.y][k] * right[k][threadIdx.x];
        __syncthreads();
    }
    if (row < n && col < n) c[row * n + col] = sum;
}
}
std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (n < 0 || a.size() != size_t(n) * n || b.size() != a.size()) throw std::invalid_argument("matrix size mismatch");
    std::vector<float> c(a.size());
    if (n == 0) return c;
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    size_t bytes = a.size() * sizeof(float);
    try {
        Check(cudaMalloc(&da, bytes)); Check(cudaMalloc(&db, bytes)); Check(cudaMalloc(&dc, bytes));
        Check(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
        Check(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
        Multiply<<<dim3((n + 15) / 16, (n + 15) / 16), dim3(16, 16)>>>(da, db, dc, n);
        Check(cudaGetLastError());
        Check(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost));
    } catch (...) { cudaFree(da); cudaFree(db); cudaFree(dc); throw; }
    cudaFree(da); cudaFree(db); cudaFree(dc);
    return c;
}
