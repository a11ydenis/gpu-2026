#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>
namespace {
void Check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
__global__ void Multiply(const float* a, const float* b, float* c, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= n) return;
    float sum = 0;
    for (int k = 0; k < n; ++k) sum += a[row * n + k] * b[k * n + col];
    c[row * n + col] = sum;
}
}
std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (n < 0 || a.size() != size_t(n) * n || b.size() != a.size()) throw std::invalid_argument("matrix size mismatch");
    std::vector<float> c(a.size());
    if (n == 0) return c;
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    size_t bytes = a.size() * sizeof(float);
    try {
        Check(cudaMalloc(&da, bytes)); Check(cudaMalloc(&db, bytes)); Check(cudaMalloc(&dc, bytes));
        Check(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
        Check(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
        dim3 threads(16, 16);
        dim3 blocks((n + 15) / 16, (n + 15) / 16);
        Multiply<<<blocks, threads>>>(da, db, dc, n);
        Check(cudaGetLastError());
        Check(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost));
    } catch (...) { cudaFree(da); cudaFree(db); cudaFree(dc); throw; }
    cudaFree(da); cudaFree(db); cudaFree(dc);
    return c;
}
