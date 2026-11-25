#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace {

cublasHandle_t handle = nullptr;
float* d_a = nullptr;
float* d_b = nullptr;
float* d_c = nullptr;
size_t d_capacity = 0;

void EnsureCapacity(size_t count) {
    if (!handle) cublasCreate(&handle);  // дескриптор создаётся один раз
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

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    const size_t count = (size_t)n * n;
    std::vector<float> c(count);
    if (count == 0) return c;

    EnsureCapacity(count);
    cudaMemcpy(d_a, a.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), count * sizeof(float), cudaMemcpyHostToDevice);

    // cuBLAS работает с хранением по столбцам. Матрица, лежащая по строкам, для него
    // это транспонированная. Считаем C^T = B^T * A^T, тогда результат лежит по строкам.
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                d_b, n,
                d_a, n,
                &beta,
                d_c, n);

    cudaMemcpy(c.data(), d_c, count * sizeof(float), cudaMemcpyDeviceToHost);
    return c;
}
