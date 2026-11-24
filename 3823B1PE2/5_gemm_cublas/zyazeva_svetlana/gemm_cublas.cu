#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(
    const std::vector<float>& a,
    const std::vector<float>& b,
    int n
) {
    if (n <= 0) {
        return {};
    }

    size_t bytes = static_cast<size_t>(n) * n * sizeof(float);

    static float* d_a = nullptr;
    static float* d_b = nullptr;
    static float* d_c = nullptr;
    static int capacity = 0;

    static cublasHandle_t handle = nullptr;

    if (handle == nullptr) {
        cublasCreate(&handle);
    }

    if (n > capacity) {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);

        capacity = n;
    }

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n, n, n,
        &alpha,
        d_b, n,
        d_a, n,
        &beta,
        d_c, n
    );

    std::vector<float> result(static_cast<size_t>(n) * n);

    cudaMemcpy(
        result.data(),
        d_c,
        bytes,
        cudaMemcpyDeviceToHost
    );

    return result;
}
