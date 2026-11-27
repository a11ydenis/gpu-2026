#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
namespace {
void Check(cudaError_t error) { if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); }
void Check(cublasStatus_t status) { if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS failure"); }
}
std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (n < 0 || a.size() != size_t(n) * n || b.size() != a.size()) throw std::invalid_argument("matrix size mismatch");
    std::vector<float> c(a.size());
    if (n == 0) return c;
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    cublasHandle_t handle = nullptr;
    size_t bytes = a.size() * sizeof(float);
    try {
        Check(cudaMalloc(&da, bytes)); Check(cudaMalloc(&db, bytes)); Check(cudaMalloc(&dc, bytes));
        Check(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
        Check(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
        Check(cublasCreate(&handle));
        const float alpha = 1, beta = 0;
        // Row-major A*B corresponds to column-major B^T*A^T.
        Check(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, db, n, da, n, &beta, dc, n));
        Check(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost));
    } catch (...) { if (handle) cublasDestroy(handle); cudaFree(da); cudaFree(db); cudaFree(dc); throw; }
    cublasDestroy(handle); cudaFree(da); cudaFree(db); cudaFree(dc);
    return c;
}
