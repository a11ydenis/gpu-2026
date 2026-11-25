#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <stdexcept>
#include <cublas_v2.h>

namespace {
struct BlasHandle {
    cublasHandle_t handle{};
    BlasHandle() {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasCreate failed");
    }
    ~BlasHandle() { cublasDestroy(handle); }
};
}

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (n < 0 || a.size() != size_t(n)*n || a.size() != b.size())
        throw std::invalid_argument("invalid matrix dimensions");
    if (a.empty()) return {};
    thrust::device_vector<float> left(a.begin(), a.end()), right(b.begin(), b.end()), output(a.size());
    auto da = left.data().get(), db = right.data().get(), dc = output.data().get();
    BlasHandle blas;
    float alpha = 1, beta = 0;
    // Row-major C = A B is column-major C^T = B^T A^T.
    if (cublasSgemm(blas.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                    &alpha, db, n, da, n, &beta, dc, n) != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cublasSgemm failed");
    std::vector<float> result(output.size());
    thrust::copy(output.begin(), output.end(), result.begin());
    return result;
}
