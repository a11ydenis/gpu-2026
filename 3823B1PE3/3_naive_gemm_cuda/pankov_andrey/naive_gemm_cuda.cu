#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <stdexcept>

namespace {
__global__ void multiply(const float* a, const float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;
    float sum = 0.0f;
    for (int k = 0; k < n; ++k)
        sum += a[size_t(row) * n + k] * b[size_t(k) * n + col];
    c[size_t(row) * n + col] = sum;
}
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (n < 0 || a.size() != size_t(n)*n || a.size() != b.size())
        throw std::invalid_argument("invalid matrix dimensions");
    if (a.empty()) return {};
    thrust::device_vector<float> left(a.begin(), a.end()), right(b.begin(), b.end()), output(a.size());
    auto da = left.data().get(), db = right.data().get(), dc = output.data().get();
    dim3 block(16, 16), grid((n-1)/16+1, (n-1)/16+1);
    multiply<<<grid, block>>>(da, db, dc, n);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    std::vector<float> result(output.size());
    thrust::copy(output.begin(), output.end(), result.begin());
    return result;
}
