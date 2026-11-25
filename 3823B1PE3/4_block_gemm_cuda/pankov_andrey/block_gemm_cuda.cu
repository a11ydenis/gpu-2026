#include "block_gemm_cuda.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <stdexcept>

namespace {
constexpr int tile = 16;

__global__ void multiply(const float* a, const float* b, float* c, int n) {
    __shared__ float left[tile][tile], right[tile][tile];
    int x = threadIdx.x, y = threadIdx.y;
    int row = blockIdx.y * tile + y, col = blockIdx.x * tile + x;
    float sum = 0.0f;
    for (int start = 0; start < n; start += tile) {
        left[y][x] = row < n && start+x < n ? a[size_t(row)*n + start+x] : 0;
        right[y][x] = start+y < n && col < n ? b[size_t(start+y)*n + col] : 0;
        __syncthreads();
        for (int k = 0; k < tile; ++k) sum += left[y][k] * right[k][x];
        __syncthreads();
    }
    if (row < n && col < n) c[size_t(row)*n + col] = sum;
}
}

std::vector<float> BlockGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
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
