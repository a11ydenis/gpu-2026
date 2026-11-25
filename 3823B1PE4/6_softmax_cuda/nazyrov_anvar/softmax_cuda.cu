#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <math_constants.h>

namespace {

__device__ __forceinline__ float WarpMax(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ float WarpSum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ float BlockMax(float v, float* sh) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    v = WarpMax(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    float r = -CUDART_INF_F;
    for (int i = 0; i < nw; ++i) r = fmaxf(r, sh[i]);
    return r;
}

__device__ float BlockSum(float v, float* sh) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    v = WarpSum(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    float r = 0.0f;
    for (int i = 0; i < nw; ++i) r += sh[i];
    return r;
}

// Один блок обрабатывает одну строку
__global__ void SoftmaxKernel(const float* __restrict__ in, float* __restrict__ out, int row_size) {
    __shared__ float sh_max[32];
    __shared__ float sh_sum[32];

    const float* row_in = in + (size_t)blockIdx.x * row_size;
    float* row_out = out + (size_t)blockIdx.x * row_size;

    float m = -CUDART_INF_F;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        m = fmaxf(m, row_in[i]);
    }
    m = BlockMax(m, sh_max);

    float s = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        s += expf(row_in[i] - m);
    }
    s = BlockSum(s, sh_sum);

    const float inv = 1.0f / s;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        row_out[i] = expf(row_in[i] - m) * inv;
    }
}

float* d_in = nullptr;
float* d_out = nullptr;
size_t d_capacity = 0;

void EnsureCapacity(size_t count) {
    if (count <= d_capacity) return;
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    cudaMalloc(&d_in, count * sizeof(float));
    cudaMalloc(&d_out, count * sizeof(float));
    d_capacity = count;
}

} // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    const size_t total = input.size();
    std::vector<float> output(total);
    if (total == 0 || row_count <= 0) return output;

    const int row_size = (int)(total / row_count);

    EnsureCapacity(total);
    cudaMemcpy(d_in, input.data(), total * sizeof(float), cudaMemcpyHostToDevice);

    int threads = row_size < 1024 ? row_size : 1024;
    if (threads < 32) threads = 32;
    SoftmaxKernel<<<row_count, threads>>>(d_in, d_out, row_size);

    cudaMemcpy(output.data(), d_out, total * sizeof(float), cudaMemcpyDeviceToHost);
    return output;
}
