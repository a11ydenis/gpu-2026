#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <stdexcept>
#include <math_constants.h>

namespace {
constexpr int threads = 256;

__global__ void softmax(float* data, size_t width) {
    __shared__ float partial[threads];
    float* row = data + size_t(blockIdx.x) * width;
    int lane = threadIdx.x;
    float maximum = -CUDART_INF_F;
    for (size_t j = lane; j < width; j += threads) maximum = fmaxf(maximum, row[j]);
    partial[lane] = maximum;
    __syncthreads();
    for (int step = threads/2; step; step /= 2) {
        if (lane < step) partial[lane] = fmaxf(partial[lane], partial[lane+step]);
        __syncthreads();
    }
    maximum = partial[0];
    __syncthreads();
    float sum = 0;
    for (size_t j = lane; j < width; j += threads) {
        row[j] = expf(row[j] - maximum);
        sum += row[j];
    }
    partial[lane] = sum;
    __syncthreads();
    for (int step = threads/2; step; step /= 2) {
        if (lane < step) partial[lane] += partial[lane+step];
        __syncthreads();
    }
    for (size_t j = lane; j < width; j += threads) row[j] /= partial[0];
}
}

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (input.empty()) return {};
    if (row_count <= 0 || input.size() % row_count)
        throw std::invalid_argument("invalid row count");
    thrust::device_vector<float> output(input.begin(), input.end());
    softmax<<<row_count, threads>>>(output.data().get(), input.size()/row_count);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    std::vector<float> result(output.size());
    thrust::copy(output.begin(), output.end(), result.begin());
    return result;
}
