#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>
namespace {
void Check(cudaError_t error) { if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); }
__global__ void Softmax(const float* input, float* output, int cols) {
    __shared__ float values[256];
    int row = blockIdx.x;
    int lane = threadIdx.x;
    float maximum = -INFINITY;
    for (int col = lane; col < cols; col += 256) maximum = fmaxf(maximum, input[row * cols + col]);
    values[lane] = maximum;
    __syncthreads();
    for (int step = 128; step; step >>= 1) {
        if (lane < step) values[lane] = fmaxf(values[lane], values[lane + step]);
        __syncthreads();
    }
    maximum = values[0];
    float sum = 0;
    for (int col = lane; col < cols; col += 256) sum += expf(input[row * cols + col] - maximum);
    values[lane] = sum;
    __syncthreads();
    for (int step = 128; step; step >>= 1) {
        if (lane < step) values[lane] += values[lane + step];
        __syncthreads();
    }
    sum = values[0];
    for (int col = lane; col < cols; col += 256) output[row * cols + col] = expf(input[row * cols + col] - maximum) / sum;
}
}
std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (row_count < 0 || (row_count == 0 && !input.empty()) || (row_count && input.size() % row_count)) throw std::invalid_argument("invalid row count");
    std::vector<float> output(input.size());
    if (input.empty()) return output;
    int cols = input.size() / row_count;
    float *di = nullptr, *dout = nullptr;
    size_t bytes = input.size() * sizeof(float);
    try {
        Check(cudaMalloc(&di, bytes)); Check(cudaMalloc(&dout, bytes));
        Check(cudaMemcpy(di, input.data(), bytes, cudaMemcpyHostToDevice));
        Softmax<<<row_count, 256>>>(di, dout, cols);
        Check(cudaGetLastError());
        Check(cudaMemcpy(output.data(), dout, bytes, cudaMemcpyDeviceToHost));
    } catch (...) { cudaFree(di); cudaFree(dout); throw; }
    cudaFree(di); cudaFree(dout);
    return output;
}
