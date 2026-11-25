#include "softmax_cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <stdexcept>

__global__ void softmax_kernel(
    const float* input,
    float* output,
    int row_size
) {
    extern __shared__ float buffer[];

    int row_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int threads = blockDim.x;

    int offset = row_id * row_size;

    const float* row = input + offset;
    float* result = output + offset;

    float max_val = -INFINITY;

    for (int idx = thread_id; idx < row_size; idx += threads) {
        max_val = fmaxf(max_val, row[idx]);
    }

    buffer[thread_id] = max_val;
    __syncthreads();

    for (int step = threads / 2; step > 0; step /= 2) {
        if (thread_id < step) {
            buffer[thread_id] = fmaxf(
                buffer[thread_id],
                buffer[thread_id + step]
            );
        }
        __syncthreads();
    }

    float maximum = buffer[0];

    float sum = 0.0f;

    for (int idx = thread_id; idx < row_size; idx += threads) {
        sum += expf(row[idx] - maximum);
    }

    buffer[thread_id] = sum;
    __syncthreads();

    for (int step = threads / 2; step > 0; step /= 2) {
        if (thread_id < step) {
            buffer[thread_id] += buffer[thread_id + step];
        }
        __syncthreads();
    }

    float denominator = buffer[0];

    for (int idx = thread_id; idx < row_size; idx += threads) {
        result[idx] = expf(row[idx] - maximum) / denominator;
    }
}

std::vector<float> SoftmaxCUDA(
    const std::vector<float>& input,
    int row_count
) {
    if (row_count <= 0) {
        throw std::invalid_argument("Invalid row count");
    }

    if (input.empty()) {
        return {};
    }

    if (input.size() % row_count != 0) {
        throw std::invalid_argument("Invalid input size");
    }

    int row_size = static_cast<int>(input.size() / row_count);
    size_t bytes = input.size() * sizeof(float);

    float* device_input = nullptr;
    float* device_output = nullptr;

    cudaMalloc(&device_input, bytes);
    cudaMalloc(&device_output, bytes);

    cudaMemcpy(
        device_input,
        input.data(),
        bytes,
        cudaMemcpyHostToDevice
    );

    int threads = 32;

    while (threads < 512 && threads < row_size) {
        threads *= 2;
    }

    softmax_kernel<<<
        row_count,
        threads,
        threads * sizeof(float)
    >>>(
        device_input,
        device_output,
        row_size
    );

    cudaDeviceSynchronize();

    std::vector<float> output(input.size());

    cudaMemcpy(
        output.data(),
        device_output,
        bytes,
        cudaMemcpyDeviceToHost
    );

    cudaFree(device_input);
    cudaFree(device_output);

    return output;
}
