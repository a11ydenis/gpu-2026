#include "softmax_cuda.h"

#include <cuda_runtime.h>
#include <cfloat>
#include <cstddef>
#include <climits>
#include <stdexcept>

#define CUDA_CHECK(call) do {                              \
    cudaError_t err = (call);                              \
    if (err != cudaSuccess)                                \
        throw std::runtime_error(cudaGetErrorString(err)); \
} while (0)

constexpr int THREADS = 256;

// Кэш видеопамяти для повторных вызовов
struct Workspace {
    float* d_input = nullptr;
    float* d_output = nullptr;
    size_t capacity = 0;

    void reserve(size_t bytes) {
        if (bytes <= capacity) return;

        float* new_input = nullptr;
        float* new_output = nullptr;

        CUDA_CHECK(cudaMalloc(&new_input, bytes));

        cudaError_t err = cudaMalloc(&new_output, bytes);
        if (err != cudaSuccess) {
            cudaFree(new_input);
            throw std::runtime_error(cudaGetErrorString(err));
        }

        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);

        d_input = new_input;
        d_output = new_output;
        capacity = bytes;
    }

    ~Workspace() {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
    }
};

// Каждый блок CUDA обрабатывает одну строку
__global__ void softmaxKernel(
    const float* input,
    float* output,
    int row_size
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    size_t offset = static_cast<size_t>(row) * row_size;

    __shared__ float shared[THREADS];

    // 1. Максимум строки
    float local_max = -FLT_MAX;

    for (int i = tid; i < row_size; i += blockDim.x) {
        local_max = fmaxf(
            local_max, input[offset + i]
        );
    }

    shared[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2;
         stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] = fmaxf(
                shared[tid], shared[tid + stride]
            );
        }
        __syncthreads();
    }

    float row_max = shared[0];

    // 2. Сумма экспонент
    float local_sum = 0.0f;

for (int i = tid; i < row_size; i += blockDim.x) {
    float value = expf(input[offset + i] - row_max);

    output[offset + i] = value;
    local_sum += value;
}

    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2;
         stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float row_sum = shared[0];

    // 3. Записываем результат
    for (int i = tid; i < row_size; i += blockDim.x) {
    output[offset + i] /= row_sum;
    }
}

std::vector<float> SoftmaxCUDA(
    const std::vector<float>& input,
    int row_count
) {
    if (row_count <= 0 ||
        input.size() % static_cast<size_t>(row_count) != 0) {
        throw std::invalid_argument("Invalid matrix size");
    }

    std::vector<float> result(input.size());

    if (input.empty()) return result;

    size_t row_size = input.size() / row_count;

    if (row_size > INT_MAX) {
        throw std::invalid_argument("Row too large");
    }

    size_t bytes = input.size() * sizeof(float);

    // Повторно используем память GPU
    static thread_local Workspace ws;
    ws.reserve(bytes);

    CUDA_CHECK(cudaMemcpy(
        ws.d_input, input.data(), bytes,
        cudaMemcpyHostToDevice
    ));

    softmaxKernel<<<row_count, THREADS>>>(
        ws.d_input,
        ws.d_output,
        static_cast<int>(row_size)
    );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(
        result.data(), ws.d_output, bytes,
        cudaMemcpyDeviceToHost
    ));

    return result;
}
