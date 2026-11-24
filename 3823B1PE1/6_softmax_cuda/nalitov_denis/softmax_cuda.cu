#include "softmax_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace {

constexpr int kThreads = 256;
constexpr int kWarp = 32;
constexpr int kMaxTiles = 16;
constexpr float kNegInf = -3.402823466e+38f;

void throw_if_cuda_error(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ float block_max(float value) {
    __shared__ float partial[kThreads / kWarp];
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp_id = threadIdx.x / kWarp;
    value = warp_max(value);
    if (lane == 0) {
        partial[warp_id] = value;
    }
    __syncthreads();
    value = (threadIdx.x < kThreads / kWarp) ? partial[threadIdx.x] : kNegInf;
    if (warp_id == 0) {
        value = warp_max(value);
        if (lane == 0) {
            partial[0] = value;
        }
    }
    __syncthreads();
    return partial[0];
}

__device__ float block_sum(float value) {
    __shared__ float partial[kThreads / kWarp];
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp_id = threadIdx.x / kWarp;
    value = warp_sum(value);
    if (lane == 0) {
        partial[warp_id] = value;
    }
    __syncthreads();
    value = (threadIdx.x < kThreads / kWarp) ? partial[threadIdx.x] : 0.0f;
    if (warp_id == 0) {
        value = warp_sum(value);
        if (lane == 0) {
            partial[0] = value;
        }
    }
    __syncthreads();
    return partial[0];
}

__device__ __forceinline__ float max4(float4 v) {
    return fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
}

__device__ __forceinline__ float4 exp4(float4 v, float row_max) {
    v.x = __expf(v.x - row_max);
    v.y = __expf(v.y - row_max);
    v.z = __expf(v.z - row_max);
    v.w = __expf(v.w - row_max);
    return v;
}

__global__ void softmax_kernel(const float* __restrict__ in, float* __restrict__ out, int row_size) {
    const float* row_in = in + static_cast<std::size_t>(blockIdx.x) * row_size;
    float* row_out = out + static_cast<std::size_t>(blockIdx.x) * row_size;
    const int n4 = row_size >> 2;
    const float4* in4 = reinterpret_cast<const float4*>(row_in);
    float4* out4 = reinterpret_cast<float4*>(row_out);

    float4 tiles[kMaxTiles];
    int tile_count = 0;
    int cursor = threadIdx.x;
    for (; cursor < n4 && tile_count < kMaxTiles; cursor += kThreads) {
        tiles[tile_count++] = in4[cursor];
    }

    float local_max = kNegInf;
    for (int t = 0; t < tile_count; ++t) {
        local_max = fmaxf(local_max, max4(tiles[t]));
    }
    for (int i = cursor; i < n4; i += kThreads) {
        local_max = fmaxf(local_max, max4(in4[i]));
    }
    const float row_max = block_max(local_max);

    float local_sum = 0.0f;
    for (int t = 0; t < tile_count; ++t) {
        tiles[t] = exp4(tiles[t], row_max);
        local_sum += tiles[t].x + tiles[t].y + tiles[t].z + tiles[t].w;
    }
    for (int i = cursor; i < n4; i += kThreads) {
        const float4 e = exp4(in4[i], row_max);
        local_sum += e.x + e.y + e.z + e.w;
    }
    const float scale = 1.0f / block_sum(local_sum);

    int out_index = threadIdx.x;
    for (int t = 0; t < tile_count; ++t) {
        float4 e = tiles[t];
        e.x *= scale;
        e.y *= scale;
        e.z *= scale;
        e.w *= scale;
        out4[out_index] = e;
        out_index += kThreads;
    }
    for (int i = cursor; i < n4; i += kThreads) {
        float4 e = exp4(in4[i], row_max);
        e.x *= scale;
        e.y *= scale;
        e.z *= scale;
        e.w *= scale;
        out4[i] = e;
    }
}

__global__ void softmax_scalar_kernel(const float* __restrict__ in, float* __restrict__ out,
                                      int row_size) {
    const float* row_in = in + static_cast<std::size_t>(blockIdx.x) * row_size;
    float* row_out = out + static_cast<std::size_t>(blockIdx.x) * row_size;

    float local_max = kNegInf;
    for (int i = threadIdx.x; i < row_size; i += kThreads) {
        local_max = fmaxf(local_max, row_in[i]);
    }
    const float row_max = block_max(local_max);

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += kThreads) {
        const float e = __expf(row_in[i] - row_max);
        row_out[i] = e;
        local_sum += e;
    }
    const float scale = 1.0f / block_sum(local_sum);
    for (int i = threadIdx.x; i < row_size; i += kThreads) {
        row_out[i] *= scale;
    }
}

struct GpuBuffers {
    float* in = nullptr;
    float* out = nullptr;
    std::size_t capacity = 0;
    cudaStream_t stream = nullptr;

    GpuBuffers() {
        throw_if_cuda_error(cudaStreamCreate(&stream));
    }

    ~GpuBuffers() {
        cudaFree(in);
        cudaFree(out);
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }

    GpuBuffers(const GpuBuffers&) = delete;
    GpuBuffers& operator=(const GpuBuffers&) = delete;
    GpuBuffers(GpuBuffers&&) = delete;
    GpuBuffers& operator=(GpuBuffers&&) = delete;

    void grow(std::size_t elements) {
        if (elements <= capacity) {
            return;
        }
        throw_if_cuda_error(cudaFree(in));
        throw_if_cuda_error(cudaFree(out));
        const std::size_t bytes = elements * sizeof(float);
        throw_if_cuda_error(cudaMalloc(&in, bytes));
        throw_if_cuda_error(cudaMalloc(&out, bytes));
        capacity = elements;
    }
};

GpuBuffers& gpu_buffers() {
    static GpuBuffers buffers;
    return buffers;
}

}  // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (input.empty() || row_count <= 0) {
        return {};
    }

    const std::size_t elements = input.size();
    const int row_size = static_cast<int>(elements / static_cast<std::size_t>(row_count));
    const std::size_t bytes = elements * sizeof(float);
    GpuBuffers& buffers = gpu_buffers();
    buffers.grow(elements);

    throw_if_cuda_error(
        cudaMemcpyAsync(buffers.in, input.data(), bytes, cudaMemcpyHostToDevice, buffers.stream));

    if ((row_size & 3) == 0) {
        softmax_kernel<<<row_count, kThreads, 0, buffers.stream>>>(buffers.in, buffers.out,
                                                                   row_size);
    } else {
        softmax_scalar_kernel<<<row_count, kThreads, 0, buffers.stream>>>(buffers.in, buffers.out,
                                                                         row_size);
    }
    throw_if_cuda_error(cudaGetLastError());

    std::vector<float> output(elements);

    throw_if_cuda_error(
        cudaMemcpyAsync(output.data(), buffers.out, bytes, cudaMemcpyDeviceToHost, buffers.stream));
    throw_if_cuda_error(cudaStreamSynchronize(buffers.stream));
    return output;
}
