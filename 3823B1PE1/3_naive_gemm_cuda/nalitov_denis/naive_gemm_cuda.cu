#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace {

constexpr int kThreadsX = 32;
constexpr int kThreadsY = 8;
constexpr int kVecWidth = 4;

void throw_if_cuda_error(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

__global__ void gemm_vec4_kernel(const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 float* __restrict__ c,
                                 int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = (blockIdx.x * blockDim.x + threadIdx.x) * kVecWidth;
    if (row >= n || col >= n) {
        return;
    }

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float* a_row = a + static_cast<std::size_t>(row) * n;

#pragma unroll 8
    for (int k = 0; k < n; ++k) {
        const float av = a_row[k];
        const float4 bv =
            *reinterpret_cast<const float4*>(b + static_cast<std::size_t>(k) * n + col);
        acc.x = fmaf(av, bv.x, acc.x);
        acc.y = fmaf(av, bv.y, acc.y);
        acc.z = fmaf(av, bv.z, acc.z);
        acc.w = fmaf(av, bv.w, acc.w);
    }

    *reinterpret_cast<float4*>(c + static_cast<std::size_t>(row) * n + col) = acc;
}

__global__ void gemm_scalar_kernel(const float* __restrict__ a,
                                   const float* __restrict__ b,
                                   float* __restrict__ c,
                                   int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) {
        return;
    }

    float sum = 0.0f;
    const float* a_row = a + static_cast<std::size_t>(row) * n;
    for (int k = 0; k < n; ++k) {
        sum = fmaf(a_row[k], b[static_cast<std::size_t>(k) * n + col], sum);
    }
    c[static_cast<std::size_t>(row) * n + col] = sum;
}

struct GpuBuffers {
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    std::size_t capacity = 0;
    cudaStream_t stream = nullptr;

    GpuBuffers() {
        throw_if_cuda_error(cudaStreamCreate(&stream));
    }

    ~GpuBuffers() {
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
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
        throw_if_cuda_error(cudaFree(a));
        throw_if_cuda_error(cudaFree(b));
        throw_if_cuda_error(cudaFree(c));
        const std::size_t bytes = elements * sizeof(float);
        throw_if_cuda_error(cudaMalloc(&a, bytes));
        throw_if_cuda_error(cudaMalloc(&b, bytes));
        throw_if_cuda_error(cudaMalloc(&c, bytes));
        capacity = elements;
    }
};

GpuBuffers& gpu_buffers() {
    static GpuBuffers buffers;
    return buffers;
}

}  // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return {};
    }

    const std::size_t elements = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    const std::size_t bytes = elements * sizeof(float);
    GpuBuffers& buffers = gpu_buffers();
    buffers.grow(elements);

    throw_if_cuda_error(
        cudaMemcpyAsync(buffers.a, a.data(), bytes, cudaMemcpyHostToDevice, buffers.stream));
    throw_if_cuda_error(
        cudaMemcpyAsync(buffers.b, b.data(), bytes, cudaMemcpyHostToDevice, buffers.stream));

    const dim3 threads(kThreadsX, kThreadsY);
    if (n >= kVecWidth) {
        const dim3 blocks((n / kVecWidth + kThreadsX - 1) / kThreadsX,
                          (n + kThreadsY - 1) / kThreadsY);
        gemm_vec4_kernel<<<blocks, threads, 0, buffers.stream>>>(buffers.a, buffers.b, buffers.c, n);
    } else {
        const dim3 blocks((n + kThreadsX - 1) / kThreadsX, (n + kThreadsY - 1) / kThreadsY);
        gemm_scalar_kernel<<<blocks, threads, 0, buffers.stream>>>(buffers.a, buffers.b, buffers.c,
                                                                   n);
    }
    throw_if_cuda_error(cudaGetLastError());

    std::vector<float> output(elements);

    throw_if_cuda_error(
        cudaMemcpyAsync(output.data(), buffers.c, bytes, cudaMemcpyDeviceToHost, buffers.stream));
    throw_if_cuda_error(cudaStreamSynchronize(buffers.stream));
    return output;
}
