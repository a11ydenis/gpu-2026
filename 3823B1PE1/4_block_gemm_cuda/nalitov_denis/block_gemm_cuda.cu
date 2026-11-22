#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace {

constexpr int kTile = 32;
constexpr int kVec = 4;
constexpr int kScalarThreadsX = 32;
constexpr int kScalarThreadsY = 8;

void throw_if_cuda_error(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

__device__ __forceinline__ float4 load4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

__device__ __forceinline__ void store4(float* ptr, float4 value) {
    *reinterpret_cast<float4*>(ptr) = value;
}

template <int kBlock>
__global__ void tiled_gemm_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  int n) {
    __shared__ __align__(16) float tile_a[kBlock][kBlock];
    __shared__ __align__(16) float tile_b[kBlock][kBlock];
    __shared__ __align__(16) float tile_c[kBlock][kBlock];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * kBlock + ty;
    const int col = blockIdx.x * kBlock + tx;

    float acc = 0.0f;
    for (int origin = 0; origin < n; origin += kBlock) {
        if (tx % kVec == 0) {
            store4(&tile_a[ty][tx],
                   load4(a + static_cast<std::size_t>(row) * n + origin + tx));
            store4(&tile_b[ty][tx],
                   load4(b + static_cast<std::size_t>(origin + ty) * n + col));
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kBlock; k += kVec) {
            const float4 av = load4(&tile_a[ty][k]);
            acc = fmaf(av.x, tile_b[k][tx], acc);
            acc = fmaf(av.y, tile_b[k + 1][tx], acc);
            acc = fmaf(av.z, tile_b[k + 2][tx], acc);
            acc = fmaf(av.w, tile_b[k + 3][tx], acc);
        }
        __syncthreads();
    }

    tile_c[ty][tx] = acc;
    __syncthreads();

    if (tx % kVec == 0) {
        store4(c + static_cast<std::size_t>(row) * n + col, load4(&tile_c[ty][tx]));
    }
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

template <int kBlock>
void launch_tiled(float* a, float* b, float* c, int n, cudaStream_t stream) {
    const dim3 threads(kBlock, kBlock);
    const dim3 blocks(n / kBlock, n / kBlock);
    tiled_gemm_kernel<kBlock><<<blocks, threads, 0, stream>>>(a, b, c, n);
}

}  // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
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

    if (n % kTile == 0) {
        launch_tiled<kTile>(buffers.a, buffers.b, buffers.c, n, buffers.stream);
    } else if (n % 16 == 0) {
        launch_tiled<16>(buffers.a, buffers.b, buffers.c, n, buffers.stream);
    } else if (n % 8 == 0) {
        launch_tiled<8>(buffers.a, buffers.b, buffers.c, n, buffers.stream);
    } else {
        const dim3 threads(kScalarThreadsX, kScalarThreadsY);
        const dim3 blocks((n + kScalarThreadsX - 1) / kScalarThreadsX,
                          (n + kScalarThreadsY - 1) / kScalarThreadsY);
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
