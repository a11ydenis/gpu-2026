#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace {

constexpr float kAlpha = 1.0f;
constexpr float kBeta = 0.0f;

void throw_if_cuda_error(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

void throw_if_cublas_error(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS error");
    }
}

struct GpuBuffers {
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    std::size_t capacity = 0;
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;

    GpuBuffers() {
        throw_if_cuda_error(cudaStreamCreate(&stream));
        throw_if_cublas_error(cublasCreate(&handle));
        throw_if_cublas_error(cublasSetStream(handle, stream));
    }

    ~GpuBuffers() {
        if (handle != nullptr) {
            cublasDestroy(handle);
        }
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

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
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

    throw_if_cublas_error(cublasSgemm(buffers.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &kAlpha,
                                      buffers.b, n, buffers.a, n, &kBeta, buffers.c, n));

    std::vector<float> output(elements);

    throw_if_cuda_error(
        cudaMemcpyAsync(output.data(), buffers.c, bytes, cudaMemcpyDeviceToHost, buffers.stream));
    throw_if_cuda_error(cudaStreamSynchronize(buffers.stream));
    return output;
}
