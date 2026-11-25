#include "gelu_cuda.h"
#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ float Gelu(float x) {
    // 2 * sqrt(2/pi) = 1.5957691216
    const float z = 1.5957691216f * (x + 0.044715f * x * x * x);
    return x / (1.0f + __expf(-z));
}

__global__ void GeluKernel(const float4* in, float4* out, size_t n4) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = idx; i < n4; i += stride) {
        float4 v = in[i];
        v.x = Gelu(v.x);
        v.y = Gelu(v.y);
        v.z = Gelu(v.z);
        v.w = Gelu(v.w);
        out[i] = v;
    }
}

__global__ void GeluTailKernel(const float* in, float* out, size_t start, size_t n) {
    size_t i = start + threadIdx.x;
    if (i < n) out[i] = Gelu(in[i]);
}

// Память на устройстве выделяется один раз и переиспользуется
float* d_in = nullptr;
float* d_out = nullptr;
size_t d_capacity = 0;

void EnsureCapacity(size_t n) {
    if (n <= d_capacity) return;
    if (d_in) cudaFree(d_in);
    if (d_out) cudaFree(d_out);
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    d_capacity = n;
}

} // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t n = input.size();
    std::vector<float> output(n);
    if (n == 0) return output;

    EnsureCapacity(n);

    cudaMemcpy(d_in, input.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    const size_t n4 = n / 4;
    if (n4 > 0) {
        const int block = 256;
        size_t blocks = (n4 + block - 1) / block;
        if (blocks > 65535) blocks = 65535;
        GeluKernel<<<(unsigned)blocks, block>>>(
            reinterpret_cast<const float4*>(d_in),
            reinterpret_cast<float4*>(d_out), n4);
    }
    const size_t tail = n - n4 * 4;
    if (tail > 0) {
        GeluTailKernel<<<1, 4>>>(d_in, d_out, n4 * 4, n);
    }

    cudaMemcpy(output.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    return output;
}