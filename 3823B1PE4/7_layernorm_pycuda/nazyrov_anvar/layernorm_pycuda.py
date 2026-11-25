import numpy as np
import pycuda.autoinit  # noqa: F401  (создаёт контекст CUDA)
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

_KERNEL_SRC = r"""
__device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();              // sh может использоваться повторно
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    float r = 0.0f;
    for (int i = 0; i < nw; ++i) r += sh[i];
    return r;
}

// Один блок обрабатывает одну строку
__global__ void layernorm(const float* x, const float* gamma, const float* beta,
                          float* y, int row_size, float eps) {
    __shared__ float sh[32];
    const float* row_in = x + (size_t)blockIdx.x * row_size;
    float* row_out = y + (size_t)blockIdx.x * row_size;

    float s = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) s += row_in[i];
    const float mean = block_sum(s, sh) / (float)row_size;

    float v = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        const float d = row_in[i] - mean;
        v += d * d;
    }
    const float var = block_sum(v, sh) / (float)row_size;
    const float inv = rsqrtf(var + eps);

    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        row_out[i] = gamma[i] * ((row_in[i] - mean) * inv) + beta[i];
    }
}
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = SourceModule(_KERNEL_SRC).get_function("layernorm")
    return _kernel


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """
    Apply Layer Normalization to each row of the input matrix.
    """
    x = np.ascontiguousarray(input, dtype=np.float32).ravel()
    g = np.ascontiguousarray(gamma, dtype=np.float32).ravel()
    b = np.ascontiguousarray(beta, dtype=np.float32).ravel()
    row_size = int(row_size)
    rows = x.size // row_size

    kernel = _get_kernel()

    d_x = gpuarray.to_gpu(x)
    d_g = gpuarray.to_gpu(g)
    d_b = gpuarray.to_gpu(b)
    d_y = gpuarray.empty_like(d_x)

    threads = max(32, min(row_size, 1024))
    kernel(d_x.gpudata, d_g.gpudata, d_b.gpudata, d_y.gpudata,
           np.int32(row_size), np.float32(eps),
           block=(threads, 1, 1), grid=(rows, 1))

    return d_y.get()
