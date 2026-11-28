import numpy as np

_KERNEL = r"""
constexpr int kThreads = 256;
constexpr int kWarp = 32;
constexpr int kMaxTiles = 16;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
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

__global__ void layernorm_kernel(const float* __restrict__ in,
                                 const float* __restrict__ gamma,
                                 const float* __restrict__ beta,
                                 float* __restrict__ out,
                                 int row_size,
                                 float eps) {
    const float* row_in = in + static_cast<size_t>(blockIdx.x) * row_size;
    float* row_out = out + static_cast<size_t>(blockIdx.x) * row_size;
    const int n4 = row_size >> 2;
    const float4* in4 = reinterpret_cast<const float4*>(row_in);
    const float4* gamma4 = reinterpret_cast<const float4*>(gamma);
    const float4* beta4 = reinterpret_cast<const float4*>(beta);
    float4* out4 = reinterpret_cast<float4*>(row_out);

    float4 tiles[kMaxTiles];
    int tile_count = 0;
    int cursor = threadIdx.x;
    for (; cursor < n4 && tile_count < kMaxTiles; cursor += kThreads) {
        tiles[tile_count++] = in4[cursor];
    }

    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    for (int t = 0; t < tile_count; ++t) {
        const float4 v = tiles[t];
        local_sum += v.x + v.y + v.z + v.w;
        local_sumsq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (int i = cursor; i < n4; i += kThreads) {
        const float4 v = in4[i];
        local_sum += v.x + v.y + v.z + v.w;
        local_sumsq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    const float inv_n = 1.0f / static_cast<float>(row_size);
    const float mean = block_sum(local_sum) * inv_n;
    const float mean_sq = block_sum(local_sumsq) * inv_n;
    const float inv_std = rsqrtf(fmaxf(mean_sq - mean * mean, 0.0f) + eps);

    int out_index = threadIdx.x;
    for (int t = 0; t < tile_count; ++t) {
        const float4 v = tiles[t];
        const float4 g = gamma4[out_index];
        const float4 b = beta4[out_index];
        float4 y;
        y.x = g.x * (v.x - mean) * inv_std + b.x;
        y.y = g.y * (v.y - mean) * inv_std + b.y;
        y.z = g.z * (v.z - mean) * inv_std + b.z;
        y.w = g.w * (v.w - mean) * inv_std + b.w;
        out4[out_index] = y;
        out_index += kThreads;
    }
    for (int i = cursor; i < n4; i += kThreads) {
        const float4 v = in4[i];
        const float4 g = gamma4[i];
        const float4 b = beta4[i];
        float4 y;
        y.x = g.x * (v.x - mean) * inv_std + b.x;
        y.y = g.y * (v.y - mean) * inv_std + b.y;
        y.z = g.z * (v.z - mean) * inv_std + b.z;
        y.w = g.w * (v.w - mean) * inv_std + b.w;
        out4[i] = y;
    }
}

__global__ void layernorm_scalar_kernel(const float* __restrict__ in,
                                        const float* __restrict__ gamma,
                                        const float* __restrict__ beta,
                                        float* __restrict__ out,
                                        int row_size,
                                        float eps) {
    const float* row_in = in + static_cast<size_t>(blockIdx.x) * row_size;
    float* row_out = out + static_cast<size_t>(blockIdx.x) * row_size;

    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += kThreads) {
        const float v = row_in[i];
        local_sum += v;
        local_sumsq += v * v;
    }
    const float inv_n = 1.0f / static_cast<float>(row_size);
    const float mean = block_sum(local_sum) * inv_n;
    const float mean_sq = block_sum(local_sumsq) * inv_n;
    const float inv_std = rsqrtf(fmaxf(mean_sq - mean * mean, 0.0f) + eps);

    for (int i = threadIdx.x; i < row_size; i += kThreads) {
        row_out[i] = gamma[i] * (row_in[i] - mean) * inv_std + beta[i];
    }
}
"""

_THREADS = 256
_state = {}


class _GpuBuffers:
    def __init__(self, cuda):
        self.in_ = None
        self.out = None
        self.gamma = None
        self.beta = None
        self.cap = 0
        self.row_cap = 0
        self.stream = cuda.Stream()

    def grow(self, cuda, elements, row_size):
        if elements > self.cap:
            self.in_ = cuda.mem_alloc(elements * 4)
            self.out = cuda.mem_alloc(elements * 4)
            self.cap = elements
        if row_size > self.row_cap:
            self.gamma = cuda.mem_alloc(row_size * 4)
            self.beta = cuda.mem_alloc(row_size * 4)
            self.row_cap = row_size


def _prepare():
    if "kernel" in _state:
        return
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule

    module = SourceModule(_KERNEL, options=["-O3"])
    _state["cuda"] = cuda
    _state["kernel"] = module.get_function("layernorm_kernel")
    _state["scalar"] = module.get_function("layernorm_scalar_kernel")
    _state["buffers"] = _GpuBuffers(cuda)


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """
    Apply Layer Normalization to each row of the input matrix.

    Parameters
    ----------
    input : list or numpy.ndarray of float
        Flattened matrix in row‑major order. Its length must be divisible by row_size.
    gamma : list or numpy.ndarray of float
        Scale parameter, length = row_size.
    beta : list or numpy.ndarray of float
        Shift parameter, length = row_size.
    row_size : int
        Number of features per row (i.e., number of columns).
    eps : float, optional
        Small constant for numerical stability.

    Returns
    -------
    numpy.ndarray
        Flattened matrix of the same shape as input, containing the row‑wise
        normalized results.
    """
    values = np.ascontiguousarray(input, dtype=np.float32).ravel()
    scale = np.ascontiguousarray(gamma, dtype=np.float32).ravel()
    shift = np.ascontiguousarray(beta, dtype=np.float32).ravel()
    row_size = int(row_size)

    if values.size == 0 or row_size <= 0:
        return np.empty(0, dtype=np.float32)

    _prepare()
    cuda = _state["cuda"]
    buffers = _state["buffers"]
    buffers.grow(cuda, values.size, row_size)

    cuda.memcpy_htod(buffers.in_, values)
    cuda.memcpy_htod(buffers.gamma, scale)
    cuda.memcpy_htod(buffers.beta, shift)

    row_count = int(values.size // row_size)
    kernel = _state["kernel"] if (row_size & 3) == 0 else _state["scalar"]
    kernel(
        buffers.in_,
        buffers.gamma,
        buffers.beta,
        buffers.out,
        np.int32(row_size),
        np.float32(eps),
        block=(_THREADS, 1, 1),
        grid=(row_count, 1),
        stream=buffers.stream,
    )

    output = np.empty(values.size, dtype=np.float32)
    buffers.stream.synchronize()
    cuda.memcpy_dtoh(output, buffers.out)
    return output
