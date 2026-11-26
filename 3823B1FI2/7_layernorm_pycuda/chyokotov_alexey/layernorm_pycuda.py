import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

cuda_kernel_code = """
extern "C" __global__ void layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    const int row_size,
    const float eps
) {
    int row = blockIdx.x;
    const float* x = input + (size_t)row * row_size;
    float* y = output + (size_t)row * row_size;
    extern __shared__ float sdata[];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        sum += x[i];
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
        sdata[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        sum = (lane < num_warps) ? sdata[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            sdata[0] = sum;
        }
    }
    __syncthreads();
    float mean = sdata[0] / row_size;
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        float diff = x[i] - mean;
        var_sum += diff * diff;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        var_sum += __shfl_xor_sync(0xffffffff, var_sum, offset);
    }
    if (lane == 0) {
        sdata[warp_id] = var_sum;
    }
    __syncthreads();
    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        var_sum = (lane < num_warps) ? sdata[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            var_sum += __shfl_xor_sync(0xffffffff, var_sum, offset);
        }
        if (lane == 0) {
            sdata[0] = var_sum;
        }
    }
    __syncthreads();
    float variance = sdata[0] / row_size;
    float inv_std = rsqrtf(variance + eps);
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        float x_hat = (x[i] - mean) * inv_std;
        y[i] = x_hat * gamma[i] + beta[i];
    }
}
"""

_mod = SourceModule(cuda_kernel_code)
_layernorm_kernel = _mod.get_function("layernorm_kernel")

def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    x = np.asarray(input, dtype=np.float32).ravel()
    g = np.asarray(gamma, dtype=np.float32).ravel()
    b = np.asarray(beta, dtype=np.float32).ravel()
    
    total_size = x.size
    if total_size % row_size != 0:
        raise ValueError("Input size must be divisible by row_size.")
    num_rows = total_size // row_size
    
    x_gpu = cuda.mem_alloc(x.nbytes)
    g_gpu = cuda.mem_alloc(g.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    out_gpu = cuda.mem_alloc(x.nbytes)
    
    cuda.memcpy_htod(x_gpu, x)
    cuda.memcpy_htod(g_gpu, g)
    cuda.memcpy_htod(b_gpu, b)
    
    grid = (num_rows, 1, 1)
    block_size = 32
    while block_size < 512 and block_size < row_size:
        block_size += 32
    block = (block_size, 1, 1)
    
    shared_mem_size = (block_size // 32) * 4
    
    _layernorm_kernel(
        x_gpu, g_gpu, b_gpu, out_gpu,
        np.int32(row_size), np.float32(eps),
        block=block, grid=grid, shared=shared_mem_size
    )
    
    output = np.empty_like(x)
    cuda.memcpy_dtoh(output, out_gpu)
    
    return output