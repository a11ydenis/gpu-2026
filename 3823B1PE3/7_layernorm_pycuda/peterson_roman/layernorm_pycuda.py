import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

_KERNEL = SourceModule(r"""
__global__ void layer_norm(const float* x, const float* gamma, const float* beta,
                           float* y, int width, float eps) {
    __shared__ float scratch[256];
    int row = blockIdx.x;
    int lane = threadIdx.x;
    float sum = 0.0f;
    for (int col = lane; col < width; col += 256) sum += x[row * width + col];
    scratch[lane] = sum;
    __syncthreads();
    for (int step = 128; step; step >>= 1) {
        if (lane < step) scratch[lane] += scratch[lane + step];
        __syncthreads();
    }
    float mean = scratch[0] / width;
    float variance = 0.0f;
    for (int col = lane; col < width; col += 256) {
        float delta = x[row * width + col] - mean;
        variance += delta * delta;
    }
    scratch[lane] = variance;
    __syncthreads();
    for (int step = 128; step; step >>= 1) {
        if (lane < step) scratch[lane] += scratch[lane + step];
        __syncthreads();
    }
    float scale = rsqrtf(scratch[0] / width + eps);
    for (int col = lane; col < width; col += 256) {
        y[row * width + col] = (x[row * width + col] - mean) * scale * gamma[col] + beta[col];
    }
}
""").get_function("layer_norm")


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    values = np.ascontiguousarray(input, dtype=np.float32).ravel()
    gamma = np.ascontiguousarray(gamma, dtype=np.float32).ravel()
    beta = np.ascontiguousarray(beta, dtype=np.float32).ravel()
    if row_size <= 0 or values.size % row_size or gamma.size != row_size or beta.size != row_size:
        raise ValueError("invalid layer normalization dimensions")
    output = np.empty_like(values)
    if not values.size:
        return output
    dx = cuda.mem_alloc(values.nbytes)
    dg = cuda.mem_alloc(gamma.nbytes)
    db = cuda.mem_alloc(beta.nbytes)
    dy = cuda.mem_alloc(output.nbytes)
    cuda.memcpy_htod(dx, values)
    cuda.memcpy_htod(dg, gamma)
    cuda.memcpy_htod(db, beta)
    _KERNEL(dx, dg, db, dy, np.int32(row_size), np.float32(eps),
            block=(256, 1, 1), grid=(values.size // row_size, 1, 1))
    cuda.memcpy_dtoh(output, dy)
    return output
