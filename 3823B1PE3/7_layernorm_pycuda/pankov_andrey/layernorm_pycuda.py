import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.tools import context_dependent_memoize


@context_dependent_memoize
def _kernel():
    return SourceModule(r"""
    __global__ void normalize(float *data, const float *gamma, const float *beta,
                              int width, float eps) {
        __shared__ float partial[256];
        int lane = threadIdx.x;
        float *row = data + (size_t)blockIdx.x * width;
        float mean = 0.0f;
        for (int pass = 0; pass < 2; ++pass) {
            float sum = 0.0f;
            for (int i = lane; i < width; i += blockDim.x) {
                float x = row[i];
                sum += pass == 0 ? x : (x - mean) * (x - mean);
            }
            partial[lane] = sum;
            __syncthreads();
            for (int step = blockDim.x / 2; step > 0; step /= 2) {
                if (lane < step) partial[lane] += partial[lane + step];
                __syncthreads();
            }
            float average = partial[0] / width;
            __syncthreads();
            if (pass == 0) {
                mean = average;
            } else {
                float scale = rsqrtf(average + eps);
                for (int i = lane; i < width; i += blockDim.x)
                    row[i] = (row[i] - mean) * scale * gamma[i] + beta[i];
            }
        }
    }
    """).get_function("normalize")


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    result = np.array(input, dtype=np.float32, order="C", copy=True).reshape(-1)
    gamma = np.ascontiguousarray(gamma, dtype=np.float32).reshape(-1)
    beta = np.ascontiguousarray(beta, dtype=np.float32).reshape(-1)
    if row_size <= 0 or result.size % row_size or gamma.size != row_size or beta.size != row_size:
        raise ValueError("Incompatible input, gamma, beta or row_size")
    if result.size:
        cuda.init()
        if cuda.Context.get_current() is None:
            import pycuda.autoinit  # Create a default context for standalone callers.
        _kernel()(cuda.InOut(result), cuda.In(gamma), cuda.In(beta),
                  np.int32(row_size), np.float32(eps),
                  block=(256, 1, 1), grid=(result.size // row_size, 1))
    return result
