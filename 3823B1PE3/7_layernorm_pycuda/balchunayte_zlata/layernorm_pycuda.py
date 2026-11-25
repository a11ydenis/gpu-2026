import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

from pycuda.compiler import SourceModule
from functools import lru_cache


@lru_cache(maxsize=1)
def get_kernel():
    source = r"""
    #include <math.h>

    #define THREADS 256

    __global__ void layernorm_kernel(
        const float* input,
        const float* gamma,
        const float* beta,
        float* output,
        int row_size,
        float eps
    ) {
        int row = blockIdx.x;
        int tid = threadIdx.x;
        int offset = row * row_size;

        __shared__ float shared[THREADS];
        __shared__ float mean;
        __shared__ float inv_std;

        // 1. Вычисляем среднее значение строки
        float local_sum = 0.0f;

        for (int i = tid; i < row_size; i += THREADS) {
            local_sum += input[offset + i];
        }

        shared[tid] = local_sum;
        __syncthreads();

        for (int stride = THREADS / 2;
             stride > 0; stride /= 2) {
            if (tid < stride) {
                shared[tid] += shared[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            mean = shared[0] / row_size;
        }
        __syncthreads();

        // 2. Вычисляем дисперсию строки
        float local_var = 0.0f;

        for (int i = tid; i < row_size; i += THREADS) {
            float diff = input[offset + i] - mean;
            local_var += diff * diff;
        }

        shared[tid] = local_var;
        __syncthreads();

        for (int stride = THREADS / 2;
             stride > 0; stride /= 2) {
            if (tid < stride) {
                shared[tid] += shared[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            float variance = shared[0] / row_size;
            inv_std = rsqrtf(variance + eps);
        }
        __syncthreads();

        // 3. Нормализация, масштабирование и сдвиг
        for (int i = tid; i < row_size; i += THREADS) {
            float x = input[offset + i];
            float normalized = (x - mean) * inv_std;

            output[offset + i] =
                normalized * gamma[i] + beta[i];
        }
    }
    """

    module = SourceModule(source)
    return module.get_function("layernorm_kernel")


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """
    Apply Layer Normalization to each row of the input matrix.

    Parameters
    ----------
    input : list or numpy.ndarray of float
        Flattened matrix in row-major order.
    gamma : list or numpy.ndarray of float
        Scale parameter, length = row_size.
    beta : list or numpy.ndarray of float
        Shift parameter, length = row_size.
    row_size : int
        Number of features per row.
    eps : float, optional
        Small constant for numerical stability.

    Returns
    -------
    numpy.ndarray
        Flattened float32 normalized results.
    """

    x = np.ascontiguousarray(
        np.asarray(input, dtype=np.float32).ravel()
    )

    gamma = np.ascontiguousarray(
        np.asarray(gamma, dtype=np.float32).ravel()
    )

    beta = np.ascontiguousarray(
        np.asarray(beta, dtype=np.float32).ravel()
    )

    if row_size <= 0 or x.size % row_size != 0:
        raise ValueError("Invalid row_size")

    if gamma.size != row_size or beta.size != row_size:
        raise ValueError("Invalid gamma or beta size")

    if eps <= 0:
        raise ValueError("eps must be positive")

    if x.size == 0:
        return np.empty(0, dtype=np.float32)

    row_count = x.size // row_size

    # Передаём данные на GPU
    d_input = gpuarray.to_gpu(x)
    d_gamma = gpuarray.to_gpu(gamma)
    d_beta = gpuarray.to_gpu(beta)
    d_output = gpuarray.empty_like(d_input)

    # Получаем скомпилированное CUDA-ядро
    kernel = get_kernel()

    # Запускаем вычисления на видеокарте
    kernel(
        d_input,
        d_gamma,
        d_beta,
        d_output,
        np.int32(row_size),
        np.float32(eps),
        block=(256, 1, 1),
        grid=(int(row_count), 1, 1)
    )

    # Возвращаем результат на CPU
    return d_output.get()
