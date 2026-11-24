#include "gelu_omp.h"
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t n = input.size();
    std::vector<float> output(n);

    const float k = 2.0f * 0.7978845608f; // 2 * sqrt(2/pi)

    #pragma omp parallel for simd
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        float x = input[i];
        float z = k * (x + 0.044715f * x * x * x);
        output[i] = x / (1.0f + std::exp(-z));
    }
    return output;
}