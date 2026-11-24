#include "gelu_omp.h"
#include <cmath>
#include <cstddef>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> result(input.size());
    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(input.size()); ++i) {
        float x = input[i];
        result[i] = 0.5f * x * (1.0f + std::tanh(0.7978845608f * x * (1.0f + 0.044715f * x * x)));
    }
    return result;
}
