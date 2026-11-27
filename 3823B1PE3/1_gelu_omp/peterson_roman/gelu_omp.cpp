#include "gelu_omp.h"
#include <cmath>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < input.size(); ++i) {
        const float x = input[i];
        const float z = 0.7978845608f * (x + 0.044715f * x * x * x);
        output[i] = 0.5f * x * (1.0f + std::tanh(z));
    }
    return output;
}
