#include "gelu_omp.h"

#include <cmath>
#include <cstddef>
#include <vector>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::size_t n = input.size();
    std::vector<float> output(n);

    constexpr float kLinear = 1.5957691216f;
    constexpr float kCubic = 0.0713548163f;

    #pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        const float x = input[static_cast<std::size_t>(i)];
        const float argument = x * (kLinear + kCubic * x * x);
        output[static_cast<std::size_t>(i)] =
            x / (1.0f + std::exp(-argument));
    }

    return output;
}
