#include "gelu_omp.h"

#include <cmath>
#include <cstddef>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    if (input.empty()) {
        return {};
    }

    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(input.size());
    std::vector<float> output(static_cast<std::size_t>(n));
    const float* __restrict__ in = input.data();
    float* __restrict__ out = output.data();

    constexpr float kLog2e = 1.4426950408889634f;
    constexpr float kTwoSqrt2OverPi = 1.5957691216057308f;
    constexpr float kScale = kTwoSqrt2OverPi * kLog2e;
    constexpr float kCubic = kScale * 0.044715f;

#pragma omp parallel for simd schedule(static)
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        const float x = in[i];
        const float t = x * (kScale + kCubic * x * x);
        out[i] = x / (1.0f + std::exp2(-t));
    }

    return output;
}
