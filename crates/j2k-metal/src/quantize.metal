// SPDX-License-Identifier: MIT OR Apache-2.0

#include <metal_stdlib>
using namespace metal;

struct J2kQuantizeSubbandParams {
    uint len;
    uint step_exponent;
    uint step_mantissa;
    uint range_bits;
    uint reversible;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

inline int j2k_quantize_round_to_i32(float sample) {
    const float rounded = sample >= 0.0f
        ? floor(sample + 0.5f)
        : -floor(-sample + 0.5f);
    return int(rounded);
}

inline int j2k_quantize_sample(float sample, constant J2kQuantizeSubbandParams &params) {
    if (params.reversible != 0u) {
        return j2k_quantize_round_to_i32(sample);
    }

    const int exponent = int(params.range_bits) - int(params.step_exponent);
    const float base = exp2(float(exponent));
    const float delta = base * (1.0f + float(params.step_mantissa) / 2048.0f);
    if (delta <= 0.0f) {
        return 0;
    }

    const int sign = sample < 0.0f ? -1 : 1;
    const int magnitude = int(floor(fabs(sample) / delta));
    return sign * magnitude;
}

kernel void j2k_quantize_subband(
    device const float *samples [[buffer(0)]],
    device int *coefficients [[buffer(1)]],
    constant J2kQuantizeSubbandParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.len) {
        return;
    }

    coefficients[gid] = j2k_quantize_sample(samples[gid], params);
}
