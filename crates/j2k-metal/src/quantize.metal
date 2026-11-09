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

// 2^exponent built from its IEEE-754 bits, exact in every math mode as the
// CPU's `pow2i` is; fast-math `exp2` is an approximation.
inline float j2k_quantize_pow2i(int exponent) {
    if (exponent > 127) {
        return INFINITY;
    }
    if (exponent >= -126) {
        return as_type<float>(uint(exponent + 127) << 23u);
    }
    if (exponent >= -149) {
        return as_type<float>(1u << uint(exponent + 149));
    }
    return 0.0f;
}

inline int j2k_quantize_sample(float sample, constant J2kQuantizeSubbandParams &params) {
    if (params.reversible != 0u) {
        return j2k_quantize_round_to_i32(sample);
    }

    // The mantissa factor has at most 12 significant bits, so `delta` is exact.
    const int exponent = int(params.range_bits) - int(params.step_exponent);
    const float base = j2k_quantize_pow2i(exponent);
    const float delta = base * (1.0f + float(params.step_mantissa) / 2048.0f);
    if (delta <= 0.0f) {
        return 0;
    }

    const int sign = sample < 0.0f ? -1 : 1;
    // Fast division can cross a deadzone boundary; match the CPU quotient.
    const int magnitude = int(floor(precise::divide(fabs(sample), delta)));
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

struct J2kLossyCoefficientJob {
    uint coefficient_offset;
    uint component;
    uint source_x;
    uint source_y;
    uint width;
    uint height;
    uint full_width;
    J2kQuantizeSubbandParams quantize;
};

kernel void j2k_lossy_extract_quantized_coefficients(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device int *coefficients [[buffer(3)]],
    constant J2kLossyCoefficientJob *jobs [[buffer(4)]],
    constant uint &job_count [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.z >= job_count) return;
    constant J2kLossyCoefficientJob &job = jobs[gid.z];
    if (gid.x >= job.width || gid.y >= job.height) return;
    device const float *plane = job.component == 0u ? plane0 : (job.component == 1u ? plane1 : plane2);
    const uint source = (job.source_y + gid.y) * job.full_width + job.source_x + gid.x;
    const uint destination = job.coefficient_offset + gid.y * job.width + gid.x;
    coefficients[destination] = j2k_quantize_sample(plane[source], job.quantize);
}
