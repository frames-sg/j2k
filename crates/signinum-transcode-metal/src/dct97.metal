// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>

using namespace metal;

struct Dct97ProjectionParams {
    uint width;
    uint height;
    uint block_cols;
    uint band_width;
    uint band_height;
};

struct Reversible53ProjectionParams {
    uint width;
    uint height;
    uint block_cols;
    uint band_width;
    uint band_height;
    uint vertical_low;
    uint horizontal_low;
};

struct Dct97SparseRow {
    uint offset;
    uint count;
};

struct Dct97WeightTap {
    uint sample_idx;
    float weight;
};

kernel void dct97_project_band(
    device const float *blocks [[buffer(0)]],
    device const Dct97SparseRow *x_rows [[buffer(1)]],
    device const Dct97WeightTap *x_taps [[buffer(2)]],
    device const Dct97SparseRow *y_rows [[buffer(3)]],
    device const Dct97WeightTap *y_taps [[buffer(4)]],
    device const float *idct_basis [[buffer(5)]],
    device float *output [[buffer(6)]],
    constant Dct97ProjectionParams &params [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.band_width || gid.y >= params.band_height) {
        return;
    }

    const Dct97SparseRow x_row = x_rows[gid.x];
    const Dct97SparseRow y_row = y_rows[gid.y];
    float value = 0.0f;
    for (uint y_tap_idx = 0; y_tap_idx < y_row.count; ++y_tap_idx) {
        const Dct97WeightTap y_tap = y_taps[y_row.offset + y_tap_idx];
        const uint sample_y = y_tap.sample_idx;
        const float y_weight = y_tap.weight;
        const uint block_y = sample_y / 8u;
        const uint local_y = sample_y % 8u;

        for (uint x_tap_idx = 0; x_tap_idx < x_row.count; ++x_tap_idx) {
            const Dct97WeightTap x_tap = x_taps[x_row.offset + x_tap_idx];
            const uint sample_x = x_tap.sample_idx;
            const float x_weight = x_tap.weight;
            const uint block_x = sample_x / 8u;
            const uint local_x = sample_x % 8u;
            const uint block_base = (block_y * params.block_cols + block_x) * 64u;
            const float sample_weight = y_weight * x_weight;

            for (uint freq_y = 0; freq_y < 8u; ++freq_y) {
                const float y_basis = idct_basis[local_y * 8u + freq_y];
                for (uint freq_x = 0; freq_x < 8u; ++freq_x) {
                    const float coefficient = blocks[block_base + freq_y * 8u + freq_x];
                    const float x_basis = idct_basis[local_x * 8u + freq_x];
                    value += sample_weight * y_basis * x_basis * coefficient;
                }
            }
        }
    }

    output[gid.y * params.band_width + gid.x] = value;
}

static inline int floor_div_i32(int numerator, int denominator) {
    const int quotient = numerator / denominator;
    const int remainder = numerator % denominator;
    return (remainder < 0) ? quotient - 1 : quotient;
}

static inline int reversible53_sample(
    device const int *blocks,
    uint block_cols,
    uint x,
    uint y
) {
    const uint block_x = x / 8u;
    const uint block_y = y / 8u;
    const uint local_x = x % 8u;
    const uint local_y = y % 8u;
    const uint block_base = (block_y * block_cols + block_x) * 64u;
    return blocks[block_base + local_y * 8u + local_x];
}

static inline int reversible53_vertical_high(
    device const int *blocks,
    constant Reversible53ProjectionParams &params,
    uint x,
    uint high_idx
) {
    const uint odd_idx = high_idx * 2u + 1u;
    const int current = reversible53_sample(blocks, params.block_cols, x, odd_idx);
    const int left = reversible53_sample(blocks, params.block_cols, x, odd_idx - 1u);
    if ((params.height % 2u) == 0u && odd_idx + 1u == params.height) {
        return current - left;
    }

    const uint right_idx = (odd_idx + 1u < params.height) ? odd_idx + 1u : params.height - 1u;
    const int right = reversible53_sample(blocks, params.block_cols, x, right_idx);
    return current - floor_div_i32(left + right, 2);
}

static inline int reversible53_vertical_low(
    device const int *blocks,
    constant Reversible53ProjectionParams &params,
    uint x,
    uint low_idx
) {
    const uint even_idx = low_idx * 2u;
    const int current = reversible53_sample(blocks, params.block_cols, x, even_idx);
    if (params.height < 2u) {
        return current;
    }

    if ((params.height % 2u) == 0u) {
        const int right = reversible53_vertical_high(blocks, params, x, low_idx);
        if (low_idx == 0u) {
            return current + floor_div_i32(right + 1, 2);
        }
        const int left = reversible53_vertical_high(blocks, params, x, low_idx - 1u);
        return current + floor_div_i32(left + right + 2, 4);
    }

    const uint high_len = params.height / 2u;
    if (high_len == 0u) {
        return current;
    }
    const int left = low_idx > 0u
        ? reversible53_vertical_high(blocks, params, x, low_idx - 1u)
        : reversible53_vertical_high(blocks, params, x, 0u);
    const int right = low_idx < high_len
        ? reversible53_vertical_high(blocks, params, x, low_idx)
        : left;
    return current + floor_div_i32(left + right + 2, 4);
}

static inline int reversible53_vertical_value(
    device const int *blocks,
    constant Reversible53ProjectionParams &params,
    uint x,
    uint output_y
) {
    return params.vertical_low != 0u
        ? reversible53_vertical_low(blocks, params, x, output_y)
        : reversible53_vertical_high(blocks, params, x, output_y);
}

static inline int reversible53_horizontal_high(
    device const int *blocks,
    constant Reversible53ProjectionParams &params,
    uint high_idx,
    uint output_y
) {
    const uint odd_idx = high_idx * 2u + 1u;
    const int current = reversible53_vertical_value(blocks, params, odd_idx, output_y);
    const int left = reversible53_vertical_value(blocks, params, odd_idx - 1u, output_y);
    if ((params.width % 2u) == 0u && odd_idx + 1u == params.width) {
        return current - left;
    }

    const uint right_idx = (odd_idx + 1u < params.width) ? odd_idx + 1u : params.width - 1u;
    const int right = reversible53_vertical_value(blocks, params, right_idx, output_y);
    return current - floor_div_i32(left + right, 2);
}

static inline int reversible53_horizontal_low(
    device const int *blocks,
    constant Reversible53ProjectionParams &params,
    uint low_idx,
    uint output_y
) {
    const uint even_idx = low_idx * 2u;
    const int current = reversible53_vertical_value(blocks, params, even_idx, output_y);
    if (params.width < 2u) {
        return current;
    }

    if ((params.width % 2u) == 0u) {
        const int right = reversible53_horizontal_high(blocks, params, low_idx, output_y);
        if (low_idx == 0u) {
            return current + floor_div_i32(right + 1, 2);
        }
        const int left = reversible53_horizontal_high(blocks, params, low_idx - 1u, output_y);
        return current + floor_div_i32(left + right + 2, 4);
    }

    const uint high_len = params.width / 2u;
    if (high_len == 0u) {
        return current;
    }
    const int left = low_idx > 0u
        ? reversible53_horizontal_high(blocks, params, low_idx - 1u, output_y)
        : reversible53_horizontal_high(blocks, params, 0u, output_y);
    const int right = low_idx < high_len
        ? reversible53_horizontal_high(blocks, params, low_idx, output_y)
        : left;
    return current + floor_div_i32(left + right + 2, 4);
}

kernel void reversible53_project_band(
    device const int *blocks [[buffer(0)]],
    device int *output [[buffer(1)]],
    constant Reversible53ProjectionParams &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.band_width || gid.y >= params.band_height) {
        return;
    }

    const int value = params.horizontal_low != 0u
        ? reversible53_horizontal_low(blocks, params, gid.x, gid.y)
        : reversible53_horizontal_high(blocks, params, gid.x, gid.y);
    output[gid.y * params.band_width + gid.x] = value;
}
