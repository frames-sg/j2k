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
