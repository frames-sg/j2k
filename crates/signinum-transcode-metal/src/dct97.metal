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

kernel void dct97_project_band(
    device const float *blocks [[buffer(0)]],
    device const float *x_weights [[buffer(1)]],
    device const float *y_weights [[buffer(2)]],
    device const float *idct_basis [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant Dct97ProjectionParams &params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.band_width || gid.y >= params.band_height) {
        return;
    }

    float value = 0.0f;
    for (uint sample_y = 0; sample_y < params.height; ++sample_y) {
        const float y_weight = y_weights[gid.y * params.height + sample_y];
        if (y_weight == 0.0f) {
            continue;
        }
        const uint block_y = sample_y / 8u;
        const uint local_y = sample_y % 8u;

        for (uint sample_x = 0; sample_x < params.width; ++sample_x) {
            const float x_weight = x_weights[gid.x * params.width + sample_x];
            if (x_weight == 0.0f) {
                continue;
            }

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
