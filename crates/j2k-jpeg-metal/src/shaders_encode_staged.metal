// SPDX-License-Identifier: MIT OR Apache-2.0

kernel void jpeg_encode_baseline_precompute_batch(
    device const uchar *input [[buffer(0)]],
    device int *coefficients [[buffer(1)]],
    constant JpegBaselineEncodeParams *params [[buffer(2)]],
    constant uchar *q_luma [[buffer(3)]],
    constant uchar *q_chroma [[buffer(4)]],
    constant uint &tile_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint tile_index = 0u;
    uint local_mcu = gid;
    uint coefficient_block_base = 0u;
    for (; tile_index < tile_count; tile_index++) {
        constant JpegBaselineEncodeParams &tile_params = params[tile_index];
        const uint tile_mcus = tile_params.mcus_per_row * tile_params.mcu_rows;
        const uint tile_blocks = tile_mcus * jpeg_encode_blocks_per_mcu(tile_params);
        if (local_mcu < tile_mcus) {
            break;
        }
        local_mcu -= tile_mcus;
        coefficient_block_base += tile_blocks;
    }
    if (tile_index >= tile_count) {
        return;
    }

    constant JpegBaselineEncodeParams &tile_params = params[tile_index];
    const uint blocks_per_mcu = jpeg_encode_blocks_per_mcu(tile_params);
    const uint mcu_x = local_mcu % tile_params.mcus_per_row;
    const uint mcu_y = local_mcu / tile_params.mcus_per_row;
    uint block_index = 0u;
    for (uint component = 0u; component < tile_params.components; component++) {
        const uint h = component_h(tile_params, component);
        const uint v = component_v(tile_params, component);
        for (uint block_y = 0u; block_y < v; block_y++) {
            for (uint block_x = 0u; block_x < h; block_x++) {
                thread uchar block[64];
                thread int coeffs[64];
                jpeg_encode_sample_block(
                    input + tile_params.input_offset_bytes,
                    tile_params,
                    component,
                    mcu_x,
                    mcu_y,
                    block_x,
                    block_y,
                    block
                );
                jpeg_encode_fdct_quantize(
                    block,
                    component == 0u ? q_luma : q_chroma,
                    coeffs
                );
                device int *destination = coefficients
                    + (coefficient_block_base + local_mcu * blocks_per_mcu + block_index) * 64u;
                for (uint index = 0u; index < 64u; index++) {
                    destination[index] = coeffs[index];
                }
                block_index += 1u;
            }
        }
    }
}

kernel void jpeg_encode_baseline_entropy_from_coeffs_batch(
    device const int *coefficients [[buffer(0)]],
    device uchar *entropy [[buffer(1)]],
    device JpegBaselineEncodeStatus *status [[buffer(2)]],
    constant JpegBaselineEncodeParams *params [[buffer(3)]],
    constant JpegBaselineEncodeHuffmanTable &dc_luma [[buffer(4)]],
    constant JpegBaselineEncodeHuffmanTable &ac_luma [[buffer(5)]],
    constant JpegBaselineEncodeHuffmanTable &dc_chroma [[buffer(6)]],
    constant JpegBaselineEncodeHuffmanTable &ac_chroma [[buffer(7)]],
    constant uint &tile_count [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= tile_count) {
        return;
    }
    uint coefficient_block_base = 0u;
    for (uint tile_index = 0u; tile_index < gid; tile_index++) {
        constant JpegBaselineEncodeParams &prior = params[tile_index];
        coefficient_block_base += prior.mcus_per_row
            * prior.mcu_rows
            * jpeg_encode_blocks_per_mcu(prior);
    }
    constant JpegBaselineEncodeParams &tile_params = params[gid];
    jpeg_encode_baseline_entropy_from_coeffs_one(
        coefficients + coefficient_block_base * 64u,
        entropy + tile_params.entropy_offset_bytes,
        status + gid,
        tile_params,
        dc_luma,
        ac_luma,
        dc_chroma,
        ac_chroma
    );
}
