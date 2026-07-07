kernel void jpeg_decode_fast444(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFast444Params &params [[buffer(4)]],
    constant ushort *y_quant [[buffer(5)]],
    constant ushort *cb_quant [[buffer(6)]],
    constant ushort *cr_quant [[buffer(7)]],
    constant PreparedHuffman &y_dc [[buffer(8)]],
    constant PreparedHuffman &y_ac [[buffer(9)]],
    constant PreparedHuffman &cb_dc [[buffer(10)]],
    constant PreparedHuffman &cb_ac [[buffer(11)]],
    constant PreparedHuffman &cr_dc [[buffer(12)]],
    constant PreparedHuffman &cr_ac [[buffer(13)]],
    device const uint *restart_offsets [[buffer(14)]],
    device JpegDecodeStatus *status [[buffer(15)]],
    device const JpegEntropyCheckpoint *entropy_checkpoints [[buffer(16)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_mcus = params.mcus_per_row * params.mcu_rows;
    JPEG_ENTROPY_THREAD_VARS();
    if (!JPEG_CONFIGURE_ENTROPY_THREAD(
        gid,
        total_mcus,
        params,
        restart_offsets,
        entropy_checkpoints
    )) {
        return;
    }
    device JpegDecodeStatus *thread_status = status + gid;
    init_decode_status(thread_status);

    thread short coeffs[64];
    thread uchar pixels[64];
    uint mx = 0u;
    uint my = 0u;
    init_mcu_cursor(start_mcu, params.mcus_per_row, mx, my);
    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint block_x = mx * 8u;
        const uint block_y = my * 8u;

        if (!decode_idct_deposit_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.width, params.width, params.height, block_x, block_y, coeffs, pixels)) {
            return;
        }

        if (!decode_idct_deposit_block(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, cb_plane, params.width, params.width, params.height, block_x, block_y, coeffs, pixels)) {
            return;
        }

        if (!decode_idct_deposit_block(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, cr_plane, params.width, params.width, params.height, block_x, block_y, coeffs, pixels)) {
            return;
        }
        advance_mcu_cursor(mx, my, params.mcus_per_row);
    }
}

kernel void jpeg_decode_fast444_region(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFast444Params &params [[buffer(4)]],
    constant ushort *y_quant [[buffer(5)]],
    constant ushort *cb_quant [[buffer(6)]],
    constant ushort *cr_quant [[buffer(7)]],
    constant PreparedHuffman &y_dc [[buffer(8)]],
    constant PreparedHuffman &y_ac [[buffer(9)]],
    constant PreparedHuffman &cb_dc [[buffer(10)]],
    constant PreparedHuffman &cb_ac [[buffer(11)]],
    constant PreparedHuffman &cr_dc [[buffer(12)]],
    constant PreparedHuffman &cr_ac [[buffer(13)]],
    device const uint *restart_offsets [[buffer(14)]],
    device JpegDecodeStatus *status [[buffer(15)]],
    device const JpegEntropyCheckpoint *entropy_checkpoints [[buffer(16)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_mcus = params.mcus_per_row * params.mcu_rows;
    JPEG_ENTROPY_THREAD_VARS();
    if (!JPEG_CONFIGURE_ENTROPY_THREAD(
        gid,
        total_mcus,
        params,
        restart_offsets,
        entropy_checkpoints
    )) {
        return;
    }
    device JpegDecodeStatus *thread_status = status + gid;
    init_decode_status(thread_status);

    thread short coeffs[64];
    thread uchar pixels[64];
    uint mx = 0u;
    uint my = 0u;
    init_mcu_cursor(start_mcu, params.mcus_per_row, mx, my);
    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint block_x = mx * 8u;
        const uint block_y = my * 8u;

        if (!jpeg_decode_idct_deposit_region_block_or_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.width, params.width, params.height, params.origin_x, params.origin_y, block_x, block_y, 8u, 8u, coeffs, pixels)) {
            return;
        }
        if (!jpeg_decode_idct_deposit_region_block_or_skip(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, cb_plane, params.width, params.width, params.height, params.origin_x, params.origin_y, block_x, block_y, 8u, 8u, coeffs, pixels)) {
            return;
        }
        if (!jpeg_decode_idct_deposit_region_block_or_skip(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, cr_plane, params.width, params.width, params.height, params.origin_x, params.origin_y, block_x, block_y, 8u, 8u, coeffs, pixels)) {
            return;
        }
        advance_mcu_cursor(mx, my, params.mcus_per_row);
    }
}

kernel void jpeg_decode_fast444_scaled(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFast444ScaledParams &params [[buffer(4)]],
    constant ushort *y_quant [[buffer(5)]],
    constant ushort *cb_quant [[buffer(6)]],
    constant ushort *cr_quant [[buffer(7)]],
    constant PreparedHuffman &y_dc [[buffer(8)]],
    constant PreparedHuffman &y_ac [[buffer(9)]],
    constant PreparedHuffman &cb_dc [[buffer(10)]],
    constant PreparedHuffman &cb_ac [[buffer(11)]],
    constant PreparedHuffman &cr_dc [[buffer(12)]],
    constant PreparedHuffman &cr_ac [[buffer(13)]],
    device const uint *restart_offsets [[buffer(14)]],
    device JpegDecodeStatus *status [[buffer(15)]],
    device const JpegEntropyCheckpoint *entropy_checkpoints [[buffer(16)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_mcus = params.mcus_per_row * params.mcu_rows;
    JPEG_ENTROPY_THREAD_VARS();
    if (!JPEG_CONFIGURE_ENTROPY_THREAD(
        gid,
        total_mcus,
        params,
        restart_offsets,
        entropy_checkpoints
    )) {
        return;
    }
    device JpegDecodeStatus *thread_status = status + gid;
    init_decode_status(thread_status);

    thread short coeffs[64];
    const uint block_size = 8u >> params.scale_shift;

    uint mx = 0u;
    uint my = 0u;
    init_mcu_cursor(start_mcu, params.mcus_per_row, mx, my);
    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint block_x = mx * block_size;
        const uint block_y = my * block_size;
        if (!jpeg_decode_deposit_scaled_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.scaled_width, params.scaled_width, params.scaled_height, block_x, block_y, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_block(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, cb_plane, params.scaled_width, params.scaled_width, params.scaled_height, block_x, block_y, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_block(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, cr_plane, params.scaled_width, params.scaled_width, params.scaled_height, block_x, block_y, params.scale_shift, coeffs)) {
            return;
        }
        advance_mcu_cursor(mx, my, params.mcus_per_row);
    }
}

kernel void jpeg_decode_fast444_scaled_region(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFast444ScaledParams &params [[buffer(4)]],
    constant ushort *y_quant [[buffer(5)]],
    constant ushort *cb_quant [[buffer(6)]],
    constant ushort *cr_quant [[buffer(7)]],
    constant PreparedHuffman &y_dc [[buffer(8)]],
    constant PreparedHuffman &y_ac [[buffer(9)]],
    constant PreparedHuffman &cb_dc [[buffer(10)]],
    constant PreparedHuffman &cb_ac [[buffer(11)]],
    constant PreparedHuffman &cr_dc [[buffer(12)]],
    constant PreparedHuffman &cr_ac [[buffer(13)]],
    device const uint *restart_offsets [[buffer(14)]],
    device JpegDecodeStatus *status [[buffer(15)]],
    device const JpegEntropyCheckpoint *entropy_checkpoints [[buffer(16)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_mcus = params.mcus_per_row * params.mcu_rows;
    JPEG_ENTROPY_THREAD_VARS();
    if (!JPEG_CONFIGURE_ENTROPY_THREAD(
        gid,
        total_mcus,
        params,
        restart_offsets,
        entropy_checkpoints
    )) {
        return;
    }
    device JpegDecodeStatus *thread_status = status + gid;
    init_decode_status(thread_status);

    thread short coeffs[64];
    const uint block_size = 8u >> params.scale_shift;

    uint mx = 0u;
    uint my = 0u;
    init_mcu_cursor(start_mcu, params.mcus_per_row, mx, my);
    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint block_x = mx * block_size;
        const uint block_y = my * block_size;

        if (!jpeg_decode_deposit_scaled_region_block_or_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.scaled_width, params.scaled_width, params.scaled_height, params.origin_x, params.origin_y, block_x, block_y, block_size, block_size, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_region_block_or_skip(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, cb_plane, params.scaled_width, params.scaled_width, params.scaled_height, params.origin_x, params.origin_y, block_x, block_y, block_size, block_size, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_region_block_or_skip(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, cr_plane, params.scaled_width, params.scaled_width, params.scaled_height, params.origin_x, params.origin_y, block_x, block_y, block_size, block_size, params.scale_shift, coeffs)) {
            return;
        }
        advance_mcu_cursor(mx, my, params.mcus_per_row);
    }
}

kernel void jpeg_decode_fast444_scaled_region_batch(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFastRegionScaledBatchParams &params [[buffer(4)]],
    constant ushort *y_quant [[buffer(5)]],
    constant ushort *cb_quant [[buffer(6)]],
    constant ushort *cr_quant [[buffer(7)]],
    constant PreparedHuffman &y_dc [[buffer(8)]],
    constant PreparedHuffman &y_ac [[buffer(9)]],
    constant PreparedHuffman &cb_dc [[buffer(10)]],
    constant PreparedHuffman &cb_ac [[buffer(11)]],
    constant PreparedHuffman &cr_dc [[buffer(12)]],
    constant PreparedHuffman &cr_ac [[buffer(13)]],
    device const uint *entropy_offsets [[buffer(14)]],
    device const uint *entropy_lens [[buffer(15)]],
    device JpegDecodeStatus *status [[buffer(16)]],
    device const JpegEntropyCheckpoint *entropy_checkpoints [[buffer(17)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_mcus = params.mcus_per_row * params.mcu_rows;
    JPEG_BATCH_ENTROPY_THREAD_VARS();
    if (!JPEG_CONFIGURE_BATCH_ENTROPY_THREAD(
        gid,
        total_mcus,
        params,
        entropy_offsets,
        entropy_lens,
        entropy_checkpoints
    )) {
        return;
    }
    device JpegDecodeStatus *thread_status = status + gid;
    init_decode_status(thread_status);

    const uint plane_base = tile_index * params.scaled_width * params.scaled_height;
    device uchar *tile_y_plane = y_plane + plane_base;
    device uchar *tile_cb_plane = cb_plane + plane_base;
    device uchar *tile_cr_plane = cr_plane + plane_base;

    thread short coeffs[64];
    const uint block_size = 8u >> params.scale_shift;

    uint mx = 0u;
    uint my = 0u;
    init_mcu_cursor(start_mcu, params.mcus_per_row, mx, my);
    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint block_x = mx * block_size;
        const uint block_y = my * block_size;

        if (!jpeg_decode_deposit_scaled_region_block_or_skip(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, tile_y_plane, params.scaled_width, params.scaled_width, params.scaled_height, params.origin_x, params.origin_y, block_x, block_y, block_size, block_size, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_region_block_or_skip(br, entropy, entropy_end, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, tile_cb_plane, params.scaled_width, params.scaled_width, params.scaled_height, params.origin_x, params.origin_y, block_x, block_y, block_size, block_size, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_region_block_or_skip(br, entropy, entropy_end, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, tile_cr_plane, params.scaled_width, params.scaled_width, params.scaled_height, params.origin_x, params.origin_y, block_x, block_y, block_size, block_size, params.scale_shift, coeffs)) {
            return;
        }
        advance_mcu_cursor(mx, my, params.mcus_per_row);
    }
}

kernel void jpeg_decode_fast444_rgba_texture_batch(
    device const uchar *entropy [[buffer(0)]],
    constant JpegFast444TextureBatchParams &params [[buffer(4)]],
    constant ushort *y_quant [[buffer(5)]],
    constant ushort *cb_quant [[buffer(6)]],
    constant ushort *cr_quant [[buffer(7)]],
    constant PreparedHuffman &y_dc [[buffer(8)]],
    constant PreparedHuffman &y_ac [[buffer(9)]],
    constant PreparedHuffman &cb_dc [[buffer(10)]],
    constant PreparedHuffman &cb_ac [[buffer(11)]],
    constant PreparedHuffman &cr_dc [[buffer(12)]],
    constant PreparedHuffman &cr_ac [[buffer(13)]],
    device const uint *entropy_offsets [[buffer(14)]],
    device const uint *entropy_lens [[buffer(15)]],
    device JpegDecodeStatus *status [[buffer(16)]],
    device const JpegEntropyCheckpoint *entropy_checkpoints [[buffer(17)]],
    texture2d<float, access::write> out [[texture(0)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.segment_count) {
        return;
    }

    const uint total_mcus = params.mcus_per_row * params.mcu_rows;
    const uint status_index = params.tile_index * params.segment_count + gid;
    device JpegDecodeStatus *thread_status = status + status_index;
    init_decode_status(thread_status);

    JPEG_BATCH_ENTROPY_THREAD_VARS();
    if (!configure_batch_entropy_thread(
        status_index,
        total_mcus,
        params.segment_count,
        params.tile_index + 1u,
        entropy_offsets,
        entropy_lens,
        entropy_checkpoints,
        br,
        tile_index,
        start_mcu,
        end_mcu,
        entropy_end,
        y_prev_dc,
        cb_prev_dc,
        cr_prev_dc
    )) {
        return;
    }
    if (tile_index != params.tile_index) {
        return;
    }

    thread short coeffs[64];
    thread uchar y_pixels[64];
    thread uchar cb_pixels[64];
    thread uchar cr_pixels[64];
    uint mx = 0u;
    uint my = 0u;
    init_mcu_cursor(start_mcu, params.mcus_per_row, mx, my);
    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint block_x = mx * 8u;
        const uint block_y = my * 8u;
        bool dc_only = false;

        if (!decode_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        idct_block(coeffs, dc_only, y_pixels);

        if (!decode_block(br, entropy, entropy_end, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        idct_block(coeffs, dc_only, cb_pixels);

        if (!decode_block(br, entropy, entropy_end, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        idct_block(coeffs, dc_only, cr_pixels);

        const uint copy_width = min(8u, params.width - min(block_x, params.width));
        const uint copy_height = min(8u, params.height - min(block_y, params.height));
        for (uint by = 0u; by < copy_height; ++by) {
            for (uint bx = 0u; bx < copy_width; ++bx) {
                const uint idx = by * 8u + bx;
                const uint2 pos = uint2(block_x + bx, block_y + by);
                if (params.mode == MODE_GRAY) {
                    const uchar gray = y_pixels[idx];
                    out.write(rgba_float_direct(gray, gray, gray, params.alpha), pos);
                } else if (params.mode == MODE_RGB) {
                    out.write(
                        rgba_float_direct(y_pixels[idx], cb_pixels[idx], cr_pixels[idx], params.alpha),
                        pos
                    );
                } else {
                    jpeg_write_ycbcr_rgba(
                        out,
                        pos,
                        y_pixels[idx],
                        cb_pixels[idx],
                        cr_pixels[idx],
                        params.alpha
                    );
                }
            }
        }
        advance_mcu_cursor(mx, my, params.mcus_per_row);
    }
}
