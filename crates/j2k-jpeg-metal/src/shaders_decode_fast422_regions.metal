kernel void jpeg_decode_fast422(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFast420Params &params [[buffer(4)]],
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
        const uint y_x = mx * 16u;
        const uint y_y = my * 8u;
        const uint c_x = mx * 8u;
        const uint c_y = my * 8u;

        if (!decode_idct_deposit_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.width, params.width, params.height, y_x, y_y, coeffs, pixels)) {
            return;
        }

        if (!decode_idct_deposit_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.width, params.width, params.height, y_x + 8u, y_y, coeffs, pixels)) {
            return;
        }

        if (!decode_idct_deposit_block(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, cb_plane, params.chroma_width, params.chroma_width, params.chroma_height, c_x, c_y, coeffs, pixels)) {
            return;
        }

        if (!decode_idct_deposit_block(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, cr_plane, params.chroma_width, params.chroma_width, params.chroma_height, c_x, c_y, coeffs, pixels)) {
            return;
        }
        advance_mcu_cursor(mx, my, params.mcus_per_row);
    }
}

kernel void jpeg_decode_fast422_batch(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFast420BatchParams &params [[buffer(4)]],
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

    const uint y_plane_base = tile_index * params.width * params.height;
    const uint chroma_plane_base = tile_index * params.chroma_width * params.chroma_height;
    device uchar *tile_y_plane = y_plane + y_plane_base;
    device uchar *tile_cb_plane = cb_plane + chroma_plane_base;
    device uchar *tile_cr_plane = cr_plane + chroma_plane_base;

    thread short coeffs[64];
    thread uchar pixels[64];

    uint mx = 0u;
    uint my = 0u;
    init_mcu_cursor(start_mcu, params.mcus_per_row, mx, my);
    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint y_x = mx * 16u;
        const uint y_y = my * 8u;
        const uint c_x = mx * 8u;
        const uint c_y = my * 8u;

        if (!decode_idct_deposit_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, tile_y_plane, params.width, params.width, params.height, y_x, y_y, coeffs, pixels)) {
            return;
        }

        if (!decode_idct_deposit_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, tile_y_plane, params.width, params.width, params.height, y_x + 8u, y_y, coeffs, pixels)) {
            return;
        }

        if (!decode_idct_deposit_block(br, entropy, entropy_end, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, tile_cb_plane, params.chroma_width, params.chroma_width, params.chroma_height, c_x, c_y, coeffs, pixels)) {
            return;
        }

        if (!decode_idct_deposit_block(br, entropy, entropy_end, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, tile_cr_plane, params.chroma_width, params.chroma_width, params.chroma_height, c_x, c_y, coeffs, pixels)) {
            return;
        }
        advance_mcu_cursor(mx, my, params.mcus_per_row);
    }
}

kernel void jpeg_decode_fast422_rgba_texture_batch(
    device const uchar *entropy [[buffer(0)]],
    constant JpegFast422TextureBatchParams &params [[buffer(4)]],
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
    device uint *boundary_meta [[buffer(18)]],
    device uchar *boundary_samples [[buffer(19)]],
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
    const uint boundary_meta_base = fast422_boundary_meta_base(status_index);
    jpeg_decode_clear_meta_quad(boundary_meta, boundary_meta_base);

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
    thread uchar y_left_pixels[64];
    thread uchar y_right_pixels[64];
    thread uchar cb_pixels[64];
    thread uchar cr_pixels[64];
    thread uchar prev_y_right_pixels[64];
    thread uchar prev_cb_pixels[64];
    thread uchar prev_cr_pixels[64];
    bool have_prev_horizontal = false;

    uint mx = 0u;
    uint my = 0u;
    init_mcu_cursor(start_mcu, params.mcus_per_row, mx, my);
    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint y_x = mx * 16u;
        const uint y_y = my * 8u;
        bool dc_only = false;

        if (!decode_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        idct_block(coeffs, dc_only, y_left_pixels);

        if (!decode_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        idct_block(coeffs, dc_only, y_right_pixels);

        if (!decode_block(br, entropy, entropy_end, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        idct_block(coeffs, dc_only, cb_pixels);

        if (!decode_block(br, entropy, entropy_end, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        idct_block(coeffs, dc_only, cr_pixels);

        const uint copy_width = jpeg_clamped_extent(y_x, 16u, params.width);
        const uint copy_height = jpeg_clamped_extent(y_y, 8u, params.height);
        const bool starts_mid_row = mcu_index == start_mcu && mx > 0u;
        const bool ends_mid_row = mcu_index + 1u >= end_mcu && mx + 1u < params.mcus_per_row;
        const uint boundary_sample_base = fast422_boundary_sample_base(status_index);
        if (starts_mid_row) {
            boundary_meta[boundary_meta_base] = y_x;
            boundary_meta[boundary_meta_base + 1u] = y_y;
            boundary_meta[boundary_meta_base + 2u] = 1u;
            for (uint by = 0u; by < copy_height; ++by) {
                boundary_samples[boundary_sample_base + by] = y_left_pixels[by * 8u];
                boundary_samples[boundary_sample_base + 8u + by] = cb_pixels[by * 8u];
                boundary_samples[boundary_sample_base + 16u + by] = cr_pixels[by * 8u];
            }
        }
        if (ends_mid_row) {
            boundary_meta[boundary_meta_base + 3u] = 1u;
            for (uint by = 0u; by < copy_height; ++by) {
                boundary_samples[boundary_sample_base + 24u + by] = y_right_pixels[by * 8u + 7u];
                boundary_samples[boundary_sample_base + 32u + by] = cb_pixels[by * 8u + 7u];
                boundary_samples[boundary_sample_base + 40u + by] = cr_pixels[by * 8u + 7u];
            }
        }
        if (have_prev_horizontal && mx > 0u) {
            const uint prev_x = y_x - 1u;
            for (uint by = 0u; by < copy_height; ++by) {
                thread const uchar *cb_row = cb_pixels + by * 8u;
                thread const uchar *cr_row = cr_pixels + by * 8u;
                thread const uchar *prev_cb_row = prev_cb_pixels + by * 8u;
                thread const uchar *prev_cr_row = prev_cr_pixels + by * 8u;
                const uchar y_value = prev_y_right_pixels[by * 8u + 7u];
                const uchar cb_value = h2v1_boundary_left_from_samples(prev_cb_row[7], cb_row[0]);
                const uchar cr_value = h2v1_boundary_left_from_samples(prev_cr_row[7], cr_row[0]);
                jpeg_write_ycbcr_rgba(out, uint2(prev_x, y_y + by), y_value, cb_value, cr_value, params.alpha);
            }
        }

        const uint local_sample_base = mx * 8u;
        const bool has_right_mcu = mx + 1u < params.mcus_per_row;
        const uint last_chroma_x = params.chroma_width * 2u - 1u;
        for (uint by = 0u; by < copy_height; ++by) {
            thread const uchar *cb_row = cb_pixels + by * 8u;
            thread const uchar *cr_row = cr_pixels + by * 8u;
            const uchar left_cb = mx == 0u || starts_mid_row
                ? cb_row[0]
                : prev_cb_pixels[by * 8u + 7u];
            const uchar left_cr = mx == 0u || starts_mid_row
                ? cr_row[0]
                : prev_cr_pixels[by * 8u + 7u];
            for (uint bx = 0u; bx < copy_width; ++bx) {
                const uint x = y_x + bx;
                if (starts_mid_row && bx == 0u) {
                    continue;
                }
                if (has_right_mcu && bx == 15u && (x & 1u) == 1u && x != last_chroma_x) {
                    continue;
                }
                const uchar y_value = bx < 8u
                    ? y_left_pixels[by * 8u + bx]
                    : y_right_pixels[by * 8u + (bx - 8u)];
                const uchar cb_value = h2v1_sample_thread_local(
                    cb_row,
                    params.chroma_width,
                    x,
                    local_sample_base,
                    left_cb
                );
                const uchar cr_value = h2v1_sample_thread_local(
                    cr_row,
                    params.chroma_width,
                    x,
                    local_sample_base,
                    left_cr
                );
                jpeg_write_ycbcr_rgba(out, uint2(x, y_y + by), y_value, cb_value, cr_value, params.alpha);
            }
        }

        for (uint i = 0u; i < 64u; ++i) {
            prev_y_right_pixels[i] = y_right_pixels[i];
            prev_cb_pixels[i] = cb_pixels[i];
            prev_cr_pixels[i] = cr_pixels[i];
        }
        have_prev_horizontal = true;
        advance_mcu_cursor(mx, my, params.mcus_per_row);
    }
}

kernel void jpeg_resolve_fast422_rgba_texture_boundaries(
    device const uint *boundary_meta [[buffer(0)]],
    device const uchar *boundary_samples [[buffer(1)]],
    constant JpegFast422TextureBatchParams &params [[buffer(2)]],
    texture2d<float, access::write> out [[texture(0)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0u || gid >= params.segment_count) {
        return;
    }

    const uint record_index = params.tile_index * params.segment_count + gid;
    const uint previous_record_index = record_index - 1u;
    const uint meta_base = fast422_boundary_meta_base(record_index);
    const uint previous_meta_base = fast422_boundary_meta_base(previous_record_index);
    if (boundary_meta[meta_base + 2u] == 0u || boundary_meta[previous_meta_base + 3u] == 0u) {
        return;
    }

    const uint x = boundary_meta[meta_base];
    const uint y = boundary_meta[meta_base + 1u];
    if (x == 0u || x >= params.width || y >= params.height) {
        return;
    }

    const uint sample_base = fast422_boundary_sample_base(record_index);
    const uint previous_sample_base = fast422_boundary_sample_base(previous_record_index);
    const uint copy_height = min(8u, params.height - y);
    for (uint by = 0u; by < copy_height; ++by) {
        const uchar left_y = boundary_samples[previous_sample_base + 24u + by];
        const uchar left_cb = boundary_samples[previous_sample_base + 32u + by];
        const uchar left_cr = boundary_samples[previous_sample_base + 40u + by];
        const uchar right_y = boundary_samples[sample_base + by];
        const uchar right_cb = boundary_samples[sample_base + 8u + by];
        const uchar right_cr = boundary_samples[sample_base + 16u + by];
        const uchar resolved_left_cb = h2v1_boundary_left_from_samples(left_cb, right_cb);
        const uchar resolved_left_cr = h2v1_boundary_left_from_samples(left_cr, right_cr);
        const uchar resolved_right_cb = h2v1_boundary_right_from_samples(left_cb, right_cb);
        const uchar resolved_right_cr = h2v1_boundary_right_from_samples(left_cr, right_cr);
        const uint row_y = y + by;
        jpeg_write_ycbcr_rgba(out, uint2(x - 1u, row_y), left_y, resolved_left_cb, resolved_left_cr, params.alpha);
        jpeg_write_ycbcr_rgba(out, uint2(x, row_y), right_y, resolved_right_cb, resolved_right_cr, params.alpha);
    }
}

kernel void jpeg_decode_fast422_region(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFast420Params &params [[buffer(4)]],
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

    const uint chroma_origin_x = params.origin_x / 2u;
    const uint chroma_origin_y = params.origin_y;

    uint mx = 0u;
    uint my = 0u;
    init_mcu_cursor(start_mcu, params.mcus_per_row, mx, my);
    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint y_x = mx * 16u;
        const uint y_y = my * 8u;
        const uint c_x = mx * 8u;
        const uint c_y = my * 8u;
        if (!jpeg_decode_idct_deposit_region_block_or_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.width, params.width, params.height, params.origin_x, params.origin_y, y_x, y_y, 8u, 8u, coeffs, pixels)) {
            return;
        }
        if (!jpeg_decode_idct_deposit_region_block_or_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.width, params.width, params.height, params.origin_x, params.origin_y, y_x + 8u, y_y, 8u, 8u, coeffs, pixels)) {
            return;
        }
        if (!jpeg_decode_idct_deposit_region_block_or_skip(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, cb_plane, params.chroma_width, params.chroma_width, params.chroma_height, chroma_origin_x, chroma_origin_y, c_x, c_y, 8u, 8u, coeffs, pixels)) {
            return;
        }
        if (!jpeg_decode_idct_deposit_region_block_or_skip(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, cr_plane, params.chroma_width, params.chroma_width, params.chroma_height, chroma_origin_x, chroma_origin_y, c_x, c_y, 8u, 8u, coeffs, pixels)) {
            return;
        }
        advance_mcu_cursor(mx, my, params.mcus_per_row);
    }
}

kernel void jpeg_decode_fast422_scaled(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFast420ScaledParams &params [[buffer(4)]],
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
    const uint mcu_width = 16u >> params.scale_shift;
    const uint mcu_height = 8u >> params.scale_shift;

    uint mx = 0u;
    uint my = 0u;
    init_mcu_cursor(start_mcu, params.mcus_per_row, mx, my);
    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint y_x = mx * mcu_width;
        const uint y_y = my * mcu_height;
        const uint c_x = mx * block_size;
        const uint c_y = my * block_size;
        if (!jpeg_decode_deposit_scaled_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.scaled_width, params.scaled_width, params.scaled_height, y_x, y_y, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.scaled_width, params.scaled_width, params.scaled_height, y_x + block_size, y_y, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_block(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, cb_plane, params.chroma_width, params.chroma_width, params.chroma_height, c_x, c_y, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_block(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, cr_plane, params.chroma_width, params.chroma_width, params.chroma_height, c_x, c_y, params.scale_shift, coeffs)) {
            return;
        }
        advance_mcu_cursor(mx, my, params.mcus_per_row);
    }
}

kernel void jpeg_decode_fast422_scaled_region(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFast420ScaledParams &params [[buffer(4)]],
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
    const uint mcu_width = 16u >> params.scale_shift;
    const uint mcu_height = 8u >> params.scale_shift;
    const uint chroma_origin_x = params.origin_x / 2u;
    const uint chroma_origin_y = params.origin_y;

    uint mx = 0u;
    uint my = 0u;
    init_mcu_cursor(start_mcu, params.mcus_per_row, mx, my);
    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint y_x = mx * mcu_width;
        const uint y_y = my * mcu_height;
        const uint c_x = mx * block_size;
        const uint c_y = my * block_size;
        if (!jpeg_decode_deposit_scaled_region_block_or_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.scaled_width, params.scaled_width, params.scaled_height, params.origin_x, params.origin_y, y_x, y_y, block_size, block_size, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_region_block_or_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.scaled_width, params.scaled_width, params.scaled_height, params.origin_x, params.origin_y, y_x + block_size, y_y, block_size, block_size, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_region_block_or_skip(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, cb_plane, params.chroma_width, params.chroma_width, params.chroma_height, chroma_origin_x, chroma_origin_y, c_x, c_y, block_size, block_size, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_region_block_or_skip(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, cr_plane, params.chroma_width, params.chroma_width, params.chroma_height, chroma_origin_x, chroma_origin_y, c_x, c_y, block_size, block_size, params.scale_shift, coeffs)) {
            return;
        }
        advance_mcu_cursor(mx, my, params.mcus_per_row);
    }
}

kernel void jpeg_decode_fast422_scaled_region_batch(
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

    const uint y_plane_base = tile_index * params.scaled_width * params.scaled_height;
    const uint chroma_plane_base = tile_index * params.chroma_width * params.chroma_height;
    device uchar *tile_y_plane = y_plane + y_plane_base;
    device uchar *tile_cb_plane = cb_plane + chroma_plane_base;
    device uchar *tile_cr_plane = cr_plane + chroma_plane_base;

    thread short coeffs[64];

    const uint block_size = 8u >> params.scale_shift;
    const uint mcu_width = 16u >> params.scale_shift;
    const uint mcu_height = 8u >> params.scale_shift;
    const uint chroma_origin_x = params.origin_x / 2u;
    const uint chroma_origin_y = params.origin_y;

    uint mx = 0u;
    uint my = 0u;
    init_mcu_cursor(start_mcu, params.mcus_per_row, mx, my);
    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint y_x = mx * mcu_width;
        const uint y_y = my * mcu_height;
        const uint c_x = mx * block_size;
        const uint c_y = my * block_size;
        if (!jpeg_decode_deposit_scaled_region_block_or_skip(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, tile_y_plane, params.scaled_width, params.scaled_width, params.scaled_height, params.origin_x, params.origin_y, y_x, y_y, block_size, block_size, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_region_block_or_skip(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, tile_y_plane, params.scaled_width, params.scaled_width, params.scaled_height, params.origin_x, params.origin_y, y_x + block_size, y_y, block_size, block_size, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_region_block_or_skip(br, entropy, entropy_end, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, tile_cb_plane, params.chroma_width, params.chroma_width, params.chroma_height, chroma_origin_x, chroma_origin_y, c_x, c_y, block_size, block_size, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_region_block_or_skip(br, entropy, entropy_end, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, tile_cr_plane, params.chroma_width, params.chroma_width, params.chroma_height, chroma_origin_x, chroma_origin_y, c_x, c_y, block_size, block_size, params.scale_shift, coeffs)) {
            return;
        }
        advance_mcu_cursor(mx, my, params.mcus_per_row);
    }
}

kernel void jpeg_decode_fast420_region(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFast420Params &params [[buffer(4)]],
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

    const uint chroma_origin_x = params.origin_x / 2u;
    const uint chroma_origin_y = params.origin_y / 2u;

    uint mx = 0u;
    uint my = 0u;
    init_mcu_cursor(start_mcu, params.mcus_per_row, mx, my);
    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint y_x = mx * 16u;
        const uint y_y = my * 16u;
        const uint c_x = mx * 8u;
        const uint c_y = my * 8u;
        if (!jpeg_decode_idct_deposit_region_block_or_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.width, params.width, params.height, params.origin_x, params.origin_y, y_x, y_y, 8u, 8u, coeffs, pixels)) {
            return;
        }
        if (!jpeg_decode_idct_deposit_region_block_or_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.width, params.width, params.height, params.origin_x, params.origin_y, y_x + 8u, y_y, 8u, 8u, coeffs, pixels)) {
            return;
        }
        if (!jpeg_decode_idct_deposit_region_block_or_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.width, params.width, params.height, params.origin_x, params.origin_y, y_x, y_y + 8u, 8u, 8u, coeffs, pixels)) {
            return;
        }
        if (!jpeg_decode_idct_deposit_region_block_or_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.width, params.width, params.height, params.origin_x, params.origin_y, y_x + 8u, y_y + 8u, 8u, 8u, coeffs, pixels)) {
            return;
        }
        if (!jpeg_decode_idct_deposit_region_block_or_skip(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, cb_plane, params.chroma_width, params.chroma_width, params.chroma_height, chroma_origin_x, chroma_origin_y, c_x, c_y, 8u, 8u, coeffs, pixels)) {
            return;
        }
        if (!jpeg_decode_idct_deposit_region_block_or_skip(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, cr_plane, params.chroma_width, params.chroma_width, params.chroma_height, chroma_origin_x, chroma_origin_y, c_x, c_y, 8u, 8u, coeffs, pixels)) {
            return;
        }
        advance_mcu_cursor(mx, my, params.mcus_per_row);
    }
}

kernel void jpeg_decode_fast420_scaled(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFast420ScaledParams &params [[buffer(4)]],
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

    const uint y_block_size = 8u >> params.scale_shift;
    const uint c_block_size = 8u >> params.scale_shift;
    const uint y_mcu_size = 16u >> params.scale_shift;

    uint mx = 0u;
    uint my = 0u;
    init_mcu_cursor(start_mcu, params.mcus_per_row, mx, my);
    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint y_x = mx * y_mcu_size;
        const uint y_y = my * y_mcu_size;
        const uint c_x = mx * c_block_size;
        const uint c_y = my * c_block_size;
        if (!jpeg_decode_deposit_scaled_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.scaled_width, params.scaled_width, params.scaled_height, y_x, y_y, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.scaled_width, params.scaled_width, params.scaled_height, y_x + y_block_size, y_y, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.scaled_width, params.scaled_width, params.scaled_height, y_x, y_y + y_block_size, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.scaled_width, params.scaled_width, params.scaled_height, y_x + y_block_size, y_y + y_block_size, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_block(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, cb_plane, params.chroma_width, params.chroma_width, params.chroma_height, c_x, c_y, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_block(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, cr_plane, params.chroma_width, params.chroma_width, params.chroma_height, c_x, c_y, params.scale_shift, coeffs)) {
            return;
        }
        advance_mcu_cursor(mx, my, params.mcus_per_row);
    }
}

kernel void jpeg_decode_fast420_scaled_region(
    device const uchar *entropy [[buffer(0)]],
    device uchar *y_plane [[buffer(1)]],
    device uchar *cb_plane [[buffer(2)]],
    device uchar *cr_plane [[buffer(3)]],
    constant JpegFast420ScaledParams &params [[buffer(4)]],
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

    const uint y_block_size = 8u >> params.scale_shift;
    const uint c_block_size = 8u >> params.scale_shift;
    const uint y_mcu_size = 16u >> params.scale_shift;
    const uint chroma_origin_x = params.origin_x / 2u;
    const uint chroma_origin_y = params.origin_y / 2u;

    uint mx = 0u;
    uint my = 0u;
    init_mcu_cursor(start_mcu, params.mcus_per_row, mx, my);
    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint y_x = mx * y_mcu_size;
        const uint y_y = my * y_mcu_size;
        const uint c_x = mx * c_block_size;
        const uint c_y = my * c_block_size;
        if (!jpeg_decode_deposit_scaled_region_block_or_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.scaled_width, params.scaled_width, params.scaled_height, params.origin_x, params.origin_y, y_x, y_y, y_block_size, y_block_size, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_region_block_or_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.scaled_width, params.scaled_width, params.scaled_height, params.origin_x, params.origin_y, y_x + y_block_size, y_y, y_block_size, y_block_size, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_region_block_or_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.scaled_width, params.scaled_width, params.scaled_height, params.origin_x, params.origin_y, y_x, y_y + y_block_size, y_block_size, y_block_size, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_region_block_or_skip(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.scaled_width, params.scaled_width, params.scaled_height, params.origin_x, params.origin_y, y_x + y_block_size, y_y + y_block_size, y_block_size, y_block_size, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_region_block_or_skip(br, entropy, params.entropy_len, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, cb_plane, params.chroma_width, params.chroma_width, params.chroma_height, chroma_origin_x, chroma_origin_y, c_x, c_y, c_block_size, c_block_size, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_region_block_or_skip(br, entropy, params.entropy_len, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, cr_plane, params.chroma_width, params.chroma_width, params.chroma_height, chroma_origin_x, chroma_origin_y, c_x, c_y, c_block_size, c_block_size, params.scale_shift, coeffs)) {
            return;
        }
        advance_mcu_cursor(mx, my, params.mcus_per_row);
    }
}

kernel void jpeg_decode_fast420_scaled_region_batch(
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

    const uint y_plane_base = tile_index * params.scaled_width * params.scaled_height;
    const uint chroma_plane_base = tile_index * params.chroma_width * params.chroma_height;
    device uchar *tile_y_plane = y_plane + y_plane_base;
    device uchar *tile_cb_plane = cb_plane + chroma_plane_base;
    device uchar *tile_cr_plane = cr_plane + chroma_plane_base;

    thread short coeffs[64];

    const uint y_block_size = 8u >> params.scale_shift;
    const uint c_block_size = 8u >> params.scale_shift;
    const uint y_mcu_size = 16u >> params.scale_shift;
    const uint chroma_origin_x = params.origin_x / 2u;
    const uint chroma_origin_y = params.origin_y / 2u;

    uint mx = 0u;
    uint my = 0u;
    init_mcu_cursor(start_mcu, params.mcus_per_row, mx, my);
    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint y_x = mx * y_mcu_size;
        const uint y_y = my * y_mcu_size;
        const uint c_x = mx * c_block_size;
        const uint c_y = my * c_block_size;
        if (!jpeg_decode_deposit_scaled_region_block_or_skip(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, tile_y_plane, params.scaled_width, params.scaled_width, params.scaled_height, params.origin_x, params.origin_y, y_x, y_y, y_block_size, y_block_size, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_region_block_or_skip(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, tile_y_plane, params.scaled_width, params.scaled_width, params.scaled_height, params.origin_x, params.origin_y, y_x + y_block_size, y_y, y_block_size, y_block_size, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_region_block_or_skip(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, tile_y_plane, params.scaled_width, params.scaled_width, params.scaled_height, params.origin_x, params.origin_y, y_x, y_y + y_block_size, y_block_size, y_block_size, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_region_block_or_skip(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, tile_y_plane, params.scaled_width, params.scaled_width, params.scaled_height, params.origin_x, params.origin_y, y_x + y_block_size, y_y + y_block_size, y_block_size, y_block_size, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_region_block_or_skip(br, entropy, entropy_end, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, tile_cb_plane, params.chroma_width, params.chroma_width, params.chroma_height, chroma_origin_x, chroma_origin_y, c_x, c_y, c_block_size, c_block_size, params.scale_shift, coeffs)) {
            return;
        }
        if (!jpeg_decode_deposit_scaled_region_block_or_skip(br, entropy, entropy_end, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, tile_cr_plane, params.chroma_width, params.chroma_width, params.chroma_height, chroma_origin_x, chroma_origin_y, c_x, c_y, c_block_size, c_block_size, params.scale_shift, coeffs)) {
            return;
        }
        advance_mcu_cursor(mx, my, params.mcus_per_row);
    }
}
