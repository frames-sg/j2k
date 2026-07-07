kernel void jpeg_decode_fast420(
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
            const uint y_y = my * 16u;
            const uint c_x = mx * 8u;
            const uint c_y = my * 8u;

            if (!decode_idct_deposit_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.width, params.width, params.height, y_x, y_y, coeffs, pixels)) {
                return;
            }

            if (!decode_idct_deposit_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.width, params.width, params.height, y_x + 8u, y_y, coeffs, pixels)) {
                return;
            }

            if (!decode_idct_deposit_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.width, params.width, params.height, y_x, y_y + 8u, coeffs, pixels)) {
                return;
            }

            if (!decode_idct_deposit_block(br, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, y_plane, params.width, params.width, params.height, y_x + 8u, y_y + 8u, coeffs, pixels)) {
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

kernel void jpeg_decode_fast420_batch(
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
        const uint y_y = my * 16u;
        const uint c_x = mx * 8u;
        const uint c_y = my * 8u;

        if (!decode_idct_deposit_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, tile_y_plane, params.width, params.width, params.height, y_x, y_y, coeffs, pixels)) {
            return;
        }

        if (!decode_idct_deposit_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, tile_y_plane, params.width, params.width, params.height, y_x + 8u, y_y, coeffs, pixels)) {
            return;
        }

        if (!decode_idct_deposit_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, tile_y_plane, params.width, params.width, params.height, y_x, y_y + 8u, coeffs, pixels)) {
            return;
        }

        if (!decode_idct_deposit_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, tile_y_plane, params.width, params.width, params.height, y_x + 8u, y_y + 8u, coeffs, pixels)) {
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

kernel void jpeg_decode_fast420_rgba_texture_batch(
    device const uchar *entropy [[buffer(0)]],
    constant JpegFast420TextureBatchParams &params [[buffer(4)]],
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
    device uint *vertical_meta [[buffer(20)]],
    device uchar *vertical_samples [[buffer(21)]],
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
    thread uchar y00_pixels[64];
    thread uchar y10_pixels[64];
    thread uchar y01_pixels[64];
    thread uchar y11_pixels[64];
    thread uchar cb_pixels[64];
    thread uchar cr_pixels[64];
    thread uchar prev_y10_pixels[64];
    thread uchar prev_y11_pixels[64];
    thread uchar prev_cb_pixels[64];
    thread uchar prev_cr_pixels[64];
    bool have_prev_horizontal = false;
    thread uchar prev_vertical_y01_pixels[64];
    thread uchar prev_vertical_y11_pixels[64];
    thread uchar prev_vertical_cb_pixels[64];
    thread uchar prev_vertical_cr_pixels[64];
    bool have_prev_vertical = false;

    uint mx = 0u;
    uint my = 0u;
    init_mcu_cursor(start_mcu, params.mcus_per_row, mx, my);
    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        const uint y_x = mx * 16u;
        const uint y_y = my * 16u;
        bool dc_only = false;

        if (!decode_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        idct_block(coeffs, dc_only, y00_pixels);

        if (!decode_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        idct_block(coeffs, dc_only, y10_pixels);

        if (!decode_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        idct_block(coeffs, dc_only, y01_pixels);

        if (!decode_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        idct_block(coeffs, dc_only, y11_pixels);

        if (!decode_block(br, entropy, entropy_end, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        idct_block(coeffs, dc_only, cb_pixels);

        if (!decode_block(br, entropy, entropy_end, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        idct_block(coeffs, dc_only, cr_pixels);

        const uint copy_width = jpeg_clamped_extent(y_x, 16u, params.width);
        const uint copy_height = jpeg_clamped_extent(y_y, 16u, params.height);
        const uint chroma_y_base = my * 8u;
        const uint copy_chroma_height = jpeg_clamped_extent(chroma_y_base, 8u, params.chroma_height);
        const bool starts_mid_row = mcu_index == start_mcu && mx > 0u;
        const bool has_left_mcu = mx > 0u;
        const bool has_right_mcu = mx + 1u < params.mcus_per_row;
        const bool has_top_mcu = my > 0u;
        const bool has_bottom_mcu = my + 1u < params.mcu_rows;
        const uint repair_record_index = params.tile_index * total_mcus + mcu_index;
        const uint boundary_meta_base = fast420_boundary_meta_base(repair_record_index);
        const uint vertical_meta_base = fast420_vertical_meta_base(repair_record_index);
        jpeg_decode_clear_meta_quad(boundary_meta, boundary_meta_base);
        jpeg_decode_clear_meta_quad(vertical_meta, vertical_meta_base);
        const uint boundary_sample_base = fast420_boundary_sample_base(repair_record_index);
        const uint vertical_sample_base = fast420_vertical_sample_base(repair_record_index);
        const uint local_sample_base = mx * 8u;
        if (has_left_mcu) {
            boundary_meta[boundary_meta_base] = y_x;
            boundary_meta[boundary_meta_base + 1u] = y_y;
            boundary_meta[boundary_meta_base + 2u] = 1u;
            for (uint by = 0u; by < copy_height; ++by) {
                boundary_samples[boundary_sample_base + by] = by < 8u
                    ? y00_pixels[by * 8u]
                    : y01_pixels[(by - 8u) * 8u];
            }
            for (uint cy = 0u; cy < copy_chroma_height; ++cy) {
                boundary_samples[boundary_sample_base + 16u + cy] = cb_pixels[cy * 8u];
                boundary_samples[boundary_sample_base + 24u + cy] = cr_pixels[cy * 8u];
            }
        }
        if (has_right_mcu) {
            boundary_meta[boundary_meta_base + 3u] = 1u;
            for (uint by = 0u; by < copy_height; ++by) {
                boundary_samples[boundary_sample_base + 32u + by] = by < 8u
                    ? y10_pixels[by * 8u + 7u]
                    : y11_pixels[(by - 8u) * 8u + 7u];
            }
            for (uint cy = 0u; cy < copy_chroma_height; ++cy) {
                boundary_samples[boundary_sample_base + 48u + cy] = cb_pixels[cy * 8u + 7u];
                boundary_samples[boundary_sample_base + 56u + cy] = cr_pixels[cy * 8u + 7u];
            }
        }
        if (has_top_mcu) {
            vertical_meta[vertical_meta_base] = y_x;
            vertical_meta[vertical_meta_base + 1u] = y_y;
            vertical_meta[vertical_meta_base + 2u] = 1u;
            for (uint bx = 0u; bx < copy_width; ++bx) {
                vertical_samples[vertical_sample_base + bx] = bx < 8u
                    ? y00_pixels[bx]
                    : y10_pixels[bx - 8u];
            }
            for (uint cx = 0u; cx < jpeg_clamped_extent(mx * 8u, 8u, params.chroma_width); ++cx) {
                vertical_samples[vertical_sample_base + 16u + cx] = cb_pixels[cx];
                vertical_samples[vertical_sample_base + 24u + cx] = cr_pixels[cx];
            }
        }
        if (has_bottom_mcu) {
            vertical_meta[vertical_meta_base + 3u] = 1u;
            for (uint bx = 0u; bx < copy_width; ++bx) {
                vertical_samples[vertical_sample_base + 32u + bx] = bx < 8u
                    ? y01_pixels[7u * 8u + bx]
                    : y11_pixels[7u * 8u + (bx - 8u)];
            }
            for (uint cx = 0u; cx < jpeg_clamped_extent(mx * 8u, 8u, params.chroma_width); ++cx) {
                vertical_samples[vertical_sample_base + 48u + cx] = cb_pixels[7u * 8u + cx];
                vertical_samples[vertical_sample_base + 56u + cx] = cr_pixels[7u * 8u + cx];
            }
        }
        if (have_prev_vertical && params.mcus_per_row == 1u && has_top_mcu) {
            const uint top_y = y_y;
            const uint bottom_y = y_y - 1u;
            for (uint bx = 0u; bx < copy_width; ++bx) {
                const uint out_x = y_x + bx;
                const uchar top_y_value = bx < 8u ? y00_pixels[bx] : y10_pixels[bx - 8u];
                const uchar bottom_y_value = bx < 8u
                    ? prev_vertical_y01_pixels[7u * 8u + bx]
                    : prev_vertical_y11_pixels[7u * 8u + (bx - 8u)];
                jpeg_write_ycbcr_rgba(
                    out,
                    uint2(out_x, bottom_y),
                    bottom_y_value,
                    h2v2_sample_thread_local(
                        cb_pixels,
                        prev_vertical_cb_pixels + 7u * 8u,
                        params.chroma_width,
                        out_x,
                        local_sample_base,
                        cb_pixels[0],
                        prev_vertical_cb_pixels[7u * 8u]
                    ),
                    h2v2_sample_thread_local(
                        cr_pixels,
                        prev_vertical_cr_pixels + 7u * 8u,
                        params.chroma_width,
                        out_x,
                        local_sample_base,
                        cr_pixels[0],
                        prev_vertical_cr_pixels[7u * 8u]
                    ),
                    params.alpha
                );
                jpeg_write_ycbcr_rgba(
                    out,
                    uint2(out_x, top_y),
                    top_y_value,
                    h2v2_sample_thread_local(
                        prev_vertical_cb_pixels + 7u * 8u,
                        cb_pixels,
                        params.chroma_width,
                        out_x,
                        local_sample_base,
                        prev_vertical_cb_pixels[7u * 8u],
                        cb_pixels[0]
                    ),
                    h2v2_sample_thread_local(
                        prev_vertical_cr_pixels + 7u * 8u,
                        cr_pixels,
                        params.chroma_width,
                        out_x,
                        local_sample_base,
                        prev_vertical_cr_pixels[7u * 8u],
                        cr_pixels[0]
                    ),
                    params.alpha
                );
            }
        }
        if (have_prev_horizontal && mx > 0u) {
            const uint left_x = y_x - 1u;
            for (uint by = 0u; by < copy_height; ++by) {
                const uint out_y = y_y + by;
                if (jpeg_skip_h2v2_boundary_repair_row(by, copy_height, params.mcu_rows, has_top_mcu, has_bottom_mcu)) {
                    continue;
                }
                const uint chroma_y = min(out_y / 2u, params.chroma_height - 1u);
                const uint near_y = (out_y & 1u) == 0u
                    ? (chroma_y == 0u ? 0u : chroma_y - 1u)
                    : min(chroma_y + 1u, params.chroma_height - 1u);
                const uint local_chroma_y = chroma_y - chroma_y_base;
                const uint local_near_y = near_y - chroma_y_base;
                const uint left_cb_sum = h2v2_weighted_sample_sum(prev_cb_pixels[local_chroma_y * 8u + 7u], prev_cb_pixels[local_near_y * 8u + 7u]);
                const uint left_cr_sum = h2v2_weighted_sample_sum(prev_cr_pixels[local_chroma_y * 8u + 7u], prev_cr_pixels[local_near_y * 8u + 7u]);
                const uint right_cb_sum = h2v2_weighted_sample_sum(cb_pixels[local_chroma_y * 8u], cb_pixels[local_near_y * 8u]);
                const uint right_cr_sum = h2v2_weighted_sample_sum(cr_pixels[local_chroma_y * 8u], cr_pixels[local_near_y * 8u]);
                const uchar left_y = by < 8u
                    ? prev_y10_pixels[by * 8u + 7u]
                    : prev_y11_pixels[(by - 8u) * 8u + 7u];
                const uchar right_y = by < 8u
                    ? y00_pixels[by * 8u]
                    : y01_pixels[(by - 8u) * 8u];
                jpeg_write_h2v2_boundary_pair(
                    out,
                    uint2(left_x, out_y),
                    uint2(y_x, out_y),
                    left_y,
                    right_y,
                    left_cb_sum,
                    left_cr_sum,
                    right_cb_sum,
                    right_cr_sum,
                    params.alpha
                );
            }
        }

        const uint last_chroma_x = params.chroma_width * 2u - 1u;
        for (uint by = 0u; by < copy_height; ++by) {
            const uint out_y = y_y + by;
            if (has_top_mcu && by == 0u) {
                continue;
            }
            if (has_bottom_mcu && by + 1u == copy_height) {
                continue;
            }
            const uint chroma_y = min(out_y / 2u, params.chroma_height - 1u);
            const uint near_y = (out_y & 1u) == 0u
                ? (chroma_y == 0u ? 0u : chroma_y - 1u)
                : min(chroma_y + 1u, params.chroma_height - 1u);
            const uint local_chroma_y = chroma_y - chroma_y_base;
            const uint local_near_y = near_y - chroma_y_base;
            thread const uchar *curr_cb = cb_pixels + local_chroma_y * 8u;
            thread const uchar *curr_cr = cr_pixels + local_chroma_y * 8u;
            thread const uchar *near_cb = cb_pixels + local_near_y * 8u;
            thread const uchar *near_cr = cr_pixels + local_near_y * 8u;
            const uchar left_curr_cb = mx == 0u || starts_mid_row
                ? curr_cb[0]
                : prev_cb_pixels[local_chroma_y * 8u + 7u];
            const uchar left_curr_cr = mx == 0u || starts_mid_row
                ? curr_cr[0]
                : prev_cr_pixels[local_chroma_y * 8u + 7u];
            const uchar left_near_cb = mx == 0u || starts_mid_row
                ? near_cb[0]
                : prev_cb_pixels[local_near_y * 8u + 7u];
            const uchar left_near_cr = mx == 0u || starts_mid_row
                ? near_cr[0]
                : prev_cr_pixels[local_near_y * 8u + 7u];
            for (uint bx = 0u; bx < copy_width; ++bx) {
                const uint out_x = y_x + bx;
                if (mx > 0u && bx == 0u) {
                    continue;
                }
                if (has_right_mcu && bx == 15u && out_x != last_chroma_x) {
                    continue;
                }
                uchar y_value;
                if (by < 8u) {
                    y_value = bx < 8u
                        ? y00_pixels[by * 8u + bx]
                        : y10_pixels[by * 8u + (bx - 8u)];
                } else {
                    y_value = bx < 8u
                        ? y01_pixels[(by - 8u) * 8u + bx]
                        : y11_pixels[(by - 8u) * 8u + (bx - 8u)];
                }
                const uchar cb_value = h2v2_sample_thread_local(
                    near_cb,
                    curr_cb,
                    params.chroma_width,
                    out_x,
                    local_sample_base,
                    left_near_cb,
                    left_curr_cb
                );
                const uchar cr_value = h2v2_sample_thread_local(
                    near_cr,
                    curr_cr,
                    params.chroma_width,
                    out_x,
                    local_sample_base,
                    left_near_cr,
                    left_curr_cr
                );
                jpeg_write_ycbcr_rgba(out, uint2(out_x, out_y), y_value, cb_value, cr_value, params.alpha);
            }
        }

        for (uint i = 0u; i < 64u; ++i) {
            prev_y10_pixels[i] = y10_pixels[i];
            prev_y11_pixels[i] = y11_pixels[i];
            prev_cb_pixels[i] = cb_pixels[i];
            prev_cr_pixels[i] = cr_pixels[i];
            prev_vertical_y01_pixels[i] = y01_pixels[i];
            prev_vertical_y11_pixels[i] = y11_pixels[i];
            prev_vertical_cb_pixels[i] = cb_pixels[i];
            prev_vertical_cr_pixels[i] = cr_pixels[i];
        }
        have_prev_horizontal = true;
        have_prev_vertical = true;
        advance_mcu_cursor(mx, my, params.mcus_per_row);
    }
}

kernel void jpeg_resolve_fast420_rgba_texture_boundaries(
    device const uint *boundary_meta [[buffer(0)]],
    device const uchar *boundary_samples [[buffer(1)]],
    constant JpegFast420TextureBatchParams &params [[buffer(2)]],
    texture2d<float, access::write> out [[texture(0)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_mcus = params.mcus_per_row * params.mcu_rows;
    if (gid >= total_mcus || params.mcus_per_row <= 1u) {
        return;
    }
    const uint mx = gid % params.mcus_per_row;
    if (mx == 0u) {
        return;
    }

    const uint record_index = params.tile_index * total_mcus + gid;
    const uint previous_record_index = record_index - 1u;
    const uint meta_base = fast420_boundary_meta_base(record_index);
    const uint previous_meta_base = fast420_boundary_meta_base(previous_record_index);
    if (boundary_meta[meta_base + 2u] == 0u || boundary_meta[previous_meta_base + 3u] == 0u) {
        return;
    }

    const uint x = boundary_meta[meta_base];
    const uint y = boundary_meta[meta_base + 1u];
    if (x == 0u || x >= params.width || y >= params.height) {
        return;
    }

    const uint sample_base = fast420_boundary_sample_base(record_index);
    const uint previous_sample_base = fast420_boundary_sample_base(previous_record_index);
    const uint copy_height = min(16u, params.height - y);
    const uint chroma_y_base = y / 2u;
    const bool has_top_row = y > 0u;
    const bool has_bottom_row = y + copy_height < params.height;
    for (uint by = 0u; by < copy_height; ++by) {
        const uint out_y = y + by;
        if (jpeg_skip_h2v2_boundary_repair_row(by, copy_height, params.mcu_rows, has_top_row, has_bottom_row)) {
            continue;
        }
        const uint chroma_y = min(out_y / 2u, params.chroma_height - 1u);
        const uint near_y = (out_y & 1u) == 0u
            ? (chroma_y == 0u ? 0u : chroma_y - 1u)
            : min(chroma_y + 1u, params.chroma_height - 1u);
        const uint local_chroma_y = chroma_y - chroma_y_base;
        const uint local_near_y = near_y - chroma_y_base;
        const uint left_cb_sum = h2v2_weighted_sample_sum(boundary_samples[previous_sample_base + 48u + local_chroma_y], boundary_samples[previous_sample_base + 48u + local_near_y]);
        const uint left_cr_sum = h2v2_weighted_sample_sum(boundary_samples[previous_sample_base + 56u + local_chroma_y], boundary_samples[previous_sample_base + 56u + local_near_y]);
        const uint right_cb_sum = h2v2_weighted_sample_sum(boundary_samples[sample_base + 16u + local_chroma_y], boundary_samples[sample_base + 16u + local_near_y]);
        const uint right_cr_sum = h2v2_weighted_sample_sum(boundary_samples[sample_base + 24u + local_chroma_y], boundary_samples[sample_base + 24u + local_near_y]);
        const uchar left_y = boundary_samples[previous_sample_base + 32u + by];
        const uchar right_y = boundary_samples[sample_base + by];
        jpeg_write_h2v2_boundary_pair(
            out,
            uint2(x - 1u, out_y),
            uint2(x, out_y),
            left_y,
            right_y,
            left_cb_sum,
            left_cr_sum,
            right_cb_sum,
            right_cr_sum,
            params.alpha
        );
    }
}

kernel void jpeg_resolve_fast420_rgba_texture_vertical_boundaries(
    device const uint *vertical_meta [[buffer(0)]],
    device const uchar *vertical_samples [[buffer(1)]],
    constant JpegFast420TextureBatchParams &params [[buffer(2)]],
    texture2d<float, access::write> out [[texture(0)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total_mcus = params.mcus_per_row * params.mcu_rows;
    if (gid >= total_mcus || params.mcu_rows <= 1u) {
        return;
    }
    const uint my = gid / params.mcus_per_row;
    if (my == 0u) {
        return;
    }

    const uint record_index = params.tile_index * total_mcus + gid;
    const uint previous_record_index = record_index - params.mcus_per_row;
    const uint meta_base = fast420_vertical_meta_base(record_index);
    const uint previous_meta_base = fast420_vertical_meta_base(previous_record_index);
    if (vertical_meta[meta_base + 2u] == 0u || vertical_meta[previous_meta_base + 3u] == 0u) {
        return;
    }

    const uint x = vertical_meta[meta_base];
    const uint y = vertical_meta[meta_base + 1u];
    if (x >= params.width || y == 0u || y >= params.height) {
        return;
    }

    const uint sample_base = fast420_vertical_sample_base(record_index);
    const uint previous_sample_base = fast420_vertical_sample_base(previous_record_index);
    const uint copy_width = min(16u, params.width - x);
    const uint local_sample_base = x / 2u;
    device const uchar *top_cb = vertical_samples + sample_base + 16u;
    device const uchar *top_cr = vertical_samples + sample_base + 24u;
    device const uchar *bottom_cb = vertical_samples + previous_sample_base + 48u;
    device const uchar *bottom_cr = vertical_samples + previous_sample_base + 56u;
    const bool has_left_column = params.mcus_per_row > 1u && x > 0u;
    const bool has_right_column = params.mcus_per_row > 1u && x + copy_width < params.width;
    for (uint bx = 0u; bx < copy_width; ++bx) {
        const uint out_x = x + bx;
        if (has_left_column && bx == 0u) {
            continue;
        }
        if (has_right_column && bx + 1u == copy_width) {
            continue;
        }
        const uchar bottom_y = vertical_samples[previous_sample_base + 32u + bx];
        const uchar top_y = vertical_samples[sample_base + bx];
        jpeg_write_ycbcr_rgba(
            out,
            uint2(out_x, y - 1u),
            bottom_y,
            h2v2_sample_device_local(top_cb, bottom_cb, params.chroma_width, out_x, local_sample_base),
            h2v2_sample_device_local(top_cr, bottom_cr, params.chroma_width, out_x, local_sample_base),
            params.alpha
        );
        jpeg_write_ycbcr_rgba(
            out,
            uint2(out_x, y),
            top_y,
            h2v2_sample_device_local(bottom_cb, top_cb, params.chroma_width, out_x, local_sample_base),
            h2v2_sample_device_local(bottom_cr, top_cr, params.chroma_width, out_x, local_sample_base),
            params.alpha
        );
    }
}

kernel void jpeg_resolve_fast420_rgba_texture_corners(
    device const uint *boundary_meta [[buffer(0)]],
    device const uint *vertical_meta [[buffer(1)]],
    device const uchar *vertical_samples [[buffer(2)]],
    constant JpegFast420TextureBatchParams &params [[buffer(3)]],
    texture2d<float, access::write> out [[texture(0)]],
    uint gid [[thread_position_in_grid]]
) {
    if (params.mcus_per_row <= 1u || params.mcu_rows <= 1u) {
        return;
    }
    const uint total_mcus = params.mcus_per_row * params.mcu_rows;
    if (gid >= total_mcus) {
        return;
    }
    const uint mx = gid % params.mcus_per_row;
    const uint my = gid / params.mcus_per_row;
    if (mx == 0u || my == 0u) {
        return;
    }

    const uint br_record = params.tile_index * total_mcus + gid;
    const uint bl_record = br_record - 1u;
    const uint tr_record = br_record - params.mcus_per_row;
    const uint tl_record = tr_record - 1u;
    const uint br_boundary_meta_base = fast420_boundary_meta_base(br_record);
    const uint bl_boundary_meta_base = fast420_boundary_meta_base(bl_record);
    const uint tr_boundary_meta_base = fast420_boundary_meta_base(tr_record);
    const uint tl_boundary_meta_base = fast420_boundary_meta_base(tl_record);
    const uint br_vertical_meta_base = fast420_vertical_meta_base(br_record);
    const uint bl_vertical_meta_base = fast420_vertical_meta_base(bl_record);
    const uint tr_vertical_meta_base = fast420_vertical_meta_base(tr_record);
    const uint tl_vertical_meta_base = fast420_vertical_meta_base(tl_record);
    if (
        boundary_meta[br_boundary_meta_base + 2u] == 0u ||
        boundary_meta[bl_boundary_meta_base + 3u] == 0u ||
        boundary_meta[tr_boundary_meta_base + 2u] == 0u ||
        boundary_meta[tl_boundary_meta_base + 3u] == 0u ||
        vertical_meta[br_vertical_meta_base + 2u] == 0u ||
        vertical_meta[bl_vertical_meta_base + 2u] == 0u ||
        vertical_meta[tr_vertical_meta_base + 3u] == 0u ||
        vertical_meta[tl_vertical_meta_base + 3u] == 0u
    ) {
        return;
    }

    const uint x = vertical_meta[br_vertical_meta_base];
    const uint y = vertical_meta[br_vertical_meta_base + 1u];
    if (x == 0u || y == 0u || x >= params.width || y >= params.height) {
        return;
    }

    const uint br_sample_base = fast420_vertical_sample_base(br_record);
    const uint bl_sample_base = fast420_vertical_sample_base(bl_record);
    const uint tr_sample_base = fast420_vertical_sample_base(tr_record);
    const uint tl_sample_base = fast420_vertical_sample_base(tl_record);

    const uchar tl_y = vertical_samples[tl_sample_base + 32u + 15u];
    const uchar tr_y = vertical_samples[tr_sample_base + 32u];
    const uchar bl_y = vertical_samples[bl_sample_base + 15u];
    const uchar br_y = vertical_samples[br_sample_base];

    const uchar tl_cb = vertical_samples[tl_sample_base + 48u + 7u];
    const uchar tr_cb = vertical_samples[tr_sample_base + 48u];
    const uchar bl_cb = vertical_samples[bl_sample_base + 16u + 7u];
    const uchar br_cb = vertical_samples[br_sample_base + 16u];
    const uchar tl_cr = vertical_samples[tl_sample_base + 56u + 7u];
    const uchar tr_cr = vertical_samples[tr_sample_base + 56u];
    const uchar bl_cr = vertical_samples[bl_sample_base + 24u + 7u];
    const uchar br_cr = vertical_samples[br_sample_base + 24u];

    jpeg_write_ycbcr_rgba(
        out,
        uint2(x - 1u, y - 1u),
        tl_y,
        h2v2_corner_sample(tl_cb, tr_cb, bl_cb, br_cb, false, false),
        h2v2_corner_sample(tl_cr, tr_cr, bl_cr, br_cr, false, false),
        params.alpha
    );
    jpeg_write_ycbcr_rgba(
        out,
        uint2(x, y - 1u),
        tr_y,
        h2v2_corner_sample(tl_cb, tr_cb, bl_cb, br_cb, false, true),
        h2v2_corner_sample(tl_cr, tr_cr, bl_cr, br_cr, false, true),
        params.alpha
    );
    jpeg_write_ycbcr_rgba(
        out,
        uint2(x - 1u, y),
        bl_y,
        h2v2_corner_sample(tl_cb, tr_cb, bl_cb, br_cb, true, false),
        h2v2_corner_sample(tl_cr, tr_cr, bl_cr, br_cr, true, false),
        params.alpha
    );
    jpeg_write_ycbcr_rgba(
        out,
        uint2(x, y),
        br_y,
        h2v2_corner_sample(tl_cb, tr_cb, bl_cb, br_cb, true, true),
        h2v2_corner_sample(tl_cr, tr_cr, bl_cr, br_cr, true, true),
        params.alpha
    );
}

inline uint fast420_total_mcus(constant JpegFast420BatchParams &params) {
    return params.mcus_per_row * params.mcu_rows;
}

inline uint fast420_y_blocks_per_tile(constant JpegFast420BatchParams &params) {
    return fast420_total_mcus(params) * 4u;
}

inline uint fast420_blocks_per_tile(constant JpegFast420BatchParams &params) {
    return fast420_total_mcus(params) * 6u;
}

inline void store_coeff_block(
    device short *coeff_blocks,
    device uchar *dc_only_flags,
    uint block_index,
    thread const short coeffs[64],
    bool dc_only
) {
    device short *dst = coeff_blocks + block_index * 64u;
    for (uint i = 0u; i < 64u; ++i) {
        dst[i] = coeffs[i];
    }
    dc_only_flags[block_index] = dc_only ? uchar(1) : uchar(0);
}

inline void idct_deposit_coeff_block(
    device const short *coeff_blocks,
    device const uchar *dc_only_flags,
    uint block_index,
    device uchar *plane,
    uint stride,
    uint width,
    uint height,
    uint x,
    uint y
) {
    thread uchar pixels[64];
    device const short *src = coeff_blocks + block_index * 64u;
    if (dc_only_flags[block_index] != 0u) {
        idct_islow_dc_only(src[0], pixels);
    } else {
        thread short coeffs[64];
        for (uint i = 0u; i < 64u; ++i) {
            coeffs[i] = src[i];
        }
        idct_islow(coeffs, pixels);
    }
    deposit_block(plane, stride, width, height, x, y, pixels);
}

kernel void jpeg_decode_fast420_batch_coeffs(
    device const uchar *entropy [[buffer(0)]],
    device short *coeff_blocks [[buffer(1)]],
    device uchar *dc_only_flags [[buffer(2)]],
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
    const uint total_mcus = fast420_total_mcus(params);
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

    const uint blocks_per_tile = fast420_blocks_per_tile(params);
    const uint y_blocks_per_tile = fast420_y_blocks_per_tile(params);
    const uint tile_block_base = tile_index * blocks_per_tile;
    const uint cb_block_base = tile_block_base + y_blocks_per_tile;
    const uint cr_block_base = cb_block_base + total_mcus;

    thread short coeffs[64];
    for (uint mcu_index = start_mcu; mcu_index < end_mcu; ++mcu_index) {
        bool dc_only = false;
        const uint y_block_base = tile_block_base + mcu_index * 4u;

        if (!decode_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        store_coeff_block(coeff_blocks, dc_only_flags, y_block_base, coeffs, dc_only);

        if (!decode_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        store_coeff_block(coeff_blocks, dc_only_flags, y_block_base + 1u, coeffs, dc_only);

        if (!decode_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        store_coeff_block(coeff_blocks, dc_only_flags, y_block_base + 2u, coeffs, dc_only);

        if (!decode_block(br, entropy, entropy_end, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        store_coeff_block(coeff_blocks, dc_only_flags, y_block_base + 3u, coeffs, dc_only);

        if (!decode_block(br, entropy, entropy_end, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        store_coeff_block(coeff_blocks, dc_only_flags, cb_block_base + mcu_index, coeffs, dc_only);

        if (!decode_block(br, entropy, entropy_end, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, coeffs, dc_only)) {
            return;
        }
        store_coeff_block(coeff_blocks, dc_only_flags, cr_block_base + mcu_index, coeffs, dc_only);
    }
}

kernel void jpeg_idct_deposit_fast420_batch(
    device const short *coeff_blocks [[buffer(0)]],
    device const uchar *dc_only_flags [[buffer(1)]],
    device uchar *y_plane [[buffer(2)]],
    device uchar *cb_plane [[buffer(3)]],
    device uchar *cr_plane [[buffer(4)]],
    constant JpegFast420BatchParams &params [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.mcus_per_row || gid.y >= params.mcu_rows) {
        return;
    }

    const uint tile_index = gid.z / 6u;
    const uint component = gid.z - tile_index * 6u;
    if (tile_index >= params.tile_count || component >= 6u) {
        return;
    }

    const uint total_mcus = fast420_total_mcus(params);
    const uint y_blocks_per_tile = fast420_y_blocks_per_tile(params);
    const uint blocks_per_tile = fast420_blocks_per_tile(params);
    const uint mcu_index = gid.y * params.mcus_per_row + gid.x;
    const uint tile_block_base = tile_index * blocks_per_tile;
    const uint y_plane_base = tile_index * params.width * params.height;
    const uint chroma_plane_base = tile_index * params.chroma_width * params.chroma_height;

    if (component < 4u) {
        const uint block_index = tile_block_base + mcu_index * 4u + component;
        const uint x = gid.x * 16u + (component & 1u) * 8u;
        const uint y = gid.y * 16u + (component >> 1u) * 8u;
        idct_deposit_coeff_block(
            coeff_blocks,
            dc_only_flags,
            block_index,
            y_plane + y_plane_base,
            params.width,
            params.width,
            params.height,
            x,
            y
        );
        return;
    }

    const uint x = gid.x * 8u;
    const uint y = gid.y * 8u;
    if (component == 4u) {
        idct_deposit_coeff_block(
            coeff_blocks,
            dc_only_flags,
            tile_block_base + y_blocks_per_tile + mcu_index,
            cb_plane + chroma_plane_base,
            params.chroma_width,
            params.chroma_width,
            params.chroma_height,
            x,
            y
        );
    } else {
        idct_deposit_coeff_block(
            coeff_blocks,
            dc_only_flags,
            tile_block_base + y_blocks_per_tile + total_mcus + mcu_index,
            cr_plane + chroma_plane_base,
            params.chroma_width,
            params.chroma_width,
            params.chroma_height,
            x,
            y
        );
    }
}
