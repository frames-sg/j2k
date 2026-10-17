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

kernel void jpeg_encode_baseline_entropy_segments_batch(
    device const int *coefficients [[buffer(0)]],
    device uchar *segments [[buffer(1)]],
    device JpegBaselineEncodeStatus *segment_status [[buffer(2)]],
    constant JpegBaselineEncodeParams *params [[buffer(3)]],
    constant JpegBaselineEncodeHuffmanTable &dc_luma [[buffer(4)]],
    constant JpegBaselineEncodeHuffmanTable &ac_luma [[buffer(5)]],
    constant JpegBaselineEncodeHuffmanTable &dc_chroma [[buffer(6)]],
    constant JpegBaselineEncodeHuffmanTable &ac_chroma [[buffer(7)]],
    constant uint &tile_count [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    uint tile_index = 0u;
    uint local_mcu = gid;
    uint coefficient_block_base = 0u;
    for (; tile_index < tile_count; ++tile_index) {
        constant JpegBaselineEncodeParams &candidate = params[tile_index];
        const uint tile_mcus = candidate.mcus_per_row * candidate.mcu_rows;
        const uint tile_blocks = tile_mcus * jpeg_encode_blocks_per_mcu(candidate);
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
    device JpegBaselineEncodeStatus *status = segment_status + gid;
    status->code = JPEG_BASELINE_ENCODE_STATUS_OK;
    status->entropy_len = 0u;
    status->detail = 0u;
    status->reserved = 0u;
    const uint blocks_per_mcu = jpeg_encode_blocks_per_mcu(tile_params);
    if (blocks_per_mcu == 0u || tile_params.components == 0u || tile_params.components > 3u) {
        status->code = JPEG_BASELINE_ENCODE_STATUS_INVALID_PARAMS;
        return;
    }

    const bool starts_predictor = local_mcu == 0u || (
        tile_params.restart_interval_mcus != 0u
        && local_mcu % tile_params.restart_interval_mcus == 0u
    );
    thread int prev_dc[3] = {0, 0, 0};
    if (!starts_predictor) {
        device const int *prior = coefficients
            + (coefficient_block_base + (local_mcu - 1u) * blocks_per_mcu) * 64u;
        uint component_block = 0u;
        for (uint component = 0u; component < tile_params.components; ++component) {
            const uint component_blocks = component_h(tile_params, component)
                * component_v(tile_params, component);
            if (component_blocks == 0u) {
                status->code = JPEG_BASELINE_ENCODE_STATUS_INVALID_PARAMS;
                return;
            }
            prev_dc[component] = prior[(component_block + component_blocks - 1u) * 64u];
            component_block += component_blocks;
        }
    }

    device uchar *segment = segments + gid * JPEG_BASELINE_ENCODE_SEGMENT_BYTES;
    thread JpegBaselineRawBitWriter writer;
    writer.bits = 0u;
    writer.overflow = false;
    device const int *source = coefficients
        + (coefficient_block_base + local_mcu * blocks_per_mcu) * 64u;
    uint block_index = 0u;
    for (uint component = 0u; component < tile_params.components; ++component) {
        const uint component_blocks = component_h(tile_params, component)
            * component_v(tile_params, component);
        if (component_blocks == 0u) {
            status->code = JPEG_BASELINE_ENCODE_STATUS_INVALID_PARAMS;
            return;
        }
        for (uint block = 0u; block < component_blocks; ++block) {
            const bool ok = component == 0u
                ? jpeg_encode_raw_block(
                    source + block_index * 64u,
                    prev_dc[component],
                    dc_luma,
                    ac_luma,
                    segment,
                    JPEG_BASELINE_ENCODE_SEGMENT_BYTES,
                    writer,
                    status
                )
                : jpeg_encode_raw_block(
                    source + block_index * 64u,
                    prev_dc[component],
                    dc_chroma,
                    ac_chroma,
                    segment,
                    JPEG_BASELINE_ENCODE_SEGMENT_BYTES,
                    writer,
                    status
                );
            if (!ok) {
                return;
            }
            block_index += 1u;
        }
    }
    status->entropy_len = writer.bits;
}

kernel void jpeg_encode_baseline_prepare_segments_batch(
    device const uchar *segments [[buffer(0)]],
    device JpegBaselineEncodeStatus *segment_status [[buffer(1)]],
    device uchar *entropy [[buffer(2)]],
    device JpegBaselineEncodeStatus *status [[buffer(3)]],
    constant JpegBaselineEncodeParams *params [[buffer(4)]],
    constant uint &tile_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= tile_count) {
        return;
    }
    uint segment_base = 0u;
    for (uint tile_index = 0u; tile_index < gid; ++tile_index) {
        segment_base += params[tile_index].mcus_per_row * params[tile_index].mcu_rows;
    }

    constant JpegBaselineEncodeParams &tile_params = params[gid];
    device JpegBaselineEncodeStatus *tile_status = status + gid;
    tile_status->code = JPEG_BASELINE_ENCODE_STATUS_OK;
    tile_status->entropy_len = 0u;
    tile_status->detail = 0u;
    tile_status->reserved = 0u;
    const uint tile_mcus = tile_params.mcus_per_row * tile_params.mcu_rows;
    if (tile_mcus == 0u) {
        tile_status->code = JPEG_BASELINE_ENCODE_STATUS_INVALID_PARAMS;
        return;
    }

    if (tile_params.restart_interval_mcus == 0u) {
        uint bit_offset = 0u;
        for (uint local_mcu = 0u; local_mcu < tile_mcus; ++local_mcu) {
            device JpegBaselineEncodeStatus *source_status = segment_status
                + segment_base + local_mcu;
            if (source_status->code != JPEG_BASELINE_ENCODE_STATUS_OK) {
                *tile_status = *source_status;
                return;
            }
            if (source_status->entropy_len > JPEG_BASELINE_ENCODE_SEGMENT_BYTES * 8u) {
                tile_status->code = JPEG_BASELINE_ENCODE_STATUS_OVERFLOW;
                return;
            }
            source_status->detail = bit_offset;
            if (source_status->entropy_len > 0xffffffffu - bit_offset) {
                tile_status->code = JPEG_BASELINE_ENCODE_STATUS_OVERFLOW;
                return;
            }
            bit_offset += source_status->entropy_len;
        }
        tile_status->reserved = bit_offset;
        return;
    }

    device uchar *tile_entropy = entropy + tile_params.entropy_offset_bytes;
    thread JpegBaselineBitWriter writer;
    writer.pos = 0u;
    writer.current = 0;
    writer.used = 0u;
    writer.overflow = false;
    uint rst = 0u;
    for (uint local_mcu = 0u; local_mcu < tile_mcus; ++local_mcu) {
        if (
            local_mcu > 0u
            && tile_params.restart_interval_mcus != 0u
            && local_mcu % tile_params.restart_interval_mcus == 0u
        ) {
            jpeg_encode_push_restart_marker(
                tile_entropy,
                tile_params.entropy_capacity,
                writer,
                rst
            );
            if (writer.overflow) {
                tile_status->code = JPEG_BASELINE_ENCODE_STATUS_OVERFLOW;
                return;
            }
            rst = (rst + 1u) & 7u;
        }

        const uint segment_index = segment_base + local_mcu;
        const JpegBaselineEncodeStatus source_status = segment_status[segment_index];
        if (source_status.code != JPEG_BASELINE_ENCODE_STATUS_OK) {
            *tile_status = source_status;
            return;
        }
        const uint bit_count = source_status.entropy_len;
        if (bit_count > JPEG_BASELINE_ENCODE_SEGMENT_BYTES * 8u) {
            tile_status->code = JPEG_BASELINE_ENCODE_STATUS_OVERFLOW;
            return;
        }
        device const uchar *segment = segments
            + segment_index * JPEG_BASELINE_ENCODE_SEGMENT_BYTES;
        uint bit_offset = 0u;
        while (bit_offset < bit_count) {
            const uint len = min(8u, bit_count - bit_offset);
            ushort bits = ushort(segment[bit_offset >> 3u]);
            if (len < 8u) {
                bits >>= 8u - len;
            }
            jpeg_encode_append_raw_bits(
                tile_entropy,
                tile_params.entropy_capacity,
                writer,
                bits,
                len
            );
            if (writer.overflow) {
                tile_status->code = JPEG_BASELINE_ENCODE_STATUS_OVERFLOW;
                return;
            }
            bit_offset += len;
        }
    }

    jpeg_encode_align_with_ones(tile_entropy, tile_params.entropy_capacity, writer);
    if (writer.overflow) {
        tile_status->code = JPEG_BASELINE_ENCODE_STATUS_OVERFLOW;
        return;
    }
    tile_status->entropy_len = writer.pos;
}

inline uchar jpeg_encode_read_segment_bits(
    device const uchar *segment,
    uint bit_offset,
    uint len
) {
    const uint byte_offset = bit_offset >> 3u;
    const uint intra_byte = bit_offset & 7u;
    ushort pair = ushort(segment[byte_offset]) << 8u;
    if (intra_byte + len > 8u) {
        pair |= ushort(segment[byte_offset + 1u]);
    }
    const uint shift = 16u - intra_byte - len;
    const ushort mask = len == 8u ? 0xffu : ushort((1u << len) - 1u);
    return uchar((pair >> shift) & mask);
}

inline uchar jpeg_encode_materialize_raw_byte(
    device const uchar *segments,
    device const JpegBaselineEncodeStatus *segment_status,
    uint segment_index,
    uint segment_end,
    uint global_bit,
    uint total_bits
) {
    uint remaining = min(8u, total_bits - global_bit);
    uint produced = 0u;
    uchar byte = 0u;
    uint source_index = segment_index;
    uint source_bit = global_bit - segment_status[source_index].detail;
    while (remaining > 0u && source_index < segment_end) {
        const uint source_bits = segment_status[source_index].entropy_len;
        if (source_bit >= source_bits) {
            source_index += 1u;
            source_bit = 0u;
            continue;
        }
        const uint take = min(remaining, source_bits - source_bit);
        device const uchar *source = segments
            + source_index * JPEG_BASELINE_ENCODE_SEGMENT_BYTES;
        const uchar chunk = jpeg_encode_read_segment_bits(source, source_bit, take);
        byte = uchar((ushort(byte) << take) | ushort(chunk));
        produced += take;
        remaining -= take;
        source_index += 1u;
        source_bit = 0u;
    }
    if (produced < 8u) {
        const uint pad = 8u - produced;
        byte = uchar((ushort(byte) << pad) | ushort((1u << pad) - 1u));
    }
    return byte;
}

inline uint jpeg_encode_bytes_for_bits(uint bits) {
    return (bits >> 3u) + ((bits & 7u) != 0u ? 1u : 0u);
}

kernel void jpeg_encode_baseline_count_stuffed_segments_batch(
    device const uchar *segments [[buffer(0)]],
    device JpegBaselineEncodeStatus *segment_status [[buffer(1)]],
    device const JpegBaselineEncodeStatus *status [[buffer(2)]],
    constant JpegBaselineEncodeParams *params [[buffer(3)]],
    constant uint &tile_count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint tile_index = 0u;
    uint local_mcu = gid;
    uint segment_base = 0u;
    for (; tile_index < tile_count; ++tile_index) {
        const uint tile_mcus = params[tile_index].mcus_per_row * params[tile_index].mcu_rows;
        if (local_mcu < tile_mcus) {
            break;
        }
        local_mcu -= tile_mcus;
        segment_base += tile_mcus;
    }
    if (
        tile_index >= tile_count
        || params[tile_index].restart_interval_mcus != 0u
        || status[tile_index].code != JPEG_BASELINE_ENCODE_STATUS_OK
    ) {
        return;
    }

    const uint tile_mcus = params[tile_index].mcus_per_row * params[tile_index].mcu_rows;
    const uint segment_index = segment_base + local_mcu;
    device JpegBaselineEncodeStatus *source_status = segment_status + segment_index;
    const uint bit_start = source_status->detail;
    const uint bit_end = bit_start + source_status->entropy_len;
    const uint first_byte = jpeg_encode_bytes_for_bits(bit_start);
    const uint end_byte = jpeg_encode_bytes_for_bits(bit_end);
    const uint total_bits = status[tile_index].reserved;
    uint stuffed = 0u;
    for (uint byte_index = first_byte; byte_index < end_byte; ++byte_index) {
        const uchar byte = jpeg_encode_materialize_raw_byte(
            segments,
            segment_status,
            segment_index,
            segment_base + tile_mcus,
            byte_index * 8u,
            total_bits
        );
        stuffed += byte == 0xffu ? 1u : 0u;
    }
    source_status->reserved = stuffed;
}

kernel void jpeg_encode_baseline_offset_stuffed_segments_batch(
    device JpegBaselineEncodeStatus *segment_status [[buffer(0)]],
    device JpegBaselineEncodeStatus *status [[buffer(1)]],
    constant JpegBaselineEncodeParams *params [[buffer(2)]],
    constant uint &tile_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= tile_count || params[gid].restart_interval_mcus != 0u) {
        return;
    }
    device JpegBaselineEncodeStatus *tile_status = status + gid;
    if (tile_status->code != JPEG_BASELINE_ENCODE_STATUS_OK) {
        return;
    }

    uint segment_base = 0u;
    for (uint tile_index = 0u; tile_index < gid; ++tile_index) {
        segment_base += params[tile_index].mcus_per_row * params[tile_index].mcu_rows;
    }
    const uint tile_mcus = params[gid].mcus_per_row * params[gid].mcu_rows;
    uint output_offset = 0u;
    for (uint local_mcu = 0u; local_mcu < tile_mcus; ++local_mcu) {
        device JpegBaselineEncodeStatus *source_status = segment_status
            + segment_base + local_mcu;
        const uint bit_start = source_status->detail;
        const uint bit_end = bit_start + source_status->entropy_len;
        const uint raw_bytes = jpeg_encode_bytes_for_bits(bit_end)
            - jpeg_encode_bytes_for_bits(bit_start);
        const uint stuffed = source_status->reserved;
        source_status->reserved = output_offset;
        if (raw_bytes > 0xffffffffu - stuffed || raw_bytes + stuffed > 0xffffffffu - output_offset) {
            tile_status->code = JPEG_BASELINE_ENCODE_STATUS_OVERFLOW;
            return;
        }
        output_offset += raw_bytes + stuffed;
    }
    if (output_offset > params[gid].entropy_capacity) {
        tile_status->code = JPEG_BASELINE_ENCODE_STATUS_OVERFLOW;
        return;
    }
    tile_status->entropy_len = output_offset;
}

kernel void jpeg_encode_baseline_scatter_stuffed_segments_batch(
    device const uchar *segments [[buffer(0)]],
    device const JpegBaselineEncodeStatus *segment_status [[buffer(1)]],
    device uchar *entropy [[buffer(2)]],
    device const JpegBaselineEncodeStatus *status [[buffer(3)]],
    constant JpegBaselineEncodeParams *params [[buffer(4)]],
    constant uint &tile_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint tile_index = 0u;
    uint local_mcu = gid;
    uint segment_base = 0u;
    for (; tile_index < tile_count; ++tile_index) {
        const uint tile_mcus = params[tile_index].mcus_per_row * params[tile_index].mcu_rows;
        if (local_mcu < tile_mcus) {
            break;
        }
        local_mcu -= tile_mcus;
        segment_base += tile_mcus;
    }
    if (
        tile_index >= tile_count
        || params[tile_index].restart_interval_mcus != 0u
        || status[tile_index].code != JPEG_BASELINE_ENCODE_STATUS_OK
    ) {
        return;
    }

    const uint tile_mcus = params[tile_index].mcus_per_row * params[tile_index].mcu_rows;
    const uint segment_index = segment_base + local_mcu;
    const JpegBaselineEncodeStatus source_status = segment_status[segment_index];
    const uint bit_start = source_status.detail;
    const uint bit_end = bit_start + source_status.entropy_len;
    const uint first_byte = jpeg_encode_bytes_for_bits(bit_start);
    const uint end_byte = jpeg_encode_bytes_for_bits(bit_end);
    const uint total_bits = status[tile_index].reserved;
    uint output_offset = params[tile_index].entropy_offset_bytes + source_status.reserved;
    for (uint byte_index = first_byte; byte_index < end_byte; ++byte_index) {
        const uchar byte = jpeg_encode_materialize_raw_byte(
            segments,
            segment_status,
            segment_index,
            segment_base + tile_mcus,
            byte_index * 8u,
            total_bits
        );
        entropy[output_offset++] = byte;
        if (byte == 0xffu) {
            entropy[output_offset++] = 0u;
        }
    }
}
