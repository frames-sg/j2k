kernel void jpeg_encode_baseline_entropy(
    device const uchar *input [[buffer(0)]],
    device uchar *entropy [[buffer(1)]],
    device JpegBaselineEncodeStatus *status [[buffer(2)]],
    constant JpegBaselineEncodeParams &params [[buffer(3)]],
    constant uchar *q_luma [[buffer(4)]],
    constant uchar *q_chroma [[buffer(5)]],
    constant JpegBaselineEncodeHuffmanTable &dc_luma [[buffer(6)]],
    constant JpegBaselineEncodeHuffmanTable &ac_luma [[buffer(7)]],
    constant JpegBaselineEncodeHuffmanTable &dc_chroma [[buffer(8)]],
    constant JpegBaselineEncodeHuffmanTable &ac_chroma [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) {
        return;
    }
    jpeg_encode_baseline_entropy_one(
        input,
        entropy,
        status,
        params,
        q_luma,
        q_chroma,
        dc_luma,
        ac_luma,
        dc_chroma,
        ac_chroma
    );
}

kernel void jpeg_encode_baseline_entropy_batch(
    device const uchar *input [[buffer(0)]],
    device uchar *entropy [[buffer(1)]],
    device JpegBaselineEncodeStatus *status [[buffer(2)]],
    constant JpegBaselineEncodeParams *params [[buffer(3)]],
    constant uchar *q_luma [[buffer(4)]],
    constant uchar *q_chroma [[buffer(5)]],
    constant JpegBaselineEncodeHuffmanTable &dc_luma [[buffer(6)]],
    constant JpegBaselineEncodeHuffmanTable &ac_luma [[buffer(7)]],
    constant JpegBaselineEncodeHuffmanTable &dc_chroma [[buffer(8)]],
    constant JpegBaselineEncodeHuffmanTable &ac_chroma [[buffer(9)]],
    constant uint &tile_count [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= tile_count) {
        return;
    }
    constant JpegBaselineEncodeParams &tile_params = params[gid];
    jpeg_encode_baseline_entropy_one(
        input + tile_params.input_offset_bytes,
        entropy + tile_params.entropy_offset_bytes,
        status + gid,
        tile_params,
        q_luma,
        q_chroma,
        dc_luma,
        ac_luma,
        dc_chroma,
        ac_chroma
    );
}

inline void init_mcu_cursor(
    uint start_mcu,
    uint mcus_per_row,
    thread uint &mx,
    thread uint &my
) {
    my = start_mcu / mcus_per_row;
    mx = start_mcu - my * mcus_per_row;
}

inline void advance_mcu_cursor(thread uint &mx, thread uint &my, uint mcus_per_row) {
    mx += 1u;
    if (mx == mcus_per_row) {
        mx = 0u;
        my += 1u;
    }
}

inline bool refill_one_byte(
    thread BitReader &br,
    device const uchar *bytes,
    uint len
) {
    if (br.pos >= len) {
        return false;
    }
    const uint shift = 64u - 8u - br.bits;
    br.acc |= ulong(bytes[br.pos]) << shift;
    br.pos += 1;
    br.bits += 8;
    return true;
}

inline bool refill_four_bytes(
    thread BitReader &br,
    device const uchar *bytes,
    uint len
) {
    if (br.bits > 32u || br.pos + 4u > len) {
        return false;
    }
    const uint word = (uint(bytes[br.pos]) << 24)
        | (uint(bytes[br.pos + 1u]) << 16)
        | (uint(bytes[br.pos + 2u]) << 8)
        | uint(bytes[br.pos + 3u]);
    const uint shift = 64u - 32u - br.bits;
    br.acc |= ulong(word) << shift;
    br.pos += 4u;
    br.bits += 32u;
    return true;
}

inline bool refill_bits(
    thread BitReader &br,
    device const uchar *bytes,
    uint len
) {
    return refill_four_bytes(br, bytes, len) || refill_one_byte(br, bytes, len);
}

inline bool ensure_bits(
    thread BitReader &br,
    device const uchar *bytes,
    uint len,
    uint n,
    device JpegDecodeStatus *status
) {
    while (br.bits < n) {
        if (!refill_bits(br, bytes, len)) {
            status->code = FAST420_STATUS_TRUNCATED;
            status->position = br.pos;
            return false;
        }
    }
    return true;
}

inline void ensure_bits_padded(
    thread BitReader &br,
    device const uchar *bytes,
    uint len,
    uint n
) {
    while (br.bits < n) {
        if (!refill_bits(br, bytes, len)) {
            br.acc |= ulong(1) << (63u - br.bits);
            br.bits += 1;
        }
    }
}

inline uint peek_bits(thread const BitReader &br, uint n) {
    if (n == 0) {
        return 0;
    }
    return uint(br.acc >> (64u - n));
}

inline void consume_bits(thread BitReader &br, uint n) {
    br.acc <<= n;
    br.bits -= n;
}

inline int huff_extend(int value, uchar ssss) {
    if (ssss == 0) {
        return 0;
    }
    const int threshold = 1 << (ssss - 1);
    if (value < threshold) {
        return value + ((-1) << ssss) + 1;
    }
    return value;
}

inline bool receive_extend(
    thread BitReader &br,
    device const uchar *bytes,
    uint len,
    uchar ssss,
    device JpegDecodeStatus *status,
    thread int &value
) {
    if (ssss == 0) {
        value = 0;
        return true;
    }
    if (!ensure_bits(br, bytes, len, uint(ssss), status)) {
        return false;
    }
    value = huff_extend(int(peek_bits(br, uint(ssss))), ssss);
    consume_bits(br, uint(ssss));
    return true;
}

inline bool skip_receive_extend(
    thread BitReader &br,
    device const uchar *bytes,
    uint len,
    uchar ssss,
    device JpegDecodeStatus *status
) {
    if (ssss == 0) {
        return true;
    }
    if (!ensure_bits(br, bytes, len, uint(ssss), status)) {
        return false;
    }
    consume_bits(br, uint(ssss));
    return true;
}

inline bool configure_restart_thread(
    uint gid,
    uint total_mcus,
    uint restart_interval_mcus,
    uint restart_offset_count,
    uint restart_start_mcu,
    device const uint *restart_offsets,
    thread BitReader &br,
    thread uint &start_mcu,
    thread uint &end_mcu
) {
    br.pos = 0u;
    br.acc = 0u;
    br.bits = 0u;

    if (restart_interval_mcus == 0u) {
        if (gid != 0u) {
            return false;
        }
        start_mcu = 0u;
        end_mcu = total_mcus;
        return true;
    }

    if (gid >= restart_offset_count) {
        return false;
    }

    start_mcu = restart_start_mcu + gid * restart_interval_mcus;
    if (start_mcu >= total_mcus) {
        return false;
    }
    end_mcu = min(total_mcus, start_mcu + restart_interval_mcus);
    br.pos = restart_offsets[gid];
    return true;
}

inline bool configure_entropy_thread(
    uint gid,
    uint total_mcus,
    uint restart_interval_mcus,
    uint segment_count,
    uint restart_start_mcu,
    device const uint *restart_offsets,
    device const JpegEntropyCheckpoint *entropy_checkpoints,
    thread BitReader &br,
    thread uint &start_mcu,
    thread uint &end_mcu,
    thread int &y_prev_dc,
    thread int &cb_prev_dc,
    thread int &cr_prev_dc
) {
    y_prev_dc = 0;
    cb_prev_dc = 0;
    cr_prev_dc = 0;

    if (restart_interval_mcus != 0u) {
        return configure_restart_thread(
            gid,
            total_mcus,
            restart_interval_mcus,
            segment_count,
            restart_start_mcu,
            restart_offsets,
            br,
            start_mcu,
            end_mcu
        );
    }

    br.pos = 0u;
    br.acc = 0u;
    br.bits = 0u;

    if (gid >= segment_count) {
        return false;
    }

    const JpegEntropyCheckpoint checkpoint = entropy_checkpoints[gid];
    start_mcu = checkpoint.mcu_index;
    if (start_mcu >= total_mcus) {
        return false;
    }
    if (gid + 1u < segment_count) {
        end_mcu = min(total_mcus, entropy_checkpoints[gid + 1u].mcu_index);
    } else {
        end_mcu = total_mcus;
    }
    if (end_mcu <= start_mcu) {
        return false;
    }

    br.pos = checkpoint.entropy_pos;
    br.acc = checkpoint.bit_acc;
    br.bits = checkpoint.bit_count;
    y_prev_dc = checkpoint.y_prev_dc;
    cb_prev_dc = checkpoint.cb_prev_dc;
    cr_prev_dc = checkpoint.cr_prev_dc;
    return true;
}

inline bool configure_batch_entropy_thread(
    uint gid,
    uint total_mcus,
    uint segment_count,
    uint tile_count,
    device const uint *entropy_offsets,
    device const uint *entropy_lens,
    device const JpegEntropyCheckpoint *entropy_checkpoints,
    thread BitReader &br,
    thread uint &tile_index,
    thread uint &start_mcu,
    thread uint &end_mcu,
    thread uint &entropy_end,
    thread int &y_prev_dc,
    thread int &cb_prev_dc,
    thread int &cr_prev_dc
) {
    if (segment_count == 0u) {
        return false;
    }

    tile_index = gid / segment_count;
    const uint local_gid = gid - tile_index * segment_count;
    if (tile_index >= tile_count) {
        return false;
    }

    const uint checkpoint_base = tile_index * segment_count;
    const JpegEntropyCheckpoint checkpoint = entropy_checkpoints[checkpoint_base + local_gid];
    start_mcu = checkpoint.mcu_index;
    if (start_mcu >= total_mcus) {
        return false;
    }
    end_mcu = total_mcus;
    if (local_gid + 1u < segment_count) {
        end_mcu = min(total_mcus, entropy_checkpoints[checkpoint_base + local_gid + 1u].mcu_index);
    }
    if (end_mcu <= start_mcu) {
        return false;
    }

    const uint entropy_base = entropy_offsets[tile_index];
    entropy_end = entropy_base + entropy_lens[tile_index];
    br.pos = entropy_base + checkpoint.entropy_pos;
    br.acc = checkpoint.bit_acc;
    br.bits = checkpoint.bit_count;
    y_prev_dc = checkpoint.y_prev_dc;
    cb_prev_dc = checkpoint.cb_prev_dc;
    cr_prev_dc = checkpoint.cr_prev_dc;
    return true;
}

#define JPEG_ENTROPY_THREAD_VARS() \
    thread BitReader br; \
    uint start_mcu = 0u; \
    uint end_mcu = 0u; \
    int y_prev_dc = 0; \
    int cb_prev_dc = 0; \
    int cr_prev_dc = 0

#define JPEG_CONFIGURE_ENTROPY_THREAD(GID, TOTAL_MCUS, PARAMS, RESTART_OFFSETS, CHECKPOINTS) \
    configure_entropy_thread( \
        (GID), \
        (TOTAL_MCUS), \
        (PARAMS).restart_interval_mcus, \
        (PARAMS).restart_offset_count, \
        (PARAMS).restart_start_mcu, \
        (RESTART_OFFSETS), \
        (CHECKPOINTS), \
        br, \
        start_mcu, \
        end_mcu, \
        y_prev_dc, \
        cb_prev_dc, \
        cr_prev_dc \
    )

#define JPEG_BATCH_ENTROPY_THREAD_VARS() \
    thread BitReader br; \
    uint tile_index = 0u; \
    uint start_mcu = 0u; \
    uint end_mcu = 0u; \
    uint entropy_end = 0u; \
    int y_prev_dc = 0; \
    int cb_prev_dc = 0; \
    int cr_prev_dc = 0

#define JPEG_CONFIGURE_BATCH_ENTROPY_THREAD(GID, TOTAL_MCUS, PARAMS, OFFSETS, LENS, CHECKPOINTS) \
    configure_batch_entropy_thread( \
        (GID), \
        (TOTAL_MCUS), \
        (PARAMS).segment_count, \
        (PARAMS).tile_count, \
        (OFFSETS), \
        (LENS), \
        (CHECKPOINTS), \
        br, \
        tile_index, \
        start_mcu, \
        end_mcu, \
        entropy_end, \
        y_prev_dc, \
        cb_prev_dc, \
        cr_prev_dc \
    )

inline void prepare_huffman(
    constant JpegHuffmanTable &raw,
    thread PreparedHuffman &out
) {
    uchar huffsize[256];
    ushort huffcode[256];
    ushort huffsize_len = 0;
    for (uint i = 0; i < 17; ++i) {
        out.min_code[i] = 0x7fffffff;
        out.max_code[i] = -1;
        out.val_offset[i] = 0;
    }
    for (uint i = 0; i < raw.values_len; ++i) {
        out.values[i] = raw.values[i];
    }
    for (uint i = 0; i < 512; ++i) {
        out.fast_symbol[i] = 0;
        out.fast_len[i] = 0;
    }
    out.values_len = raw.values_len;
    for (uint len_minus_1 = 0; len_minus_1 < 16; ++len_minus_1) {
        const uchar len = uchar(len_minus_1 + 1);
        for (uchar count = 0; count < raw.bits[len_minus_1]; ++count) {
            huffsize[huffsize_len] = len;
            huffsize_len += 1;
        }
    }

    uint code = 0;
    uchar si = huffsize_len == 0 ? 0 : huffsize[0];
    for (ushort k = 0; k < huffsize_len; ++k) {
        const uchar s = huffsize[k];
        while (s != si) {
            code <<= 1;
            si += 1;
        }
        huffcode[k] = ushort(code);
        code += 1;
    }

    ushort k = 0;
    for (uint len_minus_1 = 0; len_minus_1 < 16; ++len_minus_1) {
        const uint len = len_minus_1 + 1;
        const ushort count = raw.bits[len_minus_1];
        if (count == 0) {
            continue;
        }
        out.min_code[len] = int(huffcode[k]);
        out.max_code[len] = int(huffcode[k + count - 1]);
        out.val_offset[len] = int(k) - out.min_code[len];
        k += count;
    }

    for (uint idx = 0; idx < huffsize_len; ++idx) {
        const uint len = uint(huffsize[idx]);
        if (len == 0u || len > 9u) {
            continue;
        }
        const uint prefix = uint(huffcode[idx]) << (9u - len);
        const uint fill = 1u << (9u - len);
        for (uint suffix = 0; suffix < fill; ++suffix) {
            out.fast_symbol[prefix | suffix] = raw.values[idx];
            out.fast_len[prefix | suffix] = huffsize[idx];
        }
    }
}

inline bool decode_symbol(
    thread BitReader &br,
    device const uchar *bytes,
    uint len,
    constant PreparedHuffman &table,
    device JpegDecodeStatus *status,
    thread uchar &symbol
) {
    ensure_bits_padded(br, bytes, len, 9);
    const uint fast_index = peek_bits(br, 9);
    const uchar len9 = table.fast_len[fast_index];
    if (len9 != 0) {
        consume_bits(br, uint(len9));
        symbol = table.fast_symbol[fast_index];
        return true;
    }

    ensure_bits_padded(br, bytes, len, 16);
    const int code16 = int(peek_bits(br, 16));
    for (uint length = 1; length <= 16; ++length) {
        const int code = code16 >> (16 - int(length));
        if (code <= table.max_code[length]) {
            if (code < table.min_code[length]) {
                continue;
            }
            const int idx = code + table.val_offset[length];
            if (idx < 0 || idx >= int(table.values_len)) {
                status->code = FAST420_STATUS_HUFFMAN;
                status->position = br.pos;
                return false;
            }
            consume_bits(br, length);
            symbol = table.values[idx];
            return true;
        }
    }
    status->code = FAST420_STATUS_HUFFMAN;
    status->position = br.pos;
    return false;
}

inline bool decode_block(
    thread BitReader &br,
    device const uchar *bytes,
    uint len,
    constant PreparedHuffman &dc_table,
    constant PreparedHuffman &ac_table,
    constant ushort *quant,
    thread int &prev_dc,
    device JpegDecodeStatus *status,
    thread short coeffs[64],
    thread bool &dc_only
) {
    thread short4 *coeff_chunks = reinterpret_cast<thread short4 *>(coeffs);
    for (uint i = 0; i < 16; ++i) {
        coeff_chunks[i] = short4(0);
    }
    uchar ssss = 0;
    if (!decode_symbol(br, bytes, len, dc_table, status, ssss)) {
        return false;
    }
    if (ssss > 15) {
        status->code = FAST420_STATUS_HUFFMAN;
        status->position = br.pos;
        return false;
    }

    int diff = 0;
    if (!receive_extend(br, bytes, len, ssss, status, diff)) {
        return false;
    }
    prev_dc += diff;
    coeffs[0] = clamp_i16(prev_dc * int(quant[0]));

    dc_only = true;
    uint k = 1;
    while (k < 64) {
        uchar symbol = 0;
        if (!decode_symbol(br, bytes, len, ac_table, status, symbol)) {
            return false;
        }
        const uint run = uint(symbol >> 4);
        ssss = symbol & 0x0F;
        if (ssss == 0) {
            if (run == 15) {
                k += 16;
                continue;
            }
            break;
        }

        k += run;
        if (k >= 64) {
            status->code = FAST420_STATUS_HUFFMAN;
            status->position = br.pos;
            return false;
        }

        int value = 0;
        if (!receive_extend(br, bytes, len, ssss, status, value)) {
            return false;
        }
        coeffs[ZIGZAG[k]] = clamp_i16(value * int(quant[k]));
        dc_only = false;
        k += 1;
    }
    return true;
}

inline bool decode_block_skip(
    thread BitReader &br,
    device const uchar *bytes,
    uint len,
    constant PreparedHuffman &dc_table,
    constant PreparedHuffman &ac_table,
    thread int &prev_dc,
    device JpegDecodeStatus *status
) {
    uchar ssss = 0;
    if (!decode_symbol(br, bytes, len, dc_table, status, ssss)) {
        return false;
    }
    if (ssss > 15) {
        status->code = FAST420_STATUS_HUFFMAN;
        status->position = br.pos;
        return false;
    }

    int diff = 0;
    if (!receive_extend(br, bytes, len, ssss, status, diff)) {
        return false;
    }
    prev_dc += diff;

    uint k = 1;
    while (k < 64) {
        uchar symbol = 0;
        if (!decode_symbol(br, bytes, len, ac_table, status, symbol)) {
            return false;
        }
        const uint run = uint(symbol >> 4);
        ssss = symbol & 0x0F;
        if (ssss == 0) {
            if (run == 15) {
                k += 16;
                continue;
            }
            break;
        }

        k += run;
        if (k >= 64) {
            status->code = FAST420_STATUS_HUFFMAN;
            status->position = br.pos;
            return false;
        }

        if (!skip_receive_extend(br, bytes, len, ssss, status)) {
            return false;
        }
        k += 1;
    }
    return true;
}

inline bool block_intersects_rect(
    uint block_x,
    uint block_y,
    uint block_width,
    uint block_height,
    uint rect_x,
    uint rect_y,
    uint rect_width,
    uint rect_height
) {
    const uint block_x1 = block_x + block_width;
    const uint block_y1 = block_y + block_height;
    const uint rect_x1 = rect_x + rect_width;
    const uint rect_y1 = rect_y + rect_height;
    return block_x < rect_x1 && rect_x < block_x1 && block_y < rect_y1 && rect_y < block_y1;
}

inline int descale(int value, int shift) {
    return value >> shift;
}

inline uchar descale_and_clamp(int value, int shift) {
    const int shifted = value >> shift;
    return clamp_u8(shifted + 128);
}

inline void idct_1d_column(
    thread const short input[64],
    thread int work[64],
    uint col
) {
    const int p0 = int(input[col]);
    const int p1 = int(input[col + 8]);
    const int p2 = int(input[col + 16]);
    const int p3 = int(input[col + 24]);
    const int p4 = int(input[col + 32]);
    const int p5 = int(input[col + 40]);
    const int p6 = int(input[col + 48]);
    const int p7 = int(input[col + 56]);

    if (p1 == 0 && p2 == 0 && p3 == 0 && p4 == 0 && p5 == 0 && p6 == 0 && p7 == 0) {
        const int dc = p0 << PASS1_BITS;
        work[col] = dc;
        work[col + 8] = dc;
        work[col + 16] = dc;
        work[col + 24] = dc;
        work[col + 32] = dc;
        work[col + 40] = dc;
        work[col + 48] = dc;
        work[col + 56] = dc;
        return;
    }

    const int z2 = p2;
    const int z3 = p6;
    const int z1 = (z2 + z3) * FIX_0_541196100;
    const int tmp2 = z1 - z3 * FIX_1_847759065;
    const int tmp3 = z1 + z2 * FIX_0_765366865;

    const int tmp0 = (p0 + p4) << CONST_BITS;
    const int tmp1 = (p0 - p4) << CONST_BITS;

    const int tmp10 = tmp0 + tmp3;
    const int tmp13 = tmp0 - tmp3;
    const int tmp11 = tmp1 + tmp2;
    const int tmp12 = tmp1 - tmp2;

    const int z1o = p7 + p1;
    const int z2o = p5 + p3;
    const int z3o = p7 + p3;
    const int z4o = p5 + p1;
    const int z5 = (z3o + z4o) * FIX_1_175875602;

    const int tmp0o = p7 * FIX_0_298631336;
    const int tmp1o = p5 * FIX_2_053119869;
    const int tmp2o = p3 * FIX_3_072711026;
    const int tmp3o = p1 * FIX_1_501321110;
    const int z1m = z1o * -FIX_0_899976223;
    const int z2m = z2o * -FIX_2_562915447;
    const int z3m = z3o * -FIX_1_961570560 + z5;
    const int z4m = z4o * -FIX_0_390180644 + z5;

    const int out0 = tmp0o + z1m + z3m;
    const int out1 = tmp1o + z2m + z4m;
    const int out2 = tmp2o + z2m + z3m;
    const int out3 = tmp3o + z1m + z4m;

    const int shift = CONST_BITS - PASS1_BITS;
    const int rounding = 1 << (shift - 1);
    work[col] = descale(tmp10 + out3 + rounding, shift);
    work[col + 56] = descale(tmp10 - out3 + rounding, shift);
    work[col + 8] = descale(tmp11 + out2 + rounding, shift);
    work[col + 48] = descale(tmp11 - out2 + rounding, shift);
    work[col + 16] = descale(tmp12 + out1 + rounding, shift);
    work[col + 40] = descale(tmp12 - out1 + rounding, shift);
    work[col + 24] = descale(tmp13 + out0 + rounding, shift);
    work[col + 32] = descale(tmp13 - out0 + rounding, shift);
}

inline void idct_1d_column_bottom_half_zero(
    thread const short input[64],
    thread int work[64],
    uint col
) {
    const int p0 = int(input[col]);
    const int p1 = int(input[col + 8]);
    const int p2 = int(input[col + 16]);
    const int p3 = int(input[col + 24]);

    if (p1 == 0 && p2 == 0 && p3 == 0) {
        const int dc = p0 << PASS1_BITS;
        work[col] = dc;
        work[col + 8] = dc;
        work[col + 16] = dc;
        work[col + 24] = dc;
        work[col + 32] = dc;
        work[col + 40] = dc;
        work[col + 48] = dc;
        work[col + 56] = dc;
        return;
    }

    const int z1 = p2 * FIX_0_541196100;
    const int tmp2 = z1;
    const int tmp3 = z1 + p2 * FIX_0_765366865;

    const int tmp0 = p0 << CONST_BITS;
    const int tmp1 = p0 << CONST_BITS;

    const int tmp10 = tmp0 + tmp3;
    const int tmp13 = tmp0 - tmp3;
    const int tmp11 = tmp1 + tmp2;
    const int tmp12 = tmp1 - tmp2;

    const int z5 = (p1 + p3) * FIX_1_175875602;
    const int z1m = p1 * -FIX_0_899976223;
    const int z2m = p3 * -FIX_2_562915447;
    const int z3m = p3 * -FIX_1_961570560 + z5;
    const int z4m = p1 * -FIX_0_390180644 + z5;

    const int out0 = z1m + z3m;
    const int out1 = z2m + z4m;
    const int out2 = p3 * FIX_3_072711026 + z2m + z3m;
    const int out3 = p1 * FIX_1_501321110 + z1m + z4m;

    const int shift = CONST_BITS - PASS1_BITS;
    const int rounding = 1 << (shift - 1);
    work[col] = descale(tmp10 + out3 + rounding, shift);
    work[col + 56] = descale(tmp10 - out3 + rounding, shift);
    work[col + 8] = descale(tmp11 + out2 + rounding, shift);
    work[col + 48] = descale(tmp11 - out2 + rounding, shift);
    work[col + 16] = descale(tmp12 + out1 + rounding, shift);
    work[col + 40] = descale(tmp12 - out1 + rounding, shift);
    work[col + 24] = descale(tmp13 + out0 + rounding, shift);
    work[col + 32] = descale(tmp13 - out0 + rounding, shift);
}

inline void idct_1d_row(
    thread const int work[64],
    thread uchar output[64],
    uint row
) {
    const uint base = row * 8;
    const int p0 = work[base];
    const int p1 = work[base + 1];
    const int p2 = work[base + 2];
    const int p3 = work[base + 3];
    const int p4 = work[base + 4];
    const int p5 = work[base + 5];
    const int p6 = work[base + 6];
    const int p7 = work[base + 7];

    const int shift = CONST_BITS + PASS1_BITS + 3;
    const int rounding = 1 << (shift - 1);

    if (p1 == 0 && p2 == 0 && p3 == 0 && p4 == 0 && p5 == 0 && p6 == 0 && p7 == 0) {
        const int dc_shift = PASS1_BITS + 3;
        const int dc_rounding = 1 << (dc_shift - 1);
        const uchar pixel = descale_and_clamp(p0 + dc_rounding, dc_shift);
        for (uint i = 0; i < 8; ++i) {
            output[base + i] = pixel;
        }
        return;
    }

    const int z2 = p2;
    const int z3 = p6;
    const int z1 = (z2 + z3) * FIX_0_541196100;
    const int tmp2 = z1 - z3 * FIX_1_847759065;
    const int tmp3 = z1 + z2 * FIX_0_765366865;

    const int tmp0 = (p0 + p4) << CONST_BITS;
    const int tmp1 = (p0 - p4) << CONST_BITS;

    const int tmp10 = tmp0 + tmp3;
    const int tmp13 = tmp0 - tmp3;
    const int tmp11 = tmp1 + tmp2;
    const int tmp12 = tmp1 - tmp2;

    const int z1o = p7 + p1;
    const int z2o = p5 + p3;
    const int z3o = p7 + p3;
    const int z4o = p5 + p1;
    const int z5 = (z3o + z4o) * FIX_1_175875602;

    const int tmp0o = p7 * FIX_0_298631336;
    const int tmp1o = p5 * FIX_2_053119869;
    const int tmp2o = p3 * FIX_3_072711026;
    const int tmp3o = p1 * FIX_1_501321110;
    const int z1m = z1o * -FIX_0_899976223;
    const int z2m = z2o * -FIX_2_562915447;
    const int z3m = z3o * -FIX_1_961570560 + z5;
    const int z4m = z4o * -FIX_0_390180644 + z5;

    const int out0 = tmp0o + z1m + z3m;
    const int out1 = tmp1o + z2m + z4m;
    const int out2 = tmp2o + z2m + z3m;
    const int out3 = tmp3o + z1m + z4m;

    output[base] = descale_and_clamp(tmp10 + out3 + rounding, shift);
    output[base + 7] = descale_and_clamp(tmp10 - out3 + rounding, shift);
    output[base + 1] = descale_and_clamp(tmp11 + out2 + rounding, shift);
    output[base + 6] = descale_and_clamp(tmp11 - out2 + rounding, shift);
    output[base + 2] = descale_and_clamp(tmp12 + out1 + rounding, shift);
    output[base + 5] = descale_and_clamp(tmp12 - out1 + rounding, shift);
    output[base + 3] = descale_and_clamp(tmp13 + out0 + rounding, shift);
    output[base + 4] = descale_and_clamp(tmp13 - out0 + rounding, shift);
}

inline void idct_islow(
    thread const short input[64],
    thread uchar output[64]
) {
    thread int work[64];
    bool upper_half_zero = true;
    for (uint i = 32; i < 64; ++i) {
        if (input[i] != 0) {
            upper_half_zero = false;
            break;
        }
    }
    for (uint col = 0; col < 8; ++col) {
        if (upper_half_zero) {
            idct_1d_column_bottom_half_zero(input, work, col);
        } else {
            idct_1d_column(input, work, col);
        }
    }
    for (uint row = 0; row < 8; ++row) {
        idct_1d_row(work, output, row);
    }
}

inline void idct_islow_dc_only(
    short dc_coeff,
    thread uchar output[64]
) {
    const uchar pixel = clamp_u8(((int(dc_coeff) + 4) >> 3) + 128);
    for (uint i = 0; i < 64; ++i) {
        output[i] = pixel;
    }
}

inline void idct_block(
    thread const short coeffs[64],
    bool dc_only,
    thread uchar pixels[64]
) {
    if (dc_only) {
        idct_islow_dc_only(coeffs[0], pixels);
    } else {
        idct_islow(coeffs, pixels);
    }
}

inline void deposit_block(
    device uchar *plane,
    uint stride,
    uint width,
    uint height,
    uint x,
    uint y,
    thread const uchar block[64]
) {
    if (x >= width || y >= height) {
        return;
    }
    const uint copy_width = min(8u, width - x);
    const uint copy_height = min(8u, height - y);
    if (copy_width == 8u && copy_height == 8u && (stride & 3u) == 0u) {
        for (uint by = 0; by < 8u; ++by) {
            const uint src = by * 8u;
            const uint dst = (y + by) * stride + x;
            *(device uchar4 *)(plane + dst) = uchar4(
                block[src],
                block[src + 1u],
                block[src + 2u],
                block[src + 3u]
            );
            *(device uchar4 *)(plane + dst + 4u) = uchar4(
                block[src + 4u],
                block[src + 5u],
                block[src + 6u],
                block[src + 7u]
            );
        }
        return;
    }
    for (uint by = 0; by < copy_height; ++by) {
        const uint dst = (y + by) * stride + x;
        for (uint bx = 0; bx < copy_width; ++bx) {
            plane[dst + bx] = block[by * 8 + bx];
        }
    }
}

inline bool decode_idct_deposit_block(
    thread BitReader &br,
    device const uchar *bytes,
    uint len,
    constant PreparedHuffman &dc_table,
    constant PreparedHuffman &ac_table,
    constant ushort *quant,
    thread int &prev_dc,
    device JpegDecodeStatus *status,
    device uchar *plane,
    uint stride,
    uint width,
    uint height,
    uint x,
    uint y,
    thread short coeffs[64],
    thread uchar pixels[64]
) {
    bool dc_only = false;
    if (!decode_block(br, bytes, len, dc_table, ac_table, quant, prev_dc, status, coeffs, dc_only)) {
        return false;
    }
    idct_block(coeffs, dc_only, pixels);
    deposit_block(plane, stride, width, height, x, y, pixels);
    return true;
}

inline void idct_4x4_column(
    thread const short input[64],
    thread int work[32],
    uint col
) {
    const int p0 = int(input[col]);
    const int p1 = int(input[col + 8]);
    const int p2 = int(input[col + 16]);
    const int p3 = int(input[col + 24]);
    const int p5 = int(input[col + 40]);
    const int p6 = int(input[col + 48]);
    const int p7 = int(input[col + 56]);

    if (p1 == 0 && p2 == 0 && p3 == 0 && p5 == 0 && p6 == 0 && p7 == 0) {
        const int dc = p0 << PASS1_BITS;
        work[col] = dc;
        work[8 + col] = dc;
        work[16 + col] = dc;
        work[24 + col] = dc;
        return;
    }

    const int tmp0_base = p0 << (CONST_BITS + 1);
    const int tmp2_even = p2 * FIX_1_847759065 + p6 * -FIX_0_765366865;
    const int tmp10 = tmp0_base + tmp2_even;
    const int tmp12 = tmp0_base - tmp2_even;

    const int tmp0 = p7 * -FIX_0_211164243
        + p5 * FIX_1_451774981
        + p3 * -FIX_2_172734803
        + p1 * FIX_1_061594337;
    const int tmp2 = p7 * -FIX_0_509795579
        + p5 * -FIX_0_601344887
        + p3 * FIX_0_899976223
        + p1 * FIX_2_562915447;

    const int shift = CONST_BITS - PASS1_BITS + 1;
    work[col] = descale(tmp10 + tmp2, shift);
    work[24 + col] = descale(tmp10 - tmp2, shift);
    work[8 + col] = descale(tmp12 + tmp0, shift);
    work[16 + col] = descale(tmp12 - tmp0, shift);
}

inline void idct_4x4_row(
    thread const int work[32],
    thread uchar output[16],
    uint row
) {
    const uint base = row * 8;
    const int p0 = work[base];
    const int p1 = work[base + 1];
    const int p2 = work[base + 2];
    const int p3 = work[base + 3];
    const int p5 = work[base + 5];
    const int p6 = work[base + 6];
    const int p7 = work[base + 7];

    const uint out = row * 4;
    if (p1 == 0 && p2 == 0 && p3 == 0 && p5 == 0 && p6 == 0 && p7 == 0) {
        const uchar dc = descale_and_clamp(p0, PASS1_BITS + 3);
        output[out] = dc;
        output[out + 1] = dc;
        output[out + 2] = dc;
        output[out + 3] = dc;
        return;
    }

    const int tmp0_base = p0 << (CONST_BITS + 1);
    const int tmp2_even = p2 * FIX_1_847759065 + p6 * -FIX_0_765366865;
    const int tmp10 = tmp0_base + tmp2_even;
    const int tmp12 = tmp0_base - tmp2_even;

    const int tmp0 = p7 * -FIX_0_211164243
        + p5 * FIX_1_451774981
        + p3 * -FIX_2_172734803
        + p1 * FIX_1_061594337;
    const int tmp2 = p7 * -FIX_0_509795579
        + p5 * -FIX_0_601344887
        + p3 * FIX_0_899976223
        + p1 * FIX_2_562915447;

    const int shift = CONST_BITS + PASS1_BITS + 3 + 1;
    output[out] = descale_and_clamp(tmp10 + tmp2, shift);
    output[out + 3] = descale_and_clamp(tmp10 - tmp2, shift);
    output[out + 1] = descale_and_clamp(tmp12 + tmp0, shift);
    output[out + 2] = descale_and_clamp(tmp12 - tmp0, shift);
}

inline void idct_islow_4x4(
    thread const short input[64],
    thread uchar output[16]
) {
    thread int work[32];
    for (uint col = 0; col < 8; ++col) {
        if (col == 4) {
            continue;
        }
        idct_4x4_column(input, work, col);
    }
    for (uint row = 0; row < 4; ++row) {
        idct_4x4_row(work, output, row);
    }
}

inline void idct_2x2_column(
    thread const short input[64],
    thread int work[16],
    uint col
) {
    const int p0 = int(input[col]);
    const int p1 = int(input[col + 8]);
    const int p3 = int(input[col + 24]);
    const int p5 = int(input[col + 40]);
    const int p7 = int(input[col + 56]);

    if (p1 == 0 && p3 == 0 && p5 == 0 && p7 == 0) {
        const int dc = p0 << PASS1_BITS;
        work[col] = dc;
        work[8 + col] = dc;
        return;
    }

    const int tmp10 = p0 << (CONST_BITS + 2);
    const int tmp0 = p7 * -FIX_0_720959822
        + p5 * FIX_0_850430095
        + p3 * -FIX_1_272758580
        + p1 * FIX_3_624509785;

    const int shift = CONST_BITS - PASS1_BITS + 2;
    work[col] = descale(tmp10 + tmp0, shift);
    work[8 + col] = descale(tmp10 - tmp0, shift);
}

inline void idct_2x2_row(
    thread const int work[16],
    thread uchar output[4],
    uint row
) {
    const uint base = row * 8;
    const int p0 = work[base];
    const int p1 = work[base + 1];
    const int p3 = work[base + 3];
    const int p5 = work[base + 5];
    const int p7 = work[base + 7];

    if (p1 == 0 && p3 == 0 && p5 == 0 && p7 == 0) {
        const uchar dc = descale_and_clamp(p0, PASS1_BITS + 3);
        const uint out = row * 2;
        output[out] = dc;
        output[out + 1] = dc;
        return;
    }

    const int tmp10 = p0 << (CONST_BITS + 2);
    const int tmp0 = p7 * -FIX_0_720959822
        + p5 * FIX_0_850430095
        + p3 * -FIX_1_272758580
        + p1 * FIX_3_624509785;

    const int shift = CONST_BITS + PASS1_BITS + 5;
    const uint out = row * 2;
    output[out] = descale_and_clamp(tmp10 + tmp0, shift);
    output[out + 1] = descale_and_clamp(tmp10 - tmp0, shift);
}

inline void idct_islow_2x2(
    thread const short input[64],
    thread uchar output[4]
) {
    thread int work[16];
    for (uint col = 0; col < 8; ++col) {
        if (col == 2 || col == 4 || col == 6) {
            continue;
        }
        idct_2x2_column(input, work, col);
    }
    for (uint row = 0; row < 2; ++row) {
        idct_2x2_row(work, output, row);
    }
}

inline uchar idct_islow_1x1(thread const short input[64]) {
    return descale_and_clamp(int(input[0]), 3);
}

inline void deposit_block_region(
    device uchar *plane,
    uint stride,
    uint width,
    uint height,
    uint origin_x,
    uint origin_y,
    uint block_x,
    uint block_y,
    thread const uchar pixels[64]
) {
    const int dst_x = int(block_x) - int(origin_x);
    const int dst_y = int(block_y) - int(origin_y);
    for (uint row = 0; row < 8; ++row) {
        const int out_y = dst_y + int(row);
        if (out_y < 0 || out_y >= int(height)) {
            continue;
        }
        for (uint col = 0; col < 8; ++col) {
            const int out_x = dst_x + int(col);
            if (out_x < 0 || out_x >= int(width)) {
                continue;
            }
            plane[uint(out_y) * stride + uint(out_x)] = pixels[row * 8u + col];
        }
    }
}

inline void deposit_block_4x4(
    device uchar *plane,
    uint stride,
    uint width,
    uint height,
    uint x,
    uint y,
    thread const uchar block[16]
) {
    if (x >= width || y >= height) {
        return;
    }
    const uint copy_width = min(4u, width - x);
    const uint copy_height = min(4u, height - y);
    for (uint by = 0; by < copy_height; ++by) {
        const uint dst = (y + by) * stride + x;
        for (uint bx = 0; bx < copy_width; ++bx) {
            plane[dst + bx] = block[by * 4 + bx];
        }
    }
}

inline void deposit_block_4x4_region(
    device uchar *plane,
    uint stride,
    uint width,
    uint height,
    uint origin_x,
    uint origin_y,
    uint block_x,
    uint block_y,
    thread const uchar pixels[16]
) {
    const int dst_x = int(block_x) - int(origin_x);
    const int dst_y = int(block_y) - int(origin_y);
    for (uint row = 0; row < 4; ++row) {
        const int out_y = dst_y + int(row);
        if (out_y < 0 || out_y >= int(height)) {
            continue;
        }
        for (uint col = 0; col < 4; ++col) {
            const int out_x = dst_x + int(col);
            if (out_x < 0 || out_x >= int(width)) {
                continue;
            }
            plane[uint(out_y) * stride + uint(out_x)] = pixels[row * 4u + col];
        }
    }
}

inline void deposit_block_2x2(
    device uchar *plane,
    uint stride,
    uint width,
    uint height,
    uint x,
    uint y,
    thread const uchar block[4]
) {
    if (x >= width || y >= height) {
        return;
    }
    const uint copy_width = min(2u, width - x);
    const uint copy_height = min(2u, height - y);
    for (uint by = 0; by < copy_height; ++by) {
        const uint dst = (y + by) * stride + x;
        for (uint bx = 0; bx < copy_width; ++bx) {
            plane[dst + bx] = block[by * 2 + bx];
        }
    }
}

inline void deposit_block_2x2_region(
    device uchar *plane,
    uint stride,
    uint width,
    uint height,
    uint origin_x,
    uint origin_y,
    uint block_x,
    uint block_y,
    thread const uchar pixels[4]
) {
    const int dst_x = int(block_x) - int(origin_x);
    const int dst_y = int(block_y) - int(origin_y);
    for (uint row = 0; row < 2; ++row) {
        const int out_y = dst_y + int(row);
        if (out_y < 0 || out_y >= int(height)) {
            continue;
        }
        for (uint col = 0; col < 2; ++col) {
            const int out_x = dst_x + int(col);
            if (out_x < 0 || out_x >= int(width)) {
                continue;
            }
            plane[uint(out_y) * stride + uint(out_x)] = pixels[row * 2u + col];
        }
    }
}

inline void deposit_scaled_block(
    device uchar *plane,
    uint stride,
    uint width,
    uint height,
    uint x,
    uint y,
    uint scale_shift,
    thread const short coeffs[64],
    bool dc_only
) {
    if (scale_shift == 1u) {
        thread uchar pixels4[16];
        if (dc_only) {
            const uchar pixel = idct_islow_1x1(coeffs);
            for (uint i = 0; i < 16; ++i) {
                pixels4[i] = pixel;
            }
        } else {
            idct_islow_4x4(coeffs, pixels4);
        }
        deposit_block_4x4(plane, stride, width, height, x, y, pixels4);
        return;
    }

    if (scale_shift == 2u) {
        thread uchar pixels2[4];
        if (dc_only) {
            const uchar pixel = idct_islow_1x1(coeffs);
            for (uint i = 0; i < 4; ++i) {
                pixels2[i] = pixel;
            }
        } else {
            idct_islow_2x2(coeffs, pixels2);
        }
        deposit_block_2x2(plane, stride, width, height, x, y, pixels2);
        return;
    }

    const uchar pixel = idct_islow_1x1(coeffs);
    if (x < width && y < height) {
        plane[y * stride + x] = pixel;
    }
}

inline void deposit_scaled_block_region(
    device uchar *plane,
    uint stride,
    uint width,
    uint height,
    uint origin_x,
    uint origin_y,
    uint x,
    uint y,
    uint scale_shift,
    thread const short coeffs[64],
    bool dc_only
) {
    if (scale_shift == 0u) {
        thread uchar pixels8[64];
        idct_block(coeffs, dc_only, pixels8);
        deposit_block_region(plane, stride, width, height, origin_x, origin_y, x, y, pixels8);
    } else if (scale_shift == 1u) {
        thread uchar pixels4[16];
        if (dc_only) {
            const uchar pixel = idct_islow_1x1(coeffs);
            for (uint i = 0; i < 16; ++i) {
                pixels4[i] = pixel;
            }
        } else {
            idct_islow_4x4(coeffs, pixels4);
        }
        deposit_block_4x4_region(plane, stride, width, height, origin_x, origin_y, x, y, pixels4);
    } else if (scale_shift == 2u) {
        thread uchar pixels2[4];
        if (dc_only) {
            const uchar pixel = idct_islow_1x1(coeffs);
            for (uint i = 0; i < 4; ++i) {
                pixels2[i] = pixel;
            }
        } else {
            idct_islow_2x2(coeffs, pixels2);
        }
        deposit_block_2x2_region(plane, stride, width, height, origin_x, origin_y, x, y, pixels2);
    } else {
        const int out_x = int(x) - int(origin_x);
        const int out_y = int(y) - int(origin_y);
        if (out_x >= 0 && out_x < int(width) && out_y >= 0 && out_y < int(height)) {
            const uchar pixel = idct_islow_1x1(coeffs);
            plane[uint(out_y) * stride + uint(out_x)] = pixel;
        }
    }
}

inline uint h2v2_weighted_sample_sum(uchar primary, uchar adjacent) { return 3u * uint(primary) + uint(adjacent); }

inline uchar h2v2_sample(
    device const uchar *near_row,
    device const uchar *curr_row,
    uint n,
    uint x
) {
    if (n == 0) {
        return 0;
    }
    const uint sample = min(x / 2, n - 1);
    const uint this_sum = h2v2_weighted_sample_sum(curr_row[sample], near_row[sample]);
    if (n == 1) {
        return uchar((4u * this_sum + 8u) >> 4);
    }
    if (x == 0) {
        return uchar((this_sum * 4u + 8u) >> 4);
    }
    if (x == n * 2u - 1u) {
        return uchar((this_sum * 4u + 7u) >> 4);
    }
    if ((x & 1u) == 0u) {
        const uint last_sum = h2v2_weighted_sample_sum(curr_row[sample - 1], near_row[sample - 1]);
        return uchar((this_sum * 3u + last_sum + 8u) >> 4);
    }
    const uint next_sum = h2v2_weighted_sample_sum(curr_row[sample + 1], near_row[sample + 1]);
    return uchar((this_sum * 3u + next_sum + 7u) >> 4);
}

inline uchar h2v2_sample_thread(
    thread const uchar *near_row,
    thread const uchar *curr_row,
    uint n,
    uint x
) {
    if (n == 0) {
        return 0;
    }
    const uint sample = min(x / 2, n - 1);
    const uint this_sum = h2v2_weighted_sample_sum(curr_row[sample], near_row[sample]);
    if (n == 1) {
        return uchar((4u * this_sum + 8u) >> 4);
    }
    if (x == 0) {
        return uchar((this_sum * 4u + 8u) >> 4);
    }
    if (x == n * 2u - 1u) {
        return uchar((this_sum * 4u + 7u) >> 4);
    }
    if ((x & 1u) == 0u) {
        const uint last_sum = h2v2_weighted_sample_sum(curr_row[sample - 1], near_row[sample - 1]);
        return uchar((this_sum * 3u + last_sum + 8u) >> 4);
    }
    const uint next_sum = h2v2_weighted_sample_sum(curr_row[sample + 1], near_row[sample + 1]);
    return uchar((this_sum * 3u + next_sum + 7u) >> 4);
}

inline uchar h2v2_sample_thread_local(
    thread const uchar *near_row,
    thread const uchar *curr_row,
    uint n,
    uint x,
    uint local_sample_base,
    uchar left_near_sample,
    uchar left_curr_sample
) {
    if (n == 0) {
        return 0;
    }
    const uint sample = min(x / 2, n - 1);
    const uint local_sample = sample - local_sample_base;
    const uint this_sum = h2v2_weighted_sample_sum(curr_row[local_sample], near_row[local_sample]);
    if (n == 1) {
        return uchar((4u * this_sum + 8u) >> 4);
    }
    if (x == 0) {
        return uchar((this_sum * 4u + 8u) >> 4);
    }
    if (x == n * 2u - 1u) {
        return uchar((this_sum * 4u + 7u) >> 4);
    }
    if ((x & 1u) == 0u) {
        const uint last_sum = local_sample == 0u
            ? h2v2_weighted_sample_sum(left_curr_sample, left_near_sample)
            : h2v2_weighted_sample_sum(curr_row[local_sample - 1], near_row[local_sample - 1]);
        return uchar((this_sum * 3u + last_sum + 8u) >> 4);
    }
    const uint next_sum = h2v2_weighted_sample_sum(curr_row[local_sample + 1], near_row[local_sample + 1]);
    return uchar((this_sum * 3u + next_sum + 7u) >> 4);
}

inline uint fast420_boundary_meta_base(uint record_index) {
    return record_index * 4u;
}

inline uint fast420_boundary_sample_base(uint record_index) {
    return record_index * 64u;
}

inline uint fast420_vertical_meta_base(uint record_index) {
    return record_index * 4u;
}

inline uint fast420_vertical_sample_base(uint record_index) {
    return record_index * 64u;
}

inline uchar h2v2_sample_device_local(
    device const uchar *near_row,
    device const uchar *curr_row,
    uint n,
    uint x,
    uint local_sample_base
) {
    if (n == 0) {
        return 0;
    }
    const uint sample = min(x / 2u, n - 1u);
    const uint local_sample = sample - local_sample_base;
    const uint this_sum = h2v2_weighted_sample_sum(curr_row[local_sample], near_row[local_sample]);
    if (n == 1) {
        return uchar((4u * this_sum + 8u) >> 4);
    }
    if (x == 0u) {
        return uchar((this_sum * 4u + 8u) >> 4);
    }
    if (x == n * 2u - 1u) {
        return uchar((this_sum * 4u + 7u) >> 4);
    }
    if ((x & 1u) == 0u) {
        const uint last_sum = h2v2_weighted_sample_sum(curr_row[local_sample - 1u], near_row[local_sample - 1u]);
        return uchar((this_sum * 3u + last_sum + 8u) >> 4);
    }
    const uint next_sum = h2v2_weighted_sample_sum(curr_row[local_sample + 1u], near_row[local_sample + 1u]);
    return uchar((this_sum * 3u + next_sum + 7u) >> 4);
}

inline uchar h2v2_boundary_left_from_sums(uint left_sum, uint right_sum) {
    return uchar((left_sum * 3u + right_sum + 7u) >> 4);
}

inline uchar h2v2_boundary_right_from_sums(uint left_sum, uint right_sum) {
    return uchar((right_sum * 3u + left_sum + 8u) >> 4);
}

inline uchar h2v2_corner_sample(
    uchar top_left,
    uchar top_right,
    uchar bottom_left,
    uchar bottom_right,
    bool bottom_row,
    bool right_column
) {
    const uint left_sum = bottom_row
        ? h2v2_weighted_sample_sum(bottom_left, top_left)
        : h2v2_weighted_sample_sum(top_left, bottom_left);
    const uint right_sum = bottom_row
        ? h2v2_weighted_sample_sum(bottom_right, top_right)
        : h2v2_weighted_sample_sum(top_right, bottom_right);
    return right_column
        ? h2v2_boundary_right_from_sums(left_sum, right_sum)
        : h2v2_boundary_left_from_sums(left_sum, right_sum);
}

inline uchar h2v1_sample(
    device const uchar *row,
    uint n,
    uint x
) {
    if (n == 0) {
        return 0;
    }
    if (n == 1) {
        return row[0];
    }
    const uint sample = min(x / 2u, n - 1u);
    if (x == 0u) {
        return row[0];
    }
    if (x == n * 2u - 1u) {
        return row[n - 1u];
    }
    if ((x & 1u) == 0u) {
        const uint prev = uint(row[sample - 1u]);
        const uint curr = uint(row[sample]);
        return uchar((3u * curr + prev + 2u) >> 2);
    }
    const uint curr = uint(row[sample]);
    const uint next = uint(row[sample + 1u]);
    return uchar((3u * curr + next + 2u) >> 2);
}

inline uchar h2v1_sample_thread(
    thread const uchar *row,
    uint n,
    uint x
) {
    if (n == 0) {
        return 0;
    }
    if (n == 1) {
        return row[0];
    }
    const uint sample = min(x / 2u, n - 1u);
    if (x == 0u) {
        return row[0];
    }
    if (x == n * 2u - 1u) {
        return row[n - 1u];
    }
    if ((x & 1u) == 0u) {
        const uint prev = uint(row[sample - 1u]);
        const uint curr = uint(row[sample]);
        return uchar((3u * curr + prev + 2u) >> 2);
    }
    const uint curr = uint(row[sample]);
    const uint next = uint(row[sample + 1u]);
    return uchar((3u * curr + next + 2u) >> 2);
}

inline uchar h2v1_sample_thread_local(
    thread const uchar *row,
    uint n,
    uint x,
    uint local_sample_base,
    uchar left_sample
) {
    if (n == 0) {
        return 0;
    }
    if (n == 1) {
        return row[0];
    }
    const uint sample = min(x / 2u, n - 1u);
    const uint local_sample = sample - local_sample_base;
    if (x == 0u) {
        return row[0];
    }
    if (x == n * 2u - 1u) {
        return row[local_sample];
    }
    if ((x & 1u) == 0u) {
        const uint prev = local_sample == 0u ? uint(left_sample) : uint(row[local_sample - 1u]);
        const uint curr = uint(row[local_sample]);
        return uchar((3u * curr + prev + 2u) >> 2);
    }
    const uint curr = uint(row[local_sample]);
    const uint next = uint(row[local_sample + 1u]);
    return uchar((3u * curr + next + 2u) >> 2);
}

inline uint fast422_boundary_meta_base(uint record_index) {
    return record_index * 4u;
}

inline uint fast422_boundary_sample_base(uint record_index) {
    return record_index * 48u;
}

inline void h2v2_sample_even_pair(
    device const uchar *near_row,
    device const uchar *curr_row,
    uint n,
    uint x,
    thread uchar &left,
    thread uchar &right
) {
    if (n <= 1u) {
        left = h2v2_sample(near_row, curr_row, n, x);
        right = h2v2_sample(near_row, curr_row, n, x + 1u);
        return;
    }

    const uint last_x = n * 2u - 1u;
    if (x == 0u || x + 1u >= last_x) {
        left = h2v2_sample(near_row, curr_row, n, x);
        right = h2v2_sample(near_row, curr_row, n, x + 1u);
        return;
    }

    const uint sample = x / 2u;
    const uint this_sum = h2v2_weighted_sample_sum(curr_row[sample], near_row[sample]);
    const uint last_sum = h2v2_weighted_sample_sum(curr_row[sample - 1u], near_row[sample - 1u]);
    const uint next_sum = h2v2_weighted_sample_sum(curr_row[sample + 1u], near_row[sample + 1u]);
    left = uchar((this_sum * 3u + last_sum + 8u) >> 4);
    right = uchar((this_sum * 3u + next_sum + 7u) >> 4);
}

inline void h2v1_sample_even_pair(
    device const uchar *row,
    uint n,
    uint x,
    thread uchar &left,
    thread uchar &right
) {
    if (n <= 1u) {
        left = h2v1_sample(row, n, x);
        right = h2v1_sample(row, n, x + 1u);
        return;
    }

    const uint last_x = n * 2u - 1u;
    if (x == 0u || x + 1u >= last_x) {
        left = h2v1_sample(row, n, x);
        right = h2v1_sample(row, n, x + 1u);
        return;
    }

    const uint sample = x / 2u;
    const uint prev = uint(row[sample - 1u]);
    const uint curr = uint(row[sample]);
    const uint next = uint(row[sample + 1u]);
    left = uchar((3u * curr + prev + 2u) >> 2);
    right = uchar((3u * curr + next + 2u) >> 2);
}

inline void jpeg_sample_420_chroma(
    device const uchar *cb_plane,
    device const uchar *cr_plane,
    uint chroma_width,
    uint chroma_height,
    uint x,
    uint y,
    thread uchar &cb,
    thread uchar &cr
) {
    const uint chroma_y = min(y / 2u, chroma_height - 1u);
    const uint near_y = (y & 1u) == 0u
        ? (chroma_y == 0u ? 0u : chroma_y - 1u)
        : min(chroma_y + 1u, chroma_height - 1u);
    device const uchar *curr_cb = cb_plane + chroma_y * chroma_width;
    device const uchar *near_cb = cb_plane + near_y * chroma_width;
    device const uchar *curr_cr = cr_plane + chroma_y * chroma_width;
    device const uchar *near_cr = cr_plane + near_y * chroma_width;

    cb = h2v2_sample(near_cb, curr_cb, chroma_width, x);
    cr = h2v2_sample(near_cr, curr_cr, chroma_width, x);
}

inline void jpeg_sample_422_chroma(
    device const uchar *cb_plane,
    device const uchar *cr_plane,
    uint chroma_width,
    uint chroma_height,
    uint x,
    uint y,
    thread uchar &cb,
    thread uchar &cr
) {
    const uint chroma_y = min(y, chroma_height - 1u);
    device const uchar *curr_cb = cb_plane + chroma_y * chroma_width;
    device const uchar *curr_cr = cr_plane + chroma_y * chroma_width;

    cb = h2v1_sample(curr_cb, chroma_width, x);
    cr = h2v1_sample(curr_cr, chroma_width, x);
}

inline void store_rgb_ycbcr(
    device uchar *out,
    uint out_idx,
    uchar y_value,
    uchar cb_value,
    uchar cr_value
) {
    const int y = int(y_value);
    const int cb_centered = int(cb_value) - 128;
    const int cr_centered = int(cr_value) - 128;
    out[out_idx] = clamp_u8(y + ((91881 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 1u] = clamp_u8(y - ((22554 * cb_centered + 46802 * cr_centered + (1 << 15)) >> 16));
    out[out_idx + 2u] = clamp_u8(y + ((116130 * cb_centered + (1 << 15)) >> 16));
}

inline void store_rgba_ycbcr(
    device uchar *out,
    uint out_idx,
    uchar y_value,
    uchar cb_value,
    uchar cr_value,
    uint alpha
) {
    store_rgb_ycbcr(out, out_idx, y_value, cb_value, cr_value);
    out[out_idx + 3u] = uchar(alpha);
}

inline float4 rgba_float_ycbcr(
    uchar y_value,
    uchar cb_value,
    uchar cr_value,
    uint alpha
) {
    const int y = int(y_value);
    const int cb_centered = int(cb_value) - 128;
    const int cr_centered = int(cr_value) - 128;
    const uchar r = clamp_u8(y + ((91881 * cr_centered + (1 << 15)) >> 16));
    const uchar g = clamp_u8(y - ((22554 * cb_centered + 46802 * cr_centered + (1 << 15)) >> 16));
    const uchar b = clamp_u8(y + ((116130 * cb_centered + (1 << 15)) >> 16));
    return float4(float(r), float(g), float(b), float(uchar(alpha))) / 255.0f;
}

inline float4 rgba_float_direct(
    uchar r,
    uchar g,
    uchar b,
    uint alpha
) {
    return float4(float(r), float(g), float(b), float(uchar(alpha))) / 255.0f;
}
