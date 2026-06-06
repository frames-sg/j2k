#include <math.h>

struct SigninumJpeg420Params {
    unsigned int width;
    unsigned int height;
    unsigned int mcus_per_row;
    unsigned int mcu_rows;
    unsigned int entropy_len;
    unsigned int checkpoint_count;
    unsigned int out_stride;
    unsigned int reserved;
};

struct SigninumJpegEntropyCheckpoint {
    unsigned int mcu_index;
    unsigned int entropy_pos;
    unsigned long long bit_acc;
    unsigned int bit_count;
    int y_prev_dc;
    int cb_prev_dc;
    int cr_prev_dc;
    unsigned int reserved;
};

struct SigninumJpegHuffmanTable {
    int max_code[17];
    int val_offset[17];
    unsigned char values[256];
    unsigned int values_len;
};

struct SigninumJpegDecodeStatus {
    unsigned int code;
    unsigned int detail;
    unsigned int position;
    unsigned int reserved;
};

struct SigninumJpegEntropyChunkParams {
    unsigned int entropy_len;
    unsigned int entropy_bits;
    unsigned int subsequence_bits;
    unsigned int subsequence_count;
    unsigned int sequence_len;
    unsigned int max_overflow_subsequences;
    unsigned int reserved0;
    unsigned int reserved1;
};

struct SigninumJpegEntropySyncState {
    unsigned int code;
    unsigned int start_bit;
    unsigned int end_bit;
    unsigned int bit_pos;
    unsigned int symbol_count;
    unsigned int block_phase;
    unsigned int zigzag_index;
    unsigned int reserved;
};

struct SigninumJpegEntropyOverflowState {
    unsigned int code;
    unsigned int from_subsequence;
    unsigned int to_subsequence;
    unsigned int overflow_bits;
    unsigned int synchronized;
    unsigned int reserved[3];
};

struct SigninumJpegBitReader {
    unsigned int pos;
    unsigned long long acc;
    unsigned int bits;
};

static constexpr unsigned int JPEG_STATUS_OK = 0u;
static constexpr unsigned int JPEG_STATUS_TRUNCATED = 1u;
static constexpr unsigned int JPEG_STATUS_HUFFMAN = 2u;

__device__ __constant__ unsigned char SIGNINUM_JPEG_ZIGZAG[64] = {
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

__device__ unsigned int signinum_jpeg_zigzag(unsigned int k) {
    return SIGNINUM_JPEG_ZIGZAG[k];
}

__device__ void signinum_jpeg_set_error(
    SigninumJpegDecodeStatus *status,
    unsigned int code,
    unsigned int detail,
    unsigned int position
) {
    status->code = code;
    status->detail = detail;
    status->position = position;
}

__device__ bool signinum_jpeg_refill_one(
    SigninumJpegBitReader &reader,
    const unsigned char *entropy,
    unsigned int entropy_len
) {
    if (reader.pos >= entropy_len) {
        return false;
    }
    const unsigned int shift = 64u - 8u - reader.bits;
    reader.acc |= static_cast<unsigned long long>(entropy[reader.pos]) << shift;
    reader.pos += 1u;
    reader.bits += 8u;
    return true;
}

__device__ bool signinum_jpeg_ensure_bits(
    SigninumJpegBitReader &reader,
    const unsigned char *entropy,
    unsigned int entropy_len,
    unsigned int wanted
) {
    while (reader.bits < wanted) {
        if (!signinum_jpeg_refill_one(reader, entropy, entropy_len)) {
            return false;
        }
    }
    return true;
}

__device__ void signinum_jpeg_ensure_bits_padded(
    SigninumJpegBitReader &reader,
    const unsigned char *entropy,
    unsigned int entropy_len,
    unsigned int wanted
) {
    while (reader.bits < wanted) {
        if (!signinum_jpeg_refill_one(reader, entropy, entropy_len)) {
            reader.acc |= 1ull << (63u - reader.bits);
            reader.bits += 1u;
        }
    }
}

__device__ unsigned int signinum_jpeg_peek_bits(
    const SigninumJpegBitReader &reader,
    unsigned int count
) {
    return count == 0u ? 0u : static_cast<unsigned int>(reader.acc >> (64u - count));
}

__device__ void signinum_jpeg_consume_bits(SigninumJpegBitReader &reader, unsigned int count) {
    reader.acc <<= count;
    reader.bits -= count;
}

__device__ SigninumJpegBitReader signinum_jpeg_bit_reader_at_bit(
    const unsigned char *entropy,
    unsigned int entropy_len,
    unsigned int bit_pos
) {
    SigninumJpegBitReader reader;
    reader.pos = bit_pos / 8u;
    reader.acc = 0ull;
    reader.bits = 0u;
    const unsigned int skip = bit_pos & 7u;
    if (skip != 0u && reader.pos < entropy_len) {
        reader.acc = static_cast<unsigned long long>(entropy[reader.pos]) << 56u;
        reader.pos += 1u;
        reader.bits = 8u;
        signinum_jpeg_consume_bits(reader, skip);
    }
    return reader;
}

__device__ bool signinum_jpeg_real_bits_consumed(
    const SigninumJpegBitReader &reader,
    unsigned int before_pos,
    unsigned int before_bits,
    unsigned int &consumed
) {
    const unsigned int loaded_bits = (reader.pos - before_pos) * 8u + before_bits;
    if (reader.bits >= loaded_bits) {
        consumed = 0u;
        return false;
    }
    consumed = loaded_bits - reader.bits;
    return true;
}

__device__ bool signinum_jpeg_receive_extend(
    SigninumJpegBitReader &reader,
    const unsigned char *entropy,
    unsigned int entropy_len,
    unsigned int ssss,
    SigninumJpegDecodeStatus *status,
    int &out
) {
    if (ssss == 0u) {
        out = 0;
        return true;
    }
    if (!signinum_jpeg_ensure_bits(reader, entropy, entropy_len, ssss)) {
        signinum_jpeg_set_error(status, JPEG_STATUS_TRUNCATED, ssss, reader.pos);
        return false;
    }
    const int value = static_cast<int>(signinum_jpeg_peek_bits(reader, ssss));
    signinum_jpeg_consume_bits(reader, ssss);
    const int threshold = 1 << (ssss - 1u);
    out = value < threshold ? value + ((-1) << ssss) + 1 : value;
    return true;
}

__device__ bool signinum_jpeg_decode_symbol(
    SigninumJpegBitReader &reader,
    const unsigned char *entropy,
    unsigned int entropy_len,
    const SigninumJpegHuffmanTable *table,
    SigninumJpegDecodeStatus *status,
    unsigned char &symbol
) {
    signinum_jpeg_ensure_bits_padded(reader, entropy, entropy_len, 16u);
    const int code16 = static_cast<int>(signinum_jpeg_peek_bits(reader, 16u));
    for (unsigned int len = 1u; len <= 16u; ++len) {
        if (table->max_code[len] < 0) {
            continue;
        }
        const int code = code16 >> (16u - len);
        if (code <= table->max_code[len]) {
            const int idx = code + table->val_offset[len];
            if (idx < 0 || static_cast<unsigned int>(idx) >= table->values_len) {
                signinum_jpeg_set_error(status, JPEG_STATUS_HUFFMAN, len, reader.pos);
                return false;
            }
            signinum_jpeg_consume_bits(reader, len);
            symbol = table->values[idx];
            return true;
        }
    }
    signinum_jpeg_set_error(status, JPEG_STATUS_HUFFMAN, 16u, reader.pos);
    return false;
}

__device__ bool signinum_jpeg_decode_symbol_real(
    SigninumJpegBitReader &reader,
    const unsigned char *entropy,
    unsigned int entropy_len,
    const SigninumJpegHuffmanTable *table,
    SigninumJpegDecodeStatus *status,
    unsigned char &symbol
) {
    for (unsigned int len = 1u; len <= 16u; ++len) {
        if (!signinum_jpeg_ensure_bits(reader, entropy, entropy_len, len)) {
            signinum_jpeg_set_error(status, JPEG_STATUS_TRUNCATED, len, reader.pos);
            return false;
        }
        if (table->max_code[len] < 0) {
            continue;
        }
        const int code = static_cast<int>(signinum_jpeg_peek_bits(reader, len));
        if (code <= table->max_code[len]) {
            const int idx = code + table->val_offset[len];
            if (idx < 0 || static_cast<unsigned int>(idx) >= table->values_len) {
                signinum_jpeg_set_error(status, JPEG_STATUS_HUFFMAN, len, reader.pos);
                return false;
            }
            signinum_jpeg_consume_bits(reader, len);
            symbol = table->values[idx];
            return true;
        }
    }
    signinum_jpeg_set_error(status, JPEG_STATUS_HUFFMAN, 16u, reader.pos);
    return false;
}

__device__ bool signinum_jpeg_decode_block(
    SigninumJpegBitReader &reader,
    const unsigned char *entropy,
    unsigned int entropy_len,
    const SigninumJpegHuffmanTable *dc_table,
    const SigninumJpegHuffmanTable *ac_table,
    const unsigned short *quant,
    int &prev_dc,
    SigninumJpegDecodeStatus *status,
    int coeffs[64]
) {
    for (unsigned int i = 0u; i < 64u; ++i) {
        coeffs[i] = 0;
    }

    unsigned char ssss = 0u;
    if (!signinum_jpeg_decode_symbol(reader, entropy, entropy_len, dc_table, status, ssss)) {
        return false;
    }
    if (ssss > 15u) {
        signinum_jpeg_set_error(status, JPEG_STATUS_HUFFMAN, ssss, reader.pos);
        return false;
    }
    int diff = 0;
    if (!signinum_jpeg_receive_extend(reader, entropy, entropy_len, ssss, status, diff)) {
        return false;
    }
    prev_dc += diff;
    coeffs[0] = prev_dc * static_cast<int>(quant[0]);

    unsigned int k = 1u;
    while (k < 64u) {
        unsigned char packed = 0u;
        if (!signinum_jpeg_decode_symbol(reader, entropy, entropy_len, ac_table, status, packed)) {
            return false;
        }
        const unsigned int run = static_cast<unsigned int>(packed >> 4u);
        ssss = packed & 0x0Fu;
        if (ssss == 0u) {
            if (run == 15u) {
                k += 16u;
                continue;
            }
            break;
        }
        k += run;
        if (k >= 64u) {
            signinum_jpeg_set_error(status, JPEG_STATUS_HUFFMAN, k, reader.pos);
            return false;
        }
        int value = 0;
        if (!signinum_jpeg_receive_extend(reader, entropy, entropy_len, ssss, status, value)) {
            return false;
        }
        coeffs[signinum_jpeg_zigzag(k)] = value * static_cast<int>(quant[k]);
        k += 1u;
    }
    return true;
}

extern "C" __global__ void signinum_jpeg_entropy_sync420(
    const unsigned char *entropy,
    SigninumJpegEntropyChunkParams params,
    const SigninumJpegHuffmanTable *y_dc,
    const SigninumJpegHuffmanTable *y_ac,
    const SigninumJpegHuffmanTable *cb_dc,
    const SigninumJpegHuffmanTable *cb_ac,
    const SigninumJpegHuffmanTable *cr_dc,
    const SigninumJpegHuffmanTable *cr_ac,
    SigninumJpegEntropySyncState *states
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.subsequence_count) {
        return;
    }

    SigninumJpegEntropySyncState state;
    state.code = JPEG_STATUS_OK;
    state.start_bit = gid * params.subsequence_bits;
    if (state.start_bit >= params.entropy_bits) {
        state.end_bit = params.entropy_bits;
    } else {
        const unsigned int remaining_bits = params.entropy_bits - state.start_bit;
        state.end_bit = state.start_bit + min(params.subsequence_bits, remaining_bits);
    }
    state.bit_pos = state.start_bit;
    state.symbol_count = 0u;
    state.block_phase = 0u;
    state.zigzag_index = 0u;
    state.reserved = 0u;

    SigninumJpegBitReader reader =
        signinum_jpeg_bit_reader_at_bit(entropy, params.entropy_len, state.start_bit);
    SigninumJpegDecodeStatus status;
    status.code = JPEG_STATUS_OK;
    status.detail = 0u;
    status.position = 0u;
    status.reserved = 0u;

    while (state.bit_pos < state.end_bit && status.code == JPEG_STATUS_OK) {
        const bool dc = state.zigzag_index == 0u;
        const SigninumJpegHuffmanTable *table =
            state.block_phase < 4u
                ? (dc ? y_dc : y_ac)
                : (state.block_phase == 4u ? (dc ? cb_dc : cb_ac) : (dc ? cr_dc : cr_ac));
        unsigned char symbol = 0u;
        const unsigned int before_pos = reader.pos;
        const unsigned int before_bits = reader.bits;
        if (!signinum_jpeg_decode_symbol_real(reader, entropy, params.entropy_len, table, &status, symbol)) {
            break;
        }
        const unsigned int run = symbol >> 4u;
        const unsigned int ssss = symbol & 0x0Fu;
        if (!dc && ssss == 0u && run == 15u && state.zigzag_index + 16u > 64u) {
            signinum_jpeg_set_error(&status, JPEG_STATUS_HUFFMAN, state.zigzag_index + 16u, reader.pos);
            break;
        }
        if (!dc && ssss != 0u && state.zigzag_index + run >= 64u) {
            signinum_jpeg_set_error(&status, JPEG_STATUS_HUFFMAN, state.zigzag_index + run, reader.pos);
            break;
        }
        unsigned int coeff_bits = dc ? symbol : (symbol & 0x0Fu);
        if (coeff_bits > 15u) {
            signinum_jpeg_set_error(&status, JPEG_STATUS_HUFFMAN, coeff_bits, reader.pos);
            break;
        }
        if (!signinum_jpeg_ensure_bits(reader, entropy, params.entropy_len, coeff_bits)) {
            signinum_jpeg_set_error(&status, JPEG_STATUS_TRUNCATED, coeff_bits, reader.pos);
            break;
        }
        signinum_jpeg_consume_bits(reader, coeff_bits);
        unsigned int consumed = 0u;
        if (!signinum_jpeg_real_bits_consumed(reader, before_pos, before_bits, consumed)) {
            signinum_jpeg_set_error(&status, JPEG_STATUS_TRUNCATED, 0u, reader.pos);
            break;
        }
        state.bit_pos += consumed;
        if (dc) {
            state.zigzag_index = 1u;
            state.symbol_count += 1u;
            continue;
        }
        if (ssss == 0u && run != 15u) {
            state.symbol_count += 64u - state.zigzag_index;
            state.zigzag_index = 0u;
            state.block_phase = (state.block_phase + 1u) % 6u;
            continue;
        }
        state.zigzag_index += run + 1u;
        state.symbol_count += run + 1u;
        if (state.zigzag_index >= 64u) {
            state.zigzag_index = 0u;
            state.block_phase = (state.block_phase + 1u) % 6u;
        }
    }
    state.code = status.code;
    states[gid] = state;
}

extern "C" __global__ void signinum_jpeg_entropy_overflow420(
    SigninumJpegEntropyChunkParams params,
    const SigninumJpegEntropySyncState *states,
    SigninumJpegEntropyOverflowState *overflows
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (params.subsequence_count <= 1u || params.max_overflow_subsequences == 0u) {
        return;
    }
    const unsigned int overflow_count = params.subsequence_count - 1u;
    if (gid >= overflow_count) {
        return;
    }
    (void)states;
    (void)overflows;
}

static constexpr int JPEG_CONST_BITS = 13;
static constexpr int JPEG_PASS1_BITS = 2;
static constexpr int JPEG_FIX_0_298631336 = 2446;
static constexpr int JPEG_FIX_0_390180644 = 3196;
static constexpr int JPEG_FIX_0_541196100 = 4433;
static constexpr int JPEG_FIX_0_765366865 = 6270;
static constexpr int JPEG_FIX_0_899976223 = 7373;
static constexpr int JPEG_FIX_1_175875602 = 9633;
static constexpr int JPEG_FIX_1_501321110 = 12299;
static constexpr int JPEG_FIX_1_847759065 = 15137;
static constexpr int JPEG_FIX_1_961570560 = 16069;
static constexpr int JPEG_FIX_2_053119869 = 16819;
static constexpr int JPEG_FIX_2_562915447 = 20995;
static constexpr int JPEG_FIX_3_072711026 = 25172;

__device__ unsigned char signinum_jpeg_clamp_i32(int value) {
    return static_cast<unsigned char>(value < 0 ? 0 : (value > 255 ? 255 : value));
}

__device__ int signinum_jpeg_descale(int value, int shift) {
    return value >> shift;
}

__device__ unsigned char signinum_jpeg_descale_and_clamp(int value, int shift) {
    return signinum_jpeg_clamp_i32((value >> shift) + 128);
}

__device__ void signinum_jpeg_idct_column(const int input[64], int work[64], unsigned int col) {
    const int p0 = input[col];
    const int p1 = input[col + 8u];
    const int p2 = input[col + 16u];
    const int p3 = input[col + 24u];
    const int p4 = input[col + 32u];
    const int p5 = input[col + 40u];
    const int p6 = input[col + 48u];
    const int p7 = input[col + 56u];

    if (p1 == 0 && p2 == 0 && p3 == 0 && p4 == 0 && p5 == 0 && p6 == 0 && p7 == 0) {
        const int dc = p0 << JPEG_PASS1_BITS;
        work[col] = dc;
        work[col + 8u] = dc;
        work[col + 16u] = dc;
        work[col + 24u] = dc;
        work[col + 32u] = dc;
        work[col + 40u] = dc;
        work[col + 48u] = dc;
        work[col + 56u] = dc;
        return;
    }

    int z2 = p2;
    int z3 = p6;
    int z1 = (z2 + z3) * JPEG_FIX_0_541196100;
    int tmp2 = z1 - z3 * JPEG_FIX_1_847759065;
    int tmp3 = z1 + z2 * JPEG_FIX_0_765366865;

    z2 = p0;
    z3 = p4;
    int tmp0 = (z2 + z3) << JPEG_CONST_BITS;
    int tmp1 = (z2 - z3) << JPEG_CONST_BITS;

    int tmp10 = tmp0 + tmp3;
    int tmp13 = tmp0 - tmp3;
    int tmp11 = tmp1 + tmp2;
    int tmp12 = tmp1 - tmp2;

    tmp0 = p7;
    tmp1 = p5;
    tmp2 = p3;
    tmp3 = p1;

    z1 = tmp0 + tmp3;
    z2 = tmp1 + tmp2;
    z3 = tmp0 + tmp2;
    int z4 = tmp1 + tmp3;
    int z5 = (z3 + z4) * JPEG_FIX_1_175875602;

    tmp0 *= JPEG_FIX_0_298631336;
    tmp1 *= JPEG_FIX_2_053119869;
    tmp2 *= JPEG_FIX_3_072711026;
    tmp3 *= JPEG_FIX_1_501321110;
    z1 *= -JPEG_FIX_0_899976223;
    z2 *= -JPEG_FIX_2_562915447;
    z3 *= -JPEG_FIX_1_961570560;
    z4 *= -JPEG_FIX_0_390180644;

    z3 += z5;
    z4 += z5;

    tmp0 += z1 + z3;
    tmp1 += z2 + z4;
    tmp2 += z2 + z3;
    tmp3 += z1 + z4;

    const int shift = JPEG_CONST_BITS - JPEG_PASS1_BITS;
    const int rounding = 1 << (shift - 1);
    work[col] = signinum_jpeg_descale(tmp10 + tmp3 + rounding, shift);
    work[col + 56u] = signinum_jpeg_descale(tmp10 - tmp3 + rounding, shift);
    work[col + 8u] = signinum_jpeg_descale(tmp11 + tmp2 + rounding, shift);
    work[col + 48u] = signinum_jpeg_descale(tmp11 - tmp2 + rounding, shift);
    work[col + 16u] = signinum_jpeg_descale(tmp12 + tmp1 + rounding, shift);
    work[col + 40u] = signinum_jpeg_descale(tmp12 - tmp1 + rounding, shift);
    work[col + 24u] = signinum_jpeg_descale(tmp13 + tmp0 + rounding, shift);
    work[col + 32u] = signinum_jpeg_descale(tmp13 - tmp0 + rounding, shift);
}

__device__ void signinum_jpeg_idct_row(const int work[64], unsigned char pixels[64], unsigned int row) {
    const unsigned int base = row * 8u;
    const int p0 = work[base];
    const int p1 = work[base + 1u];
    const int p2 = work[base + 2u];
    const int p3 = work[base + 3u];
    const int p4 = work[base + 4u];
    const int p5 = work[base + 5u];
    const int p6 = work[base + 6u];
    const int p7 = work[base + 7u];

    const int shift = JPEG_CONST_BITS + JPEG_PASS1_BITS + 3;
    const int rounding = 1 << (shift - 1);

    if (p1 == 0 && p2 == 0 && p3 == 0 && p4 == 0 && p5 == 0 && p6 == 0 && p7 == 0) {
        const int dc_shift = JPEG_PASS1_BITS + 3;
        const int rounding_dc = 1 << (dc_shift - 1);
        const unsigned char pixel = signinum_jpeg_descale_and_clamp(p0 + rounding_dc, dc_shift);
        for (unsigned int i = 0u; i < 8u; ++i) {
            pixels[base + i] = pixel;
        }
        return;
    }

    int z2 = p2;
    int z3 = p6;
    int z1 = (z2 + z3) * JPEG_FIX_0_541196100;
    int tmp2 = z1 - z3 * JPEG_FIX_1_847759065;
    int tmp3 = z1 + z2 * JPEG_FIX_0_765366865;

    int tmp0 = (p0 + p4) << JPEG_CONST_BITS;
    int tmp1 = (p0 - p4) << JPEG_CONST_BITS;

    int tmp10 = tmp0 + tmp3;
    int tmp13 = tmp0 - tmp3;
    int tmp11 = tmp1 + tmp2;
    int tmp12 = tmp1 - tmp2;

    tmp0 = p7;
    tmp1 = p5;
    tmp2 = p3;
    tmp3 = p1;

    z1 = tmp0 + tmp3;
    z2 = tmp1 + tmp2;
    z3 = tmp0 + tmp2;
    int z4 = tmp1 + tmp3;
    int z5 = (z3 + z4) * JPEG_FIX_1_175875602;

    tmp0 *= JPEG_FIX_0_298631336;
    tmp1 *= JPEG_FIX_2_053119869;
    tmp2 *= JPEG_FIX_3_072711026;
    tmp3 *= JPEG_FIX_1_501321110;
    z1 *= -JPEG_FIX_0_899976223;
    z2 *= -JPEG_FIX_2_562915447;
    z3 *= -JPEG_FIX_1_961570560;
    z4 *= -JPEG_FIX_0_390180644;

    z3 += z5;
    z4 += z5;

    tmp0 += z1 + z3;
    tmp1 += z2 + z4;
    tmp2 += z2 + z3;
    tmp3 += z1 + z4;

    pixels[base] = signinum_jpeg_descale_and_clamp(tmp10 + tmp3 + rounding, shift);
    pixels[base + 7u] = signinum_jpeg_descale_and_clamp(tmp10 - tmp3 + rounding, shift);
    pixels[base + 1u] = signinum_jpeg_descale_and_clamp(tmp11 + tmp2 + rounding, shift);
    pixels[base + 6u] = signinum_jpeg_descale_and_clamp(tmp11 - tmp2 + rounding, shift);
    pixels[base + 2u] = signinum_jpeg_descale_and_clamp(tmp12 + tmp1 + rounding, shift);
    pixels[base + 5u] = signinum_jpeg_descale_and_clamp(tmp12 - tmp1 + rounding, shift);
    pixels[base + 3u] = signinum_jpeg_descale_and_clamp(tmp13 + tmp0 + rounding, shift);
    pixels[base + 4u] = signinum_jpeg_descale_and_clamp(tmp13 - tmp0 + rounding, shift);
}

__device__ void signinum_jpeg_idct_islow(const int coeffs[64], unsigned char pixels[64]) {
    int work[64];
    for (unsigned int col = 0u; col < 8u; ++col) {
        signinum_jpeg_idct_column(coeffs, work, col);
    }
    for (unsigned int row = 0u; row < 8u; ++row) {
        signinum_jpeg_idct_row(work, pixels, row);
    }
}

__device__ unsigned char signinum_jpeg_h2v2_sample(
    const unsigned char block[64],
    unsigned int chroma_cols,
    unsigned int chroma_rows,
    unsigned int output_x,
    unsigned int chroma_y,
    bool bottom
) {
    const unsigned int n = chroma_cols == 0u ? 1u : chroma_cols;
    const unsigned int curr_y = chroma_y < chroma_rows ? chroma_y : chroma_rows - 1u;
    const unsigned int near_y = bottom
        ? (curr_y + 1u < chroma_rows ? curr_y + 1u : chroma_rows - 1u)
        : (curr_y == 0u ? 0u : curr_y - 1u);
    const unsigned int sample = min(output_x / 2u, n - 1u);
    const unsigned int curr = static_cast<unsigned int>(block[curr_y * 8u + sample]);
    const unsigned int near = static_cast<unsigned int>(block[near_y * 8u + sample]);
    const unsigned int this_sum = 3u * curr + near;
    if (n == 1u) {
        return static_cast<unsigned char>((4u * this_sum + 8u) >> 4u);
    }
    if (output_x == 0u) {
        return static_cast<unsigned char>((this_sum * 4u + 8u) >> 4u);
    }
    if (output_x == n * 2u - 1u) {
        return static_cast<unsigned char>((this_sum * 4u + 7u) >> 4u);
    }
    if ((output_x & 1u) == 0u) {
        const unsigned int last_curr = static_cast<unsigned int>(block[curr_y * 8u + sample - 1u]);
        const unsigned int last_near = static_cast<unsigned int>(block[near_y * 8u + sample - 1u]);
        const unsigned int last_sum = 3u * last_curr + last_near;
        return static_cast<unsigned char>((this_sum * 3u + last_sum + 8u) >> 4u);
    }
    const unsigned int next_sample = min(sample + 1u, n - 1u);
    const unsigned int next_curr = static_cast<unsigned int>(block[curr_y * 8u + next_sample]);
    const unsigned int next_near = static_cast<unsigned int>(block[near_y * 8u + next_sample]);
    const unsigned int next_sum = 3u * next_curr + next_near;
    return static_cast<unsigned char>((this_sum * 3u + next_sum + 7u) >> 4u);
}

__device__ unsigned char signinum_jpeg_h2v1_sample(
    const unsigned char block[64],
    unsigned int chroma_cols,
    unsigned int output_x,
    unsigned int chroma_y
) {
    const unsigned int n = chroma_cols == 0u ? 1u : chroma_cols;
    const unsigned int row = min(chroma_y, 7u);
    const unsigned int base = row * 8u;
    if (n == 1u) {
        return block[base];
    }
    const unsigned int sample = min(output_x / 2u, n - 1u);
    if (output_x == 0u) {
        return block[base];
    }
    if (output_x == n * 2u - 1u) {
        return block[base + n - 1u];
    }
    const unsigned int curr = static_cast<unsigned int>(block[base + sample]);
    if ((output_x & 1u) == 0u) {
        const unsigned int prev = static_cast<unsigned int>(block[base + sample - 1u]);
        return static_cast<unsigned char>((3u * curr + prev + 2u) >> 2u);
    }
    const unsigned int next = static_cast<unsigned int>(block[base + sample + 1u]);
    return static_cast<unsigned char>((3u * curr + next + 2u) >> 2u);
}

__device__ void signinum_jpeg_ycbcr_to_rgb(
    unsigned char y,
    unsigned char cb,
    unsigned char cr,
    unsigned char &r,
    unsigned char &g,
    unsigned char &b
) {
    const int yy = static_cast<int>(y);
    const int cb_centered = static_cast<int>(cb) - 128;
    const int cr_centered = static_cast<int>(cr) - 128;
    r = signinum_jpeg_clamp_i32(yy + ((91881 * cr_centered + (1 << 15)) >> 16));
    g = signinum_jpeg_clamp_i32(yy - ((22554 * cb_centered + 46802 * cr_centered + (1 << 15)) >> 16));
    b = signinum_jpeg_clamp_i32(yy + ((116130 * cb_centered + (1 << 15)) >> 16));
}

__device__ void signinum_jpeg_store_rgb420_mcu(
    unsigned char *out,
    const SigninumJpeg420Params &params,
    unsigned int mx,
    unsigned int my,
    const unsigned char y0[64],
    const unsigned char y1[64],
    const unsigned char y2[64],
    const unsigned char y3[64],
    const unsigned char cb[64],
    const unsigned char cr[64]
) {
    const unsigned int base_x = mx * 16u;
    const unsigned int base_y = my * 16u;
    const unsigned int remaining_x = params.width > base_x ? params.width - base_x : 0u;
    const unsigned int remaining_y = params.height > base_y ? params.height - base_y : 0u;
    const unsigned int chroma_cols = min(8u, (remaining_x + 1u) / 2u);
    const unsigned int chroma_rows = min(8u, (remaining_y + 1u) / 2u);
    for (unsigned int yy = 0u; yy < 16u; ++yy) {
        const unsigned int py = base_y + yy;
        if (py >= params.height) {
            continue;
        }
        for (unsigned int xx = 0u; xx < 16u; ++xx) {
            const unsigned int px = base_x + xx;
            if (px >= params.width) {
                continue;
            }
            const unsigned char *yb = yy < 8u
                ? (xx < 8u ? y0 : y1)
                : (xx < 8u ? y2 : y3);
            const unsigned int y_idx = (yy & 7u) * 8u + (xx & 7u);
            const unsigned int chroma_y = min(yy / 2u, chroma_rows - 1u);
            const bool bottom = (yy & 1u) != 0u;
            const unsigned char cbv =
                signinum_jpeg_h2v2_sample(cb, chroma_cols, chroma_rows, xx, chroma_y, bottom);
            const unsigned char crv =
                signinum_jpeg_h2v2_sample(cr, chroma_cols, chroma_rows, xx, chroma_y, bottom);
            const unsigned int dst = py * params.out_stride + px * 3u;
            unsigned char r = 0u;
            unsigned char g = 0u;
            unsigned char b = 0u;
            signinum_jpeg_ycbcr_to_rgb(yb[y_idx], cbv, crv, r, g, b);
            out[dst] = r;
            out[dst + 1u] = g;
            out[dst + 2u] = b;
        }
    }
}

__device__ void signinum_jpeg_store_rgb422_mcu(
    unsigned char *out,
    const SigninumJpeg420Params &params,
    unsigned int mx,
    unsigned int my,
    const unsigned char y0[64],
    const unsigned char y1[64],
    const unsigned char cb[64],
    const unsigned char cr[64]
) {
    const unsigned int base_x = mx * 16u;
    const unsigned int base_y = my * 8u;
    const unsigned int remaining_x = params.width > base_x ? params.width - base_x : 0u;
    const unsigned int remaining_y = params.height > base_y ? params.height - base_y : 0u;
    const unsigned int chroma_cols = min(8u, (remaining_x + 1u) / 2u);
    const unsigned int chroma_rows = min(8u, remaining_y);
    for (unsigned int yy = 0u; yy < 8u; ++yy) {
        const unsigned int py = base_y + yy;
        if (py >= params.height) {
            continue;
        }
        const unsigned int chroma_y = min(yy, chroma_rows - 1u);
        for (unsigned int xx = 0u; xx < 16u; ++xx) {
            const unsigned int px = base_x + xx;
            if (px >= params.width) {
                continue;
            }
            const unsigned char *yb = xx < 8u ? y0 : y1;
            const unsigned int y_idx = yy * 8u + (xx & 7u);
            const unsigned char cbv = signinum_jpeg_h2v1_sample(cb, chroma_cols, xx, chroma_y);
            const unsigned char crv = signinum_jpeg_h2v1_sample(cr, chroma_cols, xx, chroma_y);
            const unsigned int dst = py * params.out_stride + px * 3u;
            unsigned char r = 0u;
            unsigned char g = 0u;
            unsigned char b = 0u;
            signinum_jpeg_ycbcr_to_rgb(yb[y_idx], cbv, crv, r, g, b);
            out[dst] = r;
            out[dst + 1u] = g;
            out[dst + 2u] = b;
        }
    }
}

__device__ void signinum_jpeg_store_rgb444_mcu(
    unsigned char *out,
    const SigninumJpeg420Params &params,
    unsigned int mx,
    unsigned int my,
    const unsigned char y[64],
    const unsigned char cb[64],
    const unsigned char cr[64]
) {
    const unsigned int base_x = mx * 8u;
    const unsigned int base_y = my * 8u;
    for (unsigned int yy = 0u; yy < 8u; ++yy) {
        const unsigned int py = base_y + yy;
        if (py >= params.height) {
            continue;
        }
        for (unsigned int xx = 0u; xx < 8u; ++xx) {
            const unsigned int px = base_x + xx;
            if (px >= params.width) {
                continue;
            }
            const unsigned int idx = yy * 8u + xx;
            const unsigned int dst = py * params.out_stride + px * 3u;
            unsigned char r = 0u;
            unsigned char g = 0u;
            unsigned char b = 0u;
            signinum_jpeg_ycbcr_to_rgb(y[idx], cb[idx], cr[idx], r, g, b);
            out[dst] = r;
            out[dst + 1u] = g;
            out[dst + 2u] = b;
        }
    }
}

extern "C" __global__ void signinum_jpeg_decode_fast420_rgb8(
    const unsigned char *entropy,
    unsigned char *out,
    SigninumJpeg420Params params,
    const unsigned short *y_quant,
    const unsigned short *cb_quant,
    const unsigned short *cr_quant,
    const SigninumJpegHuffmanTable *y_dc,
    const SigninumJpegHuffmanTable *y_ac,
    const SigninumJpegHuffmanTable *cb_dc,
    const SigninumJpegHuffmanTable *cb_ac,
    const SigninumJpegHuffmanTable *cr_dc,
    const SigninumJpegHuffmanTable *cr_ac,
    const SigninumJpegEntropyCheckpoint *checkpoints,
    SigninumJpegDecodeStatus *status
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.checkpoint_count) {
        return;
    }
    SigninumJpegDecodeStatus *thread_status = status + gid;
    thread_status->code = JPEG_STATUS_OK;
    thread_status->detail = 0u;
    thread_status->position = 0u;
    thread_status->reserved = 0u;

    const unsigned int total_mcus = params.mcus_per_row * params.mcu_rows;
    const SigninumJpegEntropyCheckpoint checkpoint = checkpoints[gid];
    unsigned int start_mcu = checkpoint.mcu_index;
    if (start_mcu >= total_mcus) {
        return;
    }
    unsigned int end_mcu = total_mcus;
    if (gid + 1u < params.checkpoint_count) {
        end_mcu = checkpoints[gid + 1u].mcu_index;
        if (end_mcu > total_mcus) {
            end_mcu = total_mcus;
        }
    }
    if (end_mcu <= start_mcu) {
        return;
    }

    SigninumJpegBitReader reader;
    reader.pos = checkpoint.entropy_pos;
    reader.acc = checkpoint.bit_acc;
    reader.bits = checkpoint.bit_count;
    int y_prev_dc = checkpoint.y_prev_dc;
    int cb_prev_dc = checkpoint.cb_prev_dc;
    int cr_prev_dc = checkpoint.cr_prev_dc;

    int coeffs[64];
    unsigned char y0[64];
    unsigned char y1[64];
    unsigned char y2[64];
    unsigned char y3[64];
    unsigned char cb[64];
    unsigned char cr[64];

    for (unsigned int mcu = start_mcu; mcu < end_mcu; ++mcu) {
        if (!signinum_jpeg_decode_block(reader, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs)) {
            return;
        }
        signinum_jpeg_idct_islow(coeffs, y0);
        if (!signinum_jpeg_decode_block(reader, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs)) {
            return;
        }
        signinum_jpeg_idct_islow(coeffs, y1);
        if (!signinum_jpeg_decode_block(reader, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs)) {
            return;
        }
        signinum_jpeg_idct_islow(coeffs, y2);
        if (!signinum_jpeg_decode_block(reader, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs)) {
            return;
        }
        signinum_jpeg_idct_islow(coeffs, y3);
        if (!signinum_jpeg_decode_block(reader, entropy, params.entropy_len, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, coeffs)) {
            return;
        }
        signinum_jpeg_idct_islow(coeffs, cb);
        if (!signinum_jpeg_decode_block(reader, entropy, params.entropy_len, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, coeffs)) {
            return;
        }
        signinum_jpeg_idct_islow(coeffs, cr);
        const unsigned int mx = mcu - (mcu / params.mcus_per_row) * params.mcus_per_row;
        const unsigned int my = mcu / params.mcus_per_row;
        signinum_jpeg_store_rgb420_mcu(out, params, mx, my, y0, y1, y2, y3, cb, cr);
    }
}

extern "C" __global__ void signinum_jpeg_decode_fast422_rgb8(
    const unsigned char *entropy,
    unsigned char *out,
    SigninumJpeg420Params params,
    const unsigned short *y_quant,
    const unsigned short *cb_quant,
    const unsigned short *cr_quant,
    const SigninumJpegHuffmanTable *y_dc,
    const SigninumJpegHuffmanTable *y_ac,
    const SigninumJpegHuffmanTable *cb_dc,
    const SigninumJpegHuffmanTable *cb_ac,
    const SigninumJpegHuffmanTable *cr_dc,
    const SigninumJpegHuffmanTable *cr_ac,
    const SigninumJpegEntropyCheckpoint *checkpoints,
    SigninumJpegDecodeStatus *status
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.checkpoint_count) {
        return;
    }
    SigninumJpegDecodeStatus *thread_status = status + gid;
    thread_status->code = JPEG_STATUS_OK;
    thread_status->detail = 0u;
    thread_status->position = 0u;
    thread_status->reserved = 0u;

    const unsigned int total_mcus = params.mcus_per_row * params.mcu_rows;
    const SigninumJpegEntropyCheckpoint checkpoint = checkpoints[gid];
    unsigned int start_mcu = checkpoint.mcu_index;
    if (start_mcu >= total_mcus) {
        return;
    }
    unsigned int end_mcu = total_mcus;
    if (gid + 1u < params.checkpoint_count) {
        end_mcu = checkpoints[gid + 1u].mcu_index;
        if (end_mcu > total_mcus) {
            end_mcu = total_mcus;
        }
    }
    if (end_mcu <= start_mcu) {
        return;
    }

    SigninumJpegBitReader reader;
    reader.pos = checkpoint.entropy_pos;
    reader.acc = checkpoint.bit_acc;
    reader.bits = checkpoint.bit_count;
    int y_prev_dc = checkpoint.y_prev_dc;
    int cb_prev_dc = checkpoint.cb_prev_dc;
    int cr_prev_dc = checkpoint.cr_prev_dc;

    int coeffs[64];
    unsigned char y0[64];
    unsigned char y1[64];
    unsigned char cb[64];
    unsigned char cr[64];

    for (unsigned int mcu = start_mcu; mcu < end_mcu; ++mcu) {
        if (!signinum_jpeg_decode_block(reader, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs)) {
            return;
        }
        signinum_jpeg_idct_islow(coeffs, y0);
        if (!signinum_jpeg_decode_block(reader, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs)) {
            return;
        }
        signinum_jpeg_idct_islow(coeffs, y1);
        if (!signinum_jpeg_decode_block(reader, entropy, params.entropy_len, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, coeffs)) {
            return;
        }
        signinum_jpeg_idct_islow(coeffs, cb);
        if (!signinum_jpeg_decode_block(reader, entropy, params.entropy_len, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, coeffs)) {
            return;
        }
        signinum_jpeg_idct_islow(coeffs, cr);
        const unsigned int mx = mcu - (mcu / params.mcus_per_row) * params.mcus_per_row;
        const unsigned int my = mcu / params.mcus_per_row;
        signinum_jpeg_store_rgb422_mcu(out, params, mx, my, y0, y1, cb, cr);
    }
}

extern "C" __global__ void signinum_jpeg_decode_fast444_rgb8(
    const unsigned char *entropy,
    unsigned char *out,
    SigninumJpeg420Params params,
    const unsigned short *y_quant,
    const unsigned short *cb_quant,
    const unsigned short *cr_quant,
    const SigninumJpegHuffmanTable *y_dc,
    const SigninumJpegHuffmanTable *y_ac,
    const SigninumJpegHuffmanTable *cb_dc,
    const SigninumJpegHuffmanTable *cb_ac,
    const SigninumJpegHuffmanTable *cr_dc,
    const SigninumJpegHuffmanTable *cr_ac,
    const SigninumJpegEntropyCheckpoint *checkpoints,
    SigninumJpegDecodeStatus *status
) {
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= params.checkpoint_count) {
        return;
    }
    SigninumJpegDecodeStatus *thread_status = status + gid;
    thread_status->code = JPEG_STATUS_OK;
    thread_status->detail = 0u;
    thread_status->position = 0u;
    thread_status->reserved = 0u;

    const unsigned int total_mcus = params.mcus_per_row * params.mcu_rows;
    const SigninumJpegEntropyCheckpoint checkpoint = checkpoints[gid];
    unsigned int start_mcu = checkpoint.mcu_index;
    if (start_mcu >= total_mcus) {
        return;
    }
    unsigned int end_mcu = total_mcus;
    if (gid + 1u < params.checkpoint_count) {
        end_mcu = checkpoints[gid + 1u].mcu_index;
        if (end_mcu > total_mcus) {
            end_mcu = total_mcus;
        }
    }
    if (end_mcu <= start_mcu) {
        return;
    }

    SigninumJpegBitReader reader;
    reader.pos = checkpoint.entropy_pos;
    reader.acc = checkpoint.bit_acc;
    reader.bits = checkpoint.bit_count;
    int y_prev_dc = checkpoint.y_prev_dc;
    int cb_prev_dc = checkpoint.cb_prev_dc;
    int cr_prev_dc = checkpoint.cr_prev_dc;

    int coeffs[64];
    unsigned char y[64];
    unsigned char cb[64];
    unsigned char cr[64];

    for (unsigned int mcu = start_mcu; mcu < end_mcu; ++mcu) {
        if (!signinum_jpeg_decode_block(reader, entropy, params.entropy_len, y_dc, y_ac, y_quant, y_prev_dc, thread_status, coeffs)) {
            return;
        }
        signinum_jpeg_idct_islow(coeffs, y);
        if (!signinum_jpeg_decode_block(reader, entropy, params.entropy_len, cb_dc, cb_ac, cb_quant, cb_prev_dc, thread_status, coeffs)) {
            return;
        }
        signinum_jpeg_idct_islow(coeffs, cb);
        if (!signinum_jpeg_decode_block(reader, entropy, params.entropy_len, cr_dc, cr_ac, cr_quant, cr_prev_dc, thread_status, coeffs)) {
            return;
        }
        signinum_jpeg_idct_islow(coeffs, cr);
        const unsigned int mx = mcu - (mcu / params.mcus_per_row) * params.mcus_per_row;
        const unsigned int my = mcu / params.mcus_per_row;
        signinum_jpeg_store_rgb444_mcu(out, params, mx, my, y, cb, cr);
    }
}
