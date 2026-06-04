// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <math.h>

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long long j2k_ulong;

__device__ inline uint j2k_min_u32(uint a, uint b) { return a < b ? a : b; }
__device__ inline int j2k_max_i32(int a, int b) { return a > b ? a : b; }

struct J2kHtCleanupParams {
    uint width;
    uint height;
    uint coded_len;
    uint cleanup_length;
    uint refinement_length;
    uint missing_msbs;
    uint num_bitplanes;
    uint number_of_coding_passes;
    uint output_stride;
    uint output_offset;
    float dequantization_step;
    uint stripe_causal;
};

struct J2kHtCleanupBatchJob {
    uint coded_offset;
    uint width;
    uint height;
    uint coded_len;
    uint cleanup_length;
    uint refinement_length;
    uint missing_msbs;
    uint num_bitplanes;
    uint number_of_coding_passes;
    uint output_stride;
    uint output_offset;
    float dequantization_step;
    uint stripe_causal;
};

struct J2kHtCleanupMultiBatchJob {
    j2k_ulong output_ptr;
    uint coded_offset;
    uint width;
    uint height;
    uint coded_len;
    uint cleanup_length;
    uint refinement_length;
    uint missing_msbs;
    uint num_bitplanes;
    uint number_of_coding_passes;
    uint output_stride;
    uint output_offset;
    float dequantization_step;
    uint stripe_causal;
};

struct J2kHtRepeatedBatchParams {
    uint job_count;
    uint output_plane_len;
    uint batch_count;
};

struct J2kHtStatus {
    uint code;
    uint detail;
    uint reserved0;
    uint reserved1;
};

struct J2kHtDequantizeJob {
    j2k_ulong output_ptr;
    uint width;
    uint height;
    uint output_stride;
    uint output_offset;
    uint num_bitplanes;
    uint reserved;
    float dequantization_step;
};

static constexpr uint J2K_HT_STATUS_OK = 0u;
static constexpr uint J2K_HT_STATUS_FAIL = 1u;
static constexpr uint J2K_HT_STATUS_UNSUPPORTED = 2u;

static constexpr uint J2K_HT_MAX_WIDTH = 256u;
static constexpr uint J2K_HT_MAX_HEIGHT = 256u;
static constexpr uint J2K_HT_MAX_COEFFICIENTS = 4096u;
static constexpr uint J2K_HT_MAX_SSTR = 264u;
static constexpr uint J2K_HT_MAX_SCRATCH = 3096u;
static constexpr uint J2K_HT_MAX_VN = 130u;
static constexpr uint J2K_HT_MAX_MSTR = 72u;
static constexpr uint J2K_HT_MAX_SIGMA = 528u;
static constexpr uint J2K_HT_MAX_PREV_ROW_SIG = 72u;

__device__ inline void set_ht_status(J2kHtStatus *status, uint code, uint detail) {
    status->code = code;
    status->detail = detail;
    status->reserved0 = 0u;
    status->reserved1 = 0u;
}

struct MelDecoder {
    const uchar *data;
    uint pos;
    uint remaining;
    bool unstuff;
    uchar current_byte;
    uchar bits_left;
    uint k;
    uint num_runs;
    j2k_ulong runs;
};

__device__ inline MelDecoder mel_decoder_new(const uchar *data, uint lcup, uint scup) {
    MelDecoder decoder;
    decoder.data = data;
    decoder.pos = lcup - scup;
    decoder.remaining = scup - 1u;
    decoder.unstuff = false;
    decoder.current_byte = 0;
    decoder.bits_left = 0;
    decoder.k = 0u;
    decoder.num_runs = 0u;
    decoder.runs = 0u;
    return decoder;
}

__device__ inline bool mel_read_bit(MelDecoder &decoder, uint &bit) {
    if (decoder.bits_left == 0u) {
        uchar byte = decoder.remaining > 0u ? decoder.data[decoder.pos] : uchar(0xFF);
        if (decoder.remaining > 0u) {
            decoder.pos += 1u;
            decoder.remaining -= 1u;
        }
        if (decoder.remaining == 0u) {
            byte |= uchar(0x0F);
        }
        decoder.current_byte = byte;
        decoder.bits_left = uchar(8u - uint(decoder.unstuff));
        decoder.unstuff = byte == uchar(0xFF);
    }

    decoder.bits_left -= 1u;
    bit = uint((decoder.current_byte >> decoder.bits_left) & uchar(1));
    return true;
}

__device__ inline bool mel_read_bits(MelDecoder &decoder, uint count, uint &value) {
    value = 0u;
    for (uint idx = 0u; idx < count; ++idx) {
        uint bit = 0u;
        if (!mel_read_bit(decoder, bit)) {
            return false;
        }
        value = (value << 1u) | bit;
    }
    return true;
}

__device__ inline bool mel_decode_more_runs(MelDecoder &decoder) {
    static constexpr uint MEL_EXP[13] = {0u, 0u, 0u, 1u, 1u, 1u, 2u, 2u, 2u, 3u, 3u, 4u, 5u};

    while (decoder.num_runs < 8u) {
        const uint eval = MEL_EXP[decoder.k];
        uint first = 0u;
        if (!mel_read_bit(decoder, first)) {
            return false;
        }

        uint run = 0u;
        if (first == 1u) {
            decoder.k = j2k_min_u32(decoder.k + 1u, 12u);
            run = ((1u << eval) - 1u) << 1u;
        } else {
            decoder.k = decoder.k == 0u ? 0u : decoder.k - 1u;
            uint bits = 0u;
            if (!mel_read_bits(decoder, eval, bits)) {
                return false;
            }
            run = (bits << 1u) | 1u;
        }

        decoder.runs |= (j2k_ulong(run) << (decoder.num_runs * 7u));
        decoder.num_runs += 1u;

        if (eval == 5u && first == 0u && decoder.num_runs >= 8u) {
            break;
        }
    }

    return true;
}

__device__ inline bool mel_get_run(MelDecoder &decoder, int &run) {
    if (decoder.num_runs == 0u && !mel_decode_more_runs(decoder)) {
        return false;
    }

    run = int(decoder.runs & 0x7Ful);
    decoder.runs >>= 7u;
    decoder.num_runs -= 1u;
    return true;
}

struct ForwardBitReader {
    const uchar *data;
    uint data_len;
    uint pos;
    j2k_ulong tmp;
    uint bits;
    bool unstuff;
    uchar pad;
};

__device__ inline ForwardBitReader forward_reader_new(const uchar *data, uint data_len, uchar pad) {
    ForwardBitReader reader;
    reader.data = data;
    reader.data_len = data_len;
    reader.pos = 0u;
    reader.tmp = 0ul;
    reader.bits = 0u;
    reader.unstuff = false;
    reader.pad = pad;
    return reader;
}

__device__ inline void forward_reader_fill(ForwardBitReader &reader) {
    while (reader.bits <= 32u) {
        const uchar byte = reader.pos < reader.data_len ? reader.data[reader.pos++] : reader.pad;
        reader.tmp |= (j2k_ulong(byte) << reader.bits);
        reader.bits += 8u - uint(reader.unstuff);
        reader.unstuff = byte == uchar(0xFF);
    }
}

__device__ inline uint forward_reader_fetch(ForwardBitReader &reader) {
    if (reader.bits < 32u) {
        forward_reader_fill(reader);
    }
    return uint(reader.tmp);
}

__device__ inline void forward_reader_advance(ForwardBitReader &reader, uint count) {
    reader.tmp >>= count;
    reader.bits -= count;
}

struct ReverseBitReader {
    const uchar *data;
    int pos;
    uint remaining;
    j2k_ulong tmp;
    uint bits;
    bool unstuff;
};

__device__ inline ReverseBitReader reverse_reader_new_vlc(
    const uchar *data,
    uint lcup,
    uint scup
) {
    const uchar d = data[lcup - 2u];
    const j2k_ulong tmp = j2k_ulong(d >> 4);

    ReverseBitReader reader;
    reader.data = data;
    reader.pos = int(lcup) - 3;
    reader.remaining = scup - 2u;
    reader.tmp = tmp;
    reader.bits = 4u - uint((tmp & 0x7ul) == 0x7ul);
    reader.unstuff = (d | uchar(0x0F)) > uchar(0x8F);
    return reader;
}

__device__ inline ReverseBitReader reverse_reader_new_mrp(
    const uchar *data,
    uint lcup,
    uint len2
) {
    ReverseBitReader reader;
    reader.data = data;
    reader.pos = int(lcup + len2) - 1;
    reader.remaining = len2;
    reader.tmp = 0ul;
    reader.bits = 0u;
    reader.unstuff = true;
    return reader;
}

__device__ inline void reverse_reader_fill(ReverseBitReader &reader) {
    while (reader.bits <= 32u) {
        const uchar byte = reader.remaining > 0u ? reader.data[reader.pos] : uchar(0u);
        if (reader.remaining > 0u) {
            reader.pos -= 1;
            reader.remaining -= 1u;
        }
        const uint d_bits = 8u - uint(reader.unstuff && (byte & uchar(0x7F)) == uchar(0x7F));
        reader.tmp |= (j2k_ulong(byte) << reader.bits);
        reader.bits += d_bits;
        reader.unstuff = byte > uchar(0x8F);
    }
}

__device__ inline uint reverse_reader_fetch(ReverseBitReader &reader) {
    if (reader.bits < 32u) {
        reverse_reader_fill(reader);
    }
    return uint(reader.tmp);
}

__device__ inline uint reverse_reader_advance(ReverseBitReader &reader, uint count) {
    reader.tmp >>= count;
    reader.bits -= count;
    return uint(reader.tmp);
}

__device__ inline uint read_u32_pair(const ushort *values, uint index) {
    return uint(values[index]) | (uint(values[index + 1u]) << 16u);
}

__device__ inline uint sample_mask(uint bit) {
    return 1u << (4u + bit);
}

__device__ inline int coefficient_to_i32(uint value, uint k_max) {
    const uint shift = 31u - k_max;
    const int magnitude = int((value & 0x7FFF'FFFFu) >> shift);
    return (value & 0x8000'0000u) != 0u ? -magnitude : magnitude;
}

__device__ inline float coefficient_to_float(uint value, uint k_max, float scale) {
    return float(coefficient_to_i32(value, k_max)) * scale;
}

__device__ inline uint coefficient_to_float_bits(uint value, uint k_max, float scale) {
    return __float_as_uint(coefficient_to_float(value, k_max, scale));
}

__device__ inline void decode_mag_sgn_sample_with_vn(
    ForwardBitReader &magsgn,
    uint inf,
    uint bit,
    uint uq,
    uint p,
    uint &value,
    uint &v_n
) {
    if ((inf & sample_mask(bit)) == 0u) {
        value = 0u;
        v_n = 0u;
        return;
    }

    const uint ms_val = forward_reader_fetch(magsgn);
    const uint m_n = uq - ((inf >> (12u + bit)) & 1u);
    forward_reader_advance(magsgn, m_n);

    value = ms_val << 31u;
    const uint mask = m_n == 0u ? 0u : (1u << m_n) - 1u;
    v_n = ms_val & mask;
    v_n |= ((inf >> (8u + bit)) & 1u) << m_n;
    v_n |= 1u;
    value |= (v_n + 2u) << (p - 1u);
}

template <bool CLEANUP_ONLY>
__device__ inline void decode_ht_cleanup_impl(
    const uchar *coded_data,
    uint *decoded_data,
    J2kHtCleanupParams params,
    const ushort *vlc_table0,
    const ushort *vlc_table1,
    const ushort *uvlc_table0,
    const ushort *uvlc_table1,
    J2kHtStatus *status
) {
    set_ht_status(status, J2K_HT_STATUS_OK, 0u);

    uint num_passes = params.number_of_coding_passes;
    if (num_passes > 1u && params.refinement_length == 0u) {
        num_passes = 1u;
    }
    if (CLEANUP_ONLY && params.refinement_length != 0u) {
        set_ht_status(status, J2K_HT_STATUS_UNSUPPORTED, 17u);
        return;
    }

    if (params.width == 0u || params.height == 0u) {
        return;
    }
    if (params.width > J2K_HT_MAX_WIDTH || params.height > J2K_HT_MAX_HEIGHT ||
        params.width * params.height > J2K_HT_MAX_COEFFICIENTS) {
        set_ht_status(status, J2K_HT_STATUS_UNSUPPORTED, 1u);
        return;
    }
    if (params.num_bitplanes == 0u || params.num_bitplanes > 31u) {
        set_ht_status(status, J2K_HT_STATUS_FAIL, 2u);
        return;
    }
    if (num_passes > 3u || params.missing_msbs > 30u || params.missing_msbs == 30u) {
        set_ht_status(status, J2K_HT_STATUS_FAIL, 3u);
        return;
    }
    if (params.missing_msbs == 29u && num_passes > 1u) {
        num_passes = 1u;
    }

    const uint lcup = params.cleanup_length;
    if (lcup < 2u || params.coded_len < lcup + params.refinement_length) {
        set_ht_status(status, J2K_HT_STATUS_FAIL, 4u);
        return;
    }

    const uint scup = (uint(coded_data[lcup - 1u]) << 4u) + uint(coded_data[lcup - 2u] & uchar(0x0F));
    if (scup < 2u || scup > lcup || scup > 4079u) {
        set_ht_status(status, J2K_HT_STATUS_FAIL, 5u);
        return;
    }

    const uint width = params.width;
    const uint height = params.height;
    const uint stride = params.output_stride;
    const uint quad_rows = (height + 1u) / 2u;
    const uint sstr = (width + 9u) & ~7u;
    if (sstr > J2K_HT_MAX_SSTR || sstr * (quad_rows + 1u) > J2K_HT_MAX_SCRATCH) {
        set_ht_status(status, J2K_HT_STATUS_UNSUPPORTED, 6u);
        return;
    }

    ushort scratch[J2K_HT_MAX_SCRATCH];
    uint v_n_scratch[J2K_HT_MAX_VN];

    {
        MelDecoder mel = mel_decoder_new(coded_data, lcup, scup);
        ReverseBitReader vlc = reverse_reader_new_vlc(coded_data, lcup, scup);
        int run = 0;
        if (!mel_get_run(mel, run)) {
            set_ht_status(status, J2K_HT_STATUS_FAIL, 6u);
            return;
        }

        uint c_q = 0u;
        uint row_offset = 0u;
        uint x = 0u;

        while (x < width) {
            uint vlc_val = reverse_reader_fetch(vlc);
            uint t0 = uint(vlc_table0[c_q + (vlc_val & 0x7Fu)]);
            if (c_q == 0u) {
                run -= 2;
                t0 = run == -1 ? t0 : 0u;
                if (run < 0 && !mel_get_run(mel, run)) {
                    set_ht_status(status, J2K_HT_STATUS_FAIL, 7u);
                    return;
                }
            }
            scratch[row_offset] = ushort(t0);
            x += 2u;
            c_q = ((t0 & 0x10u) << 3u) | ((t0 & 0xE0u) << 2u);
            vlc_val = reverse_reader_advance(vlc, t0 & 0x7u);

            uint t1 = uint(vlc_table0[c_q + (vlc_val & 0x7Fu)]);
            if (c_q == 0u && x < width) {
                run -= 2;
                t1 = run == -1 ? t1 : 0u;
                if (run < 0 && !mel_get_run(mel, run)) {
                    set_ht_status(status, J2K_HT_STATUS_FAIL, 8u);
                    return;
                }
            }
            if (x >= width) {
                t1 = 0u;
            }
            scratch[row_offset + 2u] = ushort(t1);
            x += 2u;
            c_q = ((t1 & 0x10u) << 3u) | ((t1 & 0xE0u) << 2u);
            vlc_val = reverse_reader_advance(vlc, t1 & 0x7u);

            uint uvlc_mode = ((t0 & 0x8u) << 3u) | ((t1 & 0x8u) << 4u);
            if (uvlc_mode == 0xC0u) {
                run -= 2;
                if (run == -1) {
                    uvlc_mode += 0x40u;
                }
                if (run < 0 && !mel_get_run(mel, run)) {
                    set_ht_status(status, J2K_HT_STATUS_FAIL, 9u);
                    return;
                }
            }

            uint uvlc_entry = uint(uvlc_table0[uvlc_mode + (vlc_val & 0x3Fu)]);
            vlc_val = reverse_reader_advance(vlc, uvlc_entry & 0x7u);
            uvlc_entry >>= 3u;
            uint len = uvlc_entry & 0xFu;
            const uint tmp = vlc_val & ((1u << len) - 1u);
            vlc_val = reverse_reader_advance(vlc, len);
            uvlc_entry >>= 4u;
            len = uvlc_entry & 0x7u;
            uvlc_entry >>= 3u;
            scratch[row_offset + 1u] = ushort(1u + (uvlc_entry & 0x7u) + (tmp & ~(0xFFu << len)));
            scratch[row_offset + 3u] = ushort(1u + (uvlc_entry >> 3u) + (tmp >> len));

            row_offset += 4u;
        }
        scratch[row_offset] = 0u;
        scratch[row_offset + 1u] = 0u;

        for (uint y = 2u; y < height; y += 2u) {
            const uint row_base = (y >> 1u) * sstr;
            const uint prev_base = row_base - sstr;
            uint local_x = 0u;
            uint local_c_q = 0u;
            uint local_row_offset = row_base;

            while (local_x < width) {
                local_c_q |= (uint(scratch[prev_base + (local_row_offset - row_base)]) & 0xA0u) << 2u;
                local_c_q |= (uint(scratch[prev_base + (local_row_offset - row_base) + 2u]) & 0x20u) << 4u;

                uint vlc_val = reverse_reader_fetch(vlc);
                uint t0 = uint(vlc_table1[local_c_q + (vlc_val & 0x7Fu)]);
                if (local_c_q == 0u) {
                    run -= 2;
                    t0 = run == -1 ? t0 : 0u;
                    if (run < 0 && !mel_get_run(mel, run)) {
                        set_ht_status(status, J2K_HT_STATUS_FAIL, 10u);
                        return;
                    }
                }
                scratch[local_row_offset] = ushort(t0);
                local_x += 2u;

                local_c_q = ((t0 & 0x40u) << 2u) | ((t0 & 0x80u) << 1u);
                local_c_q |= uint(scratch[prev_base + (local_row_offset - row_base)]) & 0x80u;
                local_c_q |= (uint(scratch[prev_base + (local_row_offset - row_base) + 2u]) & 0xA0u) << 2u;
                local_c_q |= (uint(scratch[prev_base + (local_row_offset - row_base) + 4u]) & 0x20u) << 4u;
                vlc_val = reverse_reader_advance(vlc, t0 & 0x7u);

                uint t1 = uint(vlc_table1[local_c_q + (vlc_val & 0x7Fu)]);
                if (local_c_q == 0u && local_x < width) {
                    run -= 2;
                    t1 = run == -1 ? t1 : 0u;
                    if (run < 0 && !mel_get_run(mel, run)) {
                        set_ht_status(status, J2K_HT_STATUS_FAIL, 11u);
                        return;
                    }
                }
                if (local_x >= width) {
                    t1 = 0u;
                }
                scratch[local_row_offset + 2u] = ushort(t1);
                local_x += 2u;

                local_c_q = ((t1 & 0x40u) << 2u) | ((t1 & 0x80u) << 1u);
                local_c_q |= uint(scratch[prev_base + (local_row_offset - row_base) + 2u]) & 0x80u;
                vlc_val = reverse_reader_advance(vlc, t1 & 0x7u);

                const uint uvlc_mode = ((t0 & 0x8u) << 3u) | ((t1 & 0x8u) << 4u);
                uint uvlc_entry = uint(uvlc_table1[uvlc_mode + (vlc_val & 0x3Fu)]);
                vlc_val = reverse_reader_advance(vlc, uvlc_entry & 0x7u);
                uvlc_entry >>= 3u;
                uint len = uvlc_entry & 0xFu;
                const uint tmp = vlc_val & ((1u << len) - 1u);
                vlc_val = reverse_reader_advance(vlc, len);
                uvlc_entry >>= 4u;
                len = uvlc_entry & 0x7u;
                uvlc_entry >>= 3u;
                scratch[local_row_offset + 1u] =
                    ushort((uvlc_entry & 0x7u) + (tmp & ~(0xFFu << len)));
                scratch[local_row_offset + 3u] = ushort((uvlc_entry >> 3u) + (tmp >> len));

                local_row_offset += 4u;
            }

            scratch[local_row_offset] = 0u;
            scratch[local_row_offset + 1u] = 0u;
        }
    }

    const uint p = 30u - params.missing_msbs;

    {
        ForwardBitReader magsgn = forward_reader_new(coded_data, lcup - scup, uchar(0xFF));
        const uint v_n_width = ((width + 1u) / 2u) + 2u;
        if (v_n_width > J2K_HT_MAX_VN) {
            set_ht_status(status, J2K_HT_STATUS_UNSUPPORTED, 12u);
            return;
        }

        uint prev_v_n = 0u;
        uint x = 0u;
        uint sp = 0u;
        uint vp = 0u;
        uint dp = params.output_offset;
        const bool second_row_present = height > 1u;

        while (x < width) {
            const uint inf = uint(scratch[sp]);
            const uint uq = uint(scratch[sp + 1u]);
            if (uq > params.missing_msbs + 2u) {
                set_ht_status(status, J2K_HT_STATUS_FAIL, 13u);
                return;
            }

            uint value0 = 0u;
            uint ignored_vn = 0u;
            decode_mag_sgn_sample_with_vn(magsgn, inf, 0u, uq, p, value0, ignored_vn);
            decoded_data[dp] = value0;

            uint value1 = 0u;
            uint v_n1 = 0u;
            decode_mag_sgn_sample_with_vn(magsgn, inf, 1u, uq, p, value1, v_n1);
            if (second_row_present) {
                decoded_data[dp + stride] = value1;
            }
            v_n_scratch[vp] = prev_v_n | v_n1;
            prev_v_n = 0u;
            dp += 1u;
            x += 1u;

            if (x >= width) {
                vp += 1u;
                break;
            }

            uint value2 = 0u;
            decode_mag_sgn_sample_with_vn(magsgn, inf, 2u, uq, p, value2, ignored_vn);
            decoded_data[dp] = value2;

            uint value3 = 0u;
            uint v_n3 = 0u;
            decode_mag_sgn_sample_with_vn(magsgn, inf, 3u, uq, p, value3, v_n3);
            if (second_row_present) {
                decoded_data[dp + stride] = value3;
            }
            prev_v_n = v_n3;
            dp += 1u;
            x += 1u;
            sp += 2u;
            vp += 1u;
        }
        v_n_scratch[vp] = prev_v_n;

        for (uint y = 2u; y < height; y += 2u) {
            const uint row_base = (y >> 1u) * sstr;
            uint local_sp = row_base;
            uint local_vp = 0u;
            uint local_dp = params.output_offset + y * stride;
            uint local_prev_v_n = 0u;
            uint local_x = 0u;
            const bool local_second_row_present = y + 1u < height;

            while (local_x < width) {
                const uint inf = uint(scratch[local_sp]);
                const uint u_q = uint(scratch[local_sp + 1u]);
                uint gamma = inf & 0xF0u;
                gamma &= gamma - 0x10u;
                uint emax = v_n_scratch[local_vp] | v_n_scratch[local_vp + 1u];
                emax = 31u - __clz(emax | 2u);
                const uint kappa = gamma != 0u ? emax : 1u;
                const uint uq = u_q + kappa;
                if (uq > params.missing_msbs + 2u) {
                    set_ht_status(status, J2K_HT_STATUS_FAIL, 14u);
                    return;
                }

                uint value0 = 0u;
                uint ignored_vn = 0u;
                decode_mag_sgn_sample_with_vn(magsgn, inf, 0u, uq, p, value0, ignored_vn);
                decoded_data[local_dp] = value0;

                uint value1 = 0u;
                uint v_n1 = 0u;
                decode_mag_sgn_sample_with_vn(magsgn, inf, 1u, uq, p, value1, v_n1);
                if (local_second_row_present) {
                    decoded_data[local_dp + stride] = value1;
                }
                v_n_scratch[local_vp] = local_prev_v_n | v_n1;
                local_prev_v_n = 0u;
                local_dp += 1u;
                local_x += 1u;

                if (local_x >= width) {
                    local_vp += 1u;
                    break;
                }

                uint value2 = 0u;
                decode_mag_sgn_sample_with_vn(magsgn, inf, 2u, uq, p, value2, ignored_vn);
                decoded_data[local_dp] = value2;

                uint value3 = 0u;
                uint v_n3 = 0u;
                decode_mag_sgn_sample_with_vn(magsgn, inf, 3u, uq, p, value3, v_n3);
                if (local_second_row_present) {
                    decoded_data[local_dp + stride] = value3;
                }
                local_prev_v_n = v_n3;
                local_dp += 1u;
                local_x += 1u;
                local_sp += 2u;
                local_vp += 1u;
            }

            v_n_scratch[local_vp] = local_prev_v_n;
        }
    }

    if (!CLEANUP_ONLY && num_passes > 1u) {
        const uint sigma_rows = ((height + 3u) / 4u) + 1u;
        const uint mstr = ((((width + 3u) / 4u) + 2u + 7u) & ~7u);
        const uint prev_row_len = ((width + 3u) / 4u) + 8u;
        if (mstr > J2K_HT_MAX_MSTR || sigma_rows * mstr > J2K_HT_MAX_SIGMA) {
            set_ht_status(status, J2K_HT_STATUS_UNSUPPORTED, 15u);
            return;
        }
        if (prev_row_len > J2K_HT_MAX_PREV_ROW_SIG) {
            set_ht_status(status, J2K_HT_STATUS_UNSUPPORTED, 16u);
            return;
        }

        ushort sigma[J2K_HT_MAX_SIGMA];
        ushort prev_row_sig[J2K_HT_MAX_PREV_ROW_SIG];

        uint y = 0u;
        while (y < height) {
            uint sp_base = (y >> 1u) * sstr;
            uint dp_base = (y >> 2u) * mstr;
            uint local_x = 0u;
            uint sigma_sp = sp_base;
            uint sigma_dp = dp_base;
            while (local_x < width) {
                uint t0 = ((uint(scratch[sigma_sp]) & 0x30u) >> 4u)
                    | ((uint(scratch[sigma_sp]) & 0xC0u) >> 2u);
                t0 |= ((uint(scratch[sigma_sp + 2u]) & 0x30u) << 4u)
                    | ((uint(scratch[sigma_sp + 2u]) & 0xC0u) << 6u);
                uint t1 = ((uint(scratch[sigma_sp + sstr]) & 0x30u) >> 2u)
                    | (uint(scratch[sigma_sp + sstr]) & 0xC0u);
                t1 |= ((uint(scratch[sigma_sp + sstr + 2u]) & 0x30u) << 6u)
                    | ((uint(scratch[sigma_sp + sstr + 2u]) & 0xC0u) << 8u);
                sigma[sigma_dp] = ushort(t0 | t1);
                local_x += 4u;
                sigma_sp += 4u;
                sigma_dp += 1u;
            }
            sigma[sigma_dp] = 0u;
            y += 4u;
        }

        const uint sigma_tail = ((height + 3u) / 4u) * mstr;
        for (uint i = 0u; i <= (width + 3u) / 4u; ++i) {
            sigma[sigma_tail + i] = 0u;
        }

        for (uint i = 0u; i < prev_row_len; ++i) {
            prev_row_sig[i] = 0u;
        }

        ForwardBitReader sigprop =
            forward_reader_new(coded_data + lcup, params.refinement_length, uchar(0x00));

        for (y = 0u; y < height; y += 4u) {
            uint pattern = 0xFFFFu;
            if (height - y < 4u) {
                pattern = 0x7777u;
                if (height - y < 3u) {
                    pattern = 0x3333u;
                    if (height - y < 2u) {
                        pattern = 0x1111u;
                    }
                }
            }

            uint prev = 0u;
            const uint cur_row = (y >> 2u) * mstr;
            const uint next_row = cur_row + mstr;
            const uint dpp = params.output_offset + y * stride;

            for (uint x4 = 0u; x4 < width; x4 += 4u) {
                uint col_pattern = pattern;
                int s = int(x4) + 4 - int(width);
                s = j2k_max_i32(s, 0);
                col_pattern >>= uint(s * 4);

                const uint idx = x4 >> 2u;
                const uint ps =
                    uint(prev_row_sig[idx]) | (uint(prev_row_sig[idx + 1u]) << 16u);
                const uint ns = read_u32_pair(sigma, next_row + idx);
                uint u = (ps & 0x8888'8888u) >> 3u;
                if (params.stripe_causal == 0u) {
                    u |= (ns & 0x1111'1111u) << 3u;
                }

                const uint cs = read_u32_pair(sigma, cur_row + idx);
                uint mbr = cs;
                mbr |= (cs & 0x7777'7777u) << 1u;
                mbr |= (cs & 0xEEEE'EEEEu) >> 1u;
                mbr |= u;
                const uint t = mbr;
                mbr |= t << 4u;
                mbr |= t >> 4u;
                mbr |= prev >> 12u;
                mbr &= col_pattern;
                mbr &= ~cs;

                uint new_sig = mbr;
                if (new_sig != 0u) {
                    uint cwd = forward_reader_fetch(sigprop);
                    uint cnt = 0u;
                    uint col_mask = 0xFu;
                    const uint inv_sig = ~cs & col_pattern;

                    for (uint i = 0u; i < 16u; i += 4u) {
                        if ((col_mask & new_sig) == 0u) {
                            col_mask <<= 4u;
                            continue;
                        }

                        uint sample_mask = 0x1111u & col_mask;
                        if ((new_sig & sample_mask) != 0u) {
                            new_sig &= ~sample_mask;
                            if ((cwd & 1u) != 0u) {
                                const uint t_bits = 0x33u << i;
                                new_sig |= t_bits & inv_sig;
                            }
                            cwd >>= 1u;
                            cnt += 1u;
                        }

                        sample_mask <<= 1u;
                        if ((new_sig & sample_mask) != 0u) {
                            new_sig &= ~sample_mask;
                            if ((cwd & 1u) != 0u) {
                                const uint t_bits = 0x76u << i;
                                new_sig |= t_bits & inv_sig;
                            }
                            cwd >>= 1u;
                            cnt += 1u;
                        }

                        sample_mask <<= 1u;
                        if ((new_sig & sample_mask) != 0u) {
                            new_sig &= ~sample_mask;
                            if ((cwd & 1u) != 0u) {
                                const uint t_bits = 0xECu << i;
                                new_sig |= t_bits & inv_sig;
                            }
                            cwd >>= 1u;
                            cnt += 1u;
                        }

                        sample_mask <<= 1u;
                        if ((new_sig & sample_mask) != 0u) {
                            new_sig &= ~sample_mask;
                            if ((cwd & 1u) != 0u) {
                                const uint t_bits = 0xC8u << i;
                                new_sig |= t_bits & inv_sig;
                            }
                            cwd >>= 1u;
                            cnt += 1u;
                        }

                        col_mask <<= 4u;
                    }

                    if (new_sig != 0u) {
                        uint sig_dp = dpp + x4;
                        const uint value = 3u << (p - 2u);
                        col_mask = 0xFu;

                        for (uint column = 0u; column < 4u; ++column) {
                            if ((col_mask & new_sig) == 0u) {
                                col_mask <<= 4u;
                                sig_dp += 1u;
                                continue;
                            }

                            uint sample_mask = 0x1111u & col_mask;
                            if ((new_sig & sample_mask) != 0u) {
                                decoded_data[sig_dp] = (cwd << 31u) | value;
                                cwd >>= 1u;
                                cnt += 1u;
                            }

                            sample_mask <<= 1u;
                            if ((new_sig & sample_mask) != 0u) {
                                decoded_data[sig_dp + stride] = (cwd << 31u) | value;
                                cwd >>= 1u;
                                cnt += 1u;
                            }

                            sample_mask <<= 1u;
                            if ((new_sig & sample_mask) != 0u) {
                                decoded_data[sig_dp + 2u * stride] = (cwd << 31u) | value;
                                cwd >>= 1u;
                                cnt += 1u;
                            }

                            sample_mask <<= 1u;
                            if ((new_sig & sample_mask) != 0u) {
                                decoded_data[sig_dp + 3u * stride] = (cwd << 31u) | value;
                                cwd >>= 1u;
                                cnt += 1u;
                            }

                            col_mask <<= 4u;
                            sig_dp += 1u;
                        }
                    }

                    forward_reader_advance(sigprop, cnt);
                }

                const uint combined_sig = new_sig | cs;
                prev_row_sig[idx] = ushort(combined_sig);
                if (idx + 1u < prev_row_len) {
                    prev_row_sig[idx + 1u] = ushort(combined_sig >> 16u);
                }

                const uint combined = combined_sig;
                uint next_prev = combined_sig;
                next_prev |= (combined & 0x7777u) << 1u;
                next_prev |= (combined & 0xEEEEu) >> 1u;
                prev = (next_prev | u) & 0xF000u;
            }
        }

        if (num_passes > 2u) {
            y = 0u;
            while (y < height) {
                uint sp_base = (y >> 1u) * sstr;
                uint dp_base = (y >> 2u) * mstr;
                uint local_x = 0u;
                uint sigma_sp = sp_base;
                uint sigma_dp = dp_base;
                while (local_x < width) {
                    uint t0 = ((uint(scratch[sigma_sp]) & 0x30u) >> 4u)
                        | ((uint(scratch[sigma_sp]) & 0xC0u) >> 2u);
                    t0 |= ((uint(scratch[sigma_sp + 2u]) & 0x30u) << 4u)
                        | ((uint(scratch[sigma_sp + 2u]) & 0xC0u) << 6u);
                    uint t1 = ((uint(scratch[sigma_sp + sstr]) & 0x30u) >> 2u)
                        | (uint(scratch[sigma_sp + sstr]) & 0xC0u);
                    t1 |= ((uint(scratch[sigma_sp + sstr + 2u]) & 0x30u) << 6u)
                        | ((uint(scratch[sigma_sp + sstr + 2u]) & 0xC0u) << 8u);
                    sigma[sigma_dp] = ushort(t0 | t1);
                    local_x += 4u;
                    sigma_sp += 4u;
                    sigma_dp += 1u;
                }
                sigma[sigma_dp] = 0u;
                y += 4u;
            }

            for (uint i = 0u; i <= (width + 3u) / 4u; ++i) {
                sigma[sigma_tail + i] = 0u;
            }

            ReverseBitReader magref =
                reverse_reader_new_mrp(coded_data, lcup, params.refinement_length);
            const uint half_value = 1u << (p - 2u);

            for (y = 0u; y < height; y += 4u) {
                uint cur_sig_idx = (y >> 2u) * mstr;
                const uint dpp = params.output_offset + y * stride;

                for (uint x8 = 0u; x8 < width; x8 += 8u) {
                    const uint cwd = reverse_reader_fetch(magref);
                    const uint sig = read_u32_pair(sigma, cur_sig_idx);
                    cur_sig_idx += 2u;
                    uint col_mask = 0xFu;
                    uint cwd_mut = cwd;

                    if (sig != 0u) {
                        for (uint column = 0u; column < 8u; ++column) {
                            if ((sig & col_mask) != 0u) {
                                uint mag_dp = dpp + x8 + column;
                                uint sample_mask = 0x1111'1111u & col_mask;

                                for (uint row = 0u; row < 4u; ++row) {
                                    if ((sig & sample_mask) != 0u) {
                                        uint sym = cwd_mut & 1u;
                                        sym = (1u - sym) << (p - 1u);
                                        sym |= half_value;
                                        decoded_data[mag_dp] ^= sym;
                                        cwd_mut >>= 1u;
                                    }
                                    sample_mask <<= 1u;
                                    mag_dp += stride;
                                }
                            }
                            col_mask <<= 4u;
                        }
                    }

                    reverse_reader_advance(magref, __popc(sig));
                }
            }
        }
    }

}


struct CudaJ2kRect {
    uint x0;
    uint y0;
    uint x1;
    uint y1;
};

struct CudaJ2kIdwtJob {
    CudaJ2kRect rect;
    CudaJ2kRect ll_rect;
    CudaJ2kRect hl_rect;
    CudaJ2kRect lh_rect;
    CudaJ2kRect hh_rect;
    uint irreversible97;
};

struct CudaJ2kIdwtMultiJob {
    j2k_ulong ll_ptr;
    j2k_ulong hl_ptr;
    j2k_ulong lh_ptr;
    j2k_ulong hh_ptr;
    j2k_ulong output_ptr;
    CudaJ2kIdwtJob job;
};

struct CudaJ2kStoreGray8Job {
    uint input_width;
    uint source_x;
    uint source_y;
    uint copy_width;
    uint copy_height;
    uint output_width;
    uint output_height;
    uint output_x;
    uint output_y;
    float addend;
    uint bit_depth;
};

struct CudaJ2kStoreGray16Job {
    uint input_width;
    uint source_x;
    uint source_y;
    uint copy_width;
    uint copy_height;
    uint output_width;
    uint output_height;
    uint output_x;
    uint output_y;
    float addend;
    uint bit_depth;
};

struct CudaJ2kInverseMctJob {
    uint len;
    uint irreversible97;
    float addend0;
    float addend1;
    float addend2;
};

struct CudaJ2kStoreRgb8Job {
    uint input_width0;
    uint input_width1;
    uint input_width2;
    uint source_x0;
    uint source_y0;
    uint source_x1;
    uint source_y1;
    uint source_x2;
    uint source_y2;
    uint copy_width;
    uint copy_height;
    uint output_width;
    uint output_height;
    uint output_x;
    uint output_y;
    float addend0;
    float addend1;
    float addend2;
    uint bit_depth0;
    uint bit_depth1;
    uint bit_depth2;
    uint rgba;
};

struct CudaJ2kStoreRgb16Job {
    uint input_width0;
    uint input_width1;
    uint input_width2;
    uint source_x0;
    uint source_y0;
    uint source_x1;
    uint source_y1;
    uint source_x2;
    uint source_y2;
    uint copy_width;
    uint copy_height;
    uint output_width;
    uint output_height;
    uint output_x;
    uint output_y;
    float addend0;
    float addend1;
    float addend2;
    uint bit_depth0;
    uint bit_depth1;
    uint bit_depth2;
    uint rgba;
};

struct CudaJ2kStoreRgb8MctJob {
    CudaJ2kStoreRgb8Job store;
    uint irreversible97;
};

struct CudaJ2kStoreRgb8MctBatchJob {
    j2k_ulong plane0_ptr;
    j2k_ulong plane1_ptr;
    j2k_ulong plane2_ptr;
    j2k_ulong output_ptr;
    CudaJ2kStoreRgb8MctJob job;
};

struct CudaJ2kStoreRgb16MctJob {
    CudaJ2kStoreRgb16Job store;
    uint irreversible97;
};

__device__ inline uint rect_width(CudaJ2kRect rect) {
    return rect.x1 - rect.x0;
}

__device__ inline uint rect_height(CudaJ2kRect rect) {
    return rect.y1 - rect.y0;
}

__device__ inline uint div_ceil_2(uint value) {
    return (value + 1u) >> 1u;
}

__device__ inline uint idwt_band_coord(uint output_origin, uint output_coord, uint band_origin, bool low) {
    const uint global = output_coord;
    uint index;
    if (low) {
        index = div_ceil_2(global) - div_ceil_2(output_origin);
    } else {
        index = (global >> 1u) - (output_origin >> 1u);
    }
    return band_origin + index;
}

__device__ inline float source_get(const float *source, CudaJ2kRect rect, uint x, uint y) {
    if (x < rect.x0 || x >= rect.x1 || y < rect.y0 || y >= rect.y1) {
        return 0.0f;
    }
    const uint local_x = x - rect.x0;
    const uint local_y = y - rect.y0;
    return source[local_y * rect_width(rect) + local_x];
}

__device__ inline uint pse_left(uint idx, uint offset) {
    return idx > offset ? idx - offset : offset - idx;
}

__device__ inline uint pse_right(uint idx, uint offset, uint length) {
    const uint new_idx = idx + offset;
    if (new_idx >= length) {
        const uint overshoot = new_idx - length;
        return length - 2u - overshoot;
    }
    return new_idx;
}

__device__ inline float lift_53_sample(float sample, float left, float right, bool update_even) {
    if (update_even) {
        return sample - floorf(fmaf(left + right, 0.25f, 0.5f));
    }
    return sample + floorf((left + right) * 0.5f);
}

__device__ inline void filter_step_horizontal_53(float *scanline, uint width, uint first, bool update_even) {
    if (first == 0u) {
        const uint left = pse_left(0u, 1u);
        const uint right = pse_right(0u, 1u, width);
        if (update_even) {
            scanline[0] = scanline[0] - floorf(fmaf(scanline[left] + scanline[right], 0.25f, 0.5f));
        } else {
            scanline[0] = scanline[0] + floorf((scanline[left] + scanline[right]) * 0.5f);
        }
    }

    const uint middle_start = first == 0u ? 2u : 1u;
    for (uint i = middle_start; i + 1u < width; i += 2u) {
        if (update_even) {
            scanline[i] = scanline[i] - floorf(fmaf(scanline[i - 1u] + scanline[i + 1u], 0.25f, 0.5f));
        } else {
            scanline[i] = scanline[i] + floorf((scanline[i - 1u] + scanline[i + 1u]) * 0.5f);
        }
    }

    if (width > 1u && ((width - 1u) & 1u) == first) {
        const uint i = width - 1u;
        const uint left = pse_left(i, 1u);
        const uint right = pse_right(i, 1u, width);
        if (update_even) {
            scanline[i] = scanline[i] - floorf(fmaf(scanline[left] + scanline[right], 0.25f, 0.5f));
        } else {
            scanline[i] = scanline[i] + floorf((scanline[left] + scanline[right]) * 0.5f);
        }
    }
}

__device__ inline void filter_step_horizontal_97(float *scanline, uint width, uint first, float coefficient) {
    if (first == 0u) {
        const uint left = pse_left(0u, 1u);
        const uint right = pse_right(0u, 1u, width);
        scanline[0] = fmaf(scanline[left] + scanline[right], coefficient, scanline[0]);
    }

    const uint middle_start = first == 0u ? 2u : 1u;
    for (uint i = middle_start; i + 1u < width; i += 2u) {
        scanline[i] = fmaf(scanline[i - 1u] + scanline[i + 1u], coefficient, scanline[i]);
    }

    if (width > 1u && ((width - 1u) & 1u) == first) {
        const uint i = width - 1u;
        const uint left = pse_left(i, 1u);
        const uint right = pse_right(i, 1u, width);
        scanline[i] = fmaf(scanline[left] + scanline[right], coefficient, scanline[i]);
    }
}

__device__ inline void filter_horizontal_scanline(float *scanline, uint width, uint rect_x0, bool irreversible97) {
    if (width == 1u) {
        if ((rect_x0 & 1u) != 0u) {
            scanline[0] *= 0.5f;
        }
        return;
    }

    const uint first_even = rect_x0 & 1u;
    const uint first_odd = 1u - first_even;
    if (!irreversible97) {
        filter_step_horizontal_53(scanline, width, first_even, true);
        filter_step_horizontal_53(scanline, width, first_odd, false);
    } else {
        const float neg_alpha = 1.5861343f;
        const float neg_beta = 0.052980117f;
        const float neg_gamma = -0.8829111f;
        const float neg_delta = -0.44350687f;
        const float kappa = 1.2301741f;
        const float inv_kappa = 1.0f / kappa;
        const float k0 = first_even == 0u ? kappa : inv_kappa;
        const float k1 = first_even == 0u ? inv_kappa : kappa;
        for (uint i = 0u; i + 1u < width; i += 2u) {
            scanline[i] *= k0;
            scanline[i + 1u] *= k1;
        }
        if ((width & 1u) != 0u) {
            scanline[width - 1u] *= k0;
        }
        filter_step_horizontal_97(scanline, width, first_even, neg_delta);
        filter_step_horizontal_97(scanline, width, first_odd, neg_gamma);
        filter_step_horizontal_97(scanline, width, first_even, neg_beta);
        filter_step_horizontal_97(scanline, width, first_odd, neg_alpha);
    }
}

__device__ inline void filter_horizontal(float *output, CudaJ2kRect rect, bool irreversible97) {
    const uint width = rect_width(rect);
    const uint height = rect_height(rect);
    for (uint y = 0u; y < height; ++y) {
        filter_horizontal_scanline(output + y * width, width, rect.x0, irreversible97);
    }
}

__device__ inline void filter_step_vertical_53(float *output, uint width, uint height, uint first, bool update_even) {
    for (uint row = first; row < height; row += 2u) {
        const uint row_above = pse_left(row, 1u);
        const uint row_below = pse_right(row, 1u, height);
        for (uint col = 0u; col < width; ++col) {
            const uint idx = row * width + col;
            const float above = output[row_above * width + col];
            const float below = output[row_below * width + col];
            if (update_even) {
                output[idx] = output[idx] - floorf(fmaf(above + below, 0.25f, 0.5f));
            } else {
                output[idx] = output[idx] + floorf((above + below) * 0.5f);
            }
        }
    }
}

__device__ inline void filter_step_vertical_97(float *output, uint width, uint height, uint first, float coefficient) {
    for (uint row = first; row < height; row += 2u) {
        const uint row_above = pse_left(row, 1u);
        const uint row_below = pse_right(row, 1u, height);
        for (uint col = 0u; col < width; ++col) {
            const uint idx = row * width + col;
            output[idx] = fmaf(
                output[row_above * width + col] + output[row_below * width + col],
                coefficient,
                output[idx]
            );
        }
    }
}

__device__ inline void filter_step_vertical_53_column(
    float *output,
    uint width,
    uint height,
    uint col,
    uint first,
    bool update_even
) {
    for (uint row = first; row < height; row += 2u) {
        const uint row_above = pse_left(row, 1u);
        const uint row_below = pse_right(row, 1u, height);
        const uint idx = row * width + col;
        const float above = output[row_above * width + col];
        const float below = output[row_below * width + col];
        if (update_even) {
            output[idx] = output[idx] - floorf(fmaf(above + below, 0.25f, 0.5f));
        } else {
            output[idx] = output[idx] + floorf((above + below) * 0.5f);
        }
    }
}

__device__ inline void filter_step_vertical_97_column(
    float *output,
    uint width,
    uint height,
    uint col,
    uint first,
    float coefficient
) {
    for (uint row = first; row < height; row += 2u) {
        const uint row_above = pse_left(row, 1u);
        const uint row_below = pse_right(row, 1u, height);
        const uint idx = row * width + col;
        output[idx] = fmaf(
            output[row_above * width + col] + output[row_below * width + col],
            coefficient,
            output[idx]
        );
    }
}

__device__ inline void filter_vertical_column(
    float *output,
    uint width,
    uint height,
    uint rect_y0,
    uint col,
    bool irreversible97
) {
    if (height == 1u) {
        if ((rect_y0 & 1u) != 0u) {
            output[col] *= 0.5f;
        }
        return;
    }

    const uint first_even = rect_y0 & 1u;
    const uint first_odd = 1u - first_even;
    if (!irreversible97) {
        filter_step_vertical_53_column(output, width, height, col, first_even, true);
        filter_step_vertical_53_column(output, width, height, col, first_odd, false);
    } else {
        const float neg_alpha = 1.5861343f;
        const float neg_beta = 0.052980117f;
        const float neg_gamma = -0.8829111f;
        const float neg_delta = -0.44350687f;
        const float kappa = 1.2301741f;
        const float inv_kappa = 1.0f / kappa;
        const float k0 = first_even == 0u ? kappa : inv_kappa;
        const float k1 = first_even == 0u ? inv_kappa : kappa;
        for (uint row = 0u; row + 1u < height; row += 2u) {
            output[row * width + col] *= k0;
            output[(row + 1u) * width + col] *= k1;
        }
        if ((height & 1u) != 0u) {
            const uint row = height - 1u;
            output[row * width + col] *= k0;
        }
        filter_step_vertical_97_column(output, width, height, col, first_even, neg_delta);
        filter_step_vertical_97_column(output, width, height, col, first_odd, neg_gamma);
        filter_step_vertical_97_column(output, width, height, col, first_even, neg_beta);
        filter_step_vertical_97_column(output, width, height, col, first_odd, neg_alpha);
    }
}

__device__ inline void filter_vertical(float *output, CudaJ2kRect rect, bool irreversible97) {
    const uint width = rect_width(rect);
    const uint height = rect_height(rect);
    for (uint col = 0u; col < width; ++col) {
        filter_vertical_column(output, width, height, rect.y0, col, irreversible97);
    }
}

extern "C" __global__ void signinum_j2k_inverse_dwt_single(
    const float *ll,
    const float *hl,
    const float *lh,
    const float *hh,
    float *output,
    const CudaJ2kIdwtJob *job_buffer
) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }

    const CudaJ2kIdwtJob job = job_buffer[0];
    const uint width = rect_width(job.rect);
    const uint height = rect_height(job.rect);
    for (uint y = job.rect.y0; y < job.rect.y1; ++y) {
        const bool low_y = (y & 1u) == 0u;
        for (uint x = job.rect.x0; x < job.rect.x1; ++x) {
            const bool low_x = (x & 1u) == 0u;
            const float *source = ll;
            CudaJ2kRect source_rect = job.ll_rect;
            uint band_x;
            uint band_y;
            if (low_x && low_y) {
                source = ll;
                source_rect = job.ll_rect;
                band_x = idwt_band_coord(job.rect.x0, x, job.ll_rect.x0, true);
                band_y = idwt_band_coord(job.rect.y0, y, job.ll_rect.y0, true);
            } else if (!low_x && low_y) {
                source = hl;
                source_rect = job.hl_rect;
                band_x = idwt_band_coord(job.rect.x0, x, job.hl_rect.x0, false);
                band_y = idwt_band_coord(job.rect.y0, y, job.hl_rect.y0, true);
            } else if (low_x && !low_y) {
                source = lh;
                source_rect = job.lh_rect;
                band_x = idwt_band_coord(job.rect.x0, x, job.lh_rect.x0, true);
                band_y = idwt_band_coord(job.rect.y0, y, job.lh_rect.y0, false);
            } else {
                source = hh;
                source_rect = job.hh_rect;
                band_x = idwt_band_coord(job.rect.x0, x, job.hh_rect.x0, false);
                band_y = idwt_band_coord(job.rect.y0, y, job.hh_rect.y0, false);
            }
            const uint dst = (y - job.rect.y0) * width + (x - job.rect.x0);
            output[dst] = source_get(source, source_rect, band_x, band_y);
        }
    }

    if (width > 0u && height > 0u) {
        const bool irreversible97 = job.irreversible97 != 0u;
        filter_horizontal(output, job.rect, irreversible97);
        filter_vertical(output, job.rect, irreversible97);
    }
}

extern "C" __global__ void signinum_j2k_idwt_interleave(
    const float *ll,
    const float *hl,
    const float *lh,
    const float *hh,
    float *output,
    const CudaJ2kIdwtJob *job_buffer
) {
    const CudaJ2kIdwtJob job = job_buffer[0];
    const uint width = rect_width(job.rect);
    const uint height = rect_height(job.rect);
    const uint local_x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint local_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (local_x >= width || local_y >= height) {
        return;
    }

    const uint x = job.rect.x0 + local_x;
    const uint y = job.rect.y0 + local_y;
    const bool low_x = (x & 1u) == 0u;
    const bool low_y = (y & 1u) == 0u;
    const float *source = ll;
    CudaJ2kRect source_rect = job.ll_rect;
    uint band_x;
    uint band_y;
    if (low_x && low_y) {
        source = ll;
        source_rect = job.ll_rect;
        band_x = idwt_band_coord(job.rect.x0, x, job.ll_rect.x0, true);
        band_y = idwt_band_coord(job.rect.y0, y, job.ll_rect.y0, true);
    } else if (!low_x && low_y) {
        source = hl;
        source_rect = job.hl_rect;
        band_x = idwt_band_coord(job.rect.x0, x, job.hl_rect.x0, false);
        band_y = idwt_band_coord(job.rect.y0, y, job.hl_rect.y0, true);
    } else if (low_x && !low_y) {
        source = lh;
        source_rect = job.lh_rect;
        band_x = idwt_band_coord(job.rect.x0, x, job.lh_rect.x0, true);
        band_y = idwt_band_coord(job.rect.y0, y, job.lh_rect.y0, false);
    } else {
        source = hh;
        source_rect = job.hh_rect;
        band_x = idwt_band_coord(job.rect.x0, x, job.hh_rect.x0, false);
        band_y = idwt_band_coord(job.rect.y0, y, job.hh_rect.y0, false);
    }
    output[local_y * width + local_x] = source_get(source, source_rect, band_x, band_y);
}

extern "C" __global__ void signinum_j2k_idwt_interleave_multi(
    const CudaJ2kIdwtMultiJob *jobs
) {
    const uint job_idx = blockIdx.z;
    const CudaJ2kIdwtMultiJob item = jobs[job_idx];
    const CudaJ2kIdwtJob job = item.job;
    const float *ll = reinterpret_cast<const float *>(static_cast<uintptr_t>(item.ll_ptr));
    const float *hl = reinterpret_cast<const float *>(static_cast<uintptr_t>(item.hl_ptr));
    const float *lh = reinterpret_cast<const float *>(static_cast<uintptr_t>(item.lh_ptr));
    const float *hh = reinterpret_cast<const float *>(static_cast<uintptr_t>(item.hh_ptr));
    float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(item.output_ptr));
    const uint width = rect_width(job.rect);
    const uint height = rect_height(job.rect);
    const uint local_x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint local_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (local_x >= width || local_y >= height) {
        return;
    }

    const uint x = job.rect.x0 + local_x;
    const uint y = job.rect.y0 + local_y;
    const bool low_x = (x & 1u) == 0u;
    const bool low_y = (y & 1u) == 0u;
    const float *source = ll;
    CudaJ2kRect source_rect = job.ll_rect;
    uint band_x;
    uint band_y;
    if (low_x && low_y) {
        source = ll;
        source_rect = job.ll_rect;
        band_x = idwt_band_coord(job.rect.x0, x, job.ll_rect.x0, true);
        band_y = idwt_band_coord(job.rect.y0, y, job.ll_rect.y0, true);
    } else if (!low_x && low_y) {
        source = hl;
        source_rect = job.hl_rect;
        band_x = idwt_band_coord(job.rect.x0, x, job.hl_rect.x0, false);
        band_y = idwt_band_coord(job.rect.y0, y, job.hl_rect.y0, true);
    } else if (low_x && !low_y) {
        source = lh;
        source_rect = job.lh_rect;
        band_x = idwt_band_coord(job.rect.x0, x, job.lh_rect.x0, true);
        band_y = idwt_band_coord(job.rect.y0, y, job.lh_rect.y0, false);
    } else {
        source = hh;
        source_rect = job.hh_rect;
        band_x = idwt_band_coord(job.rect.x0, x, job.hh_rect.x0, false);
        band_y = idwt_band_coord(job.rect.y0, y, job.hh_rect.y0, false);
    }
    output[local_y * width + local_x] = source_get(source, source_rect, band_x, band_y);
}

__device__ inline float idwt_interleave_sample(
    const float *ll,
    const float *hl,
    const float *lh,
    const float *hh,
    CudaJ2kIdwtJob job,
    uint local_x,
    uint local_y
) {
    const uint x = job.rect.x0 + local_x;
    const uint y = job.rect.y0 + local_y;
    const bool low_x = (x & 1u) == 0u;
    const bool low_y = (y & 1u) == 0u;
    const float *source = ll;
    CudaJ2kRect source_rect = job.ll_rect;
    uint band_x;
    uint band_y;
    if (low_x && low_y) {
        source = ll;
        source_rect = job.ll_rect;
        band_x = idwt_band_coord(job.rect.x0, x, job.ll_rect.x0, true);
        band_y = idwt_band_coord(job.rect.y0, y, job.ll_rect.y0, true);
    } else if (!low_x && low_y) {
        source = hl;
        source_rect = job.hl_rect;
        band_x = idwt_band_coord(job.rect.x0, x, job.hl_rect.x0, false);
        band_y = idwt_band_coord(job.rect.y0, y, job.hl_rect.y0, true);
    } else if (low_x && !low_y) {
        source = lh;
        source_rect = job.lh_rect;
        band_x = idwt_band_coord(job.rect.x0, x, job.lh_rect.x0, true);
        band_y = idwt_band_coord(job.rect.y0, y, job.lh_rect.y0, false);
    } else {
        source = hh;
        source_rect = job.hh_rect;
        band_x = idwt_band_coord(job.rect.x0, x, job.hh_rect.x0, false);
        band_y = idwt_band_coord(job.rect.y0, y, job.hh_rect.y0, false);
    }
    return source_get(source, source_rect, band_x, band_y);
}

extern "C" __global__ void signinum_j2k_idwt_interleave_horizontal_multi(
    const CudaJ2kIdwtMultiJob *jobs
) {
    const uint job_idx = blockIdx.y;
    const CudaJ2kIdwtMultiJob item = jobs[job_idx];
    const CudaJ2kIdwtJob job = item.job;
    const float *ll = reinterpret_cast<const float *>(static_cast<uintptr_t>(item.ll_ptr));
    const float *hl = reinterpret_cast<const float *>(static_cast<uintptr_t>(item.hl_ptr));
    const float *lh = reinterpret_cast<const float *>(static_cast<uintptr_t>(item.lh_ptr));
    const float *hh = reinterpret_cast<const float *>(static_cast<uintptr_t>(item.hh_ptr));
    float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(item.output_ptr));
    const uint width = rect_width(job.rect);
    const uint height = rect_height(job.rect);
    const uint local_y = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_y >= height) {
        return;
    }

    const uint y = job.rect.y0 + local_y;
    const bool low_y = (y & 1u) == 0u;
    float *row_output = output + local_y * width;
    for (uint local_x = 0u; local_x < width; ++local_x) {
        const uint x = job.rect.x0 + local_x;
        const bool low_x = (x & 1u) == 0u;
        const float *source = ll;
        CudaJ2kRect source_rect = job.ll_rect;
        uint band_x;
        uint band_y;
        if (low_x && low_y) {
            source = ll;
            source_rect = job.ll_rect;
            band_x = idwt_band_coord(job.rect.x0, x, job.ll_rect.x0, true);
            band_y = idwt_band_coord(job.rect.y0, y, job.ll_rect.y0, true);
        } else if (!low_x && low_y) {
            source = hl;
            source_rect = job.hl_rect;
            band_x = idwt_band_coord(job.rect.x0, x, job.hl_rect.x0, false);
            band_y = idwt_band_coord(job.rect.y0, y, job.hl_rect.y0, true);
        } else if (low_x && !low_y) {
            source = lh;
            source_rect = job.lh_rect;
            band_x = idwt_band_coord(job.rect.x0, x, job.lh_rect.x0, true);
            band_y = idwt_band_coord(job.rect.y0, y, job.lh_rect.y0, false);
        } else {
            source = hh;
            source_rect = job.hh_rect;
            band_x = idwt_band_coord(job.rect.x0, x, job.hh_rect.x0, false);
            band_y = idwt_band_coord(job.rect.y0, y, job.hh_rect.y0, false);
        }
        row_output[local_x] = source_get(source, source_rect, band_x, band_y);
    }
    filter_horizontal_scanline(
        row_output,
        width,
        job.rect.x0,
        job.irreversible97 != 0u
    );
}

extern "C" __global__ void signinum_j2k_idwt_interleave_horizontal_53_multi(
    const CudaJ2kIdwtMultiJob *jobs
) {
    __shared__ float row_samples[512];

    const uint local_x = threadIdx.x;
    const uint local_y = blockIdx.x;
    const uint job_idx = blockIdx.y;
    const CudaJ2kIdwtMultiJob item = jobs[job_idx];
    const CudaJ2kIdwtJob job = item.job;
    const float *ll = reinterpret_cast<const float *>(static_cast<uintptr_t>(item.ll_ptr));
    const float *hl = reinterpret_cast<const float *>(static_cast<uintptr_t>(item.hl_ptr));
    const float *lh = reinterpret_cast<const float *>(static_cast<uintptr_t>(item.lh_ptr));
    const float *hh = reinterpret_cast<const float *>(static_cast<uintptr_t>(item.hh_ptr));
    float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(item.output_ptr));
    const uint width = rect_width(job.rect);
    const uint height = rect_height(job.rect);
    if (local_y >= height) {
        return;
    }

    if (local_x < width) {
        row_samples[local_x] = idwt_interleave_sample(ll, hl, lh, hh, job, local_x, local_y);
    }
    __syncthreads();

    if (width == 1u) {
        if (local_x == 0u && (job.rect.x0 & 1u) != 0u) {
            row_samples[0] *= 0.5f;
        }
        __syncthreads();
        if (local_x == 0u) {
            output[local_y * width] = row_samples[0];
        }
        return;
    }

    const uint first_even = job.rect.x0 & 1u;
    const uint first_odd = 1u - first_even;
    if (local_x < width && ((local_x & 1u) == first_even)) {
        const uint left = pse_left(local_x, 1u);
        const uint right = pse_right(local_x, 1u, width);
        row_samples[local_x] = lift_53_sample(
            row_samples[local_x],
            row_samples[left],
            row_samples[right],
            true
        );
    }
    __syncthreads();

    if (local_x < width && ((local_x & 1u) == first_odd)) {
        const uint left = pse_left(local_x, 1u);
        const uint right = pse_right(local_x, 1u, width);
        row_samples[local_x] = lift_53_sample(
            row_samples[local_x],
            row_samples[left],
            row_samples[right],
            false
        );
    }
    __syncthreads();

    if (local_x < width) {
        output[local_y * width + local_x] = row_samples[local_x];
    }
}

extern "C" __global__ void signinum_j2k_idwt_interleave_horizontal_97_multi(
    const CudaJ2kIdwtMultiJob *jobs
) {
    __shared__ float row_samples[512];

    const uint local_x = threadIdx.x;
    const uint local_y = blockIdx.x;
    const uint job_idx = blockIdx.y;
    const CudaJ2kIdwtMultiJob item = jobs[job_idx];
    const CudaJ2kIdwtJob job = item.job;
    const float *ll = reinterpret_cast<const float *>(static_cast<uintptr_t>(item.ll_ptr));
    const float *hl = reinterpret_cast<const float *>(static_cast<uintptr_t>(item.hl_ptr));
    const float *lh = reinterpret_cast<const float *>(static_cast<uintptr_t>(item.lh_ptr));
    const float *hh = reinterpret_cast<const float *>(static_cast<uintptr_t>(item.hh_ptr));
    float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(item.output_ptr));
    const uint width = rect_width(job.rect);
    const uint height = rect_height(job.rect);
    if (local_y >= height) {
        return;
    }

    if (local_x < width) {
        row_samples[local_x] = idwt_interleave_sample(ll, hl, lh, hh, job, local_x, local_y);
    }
    __syncthreads();

    if (width == 1u) {
        if (local_x == 0u && (job.rect.x0 & 1u) != 0u) {
            row_samples[0] *= 0.5f;
        }
        __syncthreads();
        if (local_x == 0u) {
            output[local_y * width] = row_samples[0];
        }
        return;
    }

    const uint first_even = job.rect.x0 & 1u;
    const uint first_odd = 1u - first_even;
    const float neg_alpha = 1.5861343f;
    const float neg_beta = 0.052980117f;
    const float neg_gamma = -0.8829111f;
    const float neg_delta = -0.44350687f;
    const float kappa = 1.2301741f;
    const float inv_kappa = 1.0f / kappa;
    const float k0 = first_even == 0u ? kappa : inv_kappa;
    const float k1 = first_even == 0u ? inv_kappa : kappa;

    if (local_x < width) {
        row_samples[local_x] *= ((local_x & 1u) == 0u) ? k0 : k1;
    }
    __syncthreads();

    if (local_x < width && ((local_x & 1u) == first_even)) {
        const uint left = pse_left(local_x, 1u);
        const uint right = pse_right(local_x, 1u, width);
        row_samples[local_x] = fmaf(row_samples[left] + row_samples[right], neg_delta, row_samples[local_x]);
    }
    __syncthreads();

    if (local_x < width && ((local_x & 1u) == first_odd)) {
        const uint left = pse_left(local_x, 1u);
        const uint right = pse_right(local_x, 1u, width);
        row_samples[local_x] = fmaf(row_samples[left] + row_samples[right], neg_gamma, row_samples[local_x]);
    }
    __syncthreads();

    if (local_x < width && ((local_x & 1u) == first_even)) {
        const uint left = pse_left(local_x, 1u);
        const uint right = pse_right(local_x, 1u, width);
        row_samples[local_x] = fmaf(row_samples[left] + row_samples[right], neg_beta, row_samples[local_x]);
    }
    __syncthreads();

    if (local_x < width && ((local_x & 1u) == first_odd)) {
        const uint left = pse_left(local_x, 1u);
        const uint right = pse_right(local_x, 1u, width);
        row_samples[local_x] = fmaf(row_samples[left] + row_samples[right], neg_alpha, row_samples[local_x]);
    }
    __syncthreads();

    if (local_x < width) {
        output[local_y * width + local_x] = row_samples[local_x];
    }
}

extern "C" __global__ void signinum_j2k_idwt_horizontal(
    float *output,
    const CudaJ2kIdwtJob *job_buffer
) {
    const CudaJ2kIdwtJob job = job_buffer[0];
    const uint width = rect_width(job.rect);
    const uint height = rect_height(job.rect);
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) {
        return;
    }
    filter_horizontal_scanline(
        output + row * width,
        width,
        job.rect.x0,
        job.irreversible97 != 0u
    );
}

extern "C" __global__ void signinum_j2k_idwt_horizontal_53(
    float *output,
    const CudaJ2kIdwtJob *job_buffer
) {
    const CudaJ2kIdwtJob job = job_buffer[0];
    const uint width = rect_width(job.rect);
    const uint height = rect_height(job.rect);
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) {
        return;
    }
    filter_horizontal_scanline(output + row * width, width, job.rect.x0, false);
}

extern "C" __global__ void signinum_j2k_idwt_horizontal_97(
    float *output,
    const CudaJ2kIdwtJob *job_buffer
) {
    const CudaJ2kIdwtJob job = job_buffer[0];
    const uint width = rect_width(job.rect);
    const uint height = rect_height(job.rect);
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) {
        return;
    }
    filter_horizontal_scanline(output + row * width, width, job.rect.x0, true);
}

extern "C" __global__ void signinum_j2k_idwt_horizontal_multi(
    const CudaJ2kIdwtMultiJob *jobs
) {
    const uint job_idx = blockIdx.y;
    const CudaJ2kIdwtMultiJob item = jobs[job_idx];
    const CudaJ2kIdwtJob job = item.job;
    float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(item.output_ptr));
    const uint width = rect_width(job.rect);
    const uint height = rect_height(job.rect);
    const uint row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) {
        return;
    }
    filter_horizontal_scanline(
        output + row * width,
        width,
        job.rect.x0,
        job.irreversible97 != 0u
    );
}

extern "C" __global__ void signinum_j2k_idwt_vertical(
    float *output,
    const CudaJ2kIdwtJob *job_buffer
) {
    const CudaJ2kIdwtJob job = job_buffer[0];
    const uint width = rect_width(job.rect);
    const uint height = rect_height(job.rect);
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width) {
        return;
    }
    filter_vertical_column(
        output,
        width,
        height,
        job.rect.y0,
        col,
        job.irreversible97 != 0u
    );
}

extern "C" __global__ void signinum_j2k_idwt_vertical_53(
    float *output,
    const CudaJ2kIdwtJob *job_buffer
) {
    const CudaJ2kIdwtJob job = job_buffer[0];
    const uint width = rect_width(job.rect);
    const uint height = rect_height(job.rect);
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width) {
        return;
    }
    filter_vertical_column(output, width, height, job.rect.y0, col, false);
}

extern "C" __global__ void signinum_j2k_idwt_vertical_97(
    float *output,
    const CudaJ2kIdwtJob *job_buffer
) {
    const CudaJ2kIdwtJob job = job_buffer[0];
    const uint width = rect_width(job.rect);
    const uint height = rect_height(job.rect);
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width) {
        return;
    }
    filter_vertical_column(output, width, height, job.rect.y0, col, true);
}

extern "C" __global__ void signinum_j2k_idwt_vertical_multi(
    const CudaJ2kIdwtMultiJob *jobs
) {
    const uint job_idx = blockIdx.y;
    const CudaJ2kIdwtMultiJob item = jobs[job_idx];
    const CudaJ2kIdwtJob job = item.job;
    float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(item.output_ptr));
    const uint width = rect_width(job.rect);
    const uint height = rect_height(job.rect);
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= width) {
        return;
    }
    filter_vertical_column(
        output,
        width,
        height,
        job.rect.y0,
        col,
        job.irreversible97 != 0u
    );
}

extern "C" __global__ void signinum_j2k_idwt_vertical_53_multi(
    const CudaJ2kIdwtMultiJob *jobs
) {
    __shared__ float column_samples[512];

    const uint row = threadIdx.x;
    const uint col = blockIdx.x;
    const uint job_idx = blockIdx.y;
    const CudaJ2kIdwtMultiJob item = jobs[job_idx];
    const CudaJ2kIdwtJob job = item.job;
    float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(item.output_ptr));
    const uint width = rect_width(job.rect);
    const uint height = rect_height(job.rect);
    if (col >= width) {
        return;
    }

    if (row < height) {
        column_samples[row] = output[row * width + col];
    }
    __syncthreads();

    if (height == 1u) {
        if (row == 0u && (job.rect.y0 & 1u) != 0u) {
            column_samples[0] *= 0.5f;
        }
        __syncthreads();
        if (row == 0u) {
            output[col] = column_samples[0];
        }
        return;
    }

    const uint first_even = job.rect.y0 & 1u;
    const uint first_odd = 1u - first_even;
    if (row < height && ((row & 1u) == first_even)) {
        const uint above = pse_left(row, 1u);
        const uint below = pse_right(row, 1u, height);
        column_samples[row] = lift_53_sample(
            column_samples[row],
            column_samples[above],
            column_samples[below],
            true
        );
    }
    __syncthreads();

    if (row < height && ((row & 1u) == first_odd)) {
        const uint above = pse_left(row, 1u);
        const uint below = pse_right(row, 1u, height);
        column_samples[row] = lift_53_sample(
            column_samples[row],
            column_samples[above],
            column_samples[below],
            false
        );
    }
    __syncthreads();

    if (row < height) {
        output[row * width + col] = column_samples[row];
    }
}

extern "C" __global__ void signinum_j2k_idwt_vertical_97_multi(
    const CudaJ2kIdwtMultiJob *jobs
) {
    __shared__ float column_samples[512];

    const uint row = threadIdx.x;
    const uint col = blockIdx.x;
    const uint job_idx = blockIdx.y;
    const CudaJ2kIdwtMultiJob item = jobs[job_idx];
    const CudaJ2kIdwtJob job = item.job;
    float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(item.output_ptr));
    const uint width = rect_width(job.rect);
    const uint height = rect_height(job.rect);
    if (col >= width) {
        return;
    }

    if (row < height) {
        column_samples[row] = output[row * width + col];
    }
    __syncthreads();

    if (height == 1u) {
        if (row == 0u && (job.rect.y0 & 1u) != 0u) {
            column_samples[0] *= 0.5f;
        }
        __syncthreads();
        if (row == 0u) {
            output[col] = column_samples[0];
        }
        return;
    }

    const uint first_even = job.rect.y0 & 1u;
    const uint first_odd = 1u - first_even;
    const float neg_alpha = 1.5861343f;
    const float neg_beta = 0.052980117f;
    const float neg_gamma = -0.8829111f;
    const float neg_delta = -0.44350687f;
    const float kappa = 1.2301741f;
    const float inv_kappa = 1.0f / kappa;
    const float k0 = first_even == 0u ? kappa : inv_kappa;
    const float k1 = first_even == 0u ? inv_kappa : kappa;

    if (row < height) {
        column_samples[row] *= ((row & 1u) == 0u) ? k0 : k1;
    }
    __syncthreads();

    if (row < height && ((row & 1u) == first_even)) {
        const uint above = pse_left(row, 1u);
        const uint below = pse_right(row, 1u, height);
        column_samples[row] = fmaf(
            column_samples[above] + column_samples[below],
            neg_delta,
            column_samples[row]
        );
    }
    __syncthreads();

    if (row < height && ((row & 1u) == first_odd)) {
        const uint above = pse_left(row, 1u);
        const uint below = pse_right(row, 1u, height);
        column_samples[row] = fmaf(
            column_samples[above] + column_samples[below],
            neg_gamma,
            column_samples[row]
        );
    }
    __syncthreads();

    if (row < height && ((row & 1u) == first_even)) {
        const uint above = pse_left(row, 1u);
        const uint below = pse_right(row, 1u, height);
        column_samples[row] = fmaf(
            column_samples[above] + column_samples[below],
            neg_beta,
            column_samples[row]
        );
    }
    __syncthreads();

    if (row < height && ((row & 1u) == first_odd)) {
        const uint above = pse_left(row, 1u);
        const uint below = pse_right(row, 1u, height);
        column_samples[row] = fmaf(
            column_samples[above] + column_samples[below],
            neg_alpha,
            column_samples[row]
        );
    }
    __syncthreads();

    if (row < height) {
        output[row * width + col] = column_samples[row];
    }
}

extern "C" __global__ void signinum_j2k_idwt_vertical_97_multi_cols4(
    const CudaJ2kIdwtMultiJob *jobs
) {
    __shared__ float column_samples[256][4];

    const uint local_col = threadIdx.x;
    const uint row = threadIdx.y;
    const uint col = blockIdx.x * 4u + local_col;
    const uint job_idx = blockIdx.y;
    const CudaJ2kIdwtMultiJob item = jobs[job_idx];
    const CudaJ2kIdwtJob job = item.job;
    float *output = reinterpret_cast<float *>(static_cast<uintptr_t>(item.output_ptr));
    const uint width = rect_width(job.rect);
    const uint height = rect_height(job.rect);
    if (height > 256u) {
        return;
    }

    const bool valid = col < width && row < height;
    if (valid) {
        column_samples[row][local_col] = output[row * width + col];
    }
    __syncthreads();

    if (height == 1u) {
        if (valid && (job.rect.y0 & 1u) != 0u) {
            column_samples[0][local_col] *= 0.5f;
        }
        __syncthreads();
        if (valid) {
            output[col] = column_samples[0][local_col];
        }
        return;
    }

    const uint first_even = job.rect.y0 & 1u;
    const uint first_odd = 1u - first_even;
    const float neg_alpha = 1.5861343f;
    const float neg_beta = 0.052980117f;
    const float neg_gamma = -0.8829111f;
    const float neg_delta = -0.44350687f;
    const float kappa = 1.2301741f;
    const float inv_kappa = 1.0f / kappa;
    const float k0 = first_even == 0u ? kappa : inv_kappa;
    const float k1 = first_even == 0u ? inv_kappa : kappa;

    if (valid) {
        column_samples[row][local_col] *= ((row & 1u) == 0u) ? k0 : k1;
    }
    __syncthreads();

    if (valid && ((row & 1u) == first_even)) {
        const uint above = pse_left(row, 1u);
        const uint below = pse_right(row, 1u, height);
        column_samples[row][local_col] = fmaf(
            column_samples[above][local_col] + column_samples[below][local_col],
            neg_delta,
            column_samples[row][local_col]
        );
    }
    __syncthreads();

    if (valid && ((row & 1u) == first_odd)) {
        const uint above = pse_left(row, 1u);
        const uint below = pse_right(row, 1u, height);
        column_samples[row][local_col] = fmaf(
            column_samples[above][local_col] + column_samples[below][local_col],
            neg_gamma,
            column_samples[row][local_col]
        );
    }
    __syncthreads();

    if (valid && ((row & 1u) == first_even)) {
        const uint above = pse_left(row, 1u);
        const uint below = pse_right(row, 1u, height);
        column_samples[row][local_col] = fmaf(
            column_samples[above][local_col] + column_samples[below][local_col],
            neg_beta,
            column_samples[row][local_col]
        );
    }
    __syncthreads();

    if (valid && ((row & 1u) == first_odd)) {
        const uint above = pse_left(row, 1u);
        const uint below = pse_right(row, 1u, height);
        column_samples[row][local_col] = fmaf(
            column_samples[above][local_col] + column_samples[below][local_col],
            neg_alpha,
            column_samples[row][local_col]
        );
    }
    __syncthreads();

    if (valid) {
        output[row * width + col] = column_samples[row][local_col];
    }
}

__device__ inline uchar sample_as_u8(float sample, uint bit_depth) {
    const float rounded = roundf(sample);
    if (bit_depth == 8u) {
        return (uchar)fminf(fmaxf(rounded, 0.0f), 255.0f);
    }
    const float max_value = bit_depth >= 16u ? 65535.0f : (float)(((1u << bit_depth) - 1u) > 1u ? ((1u << bit_depth) - 1u) : 1u);
    return (uchar)roundf((fminf(fmaxf(rounded, 0.0f), max_value) / max_value) * 255.0f);
}

__device__ inline ushort sample_as_u16(float sample, uint bit_depth) {
    const float rounded = roundf(sample);
    if (bit_depth >= 16u) {
        return (ushort)fminf(fmaxf(rounded, 0.0f), 65535.0f);
    }
    const uint max_int = bit_depth == 0u ? 1u : ((1u << bit_depth) - 1u);
    const float max_value = (float)(max_int > 1u ? max_int : 1u);
    return (ushort)roundf((fminf(fmaxf(rounded, 0.0f), max_value) / max_value) * 65535.0f);
}

extern "C" __global__ void signinum_j2k_store_gray8(
    const float *input,
    uchar *output,
    const CudaJ2kStoreGray8Job *job_buffer
) {
    const CudaJ2kStoreGray8Job job = job_buffer[0];
    const uint pixels = job.copy_width * job.copy_height;
    const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= pixels) {
        return;
    }

    const uint row = gid / job.copy_width;
    const uint col = gid - row * job.copy_width;
    const uint src = (job.source_y + row) * job.input_width + job.source_x + col;
    const uint dst = (job.output_y + row) * job.output_width + job.output_x + col;
    output[dst] = sample_as_u8(input[src] + job.addend, job.bit_depth);
}

extern "C" __global__ void signinum_j2k_store_gray16(
    const float *input,
    ushort *output,
    const CudaJ2kStoreGray16Job *job_buffer
) {
    const CudaJ2kStoreGray16Job job = job_buffer[0];
    const uint pixels = job.copy_width * job.copy_height;
    const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= pixels) {
        return;
    }

    const uint row = gid / job.copy_width;
    const uint col = gid - row * job.copy_width;
    const uint src = (job.source_y + row) * job.input_width + job.source_x + col;
    const uint dst = (job.output_y + row) * job.output_width + job.output_x + col;
    output[dst] = sample_as_u16(input[src] + job.addend, job.bit_depth);
}

__device__ inline void inverse_mct_sample(
    float src0,
    float src1,
    float src2,
    uint irreversible97,
    float *out0,
    float *out1,
    float *out2
) {
    if (irreversible97 != 0u) {
        *out0 = src0 + 1.402f * src2;
        *out1 = src0 - 0.34413f * src1 - 0.71414f * src2;
        *out2 = src0 + 1.772f * src1;
    } else {
        const float green = src0 - floorf((src2 + src1) * 0.25f);
        *out0 = src2 + green;
        *out1 = green;
        *out2 = src1 + green;
    }
}

extern "C" __global__ void signinum_j2k_inverse_mct(
    float *plane0,
    float *plane1,
    float *plane2,
    const CudaJ2kInverseMctJob *job_buffer
) {
    const CudaJ2kInverseMctJob job = job_buffer[0];
    const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= job.len) {
        return;
    }

    const float src0 = plane0[gid];
    const float src1 = plane1[gid];
    const float src2 = plane2[gid];
    float out0;
    float out1;
    float out2;
    inverse_mct_sample(src0, src1, src2, job.irreversible97, &out0, &out1, &out2);
    plane0[gid] = out0 + job.addend0;
    plane1[gid] = out1 + job.addend1;
    plane2[gid] = out2 + job.addend2;
}

extern "C" __global__ void signinum_j2k_store_rgb8(
    const float *plane0,
    const float *plane1,
    const float *plane2,
    uchar *output,
    const CudaJ2kStoreRgb8Job *job_buffer
) {
    const CudaJ2kStoreRgb8Job job = job_buffer[0];
    const uint pixels = job.copy_width * job.copy_height;
    const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= pixels) {
        return;
    }

    const uint row = gid / job.copy_width;
    const uint col = gid - row * job.copy_width;
    const uint src0 = (job.source_y0 + row) * job.input_width0 + job.source_x0 + col;
    const uint src1 = (job.source_y1 + row) * job.input_width1 + job.source_x1 + col;
    const uint src2 = (job.source_y2 + row) * job.input_width2 + job.source_x2 + col;
    const uint channels = job.rgba != 0u ? 4u : 3u;
    const uint dst = ((job.output_y + row) * job.output_width + job.output_x + col) * channels;

    output[dst] = sample_as_u8(plane0[src0] + job.addend0, job.bit_depth0);
    output[dst + 1u] = sample_as_u8(plane1[src1] + job.addend1, job.bit_depth1);
    output[dst + 2u] = sample_as_u8(plane2[src2] + job.addend2, job.bit_depth2);
    if (job.rgba != 0u) {
        output[dst + 3u] = 255u;
    }
}

extern "C" __global__ void signinum_j2k_store_rgb16(
    const float *plane0,
    const float *plane1,
    const float *plane2,
    ushort *output,
    const CudaJ2kStoreRgb16Job *job_buffer
) {
    const CudaJ2kStoreRgb16Job job = job_buffer[0];
    const uint pixels = job.copy_width * job.copy_height;
    const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= pixels) {
        return;
    }

    const uint row = gid / job.copy_width;
    const uint col = gid - row * job.copy_width;
    const uint src0 = (job.source_y0 + row) * job.input_width0 + job.source_x0 + col;
    const uint src1 = (job.source_y1 + row) * job.input_width1 + job.source_x1 + col;
    const uint src2 = (job.source_y2 + row) * job.input_width2 + job.source_x2 + col;
    const uint channels = job.rgba != 0u ? 4u : 3u;
    const uint dst = ((job.output_y + row) * job.output_width + job.output_x + col) * channels;

    output[dst] = sample_as_u16(plane0[src0] + job.addend0, job.bit_depth0);
    output[dst + 1u] = sample_as_u16(plane1[src1] + job.addend1, job.bit_depth1);
    output[dst + 2u] = sample_as_u16(plane2[src2] + job.addend2, job.bit_depth2);
    if (job.rgba != 0u) {
        output[dst + 3u] = 65535u;
    }
}

extern "C" __global__ void signinum_j2k_store_rgb8_mct(
    const float *plane0,
    const float *plane1,
    const float *plane2,
    uchar *output,
    const CudaJ2kStoreRgb8MctJob *job_buffer
) {
    const CudaJ2kStoreRgb8MctJob mct_job = job_buffer[0];
    const CudaJ2kStoreRgb8Job job = mct_job.store;
    const uint pixels = job.copy_width * job.copy_height;
    const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= pixels) {
        return;
    }

    const uint row = gid / job.copy_width;
    const uint col = gid - row * job.copy_width;
    const uint src0 = (job.source_y0 + row) * job.input_width0 + job.source_x0 + col;
    const uint src1 = (job.source_y1 + row) * job.input_width1 + job.source_x1 + col;
    const uint src2 = (job.source_y2 + row) * job.input_width2 + job.source_x2 + col;
    const uint channels = job.rgba != 0u ? 4u : 3u;
    const uint dst = ((job.output_y + row) * job.output_width + job.output_x + col) * channels;

    float out0;
    float out1;
    float out2;
    inverse_mct_sample(
        plane0[src0],
        plane1[src1],
        plane2[src2],
        mct_job.irreversible97,
        &out0,
        &out1,
        &out2
    );
    output[dst] = sample_as_u8(out0 + job.addend0, job.bit_depth0);
    output[dst + 1u] = sample_as_u8(out1 + job.addend1, job.bit_depth1);
    output[dst + 2u] = sample_as_u8(out2 + job.addend2, job.bit_depth2);
    if (job.rgba != 0u) {
        output[dst + 3u] = 255u;
    }
}

extern "C" __global__ void signinum_j2k_store_rgb8_mct_batch(
    const CudaJ2kStoreRgb8MctBatchJob *jobs
) {
    const CudaJ2kStoreRgb8MctBatchJob item = jobs[blockIdx.y];
    const float *plane0 = reinterpret_cast<const float *>(static_cast<uintptr_t>(item.plane0_ptr));
    const float *plane1 = reinterpret_cast<const float *>(static_cast<uintptr_t>(item.plane1_ptr));
    const float *plane2 = reinterpret_cast<const float *>(static_cast<uintptr_t>(item.plane2_ptr));
    uchar *output = reinterpret_cast<uchar *>(static_cast<uintptr_t>(item.output_ptr));
    const CudaJ2kStoreRgb8MctJob mct_job = item.job;
    const CudaJ2kStoreRgb8Job job = mct_job.store;
    const uint pixels = job.copy_width * job.copy_height;
    const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= pixels) {
        return;
    }

    const uint row = gid / job.copy_width;
    const uint col = gid - row * job.copy_width;
    const uint src0 = (job.source_y0 + row) * job.input_width0 + job.source_x0 + col;
    const uint src1 = (job.source_y1 + row) * job.input_width1 + job.source_x1 + col;
    const uint src2 = (job.source_y2 + row) * job.input_width2 + job.source_x2 + col;
    const uint channels = job.rgba != 0u ? 4u : 3u;
    const uint dst = ((job.output_y + row) * job.output_width + job.output_x + col) * channels;

    float out0;
    float out1;
    float out2;
    inverse_mct_sample(
        plane0[src0],
        plane1[src1],
        plane2[src2],
        mct_job.irreversible97,
        &out0,
        &out1,
        &out2
    );
    output[dst] = sample_as_u8(out0 + job.addend0, job.bit_depth0);
    output[dst + 1u] = sample_as_u8(out1 + job.addend1, job.bit_depth1);
    output[dst + 2u] = sample_as_u8(out2 + job.addend2, job.bit_depth2);
    if (job.rgba != 0u) {
        output[dst + 3u] = 255u;
    }
}

extern "C" __global__ void signinum_j2k_store_rgb16_mct(
    const float *plane0,
    const float *plane1,
    const float *plane2,
    ushort *output,
    const CudaJ2kStoreRgb16MctJob *job_buffer
) {
    const CudaJ2kStoreRgb16MctJob mct_job = job_buffer[0];
    const CudaJ2kStoreRgb16Job job = mct_job.store;
    const uint pixels = job.copy_width * job.copy_height;
    const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= pixels) {
        return;
    }

    const uint row = gid / job.copy_width;
    const uint col = gid - row * job.copy_width;
    const uint src0 = (job.source_y0 + row) * job.input_width0 + job.source_x0 + col;
    const uint src1 = (job.source_y1 + row) * job.input_width1 + job.source_x1 + col;
    const uint src2 = (job.source_y2 + row) * job.input_width2 + job.source_x2 + col;
    const uint channels = job.rgba != 0u ? 4u : 3u;
    const uint dst = ((job.output_y + row) * job.output_width + job.output_x + col) * channels;

    float out0;
    float out1;
    float out2;
    inverse_mct_sample(
        plane0[src0],
        plane1[src1],
        plane2[src2],
        mct_job.irreversible97,
        &out0,
        &out1,
        &out2
    );
    output[dst] = sample_as_u16(out0 + job.addend0, job.bit_depth0);
    output[dst + 1u] = sample_as_u16(out1 + job.addend1, job.bit_depth1);
    output[dst + 2u] = sample_as_u16(out2 + job.addend2, job.bit_depth2);
    if (job.rgba != 0u) {
        output[dst + 3u] = 65535u;
    }
}

extern "C" __global__ void signinum_htj2k_decode_codeblocks(
    const uchar *coded_data,
    uint *decoded_data,
    const J2kHtCleanupBatchJob *jobs,
    const ushort *vlc_table0,
    const ushort *vlc_table1,
    const ushort *uvlc_table0,
    const ushort *uvlc_table1,
    J2kHtStatus *status,
    uint job_count
) {
    const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= job_count) {
        return;
    }
    const J2kHtCleanupBatchJob job = jobs[gid];

    J2kHtCleanupParams params;
    params.width = job.width;
    params.height = job.height;
    params.coded_len = job.coded_len;
    params.cleanup_length = job.cleanup_length;
    params.refinement_length = job.refinement_length;
    params.missing_msbs = job.missing_msbs;
    params.num_bitplanes = job.num_bitplanes;
    params.number_of_coding_passes = job.number_of_coding_passes;
    params.output_stride = job.output_stride;
    params.output_offset = job.output_offset;
    params.dequantization_step = job.dequantization_step;
    params.stripe_causal = job.stripe_causal;

    decode_ht_cleanup_impl<false>(
        coded_data + job.coded_offset,
        decoded_data,
        params,
        vlc_table0,
        vlc_table1,
        uvlc_table0,
        uvlc_table1,
        status + gid
    );
}

extern "C" __global__ void signinum_htj2k_decode_codeblocks_multi(
    const uchar *coded_data,
    const J2kHtCleanupMultiBatchJob *jobs,
    const ushort *vlc_table0,
    const ushort *vlc_table1,
    const ushort *uvlc_table0,
    const ushort *uvlc_table1,
    J2kHtStatus *status,
    uint job_count
) {
    const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= job_count) {
        return;
    }
    const J2kHtCleanupMultiBatchJob job = jobs[gid];
    uint *decoded_data = reinterpret_cast<uint *>(static_cast<uintptr_t>(job.output_ptr));

    J2kHtCleanupParams params;
    params.width = job.width;
    params.height = job.height;
    params.coded_len = job.coded_len;
    params.cleanup_length = job.cleanup_length;
    params.refinement_length = job.refinement_length;
    params.missing_msbs = job.missing_msbs;
    params.num_bitplanes = job.num_bitplanes;
    params.number_of_coding_passes = job.number_of_coding_passes;
    params.output_stride = job.output_stride;
    params.output_offset = job.output_offset;
    params.dequantization_step = job.dequantization_step;
    params.stripe_causal = job.stripe_causal;

    decode_ht_cleanup_impl<false>(
        coded_data + job.coded_offset,
        decoded_data,
        params,
        vlc_table0,
        vlc_table1,
        uvlc_table0,
        uvlc_table1,
        status + gid
    );
}

extern "C" __global__ void signinum_htj2k_decode_codeblocks_multi_cleanup_only(
    const uchar *coded_data,
    const J2kHtCleanupMultiBatchJob *jobs,
    const ushort *vlc_table0,
    const ushort *vlc_table1,
    const ushort *uvlc_table0,
    const ushort *uvlc_table1,
    J2kHtStatus *status,
    uint job_count
) {
    const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= job_count) {
        return;
    }
    const J2kHtCleanupMultiBatchJob job = jobs[gid];
    uint *decoded_data = reinterpret_cast<uint *>(static_cast<uintptr_t>(job.output_ptr));

    J2kHtCleanupParams params;
    params.width = job.width;
    params.height = job.height;
    params.coded_len = job.coded_len;
    params.cleanup_length = job.cleanup_length;
    params.refinement_length = job.refinement_length;
    params.missing_msbs = job.missing_msbs;
    params.num_bitplanes = job.num_bitplanes;
    params.number_of_coding_passes = job.number_of_coding_passes;
    params.output_stride = job.output_stride;
    params.output_offset = job.output_offset;
    params.dequantization_step = job.dequantization_step;
    params.stripe_causal = job.stripe_causal;

    decode_ht_cleanup_impl<true>(
        coded_data + job.coded_offset,
        decoded_data,
        params,
        vlc_table0,
        vlc_table1,
        uvlc_table0,
        uvlc_table1,
        status + gid
    );
}

extern "C" __global__ void signinum_j2k_dequantize_htj2k_codeblocks(
    uint *decoded_data,
    const J2kHtCleanupBatchJob *jobs
) {
    const uint job_idx = blockIdx.x;
    const J2kHtCleanupBatchJob job = jobs[job_idx];
    const uint sample_count = job.width * job.height;
    for (uint sample = threadIdx.x; sample < sample_count; sample += blockDim.x) {
        const uint y = sample / job.width;
        const uint x = sample - y * job.width;
        const uint idx = job.output_offset + y * job.output_stride + x;
        decoded_data[idx] = coefficient_to_float_bits(
            decoded_data[idx],
            job.num_bitplanes,
            job.dequantization_step
        );
    }
}

extern "C" __global__ void signinum_j2k_dequantize_htj2k_codeblocks_multi(
    const J2kHtDequantizeJob *jobs
) {
    const uint job_idx = blockIdx.x;
    const J2kHtDequantizeJob job = jobs[job_idx];
    uint *decoded_data = reinterpret_cast<uint *>(static_cast<uintptr_t>(job.output_ptr));
    const uint sample_count = job.width * job.height;
    for (uint sample = threadIdx.x; sample < sample_count; sample += blockDim.x) {
        const uint y = sample / job.width;
        const uint x = sample - y * job.width;
        const uint idx = job.output_offset + y * job.output_stride + x;
        decoded_data[idx] = coefficient_to_float_bits(
            decoded_data[idx],
            job.num_bitplanes,
            job.dequantization_step
        );
    }
}

extern "C" __global__ void signinum_j2k_dequantize_htj2k_cleanup_jobs_multi(
    const J2kHtCleanupMultiBatchJob *jobs
) {
    const uint job_idx = blockIdx.x;
    const J2kHtCleanupMultiBatchJob job = jobs[job_idx];
    uint *decoded_data = reinterpret_cast<uint *>(static_cast<uintptr_t>(job.output_ptr));
    const uint sample_count = job.width * job.height;
    for (uint sample = threadIdx.x; sample < sample_count; sample += blockDim.x) {
        const uint y = sample / job.width;
        const uint x = sample - y * job.width;
        const uint idx = job.output_offset + y * job.output_stride + x;
        decoded_data[idx] = coefficient_to_float_bits(
            decoded_data[idx],
            job.num_bitplanes,
            job.dequantization_step
        );
    }
}
