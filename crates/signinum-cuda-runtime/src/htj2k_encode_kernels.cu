// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long long j2k_ulong;

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif
#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

__device__ inline uint j2k_classic_magnitude(int value) {
    return value < 0 ? uint(-value) : uint(value);
}

static constexpr uint J2K_ENCODE_STATUS_OK = 0u;
static constexpr uint J2K_ENCODE_STATUS_FAIL = 1u;
static constexpr uint J2K_ENCODE_STATUS_UNSUPPORTED = 2u;

static constexpr uint J2K_HT_MAX_BITPLANES = 30u;
static constexpr uint J2K_HT_MEL_SIZE = 192u;
static constexpr uint J2K_HT_VLC_SIZE = 3072u - J2K_HT_MEL_SIZE;
static constexpr uint J2K_HT_MS_SIZE = ((16384u * 16u) + 14u) / 15u;
static constexpr uint J2K_HT_MEL_OFFSET = J2K_HT_MS_SIZE;
static constexpr uint J2K_HT_VLC_OFFSET = J2K_HT_MS_SIZE + J2K_HT_MEL_SIZE;

struct J2kHtEncodeParams {
    uint width;
    uint height;
    uint coefficient_stride;
    uint total_bitplanes;
    uint output_capacity;
    uint target_coding_passes;
};

struct J2kHtEncodeStatus {
    uint code;
    uint detail;
    uint data_len;
    uint num_coding_passes;
    uint num_zero_bitplanes;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

struct J2kHtMelEncoder {
    uint pos;
    uint remaining_bits;
    uchar tmp;
    uint run;
    uint k;
    uint threshold;
    uint failed;
};

struct J2kHtVlcEncoder {
    uint pos;
    uint used_bits;
    uchar tmp;
    uint last_greater_than_8f;
    uint failed;
};

struct J2kHtMagSgnEncoder {
    uint pos;
    uint max_bits;
    uint used_bits;
    uint tmp;
    uint failed;
};

__device__ inline uint j2k_ht_mel_exp(uint k) {
    return k < 3u ? 0u
        : k < 6u ? 1u
        : k < 9u ? 2u
        : k < 11u ? 3u
        : k == 11u ? 4u
        : 5u;
}

__device__ inline void j2k_set_ht_encode_status(
    J2kHtEncodeStatus *status,
    uint code,
    uint detail,
    uint data_len,
    uint passes,
    uint zbp
) {
    status->code = code;
    status->detail = detail;
    status->data_len = data_len;
    status->num_coding_passes = passes;
    status->num_zero_bitplanes = zbp;
    status->reserved0 = 0u;
    status->reserved1 = 0u;
    status->reserved2 = 0u;
}

__device__ inline void j2k_set_ht_encode_status_with_segments(
    J2kHtEncodeStatus *status,
    uint code,
    uint detail,
    uint data_len,
    uint passes,
    uint zbp,
    uint cleanup_len,
    uint refinement_len,
    uint reserved
) {
    status->code = code;
    status->detail = detail;
    status->data_len = data_len;
    status->num_coding_passes = passes;
    status->num_zero_bitplanes = zbp;
    status->reserved0 = cleanup_len;
    status->reserved1 = refinement_len;
    status->reserved2 = reserved;
}

__device__ inline uint j2k_ht_aligned_sign_magnitude(int coefficient, uint total_bitplanes) {
    if (coefficient == 0) {
        return 0u;
    }
    const uint sign = coefficient < 0 ? 0x80000000u : 0u;
    const uint magnitude = (coefficient < 0 ? uint(-coefficient) : uint(coefficient))
        << (31u - total_bitplanes);
    return sign | magnitude;
}

__device__ inline void j2k_ht_mel_init(J2kHtMelEncoder &mel) {
    mel.pos = 0u;
    mel.remaining_bits = 8u;
    mel.tmp = uchar(0u);
    mel.run = 0u;
    mel.k = 0u;
    mel.threshold = 1u;
    mel.failed = 0u;
}

__device__ inline void j2k_ht_vlc_init(J2kHtVlcEncoder &vlc, uchar *out) {
    vlc.pos = 1u;
    vlc.used_bits = 4u;
    vlc.tmp = uchar(0x0Fu);
    vlc.last_greater_than_8f = 1u;
    vlc.failed = 0u;
    out[J2K_HT_VLC_OFFSET + J2K_HT_VLC_SIZE - 1u] = uchar(0xFFu);
}

__device__ inline void j2k_ht_ms_init(J2kHtMagSgnEncoder &ms) {
    ms.pos = 0u;
    ms.max_bits = 8u;
    ms.used_bits = 0u;
    ms.tmp = 0u;
    ms.failed = 0u;
}

__device__ inline void j2k_ht_mel_emit_bit(J2kHtMelEncoder &mel, uchar *out, bool bit) {
    mel.tmp = uchar((uint(mel.tmp) << 1u) | (bit ? 1u : 0u));
    mel.remaining_bits -= 1u;
    if (mel.remaining_bits == 0u) {
        if (mel.pos >= J2K_HT_MEL_SIZE) {
            mel.failed = 1u;
            return;
        }
        out[J2K_HT_MEL_OFFSET + mel.pos] = mel.tmp;
        mel.pos += 1u;
        mel.remaining_bits = mel.tmp == uchar(0xFFu) ? 7u : 8u;
        mel.tmp = uchar(0u);
    }
}

__device__ inline void j2k_ht_mel_encode(J2kHtMelEncoder &mel, uchar *out, bool bit) {
    if (!bit) {
        mel.run += 1u;
        if (mel.run >= mel.threshold) {
            j2k_ht_mel_emit_bit(mel, out, true);
            mel.run = 0u;
            mel.k = min(mel.k + 1u, 12u);
            mel.threshold = 1u << j2k_ht_mel_exp(mel.k);
        }
    } else {
        j2k_ht_mel_emit_bit(mel, out, false);
        uint t = j2k_ht_mel_exp(mel.k);
        while (t > 0u) {
            t -= 1u;
            j2k_ht_mel_emit_bit(mel, out, ((mel.run >> t) & 1u) != 0u);
        }
        mel.run = 0u;
        mel.k = mel.k == 0u ? 0u : mel.k - 1u;
        mel.threshold = 1u << j2k_ht_mel_exp(mel.k);
    }
}

__device__ inline void j2k_ht_vlc_encode(
    J2kHtVlcEncoder &vlc,
    uchar *out,
    uint codeword,
    uint codeword_len
) {
    while (codeword_len > 0u) {
        if (vlc.pos >= J2K_HT_VLC_SIZE) {
            vlc.failed = 1u;
            return;
        }

        uint available_bits = 8u - vlc.last_greater_than_8f - vlc.used_bits;
        const uint take = min(available_bits, codeword_len);
        const uint mask = take == 32u ? 0xFFFFFFFFu : ((1u << take) - 1u);
        vlc.tmp = uchar(uint(vlc.tmp) | ((codeword & mask) << vlc.used_bits));
        vlc.used_bits += take;
        available_bits -= take;
        codeword_len -= take;
        codeword >>= take;

        if (available_bits == 0u) {
            if (vlc.last_greater_than_8f != 0u && vlc.tmp != uchar(0x7Fu)) {
                vlc.last_greater_than_8f = 0u;
                continue;
            }

            const uint write_index = J2K_HT_VLC_SIZE - 1u - vlc.pos;
            out[J2K_HT_VLC_OFFSET + write_index] = vlc.tmp;
            vlc.pos += 1u;
            vlc.last_greater_than_8f = vlc.tmp > uchar(0x8Fu) ? 1u : 0u;
            vlc.tmp = uchar(0u);
            vlc.used_bits = 0u;
        }
    }
}

__device__ inline void j2k_ht_ms_encode(
    J2kHtMagSgnEncoder &ms,
    uchar *out,
    uint codeword,
    uint codeword_len
) {
    while (codeword_len > 0u) {
        if (ms.pos >= J2K_HT_MS_SIZE) {
            ms.failed = 1u;
            return;
        }

        const uint take = min(ms.max_bits - ms.used_bits, codeword_len);
        const uint mask = take == 32u ? 0xFFFFFFFFu : ((1u << take) - 1u);
        ms.tmp |= (codeword & mask) << ms.used_bits;
        ms.used_bits += take;
        codeword >>= take;
        codeword_len -= take;

        if (ms.used_bits >= ms.max_bits) {
            out[ms.pos] = uchar(ms.tmp);
            ms.pos += 1u;
            ms.max_bits = ms.tmp == 0xFFu ? 7u : 8u;
            ms.tmp = 0u;
            ms.used_bits = 0u;
        }
    }
}

__device__ inline void j2k_ht_ms_terminate(J2kHtMagSgnEncoder &ms, uchar *out) {
    if (ms.used_bits > 0u) {
        const uint unused = ms.max_bits - ms.used_bits;
        ms.tmp |= (0xFFu & ((1u << unused) - 1u)) << ms.used_bits;
        ms.used_bits += unused;
        if (ms.tmp != 0xFFu) {
            if (ms.pos >= J2K_HT_MS_SIZE) {
                ms.failed = 1u;
                return;
            }
            out[ms.pos] = uchar(ms.tmp);
            ms.pos += 1u;
        }
    } else if (ms.max_bits == 7u) {
        ms.pos = ms.pos == 0u ? 0u : ms.pos - 1u;
    }
}

__device__ inline void j2k_ht_process_sample(
    uint slot,
    uint value,
    uint p,
    int *rho_acc,
    int *e_q,
    int &e_qmax,
    uint *s
) {
    uint val = value + value;
    val >>= p;
    val &= ~1u;
    if (val != 0u) {
        rho_acc[0] |= int(1u << (slot & 0x3u));
        val -= 1u;
        e_q[slot] = int(32u - __clz(val));
        e_qmax = max(e_qmax, e_q[slot]);
        val -= 1u;
        s[slot] = val + (value >> 31u);
    }
}

__device__ inline uchar j2k_ht_uvlc_byte(const uchar *table, uint index, uint field) {
    return table[index * 6u + field];
}

__device__ inline void j2k_ht_encode_uvlc_pair(
    J2kHtVlcEncoder &vlc,
    uchar *out,
    const uchar *uvlc_table,
    uint first_index,
    uint second_index
) {
    const uchar first_pre = j2k_ht_uvlc_byte(uvlc_table, first_index, 0u);
    const uchar first_pre_len = j2k_ht_uvlc_byte(uvlc_table, first_index, 1u);
    const uchar first_suf = j2k_ht_uvlc_byte(uvlc_table, first_index, 2u);
    const uchar first_suf_len = j2k_ht_uvlc_byte(uvlc_table, first_index, 3u);
    const uchar second_pre = j2k_ht_uvlc_byte(uvlc_table, second_index, 0u);
    const uchar second_pre_len = j2k_ht_uvlc_byte(uvlc_table, second_index, 1u);
    const uchar second_suf = j2k_ht_uvlc_byte(uvlc_table, second_index, 2u);
    const uchar second_suf_len = j2k_ht_uvlc_byte(uvlc_table, second_index, 3u);
    j2k_ht_vlc_encode(vlc, out, uint(first_pre), uint(first_pre_len));
    j2k_ht_vlc_encode(vlc, out, uint(second_pre), uint(second_pre_len));
    j2k_ht_vlc_encode(vlc, out, uint(first_suf), uint(first_suf_len));
    j2k_ht_vlc_encode(vlc, out, uint(second_suf), uint(second_suf_len));
}

__device__ inline void j2k_ht_encode_uvlc(
    int u_q0,
    int u_q1,
    J2kHtVlcEncoder &vlc,
    uchar *out,
    const uchar *uvlc_table
) {
    if (u_q0 > 2 && u_q1 > 2) {
        j2k_ht_encode_uvlc_pair(vlc, out, uvlc_table, uint(u_q0 - 2), uint(u_q1 - 2));
    } else if (u_q0 > 2 && u_q1 > 0) {
        const uint first_index = uint(u_q0);
        const uchar first_pre = j2k_ht_uvlc_byte(uvlc_table, first_index, 0u);
        const uchar first_pre_len = j2k_ht_uvlc_byte(uvlc_table, first_index, 1u);
        const uchar first_suf = j2k_ht_uvlc_byte(uvlc_table, first_index, 2u);
        const uchar first_suf_len = j2k_ht_uvlc_byte(uvlc_table, first_index, 3u);
        j2k_ht_vlc_encode(vlc, out, uint(first_pre), uint(first_pre_len));
        j2k_ht_vlc_encode(vlc, out, uint(u_q1 - 1), 1u);
        j2k_ht_vlc_encode(vlc, out, uint(first_suf), uint(first_suf_len));
    } else {
        j2k_ht_encode_uvlc_pair(
            vlc,
            out,
            uvlc_table,
            uint(max(u_q0, 0)),
            uint(max(u_q1, 0))
        );
    }
}

__device__ inline void j2k_ht_encode_uvlc_non_initial(
    int u_q0,
    int u_q1,
    J2kHtVlcEncoder &vlc,
    uchar *out,
    const uchar *uvlc_table
) {
    j2k_ht_encode_uvlc_pair(
        vlc,
        out,
        uvlc_table,
        uint(max(u_q0, 0)),
        uint(max(u_q1, 0))
    );
}

__device__ inline void j2k_ht_encode_mag_signs(
    int rho,
    int u_q,
    ushort tuple,
    const uint *s,
    uint offset,
    J2kHtMagSgnEncoder &ms,
    uchar *out
) {
    const uint e_k = uint(tuple & ushort(0xFu));
    for (uint bit = 0u; bit < 4u; ++bit) {
        const int sample_mask = int(1u << bit);
        if ((rho & sample_mask) == 0) {
            continue;
        }
        const int reduction = int((e_k >> bit) & 1u);
        const uint magnitude_bits = uint(u_q - reduction);
        const uint payload = magnitude_bits == 0u
            ? 0u
            : (s[offset + bit] & ((1u << magnitude_bits) - 1u));
        j2k_ht_ms_encode(ms, out, payload, magnitude_bits);
    }
}

__device__ inline int j2k_ht_encode_quad_initial_row(
    uint offset,
    uint c_q,
    int rho,
    int e_qmax,
    const int *e_q,
    const uint *s,
    uint lep,
    uint lcxp,
    uchar *e_val,
    uchar *cx_val,
    J2kHtMelEncoder &mel,
    J2kHtVlcEncoder &vlc,
    J2kHtMagSgnEncoder &ms,
    uchar *out,
    const ushort *vlc_table0
) {
    const int u_q = max(e_qmax, 1) - 1;
    uint eps = 0u;
    if (u_q > 0) {
        eps |= uint(e_q[offset] == e_qmax);
        eps |= uint(e_q[offset + 1u] == e_qmax) << 1u;
        eps |= uint(e_q[offset + 2u] == e_qmax) << 2u;
        eps |= uint(e_q[offset + 3u] == e_qmax) << 3u;
    }

    e_val[lep] = max(e_val[lep], uchar(e_q[offset + 1u]));
    e_val[lep + 1u] = uchar(e_q[offset + 3u]);
    cx_val[lcxp] = uchar(uint(cx_val[lcxp]) | uint((rho & 2) >> 1));
    cx_val[lcxp + 1u] = uchar((rho & 8) >> 3);

    const ushort tuple = vlc_table0[(c_q << 8u) | (uint(rho) << 4u) | eps];
    j2k_ht_vlc_encode(vlc, out, uint(tuple >> 8u), uint((tuple >> 4u) & ushort(0x7u)));
    if (c_q == 0u) {
        j2k_ht_mel_encode(mel, out, rho != 0);
    }
    j2k_ht_encode_mag_signs(rho, max(e_qmax, 1), tuple, s, offset, ms, out);
    return u_q;
}

__device__ inline int j2k_ht_encode_quad_non_initial_row(
    uint offset,
    uint c_q,
    int rho,
    int e_qmax,
    int max_e,
    const int *e_q,
    const uint *s,
    J2kHtMelEncoder &mel,
    J2kHtVlcEncoder &vlc,
    J2kHtMagSgnEncoder &ms,
    uchar *out,
    const ushort *vlc_table1
) {
    const int kappa = (rho & (rho - 1)) != 0 ? max(max_e, 1) : 1;
    const int u_q = max(e_qmax, kappa) - kappa;
    uint eps = 0u;
    if (u_q > 0) {
        eps |= uint(e_q[offset] == e_qmax);
        eps |= uint(e_q[offset + 1u] == e_qmax) << 1u;
        eps |= uint(e_q[offset + 2u] == e_qmax) << 2u;
        eps |= uint(e_q[offset + 3u] == e_qmax) << 3u;
    }

    const ushort tuple = vlc_table1[(c_q << 8u) | (uint(rho) << 4u) | eps];
    j2k_ht_vlc_encode(vlc, out, uint(tuple >> 8u), uint((tuple >> 4u) & ushort(0x7u)));
    if (c_q == 0u) {
        j2k_ht_mel_encode(mel, out, rho != 0);
    }
    j2k_ht_encode_mag_signs(rho, max(e_qmax, kappa), tuple, s, offset, ms, out);
    return u_q;
}

__device__ inline void j2k_ht_clear_quad_state(int *rho, int *e_q, int *e_qmax, uint *s) {
    rho[0] = 0;
    rho[1] = 0;
    for (uint idx = 0u; idx < 8u; ++idx) {
        e_q[idx] = 0;
        s[idx] = 0u;
    }
    e_qmax[0] = 0;
    e_qmax[1] = 0;
}

__device__ inline int j2k_ht_encode_first_quad_pair(
    const int *coefficients,
    uint source_stride,
    uint width,
    uint height,
    uint total_bitplanes,
    uint p,
    uint &sp,
    uint x,
    uchar *e_val,
    uchar *cx_val,
    uint &c_q0,
    int *rho,
    int *e_q,
    int *e_qmax,
    uint *s,
    J2kHtMelEncoder &mel,
    J2kHtVlcEncoder &vlc,
    J2kHtMagSgnEncoder &ms,
    uchar *out,
    const ushort *vlc_table0,
    const uchar *uvlc_table
) {
    const uint lep = x / 2u;
    const uint lcxp = x / 2u;

    j2k_ht_process_sample(0u, j2k_ht_aligned_sign_magnitude(coefficients[sp], total_bitplanes), p, &rho[0], e_q, e_qmax[0], s);
    j2k_ht_process_sample(
        1u,
        height > 1u ? j2k_ht_aligned_sign_magnitude(coefficients[sp + source_stride], total_bitplanes) : 0u,
        p,
        &rho[0],
        e_q,
        e_qmax[0],
        s
    );
    sp += 1u;

    if (x + 1u < width) {
        j2k_ht_process_sample(2u, j2k_ht_aligned_sign_magnitude(coefficients[sp], total_bitplanes), p, &rho[0], e_q, e_qmax[0], s);
        j2k_ht_process_sample(
            3u,
            height > 1u ? j2k_ht_aligned_sign_magnitude(coefficients[sp + source_stride], total_bitplanes) : 0u,
            p,
            &rho[0],
            e_q,
            e_qmax[0],
            s
        );
        sp += 1u;
    }

    const int u_q0 = j2k_ht_encode_quad_initial_row(
        0u, c_q0, rho[0], e_qmax[0], e_q, s, lep, lcxp, e_val, cx_val, mel, vlc, ms, out, vlc_table0
    );

    if (x + 2u < width) {
        j2k_ht_process_sample(4u, j2k_ht_aligned_sign_magnitude(coefficients[sp], total_bitplanes), p, &rho[1], e_q, e_qmax[1], s);
        j2k_ht_process_sample(
            5u,
            height > 1u ? j2k_ht_aligned_sign_magnitude(coefficients[sp + source_stride], total_bitplanes) : 0u,
            p,
            &rho[1],
            e_q,
            e_qmax[1],
            s
        );
        sp += 1u;

        if (x + 3u < width) {
            j2k_ht_process_sample(6u, j2k_ht_aligned_sign_magnitude(coefficients[sp], total_bitplanes), p, &rho[1], e_q, e_qmax[1], s);
            j2k_ht_process_sample(
                7u,
                height > 1u ? j2k_ht_aligned_sign_magnitude(coefficients[sp + source_stride], total_bitplanes) : 0u,
                p,
                &rho[1],
                e_q,
                e_qmax[1],
                s
            );
            sp += 1u;
        }

        const uint c_q1 = uint((rho[0] >> 1) | (rho[0] & 1));
        const int u_q1 = j2k_ht_encode_quad_initial_row(
            4u, c_q1, rho[1], e_qmax[1], e_q, s, lep + 1u, lcxp + 1u, e_val, cx_val, mel, vlc, ms, out, vlc_table0
        );

        if (u_q0 > 0 && u_q1 > 0) {
            j2k_ht_mel_encode(mel, out, min(u_q0, u_q1) > 2);
        }
        j2k_ht_encode_uvlc(u_q0, u_q1, vlc, out, uvlc_table);
        c_q0 = uint((rho[1] >> 1) | (rho[1] & 1));
    } else {
        j2k_ht_encode_uvlc(u_q0, 0, vlc, out, uvlc_table);
        c_q0 = 0u;
    }

    j2k_ht_clear_quad_state(rho, e_q, e_qmax, s);
    return 0;
}

__device__ inline int j2k_ht_encode_non_initial_quad_pair(
    const int *coefficients,
    uint stride,
    uint width,
    uint height,
    uint y,
    uint total_bitplanes,
    uint p,
    uint &sp,
    uint x,
    uchar *e_val,
    uchar *cx_val,
    uint &lep,
    uint &lcxp,
    int &max_e,
    uint &c_q0,
    int *rho,
    int *e_q,
    int *e_qmax,
    uint *s,
    J2kHtMelEncoder &mel,
    J2kHtVlcEncoder &vlc,
    J2kHtMagSgnEncoder &ms,
    uchar *out,
    const ushort *vlc_table1,
    const uchar *uvlc_table
) {
    j2k_ht_process_sample(0u, j2k_ht_aligned_sign_magnitude(coefficients[sp], total_bitplanes), p, &rho[0], e_q, e_qmax[0], s);
    j2k_ht_process_sample(
        1u,
        y + 1u < height ? j2k_ht_aligned_sign_magnitude(coefficients[sp + stride], total_bitplanes) : 0u,
        p,
        &rho[0],
        e_q,
        e_qmax[0],
        s
    );
    sp += 1u;

    if (x + 1u < width) {
        j2k_ht_process_sample(2u, j2k_ht_aligned_sign_magnitude(coefficients[sp], total_bitplanes), p, &rho[0], e_q, e_qmax[0], s);
        j2k_ht_process_sample(
            3u,
            y + 1u < height ? j2k_ht_aligned_sign_magnitude(coefficients[sp + stride], total_bitplanes) : 0u,
            p,
            &rho[0],
            e_q,
            e_qmax[0],
            s
        );
        sp += 1u;
    }

    const int prev_max = max_e;
    const int u_q0 = j2k_ht_encode_quad_non_initial_row(
        0u, c_q0, rho[0], e_qmax[0], prev_max, e_q, s, mel, vlc, ms, out, vlc_table1
    );

    e_val[lep] = max(e_val[lep], uchar(e_q[1]));
    lep += 1u;
    max_e = int(max(e_val[lep], e_val[lep + 1u])) - 1;
    e_val[lep] = uchar(e_q[3]);
    cx_val[lcxp] = uchar(uint(cx_val[lcxp]) | uint((rho[0] & 2) >> 1));
    lcxp += 1u;
    uint c_q1 = uint(cx_val[lcxp]) + (uint(cx_val[lcxp + 1u]) << 2u);
    cx_val[lcxp] = uchar((rho[0] & 8) >> 3);

    int u_q1 = 0;
    if (x + 2u < width) {
        j2k_ht_process_sample(4u, j2k_ht_aligned_sign_magnitude(coefficients[sp], total_bitplanes), p, &rho[1], e_q, e_qmax[1], s);
        j2k_ht_process_sample(
            5u,
            y + 1u < height ? j2k_ht_aligned_sign_magnitude(coefficients[sp + stride], total_bitplanes) : 0u,
            p,
            &rho[1],
            e_q,
            e_qmax[1],
            s
        );
        sp += 1u;

        if (x + 3u < width) {
            j2k_ht_process_sample(6u, j2k_ht_aligned_sign_magnitude(coefficients[sp], total_bitplanes), p, &rho[1], e_q, e_qmax[1], s);
            j2k_ht_process_sample(
                7u,
                y + 1u < height ? j2k_ht_aligned_sign_magnitude(coefficients[sp + stride], total_bitplanes) : 0u,
                p,
                &rho[1],
                e_q,
                e_qmax[1],
                s
            );
            sp += 1u;
        }

        c_q1 |= uint((rho[0] & 4) >> 1);
        c_q1 |= uint((rho[0] & 8) >> 2);
        u_q1 = j2k_ht_encode_quad_non_initial_row(
            4u, c_q1, rho[1], e_qmax[1], max_e, e_q, s, mel, vlc, ms, out, vlc_table1
        );

        e_val[lep] = max(e_val[lep], uchar(e_q[5]));
        lep += 1u;
        max_e = int(max(e_val[lep], e_val[lep + 1u])) - 1;
        e_val[lep] = uchar(e_q[7]);
        cx_val[lcxp] = uchar(uint(cx_val[lcxp]) | uint((rho[1] & 2) >> 1));
        lcxp += 1u;
        c_q0 = uint(cx_val[lcxp]) + (uint(cx_val[lcxp + 1u]) << 2u);
        cx_val[lcxp] = uchar((rho[1] & 8) >> 3);
        c_q0 |= uint((rho[1] & 4) >> 1);
        c_q0 |= uint((rho[1] & 8) >> 2);
    } else {
        c_q0 = 0u;
    }

    j2k_ht_encode_uvlc_non_initial(u_q0, u_q1, vlc, out, uvlc_table);
    j2k_ht_clear_quad_state(rho, e_q, e_qmax, s);
    return 0;
}

__device__ inline void j2k_ht_terminate_mel_vlc(
    J2kHtMelEncoder &mel,
    J2kHtVlcEncoder &vlc,
    uchar *out
) {
    if (mel.run > 0u) {
        j2k_ht_mel_emit_bit(mel, out, true);
    }

    mel.tmp = uchar(uint(mel.tmp) << mel.remaining_bits);
    const uchar mel_mask = uchar((0xFFu << mel.remaining_bits) & 0xFFu);
    const uchar vlc_mask = vlc.used_bits == 0u
        ? uchar(0u)
        : uchar((1u << vlc.used_bits) - 1u);

    if ((mel_mask | vlc_mask) == uchar(0u)) {
        return;
    }

    const uchar fused = mel.tmp | vlc.tmp;
    const bool fused_ok =
        ((((fused ^ mel.tmp) & mel_mask) | ((fused ^ vlc.tmp) & vlc_mask)) == uchar(0u)) &&
        fused != uchar(0xFFu);

    if (fused_ok && vlc.pos > 1u) {
        if (mel.pos >= J2K_HT_MEL_SIZE) {
            mel.failed = 1u;
            return;
        }
        out[J2K_HT_MEL_OFFSET + mel.pos] = fused;
        mel.pos += 1u;
    } else {
        if (mel.pos >= J2K_HT_MEL_SIZE || vlc.pos >= J2K_HT_VLC_SIZE) {
            mel.failed = 1u;
            vlc.failed = 1u;
            return;
        }
        out[J2K_HT_MEL_OFFSET + mel.pos] = mel.tmp;
        mel.pos += 1u;
        const uint write_index = J2K_HT_VLC_SIZE - 1u - vlc.pos;
        out[J2K_HT_VLC_OFFSET + write_index] = vlc.tmp;
        vlc.pos += 1u;
    }
}

__device__ inline void j2k_encode_ht_code_block_impl_with_max_and_assembly(
    const int *coefficients,
    uchar *out,
    J2kHtEncodeParams params,
    const ushort *vlc_table0,
    const ushort *vlc_table1,
    const uchar *uvlc_table,
    J2kHtEncodeStatus *status,
    uint max_magnitude,
    bool assemble_final
) {
    j2k_set_ht_encode_status(status, J2K_ENCODE_STATUS_FAIL, 0u, 0u, 0u, 0u);

    if (params.width == 0u || params.height == 0u || params.coefficient_stride < params.width ||
        params.total_bitplanes == 0u || params.total_bitplanes > J2K_HT_MAX_BITPLANES ||
        params.output_capacity < J2K_HT_MS_SIZE + J2K_HT_MEL_SIZE + J2K_HT_VLC_SIZE) {
        j2k_set_ht_encode_status(status, J2K_ENCODE_STATUS_UNSUPPORTED, 1u, 0u, 0u, 0u);
        return;
    }
    if (params.target_coding_passes == 0u || params.target_coding_passes > 164u) {
        j2k_set_ht_encode_status(status, J2K_ENCODE_STATUS_UNSUPPORTED, 5u, 0u, 0u, 0u);
        return;
    }
    if (params.target_coding_passes > 2u) {
        j2k_set_ht_encode_status(status, J2K_ENCODE_STATUS_UNSUPPORTED, 5u, 0u, 0u, 0u);
        return;
    }

    if (max_magnitude == 0u) {
        j2k_set_ht_encode_status(status, J2K_ENCODE_STATUS_OK, 0u, 0u, 1u, params.total_bitplanes);
        return;
    }

    const uint block_bitplanes = 32u - __clz(max_magnitude);
    if (block_bitplanes > params.total_bitplanes) {
        j2k_set_ht_encode_status(status, J2K_ENCODE_STATUS_FAIL, 2u, 0u, 0u, 0u);
        return;
    }

    const uint missing_msbs = params.total_bitplanes - 1u;
    const uint p = 30u - missing_msbs;

    J2kHtMelEncoder mel;
    J2kHtVlcEncoder vlc;
    J2kHtMagSgnEncoder ms;
    j2k_ht_mel_init(mel);
    j2k_ht_vlc_init(vlc, out);
    j2k_ht_ms_init(ms);

    uchar e_val[513];
    uchar cx_val[513];
    for (uint idx = 0u; idx < 513u; ++idx) {
        e_val[idx] = uchar(0u);
        cx_val[idx] = uchar(0u);
    }

    int e_qmax[2];
    int e_q[8];
    int rho[2];
    uint s[8];
    j2k_ht_clear_quad_state(rho, e_q, e_qmax, s);

    uint c_q0 = 0u;
    uint sp = 0u;
    uint x = 0u;
    while (x < params.width) {
        j2k_ht_encode_first_quad_pair(
            coefficients,
            params.coefficient_stride,
            params.width,
            params.height,
            params.total_bitplanes,
            p,
            sp,
            x,
            e_val,
            cx_val,
            c_q0,
            rho,
            e_q,
            e_qmax,
            s,
            mel,
            vlc,
            ms,
            out,
            vlc_table0,
            uvlc_table
        );
        x += 4u;
    }

    const uint e_val_sentinel = (params.width + 1u) / 2u + 1u;
    if (e_val_sentinel < 513u) {
        e_val[e_val_sentinel] = uchar(0u);
    }

    uint y = 2u;
    while (y < params.height) {
        uint lep = 0u;
        int max_e = int(max(e_val[lep], e_val[lep + 1u])) - 1;
        e_val[lep] = uchar(0u);

        uint lcxp = 0u;
        c_q0 = uint(cx_val[lcxp]) + (uint(cx_val[lcxp + 1u]) << 2u);
        cx_val[lcxp] = uchar(0u);

        sp = y * params.coefficient_stride;
        x = 0u;
        while (x < params.width) {
            j2k_ht_encode_non_initial_quad_pair(
                coefficients,
                params.coefficient_stride,
                params.width,
                params.height,
                y,
                params.total_bitplanes,
                p,
                sp,
                x,
                e_val,
                cx_val,
                lep,
                lcxp,
                max_e,
                c_q0,
                rho,
                e_q,
                e_qmax,
                s,
                mel,
                vlc,
                ms,
                out,
                vlc_table1,
                uvlc_table
            );
            x += 4u;
        }

        y += 2u;
    }

    j2k_ht_terminate_mel_vlc(mel, vlc, out);
    j2k_ht_ms_terminate(ms, out);

    if (mel.failed != 0u || vlc.failed != 0u || ms.failed != 0u) {
        j2k_set_ht_encode_status(status, J2K_ENCODE_STATUS_FAIL, 3u, 0u, 0u, 0u);
        return;
    }

    const uint ms_len = ms.pos;
    const uint mel_len = mel.pos;
    const uint vlc_len = vlc.pos;
    const uint cleanup_len = ms_len + mel_len + vlc_len;
    const uint refinement_len = params.target_coding_passes > 1u ? 1u : 0u;
    const uint total_len = cleanup_len + refinement_len;
    if (cleanup_len < 2u || total_len > params.output_capacity) {
        j2k_set_ht_encode_status(status, J2K_ENCODE_STATUS_FAIL, 4u, 0u, 0u, 0u);
        return;
    }

    if (assemble_final) {
        for (uint idx = 0u; idx < mel_len; ++idx) {
            out[ms_len + idx] = out[J2K_HT_MEL_OFFSET + idx];
        }
        const uint vlc_start = J2K_HT_VLC_SIZE - vlc_len;
        for (uint idx = 0u; idx < vlc_len; ++idx) {
            out[ms_len + mel_len + idx] = out[J2K_HT_VLC_OFFSET + vlc_start + idx];
        }

        const uint locator_bytes = mel_len + vlc_len;
        const uint cleanup_last = cleanup_len - 1u;
        const uint cleanup_prev = cleanup_len - 2u;
        out[cleanup_last] = uchar(locator_bytes >> 4u);
        out[cleanup_prev] = uchar((out[cleanup_prev] & uchar(0xF0u)) | uchar(locator_bytes & 0x0Fu));
        if (refinement_len != 0u) {
            out[cleanup_len] = uchar(0u);
        }
    }

    j2k_set_ht_encode_status_with_segments(
        status,
        J2K_ENCODE_STATUS_OK,
        0u,
        total_len,
        params.target_coding_passes,
        missing_msbs,
        cleanup_len,
        refinement_len,
        0u
    );
}

__device__ inline void j2k_encode_ht_code_block_impl_with_max(
    const int *coefficients,
    uchar *out,
    J2kHtEncodeParams params,
    const ushort *vlc_table0,
    const ushort *vlc_table1,
    const uchar *uvlc_table,
    J2kHtEncodeStatus *status,
    uint max_magnitude
) {
    j2k_encode_ht_code_block_impl_with_max_and_assembly(
        coefficients,
        out,
        params,
        vlc_table0,
        vlc_table1,
        uvlc_table,
        status,
        max_magnitude,
        true
    );
}

__device__ inline void j2k_encode_ht_code_block_impl(
    const int *coefficients,
    uchar *out,
    J2kHtEncodeParams params,
    const ushort *vlc_table0,
    const ushort *vlc_table1,
    const uchar *uvlc_table,
    J2kHtEncodeStatus *status
) {
    uint max_magnitude = 0u;
    for (uint y = 0u; y < params.height; ++y) {
        for (uint x = 0u; x < params.width; ++x) {
            max_magnitude = max(max_magnitude, j2k_classic_magnitude(coefficients[y * params.coefficient_stride + x]));
        }
    }
    j2k_encode_ht_code_block_impl_with_max(
        coefficients,
        out,
        params,
        vlc_table0,
        vlc_table1,
        uvlc_table,
        status,
        max_magnitude
    );
}

__device__ inline uint j2k_ht_reduce_max_magnitude_cooperative(
    const int *coefficients,
    uint width,
    uint height,
    uint coefficient_stride,
    uint *block_max
) {
    if (width == 0u || height == 0u) {
        return 0u;
    }

    const uint tid = threadIdx.x;
    uint local_max = 0u;
    const uint sample_count = width * height;
    for (uint sample = tid; sample < sample_count; sample += blockDim.x) {
        const uint y = sample / width;
        const uint x = sample - y * width;
        local_max = max(
            local_max,
            j2k_classic_magnitude(coefficients[y * coefficient_stride + x])
        );
    }

    block_max[tid] = local_max;
    __syncthreads();

    for (uint stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            block_max[tid] = max(block_max[tid], block_max[tid + stride]);
        }
        __syncthreads();
    }

    return block_max[0];
}


extern "C" __global__ void signinum_htj2k_encode_codeblock(
    const int *coefficients,
    uchar *out,
    const J2kHtEncodeParams *params,
    const ushort *vlc_table0,
    const ushort *vlc_table1,
    const uchar *uvlc_table,
    J2kHtEncodeStatus *status
) {
    if (blockIdx.x != 0u) {
        return;
    }
    __shared__ uint block_max[256];
    const J2kHtEncodeParams params_value = params[0];
    const uint max_magnitude = j2k_ht_reduce_max_magnitude_cooperative(
        coefficients,
        params_value.width,
        params_value.height,
        params_value.coefficient_stride,
        block_max
    );
    if (threadIdx.x != 0u) {
        return;
    }
    j2k_encode_ht_code_block_impl_with_max(
        coefficients,
        out,
        params_value,
        vlc_table0,
        vlc_table1,
        uvlc_table,
        status,
        max_magnitude
    );
}

struct J2kHtEncodeJob {
    uint coefficient_offset;
    uint coefficient_stride;
    uint width;
    uint height;
    uint total_bitplanes;
    uint output_offset;
    uint output_capacity;
    uint target_coding_passes;
};

extern "C" __global__ void signinum_htj2k_encode_codeblocks(
    const int *coefficients,
    uchar *out,
    const J2kHtEncodeJob *jobs,
    const ushort *vlc_table0,
    const ushort *vlc_table1,
    const uchar *uvlc_table,
    J2kHtEncodeStatus *statuses,
    unsigned long long job_count
) {
    const uint job_idx = blockIdx.x;
    if (j2k_ulong(job_idx) >= job_count) {
        return;
    }

    const J2kHtEncodeJob job = jobs[job_idx];
    J2kHtEncodeParams params;
    params.width = job.width;
    params.height = job.height;
    params.coefficient_stride = job.coefficient_stride;
    params.total_bitplanes = job.total_bitplanes;
    params.output_capacity = job.output_capacity;
    params.target_coding_passes = job.target_coding_passes;

    __shared__ uint block_max[256];
    const int *codeblock_coefficients = coefficients + job.coefficient_offset;
    const uint max_magnitude = j2k_ht_reduce_max_magnitude_cooperative(
        codeblock_coefficients,
        params.width,
        params.height,
        params.coefficient_stride,
        block_max
    );
    if (threadIdx.x != 0u) {
        return;
    }

    j2k_encode_ht_code_block_impl_with_max(
        codeblock_coefficients,
        out + job.output_offset,
        params,
        vlc_table0,
        vlc_table1,
        uvlc_table,
        &statuses[job_idx],
        max_magnitude
    );
}

struct J2kHtPacketJob {
    uint block_start;
    uint block_count;
    uint subband_start;
    uint subband_count;
    uint output_offset;
    uint output_capacity;
    uint layer;
};

struct J2kHtPacketSubband {
    uint block_start;
    uint block_count;
    uint num_cbs_x;
    uint num_cbs_y;
};

struct J2kHtPacketBlock {
    uint data_offset;
    uint data_len;
    uint cleanup_length;
    uint refinement_length;
    uint num_coding_passes;
    uint num_zero_bitplanes;
    uint l_block;
    uint previously_included;
    uint inclusion_layer;
};

struct J2kHtPacketSubbandTagState {
    uint inclusion_node_start;
    uint zero_bitplane_node_start;
    uint node_count;
    uint reserved0;
};

struct J2kHtPacketTagNodeState {
    uint current;
    uint known;
};

struct J2kHtPacketStatus {
    uint code;
    uint detail;
    uint output_len;
    uint reserved0;
};

struct J2kPacketBitWriter {
    uchar *out;
    uint pos;
    uint capacity;
    uint buffer;
    uint bits_in_buffer;
    uint last_byte_was_ff;
    uint failed;
};

static constexpr uint J2K_PACKET_TAG_INF = 0x7FFFFFFFu;
static constexpr uint J2K_PACKET_MAX_TAG_NODES = 2048u;
static constexpr uint J2K_PACKET_MAX_TAG_LEVELS = 16u;

struct J2kPacketTagTree {
    uint values[J2K_PACKET_MAX_TAG_NODES];
    uint current[J2K_PACKET_MAX_TAG_NODES];
    uint known[J2K_PACKET_MAX_TAG_NODES];
    uint widths[J2K_PACKET_MAX_TAG_LEVELS];
    uint heights[J2K_PACKET_MAX_TAG_LEVELS];
    uint offsets[J2K_PACKET_MAX_TAG_LEVELS];
    uint levels;
    uint total_nodes;
    uint failed;
};

__device__ inline void j2k_packet_status(
    J2kHtPacketStatus *status,
    uint code,
    uint detail,
    uint output_len
) {
    status->code = code;
    status->detail = detail;
    status->output_len = output_len;
    status->reserved0 = 0u;
}

__device__ inline void j2k_packet_writer_init(J2kPacketBitWriter &writer, uchar *out, uint capacity) {
    writer.out = out;
    writer.pos = 0u;
    writer.capacity = capacity;
    writer.buffer = 0u;
    writer.bits_in_buffer = 0u;
    writer.last_byte_was_ff = 0u;
    writer.failed = 0u;
}

__device__ inline void j2k_packet_push_byte(J2kPacketBitWriter &writer, uchar byte) {
    if (writer.pos >= writer.capacity) {
        writer.failed = 1u;
        return;
    }
    writer.out[writer.pos] = byte;
    writer.pos += 1u;
    writer.last_byte_was_ff = byte == uchar(0xFFu) ? 1u : 0u;
}

__device__ inline void j2k_packet_flush_byte(J2kPacketBitWriter &writer) {
    const uint limit = writer.last_byte_was_ff != 0u ? 7u : 8u;
    const uchar byte = uchar((writer.buffer >> (writer.bits_in_buffer - limit)) & 0xFFu);
    j2k_packet_push_byte(writer, byte);
    writer.bits_in_buffer -= limit;
    writer.buffer &= writer.bits_in_buffer == 0u ? 0u : ((1u << writer.bits_in_buffer) - 1u);
}

__device__ inline void j2k_packet_write_bit(J2kPacketBitWriter &writer, uint bit) {
    writer.buffer = (writer.buffer << 1u) | (bit & 1u);
    writer.bits_in_buffer += 1u;
    const uint limit = writer.last_byte_was_ff != 0u ? 7u : 8u;
    if (writer.bits_in_buffer >= limit) {
        j2k_packet_flush_byte(writer);
    }
}

__device__ inline void j2k_packet_write_bits(
    J2kPacketBitWriter &writer,
    uint value,
    uint count
) {
    while (count > 0u) {
        count -= 1u;
        j2k_packet_write_bit(writer, (value >> count) & 1u);
    }
}

__device__ inline void j2k_packet_finish(J2kPacketBitWriter &writer) {
    if (writer.bits_in_buffer == 0u) {
        return;
    }
    const uint limit = writer.last_byte_was_ff != 0u ? 7u : 8u;
    const uint shift = limit - writer.bits_in_buffer;
    const uchar byte = uchar((writer.buffer << shift) & 0xFFu);
    j2k_packet_push_byte(writer, byte);
    writer.buffer = 0u;
    writer.bits_in_buffer = 0u;
}

__device__ inline void j2k_packet_encode_num_ht_passes(
    J2kPacketBitWriter &writer,
    uint num_passes
) {
    if (num_passes == 1u) {
        j2k_packet_write_bit(writer, 0u);
    } else if (num_passes == 2u) {
        j2k_packet_write_bits(writer, 0b10u, 2u);
    } else if (num_passes <= 5u) {
        j2k_packet_write_bits(writer, 0b11u, 2u);
        j2k_packet_write_bits(writer, num_passes - 3u, 2u);
    } else if (num_passes <= 36u) {
        j2k_packet_write_bits(writer, 0b11u, 2u);
        j2k_packet_write_bits(writer, 0b11u, 2u);
        j2k_packet_write_bits(writer, num_passes - 6u, 5u);
    } else {
        j2k_packet_write_bits(writer, 0b11u, 2u);
        j2k_packet_write_bits(writer, 0b11u, 2u);
        j2k_packet_write_bits(writer, 31u, 5u);
        j2k_packet_write_bits(writer, num_passes - 37u, 7u);
    }
}

__device__ inline uint j2k_packet_value_fits(uint value, uint bits) {
    return bits >= 32u || value < (1u << bits);
}

__device__ inline uint j2k_packet_ht_length_bits(uint l_block, uint num_passes) {
    const uint placeholder_groups = (num_passes > 0u ? num_passes - 1u : 0u) / 3u;
    const uint placeholder_passes = placeholder_groups * 3u;
    uint value = placeholder_passes + 1u;
    uint log2_value = 0u;
    while (value > 1u) {
        value >>= 1u;
        log2_value += 1u;
    }
    return l_block + log2_value;
}

__device__ inline void j2k_packet_encode_length(
    J2kPacketBitWriter &writer,
    uint length,
    uint l_block,
    uint num_bits
) {
    while (!j2k_packet_value_fits(length, num_bits)) {
        j2k_packet_write_bit(writer, 1u);
        l_block += 1u;
        num_bits += 1u;
    }
    j2k_packet_write_bit(writer, 0u);
    j2k_packet_write_bits(writer, length, num_bits);
}

__device__ inline void j2k_packet_encode_ht_segment_lengths(
    J2kPacketBitWriter &writer,
    J2kHtPacketBlock block
) {
    const uint cleanup_length =
        (block.num_coding_passes == 1u && block.cleanup_length == 0u)
            ? block.data_len
            : block.cleanup_length;
    uint l_block = block.l_block;
    uint cleanup_bits = j2k_packet_ht_length_bits(l_block, block.num_coding_passes);
    const uint refinement_extra_bits = block.num_coding_passes > 2u ? 1u : 0u;
    while (!j2k_packet_value_fits(cleanup_length, cleanup_bits)
        || (block.num_coding_passes > 1u
            && !j2k_packet_value_fits(block.refinement_length, l_block + refinement_extra_bits))) {
        j2k_packet_write_bit(writer, 1u);
        l_block += 1u;
        cleanup_bits += 1u;
    }
    j2k_packet_write_bit(writer, 0u);
    j2k_packet_write_bits(writer, cleanup_length, cleanup_bits);

    if (block.num_coding_passes > 1u) {
        j2k_packet_write_bits(writer, block.refinement_length, l_block + refinement_extra_bits);
    }
}

__device__ inline uint j2k_packet_tag_tree_init(
    J2kPacketTagTree &tree,
    uint width,
    uint height
) {
    if (width == 0u || height == 0u) {
        tree.failed = 1u;
        return 0u;
    }

    uint w = width;
    uint h = height;
    uint total = 0u;
    uint levels = 0u;
    while (true) {
        if (levels >= J2K_PACKET_MAX_TAG_LEVELS) {
            tree.failed = 1u;
            return 0u;
        }
        const uint nodes = w * h;
        if (w != 0u && nodes / w != h) {
            tree.failed = 1u;
            return 0u;
        }
        if (total + nodes < total || total + nodes > J2K_PACKET_MAX_TAG_NODES) {
            tree.failed = 1u;
            return 0u;
        }
        tree.widths[levels] = w;
        tree.heights[levels] = h;
        tree.offsets[levels] = total;
        total += nodes;
        levels += 1u;
        if (w <= 1u && h <= 1u) {
            break;
        }
        w = (w + 1u) >> 1u;
        h = (h + 1u) >> 1u;
    }

    tree.levels = levels;
    tree.total_nodes = total;
    tree.failed = 0u;
    for (uint idx = 0u; idx < total; ++idx) {
        tree.values[idx] = 0u;
        tree.current[idx] = 0u;
        tree.known[idx] = 0u;
    }
    return 1u;
}

__device__ inline void j2k_packet_tag_tree_propagate(J2kPacketTagTree &tree) {
    for (uint level = 1u; level < tree.levels; ++level) {
        const uint prev_w = tree.widths[level - 1u];
        const uint prev_h = tree.heights[level - 1u];
        const uint curr_w = tree.widths[level];
        const uint curr_h = tree.heights[level];
        for (uint cy = 0u; cy < curr_h; ++cy) {
            for (uint cx = 0u; cx < curr_w; ++cx) {
                uint min_value = 0xFFFFFFFFu;
                const uint child_x0 = cx << 1u;
                const uint child_y0 = cy << 1u;
                const uint child_x1 = min(child_x0 + 2u, prev_w);
                const uint child_y1 = min(child_y0 + 2u, prev_h);
                for (uint y = child_y0; y < child_y1; ++y) {
                    for (uint x = child_x0; x < child_x1; ++x) {
                        const uint child_idx = tree.offsets[level - 1u] + y * prev_w + x;
                        min_value = min(min_value, tree.values[child_idx]);
                    }
                }
                tree.values[tree.offsets[level] + cy * curr_w + cx] = min_value;
            }
        }
    }
}

__device__ inline void j2k_packet_build_tag_trees(
    J2kPacketTagTree &inclusion_tree,
    J2kPacketTagTree &zbp_tree,
    const J2kHtPacketSubband &subband,
    const J2kHtPacketBlock *blocks,
    const J2kHtPacketSubbandTagState *tag_states,
    const J2kHtPacketTagNodeState *tag_nodes,
    j2k_ulong tag_state_count,
    j2k_ulong tag_node_count,
    uint subband_meta_idx
) {
    if (j2k_packet_tag_tree_init(inclusion_tree, subband.num_cbs_x, subband.num_cbs_y) == 0u
        || j2k_packet_tag_tree_init(zbp_tree, subband.num_cbs_x, subband.num_cbs_y) == 0u) {
        inclusion_tree.failed = 1u;
        zbp_tree.failed = 1u;
        return;
    }

    for (uint idx = 0u; idx < subband.block_count; ++idx) {
        const J2kHtPacketBlock block = blocks[subband.block_start + idx];
        const uint x = idx % subband.num_cbs_x;
        const uint y = idx / subband.num_cbs_x;
        const uint leaf_idx = y * subband.num_cbs_x + x;
        inclusion_tree.values[leaf_idx] =
            block.previously_included == 0u
                ? block.inclusion_layer
                : J2K_PACKET_TAG_INF;
        zbp_tree.values[leaf_idx] = block.num_zero_bitplanes;
    }
    j2k_packet_tag_tree_propagate(inclusion_tree);
    j2k_packet_tag_tree_propagate(zbp_tree);

    if (tag_state_count == 0u) {
        return;
    }
    if (j2k_ulong(subband_meta_idx) >= tag_state_count) {
        inclusion_tree.failed = 1u;
        zbp_tree.failed = 1u;
        return;
    }
    const J2kHtPacketSubbandTagState state = tag_states[subband_meta_idx];
    if (state.node_count != inclusion_tree.total_nodes) {
        inclusion_tree.failed = 1u;
        zbp_tree.failed = 1u;
        return;
    }
    const j2k_ulong inclusion_end = j2k_ulong(state.inclusion_node_start) + j2k_ulong(state.node_count);
    const j2k_ulong zbp_end = j2k_ulong(state.zero_bitplane_node_start) + j2k_ulong(state.node_count);
    if (inclusion_end < j2k_ulong(state.inclusion_node_start)
        || zbp_end < j2k_ulong(state.zero_bitplane_node_start)
        || inclusion_end > tag_node_count
        || zbp_end > tag_node_count) {
        inclusion_tree.failed = 1u;
        zbp_tree.failed = 1u;
        return;
    }
    for (uint idx = 0u; idx < state.node_count; ++idx) {
        const J2kHtPacketTagNodeState inclusion_node =
            tag_nodes[state.inclusion_node_start + idx];
        const J2kHtPacketTagNodeState zbp_node =
            tag_nodes[state.zero_bitplane_node_start + idx];
        inclusion_tree.current[idx] = inclusion_node.current;
        inclusion_tree.known[idx] = inclusion_node.known;
        zbp_tree.current[idx] = zbp_node.current;
        zbp_tree.known[idx] = zbp_node.known;
    }
}

__device__ inline void j2k_packet_tag_tree_encode(
    J2kPacketTagTree &tree,
    uint x,
    uint y,
    uint max_value,
    J2kPacketBitWriter &writer
) {
    uint path[J2K_PACKET_MAX_TAG_LEVELS];
    uint cx = x;
    uint cy = y;
    for (uint level = 0u; level < tree.levels; ++level) {
        path[level] = tree.offsets[level] + cy * tree.widths[level] + cx;
        cx >>= 1u;
        cy >>= 1u;
    }

    uint parent_value = 0u;
    for (uint reverse = tree.levels; reverse > 0u; --reverse) {
        const uint node_idx = path[reverse - 1u];
        uint start = max(tree.current[node_idx], parent_value);
        if (tree.known[node_idx] == 0u) {
            const uint target = min(tree.values[node_idx], max_value);
            while (start < target) {
                j2k_packet_write_bit(writer, 0u);
                start += 1u;
            }
            if (tree.values[node_idx] < max_value) {
                j2k_packet_write_bit(writer, 1u);
                tree.known[node_idx] = 1u;
            }
            tree.current[node_idx] = target;
        }
        parent_value = tree.current[node_idx];
    }
}

struct J2kPacketHeaderResult {
    uint code;
    uint detail;
    uint header_len;
    uint body_len;
    uint output_len;
};

__device__ inline J2kPacketHeaderResult j2k_packet_header_result(
    uint code,
    uint detail,
    uint header_len,
    uint body_len,
    uint output_len
) {
    J2kPacketHeaderResult result;
    result.code = code;
    result.detail = detail;
    result.header_len = header_len;
    result.body_len = body_len;
    result.output_len = output_len;
    return result;
}

__device__ inline J2kPacketHeaderResult j2k_packet_build_header_serial(
    unsigned long long payload_len,
    J2kHtPacketJob packet,
    const J2kHtPacketSubband *subbands,
    const J2kHtPacketBlock *blocks,
    const J2kHtPacketSubbandTagState *tag_states,
    const J2kHtPacketTagNodeState *tag_nodes,
    unsigned long long tag_state_count,
    unsigned long long tag_node_count,
    uchar *packet_out
) {
    J2kPacketBitWriter writer;
    j2k_packet_writer_init(writer, packet_out, packet.output_capacity);

    uint any_data = 0u;
    for (uint subband_idx = 0u; subband_idx < packet.subband_count; ++subband_idx) {
        const J2kHtPacketSubband subband = subbands[packet.subband_start + subband_idx];
        if (subband.num_cbs_x == 0u
            || subband.num_cbs_y == 0u
            || subband.num_cbs_x * subband.num_cbs_y != subband.block_count) {
            return j2k_packet_header_result(J2K_ENCODE_STATUS_FAIL, 7u, 0u, 0u, 0u);
        }
        for (uint idx = 0u; idx < subband.block_count; ++idx) {
            const J2kHtPacketBlock block = blocks[subband.block_start + idx];
            if (block.num_coding_passes > 0u) {
                any_data = 1u;
            }
            if (block.num_coding_passes > 164u) {
                return j2k_packet_header_result(
                    J2K_ENCODE_STATUS_UNSUPPORTED,
                    1u,
                    0u,
                    0u,
                    0u
                );
            }
            if (block.data_offset + block.data_len < block.data_offset
                || j2k_ulong(block.data_offset) + j2k_ulong(block.data_len) > payload_len) {
                return j2k_packet_header_result(J2K_ENCODE_STATUS_FAIL, 2u, 0u, 0u, 0u);
            }
            if (block.num_coding_passes == 0u) {
                if (block.data_len != 0u || block.cleanup_length != 0u || block.refinement_length != 0u) {
                    return j2k_packet_header_result(J2K_ENCODE_STATUS_FAIL, 10u, 0u, 0u, 0u);
                }
            } else if (block.num_coding_passes == 1u) {
                const uint cleanup_length =
                    block.cleanup_length == 0u ? block.data_len : block.cleanup_length;
                if (cleanup_length != block.data_len || block.refinement_length != 0u) {
                    return j2k_packet_header_result(J2K_ENCODE_STATUS_FAIL, 11u, 0u, 0u, 0u);
                }
            } else {
                const j2k_ulong segment_len =
                    j2k_ulong(block.cleanup_length) + j2k_ulong(block.refinement_length);
                if (block.cleanup_length == 0u
                    || block.refinement_length == 0u
                    || segment_len != j2k_ulong(block.data_len)
                    || block.cleanup_length < 2u
                    || block.cleanup_length >= 65535u
                    || block.refinement_length >= 2047u) {
                    return j2k_packet_header_result(J2K_ENCODE_STATUS_FAIL, 12u, 0u, 0u, 0u);
                }
            }
        }
    }

    if (any_data == 0u) {
        j2k_packet_write_bit(writer, 0u);
        j2k_packet_finish(writer);
        if (writer.failed != 0u) {
            return j2k_packet_header_result(J2K_ENCODE_STATUS_FAIL, 3u, 0u, 0u, 0u);
        }
        return j2k_packet_header_result(
            J2K_ENCODE_STATUS_OK,
            0u,
            writer.pos,
            0u,
            writer.pos
        );
    }

    j2k_packet_write_bit(writer, 1u);
    for (uint subband_idx = 0u; subband_idx < packet.subband_count; ++subband_idx) {
        const uint subband_meta_idx = packet.subband_start + subband_idx;
        const J2kHtPacketSubband subband = subbands[subband_meta_idx];
        J2kPacketTagTree inclusion_tree;
        J2kPacketTagTree zbp_tree;
        j2k_packet_build_tag_trees(
            inclusion_tree,
            zbp_tree,
            subband,
            blocks,
            tag_states,
            tag_nodes,
            tag_state_count,
            tag_node_count,
            subband_meta_idx
        );
        if (inclusion_tree.failed != 0u || zbp_tree.failed != 0u) {
            return j2k_packet_header_result(
                J2K_ENCODE_STATUS_UNSUPPORTED,
                8u,
                0u,
                0u,
                0u
            );
        }

        for (uint idx = 0u; idx < subband.block_count; ++idx) {
            const J2kHtPacketBlock block = blocks[subband.block_start + idx];
            const uint x = idx % subband.num_cbs_x;
            const uint y = idx / subband.num_cbs_x;
            if (block.previously_included == 0u) {
                if (block.num_coding_passes > 0u && block.inclusion_layer != packet.layer) {
                    return j2k_packet_header_result(J2K_ENCODE_STATUS_FAIL, 9u, 0u, 0u, 0u);
                }
                j2k_packet_tag_tree_encode(inclusion_tree, x, y, packet.layer + 1u, writer);
                if (block.num_coding_passes == 0u) {
                    continue;
                }
                j2k_packet_tag_tree_encode(zbp_tree, x, y, block.num_zero_bitplanes + 1u, writer);
            } else if (block.num_coding_passes > 0u) {
                j2k_packet_write_bit(writer, 1u);
            } else {
                j2k_packet_write_bit(writer, 0u);
                continue;
            }
            j2k_packet_encode_num_ht_passes(writer, block.num_coding_passes);
            j2k_packet_encode_ht_segment_lengths(writer, block);
        }
    }

    j2k_packet_finish(writer);
    if (writer.failed != 0u) {
        return j2k_packet_header_result(J2K_ENCODE_STATUS_FAIL, 4u, 0u, 0u, 0u);
    }
    if (writer.pos > 0u && packet_out[writer.pos - 1u] == uchar(0xFFu)) {
        if (writer.pos >= packet.output_capacity) {
            return j2k_packet_header_result(J2K_ENCODE_STATUS_FAIL, 5u, 0u, 0u, 0u);
        }
        packet_out[writer.pos] = uchar(0u);
        writer.pos += 1u;
    }

    const uint header_len = writer.pos;
    uint body_len = 0u;
    for (uint subband_idx = 0u; subband_idx < packet.subband_count; ++subband_idx) {
        const J2kHtPacketSubband subband = subbands[packet.subband_start + subband_idx];
        for (uint idx = 0u; idx < subband.block_count; ++idx) {
            const J2kHtPacketBlock block = blocks[subband.block_start + idx];
            if (block.num_coding_passes == 0u || block.data_len == 0u) {
                continue;
            }
            if (body_len + block.data_len < body_len
                || header_len + body_len + block.data_len < header_len
                || header_len + body_len + block.data_len > packet.output_capacity) {
                return j2k_packet_header_result(J2K_ENCODE_STATUS_FAIL, 6u, 0u, 0u, 0u);
            }
            body_len += block.data_len;
        }
    }

    return j2k_packet_header_result(
        J2K_ENCODE_STATUS_OK,
        0u,
        header_len,
        body_len,
        header_len + body_len
    );
}

__device__ inline void j2k_packet_copy_body_cooperative(
    const uchar *payload,
    J2kHtPacketJob packet,
    const J2kHtPacketSubband *subbands,
    const J2kHtPacketBlock *blocks,
    uchar *packet_out,
    uint header_len,
    uint body_len
) {
    for (uint body_byte = threadIdx.x; body_byte < body_len; body_byte += blockDim.x) {
        uint remaining = body_byte;
        for (uint subband_idx = 0u; subband_idx < packet.subband_count; ++subband_idx) {
            const J2kHtPacketSubband subband = subbands[packet.subband_start + subband_idx];
            for (uint idx = 0u; idx < subband.block_count; ++idx) {
                const J2kHtPacketBlock block = blocks[subband.block_start + idx];
                if (block.num_coding_passes == 0u || block.data_len == 0u) {
                    continue;
                }
                if (remaining < block.data_len) {
                    packet_out[header_len + body_byte] = payload[block.data_offset + remaining];
                    subband_idx = packet.subband_count;
                    break;
                }
                remaining -= block.data_len;
            }
        }
    }
}

extern "C" __global__ void signinum_htj2k_packetize_cleanup(
    const uchar *payload,
    unsigned long long payload_len,
    const J2kHtPacketJob *packets,
    const J2kHtPacketSubband *subbands,
    const J2kHtPacketBlock *blocks,
    const J2kHtPacketSubbandTagState *tag_states,
    const J2kHtPacketTagNodeState *tag_nodes,
    unsigned long long tag_state_count,
    unsigned long long tag_node_count,
    uchar *out,
    J2kHtPacketStatus *statuses,
    unsigned long long packet_count
) {
    const unsigned long long packet_idx = blockIdx.x;
    if (packet_idx >= packet_count) {
        return;
    }

    const J2kHtPacketJob packet = packets[packet_idx];
    J2kHtPacketStatus *status = &statuses[packet_idx];
    uchar *packet_out = out + packet.output_offset;
    __shared__ uint shared_code;
    __shared__ uint shared_header_len;
    __shared__ uint shared_body_len;

    if (threadIdx.x == 0u) {
        const J2kPacketHeaderResult result = j2k_packet_build_header_serial(
            payload_len,
            packet,
            subbands,
            blocks,
            tag_states,
            tag_nodes,
            tag_state_count,
            tag_node_count,
            packet_out
        );
        shared_code = result.code;
        shared_header_len = result.header_len;
        shared_body_len = result.body_len;
        j2k_packet_status(status, result.code, result.detail, result.output_len);
    }
    __syncthreads();

    if (shared_code != J2K_ENCODE_STATUS_OK || shared_body_len == 0u) {
        return;
    }
    j2k_packet_copy_body_cooperative(
        payload,
        packet,
        subbands,
        blocks,
        packet_out,
        shared_header_len,
        shared_body_len
    );
}
