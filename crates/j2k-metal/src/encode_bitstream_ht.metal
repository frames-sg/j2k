constant uint J2K_HT_MAX_BITPLANES = 30u;
constant uint J2K_HT_MAX_SAMPLES = 16384u;
constant uint J2K_HT_MS_BYTES_PER_SAMPLE_FLOOR = 5u;
constant uint J2K_HT_MEL_SIZE = 192u;
constant uint J2K_HT_VLC_SIZE = 3072u - J2K_HT_MEL_SIZE;
constant uint J2K_HT_MS_SIZE = ((16384u * 16u) + 14u) / 15u;
constant uint J2K_HT_MEL_OFFSET = J2K_HT_MS_SIZE;
constant uint J2K_HT_VLC_OFFSET = J2K_HT_MS_SIZE + J2K_HT_MEL_SIZE;

struct J2kHtEncodeParams {
    uint width;
    uint height;
    uint total_bitplanes;
    uint output_capacity;
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
    uint offset;
    uint capacity;
};

struct J2kHtVlcEncoder {
    uint pos;
    uint used_bits;
    uchar tmp;
    uint last_greater_than_8f;
    uint failed;
    uint offset;
    uint capacity;
};

struct J2kHtMagSgnEncoder {
    uint pos;
    uint max_bits;
    uint used_bits;
    uint tmp;
    uint failed;
    uint capacity;
};

constant uint J2K_HT_MEL_EXP[13] = {
    0u, 0u, 0u, 1u, 1u, 1u, 2u, 2u, 2u, 3u, 3u, 4u, 5u
};

inline uint j2k_ht_scaled_scratch_size(uint max_size, uint sample_count) {
    const ulong scaled =
        (ulong(max_size) * ulong(sample_count) + ulong(J2K_HT_MAX_SAMPLES - 1u)) /
        ulong(J2K_HT_MAX_SAMPLES);
    return uint(max(ulong(1u), scaled));
}

inline uint j2k_ht_sample_count(J2kHtEncodeParams params) {
    return params.width * params.height;
}

inline uint j2k_ht_ms_size(J2kHtEncodeParams params) {
    const uint sample_count = j2k_ht_sample_count(params);
    const uint scaled = j2k_ht_scaled_scratch_size(J2K_HT_MS_SIZE, sample_count);
    const uint floor = sample_count * J2K_HT_MS_BYTES_PER_SAMPLE_FLOOR;
    return min(J2K_HT_MS_SIZE, max(scaled, floor));
}

inline uint j2k_ht_mel_size(J2kHtEncodeParams params) {
    return J2K_HT_MEL_SIZE;
}

inline uint j2k_ht_vlc_size(J2kHtEncodeParams params) {
    return J2K_HT_VLC_SIZE;
}

inline uint j2k_ht_mel_offset(J2kHtEncodeParams params) {
    return j2k_ht_ms_size(params);
}

inline uint j2k_ht_vlc_offset(J2kHtEncodeParams params) {
    return j2k_ht_ms_size(params) + j2k_ht_mel_size(params);
}

inline uint j2k_ht_output_size(J2kHtEncodeParams params) {
    return j2k_ht_ms_size(params) + j2k_ht_mel_size(params) + j2k_ht_vlc_size(params);
}

inline void j2k_set_ht_encode_status(
    device J2kHtEncodeStatus *status,
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

inline void j2k_set_ht_encode_status_with_segments(
    device J2kHtEncodeStatus *status,
    uint code,
    uint detail,
    uint data_len,
    uint passes,
    uint zbp,
    uint ms_len,
    uint mel_len,
    uint vlc_len
) {
    status->code = code;
    status->detail = detail;
    status->data_len = data_len;
    status->num_coding_passes = passes;
    status->num_zero_bitplanes = zbp;
    status->reserved0 = ms_len;
    status->reserved1 = mel_len;
    status->reserved2 = vlc_len;
}

inline uint j2k_ht_aligned_sign_magnitude(int coefficient, uint total_bitplanes) {
    if (coefficient == 0) {
        return 0u;
    }
    const uint sign = coefficient < 0 ? 0x80000000u : 0u;
    const uint magnitude = (coefficient < 0 ? uint(-coefficient) : uint(coefficient))
        << (31u - total_bitplanes);
    return sign | magnitude;
}

inline void j2k_ht_mel_init(thread J2kHtMelEncoder &mel, J2kHtEncodeParams params) {
    mel.pos = 0u;
    mel.remaining_bits = 8u;
    mel.tmp = uchar(0u);
    mel.run = 0u;
    mel.k = 0u;
    mel.threshold = 1u;
    mel.failed = 0u;
    mel.offset = j2k_ht_mel_offset(params);
    mel.capacity = j2k_ht_mel_size(params);
}

inline void j2k_ht_vlc_init(thread J2kHtVlcEncoder &vlc, device uchar *out, J2kHtEncodeParams params) {
    vlc.pos = 1u;
    vlc.used_bits = 4u;
    vlc.tmp = uchar(0x0Fu);
    vlc.last_greater_than_8f = 1u;
    vlc.failed = 0u;
    vlc.offset = j2k_ht_vlc_offset(params);
    vlc.capacity = j2k_ht_vlc_size(params);
    out[vlc.offset + vlc.capacity - 1u] = uchar(0xFFu);
}

inline void j2k_ht_ms_init(thread J2kHtMagSgnEncoder &ms, J2kHtEncodeParams params) {
    ms.pos = 0u;
    ms.max_bits = 8u;
    ms.used_bits = 0u;
    ms.tmp = 0u;
    ms.failed = 0u;
    ms.capacity = j2k_ht_ms_size(params);
}

inline void j2k_ht_mel_emit_bit(thread J2kHtMelEncoder &mel, device uchar *out, bool bit) {
    mel.tmp = uchar((uint(mel.tmp) << 1u) | (bit ? 1u : 0u));
    mel.remaining_bits -= 1u;
    if (mel.remaining_bits == 0u) {
        if (mel.pos >= mel.capacity) {
            mel.failed = 1u;
            return;
        }
        out[mel.offset + mel.pos] = mel.tmp;
        mel.pos += 1u;
        mel.remaining_bits = mel.tmp == uchar(0xFFu) ? 7u : 8u;
        mel.tmp = uchar(0u);
    }
}

inline void j2k_ht_mel_encode(thread J2kHtMelEncoder &mel, device uchar *out, bool bit) {
    if (!bit) {
        mel.run += 1u;
        if (mel.run >= mel.threshold) {
           j2k_ht_mel_emit_bit(mel, out, true);
            mel.run = 0u;
            mel.k = min(mel.k + 1u, 12u);
            mel.threshold = 1u << J2K_HT_MEL_EXP[mel.k];
        }
    } else {
       j2k_ht_mel_emit_bit(mel, out, false);
        uint t = J2K_HT_MEL_EXP[mel.k];
        while (t > 0u) {
            t -= 1u;
           j2k_ht_mel_emit_bit(mel, out, ((mel.run >> t) & 1u) != 0u);
        }
        mel.run = 0u;
        mel.k = mel.k == 0u ? 0u : mel.k - 1u;
        mel.threshold = 1u << J2K_HT_MEL_EXP[mel.k];
    }
}

inline void j2k_ht_vlc_encode(
    thread J2kHtVlcEncoder &vlc,
    device uchar *out,
    uint codeword,
    uint codeword_len
) {
    while (codeword_len > 0u) {
        if (vlc.pos >= vlc.capacity) {
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

            const uint write_index = vlc.capacity - 1u - vlc.pos;
            out[vlc.offset + write_index] = vlc.tmp;
            vlc.pos += 1u;
            vlc.last_greater_than_8f = vlc.tmp > uchar(0x8Fu) ? 1u : 0u;
            vlc.tmp = uchar(0u);
            vlc.used_bits = 0u;
        }
    }
}

inline void j2k_ht_ms_encode(
    thread J2kHtMagSgnEncoder &ms,
    device uchar *out,
    uint codeword,
    uint codeword_len
) {
    while (codeword_len > 0u) {
        if (ms.pos >= ms.capacity) {
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

inline void j2k_ht_ms_terminate(thread J2kHtMagSgnEncoder &ms, device uchar *out) {
    if (ms.used_bits > 0u) {
        const uint unused = ms.max_bits - ms.used_bits;
        ms.tmp |= (0xFFu & ((1u << unused) - 1u)) << ms.used_bits;
        ms.used_bits += unused;
        if (ms.tmp != 0xFFu) {
            if (ms.pos >= ms.capacity) {
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

inline void j2k_ht_process_sample(
    uint slot,
    uint value,
    uint p,
    thread int *rho_acc,
    thread int *e_q,
    thread int &e_qmax,
    thread uint *s
) {
    uint val = value + value;
    val >>= p;
    val &= ~1u;
    if (val != 0u) {
        rho_acc[0] |= int(1u << (slot & 0x3u));
        val -= 1u;
        e_q[slot] = int(32u - clz(val));
        e_qmax = max(e_qmax, e_q[slot]);
        val -= 1u;
        s[slot] = val + (value >> 31u);
    }
}

inline uchar j2k_ht_uvlc_byte(device const uchar *table, uint index, uint field) {
    return table[index * 6u + field];
}

inline void j2k_ht_encode_uvlc_pair(
    thread J2kHtVlcEncoder &vlc,
    device uchar *out,
    device const uchar *uvlc_table,
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

inline void j2k_ht_encode_uvlc(
    int u_q0,
    int u_q1,
    thread J2kHtVlcEncoder &vlc,
    device uchar *out,
    device const uchar *uvlc_table
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

inline void j2k_ht_encode_uvlc_non_initial(
    int u_q0,
    int u_q1,
    thread J2kHtVlcEncoder &vlc,
    device uchar *out,
    device const uchar *uvlc_table
) {
   j2k_ht_encode_uvlc_pair(
        vlc,
        out,
        uvlc_table,
        uint(max(u_q0, 0)),
        uint(max(u_q1, 0))
    );
}

inline void j2k_ht_encode_mag_signs(
    int rho,
    int u_q,
    ushort tuple,
    thread const uint *s,
    uint offset,
    thread J2kHtMagSgnEncoder &ms,
    device uchar *out
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

inline int j2k_ht_encode_quad_initial_row(
    uint offset,
    uint c_q,
    int rho,
    int e_qmax,
    thread const int *e_q,
    thread const uint *s,
    uint lep,
    uint lcxp,
    thread uchar *e_val,
    thread uchar *cx_val,
    thread J2kHtMelEncoder &mel,
    thread J2kHtVlcEncoder &vlc,
    thread J2kHtMagSgnEncoder &ms,
    device uchar *out,
    device const ushort *vlc_table0
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

inline int j2k_ht_encode_quad_non_initial_row(
    uint offset,
    uint c_q,
    int rho,
    int e_qmax,
    int max_e,
    thread const int *e_q,
    thread const uint *s,
    thread J2kHtMelEncoder &mel,
    thread J2kHtVlcEncoder &vlc,
    thread J2kHtMagSgnEncoder &ms,
    device uchar *out,
    device const ushort *vlc_table1
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

inline void j2k_ht_clear_quad_state(thread int *rho, thread int *e_q, thread int *e_qmax, thread uint *s) {
    rho[0] = 0;
    rho[1] = 0;
    for (uint idx = 0u; idx < 8u; ++idx) {
        e_q[idx] = 0;
        s[idx] = 0u;
    }
    e_qmax[0] = 0;
    e_qmax[1] = 0;
}

inline int j2k_ht_encode_first_quad_pair(
    device const int *coefficients,
    uint stride,
    uint height,
    uint total_bitplanes,
    uint p,
    thread uint &sp,
    uint x,
    thread uchar *e_val,
    thread uchar *cx_val,
    thread uint &c_q0,
    thread int *rho,
    thread int *e_q,
    thread int *e_qmax,
    thread uint *s,
    thread J2kHtMelEncoder &mel,
    thread J2kHtVlcEncoder &vlc,
    thread J2kHtMagSgnEncoder &ms,
    device uchar *out,
    device const ushort *vlc_table0,
    device const uchar *uvlc_table
) {
    const uint lep = x / 2u;
    const uint lcxp = x / 2u;

   j2k_ht_process_sample(0u, j2k_ht_aligned_sign_magnitude(coefficients[sp], total_bitplanes), p, &rho[0], e_q, e_qmax[0], s);
   j2k_ht_process_sample(
        1u,
        height > 1u ? j2k_ht_aligned_sign_magnitude(coefficients[sp + stride], total_bitplanes) : 0u,
        p,
        &rho[0],
        e_q,
        e_qmax[0],
        s
    );
    sp += 1u;

    if (x + 1u < stride) {
       j2k_ht_process_sample(2u, j2k_ht_aligned_sign_magnitude(coefficients[sp], total_bitplanes), p, &rho[0], e_q, e_qmax[0], s);
       j2k_ht_process_sample(
            3u,
            height > 1u ? j2k_ht_aligned_sign_magnitude(coefficients[sp + stride], total_bitplanes) : 0u,
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

    if (x + 2u < stride) {
       j2k_ht_process_sample(4u, j2k_ht_aligned_sign_magnitude(coefficients[sp], total_bitplanes), p, &rho[1], e_q, e_qmax[1], s);
       j2k_ht_process_sample(
            5u,
            height > 1u ? j2k_ht_aligned_sign_magnitude(coefficients[sp + stride], total_bitplanes) : 0u,
            p,
            &rho[1],
            e_q,
            e_qmax[1],
            s
        );
        sp += 1u;

        if (x + 3u < stride) {
           j2k_ht_process_sample(6u, j2k_ht_aligned_sign_magnitude(coefficients[sp], total_bitplanes), p, &rho[1], e_q, e_qmax[1], s);
           j2k_ht_process_sample(
                7u,
                height > 1u ? j2k_ht_aligned_sign_magnitude(coefficients[sp + stride], total_bitplanes) : 0u,
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

inline int j2k_ht_encode_non_initial_quad_pair(
    device const int *coefficients,
    uint stride,
    uint width,
    uint height,
    uint y,
    uint total_bitplanes,
    uint p,
    thread uint &sp,
    uint x,
    thread uchar *e_val,
    thread uchar *cx_val,
    thread uint &lep,
    thread uint &lcxp,
    thread int &max_e,
    thread uint &c_q0,
    thread int *rho,
    thread int *e_q,
    thread int *e_qmax,
    thread uint *s,
    thread J2kHtMelEncoder &mel,
    thread J2kHtVlcEncoder &vlc,
    thread J2kHtMagSgnEncoder &ms,
    device uchar *out,
    device const ushort *vlc_table1,
    device const uchar *uvlc_table
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

inline void j2k_ht_terminate_mel_vlc(
    thread J2kHtMelEncoder &mel,
    thread J2kHtVlcEncoder &vlc,
    device uchar *out
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
        if (mel.pos >= mel.capacity) {
            mel.failed = 1u;
            return;
        }
        out[mel.offset + mel.pos] = fused;
        mel.pos += 1u;
    } else {
        if (mel.pos >= mel.capacity || vlc.pos >= vlc.capacity) {
            mel.failed = 1u;
            vlc.failed = 1u;
            return;
        }
        out[mel.offset + mel.pos] = mel.tmp;
        mel.pos += 1u;
        const uint write_index = vlc.capacity - 1u - vlc.pos;
        out[vlc.offset + write_index] = vlc.tmp;
        vlc.pos += 1u;
    }
}

inline void j2k_encode_ht_code_block_impl_with_max_and_assembly(
    device const int *coefficients,
    device uchar *out,
    J2kHtEncodeParams params,
    device const ushort *vlc_table0,
    device const ushort *vlc_table1,
    device const uchar *uvlc_table,
    device J2kHtEncodeStatus *status,
    uint max_magnitude,
    bool assemble_final
) {
   j2k_set_ht_encode_status(status, J2K_ENCODE_STATUS_FAIL, 0u, 0u, 0u, 0u);

    if (params.width == 0u || params.height == 0u ||
        params.total_bitplanes == 0u || params.total_bitplanes > J2K_HT_MAX_BITPLANES ||
        params.width * params.height > J2K_HT_MAX_SAMPLES ||
        params.output_capacity < j2k_ht_output_size(params)) {
       j2k_set_ht_encode_status(status, J2K_ENCODE_STATUS_UNSUPPORTED, 1u, 0u, 0u, 0u);
        return;
    }

    if (max_magnitude == 0u) {
       j2k_set_ht_encode_status(status, J2K_ENCODE_STATUS_OK, 0u, 0u, 0u, params.total_bitplanes);
        return;
    }

    const uint block_bitplanes = 32u - clz(max_magnitude);
    if (block_bitplanes > params.total_bitplanes) {
       j2k_set_ht_encode_status(status, J2K_ENCODE_STATUS_FAIL, 2u, 0u, 0u, 0u);
        return;
    }

    const uint missing_msbs = params.total_bitplanes - 1u;
    const uint p = 30u - missing_msbs;

    thread J2kHtMelEncoder mel;
    thread J2kHtVlcEncoder vlc;
    thread J2kHtMagSgnEncoder ms;
   j2k_ht_mel_init(mel, params);
   j2k_ht_vlc_init(vlc, out, params);
   j2k_ht_ms_init(ms, params);

    thread uchar e_val[513];
    thread uchar cx_val[513];
    for (uint idx = 0u; idx < 513u; ++idx) {
        e_val[idx] = uchar(0u);
        cx_val[idx] = uchar(0u);
    }

    thread int e_qmax[2];
    thread int e_q[8];
    thread int rho[2];
    thread uint s[8];
   j2k_ht_clear_quad_state(rho, e_q, e_qmax, s);

    uint c_q0 = 0u;
    uint sp = 0u;
    uint x = 0u;
    while (x < params.width) {
       j2k_ht_encode_first_quad_pair(
            coefficients,
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

        sp = y * params.width;
        x = 0u;
        while (x < params.width) {
           j2k_ht_encode_non_initial_quad_pair(
                coefficients,
                params.width,
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
        const uint fail_detail = ms.failed != 0u ? 32u : (vlc.failed != 0u ? 31u : 30u);
       j2k_set_ht_encode_status(status, J2K_ENCODE_STATUS_FAIL, fail_detail, 0u, 0u, 0u);
        return;
    }

    const uint ms_len = ms.pos;
    const uint mel_len = mel.pos;
    const uint vlc_len = vlc.pos;
    const uint total_len = ms_len + mel_len + vlc_len;
    if (total_len < 2u || total_len > params.output_capacity) {
       j2k_set_ht_encode_status(status, J2K_ENCODE_STATUS_FAIL, 4u, 0u, 0u, 0u);
        return;
    }

    if (assemble_final) {
        for (uint idx = 0u; idx < mel_len; ++idx) {
            out[ms_len + idx] = out[mel.offset + idx];
        }
        const uint vlc_start = vlc.capacity - vlc_len;
        for (uint idx = 0u; idx < vlc_len; ++idx) {
            out[ms_len + mel_len + idx] = out[vlc.offset + vlc_start + idx];
        }

        const uint last = total_len - 1u;
        const uint prev = total_len - 2u;
        const uint locator_bytes = mel_len + vlc_len;
        out[last] = uchar(locator_bytes >> 4u);
        out[prev] = uchar((out[prev] & uchar(0xF0u)) | uchar(locator_bytes & 0x0Fu));
    }

   j2k_set_ht_encode_status_with_segments(
        status,
        J2K_ENCODE_STATUS_OK,
        0u,
        total_len,
        1u,
        missing_msbs,
        ms_len,
        mel_len,
        vlc_len
    );
}

inline void j2k_encode_ht_code_block_impl_with_max(
    device const int *coefficients,
    device uchar *out,
    J2kHtEncodeParams params,
    device const ushort *vlc_table0,
    device const ushort *vlc_table1,
    device const uchar *uvlc_table,
    device J2kHtEncodeStatus *status,
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

inline void j2k_encode_ht_code_block_impl(
    device const int *coefficients,
    device uchar *out,
    J2kHtEncodeParams params,
    device const ushort *vlc_table0,
    device const ushort *vlc_table1,
    device const uchar *uvlc_table,
    device J2kHtEncodeStatus *status
) {
    uint max_magnitude = 0u;
    for (uint y = 0u; y < params.height; ++y) {
        for (uint x = 0u; x < params.width; ++x) {
            max_magnitude = max(max_magnitude, j2k_classic_magnitude(coefficients[y * params.width + x]));
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

struct J2kHtEncodeBatchJob {
    uint coefficient_offset;
    uint output_offset;
    uint width;
    uint height;
    uint total_bitplanes;
    uint output_capacity;
};

kernel void j2k_encode_ht_code_block(
    device const int *coefficients [[buffer(0)]],
    device uchar *out [[buffer(1)]],
    constant J2kHtEncodeParams &params [[buffer(2)]],
    device const ushort *vlc_table0 [[buffer(3)]],
    device const ushort *vlc_table1 [[buffer(4)]],
    device const uchar *uvlc_table [[buffer(5)]],
    device J2kHtEncodeStatus *status [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) {
        return;
    }
   j2k_encode_ht_code_block_impl(
        coefficients,
        out,
        params,
        vlc_table0,
        vlc_table1,
        uvlc_table,
        status
    );
}

kernel void j2k_encode_ht_code_blocks(
    device const int *coefficients [[buffer(0)]],
    device uchar *out [[buffer(1)]],
    device const J2kHtEncodeBatchJob *jobs [[buffer(2)]],
    device const ushort *vlc_table0 [[buffer(3)]],
    device const ushort *vlc_table1 [[buffer(4)]],
    device const uchar *uvlc_table [[buffer(5)]],
    device J2kHtEncodeStatus *statuses [[buffer(6)]],
    constant uint &job_count [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= job_count) {
        return;
    }
    const J2kHtEncodeBatchJob job = jobs[gid];
    J2kHtEncodeParams params;
    params.width = job.width;
    params.height = job.height;
    params.total_bitplanes = job.total_bitplanes;
    params.output_capacity = job.output_capacity;
   j2k_encode_ht_code_block_impl(
        coefficients + job.coefficient_offset,
        out + job.output_offset,
        params,
        vlc_table0,
        vlc_table1,
        uvlc_table,
        statuses + gid
    );
}
