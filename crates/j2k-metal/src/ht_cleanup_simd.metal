// SPDX-License-Identifier: MIT OR Apache-2.0

// Two-kernel HT cleanup decode for cleanup-only code blocks.
//
// 1. `..._vlc_...`: one thread per code block walks the serial MEL/VLC
//    streams. Thirty-two blocks share a SIMD group, so this latency-bound
//    walk uses every lane. For quad (q, r) it stores `(u_q << 16) | inf` in
//    the block's own output word at column 2q, row 2r, which the next kernel
//    overwrites; no scratch allocation is needed.
// 2. `..._magsgn_...`: one SIMD group per code block decodes each quad row's
//    MagSgn samples in parallel. Each lane owns one quad, derives its bit
//    count from the stored VLC data and the previous row's exponents, and
//    locates its bits with a SIMD prefix sum over a cooperatively destuffed
//    threadgroup window.
//
// Arithmetic, bit consumption, validation and status codes mirror
// `decode_ht_cleanup_common<true>`; only the schedule differs.

constant uint J2K_HT_SIMD_LANES = 32u;
constant uint J2K_HT_SIMD_BLOCKS_PER_GROUP = 4u;
// 8192 destuffed bits. One 32-quad chunk consumes at most 3968 bits, and each
// refill appends at most 1024 bits, so live bits never wrap onto themselves.
constant uint J2K_HT_SIMD_WINDOW_WORDS = 256u;
constant uint J2K_HT_SIMD_WINDOW_MASK = J2K_HT_SIMD_WINDOW_WORDS - 1u;
constant uint J2K_HT_SIMD_REFILL_BYTES = J2K_HT_SIMD_LANES * 4u;
// Previous/current row significance context: two bits per quad, 16 quads per
// word, covering J2K_HT_MAX_WIDTH / 2 quads plus the zero quad read past the
// row end. The odd per-thread stride keeps lanes on distinct banks.
constant uint J2K_HT_VLC_CTX_WORDS = 9u;
constant uint J2K_HT_VLC_CTX_STRIDE = 2u * J2K_HT_VLC_CTX_WORDS + 1u;
constant uint J2K_HT_VLC_THREADS_PER_GROUP = 32u;

// The VLC kernels run `lanes_per_simd` blocks per 32-thread SIMD group (the
// remaining lanes exit). Fewer blocks per group means less divergence between
// the serial walks; more means fewer groups to issue. The host picks the ratio
// from the job count.
struct J2kHtVlcDispatchParams {
    uint job_count;
    uint lanes_per_simd;
};

inline bool ht_vlc_job_index(
    J2kHtVlcDispatchParams dispatch,
    uint grid_x,
    thread uint &job_index
) {
    const uint lane = grid_x % J2K_HT_SIMD_LANES;
    job_index = (grid_x / J2K_HT_SIMD_LANES) * dispatch.lanes_per_simd + lane;
    return lane < dispatch.lanes_per_simd && job_index < dispatch.job_count;
}

struct HtCleanupGeometry {
    uint scup;
    uint quads;
    uint quad_rows;
};

// Validation shared by both kernels, in `decode_ht_cleanup_common` order.
// Returns the failing status code (and sets `detail`) or J2K_HT_STATUS_OK.
inline uint ht_simd_validate(
    device const uchar *coded_data,
    J2kHtCleanupParams params,
    thread HtCleanupGeometry &geometry,
    thread uint &detail
) {
    detail = 0u;
    geometry.scup = 0u;
    geometry.quads = 0u;
    geometry.quad_rows = 0u;
    if (params.number_of_coding_passes > 3u) {
        detail = 3u;
        return J2K_HT_STATUS_FAIL;
    }
    // Empty blocks decode nothing and report OK; callers then see zero rows.
    if (params.width == 0u || params.height == 0u) {
        return J2K_HT_STATUS_OK;
    }
    if (params.width > J2K_HT_MAX_WIDTH || params.height > J2K_HT_MAX_HEIGHT ||
        params.width * params.height > J2K_HT_MAX_COEFFICIENTS) {
        detail = 1u;
        return J2K_HT_STATUS_UNSUPPORTED;
    }
    if (params.num_bitplanes == 0u || params.num_bitplanes > 31u ||
        params.roi_shift > 31u - params.num_bitplanes) {
        detail = 2u;
        return J2K_HT_STATUS_FAIL;
    }
    if (params.missing_msbs > 30u || params.missing_msbs == 30u) {
        detail = 3u;
        return J2K_HT_STATUS_FAIL;
    }
    const uint lcup = params.cleanup_length;
    if (lcup < 2u || params.coded_len < lcup + params.refinement_length) {
        detail = 4u;
        return J2K_HT_STATUS_FAIL;
    }
    const uint scup = (uint(coded_data[lcup - 1u]) << 4u) + uint(coded_data[lcup - 2u] & uchar(0x0F));
    if (scup < 2u || scup > lcup || scup > 4079u) {
        detail = 5u;
        return J2K_HT_STATUS_FAIL;
    }
    const uint quad_rows = (params.height + 1u) / 2u;
    const uint sstr = (params.width + 9u) & ~7u;
    if (sstr > J2K_HT_MAX_SSTR || sstr * (quad_rows + 1u) > J2K_HT_MAX_SCRATCH) {
        detail = 6u;
        return J2K_HT_STATUS_UNSUPPORTED;
    }
    const uint quads = (params.width + 1u) / 2u;
    if (quads + 2u > J2K_HT_MAX_VN) {
        detail = 12u;
        return J2K_HT_STATUS_UNSUPPORTED;
    }
    geometry.scup = scup;
    geometry.quads = quads;
    geometry.quad_rows = quad_rows;
    return J2K_HT_STATUS_OK;
}

// MEL decoder yielding the same run sequence as `mel_get_run`, one run on
// demand from a 64-bit MSB-first bit buffer instead of bit by bit. A byte
// following 0xFF contributes seven bits, and the final segment byte has its
// low nibble forced to ones.
struct HtSimdMel {
    device const uchar *data;
    uint pos;
    uint remaining;
    ulong acc;
    uint acc_bits;
    uint unstuff;
    uint k;
};

inline HtSimdMel ht_simd_mel_new(device const uchar *data, uint lcup, uint scup) {
    HtSimdMel mel;
    mel.data = data;
    mel.pos = lcup - scup;
    mel.remaining = scup - 1u;
    mel.acc = 0ul;
    mel.acc_bits = 0u;
    mel.unstuff = 0u;
    mel.k = 0u;
    return mel;
}

inline int ht_simd_mel_get_run(thread HtSimdMel &mel) {
    // A run consumes at most six bits; one byte always supplies at least seven.
    if (mel.acc_bits < 6u) {
        uint byte = 0xFFu;
        if (mel.remaining > 0u) {
            byte = uint(mel.data[mel.pos]);
            mel.pos += 1u;
            mel.remaining -= 1u;
        }
        if (mel.remaining == 0u) {
            byte |= 0x0Fu;
        }
        const uint nbits = 8u - mel.unstuff;
        const uint value = byte & ((1u << nbits) - 1u);
        mel.acc |= ulong(value) << (64u - mel.acc_bits - nbits);
        mel.acc_bits += nbits;
        mel.unstuff = byte == 0xFFu ? 1u : 0u;
    }

    // MEL_EXP = {0,0,0,1,1,1,2,2,2,3,3,4,5}
    const uint eval = mel.k < 11u ? mel.k / 3u : mel.k - 7u;
    if ((mel.acc >> 63) != 0ul) {
        mel.k = min(mel.k + 1u, 12u);
        mel.acc <<= 1;
        mel.acc_bits -= 1u;
        return int(((1u << eval) - 1u) << 1u);
    }
    mel.k = mel.k == 0u ? 0u : mel.k - 1u;
    const uint bits = eval == 0u ? 0u : uint((mel.acc << 1) >> (64u - eval));
    mel.acc <<= 1u + eval;
    mel.acc_bits -= 1u + eval;
    return int((bits << 1u) | 1u);
}

// The non-initial VLC context reads only inf bits 5 and 7 of the quads above
// (the bottom samples' significance), so each quad is recorded as two bits.
inline uint ht_simd_ctx_bits(uint inf) {
    return ((inf >> 5u) & 1u) | ((inf >> 6u) & 2u);
}

inline uint ht_simd_ctx_inf(uint bits) {
    return ((bits & 1u) << 5u) | ((bits & 2u) << 6u);
}

// Context for blocks at most 64 samples wide (32 quads): the previous row is
// a register consumed two quads per pair, and the current row is accumulated
// from the top four bits at a time, so no loop step shifts by a variable.
struct HtNarrowVlcContext {
    ulong prev;
    ulong cur;
    uint pairs;

    uint above(uint offset) const {
        return ht_simd_ctx_inf(uint(prev >> (2u * offset)) & 3u);
    }

    void record_pair(uint inf0, uint inf1) {
        const ulong bits = ulong(ht_simd_ctx_bits(inf0) | (ht_simd_ctx_bits(inf1) << 2u));
        cur = (cur >> 4u) | (bits << 60u);
        prev >>= 4u;
        pairs += 1u;
    }

    void finish_row() {
        prev = pairs < 16u ? cur >> (4u * (16u - pairs)) : cur;
        cur = 0ul;
        pairs = 0u;
    }
};

inline HtNarrowVlcContext ht_narrow_vlc_context() {
    HtNarrowVlcContext ctx;
    ctx.prev = 0ul;
    ctx.cur = 0ul;
    ctx.pairs = 0u;
    return ctx;
}

// Context for wider blocks: two rows of 2-bit entries in this thread's slice
// of threadgroup memory, covering J2K_HT_MAX_WIDTH / 2 quads plus the zero
// quad read past the row end.
struct HtWideVlcContext {
    threadgroup uint *prev;
    threadgroup uint *cur;
    uint quad;

    uint above(uint offset) const {
        const uint q = quad + offset;
        return ht_simd_ctx_inf((prev[q >> 4] >> ((q & 15u) * 2u)) & 3u);
    }

    void record_pair(uint inf0, uint inf1) {
        const uint bits = ht_simd_ctx_bits(inf0) | (ht_simd_ctx_bits(inf1) << 2u);
        cur[quad >> 4] |= bits << ((quad & 15u) * 2u);
        quad += 2u;
    }

    void finish_row() {
        threadgroup uint *done = cur;
        cur = prev;
        prev = done;
        for (uint word = 0u; word < J2K_HT_VLC_CTX_WORDS; ++word) {
            cur[word] = 0u;
        }
        quad = 0u;
    }
};

inline HtWideVlcContext ht_wide_vlc_context(threadgroup uint *slice) {
    HtWideVlcContext ctx;
    ctx.prev = slice;
    ctx.cur = slice + J2K_HT_VLC_CTX_WORDS;
    ctx.quad = 0u;
    for (uint word = 0u; word < 2u * J2K_HT_VLC_CTX_WORDS; ++word) {
        slice[word] = 0u;
    }
    return ctx;
}

// Reverse VLC reader for the thread-per-block kernel. It exposes the same bit
// sequence as `ReverseBitReader`, but refills four bytes at a time from a
// register that was loaded during the previous refill, so a refill never waits
// on memory. Buffering extra bits cannot change results: a quad pair reads at
// most 30 bits (two 7-bit codewords and a 16-bit UVLC prefix plus suffix) and
// every fetch leaves more than 32 valid bits.
struct HtVlcReader {
    device const uchar *data;
    int pos;
    uint remaining;
    ulong tmp;
    uint bits;
    uint unstuff;
    uint ahead;
};

inline uint ht_vlc_load_ahead(device const uchar *data, int pos, uint remaining) {
    uint word = 0u;
    for (uint j = 0u; j < 4u; ++j) {
        const uint byte = remaining > j ? uint(data[pos - int(j)]) : 0u;
        word |= byte << (8u * j);
    }
    return word;
}

inline HtVlcReader ht_vlc_reader_new(device const uchar *data, uint lcup, uint scup) {
    const uchar d = data[lcup - 2u];
    const ulong tmp = ulong(d >> 4);
    HtVlcReader reader;
    reader.data = data;
    reader.pos = int(lcup) - 3;
    reader.remaining = scup - 2u;
    reader.tmp = tmp;
    reader.bits = 4u - uint((tmp & 0x7ul) == 0x7ul);
    reader.unstuff = (d | uchar(0x0F)) > uchar(0x8F) ? 1u : 0u;
    reader.ahead = ht_vlc_load_ahead(data, reader.pos, reader.remaining);
    return reader;
}

inline uint ht_vlc_fetch(thread HtVlcReader &reader) {
    while (reader.bits <= 32u) {
        const uint taken = min(reader.remaining, 4u);
        const uint word = reader.ahead;
        reader.pos -= int(taken);
        reader.remaining -= taken;
        reader.ahead = ht_vlc_load_ahead(reader.data, reader.pos, reader.remaining);
        for (uint j = 0u; j < 4u; ++j) {
            const uint raw = (word >> (8u * j)) & 0xFFu;
            const bool stuffed = reader.unstuff != 0u && (raw & 0x7Fu) == 0x7Fu;
            reader.tmp |= ulong(stuffed ? raw & 0x7Fu : raw) << reader.bits;
            reader.bits += stuffed ? 7u : 8u;
            reader.unstuff = raw > 0x8Fu ? 1u : 0u;
        }
    }
    return uint(reader.tmp);
}

inline void ht_simd_vlc_consume(thread HtVlcReader &vlc, uint count) {
    vlc.tmp >>= count;
    vlc.bits -= count;
}

inline uint ht_simd_quad_word(uint inf, uint u_q) {
    return (uint(ushort(u_q)) << 16u) | inf;
}

// Each quad pair's VLC codewords, UVLC prefix and UVLC suffix are located in
// one fetched window, and the reader advances once per pair. Every field is
// taken from the same bits `reverse_reader_advance` would expose.
template<typename Context>
inline void ht_simd_vlc_initial_row(
    thread HtSimdMel &mel,
    thread HtVlcReader &vlc,
    thread int &run,
    thread Context &ctx,
    device uint *quad_words,
    uint width,
    constant ushort *vlc_table0,
    constant ushort *uvlc_table0
) {
    uint c_q = 0u;
    uint quad = 0u;
    uint x = 0u;

    while (x < width) {
        const uint window = ht_vlc_fetch(vlc);
        uint t0 = uint(vlc_table0[c_q + (window & 0x7Fu)]);
        if (c_q == 0u) {
            run -= 2;
            t0 = run == -1 ? t0 : 0u;
            if (run < 0) {
                run = ht_simd_mel_get_run(mel);
            }
        }
        x += 2u;
        c_q = ((t0 & 0x10u) << 3u) | ((t0 & 0xE0u) << 2u);
        uint used = t0 & 0x7u;

        const bool second = x < width;
        uint t1 = uint(vlc_table0[c_q + ((window >> used) & 0x7Fu)]);
        if (c_q == 0u && second) {
            run -= 2;
            t1 = run == -1 ? t1 : 0u;
            if (run < 0) {
                run = ht_simd_mel_get_run(mel);
            }
        }
        if (!second) {
            t1 = 0u;
        }
        x += 2u;
        c_q = ((t1 & 0x10u) << 3u) | ((t1 & 0xE0u) << 2u);
        used += t1 & 0x7u;

        uint uvlc_mode = ((t0 & 0x8u) << 3u) | ((t1 & 0x8u) << 4u);
        if (uvlc_mode == 0xC0u) {
            run -= 2;
            if (run == -1) {
                uvlc_mode += 0x40u;
            }
            if (run < 0) {
                run = ht_simd_mel_get_run(mel);
            }
        }

        uint uvlc_entry = uint(uvlc_table0[uvlc_mode + ((window >> used) & 0x3Fu)]);
        used += uvlc_entry & 0x7u;
        uvlc_entry >>= 3u;
        uint len = uvlc_entry & 0xFu;
        const uint tmp = uint(vlc.tmp >> used) & ((1u << len) - 1u);
        ht_simd_vlc_consume(vlc, used + len);
        uvlc_entry >>= 4u;
        len = uvlc_entry & 0x7u;
        uvlc_entry >>= 3u;

        ctx.record_pair(t0, t1);
        quad_words[quad * 2u] =
            ht_simd_quad_word(t0, 1u + (uvlc_entry & 0x7u) + (tmp & ~(0xFFu << len)));
        if (second) {
            quad_words[quad * 2u + 2u] =
                ht_simd_quad_word(t1, 1u + (uvlc_entry >> 3u) + (tmp >> len));
        }
        quad += 2u;
    }
}

template<typename Context>
inline void ht_simd_vlc_non_initial_row(
    thread HtSimdMel &mel,
    thread HtVlcReader &vlc,
    thread int &run,
    thread Context &ctx,
    device uint *quad_words,
    uint width,
    constant ushort *vlc_table1,
    constant ushort *uvlc_table1
) {
    uint local_x = 0u;
    uint local_c_q = 0u;
    uint quad = 0u;

    while (local_x < width) {
        const uint above0 = ctx.above(0u);
        const uint above1 = ctx.above(1u);
        const uint above2 = ctx.above(2u);
        local_c_q |= (above0 & 0xA0u) << 2u;
        local_c_q |= (above1 & 0x20u) << 4u;

        const uint window = ht_vlc_fetch(vlc);
        uint t0 = uint(vlc_table1[local_c_q + (window & 0x7Fu)]);
        if (local_c_q == 0u) {
            run -= 2;
            t0 = run == -1 ? t0 : 0u;
            if (run < 0) {
                run = ht_simd_mel_get_run(mel);
            }
        }
        local_x += 2u;

        local_c_q = ((t0 & 0x40u) << 2u) | ((t0 & 0x80u) << 1u);
        local_c_q |= above0 & 0x80u;
        local_c_q |= (above1 & 0xA0u) << 2u;
        local_c_q |= (above2 & 0x20u) << 4u;
        uint used = t0 & 0x7u;

        const bool second = local_x < width;
        uint t1 = uint(vlc_table1[local_c_q + ((window >> used) & 0x7Fu)]);
        if (local_c_q == 0u && second) {
            run -= 2;
            t1 = run == -1 ? t1 : 0u;
            if (run < 0) {
                run = ht_simd_mel_get_run(mel);
            }
        }
        if (!second) {
            t1 = 0u;
        }
        local_x += 2u;

        local_c_q = ((t1 & 0x40u) << 2u) | ((t1 & 0x80u) << 1u);
        local_c_q |= above1 & 0x80u;
        used += t1 & 0x7u;

        const uint uvlc_mode = ((t0 & 0x8u) << 3u) | ((t1 & 0x8u) << 4u);
        uint uvlc_entry = uint(uvlc_table1[uvlc_mode + ((window >> used) & 0x3Fu)]);
        used += uvlc_entry & 0x7u;
        uvlc_entry >>= 3u;
        uint len = uvlc_entry & 0xFu;
        const uint tmp = uint(vlc.tmp >> used) & ((1u << len) - 1u);
        ht_simd_vlc_consume(vlc, used + len);
        uvlc_entry >>= 4u;
        len = uvlc_entry & 0x7u;
        uvlc_entry >>= 3u;

        ctx.record_pair(t0, t1);
        quad_words[quad * 2u] =
            ht_simd_quad_word(t0, (uvlc_entry & 0x7u) + (tmp & ~(0xFFu << len)));
        if (second) {
            quad_words[quad * 2u + 2u] =
                ht_simd_quad_word(t1, (uvlc_entry >> 3u) + (tmp >> len));
        }
        quad += 2u;
    }
}

template<typename Context>
inline void ht_simd_vlc_rows(
    thread HtSimdMel &mel,
    thread HtVlcReader &vlc,
    thread int &run,
    thread Context &ctx,
    device uint *decoded_data,
    J2kHtCleanupParams params,
    uint quad_rows,
    constant ushort *vlc_table0,
    constant ushort *vlc_table1,
    constant ushort *uvlc_table0,
    constant ushort *uvlc_table1
) {
    for (uint row = 0u; row < quad_rows; ++row) {
        device uint *quad_words =
            decoded_data + params.output_offset + row * 2u * params.output_stride;
        if (row == 0u) {
            ht_simd_vlc_initial_row(
                mel, vlc, run, ctx, quad_words, params.width, vlc_table0, uvlc_table0
            );
        } else {
            ht_simd_vlc_non_initial_row(
                mel, vlc, run, ctx, quad_words, params.width, vlc_table1, uvlc_table1
            );
        }
        ctx.finish_row();
    }
}

inline void decode_ht_cleanup_vlc(
    device const uchar *coded_data,
    device uint *decoded_data,
    J2kHtCleanupParams params,
    constant ushort *vlc_table0,
    constant ushort *vlc_table1,
    constant ushort *uvlc_table0,
    constant ushort *uvlc_table1,
    device J2kHtStatus *status,
    threadgroup uint *ctx_slice
) {
    HtCleanupGeometry geometry;
    uint detail = 0u;
    const uint code = ht_simd_validate(coded_data, params, geometry, detail);
    set_ht_status(status, code, detail);
    if (code != J2K_HT_STATUS_OK || geometry.quad_rows == 0u) {
        return;
    }

    const uint lcup = params.cleanup_length;
    HtSimdMel mel = ht_simd_mel_new(coded_data, lcup, geometry.scup);
    HtVlcReader vlc = ht_vlc_reader_new(coded_data, lcup, geometry.scup);
    int run = ht_simd_mel_get_run(mel);

    if (geometry.quads <= 32u) {
        HtNarrowVlcContext ctx = ht_narrow_vlc_context();
        ht_simd_vlc_rows(
            mel, vlc, run, ctx, decoded_data, params, geometry.quad_rows,
            vlc_table0, vlc_table1, uvlc_table0, uvlc_table1
        );
    } else {
        HtWideVlcContext ctx = ht_wide_vlc_context(ctx_slice);
        ht_simd_vlc_rows(
            mel, vlc, run, ctx, decoded_data, params, geometry.quad_rows,
            vlc_table0, vlc_table1, uvlc_table0, uvlc_table1
        );
    }
}

// Appends the next 128 raw MagSgn bytes, destuffed, to the window. Bytes past
// `len` read as 0xFF, and a byte following 0xFF contributes only its low seven
// bits, exactly as `forward_reader_fill` consumes them.
inline void ht_simd_window_refill(
    device const uchar *data,
    uint len,
    threadgroup uint *window,
    thread uint &raw_pos,
    thread uint &window_end,
    thread uint &prev_ff,
    uint lane
) {
    const uint first_empty = (window_end + 31u) >> 5;
    window[(first_empty + lane) & J2K_HT_SIMD_WINDOW_MASK] = 0u;

    const uint base = raw_pos + lane * 4u;
    uint raw[4];
    for (uint j = 0u; j < 4u; ++j) {
        const uint k = base + j;
        raw[j] = k < len ? uint(data[k]) : 0xFFu;
    }
    const uint last_ff = raw[3] == 0xFFu ? 1u : 0u;
    const uint left_ff = simd_shuffle_up(last_ff, 1u);
    uint stuffed = lane == 0u ? prev_ff : left_ff;

    uint bits = 0u;
    uint count = 0u;
    for (uint j = 0u; j < 4u; ++j) {
        const uint value = stuffed != 0u ? (raw[j] & 0x7Fu) : raw[j];
        bits |= value << count;
        count += 8u - stuffed;
        stuffed = raw[j] == 0xFFu ? 1u : 0u;
    }

    const uint offset = window_end + simd_prefix_exclusive_sum(count);
    const uint total = simd_sum(count);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const uint word = offset >> 5;
    const uint shift = offset & 31u;
    threadgroup atomic_uint *atomic_window = (threadgroup atomic_uint *)window;
    atomic_fetch_or_explicit(
        &atomic_window[word & J2K_HT_SIMD_WINDOW_MASK], bits << shift, memory_order_relaxed
    );
    if (shift != 0u && shift + count > 32u) {
        atomic_fetch_or_explicit(
            &atomic_window[(word + 1u) & J2K_HT_SIMD_WINDOW_MASK],
            bits >> (32u - shift),
            memory_order_relaxed
        );
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    raw_pos += J2K_HT_SIMD_REFILL_BYTES;
    window_end += total;
    prev_ff = simd_broadcast(last_ff, J2K_HT_SIMD_LANES - 1u);
}

inline uint ht_simd_window_read32(threadgroup const uint *window, uint bit) {
    const uint word = bit >> 5;
    const uint shift = bit & 31u;
    const uint lo = window[word & J2K_HT_SIMD_WINDOW_MASK];
    if (shift == 0u) {
        return lo;
    }
    const uint hi = window[(word + 1u) & J2K_HT_SIMD_WINDOW_MASK];
    return (lo >> shift) | (hi << (32u - shift));
}

// Same result as `decode_mag_sgn_sample_with_vn` for a significant sample
// whose MagSgn bits start at `ms_val`.
inline uint ht_simd_mag_sgn_value(
    uint ms_val,
    uint m_n,
    uint inf,
    uint bit,
    uint p,
    thread uint &v_n
) {
    uint value = ms_val << 31u;
    const uint mask = m_n == 0u ? 0u : (1u << m_n) - 1u;
    v_n = ms_val & mask;
    v_n |= ((inf >> (8u + bit)) & 1u) << m_n;
    v_n |= 1u;
    value |= (v_n + 2u) << (p - 1u);
    return value;
}

inline uint ht_simd_output_bits(uint value, J2kHtCleanupParams params, uint k_max) {
    return coefficient_to_float_bits(
        value, k_max, params.dequantization_step, params.irreversible_midpoint, params.roi_shift
    );
}

inline void decode_ht_cleanup_magsgn(
    device const uchar *coded_data,
    device uint *decoded_data,
    J2kHtCleanupParams params,
    device J2kHtStatus *status,
    threadgroup uint *v_n,
    threadgroup uint *window,
    uint lane
) {
    // The VLC kernel already reported validation failures; skip those blocks.
    HtCleanupGeometry geometry;
    uint detail = 0u;
    if (ht_simd_validate(coded_data, params, geometry, detail) != J2K_HT_STATUS_OK) {
        return;
    }

    const uint width = params.width;
    const uint height = params.height;
    const uint stride = params.output_stride;
    const uint quads = geometry.quads;
    const uint p = 30u - params.missing_msbs;
    const uint uq_limit = params.missing_msbs + 2u;
    const uint k_max = params.num_bitplanes + params.roi_shift;
    const uint magsgn_len = params.cleanup_length - geometry.scup;

    uint raw_pos = 0u;
    uint window_end = 0u;
    uint prev_ff = 0u;
    uint bit_pos = 0u;

    for (uint row = 0u; row < geometry.quad_rows; ++row) {
        const uint y = row * 2u;
        const bool second_row_present = y + 1u < height;
        device uint *row_out = decoded_data + params.output_offset + y * stride;
        uint carry_v_n3 = 0u;

        for (uint chunk = 0u; chunk < quads; chunk += J2K_HT_SIMD_LANES) {
            const uint q = chunk + lane;
            const bool active = q < quads;
            const bool has_right = active && q * 2u + 1u < width;

            uint inf = 0u;
            uint uq = 0u;
            if (active) {
                const uint quad_word = row_out[q * 2u];
                inf = quad_word & 0xFFFFu;
                const uint u_q = quad_word >> 16u;
                if (row == 0u) {
                    uq = u_q;
                } else {
                    uint gamma = inf & 0xF0u;
                    gamma &= gamma - 0x10u;
                    uint emax = v_n[q] | v_n[q + 1u];
                    emax = 31u - clz(emax | 2u);
                    const uint kappa = gamma != 0u ? emax : 1u;
                    uq = u_q + kappa;
                }
            }
            if (simd_any(active && uq > uq_limit)) {
                if (lane == 0u) {
                    set_ht_status(status, J2K_HT_STATUS_FAIL, row == 0u ? 13u : 14u);
                }
                return;
            }

            // Samples are consumed in column-major quad order: 0 and 1 always,
            // 2 and 3 only when the quad's right column exists.
            const uint m0 = (inf & sample_mask(0u)) != 0u ? uq - ((inf >> 12u) & 1u) : 0u;
            const uint m1 = (inf & sample_mask(1u)) != 0u ? uq - ((inf >> 13u) & 1u) : 0u;
            const uint m2 = has_right && (inf & sample_mask(2u)) != 0u
                ? uq - ((inf >> 14u) & 1u) : 0u;
            const uint m3 = has_right && (inf & sample_mask(3u)) != 0u
                ? uq - ((inf >> 15u) & 1u) : 0u;
            const uint quad_bits = m0 + m1 + m2 + m3;
            const uint lane_bit = simd_prefix_exclusive_sum(quad_bits);
            const uint chunk_bits = simd_sum(quad_bits);

            while (window_end < bit_pos + chunk_bits + 32u) {
                ht_simd_window_refill(
                    coded_data, magsgn_len, window, raw_pos, window_end, prev_ff, lane
                );
            }

            uint v_n1 = 0u;
            uint v_n3 = 0u;
            if (active) {
                uint cursor = bit_pos + lane_bit;
                device uint *out = row_out + q * 2u;
                uint ignored_vn = 0u;

                uint value0 = 0u;
                if ((inf & sample_mask(0u)) != 0u) {
                    value0 = ht_simd_mag_sgn_value(
                        ht_simd_window_read32(window, cursor), m0, inf, 0u, p, ignored_vn
                    );
                    cursor += m0;
                }
                out[0] = ht_simd_output_bits(value0, params, k_max);

                uint value1 = 0u;
                if ((inf & sample_mask(1u)) != 0u) {
                    value1 = ht_simd_mag_sgn_value(
                        ht_simd_window_read32(window, cursor), m1, inf, 1u, p, v_n1
                    );
                    cursor += m1;
                }
                if (second_row_present) {
                    out[stride] = ht_simd_output_bits(value1, params, k_max);
                }

                if (has_right) {
                    uint value2 = 0u;
                    if ((inf & sample_mask(2u)) != 0u) {
                        value2 = ht_simd_mag_sgn_value(
                            ht_simd_window_read32(window, cursor), m2, inf, 2u, p, ignored_vn
                        );
                        cursor += m2;
                    }
                    out[1] = ht_simd_output_bits(value2, params, k_max);

                    uint value3 = 0u;
                    if ((inf & sample_mask(3u)) != 0u) {
                        value3 = ht_simd_mag_sgn_value(
                            ht_simd_window_read32(window, cursor), m3, inf, 3u, p, v_n3
                        );
                    }
                    if (second_row_present) {
                        out[stride + 1u] = ht_simd_output_bits(value3, params, k_max);
                    }
                }
            }
            bit_pos += chunk_bits;

            // Every lane has read the previous row's exponents; publish this row's.
            simdgroup_barrier(mem_flags::mem_threadgroup);
            const uint left_v_n3 = simd_shuffle_up(v_n3, 1u);
            if (active) {
                v_n[q] = (lane == 0u ? carry_v_n3 : left_v_n3) | v_n1;
            }
            const uint last_lane = min(J2K_HT_SIMD_LANES - 1u, quads - 1u - chunk);
            carry_v_n3 = simd_shuffle(v_n3, ushort(last_lane));
        }
        if (lane == 0u) {
            v_n[quads] = carry_v_n3;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void j2k_decode_ht_cleanup_vlc_batched(
    device const uchar *coded_data [[buffer(0)]],
    device uint *decoded_data [[buffer(1)]],
    constant J2kHtCleanupBatchJob *jobs [[buffer(2)]],
    constant ushort *vlc_table0 [[buffer(3)]],
    constant ushort *vlc_table1 [[buffer(4)]],
    constant ushort *uvlc_table0 [[buffer(5)]],
    constant ushort *uvlc_table1 [[buffer(6)]],
    device J2kHtStatus *status [[buffer(7)]],
    constant J2kHtVlcDispatchParams &dispatch [[buffer(9)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup uint ctx[J2K_HT_VLC_THREADS_PER_GROUP * J2K_HT_VLC_CTX_STRIDE];
    uint job_index = 0u;
    if (!ht_vlc_job_index(dispatch, gid, job_index)) {
        return;
    }
    const constant J2kHtCleanupBatchJob &job = jobs[job_index];
    const J2kHtCleanupParams params = ht_cleanup_params_from_job(job, 1u, job.output_offset);
    decode_ht_cleanup_vlc(
        coded_data + job.coded_offset, decoded_data, params,
        vlc_table0, vlc_table1, uvlc_table0, uvlc_table1, status + job_index,
        ctx + tid * J2K_HT_VLC_CTX_STRIDE
    );
}

kernel void j2k_decode_ht_cleanup_vlc_repeated_batched(
    device const uchar *coded_data [[buffer(0)]],
    device uint *decoded_data [[buffer(1)]],
    constant J2kHtCleanupBatchJob *jobs [[buffer(2)]],
    constant J2kHtRepeatedBatchParams &repeated [[buffer(3)]],
    constant ushort *vlc_table0 [[buffer(4)]],
    constant ushort *vlc_table1 [[buffer(5)]],
    constant ushort *uvlc_table0 [[buffer(6)]],
    constant ushort *uvlc_table1 [[buffer(7)]],
    device J2kHtStatus *status [[buffer(8)]],
    constant J2kHtVlcDispatchParams &dispatch [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup uint ctx[J2K_HT_VLC_THREADS_PER_GROUP * J2K_HT_VLC_CTX_STRIDE];
    uint job_index = 0u;
    if (!ht_vlc_job_index(dispatch, gid.x, job_index) || gid.y >= repeated.batch_count) {
        return;
    }
    const constant J2kHtCleanupBatchJob &job = jobs[job_index];
    const J2kHtCleanupParams params = ht_cleanup_params_from_job(
        job, 1u, job.output_offset + gid.y * repeated.output_plane_len
    );
    decode_ht_cleanup_vlc(
        coded_data + job.coded_offset, decoded_data, params,
        vlc_table0, vlc_table1, uvlc_table0, uvlc_table1,
        status + gid.y * repeated.job_count + job_index,
        ctx + tid * J2K_HT_VLC_CTX_STRIDE
    );
}

kernel void j2k_decode_ht_cleanup_magsgn_batched(
    device const uchar *coded_data [[buffer(0)]],
    device uint *decoded_data [[buffer(1)]],
    constant J2kHtCleanupBatchJob *jobs [[buffer(2)]],
    device J2kHtStatus *status [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup uint v_n[J2K_HT_SIMD_BLOCKS_PER_GROUP][J2K_HT_MAX_VN];
    threadgroup uint window[J2K_HT_SIMD_BLOCKS_PER_GROUP][J2K_HT_SIMD_WINDOW_WORDS];

    const uint job_index = gid / J2K_HT_SIMD_LANES;
    const constant J2kHtCleanupBatchJob &job = jobs[job_index];
    const J2kHtCleanupParams params = ht_cleanup_params_from_job(job, 1u, job.output_offset);
    decode_ht_cleanup_magsgn(
        coded_data + job.coded_offset, decoded_data, params, status + job_index,
        v_n[simd_index], window[simd_index], lane
    );
}

kernel void j2k_decode_ht_cleanup_magsgn_repeated_batched(
    device const uchar *coded_data [[buffer(0)]],
    device uint *decoded_data [[buffer(1)]],
    constant J2kHtCleanupBatchJob *jobs [[buffer(2)]],
    constant J2kHtRepeatedBatchParams &repeated [[buffer(3)]],
    device J2kHtStatus *status [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup uint v_n[J2K_HT_SIMD_BLOCKS_PER_GROUP][J2K_HT_MAX_VN];
    threadgroup uint window[J2K_HT_SIMD_BLOCKS_PER_GROUP][J2K_HT_SIMD_WINDOW_WORDS];

    const uint job_index = gid.x / J2K_HT_SIMD_LANES;
    if (job_index >= repeated.job_count || gid.y >= repeated.batch_count) {
        return;
    }
    const constant J2kHtCleanupBatchJob &job = jobs[job_index];
    const J2kHtCleanupParams params = ht_cleanup_params_from_job(
        job, 1u, job.output_offset + gid.y * repeated.output_plane_len
    );
    decode_ht_cleanup_magsgn(
        coded_data + job.coded_offset, decoded_data, params,
        status + gid.y * repeated.job_count + job_index,
        v_n[simd_index], window[simd_index], lane
    );
}
