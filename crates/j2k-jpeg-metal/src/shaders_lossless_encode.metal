// ---------------------------------------------------------------------------
// Lossless (SOF3) JPEG encode
//
// The stages, buffers, and parameters follow
// j2k_jpeg::adapter::JpegLosslessGpuEncodeParams (lossless_encode/gpu.rs),
// whose host emulation is tested against CPU encode. Prediction and
// difference categories mirror j2k_codec_math::jpeg::lossless.
// ---------------------------------------------------------------------------

struct JpegLosslessEncodeParams {
    uint width;
    uint height;
    uint components;
    uint sample_bytes;
    uint pitch_bytes;
    uint input_offset_bytes;
    uint predictor;
    uint point_transform;
    uint initial_predictor;
    uint max_sample;
    uint interval_rows;
    uint chunk_pixels;
    uint chunks_per_row;
    uint intervals;
    uint sample_count;
    uint packed_words;
};

// `code << 8 | code length` per difference category 0..=16.
struct JpegLosslessEncodeTable {
    uint entries[17];
};

constant uint JPEG_LOSSLESS_CATEGORIES = 17u;
// Status words: first out-of-range sample index, its value, packed bytes.
constant uint JPEG_LOSSLESS_STATUS_BAD_INDEX = 0u;
constant uint JPEG_LOSSLESS_STATUS_BAD_VALUE = 1u;
constant uint JPEG_LOSSLESS_STATUS_PACKED_BYTES = 2u;

static inline uint jpeg_lossless_sample(
    device const uchar* input,
    constant JpegLosslessEncodeParams& p,
    uint row,
    uint index
) {
    const uint offset = p.input_offset_bytes + row * p.pitch_bytes + index * p.sample_bytes;
    // The host checked that 16-bit samples are 2-byte aligned.
    return p.sample_bytes == 2u ? uint(*((device const ushort*)(input + offset)))
                                : uint(input[offset]);
}

static inline uint jpeg_lossless_predict(uint predictor, int ra, int rb, int rc) {
    int prediction;
    switch (predictor) {
        case 1u: prediction = ra; break;
        case 2u: prediction = rb; break;
        case 3u: prediction = rc; break;
        case 4u: prediction = ra + rb - rc; break;
        case 5u: prediction = ra + ((rb - rc) >> 1); break;
        case 6u: prediction = rb + ((ra - rc) >> 1); break;
        default: prediction = (ra + rb) >> 1; break;
    }
    return uint(prediction) & 0xFFFFu;
}

static inline uint jpeg_lossless_additional_bits(uint category) {
    return category >= 16u ? 0u : category;
}

// The `[start, end)` token range of one chunk.
static inline uint2 jpeg_lossless_chunk_tokens(constant JpegLosslessEncodeParams& p, uint chunk) {
    const uint row = chunk / p.chunks_per_row;
    const uint x0 = (chunk - row * p.chunks_per_row) * p.chunk_pixels;
    const uint x1 = min(x0 + p.chunk_pixels, p.width);
    const uint base = row * p.width * p.components;
    return uint2(base + x0 * p.components, base + x1 * p.components);
}

// Memory word whose bytes are the stream word `w` in stream (big-endian)
// order.
static inline uint jpeg_lossless_stream_word(uint w) {
    return (w >> 24) | ((w >> 8) & 0xFF00u) | ((w << 8) & 0xFF0000u) | (w << 24);
}

// Stage 1, one thread per sample: token `category | bits << 16` and the
// category histogram; out-of-range samples record their least index.
kernel void jpeg_lossless_encode_classify(
    device const uchar* input [[buffer(0)]],
    device uint* tokens [[buffer(1)]],
    device atomic_uint* histogram [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    constant JpegLosslessEncodeParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    threadgroup atomic_uint local_histogram[JPEG_LOSSLESS_CATEGORIES];
    for (uint bin = lid; bin < JPEG_LOSSLESS_CATEGORIES; bin += group_size) {
        atomic_store_explicit(&local_histogram[bin], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gid < p.sample_count) {
        const uint row_samples = p.width * p.components;
        const uint row = gid / row_samples;
        const uint index = gid - row * row_samples;
        const uint x = index / p.components;
        const uint value = jpeg_lossless_sample(input, p, row, index);
        if (value > p.max_sample) {
            atomic_fetch_min_explicit(
                &status[JPEG_LOSSLESS_STATUS_BAD_INDEX], gid, memory_order_relaxed);
        }
        const uint shift = p.point_transform;
        uint prediction;
        if (row % p.interval_rows == 0u) {
            prediction = x == 0u
                ? p.initial_predictor
                : jpeg_lossless_sample(input, p, row, index - p.components) >> shift;
        } else if (x == 0u) {
            prediction = jpeg_lossless_sample(input, p, row - 1u, index) >> shift;
        } else {
            const int ra = int(jpeg_lossless_sample(input, p, row, index - p.components) >> shift);
            const int rb = int(jpeg_lossless_sample(input, p, row - 1u, index) >> shift);
            const int rc =
                int(jpeg_lossless_sample(input, p, row - 1u, index - p.components) >> shift);
            prediction = jpeg_lossless_predict(p.predictor, ra, rb, rc);
        }
        const uint difference = ((value >> shift) - prediction) & 0xFFFFu;
        uint category;
        uint bits;
        if (difference == 0x8000u) {
            category = 16u;
            bits = 0u;
        } else {
            const int signed_difference = int(difference << 16) >> 16;
            const uint magnitude = uint(abs(signed_difference));
            category = 32u - clz(magnitude);
            bits = signed_difference >= 0
                ? magnitude
                : (difference - 1u) & ((1u << category) - 1u);
        }
        tokens[gid] = category | (bits << 16);
        atomic_fetch_add_explicit(&local_histogram[category], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint bin = lid; bin < JPEG_LOSSLESS_CATEGORIES; bin += group_size) {
        const uint count = atomic_load_explicit(&local_histogram[bin], memory_order_relaxed);
        if (count != 0u) {
            atomic_fetch_add_explicit(&histogram[bin], count, memory_order_relaxed);
        }
    }
}

// One thread: the value of the first out-of-range sample, for the error.
kernel void jpeg_lossless_encode_read_sample(
    device const uchar* input [[buffer(0)]],
    device uint* status [[buffer(1)]],
    constant JpegLosslessEncodeParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint index = status[JPEG_LOSSLESS_STATUS_BAD_INDEX];
    if (gid != 0u || index >= p.sample_count) {
        return;
    }
    const uint row_samples = p.width * p.components;
    const uint row = index / row_samples;
    status[JPEG_LOSSLESS_STATUS_BAD_VALUE] =
        jpeg_lossless_sample(input, p, row, index - row * row_samples);
}

// Stage 3, one thread per chunk: bits its tokens take.
kernel void jpeg_lossless_encode_chunk_bits(
    device const uint* tokens [[buffer(0)]],
    device uint* chunk_bits [[buffer(1)]],
    constant JpegLosslessEncodeParams& p [[buffer(2)]],
    constant JpegLosslessEncodeTable& table [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.chunks_per_row * p.height) {
        return;
    }
    const uint2 range = jpeg_lossless_chunk_tokens(p, gid);
    uint bits = 0u;
    for (uint i = range.x; i < range.y; ++i) {
        const uint category = tokens[i] & 0xFFu;
        bits += (table.entries[category] & 0xFFu) + jpeg_lossless_additional_bits(category);
    }
    chunk_bits[gid] = bits;
}

// Stage 4a, one thread per row: each chunk's bit offset within its row.
kernel void jpeg_lossless_encode_row_offsets(
    device const uint* chunk_bits [[buffer(0)]],
    device uint* chunk_offsets [[buffer(1)]],
    device uint* row_bits [[buffer(2)]],
    constant JpegLosslessEncodeParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.height) {
        return;
    }
    uint running = 0u;
    for (uint k = 0u; k < p.chunks_per_row; ++k) {
        const uint chunk = gid * p.chunks_per_row + k;
        chunk_offsets[chunk] = running;
        running += chunk_bits[chunk];
    }
    row_bits[gid] = running;
}

// Stage 4b, one thread per restart interval: each row's bit offset within
// its interval.
kernel void jpeg_lossless_encode_interval_offsets(
    device const uint* row_bits [[buffer(0)]],
    device uint* row_offsets [[buffer(1)]],
    device uint* interval_bits [[buffer(2)]],
    constant JpegLosslessEncodeParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.intervals) {
        return;
    }
    const uint first = gid * p.interval_rows;
    const uint last = min(first + p.interval_rows, p.height);
    uint running = 0u;
    for (uint row = first; row < last; ++row) {
        row_offsets[row] = running;
        running += row_bits[row];
    }
    interval_bits[gid] = running;
}

// Stage 4c, one thread: byte lengths and byte bases of the intervals.
kernel void jpeg_lossless_encode_interval_bases(
    device const uint* interval_bits [[buffer(0)]],
    device uint* interval_bases [[buffer(1)]],
    device uint* interval_bytes [[buffer(2)]],
    device uint* status [[buffer(3)]],
    constant JpegLosslessEncodeParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) {
        return;
    }
    uint running = 0u;
    for (uint interval = 0u; interval < p.intervals; ++interval) {
        const uint bytes = (interval_bits[interval] + 7u) / 8u;
        interval_bytes[interval] = bytes;
        interval_bases[interval] = running;
        running += bytes;
    }
    status[JPEG_LOSSLESS_STATUS_PACKED_BYTES] = running;
}

// Zero the words the pack stage ORs into.
kernel void jpeg_lossless_encode_clear(
    device uint* packed [[buffer(0)]],
    device const uint* status [[buffer(1)]],
    constant JpegLosslessEncodeParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint used_words = status[JPEG_LOSSLESS_STATUS_PACKED_BYTES] / 4u + 1u;
    if (gid < min(used_words, p.packed_words)) {
        packed[gid] = 0u;
    }
}

// Stage 5, one thread per chunk: OR its codes into the packed stream, and
// pad the last byte of an interval with 1-bits.
kernel void jpeg_lossless_encode_pack(
    device const uint* tokens [[buffer(0)]],
    device atomic_uint* packed [[buffer(1)]],
    device const uint* chunk_offsets [[buffer(2)]],
    device const uint* row_offsets [[buffer(3)]],
    device const uint* interval_bases [[buffer(4)]],
    device const uint* interval_bits [[buffer(5)]],
    constant JpegLosslessEncodeParams& p [[buffer(6)]],
    constant JpegLosslessEncodeTable& table [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.chunks_per_row * p.height) {
        return;
    }
    const uint row = gid / p.chunks_per_row;
    const uint interval = row / p.interval_rows;
    const uint start = interval_bases[interval] * 8u + row_offsets[row] + chunk_offsets[gid];
    uint word = start / 32u;
    const uint lead = start % 32u;
    ulong acc = 0ul;
    uint acc_bits = lead;
    // A word is shared with the previous chunk only when it starts mid-word.
    bool shared_first = lead != 0u;
    const uint2 range = jpeg_lossless_chunk_tokens(p, gid);
    for (uint i = range.x; i < range.y; ++i) {
        const uint token = tokens[i];
        const uint category = token & 0xFFu;
        const uint entry = table.entries[category];
        const uint extra = jpeg_lossless_additional_bits(category);
        const uint len = (entry & 0xFFu) + extra;
        const ulong bits = ulong(token >> 16) & ((1ul << extra) - 1ul);
        acc = (acc << len) | (ulong(entry >> 8) << extra) | bits;
        acc_bits += len;
        while (acc_bits >= 32u) {
            acc_bits -= 32u;
            const uint w = jpeg_lossless_stream_word(uint(acc >> acc_bits));
            if (shared_first) {
                atomic_fetch_or_explicit(&packed[word], w, memory_order_relaxed);
                shared_first = false;
            } else {
                atomic_store_explicit(&packed[word], w, memory_order_relaxed);
            }
            acc &= (1ul << acc_bits) - 1ul;
            ++word;
        }
    }
    if (acc_bits > 0u) {
        const uint w = jpeg_lossless_stream_word(uint(acc << (32u - acc_bits)));
        atomic_fetch_or_explicit(&packed[word], w, memory_order_relaxed);
    }
    const uint last_row = min((interval + 1u) * p.interval_rows, p.height) - 1u;
    if (row == last_row && gid - row * p.chunks_per_row == p.chunks_per_row - 1u) {
        const uint end = interval_bases[interval] * 8u + interval_bits[interval];
        const uint pad = (8u - end % 8u) % 8u;
        if (pad > 0u) {
            const uint ones = ((1u << pad) - 1u) << (32u - end % 32u - pad);
            atomic_fetch_or_explicit(
                &packed[end / 32u], jpeg_lossless_stream_word(ones), memory_order_relaxed);
        }
    }
}
