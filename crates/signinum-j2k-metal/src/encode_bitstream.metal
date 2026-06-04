#include <metal_stdlib>
using namespace metal;

constant uint J2K_ENCODE_STATUS_OK = 0u;
constant uint J2K_ENCODE_STATUS_FAIL = 1u;
constant uint J2K_ENCODE_STATUS_UNSUPPORTED = 2u;
constant uint J2K_PACKET_PAYLOAD_COPY_SMALL_JOB_BYTES = 64u;
constant uint J2K_PACKET_PAYLOAD_COPY_MEDIUM_JOB_BYTES = 512u;

constant uchar J2K_ENCODE_SIGNIFICANT = uchar(1u << 7u);
constant uchar J2K_ENCODE_MAGNITUDE_REFINED = uchar(1u << 6u);
constant uchar J2K_ENCODE_SIGN = uchar(1u << 5u);
constant uchar J2K_ENCODE_CODED_MARKER_MASK = uchar(0x1Fu);
constant uint J2K_CLASSIC_ENCODE_32_MAX_WIDTH = 32u;
constant uint J2K_CLASSIC_ENCODE_32_MAX_HEIGHT = 32u;
constant uint J2K_CLASSIC_ENCODE_32_PADDED_WIDTH = J2K_CLASSIC_ENCODE_32_MAX_WIDTH + 2u;
constant uint J2K_CLASSIC_ENCODE_32_PADDED_HEIGHT = J2K_CLASSIC_ENCODE_32_MAX_HEIGHT + 2u;
constant uint J2K_CLASSIC_ENCODE_32_MAX_COEFF_COUNT =
    J2K_CLASSIC_ENCODE_32_PADDED_WIDTH * J2K_CLASSIC_ENCODE_32_PADDED_HEIGHT;
constant uint J2K_CLASSIC_TIER1_PASS_PLAN_CAPACITY = 48u;

struct J2kClassicEncodeParams {
    uint width;
    uint height;
    uint sub_band_type;
    uint total_bitplanes;
    uint style_flags;
    uint output_capacity;
    uint segment_capacity;
};

struct J2kClassicEncodeStatus {
    uint code;
    uint detail;
    uint data_len;
    uint number_of_coding_passes;
    uint missing_bit_planes;
    uint segment_count;
    uint reserved0;
    uint reserved1;
};

struct J2kMqEncoder {
    device uchar *data;
    uint max_len;
    uint len;
    uint a;
    uint c;
    uint ct;
    uint failed;
};

struct J2kRawBitWriter {
    device uchar *data;
    uint max_len;
    uint len;
    uint buffer;
    uint bits_in_buffer;
    uint last_byte_was_ff;
    uint failed;
};

inline void j2k_set_encode_status(
    device J2kClassicEncodeStatus *status,
    uint code,
    uint detail,
    uint data_len,
    uint passes,
    uint missing,
    uint segments
) {
    status->code = code;
    status->detail = detail;
    status->data_len = data_len;
    status->number_of_coding_passes = passes;
    status->missing_bit_planes = missing;
    status->segment_count = segments;
    status->reserved0 = 0u;
    status->reserved1 = 0u;
}

inline void j2k_set_encode_status_with_payload_skip(
    device J2kClassicEncodeStatus *status,
    uint code,
    uint detail,
    uint data_len,
    uint passes,
    uint missing,
    uint segments,
    uint payload_skip
) {
    j2k_set_encode_status(status, code, detail, data_len, passes, missing, segments);
    status->reserved0 = payload_skip;
}

inline void j2k_mq_init(thread J2kMqEncoder &encoder, device uchar *out, uint capacity) {
    encoder.data = out;
    encoder.max_len = capacity;
    encoder.len = 0u;
    encoder.a = 0x8000u;
    encoder.c = 0u;
    encoder.ct = 12u;
    encoder.failed = 0u;
    if (capacity == 0u) {
        encoder.failed = 1u;
        return;
    }
    encoder.data[0] = uchar(0);
    encoder.len = 1u;
}

inline void j2k_mq_push(thread J2kMqEncoder &encoder, uchar value) {
    if (encoder.len >= encoder.max_len) {
        encoder.failed = 1u;
        return;
    }
    encoder.data[encoder.len] = value;
    encoder.len += 1u;
}

inline void j2k_mq_byte_out(thread J2kMqEncoder &encoder) {
    if (encoder.failed != 0u || encoder.len == 0u) {
        encoder.failed = 1u;
        return;
    }

    uchar last_byte = encoder.data[encoder.len - 1u];
    if (last_byte == uchar(0xFFu)) {
        const uchar b = uchar(encoder.c >> 20u);
        j2k_mq_push(encoder, b);
        encoder.c &= 0xFFFFFu;
        encoder.ct = 7u;
    } else if ((encoder.c & 0x8000000u) == 0u) {
        const uchar b = uchar(encoder.c >> 19u);
        j2k_mq_push(encoder, b);
        encoder.c &= 0x7FFFFu;
        encoder.ct = 8u;
    } else {
        encoder.data[encoder.len - 1u] = uchar(encoder.data[encoder.len - 1u] + uchar(1u));
        encoder.c &= 0x7FFFFFFu;
        if (encoder.data[encoder.len - 1u] == uchar(0xFFu)) {
            const uchar b = uchar(encoder.c >> 20u);
            j2k_mq_push(encoder, b);
            encoder.c &= 0xFFFFFu;
            encoder.ct = 7u;
        } else {
            const uchar b = uchar(encoder.c >> 19u);
            j2k_mq_push(encoder, b);
            encoder.c &= 0x7FFFFu;
            encoder.ct = 8u;
        }
    }
}

inline void j2k_mq_renormalize(thread J2kMqEncoder &encoder) {
    do {
        encoder.a <<= 1u;
        encoder.c <<= 1u;
        encoder.ct -= 1u;
        if (encoder.ct == 0u) {
            j2k_mq_byte_out(encoder);
        }
    } while ((encoder.a & 0x8000u) == 0u && encoder.failed == 0u);
}

inline void j2k_mq_encode(
    thread J2kMqEncoder &encoder,
    thread uchar *contexts,
    uint ctx_label,
    uint bit
) {
    uchar ctx = contexts[ctx_label];
    const J2kQeData qe = J2K_QE_TABLE[ctx & uchar(0x7Fu)];
    const uint mps = uint(ctx >> 7u);
    encoder.a -= qe.qe;

    if (bit == mps) {
        if ((encoder.a & 0x8000u) != 0u) {
            encoder.c += qe.qe;
            return;
        }
        if (encoder.a < qe.qe) {
            encoder.a = qe.qe;
        } else {
            encoder.c += qe.qe;
        }
        ctx = uchar((ctx & uchar(0x80u)) | qe.nmps);
    } else {
        if (encoder.a < qe.qe) {
            encoder.c += qe.qe;
        } else {
            encoder.a = qe.qe;
        }
        if (qe.switch_mps != 0u) {
            ctx ^= uchar(0x80u);
        }
        ctx = uchar((ctx & uchar(0x80u)) | qe.nlps);
    }

    contexts[ctx_label] = ctx;
    j2k_mq_renormalize(encoder);
}

inline void j2k_mq_set_bits(thread J2kMqEncoder &encoder) {
    const uint temp = encoder.c + encoder.a;
    encoder.c |= 0xFFFFu;
    if (encoder.c >= temp) {
        encoder.c -= 0x8000u;
    }
}

inline void j2k_mq_finish(thread J2kMqEncoder &encoder) {
    j2k_mq_set_bits(encoder);
    encoder.c <<= encoder.ct;
    j2k_mq_byte_out(encoder);
    encoder.c <<= encoder.ct;
    j2k_mq_byte_out(encoder);
}

inline void j2k_raw_writer_init(thread J2kRawBitWriter &writer, device uchar *out, uint capacity) {
    writer.data = out;
    writer.max_len = capacity;
    writer.len = 0u;
    writer.buffer = 0u;
    writer.bits_in_buffer = 0u;
    writer.last_byte_was_ff = 0u;
    writer.failed = 0u;
}

inline void j2k_raw_writer_push(thread J2kRawBitWriter &writer, uchar value) {
    if (writer.len >= writer.max_len) {
        writer.failed = 1u;
        return;
    }
    writer.data[writer.len] = value;
    writer.len += 1u;
    writer.last_byte_was_ff = value == uchar(0xFFu) ? 1u : 0u;
}

inline void j2k_raw_writer_flush_byte(thread J2kRawBitWriter &writer) {
    const uint limit = writer.last_byte_was_ff != 0u ? 7u : 8u;
    const uchar byte = uchar(writer.buffer >> (writer.bits_in_buffer - limit));
    j2k_raw_writer_push(writer, byte);
    writer.bits_in_buffer -= limit;
    writer.buffer &= writer.bits_in_buffer == 0u ? 0u : ((1u << writer.bits_in_buffer) - 1u);
}

inline void j2k_raw_writer_write_bit(thread J2kRawBitWriter &writer, uint bit) {
    writer.buffer = (writer.buffer << 1u) | (bit & 1u);
    writer.bits_in_buffer += 1u;
    const uint limit = writer.last_byte_was_ff != 0u ? 7u : 8u;
    if (writer.bits_in_buffer >= limit) {
        j2k_raw_writer_flush_byte(writer);
    }
}

inline void j2k_raw_writer_finish(thread J2kRawBitWriter &writer) {
    if (writer.bits_in_buffer == 0u) {
        return;
    }
    const uint limit = writer.last_byte_was_ff != 0u ? 7u : 8u;
    const uint shift = limit - writer.bits_in_buffer;
    j2k_raw_writer_push(writer, uchar(writer.buffer << shift));
    writer.buffer = 0u;
    writer.bits_in_buffer = 0u;
}

inline uint j2k_classic_magnitude(int value) {
    return value < 0 ? uint(-value) : uint(value);
}

inline void j2k_classic_clear_state_border(
    thread uchar *states,
    uint padded_width,
    uint width,
    uint height
) {
    const uint bottom_row = height + 1u;
    for (uint x = 0u; x < padded_width; ++x) {
        states[coeff_index(padded_width, x, 0u)] = uchar(0u);
        states[coeff_index(padded_width, x, bottom_row)] = uchar(0u);
    }
    for (uint y = 1u; y <= height; ++y) {
        states[coeff_index(padded_width, 0u, y)] = uchar(0u);
        states[coeff_index(padded_width, width + 1u, y)] = uchar(0u);
    }
}

inline uchar j2k_classic_effective_neighbors(
    thread const uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y,
    uint height,
    uint style_flags
) {
    return effective_neighborhood_states(states, padded_width, index_x, index_y, height, style_flags);
}

inline bool j2k_classic_coded_marker_matches(
    thread const uchar *states,
    uint idx,
    uchar coded_marker
) {
    return (states[idx] & J2K_ENCODE_CODED_MARKER_MASK) ==
        (coded_marker & J2K_ENCODE_CODED_MARKER_MASK);
}

inline void j2k_classic_set_coded_marker(
    thread uchar *states,
    uint idx,
    uchar coded_marker
) {
    states[idx] =
        (states[idx] & uchar(~J2K_ENCODE_CODED_MARKER_MASK)) |
        (coded_marker & J2K_ENCODE_CODED_MARKER_MASK);
}

inline void j2k_classic_encode_sign_from_neighbors(
    uint idx,
    thread const uchar *states,
    thread J2kMqEncoder &encoder,
    thread uchar *contexts,
    uint padded_width,
    uint index_x,
    uint index_y,
    uint height,
    uint style_flags,
    uchar neighbor_sig
) {
    const uchar significances = neighbor_sig & uchar(0b01010101);
    const uint left_sign = coeff_sign(states, coeff_index(padded_width, index_x - 1u, index_y));
    const uint right_sign = coeff_sign(states, coeff_index(padded_width, index_x + 1u, index_y));
    const uint top_sign = coeff_sign(states, coeff_index(padded_width, index_x, index_y - 1u));
    const uint bottom_sign =
        ((style_flags & J2K_CLASSIC_STYLE_VERTICALLY_CAUSAL_CONTEXT) != 0u &&
            neighbor_in_next_stripe(index_y, height))
        ? 0u
        : coeff_sign(states, coeff_index(padded_width, index_x, index_y + 1u));
    const uchar signs = uchar((top_sign << 6u) | (left_sign << 4u) | (right_sign << 2u) | bottom_sign);
    const uchar negative = significances & signs;
    const uchar positive = significances & uchar(~signs);
    const uchar2 sign_ctx = SIGN_CONTEXT_LOOKUP[uchar((negative << 1u) | positive)];
    const uint sign_bit = (uint(states[idx]) >> 5u) & 1u;
    j2k_mq_encode(encoder, contexts, uint(sign_ctx.x), sign_bit ^ uint(sign_ctx.y));
}

template<typename Magnitude>
inline void j2k_classic_significance_pass(
    thread const Magnitude *magnitudes,
    thread uchar *states,
    uchar coded_marker,
    thread J2kMqEncoder &encoder,
    thread uchar *contexts,
    uint width,
    uint height,
    uint padded_width,
    uint bit_mask,
    uint sub_band_type,
    uint style_flags
) {
    for (uint y_base = 0u; y_base < height; y_base += 4u) {
        for (uint x = 0u; x < width; ++x) {
            const uint y_end = min(y_base + 4u, height);
            for (uint y = y_base; y < y_end; ++y) {
                const uint ix = x + 1u;
                const uint iy = y + 1u;
                const uint idx = coeff_index(padded_width, ix, iy);
                const uchar neighbor_sig =
                    j2k_classic_effective_neighbors(states, padded_width, ix, iy, height, style_flags);
                if ((states[idx] & J2K_ENCODE_SIGNIFICANT) == 0u && neighbor_sig != 0u) {
                    const uint ctx_label = uint(zero_context_label(neighbor_sig, sub_band_type));
                    const uint bit = (magnitudes[idx] & bit_mask) != 0u ? 1u : 0u;
                    j2k_mq_encode(encoder, contexts, ctx_label, bit);
                    j2k_classic_set_coded_marker(states, idx, coded_marker);
                    if (bit != 0u) {
                        j2k_classic_encode_sign_from_neighbors(
                            idx,
                            states,
                            encoder,
                            contexts,
                            padded_width,
                            ix,
                            iy,
                            height,
                            style_flags,
                            neighbor_sig
                        );
                        set_significant(states, padded_width, ix, iy);
                    }
                }
            }
        }
    }
}

template<typename Magnitude>
inline void j2k_classic_magnitude_refinement_pass(
    thread const Magnitude *magnitudes,
    thread uchar *states,
    uchar coded_marker,
    thread J2kMqEncoder &encoder,
    thread uchar *contexts,
    uint width,
    uint height,
    uint padded_width,
    uint bit_mask,
    uint style_flags
) {
    for (uint y_base = 0u; y_base < height; y_base += 4u) {
        for (uint x = 0u; x < width; ++x) {
            const uint y_end = min(y_base + 4u, height);
            for (uint y = y_base; y < y_end; ++y) {
                const uint ix = x + 1u;
                const uint iy = y + 1u;
                const uint idx = coeff_index(padded_width, ix, iy);
                if ((states[idx] & J2K_ENCODE_SIGNIFICANT) != 0u &&
                    !j2k_classic_coded_marker_matches(states, idx, coded_marker)) {
                    const uint ctx_label =
                        uint(magnitude_refinement_context(states, padded_width, ix, iy, height, style_flags));
                    const uint bit = (magnitudes[idx] & bit_mask) != 0u ? 1u : 0u;
                    j2k_mq_encode(encoder, contexts, ctx_label, bit);
                    states[idx] |= J2K_ENCODE_MAGNITUDE_REFINED;
                }
            }
        }
    }
}

template<typename Magnitude>
inline void j2k_classic_significance_pass_raw(
    thread const Magnitude *magnitudes,
    thread uchar *states,
    uchar coded_marker,
    thread J2kRawBitWriter &writer,
    uint width,
    uint height,
    uint padded_width,
    uint bit_mask,
    uint style_flags
) {
    for (uint y_base = 0u; y_base < height; y_base += 4u) {
        for (uint x = 0u; x < width; ++x) {
            const uint y_end = min(y_base + 4u, height);
            for (uint y = y_base; y < y_end; ++y) {
                const uint ix = x + 1u;
                const uint iy = y + 1u;
                const uint idx = coeff_index(padded_width, ix, iy);
                const uchar neighbor_sig =
                    j2k_classic_effective_neighbors(states, padded_width, ix, iy, height, style_flags);
                if ((states[idx] & J2K_ENCODE_SIGNIFICANT) == 0u && neighbor_sig != 0u) {
                    const uint bit = (magnitudes[idx] & bit_mask) != 0u ? 1u : 0u;
                    j2k_raw_writer_write_bit(writer, bit);
                    j2k_classic_set_coded_marker(states, idx, coded_marker);
                    if (bit != 0u) {
                        j2k_raw_writer_write_bit(writer, (uint(states[idx]) >> 5u) & 1u);
                        set_significant(states, padded_width, ix, iy);
                    }
                }
            }
        }
    }
}

template<typename Magnitude>
inline void j2k_classic_magnitude_refinement_pass_raw(
    thread const Magnitude *magnitudes,
    thread uchar *states,
    uchar coded_marker,
    thread J2kRawBitWriter &writer,
    uint width,
    uint height,
    uint padded_width,
    uint bit_mask
) {
    for (uint y_base = 0u; y_base < height; y_base += 4u) {
        for (uint x = 0u; x < width; ++x) {
            const uint y_end = min(y_base + 4u, height);
            for (uint y = y_base; y < y_end; ++y) {
                const uint ix = x + 1u;
                const uint iy = y + 1u;
                const uint idx = coeff_index(padded_width, ix, iy);
                if ((states[idx] & J2K_ENCODE_SIGNIFICANT) != 0u &&
                    !j2k_classic_coded_marker_matches(states, idx, coded_marker)) {
                    const uint bit = (magnitudes[idx] & bit_mask) != 0u ? 1u : 0u;
                    j2k_raw_writer_write_bit(writer, bit);
                    states[idx] |= J2K_ENCODE_MAGNITUDE_REFINED;
                }
            }
        }
    }
}

template<typename Magnitude>
inline void j2k_classic_cleanup_pass(
    thread const Magnitude *magnitudes,
    thread uchar *states,
    uchar coded_marker,
    thread J2kMqEncoder &encoder,
    thread uchar *contexts,
    uint width,
    uint height,
    uint padded_width,
    uint bit_mask,
    uint sub_band_type,
    uint style_flags
) {
    for (uint y_base = 0u; y_base < height; y_base += 4u) {
        for (uint x = 0u; x < width; ++x) {
            const uint y_end = min(y_base + 4u, height);
            const uint stripe_height = y_end - y_base;

            if (stripe_height == 4u) {
                bool all_zero_uncoded = true;
                for (uint y = y_base; y < y_end; ++y) {
                    const uint ix = x + 1u;
                    const uint iy = y + 1u;
                    const uint idx = coeff_index(padded_width, ix, iy);
                    const uchar neighbor_sig =
                        j2k_classic_effective_neighbors(states, padded_width, ix, iy, height, style_flags);
                    if ((states[idx] & J2K_ENCODE_SIGNIFICANT) != 0u ||
                        j2k_classic_coded_marker_matches(states, idx, coded_marker) ||
                        neighbor_sig != 0u) {
                        all_zero_uncoded = false;
                        break;
                    }
                }

                if (all_zero_uncoded) {
                    uint first_sig = 4u;
                    for (uint pos = 0u; pos < 4u; ++pos) {
                        const uint idx = coeff_index(padded_width, x + 1u, y_base + pos + 1u);
                        if ((magnitudes[idx] & bit_mask) != 0u) {
                            first_sig = pos;
                            break;
                        }
                    }

                    if (first_sig < 4u) {
                        j2k_mq_encode(encoder, contexts, 17u, 1u);
                        j2k_mq_encode(encoder, contexts, 18u, (first_sig >> 1u) & 1u);
                        j2k_mq_encode(encoder, contexts, 18u, first_sig & 1u);

                        const uint sig_y = y_base + first_sig;
                        const uint sig_idx = coeff_index(padded_width, x + 1u, sig_y + 1u);
                        j2k_classic_encode_sign_from_neighbors(
                            sig_idx,
                            states,
                            encoder,
                            contexts,
                            padded_width,
                            x + 1u,
                            sig_y + 1u,
                            height,
                            style_flags,
                            uchar(0u)
                        );
                        set_significant(states, padded_width, x + 1u, sig_y + 1u);

                        for (uint y = sig_y + 1u; y < y_end; ++y) {
                            const uint ix = x + 1u;
                            const uint iy = y + 1u;
                            const uint idx = coeff_index(padded_width, ix, iy);
                            if ((states[idx] & J2K_ENCODE_SIGNIFICANT) == 0u &&
                                !j2k_classic_coded_marker_matches(states, idx, coded_marker)) {
                                const uchar neighbor_sig =
                                    j2k_classic_effective_neighbors(
                                        states,
                                        padded_width,
                                        ix,
                                        iy,
                                        height,
                                        style_flags
                                    );
                                const uint ctx_label = uint(zero_context_label(neighbor_sig, sub_band_type));
                                const uint bit = (magnitudes[idx] & bit_mask) != 0u ? 1u : 0u;
                                j2k_mq_encode(encoder, contexts, ctx_label, bit);
                                if (bit != 0u) {
                                    j2k_classic_encode_sign_from_neighbors(
                                        idx,
                                        states,
                                        encoder,
                                        contexts,
                                        padded_width,
                                        ix,
                                        iy,
                                        height,
                                        style_flags,
                                        neighbor_sig
                                    );
                                    set_significant(states, padded_width, ix, iy);
                                }
                            }
                        }
                        continue;
                    }

                    j2k_mq_encode(encoder, contexts, 17u, 0u);
                    continue;
                }
            }

            for (uint y = y_base; y < y_end; ++y) {
                const uint ix = x + 1u;
                const uint iy = y + 1u;
                const uint idx = coeff_index(padded_width, ix, iy);
                if ((states[idx] & J2K_ENCODE_SIGNIFICANT) == 0u &&
                    !j2k_classic_coded_marker_matches(states, idx, coded_marker)) {
                    const uchar neighbor_sig =
                        j2k_classic_effective_neighbors(states, padded_width, ix, iy, height, style_flags);
                    const uint ctx_label = uint(zero_context_label(neighbor_sig, sub_band_type));
                    const uint bit = (magnitudes[idx] & bit_mask) != 0u ? 1u : 0u;
                    j2k_mq_encode(encoder, contexts, ctx_label, bit);
                    if (bit != 0u) {
                        j2k_classic_encode_sign_from_neighbors(
                            idx,
                            states,
                            encoder,
                            contexts,
                            padded_width,
                            ix,
                            iy,
                            height,
                            style_flags,
                            neighbor_sig
                        );
                        set_significant(states, padded_width, ix, iy);
                    }
                }
            }
        }
    }
}

inline void j2k_classic_encode_segmentation_symbols(
    thread J2kMqEncoder &encoder,
    thread uchar *contexts
) {
    j2k_mq_encode(encoder, contexts, 18u, 1u);
    j2k_mq_encode(encoder, contexts, 18u, 0u);
    j2k_mq_encode(encoder, contexts, 18u, 1u);
    j2k_mq_encode(encoder, contexts, 18u, 0u);
}

inline uint j2k_classic_bypass_segment_idx(uint pass_idx) {
    if (pass_idx < 10u) {
        return 0u;
    }
    return 1u + (2u * ((pass_idx - 10u) / 3u)) + (((pass_idx - 10u) % 3u) == 2u ? 1u : 0u);
}

inline bool j2k_classic_push_segment(
    device J2kClassicSegment *segments,
    uint segment_capacity,
    thread uint &segment_count,
    uint data_offset,
    uint data_length,
    uint start_pass,
    uint end_pass,
    bool use_arithmetic
) {
    if (segment_count >= segment_capacity) {
        return false;
    }
    segments[segment_count].data_offset = data_offset;
    segments[segment_count].data_length = data_length;
    segments[segment_count].start_coding_pass = start_pass;
    segments[segment_count].end_coding_pass = end_pass;
    segments[segment_count].use_arithmetic = use_arithmetic ? 1u : 0u;
    segment_count += 1u;
    return true;
}

inline uint j2k_classic_finish_arithmetic_segment(thread J2kMqEncoder &encoder) {
    j2k_mq_finish(encoder);
    if (encoder.failed != 0u || encoder.len == 0u) {
        encoder.failed = 1u;
        return 0u;
    }
    const uint data_len = encoder.len - 1u;
    for (uint idx = 0u; idx < data_len; ++idx) {
        encoder.data[idx] = encoder.data[idx + 1u];
    }
    return data_len;
}

inline uint j2k_classic_finish_arithmetic_segment_unshifted(thread J2kMqEncoder &encoder) {
    j2k_mq_finish(encoder);
    if (encoder.failed != 0u || encoder.len == 0u) {
        encoder.failed = 1u;
        return 0u;
    }
    return encoder.len - 1u;
}

inline void j2k_encode_classic_code_block_impl_with_scratch(
    device const int *coefficients,
    device uchar *out,
    J2kClassicEncodeParams params,
    device J2kClassicEncodeStatus *status,
    device J2kClassicSegment *segments,
    thread uint *magnitudes,
    thread uchar *states,
    uint max_width,
    uint max_height
) {
    j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 0u, 0u, 0u, 0u, 0u);

    if (params.width == 0u || params.height == 0u ||
        params.width > max_width ||
        params.height > max_height ||
        params.total_bitplanes > 31u) {
        j2k_set_encode_status(status, J2K_ENCODE_STATUS_UNSUPPORTED, 1u, 0u, 0u, 0u, 0u);
        return;
    }

    const uint padded_width = params.width + 2u;

    j2k_classic_clear_state_border(states, padded_width, params.width, params.height);
    uint max_magnitude = 0u;
    for (uint y = 0u; y < params.height; ++y) {
        for (uint x = 0u; x < params.width; ++x) {
            const uint src_idx = y * params.width + x;
            const int value = coefficients[src_idx];
            const uint dst_idx = coeff_index(padded_width, x + 1u, y + 1u);
            const uint magnitude = j2k_classic_magnitude(value);
            magnitudes[dst_idx] = magnitude;
            states[dst_idx] = value < 0 ? J2K_ENCODE_SIGN : uchar(0u);
            max_magnitude = max(max_magnitude, magnitude);
        }
    }

    if (max_magnitude == 0u) {
        j2k_set_encode_status(
            status,
            J2K_ENCODE_STATUS_OK,
            0u,
            0u,
            0u,
            params.total_bitplanes,
            0u
        );
        return;
    }

    const uint num_bitplanes = 32u - clz(max_magnitude);
    if (num_bitplanes > params.total_bitplanes) {
        j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 3u, 0u, 0u, 0u, 0u);
        return;
    }
    const uint missing_bit_planes = params.total_bitplanes - num_bitplanes;

    thread uchar contexts[19];
    reset_contexts(contexts);
    uchar coded_marker = uchar(1u);

    if ((params.style_flags & (J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS |
                               J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS)) != 0u) {
        const uint total_passes = 1u + 3u * (num_bitplanes - 1u);
        uint data_cursor = 0u;
        uint segment_count = 0u;
        uint current_segment_idx = 0xFFFFFFFFu;
        uint current_segment_start_pass = 0u;
        bool current_use_arithmetic = true;
        bool have_segment = false;
        thread J2kMqEncoder arithmetic_encoder;
        thread J2kRawBitWriter raw_writer;

        for (uint coding_pass = 0u; coding_pass < total_passes; ++coding_pass) {
            const uint segment_idx =
                (params.style_flags & J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS) != 0u
                    ? coding_pass
                    : j2k_classic_bypass_segment_idx(coding_pass);
            const bool use_arithmetic =
                (params.style_flags & J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS) == 0u ||
                coding_pass <= 9u ||
                (coding_pass % 3u) == 0u;

            if (!have_segment || current_segment_idx != segment_idx) {
                if (have_segment) {
                    uint segment_len = 0u;
                    if (current_use_arithmetic) {
                        segment_len = j2k_classic_finish_arithmetic_segment(arithmetic_encoder);
                        if (arithmetic_encoder.failed != 0u) {
                            j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 6u, 0u, 0u, 0u, 0u);
                            return;
                        }
                    } else {
                        j2k_raw_writer_finish(raw_writer);
                        if (raw_writer.failed != 0u) {
                            j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 7u, 0u, 0u, 0u, 0u);
                            return;
                        }
                        segment_len = raw_writer.len;
                    }
                    if (!j2k_classic_push_segment(
                            segments,
                            params.segment_capacity,
                            segment_count,
                            data_cursor,
                            segment_len,
                            current_segment_start_pass,
                            coding_pass,
                            current_use_arithmetic)) {
                        j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 8u, 0u, 0u, 0u, 0u);
                        return;
                    }
                    data_cursor += segment_len;
                }

                current_segment_idx = segment_idx;
                current_segment_start_pass = coding_pass;
                current_use_arithmetic = use_arithmetic;
                have_segment = true;
                if (data_cursor > params.output_capacity) {
                    j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 9u, 0u, 0u, 0u, 0u);
                    return;
                }
                const uint remaining_capacity = params.output_capacity - data_cursor;
                if (use_arithmetic) {
                    j2k_mq_init(arithmetic_encoder, out + data_cursor, remaining_capacity);
                } else {
                    j2k_raw_writer_init(raw_writer, out + data_cursor, remaining_capacity);
                }
            }

            const uint current_bitplane = (coding_pass + 2u) / 3u;
            const uint bit_mask = 1u << (num_bitplanes - 1u - current_bitplane);
            switch (coding_pass % 3u) {
                case 0u:
                    j2k_classic_cleanup_pass(
                        magnitudes,
                        states,
                        coded_marker,
                        arithmetic_encoder,
                        contexts,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        params.sub_band_type,
                        params.style_flags
                    );
                    if ((params.style_flags & J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS) != 0u) {
                        j2k_classic_encode_segmentation_symbols(arithmetic_encoder, contexts);
                    }
                    coded_marker += uchar(1u);
                    break;
                case 1u:
                    if (use_arithmetic) {
                        j2k_classic_significance_pass(
                            magnitudes,
                            states,
                            coded_marker,
                            arithmetic_encoder,
                            contexts,
                            params.width,
                            params.height,
                            padded_width,
                            bit_mask,
                            params.sub_band_type,
                            params.style_flags
                        );
                    } else {
                        j2k_classic_significance_pass_raw(
                            magnitudes,
                            states,
                            coded_marker,
                            raw_writer,
                            params.width,
                            params.height,
                            padded_width,
                            bit_mask,
                            params.style_flags
                        );
                    }
                    break;
                default:
                    if (use_arithmetic) {
                        j2k_classic_magnitude_refinement_pass(
                            magnitudes,
                            states,
                            coded_marker,
                            arithmetic_encoder,
                            contexts,
                            params.width,
                            params.height,
                            padded_width,
                            bit_mask,
                            params.style_flags
                        );
                    } else {
                        j2k_classic_magnitude_refinement_pass_raw(
                            magnitudes,
                            states,
                            coded_marker,
                            raw_writer,
                            params.width,
                            params.height,
                            padded_width,
                            bit_mask
                        );
                    }
                    break;
            }

            if ((params.style_flags & J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES) != 0u) {
                reset_contexts(contexts);
            }
            const bool current_failed = use_arithmetic
                ? arithmetic_encoder.failed != 0u
                : raw_writer.failed != 0u;
            if (current_failed) {
                j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 10u, 0u, 0u, 0u, 0u);
        return;
    }
}

        if (have_segment) {
            uint segment_len = 0u;
            if (current_use_arithmetic) {
                segment_len = j2k_classic_finish_arithmetic_segment(arithmetic_encoder);
                if (arithmetic_encoder.failed != 0u) {
                    j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 11u, 0u, 0u, 0u, 0u);
                    return;
                }
            } else {
                j2k_raw_writer_finish(raw_writer);
                if (raw_writer.failed != 0u) {
                    j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 12u, 0u, 0u, 0u, 0u);
                    return;
                }
                segment_len = raw_writer.len;
            }
            if (!j2k_classic_push_segment(
                    segments,
                    params.segment_capacity,
                    segment_count,
                    data_cursor,
                    segment_len,
                    current_segment_start_pass,
                    total_passes,
                    current_use_arithmetic)) {
                j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 13u, 0u, 0u, 0u, 0u);
                return;
            }
            data_cursor += segment_len;
        }

        j2k_set_encode_status(
            status,
            J2K_ENCODE_STATUS_OK,
            0u,
            data_cursor,
            total_passes,
            missing_bit_planes,
            segment_count
        );
        return;
    }

    thread J2kMqEncoder encoder;
    j2k_mq_init(encoder, out, params.output_capacity);

    uint pass_count = 0u;
    for (int bp = int(num_bitplanes) - 1; bp >= 0; --bp) {
        const uint bit_mask = 1u << uint(bp);
        const bool first_bitplane = uint(bp) == num_bitplanes - 1u;

        if (first_bitplane) {
            j2k_classic_cleanup_pass(
                magnitudes,
                states,
                coded_marker,
                encoder,
                contexts,
                params.width,
                params.height,
                padded_width,
                bit_mask,
                params.sub_band_type,
                params.style_flags
            );
            if ((params.style_flags & J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS) != 0u) {
                j2k_classic_encode_segmentation_symbols(encoder, contexts);
            }
            pass_count += 1u;
            if ((params.style_flags & J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES) != 0u) {
                reset_contexts(contexts);
            }
        } else {
            j2k_classic_significance_pass(
                magnitudes,
                states,
                coded_marker,
                encoder,
                contexts,
                params.width,
                params.height,
                padded_width,
                bit_mask,
                params.sub_band_type,
                params.style_flags
            );
            pass_count += 1u;
            if ((params.style_flags & J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES) != 0u) {
                reset_contexts(contexts);
            }

            j2k_classic_magnitude_refinement_pass(
                magnitudes,
                states,
                coded_marker,
                encoder,
                contexts,
                params.width,
                params.height,
                padded_width,
                bit_mask,
                params.style_flags
            );
            pass_count += 1u;
            if ((params.style_flags & J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES) != 0u) {
                reset_contexts(contexts);
            }

            j2k_classic_cleanup_pass(
                magnitudes,
                states,
                coded_marker,
                encoder,
                contexts,
                params.width,
                params.height,
                padded_width,
                bit_mask,
                params.sub_band_type,
                params.style_flags
            );
            if ((params.style_flags & J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS) != 0u) {
                j2k_classic_encode_segmentation_symbols(encoder, contexts);
            }
            pass_count += 1u;
            if ((params.style_flags & J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES) != 0u) {
                reset_contexts(contexts);
            }
        }

        coded_marker += uchar(1u);

        if (encoder.failed != 0u) {
            j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 4u, 0u, 0u, 0u, 0u);
            return;
        }
    }

    const uint payload_skip = 1u;
    const uint data_len = j2k_classic_finish_arithmetic_segment_unshifted(encoder);
    if (encoder.failed != 0u || encoder.len == 0u) {
        j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 5u, 0u, 0u, 0u, 0u);
        return;
    }
    uint segment_count = 0u;
    if (!j2k_classic_push_segment(
            segments,
            params.segment_capacity,
            segment_count,
            0u,
            data_len,
            0u,
            pass_count,
            true)) {
        j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 14u, 0u, 0u, 0u, 0u);
        return;
    }

    j2k_set_encode_status_with_payload_skip(
        status,
        J2K_ENCODE_STATUS_OK,
        0u,
        data_len,
        pass_count,
        missing_bit_planes,
        segment_count,
        payload_skip
    );
}

inline void j2k_encode_classic_code_block_impl(
    device const int *coefficients,
    device uchar *out,
    J2kClassicEncodeParams params,
    device J2kClassicEncodeStatus *status,
    device J2kClassicSegment *segments
) {
    thread uint magnitudes[J2K_CLASSIC_MAX_COEFF_COUNT];
    thread uchar states[J2K_CLASSIC_MAX_COEFF_COUNT];
    j2k_encode_classic_code_block_impl_with_scratch(
        coefficients,
        out,
        params,
        status,
        segments,
        magnitudes,
        states,
        J2K_CLASSIC_MAX_WIDTH,
        J2K_CLASSIC_MAX_HEIGHT
    );
}

inline void j2k_encode_classic_code_block_impl_style0_with_scratch(
    device const int *coefficients,
    device uchar *out,
    J2kClassicEncodeParams params,
    device J2kClassicEncodeStatus *status,
    device J2kClassicSegment *segments,
    thread uint *magnitudes,
    thread uchar *states,
    uint max_width,
    uint max_height
) {
    j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 0u, 0u, 0u, 0u, 0u);

    if (params.width == 0u || params.height == 0u ||
        params.width > max_width ||
        params.height > max_height ||
        params.total_bitplanes > 31u) {
        j2k_set_encode_status(status, J2K_ENCODE_STATUS_UNSUPPORTED, 1u, 0u, 0u, 0u, 0u);
        return;
    }

    const uint padded_width = params.width + 2u;

    j2k_classic_clear_state_border(states, padded_width, params.width, params.height);
    uint max_magnitude = 0u;
    for (uint y = 0u; y < params.height; ++y) {
        for (uint x = 0u; x < params.width; ++x) {
            const uint src_idx = y * params.width + x;
            const int value = coefficients[src_idx];
            const uint dst_idx = coeff_index(padded_width, x + 1u, y + 1u);
            const uint magnitude = j2k_classic_magnitude(value);
            magnitudes[dst_idx] = magnitude;
            states[dst_idx] = value < 0 ? J2K_ENCODE_SIGN : uchar(0u);
            max_magnitude = max(max_magnitude, magnitude);
        }
    }

    if (max_magnitude == 0u) {
        j2k_set_encode_status(
            status,
            J2K_ENCODE_STATUS_OK,
            0u,
            0u,
            0u,
            params.total_bitplanes,
            0u
        );
        return;
    }

    const uint num_bitplanes = 32u - clz(max_magnitude);
    if (num_bitplanes > params.total_bitplanes) {
        j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 3u, 0u, 0u, 0u, 0u);
        return;
    }
    const uint missing_bit_planes = params.total_bitplanes - num_bitplanes;

    thread uchar contexts[19];
    reset_contexts(contexts);
    uchar coded_marker = uchar(1u);
    thread J2kMqEncoder encoder;
    j2k_mq_init(encoder, out, params.output_capacity);

    uint pass_count = 0u;
    for (int bp = int(num_bitplanes) - 1; bp >= 0; --bp) {
        const uint bit_mask = 1u << uint(bp);
        const bool first_bitplane = uint(bp) == num_bitplanes - 1u;

        if (first_bitplane) {
            j2k_classic_cleanup_pass(
                magnitudes,
                states,
                coded_marker,
                encoder,
                contexts,
                params.width,
                params.height,
                padded_width,
                bit_mask,
                params.sub_band_type,
                0u
            );
            pass_count += 1u;
        } else {
            j2k_classic_significance_pass(
                magnitudes,
                states,
                coded_marker,
                encoder,
                contexts,
                params.width,
                params.height,
                padded_width,
                bit_mask,
                params.sub_band_type,
                0u
            );
            pass_count += 1u;

            j2k_classic_magnitude_refinement_pass(
                magnitudes,
                states,
                coded_marker,
                encoder,
                contexts,
                params.width,
                params.height,
                padded_width,
                bit_mask,
                0u
            );
            pass_count += 1u;

            j2k_classic_cleanup_pass(
                magnitudes,
                states,
                coded_marker,
                encoder,
                contexts,
                params.width,
                params.height,
                padded_width,
                bit_mask,
                params.sub_band_type,
                0u
            );
            pass_count += 1u;
        }

        coded_marker += uchar(1u);

        if (encoder.failed != 0u) {
            j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 4u, 0u, 0u, 0u, 0u);
            return;
        }
    }

    const uint payload_skip = 1u;
    const uint data_len = j2k_classic_finish_arithmetic_segment_unshifted(encoder);
    if (encoder.failed != 0u || encoder.len == 0u) {
        j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 5u, 0u, 0u, 0u, 0u);
        return;
    }
    uint segment_count = 0u;
    if (!j2k_classic_push_segment(
            segments,
            params.segment_capacity,
            segment_count,
            0u,
            data_len,
            0u,
            pass_count,
            true)) {
        j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 14u, 0u, 0u, 0u, 0u);
        return;
    }

    j2k_set_encode_status_with_payload_skip(
        status,
        J2K_ENCODE_STATUS_OK,
        0u,
        data_len,
        pass_count,
        missing_bit_planes,
        segment_count,
        payload_skip
    );
}

inline void j2k_encode_classic_code_block_impl_style0(
    device const int *coefficients,
    device uchar *out,
    J2kClassicEncodeParams params,
    device J2kClassicEncodeStatus *status,
    device J2kClassicSegment *segments
) {
    thread uint magnitudes[J2K_CLASSIC_MAX_COEFF_COUNT];
    thread uchar states[J2K_CLASSIC_MAX_COEFF_COUNT];
    j2k_encode_classic_code_block_impl_style0_with_scratch(
        coefficients,
        out,
        params,
        status,
        segments,
        magnitudes,
        states,
        J2K_CLASSIC_MAX_WIDTH,
        J2K_CLASSIC_MAX_HEIGHT
    );
}

inline void j2k_encode_classic_code_block_impl_32(
    device const int *coefficients,
    device uchar *out,
    J2kClassicEncodeParams params,
    device J2kClassicEncodeStatus *status,
    device J2kClassicSegment *segments
) {
    thread uint magnitudes[J2K_CLASSIC_ENCODE_32_MAX_COEFF_COUNT];
    thread uchar states[J2K_CLASSIC_ENCODE_32_MAX_COEFF_COUNT];
    j2k_encode_classic_code_block_impl_with_scratch(
        coefficients,
        out,
        params,
        status,
        segments,
        magnitudes,
        states,
        J2K_CLASSIC_ENCODE_32_MAX_WIDTH,
        J2K_CLASSIC_ENCODE_32_MAX_HEIGHT
    );
}

template<typename Magnitude>
inline void j2k_encode_classic_code_block_impl_bypass_32_with_scratch(
    device const int *coefficients,
    device uchar *out,
    J2kClassicEncodeParams params,
    device J2kClassicEncodeStatus *status,
    device J2kClassicSegment *segments,
    thread Magnitude *magnitudes,
    thread uchar *states,
    uint max_bitplanes
) {
    j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 0u, 0u, 0u, 0u, 0u);

    if (params.width == 0u || params.height == 0u ||
        params.width > J2K_CLASSIC_ENCODE_32_MAX_WIDTH ||
        params.height > J2K_CLASSIC_ENCODE_32_MAX_HEIGHT ||
        params.total_bitplanes > max_bitplanes) {
        j2k_set_encode_status(status, J2K_ENCODE_STATUS_UNSUPPORTED, 1u, 0u, 0u, 0u, 0u);
        return;
    }

    const uint padded_width = params.width + 2u;

    j2k_classic_clear_state_border(states, padded_width, params.width, params.height);
    uint max_magnitude = 0u;
    for (uint y = 0u; y < params.height; ++y) {
        for (uint x = 0u; x < params.width; ++x) {
            const uint src_idx = y * params.width + x;
            const int value = coefficients[src_idx];
            const uint dst_idx = coeff_index(padded_width, x + 1u, y + 1u);
            const uint magnitude = j2k_classic_magnitude(value);
            magnitudes[dst_idx] = Magnitude(magnitude);
            states[dst_idx] = value < 0 ? J2K_ENCODE_SIGN : uchar(0u);
            max_magnitude = max(max_magnitude, magnitude);
        }
    }

    if (max_magnitude == 0u) {
        j2k_set_encode_status(
            status,
            J2K_ENCODE_STATUS_OK,
            0u,
            0u,
            0u,
            params.total_bitplanes,
            0u
        );
        return;
    }

    const uint num_bitplanes = 32u - clz(max_magnitude);
    if (num_bitplanes > params.total_bitplanes) {
        j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 3u, 0u, 0u, 0u, 0u);
        return;
    }
    const uint missing_bit_planes = params.total_bitplanes - num_bitplanes;

    thread uchar contexts[19];
    reset_contexts(contexts);
    uchar coded_marker = uchar(1u);
    const uint total_passes = 1u + 3u * (num_bitplanes - 1u);
    uint data_cursor = 0u;
    uint physical_cursor = 0u;
    uint segment_count = 0u;
    uint current_segment_idx = 0xFFFFFFFFu;
    uint current_segment_start_pass = 0u;
    bool current_use_arithmetic = true;
    bool have_segment = false;
    thread J2kMqEncoder arithmetic_encoder;
    thread J2kRawBitWriter raw_writer;

    for (uint coding_pass = 0u; coding_pass < total_passes; ++coding_pass) {
        const uint segment_idx = j2k_classic_bypass_segment_idx(coding_pass);
        const bool use_arithmetic = coding_pass <= 9u || (coding_pass % 3u) == 0u;

        if (!have_segment || current_segment_idx != segment_idx) {
            if (have_segment) {
                uint segment_len = 0u;
                if (current_use_arithmetic) {
                    segment_len = j2k_classic_finish_arithmetic_segment(arithmetic_encoder);
                    if (arithmetic_encoder.failed != 0u) {
                        j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 6u, 0u, 0u, 0u, 0u);
                        return;
                    }
                    physical_cursor += segment_len;
                } else {
                    j2k_raw_writer_finish(raw_writer);
                    if (raw_writer.failed != 0u) {
                        j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 7u, 0u, 0u, 0u, 0u);
                        return;
                    }
                    segment_len = raw_writer.len;
                    physical_cursor += segment_len;
                }
                if (!j2k_classic_push_segment(
                        segments,
                        params.segment_capacity,
                        segment_count,
                        data_cursor,
                        segment_len,
                        current_segment_start_pass,
                        coding_pass,
                        current_use_arithmetic)) {
                    j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 8u, 0u, 0u, 0u, 0u);
                    return;
                }
                data_cursor += segment_len;
            }

            current_segment_idx = segment_idx;
            current_segment_start_pass = coding_pass;
            current_use_arithmetic = use_arithmetic;
            have_segment = true;
            if (physical_cursor > params.output_capacity) {
                j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 9u, 0u, 0u, 0u, 0u);
                return;
            }
            const uint remaining_capacity = params.output_capacity - physical_cursor;
            if (use_arithmetic) {
                j2k_mq_init(arithmetic_encoder, out + physical_cursor, remaining_capacity);
            } else {
                j2k_raw_writer_init(raw_writer, out + physical_cursor, remaining_capacity);
            }
        }

        const uint current_bitplane = (coding_pass + 2u) / 3u;
            const uint bit_mask = 1u << (num_bitplanes - 1u - current_bitplane);
            switch (coding_pass % 3u) {
            case 0u:
                j2k_classic_cleanup_pass(
                    magnitudes,
                    states,
                    coded_marker,
                    arithmetic_encoder,
                    contexts,
                    params.width,
                    params.height,
                    padded_width,
                    bit_mask,
                    params.sub_band_type,
                    0u
                );
                coded_marker += uchar(1u);
                break;
            case 1u:
                if (use_arithmetic) {
                    j2k_classic_significance_pass(
                        magnitudes,
                        states,
                        coded_marker,
                        arithmetic_encoder,
                        contexts,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        params.sub_band_type,
                        0u
                    );
                } else {
                    j2k_classic_significance_pass_raw(
                        magnitudes,
                        states,
                        coded_marker,
                        raw_writer,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        0u
                    );
                }
                break;
            default:
                if (use_arithmetic) {
                    j2k_classic_magnitude_refinement_pass(
                        magnitudes,
                        states,
                        coded_marker,
                        arithmetic_encoder,
                        contexts,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        0u
                    );
                } else {
                    j2k_classic_magnitude_refinement_pass_raw(
                        magnitudes,
                        states,
                        coded_marker,
                        raw_writer,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask
                    );
                }
                break;
        }

        const bool current_failed = use_arithmetic
            ? arithmetic_encoder.failed != 0u
            : raw_writer.failed != 0u;
        if (current_failed) {
            j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 10u, 0u, 0u, 0u, 0u);
            return;
        }
    }

    if (have_segment) {
        uint segment_len = 0u;
        if (current_use_arithmetic) {
            segment_len = j2k_classic_finish_arithmetic_segment(arithmetic_encoder);
            if (arithmetic_encoder.failed != 0u) {
                j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 11u, 0u, 0u, 0u, 0u);
                return;
            }
            physical_cursor += segment_len;
        } else {
            j2k_raw_writer_finish(raw_writer);
            if (raw_writer.failed != 0u) {
                j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 12u, 0u, 0u, 0u, 0u);
                return;
            }
            segment_len = raw_writer.len;
            physical_cursor += segment_len;
        }
        if (!j2k_classic_push_segment(
                segments,
                params.segment_capacity,
                segment_count,
                data_cursor,
                segment_len,
                current_segment_start_pass,
                total_passes,
                current_use_arithmetic)) {
            j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 13u, 0u, 0u, 0u, 0u);
            return;
        }
        data_cursor += segment_len;
    }

    j2k_set_encode_status(
        status,
        J2K_ENCODE_STATUS_OK,
        0u,
        data_cursor,
        total_passes,
        missing_bit_planes,
        segment_count
    );
}

inline void j2k_encode_classic_code_block_impl_bypass_32(
    device const int *coefficients,
    device uchar *out,
    J2kClassicEncodeParams params,
    device J2kClassicEncodeStatus *status,
    device J2kClassicSegment *segments
) {
    thread uint magnitudes[J2K_CLASSIC_ENCODE_32_MAX_COEFF_COUNT];
    thread uchar states[J2K_CLASSIC_ENCODE_32_MAX_COEFF_COUNT];
    j2k_encode_classic_code_block_impl_bypass_32_with_scratch(
        coefficients,
        out,
        params,
        status,
        segments,
        magnitudes,
        states,
        31u
    );
}

inline void j2k_encode_classic_code_block_impl_bypass_u16_32(
    device const int *coefficients,
    device uchar *out,
    J2kClassicEncodeParams params,
    device J2kClassicEncodeStatus *status,
    device J2kClassicSegment *segments
) {
    thread ushort magnitudes[J2K_CLASSIC_ENCODE_32_MAX_COEFF_COUNT];
    thread uchar states[J2K_CLASSIC_ENCODE_32_MAX_COEFF_COUNT];
    j2k_encode_classic_code_block_impl_bypass_32_with_scratch(
        coefficients,
        out,
        params,
        status,
        segments,
        magnitudes,
        states,
        16u
    );
}

inline void j2k_encode_classic_code_block_impl_style0_32(
    device const int *coefficients,
    device uchar *out,
    J2kClassicEncodeParams params,
    device J2kClassicEncodeStatus *status,
    device J2kClassicSegment *segments
) {
    thread uint magnitudes[J2K_CLASSIC_ENCODE_32_MAX_COEFF_COUNT];
    thread uchar states[J2K_CLASSIC_ENCODE_32_MAX_COEFF_COUNT];
    j2k_encode_classic_code_block_impl_style0_with_scratch(
        coefficients,
        out,
        params,
        status,
        segments,
        magnitudes,
        states,
        J2K_CLASSIC_ENCODE_32_MAX_WIDTH,
        J2K_CLASSIC_ENCODE_32_MAX_HEIGHT
    );
}

struct J2kClassicEncodeBatchJob {
    uint coefficient_offset;
    uint output_offset;
    uint segment_offset;
    uint width;
    uint height;
    uint sub_band_type;
    uint total_bitplanes;
    uint style_flags;
    uint output_capacity;
    uint segment_capacity;
};

struct J2kClassicTier1DensityCounters {
    uint sigprop_active_candidates;
    uint sigprop_new_significant;
    uint magref_active_candidates;
    uint cleanup_active_candidates;
    uint cleanup_new_significant;
    uint cleanup_rlc_stripes;
    uint cleanup_rlc_zero_stripes;
    uint arithmetic_sigprop_active_candidates;
    uint arithmetic_sigprop_new_significant;
    uint raw_sigprop_active_candidates;
    uint raw_sigprop_new_significant;
    uint arithmetic_magref_active_candidates;
    uint raw_magref_active_candidates;
    uint reserved0;
    uint reserved1;
};

struct J2kClassicTier1SymbolPlanCounters {
    uint code;
    uint detail;
    uint coding_passes;
    uint missing_bit_planes;
    uint segment_count;
    uint mq_symbol_count;
    uint raw_bit_count;
    uint cleanup_mq_symbol_count;
    uint sigprop_mq_symbol_count;
    uint magref_mq_symbol_count;
    uint raw_sigprop_bit_count;
    uint raw_magref_bit_count;
    uint cleanup_sign_symbol_count;
    uint sigprop_sign_symbol_count;
    uint mq_symbol_hash;
    uint raw_bit_hash;
};

struct J2kClassicTier1PassPlanCounters {
    uint code;
    uint detail;
    uint coding_passes;
    uint missing_bit_planes;
    uint segment_count;
    uint mq_symbol_count;
    uint raw_bit_count;
    uint nonempty_mq_passes;
    uint nonempty_raw_passes;
    uint max_mq_symbols_per_pass;
    uint max_raw_bits_per_pass;
    uint reserved0;
    uint reserved1;
    uint reserved2;
    uint reserved3;
    uint reserved4;
    uint mq_symbols_by_pass[J2K_CLASSIC_TIER1_PASS_PLAN_CAPACITY];
    uint raw_bits_by_pass[J2K_CLASSIC_TIER1_PASS_PLAN_CAPACITY];
};

struct J2kClassicTier1TokenSegment {
    uint token_bit_offset;
    uint token_bit_count;
    uint pass_range;
    uint flags;
};

struct J2kClassicTier1TokenBitWriter {
    device uchar *data;
    uint capacity;
    uint len;
    uint bit_count;
    uint current_byte;
    uint bits_in_current;
    uint failed;
};

struct J2kClassicTier1TokenByteWriter {
    device uchar *data;
    uint capacity;
    uint len;
    uint failed;
};

struct J2kClassicTier1TokenEmitter {
    J2kClassicTier1SymbolPlanCounters counters;
    J2kClassicTier1TokenBitWriter writer;
    device J2kClassicTier1TokenSegment *segments;
    uint segment_capacity;
    uint current_segment_start_bit;
    uint segment_failed;
};

struct J2kClassicTier1SplitTokenEmitter {
    J2kClassicTier1SymbolPlanCounters counters;
    J2kClassicTier1TokenBitWriter mq_writer;
    J2kClassicTier1TokenBitWriter raw_writer;
    device J2kClassicTier1TokenSegment *segments;
    uint segment_capacity;
    uint current_segment_start_bit;
    uint segment_failed;
};

struct J2kClassicTier1SplitMqByteRawTokenEmitter {
    J2kClassicTier1SymbolPlanCounters counters;
    J2kClassicTier1TokenByteWriter mq_writer;
    J2kClassicTier1TokenBitWriter raw_writer;
    device J2kClassicTier1TokenSegment *segments;
    uint segment_capacity;
    uint current_segment_start_bit;
    uint segment_failed;
};

struct J2kClassicTier1PassPlanEmitter {
    J2kClassicTier1PassPlanCounters counters;
    uint current_pass;
};

struct J2kClassicTier1TokenBitReader {
    device const uchar *data;
    uint bit_pos;
    uint bit_capacity;
    uint failed;
};

inline void j2k_classic_tier1_density_zero(thread J2kClassicTier1DensityCounters &counters) {
    counters.sigprop_active_candidates = 0u;
    counters.sigprop_new_significant = 0u;
    counters.magref_active_candidates = 0u;
    counters.cleanup_active_candidates = 0u;
    counters.cleanup_new_significant = 0u;
    counters.cleanup_rlc_stripes = 0u;
    counters.cleanup_rlc_zero_stripes = 0u;
    counters.arithmetic_sigprop_active_candidates = 0u;
    counters.arithmetic_sigprop_new_significant = 0u;
    counters.raw_sigprop_active_candidates = 0u;
    counters.raw_sigprop_new_significant = 0u;
    counters.arithmetic_magref_active_candidates = 0u;
    counters.raw_magref_active_candidates = 0u;
    counters.reserved0 = 0u;
    counters.reserved1 = 0u;
}

inline void j2k_classic_tier1_density_store(
    device J2kClassicTier1DensityCounters *out,
    thread const J2kClassicTier1DensityCounters &counters
) {
    out->sigprop_active_candidates = counters.sigprop_active_candidates;
    out->sigprop_new_significant = counters.sigprop_new_significant;
    out->magref_active_candidates = counters.magref_active_candidates;
    out->cleanup_active_candidates = counters.cleanup_active_candidates;
    out->cleanup_new_significant = counters.cleanup_new_significant;
    out->cleanup_rlc_stripes = counters.cleanup_rlc_stripes;
    out->cleanup_rlc_zero_stripes = counters.cleanup_rlc_zero_stripes;
    out->arithmetic_sigprop_active_candidates = counters.arithmetic_sigprop_active_candidates;
    out->arithmetic_sigprop_new_significant = counters.arithmetic_sigprop_new_significant;
    out->raw_sigprop_active_candidates = counters.raw_sigprop_active_candidates;
    out->raw_sigprop_new_significant = counters.raw_sigprop_new_significant;
    out->arithmetic_magref_active_candidates = counters.arithmetic_magref_active_candidates;
    out->raw_magref_active_candidates = counters.raw_magref_active_candidates;
    out->reserved0 = counters.reserved0;
    out->reserved1 = counters.reserved1;
}

inline void j2k_classic_symbol_plan_zero(thread J2kClassicTier1SymbolPlanCounters &counters) {
    counters.code = J2K_ENCODE_STATUS_OK;
    counters.detail = 0u;
    counters.coding_passes = 0u;
    counters.missing_bit_planes = 0u;
    counters.segment_count = 0u;
    counters.mq_symbol_count = 0u;
    counters.raw_bit_count = 0u;
    counters.cleanup_mq_symbol_count = 0u;
    counters.sigprop_mq_symbol_count = 0u;
    counters.magref_mq_symbol_count = 0u;
    counters.raw_sigprop_bit_count = 0u;
    counters.raw_magref_bit_count = 0u;
    counters.cleanup_sign_symbol_count = 0u;
    counters.sigprop_sign_symbol_count = 0u;
    counters.mq_symbol_hash = 2166136261u;
    counters.raw_bit_hash = 2166136261u;
}

inline void j2k_classic_symbol_plan_store(
    device J2kClassicTier1SymbolPlanCounters *out,
    thread const J2kClassicTier1SymbolPlanCounters &counters
) {
    out->code = counters.code;
    out->detail = counters.detail;
    out->coding_passes = counters.coding_passes;
    out->missing_bit_planes = counters.missing_bit_planes;
    out->segment_count = counters.segment_count;
    out->mq_symbol_count = counters.mq_symbol_count;
    out->raw_bit_count = counters.raw_bit_count;
    out->cleanup_mq_symbol_count = counters.cleanup_mq_symbol_count;
    out->sigprop_mq_symbol_count = counters.sigprop_mq_symbol_count;
    out->magref_mq_symbol_count = counters.magref_mq_symbol_count;
    out->raw_sigprop_bit_count = counters.raw_sigprop_bit_count;
    out->raw_magref_bit_count = counters.raw_magref_bit_count;
    out->cleanup_sign_symbol_count = counters.cleanup_sign_symbol_count;
    out->sigprop_sign_symbol_count = counters.sigprop_sign_symbol_count;
    out->mq_symbol_hash = counters.mq_symbol_hash;
    out->raw_bit_hash = counters.raw_bit_hash;
}

inline void j2k_classic_pass_plan_zero(thread J2kClassicTier1PassPlanCounters &counters) {
    counters.code = J2K_ENCODE_STATUS_OK;
    counters.detail = 0u;
    counters.coding_passes = 0u;
    counters.missing_bit_planes = 0u;
    counters.segment_count = 0u;
    counters.mq_symbol_count = 0u;
    counters.raw_bit_count = 0u;
    counters.nonempty_mq_passes = 0u;
    counters.nonempty_raw_passes = 0u;
    counters.max_mq_symbols_per_pass = 0u;
    counters.max_raw_bits_per_pass = 0u;
    counters.reserved0 = 0u;
    counters.reserved1 = 0u;
    counters.reserved2 = 0u;
    counters.reserved3 = 0u;
    counters.reserved4 = 0u;
    for (uint pass = 0u; pass < J2K_CLASSIC_TIER1_PASS_PLAN_CAPACITY; ++pass) {
        counters.mq_symbols_by_pass[pass] = 0u;
        counters.raw_bits_by_pass[pass] = 0u;
    }
}

inline void j2k_classic_pass_plan_store(
    device J2kClassicTier1PassPlanCounters *out,
    thread const J2kClassicTier1PassPlanCounters &counters
) {
    out->code = counters.code;
    out->detail = counters.detail;
    out->coding_passes = counters.coding_passes;
    out->missing_bit_planes = counters.missing_bit_planes;
    out->segment_count = counters.segment_count;
    out->mq_symbol_count = counters.mq_symbol_count;
    out->raw_bit_count = counters.raw_bit_count;
    out->nonempty_mq_passes = counters.nonempty_mq_passes;
    out->nonempty_raw_passes = counters.nonempty_raw_passes;
    out->max_mq_symbols_per_pass = counters.max_mq_symbols_per_pass;
    out->max_raw_bits_per_pass = counters.max_raw_bits_per_pass;
    out->reserved0 = counters.reserved0;
    out->reserved1 = counters.reserved1;
    out->reserved2 = counters.reserved2;
    out->reserved3 = counters.reserved3;
    out->reserved4 = counters.reserved4;
    for (uint pass = 0u; pass < J2K_CLASSIC_TIER1_PASS_PLAN_CAPACITY; ++pass) {
        out->mq_symbols_by_pass[pass] = counters.mq_symbols_by_pass[pass];
        out->raw_bits_by_pass[pass] = counters.raw_bits_by_pass[pass];
    }
}

inline void j2k_classic_tier1_token_writer_init(
    thread J2kClassicTier1TokenBitWriter &writer,
    device uchar *data,
    uint capacity
) {
    writer.data = data;
    writer.capacity = capacity;
    writer.len = 0u;
    writer.bit_count = 0u;
    writer.current_byte = 0u;
    writer.bits_in_current = 0u;
    writer.failed = 0u;
}

inline void j2k_classic_tier1_token_writer_push_byte(
    thread J2kClassicTier1TokenBitWriter &writer,
    uint value
) {
    if (writer.len >= writer.capacity) {
        writer.failed = 1u;
        return;
    }
    writer.data[writer.len] = uchar(value & 0xFFu);
    writer.len += 1u;
}

inline void j2k_classic_tier1_token_writer_write_bit(
    thread J2kClassicTier1TokenBitWriter &writer,
    uint bit
) {
    writer.current_byte = (writer.current_byte << 1u) | (bit & 1u);
    writer.bits_in_current += 1u;
    writer.bit_count += 1u;
    if (writer.bits_in_current == 8u) {
        j2k_classic_tier1_token_writer_push_byte(writer, writer.current_byte);
        writer.current_byte = 0u;
        writer.bits_in_current = 0u;
    }
}

inline void j2k_classic_tier1_token_writer_write_bits(
    thread J2kClassicTier1TokenBitWriter &writer,
    uint value,
    uint width
) {
    for (uint bit_idx = 0u; bit_idx < width; ++bit_idx) {
        const uint shift = width - 1u - bit_idx;
        j2k_classic_tier1_token_writer_write_bit(writer, (value >> shift) & 1u);
    }
}

inline void j2k_classic_tier1_token_writer_finish(
    thread J2kClassicTier1TokenBitWriter &writer
) {
    if (writer.bits_in_current == 0u) {
        return;
    }
    j2k_classic_tier1_token_writer_push_byte(
        writer,
        writer.current_byte << (8u - writer.bits_in_current)
    );
    writer.current_byte = 0u;
    writer.bits_in_current = 0u;
}

inline void j2k_classic_tier1_token_byte_writer_init(
    thread J2kClassicTier1TokenByteWriter &writer,
    device uchar *data,
    uint capacity
) {
    writer.data = data;
    writer.capacity = capacity;
    writer.len = 0u;
    writer.failed = 0u;
}

inline void j2k_classic_tier1_token_byte_writer_push(
    thread J2kClassicTier1TokenByteWriter &writer,
    uint value
) {
    if (writer.len >= writer.capacity) {
        writer.failed = 1u;
        return;
    }
    writer.data[writer.len] = uchar(value & 0xFFu);
    writer.len += 1u;
}

inline void j2k_classic_tier1_token_emit_init(
    thread J2kClassicTier1TokenEmitter &emitter,
    device uchar *token_data,
    uint token_capacity,
    device J2kClassicTier1TokenSegment *segments,
    uint segment_capacity
) {
    j2k_classic_symbol_plan_zero(emitter.counters);
    j2k_classic_tier1_token_writer_init(emitter.writer, token_data, token_capacity);
    emitter.segments = segments;
    emitter.segment_capacity = segment_capacity;
    emitter.current_segment_start_bit = 0u;
    emitter.segment_failed = 0u;
}

inline void j2k_classic_tier1_token_emit_push_segment(
    thread J2kClassicTier1TokenEmitter &emitter,
    uint start_pass,
    uint end_pass,
    bool use_arithmetic
) {
    const uint segment_idx = emitter.counters.segment_count;
    emitter.counters.segment_count += 1u;
    if (segment_idx >= emitter.segment_capacity) {
        emitter.segment_failed = 1u;
        return;
    }
    const uint token_bit_count = emitter.writer.bit_count - emitter.current_segment_start_bit;
    emitter.segments[segment_idx].token_bit_offset = emitter.current_segment_start_bit;
    emitter.segments[segment_idx].token_bit_count = token_bit_count;
    emitter.segments[segment_idx].pass_range =
        (start_pass & 0xFFFFu) | ((end_pass & 0xFFFFu) << 16u);
    emitter.segments[segment_idx].flags = use_arithmetic ? 1u : 0u;
    emitter.current_segment_start_bit = emitter.writer.bit_count;
}

inline void j2k_classic_tier1_token_emit_store(
    device J2kClassicTier1SymbolPlanCounters *out,
    thread J2kClassicTier1TokenEmitter &emitter
) {
    if (emitter.writer.failed != 0u) {
        emitter.counters.code = J2K_ENCODE_STATUS_FAIL;
        emitter.counters.detail = 21u;
    } else if (emitter.segment_failed != 0u) {
        emitter.counters.code = J2K_ENCODE_STATUS_FAIL;
        emitter.counters.detail = 22u;
    }
    j2k_classic_symbol_plan_store(out, emitter.counters);
}

inline void j2k_classic_tier1_split_token_emit_init(
    thread J2kClassicTier1SplitTokenEmitter &emitter,
    device uchar *mq_token_data,
    uint mq_token_capacity,
    device uchar *raw_token_data,
    uint raw_token_capacity,
    device J2kClassicTier1TokenSegment *segments,
    uint segment_capacity
) {
    j2k_classic_symbol_plan_zero(emitter.counters);
    j2k_classic_tier1_token_writer_init(emitter.mq_writer, mq_token_data, mq_token_capacity);
    j2k_classic_tier1_token_writer_init(emitter.raw_writer, raw_token_data, raw_token_capacity);
    emitter.segments = segments;
    emitter.segment_capacity = segment_capacity;
    emitter.current_segment_start_bit = 0u;
    emitter.segment_failed = 0u;
}

inline uint j2k_classic_tier1_split_token_emit_stream_bit_count(
    thread J2kClassicTier1SplitTokenEmitter &emitter,
    bool use_arithmetic
) {
    return use_arithmetic ? emitter.mq_writer.bit_count : emitter.raw_writer.bit_count;
}

inline void j2k_classic_tier1_split_token_emit_push_segment(
    thread J2kClassicTier1SplitTokenEmitter &emitter,
    uint start_pass,
    uint end_pass,
    bool use_arithmetic
) {
    const uint segment_idx = emitter.counters.segment_count;
    emitter.counters.segment_count += 1u;
    if (segment_idx >= emitter.segment_capacity) {
        emitter.segment_failed = 1u;
        return;
    }
    const uint stream_bit_count =
        j2k_classic_tier1_split_token_emit_stream_bit_count(emitter, use_arithmetic);
    const uint token_bit_count = stream_bit_count - emitter.current_segment_start_bit;
    emitter.segments[segment_idx].token_bit_offset = emitter.current_segment_start_bit;
    emitter.segments[segment_idx].token_bit_count = token_bit_count;
    emitter.segments[segment_idx].pass_range =
        (start_pass & 0xFFFFu) | ((end_pass & 0xFFFFu) << 16u);
    emitter.segments[segment_idx].flags = use_arithmetic ? 1u : 0u;
}

inline void j2k_classic_tier1_split_token_emit_store(
    device J2kClassicTier1SymbolPlanCounters *out,
    thread J2kClassicTier1SplitTokenEmitter &emitter
) {
    if (emitter.mq_writer.failed != 0u) {
        emitter.counters.code = J2K_ENCODE_STATUS_FAIL;
        emitter.counters.detail = 23u;
    } else if (emitter.raw_writer.failed != 0u) {
        emitter.counters.code = J2K_ENCODE_STATUS_FAIL;
        emitter.counters.detail = 24u;
    } else if (emitter.segment_failed != 0u) {
        emitter.counters.code = J2K_ENCODE_STATUS_FAIL;
        emitter.counters.detail = 22u;
    }
    j2k_classic_symbol_plan_store(out, emitter.counters);
}

inline void j2k_classic_tier1_split_mq_byte_token_emit_init(
    thread J2kClassicTier1SplitMqByteRawTokenEmitter &emitter,
    device uchar *mq_token_data,
    uint mq_token_capacity,
    device uchar *raw_token_data,
    uint raw_token_capacity,
    device J2kClassicTier1TokenSegment *segments,
    uint segment_capacity
) {
    j2k_classic_symbol_plan_zero(emitter.counters);
    j2k_classic_tier1_token_byte_writer_init(
        emitter.mq_writer,
        mq_token_data,
        mq_token_capacity
    );
    j2k_classic_tier1_token_writer_init(emitter.raw_writer, raw_token_data, raw_token_capacity);
    emitter.segments = segments;
    emitter.segment_capacity = segment_capacity;
    emitter.current_segment_start_bit = 0u;
    emitter.segment_failed = 0u;
}

inline uint j2k_classic_tier1_split_mq_byte_token_emit_stream_bit_count(
    thread J2kClassicTier1SplitMqByteRawTokenEmitter &emitter,
    bool use_arithmetic
) {
    return use_arithmetic ? emitter.mq_writer.len * 8u : emitter.raw_writer.bit_count;
}

inline void j2k_classic_tier1_split_mq_byte_token_emit_push_segment(
    thread J2kClassicTier1SplitMqByteRawTokenEmitter &emitter,
    uint start_pass,
    uint end_pass,
    bool use_arithmetic
) {
    const uint segment_idx = emitter.counters.segment_count;
    emitter.counters.segment_count += 1u;
    if (segment_idx >= emitter.segment_capacity) {
        emitter.segment_failed = 1u;
        return;
    }
    const uint stream_bit_count =
        j2k_classic_tier1_split_mq_byte_token_emit_stream_bit_count(emitter, use_arithmetic);
    const uint token_bit_count = stream_bit_count - emitter.current_segment_start_bit;
    emitter.segments[segment_idx].token_bit_offset = emitter.current_segment_start_bit;
    emitter.segments[segment_idx].token_bit_count = token_bit_count;
    emitter.segments[segment_idx].pass_range =
        (start_pass & 0xFFFFu) | ((end_pass & 0xFFFFu) << 16u);
    emitter.segments[segment_idx].flags = use_arithmetic ? 3u : 0u;
}

inline void j2k_classic_tier1_split_mq_byte_token_emit_store(
    device J2kClassicTier1SymbolPlanCounters *out,
    thread J2kClassicTier1SplitMqByteRawTokenEmitter &emitter
) {
    if (emitter.mq_writer.failed != 0u) {
        emitter.counters.code = J2K_ENCODE_STATUS_FAIL;
        emitter.counters.detail = 25u;
    } else if (emitter.raw_writer.failed != 0u) {
        emitter.counters.code = J2K_ENCODE_STATUS_FAIL;
        emitter.counters.detail = 24u;
    } else if (emitter.segment_failed != 0u) {
        emitter.counters.code = J2K_ENCODE_STATUS_FAIL;
        emitter.counters.detail = 22u;
    }
    j2k_classic_symbol_plan_store(out, emitter.counters);
}

inline void j2k_classic_tier1_token_reader_init(
    thread J2kClassicTier1TokenBitReader &reader,
    device const uchar *data,
    uint capacity_bits
) {
    reader.data = data;
    reader.bit_pos = 0u;
    reader.bit_capacity = capacity_bits;
    reader.failed = 0u;
}

inline void j2k_classic_tier1_token_reader_seek(
    thread J2kClassicTier1TokenBitReader &reader,
    uint bit_pos
) {
    if (bit_pos > reader.bit_capacity) {
        reader.failed = 1u;
        return;
    }
    reader.bit_pos = bit_pos;
}

inline uint j2k_classic_tier1_token_reader_read_bits(
    thread J2kClassicTier1TokenBitReader &reader,
    uint count
) {
    if (count > 32u || reader.bit_pos > reader.bit_capacity ||
        count > (reader.bit_capacity - reader.bit_pos)) {
        reader.failed = 1u;
        return 0u;
    }
    uint value = 0u;
    for (uint idx = 0u; idx < count; ++idx) {
        const uint byte_idx = reader.bit_pos >> 3u;
        const uint shift = 7u - (reader.bit_pos & 7u);
        value = (value << 1u) | ((uint(reader.data[byte_idx]) >> shift) & 1u);
        reader.bit_pos += 1u;
    }
    return value;
}

inline void j2k_pack_classic_tier1_tokens_bypass_u16_32_impl(
    J2kClassicEncodeParams params,
    J2kClassicTier1SymbolPlanCounters counters,
    device const uchar *token_data,
    device const J2kClassicTier1TokenSegment *token_segments,
    uint token_capacity_bytes,
    uint token_segment_capacity,
    device uchar *out,
    device J2kClassicEncodeStatus *status,
    device J2kClassicSegment *segments
) {
    j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 30u, 0u, 0u, 0u, 0u);

    if (params.style_flags != J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS) {
        j2k_set_encode_status(
            status,
            J2K_ENCODE_STATUS_UNSUPPORTED,
            2u,
            0u,
            counters.coding_passes,
            counters.missing_bit_planes,
            0u
        );
        return;
    }
    if (counters.code != J2K_ENCODE_STATUS_OK) {
        j2k_set_encode_status(
            status,
            counters.code,
            counters.detail,
            0u,
            counters.coding_passes,
            counters.missing_bit_planes,
            0u
        );
        return;
    }
    if (counters.segment_count > token_segment_capacity ||
        counters.segment_count > params.segment_capacity) {
        j2k_set_encode_status(
            status,
            J2K_ENCODE_STATUS_FAIL,
            31u,
            0u,
            counters.coding_passes,
            counters.missing_bit_planes,
            0u
        );
        return;
    }

    const uint token_capacity_bits = token_capacity_bytes * 8u;
    thread J2kClassicTier1TokenBitReader reader;
    j2k_classic_tier1_token_reader_init(reader, token_data, token_capacity_bits);
    thread uchar contexts[19];
    reset_contexts(contexts);

    uint data_cursor = 0u;
    uint segment_count = 0u;
    for (uint segment_idx = 0u; segment_idx < counters.segment_count; ++segment_idx) {
        const J2kClassicTier1TokenSegment token_segment = token_segments[segment_idx];
        const uint start_pass = token_segment.pass_range & 0xFFFFu;
        const uint end_pass = token_segment.pass_range >> 16u;
        const bool use_arithmetic = (token_segment.flags & 1u) != 0u;
        if ((token_segment.flags & ~1u) != 0u ||
            start_pass > end_pass ||
            end_pass > counters.coding_passes) {
            j2k_set_encode_status(
                status,
                J2K_ENCODE_STATUS_FAIL,
                32u,
                0u,
                counters.coding_passes,
                counters.missing_bit_planes,
                0u
            );
            return;
        }
        if (token_segment.token_bit_offset > token_capacity_bits ||
            token_segment.token_bit_count > (token_capacity_bits - token_segment.token_bit_offset)) {
            j2k_set_encode_status(
                status,
                J2K_ENCODE_STATUS_FAIL,
                33u,
                0u,
                counters.coding_passes,
                counters.missing_bit_planes,
                0u
            );
            return;
        }
        if (data_cursor > params.output_capacity) {
            j2k_set_encode_status(
                status,
                J2K_ENCODE_STATUS_FAIL,
                34u,
                0u,
                counters.coding_passes,
                counters.missing_bit_planes,
                0u
            );
            return;
        }
        j2k_classic_tier1_token_reader_seek(reader, token_segment.token_bit_offset);
        if (reader.failed != 0u) {
            j2k_set_encode_status(
                status,
                J2K_ENCODE_STATUS_FAIL,
                35u,
                0u,
                counters.coding_passes,
                counters.missing_bit_planes,
                0u
            );
            return;
        }

        uint segment_len = 0u;
        if (use_arithmetic) {
            if ((token_segment.token_bit_count % 6u) != 0u) {
                j2k_set_encode_status(
                    status,
                    J2K_ENCODE_STATUS_FAIL,
                    36u,
                    0u,
                    counters.coding_passes,
                    counters.missing_bit_planes,
                    0u
                );
                return;
            }
            thread J2kMqEncoder encoder;
            j2k_mq_init(encoder, out + data_cursor, params.output_capacity - data_cursor);
            const uint symbol_count = token_segment.token_bit_count / 6u;
            for (uint symbol_idx = 0u; symbol_idx < symbol_count; ++symbol_idx) {
                const uint token = j2k_classic_tier1_token_reader_read_bits(reader, 6u);
                if (reader.failed != 0u) {
                    j2k_set_encode_status(
                        status,
                        J2K_ENCODE_STATUS_FAIL,
                        37u,
                        0u,
                        counters.coding_passes,
                        counters.missing_bit_planes,
                        0u
                    );
                    return;
                }
                const uint ctx_label = token & 0x1Fu;
                if (ctx_label >= 19u) {
                    j2k_set_encode_status(
                        status,
                        J2K_ENCODE_STATUS_FAIL,
                        38u,
                        0u,
                        counters.coding_passes,
                        counters.missing_bit_planes,
                        0u
                    );
                    return;
                }
                j2k_mq_encode(encoder, contexts, ctx_label, (token >> 5u) & 1u);
                if (encoder.failed != 0u) {
                    j2k_set_encode_status(
                        status,
                        J2K_ENCODE_STATUS_FAIL,
                        39u,
                        0u,
                        counters.coding_passes,
                        counters.missing_bit_planes,
                        0u
                    );
                    return;
                }
            }
            segment_len = j2k_classic_finish_arithmetic_segment(encoder);
            if (encoder.failed != 0u) {
                j2k_set_encode_status(
                    status,
                    J2K_ENCODE_STATUS_FAIL,
                    40u,
                    0u,
                    counters.coding_passes,
                    counters.missing_bit_planes,
                    0u
                );
                return;
            }
        } else {
            thread J2kRawBitWriter writer;
            j2k_raw_writer_init(writer, out + data_cursor, params.output_capacity - data_cursor);
            for (uint bit_idx = 0u; bit_idx < token_segment.token_bit_count; ++bit_idx) {
                const uint bit = j2k_classic_tier1_token_reader_read_bits(reader, 1u);
                if (reader.failed != 0u) {
                    j2k_set_encode_status(
                        status,
                        J2K_ENCODE_STATUS_FAIL,
                        41u,
                        0u,
                        counters.coding_passes,
                        counters.missing_bit_planes,
                        0u
                    );
                    return;
                }
                j2k_raw_writer_write_bit(writer, bit);
                if (writer.failed != 0u) {
                    j2k_set_encode_status(
                        status,
                        J2K_ENCODE_STATUS_FAIL,
                        42u,
                        0u,
                        counters.coding_passes,
                        counters.missing_bit_planes,
                        0u
                    );
                    return;
                }
            }
            j2k_raw_writer_finish(writer);
            if (writer.failed != 0u) {
                j2k_set_encode_status(
                    status,
                    J2K_ENCODE_STATUS_FAIL,
                    43u,
                    0u,
                    counters.coding_passes,
                    counters.missing_bit_planes,
                    0u
                );
                return;
            }
            segment_len = writer.len;
        }

        if (segment_len > (params.output_capacity - data_cursor)) {
            j2k_set_encode_status(
                status,
                J2K_ENCODE_STATUS_FAIL,
                44u,
                0u,
                counters.coding_passes,
                counters.missing_bit_planes,
                0u
            );
            return;
        }
        if (!j2k_classic_push_segment(
                segments,
                params.segment_capacity,
                segment_count,
                data_cursor,
                segment_len,
                start_pass,
                end_pass,
                use_arithmetic)) {
            j2k_set_encode_status(
                status,
                J2K_ENCODE_STATUS_FAIL,
                45u,
                0u,
                counters.coding_passes,
                counters.missing_bit_planes,
                0u
            );
            return;
        }
        data_cursor += segment_len;
    }

    j2k_set_encode_status(
        status,
        J2K_ENCODE_STATUS_OK,
        0u,
        data_cursor,
        counters.coding_passes,
        counters.missing_bit_planes,
        segment_count
    );
}

inline void j2k_pack_classic_tier1_split_tokens_bypass_u16_32_impl(
    J2kClassicEncodeParams params,
    J2kClassicTier1SymbolPlanCounters counters,
    device const uchar *mq_token_data,
    device const uchar *raw_token_data,
    device const J2kClassicTier1TokenSegment *token_segments,
    uint mq_token_capacity_bytes,
    uint raw_token_capacity_bytes,
    uint token_segment_capacity,
    device uchar *out,
    device J2kClassicEncodeStatus *status,
    device J2kClassicSegment *segments
) {
    j2k_set_encode_status(status, J2K_ENCODE_STATUS_FAIL, 70u, 0u, 0u, 0u, 0u);

    if (params.style_flags != J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS) {
        j2k_set_encode_status(
            status,
            J2K_ENCODE_STATUS_UNSUPPORTED,
            2u,
            0u,
            counters.coding_passes,
            counters.missing_bit_planes,
            0u
        );
        return;
    }
    if (counters.code != J2K_ENCODE_STATUS_OK) {
        j2k_set_encode_status(
            status,
            counters.code,
            counters.detail,
            0u,
            counters.coding_passes,
            counters.missing_bit_planes,
            0u
        );
        return;
    }
    if (counters.segment_count > token_segment_capacity ||
        counters.segment_count > params.segment_capacity) {
        j2k_set_encode_status(
            status,
            J2K_ENCODE_STATUS_FAIL,
            71u,
            0u,
            counters.coding_passes,
            counters.missing_bit_planes,
            0u
        );
        return;
    }

    const uint mq_token_capacity_bits = mq_token_capacity_bytes * 8u;
    const uint raw_token_capacity_bits = raw_token_capacity_bytes * 8u;
    thread J2kClassicTier1TokenBitReader mq_reader;
    thread J2kClassicTier1TokenBitReader raw_reader;
    j2k_classic_tier1_token_reader_init(mq_reader, mq_token_data, mq_token_capacity_bits);
    j2k_classic_tier1_token_reader_init(raw_reader, raw_token_data, raw_token_capacity_bits);
    thread uchar contexts[19];
    reset_contexts(contexts);

    uint data_cursor = 0u;
    uint segment_count = 0u;
    for (uint segment_idx = 0u; segment_idx < counters.segment_count; ++segment_idx) {
        const J2kClassicTier1TokenSegment token_segment = token_segments[segment_idx];
        const uint start_pass = token_segment.pass_range & 0xFFFFu;
        const uint end_pass = token_segment.pass_range >> 16u;
        const bool use_arithmetic = (token_segment.flags & 1u) != 0u;
        const bool mq_tokens_are_bytes = (token_segment.flags & 2u) != 0u;
        const uint token_capacity_bits =
            use_arithmetic ? mq_token_capacity_bits : raw_token_capacity_bits;
        if ((token_segment.flags & ~3u) != 0u ||
            (!use_arithmetic && mq_tokens_are_bytes) ||
            start_pass > end_pass ||
            end_pass > counters.coding_passes) {
            j2k_set_encode_status(
                status,
                J2K_ENCODE_STATUS_FAIL,
                72u,
                0u,
                counters.coding_passes,
                counters.missing_bit_planes,
                0u
            );
            return;
        }
        if (token_segment.token_bit_offset > token_capacity_bits ||
            token_segment.token_bit_count > (token_capacity_bits - token_segment.token_bit_offset)) {
            j2k_set_encode_status(
                status,
                J2K_ENCODE_STATUS_FAIL,
                73u,
                0u,
                counters.coding_passes,
                counters.missing_bit_planes,
                0u
            );
            return;
        }
        if (data_cursor > params.output_capacity) {
            j2k_set_encode_status(
                status,
                J2K_ENCODE_STATUS_FAIL,
                74u,
                0u,
                counters.coding_passes,
                counters.missing_bit_planes,
                0u
            );
            return;
        }

        uint segment_len = 0u;
        if (use_arithmetic) {
            if (!mq_tokens_are_bytes && (token_segment.token_bit_count % 6u) != 0u) {
                j2k_set_encode_status(
                    status,
                    J2K_ENCODE_STATUS_FAIL,
                    75u,
                    0u,
                    counters.coding_passes,
                    counters.missing_bit_planes,
                    0u
                );
                return;
            }
            if (mq_tokens_are_bytes &&
                ((token_segment.token_bit_offset | token_segment.token_bit_count) & 7u) != 0u) {
                j2k_set_encode_status(
                    status,
                    J2K_ENCODE_STATUS_FAIL,
                    75u,
                    0u,
                    counters.coding_passes,
                    counters.missing_bit_planes,
                    0u
                );
                return;
            }
            if (!mq_tokens_are_bytes) {
                j2k_classic_tier1_token_reader_seek(mq_reader, token_segment.token_bit_offset);
            }
            if (!mq_tokens_are_bytes && mq_reader.failed != 0u) {
                j2k_set_encode_status(
                    status,
                    J2K_ENCODE_STATUS_FAIL,
                    76u,
                    0u,
                    counters.coding_passes,
                    counters.missing_bit_planes,
                    0u
                );
                return;
            }
            thread J2kMqEncoder encoder;
            j2k_mq_init(encoder, out + data_cursor, params.output_capacity - data_cursor);
            const uint symbol_count =
                mq_tokens_are_bytes ? (token_segment.token_bit_count >> 3u) :
                (token_segment.token_bit_count / 6u);
            const uint mq_byte_offset = token_segment.token_bit_offset >> 3u;
            for (uint symbol_idx = 0u; symbol_idx < symbol_count; ++symbol_idx) {
                const uint token = mq_tokens_are_bytes
                    ? uint(mq_token_data[mq_byte_offset + symbol_idx])
                    : j2k_classic_tier1_token_reader_read_bits(mq_reader, 6u);
                if (!mq_tokens_are_bytes && mq_reader.failed != 0u) {
                    j2k_set_encode_status(
                        status,
                        J2K_ENCODE_STATUS_FAIL,
                        77u,
                        0u,
                        counters.coding_passes,
                        counters.missing_bit_planes,
                        0u
                    );
                    return;
                }
                const uint ctx_label = token & 0x1Fu;
                if (ctx_label >= 19u) {
                    j2k_set_encode_status(
                        status,
                        J2K_ENCODE_STATUS_FAIL,
                        78u,
                        0u,
                        counters.coding_passes,
                        counters.missing_bit_planes,
                        0u
                    );
                    return;
                }
                j2k_mq_encode(encoder, contexts, ctx_label, (token >> 5u) & 1u);
                if (encoder.failed != 0u) {
                    j2k_set_encode_status(
                        status,
                        J2K_ENCODE_STATUS_FAIL,
                        79u,
                        0u,
                        counters.coding_passes,
                        counters.missing_bit_planes,
                        0u
                    );
                    return;
                }
            }
            segment_len = j2k_classic_finish_arithmetic_segment(encoder);
            if (encoder.failed != 0u) {
                j2k_set_encode_status(
                    status,
                    J2K_ENCODE_STATUS_FAIL,
                    80u,
                    0u,
                    counters.coding_passes,
                    counters.missing_bit_planes,
                    0u
                );
                return;
            }
        } else {
            j2k_classic_tier1_token_reader_seek(raw_reader, token_segment.token_bit_offset);
            if (raw_reader.failed != 0u) {
                j2k_set_encode_status(
                    status,
                    J2K_ENCODE_STATUS_FAIL,
                    81u,
                    0u,
                    counters.coding_passes,
                    counters.missing_bit_planes,
                    0u
                );
                return;
            }
            thread J2kRawBitWriter writer;
            j2k_raw_writer_init(writer, out + data_cursor, params.output_capacity - data_cursor);
            for (uint bit_idx = 0u; bit_idx < token_segment.token_bit_count; ++bit_idx) {
                const uint bit = j2k_classic_tier1_token_reader_read_bits(raw_reader, 1u);
                if (raw_reader.failed != 0u) {
                    j2k_set_encode_status(
                        status,
                        J2K_ENCODE_STATUS_FAIL,
                        82u,
                        0u,
                        counters.coding_passes,
                        counters.missing_bit_planes,
                        0u
                    );
                    return;
                }
                j2k_raw_writer_write_bit(writer, bit);
                if (writer.failed != 0u) {
                    j2k_set_encode_status(
                        status,
                        J2K_ENCODE_STATUS_FAIL,
                        83u,
                        0u,
                        counters.coding_passes,
                        counters.missing_bit_planes,
                        0u
                    );
                    return;
                }
            }
            j2k_raw_writer_finish(writer);
            if (writer.failed != 0u) {
                j2k_set_encode_status(
                    status,
                    J2K_ENCODE_STATUS_FAIL,
                    84u,
                    0u,
                    counters.coding_passes,
                    counters.missing_bit_planes,
                    0u
                );
                return;
            }
            segment_len = writer.len;
        }

        if (segment_len > (params.output_capacity - data_cursor)) {
            j2k_set_encode_status(
                status,
                J2K_ENCODE_STATUS_FAIL,
                85u,
                0u,
                counters.coding_passes,
                counters.missing_bit_planes,
                0u
            );
            return;
        }
        if (!j2k_classic_push_segment(
                segments,
                params.segment_capacity,
                segment_count,
                data_cursor,
                segment_len,
                start_pass,
                end_pass,
                use_arithmetic)) {
            j2k_set_encode_status(
                status,
                J2K_ENCODE_STATUS_FAIL,
                86u,
                0u,
                counters.coding_passes,
                counters.missing_bit_planes,
                0u
            );
            return;
        }
        data_cursor += segment_len;
    }

    j2k_set_encode_status(
        status,
        J2K_ENCODE_STATUS_OK,
        0u,
        data_cursor,
        counters.coding_passes,
        counters.missing_bit_planes,
        segment_count
    );
}

inline void j2k_classic_symbol_plan_record_mq(
    thread J2kClassicTier1SymbolPlanCounters &counters,
    uint ctx_label,
    uint bit,
    uint pass_type
) {
    counters.mq_symbol_count += 1u;
    switch (pass_type) {
        case 0u:
            counters.cleanup_mq_symbol_count += 1u;
            break;
        case 1u:
            counters.sigprop_mq_symbol_count += 1u;
            break;
        default:
            counters.magref_mq_symbol_count += 1u;
            break;
    }
    const uint packed = (ctx_label & 0x1Fu) | ((bit & 1u) << 5u) | ((pass_type & 3u) << 6u);
    counters.mq_symbol_hash = (counters.mq_symbol_hash ^ packed) * 16777619u;
}

inline void j2k_classic_symbol_plan_record_mq(
    thread J2kClassicTier1TokenEmitter &emitter,
    uint ctx_label,
    uint bit,
    uint pass_type
) {
    emitter.counters.mq_symbol_count += 1u;
    switch (pass_type) {
        case 0u:
            emitter.counters.cleanup_mq_symbol_count += 1u;
            break;
        case 1u:
            emitter.counters.sigprop_mq_symbol_count += 1u;
            break;
        default:
            emitter.counters.magref_mq_symbol_count += 1u;
            break;
    }
    const uint packed_hash =
        (ctx_label & 0x1Fu) | ((bit & 1u) << 5u) | ((pass_type & 3u) << 6u);
    emitter.counters.mq_symbol_hash =
        (emitter.counters.mq_symbol_hash ^ packed_hash) * 16777619u;
    const uint packed_token = (ctx_label & 0x1Fu) | ((bit & 1u) << 5u);
    j2k_classic_tier1_token_writer_write_bits(emitter.writer, packed_token, 6u);
}

inline void j2k_classic_symbol_plan_record_mq(
    thread J2kClassicTier1SplitTokenEmitter &emitter,
    uint ctx_label,
    uint bit,
    uint pass_type
) {
    emitter.counters.mq_symbol_count += 1u;
    switch (pass_type) {
        case 0u:
            emitter.counters.cleanup_mq_symbol_count += 1u;
            break;
        case 1u:
            emitter.counters.sigprop_mq_symbol_count += 1u;
            break;
        default:
            emitter.counters.magref_mq_symbol_count += 1u;
            break;
    }
    const uint packed_hash =
        (ctx_label & 0x1Fu) | ((bit & 1u) << 5u) | ((pass_type & 3u) << 6u);
    emitter.counters.mq_symbol_hash =
        (emitter.counters.mq_symbol_hash ^ packed_hash) * 16777619u;
    const uint packed_token = (ctx_label & 0x1Fu) | ((bit & 1u) << 5u);
    j2k_classic_tier1_token_writer_write_bits(emitter.mq_writer, packed_token, 6u);
}

inline void j2k_classic_symbol_plan_record_mq(
    thread J2kClassicTier1SplitMqByteRawTokenEmitter &emitter,
    uint ctx_label,
    uint bit,
    uint pass_type
) {
    emitter.counters.mq_symbol_count += 1u;
    switch (pass_type) {
        case 0u:
            emitter.counters.cleanup_mq_symbol_count += 1u;
            break;
        case 1u:
            emitter.counters.sigprop_mq_symbol_count += 1u;
            break;
        default:
            emitter.counters.magref_mq_symbol_count += 1u;
            break;
    }
    const uint packed_hash =
        (ctx_label & 0x1Fu) | ((bit & 1u) << 5u) | ((pass_type & 3u) << 6u);
    emitter.counters.mq_symbol_hash =
        (emitter.counters.mq_symbol_hash ^ packed_hash) * 16777619u;
    const uint packed_token = (ctx_label & 0x1Fu) | ((bit & 1u) << 5u);
    j2k_classic_tier1_token_byte_writer_push(emitter.mq_writer, packed_token);
}

inline void j2k_classic_pass_plan_record_mq_symbol(
    thread J2kClassicTier1PassPlanEmitter &emitter
) {
    emitter.counters.mq_symbol_count += 1u;
    if (emitter.current_pass >= J2K_CLASSIC_TIER1_PASS_PLAN_CAPACITY) {
        emitter.counters.code = J2K_ENCODE_STATUS_FAIL;
        emitter.counters.detail = 60u;
        return;
    }
    thread uint &pass_count = emitter.counters.mq_symbols_by_pass[emitter.current_pass];
    if (pass_count == 0u) {
        emitter.counters.nonempty_mq_passes += 1u;
    }
    pass_count += 1u;
    emitter.counters.max_mq_symbols_per_pass =
        max(emitter.counters.max_mq_symbols_per_pass, pass_count);
}

inline void j2k_classic_symbol_plan_record_mq(
    thread J2kClassicTier1PassPlanEmitter &emitter,
    uint ctx_label,
    uint bit,
    uint pass_type
) {
    (void)ctx_label;
    (void)bit;
    (void)pass_type;
    j2k_classic_pass_plan_record_mq_symbol(emitter);
}

inline void j2k_classic_symbol_plan_record_raw(
    thread J2kClassicTier1SymbolPlanCounters &counters,
    uint bit,
    uint pass_type
) {
    counters.raw_bit_count += 1u;
    if (pass_type == 1u) {
        counters.raw_sigprop_bit_count += 1u;
    } else {
        counters.raw_magref_bit_count += 1u;
    }
    const uint packed = (bit & 1u) | ((pass_type & 3u) << 1u);
    counters.raw_bit_hash = (counters.raw_bit_hash ^ packed) * 16777619u;
}

inline void j2k_classic_symbol_plan_record_raw(
    thread J2kClassicTier1TokenEmitter &emitter,
    uint bit,
    uint pass_type
) {
    emitter.counters.raw_bit_count += 1u;
    if (pass_type == 1u) {
        emitter.counters.raw_sigprop_bit_count += 1u;
    } else {
        emitter.counters.raw_magref_bit_count += 1u;
    }
    const uint packed = (bit & 1u) | ((pass_type & 3u) << 1u);
    emitter.counters.raw_bit_hash = (emitter.counters.raw_bit_hash ^ packed) * 16777619u;
    j2k_classic_tier1_token_writer_write_bit(emitter.writer, bit);
}

inline void j2k_classic_symbol_plan_record_raw(
    thread J2kClassicTier1SplitTokenEmitter &emitter,
    uint bit,
    uint pass_type
) {
    emitter.counters.raw_bit_count += 1u;
    if (pass_type == 1u) {
        emitter.counters.raw_sigprop_bit_count += 1u;
    } else {
        emitter.counters.raw_magref_bit_count += 1u;
    }
    const uint packed = (bit & 1u) | ((pass_type & 3u) << 1u);
    emitter.counters.raw_bit_hash = (emitter.counters.raw_bit_hash ^ packed) * 16777619u;
    j2k_classic_tier1_token_writer_write_bit(emitter.raw_writer, bit);
}

inline void j2k_classic_symbol_plan_record_raw(
    thread J2kClassicTier1SplitMqByteRawTokenEmitter &emitter,
    uint bit,
    uint pass_type
) {
    emitter.counters.raw_bit_count += 1u;
    if (pass_type == 1u) {
        emitter.counters.raw_sigprop_bit_count += 1u;
    } else {
        emitter.counters.raw_magref_bit_count += 1u;
    }
    const uint packed = (bit & 1u) | ((pass_type & 3u) << 1u);
    emitter.counters.raw_bit_hash = (emitter.counters.raw_bit_hash ^ packed) * 16777619u;
    j2k_classic_tier1_token_writer_write_bit(emitter.raw_writer, bit);
}

inline void j2k_classic_symbol_plan_record_raw(
    thread J2kClassicTier1PassPlanEmitter &emitter,
    uint bit,
    uint pass_type
) {
    (void)bit;
    (void)pass_type;
    emitter.counters.raw_bit_count += 1u;
    if (emitter.current_pass >= J2K_CLASSIC_TIER1_PASS_PLAN_CAPACITY) {
        emitter.counters.code = J2K_ENCODE_STATUS_FAIL;
        emitter.counters.detail = 60u;
        return;
    }
    thread uint &pass_count = emitter.counters.raw_bits_by_pass[emitter.current_pass];
    if (pass_count == 0u) {
        emitter.counters.nonempty_raw_passes += 1u;
    }
    pass_count += 1u;
    emitter.counters.max_raw_bits_per_pass =
        max(emitter.counters.max_raw_bits_per_pass, pass_count);
}

inline void j2k_classic_symbol_plan_record_sign_count(
    thread J2kClassicTier1SymbolPlanCounters &counters,
    uint pass_type
) {
    if (pass_type == 0u) {
        counters.cleanup_sign_symbol_count += 1u;
    } else {
        counters.sigprop_sign_symbol_count += 1u;
    }
}

inline void j2k_classic_symbol_plan_record_sign_count(
    thread J2kClassicTier1TokenEmitter &emitter,
    uint pass_type
) {
    if (pass_type == 0u) {
        emitter.counters.cleanup_sign_symbol_count += 1u;
    } else {
        emitter.counters.sigprop_sign_symbol_count += 1u;
    }
}

inline void j2k_classic_symbol_plan_record_sign_count(
    thread J2kClassicTier1SplitTokenEmitter &emitter,
    uint pass_type
) {
    if (pass_type == 0u) {
        emitter.counters.cleanup_sign_symbol_count += 1u;
    } else {
        emitter.counters.sigprop_sign_symbol_count += 1u;
    }
}

inline void j2k_classic_symbol_plan_record_sign_count(
    thread J2kClassicTier1SplitMqByteRawTokenEmitter &emitter,
    uint pass_type
) {
    if (pass_type == 0u) {
        emitter.counters.cleanup_sign_symbol_count += 1u;
    } else {
        emitter.counters.sigprop_sign_symbol_count += 1u;
    }
}

inline void j2k_classic_symbol_plan_record_sign_count(
    thread J2kClassicTier1PassPlanEmitter &emitter,
    uint pass_type
) {
    (void)emitter;
    (void)pass_type;
}

template<typename Recorder>
inline void j2k_classic_symbol_plan_record_sign(
    uint idx,
    thread const uchar *states,
    thread Recorder &counters,
    uint padded_width,
    uint index_x,
    uint index_y,
    uint height,
    uint style_flags,
    uchar neighbor_sig,
    uint pass_type
) {
    const uchar significances = neighbor_sig & uchar(0b01010101);
    const uint left_sign = coeff_sign(states, coeff_index(padded_width, index_x - 1u, index_y));
    const uint right_sign = coeff_sign(states, coeff_index(padded_width, index_x + 1u, index_y));
    const uint top_sign = coeff_sign(states, coeff_index(padded_width, index_x, index_y - 1u));
    const uint bottom_sign =
        ((style_flags & J2K_CLASSIC_STYLE_VERTICALLY_CAUSAL_CONTEXT) != 0u &&
            neighbor_in_next_stripe(index_y, height))
        ? 0u
        : coeff_sign(states, coeff_index(padded_width, index_x, index_y + 1u));
    const uchar signs = uchar((top_sign << 6u) | (left_sign << 4u) | (right_sign << 2u) | bottom_sign);
    const uchar negative = significances & signs;
    const uchar positive = significances & uchar(~signs);
    const uchar2 sign_ctx = SIGN_CONTEXT_LOOKUP[uchar((negative << 1u) | positive)];
    const uint sign_bit = (uint(states[idx]) >> 5u) & 1u;
    j2k_classic_symbol_plan_record_sign_count(counters, pass_type);
    j2k_classic_symbol_plan_record_mq(
        counters,
        uint(sign_ctx.x),
        sign_bit ^ uint(sign_ctx.y),
        pass_type
    );
}

inline void j2k_classic_profile_significance_density(
    thread const ushort *magnitudes,
    thread uchar *states,
    uchar coded_marker,
    uint width,
    uint height,
    uint padded_width,
    uint bit_mask,
    uint style_flags,
    bool use_arithmetic,
    thread J2kClassicTier1DensityCounters &counters
) {
    for (uint y_base = 0u; y_base < height; y_base += 4u) {
        for (uint x = 0u; x < width; ++x) {
            const uint y_end = min(y_base + 4u, height);
            for (uint y = y_base; y < y_end; ++y) {
                const uint ix = x + 1u;
                const uint iy = y + 1u;
                const uint idx = coeff_index(padded_width, ix, iy);
                const uchar neighbor_sig =
                    j2k_classic_effective_neighbors(states, padded_width, ix, iy, height, style_flags);
                if ((states[idx] & J2K_ENCODE_SIGNIFICANT) == 0u && neighbor_sig != 0u) {
                    counters.sigprop_active_candidates += 1u;
                    if (use_arithmetic) {
                        counters.arithmetic_sigprop_active_candidates += 1u;
                    } else {
                        counters.raw_sigprop_active_candidates += 1u;
                    }
                    j2k_classic_set_coded_marker(states, idx, coded_marker);
                    if ((magnitudes[idx] & ushort(bit_mask)) != 0u) {
                        counters.sigprop_new_significant += 1u;
                        if (use_arithmetic) {
                            counters.arithmetic_sigprop_new_significant += 1u;
                        } else {
                            counters.raw_sigprop_new_significant += 1u;
                        }
                        set_significant(states, padded_width, ix, iy);
                    }
                }
            }
        }
    }
}

inline void j2k_classic_profile_magnitude_density(
    thread uchar *states,
    uchar coded_marker,
    uint width,
    uint height,
    uint padded_width,
    bool use_arithmetic,
    thread J2kClassicTier1DensityCounters &counters
) {
    for (uint y_base = 0u; y_base < height; y_base += 4u) {
        for (uint x = 0u; x < width; ++x) {
            const uint y_end = min(y_base + 4u, height);
            for (uint y = y_base; y < y_end; ++y) {
                const uint idx = coeff_index(padded_width, x + 1u, y + 1u);
                if ((states[idx] & J2K_ENCODE_SIGNIFICANT) != 0u &&
                    !j2k_classic_coded_marker_matches(states, idx, coded_marker)) {
                    counters.magref_active_candidates += 1u;
                    if (use_arithmetic) {
                        counters.arithmetic_magref_active_candidates += 1u;
                    } else {
                        counters.raw_magref_active_candidates += 1u;
                    }
                    states[idx] |= J2K_ENCODE_MAGNITUDE_REFINED;
                }
            }
        }
    }
}

inline void j2k_classic_profile_cleanup_density(
    thread const ushort *magnitudes,
    thread uchar *states,
    uchar coded_marker,
    uint width,
    uint height,
    uint padded_width,
    uint bit_mask,
    uint style_flags,
    thread J2kClassicTier1DensityCounters &counters
) {
    for (uint y_base = 0u; y_base < height; y_base += 4u) {
        for (uint x = 0u; x < width; ++x) {
            const uint y_end = min(y_base + 4u, height);
            const uint stripe_height = y_end - y_base;

            if (stripe_height == 4u) {
                bool all_zero_uncoded = true;
                for (uint y = y_base; y < y_end; ++y) {
                    const uint ix = x + 1u;
                    const uint iy = y + 1u;
                    const uint idx = coeff_index(padded_width, ix, iy);
                    const uchar neighbor_sig =
                        j2k_classic_effective_neighbors(states, padded_width, ix, iy, height, style_flags);
                    if ((states[idx] & J2K_ENCODE_SIGNIFICANT) != 0u ||
                        j2k_classic_coded_marker_matches(states, idx, coded_marker) ||
                        neighbor_sig != 0u) {
                        all_zero_uncoded = false;
                        break;
                    }
                }

                if (all_zero_uncoded) {
                    counters.cleanup_rlc_stripes += 1u;
                    uint first_sig = 4u;
                    for (uint pos = 0u; pos < 4u; ++pos) {
                        const uint idx = coeff_index(padded_width, x + 1u, y_base + pos + 1u);
                        if ((magnitudes[idx] & ushort(bit_mask)) != 0u) {
                            first_sig = pos;
                            break;
                        }
                    }

                    if (first_sig < 4u) {
                        counters.cleanup_new_significant += 1u;
                        const uint sig_y = y_base + first_sig;
                        set_significant(states, padded_width, x + 1u, sig_y + 1u);

                        for (uint y = sig_y + 1u; y < y_end; ++y) {
                            const uint ix = x + 1u;
                            const uint iy = y + 1u;
                            const uint idx = coeff_index(padded_width, ix, iy);
                            if ((states[idx] & J2K_ENCODE_SIGNIFICANT) == 0u &&
                                !j2k_classic_coded_marker_matches(states, idx, coded_marker)) {
                                counters.cleanup_active_candidates += 1u;
                                if ((magnitudes[idx] & ushort(bit_mask)) != 0u) {
                                    counters.cleanup_new_significant += 1u;
                                    set_significant(states, padded_width, ix, iy);
                                }
                            }
                        }
                        continue;
                    }

                    counters.cleanup_rlc_zero_stripes += 1u;
                    continue;
                }
            }

            for (uint y = y_base; y < y_end; ++y) {
                const uint ix = x + 1u;
                const uint iy = y + 1u;
                const uint idx = coeff_index(padded_width, ix, iy);
                if ((states[idx] & J2K_ENCODE_SIGNIFICANT) == 0u &&
                    !j2k_classic_coded_marker_matches(states, idx, coded_marker)) {
                    counters.cleanup_active_candidates += 1u;
                    if ((magnitudes[idx] & ushort(bit_mask)) != 0u) {
                        counters.cleanup_new_significant += 1u;
                        set_significant(states, padded_width, ix, iy);
                    }
                }
            }
        }
    }
}

template<typename Recorder>
inline void j2k_classic_symbol_plan_significance_pass(
    thread const ushort *magnitudes,
    thread uchar *states,
    uchar coded_marker,
    uint width,
    uint height,
    uint padded_width,
    uint bit_mask,
    uint sub_band_type,
    uint style_flags,
    thread Recorder &counters
) {
    for (uint y_base = 0u; y_base < height; y_base += 4u) {
        for (uint x = 0u; x < width; ++x) {
            const uint y_end = min(y_base + 4u, height);
            for (uint y = y_base; y < y_end; ++y) {
                const uint ix = x + 1u;
                const uint iy = y + 1u;
                const uint idx = coeff_index(padded_width, ix, iy);
                const uchar neighbor_sig =
                    j2k_classic_effective_neighbors(states, padded_width, ix, iy, height, style_flags);
                if ((states[idx] & J2K_ENCODE_SIGNIFICANT) == 0u && neighbor_sig != 0u) {
                    const uint ctx_label = uint(zero_context_label(neighbor_sig, sub_band_type));
                    const uint bit = (magnitudes[idx] & ushort(bit_mask)) != 0u ? 1u : 0u;
                    j2k_classic_symbol_plan_record_mq(counters, ctx_label, bit, 1u);
                    j2k_classic_set_coded_marker(states, idx, coded_marker);
                    if (bit != 0u) {
                        j2k_classic_symbol_plan_record_sign(
                            idx,
                            states,
                            counters,
                            padded_width,
                            ix,
                            iy,
                            height,
                            style_flags,
                            neighbor_sig,
                            1u
                        );
                        set_significant(states, padded_width, ix, iy);
                    }
                }
            }
        }
    }
}

template<typename Recorder>
inline void j2k_classic_symbol_plan_magnitude_pass(
    thread const ushort *magnitudes,
    thread uchar *states,
    uchar coded_marker,
    uint width,
    uint height,
    uint padded_width,
    uint bit_mask,
    uint style_flags,
    thread Recorder &counters
) {
    for (uint y_base = 0u; y_base < height; y_base += 4u) {
        for (uint x = 0u; x < width; ++x) {
            const uint y_end = min(y_base + 4u, height);
            for (uint y = y_base; y < y_end; ++y) {
                const uint ix = x + 1u;
                const uint iy = y + 1u;
                const uint idx = coeff_index(padded_width, ix, iy);
                if ((states[idx] & J2K_ENCODE_SIGNIFICANT) != 0u &&
                    !j2k_classic_coded_marker_matches(states, idx, coded_marker)) {
                    const uint ctx_label =
                        uint(magnitude_refinement_context(states, padded_width, ix, iy, height, style_flags));
                    const uint bit = (magnitudes[idx] & ushort(bit_mask)) != 0u ? 1u : 0u;
                    j2k_classic_symbol_plan_record_mq(counters, ctx_label, bit, 2u);
                    states[idx] |= J2K_ENCODE_MAGNITUDE_REFINED;
                }
            }
        }
    }
}

template<typename Recorder>
inline void j2k_classic_symbol_plan_significance_pass_raw(
    thread const ushort *magnitudes,
    thread uchar *states,
    uchar coded_marker,
    uint width,
    uint height,
    uint padded_width,
    uint bit_mask,
    uint style_flags,
    thread Recorder &counters
) {
    for (uint y_base = 0u; y_base < height; y_base += 4u) {
        for (uint x = 0u; x < width; ++x) {
            const uint y_end = min(y_base + 4u, height);
            for (uint y = y_base; y < y_end; ++y) {
                const uint ix = x + 1u;
                const uint iy = y + 1u;
                const uint idx = coeff_index(padded_width, ix, iy);
                const uchar neighbor_sig =
                    j2k_classic_effective_neighbors(states, padded_width, ix, iy, height, style_flags);
                if ((states[idx] & J2K_ENCODE_SIGNIFICANT) == 0u && neighbor_sig != 0u) {
                    const uint bit = (magnitudes[idx] & ushort(bit_mask)) != 0u ? 1u : 0u;
                    j2k_classic_symbol_plan_record_raw(counters, bit, 1u);
                    j2k_classic_set_coded_marker(states, idx, coded_marker);
                    if (bit != 0u) {
                        j2k_classic_symbol_plan_record_raw(counters, (uint(states[idx]) >> 5u) & 1u, 1u);
                        set_significant(states, padded_width, ix, iy);
                    }
                }
            }
        }
    }
}

template<typename Recorder>
inline void j2k_classic_symbol_plan_magnitude_pass_raw(
    thread const ushort *magnitudes,
    thread uchar *states,
    uchar coded_marker,
    uint width,
    uint height,
    uint padded_width,
    uint bit_mask,
    thread Recorder &counters
) {
    for (uint y_base = 0u; y_base < height; y_base += 4u) {
        for (uint x = 0u; x < width; ++x) {
            const uint y_end = min(y_base + 4u, height);
            for (uint y = y_base; y < y_end; ++y) {
                const uint ix = x + 1u;
                const uint iy = y + 1u;
                const uint idx = coeff_index(padded_width, ix, iy);
                if ((states[idx] & J2K_ENCODE_SIGNIFICANT) != 0u &&
                    !j2k_classic_coded_marker_matches(states, idx, coded_marker)) {
                    const uint bit = (magnitudes[idx] & ushort(bit_mask)) != 0u ? 1u : 0u;
                    j2k_classic_symbol_plan_record_raw(counters, bit, 2u);
                    states[idx] |= J2K_ENCODE_MAGNITUDE_REFINED;
                }
            }
        }
    }
}

template<typename Recorder>
inline void j2k_classic_symbol_plan_cleanup_pass(
    thread const ushort *magnitudes,
    thread uchar *states,
    uchar coded_marker,
    uint width,
    uint height,
    uint padded_width,
    uint bit_mask,
    uint sub_band_type,
    uint style_flags,
    thread Recorder &counters
) {
    for (uint y_base = 0u; y_base < height; y_base += 4u) {
        for (uint x = 0u; x < width; ++x) {
            const uint y_end = min(y_base + 4u, height);
            const uint stripe_height = y_end - y_base;

            if (stripe_height == 4u) {
                bool all_zero_uncoded = true;
                for (uint y = y_base; y < y_end; ++y) {
                    const uint ix = x + 1u;
                    const uint iy = y + 1u;
                    const uint idx = coeff_index(padded_width, ix, iy);
                    const uchar neighbor_sig =
                        j2k_classic_effective_neighbors(states, padded_width, ix, iy, height, style_flags);
                    if ((states[idx] & J2K_ENCODE_SIGNIFICANT) != 0u ||
                        j2k_classic_coded_marker_matches(states, idx, coded_marker) ||
                        neighbor_sig != 0u) {
                        all_zero_uncoded = false;
                        break;
                    }
                }

                if (all_zero_uncoded) {
                    uint first_sig = 4u;
                    for (uint pos = 0u; pos < 4u; ++pos) {
                        const uint idx = coeff_index(padded_width, x + 1u, y_base + pos + 1u);
                        if ((magnitudes[idx] & ushort(bit_mask)) != 0u) {
                            first_sig = pos;
                            break;
                        }
                    }

                    if (first_sig < 4u) {
                        j2k_classic_symbol_plan_record_mq(counters, 17u, 1u, 0u);
                        j2k_classic_symbol_plan_record_mq(counters, 18u, (first_sig >> 1u) & 1u, 0u);
                        j2k_classic_symbol_plan_record_mq(counters, 18u, first_sig & 1u, 0u);

                        const uint sig_y = y_base + first_sig;
                        const uint sig_idx = coeff_index(padded_width, x + 1u, sig_y + 1u);
                        j2k_classic_symbol_plan_record_sign(
                            sig_idx,
                            states,
                            counters,
                            padded_width,
                            x + 1u,
                            sig_y + 1u,
                            height,
                            style_flags,
                            uchar(0u),
                            0u
                        );
                        set_significant(states, padded_width, x + 1u, sig_y + 1u);

                        for (uint y = sig_y + 1u; y < y_end; ++y) {
                            const uint ix = x + 1u;
                            const uint iy = y + 1u;
                            const uint idx = coeff_index(padded_width, ix, iy);
                            if ((states[idx] & J2K_ENCODE_SIGNIFICANT) == 0u &&
                                !j2k_classic_coded_marker_matches(states, idx, coded_marker)) {
                                const uchar neighbor_sig =
                                    j2k_classic_effective_neighbors(
                                        states,
                                        padded_width,
                                        ix,
                                        iy,
                                        height,
                                        style_flags
                                    );
                                const uint ctx_label = uint(zero_context_label(neighbor_sig, sub_band_type));
                                const uint bit = (magnitudes[idx] & ushort(bit_mask)) != 0u ? 1u : 0u;
                                j2k_classic_symbol_plan_record_mq(counters, ctx_label, bit, 0u);
                                if (bit != 0u) {
                                    j2k_classic_symbol_plan_record_sign(
                                        idx,
                                        states,
                                        counters,
                                        padded_width,
                                        ix,
                                        iy,
                                        height,
                                        style_flags,
                                        neighbor_sig,
                                        0u
                                    );
                                    set_significant(states, padded_width, ix, iy);
                                }
                            }
                        }
                        continue;
                    }

                    j2k_classic_symbol_plan_record_mq(counters, 17u, 0u, 0u);
                    continue;
                }
            }

            for (uint y = y_base; y < y_end; ++y) {
                const uint ix = x + 1u;
                const uint iy = y + 1u;
                const uint idx = coeff_index(padded_width, ix, iy);
                if ((states[idx] & J2K_ENCODE_SIGNIFICANT) == 0u &&
                    !j2k_classic_coded_marker_matches(states, idx, coded_marker)) {
                    const uchar neighbor_sig =
                        j2k_classic_effective_neighbors(states, padded_width, ix, iy, height, style_flags);
                    const uint ctx_label = uint(zero_context_label(neighbor_sig, sub_band_type));
                    const uint bit = (magnitudes[idx] & ushort(bit_mask)) != 0u ? 1u : 0u;
                    j2k_classic_symbol_plan_record_mq(counters, ctx_label, bit, 0u);
                    if (bit != 0u) {
                        j2k_classic_symbol_plan_record_sign(
                            idx,
                            states,
                            counters,
                            padded_width,
                            ix,
                            iy,
                            height,
                            style_flags,
                            neighbor_sig,
                            0u
                        );
                        set_significant(states, padded_width, ix, iy);
                    }
                }
            }
        }
    }
}

inline void j2k_profile_classic_tier1_density_bypass_u16_32_impl(
    device const int *coefficients,
    J2kClassicEncodeParams params,
    device J2kClassicTier1DensityCounters *out
) {
    thread J2kClassicTier1DensityCounters counters;
    j2k_classic_tier1_density_zero(counters);

    if (params.width == 0u || params.height == 0u ||
        params.width > J2K_CLASSIC_ENCODE_32_MAX_WIDTH ||
        params.height > J2K_CLASSIC_ENCODE_32_MAX_HEIGHT ||
        params.total_bitplanes > 16u) {
        j2k_classic_tier1_density_store(out, counters);
        return;
    }

    thread ushort magnitudes[J2K_CLASSIC_ENCODE_32_MAX_COEFF_COUNT];
    thread uchar states[J2K_CLASSIC_ENCODE_32_MAX_COEFF_COUNT];
    const uint padded_width = params.width + 2u;
    j2k_classic_clear_state_border(states, padded_width, params.width, params.height);

    uint max_magnitude = 0u;
    for (uint y = 0u; y < params.height; ++y) {
        for (uint x = 0u; x < params.width; ++x) {
            const uint src_idx = y * params.width + x;
            const int value = coefficients[src_idx];
            const uint dst_idx = coeff_index(padded_width, x + 1u, y + 1u);
            const uint magnitude = j2k_classic_magnitude(value);
            magnitudes[dst_idx] = ushort(magnitude);
            states[dst_idx] = value < 0 ? J2K_ENCODE_SIGN : uchar(0u);
            max_magnitude = max(max_magnitude, magnitude);
        }
    }

    if (max_magnitude == 0u) {
        j2k_classic_tier1_density_store(out, counters);
        return;
    }

    const uint num_bitplanes = 32u - clz(max_magnitude);
    if (num_bitplanes > params.total_bitplanes) {
        j2k_classic_tier1_density_store(out, counters);
        return;
    }

    uchar coded_marker = uchar(1u);
    const uint total_passes = 1u + 3u * (num_bitplanes - 1u);
    for (uint coding_pass = 0u; coding_pass < total_passes; ++coding_pass) {
        const uint current_bitplane = (coding_pass + 2u) / 3u;
        const uint bit_mask = 1u << (num_bitplanes - 1u - current_bitplane);
        const bool use_arithmetic =
            (params.style_flags & J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS) == 0u ||
            coding_pass <= 9u ||
            (coding_pass % 3u) == 0u;
        switch (coding_pass % 3u) {
            case 0u:
                j2k_classic_profile_cleanup_density(
                    magnitudes,
                    states,
                    coded_marker,
                    params.width,
                    params.height,
                    padded_width,
                    bit_mask,
                    params.style_flags,
                    counters
                );
                coded_marker += uchar(1u);
                break;
            case 1u:
                j2k_classic_profile_significance_density(
                    magnitudes,
                    states,
                    coded_marker,
                    params.width,
                    params.height,
                    padded_width,
                    bit_mask,
                    params.style_flags,
                    use_arithmetic,
                    counters
                );
                break;
            default:
                j2k_classic_profile_magnitude_density(
                    states,
                    coded_marker,
                    params.width,
                    params.height,
                    padded_width,
                    use_arithmetic,
                    counters
                );
                break;
        }
    }

    j2k_classic_tier1_density_store(out, counters);
}

inline void j2k_plan_classic_tier1_symbols_bypass_u16_32_impl(
    device const int *coefficients,
    J2kClassicEncodeParams params,
    device J2kClassicTier1SymbolPlanCounters *out
) {
    thread J2kClassicTier1SymbolPlanCounters counters;
    j2k_classic_symbol_plan_zero(counters);

    if (params.width == 0u || params.height == 0u ||
        params.width > J2K_CLASSIC_ENCODE_32_MAX_WIDTH ||
        params.height > J2K_CLASSIC_ENCODE_32_MAX_HEIGHT ||
        params.total_bitplanes > 16u) {
        counters.code = J2K_ENCODE_STATUS_UNSUPPORTED;
        counters.detail = 1u;
        j2k_classic_symbol_plan_store(out, counters);
        return;
    }
    if (params.style_flags != J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS) {
        counters.code = J2K_ENCODE_STATUS_UNSUPPORTED;
        counters.detail = 2u;
        j2k_classic_symbol_plan_store(out, counters);
        return;
    }

    thread ushort magnitudes[J2K_CLASSIC_ENCODE_32_MAX_COEFF_COUNT];
    thread uchar states[J2K_CLASSIC_ENCODE_32_MAX_COEFF_COUNT];
    const uint padded_width = params.width + 2u;
    j2k_classic_clear_state_border(states, padded_width, params.width, params.height);

    uint max_magnitude = 0u;
    for (uint y = 0u; y < params.height; ++y) {
        for (uint x = 0u; x < params.width; ++x) {
            const uint src_idx = y * params.width + x;
            const int value = coefficients[src_idx];
            const uint dst_idx = coeff_index(padded_width, x + 1u, y + 1u);
            const uint magnitude = j2k_classic_magnitude(value);
            magnitudes[dst_idx] = ushort(magnitude);
            states[dst_idx] = value < 0 ? J2K_ENCODE_SIGN : uchar(0u);
            max_magnitude = max(max_magnitude, magnitude);
        }
    }

    if (max_magnitude == 0u) {
        counters.missing_bit_planes = params.total_bitplanes;
        j2k_classic_symbol_plan_store(out, counters);
        return;
    }

    const uint num_bitplanes = 32u - clz(max_magnitude);
    if (num_bitplanes > params.total_bitplanes) {
        counters.code = J2K_ENCODE_STATUS_FAIL;
        counters.detail = 3u;
        j2k_classic_symbol_plan_store(out, counters);
        return;
    }
    counters.missing_bit_planes = params.total_bitplanes - num_bitplanes;

    uchar coded_marker = uchar(1u);
    const uint total_passes = 1u + 3u * (num_bitplanes - 1u);
    counters.coding_passes = total_passes;
    uint current_segment_idx = 0xFFFFFFFFu;
    for (uint coding_pass = 0u; coding_pass < total_passes; ++coding_pass) {
        const uint segment_idx = j2k_classic_bypass_segment_idx(coding_pass);
        if (current_segment_idx != segment_idx) {
            current_segment_idx = segment_idx;
            counters.segment_count += 1u;
        }

        const uint pass_type = coding_pass % 3u;
        const bool use_arithmetic = coding_pass <= 9u || pass_type == 0u;
        const uint current_bitplane = (coding_pass + 2u) / 3u;
        const uint bit_mask = 1u << (num_bitplanes - 1u - current_bitplane);
        switch (pass_type) {
            case 0u:
                j2k_classic_symbol_plan_cleanup_pass(
                    magnitudes,
                    states,
                    coded_marker,
                    params.width,
                    params.height,
                    padded_width,
                    bit_mask,
                    params.sub_band_type,
                    0u,
                    counters
                );
                coded_marker += uchar(1u);
                break;
            case 1u:
                if (use_arithmetic) {
                    j2k_classic_symbol_plan_significance_pass(
                        magnitudes,
                        states,
                        coded_marker,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        params.sub_band_type,
                        0u,
                        counters
                    );
                } else {
                    j2k_classic_symbol_plan_significance_pass_raw(
                        magnitudes,
                        states,
                        coded_marker,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        0u,
                        counters
                    );
                }
                break;
            default:
                if (use_arithmetic) {
                    j2k_classic_symbol_plan_magnitude_pass(
                        magnitudes,
                        states,
                        coded_marker,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        0u,
                        counters
                    );
                } else {
                    j2k_classic_symbol_plan_magnitude_pass_raw(
                        magnitudes,
                        states,
                        coded_marker,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        counters
                    );
                }
                break;
        }
    }

    j2k_classic_symbol_plan_store(out, counters);
}

inline void j2k_plan_classic_tier1_passes_bypass_u16_32_impl(
    device const int *coefficients,
    J2kClassicEncodeParams params,
    device J2kClassicTier1PassPlanCounters *out
) {
    thread J2kClassicTier1PassPlanEmitter emitter;
    j2k_classic_pass_plan_zero(emitter.counters);
    emitter.current_pass = 0u;

    if (params.width == 0u || params.height == 0u ||
        params.width > J2K_CLASSIC_ENCODE_32_MAX_WIDTH ||
        params.height > J2K_CLASSIC_ENCODE_32_MAX_HEIGHT ||
        params.total_bitplanes > 16u) {
        emitter.counters.code = J2K_ENCODE_STATUS_UNSUPPORTED;
        emitter.counters.detail = 1u;
        j2k_classic_pass_plan_store(out, emitter.counters);
        return;
    }
    if (params.style_flags != J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS) {
        emitter.counters.code = J2K_ENCODE_STATUS_UNSUPPORTED;
        emitter.counters.detail = 2u;
        j2k_classic_pass_plan_store(out, emitter.counters);
        return;
    }

    thread ushort magnitudes[J2K_CLASSIC_ENCODE_32_MAX_COEFF_COUNT];
    thread uchar states[J2K_CLASSIC_ENCODE_32_MAX_COEFF_COUNT];
    const uint padded_width = params.width + 2u;
    j2k_classic_clear_state_border(states, padded_width, params.width, params.height);

    uint max_magnitude = 0u;
    for (uint y = 0u; y < params.height; ++y) {
        for (uint x = 0u; x < params.width; ++x) {
            const uint src_idx = y * params.width + x;
            const int value = coefficients[src_idx];
            const uint dst_idx = coeff_index(padded_width, x + 1u, y + 1u);
            const uint magnitude = j2k_classic_magnitude(value);
            magnitudes[dst_idx] = ushort(magnitude);
            states[dst_idx] = value < 0 ? J2K_ENCODE_SIGN : uchar(0u);
            max_magnitude = max(max_magnitude, magnitude);
        }
    }

    if (max_magnitude == 0u) {
        emitter.counters.missing_bit_planes = params.total_bitplanes;
        j2k_classic_pass_plan_store(out, emitter.counters);
        return;
    }

    const uint num_bitplanes = 32u - clz(max_magnitude);
    if (num_bitplanes > params.total_bitplanes) {
        emitter.counters.code = J2K_ENCODE_STATUS_FAIL;
        emitter.counters.detail = 3u;
        j2k_classic_pass_plan_store(out, emitter.counters);
        return;
    }
    emitter.counters.missing_bit_planes = params.total_bitplanes - num_bitplanes;

    uchar coded_marker = uchar(1u);
    const uint total_passes = 1u + 3u * (num_bitplanes - 1u);
    if (total_passes > J2K_CLASSIC_TIER1_PASS_PLAN_CAPACITY) {
        emitter.counters.code = J2K_ENCODE_STATUS_FAIL;
        emitter.counters.detail = 61u;
        j2k_classic_pass_plan_store(out, emitter.counters);
        return;
    }
    emitter.counters.coding_passes = total_passes;
    uint current_segment_idx = 0xFFFFFFFFu;
    for (uint coding_pass = 0u; coding_pass < total_passes; ++coding_pass) {
        emitter.current_pass = coding_pass;
        const uint segment_idx = j2k_classic_bypass_segment_idx(coding_pass);
        if (current_segment_idx != segment_idx) {
            current_segment_idx = segment_idx;
            emitter.counters.segment_count += 1u;
        }

        const uint pass_type = coding_pass % 3u;
        const bool use_arithmetic = coding_pass <= 9u || pass_type == 0u;
        const uint current_bitplane = (coding_pass + 2u) / 3u;
        const uint bit_mask = 1u << (num_bitplanes - 1u - current_bitplane);
        switch (pass_type) {
            case 0u:
                j2k_classic_symbol_plan_cleanup_pass(
                    magnitudes,
                    states,
                    coded_marker,
                    params.width,
                    params.height,
                    padded_width,
                    bit_mask,
                    params.sub_band_type,
                    0u,
                    emitter
                );
                coded_marker += uchar(1u);
                break;
            case 1u:
                if (use_arithmetic) {
                    j2k_classic_symbol_plan_significance_pass(
                        magnitudes,
                        states,
                        coded_marker,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        params.sub_band_type,
                        0u,
                        emitter
                    );
                } else {
                    j2k_classic_symbol_plan_significance_pass_raw(
                        magnitudes,
                        states,
                        coded_marker,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        0u,
                        emitter
                    );
                }
                break;
            default:
                if (use_arithmetic) {
                    j2k_classic_symbol_plan_magnitude_pass(
                        magnitudes,
                        states,
                        coded_marker,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        0u,
                        emitter
                    );
                } else {
                    j2k_classic_symbol_plan_magnitude_pass_raw(
                        magnitudes,
                        states,
                        coded_marker,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        emitter
                    );
                }
                break;
        }
    }

    j2k_classic_pass_plan_store(out, emitter.counters);
}

inline void j2k_emit_classic_tier1_tokens_bypass_u16_32_impl(
    device const int *coefficients,
    J2kClassicEncodeParams params,
    device J2kClassicTier1SymbolPlanCounters *out,
    device uchar *token_data,
    device J2kClassicTier1TokenSegment *token_segments,
    uint token_capacity,
    uint token_segment_capacity
) {
    thread J2kClassicTier1TokenEmitter emitter;
    j2k_classic_tier1_token_emit_init(
        emitter,
        token_data,
        token_capacity,
        token_segments,
        token_segment_capacity
    );

    if (params.width == 0u || params.height == 0u ||
        params.width > J2K_CLASSIC_ENCODE_32_MAX_WIDTH ||
        params.height > J2K_CLASSIC_ENCODE_32_MAX_HEIGHT ||
        params.total_bitplanes > 16u) {
        emitter.counters.code = J2K_ENCODE_STATUS_UNSUPPORTED;
        emitter.counters.detail = 1u;
        j2k_classic_tier1_token_emit_store(out, emitter);
        return;
    }
    if (params.style_flags != J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS) {
        emitter.counters.code = J2K_ENCODE_STATUS_UNSUPPORTED;
        emitter.counters.detail = 2u;
        j2k_classic_tier1_token_emit_store(out, emitter);
        return;
    }

    thread ushort magnitudes[J2K_CLASSIC_ENCODE_32_MAX_COEFF_COUNT];
    thread uchar states[J2K_CLASSIC_ENCODE_32_MAX_COEFF_COUNT];
    const uint padded_width = params.width + 2u;
    j2k_classic_clear_state_border(states, padded_width, params.width, params.height);

    uint max_magnitude = 0u;
    for (uint y = 0u; y < params.height; ++y) {
        for (uint x = 0u; x < params.width; ++x) {
            const uint src_idx = y * params.width + x;
            const int value = coefficients[src_idx];
            const uint dst_idx = coeff_index(padded_width, x + 1u, y + 1u);
            const uint magnitude = j2k_classic_magnitude(value);
            magnitudes[dst_idx] = ushort(magnitude);
            states[dst_idx] = value < 0 ? J2K_ENCODE_SIGN : uchar(0u);
            max_magnitude = max(max_magnitude, magnitude);
        }
    }

    if (max_magnitude == 0u) {
        emitter.counters.missing_bit_planes = params.total_bitplanes;
        j2k_classic_tier1_token_emit_store(out, emitter);
        return;
    }

    const uint num_bitplanes = 32u - clz(max_magnitude);
    if (num_bitplanes > params.total_bitplanes) {
        emitter.counters.code = J2K_ENCODE_STATUS_FAIL;
        emitter.counters.detail = 3u;
        j2k_classic_tier1_token_emit_store(out, emitter);
        return;
    }
    emitter.counters.missing_bit_planes = params.total_bitplanes - num_bitplanes;

    uchar coded_marker = uchar(1u);
    const uint total_passes = 1u + 3u * (num_bitplanes - 1u);
    emitter.counters.coding_passes = total_passes;
    uint current_segment_idx = 0xFFFFFFFFu;
    uint current_segment_start_pass = 0u;
    bool current_use_arithmetic = true;
    bool have_segment = false;
    for (uint coding_pass = 0u; coding_pass < total_passes; ++coding_pass) {
        const uint pass_type = coding_pass % 3u;
        const uint segment_idx = j2k_classic_bypass_segment_idx(coding_pass);
        const bool use_arithmetic = coding_pass <= 9u || pass_type == 0u;
        if (!have_segment || current_segment_idx != segment_idx) {
            if (have_segment) {
                j2k_classic_tier1_token_emit_push_segment(
                    emitter,
                    current_segment_start_pass,
                    coding_pass,
                    current_use_arithmetic
                );
            }
            current_segment_idx = segment_idx;
            current_segment_start_pass = coding_pass;
            current_use_arithmetic = use_arithmetic;
            have_segment = true;
        }

        const uint current_bitplane = (coding_pass + 2u) / 3u;
        const uint bit_mask = 1u << (num_bitplanes - 1u - current_bitplane);
        switch (pass_type) {
            case 0u:
                j2k_classic_symbol_plan_cleanup_pass(
                    magnitudes,
                    states,
                    coded_marker,
                    params.width,
                    params.height,
                    padded_width,
                    bit_mask,
                    params.sub_band_type,
                    0u,
                    emitter
                );
                coded_marker += uchar(1u);
                break;
            case 1u:
                if (use_arithmetic) {
                    j2k_classic_symbol_plan_significance_pass(
                        magnitudes,
                        states,
                        coded_marker,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        params.sub_band_type,
                        0u,
                        emitter
                    );
                } else {
                    j2k_classic_symbol_plan_significance_pass_raw(
                        magnitudes,
                        states,
                        coded_marker,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        0u,
                        emitter
                    );
                }
                break;
            default:
                if (use_arithmetic) {
                    j2k_classic_symbol_plan_magnitude_pass(
                        magnitudes,
                        states,
                        coded_marker,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        0u,
                        emitter
                    );
                } else {
                    j2k_classic_symbol_plan_magnitude_pass_raw(
                        magnitudes,
                        states,
                        coded_marker,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        emitter
                    );
                }
                break;
        }
    }

    if (have_segment) {
        j2k_classic_tier1_token_emit_push_segment(
            emitter,
            current_segment_start_pass,
            total_passes,
            current_use_arithmetic
        );
    }
    j2k_classic_tier1_token_writer_finish(emitter.writer);
    j2k_classic_tier1_token_emit_store(out, emitter);
}

inline void j2k_emit_classic_tier1_split_tokens_bypass_u16_32_impl(
    device const int *coefficients,
    J2kClassicEncodeParams params,
    device J2kClassicTier1SymbolPlanCounters *out,
    device uchar *mq_token_data,
    device uchar *raw_token_data,
    device J2kClassicTier1TokenSegment *token_segments,
    uint mq_token_capacity,
    uint raw_token_capacity,
    uint token_segment_capacity
) {
    thread J2kClassicTier1SplitTokenEmitter emitter;
    j2k_classic_tier1_split_token_emit_init(
        emitter,
        mq_token_data,
        mq_token_capacity,
        raw_token_data,
        raw_token_capacity,
        token_segments,
        token_segment_capacity
    );

    if (params.width == 0u || params.height == 0u ||
        params.width > J2K_CLASSIC_ENCODE_32_MAX_WIDTH ||
        params.height > J2K_CLASSIC_ENCODE_32_MAX_HEIGHT ||
        params.total_bitplanes > 16u) {
        emitter.counters.code = J2K_ENCODE_STATUS_UNSUPPORTED;
        emitter.counters.detail = 1u;
        j2k_classic_tier1_split_token_emit_store(out, emitter);
        return;
    }
    if (params.style_flags != J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS) {
        emitter.counters.code = J2K_ENCODE_STATUS_UNSUPPORTED;
        emitter.counters.detail = 2u;
        j2k_classic_tier1_split_token_emit_store(out, emitter);
        return;
    }

    thread ushort magnitudes[J2K_CLASSIC_ENCODE_32_MAX_COEFF_COUNT];
    thread uchar states[J2K_CLASSIC_ENCODE_32_MAX_COEFF_COUNT];
    const uint padded_width = params.width + 2u;
    j2k_classic_clear_state_border(states, padded_width, params.width, params.height);

    uint max_magnitude = 0u;
    for (uint y = 0u; y < params.height; ++y) {
        for (uint x = 0u; x < params.width; ++x) {
            const uint src_idx = y * params.width + x;
            const int value = coefficients[src_idx];
            const uint dst_idx = coeff_index(padded_width, x + 1u, y + 1u);
            const uint magnitude = j2k_classic_magnitude(value);
            magnitudes[dst_idx] = ushort(magnitude);
            states[dst_idx] = value < 0 ? J2K_ENCODE_SIGN : uchar(0u);
            max_magnitude = max(max_magnitude, magnitude);
        }
    }

    if (max_magnitude == 0u) {
        emitter.counters.missing_bit_planes = params.total_bitplanes;
        j2k_classic_tier1_split_token_emit_store(out, emitter);
        return;
    }

    const uint num_bitplanes = 32u - clz(max_magnitude);
    if (num_bitplanes > params.total_bitplanes) {
        emitter.counters.code = J2K_ENCODE_STATUS_FAIL;
        emitter.counters.detail = 3u;
        j2k_classic_tier1_split_token_emit_store(out, emitter);
        return;
    }
    emitter.counters.missing_bit_planes = params.total_bitplanes - num_bitplanes;

    uchar coded_marker = uchar(1u);
    const uint total_passes = 1u + 3u * (num_bitplanes - 1u);
    emitter.counters.coding_passes = total_passes;
    uint current_segment_idx = 0xFFFFFFFFu;
    uint current_segment_start_pass = 0u;
    bool current_use_arithmetic = true;
    bool have_segment = false;
    for (uint coding_pass = 0u; coding_pass < total_passes; ++coding_pass) {
        const uint pass_type = coding_pass % 3u;
        const uint segment_idx = j2k_classic_bypass_segment_idx(coding_pass);
        const bool use_arithmetic = coding_pass <= 9u || pass_type == 0u;
        if (!have_segment || current_segment_idx != segment_idx) {
            if (have_segment) {
                j2k_classic_tier1_split_token_emit_push_segment(
                    emitter,
                    current_segment_start_pass,
                    coding_pass,
                    current_use_arithmetic
                );
            }
            current_segment_idx = segment_idx;
            current_segment_start_pass = coding_pass;
            current_use_arithmetic = use_arithmetic;
            emitter.current_segment_start_bit =
                j2k_classic_tier1_split_token_emit_stream_bit_count(emitter, use_arithmetic);
            have_segment = true;
        }

        const uint current_bitplane = (coding_pass + 2u) / 3u;
        const uint bit_mask = 1u << (num_bitplanes - 1u - current_bitplane);
        switch (pass_type) {
            case 0u:
                j2k_classic_symbol_plan_cleanup_pass(
                    magnitudes,
                    states,
                    coded_marker,
                    params.width,
                    params.height,
                    padded_width,
                    bit_mask,
                    params.sub_band_type,
                    0u,
                    emitter
                );
                coded_marker += uchar(1u);
                break;
            case 1u:
                if (use_arithmetic) {
                    j2k_classic_symbol_plan_significance_pass(
                        magnitudes,
                        states,
                        coded_marker,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        params.sub_band_type,
                        0u,
                        emitter
                    );
                } else {
                    j2k_classic_symbol_plan_significance_pass_raw(
                        magnitudes,
                        states,
                        coded_marker,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        0u,
                        emitter
                    );
                }
                break;
            default:
                if (use_arithmetic) {
                    j2k_classic_symbol_plan_magnitude_pass(
                        magnitudes,
                        states,
                        coded_marker,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        0u,
                        emitter
                    );
                } else {
                    j2k_classic_symbol_plan_magnitude_pass_raw(
                        magnitudes,
                        states,
                        coded_marker,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        emitter
                    );
                }
                break;
        }
    }

    if (have_segment) {
        j2k_classic_tier1_split_token_emit_push_segment(
            emitter,
            current_segment_start_pass,
            total_passes,
            current_use_arithmetic
        );
    }
    j2k_classic_tier1_token_writer_finish(emitter.mq_writer);
    j2k_classic_tier1_token_writer_finish(emitter.raw_writer);
    j2k_classic_tier1_split_token_emit_store(out, emitter);
}

inline void j2k_emit_classic_tier1_split_mq_byte_raw_tokens_bypass_u16_32_impl(
    device const int *coefficients,
    J2kClassicEncodeParams params,
    device J2kClassicTier1SymbolPlanCounters *out,
    device uchar *mq_token_data,
    device uchar *raw_token_data,
    device J2kClassicTier1TokenSegment *token_segments,
    uint mq_token_capacity,
    uint raw_token_capacity,
    uint token_segment_capacity
) {
    thread J2kClassicTier1SplitMqByteRawTokenEmitter emitter;
    j2k_classic_tier1_split_mq_byte_token_emit_init(
        emitter,
        mq_token_data,
        mq_token_capacity,
        raw_token_data,
        raw_token_capacity,
        token_segments,
        token_segment_capacity
    );

    if (params.width == 0u || params.height == 0u ||
        params.width > J2K_CLASSIC_ENCODE_32_MAX_WIDTH ||
        params.height > J2K_CLASSIC_ENCODE_32_MAX_HEIGHT ||
        params.total_bitplanes > 16u) {
        emitter.counters.code = J2K_ENCODE_STATUS_UNSUPPORTED;
        emitter.counters.detail = 1u;
        j2k_classic_tier1_split_mq_byte_token_emit_store(out, emitter);
        return;
    }
    if (params.style_flags != J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS) {
        emitter.counters.code = J2K_ENCODE_STATUS_UNSUPPORTED;
        emitter.counters.detail = 2u;
        j2k_classic_tier1_split_mq_byte_token_emit_store(out, emitter);
        return;
    }

    thread ushort magnitudes[J2K_CLASSIC_ENCODE_32_MAX_COEFF_COUNT];
    thread uchar states[J2K_CLASSIC_ENCODE_32_MAX_COEFF_COUNT];
    const uint padded_width = params.width + 2u;
    j2k_classic_clear_state_border(states, padded_width, params.width, params.height);

    uint max_magnitude = 0u;
    for (uint y = 0u; y < params.height; ++y) {
        for (uint x = 0u; x < params.width; ++x) {
            const uint src_idx = y * params.width + x;
            const int value = coefficients[src_idx];
            const uint dst_idx = coeff_index(padded_width, x + 1u, y + 1u);
            const uint magnitude = j2k_classic_magnitude(value);
            magnitudes[dst_idx] = ushort(magnitude);
            states[dst_idx] = value < 0 ? J2K_ENCODE_SIGN : uchar(0u);
            max_magnitude = max(max_magnitude, magnitude);
        }
    }

    if (max_magnitude == 0u) {
        emitter.counters.missing_bit_planes = params.total_bitplanes;
        j2k_classic_tier1_split_mq_byte_token_emit_store(out, emitter);
        return;
    }

    const uint num_bitplanes = 32u - clz(max_magnitude);
    if (num_bitplanes > params.total_bitplanes) {
        emitter.counters.code = J2K_ENCODE_STATUS_FAIL;
        emitter.counters.detail = 3u;
        j2k_classic_tier1_split_mq_byte_token_emit_store(out, emitter);
        return;
    }
    emitter.counters.missing_bit_planes = params.total_bitplanes - num_bitplanes;

    uchar coded_marker = uchar(1u);
    const uint total_passes = 1u + 3u * (num_bitplanes - 1u);
    emitter.counters.coding_passes = total_passes;
    uint current_segment_idx = 0xFFFFFFFFu;
    uint current_segment_start_pass = 0u;
    bool current_use_arithmetic = true;
    bool have_segment = false;
    for (uint coding_pass = 0u; coding_pass < total_passes; ++coding_pass) {
        const uint pass_type = coding_pass % 3u;
        const uint segment_idx = j2k_classic_bypass_segment_idx(coding_pass);
        const bool use_arithmetic = coding_pass <= 9u || pass_type == 0u;
        if (!have_segment || current_segment_idx != segment_idx) {
            if (have_segment) {
                j2k_classic_tier1_split_mq_byte_token_emit_push_segment(
                    emitter,
                    current_segment_start_pass,
                    coding_pass,
                    current_use_arithmetic
                );
            }
            current_segment_idx = segment_idx;
            current_segment_start_pass = coding_pass;
            current_use_arithmetic = use_arithmetic;
            emitter.current_segment_start_bit =
                j2k_classic_tier1_split_mq_byte_token_emit_stream_bit_count(
                    emitter,
                    use_arithmetic
                );
            have_segment = true;
        }

        const uint current_bitplane = (coding_pass + 2u) / 3u;
        const uint bit_mask = 1u << (num_bitplanes - 1u - current_bitplane);
        switch (pass_type) {
            case 0u:
                j2k_classic_symbol_plan_cleanup_pass(
                    magnitudes,
                    states,
                    coded_marker,
                    params.width,
                    params.height,
                    padded_width,
                    bit_mask,
                    params.sub_band_type,
                    0u,
                    emitter
                );
                coded_marker += uchar(1u);
                break;
            case 1u:
                if (use_arithmetic) {
                    j2k_classic_symbol_plan_significance_pass(
                        magnitudes,
                        states,
                        coded_marker,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        params.sub_band_type,
                        0u,
                        emitter
                    );
                } else {
                    j2k_classic_symbol_plan_significance_pass_raw(
                        magnitudes,
                        states,
                        coded_marker,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        0u,
                        emitter
                    );
                }
                break;
            default:
                if (use_arithmetic) {
                    j2k_classic_symbol_plan_magnitude_pass(
                        magnitudes,
                        states,
                        coded_marker,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        0u,
                        emitter
                    );
                } else {
                    j2k_classic_symbol_plan_magnitude_pass_raw(
                        magnitudes,
                        states,
                        coded_marker,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        emitter
                    );
                }
                break;
        }
    }

    if (have_segment) {
        j2k_classic_tier1_split_mq_byte_token_emit_push_segment(
            emitter,
            current_segment_start_pass,
            total_passes,
            current_use_arithmetic
        );
    }
    j2k_classic_tier1_token_writer_finish(emitter.raw_writer);
    j2k_classic_tier1_split_mq_byte_token_emit_store(out, emitter);
}

inline void j2k_profile_classic_tier1_raw_pack_bypass_u16_32_impl(
    device const int *coefficients,
    device uchar *out,
    J2kClassicEncodeParams params
) {
    if (params.width == 0u || params.height == 0u ||
        params.width > J2K_CLASSIC_ENCODE_32_MAX_WIDTH ||
        params.height > J2K_CLASSIC_ENCODE_32_MAX_HEIGHT ||
        params.total_bitplanes > 16u) {
        return;
    }

    thread ushort magnitudes[J2K_CLASSIC_ENCODE_32_MAX_COEFF_COUNT];
    thread uchar states[J2K_CLASSIC_ENCODE_32_MAX_COEFF_COUNT];
    const uint padded_width = params.width + 2u;
    j2k_classic_clear_state_border(states, padded_width, params.width, params.height);

    uint max_magnitude = 0u;
    for (uint y = 0u; y < params.height; ++y) {
        for (uint x = 0u; x < params.width; ++x) {
            const uint src_idx = y * params.width + x;
            const int value = coefficients[src_idx];
            const uint dst_idx = coeff_index(padded_width, x + 1u, y + 1u);
            const uint magnitude = j2k_classic_magnitude(value);
            magnitudes[dst_idx] = ushort(magnitude);
            states[dst_idx] = value < 0 ? J2K_ENCODE_SIGN : uchar(0u);
            max_magnitude = max(max_magnitude, magnitude);
        }
    }

    if (max_magnitude == 0u) {
        return;
    }

    const uint num_bitplanes = 32u - clz(max_magnitude);
    if (num_bitplanes > params.total_bitplanes) {
        return;
    }

    thread J2kClassicTier1DensityCounters counters;
    j2k_classic_tier1_density_zero(counters);

    uchar coded_marker = uchar(1u);
    const uint total_passes = 1u + 3u * (num_bitplanes - 1u);
    uint data_cursor = 0u;
    uint current_raw_segment_idx = 0xFFFFFFFFu;
    bool have_raw_segment = false;
    thread J2kRawBitWriter raw_writer;

    for (uint coding_pass = 0u; coding_pass < total_passes; ++coding_pass) {
        const uint current_bitplane = (coding_pass + 2u) / 3u;
        const uint bit_mask = 1u << (num_bitplanes - 1u - current_bitplane);
        const uint pass_type = coding_pass % 3u;
        const bool use_arithmetic =
            (params.style_flags & J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS) == 0u ||
            coding_pass <= 9u ||
            pass_type == 0u;

        if (!use_arithmetic) {
            const uint segment_idx =
                (params.style_flags & J2K_CLASSIC_STYLE_TERMINATION_ON_EACH_PASS) != 0u
                    ? coding_pass
                    : j2k_classic_bypass_segment_idx(coding_pass);
            if (!have_raw_segment || current_raw_segment_idx != segment_idx) {
                if (have_raw_segment) {
                    j2k_raw_writer_finish(raw_writer);
                    if (raw_writer.failed != 0u) {
                        return;
                    }
                    data_cursor += raw_writer.len;
                }
                if (data_cursor > params.output_capacity) {
                    return;
                }
                current_raw_segment_idx = segment_idx;
                have_raw_segment = true;
                j2k_raw_writer_init(raw_writer, out + data_cursor, params.output_capacity - data_cursor);
            }
        }

        switch (pass_type) {
            case 0u:
                j2k_classic_profile_cleanup_density(
                    magnitudes,
                    states,
                    coded_marker,
                    params.width,
                    params.height,
                    padded_width,
                    bit_mask,
                    params.style_flags,
                    counters
                );
                coded_marker += uchar(1u);
                break;
            case 1u:
                if (use_arithmetic) {
                    j2k_classic_profile_significance_density(
                        magnitudes,
                        states,
                        coded_marker,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        params.style_flags,
                        true,
                        counters
                    );
                } else {
                    j2k_classic_significance_pass_raw(
                        magnitudes,
                        states,
                        coded_marker,
                        raw_writer,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        params.style_flags
                    );
                }
                break;
            default:
                if (use_arithmetic) {
                    j2k_classic_profile_magnitude_density(
                        states,
                        coded_marker,
                        params.width,
                        params.height,
                        padded_width,
                        true,
                        counters
                    );
                } else {
                    j2k_classic_magnitude_refinement_pass_raw(
                        magnitudes,
                        states,
                        coded_marker,
                        raw_writer,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask
                    );
                }
                break;
        }
    }

    if (have_raw_segment) {
        j2k_raw_writer_finish(raw_writer);
    }
}

inline void j2k_profile_classic_tier1_arithmetic_pack_bypass_u16_32_impl(
    device const int *coefficients,
    device uchar *out,
    J2kClassicEncodeParams params
) {
    if (params.width == 0u || params.height == 0u ||
        params.width > J2K_CLASSIC_ENCODE_32_MAX_WIDTH ||
        params.height > J2K_CLASSIC_ENCODE_32_MAX_HEIGHT ||
        params.total_bitplanes > 16u ||
        params.style_flags != J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS) {
        return;
    }

    thread ushort magnitudes[J2K_CLASSIC_ENCODE_32_MAX_COEFF_COUNT];
    thread uchar states[J2K_CLASSIC_ENCODE_32_MAX_COEFF_COUNT];
    const uint padded_width = params.width + 2u;
    j2k_classic_clear_state_border(states, padded_width, params.width, params.height);

    uint max_magnitude = 0u;
    for (uint y = 0u; y < params.height; ++y) {
        for (uint x = 0u; x < params.width; ++x) {
            const uint src_idx = y * params.width + x;
            const int value = coefficients[src_idx];
            const uint dst_idx = coeff_index(padded_width, x + 1u, y + 1u);
            const uint magnitude = j2k_classic_magnitude(value);
            magnitudes[dst_idx] = ushort(magnitude);
            states[dst_idx] = value < 0 ? J2K_ENCODE_SIGN : uchar(0u);
            max_magnitude = max(max_magnitude, magnitude);
        }
    }

    if (max_magnitude == 0u) {
        return;
    }

    const uint num_bitplanes = 32u - clz(max_magnitude);
    if (num_bitplanes > params.total_bitplanes) {
        return;
    }

    thread uchar contexts[19];
    reset_contexts(contexts);
    thread J2kMqEncoder arithmetic_encoder;
    thread J2kClassicTier1DensityCounters counters;
    j2k_classic_tier1_density_zero(counters);

    uchar coded_marker = uchar(1u);
    const uint total_passes = 1u + 3u * (num_bitplanes - 1u);
    uint data_cursor = 0u;
    uint current_arithmetic_segment_idx = 0xFFFFFFFFu;
    bool have_arithmetic_segment = false;

    for (uint coding_pass = 0u; coding_pass < total_passes; ++coding_pass) {
        const uint current_bitplane = (coding_pass + 2u) / 3u;
        const uint bit_mask = 1u << (num_bitplanes - 1u - current_bitplane);
        const uint pass_type = coding_pass % 3u;
        const bool use_arithmetic = coding_pass <= 9u || pass_type == 0u;

        if (use_arithmetic) {
            const uint segment_idx = j2k_classic_bypass_segment_idx(coding_pass);
            if (!have_arithmetic_segment || current_arithmetic_segment_idx != segment_idx) {
                if (have_arithmetic_segment) {
                    const uint segment_len = j2k_classic_finish_arithmetic_segment(arithmetic_encoder);
                    if (arithmetic_encoder.failed != 0u) {
                        return;
                    }
                    data_cursor += segment_len;
                }
                if (data_cursor > params.output_capacity) {
                    return;
                }
                current_arithmetic_segment_idx = segment_idx;
                have_arithmetic_segment = true;
                j2k_mq_init(arithmetic_encoder, out + data_cursor, params.output_capacity - data_cursor);
            }
        }

        switch (pass_type) {
            case 0u:
                j2k_classic_cleanup_pass(
                    magnitudes,
                    states,
                    coded_marker,
                    arithmetic_encoder,
                    contexts,
                    params.width,
                    params.height,
                    padded_width,
                    bit_mask,
                    params.sub_band_type,
                    0u
                );
                coded_marker += uchar(1u);
                break;
            case 1u:
                if (use_arithmetic) {
                    j2k_classic_significance_pass(
                        magnitudes,
                        states,
                        coded_marker,
                        arithmetic_encoder,
                        contexts,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        params.sub_band_type,
                        0u
                    );
                } else {
                    j2k_classic_profile_significance_density(
                        magnitudes,
                        states,
                        coded_marker,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        0u,
                        false,
                        counters
                    );
                }
                break;
            default:
                if (use_arithmetic) {
                    j2k_classic_magnitude_refinement_pass(
                        magnitudes,
                        states,
                        coded_marker,
                        arithmetic_encoder,
                        contexts,
                        params.width,
                        params.height,
                        padded_width,
                        bit_mask,
                        0u
                    );
                } else {
                    j2k_classic_profile_magnitude_density(
                        states,
                        coded_marker,
                        params.width,
                        params.height,
                        padded_width,
                        false,
                        counters
                    );
                }
                break;
        }

        if (use_arithmetic && arithmetic_encoder.failed != 0u) {
            return;
        }
    }

    if (have_arithmetic_segment) {
        j2k_classic_finish_arithmetic_segment(arithmetic_encoder);
    }
}

kernel void j2k_encode_classic_code_block(
    device const int *coefficients [[buffer(0)]],
    device uchar *out [[buffer(1)]],
    constant J2kClassicEncodeParams &params [[buffer(2)]],
    device J2kClassicEncodeStatus *status [[buffer(3)]],
    device J2kClassicSegment *segments [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) {
        return;
    }
    j2k_encode_classic_code_block_impl(coefficients, out, params, status, segments);
}

kernel void j2k_encode_classic_code_blocks(
    device const int *coefficients [[buffer(0)]],
    device uchar *out [[buffer(1)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(2)]],
    device J2kClassicEncodeStatus *statuses [[buffer(3)]],
    device J2kClassicSegment *segments [[buffer(4)]],
    constant uint &job_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= job_count) {
        return;
    }
    const J2kClassicEncodeBatchJob job = jobs[gid];
    J2kClassicEncodeParams params;
    params.width = job.width;
    params.height = job.height;
    params.sub_band_type = job.sub_band_type;
    params.total_bitplanes = job.total_bitplanes;
    params.style_flags = job.style_flags;
    params.output_capacity = job.output_capacity;
    params.segment_capacity = job.segment_capacity;
    j2k_encode_classic_code_block_impl(
        coefficients + job.coefficient_offset,
        out + job.output_offset,
        params,
        statuses + gid,
        segments + job.segment_offset
    );
}

kernel void j2k_encode_classic_code_blocks_style0(
    device const int *coefficients [[buffer(0)]],
    device uchar *out [[buffer(1)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(2)]],
    device J2kClassicEncodeStatus *statuses [[buffer(3)]],
    device J2kClassicSegment *segments [[buffer(4)]],
    constant uint &job_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= job_count) {
        return;
    }
    const J2kClassicEncodeBatchJob job = jobs[gid];
    J2kClassicEncodeParams params;
    params.width = job.width;
    params.height = job.height;
    params.sub_band_type = job.sub_band_type;
    params.total_bitplanes = job.total_bitplanes;
    params.style_flags = 0u;
    params.output_capacity = job.output_capacity;
    params.segment_capacity = job.segment_capacity;
    j2k_encode_classic_code_block_impl_style0(
        coefficients + job.coefficient_offset,
        out + job.output_offset,
        params,
        statuses + gid,
        segments + job.segment_offset
    );
}

kernel void j2k_encode_classic_code_blocks_32(
    device const int *coefficients [[buffer(0)]],
    device uchar *out [[buffer(1)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(2)]],
    device J2kClassicEncodeStatus *statuses [[buffer(3)]],
    device J2kClassicSegment *segments [[buffer(4)]],
    constant uint &job_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= job_count) {
        return;
    }
    const J2kClassicEncodeBatchJob job = jobs[gid];
    J2kClassicEncodeParams params;
    params.width = job.width;
    params.height = job.height;
    params.sub_band_type = job.sub_band_type;
    params.total_bitplanes = job.total_bitplanes;
    params.style_flags = job.style_flags;
    params.output_capacity = job.output_capacity;
    params.segment_capacity = job.segment_capacity;
    j2k_encode_classic_code_block_impl_32(
        coefficients + job.coefficient_offset,
        out + job.output_offset,
        params,
        statuses + gid,
        segments + job.segment_offset
    );
}

kernel void j2k_encode_classic_code_blocks_bypass_32(
    device const int *coefficients [[buffer(0)]],
    device uchar *out [[buffer(1)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(2)]],
    device J2kClassicEncodeStatus *statuses [[buffer(3)]],
    device J2kClassicSegment *segments [[buffer(4)]],
    constant uint &job_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= job_count) {
        return;
    }
    const J2kClassicEncodeBatchJob job = jobs[gid];
    J2kClassicEncodeParams params;
    params.width = job.width;
    params.height = job.height;
    params.sub_band_type = job.sub_band_type;
    params.total_bitplanes = job.total_bitplanes;
    params.style_flags = J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS;
    params.output_capacity = job.output_capacity;
    params.segment_capacity = job.segment_capacity;
    j2k_encode_classic_code_block_impl_bypass_32(
        coefficients + job.coefficient_offset,
        out + job.output_offset,
        params,
        statuses + gid,
        segments + job.segment_offset
    );
}

kernel void j2k_encode_classic_code_blocks_bypass_u16_32(
    device const int *coefficients [[buffer(0)]],
    device uchar *out [[buffer(1)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(2)]],
    device J2kClassicEncodeStatus *statuses [[buffer(3)]],
    device J2kClassicSegment *segments [[buffer(4)]],
    constant uint &job_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= job_count) {
        return;
    }
    const J2kClassicEncodeBatchJob job = jobs[gid];
    J2kClassicEncodeParams params;
    params.width = job.width;
    params.height = job.height;
    params.sub_band_type = job.sub_band_type;
    params.total_bitplanes = job.total_bitplanes;
    params.style_flags = J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS;
    params.output_capacity = job.output_capacity;
    params.segment_capacity = job.segment_capacity;
    j2k_encode_classic_code_block_impl_bypass_u16_32(
        coefficients + job.coefficient_offset,
        out + job.output_offset,
        params,
        statuses + gid,
        segments + job.segment_offset
    );
}

kernel void j2k_profile_classic_tier1_density_bypass_u16_32(
    device const int *coefficients [[buffer(0)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(1)]],
    device J2kClassicTier1DensityCounters *counters [[buffer(2)]],
    constant uint &job_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= job_count) {
        return;
    }
    const J2kClassicEncodeBatchJob job = jobs[gid];
    J2kClassicEncodeParams params;
    params.width = job.width;
    params.height = job.height;
    params.sub_band_type = job.sub_band_type;
    params.total_bitplanes = job.total_bitplanes;
    params.style_flags = job.style_flags;
    params.output_capacity = job.output_capacity;
    params.segment_capacity = job.segment_capacity;
    j2k_profile_classic_tier1_density_bypass_u16_32_impl(
        coefficients + job.coefficient_offset,
        params,
        counters + gid
    );
}

kernel void j2k_plan_classic_tier1_symbols_bypass_u16_32(
    device const int *coefficients [[buffer(0)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(1)]],
    device J2kClassicTier1SymbolPlanCounters *counters [[buffer(2)]],
    constant uint &job_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= job_count) {
        return;
    }
    const J2kClassicEncodeBatchJob job = jobs[gid];
    J2kClassicEncodeParams params;
    params.width = job.width;
    params.height = job.height;
    params.sub_band_type = job.sub_band_type;
    params.total_bitplanes = job.total_bitplanes;
    params.style_flags = job.style_flags;
    params.output_capacity = job.output_capacity;
    params.segment_capacity = job.segment_capacity;
    j2k_plan_classic_tier1_symbols_bypass_u16_32_impl(
        coefficients + job.coefficient_offset,
        params,
        counters + gid
    );
}

kernel void j2k_plan_classic_tier1_passes_bypass_u16_32(
    device const int *coefficients [[buffer(0)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(1)]],
    device J2kClassicTier1PassPlanCounters *counters [[buffer(2)]],
    constant uint &job_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= job_count) {
        return;
    }
    const J2kClassicEncodeBatchJob job = jobs[gid];
    J2kClassicEncodeParams params;
    params.width = job.width;
    params.height = job.height;
    params.sub_band_type = job.sub_band_type;
    params.total_bitplanes = job.total_bitplanes;
    params.style_flags = job.style_flags;
    params.output_capacity = job.output_capacity;
    params.segment_capacity = job.segment_capacity;
    j2k_plan_classic_tier1_passes_bypass_u16_32_impl(
        coefficients + job.coefficient_offset,
        params,
        counters + gid
    );
}

kernel void j2k_emit_classic_tier1_tokens_bypass_u16_32(
    device const int *coefficients [[buffer(0)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(1)]],
    device J2kClassicTier1SymbolPlanCounters *counters [[buffer(2)]],
    device uchar *token_data [[buffer(3)]],
    device J2kClassicTier1TokenSegment *token_segments [[buffer(4)]],
    constant uint &token_stride_bytes [[buffer(5)]],
    constant uint &token_segment_stride [[buffer(6)]],
    constant uint &job_count [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= job_count) {
        return;
    }
    const J2kClassicEncodeBatchJob job = jobs[gid];
    J2kClassicEncodeParams params;
    params.width = job.width;
    params.height = job.height;
    params.sub_band_type = job.sub_band_type;
    params.total_bitplanes = job.total_bitplanes;
    params.style_flags = job.style_flags;
    params.output_capacity = job.output_capacity;
    params.segment_capacity = job.segment_capacity;
    j2k_emit_classic_tier1_tokens_bypass_u16_32_impl(
        coefficients + job.coefficient_offset,
        params,
        counters + gid,
        token_data + (gid * token_stride_bytes),
        token_segments + (gid * token_segment_stride),
        token_stride_bytes,
        token_segment_stride
    );
}

kernel void j2k_emit_classic_tier1_split_tokens_bypass_u16_32(
    device const int *coefficients [[buffer(0)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(1)]],
    device J2kClassicTier1SymbolPlanCounters *counters [[buffer(2)]],
    device uchar *mq_token_data [[buffer(3)]],
    device uchar *raw_token_data [[buffer(4)]],
    device J2kClassicTier1TokenSegment *token_segments [[buffer(5)]],
    constant uint &mq_token_stride_bytes [[buffer(6)]],
    constant uint &raw_token_stride_bytes [[buffer(7)]],
    constant uint &token_segment_stride [[buffer(8)]],
    constant uint &job_count [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= job_count) {
        return;
    }
    const J2kClassicEncodeBatchJob job = jobs[gid];
    J2kClassicEncodeParams params;
    params.width = job.width;
    params.height = job.height;
    params.sub_band_type = job.sub_band_type;
    params.total_bitplanes = job.total_bitplanes;
    params.style_flags = job.style_flags;
    params.output_capacity = job.output_capacity;
    params.segment_capacity = job.segment_capacity;
    j2k_emit_classic_tier1_split_tokens_bypass_u16_32_impl(
        coefficients + job.coefficient_offset,
        params,
        counters + gid,
        mq_token_data + (gid * mq_token_stride_bytes),
        raw_token_data + (gid * raw_token_stride_bytes),
        token_segments + (gid * token_segment_stride),
        mq_token_stride_bytes,
        raw_token_stride_bytes,
        token_segment_stride
    );
}

kernel void j2k_emit_classic_tier1_split_mq_byte_raw_tokens_bypass_u16_32(
    device const int *coefficients [[buffer(0)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(1)]],
    device J2kClassicTier1SymbolPlanCounters *counters [[buffer(2)]],
    device uchar *mq_token_data [[buffer(3)]],
    device uchar *raw_token_data [[buffer(4)]],
    device J2kClassicTier1TokenSegment *token_segments [[buffer(5)]],
    constant uint &mq_token_stride_bytes [[buffer(6)]],
    constant uint &raw_token_stride_bytes [[buffer(7)]],
    constant uint &token_segment_stride [[buffer(8)]],
    constant uint &job_count [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= job_count) {
        return;
    }
    const J2kClassicEncodeBatchJob job = jobs[gid];
    J2kClassicEncodeParams params;
    params.width = job.width;
    params.height = job.height;
    params.sub_band_type = job.sub_band_type;
    params.total_bitplanes = job.total_bitplanes;
    params.style_flags = job.style_flags;
    params.output_capacity = job.output_capacity;
    params.segment_capacity = job.segment_capacity;
    j2k_emit_classic_tier1_split_mq_byte_raw_tokens_bypass_u16_32_impl(
        coefficients + job.coefficient_offset,
        params,
        counters + gid,
        mq_token_data + (gid * mq_token_stride_bytes),
        raw_token_data + (gid * raw_token_stride_bytes),
        token_segments + (gid * token_segment_stride),
        mq_token_stride_bytes,
        raw_token_stride_bytes,
        token_segment_stride
    );
}

kernel void j2k_pack_classic_tier1_tokens_bypass_u16_32(
    device const J2kClassicEncodeBatchJob *jobs [[buffer(0)]],
    device const J2kClassicTier1SymbolPlanCounters *counters [[buffer(1)]],
    device const uchar *token_data [[buffer(2)]],
    device const J2kClassicTier1TokenSegment *token_segments [[buffer(3)]],
    device uchar *out [[buffer(4)]],
    device J2kClassicEncodeStatus *statuses [[buffer(5)]],
    device J2kClassicSegment *segments [[buffer(6)]],
    constant uint &token_stride_bytes [[buffer(7)]],
    constant uint &token_segment_stride [[buffer(8)]],
    constant uint &job_count [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= job_count) {
        return;
    }
    const J2kClassicEncodeBatchJob job = jobs[gid];
    J2kClassicEncodeParams params;
    params.width = job.width;
    params.height = job.height;
    params.sub_band_type = job.sub_band_type;
    params.total_bitplanes = job.total_bitplanes;
    params.style_flags = job.style_flags;
    params.output_capacity = job.output_capacity;
    params.segment_capacity = job.segment_capacity;
    j2k_pack_classic_tier1_tokens_bypass_u16_32_impl(
        params,
        counters[gid],
        token_data + (gid * token_stride_bytes),
        token_segments + (gid * token_segment_stride),
        token_stride_bytes,
        token_segment_stride,
        out + job.output_offset,
        statuses + gid,
        segments + job.segment_offset
    );
}

kernel void j2k_pack_classic_tier1_split_tokens_bypass_u16_32(
    device const J2kClassicEncodeBatchJob *jobs [[buffer(0)]],
    device const J2kClassicTier1SymbolPlanCounters *counters [[buffer(1)]],
    device const uchar *mq_token_data [[buffer(2)]],
    device const uchar *raw_token_data [[buffer(3)]],
    device const J2kClassicTier1TokenSegment *token_segments [[buffer(4)]],
    device uchar *out [[buffer(5)]],
    device J2kClassicEncodeStatus *statuses [[buffer(6)]],
    device J2kClassicSegment *segments [[buffer(7)]],
    constant uint &mq_token_stride_bytes [[buffer(8)]],
    constant uint &raw_token_stride_bytes [[buffer(9)]],
    constant uint &token_segment_stride [[buffer(10)]],
    constant uint &job_count [[buffer(11)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= job_count) {
        return;
    }
    const J2kClassicEncodeBatchJob job = jobs[gid];
    J2kClassicEncodeParams params;
    params.width = job.width;
    params.height = job.height;
    params.sub_band_type = job.sub_band_type;
    params.total_bitplanes = job.total_bitplanes;
    params.style_flags = job.style_flags;
    params.output_capacity = job.output_capacity;
    params.segment_capacity = job.segment_capacity;
    j2k_pack_classic_tier1_split_tokens_bypass_u16_32_impl(
        params,
        counters[gid],
        mq_token_data + gid * mq_token_stride_bytes,
        raw_token_data + gid * raw_token_stride_bytes,
        token_segments + gid * token_segment_stride,
        mq_token_stride_bytes,
        raw_token_stride_bytes,
        token_segment_stride,
        out + job.output_offset,
        statuses + gid,
        segments + job.segment_offset
    );
}

kernel void j2k_profile_classic_tier1_raw_pack_bypass_u16_32(
    device const int *coefficients [[buffer(0)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(1)]],
    device uchar *out [[buffer(2)]],
    constant uint &job_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= job_count) {
        return;
    }
    const J2kClassicEncodeBatchJob job = jobs[gid];
    J2kClassicEncodeParams params;
    params.width = job.width;
    params.height = job.height;
    params.sub_band_type = job.sub_band_type;
    params.total_bitplanes = job.total_bitplanes;
    params.style_flags = job.style_flags;
    params.output_capacity = job.output_capacity;
    params.segment_capacity = job.segment_capacity;
    j2k_profile_classic_tier1_raw_pack_bypass_u16_32_impl(
        coefficients + job.coefficient_offset,
        out + job.output_offset,
        params
    );
}

kernel void j2k_profile_classic_tier1_arithmetic_pack_bypass_u16_32(
    device const int *coefficients [[buffer(0)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(1)]],
    device uchar *out [[buffer(2)]],
    constant uint &job_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= job_count) {
        return;
    }
    const J2kClassicEncodeBatchJob job = jobs[gid];
    J2kClassicEncodeParams params;
    params.width = job.width;
    params.height = job.height;
    params.sub_band_type = job.sub_band_type;
    params.total_bitplanes = job.total_bitplanes;
    params.style_flags = job.style_flags;
    params.output_capacity = job.output_capacity;
    params.segment_capacity = job.segment_capacity;
    j2k_profile_classic_tier1_arithmetic_pack_bypass_u16_32_impl(
        coefficients + job.coefficient_offset,
        out + job.output_offset,
        params
    );
}

kernel void j2k_encode_classic_code_blocks_style0_32(
    device const int *coefficients [[buffer(0)]],
    device uchar *out [[buffer(1)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(2)]],
    device J2kClassicEncodeStatus *statuses [[buffer(3)]],
    device J2kClassicSegment *segments [[buffer(4)]],
    constant uint &job_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= job_count) {
        return;
    }
    const J2kClassicEncodeBatchJob job = jobs[gid];
    J2kClassicEncodeParams params;
    params.width = job.width;
    params.height = job.height;
    params.sub_band_type = job.sub_band_type;
    params.total_bitplanes = job.total_bitplanes;
    params.style_flags = 0u;
    params.output_capacity = job.output_capacity;
    params.segment_capacity = job.segment_capacity;
    j2k_encode_classic_code_block_impl_style0_32(
        coefficients + job.coefficient_offset,
        out + job.output_offset,
        params,
        statuses + gid,
        segments + job.segment_offset
    );
}

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

struct J2kPacketEncodeParams {
    uint resolution_count;
    uint num_layers;
    uint num_components;
    uint code_block_count;
    uint subband_count;
    uint descriptor_count;
    uint output_capacity;
    uint header_capacity;
    uint scratch_node_capacity;
};

struct J2kPacketDescriptor {
    uint packet_index;
    uint state_index;
    uint layer;
    uint resolution;
    uint component;
    uint precinct_lo;
    uint precinct_hi;
    uint state_block_offset;
};

struct J2kPacketResolution {
    uint subband_offset;
    uint subband_count;
};

struct J2kPacketSubband {
    uint block_offset;
    uint block_count;
    uint num_cbs_x;
    uint num_cbs_y;
};

struct J2kPacketBlock {
    uint data_offset;
    uint data_len;
    uint num_coding_passes;
    uint num_zero_bitplanes;
    uint previously_included;
    uint l_block;
    uint block_coding_mode;
    uint reserved0;
};

struct J2kResidentPacketBlock {
    uint tier1_job_index;
    uint previously_included;
    uint l_block;
    uint block_coding_mode;
};

struct J2kResidentPacketBlockParams {
    uint block_count;
    uint tier1_job_count;
};

struct J2kPacketStateBlock {
    uint previously_included;
    uint l_block;
};

struct J2kPacketEncodeStatus {
    uint code;
    uint detail;
    uint data_len;
    uint reserved0;
    uint payload_copy_bytes;
    uint payload_copy_small_jobs;
    uint payload_copy_medium_jobs;
    uint payload_copy_large_jobs;
};

struct J2kLosslessCodestreamAssemblyParams {
    uint width;
    uint height;
    uint num_components;
    uint bit_depth;
    uint signed_samples;
    uint num_decomposition_levels;
    uint use_mct;
    uint guard_bits;
    uint progression_order;
    uint write_tlm;
    uint high_throughput;
    uint code_block_style;
    uint code_block_width_exp;
    uint code_block_height_exp;
    uint output_capacity;
};

struct J2kCodestreamAssemblyStatus {
    uint code;
    uint detail;
    uint data_len;
    uint reserved0;
};

struct J2kPacketBitWriter {
    device uchar *data;
    uint capacity;
    uint len;
    uint buffer;
    uint bits_in_buffer;
    uint last_byte_was_ff;
    uint failed;
};

inline void j2k_set_packet_status(device J2kPacketEncodeStatus *status, uint code, uint detail, uint len) {
    status->code = code;
    status->detail = detail;
    status->data_len = len;
    status->reserved0 = 0u;
    status->payload_copy_bytes = 0u;
    status->payload_copy_small_jobs = 0u;
    status->payload_copy_medium_jobs = 0u;
    status->payload_copy_large_jobs = 0u;
}

inline void j2k_set_packet_status_payload_copy(
    device J2kPacketEncodeStatus *status,
    uint code,
    uint detail,
    uint len,
    uint payload_copy_bytes,
    uint payload_copy_small_jobs,
    uint payload_copy_medium_jobs,
    uint payload_copy_large_jobs
) {
    status->code = code;
    status->detail = detail;
    status->data_len = len;
    status->reserved0 = 0u;
    status->payload_copy_bytes = payload_copy_bytes;
    status->payload_copy_small_jobs = payload_copy_small_jobs;
    status->payload_copy_medium_jobs = payload_copy_medium_jobs;
    status->payload_copy_large_jobs = payload_copy_large_jobs;
}

inline void j2k_set_codestream_status(
    device J2kCodestreamAssemblyStatus *status,
    uint code,
    uint detail,
    uint len
) {
    status->code = code;
    status->detail = detail;
    status->data_len = len;
    status->reserved0 = 0u;
}

inline bool j2k_codestream_write_u8(
    device uchar *out,
    uint capacity,
    thread uint &cursor,
    uint value
) {
    if (cursor >= capacity) {
        return false;
    }
    out[cursor] = uchar(value & 0xFFu);
    cursor += 1u;
    return true;
}

inline bool j2k_codestream_write_u16(
    device uchar *out,
    uint capacity,
    thread uint &cursor,
    uint value
) {
    return j2k_codestream_write_u8(out, capacity, cursor, value >> 8u) &&
        j2k_codestream_write_u8(out, capacity, cursor, value);
}

inline bool j2k_codestream_write_u32(
    device uchar *out,
    uint capacity,
    thread uint &cursor,
    uint value
) {
    return j2k_codestream_write_u8(out, capacity, cursor, value >> 24u) &&
        j2k_codestream_write_u8(out, capacity, cursor, value >> 16u) &&
        j2k_codestream_write_u8(out, capacity, cursor, value >> 8u) &&
        j2k_codestream_write_u8(out, capacity, cursor, value);
}

inline bool j2k_codestream_write_marker(
    device uchar *out,
    uint capacity,
    thread uint &cursor,
    uint marker
) {
    return j2k_codestream_write_u8(out, capacity, cursor, 0xFFu) &&
        j2k_codestream_write_u8(out, capacity, cursor, marker);
}

kernel void j2k_assemble_lossless_classic_codestream(
    device const uchar *tile_data [[buffer(0)]],
    device const J2kPacketEncodeStatus *tile_status [[buffer(1)]],
    device uchar *out [[buffer(2)]],
    constant J2kLosslessCodestreamAssemblyParams &params [[buffer(3)]],
    device J2kCodestreamAssemblyStatus *status [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) {
        return;
    }

    j2k_set_codestream_status(status, J2K_ENCODE_STATUS_FAIL, 0u, 0u);
    const J2kPacketEncodeStatus packet_status = tile_status[0];
    if (packet_status.code != J2K_ENCODE_STATUS_OK) {
        j2k_set_codestream_status(status, J2K_ENCODE_STATUS_FAIL, packet_status.detail, 0u);
        return;
    }
    if (params.num_components == 0u || params.num_components > 255u ||
        params.bit_depth == 0u || params.bit_depth > 16u ||
        params.num_decomposition_levels > 31u) {
        j2k_set_codestream_status(status, J2K_ENCODE_STATUS_UNSUPPORTED, 1u, 0u);
        return;
    }

    const uint tile_len = packet_status.data_len;
    const uint tile_part_len = 14u + tile_len;
    uint cursor = 0u;
    bool ok = true;

    ok = ok && j2k_codestream_write_marker(out, params.output_capacity, cursor, 0x4Fu);

    ok = ok && j2k_codestream_write_marker(out, params.output_capacity, cursor, 0x51u);
    const uint siz_len = 38u + 3u * params.num_components;
    ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, siz_len);
    ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, params.width);
    ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, params.height);
    ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, params.width);
    ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, params.height);
    ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, params.num_components);
    const uint ssiz = (params.bit_depth - 1u) | (params.signed_samples != 0u ? 0x80u : 0u);
    for (uint comp = 0u; comp < params.num_components && ok; ++comp) {
        ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, ssiz);
        ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, 1u);
        ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, 1u);
    }

    if (params.high_throughput != 0u) {
        const uint magnitude_bits = params.bit_depth - 1u;
        const uint bp = magnitude_bits <= 8u ? 0u :
            (magnitude_bits < 28u ? magnitude_bits - 8u : 13u + (magnitude_bits >> 2u));
        ok = ok && j2k_codestream_write_marker(out, params.output_capacity, cursor, 0x50u);
        ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, 8u);
        ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, 0x00020000u);
        ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, bp);
    }

    ok = ok && j2k_codestream_write_marker(out, params.output_capacity, cursor, 0x52u);
    ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, 12u);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, params.progression_order);
    ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, 1u);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, params.use_mct != 0u ? 1u : 0u);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, params.num_decomposition_levels);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, params.code_block_width_exp);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, params.code_block_height_exp);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, params.code_block_style);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, 1u);

    ok = ok && j2k_codestream_write_marker(out, params.output_capacity, cursor, 0x5Cu);
    const uint qcd_steps = 1u + 3u * params.num_decomposition_levels;
    ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, 3u + qcd_steps);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, params.guard_bits << 5u);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, params.bit_depth << 3u);
    for (uint level = 0u; level < params.num_decomposition_levels && ok; ++level) {
        ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, (params.bit_depth + 1u) << 3u);
        ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, (params.bit_depth + 1u) << 3u);
        ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, (params.bit_depth + 2u) << 3u);
    }

    if (params.write_tlm != 0u) {
        ok = ok && j2k_codestream_write_marker(out, params.output_capacity, cursor, 0x55u);
        ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, 10u);
        ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, 0u);
        ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, 0x22u);
        ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, 0u);
        ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, tile_part_len);
    }

    ok = ok && j2k_codestream_write_marker(out, params.output_capacity, cursor, 0x90u);
    ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, 10u);
    ok = ok && j2k_codestream_write_u16(out, params.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(out, params.output_capacity, cursor, tile_part_len);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u8(out, params.output_capacity, cursor, 1u);
    ok = ok && j2k_codestream_write_marker(out, params.output_capacity, cursor, 0x93u);

    if (!ok || cursor + tile_len + 2u > params.output_capacity) {
        j2k_set_codestream_status(status, J2K_ENCODE_STATUS_FAIL, 2u, cursor);
        return;
    }
    for (uint idx = 0u; idx < tile_len; ++idx) {
        out[cursor + idx] = tile_data[idx];
    }
    cursor += tile_len;
    ok = ok && j2k_codestream_write_marker(out, params.output_capacity, cursor, 0xD9u);
    if (!ok) {
        j2k_set_codestream_status(status, J2K_ENCODE_STATUS_FAIL, 3u, cursor);
        return;
    }

    j2k_set_codestream_status(status, J2K_ENCODE_STATUS_OK, 0u, cursor);
}

struct J2kBatchedCodestreamAssemblyJob {
    uint tile_data_offset;
    uint codestream_offset;
    uint width;
    uint height;
    uint num_components;
    uint bit_depth;
    uint signed_samples;
    uint num_decomposition_levels;
    uint use_mct;
    uint guard_bits;
    uint progression_order;
    uint write_tlm;
    uint high_throughput;
    uint code_block_style;
    uint code_block_width_exp;
    uint code_block_height_exp;
    uint output_capacity;
};

kernel void j2k_assemble_lossless_codestream_batched(
    device const uchar *tile_data [[buffer(0)]],
    device const J2kPacketEncodeStatus *tile_status [[buffer(1)]],
    device uchar *out [[buffer(2)]],
    device const J2kBatchedCodestreamAssemblyJob *jobs [[buffer(3)]],
    device J2kCodestreamAssemblyStatus *status [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const J2kBatchedCodestreamAssemblyJob job = jobs[gid];
    device J2kCodestreamAssemblyStatus *tile_status_out = status + gid;
    device uchar *tile_out = out + job.codestream_offset;
    const J2kPacketEncodeStatus packet_status = tile_status[gid];

    j2k_set_codestream_status(tile_status_out, J2K_ENCODE_STATUS_FAIL, 0u, 0u);
    if (packet_status.code != J2K_ENCODE_STATUS_OK) {
        j2k_set_codestream_status(tile_status_out, J2K_ENCODE_STATUS_FAIL, packet_status.detail, 0u);
        return;
    }
    if (job.num_components == 0u || job.num_components > 255u ||
        job.bit_depth == 0u || job.bit_depth > 16u ||
        job.num_decomposition_levels > 31u) {
        j2k_set_codestream_status(tile_status_out, J2K_ENCODE_STATUS_UNSUPPORTED, 1u, 0u);
        return;
    }

    const uint tile_len = packet_status.data_len;
    const uint tile_part_len = 14u + tile_len;
    uint cursor = 0u;
    bool ok = true;

    ok = ok && j2k_codestream_write_marker(tile_out, job.output_capacity, cursor, 0x4Fu);

    ok = ok && j2k_codestream_write_marker(tile_out, job.output_capacity, cursor, 0x51u);
    const uint siz_len = 38u + 3u * job.num_components;
    ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, siz_len);
    ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, job.width);
    ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, job.height);
    ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, job.width);
    ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, job.height);
    ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, job.num_components);
    const uint ssiz = (job.bit_depth - 1u) | (job.signed_samples != 0u ? 0x80u : 0u);
    for (uint comp = 0u; comp < job.num_components && ok; ++comp) {
        ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, ssiz);
        ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, 1u);
        ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, 1u);
    }

    if (job.high_throughput != 0u) {
        const uint magnitude_bits = job.bit_depth - 1u;
        const uint bp = magnitude_bits <= 8u ? 0u :
            (magnitude_bits < 28u ? magnitude_bits - 8u : 13u + (magnitude_bits >> 2u));
        ok = ok && j2k_codestream_write_marker(tile_out, job.output_capacity, cursor, 0x50u);
        ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, 8u);
        ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, 0x00020000u);
        ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, bp);
    }

    ok = ok && j2k_codestream_write_marker(tile_out, job.output_capacity, cursor, 0x52u);
    ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, 12u);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, job.progression_order);
    ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, 1u);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, job.use_mct != 0u ? 1u : 0u);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, job.num_decomposition_levels);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, job.code_block_width_exp);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, job.code_block_height_exp);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, job.code_block_style);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, 1u);

    ok = ok && j2k_codestream_write_marker(tile_out, job.output_capacity, cursor, 0x5Cu);
    const uint qcd_steps = 1u + 3u * job.num_decomposition_levels;
    ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, 3u + qcd_steps);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, job.guard_bits << 5u);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, job.bit_depth << 3u);
    for (uint level = 0u; level < job.num_decomposition_levels && ok; ++level) {
        ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, (job.bit_depth + 1u) << 3u);
        ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, (job.bit_depth + 1u) << 3u);
        ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, (job.bit_depth + 2u) << 3u);
    }

    if (job.write_tlm != 0u) {
        ok = ok && j2k_codestream_write_marker(tile_out, job.output_capacity, cursor, 0x55u);
        ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, 10u);
        ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, 0u);
        ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, 0x22u);
        ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, 0u);
        ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, tile_part_len);
    }

    ok = ok && j2k_codestream_write_marker(tile_out, job.output_capacity, cursor, 0x90u);
    ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, 10u);
    ok = ok && j2k_codestream_write_u16(tile_out, job.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u32(tile_out, job.output_capacity, cursor, tile_part_len);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, 0u);
    ok = ok && j2k_codestream_write_u8(tile_out, job.output_capacity, cursor, 1u);
    ok = ok && j2k_codestream_write_marker(tile_out, job.output_capacity, cursor, 0x93u);
    const uint payload_offset = cursor;

    if (!ok || cursor + tile_len + 2u > job.output_capacity) {
        j2k_set_codestream_status(tile_status_out, J2K_ENCODE_STATUS_FAIL, 2u, cursor);
        return;
    }
    cursor += tile_len;
    ok = ok && j2k_codestream_write_marker(tile_out, job.output_capacity, cursor, 0xD9u);
    if (!ok) {
        j2k_set_codestream_status(tile_status_out, J2K_ENCODE_STATUS_FAIL, 3u, cursor);
        return;
    }

    j2k_set_codestream_status(tile_status_out, J2K_ENCODE_STATUS_OK, payload_offset, cursor);
}

inline J2kPacketBlock j2k_classic_packet_block_from_resident(
    device const J2kResidentPacketBlock *resident_blocks,
    device const J2kClassicEncodeBatchJob *tier1_jobs,
    device const J2kClassicEncodeStatus *tier1_statuses,
    uint tier1_job_count,
    uint block_index
) {
    const J2kResidentPacketBlock resident = resident_blocks[block_index];
    J2kPacketBlock packet;
    packet.data_offset = 0u;
    packet.data_len = 0u;
    packet.num_coding_passes = 1u;
    packet.num_zero_bitplanes = 0u;
    packet.previously_included = resident.previously_included;
    packet.l_block = resident.l_block;
    packet.block_coding_mode = 0xFFFFFFFFu;
    packet.reserved0 = 0u;

    if (resident.tier1_job_index < tier1_job_count) {
        const J2kClassicEncodeBatchJob job = tier1_jobs[resident.tier1_job_index];
        const J2kClassicEncodeStatus tier1_status = tier1_statuses[resident.tier1_job_index];
        if (tier1_status.code == J2K_ENCODE_STATUS_OK &&
            tier1_status.data_len <= job.output_capacity &&
            tier1_status.reserved0 <= 1u &&
            tier1_status.reserved1 == 0u &&
            tier1_status.data_len + tier1_status.reserved0 >= tier1_status.data_len &&
            tier1_status.data_len + tier1_status.reserved0 <= job.output_capacity &&
            tier1_status.segment_count <= job.segment_capacity) {
            packet.data_offset = job.output_offset + tier1_status.reserved0;
            packet.data_len = tier1_status.data_len;
            packet.num_coding_passes = tier1_status.number_of_coding_passes;
            packet.num_zero_bitplanes = tier1_status.missing_bit_planes;
            packet.block_coding_mode = resident.block_coding_mode;
        } else {
            packet.reserved0 = tier1_status.detail;
        }
    }

    return packet;
}

inline J2kPacketBlock j2k_ht_packet_block_from_resident(
    device const J2kResidentPacketBlock *resident_blocks,
    device const J2kHtEncodeBatchJob *tier1_jobs,
    device const J2kHtEncodeStatus *tier1_statuses,
    uint tier1_job_count,
    uint block_index
) {
    const J2kResidentPacketBlock resident = resident_blocks[block_index];
    J2kPacketBlock packet;
    packet.data_offset = 0u;
    packet.data_len = 0u;
    packet.num_coding_passes = 1u;
    packet.num_zero_bitplanes = 0u;
    packet.previously_included = resident.previously_included;
    packet.l_block = resident.l_block;
    packet.block_coding_mode = 0xFFFFFFFFu;
    packet.reserved0 = 0u;

    if (resident.tier1_job_index < tier1_job_count) {
        const J2kHtEncodeBatchJob job = tier1_jobs[resident.tier1_job_index];
        const J2kHtEncodeStatus tier1_status = tier1_statuses[resident.tier1_job_index];
        if (tier1_status.code == J2K_ENCODE_STATUS_OK &&
            tier1_status.data_len <= job.output_capacity) {
            packet.data_offset = job.output_offset;
            packet.data_len = tier1_status.data_len;
            packet.num_coding_passes = tier1_status.num_coding_passes;
            packet.num_zero_bitplanes = tier1_status.num_zero_bitplanes;
            packet.block_coding_mode = resident.block_coding_mode;
        } else {
            packet.reserved0 = tier1_status.detail;
        }
    }

    return packet;
}

kernel void j2k_prepare_packet_blocks_from_classic_status(
    device const J2kResidentPacketBlock *resident_blocks [[buffer(0)]],
    device const J2kClassicEncodeBatchJob *tier1_jobs [[buffer(1)]],
    device const J2kClassicEncodeStatus *tier1_statuses [[buffer(2)]],
    device J2kPacketBlock *packet_blocks [[buffer(3)]],
    constant J2kResidentPacketBlockParams &params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.block_count) {
        return;
    }

    packet_blocks[gid] = j2k_classic_packet_block_from_resident(
        resident_blocks,
        tier1_jobs,
        tier1_statuses,
        params.tier1_job_count,
        gid
    );
}

kernel void j2k_prepare_packet_blocks_from_ht_status(
    device const J2kResidentPacketBlock *resident_blocks [[buffer(0)]],
    device const J2kHtEncodeBatchJob *tier1_jobs [[buffer(1)]],
    device const J2kHtEncodeStatus *tier1_statuses [[buffer(2)]],
    device J2kPacketBlock *packet_blocks [[buffer(3)]],
    constant J2kResidentPacketBlockParams &params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.block_count) {
        return;
    }

    packet_blocks[gid] = j2k_ht_packet_block_from_resident(
        resident_blocks,
        tier1_jobs,
        tier1_statuses,
        params.tier1_job_count,
        gid
    );
}

inline void j2k_packet_writer_init(thread J2kPacketBitWriter &writer, device uchar *data, uint capacity) {
    writer.data = data;
    writer.capacity = capacity;
    writer.len = 0u;
    writer.buffer = 0u;
    writer.bits_in_buffer = 0u;
    writer.last_byte_was_ff = 0u;
    writer.failed = 0u;
}

inline void j2k_packet_flush_byte(thread J2kPacketBitWriter &writer) {
    const uint limit = writer.last_byte_was_ff != 0u ? 7u : 8u;
    const uchar byte = uchar(writer.buffer >> (writer.bits_in_buffer - limit));
    if (writer.len >= writer.capacity) {
        writer.failed = 1u;
        return;
    }
    writer.data[writer.len] = byte;
    writer.len += 1u;
    writer.last_byte_was_ff = byte == uchar(0xFFu) ? 1u : 0u;
    writer.bits_in_buffer -= limit;
    writer.buffer &= writer.bits_in_buffer == 0u ? 0u : ((1u << writer.bits_in_buffer) - 1u);
}

inline void j2k_packet_write_bit(thread J2kPacketBitWriter &writer, uint bit) {
    writer.buffer = (writer.buffer << 1u) | (bit & 1u);
    writer.bits_in_buffer += 1u;
    const uint limit = writer.last_byte_was_ff != 0u ? 7u : 8u;
    if (writer.bits_in_buffer >= limit) {
        j2k_packet_flush_byte(writer);
    }
}

inline void j2k_packet_write_bits(thread J2kPacketBitWriter &writer, uint value, uint count) {
    for (int bit = int(count) - 1; bit >= 0; --bit) {
        j2k_packet_write_bit(writer, (value >> uint(bit)) & 1u);
    }
}

inline void j2k_packet_writer_finish(thread J2kPacketBitWriter &writer) {
    if (writer.bits_in_buffer > 0u) {
        const uint limit = writer.last_byte_was_ff != 0u ? 7u : 8u;
        const uint shift = limit - writer.bits_in_buffer;
        const uchar byte = uchar(writer.buffer << shift);
        if (writer.len >= writer.capacity) {
            writer.failed = 1u;
            return;
        }
        writer.data[writer.len] = byte;
        writer.len += 1u;
        writer.last_byte_was_ff = byte == uchar(0xFFu) ? 1u : 0u;
        writer.buffer = 0u;
        writer.bits_in_buffer = 0u;
    }
}

inline uint j2k_packet_ilog2(uint value) {
    return value == 0u ? 0u : 31u - clz(value);
}

inline bool j2k_packet_value_fits(uint value, uint bits) {
    return bits >= 32u || value < (1u << bits);
}

inline uint j2k_packet_bits_for_length(uint l_block, uint passes) {
    const uint log2_passes = passes <= 1u ? 0u : j2k_packet_ilog2(passes);
    return l_block + log2_passes;
}

inline uint j2k_packet_bits_for_ht_length(uint l_block, uint passes) {
    const uint placeholder_groups = (passes > 0u ? passes - 1u : 0u) / 3u;
    const uint placeholder_passes = placeholder_groups * 3u;
    return l_block + j2k_packet_ilog2(placeholder_passes + 1u);
}

inline void j2k_packet_encode_num_passes(uint passes, thread J2kPacketBitWriter &writer) {
    if (passes == 1u) {
        j2k_packet_write_bit(writer, 0u);
    } else if (passes == 2u) {
        j2k_packet_write_bits(writer, 0b10u, 2u);
    } else if (passes == 3u) {
        j2k_packet_write_bits(writer, 0b1100u, 4u);
    } else if (passes == 4u) {
        j2k_packet_write_bits(writer, 0b1101u, 4u);
    } else if (passes == 5u) {
        j2k_packet_write_bits(writer, 0b1110u, 4u);
    } else if (passes <= 36u) {
        j2k_packet_write_bits(writer, 0b1111u, 4u);
        j2k_packet_write_bits(writer, passes - 6u, 5u);
    } else {
        j2k_packet_write_bits(writer, 0x1FFu, 9u);
        j2k_packet_write_bits(writer, passes - 37u, 7u);
    }
}

inline void j2k_packet_encode_num_ht_passes(uint passes, thread J2kPacketBitWriter &writer) {
    if (passes == 1u) {
        j2k_packet_write_bit(writer, 0u);
    } else if (passes == 2u) {
        j2k_packet_write_bits(writer, 0b10u, 2u);
    } else if (passes <= 5u) {
        j2k_packet_write_bits(writer, 0b11u, 2u);
        j2k_packet_write_bits(writer, passes - 3u, 2u);
    } else if (passes <= 36u) {
        j2k_packet_write_bits(writer, 0b11u, 2u);
        j2k_packet_write_bits(writer, 0b11u, 2u);
        j2k_packet_write_bits(writer, passes - 6u, 5u);
    } else {
        j2k_packet_write_bits(writer, 0b11u, 2u);
        j2k_packet_write_bits(writer, 0b11u, 2u);
        j2k_packet_write_bits(writer, 31u, 5u);
        j2k_packet_write_bits(writer, passes - 37u, 7u);
    }
}

inline void j2k_packet_encode_length(
    uint length,
    thread uint &l_block,
    uint num_bits,
    thread J2kPacketBitWriter &writer
) {
    while (!j2k_packet_value_fits(length, num_bits)) {
        j2k_packet_write_bit(writer, 1u);
        l_block += 1u;
        num_bits += 1u;
    }
    j2k_packet_write_bit(writer, 0u);
    j2k_packet_write_bits(writer, length, num_bits);
}

inline bool j2k_packet_encode_classic_segment_lengths_resident(
    const J2kPacketBlock block,
    const J2kClassicEncodeBatchJob tier1_job,
    const J2kClassicEncodeStatus tier1_status,
    device const J2kClassicSegment *tier1_segments,
    thread uint &l_block,
    thread J2kPacketBitWriter &writer
) {
    if (tier1_status.segment_count <= 1u) {
        const uint num_bits = j2k_packet_bits_for_length(l_block, block.num_coding_passes);
        j2k_packet_encode_length(block.data_len, l_block, num_bits, writer);
        return true;
    }

    uint segment_data_sum = 0u;
    uint segment_pass_sum = 0u;
    uint required_l_block = l_block;
    for (uint segment_idx = 0u; segment_idx < tier1_status.segment_count; ++segment_idx) {
        const J2kClassicSegment segment =
            tier1_segments[tier1_job.segment_offset + segment_idx];
        if (segment.start_coding_pass >= segment.end_coding_pass ||
            segment.data_offset != segment_data_sum ||
            segment_data_sum > 0xffffffffu - segment.data_length) {
            return false;
        }
        const uint segment_passes = segment.end_coding_pass - segment.start_coding_pass;
        segment_data_sum += segment.data_length;
        segment_pass_sum += segment_passes;
        while (!j2k_packet_value_fits(
                segment.data_length,
                j2k_packet_bits_for_length(required_l_block, segment_passes))) {
            required_l_block += 1u;
        }
    }
    if (segment_data_sum != block.data_len || segment_pass_sum != block.num_coding_passes) {
        return false;
    }

    while (l_block < required_l_block) {
        j2k_packet_write_bit(writer, 1u);
        l_block += 1u;
    }
    j2k_packet_write_bit(writer, 0u);

    for (uint segment_idx = 0u; segment_idx < tier1_status.segment_count; ++segment_idx) {
        const J2kClassicSegment segment =
            tier1_segments[tier1_job.segment_offset + segment_idx];
        const uint segment_passes = segment.end_coding_pass - segment.start_coding_pass;
        const uint length_bits = j2k_packet_bits_for_length(l_block, segment_passes);
        j2k_packet_write_bits(writer, segment.data_length, length_bits);
    }
    return writer.failed == 0u;
}

inline uint j2k_packet_tree_offsets(
    uint width,
    uint height,
    thread uint *level_offsets,
    thread uint *level_widths,
    thread uint *level_heights,
    thread uint &levels
) {
    uint total = 0u;
    uint w = width;
    uint h = height;
    levels = 0u;
    while (true) {
        level_offsets[levels] = total;
        level_widths[levels] = w;
        level_heights[levels] = h;
        total += w * h;
        levels += 1u;
        if (w <= 1u && h <= 1u) {
            break;
        }
        w = (w + 1u) / 2u;
        h = (h + 1u) / 2u;
    }
    return total;
}

inline bool j2k_packet_prepare_tree(
    device const J2kPacketBlock *blocks,
    uint block_offset,
    uint block_count,
    uint num_cbs_x,
    uint num_cbs_y,
    bool zero_bitplanes,
    uint inclusion_layer,
    device uint *value,
    device uint *current,
    device uint *known,
    uint node_capacity,
    thread uint *level_offsets,
    thread uint *level_widths,
    thread uint *level_heights,
    thread uint &levels
) {
    if (num_cbs_x == 0u || num_cbs_y == 0u || num_cbs_x * num_cbs_y != block_count) {
        return false;
    }
    const uint node_count =
        j2k_packet_tree_offsets(num_cbs_x, num_cbs_y, level_offsets, level_widths, level_heights, levels);
    if (node_count > node_capacity || levels > 16u) {
        return false;
    }
    for (uint idx = 0u; idx < node_count; ++idx) {
        value[idx] = 0u;
        current[idx] = 0u;
        known[idx] = 0u;
    }
    for (uint idx = 0u; idx < block_count; ++idx) {
        const J2kPacketBlock block = blocks[block_offset + idx];
        value[idx] = zero_bitplanes
            ? block.num_zero_bitplanes
            : (block.num_coding_passes > 0u ? inclusion_layer : 0x7FFFFFFFu);
    }
    for (uint level = 1u; level < levels; ++level) {
        const uint prev_w = level_widths[level - 1u];
        const uint prev_h = level_heights[level - 1u];
        const uint cur_w = level_widths[level];
        const uint cur_h = level_heights[level];
        for (uint py = 0u; py < cur_h; ++py) {
            for (uint px = 0u; px < cur_w; ++px) {
                uint min_value = 0xFFFFFFFFu;
                for (uint dy = 0u; dy < 2u; ++dy) {
                    const uint cy = py * 2u + dy;
                    if (cy >= prev_h) {
                        continue;
                    }
                    for (uint dx = 0u; dx < 2u; ++dx) {
                        const uint cx = px * 2u + dx;
                        if (cx >= prev_w) {
                            continue;
                        }
                        const uint child = level_offsets[level - 1u] + cy * prev_w + cx;
                        min_value = min(min_value, value[child]);
                    }
                }
                value[level_offsets[level] + py * cur_w + px] = min_value;
            }
        }
    }
    return true;
}

inline bool j2k_packet_prepare_tree_resident_classic(
    device const J2kResidentPacketBlock *resident_blocks,
    device const J2kClassicEncodeBatchJob *tier1_jobs,
    device const J2kClassicEncodeStatus *tier1_statuses,
    uint tier1_job_count,
    uint block_offset,
    uint block_count,
    uint num_cbs_x,
    uint num_cbs_y,
    bool zero_bitplanes,
    uint inclusion_layer,
    device uint *value,
    device uint *current,
    device uint *known,
    uint node_capacity,
    thread uint *level_offsets,
    thread uint *level_widths,
    thread uint *level_heights,
    thread uint &levels
) {
    if (num_cbs_x == 0u || num_cbs_y == 0u || num_cbs_x * num_cbs_y != block_count) {
        return false;
    }
    const uint node_count =
        j2k_packet_tree_offsets(num_cbs_x, num_cbs_y, level_offsets, level_widths, level_heights, levels);
    if (node_count > node_capacity || levels > 16u) {
        return false;
    }
    for (uint idx = 0u; idx < node_count; ++idx) {
        value[idx] = 0u;
        current[idx] = 0u;
        known[idx] = 0u;
    }
    for (uint idx = 0u; idx < block_count; ++idx) {
        const J2kPacketBlock block = j2k_classic_packet_block_from_resident(
            resident_blocks,
            tier1_jobs,
            tier1_statuses,
            tier1_job_count,
            block_offset + idx
        );
        value[idx] = zero_bitplanes
            ? block.num_zero_bitplanes
            : (block.num_coding_passes > 0u ? inclusion_layer : 0x7FFFFFFFu);
    }
    for (uint level = 1u; level < levels; ++level) {
        const uint prev_w = level_widths[level - 1u];
        const uint prev_h = level_heights[level - 1u];
        const uint cur_w = level_widths[level];
        const uint cur_h = level_heights[level];
        for (uint py = 0u; py < cur_h; ++py) {
            for (uint px = 0u; px < cur_w; ++px) {
                uint min_value = 0xFFFFFFFFu;
                for (uint dy = 0u; dy < 2u; ++dy) {
                    const uint cy = py * 2u + dy;
                    if (cy >= prev_h) {
                        continue;
                    }
                    for (uint dx = 0u; dx < 2u; ++dx) {
                        const uint cx = px * 2u + dx;
                        if (cx >= prev_w) {
                            continue;
                        }
                        const uint child = level_offsets[level - 1u] + cy * prev_w + cx;
                        min_value = min(min_value, value[child]);
                    }
                }
                value[level_offsets[level] + py * cur_w + px] = min_value;
            }
        }
    }
    return true;
}

inline void j2k_packet_tree_encode(
    uint x,
    uint y,
    uint threshold,
    device uint *value,
    device uint *current,
    device uint *known,
    thread uint *level_offsets,
    thread uint *level_widths,
    uint levels,
    thread J2kPacketBitWriter &writer
) {
    thread uint path[16];
    uint cx = x;
    uint cy = y;
    for (uint level = 0u; level < levels; ++level) {
        path[level] = level_offsets[level] + cy * level_widths[level] + cx;
        cx /= 2u;
        cy /= 2u;
    }

    uint parent_val = 0u;
    for (int level = int(levels) - 1; level >= 0; --level) {
        const uint node = path[uint(level)];
        const uint start = max(current[node], parent_val);
        if (known[node] == 0u) {
            const uint target = min(value[node], threshold);
            for (uint v = start; v < target; ++v) {
                j2k_packet_write_bit(writer, 0u);
            }
            if (value[node] < threshold) {
                j2k_packet_write_bit(writer, 1u);
                known[node] = 1u;
            }
            current[node] = target;
        }
        parent_val = current[node];
    }
}

kernel void j2k_encode_packetization(
    device const J2kPacketResolution *resolutions [[buffer(0)]],
    device const J2kPacketSubband *subbands [[buffer(1)]],
    device const J2kPacketBlock *blocks [[buffer(2)]],
    device const uchar *payload [[buffer(3)]],
    device uchar *out [[buffer(4)]],
    device uchar *header [[buffer(5)]],
    device uint *tree_scratch [[buffer(6)]],
    constant J2kPacketEncodeParams &params [[buffer(7)]],
    device J2kPacketEncodeStatus *status [[buffer(8)]],
    device const J2kPacketDescriptor *descriptors [[buffer(9)]],
    device J2kPacketStateBlock *state_blocks [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) {
        return;
    }

    j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 0u, 0u);

    const uint node_capacity = params.scratch_node_capacity;
    device uint *inc_value = tree_scratch;
    device uint *inc_current = tree_scratch + node_capacity;
    device uint *inc_known = tree_scratch + node_capacity * 2u;
    device uint *zbp_value = tree_scratch + node_capacity * 3u;
    device uint *zbp_current = tree_scratch + node_capacity * 4u;
    device uint *zbp_known = tree_scratch + node_capacity * 5u;

    uint out_len = 0u;
    const uint packet_count =
        params.descriptor_count > 0u ? params.descriptor_count : params.resolution_count;
    for (uint packet_order_idx = 0u; packet_order_idx < packet_count; ++packet_order_idx) {
        const bool has_descriptor = params.descriptor_count > 0u;
        const J2kPacketDescriptor descriptor = has_descriptor
            ? descriptors[packet_order_idx]
            : J2kPacketDescriptor{packet_order_idx, packet_order_idx, 0u, packet_order_idx, 0u, 0u, 0u, 0u};
        const uint packet_index = descriptor.packet_index;
        if (packet_index >= params.resolution_count) {
            j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 6u, 0u);
            return;
        }
        const J2kPacketResolution resolution = resolutions[packet_index];
        uint state_block_cursor = descriptor.state_block_offset;
        bool any_data = false;
        for (uint sb_idx = 0u; sb_idx < resolution.subband_count; ++sb_idx) {
            const J2kPacketSubband subband = subbands[resolution.subband_offset + sb_idx];
            for (uint block_idx = 0u; block_idx < subband.block_count; ++block_idx) {
                if (blocks[subband.block_offset + block_idx].num_coding_passes > 0u) {
                    any_data = true;
                    break;
                }
            }
        }

        thread J2kPacketBitWriter writer;
        j2k_packet_writer_init(writer, header, params.header_capacity);
        if (!any_data) {
            j2k_packet_write_bit(writer, 0u);
            j2k_packet_writer_finish(writer);
        } else {
            j2k_packet_write_bit(writer, 1u);
            for (uint sb_idx = 0u; sb_idx < resolution.subband_count; ++sb_idx) {
                const J2kPacketSubband subband = subbands[resolution.subband_offset + sb_idx];
                const uint subband_state_block_offset = state_block_cursor;
                state_block_cursor += subband.block_count;
                thread uint level_offsets[16];
                thread uint level_widths[16];
                thread uint level_heights[16];
                uint levels = 0u;
                if (!j2k_packet_prepare_tree(
                        blocks,
                        subband.block_offset,
                        subband.block_count,
                        subband.num_cbs_x,
                        subband.num_cbs_y,
                        false,
                        descriptor.layer,
                        inc_value,
                        inc_current,
                        inc_known,
                        node_capacity,
                        level_offsets,
                        level_widths,
                        level_heights,
                        levels)) {
                    j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 1u, 0u);
                    return;
                }
                thread uint z_level_offsets[16];
                thread uint z_level_widths[16];
                thread uint z_level_heights[16];
                uint z_levels = 0u;
                if (!j2k_packet_prepare_tree(
                        blocks,
                        subband.block_offset,
                        subband.block_count,
                        subband.num_cbs_x,
                        subband.num_cbs_y,
                        true,
                        descriptor.layer,
                        zbp_value,
                        zbp_current,
                        zbp_known,
                        node_capacity,
                        z_level_offsets,
                        z_level_widths,
                        z_level_heights,
                        z_levels)) {
                    j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 2u, 0u);
                    return;
                }

                for (uint block_idx = 0u; block_idx < subband.block_count; ++block_idx) {
                    const uint x = block_idx % subband.num_cbs_x;
                    const uint y = block_idx / subband.num_cbs_x;
                    const J2kPacketBlock block = blocks[subband.block_offset + block_idx];
                    const uint state_block_index = subband_state_block_offset + block_idx;
                    uint previously_included = block.previously_included;
                    uint local_l_block = block.l_block;
                    if (has_descriptor) {
                        previously_included = state_blocks[state_block_index].previously_included;
                        local_l_block = state_blocks[state_block_index].l_block;
                    }
                    if (previously_included == 0u) {
                        j2k_packet_tree_encode(
                            x,
                            y,
                            descriptor.layer + 1u,
                            inc_value,
                            inc_current,
                            inc_known,
                            level_offsets,
                            level_widths,
                            levels,
                            writer
                        );
                        if (block.num_coding_passes == 0u) {
                            continue;
                        }
                        j2k_packet_tree_encode(
                            x,
                            y,
                            block.num_zero_bitplanes + 1u,
                            zbp_value,
                            zbp_current,
                            zbp_known,
                            z_level_offsets,
                            z_level_widths,
                            z_levels,
                            writer
                        );
                    } else if (block.num_coding_passes > 0u) {
                        j2k_packet_write_bit(writer, 1u);
                    } else {
                        j2k_packet_write_bit(writer, 0u);
                        continue;
                    }

                    if (block.block_coding_mode == 0u) {
                        const uint num_bits =
                            j2k_packet_bits_for_length(local_l_block, block.num_coding_passes);
                        j2k_packet_encode_num_passes(block.num_coding_passes, writer);
                        j2k_packet_encode_length(block.data_len, local_l_block, num_bits, writer);
                    } else if (block.block_coding_mode == 1u) {
                        const uint num_bits =
                            j2k_packet_bits_for_ht_length(local_l_block, block.num_coding_passes);
                        j2k_packet_encode_num_ht_passes(block.num_coding_passes, writer);
                        j2k_packet_encode_length(block.data_len, local_l_block, num_bits, writer);
                    } else {
                        j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 7u, block.reserved0);
                        return;
                    }
                    if (has_descriptor) {
                        state_blocks[state_block_index].previously_included = 1u;
                        state_blocks[state_block_index].l_block = local_l_block;
                    }
                }
            }
            j2k_packet_writer_finish(writer);
        }

        if (writer.failed != 0u || out_len + writer.len > params.output_capacity) {
            j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 3u, 0u);
            return;
        }
        for (uint idx = 0u; idx < writer.len; ++idx) {
            out[out_len + idx] = header[idx];
        }
        out_len += writer.len;
        if (writer.len > 0u && header[writer.len - 1u] == uchar(0xFFu)) {
            if (out_len >= params.output_capacity) {
                j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 4u, 0u);
                return;
            }
            out[out_len] = uchar(0u);
            out_len += 1u;
        }

        if (any_data) {
            for (uint sb_idx = 0u; sb_idx < resolution.subband_count; ++sb_idx) {
                const J2kPacketSubband subband = subbands[resolution.subband_offset + sb_idx];
                for (uint block_idx = 0u; block_idx < subband.block_count; ++block_idx) {
                    const J2kPacketBlock block = blocks[subband.block_offset + block_idx];
                    if (block.num_coding_passes == 0u) {
                        continue;
                    }
                    if (out_len + block.data_len > params.output_capacity) {
                        j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 5u, 0u);
                        return;
                    }
                    for (uint byte_idx = 0u; byte_idx < block.data_len; ++byte_idx) {
                        out[out_len + byte_idx] = payload[block.data_offset + byte_idx];
                    }
                    out_len += block.data_len;
                }
            }
        }
    }

    j2k_set_packet_status(status, J2K_ENCODE_STATUS_OK, 0u, out_len);
}

struct J2kPacketPayloadCopyJob {
    uint src_offset;
    uint dst_offset;
    uint byte_len;
    uint reserved0;
};

struct J2kPacketPayloadCopyParams {
    uint bytes_per_thread;
    uint stripes_per_job;
};

inline void j2k_packet_record_payload_copy_job(
    uint byte_len,
    thread uint &payload_copy_bytes,
    thread uint &payload_copy_small_jobs,
    thread uint &payload_copy_medium_jobs,
    thread uint &payload_copy_large_jobs
) {
    payload_copy_bytes = payload_copy_bytes > 0xffffffffu - byte_len
        ? 0xffffffffu
        : payload_copy_bytes + byte_len;
    if (byte_len <= J2K_PACKET_PAYLOAD_COPY_SMALL_JOB_BYTES) {
        payload_copy_small_jobs += 1u;
    } else if (byte_len <= J2K_PACKET_PAYLOAD_COPY_MEDIUM_JOB_BYTES) {
        payload_copy_medium_jobs += 1u;
    } else {
        payload_copy_large_jobs += 1u;
    }
}

inline bool j2k_packet_push_payload_copy_job(
    device J2kPacketPayloadCopyJob *payload_copy_jobs,
    uint payload_copy_capacity,
    uint src_offset,
    uint dst_offset,
    uint byte_len,
    thread uint &payload_copy_count,
    thread uint &payload_copy_bytes,
    thread uint &payload_copy_small_jobs,
    thread uint &payload_copy_medium_jobs,
    thread uint &payload_copy_large_jobs
) {
    if (byte_len == 0u) {
        return true;
    }
    if (payload_copy_count >= payload_copy_capacity) {
        return false;
    }
    payload_copy_jobs[payload_copy_count] =
        J2kPacketPayloadCopyJob{src_offset, dst_offset, byte_len, 0u};
    payload_copy_count += 1u;
    j2k_packet_record_payload_copy_job(
        byte_len,
        payload_copy_bytes,
        payload_copy_small_jobs,
        payload_copy_medium_jobs,
        payload_copy_large_jobs
    );
    return true;
}

struct J2kBatchedPacketEncodeJob {
    uint resolution_offset;
    uint subband_offset;
    uint block_offset;
    uint descriptor_offset;
    uint state_block_offset;
    uint output_offset;
    uint header_offset;
    uint scratch_offset;
    uint payload_copy_offset;
    uint payload_copy_capacity;
    uint resolution_count;
    uint num_layers;
    uint num_components;
    uint code_block_count;
    uint subband_count;
    uint descriptor_count;
    uint output_capacity;
    uint header_capacity;
    uint scratch_node_capacity;
};

kernel void j2k_encode_packetization_batched(
    device const J2kPacketResolution *all_resolutions [[buffer(0)]],
    device const J2kPacketSubband *all_subbands [[buffer(1)]],
    device const J2kPacketBlock *all_blocks [[buffer(2)]],
    device const uchar *payload [[buffer(3)]],
    device uchar *all_out [[buffer(4)]],
    device uchar *all_header [[buffer(5)]],
    device uint *all_tree_scratch [[buffer(6)]],
    device const J2kBatchedPacketEncodeJob *jobs [[buffer(7)]],
    device J2kPacketEncodeStatus *all_status [[buffer(8)]],
    device const J2kPacketDescriptor *all_descriptors [[buffer(9)]],
    device J2kPacketStateBlock *all_state_blocks [[buffer(10)]],
    device J2kPacketPayloadCopyJob *all_payload_copy_jobs [[buffer(11)]],
    uint gid [[thread_position_in_grid]]
) {
    const J2kBatchedPacketEncodeJob job = jobs[gid];
    device const J2kPacketResolution *resolutions = all_resolutions + job.resolution_offset;
    device const J2kPacketSubband *subbands = all_subbands + job.subband_offset;
    device const J2kPacketBlock *blocks = all_blocks + job.block_offset;
    device uchar *out = all_out + job.output_offset;
    device uchar *header = all_header + job.header_offset;
    device uint *tree_scratch = all_tree_scratch + job.scratch_offset;
    device J2kPacketEncodeStatus *status = all_status + gid;
    device const J2kPacketDescriptor *descriptors = all_descriptors + job.descriptor_offset;
    device J2kPacketStateBlock *state_blocks = all_state_blocks + job.state_block_offset;
    device J2kPacketPayloadCopyJob *payload_copy_jobs =
        all_payload_copy_jobs + job.payload_copy_offset;

    J2kPacketEncodeParams params;
    params.resolution_count = job.resolution_count;
    params.num_layers = job.num_layers;
    params.num_components = job.num_components;
    params.code_block_count = job.code_block_count;
    params.subband_count = job.subband_count;
    params.descriptor_count = job.descriptor_count;
    params.output_capacity = job.output_capacity;
    params.header_capacity = job.header_capacity;
    params.scratch_node_capacity = job.scratch_node_capacity;

    j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 0u, 0u);

    const uint node_capacity = params.scratch_node_capacity;
    device uint *inc_value = tree_scratch;
    device uint *inc_current = tree_scratch + node_capacity;
    device uint *inc_known = tree_scratch + node_capacity * 2u;
    device uint *zbp_value = tree_scratch + node_capacity * 3u;
    device uint *zbp_current = tree_scratch + node_capacity * 4u;
    device uint *zbp_known = tree_scratch + node_capacity * 5u;

    uint out_len = 0u;
    uint payload_copy_count = 0u;
    uint payload_copy_bytes = 0u;
    uint payload_copy_small_jobs = 0u;
    uint payload_copy_medium_jobs = 0u;
    uint payload_copy_large_jobs = 0u;
    const uint packet_count =
        params.descriptor_count > 0u ? params.descriptor_count : params.resolution_count;
    for (uint packet_order_idx = 0u; packet_order_idx < packet_count; ++packet_order_idx) {
        const bool has_descriptor = params.descriptor_count > 0u;
        const J2kPacketDescriptor descriptor = has_descriptor
            ? descriptors[packet_order_idx]
            : J2kPacketDescriptor{packet_order_idx, packet_order_idx, 0u, packet_order_idx, 0u, 0u, 0u, 0u};
        const uint packet_index = descriptor.packet_index;
        if (packet_index >= params.resolution_count) {
            j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 6u, 0u);
            return;
        }
        const J2kPacketResolution resolution = resolutions[packet_index];
        uint state_block_cursor = descriptor.state_block_offset;
        bool any_data = false;
        for (uint sb_idx = 0u; sb_idx < resolution.subband_count; ++sb_idx) {
            const J2kPacketSubband subband = subbands[resolution.subband_offset + sb_idx];
            for (uint block_idx = 0u; block_idx < subband.block_count; ++block_idx) {
                if (blocks[subband.block_offset + block_idx].num_coding_passes > 0u) {
                    any_data = true;
                    break;
                }
            }
        }

        thread J2kPacketBitWriter writer;
        j2k_packet_writer_init(writer, header, params.header_capacity);
        if (!any_data) {
            j2k_packet_write_bit(writer, 0u);
            j2k_packet_writer_finish(writer);
        } else {
            j2k_packet_write_bit(writer, 1u);
            for (uint sb_idx = 0u; sb_idx < resolution.subband_count; ++sb_idx) {
                const J2kPacketSubband subband = subbands[resolution.subband_offset + sb_idx];
                const uint subband_state_block_offset = state_block_cursor;
                state_block_cursor += subband.block_count;
                thread uint level_offsets[16];
                thread uint level_widths[16];
                thread uint level_heights[16];
                uint levels = 0u;
                if (!j2k_packet_prepare_tree(
                        blocks,
                        subband.block_offset,
                        subband.block_count,
                        subband.num_cbs_x,
                        subband.num_cbs_y,
                        false,
                        descriptor.layer,
                        inc_value,
                        inc_current,
                        inc_known,
                        node_capacity,
                        level_offsets,
                        level_widths,
                        level_heights,
                        levels)) {
                    j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 1u, 0u);
                    return;
                }
                thread uint z_level_offsets[16];
                thread uint z_level_widths[16];
                thread uint z_level_heights[16];
                uint z_levels = 0u;
                if (!j2k_packet_prepare_tree(
                        blocks,
                        subband.block_offset,
                        subband.block_count,
                        subband.num_cbs_x,
                        subband.num_cbs_y,
                        true,
                        descriptor.layer,
                        zbp_value,
                        zbp_current,
                        zbp_known,
                        node_capacity,
                        z_level_offsets,
                        z_level_widths,
                        z_level_heights,
                        z_levels)) {
                    j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 2u, 0u);
                    return;
                }

                for (uint block_idx = 0u; block_idx < subband.block_count; ++block_idx) {
                    const uint x = block_idx % subband.num_cbs_x;
                    const uint y = block_idx / subband.num_cbs_x;
                    const J2kPacketBlock block = blocks[subband.block_offset + block_idx];
                    const uint state_block_index = subband_state_block_offset + block_idx;
                    uint previously_included = block.previously_included;
                    uint local_l_block = block.l_block;
                    if (has_descriptor) {
                        previously_included = state_blocks[state_block_index].previously_included;
                        local_l_block = state_blocks[state_block_index].l_block;
                    }
                    if (previously_included == 0u) {
                        j2k_packet_tree_encode(
                            x,
                            y,
                            descriptor.layer + 1u,
                            inc_value,
                            inc_current,
                            inc_known,
                            level_offsets,
                            level_widths,
                            levels,
                            writer
                        );
                        if (block.num_coding_passes == 0u) {
                            continue;
                        }
                        j2k_packet_tree_encode(
                            x,
                            y,
                            block.num_zero_bitplanes + 1u,
                            zbp_value,
                            zbp_current,
                            zbp_known,
                            z_level_offsets,
                            z_level_widths,
                            z_levels,
                            writer
                        );
                    } else if (block.num_coding_passes > 0u) {
                        j2k_packet_write_bit(writer, 1u);
                    } else {
                        j2k_packet_write_bit(writer, 0u);
                        continue;
                    }

                    if (block.block_coding_mode == 0u) {
                        const uint num_bits =
                            j2k_packet_bits_for_length(local_l_block, block.num_coding_passes);
                        j2k_packet_encode_num_passes(block.num_coding_passes, writer);
                        j2k_packet_encode_length(block.data_len, local_l_block, num_bits, writer);
                    } else if (block.block_coding_mode == 1u) {
                        const uint num_bits =
                            j2k_packet_bits_for_ht_length(local_l_block, block.num_coding_passes);
                        j2k_packet_encode_num_ht_passes(block.num_coding_passes, writer);
                        j2k_packet_encode_length(block.data_len, local_l_block, num_bits, writer);
                    } else {
                        j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 7u, block.reserved0);
                        return;
                    }
                    if (has_descriptor) {
                        state_blocks[state_block_index].previously_included = 1u;
                        state_blocks[state_block_index].l_block = local_l_block;
                    }
                }
            }
            j2k_packet_writer_finish(writer);
        }

        if (writer.failed != 0u || out_len + writer.len > params.output_capacity) {
            j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 3u, 0u);
            return;
        }
        for (uint idx = 0u; idx < writer.len; ++idx) {
            out[out_len + idx] = header[idx];
        }
        out_len += writer.len;
        if (writer.len > 0u && header[writer.len - 1u] == uchar(0xFFu)) {
            if (out_len >= params.output_capacity) {
                j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 4u, 0u);
                return;
            }
            out[out_len] = uchar(0u);
            out_len += 1u;
        }

        if (any_data) {
            for (uint sb_idx = 0u; sb_idx < resolution.subband_count; ++sb_idx) {
                const J2kPacketSubband subband = subbands[resolution.subband_offset + sb_idx];
                for (uint block_idx = 0u; block_idx < subband.block_count; ++block_idx) {
                    const J2kPacketBlock block = blocks[subband.block_offset + block_idx];
                    if (block.num_coding_passes == 0u) {
                        continue;
                    }
                    if (out_len + block.data_len > params.output_capacity) {
                        j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 5u, 0u);
                        return;
                    }
                    if (!j2k_packet_push_payload_copy_job(
                            payload_copy_jobs,
                            job.payload_copy_capacity,
                            block.data_offset,
                            out_len,
                            block.data_len,
                            payload_copy_count,
                            payload_copy_bytes,
                            payload_copy_small_jobs,
                            payload_copy_medium_jobs,
                            payload_copy_large_jobs)) {
                        j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 8u, payload_copy_count);
                        return;
                    }
                    out_len += block.data_len;
                }
            }
        }
    }

    j2k_set_packet_status_payload_copy(
        status,
        J2K_ENCODE_STATUS_OK,
        payload_copy_count,
        out_len,
        payload_copy_bytes,
        payload_copy_small_jobs,
        payload_copy_medium_jobs,
        payload_copy_large_jobs
    );
}

kernel void j2k_encode_packetization_resident_classic_batched(
    device const J2kPacketResolution *all_resolutions [[buffer(0)]],
    device const J2kPacketSubband *all_subbands [[buffer(1)]],
    device const J2kResidentPacketBlock *all_resident_blocks [[buffer(2)]],
    device const uchar *payload [[buffer(3)]],
    device uchar *all_out [[buffer(4)]],
    device uchar *all_header [[buffer(5)]],
    device uint *all_tree_scratch [[buffer(6)]],
    device const J2kBatchedPacketEncodeJob *jobs [[buffer(7)]],
    device J2kPacketEncodeStatus *all_status [[buffer(8)]],
    device const J2kPacketDescriptor *all_descriptors [[buffer(9)]],
    device J2kPacketStateBlock *all_state_blocks [[buffer(10)]],
    device J2kPacketPayloadCopyJob *all_payload_copy_jobs [[buffer(11)]],
    device const J2kClassicEncodeBatchJob *tier1_jobs [[buffer(12)]],
    device const J2kClassicEncodeStatus *tier1_statuses [[buffer(13)]],
    device const J2kClassicSegment *tier1_segments [[buffer(14)]],
    constant J2kResidentPacketBlockParams &resident_params [[buffer(15)]],
    uint gid [[thread_position_in_grid]]
) {
    const J2kBatchedPacketEncodeJob job = jobs[gid];
    device const J2kPacketResolution *resolutions = all_resolutions + job.resolution_offset;
    device const J2kPacketSubband *subbands = all_subbands + job.subband_offset;
    device const J2kResidentPacketBlock *resident_blocks = all_resident_blocks + job.block_offset;
    device uchar *out = all_out + job.output_offset;
    device uchar *header = all_header + job.header_offset;
    device uint *tree_scratch = all_tree_scratch + job.scratch_offset;
    device J2kPacketEncodeStatus *status = all_status + gid;
    device const J2kPacketDescriptor *descriptors = all_descriptors + job.descriptor_offset;
    device J2kPacketStateBlock *state_blocks = all_state_blocks + job.state_block_offset;
    device J2kPacketPayloadCopyJob *payload_copy_jobs =
        all_payload_copy_jobs + job.payload_copy_offset;

    J2kPacketEncodeParams params;
    params.resolution_count = job.resolution_count;
    params.num_layers = job.num_layers;
    params.num_components = job.num_components;
    params.code_block_count = job.code_block_count;
    params.subband_count = job.subband_count;
    params.descriptor_count = job.descriptor_count;
    params.output_capacity = job.output_capacity;
    params.header_capacity = job.header_capacity;
    params.scratch_node_capacity = job.scratch_node_capacity;

    j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 0u, 0u);

    const uint node_capacity = params.scratch_node_capacity;
    device uint *inc_value = tree_scratch;
    device uint *inc_current = tree_scratch + node_capacity;
    device uint *inc_known = tree_scratch + node_capacity * 2u;
    device uint *zbp_value = tree_scratch + node_capacity * 3u;
    device uint *zbp_current = tree_scratch + node_capacity * 4u;
    device uint *zbp_known = tree_scratch + node_capacity * 5u;

    uint out_len = 0u;
    uint payload_copy_count = 0u;
    uint payload_copy_bytes = 0u;
    uint payload_copy_small_jobs = 0u;
    uint payload_copy_medium_jobs = 0u;
    uint payload_copy_large_jobs = 0u;
    const uint packet_count =
        params.descriptor_count > 0u ? params.descriptor_count : params.resolution_count;
    for (uint packet_order_idx = 0u; packet_order_idx < packet_count; ++packet_order_idx) {
        const bool has_descriptor = params.descriptor_count > 0u;
        const J2kPacketDescriptor descriptor = has_descriptor
            ? descriptors[packet_order_idx]
            : J2kPacketDescriptor{packet_order_idx, packet_order_idx, 0u, packet_order_idx, 0u, 0u, 0u, 0u};
        const uint packet_index = descriptor.packet_index;
        if (packet_index >= params.resolution_count) {
            j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 6u, 0u);
            return;
        }
        const J2kPacketResolution resolution = resolutions[packet_index];
        uint state_block_cursor = descriptor.state_block_offset;
        bool any_data = false;
        for (uint sb_idx = 0u; sb_idx < resolution.subband_count; ++sb_idx) {
            const J2kPacketSubband subband = subbands[resolution.subband_offset + sb_idx];
            for (uint block_idx = 0u; block_idx < subband.block_count; ++block_idx) {
                const J2kPacketBlock block = j2k_classic_packet_block_from_resident(
                    resident_blocks,
                    tier1_jobs,
                    tier1_statuses,
                    resident_params.tier1_job_count,
                    subband.block_offset + block_idx
                );
                if (block.num_coding_passes > 0u) {
                    any_data = true;
                    break;
                }
            }
        }

        thread J2kPacketBitWriter writer;
        j2k_packet_writer_init(writer, header, params.header_capacity);
        if (!any_data) {
            j2k_packet_write_bit(writer, 0u);
            j2k_packet_writer_finish(writer);
        } else {
            j2k_packet_write_bit(writer, 1u);
            for (uint sb_idx = 0u; sb_idx < resolution.subband_count; ++sb_idx) {
                const J2kPacketSubband subband = subbands[resolution.subband_offset + sb_idx];
                const uint subband_state_block_offset = state_block_cursor;
                state_block_cursor += subband.block_count;
	                thread uint level_offsets[16];
	                thread uint level_widths[16];
	                thread uint level_heights[16];
	                uint levels = 0u;
	                if (!j2k_packet_prepare_tree_resident_classic(
	                        resident_blocks,
	                        tier1_jobs,
	                        tier1_statuses,
	                        resident_params.tier1_job_count,
	                        subband.block_offset,
	                        subband.block_count,
	                        subband.num_cbs_x,
	                        subband.num_cbs_y,
	                        false,
	                        descriptor.layer,
	                        inc_value,
	                        inc_current,
	                        inc_known,
	                        node_capacity,
	                        level_offsets,
	                        level_widths,
	                        level_heights,
	                        levels)) {
	                    j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 1u, 0u);
	                    return;
	                }
	                thread uint z_level_offsets[16];
	                thread uint z_level_widths[16];
	                thread uint z_level_heights[16];
	                uint z_levels = 0u;
	                if (!j2k_packet_prepare_tree_resident_classic(
	                        resident_blocks,
	                        tier1_jobs,
	                        tier1_statuses,
	                        resident_params.tier1_job_count,
	                        subband.block_offset,
	                        subband.block_count,
	                        subband.num_cbs_x,
	                        subband.num_cbs_y,
	                        true,
	                        descriptor.layer,
	                        zbp_value,
	                        zbp_current,
	                        zbp_known,
	                        node_capacity,
	                        z_level_offsets,
	                        z_level_widths,
	                        z_level_heights,
	                        z_levels)) {
	                    j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 2u, 0u);
	                    return;
	                }

                for (uint block_idx = 0u; block_idx < subband.block_count; ++block_idx) {
                    const uint x = block_idx % subband.num_cbs_x;
                    const uint y = block_idx / subband.num_cbs_x;
                    const J2kPacketBlock block = j2k_classic_packet_block_from_resident(
                        resident_blocks,
                        tier1_jobs,
                        tier1_statuses,
                        resident_params.tier1_job_count,
                        subband.block_offset + block_idx
                    );
                    const uint state_block_index = subband_state_block_offset + block_idx;
                    uint previously_included = block.previously_included;
                    uint local_l_block = block.l_block;
                    if (has_descriptor) {
                        previously_included = state_blocks[state_block_index].previously_included;
                        local_l_block = state_blocks[state_block_index].l_block;
                    }
                    if (previously_included == 0u) {
                        j2k_packet_tree_encode(
                            x,
                            y,
                            descriptor.layer + 1u,
                            inc_value,
                            inc_current,
                            inc_known,
                            level_offsets,
                            level_widths,
                            levels,
                            writer
                        );
                        if (block.num_coding_passes == 0u) {
                            continue;
                        }
                        j2k_packet_tree_encode(
                            x,
                            y,
	                            block.num_zero_bitplanes + 1u,
	                            zbp_value,
	                            zbp_current,
	                            zbp_known,
	                            z_level_offsets,
	                            z_level_widths,
	                            z_levels,
	                            writer
	                        );
                    } else if (block.num_coding_passes > 0u) {
                        j2k_packet_write_bit(writer, 1u);
                    } else {
                        j2k_packet_write_bit(writer, 0u);
                        continue;
                    }

                    if (block.block_coding_mode == 0u) {
                        j2k_packet_encode_num_passes(block.num_coding_passes, writer);
                        const J2kResidentPacketBlock resident =
                            resident_blocks[subband.block_offset + block_idx];
                        if (resident.tier1_job_index >= resident_params.tier1_job_count) {
                            j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 7u, block.reserved0);
                            return;
                        }
                        const J2kClassicEncodeBatchJob tier1_job =
                            tier1_jobs[resident.tier1_job_index];
                        const J2kClassicEncodeStatus tier1_status =
                            tier1_statuses[resident.tier1_job_index];
                        if (!j2k_packet_encode_classic_segment_lengths_resident(
                                block,
                                tier1_job,
                                tier1_status,
                                tier1_segments,
                                local_l_block,
                                writer)) {
                            j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 9u, block.reserved0);
                            return;
                        }
                    } else {
                        j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 7u, block.reserved0);
                        return;
                    }
                    if (has_descriptor) {
                        state_blocks[state_block_index].previously_included = 1u;
                        state_blocks[state_block_index].l_block = local_l_block;
                    }
                }
            }
            j2k_packet_writer_finish(writer);
        }

        if (writer.failed != 0u || out_len + writer.len > params.output_capacity) {
            j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 3u, 0u);
            return;
        }
        for (uint idx = 0u; idx < writer.len; ++idx) {
            out[out_len + idx] = header[idx];
        }
        out_len += writer.len;
        if (writer.len > 0u && header[writer.len - 1u] == uchar(0xFFu)) {
            if (out_len >= params.output_capacity) {
                j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 4u, 0u);
                return;
            }
            out[out_len] = uchar(0u);
            out_len += 1u;
        }

        if (any_data) {
            for (uint sb_idx = 0u; sb_idx < resolution.subband_count; ++sb_idx) {
                const J2kPacketSubband subband = subbands[resolution.subband_offset + sb_idx];
                for (uint block_idx = 0u; block_idx < subband.block_count; ++block_idx) {
                    const J2kPacketBlock block = j2k_classic_packet_block_from_resident(
                        resident_blocks,
                        tier1_jobs,
                        tier1_statuses,
                        resident_params.tier1_job_count,
                        subband.block_offset + block_idx
                    );
                    if (block.num_coding_passes == 0u) {
                        continue;
                    }
                    if (out_len + block.data_len > params.output_capacity) {
                        j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 5u, 0u);
                        return;
                    }
                    if (!j2k_packet_push_payload_copy_job(
                            payload_copy_jobs,
                            job.payload_copy_capacity,
                            block.data_offset,
                            out_len,
                            block.data_len,
                            payload_copy_count,
                            payload_copy_bytes,
                            payload_copy_small_jobs,
                            payload_copy_medium_jobs,
                            payload_copy_large_jobs)) {
                        j2k_set_packet_status(status, J2K_ENCODE_STATUS_FAIL, 8u, payload_copy_count);
                        return;
                    }
                    out_len += block.data_len;
                }
            }
        }
    }

    j2k_set_packet_status_payload_copy(
        status,
        J2K_ENCODE_STATUS_OK,
        payload_copy_count,
        out_len,
        payload_copy_bytes,
        payload_copy_small_jobs,
        payload_copy_medium_jobs,
        payload_copy_large_jobs
    );
}

kernel void j2k_copy_packet_payload_batched(
    device const uchar *payload [[buffer(0)]],
    device uchar *all_out [[buffer(1)]],
    device const J2kBatchedPacketEncodeJob *jobs [[buffer(2)]],
    device const J2kPacketEncodeStatus *all_status [[buffer(3)]],
    device const J2kPacketPayloadCopyJob *all_payload_copy_jobs [[buffer(4)]],
    constant J2kPacketPayloadCopyParams &params [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint copy_index = gid.x;
    const uint tile = gid.y;
    const uint stripe = gid.z;
    if (stripe >= params.stripes_per_job || params.bytes_per_thread == 0u) {
        return;
    }

    const J2kPacketEncodeStatus status = all_status[tile];
    if (status.code != J2K_ENCODE_STATUS_OK || copy_index >= status.detail) {
        return;
    }

    const J2kBatchedPacketEncodeJob job = jobs[tile];
    const J2kPacketPayloadCopyJob copy_job =
        all_payload_copy_jobs[job.payload_copy_offset + copy_index];
    device uchar *out = all_out + job.output_offset + copy_job.dst_offset;
    device const uchar *src = payload + copy_job.src_offset;
    const uint stride = params.stripes_per_job * params.bytes_per_thread;
    for (uint byte_base = stripe * params.bytes_per_thread;
         byte_base < copy_job.byte_len;
         byte_base += stride) {
        const uint byte_end = min(copy_job.byte_len, byte_base + params.bytes_per_thread);
        for (uint byte_idx = byte_base; byte_idx < byte_end; ++byte_idx) {
            out[byte_idx] = src[byte_idx];
        }
    }
}
