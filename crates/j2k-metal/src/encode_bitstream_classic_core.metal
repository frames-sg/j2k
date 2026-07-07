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
