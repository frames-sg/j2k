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
