inline bool decode_classic_job(
    J2kClassicCleanupBatchJob job,
    device const uchar *coded_data,
    device const J2kClassicSegment *segments,
    device uint *coefficients_scratch,
    uint scratch_offset,
    device float *output,
    bool store_output,
    device J2kClassicStatus *status
) {
    if (job.width == 0u || job.height == 0u) {
        return true;
    }
    if (job.width > J2K_CLASSIC_MAX_WIDTH || job.height > J2K_CLASSIC_MAX_HEIGHT) {
        set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 0u);
        return false;
    }
    uint bitplanes = 0u;
    if (!classic_decoded_bitplanes(job, bitplanes)) {
        set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 1u);
        return false;
    }

    const uint max_coding_passes = bitplanes == 0u ? 0u : 1u + 3u * (bitplanes - 1u);
    if (job.coded_len == 0u || max_coding_passes == 0u || job.number_of_coding_passes == 0u) {
        return true;
    }
    if (job.number_of_coding_passes > max_coding_passes) {
        set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 2u);
        return false;
    }
    const uint padded_width = job.width + J2K_CLASSIC_PADDING * 2u;
    const uint padded_height = job.height + J2K_CLASSIC_PADDING * 2u;
    const uint coeff_count = padded_width * padded_height;

    device uint *coefficients = coefficients_scratch + scratch_offset;
    thread uchar states[J2K_CLASSIC_MAX_COEFF_COUNT];
    for (uint idx = 0u; idx < coeff_count; ++idx) {
        coefficients[idx] = 0u;
        states[idx] = uchar(0);
    }

    thread uint contexts[19];
    reset_decode_contexts(contexts);

    if (job.segment_count == 0u) {
        set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 3u);
        return false;
    }

    const ulong coded_begin = ulong(job.coded_offset);
    const ulong coded_end = coded_begin + ulong(job.coded_len);
    uint expected_start = 0u;
    uint expected_offset = job.coded_offset;
    for (uint segment_idx = 0u; segment_idx < job.segment_count; ++segment_idx) {
        const J2kClassicSegment segment = segments[job.segment_offset + segment_idx];
        if (segment.start_coding_pass != expected_start || segment.start_coding_pass > segment.end_coding_pass) {
            set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 4u);
            return false;
        }
        if (segment.data_offset != expected_offset) {
            set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 6u);
            return false;
        }
        const ulong segment_end = ulong(segment.data_offset) + ulong(segment.data_length);
        if (ulong(segment.data_offset) < coded_begin || segment_end > coded_end) {
            set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 7u);
            return false;
        }
        expected_start = segment.end_coding_pass;
        expected_offset = segment.data_offset + segment.data_length;

        if (segment.start_coding_pass == segment.end_coding_pass) {
            continue;
        }

        J2kArithmeticDecoder decoder;
        J2kBypassDecoder bypass_decoder;
        const bool use_arithmetic = segment.use_arithmetic != 0u;
        if (use_arithmetic) {
            decoder.data = coded_data + segment.data_offset;
            decoder.data_len = segment.data_length;
            decoder.c = 0u;
            decoder.a = 0u;
            decoder.base_pointer = 0u;
            decoder.shift_count = 0u;
            arithmetic_initialize(decoder);
        } else {
            bypass_decoder.data = coded_data + segment.data_offset;
            bypass_decoder.data_len = segment.data_length;
            bypass_decoder.bit_pos = 0u;
            bypass_decoder.strict = job.strict;
        }

        uchar zero_coded_epoch = uchar((segment.start_coding_pass + 2u) / 3u);
        for (uint coding_pass = segment.start_coding_pass; coding_pass < segment.end_coding_pass; ++coding_pass) {
            const uint current_bitplane = (coding_pass + 2u) / 3u;
            const uint current_bit_position = bitplanes - 1u - current_bitplane;
            const uint pass_type = coding_pass % 3u;

            for (uint base_row = 0u; base_row < job.height; base_row += 4u) {
                const uint stripe_end = min(base_row + 4u, job.height);
                for (uint x = 0u; x < job.width; ++x) {
                    uint index_x = x + J2K_CLASSIC_PADDING;
                    uint index_y = base_row + J2K_CLASSIC_PADDING;
                    while (index_y < stripe_end + J2K_CLASSIC_PADDING) {
                        const uint idx = coeff_index(padded_width, index_x, index_y);
                        if (pass_type == 0u) {
                            if (!use_arithmetic) {
                                set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 5u);
                                return false;
                            }
                            if (coeff_is_significant(states, idx) == 0u &&
                                coeff_is_zero_coded(states, idx, zero_coded_epoch) == 0u) {
                                const bool use_rl =
                                    ((index_y - J2K_CLASSIC_PADDING) % 4u) == 0u &&
                                    (job.height - (index_y - J2K_CLASSIC_PADDING)) >= 4u &&
                                    effective_neighborhood_states(states, padded_width, index_x, index_y, job.height, job.style_flags) == 0u &&
                                    effective_neighborhood_states(states, padded_width, index_x, index_y + 1u, job.height, job.style_flags) == 0u &&
                                    effective_neighborhood_states(states, padded_width, index_x, index_y + 2u, job.height, job.style_flags) == 0u &&
                                    effective_neighborhood_states(states, padded_width, index_x, index_y + 3u, job.height, job.style_flags) == 0u;

                                uint bit = 0u;
                                if (use_rl) {
                                    bit = arithmetic_decode_bit(decoder, contexts, 17u);
                                    if (bit == 0u) {
                                        index_y += 4u;
                                        continue;
                                    }

                                    uint num_zeroes = arithmetic_decode_bit(decoder, contexts, 18u);
                                    num_zeroes = (num_zeroes << 1u) | arithmetic_decode_bit(decoder, contexts, 18u);
                                    index_y += num_zeroes;
                                } else {
                                    const uchar ctx_label = zero_context_label(
                                        effective_neighborhood_states(
                                            states,
                                            padded_width,
                                            index_x,
                                            index_y,
                                            job.height,
                                            job.style_flags
                                        ),
                                        job.sub_band_type
                                    );
                                    bit = arithmetic_decode_bit(decoder, contexts, uint(ctx_label));
                                }

                                if (bit == 1u) {
                                    coeff_push_bit(coefficients, coeff_index(padded_width, index_x, index_y), 1u, current_bit_position);
                                    decode_sign_bit(
                                        decoder,
                                        contexts,
                                        states,
                                        coefficients,
                                        padded_width,
                                        index_x,
                                        index_y,
                                        job.height,
                                        job.style_flags
                                    );
                                }
                            }
                        } else if (pass_type == 1u) {
                            if (coeff_is_significant(states, idx) == 0u &&
                                effective_neighborhood_states(
                                    states,
                                    padded_width,
                                    index_x,
                                    index_y,
                                    job.height,
                                    job.style_flags
                                ) != 0u) {
                                const uchar ctx_label = zero_context_label(
                                    effective_neighborhood_states(
                                        states,
                                        padded_width,
                                        index_x,
                                        index_y,
                                        job.height,
                                        job.style_flags
                                    ),
                                    job.sub_band_type
                                );
                                uint bit = 0u;
                                if (use_arithmetic) {
                                    bit = arithmetic_decode_bit(decoder, contexts, uint(ctx_label));
                                } else if (!bypass_read_bit(bypass_decoder, bit)) {
                                    set_classic_status(status, J2K_CLASSIC_STATUS_FAIL, 11u);
                                    return false;
                                }
                                coeff_set_zero_coded_marker(states, idx, zero_coded_epoch);
                                if (bit == 1u) {
                                    coeff_push_bit(coefficients, idx, 1u, current_bit_position);
                                    if (use_arithmetic) {
                                        decode_sign_bit(
                                            decoder,
                                            contexts,
                                            states,
                                            coefficients,
                                            padded_width,
                                            index_x,
                                            index_y,
                                            job.height,
                                            job.style_flags
                                        );
                                    } else if (!decode_sign_bit_bypass(
                                        bypass_decoder,
                                        states,
                                        coefficients,
                                        padded_width,
                                        index_x,
                                        index_y
                                    )) {
                                        set_classic_status(status, J2K_CLASSIC_STATUS_FAIL, 12u);
                                        return false;
                                    }
                                }
                            }
                        } else {
                            if (coeff_is_significant(states, idx) != 0u &&
                                coeff_is_zero_coded(states, idx, zero_coded_epoch) == 0u) {
                                const uchar ctx_label = magnitude_refinement_context(
                                    states,
                                    padded_width,
                                    index_x,
                                    index_y,
                                    job.height,
                                    job.style_flags
                                );
                                uint bit = 0u;
                                if (use_arithmetic) {
                                    bit = arithmetic_decode_bit(decoder, contexts, uint(ctx_label));
                                } else if (!bypass_read_bit(bypass_decoder, bit)) {
                                    set_classic_status(status, J2K_CLASSIC_STATUS_FAIL, 13u);
                                    return false;
                                }
                                if (bit == 1u) {
                                    coeff_push_bit(coefficients, idx, 1u, current_bit_position);
                                }
                                coeff_set_magnitude_refined(states, idx);
                            }
                        }

                        index_y += 1u;
                    }
                }
            }

            if (pass_type == 0u) {
                if ((job.style_flags & J2K_CLASSIC_STYLE_SEGMENTATION_SYMBOLS) != 0u) {
                    const uint b0 = arithmetic_decode_bit(decoder, contexts, 18u);
                    const uint b1 = arithmetic_decode_bit(decoder, contexts, 18u);
                    const uint b2 = arithmetic_decode_bit(decoder, contexts, 18u);
                    const uint b3 = arithmetic_decode_bit(decoder, contexts, 18u);
                    if ((b0 != 1u || b1 != 0u || b2 != 1u || b3 != 0u) && job.strict != 0u) {
                        set_classic_status(status, J2K_CLASSIC_STATUS_FAIL, 10u);
                        return false;
                    }
                }
                zero_coded_epoch = uchar(min(uint(zero_coded_epoch) + 1u, uint(J2K_STATE_MARKER_MASK)));
            }

            if ((job.style_flags & J2K_CLASSIC_STYLE_RESET_CONTEXT_PROBABILITIES) != 0u) {
                reset_decode_contexts(contexts);
            }
        }
    }

    if (expected_start != job.number_of_coding_passes || expected_offset != job.coded_offset + job.coded_len) {
        set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 8u);
        return false;
    }

    if (store_output) {
        for (uint y = 0u; y < job.height; ++y) {
            const uint output_row = job.output_offset + y * job.output_stride;
            for (uint x = 0u; x < job.width; ++x) {
                const uint coeff =
                    coefficients[coeff_index(padded_width, x + J2K_CLASSIC_PADDING, y + J2K_CLASSIC_PADDING)];
                output[output_row + x] =
                    reconstructed_classic_sample(coeff, job) * job.dequantization_step;
            }
        }
    }

    return true;
}

template<uint PassType, typename FlagPointer, typename CoefficientPointer, typename ContextPointer>
inline void decode_classic_job_plain_row(
    thread J2kArithmeticDecoder &decoder,
    ContextPointer contexts,
    CoefficientPointer coefficients,
    FlagPointer flags,
    thread uint &own,
    uint flag_index,
    uint flag_stride,
    uint flag_count,
    uint row,
    uint top_coefficient,
    uint padded_width,
    uint zero_context_shift,
    uint current_bit_position
) {
    const uint shift = 3u * row;
    const uint window = own >> shift;
    const uint coefficient = top_coefficient + row * padded_width;
    if (PassType == 0u) {
        if ((window & (J2K_CLASSIC_SIGMA_THIS | J2K_CLASSIC_PI_THIS)) == 0u) {
            const uchar ctx_label = plain_zero_context(window, zero_context_shift);
            if (arithmetic_decode_bit(decoder, contexts, uint(ctx_label)) != 0u) {
                plain_decode_significant(
                    decoder,
                    contexts,
                    flags,
                    own,
                    flag_index,
                    flag_stride,
                    flag_count,
                    row,
                    coefficients,
                    coefficient,
                    current_bit_position
                );
            }
        }
    } else if (PassType == 1u) {
        if ((window & (J2K_CLASSIC_SIGMA_THIS | J2K_CLASSIC_PI_THIS)) == 0u &&
            (window & J2K_CLASSIC_SIGMA_NEIGHBORS) != 0u) {
            const uchar ctx_label = plain_zero_context(window, zero_context_shift);
            if (arithmetic_decode_bit(decoder, contexts, uint(ctx_label)) != 0u) {
                plain_decode_significant(
                    decoder,
                    contexts,
                    flags,
                    own,
                    flag_index,
                    flag_stride,
                    flag_count,
                    row,
                    coefficients,
                    coefficient,
                    current_bit_position
                );
            }
            own |= J2K_CLASSIC_PI_THIS << shift;
        }
    } else if (
        (window & (J2K_CLASSIC_SIGMA_THIS | J2K_CLASSIC_PI_THIS)) ==
            J2K_CLASSIC_SIGMA_THIS
    ) {
        const uint bit = arithmetic_decode_bit(
            decoder,
            contexts,
            uint(plain_magnitude_context(window))
        );
        if (bit != 0u) {
            coefficients[coefficient] |= 1u << current_bit_position;
        }
        own |= J2K_CLASSIC_MU_THIS << shift;
    }
}

template<uint PassType, bool Full64, typename FlagPointer, typename CoefficientPointer, typename ContextPointer>
inline void decode_classic_job_plain_pass(
    J2kClassicCleanupBatchJob job,
    thread J2kArithmeticDecoder &decoder,
    ContextPointer contexts,
    CoefficientPointer coefficients,
    FlagPointer flags,
    uint padded_width,
    uint flag_stride,
    uint stripe_count,
    uint flag_count,
    uint zero_context_shift,
    uint current_bit_position
) {
    const uint pass_width = Full64 ? J2K_CLASSIC_MAX_WIDTH : job.width;
    const uint pass_height = Full64 ? J2K_CLASSIC_MAX_HEIGHT : job.height;
    const uint pass_padded_width = Full64 ? J2K_CLASSIC_MAX_PADDED_WIDTH : padded_width;
    const uint pass_flag_stride = Full64 ? J2K_CLASSIC_MAX_PADDED_WIDTH : flag_stride;
    const uint pass_stripe_count = Full64 ? (J2K_CLASSIC_MAX_HEIGHT / 4u) : stripe_count;
    const uint pass_flag_count = Full64 ? J2K_CLASSIC_MAX_FLAG_WORDS : flag_count;
    for (uint stripe = 0u; stripe < pass_stripe_count; ++stripe) {
        const uint base_row = 4u * stripe;
        const uint rows = Full64 ? 4u : min(pass_height - base_row, 4u);
        for (uint x = 0u; x < pass_width; ++x) {
            const uint flag_index = stripe * pass_flag_stride + x + 1u;
            uint own = flags[flag_index];
            const uint initial_own = own;
            const uint own_sigma = own & J2K_CLASSIC_SIGMA_ROWS;
            const uint top_coefficient =
                coeff_index(pass_padded_width, x + J2K_CLASSIC_PADDING, base_row + J2K_CLASSIC_PADDING);

            if (PassType == 1u &&
                (own == 0u || (rows == 4u && own_sigma == J2K_CLASSIC_SIGMA_ROWS))) {
                continue;
            }
            if (PassType == 2u && own_sigma == 0u) {
                continue;
            }
            if (PassType == 0u && rows == 4u && own_sigma == J2K_CLASSIC_SIGMA_ROWS) {
                own &= ~J2K_CLASSIC_PI_ALL;
                if (own != initial_own) {
                    flags[flag_index] = own;
                }
                continue;
            }

            if (PassType == 0u && rows == 4u && own == 0u) {
                const uint bit = arithmetic_decode_bit(decoder, contexts, 17u);
                if (bit != 0u) {
                    const uint first =
                        (arithmetic_decode_bit(decoder, contexts, 18u) << 1u) |
                        arithmetic_decode_bit(decoder, contexts, 18u);
                    plain_decode_significant(
                        decoder,
                        contexts,
                        flags,
                        own,
                        flag_index,
                        pass_flag_stride,
                        pass_flag_count,
                        first,
                        coefficients,
                        top_coefficient + first * pass_padded_width,
                        current_bit_position
                    );
                    for (uint row = first + 1u; row < 4u; ++row) {
                        const uint window = own >> (3u * row);
                        const uchar ctx_label = plain_zero_context(window, zero_context_shift);
                        if (arithmetic_decode_bit(decoder, contexts, uint(ctx_label)) != 0u) {
                            plain_decode_significant(
                                decoder,
                                contexts,
                                flags,
                                own,
                                flag_index,
                                pass_flag_stride,
                                pass_flag_count,
                                row,
                                coefficients,
                                top_coefficient + row * pass_padded_width,
                                current_bit_position
                            );
                        }
                    }
                }
                if (own != initial_own) {
                    flags[flag_index] = own;
                }
                continue;
            }

            if (PassType == 2u) {
                #pragma clang loop unroll(full)
                for (uint row = 0u; row < 4u; ++row) {
                    if (row < rows) {
                        decode_classic_job_plain_row<PassType>(
                            decoder,
                            contexts,
                            coefficients,
                            flags,
                            own,
                            flag_index,
                            pass_flag_stride,
                            pass_flag_count,
                            row,
                            top_coefficient,
                            pass_padded_width,
                            zero_context_shift,
                            current_bit_position
                        );
                    }
                }
            } else {
                for (uint row = 0u; row < rows; ++row) {
                    decode_classic_job_plain_row<PassType>(
                        decoder,
                        contexts,
                        coefficients,
                        flags,
                        own,
                        flag_index,
                        pass_flag_stride,
                        pass_flag_count,
                        row,
                        top_coefficient,
                        pass_padded_width,
                        zero_context_shift,
                        current_bit_position
                    );
                }
            }

            if (PassType == 0u) {
                own &= ~J2K_CLASSIC_PI_ALL;
            }
            if (own != initial_own) {
                flags[flag_index] = own;
            }
        }
    }
}

template<uint PassType, typename FlagPointer, typename CoefficientPointer, typename ContextPointer>
inline void decode_classic_job_plain_pass_geometry(
    J2kClassicCleanupBatchJob job,
    thread J2kArithmeticDecoder &decoder,
    ContextPointer contexts,
    CoefficientPointer coefficients,
    FlagPointer flags,
    uint padded_width,
    uint flag_stride,
    uint stripe_count,
    uint flag_count,
    uint zero_context_shift,
    uint current_bit_position
) {
    if (job.width == J2K_CLASSIC_MAX_WIDTH && job.height == J2K_CLASSIC_MAX_HEIGHT) {
        decode_classic_job_plain_pass<PassType, true>(
            job,
            decoder,
            contexts,
            coefficients,
            flags,
            padded_width,
            flag_stride,
            stripe_count,
            flag_count,
            zero_context_shift,
            current_bit_position
        );
    } else {
        decode_classic_job_plain_pass<PassType, false>(
            job,
            decoder,
            contexts,
            coefficients,
            flags,
            padded_width,
            flag_stride,
            stripe_count,
            flag_count,
            zero_context_shift,
            current_bit_position
        );
    }
}

template<typename FlagPointer, typename CoefficientPointer, typename ContextPointer>
inline bool decode_classic_job_plain_flags_with_contexts(
    J2kClassicCleanupBatchJob job,
    device const uchar *coded_data,
    device const J2kClassicSegment *segments,
    CoefficientPointer coefficients,
    FlagPointer flags,
    ContextPointer contexts,
    bool initialize_storage,
    device float *output,
    bool store_output,
    device J2kClassicStatus *status
) {
    if (job.width == 0u || job.height == 0u) {
        return true;
    }
    if (job.style_flags != 0u) {
        set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 12u);
        return false;
    }
    if (job.width > J2K_CLASSIC_MAX_WIDTH || job.height > J2K_CLASSIC_MAX_HEIGHT) {
        set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 0u);
        return false;
    }
    uint bitplanes = 0u;
    if (!classic_decoded_bitplanes(job, bitplanes)) {
        set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 1u);
        return false;
    }

    const uint max_coding_passes = bitplanes == 0u ? 0u : 1u + 3u * (bitplanes - 1u);
    if (job.coded_len == 0u || max_coding_passes == 0u || job.number_of_coding_passes == 0u) {
        return true;
    }
    if (job.number_of_coding_passes > max_coding_passes) {
        set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 2u);
        return false;
    }

    const uint zero_context_shift = job.sub_band_type == 1u ? 4u : (job.sub_band_type == 3u ? 8u : 0u);
    const uint padded_width = job.width + J2K_CLASSIC_PADDING * 2u;
    const uint coeff_count = padded_width * (job.height + J2K_CLASSIC_PADDING * 2u);
    const uint flag_stride = job.width + 2u;
    const uint stripe_count = (job.height + 3u) / 4u;
    const uint flag_count = flag_stride * stripe_count;
    if (initialize_storage) {
        for (uint idx = 0u; idx < coeff_count; ++idx) {
            coefficients[idx] = 0u;
        }
        for (uint idx = 0u; idx < flag_count; ++idx) {
            flags[idx] = 0u;
        }
    }

    reset_decode_contexts(contexts);

    if (job.segment_count == 0u) {
        set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 3u);
        return false;
    }

    const ulong coded_begin = ulong(job.coded_offset);
    const ulong coded_end = coded_begin + ulong(job.coded_len);
    uint expected_start = 0u;
    uint expected_offset = job.coded_offset;
    for (uint segment_idx = 0u; segment_idx < job.segment_count; ++segment_idx) {
        const J2kClassicSegment segment = segments[job.segment_offset + segment_idx];
        if (segment.use_arithmetic == 0u) {
            set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 5u);
            return false;
        }
        if (segment.start_coding_pass != expected_start || segment.start_coding_pass > segment.end_coding_pass) {
            set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 4u);
            return false;
        }
        if (segment.data_offset != expected_offset) {
            set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 6u);
            return false;
        }
        const ulong segment_end = ulong(segment.data_offset) + ulong(segment.data_length);
        if (ulong(segment.data_offset) < coded_begin || segment_end > coded_end) {
            set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 7u);
            return false;
        }
        expected_start = segment.end_coding_pass;
        expected_offset = segment.data_offset + segment.data_length;

        if (segment.start_coding_pass == segment.end_coding_pass) {
            continue;
        }

        J2kArithmeticDecoder decoder;
        decoder.data = coded_data + segment.data_offset;
        decoder.data_len = segment.data_length;
        decoder.c = 0u;
        decoder.a = 0u;
        decoder.base_pointer = 0u;
        decoder.shift_count = 0u;
        arithmetic_initialize(decoder);

        for (uint coding_pass = segment.start_coding_pass; coding_pass < segment.end_coding_pass; ++coding_pass) {
            const uint current_bitplane = (coding_pass + 2u) / 3u;
            const uint current_bit_position = bitplanes - 1u - current_bitplane;
            const uint pass_type = coding_pass % 3u;

            switch (pass_type) {
            case 0u:
                decode_classic_job_plain_pass_geometry<0u>(
                    job,
                    decoder,
                    contexts,
                    coefficients,
                    flags,
                    padded_width,
                    flag_stride,
                    stripe_count,
                    flag_count,
                    zero_context_shift,
                    current_bit_position
                );
                break;
            case 1u:
                decode_classic_job_plain_pass_geometry<1u>(
                    job,
                    decoder,
                    contexts,
                    coefficients,
                    flags,
                    padded_width,
                    flag_stride,
                    stripe_count,
                    flag_count,
                    zero_context_shift,
                    current_bit_position
                );
                break;
            default:
                decode_classic_job_plain_pass_geometry<2u>(
                    job,
                    decoder,
                    contexts,
                    coefficients,
                    flags,
                    padded_width,
                    flag_stride,
                    stripe_count,
                    flag_count,
                    zero_context_shift,
                    current_bit_position
                );
                break;
            }
        }
    }

    if (expected_start != job.number_of_coding_passes || expected_offset != job.coded_offset + job.coded_len) {
        set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 8u);
        return false;
    }

    if (store_output) {
        for (uint y = 0u; y < job.height; ++y) {
            const uint output_row = job.output_offset + y * job.output_stride;
            for (uint x = 0u; x < job.width; ++x) {
                const uint coeff =
                    coefficients[coeff_index(padded_width, x + J2K_CLASSIC_PADDING, y + J2K_CLASSIC_PADDING)];
                output[output_row + x] =
                    reconstructed_classic_sample(coeff, job) * job.dequantization_step;
            }
        }
    }

    return true;
}

template<typename FlagPointer, typename CoefficientPointer>
inline bool decode_classic_job_plain_flags(
    J2kClassicCleanupBatchJob job,
    device const uchar *coded_data,
    device const J2kClassicSegment *segments,
    CoefficientPointer coefficients,
    FlagPointer flags,
    bool initialize_storage,
    device float *output,
    bool store_output,
    device J2kClassicStatus *status
) {
    thread uint contexts[19];
    return decode_classic_job_plain_flags_with_contexts(
        job,
        coded_data,
        segments,
        coefficients,
        flags,
        contexts,
        initialize_storage,
        output,
        store_output,
        status
    );
}

inline bool decode_classic_job_plain(
    J2kClassicCleanupBatchJob job,
    device const uchar *coded_data,
    device const J2kClassicSegment *segments,
    device uint *coefficients_scratch,
    uint scratch_offset,
    threadgroup uint *flags,
    device float *output,
    device J2kClassicStatus *status
) {
    device uint *coefficients = coefficients_scratch + scratch_offset;
    return decode_classic_job_plain_flags(
        job,
        coded_data,
        segments,
        coefficients,
        flags,
        false,
        output,
        false,
        status
    );
}

inline bool decode_classic_job_plain_dev(
    J2kClassicCleanupBatchJob job,
    device const uchar *coded_data,
    device const J2kClassicSegment *segments,
    device uint *coefficients_scratch,
    uint scratch_offset,
    device uint *flags_scratch,
    device float *output,
    bool store_output,
    device J2kClassicStatus *status
) {
    device uint *coefficients = coefficients_scratch + scratch_offset;
    device uint *flags = flags_scratch + scratch_offset / 4u;
    return decode_classic_job_plain_flags(
        job,
        coded_data,
        segments,
        coefficients,
        flags,
        true,
        output,
        store_output,
        status
    );
}

inline bool decode_classic_job_plain_dense(
    J2kClassicCleanupBatchJob job,
    device const uchar *coded_data,
    device const J2kClassicSegment *segments,
    device uint *coefficients,
    device uint *flags,
    J2kPlainThreadgroupSoaContexts contexts,
    device float *output,
    device J2kClassicStatus *status
) {
    return decode_classic_job_plain_flags_with_contexts(
        job,
        coded_data,
        segments,
        coefficients,
        flags,
        contexts,
        false,
        output,
        false,
        status
    );
}

inline void store_classic_job_plain_output_tg(
    J2kClassicCleanupBatchJob job,
    device uint *coefficients_scratch,
    uint scratch_offset,
    device float *output,
    uint lane
) {
    const uint padded_width = job.width + J2K_CLASSIC_PADDING * 2u;
    device uint *coefficients = coefficients_scratch + scratch_offset;
    const uint sample_count = job.width * job.height;
    for (uint sample_idx = lane; sample_idx < sample_count; sample_idx += 32u) {
        const uint x = sample_idx % job.width;
        const uint y = sample_idx / job.width;
        const uint coeff_idx =
            coeff_index(padded_width, x + J2K_CLASSIC_PADDING, y + J2K_CLASSIC_PADDING);
        const uint coeff = coefficients[coeff_idx];
        output[job.output_offset + y * job.output_stride + x] =
            reconstructed_classic_sample(coeff, job) * job.dequantization_step;
    }
}
