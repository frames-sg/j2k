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

    thread uchar contexts[19];
    for (uint idx = 0u; idx < 19u; ++idx) {
        contexts[idx] = uchar(0);
    }
    contexts[0] = uchar(4u);
    contexts[17] = uchar(3u);
    contexts[18] = uchar(46u);

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
                reset_contexts(contexts);
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

inline bool decode_classic_job_plain(
    J2kClassicCleanupBatchJob job,
    device const uchar *coded_data,
    device const J2kClassicSegment *segments,
    device uint *coefficients_scratch,
    uint scratch_offset,
    threadgroup uchar *states,
    device float *output,
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

    const uint padded_width = job.width + J2K_CLASSIC_PADDING * 2u;
    device uint *coefficients = coefficients_scratch + scratch_offset;

    thread uchar contexts[19];
    reset_contexts(contexts);

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

        uchar zero_coded_epoch = uchar((segment.start_coding_pass + 2u) / 3u);
        for (uint coding_pass = segment.start_coding_pass; coding_pass < segment.end_coding_pass; ++coding_pass) {
            const uint current_bitplane = (coding_pass + 2u) / 3u;
            const uint current_bit_position = bitplanes - 1u - current_bitplane;
            const uint pass_type = coding_pass % 3u;

            for (uint base_row = 0u; base_row < job.height; base_row += 4u) {
                const uint stripe_end = min(base_row + 4u, job.height);
                for (uint x = 0u; x < job.width; ++x) {
                    const uint index_x = x + J2K_CLASSIC_PADDING;
                    uint index_y = base_row + J2K_CLASSIC_PADDING;
                    while (index_y < stripe_end + J2K_CLASSIC_PADDING) {
                        const uint idx = coeff_index(padded_width, index_x, index_y);
                        if (pass_type == 0u) {
                            if (coeff_is_significant_tg(states, idx) == 0u &&
                                coeff_is_zero_coded_tg(states, idx, zero_coded_epoch) == 0u) {
                                const bool use_rl =
                                    ((index_y - J2K_CLASSIC_PADDING) % 4u) == 0u &&
                                    (job.height - (index_y - J2K_CLASSIC_PADDING)) >= 4u &&
                                    neighborhood_states_plain_tg(states, padded_width, index_x, index_y) == 0u &&
                                    neighborhood_states_plain_tg(states, padded_width, index_x, index_y + 1u) == 0u &&
                                    neighborhood_states_plain_tg(states, padded_width, index_x, index_y + 2u) == 0u &&
                                    neighborhood_states_plain_tg(states, padded_width, index_x, index_y + 3u) == 0u;

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
                                        neighborhood_states_plain_tg(states, padded_width, index_x, index_y),
                                        job.sub_band_type
                                    );
                                    bit = arithmetic_decode_bit(decoder, contexts, uint(ctx_label));
                                }

                                if (bit == 1u) {
                                    coeff_push_bit(coefficients, coeff_index(padded_width, index_x, index_y), 1u, current_bit_position);
                                    decode_sign_bit_plain_tg(
                                        decoder,
                                        contexts,
                                        states,
                                        coefficients,
                                        padded_width,
                                        index_x,
                                        index_y
                                    );
                                }
                            }
                        } else if (pass_type == 1u) {
                            if (coeff_is_significant_tg(states, idx) == 0u &&
                                neighborhood_states_plain_tg(states, padded_width, index_x, index_y) != 0u) {
                                const uchar ctx_label = zero_context_label(
                                    neighborhood_states_plain_tg(states, padded_width, index_x, index_y),
                                    job.sub_band_type
                                );
                                const uint bit = arithmetic_decode_bit(decoder, contexts, uint(ctx_label));
                                coeff_set_zero_coded_marker_tg(states, idx, zero_coded_epoch);
                                if (bit == 1u) {
                                    coeff_push_bit(coefficients, idx, 1u, current_bit_position);
                                    decode_sign_bit_plain_tg(
                                        decoder,
                                        contexts,
                                        states,
                                        coefficients,
                                        padded_width,
                                        index_x,
                                        index_y
                                    );
                                }
                            }
                        } else {
                            if (coeff_is_significant_tg(states, idx) != 0u &&
                                coeff_is_zero_coded_tg(states, idx, zero_coded_epoch) == 0u) {
                                const uchar ctx_label = magnitude_refinement_context_plain_tg(
                                    states,
                                    padded_width,
                                    index_x,
                                    index_y
                                );
                                const uint bit = arithmetic_decode_bit(decoder, contexts, uint(ctx_label));
                                if (bit == 1u) {
                                    coeff_push_bit(coefficients, idx, 1u, current_bit_position);
                                }
                                coeff_set_magnitude_refined_tg(states, idx);
                            }
                        }

                        index_y += 1u;
                    }
                }
            }

            if (pass_type == 0u) {
                zero_coded_epoch = uchar(min(uint(zero_coded_epoch) + 1u, uint(J2K_STATE_MARKER_MASK)));
            }
        }
    }

    if (expected_start != job.number_of_coding_passes || expected_offset != job.coded_offset + job.coded_len) {
        set_classic_status(status, J2K_CLASSIC_STATUS_UNSUPPORTED, 8u);
        return false;
    }

    return true;
}

inline bool decode_classic_job_plain_dev(
    J2kClassicCleanupBatchJob job,
    device const uchar *coded_data,
    device const J2kClassicSegment *segments,
    device uint *coefficients_scratch,
    uint scratch_offset,
    device uchar *states_scratch,
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

    const uint padded_width = job.width + J2K_CLASSIC_PADDING * 2u;
    const uint coeff_count = padded_width * (job.height + J2K_CLASSIC_PADDING * 2u);
    device uint *coefficients = coefficients_scratch + scratch_offset;
    device uchar *states = states_scratch + scratch_offset;
    for (uint idx = 0u; idx < coeff_count; ++idx) {
        coefficients[idx] = 0u;
        states[idx] = uchar(0);
    }

    thread uchar contexts[19];
    reset_contexts(contexts);

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

        uchar zero_coded_epoch = uchar((segment.start_coding_pass + 2u) / 3u);
        for (uint coding_pass = segment.start_coding_pass; coding_pass < segment.end_coding_pass; ++coding_pass) {
            const uint current_bitplane = (coding_pass + 2u) / 3u;
            const uint current_bit_position = bitplanes - 1u - current_bitplane;
            const uint pass_type = coding_pass % 3u;

            for (uint base_row = 0u; base_row < job.height; base_row += 4u) {
                const uint stripe_end = min(base_row + 4u, job.height);
                for (uint x = 0u; x < job.width; ++x) {
                    const uint index_x = x + J2K_CLASSIC_PADDING;
                    uint index_y = base_row + J2K_CLASSIC_PADDING;
                    while (index_y < stripe_end + J2K_CLASSIC_PADDING) {
                        const uint idx = coeff_index(padded_width, index_x, index_y);
                        if (pass_type == 0u) {
                            if (coeff_is_significant_dev(states, idx) == 0u &&
                                coeff_is_zero_coded_dev(states, idx, zero_coded_epoch) == 0u) {
                                const bool use_rl =
                                    ((index_y - J2K_CLASSIC_PADDING) % 4u) == 0u &&
                                    (job.height - (index_y - J2K_CLASSIC_PADDING)) >= 4u &&
                                    neighborhood_states_plain_dev(states, padded_width, index_x, index_y) == 0u &&
                                    neighborhood_states_plain_dev(states, padded_width, index_x, index_y + 1u) == 0u &&
                                    neighborhood_states_plain_dev(states, padded_width, index_x, index_y + 2u) == 0u &&
                                    neighborhood_states_plain_dev(states, padded_width, index_x, index_y + 3u) == 0u;

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
                                        neighborhood_states_plain_dev(states, padded_width, index_x, index_y),
                                        job.sub_band_type
                                    );
                                    bit = arithmetic_decode_bit(decoder, contexts, uint(ctx_label));
                                }

                                if (bit == 1u) {
                                    coeff_push_bit(coefficients, coeff_index(padded_width, index_x, index_y), 1u, current_bit_position);
                                    decode_sign_bit_plain_dev(
                                        decoder,
                                        contexts,
                                        states,
                                        coefficients,
                                        padded_width,
                                        index_x,
                                        index_y
                                    );
                                }
                            }
                        } else if (pass_type == 1u) {
                            if (coeff_is_significant_dev(states, idx) == 0u &&
                                neighborhood_states_plain_dev(states, padded_width, index_x, index_y) != 0u) {
                                const uchar ctx_label = zero_context_label(
                                    neighborhood_states_plain_dev(states, padded_width, index_x, index_y),
                                    job.sub_band_type
                                );
                                const uint bit = arithmetic_decode_bit(decoder, contexts, uint(ctx_label));
                                coeff_set_zero_coded_marker_dev(states, idx, zero_coded_epoch);
                                if (bit == 1u) {
                                    coeff_push_bit(coefficients, idx, 1u, current_bit_position);
                                    decode_sign_bit_plain_dev(
                                        decoder,
                                        contexts,
                                        states,
                                        coefficients,
                                        padded_width,
                                        index_x,
                                        index_y
                                    );
                                }
                            }
                        } else {
                            if (coeff_is_significant_dev(states, idx) != 0u &&
                                coeff_is_zero_coded_dev(states, idx, zero_coded_epoch) == 0u) {
                                const uchar ctx_label = magnitude_refinement_context_plain_dev(
                                    states,
                                    padded_width,
                                    index_x,
                                    index_y
                                );
                                const uint bit = arithmetic_decode_bit(decoder, contexts, uint(ctx_label));
                                if (bit == 1u) {
                                    coeff_push_bit(coefficients, idx, 1u, current_bit_position);
                                }
                                coeff_set_magnitude_refined_dev(states, idx);
                            }
                        }

                        index_y += 1u;
                    }
                }
            }

            if (pass_type == 0u) {
                zero_coded_epoch = uchar(min(uint(zero_coded_epoch) + 1u, uint(J2K_STATE_MARKER_MASK)));
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

inline void store_classic_job_plain_output_tg(
    J2kClassicCleanupBatchJob job,
    device uint *coefficients_scratch,
    uint scratch_offset,
    threadgroup const uchar *states,
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
