inline uint coeff_index(uint padded_width, uint index_x, uint index_y) {
    return index_x + index_y * padded_width;
}

inline void set_classic_status(device J2kClassicStatus *status, uint code, uint detail) {
    status->code = code;
    status->detail = detail;
    status->reserved0 = 0u;
    status->reserved1 = 0u;
}

inline uchar state_bit(thread const uchar *states, uint idx, uchar shift) {
    return (states[idx] >> shift) & uchar(1u);
}

inline void set_state_bit(thread uchar *states, uint idx, uchar shift, uchar value) {
    states[idx] = uchar((states[idx] & uchar(~(1u << shift))) | ((value & 1u) << shift));
}

inline uchar state_bit_dev(device const uchar *states, uint idx, uchar shift) {
    return (states[idx] >> shift) & uchar(1u);
}

inline void set_state_bit_dev(device uchar *states, uint idx, uchar shift, uchar value) {
    states[idx] = uchar((states[idx] & uchar(~(1u << shift))) | ((value & 1u) << shift));
}

inline uchar state_bit_tg(threadgroup const uchar *states, uint idx, uchar shift) {
    return (states[idx] >> shift) & uchar(1u);
}

inline void set_state_bit_tg(threadgroup uchar *states, uint idx, uchar shift, uchar value) {
    states[idx] = uchar((states[idx] & uchar(~(1u << shift))) | ((value & 1u) << shift));
}

inline uint coeff_sign(thread const uchar *states, uint idx) {
    return uint(state_bit(states, idx, J2K_SIGN_SHIFT));
}

inline uint coeff_sign_dev(device const uchar *states, uint idx) {
    return uint(state_bit_dev(states, idx, J2K_SIGN_SHIFT));
}

inline uint coeff_sign_tg(threadgroup const uchar *states, uint idx) {
    return uint(state_bit_tg(states, idx, J2K_SIGN_SHIFT));
}

inline void coeff_push_bit(device uint *coefficients, uint idx, uint bit, uint position) {
    coefficients[idx] |= (bit << position);
}

inline void coeff_set_sign_packed(device uint *coefficients, uint idx, uint sign) {
    if (sign != 0u) {
        coefficients[idx] |= 0x80000000u;
    } else {
        coefficients[idx] &= 0x7FFFFFFFu;
    }
}

inline void coeff_set_sign(thread uchar *states, uint idx, uint sign) {
    set_state_bit(states, idx, J2K_SIGN_SHIFT, uchar(sign));
}

inline void coeff_set_sign_dev(device uchar *states, uint idx, uint sign) {
    set_state_bit_dev(states, idx, J2K_SIGN_SHIFT, uchar(sign));
}

inline void coeff_set_sign_tg(threadgroup uchar *states, uint idx, uint sign) {
    set_state_bit_tg(states, idx, J2K_SIGN_SHIFT, uchar(sign));
}

inline uchar coeff_is_significant(thread const uchar *states, uint idx) {
    return state_bit(states, idx, J2K_SIG_SHIFT);
}

inline uchar coeff_is_significant_dev(device const uchar *states, uint idx) {
    return state_bit_dev(states, idx, J2K_SIG_SHIFT);
}

inline uchar coeff_is_significant_tg(threadgroup const uchar *states, uint idx) {
    return state_bit_tg(states, idx, J2K_SIG_SHIFT);
}

inline uchar coeff_zero_coded_marker(thread const uchar *states, uint idx) {
    return states[idx] & J2K_STATE_MARKER_MASK;
}

inline uchar coeff_zero_coded_marker_dev(device const uchar *states, uint idx) {
    return states[idx] & J2K_STATE_MARKER_MASK;
}

inline uchar coeff_zero_coded_marker_tg(threadgroup const uchar *states, uint idx) {
    return states[idx] & J2K_STATE_MARKER_MASK;
}

inline uchar coeff_is_zero_coded(thread const uchar *states, uint idx, uchar marker) {
    return uchar(marker != 0u && coeff_zero_coded_marker(states, idx) == marker);
}

inline uchar coeff_is_zero_coded_dev(device const uchar *states, uint idx, uchar marker) {
    return uchar(marker != 0u && coeff_zero_coded_marker_dev(states, idx) == marker);
}

inline uchar coeff_is_zero_coded_tg(threadgroup const uchar *states, uint idx, uchar marker) {
    return uchar(marker != 0u && coeff_zero_coded_marker_tg(states, idx) == marker);
}

inline void coeff_set_zero_coded_marker(thread uchar *states, uint idx, uchar marker) {
    states[idx] = uchar((states[idx] & uchar(0xE0u)) | (marker & J2K_STATE_MARKER_MASK));
}

inline void coeff_set_zero_coded_marker_dev(device uchar *states, uint idx, uchar marker) {
    states[idx] = uchar((states[idx] & uchar(0xE0u)) | (marker & J2K_STATE_MARKER_MASK));
}

inline void coeff_set_zero_coded_marker_tg(threadgroup uchar *states, uint idx, uchar marker) {
    states[idx] = uchar((states[idx] & uchar(0xE0u)) | (marker & J2K_STATE_MARKER_MASK));
}

inline uchar coeff_is_magnitude_refined(thread const uchar *states, uint idx) {
    return state_bit(states, idx, J2K_MAG_REF_SHIFT);
}

inline uchar coeff_is_magnitude_refined_dev(device const uchar *states, uint idx) {
    return state_bit_dev(states, idx, J2K_MAG_REF_SHIFT);
}

inline uchar coeff_is_magnitude_refined_tg(threadgroup const uchar *states, uint idx) {
    return state_bit_tg(states, idx, J2K_MAG_REF_SHIFT);
}

inline void coeff_set_magnitude_refined(thread uchar *states, uint idx) {
    set_state_bit(states, idx, J2K_MAG_REF_SHIFT, uchar(1u));
}

inline void coeff_set_magnitude_refined_dev(device uchar *states, uint idx) {
    set_state_bit_dev(states, idx, J2K_MAG_REF_SHIFT, uchar(1u));
}

inline void coeff_set_magnitude_refined_tg(threadgroup uchar *states, uint idx) {
    set_state_bit_tg(states, idx, J2K_MAG_REF_SHIFT, uchar(1u));
}

inline float reconstructed_classic_sample(
    uint coefficient,
    J2kClassicCleanupBatchJob job
) {
    const uint magnitude = coefficient & 0x7FFFFFFFu;
    const uint decoded_bitplanes =
        job.total_bitplanes + job.roi_shift - job.missing_msbs;
    float reconstructed;
    if (job.irreversible_midpoint != 0u && magnitude != 0u &&
        job.number_of_coding_passes != 0u) {
        const uint final_pass = job.number_of_coding_passes - 1u;
        const uint decoded_plane = (final_pass + 2u) / 3u;
        if (decoded_bitplanes > decoded_plane) {
            uint lowest_decoded_bit = decoded_bitplanes - decoded_plane - 1u;
            if (final_pass % 3u == 1u &&
                (magnitude & (1u << lowest_decoded_bit)) == 0u) {
                lowest_decoded_bit += 1u;
            }
            uint fixed_magnitude =
                (magnitude << 1u) | (1u << lowest_decoded_bit);
            if (job.roi_shift != 0u &&
                fixed_magnitude >= (1u << job.roi_shift)) {
                fixed_magnitude >>= job.roi_shift;
            }
            reconstructed = float(fixed_magnitude) * 0.5f;
        } else {
            reconstructed = float(magnitude);
        }
    } else {
        uint reconstructed_magnitude = magnitude;
        if (job.roi_shift != 0u &&
            reconstructed_magnitude >= (1u << job.roi_shift)) {
            reconstructed_magnitude >>= job.roi_shift;
        }
        reconstructed = float(reconstructed_magnitude);
    }
    return (coefficient & 0x80000000u) != 0u ? -reconstructed : reconstructed;
}

inline bool classic_decoded_bitplanes(
    J2kClassicCleanupBatchJob job,
    thread uint &bitplanes
) {
    if (job.total_bitplanes == 0u || job.total_bitplanes > 31u ||
        job.roi_shift > 31u - job.total_bitplanes) {
        return false;
    }
    const uint coded_bitplanes = job.total_bitplanes + job.roi_shift;
    if (job.missing_msbs >= coded_bitplanes) {
        return false;
    }
    bitplanes = coded_bitplanes - job.missing_msbs;
    return true;
}

inline void reset_contexts(thread uchar *contexts) {
    for (uint idx = 0u; idx < 19u; ++idx) {
        contexts[idx] = uchar(0);
    }
    contexts[0] = uchar(4u);
    contexts[17] = uchar(3u);
    contexts[18] = uchar(46u);
}

inline uchar zero_context_label(uchar neighbors, uint sub_band_type) {
    if (sub_band_type == 1u) {
        return ZERO_CTX_HL_LOOKUP[neighbors];
    }
    if (sub_band_type == 3u) {
        return ZERO_CTX_HH_LOOKUP[neighbors];
    }
    return ZERO_CTX_LL_LH_LOOKUP[neighbors];
}

inline uchar neighborhood_states(thread const uchar *states, uint padded_width, uint index_x, uint index_y) {
    return uchar(
        (uint(coeff_is_significant(states, coeff_index(padded_width, index_x, index_y + 1u))) << 0u) |
        (uint(coeff_is_significant(states, coeff_index(padded_width, index_x + 1u, index_y + 1u))) << 1u) |
        (uint(coeff_is_significant(states, coeff_index(padded_width, index_x + 1u, index_y))) << 2u) |
        (uint(coeff_is_significant(states, coeff_index(padded_width, index_x - 1u, index_y + 1u))) << 3u) |
        (uint(coeff_is_significant(states, coeff_index(padded_width, index_x - 1u, index_y))) << 4u) |
        (uint(coeff_is_significant(states, coeff_index(padded_width, index_x + 1u, index_y - 1u))) << 5u) |
        (uint(coeff_is_significant(states, coeff_index(padded_width, index_x, index_y - 1u))) << 6u) |
        (uint(coeff_is_significant(states, coeff_index(padded_width, index_x - 1u, index_y - 1u))) << 7u)
    );
}

inline bool neighbor_in_next_stripe(uint index_y, uint height) {
    const uint real_y = index_y - J2K_CLASSIC_PADDING;
    return real_y + 1u < height && ((real_y + 1u) >> 2u) > (real_y >> 2u);
}

inline uchar effective_neighborhood_states(
    thread const uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y,
    uint height,
    uint style_flags
) {
    uchar states_mask = neighborhood_states(states, padded_width, index_x, index_y);
    if ((style_flags & J2K_CLASSIC_STYLE_VERTICALLY_CAUSAL_CONTEXT) != 0u &&
        neighbor_in_next_stripe(index_y, height)) {
        states_mask &= uchar(0b11110100);
    }
    return states_mask;
}

inline void set_significant(
    thread uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y
) {
    const uint idx = coeff_index(padded_width, index_x, index_y);
    set_state_bit(states, idx, J2K_SIG_SHIFT, uchar(1u));
}

inline uchar neighborhood_states_plain_dev(
    device const uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y
) {
    return uchar(
        (uint(coeff_is_significant_dev(states, coeff_index(padded_width, index_x, index_y + 1u))) << 0u) |
        (uint(coeff_is_significant_dev(states, coeff_index(padded_width, index_x + 1u, index_y + 1u))) << 1u) |
        (uint(coeff_is_significant_dev(states, coeff_index(padded_width, index_x + 1u, index_y))) << 2u) |
        (uint(coeff_is_significant_dev(states, coeff_index(padded_width, index_x - 1u, index_y + 1u))) << 3u) |
        (uint(coeff_is_significant_dev(states, coeff_index(padded_width, index_x - 1u, index_y))) << 4u) |
        (uint(coeff_is_significant_dev(states, coeff_index(padded_width, index_x + 1u, index_y - 1u))) << 5u) |
        (uint(coeff_is_significant_dev(states, coeff_index(padded_width, index_x, index_y - 1u))) << 6u) |
        (uint(coeff_is_significant_dev(states, coeff_index(padded_width, index_x - 1u, index_y - 1u))) << 7u)
    );
}

inline uchar neighborhood_states_plain_tg(
    threadgroup const uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y
) {
    return uchar(
        (uint(coeff_is_significant_tg(states, coeff_index(padded_width, index_x, index_y + 1u))) << 0u) |
        (uint(coeff_is_significant_tg(states, coeff_index(padded_width, index_x + 1u, index_y + 1u))) << 1u) |
        (uint(coeff_is_significant_tg(states, coeff_index(padded_width, index_x + 1u, index_y))) << 2u) |
        (uint(coeff_is_significant_tg(states, coeff_index(padded_width, index_x - 1u, index_y + 1u))) << 3u) |
        (uint(coeff_is_significant_tg(states, coeff_index(padded_width, index_x - 1u, index_y))) << 4u) |
        (uint(coeff_is_significant_tg(states, coeff_index(padded_width, index_x + 1u, index_y - 1u))) << 5u) |
        (uint(coeff_is_significant_tg(states, coeff_index(padded_width, index_x, index_y - 1u))) << 6u) |
        (uint(coeff_is_significant_tg(states, coeff_index(padded_width, index_x - 1u, index_y - 1u))) << 7u)
    );
}

inline void set_significant_plain_dev(
    device uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y
) {
    const uint idx = coeff_index(padded_width, index_x, index_y);
    set_state_bit_dev(states, idx, J2K_SIG_SHIFT, uchar(1u));
}

inline void set_significant_plain_tg(
    threadgroup uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y
) {
    const uint idx = coeff_index(padded_width, index_x, index_y);
    set_state_bit_tg(states, idx, J2K_SIG_SHIFT, uchar(1u));
}

inline uchar magnitude_refinement_context(
    thread const uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y,
    uint height,
    uint style_flags
) {
    const uint idx = coeff_index(padded_width, index_x, index_y);
    const uchar m1 = coeff_is_magnitude_refined(states, idx) * uchar(16u);
    const uchar m2 = uchar(14u + min(uint(effective_neighborhood_states(
        states,
        padded_width,
        index_x,
        index_y,
        height,
        style_flags
    )), 1u));
    return max(m1, m2);
}

inline uchar2 sign_context(
    thread const uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y,
    uint height,
    uint style_flags
) {
    const uchar significances =
        effective_neighborhood_states(
            states,
            padded_width,
            index_x,
            index_y,
            height,
            style_flags
        ) & uchar(0b01010101);
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
    return SIGN_CONTEXT_LOOKUP[uchar((negative << 1u) | positive)];
}

inline uchar2 sign_context_plain_dev(
    device const uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y
) {
    const uchar significances =
        neighborhood_states_plain_dev(states, padded_width, index_x, index_y) & uchar(0b01010101);
    const uint left_sign = coeff_sign_dev(states, coeff_index(padded_width, index_x - 1u, index_y));
    const uint right_sign = coeff_sign_dev(states, coeff_index(padded_width, index_x + 1u, index_y));
    const uint top_sign = coeff_sign_dev(states, coeff_index(padded_width, index_x, index_y - 1u));
    const uint bottom_sign = coeff_sign_dev(states, coeff_index(padded_width, index_x, index_y + 1u));
    const uchar signs = uchar((top_sign << 6u) | (left_sign << 4u) | (right_sign << 2u) | bottom_sign);
    const uchar negative = significances & signs;
    const uchar positive = significances & uchar(~signs);
    return SIGN_CONTEXT_LOOKUP[uchar((negative << 1u) | positive)];
}

inline uchar2 sign_context_plain_tg(
    threadgroup const uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y
) {
    const uchar significances =
        neighborhood_states_plain_tg(states, padded_width, index_x, index_y) & uchar(0b01010101);
    const uint left_sign = coeff_sign_tg(states, coeff_index(padded_width, index_x - 1u, index_y));
    const uint right_sign = coeff_sign_tg(states, coeff_index(padded_width, index_x + 1u, index_y));
    const uint top_sign = coeff_sign_tg(states, coeff_index(padded_width, index_x, index_y - 1u));
    const uint bottom_sign = coeff_sign_tg(states, coeff_index(padded_width, index_x, index_y + 1u));
    const uchar signs = uchar((top_sign << 6u) | (left_sign << 4u) | (right_sign << 2u) | bottom_sign);
    const uchar negative = significances & signs;
    const uchar positive = significances & uchar(~signs);
    return SIGN_CONTEXT_LOOKUP[uchar((negative << 1u) | positive)];
}
