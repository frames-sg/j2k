inline void decode_sign_bit(
    thread J2kArithmeticDecoder &decoder,
    thread uchar *contexts,
    thread uchar *states,
    device uint *coefficients,
    uint padded_width,
    uint index_x,
    uint index_y,
    uint height,
    uint style_flags
) {
    const uchar2 sign_ctx = sign_context(
        states,
        padded_width,
        index_x,
        index_y,
        height,
        style_flags
    );
    const uint sign_bit = arithmetic_decode_bit(decoder, contexts, uint(sign_ctx.x)) ^ uint(sign_ctx.y);
    const uint idx = coeff_index(padded_width, index_x, index_y);
    coeff_set_sign(states, idx, sign_bit);
    coeff_set_sign_packed(coefficients, idx, sign_bit);
    set_significant(states, padded_width, index_x, index_y);
}

inline void decode_sign_bit_plain_dev(
    thread J2kArithmeticDecoder &decoder,
    thread uchar *contexts,
    device uchar *states,
    device uint *coefficients,
    uint padded_width,
    uint index_x,
    uint index_y
) {
    const uchar2 sign_ctx = sign_context_plain_dev(states, padded_width, index_x, index_y);
    const uint sign_bit = arithmetic_decode_bit(decoder, contexts, uint(sign_ctx.x)) ^ uint(sign_ctx.y);
    const uint idx = coeff_index(padded_width, index_x, index_y);
    coeff_set_sign_dev(states, idx, sign_bit);
    coeff_set_sign_packed(coefficients, idx, sign_bit);
    set_significant_plain_dev(states, padded_width, index_x, index_y);
}

inline void decode_sign_bit_plain_tg(
    thread J2kArithmeticDecoder &decoder,
    thread uchar *contexts,
    threadgroup uchar *states,
    device uint *coefficients,
    uint padded_width,
    uint index_x,
    uint index_y
) {
    const uchar2 sign_ctx = sign_context_plain_tg(states, padded_width, index_x, index_y);
    const uint sign_bit = arithmetic_decode_bit(decoder, contexts, uint(sign_ctx.x)) ^ uint(sign_ctx.y);
    const uint idx = coeff_index(padded_width, index_x, index_y);
    coeff_set_sign_tg(states, idx, sign_bit);
    coeff_set_sign_packed(coefficients, idx, sign_bit);
    set_significant_plain_tg(states, padded_width, index_x, index_y);
}

inline uchar magnitude_refinement_context_plain_dev(
    device const uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y
) {
    const uint idx = coeff_index(padded_width, index_x, index_y);
    const uchar m1 = coeff_is_magnitude_refined_dev(states, idx) * uchar(16u);
    const uchar m2 = uchar(14u + min(uint(neighborhood_states_plain_dev(
        states,
        padded_width,
        index_x,
        index_y
    )), 1u));
    return max(m1, m2);
}

inline uchar magnitude_refinement_context_plain_tg(
    threadgroup const uchar *states,
    uint padded_width,
    uint index_x,
    uint index_y
) {
    const uint idx = coeff_index(padded_width, index_x, index_y);
    const uchar m1 = coeff_is_magnitude_refined_tg(states, idx) * uchar(16u);
    const uchar m2 = uchar(14u + min(uint(neighborhood_states_plain_tg(
        states,
        padded_width,
        index_x,
        index_y
    )), 1u));
    return max(m1, m2);
}

inline bool decode_sign_bit_bypass(
    thread J2kBypassDecoder &decoder,
    thread uchar *states,
    device uint *coefficients,
    uint padded_width,
    uint index_x,
    uint index_y
) {
    uint sign_bit = 0u;
    if (!bypass_read_bit(decoder, sign_bit)) {
        return false;
    }
    const uint idx = coeff_index(padded_width, index_x, index_y);
    coeff_set_sign(states, idx, sign_bit);
    coeff_set_sign_packed(coefficients, idx, sign_bit);
    set_significant(states, padded_width, index_x, index_y);
    return true;
}
