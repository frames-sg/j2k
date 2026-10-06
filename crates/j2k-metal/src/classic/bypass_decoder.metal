inline void decode_sign_bit(
    thread J2kArithmeticDecoder &decoder,
    thread uint *contexts,
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

inline uchar plain_zero_context(uint window, uint context_shift) {
    return uchar((uint(PLAIN_ZERO_CONTEXTS[window & 0x1FFu]) >> context_shift) & 0xFu);
}

inline uchar plain_magnitude_context(uint window) {
    if ((window & J2K_CLASSIC_MU_THIS) != 0u) {
        return uchar(16u);
    }
    return uchar(14u + uint((window & J2K_CLASSIC_SIGMA_NEIGHBORS) != 0u));
}

template<typename FlagPointer>
inline void plain_or_flag(FlagPointer flags, uint index, uint bits) {
    flags[index] |= bits;
}

template<typename FlagPointer>
inline uchar2 plain_sign_context(
    FlagPointer flags,
    uint own,
    uint flag_index,
    uint row
) {
    const uint window = own >> (3u * row);
    const uint significances =
        (((window >> 1u) & 1u) << 6u) |
        (((window >> 3u) & 1u) << 4u) |
        (((window >> 5u) & 1u) << 2u) |
        ((window >> 7u) & 1u);
    const uint north_sign_bit = row == 0u ? 18u : 16u + 3u * row;
    const uint signs =
        (((own >> north_sign_bit) & 1u) << 6u) |
        (((flags[flag_index - 1u] >> (19u + 3u * row)) & 1u) << 4u) |
        (((flags[flag_index + 1u] >> (19u + 3u * row)) & 1u) << 2u) |
        ((own >> (22u + 3u * row)) & 1u);
    const uchar negative = uchar(significances & signs);
    const uchar positive = uchar(significances & ~signs);
    return SIGN_CONTEXT_LOOKUP[uchar((negative << 1u) | positive)];
}

template<typename FlagPointer>
inline void plain_set_significant(
    FlagPointer flags,
    thread uint &own,
    uint flag_index,
    uint flag_stride,
    uint flag_count,
    uint row,
    uint negative
) {
    const uint shift = 3u * row;
    own |= (J2K_CLASSIC_SIGMA_THIS | (negative << 19u)) << shift;
    plain_or_flag(flags, flag_index - 1u, (1u << 5u) << shift);
    plain_or_flag(flags, flag_index + 1u, (1u << 3u) << shift);
    if (row == 0u && flag_index >= flag_stride) {
        const uint north = flag_index - flag_stride;
        plain_or_flag(flags, north - 1u, 1u << 17u);
        plain_or_flag(flags, north, (1u << 16u) | (negative << 31u));
        plain_or_flag(flags, north + 1u, 1u << 15u);
    } else if (row == 3u && flag_index + flag_stride < flag_count) {
        const uint south = flag_index + flag_stride;
        plain_or_flag(flags, south - 1u, 1u << 2u);
        plain_or_flag(flags, south, (1u << 1u) | (negative << 18u));
        plain_or_flag(flags, south + 1u, 1u << 0u);
    }
}

template<typename FlagPointer, typename CoefficientPointer, typename ContextPointer>
inline void plain_decode_significant(
    thread J2kArithmeticDecoder &decoder,
    ContextPointer contexts,
    FlagPointer flags,
    thread uint &own,
    uint flag_index,
    uint flag_stride,
    uint flag_count,
    uint row,
    CoefficientPointer coefficients,
    uint coefficient,
    uint position
) {
    const uchar2 sign_ctx = plain_sign_context(flags, own, flag_index, row);
    const uint negative =
        arithmetic_decode_bit(decoder, contexts, uint(sign_ctx.x)) ^ uint(sign_ctx.y);
    coefficients[coefficient] = (1u << position) | (negative << 31u);
    plain_set_significant(
        flags,
        own,
        flag_index,
        flag_stride,
        flag_count,
        row,
        negative
    );
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
