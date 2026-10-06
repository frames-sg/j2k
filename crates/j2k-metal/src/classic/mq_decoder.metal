inline uchar next_byte(thread const J2kArithmeticDecoder &decoder) {
    return decoder.base_pointer + 1u < decoder.data_len ? decoder.data[decoder.base_pointer + 1u] : uchar(0xFF);
}

inline void arithmetic_read_byte(thread J2kArithmeticDecoder &decoder) {
    if (decoder.byte == uchar(0xFF)) {
        const uchar b1 = next_byte(decoder);
        if (b1 > uchar(0x8F)) {
            decoder.shift_count = 8u;
        } else {
            decoder.base_pointer += 1u;
            decoder.byte = b1;
            decoder.c = decoder.c + 0xFE00u - (uint(b1) << 9u);
            decoder.shift_count = 7u;
        }
    } else {
        decoder.byte = next_byte(decoder);
        decoder.base_pointer += 1u;
        decoder.c = decoder.c + 0xFF00u - (uint(decoder.byte) << 8u);
        decoder.shift_count = 8u;
    }
}

inline bool raw_read_bit(thread J2kBypassDecoder &decoder, thread uint &bit) {
    const uint byte_pos = decoder.bit_pos / 8u;
    if (byte_pos >= decoder.data_len) {
        if (decoder.strict != 0u) {
            return false;
        }
        bit = 1u;
        decoder.bit_pos += 1u;
        return true;
    }

    const uint bit_pos = decoder.bit_pos % 8u;
    bit = (uint(decoder.data[byte_pos]) >> (7u - bit_pos)) & 1u;
    decoder.bit_pos += 1u;
    return true;
}

inline bool bypass_read_bit(thread J2kBypassDecoder &decoder, thread uint &bit) {
    const uint byte_pos = decoder.bit_pos / 8u;
    const uint bit_pos = decoder.bit_pos % 8u;
    if (!raw_read_bit(decoder, bit)) {
        return false;
    }
    if (bit_pos == 7u && byte_pos < decoder.data_len && decoder.data[byte_pos] == uchar(0xFFu)) {
        uint stuffed_bit = 0u;
        if (!raw_read_bit(decoder, stuffed_bit)) {
            return decoder.strict == 0u;
        }
        if (stuffed_bit != 0u && decoder.strict != 0u) {
            return false;
        }
    }
    return true;
}

inline void arithmetic_initialize(thread J2kArithmeticDecoder &decoder) {
    decoder.byte = decoder.data_len == 0u ? uchar(0xFF) : decoder.data[0];
    decoder.c = (uint(decoder.byte ^ uchar(0xFF)) << 16u);
    arithmetic_read_byte(decoder);
    decoder.c <<= 7u;
    decoder.shift_count -= 7u;
    decoder.a = 0x8000u;
}

inline void arithmetic_renormalize(thread J2kArithmeticDecoder &decoder) {
    uint remaining = clz(decoder.a) - 16u;
    while (remaining != 0u) {
        if (decoder.shift_count == 0u) {
            arithmetic_read_byte(decoder);
        }
        const uint shift = min(remaining, decoder.shift_count);
        decoder.a <<= shift;
        decoder.c <<= shift;
        decoder.shift_count -= shift;
        remaining -= shift;
    }
}

template<typename ContextPointer>
inline uint arithmetic_decode_bit(
    thread J2kArithmeticDecoder &decoder,
    ContextPointer contexts,
    uint ctx_label
) {
    const uint context = contexts[ctx_label];
    const uint qe = context & 0xFFFFu;
    const uint mps = context >> 31u;
    decoder.a -= qe;

    const bool lower = (decoder.c >> 16u) < decoder.a;
    if (lower && (decoder.a & 0x8000u) != 0u) {
        return mps;
    }

    const uint lps = uint(lower == (decoder.a < qe));
    const uint decoded = mps ^ lps;
    if (!lower) {
        decoder.c -= decoder.a << 16u;
        decoder.a = qe;
    }

    const uint next_state = lps != 0u
        ? ((context >> 22u) & 0x3Fu)
        : ((context >> 16u) & 0x3Fu);
    const uint next_mps = mps ^ (lps & ((context >> 28u) & 1u));
    contexts[ctx_label] = J2K_PACKED_QE_TABLE[next_state] | (next_mps << 31u);
    const uint shifts_needed = clz(decoder.a) - 16u;
    if (shifts_needed <= decoder.shift_count) {
        decoder.a <<= shifts_needed;
        decoder.c <<= shifts_needed;
        decoder.shift_count -= shifts_needed;
    } else {
        arithmetic_renormalize(decoder);
    }
    return decoded;
}
