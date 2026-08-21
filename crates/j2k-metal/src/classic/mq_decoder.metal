inline uchar current_byte(thread const J2kArithmeticDecoder &decoder) {
    return decoder.base_pointer < decoder.data_len ? decoder.data[decoder.base_pointer] : uchar(0xFF);
}

inline uchar next_byte(thread const J2kArithmeticDecoder &decoder) {
    return decoder.base_pointer + 1u < decoder.data_len ? decoder.data[decoder.base_pointer + 1u] : uchar(0xFF);
}

inline void arithmetic_read_byte(thread J2kArithmeticDecoder &decoder) {
    if (current_byte(decoder) == uchar(0xFF)) {
        const uchar b1 = next_byte(decoder);
        if (b1 > uchar(0x8F)) {
            decoder.shift_count = 8u;
        } else {
            decoder.base_pointer += 1u;
            decoder.c = decoder.c + 0xFE00u - (uint(current_byte(decoder)) << 9u);
            decoder.shift_count = 7u;
        }
    } else {
        decoder.base_pointer += 1u;
        decoder.c = decoder.c + 0xFF00u - (uint(current_byte(decoder)) << 8u);
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
    decoder.c = (uint(current_byte(decoder) ^ uchar(0xFF)) << 16u);
    arithmetic_read_byte(decoder);
    decoder.c <<= 7u;
    decoder.shift_count -= 7u;
    decoder.a = 0x8000u;
}

inline void arithmetic_renormalize(thread J2kArithmeticDecoder &decoder) {
    while ((decoder.a & 0x8000u) == 0u) {
        if (decoder.shift_count == 0u) {
            arithmetic_read_byte(decoder);
        }
        decoder.a <<= 1u;
        decoder.c <<= 1u;
        decoder.shift_count -= 1u;
    }
}

inline uint arithmetic_decode_bit(thread J2kArithmeticDecoder &decoder, thread uchar *contexts, uint ctx_label) {
    uchar ctx = contexts[ctx_label];
    const J2kQeData qe = J2K_QE_TABLE[ctx & uchar(0x7F)];
    decoder.a -= qe.qe;

    if ((decoder.c >> 16u) < decoder.a) {
        if ((decoder.a & 0x8000u) != 0u) {
            return uint(ctx >> 7u);
        }

        uint d;
        if (decoder.a < qe.qe) {
            d = uint((ctx >> 7u) ^ 1u);
            if (qe.switch_mps != 0u) {
                ctx ^= uchar(0x80);
            }
            ctx = uchar((ctx & 0x80u) | qe.nlps);
        } else {
            d = uint(ctx >> 7u);
            ctx = uchar((ctx & 0x80u) | qe.nmps);
        }
        contexts[ctx_label] = ctx;
        arithmetic_renormalize(decoder);
        return d;
    }

    decoder.c -= decoder.a << 16u;

    uint d;
    if (decoder.a < qe.qe) {
        decoder.a = qe.qe;
        d = uint(ctx >> 7u);
        ctx = uchar((ctx & 0x80u) | qe.nmps);
    } else {
        decoder.a = qe.qe;
        d = uint((ctx >> 7u) ^ 1u);
        if (qe.switch_mps != 0u) {
            ctx ^= uchar(0x80);
        }
        ctx = uchar((ctx & 0x80u) | qe.nlps);
    }
    contexts[ctx_label] = ctx;
    arithmetic_renormalize(decoder);
    return d;
}
