// SPDX-License-Identifier: MIT OR Apache-2.0

// HT SigProp/MagRef segment writers shared by the single-block and batch kernels.

constant uint J2K_HT_SIGPROP_SCRATCH = 513u;
constant uint J2K_HT_SIGPROP_SPREAD_MASKS[16] = {
    0x33u, 0x76u, 0xECu, 0xC8u, 0x330u, 0x760u, 0xEC0u, 0xC80u,
    0x3300u, 0x7600u, 0xEC00u, 0xC800u, 0x33000u, 0x76000u, 0xEC000u, 0xC8000u
};

struct J2kHtSigPropWriter {
    uint pos;
    uint used_bits;
    uint previous_was_ff;
    uint capacity;
    uchar tmp;
    uint failed;
};

inline uint j2k_ht_sigprop_spread_mask(uint bit) {
    return bit < 16u ? J2K_HT_SIGPROP_SPREAD_MASKS[bit] : 0u;
}

inline void j2k_ht_sigprop_writer_init(
    thread J2kHtSigPropWriter &writer,
    uint capacity
) {
    writer.pos = 0u;
    writer.used_bits = 0u;
    writer.previous_was_ff = 0u;
    writer.capacity = capacity;
    writer.tmp = uchar(0u);
    writer.failed = 0u;
}

inline void j2k_ht_sigprop_write_bit(
    thread J2kHtSigPropWriter &writer,
    device uchar *out,
    uint bit,
    bool write_output
) {
    const uint max_bits = writer.previous_was_ff != 0u ? 7u : 8u;
    writer.tmp |= uchar((bit & 1u) << writer.used_bits);
    writer.used_bits += 1u;
    if (writer.used_bits < max_bits) {
        return;
    }
    if (writer.pos >= writer.capacity) {
        writer.failed = 1u;
        return;
    }
    if (write_output) {
        out[writer.pos] = writer.tmp;
    }
    writer.previous_was_ff = writer.tmp == uchar(0xFFu) ? 1u : 0u;
    writer.tmp = uchar(0u);
    writer.used_bits = 0u;
    writer.pos += 1u;
}

inline void j2k_ht_sigprop_finish(
    thread J2kHtSigPropWriter &writer,
    device uchar *out,
    bool write_output
) {
    if (writer.used_bits == 0u) {
        return;
    }
    if (writer.pos >= writer.capacity) {
        writer.failed = 1u;
        return;
    }
    if (write_output) {
        out[writer.pos] = writer.tmp;
    }
    writer.pos += 1u;
    writer.tmp = uchar(0u);
    writer.used_bits = 0u;
}

inline uint j2k_ht_sigprop_cleanup_sig16(
    device const int *coefficients,
    uint coefficient_stride,
    uint width,
    uint height,
    uint x_base,
    uint y_base,
    uint cleanup_threshold
) {
    uint mask = 0u;
    for (uint col = 0u; col < 4u; ++col) {
        const uint x = x_base + col;
        if (x >= width) {
            continue;
        }
        for (uint row = 0u; row < 4u; ++row) {
            const uint y = y_base + row;
            if (y < height &&
                j2k_classic_magnitude(coefficients[y * coefficient_stride + x]) >= cleanup_threshold) {
                mask |= 1u << (col * 4u + row);
            }
        }
    }
    return mask;
}

inline uint j2k_ht_sigprop_target_sig16(
    device const int *coefficients,
    uint coefficient_stride,
    uint width,
    uint height,
    uint x_base,
    uint y_base,
    uint cleanup_threshold,
    uint refinement_mask
) {
    uint mask = 0u;
    for (uint col = 0u; col < 4u; ++col) {
        const uint x = x_base + col;
        if (x >= width) {
            continue;
        }
        for (uint row = 0u; row < 4u; ++row) {
            const uint y = y_base + row;
            if (y >= height) {
                continue;
            }
            const uint magnitude = j2k_classic_magnitude(coefficients[y * coefficient_stride + x]);
            if (magnitude < cleanup_threshold && (magnitude & refinement_mask) != 0u) {
                mask |= 1u << (col * 4u + row);
            }
        }
    }
    return mask;
}

inline uint j2k_ht_sigprop_coefficient_sign(
    device const int *coefficients,
    uint coefficient_stride,
    uint x_base,
    uint y_base,
    uint bit
) {
    const uint col = bit >> 2u;
    const uint row = bit & 3u;
    return coefficients[(y_base + row) * coefficient_stride + x_base + col] < 0 ? 1u : 0u;
}

inline uint j2k_ht_write_sigprop_segment(
    device const int *coefficients,
    uint coefficient_stride,
    uint width,
    uint height,
    uint cleanup_threshold,
    uint refinement_mask,
    device uchar *out,
    uint capacity,
    thread uint &bytes_written,
    bool write_output
) {
    const uint group_count = (width + 3u) >> 2u;
    if (group_count + 8u > J2K_HT_SIGPROP_SCRATCH) {
        return 0u;
    }
    thread ushort prev_row_sig[J2K_HT_SIGPROP_SCRATCH];
    for (uint idx = 0u; idx < group_count + 2u; ++idx) {
        prev_row_sig[idx] = ushort(0u);
    }
    thread J2kHtSigPropWriter writer;
    j2k_ht_sigprop_writer_init(writer, capacity);

    for (uint y = 0u; y < height; y += 4u) {
        uint pattern = 0xFFFFu;
        if (height - y < 4u) {
            pattern = 0x7777u;
            if (height - y < 3u) {
                pattern = 0x3333u;
                if (height - y < 2u) {
                    pattern = 0x1111u;
                }
            }
        }
        uint prev = 0u;
        for (uint x = 0u; x < width; x += 4u) {
            uint col_pattern = pattern;
            if (x + 4u > width) {
                col_pattern >>= (x + 4u - width) * 4u;
            }
            const uint idx = x >> 2u;
            const uint ps = uint(prev_row_sig[idx]) | (uint(prev_row_sig[idx + 1u]) << 16u);
            const uint ns = j2k_ht_sigprop_cleanup_sig16(
                coefficients, coefficient_stride, width, height, x, y + 4u, cleanup_threshold)
                | (j2k_ht_sigprop_cleanup_sig16(
                    coefficients, coefficient_stride, width, height,
                    x + 4u, y + 4u, cleanup_threshold) << 16u);
            uint u = (ps & 0x88888888u) >> 3u;
            u |= (ns & 0x11111111u) << 3u;
            const uint cs = j2k_ht_sigprop_cleanup_sig16(
                coefficients, coefficient_stride, width, height, x, y, cleanup_threshold)
                | (j2k_ht_sigprop_cleanup_sig16(
                    coefficients, coefficient_stride, width, height,
                    x + 4u, y, cleanup_threshold) << 16u);
            uint mbr = cs;
            mbr |= (cs & 0x77777777u) << 1u;
            mbr |= (cs & 0xEEEEEEEEu) >> 1u;
            mbr |= u;
            const uint t_mbr = mbr;
            mbr |= t_mbr << 4u;
            mbr |= t_mbr >> 4u;
            mbr |= prev >> 12u;
            mbr &= col_pattern;
            mbr &= ~cs;

            uint new_sig = 0u;
            const uint target_sig = j2k_ht_sigprop_target_sig16(
                coefficients, coefficient_stride, width, height, x, y,
                cleanup_threshold, refinement_mask) & col_pattern;
            if (mbr != 0u) {
                uint candidates = mbr;
                uint processed = 0u;
                const uint inv_sig = ~cs & col_pattern;
                while (candidates != 0u) {
                    const uint bit = ctz(candidates);
                    const uint sample_mask = 1u << bit;
                    candidates &= ~sample_mask;
                    processed |= sample_mask;
                    const uint desired = (target_sig & sample_mask) != 0u ? 1u : 0u;
                    j2k_ht_sigprop_write_bit(writer, out, desired, write_output);
                    if (writer.failed != 0u) {
                        return 0u;
                    }
                    if (desired != 0u) {
                        new_sig |= sample_mask;
                        candidates |= j2k_ht_sigprop_spread_mask(bit) & inv_sig & ~processed;
                    }
                }
                uint sign_bits = new_sig;
                while (sign_bits != 0u) {
                    const uint bit = ctz(sign_bits);
                    const uint sample_mask = 1u << bit;
                    sign_bits &= ~sample_mask;
                    j2k_ht_sigprop_write_bit(
                        writer, out,
                        j2k_ht_sigprop_coefficient_sign(
                            coefficients, coefficient_stride, x, y, bit),
                        write_output);
                    if (writer.failed != 0u) {
                        return 0u;
                    }
                }
            }
            const uint combined_sig = new_sig | cs;
            prev_row_sig[idx] = ushort(combined_sig & 0xFFFFu);
            prev_row_sig[idx + 1u] = ushort((combined_sig >> 16u) & 0xFFFFu);
            const uint t = combined_sig;
            uint next_prev = combined_sig;
            next_prev |= (t & 0x7777u) << 1u;
            next_prev |= (t & 0xEEEEu) >> 1u;
            prev = (next_prev | u) & 0xF000u;
        }
    }
    j2k_ht_sigprop_finish(writer, out, write_output);
    if (writer.failed != 0u) {
        return 0u;
    }
    bytes_written = writer.pos;
    return 1u;
}

inline uint j2k_ht_write_magref_segment(
    device const int *coefficients,
    uint coefficient_stride,
    uint width,
    uint height,
    uint cleanup_threshold,
    uint refinement_mask,
    device uchar *out,
    uint capacity,
    uint expected_bits,
    thread uint &bytes_written,
    bool write_output
) {
    uint bit_idx = 0u;
    uint byte_from_end = 0u;
    uint used_bits = 0u;
    uint unstuff = 1u;
    uchar current = uchar(0u);
    for (uint y = 0u; y < height; y += 4u) {
        for (uint x_base = 0u; x_base < width; x_base += 8u) {
            for (uint col = 0u; col < 8u; ++col) {
                const uint x = x_base + col;
                if (x >= width) {
                    continue;
                }
                for (uint row = 0u; row < 4u; ++row) {
                    const uint yy = y + row;
                    if (yy >= height) {
                        continue;
                    }
                    const uint magnitude =
                        j2k_classic_magnitude(coefficients[yy * coefficient_stride + x]);
                    if (magnitude < cleanup_threshold) {
                        continue;
                    }
                    current |= uchar(uint((magnitude & refinement_mask) != 0u) << used_bits);
                    used_bits += 1u;
                    bit_idx += 1u;
                    const bool stuffed =
                        unstuff != 0u && used_bits == 7u && (current & uchar(0x7Fu)) == uchar(0x7Fu);
                    if (stuffed || used_bits == 8u) {
                        if (byte_from_end >= capacity) {
                            return 0u;
                        }
                        if (write_output) {
                            out[capacity - 1u - byte_from_end] = current;
                        }
                        byte_from_end += 1u;
                        unstuff = current > uchar(0x8Fu) ? 1u : 0u;
                        current = uchar(0u);
                        used_bits = 0u;
                    }
                }
            }
        }
    }
    if (used_bits != 0u) {
        if (byte_from_end >= capacity) {
            return 0u;
        }
        if (write_output) {
            out[capacity - 1u - byte_from_end] = current;
        }
        byte_from_end += 1u;
    }
    bytes_written = byte_from_end;
    return bit_idx == expected_bits ? 1u : 0u;
}
