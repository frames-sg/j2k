// SPDX-License-Identifier: MIT OR Apache-2.0

#include <metal_stdlib>
using namespace metal;

struct JpegPackParams {
    uint width;
    uint height;
    uint out_stride;
    uint alpha;
    uint mode;
    uint out_format;
};

struct JpegBaselineEncodeParams {
    uint input_offset_bytes;
    uint input_width;
    uint input_height;
    uint output_width;
    uint output_height;
    uint pitch_bytes;
    uint mcus_per_row;
    uint mcu_rows;
    uint restart_interval_mcus;
    uint format;
    uint components;
    uint max_h;
    uint max_v;
    uint h0;
    uint v0;
    uint h1;
    uint v1;
    uint h2;
    uint v2;
    uint entropy_offset_bytes;
    uint entropy_capacity;
};

struct JpegBaselineEncodeHuffmanTable {
    ushort codes[256];
    uchar lens[256];
};

struct JpegBaselineEncodeStatus {
    uint code;
    uint entropy_len;
    uint detail;
    uint reserved;
};

struct JpegBaselineBitWriter {
    uint pos;
    uchar current;
    uint used;
    bool overflow;
};

struct JpegFast420Params {
    uint width;
    uint height;
    uint chroma_width;
    uint chroma_height;
    uint mcus_per_row;
    uint mcu_rows;
    uint restart_interval_mcus;
    uint restart_offset_count;
    uint restart_start_mcu;
    uint entropy_len;
    uint out_stride;
    uint alpha;
    uint out_format;
    uint origin_x;
    uint origin_y;
};

struct JpegFast420ScaledParams {
    uint scaled_width;
    uint scaled_height;
    uint chroma_width;
    uint chroma_height;
    uint mcus_per_row;
    uint mcu_rows;
    uint restart_interval_mcus;
    uint restart_offset_count;
    uint restart_start_mcu;
    uint entropy_len;
    uint scale_shift;
    uint origin_x;
    uint origin_y;
};

struct JpegFast444Params {
    uint width;
    uint height;
    uint mcus_per_row;
    uint mcu_rows;
    uint restart_interval_mcus;
    uint restart_offset_count;
    uint restart_start_mcu;
    uint entropy_len;
    uint origin_x;
    uint origin_y;
};

struct JpegFast444ScaledParams {
    uint scaled_width;
    uint scaled_height;
    uint mcus_per_row;
    uint mcu_rows;
    uint restart_interval_mcus;
    uint restart_offset_count;
    uint restart_start_mcu;
    uint entropy_len;
    uint scale_shift;
    uint origin_x;
    uint origin_y;
};

struct JpegFast420WindowedPackParams {
    uint src_width;
    uint src_height;
    uint chroma_width;
    uint chroma_height;
    uint src_x;
    uint src_y;
    uint width;
    uint height;
    uint out_stride;
    uint alpha;
    uint out_format;
};

struct JpegFast420BatchParams {
    uint width;
    uint height;
    uint chroma_width;
    uint chroma_height;
    uint mcus_per_row;
    uint mcu_rows;
    uint segment_count;
    uint tile_count;
    uint out_stride;
    uint alpha;
};

struct JpegFastRegionScaledBatchParams {
    uint scaled_width;
    uint scaled_height;
    uint chroma_width;
    uint chroma_height;
    uint mcus_per_row;
    uint mcu_rows;
    uint segment_count;
    uint tile_count;
    uint scale_shift;
    uint origin_x;
    uint origin_y;
};

struct JpegFast444TextureBatchParams {
    uint width;
    uint height;
    uint mcus_per_row;
    uint mcu_rows;
    uint segment_count;
    uint tile_index;
    uint alpha;
    uint mode;
};

struct JpegFast422TextureBatchParams {
    uint width;
    uint height;
    uint chroma_width;
    uint chroma_height;
    uint mcus_per_row;
    uint mcu_rows;
    uint segment_count;
    uint tile_index;
    uint alpha;
};

struct JpegFast420TextureBatchParams {
    uint width;
    uint height;
    uint chroma_width;
    uint chroma_height;
    uint mcus_per_row;
    uint mcu_rows;
    uint segment_count;
    uint tile_index;
    uint alpha;
};

struct JpegWindowedPackBatchParams {
    uint src_width;
    uint src_height;
    uint chroma_width;
    uint chroma_height;
    uint src_x;
    uint src_y;
    uint width;
    uint height;
    uint tile_count;
    uint out_stride;
    uint alpha;
    uint mode;
    uint out_format;
};

struct JpegWindowedTexturePackBatchParams {
    uint src_width;
    uint src_height;
    uint chroma_width;
    uint chroma_height;
    uint src_x;
    uint src_y;
    uint width;
    uint height;
    uint tile_index;
    uint alpha;
};

struct JpegTexturePackBatchParams {
    uint width;
    uint height;
    uint chroma_width;
    uint chroma_height;
    uint tile_index;
    uint alpha;
    uint mode;
};

struct JpegRgb8ToRgbaTextureParams {
    uint width;
    uint height;
    uint in_stride;
    uint alpha;
};

struct JpegDecodeStatus {
    uint code;
    uint detail;
    uint position;
    uint reserved;
};

struct JpegEntropyCheckpoint {
    uint mcu_index;
    uint entropy_pos;
    ulong bit_acc;
    uint bit_count;
    int y_prev_dc;
    int cb_prev_dc;
    int cr_prev_dc;
    uint reserved;
    uint reserved_tail;
};

struct JpegHuffmanTable {
    uchar bits[16];
    ushort values_len;
    ushort reserved;
    uchar values[256];
};

struct PreparedHuffman {
    int min_code[17];
    int max_code[17];
    int val_offset[17];
    uchar values[256];
    uchar fast_symbol[512];
    uchar fast_len[512];
    ushort values_len;
};

struct BitReader {
    uint pos;
    ulong acc;
    uint bits;
};

constant uint MODE_GRAY = 0;
constant uint MODE_YCBCR = 1;
constant uint MODE_RGB = 2;

constant uint OUT_GRAY = 0;
constant uint OUT_RGB = 1;
constant uint OUT_RGBA = 2;

constant uint JPEG_BASELINE_ENCODE_FORMAT_GRAY8 = 0;
constant uint JPEG_BASELINE_ENCODE_FORMAT_RGB8 = 1;
constant uint JPEG_BASELINE_ENCODE_STATUS_OK = 0;
constant uint JPEG_BASELINE_ENCODE_STATUS_OVERFLOW = 1;
constant uint JPEG_BASELINE_ENCODE_STATUS_MISSING_HUFFMAN = 2;
constant uint JPEG_BASELINE_ENCODE_STATUS_INVALID_PARAMS = 3;

constant uint FAST420_STATUS_OK = 0;
constant uint FAST420_STATUS_TRUNCATED = 1;
constant uint FAST420_STATUS_HUFFMAN = 2;

inline void init_decode_status(device JpegDecodeStatus *status) {
    status->code = FAST420_STATUS_OK;
    status->detail = 0;
    status->position = 0;
    status->reserved = 0;
}

constant ushort ZIGZAG[64] = {
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

constant int CONST_BITS = 13;
constant int PASS1_BITS = 2;

constant int FIX_0_211164243 = 1730;
constant int FIX_0_298631336 = 2446;
constant int FIX_0_390180644 = 3196;
constant int FIX_0_509795579 = 4176;
constant int FIX_0_541196100 = 4433;
constant int FIX_0_601344887 = 4926;
constant int FIX_0_720959822 = 5906;
constant int FIX_0_765366865 = 6270;
constant int FIX_0_850430095 = 6967;
constant int FIX_0_899976223 = 7373;
constant int FIX_1_061594337 = 8697;
constant int FIX_1_175875602 = 9633;
constant int FIX_1_272758580 = 10426;
constant int FIX_1_451774981 = 11893;
constant int FIX_1_501321110 = 12299;
constant int FIX_1_847759065 = 15137;
constant int FIX_1_961570560 = 16069;
constant int FIX_2_053119869 = 16819;
constant int FIX_2_172734803 = 17799;
constant int FIX_2_562915447 = 20995;
constant int FIX_3_072711026 = 25172;
constant int FIX_3_624509785 = 29692;

inline uchar clamp_u8(int value) {
    return uchar(clamp(value, 0, 255));
}

inline short clamp_i16(int value) {
    return short(clamp(value, int(short(-32768)), int(short(32767))));
}

inline uint component_h(constant JpegBaselineEncodeParams &params, uint component) {
    if (component == 0u) {
        return params.h0;
    }
    if (component == 1u) {
        return params.h1;
    }
    return params.h2;
}

inline uint component_v(constant JpegBaselineEncodeParams &params, uint component) {
    if (component == 0u) {
        return params.v0;
    }
    if (component == 1u) {
        return params.v1;
    }
    return params.v2;
}

inline int round_to_int(float value) {
    return value >= 0.0f ? int(value + 0.5f) : int(value - 0.5f);
}

inline uchar rgb_to_ycbcr_component(uchar3 rgb, uint component) {
    const int r = int(rgb.x);
    const int g = int(rgb.y);
    const int b = int(rgb.z);
    if (component == 0u) {
        return clamp_u8((19595 * r + 38470 * g + 7471 * b + 32768) >> 16);
    }
    if (component == 1u) {
        return clamp_u8((-11059 * r - 21709 * g + 32768 * b + 8421376) >> 16);
    }
    return clamp_u8((32768 * r - 27439 * g - 5329 * b + 8421376) >> 16);
}

inline uchar3 jpeg_encode_read_rgb(
    device const uchar *input,
    constant JpegBaselineEncodeParams &params,
    uint x,
    uint y
) {
    if (x >= params.input_width || y >= params.input_height) {
        return uchar3(0, 0, 0);
    }
    const uint offset = y * params.pitch_bytes + x * 3u;
    return uchar3(input[offset], input[offset + 1u], input[offset + 2u]);
}

inline uchar jpeg_encode_sample_component(
    device const uchar *input,
    constant JpegBaselineEncodeParams &params,
    uint component,
    uint x,
    uint y
) {
    if (params.format == JPEG_BASELINE_ENCODE_FORMAT_GRAY8) {
        if (x >= params.input_width || y >= params.input_height) {
            return 0;
        }
        return input[y * params.pitch_bytes + x];
    }
    return rgb_to_ycbcr_component(jpeg_encode_read_rgb(input, params, x, y), component);
}

inline void jpeg_encode_sample_block(
    device const uchar *input,
    constant JpegBaselineEncodeParams &params,
    uint component,
    uint mcu_x,
    uint mcu_y,
    uint block_x,
    uint block_y,
    thread uchar block[64]
) {
    const uint comp_h = component_h(params, component);
    const uint comp_v = component_v(params, component);
    const uint x_scale = params.max_h / comp_h;
    const uint y_scale = params.max_v / comp_v;
    const uint mcu_origin_x = mcu_x * params.max_h * 8u;
    const uint mcu_origin_y = mcu_y * params.max_v * 8u;

    for (uint y = 0u; y < 8u; y++) {
        for (uint x = 0u; x < 8u; x++) {
            uchar value;
            if (component == 0u || params.components == 1u) {
                const uint sx = min(mcu_origin_x + block_x * 8u + x, params.output_width - 1u);
                const uint sy = min(mcu_origin_y + block_y * 8u + y, params.output_height - 1u);
                value = jpeg_encode_sample_component(input, params, component, sx, sy);
            } else {
                uint sum = 0u;
                for (uint dy = 0u; dy < y_scale; dy++) {
                    for (uint dx = 0u; dx < x_scale; dx++) {
                        const uint sx = min(
                            mcu_origin_x + (block_x * 8u + x) * x_scale + dx,
                            params.output_width - 1u
                        );
                        const uint sy = min(
                            mcu_origin_y + (block_y * 8u + y) * y_scale + dy,
                            params.output_height - 1u
                        );
                        sum += uint(jpeg_encode_sample_component(input, params, component, sx, sy));
                    }
                }
                value = uchar(sum / (x_scale * y_scale));
            }
            block[y * 8u + x] = value;
        }
    }
}

inline void jpeg_encode_fdct_quantize(
    thread const uchar block[64],
    constant uchar *quant,
    thread int coeffs[64]
) {
    constexpr float pi = 3.14159265358979323846f;
    constexpr float inv_sqrt_2 = 0.70710678118654752440f;
    for (uint v = 0u; v < 8u; v++) {
        for (uint u = 0u; u < 8u; u++) {
            float sum = 0.0f;
            for (uint y = 0u; y < 8u; y++) {
                for (uint x = 0u; x < 8u; x++) {
                    const float sample = float(block[y * 8u + x]) - 128.0f;
                    const float cx = cos(((float(2u * x + 1u) * float(u) * pi) / 16.0f));
                    const float cy = cos(((float(2u * y + 1u) * float(v) * pi) / 16.0f));
                    sum += sample * cx * cy;
                }
            }
            const float cu = (u == 0u) ? inv_sqrt_2 : 1.0f;
            const float cv = (v == 0u) ? inv_sqrt_2 : 1.0f;
            const uint natural = v * 8u + u;
            const float transformed = 0.25f * cu * cv * sum;
            coeffs[natural] = round_to_int(transformed / float(quant[natural]));
        }
    }
}

inline void jpeg_encode_push_raw_byte(
    device uchar *entropy,
    uint capacity,
    thread JpegBaselineBitWriter &writer,
    uchar byte
) {
    if (writer.pos >= capacity) {
        writer.overflow = true;
        return;
    }
    entropy[writer.pos] = byte;
    writer.pos += 1u;
}

inline void jpeg_encode_push_data_byte(
    device uchar *entropy,
    uint capacity,
    thread JpegBaselineBitWriter &writer,
    uchar byte
) {
    jpeg_encode_push_raw_byte(entropy, capacity, writer, byte);
    if (!writer.overflow && byte == 0xff) {
        jpeg_encode_push_raw_byte(entropy, capacity, writer, 0x00);
    }
}

inline void jpeg_encode_write_bits(
    device uchar *entropy,
    uint capacity,
    thread JpegBaselineBitWriter &writer,
    ushort code,
    uint len
) {
    for (int bit = int(len) - 1; bit >= 0; bit--) {
        const uchar value = uchar((code >> uint(bit)) & 1u);
        writer.current = uchar((writer.current << 1u) | value);
        writer.used += 1u;
        if (writer.used == 8u) {
            jpeg_encode_push_data_byte(entropy, capacity, writer, writer.current);
            writer.current = 0;
            writer.used = 0u;
            if (writer.overflow) {
                return;
            }
        }
    }
}

inline void jpeg_encode_align_with_ones(
    device uchar *entropy,
    uint capacity,
    thread JpegBaselineBitWriter &writer
) {
    if (writer.used == 0u) {
        return;
    }
    const uint remaining = 8u - writer.used;
    writer.current = uchar((writer.current << remaining) | uchar((1u << remaining) - 1u));
    jpeg_encode_push_data_byte(entropy, capacity, writer, writer.current);
    writer.current = 0;
    writer.used = 0u;
}

inline void jpeg_encode_push_restart_marker(
    device uchar *entropy,
    uint capacity,
    thread JpegBaselineBitWriter &writer,
    uint rst
) {
    jpeg_encode_align_with_ones(entropy, capacity, writer);
    if (writer.overflow) {
        return;
    }
    jpeg_encode_push_raw_byte(entropy, capacity, writer, 0xff);
    jpeg_encode_push_raw_byte(entropy, capacity, writer, uchar(0xd0u + (rst & 0x07u)));
}

inline uint jpeg_encode_magnitude_category(int value) {
    if (value == 0) {
        return 0u;
    }
    uint abs_value = value < 0 ? uint(-value) : uint(value);
    uint size = 0u;
    while (abs_value > 0u) {
        size += 1u;
        abs_value >>= 1u;
    }
    return size;
}

inline ushort jpeg_encode_magnitude_bits(int value, uint size) {
    if (size == 0u) {
        return 0;
    }
    if (value >= 0) {
        return ushort(value);
    }
    return ushort(value + int((1u << size) - 1u));
}

inline bool jpeg_encode_write_symbol(
    device uchar *entropy,
    uint capacity,
    thread JpegBaselineBitWriter &writer,
    constant JpegBaselineEncodeHuffmanTable &table,
    uint symbol,
    device JpegBaselineEncodeStatus *status
) {
    const uint len = uint(table.lens[symbol]);
    if (len == 0u) {
        status->code = JPEG_BASELINE_ENCODE_STATUS_MISSING_HUFFMAN;
        status->detail = symbol;
        return false;
    }
    jpeg_encode_write_bits(entropy, capacity, writer, table.codes[symbol], len);
    if (writer.overflow) {
        status->code = JPEG_BASELINE_ENCODE_STATUS_OVERFLOW;
        return false;
    }
    return true;
}

inline bool jpeg_encode_block(
    thread const int coeffs[64],
    thread int &prev_dc,
    constant JpegBaselineEncodeHuffmanTable &dc_table,
    constant JpegBaselineEncodeHuffmanTable &ac_table,
    device uchar *entropy,
    uint capacity,
    thread JpegBaselineBitWriter &writer,
    device JpegBaselineEncodeStatus *status
) {
    const int diff = coeffs[0] - prev_dc;
    prev_dc = coeffs[0];
    const uint dc_size = jpeg_encode_magnitude_category(diff);
    if (!jpeg_encode_write_symbol(entropy, capacity, writer, dc_table, dc_size, status)) {
        return false;
    }
    if (dc_size > 0u) {
        jpeg_encode_write_bits(
            entropy,
            capacity,
            writer,
            jpeg_encode_magnitude_bits(diff, dc_size),
            dc_size
        );
        if (writer.overflow) {
            status->code = JPEG_BASELINE_ENCODE_STATUS_OVERFLOW;
            return false;
        }
    }

    uint zero_run = 0u;
    for (uint k = 1u; k < 64u; k++) {
        const int coeff = coeffs[ZIGZAG[k]];
        if (coeff == 0) {
            zero_run += 1u;
            continue;
        }
        while (zero_run >= 16u) {
            if (!jpeg_encode_write_symbol(entropy, capacity, writer, ac_table, 0xf0u, status)) {
                return false;
            }
            zero_run -= 16u;
        }
        const uint size = jpeg_encode_magnitude_category(coeff);
        const uint symbol = (zero_run << 4u) | size;
        if (!jpeg_encode_write_symbol(entropy, capacity, writer, ac_table, symbol, status)) {
            return false;
        }
        jpeg_encode_write_bits(
            entropy,
            capacity,
            writer,
            jpeg_encode_magnitude_bits(coeff, size),
            size
        );
        if (writer.overflow) {
            status->code = JPEG_BASELINE_ENCODE_STATUS_OVERFLOW;
            return false;
        }
        zero_run = 0u;
    }
    if (zero_run > 0u) {
        return jpeg_encode_write_symbol(entropy, capacity, writer, ac_table, 0u, status);
    }
    return true;
}

inline void jpeg_encode_baseline_entropy_one(
    device const uchar *input,
    device uchar *entropy,
    device JpegBaselineEncodeStatus *status,
    constant JpegBaselineEncodeParams &params,
    constant uchar *q_luma,
    constant uchar *q_chroma,
    constant JpegBaselineEncodeHuffmanTable &dc_luma,
    constant JpegBaselineEncodeHuffmanTable &ac_luma,
    constant JpegBaselineEncodeHuffmanTable &dc_chroma,
    constant JpegBaselineEncodeHuffmanTable &ac_chroma
) {
    status->code = JPEG_BASELINE_ENCODE_STATUS_OK;
    status->entropy_len = 0u;
    status->detail = 0u;

    if (
        params.input_width == 0u ||
        params.input_height == 0u ||
        params.output_width == 0u ||
        params.output_height == 0u ||
        params.mcus_per_row == 0u ||
        params.mcu_rows == 0u ||
        params.max_h == 0u ||
        params.max_v == 0u ||
        params.h0 == 0u ||
        params.v0 == 0u
    ) {
        status->code = JPEG_BASELINE_ENCODE_STATUS_INVALID_PARAMS;
        return;
    }

    thread JpegBaselineBitWriter writer;
    writer.pos = 0u;
    writer.current = 0;
    writer.used = 0u;
    writer.overflow = false;
    thread int prev_dc[3] = {0, 0, 0};
    uint mcus_since_restart = 0u;
    uint rst = 0u;

    for (uint mcu_y = 0u; mcu_y < params.mcu_rows; mcu_y++) {
        for (uint mcu_x = 0u; mcu_x < params.mcus_per_row; mcu_x++) {
            if (params.restart_interval_mcus != 0u && mcus_since_restart == params.restart_interval_mcus) {
                jpeg_encode_push_restart_marker(entropy, params.entropy_capacity, writer, rst);
                if (writer.overflow) {
                    status->code = JPEG_BASELINE_ENCODE_STATUS_OVERFLOW;
                    return;
                }
                rst = (rst + 1u) & 7u;
                prev_dc[0] = 0;
                prev_dc[1] = 0;
                prev_dc[2] = 0;
                mcus_since_restart = 0u;
            }

            for (uint component = 0u; component < params.components; component++) {
                const uint h = component_h(params, component);
                const uint v = component_v(params, component);
                if (h == 0u || v == 0u) {
                    status->code = JPEG_BASELINE_ENCODE_STATUS_INVALID_PARAMS;
                    return;
                }
                for (uint block_y = 0u; block_y < v; block_y++) {
                    for (uint block_x = 0u; block_x < h; block_x++) {
                        thread uchar block[64];
                        thread int coeffs[64];
                        jpeg_encode_sample_block(input, params, component, mcu_x, mcu_y, block_x, block_y, block);
                        bool ok;
                        if (component == 0u) {
                            jpeg_encode_fdct_quantize(block, q_luma, coeffs);
                            ok = jpeg_encode_block(
                                coeffs,
                                prev_dc[component],
                                dc_luma,
                                ac_luma,
                                entropy,
                                params.entropy_capacity,
                                writer,
                                status
                            );
                        } else {
                            jpeg_encode_fdct_quantize(block, q_chroma, coeffs);
                            ok = jpeg_encode_block(
                                coeffs,
                                prev_dc[component],
                                dc_chroma,
                                ac_chroma,
                                entropy,
                                params.entropy_capacity,
                                writer,
                                status
                            );
                        }
                        if (!ok) {
                            return;
                        }
                    }
                }
            }
            mcus_since_restart += 1u;
        }
    }

    jpeg_encode_align_with_ones(entropy, params.entropy_capacity, writer);
    if (writer.overflow) {
        status->code = JPEG_BASELINE_ENCODE_STATUS_OVERFLOW;
        return;
    }
    status->entropy_len = writer.pos;
}
