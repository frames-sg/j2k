// SPDX-License-Identifier: MIT OR Apache-2.0

// Hybrid JPEG decode for every JPEG class the fast-packet kernels do not
// cover. The CPU entropy-decodes dequantized DCT blocks; these kernels run the
// IDCT, chroma upsampling, color conversion, and output packing. Each formula
// mirrors the CPU decoder; `j2k_jpeg::adapter::device_dct` and its host
// reference tests define the contract.

struct JpegHybridIdctParams {
    uint block_base;
    uint block_cols;
    uint block_x0;
    uint block_y0;
    uint block_x_count;
    uint block_y_count;
    uint idct_size;
    uint plane_stride;
};

struct JpegHybridComponentParams {
    uint plane_stride;
    uint origin_x;
    uint origin_y;
    uint width;
    uint height;
    uint h_ratio;
    uint v_ratio;
    uint upsample;
};

struct JpegHybridPackParams {
    JpegHybridComponentParams components[4];
    uint component_count;
    uint color_space;
    uint out_format;
    // Source rectangle in scaled image coordinates.
    uint src_x;
    uint src_y;
    uint width;
    uint height;
    // Destination origin and row pitch, in pixels, within the output buffer.
    uint dest_x;
    uint dest_y;
    uint dest_stride;
};

constant uint HYBRID_UPSAMPLE_NONE = 0;
constant uint HYBRID_UPSAMPLE_H2V1 = 1;
constant uint HYBRID_UPSAMPLE_H1V2 = 2;
constant uint HYBRID_UPSAMPLE_H2V2 = 3;
constant uint HYBRID_UPSAMPLE_REPLICATE = 4;

constant uint HYBRID_COLOR_GRAY = 0;
constant uint HYBRID_COLOR_YCBCR = 1;
constant uint HYBRID_COLOR_RGB = 2;
constant uint HYBRID_COLOR_CMYK = 3;
constant uint HYBRID_COLOR_YCCK = 4;

// ---------------------------------------------------------------------------
// IDCT
// ---------------------------------------------------------------------------

inline void hybrid_load_block(
    device const short *coeffs,
    constant JpegHybridIdctParams &params,
    uint2 gid,
    thread short input[64]
) {
    const uint block_x = params.block_x0 + gid.x;
    const uint block_y = params.block_y0 + gid.y;
    const uint block = params.block_base + block_y * params.block_cols + block_x;
    device const short *src = coeffs + block * 64u;
    for (uint i = 0; i < 64u; ++i) {
        input[i] = src[i];
    }
}

template <typename T>
inline void hybrid_store_block(
    device T *plane,
    constant JpegHybridIdctParams &params,
    uint2 gid,
    thread const T *pixels
) {
    const uint size = params.idct_size;
    device T *dst = plane + gid.y * size * params.plane_stride + gid.x * size;
    for (uint row = 0; row < size; ++row) {
        for (uint col = 0; col < size; ++col) {
            dst[row * params.plane_stride + col] = pixels[row * size + col];
        }
    }
}

kernel void jpeg_hybrid_idct8(
    device const short *coeffs [[buffer(0)]],
    device uchar *plane [[buffer(1)]],
    constant JpegHybridIdctParams &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.block_x_count || gid.y >= params.block_y_count) {
        return;
    }
    short input[64];
    hybrid_load_block(coeffs, params, gid, input);
    uchar pixels[64];
    switch (params.idct_size) {
        case 8u:
            idct_islow(input, pixels);
            break;
        case 4u:
            idct_islow_4x4(input, pixels);
            break;
        case 2u:
            idct_islow_2x2(input, pixels);
            break;
        default:
            pixels[0] = idct_islow_1x1(input);
            break;
    }
    hybrid_store_block(plane, params, gid, pixels);
}

// 12-bit IDCTs: jidctint.c / jidctred.c built with BITS_IN_JSAMPLE == 12 keep
// PASS1_BITS = 1 and 64-bit intermediates, so nothing wraps. Samples are the
// native 0..4095 values.
inline ushort hybrid_level_shift_12(long value) {
    return ushort(clamp(value + 2048, long(0), long(4095)));
}

inline void hybrid_idct12_8x8(thread const short input[64], thread ushort output[64]) {
    long work[64];
    long r[8];
    for (uint col = 0; col < 8u; ++col) {
        thread const short *p = input + col;
        islow_1d(
            long(p[0]), long(p[8]), long(p[16]), long(p[24]),
            long(p[32]), long(p[40]), long(p[48]), long(p[56]),
            r
        );
        for (uint row = 0; row < 8u; ++row) {
            work[row * 8u + col] = descale_rounded(r[row], CODEC_MATH_JPEG_IDCT_PASS1_SHIFT_12);
        }
    }
    for (uint row = 0; row < 8u; ++row) {
        thread const long *w = work + row * 8u;
        islow_1d(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], r);
        for (uint col = 0; col < 8u; ++col) {
            output[row * 8u + col] = hybrid_level_shift_12(descale_rounded(r[col], CODEC_MATH_JPEG_IDCT_PASS2_SHIFT_12));
        }
    }
}

inline void hybrid_idct12_4x4(thread const short input[64], thread ushort output[16]) {
    long work[32];
    long r[4];
    for (uint col = 0; col < 8u; ++col) {
        if (col == 4u) {
            continue;
        }
        thread const short *p = input + col;
        islow_1d_4(
            long(p[0]), long(p[8]), long(p[16]), long(p[24]),
            long(p[40]), long(p[48]), long(p[56]),
            r
        );
        const int shift = CODEC_MATH_JPEG_IDCT_PASS1_SHIFT_12 + 1;
        for (uint row = 0; row < 4u; ++row) {
            work[row * 8u + col] = descale_rounded(r[row], shift);
        }
    }
    for (uint row = 0; row < 4u; ++row) {
        thread const long *w = work + row * 8u;
        islow_1d_4(w[0], w[1], w[2], w[3], w[5], w[6], w[7], r);
        const int shift = CODEC_MATH_JPEG_IDCT_PASS2_SHIFT_12 + 1;
        for (uint col = 0; col < 4u; ++col) {
            output[row * 4u + col] = hybrid_level_shift_12(descale_rounded(r[col], shift));
        }
    }
}

inline void hybrid_idct12_2x2(thread const short input[64], thread ushort output[4]) {
    long work[16];
    long r[2];
    const uint columns[5] = {0u, 1u, 3u, 5u, 7u};
    for (uint i = 0; i < 5u; ++i) {
        const uint col = columns[i];
        thread const short *p = input + col;
        islow_1d_2(long(p[0]), long(p[8]), long(p[24]), long(p[40]), long(p[56]), r);
        const int shift = CODEC_MATH_JPEG_IDCT_PASS1_SHIFT_12 + 2;
        work[col] = descale_rounded(r[0], shift);
        work[8u + col] = descale_rounded(r[1], shift);
    }
    for (uint row = 0; row < 2u; ++row) {
        thread const long *w = work + row * 8u;
        islow_1d_2(w[0], w[1], w[3], w[5], w[7], r);
        const int shift = CODEC_MATH_JPEG_IDCT_PASS2_SHIFT_12 + 2;
        output[row * 2u] = hybrid_level_shift_12(descale_rounded(r[0], shift));
        output[row * 2u + 1u] = hybrid_level_shift_12(descale_rounded(r[1], shift));
    }
}

kernel void jpeg_hybrid_idct12(
    device const short *coeffs [[buffer(0)]],
    device ushort *plane [[buffer(1)]],
    constant JpegHybridIdctParams &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.block_x_count || gid.y >= params.block_y_count) {
        return;
    }
    short input[64];
    hybrid_load_block(coeffs, params, gid, input);
    ushort pixels[64];
    switch (params.idct_size) {
        case 8u:
            hybrid_idct12_8x8(input, pixels);
            break;
        case 4u:
            hybrid_idct12_4x4(input, pixels);
            break;
        case 2u:
            hybrid_idct12_2x2(input, pixels);
            break;
        default:
            pixels[0] = hybrid_level_shift_12(descale_rounded(long(input[0]), 3));
            break;
    }
    hybrid_store_block(plane, params, gid, pixels);
}

// ---------------------------------------------------------------------------
// Upsampling, color conversion, and packing
// ---------------------------------------------------------------------------

template <typename T>
inline uint hybrid_plane_at(
    device const T *plane,
    constant JpegHybridComponentParams &c,
    uint x,
    uint y
) {
    return uint(plane[(y - c.origin_y) * c.plane_stride + (x - c.origin_x)]);
}

// libjpeg-turbo h2v1_fancy_upsample at output column x.
template <typename T>
inline uint hybrid_h2v1(device const T *plane, constant JpegHybridComponentParams &c, uint row, uint x) {
    const uint n = c.width;
    const uint s = min(x / 2u, n - 1u);
    const uint near = 3u * hybrid_plane_at(plane, c, s, row);
    if ((x & 1u) == 0u) {
        if (s == 0u) {
            return hybrid_plane_at(plane, c, 0u, row);
        }
        return (near + hybrid_plane_at(plane, c, s - 1u, row) + 1u) >> 2;
    }
    if (s == n - 1u) {
        return hybrid_plane_at(plane, c, n - 1u, row);
    }
    return (near + hybrid_plane_at(plane, c, s + 1u, row) + 2u) >> 2;
}

// libjpeg-turbo h2v2_fancy_upsample at output column x, given the current and
// nearest neighbouring component rows.
template <typename T>
inline uint hybrid_h2v2(
    device const T *plane,
    constant JpegHybridComponentParams &c,
    uint curr,
    uint near,
    uint x
) {
    const uint n = c.width;
    if (n == 1u) {
        const uint colsum = 3u * hybrid_plane_at(plane, c, 0u, curr) + hybrid_plane_at(plane, c, 0u, near);
        return (4u * colsum + 8u) >> 4;
    }
    const uint s = min(x / 2u, n - 1u);
    const uint this_sum = 3u * hybrid_plane_at(plane, c, s, curr) + hybrid_plane_at(plane, c, s, near);
    if (x == 0u) {
        return (4u * this_sum + 8u) >> 4;
    }
    if (x == 2u * n - 1u) {
        return (4u * this_sum + 7u) >> 4;
    }
    if ((x & 1u) == 0u) {
        const uint last = 3u * hybrid_plane_at(plane, c, s - 1u, curr) + hybrid_plane_at(plane, c, s - 1u, near);
        return (3u * this_sum + last + 8u) >> 4;
    }
    const uint next = 3u * hybrid_plane_at(plane, c, s + 1u, curr) + hybrid_plane_at(plane, c, s + 1u, near);
    return (3u * this_sum + next + 7u) >> 4;
}

template <typename T>
inline uint hybrid_component_sample(
    device const T *plane,
    constant JpegHybridComponentParams &c,
    uint x,
    uint y
) {
    const uint last_row = c.height - 1u;
    switch (c.upsample) {
        case HYBRID_UPSAMPLE_NONE:
            return hybrid_plane_at(plane, c, x, y);
        case HYBRID_UPSAMPLE_H2V1:
            return hybrid_h2v1(plane, c, min(y, last_row), x);
        case HYBRID_UPSAMPLE_H2V2:
        case HYBRID_UPSAMPLE_H1V2: {
            const uint curr = min(y / 2u, last_row);
            const bool bottom = (y & 1u) != 0u;
            const uint near = bottom ? min(curr + 1u, last_row) : (curr == 0u ? 0u : curr - 1u);
            if (c.upsample == HYBRID_UPSAMPLE_H2V2) {
                return hybrid_h2v2(plane, c, curr, near, x);
            }
            return (3u * hybrid_plane_at(plane, c, x, curr) + hybrid_plane_at(plane, c, x, near) + (bottom ? 2u : 1u)) >> 2;
        }
        default:
            return hybrid_plane_at(
                plane,
                c,
                min(x / c.h_ratio, c.width - 1u),
                min(y / c.v_ratio, last_row)
            );
    }
}

// j2k_jpeg::color::ycbcr::ycbcr_to_rgb (libjpeg-turbo SIMD form).
inline uint3 hybrid_ycbcr8_to_rgb(int y, int cb, int cr) {
    cb -= 128;
    cr -= 128;
    const int r = y + ((91881 * cr + 32768) >> 16);
    const int g = y - ((22554 * cb + 46802 * cr + 32768) >> 16);
    const int b = y + ((116130 * cb + 32768) >> 16);
    return uint3(clamp(int3(r, g, b), int3(0), int3(255)));
}

// j2k_jpeg::color::ycbcr::ycbcr12_to_rgb16 (libjpeg-turbo jdcolor.c tables).
inline uint3 hybrid_ycbcr12_to_rgb(int y, int cb, int cr) {
    cb -= 2048;
    cr -= 2048;
    const int r = y + ((91881 * cr + 32768) >> 16);
    const int g = y + ((-22554 * cb - 46802 * cr + 32768) >> 16);
    const int b = y + ((116130 * cb + 32768) >> 16);
    return uint3(clamp(int3(r, g, b), int3(0), int3(4095)));
}

// Adobe-inverted CMYK: every channel is scaled by K.
inline uint3 hybrid_inverted_cmyk_to_rgb(uint3 cmy, uint k, uint max_value) {
    return (cmy * k + (max_value >> 1)) / max_value;
}

inline uint3 hybrid_rgb_of(uint color_space, bool twelve_bit, uint4 c) {
    const uint max_value = twelve_bit ? 4095u : 255u;
    switch (color_space) {
        case HYBRID_COLOR_GRAY:
            return uint3(c.x);
        case HYBRID_COLOR_RGB:
            return c.xyz;
        case HYBRID_COLOR_YCBCR:
            return twelve_bit ? hybrid_ycbcr12_to_rgb(int(c.x), int(c.y), int(c.z))
                              : hybrid_ycbcr8_to_rgb(int(c.x), int(c.y), int(c.z));
        case HYBRID_COLOR_CMYK:
            return hybrid_inverted_cmyk_to_rgb(c.xyz, c.w, max_value);
        default: {
            // YCC holds complemented CMY, as libjpeg-turbo's ycck_cmyk_convert.
            const uint3 ycc_rgb = twelve_bit ? hybrid_ycbcr12_to_rgb(int(c.x), int(c.y), int(c.z))
                                             : hybrid_ycbcr8_to_rgb(int(c.x), int(c.y), int(c.z));
            return hybrid_inverted_cmyk_to_rgb(uint3(max_value) - ycc_rgb, c.w, max_value);
        }
    }
}

template <typename T, typename Out>
inline void hybrid_pack_pixel(
    device const T *plane0,
    device const T *plane1,
    device const T *plane2,
    device const T *plane3,
    device Out *out,
    constant JpegHybridPackParams &params,
    uint2 gid,
    bool twelve_bit
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }
    const uint x = params.src_x + gid.x;
    const uint y = params.src_y + gid.y;
    uint4 c = uint4(0u);
    c.x = hybrid_component_sample(plane0, params.components[0], x, y);
    if (params.component_count > 1u) {
        c.y = hybrid_component_sample(plane1, params.components[1], x, y);
        c.z = hybrid_component_sample(plane2, params.components[2], x, y);
    }
    if (params.component_count > 3u) {
        c.w = hybrid_component_sample(plane3, params.components[3], x, y);
    }
    const uint pixel = (params.dest_y + gid.y) * params.dest_stride + params.dest_x + gid.x;
    if (params.out_format == OUT_GRAY) {
        uint gray = c.x;
        if (params.color_space != HYBRID_COLOR_GRAY && params.color_space != HYBRID_COLOR_YCBCR) {
            const uint3 rgb = hybrid_rgb_of(params.color_space, twelve_bit, c);
            gray = (77u * rgb.x + 150u * rgb.y + 29u * rgb.z + 128u) >> 8;
        }
        out[pixel] = Out(gray);
        return;
    }
    const uint3 rgb = hybrid_rgb_of(params.color_space, twelve_bit, c);
    const uint channels = params.out_format == OUT_RGBA ? 4u : 3u;
    device Out *dst = out + pixel * channels;
    dst[0] = Out(rgb.x);
    dst[1] = Out(rgb.y);
    dst[2] = Out(rgb.z);
    if (channels == 4u) {
        dst[3] = twelve_bit ? Out(0xFFFFu) : Out(0xFFu);
    }
}

kernel void jpeg_hybrid_pack8(
    device const uchar *plane0 [[buffer(0)]],
    device const uchar *plane1 [[buffer(1)]],
    device const uchar *plane2 [[buffer(2)]],
    device const uchar *plane3 [[buffer(3)]],
    device uchar *out [[buffer(4)]],
    constant JpegHybridPackParams &params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    hybrid_pack_pixel(plane0, plane1, plane2, plane3, out, params, gid, false);
}

kernel void jpeg_hybrid_pack12(
    device const ushort *plane0 [[buffer(0)]],
    device const ushort *plane1 [[buffer(1)]],
    device const ushort *plane2 [[buffer(2)]],
    device const ushort *plane3 [[buffer(3)]],
    device ushort *out [[buffer(4)]],
    constant JpegHybridPackParams &params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    hybrid_pack_pixel(plane0, plane1, plane2, plane3, out, params, gid, true);
}

// ---------------------------------------------------------------------------
// Lossless (SOF3): predictor reconstruction and packing
// ---------------------------------------------------------------------------

// A split frame codes its components in several scans, so the predictor,
// its seed, the sample limit, and the point transform are per component.
struct JpegLosslessComponentParams {
    uint width;
    uint height;
    uint restart_rows;
    uint base;
    uint predictor;
    uint initial_predictor;
    uint sample_limit;
    uint point_transform;
    uint h_expand;
    uint v_expand;
};

struct JpegLosslessReconstructParams {
    JpegLosslessComponentParams components[3];
    uint wavefront_mask;
    uint component_count;
};

struct JpegLosslessPackParams {
    uint width;
    uint height;
    uint color_space;
    uint precision;
    uint out_format;
    uint denominator;
    uint src_x;
    uint src_y;
    uint out_width;
    uint out_height;
    uint dest_x;
    uint dest_y;
    uint dest_stride;
};

// One threadgroup reconstructs one component. Sample (row, col) depends on
// its left, upper, and upper-left neighbours, so thread t owns row
// band + t and handles column step - t: every anti-diagonal of a band is
// one parallel step, and bands follow one another. Arithmetic is modulo 2^16
// and the 1-D predictor reseeds the first row of the scan and of every
// restart interval, as in libjpeg-turbo's jdlossls.c.
kernel void jpeg_lossless_reconstruct(
    device const ushort *diffs [[buffer(0)]],
    device ushort *samples [[buffer(1)]],
    device atomic_uint *status [[buffer(2)]],
    constant JpegLosslessReconstructParams &params [[buffer(3)]],
    uint component [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    if (((params.wavefront_mask >> component) & 1u) == 0u) {
        return;
    }
    constant JpegLosslessComponentParams &c = params.components[component];
    device const ushort *d = diffs + c.base;
    device ushort *x = samples + c.base;
    const uint width = c.width;
    const uint height = c.height;
    const uint steps = width + group_size - 1u;
    for (uint band = 0; band < height; band += group_size) {
        const uint row = band + tid;
        for (uint step = 0; step < steps; ++step) {
            if (row < height && step >= tid && step - tid < width) {
                const uint col = step - tid;
                const uint at = row * width + col;
                const bool first_row = row == 0u || (c.restart_rows != 0u && row % c.restart_rows == 0u);
                int prediction;
                if (first_row) {
                    prediction = col == 0u ? int(c.initial_predictor) : int(x[at - 1u]);
                } else if (col == 0u) {
                    prediction = int(x[at - width]);
                } else {
                    const int ra = int(x[at - 1u]);
                    const int rb = int(x[at - width]);
                    const int rc = int(x[at - width - 1u]);
                    switch (c.predictor) {
                        case 1u: prediction = ra; break;
                        case 2u: prediction = rb; break;
                        case 3u: prediction = rc; break;
                        case 4u: prediction = ra + rb - rc; break;
                        case 5u: prediction = ra + ((rb - rc) >> 1); break;
                        case 6u: prediction = rb + ((ra - rc) >> 1); break;
                        default: prediction = (ra + rb) >> 1; break;
                    }
                }
                const uint sample = (uint(prediction) + uint(d[at])) & 0xFFFFu;
                if (sample > c.sample_limit) {
                    atomic_store_explicit(status, 1u, memory_order_relaxed);
                }
                x[at] = ushort(sample);
            }
            threadgroup_barrier(mem_flags::mem_device);
        }
    }
}

struct JpegLosslessRowsParams {
    uint component;
    uint column_base;
};

// Predictor 1 (DICOM's SV1) adds every difference after a row's first to its
// left neighbour, so once each row's first sample is known (the CPU derives
// those from the column of first differences, applying the restart seeding)
// the rows reconstruct independently: one threadgroup per row, each thread a
// contiguous chunk of the row, joined by a threadgroup prefix sum. Sums are
// kept modulo 2^32, which preserves the low 16 bits the samples keep.
kernel void jpeg_lossless_reconstruct_rows(
    device const ushort *diffs [[buffer(0)]],
    device ushort *samples [[buffer(1)]],
    device atomic_uint *status [[buffer(2)]],
    constant JpegLosslessReconstructParams &params [[buffer(3)]],
    device const ushort *columns [[buffer(4)]],
    constant JpegLosslessRowsParams &rows [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_groups [[simdgroups_per_threadgroup]]
) {
    constant JpegLosslessComponentParams &c = params.components[rows.component];
    const uint width = c.width;
    device const ushort *d = diffs + c.base + row * width;
    device ushort *x = samples + c.base + row * width;
    const uint first = uint(columns[rows.column_base + row]);
    const uint chunk = (width - 1u + group_size - 1u) / group_size;
    const uint start = min(1u + tid * chunk, width);
    const uint end = min(start + chunk, width);

    uint local = 0u;
    for (uint j = start; j < end; ++j) {
        local += uint(d[j]);
    }
    threadgroup uint totals[32];
    const uint simd_offset = simd_prefix_exclusive_sum(local);
    const uint simd_total = simd_sum(local);
    if (lane == 0u) {
        totals[simd_group] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0u) {
        const uint total = lane < simd_groups ? totals[lane] : 0u;
        totals[lane] = simd_prefix_exclusive_sum(total);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint running = first + totals[simd_group] + simd_offset;
    bool out_of_range = false;
    if (tid == 0u) {
        x[0] = ushort(first);
        out_of_range = first > c.sample_limit;
    }
    for (uint j = start; j < end; ++j) {
        running += uint(d[j]);
        const uint sample = running & 0xFFFFu;
        out_of_range = out_of_range || sample > c.sample_limit;
        x[j] = ushort(sample);
    }
    if (out_of_range) {
        atomic_store_explicit(status, 1u, memory_order_relaxed);
    }
}

// The sample covering full-size pixel (x, y): libjpeg-turbo's lossless
// upsampler replicates each component by its sampling ratio. Output samples
// are shifted left by the component's point transform.
inline uint lossless_sample(
    device const ushort *samples,
    constant JpegLosslessComponentParams &c,
    uint x,
    uint y
) {
    // Integer division is slow on the GPU; most components are unsampled.
    const uint sx = c.h_expand == 1u ? x : x / c.h_expand;
    const uint sy = c.v_expand == 1u ? y : y / c.v_expand;
    return uint(samples[c.base + sy * c.width + sx]) << c.point_transform;
}

// j2k_jpeg::color::ycbcr::ycbcr_to_rgb_with_precision: chroma centered on
// 2^(P-1), clamped to [0, 2^P - 1]; ycbcr_to_rgb and ycbcr16_to_rgb16 at 8
// and 16 bits.
inline uint3 lossless_ycbcr_to_rgb(long y, long cb, long cr, uint precision) {
    const long center = long(1) << (precision - 1u);
    const long max_value = (center << 1) - 1;
    cb -= center;
    cr -= center;
    const long r = y + ((long(91881) * cr + 32768) >> 16);
    const long g = y - ((long(22554) * cb + long(46802) * cr + 32768) >> 16);
    const long b = y + ((long(116130) * cb + 32768) >> 16);
    return uint3(uint(clamp(r, long(0), max_value)),
                 uint(clamp(g, long(0), max_value)),
                 uint(clamp(b, long(0), max_value)));
}

template <typename Out>
inline void lossless_pack_pixel(
    device const ushort *samples,
    device Out *out,
    constant JpegLosslessPackParams &params,
    constant JpegLosslessReconstructParams &planes,
    uint2 gid,
    bool sixteen_bit
) {
    if (gid.x >= params.out_width || gid.y >= params.out_height) {
        return;
    }
    // Reduced-scale lossless output point-samples the full-size image.
    const uint sx = min((params.src_x + gid.x) * params.denominator, params.width - 1u);
    const uint sy = min((params.src_y + gid.y) * params.denominator, params.height - 1u);
    const uint pixel = (params.dest_y + gid.y) * params.dest_stride + params.dest_x + gid.x;
    const uint c0 = lossless_sample(samples, planes.components[0], sx, sy);
    if (planes.component_count == 1u) {
        out[pixel] = Out(c0);
        return;
    }
    uint3 c = uint3(c0,
                    lossless_sample(samples, planes.components[1], sx, sy),
                    lossless_sample(samples, planes.components[2], sx, sy));
    if (params.color_space == HYBRID_COLOR_YCBCR) {
        c = lossless_ycbcr_to_rgb(long(c.x), long(c.y), long(c.z), params.precision);
    }
    const uint channels = params.out_format == OUT_RGBA ? 4u : 3u;
    device Out *dst = out + pixel * channels;
    dst[0] = Out(c.x);
    dst[1] = Out(c.y);
    dst[2] = Out(c.z);
    if (channels == 4u) {
        dst[3] = sixteen_bit ? Out(0xFFFFu) : Out(0xFFu);
    }
}

kernel void jpeg_lossless_pack8(
    device const ushort *samples [[buffer(0)]],
    device uchar *out [[buffer(1)]],
    constant JpegLosslessPackParams &params [[buffer(2)]],
    constant JpegLosslessReconstructParams &planes [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    lossless_pack_pixel(samples, out, params, planes, gid, false);
}

kernel void jpeg_lossless_pack16(
    device const ushort *samples [[buffer(0)]],
    device ushort *out [[buffer(1)]],
    constant JpegLosslessPackParams &params [[buffer(2)]],
    constant JpegLosslessReconstructParams &planes [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    lossless_pack_pixel(samples, out, params, planes, gid, true);
}
