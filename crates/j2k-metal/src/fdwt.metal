// SPDX-License-Identifier: MIT OR Apache-2.0

#include <metal_stdlib>
using namespace metal;

struct J2kForwardDwt53Params {
    uint full_width;
    uint current_width;
    uint current_height;
    uint low_width;
    uint low_height;
    uint x_phase;
    uint y_phase;
};

struct J2kForwardDwt53BatchedParams {
    uint full_width;
    uint current_width;
    uint current_height;
    uint low_width;
    uint low_height;
    uint component_count;
};

struct J2kForwardDwt97Params {
    uint full_width;
    uint current_width;
    uint current_height;
    uint low_width;
    uint low_height;
    uint parity;
    float coefficient;
    uint origin_phase;
};

// T.800 F-9/F-10 with symmetric extension, for odd-origin lines. The existing
// even-origin SIMD path remains unchanged.
inline float j2k_fdwt53_odd_predict(
    device const float *src, uint base, uint stride, uint length, uint high_index
) {
    const uint i = high_index * 2u;
    if (length == 1u) return 2.0f * src[base];
    const uint left = i == 0u ? 1u : i - 1u;
    const uint right = i + 1u < length ? i + 1u : i - 1u;
    return src[base + i * stride] - floor((src[base + left * stride] + src[base + right * stride]) * 0.5f);
}

inline void j2k_fdwt53_odd_pair(
    device const float *src, device float *dst,
    uint base, uint stride, uint length, uint low_count, uint index
) {
    const uint high_count = length - low_count;
    if (index < high_count) {
        dst[base + (low_count + index) * stride] = j2k_fdwt53_odd_predict(src, base, stride, length, index);
    }
    if (index < low_count) {
        const float left = j2k_fdwt53_odd_predict(src, base, stride, length, index);
        const float right = index + 1u < high_count
            ? j2k_fdwt53_odd_predict(src, base, stride, length, index + 1u) : left;
        dst[base + index * stride] = src[base + (2u * index + 1u) * stride] + floor((left + right) * 0.25f + 0.5f);
    }
}

inline float j2k_fdwt53_predict_row(
    device const float *src,
    uint row_base,
    uint width,
    uint high_index
) {
    const uint odd = high_index * 2u + 1u;
    const uint last_even = (width % 2u == 0u) ? width - 2u : width - 1u;
    const float left = src[row_base + odd - 1u];
    const float right = (odd + 1u < width) ? src[row_base + odd + 1u] : src[row_base + last_even];
    return src[row_base + odd] - floor((left + right) * 0.5f);
}

inline float j2k_fdwt53_predict_col(
    device const float *src,
    uint x,
    uint full_width,
    uint height,
    uint high_index
) {
    const uint odd = high_index * 2u + 1u;
    const uint last_even = (height % 2u == 0u) ? height - 2u : height - 1u;
    const float top = src[(odd - 1u) * full_width + x];
    const float bottom = (odd + 1u < height)
        ? src[(odd + 1u) * full_width + x]
        : src[last_even * full_width + x];
    return src[odd * full_width + x] - floor((top + bottom) * 0.5f);
}

inline void j2k_fdwt53_horizontal_step(
    device const float *src,
    device float *dst,
    uint full_width,
    uint current_width,
    uint low_width,
    uint2 gid,
    uint local_x,
    uint simd_lane
) {
    const bool active = gid.x < low_width;
    const uint high_width = current_width - low_width;
    const uint row_base = gid.y * full_width;
    float right = 0.0f;
    if (active && gid.x < high_width) {
        right = j2k_fdwt53_predict_row(src, row_base, current_width, gid.x);
    }
    // Each lane emits a low/high pair and shares its high prediction with the
    // next lane. Nonuniform threadgroups can make SIMD groups cross rows;
    // row and SIMD boundaries need an explicit one-sample halo.
    const float shuffled_left = simd_shuffle_up(right, 1u);

    if (!active) {
        return;
    }

    float left;
    if (gid.x == 0u) {
        left = right;
    } else if (local_x == 0u || simd_lane == 0u) {
        left = j2k_fdwt53_predict_row(src, row_base, current_width, gid.x - 1u);
    } else {
        left = shuffled_left;
    }
    if (gid.x >= high_width) {
        right = left;
    }

    const uint even = gid.x * 2u;
    dst[row_base + gid.x] =
        src[row_base + even] + floor((left + right) * 0.25f + 0.5f);
    if (gid.x < high_width) {
        dst[row_base + low_width + gid.x] = right;
    }
}

inline void j2k_fdwt53_vertical_step(
    device const float *src,
    device float *dst,
    uint full_width,
    uint current_height,
    uint low_height,
    uint2 gid
) {
    if (gid.y < low_height) {
        const uint even = gid.y * 2u;
        const float top = gid.y > 0u
            ? j2k_fdwt53_predict_col(src, gid.x, full_width, current_height, gid.y - 1u)
            : j2k_fdwt53_predict_col(src, gid.x, full_width, current_height, 0u);
        const float bottom = even + 1u < current_height
            ? j2k_fdwt53_predict_col(src, gid.x, full_width, current_height, gid.y)
            : top;
        dst[gid.y * full_width + gid.x] =
            src[even * full_width + gid.x] + floor((top + bottom) * 0.25f + 0.5f);
        return;
    }

    const uint high_index = gid.y - low_height;
    dst[gid.y * full_width + gid.x] = j2k_fdwt53_predict_col(
        src,
        gid.x,
        full_width,
        current_height,
        high_index
    );
}

kernel void j2k_forward_dwt53_horizontal(
    device const float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant J2kForwardDwt53Params &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 local_id [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    if (gid.x >= params.current_width || gid.y >= params.current_height) {
        return;
    }

    if (params.x_phase != 0u) {
        j2k_fdwt53_odd_pair(src, dst, gid.y * params.full_width, 1u, params.current_width, params.low_width, gid.x);
        return;
    }

   j2k_fdwt53_horizontal_step(
        src,
        dst,
        params.full_width,
        params.current_width,
        params.low_width,
        gid,
        local_id.x,
        simd_lane
    );
}

kernel void j2k_forward_dwt53_vertical(
    device const float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant J2kForwardDwt53Params &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.current_width || gid.y >= params.current_height) {
        return;
    }

    if (params.y_phase != 0u) {
        // The vertical launch covers the whole extent; only one thread per
        // low/high pair writes so that outputs never overlap.
        if (gid.y < max(params.low_height, params.current_height - params.low_height)) {
            j2k_fdwt53_odd_pair(src, dst, gid.x, params.full_width, params.current_height, params.low_height, gid.y);
        }
        return;
    }

   j2k_fdwt53_vertical_step(
        src,
        dst,
        params.full_width,
        params.current_height,
        params.low_height,
        gid
    );
}

kernel void j2k_forward_dwt53_horizontal_batched(
    device const float *src0 [[buffer(0)]],
    device const float *src1 [[buffer(1)]],
    device const float *src2 [[buffer(2)]],
    device float *dst0 [[buffer(3)]],
    device float *dst1 [[buffer(4)]],
    device float *dst2 [[buffer(5)]],
    constant J2kForwardDwt53BatchedParams &params [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 local_id [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    if (
        gid.x >= params.current_width ||
        gid.y >= params.current_height ||
        gid.z >= params.component_count
    ) {
        return;
    }

    device const float *src = gid.z == 0u ? src0 : (gid.z == 1u ? src1 : src2);
    device float *dst = gid.z == 0u ? dst0 : (gid.z == 1u ? dst1 : dst2);
   j2k_fdwt53_horizontal_step(
        src,
        dst,
        params.full_width,
        params.current_width,
        params.low_width,
        gid.xy,
        local_id.x,
        simd_lane
    );
}

kernel void j2k_forward_dwt53_vertical_batched(
    device const float *src0 [[buffer(0)]],
    device const float *src1 [[buffer(1)]],
    device const float *src2 [[buffer(2)]],
    device float *dst0 [[buffer(3)]],
    device float *dst1 [[buffer(4)]],
    device float *dst2 [[buffer(5)]],
    constant J2kForwardDwt53BatchedParams &params [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (
        gid.x >= params.current_width ||
        gid.y >= params.current_height ||
        gid.z >= params.component_count
    ) {
        return;
    }

    device const float *src = gid.z == 0u ? src0 : (gid.z == 1u ? src1 : src2);
    device float *dst = gid.z == 0u ? dst0 : (gid.z == 1u ? dst1 : dst2);
   j2k_fdwt53_vertical_step(
        src,
        dst,
        params.full_width,
        params.current_height,
        params.low_height,
        gid.xy
    );
}

constant uint J2K_FDWT53_TILE_W = 32u;
constant uint J2K_FDWT53_TILE_H = 32u;
// A tile row or column spans two samples before the tile and one after it:
// the lifts of the tile's samples read no further.
constant uint J2K_FDWT53_TILE_STRIDE = J2K_FDWT53_TILE_W + 3u;
constant uint J2K_FDWT53_TILE_LEN = J2K_FDWT53_TILE_STRIDE * (J2K_FDWT53_TILE_H + 3u);

struct J2kFusedDwt53LevelParams {
    J2kLosslessDeinterleaveParams input;
    uint current_width;
    uint current_height;
    uint code_block_width_log2;
    uint code_block_height_log2;
    uint first_band;
    uint from_samples;
    uint low_to_coefficients;
};

// A sub-band whose code blocks are stored in raster order from
// `coefficient_offset`, each block's samples row by row.
struct J2kFusedDwt53Band {
    uint coefficient_offset;
    uint width;
    uint height;
};

// Whole-sample symmetric extension of `index` >= -2. Samples past
// `length + 1` feed no sample inside the plane; they repeat that one.
inline uint j2k_fdwt53_mirror(int index, uint length) {
    if (length == 1u) return 0u;
    const int mirrored = abs(min(index, int(length) + 1));
    return uint(mirrored < int(length) ? mirrored : abs(2 * int(length) - 2 - mirrored));
}

inline uint j2k_fdwt53_coefficient_index(
    J2kFusedDwt53Band band, uint x, uint y, uint block_width_log2, uint block_height_log2
) {
    const uint block_x = (x >> block_width_log2) << block_width_log2;
    const uint block_y = (y >> block_height_log2) << block_height_log2;
    const uint block_width = min(1u << block_width_log2, band.width - block_x);
    const uint block_height = min(1u << block_height_log2, band.height - block_y);
    return band.coefficient_offset + block_y * band.width + block_x * block_height
        + (y - block_y) * block_width + (x - block_x);
}

// One predict or update step over `count` tile samples, `per_line` per line:
// sample k sits at `first + k / per_line * line + k % per_line * step` and
// lifts from the samples `gap` before and after it.
inline void j2k_fdwt53_lift(
    threadgroup float *tile, uint components, uint count, uint per_line, uint line, uint step,
    uint first, uint gap, bool predict, uint tid, uint threads
) {
    for (uint i = tid; i < count; i += threads) {
        const uint at = first + i / per_line * line + i % per_line * step;
        for (uint c = 0u; c < components; ++c) {
            threadgroup float *t = tile + c * J2K_FDWT53_TILE_LEN;
            const float sum = t[at - gap] + t[at + gap];
            t[at] = predict ? t[at] - floor(sum * 0.5f) : t[at] + floor(sum * 0.25f + 0.5f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// One 5/3 analysis level of every component in a single pass: each
// threadgroup loads its tile and halo (level 0 straight from the interleaved
// samples, with the RCT for three components), lifts the columns, then the
// rows, in threadgroup memory, and writes the sub-bands once: detail bands,
// and the last level's LL, as code-block coefficients, other LL bands as the
// next level's planes. Mirrored loads give the edge samples the symmetric
// extension, so every sample sees the same operations on the same values as
// the vertical and horizontal kernels above.
kernel void j2k_forward_dwt53_fused_level(
    device const uchar *samples [[buffer(0)]],
    device const float *low_in [[buffer(1)]],
    device float *low_out [[buffer(2)]],
    device int *coefficients [[buffer(3)]],
    constant J2kFusedDwt53Band *bands [[buffer(4)]],
    constant J2kFusedDwt53LevelParams &params [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group_size [[threads_per_threadgroup]]
) {
    threadgroup float tile[3u * J2K_FDWT53_TILE_LEN];
    constant J2kLosslessDeinterleaveParams &input = params.input;
    const uint threads = group_size.x;
    const uint stride = J2K_FDWT53_TILE_STRIDE;
    const uint width = params.current_width;
    const uint height = params.current_height;
    const uint components = input.components;
    const uint tx0 = group.x * J2K_FDWT53_TILE_W;
    const uint ty0 = group.y * J2K_FDWT53_TILE_H;
    for (uint i = tid; i < J2K_FDWT53_TILE_LEN; i += threads) {
        const uint x = j2k_fdwt53_mirror(int(tx0 + i % stride) - 2, width);
        const uint y = j2k_fdwt53_mirror(int(ty0 + i / stride) - 2, height);
        if (params.from_samples == 0u) {
            for (uint c = 0u; c < components; ++c) {
                tile[c * J2K_FDWT53_TILE_LEN + i] = low_in[(c * height + y) * width + x];
            }
            continue;
        }
        const bool inside = x < input.src_width && y < input.src_height;
        const uint base = y * input.src_stride + x * components * input.bytes_per_sample;
        float s[3];
        for (uint c = 0u; c < components; ++c) {
            s[c] = j2k_lossless_load_sample(samples, base, c, components, input.bytes_per_sample,
                input.bit_depth, input.sample_offset, input.signed_samples, inside);
        }
        if (components == 1u) {
            tile[i] = s[0];
            continue;
        }
        tile[i] = floor((s[0] + 2.0f * s[1] + s[2]) * 0.25f);
        tile[J2K_FDWT53_TILE_LEN + i] = s[2] - s[1];
        tile[2u * J2K_FDWT53_TILE_LEN + i] = s[0] - s[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint half_w = J2K_FDWT53_TILE_W / 2u;
    const uint half_h = J2K_FDWT53_TILE_H / 2u;
    if (height >= 2u) {
        // Odd rows 1..TILE_H + 1, then even rows 2..TILE_H.
        j2k_fdwt53_lift(tile, components, (half_h + 1u) * stride, stride, 2u * stride, 1u,
            stride, stride, true, tid, threads);
        j2k_fdwt53_lift(tile, components, half_h * stride, stride, 2u * stride, 1u,
            2u * stride, stride, false, tid, threads);
    }
    if (width >= 2u) {
        // The tile's rows: odd columns 1..TILE_W + 1, then even columns 2..TILE_W.
        j2k_fdwt53_lift(tile, components, J2K_FDWT53_TILE_H * (half_w + 1u), half_w + 1u,
            stride, 2u, 2u * stride + 1u, 1u, true, tid, threads);
        j2k_fdwt53_lift(tile, components, J2K_FDWT53_TILE_H * half_w, half_w, stride, 2u,
            2u * stride + 2u, 1u, false, tid, threads);
    }

    // Band 0 is LL, 1 HL, 2 LH, 3 HH; consecutive threads write consecutive
    // samples of one band row.
    const uint low_width = (width + 1u) / 2u;
    const uint low_height = (height + 1u) / 2u;
    for (uint i = tid; i < 4u * half_w * half_h; i += threads) {
        const uint band = i / (half_w * half_h);
        const uint bx = tx0 / 2u + i % half_w;
        const uint by = ty0 / 2u + i % (half_w * half_h) / half_w;
        const uint x = 2u * bx + (band & 1u);
        const uint y = 2u * by + (band >> 1u);
        if (x >= width || y >= height) {
            continue;
        }
        const uint at = (2u + y - ty0) * stride + 2u + x - tx0;
        for (uint c = 0u; c < components; ++c) {
            const float value = tile[c * J2K_FDWT53_TILE_LEN + at];
            if (band == 0u && params.low_to_coefficients == 0u) {
                low_out[(c * low_height + by) * low_width + bx] = value;
            } else {
                coefficients[j2k_fdwt53_coefficient_index(bands[params.first_band + c * 4u + band],
                    bx, by, params.code_block_width_log2, params.code_block_height_log2)] =
                    int(round(value));
            }
        }
    }
}

inline bool j2k_fdwt97_is_active_parity(uint index, uint parity) {
    return (index & 1u) == parity;
}

// Whole-sample symmetric extension (T.800 F.3.7/F.4.7), independent of
// whether the first in-buffer sample belongs to the low or high sequence.
inline float j2k_fdwt97_horizontal_neighbor(
    device const float *data, uint row_base, uint width, uint x, bool left_neighbor
) {
    const uint neighbor = left_neighbor ? (x > 0u ? x - 1u : 1u)
        : (x + 1u < width ? x + 1u : x - 1u);
    return data[row_base + neighbor];
}

inline float j2k_fdwt97_vertical_neighbor(
    device const float *data, uint full_width, uint height, uint x, uint y, bool top_neighbor
) {
    const uint neighbor = top_neighbor ? (y > 0u ? y - 1u : 1u)
        : (y + 1u < height ? y + 1u : y - 1u);
    return data[neighbor * full_width + x];
}

kernel void j2k_forward_dwt97_lift_horizontal(
    device float *data [[buffer(0)]],
    device float *unused [[buffer(1)]],
    constant J2kForwardDwt97Params &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
    (void)unused;
    if (params.current_width < 2u) return;
    if (
        gid.x >= params.current_width ||
        gid.y >= params.current_height ||
        !j2k_fdwt97_is_active_parity(gid.x, params.parity)
    ) {
        return;
    }

    const uint row_base = gid.y * params.full_width;
    const float left = j2k_fdwt97_horizontal_neighbor(
        data,
        row_base,
        params.current_width,
        gid.x,
        true
    );
    const float right = j2k_fdwt97_horizontal_neighbor(
        data,
        row_base,
        params.current_width,
        gid.x,
        false
    );
    data[row_base + gid.x] = fma(
        params.coefficient,
        left + right,
        data[row_base + gid.x]
    );
}

kernel void j2k_forward_dwt97_lift_vertical(
    device float *data [[buffer(0)]],
    device float *unused [[buffer(1)]],
    constant J2kForwardDwt97Params &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
    (void)unused;
    if (params.current_height < 2u) return;
    if (
        gid.x >= params.current_width ||
        gid.y >= params.current_height ||
        !j2k_fdwt97_is_active_parity(gid.y, params.parity)
    ) {
        return;
    }

    const float top = j2k_fdwt97_vertical_neighbor(
        data,
        params.full_width,
        params.current_height,
        gid.x,
        gid.y,
        true
    );
    const float bottom = j2k_fdwt97_vertical_neighbor(
        data,
        params.full_width,
        params.current_height,
        gid.x,
        gid.y,
        false
    );
    const uint index = gid.y * params.full_width + gid.x;
    data[index] = fma(params.coefficient, top + bottom, data[index]);
}

kernel void j2k_forward_dwt97_deinterleave_horizontal(
    device const float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant J2kForwardDwt97Params &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.current_width || gid.y >= params.current_height) {
        return;
    }

    const uint row_base = gid.y * params.full_width;
    if (params.current_width == 1u) {
        dst[row_base] = src[row_base] * (params.origin_phase == 0u ? 1.0f : 2.0f);
        return;
    }
    if (gid.x < params.low_width) {
        dst[row_base + gid.x] = src[row_base + gid.x * 2u + params.origin_phase] * CODEC_MATH_DWT97_INV_KAPPA;
        return;
    }

    const uint high_index = gid.x - params.low_width;
    dst[row_base + gid.x] = src[row_base + high_index * 2u + 1u - params.origin_phase] * CODEC_MATH_DWT97_KAPPA;
}

kernel void j2k_forward_dwt97_deinterleave_vertical(
    device const float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant J2kForwardDwt97Params &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.current_width || gid.y >= params.current_height) {
        return;
    }

    if (params.current_height == 1u) {
        dst[gid.x] = src[gid.x] * (params.origin_phase == 0u ? 1.0f : 2.0f);
        return;
    }
    if (gid.y < params.low_height) {
        dst[gid.y * params.full_width + gid.x] =
            src[(gid.y * 2u + params.origin_phase) * params.full_width + gid.x] * CODEC_MATH_DWT97_INV_KAPPA;
        return;
    }

    const uint high_index = gid.y - params.low_height;
    dst[gid.y * params.full_width + gid.x] =
        src[(high_index * 2u + 1u - params.origin_phase) * params.full_width + gid.x] * CODEC_MATH_DWT97_KAPPA;
}
