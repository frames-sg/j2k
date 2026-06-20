// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

struct J2kForwardDwt53Params {
    uint full_width;
    uint current_width;
    uint current_height;
    uint low_width;
    uint low_height;
};

struct J2kForwardDwt53BatchedParams {
    uint full_width;
    uint current_width;
    uint current_height;
    uint low_width;
    uint low_height;
    uint component_count;
};

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
    uint2 gid
) {
    const uint row_base = gid.y * full_width;
    if (gid.x < low_width) {
        const uint even = gid.x * 2u;
        const float left = gid.x > 0u
            ? j2k_fdwt53_predict_row(src, row_base, current_width, gid.x - 1u)
            : j2k_fdwt53_predict_row(src, row_base, current_width, 0u);
        const float right = even + 1u < current_width
            ? j2k_fdwt53_predict_row(src, row_base, current_width, gid.x)
            : left;
        dst[row_base + gid.x] =
            src[row_base + even] + floor((left + right) * 0.25f + 0.5f);
        return;
    }

    const uint high_index = gid.x - low_width;
    dst[row_base + gid.x] = j2k_fdwt53_predict_row(
        src,
        row_base,
        current_width,
        high_index
    );
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
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.current_width || gid.y >= params.current_height) {
        return;
    }

   j2k_fdwt53_horizontal_step(
        src,
        dst,
        params.full_width,
        params.current_width,
        params.low_width,
        gid
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
   j2k_fdwt53_horizontal_step(
        src,
        dst,
        params.full_width,
        params.current_width,
        params.low_width,
        gid.xy
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
