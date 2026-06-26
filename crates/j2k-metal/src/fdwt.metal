// SPDX-License-Identifier: MIT OR Apache-2.0

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

struct J2kForwardDwt97Params {
    uint full_width;
    uint current_width;
    uint current_height;
    uint low_width;
    uint low_height;
    uint parity;
    float coefficient;
    uint _reserved;
};

constant float J2K_FDWT97_KAPPA = 1.2301741f;
constant float J2K_FDWT97_INV_KAPPA = 1.0f / 1.2301741f;

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

inline bool j2k_fdwt97_is_active_parity(uint index, uint parity) {
    return (index & 1u) == parity;
}

inline float j2k_fdwt97_horizontal_neighbor(
    device const float *data,
    uint row_base,
    uint width,
    uint x,
    bool update_high,
    bool left_neighbor
) {
    if (update_high) {
        if (left_neighbor) {
            return data[row_base + x - 1u];
        }
        const uint last_even = (width & 1u) == 0u ? width - 2u : width - 1u;
        return (x + 1u < width) ? data[row_base + x + 1u] : data[row_base + last_even];
    }

    if (left_neighbor) {
        return x > 0u ? data[row_base + x - 1u] : data[row_base + 1u];
    }
    return (x + 1u < width) ? data[row_base + x + 1u] : data[row_base + x - 1u];
}

inline float j2k_fdwt97_vertical_neighbor(
    device const float *data,
    uint full_width,
    uint height,
    uint x,
    uint y,
    bool update_high,
    bool top_neighbor
) {
    if (update_high) {
        if (top_neighbor) {
            return data[(y - 1u) * full_width + x];
        }
        const uint last_even = (height & 1u) == 0u ? height - 2u : height - 1u;
        const uint neighbor_y = (y + 1u < height) ? y + 1u : last_even;
        return data[neighbor_y * full_width + x];
    }

    if (top_neighbor) {
        const uint neighbor_y = y > 0u ? y - 1u : 1u;
        return data[neighbor_y * full_width + x];
    }
    const uint neighbor_y = (y + 1u < height) ? y + 1u : y - 1u;
    return data[neighbor_y * full_width + x];
}

kernel void j2k_forward_dwt97_lift_horizontal(
    device float *data [[buffer(0)]],
    device float *unused [[buffer(1)]],
    constant J2kForwardDwt97Params &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    (void)unused;
    if (
        gid.x >= params.current_width ||
        gid.y >= params.current_height ||
        !j2k_fdwt97_is_active_parity(gid.x, params.parity)
    ) {
        return;
    }

    const bool update_high = params.parity == 1u;
    const uint row_base = gid.y * params.full_width;
    const float left = j2k_fdwt97_horizontal_neighbor(
        data,
        row_base,
        params.current_width,
        gid.x,
        update_high,
        true
    );
    const float right = j2k_fdwt97_horizontal_neighbor(
        data,
        row_base,
        params.current_width,
        gid.x,
        update_high,
        false
    );
    data[row_base + gid.x] += params.coefficient * (left + right);
}

kernel void j2k_forward_dwt97_lift_vertical(
    device float *data [[buffer(0)]],
    device float *unused [[buffer(1)]],
    constant J2kForwardDwt97Params &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    (void)unused;
    if (
        gid.x >= params.current_width ||
        gid.y >= params.current_height ||
        !j2k_fdwt97_is_active_parity(gid.y, params.parity)
    ) {
        return;
    }

    const bool update_high = params.parity == 1u;
    const float top = j2k_fdwt97_vertical_neighbor(
        data,
        params.full_width,
        params.current_height,
        gid.x,
        gid.y,
        update_high,
        true
    );
    const float bottom = j2k_fdwt97_vertical_neighbor(
        data,
        params.full_width,
        params.current_height,
        gid.x,
        gid.y,
        update_high,
        false
    );
    data[gid.y * params.full_width + gid.x] += params.coefficient * (top + bottom);
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
    if (gid.x < params.low_width) {
        dst[row_base + gid.x] = src[row_base + gid.x * 2u] * J2K_FDWT97_INV_KAPPA;
        return;
    }

    const uint high_index = gid.x - params.low_width;
    dst[row_base + gid.x] = src[row_base + high_index * 2u + 1u] * J2K_FDWT97_KAPPA;
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

    if (gid.y < params.low_height) {
        dst[gid.y * params.full_width + gid.x] =
            src[(gid.y * 2u) * params.full_width + gid.x] * J2K_FDWT97_INV_KAPPA;
        return;
    }

    const uint high_index = gid.y - params.low_height;
    dst[gid.y * params.full_width + gid.x] =
        src[(high_index * 2u + 1u) * params.full_width + gid.x] * J2K_FDWT97_KAPPA;
}
