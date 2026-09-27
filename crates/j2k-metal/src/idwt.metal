// SPDX-License-Identifier: MIT OR Apache-2.0

struct J2kIdwtSingleDecompositionParams {
    uint x0;
    uint y0;
    uint output_x;
    uint output_y;
    uint width;
    uint height;
    uint ll_x;
    uint ll_y;
    uint ll_width;
    uint ll_height;
    uint hl_x;
    uint hl_y;
    uint hl_width;
    uint hl_height;
    uint lh_x;
    uint lh_y;
    uint lh_width;
    uint lh_height;
    uint hh_x;
    uint hh_y;
    uint hh_width;
    uint hh_height;
};

struct J2kRepeatedIdwtSingleDecompositionParams {
    uint x0;
    uint y0;
    uint output_x;
    uint output_y;
    uint width;
    uint height;
    uint ll_x;
    uint ll_y;
    uint ll_width;
    uint ll_height;
    uint hl_x;
    uint hl_y;
    uint hl_width;
    uint hl_height;
    uint lh_x;
    uint lh_y;
    uint lh_width;
    uint lh_height;
    uint hh_x;
    uint hh_y;
    uint hh_width;
    uint hh_height;
    uint ll_instance_stride;
    uint hl_instance_stride;
    uint lh_instance_stride;
    uint hh_instance_stride;
    uint batch_count;
};

struct J2kIdwt97StepParams {
    float coefficient;
    uint parity;
    uint reserved0;
    uint reserved1;
};

inline uint ceil_div2_u32(uint value) {
    return (value + 1u) >> 1u;
}

inline uint low_index(uint coord, uint origin) {
    return ceil_div2_u32(coord) - ceil_div2_u32(origin);
}

inline uint high_index(uint coord, uint origin) {
    return (coord >> 1u) - (origin >> 1u);
}

inline uint periodic_symmetric_extension_left_u32(uint idx, uint offset) {
    return idx >= offset ? idx - offset : offset - idx;
}

inline uint periodic_symmetric_extension_right_u32(uint idx, uint offset, uint length) {
    const uint new_idx = idx + offset;
    if (new_idx >= length) {
        const uint overshoot = new_idx - length;
        return length - 2u - overshoot;
    }
    return new_idx;
}

inline float reversible53_predict(float s, float left, float right) {
    return s - floor((left + right) * 0.25f + 0.5f);
}

inline float reversible53_update(float s, float left, float right) {
    return s + floor((left + right) * 0.5f);
}

inline void irreversible97_horizontal_step(
    device float *row_ptr,
    uint width,
    uint first,
    float coefficient
) {
    if (first == 0u) {
        const uint left = periodic_symmetric_extension_left_u32(0u, 1u);
        const uint right = periodic_symmetric_extension_right_u32(0u, 1u, width);
        row_ptr[0] = fma(row_ptr[left] + row_ptr[right], coefficient, row_ptr[0]);
    }

    const uint middle_start = first == 0u ? 2u : 1u;
    for (uint x = middle_start; x + 1u < width; x += 2u) {
        row_ptr[x] = fma(row_ptr[x - 1u] + row_ptr[x + 1u], coefficient, row_ptr[x]);
    }

    if (width > 1u && ((width - 1u) & 1u) == first) {
        const uint x = width - 1u;
        const uint left = periodic_symmetric_extension_left_u32(x, 1u);
        const uint right = periodic_symmetric_extension_right_u32(x, 1u, width);
        row_ptr[x] = fma(row_ptr[left] + row_ptr[right], coefficient, row_ptr[x]);
    }
}

kernel void j2k_idwt_interleave(
    device const float *ll [[buffer(0)]],
    device const float *hl [[buffer(1)]],
    device const float *lh [[buffer(2)]],
    device const float *hh [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant J2kIdwtSingleDecompositionParams &params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint global_x = params.x0 + params.output_x + gid.x;
    const uint global_y = params.y0 + params.output_y + gid.y;
    // JPEG 2000 assigns low-pass samples to even reference-grid
    // coordinates.  The decomposition origin only determines which band is
    // encountered first in this tile; it must not redefine that parity.
    const bool low_x = (global_x & 1u) == 0u;
    const bool low_y = (global_y & 1u) == 0u;
    const uint full_band_x = low_x ? low_index(global_x, params.x0) : high_index(global_x, params.x0);
    const uint full_band_y = low_y ? low_index(global_y, params.y0) : high_index(global_y, params.y0);
    const uint out_idx = gid.y * params.width + gid.x;

    if (low_y && low_x) {
        const uint band_x = full_band_x - params.ll_x;
        const uint band_y = full_band_y - params.ll_y;
        out[out_idx] = (band_x < params.ll_width && band_y < params.ll_height)
            ? ll[band_y * params.ll_width + band_x]
            : 0.0f;
    } else if (low_y) {
        const uint band_x = full_band_x - params.hl_x;
        const uint band_y = full_band_y - params.hl_y;
        out[out_idx] = (band_x < params.hl_width && band_y < params.hl_height)
            ? hl[band_y * params.hl_width + band_x]
            : 0.0f;
    } else if (low_x) {
        const uint band_x = full_band_x - params.lh_x;
        const uint band_y = full_band_y - params.lh_y;
        out[out_idx] = (band_x < params.lh_width && band_y < params.lh_height)
            ? lh[band_y * params.lh_width + band_x]
            : 0.0f;
    } else {
        const uint band_x = full_band_x - params.hh_x;
        const uint band_y = full_band_y - params.hh_y;
        out[out_idx] = (band_x < params.hh_width && band_y < params.hh_height)
            ? hh[band_y * params.hh_width + band_x]
            : 0.0f;
    }
}

kernel void j2k_idwt_interleave_batched(
    device const float *ll [[buffer(0)]],
    device const float *hl [[buffer(1)]],
    device const float *lh [[buffer(2)]],
    device const float *hh [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant J2kRepeatedIdwtSingleDecompositionParams &params [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.batch_count) {
        return;
    }

    const uint global_x = params.x0 + params.output_x + gid.x;
    const uint global_y = params.y0 + params.output_y + gid.y;
    // Low/high assignment is defined in the reference grid.  For an odd
    // tile origin, the first sample is therefore high-pass.
    const bool low_x = (global_x & 1u) == 0u;
    const bool low_y = (global_y & 1u) == 0u;
    const uint full_band_x = low_x ? low_index(global_x, params.x0) : high_index(global_x, params.x0);
    const uint full_band_y = low_y ? low_index(global_y, params.y0) : high_index(global_y, params.y0);
    const uint out_plane_len = params.width * params.height;
    const uint out_idx = gid.z * out_plane_len + gid.y * params.width + gid.x;

    if (low_y && low_x) {
        const uint band_x = full_band_x - params.ll_x;
        const uint band_y = full_band_y - params.ll_y;
        out[out_idx] = (band_x < params.ll_width && band_y < params.ll_height)
            ? ll[gid.z * params.ll_instance_stride + band_y * params.ll_width + band_x]
            : 0.0f;
    } else if (low_y) {
        const uint band_x = full_band_x - params.hl_x;
        const uint band_y = full_band_y - params.hl_y;
        out[out_idx] = (band_x < params.hl_width && band_y < params.hl_height)
            ? hl[gid.z * params.hl_instance_stride + band_y * params.hl_width + band_x]
            : 0.0f;
    } else if (low_x) {
        const uint band_x = full_band_x - params.lh_x;
        const uint band_y = full_band_y - params.lh_y;
        out[out_idx] = (band_x < params.lh_width && band_y < params.lh_height)
            ? lh[gid.z * params.lh_instance_stride + band_y * params.lh_width + band_x]
            : 0.0f;
    } else {
        const uint band_x = full_band_x - params.hh_x;
        const uint band_y = full_band_y - params.hh_y;
        out[out_idx] = (band_x < params.hh_width && band_y < params.hh_height)
            ? hh[gid.z * params.hh_instance_stride + band_y * params.hh_width + band_x]
            : 0.0f;
    }
}

kernel void j2k_idwt_irreversible97_interleave_horizontal_scale(
    device const float *ll [[buffer(0)]],
    device const float *hl [[buffer(1)]],
    device const float *lh [[buffer(2)]],
    device const float *hh [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant J2kIdwtSingleDecompositionParams &params [[buffer(5)]],
    constant float &high_pass [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint global_x = params.x0 + params.output_x + gid.x;
    const uint global_y = params.y0 + params.output_y + gid.y;
    const bool low_x = (global_x & 1u) == 0u;
    const bool low_y = (global_y & 1u) == 0u;
    const uint full_band_x = low_x ? low_index(global_x, params.x0) : high_index(global_x, params.x0);
    const uint full_band_y = low_y ? low_index(global_y, params.y0) : high_index(global_y, params.y0);
    const uint out_idx = gid.y * params.width + gid.x;
    float sample;

    if (low_y && low_x) {
        const uint band_x = full_band_x - params.ll_x;
        const uint band_y = full_band_y - params.ll_y;
        sample = (band_x < params.ll_width && band_y < params.ll_height)
            ? ll[band_y * params.ll_width + band_x]
            : 0.0f;
    } else if (low_y) {
        const uint band_x = full_band_x - params.hl_x;
        const uint band_y = full_band_y - params.hl_y;
        sample = (band_x < params.hl_width && band_y < params.hl_height)
            ? hl[band_y * params.hl_width + band_x]
            : 0.0f;
    } else if (low_x) {
        const uint band_x = full_band_x - params.lh_x;
        const uint band_y = full_band_y - params.lh_y;
        sample = (band_x < params.lh_width && band_y < params.lh_height)
            ? lh[band_y * params.lh_width + band_x]
            : 0.0f;
    } else {
        const uint band_x = full_band_x - params.hh_x;
        const uint band_y = full_band_y - params.hh_y;
        sample = (band_x < params.hh_width && band_y < params.hh_height)
            ? hh[band_y * params.hh_width + band_x]
            : 0.0f;
    }

    if (params.width == 1u) {
        if (((params.x0 + params.output_x) & 1u) != 0u) {
            sample *= 0.5f;
        }
    } else {
        const uint first_even_x = (params.x0 + params.output_x) & 1u;
        const float KAPPA = CODEC_MATH_DWT97_KAPPA;
        sample *= (gid.x & 1u) == first_even_x ? KAPPA : high_pass;
    }
    out[out_idx] = sample;
}

kernel void j2k_idwt_irreversible97_interleave_horizontal_scale_batched(
    device const float *ll [[buffer(0)]],
    device const float *hl [[buffer(1)]],
    device const float *lh [[buffer(2)]],
    device const float *hh [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant J2kRepeatedIdwtSingleDecompositionParams &params [[buffer(5)]],
    constant float &high_pass [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.batch_count) {
        return;
    }

    const uint global_x = params.x0 + params.output_x + gid.x;
    const uint global_y = params.y0 + params.output_y + gid.y;
    const bool low_x = (global_x & 1u) == 0u;
    const bool low_y = (global_y & 1u) == 0u;
    const uint full_band_x = low_x ? low_index(global_x, params.x0) : high_index(global_x, params.x0);
    const uint full_band_y = low_y ? low_index(global_y, params.y0) : high_index(global_y, params.y0);
    const uint out_plane_len = params.width * params.height;
    const uint out_idx = gid.z * out_plane_len + gid.y * params.width + gid.x;
    float sample;

    if (low_y && low_x) {
        const uint band_x = full_band_x - params.ll_x;
        const uint band_y = full_band_y - params.ll_y;
        sample = (band_x < params.ll_width && band_y < params.ll_height)
            ? ll[gid.z * params.ll_instance_stride + band_y * params.ll_width + band_x]
            : 0.0f;
    } else if (low_y) {
        const uint band_x = full_band_x - params.hl_x;
        const uint band_y = full_band_y - params.hl_y;
        sample = (band_x < params.hl_width && band_y < params.hl_height)
            ? hl[gid.z * params.hl_instance_stride + band_y * params.hl_width + band_x]
            : 0.0f;
    } else if (low_x) {
        const uint band_x = full_band_x - params.lh_x;
        const uint band_y = full_band_y - params.lh_y;
        sample = (band_x < params.lh_width && band_y < params.lh_height)
            ? lh[gid.z * params.lh_instance_stride + band_y * params.lh_width + band_x]
            : 0.0f;
    } else {
        const uint band_x = full_band_x - params.hh_x;
        const uint band_y = full_band_y - params.hh_y;
        sample = (band_x < params.hh_width && band_y < params.hh_height)
            ? hh[gid.z * params.hh_instance_stride + band_y * params.hh_width + band_x]
            : 0.0f;
    }

    if (params.width == 1u) {
        if (((params.x0 + params.output_x) & 1u) != 0u) {
            sample *= 0.5f;
        }
    } else {
        const uint first_even_x = (params.x0 + params.output_x) & 1u;
        const float KAPPA = CODEC_MATH_DWT97_KAPPA;
        sample *= (gid.x & 1u) == first_even_x ? KAPPA : high_pass;
    }
    out[out_idx] = sample;
}

kernel void j2k_idwt_reversible53_horizontal_pass(
    device float *out [[buffer(0)]],
    constant J2kIdwtSingleDecompositionParams &params [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.height) {
        return;
    }

    device float *row_ptr = out + gid * params.width;

    if (params.width == 1u) {
        if (((params.x0 + params.output_x) & 1u) != 0u) {
            row_ptr[0] *= 0.5f;
        }
        return;
    }

    const uint first_even_x = (params.x0 + params.output_x) & 1u;
    const uint first_odd_x = 1u - first_even_x;

    if (first_even_x == 0u) {
        const uint left = periodic_symmetric_extension_left_u32(0u, 1u);
        const uint right = periodic_symmetric_extension_right_u32(0u, 1u, params.width);
        row_ptr[0] = reversible53_predict(row_ptr[0], row_ptr[left], row_ptr[right]);
    }

    const uint even_middle_start = first_even_x == 0u ? 2u : 1u;
    for (uint x = even_middle_start; x + 1u < params.width; x += 2u) {
        row_ptr[x] = reversible53_predict(row_ptr[x], row_ptr[x - 1u], row_ptr[x + 1u]);
    }

    if (((params.width - 1u) & 1u) == first_even_x) {
        const uint x = params.width - 1u;
        const uint left = periodic_symmetric_extension_left_u32(x, 1u);
        const uint right = periodic_symmetric_extension_right_u32(x, 1u, params.width);
        row_ptr[x] = reversible53_predict(row_ptr[x], row_ptr[left], row_ptr[right]);
    }

    if (first_odd_x == 0u) {
        const uint left = periodic_symmetric_extension_left_u32(0u, 1u);
        const uint right = periodic_symmetric_extension_right_u32(0u, 1u, params.width);
        row_ptr[0] = reversible53_update(row_ptr[0], row_ptr[left], row_ptr[right]);
    }

    const uint odd_middle_start = first_odd_x == 0u ? 2u : 1u;
    for (uint x = odd_middle_start; x + 1u < params.width; x += 2u) {
        row_ptr[x] = reversible53_update(row_ptr[x], row_ptr[x - 1u], row_ptr[x + 1u]);
    }

    if (((params.width - 1u) & 1u) == first_odd_x) {
        const uint x = params.width - 1u;
        const uint left = periodic_symmetric_extension_left_u32(x, 1u);
        const uint right = periodic_symmetric_extension_right_u32(x, 1u, params.width);
        row_ptr[x] = reversible53_update(row_ptr[x], row_ptr[left], row_ptr[right]);
    }
}

kernel void j2k_idwt_reversible53_horizontal_pass_batched(
    device float *out [[buffer(0)]],
    constant J2kRepeatedIdwtSingleDecompositionParams &params [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.height || gid.y >= params.batch_count) {
        return;
    }

    const uint plane_len = params.width * params.height;
    device float *row_ptr = out + gid.y * plane_len + gid.x * params.width;

    if (params.width == 1u) {
        if (((params.x0 + params.output_x) & 1u) != 0u) {
            row_ptr[0] *= 0.5f;
        }
        return;
    }

    const uint first_even_x = (params.x0 + params.output_x) & 1u;
    const uint first_odd_x = 1u - first_even_x;

    if (first_even_x == 0u) {
        const uint left = periodic_symmetric_extension_left_u32(0u, 1u);
        const uint right = periodic_symmetric_extension_right_u32(0u, 1u, params.width);
        row_ptr[0] = reversible53_predict(row_ptr[0], row_ptr[left], row_ptr[right]);
    }

    const uint even_middle_start = first_even_x == 0u ? 2u : 1u;
    for (uint x = even_middle_start; x + 1u < params.width; x += 2u) {
        row_ptr[x] = reversible53_predict(row_ptr[x], row_ptr[x - 1u], row_ptr[x + 1u]);
    }

    if (((params.width - 1u) & 1u) == first_even_x) {
        const uint x = params.width - 1u;
        const uint left = periodic_symmetric_extension_left_u32(x, 1u);
        const uint right = periodic_symmetric_extension_right_u32(x, 1u, params.width);
        row_ptr[x] = reversible53_predict(row_ptr[x], row_ptr[left], row_ptr[right]);
    }

    if (first_odd_x == 0u) {
        const uint left = periodic_symmetric_extension_left_u32(0u, 1u);
        const uint right = periodic_symmetric_extension_right_u32(0u, 1u, params.width);
        row_ptr[0] = reversible53_update(row_ptr[0], row_ptr[left], row_ptr[right]);
    }

    const uint odd_middle_start = first_odd_x == 0u ? 2u : 1u;
    for (uint x = odd_middle_start; x + 1u < params.width; x += 2u) {
        row_ptr[x] = reversible53_update(row_ptr[x], row_ptr[x - 1u], row_ptr[x + 1u]);
    }

    if (((params.width - 1u) & 1u) == first_odd_x) {
        const uint x = params.width - 1u;
        const uint left = periodic_symmetric_extension_left_u32(x, 1u);
        const uint right = periodic_symmetric_extension_right_u32(x, 1u, params.width);
        row_ptr[x] = reversible53_update(row_ptr[x], row_ptr[left], row_ptr[right]);
    }
}

kernel void j2k_idwt_reversible53_vertical_pass(
    device float *out [[buffer(0)]],
    constant J2kIdwtSingleDecompositionParams &params [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.width) {
        return;
    }

    if (params.height == 1u) {
        if (((params.y0 + params.output_y) & 1u) != 0u) {
            out[gid] *= 0.5f;
        }
        return;
    }

    const uint first_even_y = (params.y0 + params.output_y) & 1u;
    const uint first_odd_y = 1u - first_even_y;

    for (uint row = first_even_y; row < params.height; row += 2u) {
        const uint row_above = periodic_symmetric_extension_left_u32(row, 1u);
        const uint row_below = periodic_symmetric_extension_right_u32(row, 1u, params.height);
        const uint idx = row * params.width + gid;
        out[idx] = reversible53_predict(
            out[idx],
            out[row_above * params.width + gid],
            out[row_below * params.width + gid]
        );
    }

    for (uint row = first_odd_y; row < params.height; row += 2u) {
        const uint row_above = periodic_symmetric_extension_left_u32(row, 1u);
        const uint row_below = periodic_symmetric_extension_right_u32(row, 1u, params.height);
        const uint idx = row * params.width + gid;
        out[idx] = reversible53_update(
            out[idx],
            out[row_above * params.width + gid],
            out[row_below * params.width + gid]
        );
    }
}

kernel void j2k_idwt_reversible53_vertical_pass_batched(
    device float *out [[buffer(0)]],
    constant J2kRepeatedIdwtSingleDecompositionParams &params [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.batch_count) {
        return;
    }

    const uint plane_len = params.width * params.height;
    device float *plane = out + gid.y * plane_len;

    if (params.height == 1u) {
        if (((params.y0 + params.output_y) & 1u) != 0u) {
            plane[gid.x] *= 0.5f;
        }
        return;
    }

    const uint first_even_y = (params.y0 + params.output_y) & 1u;
    const uint first_odd_y = 1u - first_even_y;

    for (uint row = first_even_y; row < params.height; row += 2u) {
        const uint row_above = periodic_symmetric_extension_left_u32(row, 1u);
        const uint row_below = periodic_symmetric_extension_right_u32(row, 1u, params.height);
        const uint idx = row * params.width + gid.x;
        plane[idx] = reversible53_predict(
            plane[idx],
            plane[row_above * params.width + gid.x],
            plane[row_below * params.width + gid.x]
        );
    }

    for (uint row = first_odd_y; row < params.height; row += 2u) {
        const uint row_above = periodic_symmetric_extension_left_u32(row, 1u);
        const uint row_below = periodic_symmetric_extension_right_u32(row, 1u, params.height);
        const uint idx = row * params.width + gid.x;
        plane[idx] = reversible53_update(
            plane[idx],
            plane[row_above * params.width + gid.x],
            plane[row_below * params.width + gid.x]
        );
    }
}

kernel void j2k_idwt_irreversible97_horizontal_scale(
    device float *out [[buffer(0)]],
    constant J2kIdwtSingleDecompositionParams &params [[buffer(1)]],
    constant float &high_pass [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    out += ulong(gid.z) * params.width * params.height;
    const float KAPPA = CODEC_MATH_DWT97_KAPPA;
    float sample = out[gid.y * params.width + gid.x];

    if (params.width == 1u) {
        if (((params.x0 + params.output_x) & 1u) != 0u) {
            sample *= 0.5f;
        }
    } else {
        const uint first_even_x = (params.x0 + params.output_x) & 1u;
        sample *= (gid.x & 1u) == first_even_x ? KAPPA : high_pass;
    }

    out[gid.y * params.width + gid.x] = sample;
}

// Fused 9/7 lifting. Each dispatch replaces four (horizontal) or five
// (vertical scale + four lifts) full-plane passes. A threadgroup owns whole
// rows (horizontal) or a strip of columns (vertical) and walks its tiles in
// order: a tile is loaded with a four-sample halo, every step is applied in
// threadgroup memory, and the tile is written back once. The halo before a
// tile comes from a carry of the previous tile's pre-lift samples (its device
// copy has already been overwritten); the halo after it is still unwritten.
// Step s only updates positions within 3 - s samples of the tile, so each
// neighbour it reads was produced by step s - 1 in the same buffer. Updates
// use the original one-pass-per-step kernels' exact expressions and edge
// mirroring (kept as the oracle in `irreversible/parity_tests.rs`), so results
// are bit-identical to running one full pass per step.

constant uint J2K_IDWT97_HALO = 4u;
constant uint J2K_IDWT97_ROW_TILE = 128u;
constant uint J2K_IDWT97_ROWS_PER_GROUP = 4u;
constant uint J2K_IDWT97_ROW_THREADS = 64u;
constant uint J2K_IDWT97_COL_TILE = 32u;
constant uint J2K_IDWT97_COL_ROWS = 64u;
constant uint J2K_IDWT97_COL_ROW_THREADS = 8u;

struct J2kIdwt97LiftSteps {
    float4 coefficients;
    uint first_parity;
    uint high_pass_bits;
    uint reserved0;
    uint reserved1;
};

inline uint idwt97_left(uint x) {
    return periodic_symmetric_extension_left_u32(x, 1u);
}

inline uint idwt97_right(uint x, uint length) {
    return periodic_symmetric_extension_right_u32(x, 1u, length);
}

inline uint idwt97_first_of_parity(uint lo, uint parity) {
    return lo + ((lo & 1u) != parity ? 1u : 0u);
}

kernel void j2k_idwt_irreversible97_horizontal_lift_fused(
    device float *out [[buffer(0)]],
    constant J2kIdwtSingleDecompositionParams &params [[buffer(1)]],
    constant J2kIdwt97LiftSteps &steps [[buffer(2)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 local [[thread_position_in_threadgroup]]
) {
    threadgroup float rows[J2K_IDWT97_ROWS_PER_GROUP][J2K_IDWT97_ROW_TILE + 2u * J2K_IDWT97_HALO];
    threadgroup float carry[J2K_IDWT97_ROWS_PER_GROUP][J2K_IDWT97_HALO];
    const uint width = params.width;
    const uint y = group.y * J2K_IDWT97_ROWS_PER_GROUP + local.y;
    const bool row_active = y < params.height && width > 1u;
    device float *row_ptr = out + ulong(group.z) * width * params.height + ulong(y) * width;
    threadgroup float *row = rows[local.y];

    for (uint tile_start = 0u; tile_start < width; tile_start += J2K_IDWT97_ROW_TILE) {
        const uint tile_end = min(tile_start + J2K_IDWT97_ROW_TILE, width);
        const uint load_start = tile_start > J2K_IDWT97_HALO ? tile_start - J2K_IDWT97_HALO : 0u;
        const uint load_end = min(tile_end + J2K_IDWT97_HALO, width);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row_active) {
            for (uint x = tile_start + local.x; x < load_end; x += J2K_IDWT97_ROW_THREADS) {
                row[x - load_start] = row_ptr[x];
            }
            for (uint x = load_start + local.x; x < tile_start; x += J2K_IDWT97_ROW_THREADS) {
                row[x - load_start] = carry[local.y][x - (tile_start - J2K_IDWT97_HALO)];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row_active && tile_end < width && local.x < J2K_IDWT97_HALO) {
            carry[local.y][local.x] = row[tile_end - J2K_IDWT97_HALO + local.x - load_start];
        }

        for (uint step = 0u; step < 4u; ++step) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (!row_active) {
                continue;
            }
            const uint reach = 3u - step;
            const uint lo = tile_start > reach ? tile_start - reach : 0u;
            const uint hi = min(tile_end + reach, width);
            const uint first = idwt97_first_of_parity(lo, steps.first_parity ^ (step & 1u));
            const float coefficient = steps.coefficients[step];
            for (uint x = first + 2u * local.x; x < hi; x += 2u * J2K_IDWT97_ROW_THREADS) {
                const uint i = x - load_start;
                row[i] = fma(
                    row[idwt97_left(x) - load_start] + row[idwt97_right(x, width) - load_start],
                    coefficient,
                    row[i]
                );
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row_active) {
            for (uint x = tile_start + local.x; x < tile_end; x += J2K_IDWT97_ROW_THREADS) {
                row_ptr[x] = row[x - load_start];
            }
        }
    }
}

kernel void j2k_idwt_irreversible97_vertical_fused(
    device float *out [[buffer(0)]],
    constant J2kIdwtSingleDecompositionParams &params [[buffer(1)]],
    constant J2kIdwt97LiftSteps &steps [[buffer(2)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 local [[thread_position_in_threadgroup]]
) {
    threadgroup float tile[J2K_IDWT97_COL_ROWS + 2u * J2K_IDWT97_HALO][J2K_IDWT97_COL_TILE];
    threadgroup float carry[J2K_IDWT97_HALO][J2K_IDWT97_COL_TILE];
    const uint width = params.width;
    const uint height = params.height;
    const uint x = group.x * J2K_IDWT97_COL_TILE + local.x;
    const bool column_active = x < width;
    const bool lift = column_active && height > 1u;
    device float *plane = out + ulong(group.z) * width * height;

    // Vertical scale, exactly as the original full-plane vertical scale pass.
    const float KAPPA = CODEC_MATH_DWT97_KAPPA;
    const float high_pass = as_type<float>(steps.high_pass_bits);
    const uint first_even_y = (params.y0 + params.output_y) & 1u;

    for (uint tile_start = 0u; tile_start < height; tile_start += J2K_IDWT97_COL_ROWS) {
        const uint tile_end = min(tile_start + J2K_IDWT97_COL_ROWS, height);
        const uint load_start = tile_start > J2K_IDWT97_HALO ? tile_start - J2K_IDWT97_HALO : 0u;
        const uint load_end = min(tile_end + J2K_IDWT97_HALO, height);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (column_active) {
            for (uint y = tile_start + local.y; y < load_end; y += J2K_IDWT97_COL_ROW_THREADS) {
                float sample = plane[y * width + x];
                if (height == 1u) {
                    if (first_even_y != 0u) {
                        sample *= 0.5f;
                    }
                } else {
                    sample *= (y & 1u) == first_even_y ? KAPPA : high_pass;
                }
                tile[y - load_start][local.x] = sample;
            }
            for (uint y = load_start + local.y; y < tile_start; y += J2K_IDWT97_COL_ROW_THREADS) {
                tile[y - load_start][local.x] = carry[y - (tile_start - J2K_IDWT97_HALO)][local.x];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (column_active && tile_end < height && local.y < J2K_IDWT97_HALO) {
            carry[local.y][local.x] = tile[tile_end - J2K_IDWT97_HALO + local.y - load_start][local.x];
        }

        for (uint step = 0u; step < 4u; ++step) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (!lift) {
                continue;
            }
            const uint reach = 3u - step;
            const uint lo = tile_start > reach ? tile_start - reach : 0u;
            const uint hi = min(tile_end + reach, height);
            const uint first = idwt97_first_of_parity(lo, steps.first_parity ^ (step & 1u));
            const float coefficient = steps.coefficients[step];
            for (uint y = first + 2u * local.y; y < hi; y += 2u * J2K_IDWT97_COL_ROW_THREADS) {
                const uint i = y - load_start;
                tile[i][local.x] = fma(
                    tile[idwt97_left(y) - load_start][local.x] +
                        tile[idwt97_right(y, height) - load_start][local.x],
                    coefficient,
                    tile[i][local.x]
                );
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (column_active) {
            for (uint y = tile_start + local.y; y < tile_end; y += J2K_IDWT97_COL_ROW_THREADS) {
                plane[y * width + x] = tile[y - load_start][local.x];
            }
        }
    }
}
