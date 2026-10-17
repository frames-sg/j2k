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

template<typename Params>
inline float idwt_interleaved_sample(
    device const float *ll,
    device const float *hl,
    device const float *lh,
    device const float *hh,
    constant Params &params,
    uint ll_instance_offset,
    uint hl_instance_offset,
    uint lh_instance_offset,
    uint hh_instance_offset,
    uint x,
    uint y
) {
    const uint global_x = params.x0 + params.output_x + x;
    const uint global_y = params.y0 + params.output_y + y;
    const bool low_x = (global_x & 1u) == 0u;
    const bool low_y = (global_y & 1u) == 0u;
    const uint full_band_x = low_x ? low_index(global_x, params.x0) : high_index(global_x, params.x0);
    const uint full_band_y = low_y ? low_index(global_y, params.y0) : high_index(global_y, params.y0);
    float sample;

    if (low_y && low_x) {
        const uint band_x = full_band_x - params.ll_x;
        const uint band_y = full_band_y - params.ll_y;
        sample = (band_x < params.ll_width && band_y < params.ll_height)
            ? ll[ll_instance_offset + band_y * params.ll_width + band_x]
            : 0.0f;
    } else if (low_y) {
        const uint band_x = full_band_x - params.hl_x;
        const uint band_y = full_band_y - params.hl_y;
        sample = (band_x < params.hl_width && band_y < params.hl_height)
            ? hl[hl_instance_offset + band_y * params.hl_width + band_x]
            : 0.0f;
    } else if (low_x) {
        const uint band_x = full_band_x - params.lh_x;
        const uint band_y = full_band_y - params.lh_y;
        sample = (band_x < params.lh_width && band_y < params.lh_height)
            ? lh[lh_instance_offset + band_y * params.lh_width + band_x]
            : 0.0f;
    } else {
        const uint band_x = full_band_x - params.hh_x;
        const uint band_y = full_band_y - params.hh_y;
        sample = (band_x < params.hh_width && band_y < params.hh_height)
            ? hh[hh_instance_offset + band_y * params.hh_width + band_x]
            : 0.0f;
    }

    return sample;
}

constant uint J2K_IDWT53_HALO = 2u;
constant uint J2K_IDWT53_ROW_TILE = 128u;
constant uint J2K_IDWT53_ROWS_PER_GROUP = 4u;
constant uint J2K_IDWT53_ROW_THREADS = 64u;

template<typename Params>
inline void idwt53_interleave_horizontal_fused_core(
    device const float *ll,
    device const float *hl,
    device const float *lh,
    device const float *hh,
    device float *out,
    constant Params &params,
    uint ll_instance_offset,
    uint hl_instance_offset,
    uint lh_instance_offset,
    uint hh_instance_offset,
    uint instance,
    uint3 group,
    uint3 local,
    threadgroup float *rows
) {
    const uint row_stride = J2K_IDWT53_ROW_TILE + 2u * J2K_IDWT53_HALO;
    threadgroup float *row = rows + local.y * row_stride;
    const uint width = params.width;
    const uint y = group.y * J2K_IDWT53_ROWS_PER_GROUP + local.y;
    const uint tile_start = group.x * J2K_IDWT53_ROW_TILE;
    const uint tile_end = min(tile_start + J2K_IDWT53_ROW_TILE, width);
    const uint load_start = tile_start > J2K_IDWT53_HALO ? tile_start - J2K_IDWT53_HALO : 0u;
    const uint load_end = min(tile_end + J2K_IDWT53_HALO, width);
    const bool row_active = y < params.height;
    const uint first_even_x = (params.x0 + params.output_x) & 1u;

    if (row_active) {
        for (uint x = load_start + local.x; x < load_end; x += J2K_IDWT53_ROW_THREADS) {
            row[x - load_start] = idwt_interleaved_sample(
                ll,
                hl,
                lh,
                hh,
                params,
                ll_instance_offset,
                hl_instance_offset,
                lh_instance_offset,
                hh_instance_offset,
                x,
                y
            );
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row_active && width == 1u) {
        if (local.x == 0u && first_even_x != 0u) {
            row[0] *= 0.5f;
        }
    } else if (row_active) {
        const uint lo = tile_start > 1u ? tile_start - 1u : 0u;
        const uint hi = min(tile_end + 1u, width);
        const uint first = lo + ((lo & 1u) != first_even_x ? 1u : 0u);
        for (uint x = first + 2u * local.x; x < hi; x += 2u * J2K_IDWT53_ROW_THREADS) {
            const uint i = x - load_start;
            row[i] = reversible53_predict(
                row[i],
                row[periodic_symmetric_extension_left_u32(x, 1u) - load_start],
                row[periodic_symmetric_extension_right_u32(x, 1u, width) - load_start]
            );
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row_active && width > 1u) {
        const uint first_odd_x = 1u - first_even_x;
        const uint first = tile_start + ((tile_start & 1u) != first_odd_x ? 1u : 0u);
        for (uint x = first + 2u * local.x; x < tile_end; x += 2u * J2K_IDWT53_ROW_THREADS) {
            const uint i = x - load_start;
            row[i] = reversible53_update(
                row[i],
                row[periodic_symmetric_extension_left_u32(x, 1u) - load_start],
                row[periodic_symmetric_extension_right_u32(x, 1u, width) - load_start]
            );
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row_active) {
        const ulong plane_offset = ulong(instance) * width * params.height;
        for (uint x = tile_start + local.x; x < tile_end; x += J2K_IDWT53_ROW_THREADS) {
            out[plane_offset + ulong(y) * width + x] = row[x - load_start];
        }
    }
}

kernel void j2k_idwt_reversible53_interleave_horizontal_fused(
    device const float *ll [[buffer(0)]],
    device const float *hl [[buffer(1)]],
    device const float *lh [[buffer(2)]],
    device const float *hh [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant J2kIdwtSingleDecompositionParams &params [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 local [[thread_position_in_threadgroup]]
) {
    threadgroup float rows[
        J2K_IDWT53_ROWS_PER_GROUP * (J2K_IDWT53_ROW_TILE + 2u * J2K_IDWT53_HALO)
    ];
    idwt53_interleave_horizontal_fused_core(
        ll, hl, lh, hh, out, params,
        0u, 0u, 0u, 0u, 0u, group, local, rows
    );
}

kernel void j2k_idwt_reversible53_interleave_horizontal_fused_batched(
    device const float *ll [[buffer(0)]],
    device const float *hl [[buffer(1)]],
    device const float *lh [[buffer(2)]],
    device const float *hh [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant J2kRepeatedIdwtSingleDecompositionParams &params [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 local [[thread_position_in_threadgroup]]
) {
    threadgroup float rows[
        J2K_IDWT53_ROWS_PER_GROUP * (J2K_IDWT53_ROW_TILE + 2u * J2K_IDWT53_HALO)
    ];
    idwt53_interleave_horizontal_fused_core(
        ll, hl, lh, hh, out, params,
        group.z * params.ll_instance_stride,
        group.z * params.hl_instance_stride,
        group.z * params.lh_instance_stride,
        group.z * params.hh_instance_stride,
        group.z,
        group,
        local,
        rows
    );
}

constant uint J2K_IDWT53_2D_TILE_W = 60u;
constant uint J2K_IDWT53_2D_TILE_H = 60u;
constant uint J2K_IDWT53_2D_LANES = 32u;
constant uint J2K_IDWT53_2D_ROWS = 8u;
// A loaded row spans the tile and both halos: 32 low-pass and 32 high-pass
// samples, one of each per lane.
constant uint J2K_IDWT53_2D_STRIDE = J2K_IDWT53_2D_TILE_W + 2u * J2K_IDWT53_HALO;

inline float idwt53_band_sample(
    device const float *band,
    uint band_width,
    uint band_height,
    uint band_x,
    uint band_y
) {
    return (band_x < band_width && band_y < band_height)
        ? band[band_y * band_width + band_x]
        : 0.0f;
}

// One 5/3 synthesis level in a single pass: each threadgroup loads its output
// tile plus a two-sample halo from the four sub-bands, lifts the rows, then
// the tile's columns, and writes the tile once. Every sample sees the same
// operations on the same values as the interleave/horizontal and vertical
// kernels, so the output is identical. Each 32-lane SIMD group keeps a row's
// low/high pairs in registers for the horizontal lifts; the vertical lifts
// use the shared tile. Needs width and height of at least 2.
// Writes one synthesized sample of a level to its float plane.
struct J2kIdwtPlaneWriter {
    device float *out;
    uint width;

    void write(uint x, uint y, float value) const {
        out[ulong(y) * width + x] = value;
    }
};

template<typename Params, typename Writer>
inline void idwt53_fused_2d_core(
    device const float *ll,
    device const float *hl,
    device const float *lh,
    device const float *hh,
    Writer writer,
    constant Params &params,
    uint2 group,
    uint2 local,
    threadgroup float *tile
) {
    const uint width = params.width;
    const uint height = params.height;
    const uint tx0 = group.x * J2K_IDWT53_2D_TILE_W;
    const uint ty0 = group.y * J2K_IDWT53_2D_TILE_H;
    if (tx0 >= width || ty0 >= height) {
        return;
    }
    const uint tx1 = min(tx0 + J2K_IDWT53_2D_TILE_W, width);
    const uint ty1 = min(ty0 + J2K_IDWT53_2D_TILE_H, height);
    const uint lx0 = tx0 > J2K_IDWT53_HALO ? tx0 - J2K_IDWT53_HALO : 0u;
    const uint ly0 = ty0 > J2K_IDWT53_HALO ? ty0 - J2K_IDWT53_HALO : 0u;
    const uint lx1 = min(tx1 + J2K_IDWT53_HALO, width);
    const uint ly1 = min(ty1 + J2K_IDWT53_HALO, height);
    const uint origin_x = params.x0 + params.output_x;
    const uint origin_y = params.y0 + params.output_y;

    // Low-pass columns sit at even global x and come from LL or LH; the k-th
    // low-pass column of the row reads band column `low_band_x + k`.
    const uint low_x = lx0 + ((origin_x + lx0) & 1u);
    const uint high_x = lx0 + ((origin_x + lx0 + 1u) & 1u);
    const uint low_band_x = low_index(origin_x + low_x, params.x0);
    const uint high_band_x = high_index(origin_x + high_x, params.x0);
    const uint xl = low_x + 2u * local.x;
    const uint xh = high_x + 2u * local.x;
    const bool low_first = low_x < high_x;
    for (uint y = ly0 + local.y; y < ly1; y += J2K_IDWT53_2D_ROWS) {
        const uint global_y = origin_y + y;
        const bool low_y = (global_y & 1u) == 0u;
        const uint band_y = low_y ? low_index(global_y, params.y0) : high_index(global_y, params.y0);
        float low = 0.0f;
        float high = 0.0f;
        if (xl < lx1) {
            low = low_y
                ? idwt53_band_sample(ll, params.ll_width, params.ll_height,
                    low_band_x + local.x - params.ll_x, band_y - params.ll_y)
                : idwt53_band_sample(lh, params.lh_width, params.lh_height,
                    low_band_x + local.x - params.lh_x, band_y - params.lh_y);
        }
        if (xh < lx1) {
            high = low_y
                ? idwt53_band_sample(hl, params.hl_width, params.hl_height,
                    high_band_x + local.x - params.hl_x, band_y - params.hl_y)
                : idwt53_band_sample(hh, params.hh_width, params.hh_height,
                    high_band_x + local.x - params.hh_x, band_y - params.hh_y);
        }
        float adjacent = low_first
            ? simd_shuffle_up(high, 1u) : simd_shuffle_down(high, 1u);
        float left = xl == 0u ? high : (low_first ? adjacent : high);
        float right = xl + 1u == width ? left : (low_first ? high : adjacent);
        const uint lo = tx0 > 1u ? tx0 - 1u : 0u;
        const uint hi = min(tx1 + 1u, width);
        if (xl >= lo && xl < hi) {
            low = reversible53_predict(low, left, right);
        }
        adjacent = low_first
            ? simd_shuffle_down(low, 1u) : simd_shuffle_up(low, 1u);
        left = xh == 0u ? low : (low_first ? low : adjacent);
        right = xh + 1u == width ? left : (low_first ? adjacent : low);
        if (xh >= tx0 && xh < tx1) {
            high = reversible53_update(high, left, right);
        }
        threadgroup float *row = tile + (y - ly0) * J2K_IDWT53_2D_STRIDE;
        if (xl >= tx0 && xl < tx1) {
            row[xl - lx0] = low;
        }
        if (xh >= tx0 && xh < tx1) {
            row[xh - lx0] = high;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint first_even_y = origin_y & 1u;
    {
        const uint lo = ty0 > 1u ? ty0 - 1u : 0u;
        const uint hi = min(ty1 + 1u, height);
        const uint first = lo + ((lo & 1u) != first_even_y ? 1u : 0u);
        for (uint y = first + 2u * local.y; y < hi; y += 2u * J2K_IDWT53_2D_ROWS) {
            threadgroup float *row = tile + (y - ly0) * J2K_IDWT53_2D_STRIDE;
            threadgroup float *above = tile +
                (periodic_symmetric_extension_left_u32(y, 1u) - ly0) * J2K_IDWT53_2D_STRIDE;
            threadgroup float *below = tile +
                (periodic_symmetric_extension_right_u32(y, 1u, height) - ly0) * J2K_IDWT53_2D_STRIDE;
            for (uint x = tx0 + local.x; x < tx1; x += J2K_IDWT53_2D_LANES) {
                const uint c = x - lx0;
                row[c] = reversible53_predict(row[c], above[c], below[c]);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        const uint first_odd_y = 1u - first_even_y;
        const uint first = ty0 + ((ty0 & 1u) != first_odd_y ? 1u : 0u);
        for (uint y = first + 2u * local.y; y < ty1; y += 2u * J2K_IDWT53_2D_ROWS) {
            threadgroup float *row = tile + (y - ly0) * J2K_IDWT53_2D_STRIDE;
            threadgroup float *above = tile +
                (periodic_symmetric_extension_left_u32(y, 1u) - ly0) * J2K_IDWT53_2D_STRIDE;
            threadgroup float *below = tile +
                (periodic_symmetric_extension_right_u32(y, 1u, height) - ly0) * J2K_IDWT53_2D_STRIDE;
            for (uint x = tx0 + local.x; x < tx1; x += J2K_IDWT53_2D_LANES) {
                const uint c = x - lx0;
                row[c] = reversible53_update(row[c], above[c], below[c]);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint y = ty0 + local.y; y < ty1; y += J2K_IDWT53_2D_ROWS) {
        threadgroup const float *row = tile + (y - ly0) * J2K_IDWT53_2D_STRIDE;
        for (uint x = tx0 + local.x; x < tx1; x += J2K_IDWT53_2D_LANES) {
            writer.write(x, y, row[x - lx0]);
        }
    }
}

kernel void j2k_idwt_reversible53_fused_2d(
    device const float *ll [[buffer(0)]],
    device const float *hl [[buffer(1)]],
    device const float *lh [[buffer(2)]],
    device const float *hh [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant J2kIdwtSingleDecompositionParams &params [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]]
) {
    threadgroup float tile[(J2K_IDWT53_2D_TILE_H + 2u * J2K_IDWT53_HALO) * J2K_IDWT53_2D_STRIDE];
    idwt53_fused_2d_core(
        ll, hl, lh, hh, J2kIdwtPlaneWriter{out, params.width}, params, group, local, tile
    );
}

kernel void j2k_idwt_reversible53_fused_2d_batched(
    device const float *ll [[buffer(0)]],
    device const float *hl [[buffer(1)]],
    device const float *lh [[buffer(2)]],
    device const float *hh [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant J2kRepeatedIdwtSingleDecompositionParams &params [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 local [[thread_position_in_threadgroup]]
) {
    threadgroup float tile[(J2K_IDWT53_2D_TILE_H + 2u * J2K_IDWT53_HALO) * J2K_IDWT53_2D_STRIDE];
    if (group.z >= params.batch_count) {
        return;
    }
    idwt53_fused_2d_core(
        ll + group.z * params.ll_instance_stride,
        hl + group.z * params.hl_instance_stride,
        lh + group.z * params.lh_instance_stride,
        hh + group.z * params.hh_instance_stride,
        J2kIdwtPlaneWriter{out + ulong(group.z) * params.width * params.height, params.width},
        params,
        group.xy,
        local.xy,
        tile
    );
}

constant uint J2K_IDWT53_COLS_PER_GROUP = 16u;
constant uint J2K_IDWT53_COL_TILE = 128u;
constant uint J2K_IDWT53_COL_ROW_THREADS = 16u;

template<typename Params>
inline void idwt53_vertical_fused_core(
    device float *out,
    constant Params &params,
    uint instance,
    uint3 group,
    uint3 local,
    threadgroup float *columns,
    threadgroup float *carry_rows
) {
    threadgroup float *carry = carry_rows + local.x * J2K_IDWT53_HALO;
    const uint width = params.width;
    const uint height = params.height;
    const uint x = group.x * J2K_IDWT53_COLS_PER_GROUP + local.x;
    const bool column_active = x < width;
    const uint first_even_y = (params.y0 + params.output_y) & 1u;
    device float *plane = out + ulong(instance) * width * height;

    for (uint tile_start = 0u; tile_start < height; tile_start += J2K_IDWT53_COL_TILE) {
        const uint tile_end = min(tile_start + J2K_IDWT53_COL_TILE, height);
        const uint load_start = tile_start > J2K_IDWT53_HALO
            ? tile_start - J2K_IDWT53_HALO
            : 0u;
        const uint load_end = min(tile_end + J2K_IDWT53_HALO, height);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (column_active) {
            for (uint y = tile_start + local.y; y < load_end; y += J2K_IDWT53_COL_ROW_THREADS) {
                columns[(y - load_start) * J2K_IDWT53_COLS_PER_GROUP + local.x] =
                    plane[ulong(y) * width + x];
            }
            for (uint y = load_start + local.y; y < tile_start; y += J2K_IDWT53_COL_ROW_THREADS) {
                columns[(y - load_start) * J2K_IDWT53_COLS_PER_GROUP + local.x] =
                    carry[y - (tile_start - J2K_IDWT53_HALO)];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (column_active && tile_end < height && local.y < J2K_IDWT53_HALO) {
            const uint row = tile_end - J2K_IDWT53_HALO + local.y - load_start;
            carry[local.y] = columns[row * J2K_IDWT53_COLS_PER_GROUP + local.x];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (column_active && height == 1u) {
            if (local.y == 0u && first_even_y != 0u) {
                columns[local.x] *= 0.5f;
            }
        } else if (column_active) {
            const uint lo = tile_start > 1u ? tile_start - 1u : 0u;
            const uint hi = min(tile_end + 1u, height);
            const uint first = lo + ((lo & 1u) != first_even_y ? 1u : 0u);
            for (uint y = first + 2u * local.y; y < hi; y += 2u * J2K_IDWT53_COL_ROW_THREADS) {
                const uint i = (y - load_start) * J2K_IDWT53_COLS_PER_GROUP + local.x;
                const uint left = (periodic_symmetric_extension_left_u32(y, 1u) - load_start) *
                    J2K_IDWT53_COLS_PER_GROUP + local.x;
                const uint right = (periodic_symmetric_extension_right_u32(y, 1u, height) - load_start) *
                    J2K_IDWT53_COLS_PER_GROUP + local.x;
                columns[i] = reversible53_predict(
                    columns[i],
                    columns[left],
                    columns[right]
                );
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (column_active && height > 1u) {
            const uint first_odd_y = 1u - first_even_y;
            const uint first = tile_start + ((tile_start & 1u) != first_odd_y ? 1u : 0u);
            for (uint y = first + 2u * local.y; y < tile_end; y += 2u * J2K_IDWT53_COL_ROW_THREADS) {
                const uint i = (y - load_start) * J2K_IDWT53_COLS_PER_GROUP + local.x;
                const uint left = (periodic_symmetric_extension_left_u32(y, 1u) - load_start) *
                    J2K_IDWT53_COLS_PER_GROUP + local.x;
                const uint right = (periodic_symmetric_extension_right_u32(y, 1u, height) - load_start) *
                    J2K_IDWT53_COLS_PER_GROUP + local.x;
                columns[i] = reversible53_update(
                    columns[i],
                    columns[left],
                    columns[right]
                );
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (column_active) {
            for (uint y = tile_start + local.y; y < tile_end; y += J2K_IDWT53_COL_ROW_THREADS) {
                plane[ulong(y) * width + x] =
                    columns[(y - load_start) * J2K_IDWT53_COLS_PER_GROUP + local.x];
            }
        }
    }
}

kernel void j2k_idwt_reversible53_vertical_fused(
    device float *out [[buffer(0)]],
    constant J2kIdwtSingleDecompositionParams &params [[buffer(1)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 local [[thread_position_in_threadgroup]]
) {
    threadgroup float columns[
        J2K_IDWT53_COLS_PER_GROUP * (J2K_IDWT53_COL_TILE + 2u * J2K_IDWT53_HALO)
    ];
    threadgroup float carry[J2K_IDWT53_COLS_PER_GROUP * J2K_IDWT53_HALO];
    idwt53_vertical_fused_core(out, params, 0u, group, local, columns, carry);
}

kernel void j2k_idwt_reversible53_vertical_fused_batched(
    device float *out [[buffer(0)]],
    constant J2kRepeatedIdwtSingleDecompositionParams &params [[buffer(1)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 local [[thread_position_in_threadgroup]]
) {
    threadgroup float columns[
        J2K_IDWT53_COLS_PER_GROUP * (J2K_IDWT53_COL_TILE + 2u * J2K_IDWT53_HALO)
    ];
    threadgroup float carry[J2K_IDWT53_COLS_PER_GROUP * J2K_IDWT53_HALO];
    idwt53_vertical_fused_core(out, params, group.z, group, local, columns, carry);
}

template<typename Params>
inline float idwt97_interleaved_horizontal_sample(
    device const float *ll,
    device const float *hl,
    device const float *lh,
    device const float *hh,
    constant Params &params,
    uint ll_instance_offset,
    uint hl_instance_offset,
    uint lh_instance_offset,
    uint hh_instance_offset,
    uint x,
    uint y,
    float high_pass
) {
    float sample = idwt_interleaved_sample(
        ll,
        hl,
        lh,
        hh,
        params,
        ll_instance_offset,
        hl_instance_offset,
        lh_instance_offset,
        hh_instance_offset,
        x,
        y
    );
    if (params.width == 1u) {
        if (((params.x0 + params.output_x) & 1u) != 0u) {
            sample *= 0.5f;
        }
    } else {
        const uint first_even_x = (params.x0 + params.output_x) & 1u;
        const float KAPPA = CODEC_MATH_DWT97_KAPPA;
        sample *= (x & 1u) == first_even_x ? KAPPA : high_pass;
    }
    return sample;
}

template<typename Params>
inline void idwt97_interleave_horizontal_fused_core(
    device const float *ll,
    device const float *hl,
    device const float *lh,
    device const float *hh,
    device float *out,
    constant Params &params,
    constant J2kIdwt97LiftSteps &steps,
    uint ll_instance_offset,
    uint hl_instance_offset,
    uint lh_instance_offset,
    uint hh_instance_offset,
    uint instance,
    uint3 group,
    uint3 local,
    threadgroup float *rows
) {
    const uint row_stride = J2K_IDWT97_ROW_TILE + 2u * J2K_IDWT97_HALO;
    threadgroup float *row = rows + local.y * row_stride;
    const uint width = params.width;
    const uint y = group.y * J2K_IDWT97_ROWS_PER_GROUP + local.y;
    const uint tile_start = group.x * J2K_IDWT97_ROW_TILE;
    const uint tile_end = min(tile_start + J2K_IDWT97_ROW_TILE, width);
    const uint load_start = tile_start > J2K_IDWT97_HALO ? tile_start - J2K_IDWT97_HALO : 0u;
    const uint load_end = min(tile_end + J2K_IDWT97_HALO, width);
    const bool row_active = y < params.height;
    const bool lift = row_active && width > 1u;
    const float high_pass = as_type<float>(steps.high_pass_bits);

    if (row_active) {
        for (uint x = load_start + local.x; x < load_end; x += J2K_IDWT97_ROW_THREADS) {
            row[x - load_start] = idwt97_interleaved_horizontal_sample(
                ll,
                hl,
                lh,
                hh,
                params,
                ll_instance_offset,
                hl_instance_offset,
                lh_instance_offset,
                hh_instance_offset,
                x,
                y,
                high_pass
            );
        }
    }

    for (uint step = 0u; step < 4u; ++step) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!lift) {
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
        const ulong plane_offset = ulong(instance) * width * params.height;
        for (uint x = tile_start + local.x; x < tile_end; x += J2K_IDWT97_ROW_THREADS) {
            out[plane_offset + ulong(y) * width + x] = row[x - load_start];
        }
    }
}

kernel void j2k_idwt_irreversible97_interleave_horizontal_fused(
    device const float *ll [[buffer(0)]],
    device const float *hl [[buffer(1)]],
    device const float *lh [[buffer(2)]],
    device const float *hh [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant J2kIdwtSingleDecompositionParams &params [[buffer(5)]],
    constant J2kIdwt97LiftSteps &steps [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 local [[thread_position_in_threadgroup]]
) {
    threadgroup float rows[
        J2K_IDWT97_ROWS_PER_GROUP * (J2K_IDWT97_ROW_TILE + 2u * J2K_IDWT97_HALO)
    ];
    idwt97_interleave_horizontal_fused_core(
        ll, hl, lh, hh, out, params, steps,
        0u, 0u, 0u, 0u, 0u, group, local, rows
    );
}

kernel void j2k_idwt_irreversible97_interleave_horizontal_fused_batched(
    device const float *ll [[buffer(0)]],
    device const float *hl [[buffer(1)]],
    device const float *lh [[buffer(2)]],
    device const float *hh [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant J2kRepeatedIdwtSingleDecompositionParams &params [[buffer(5)]],
    constant J2kIdwt97LiftSteps &steps [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 local [[thread_position_in_threadgroup]]
) {
    threadgroup float rows[
        J2K_IDWT97_ROWS_PER_GROUP * (J2K_IDWT97_ROW_TILE + 2u * J2K_IDWT97_HALO)
    ];
    idwt97_interleave_horizontal_fused_core(
        ll, hl, lh, hh, out, params, steps,
        group.z * params.ll_instance_stride,
        group.z * params.hl_instance_stride,
        group.z * params.lh_instance_stride,
        group.z * params.hh_instance_stride,
        group.z,
        group,
        local,
        rows
    );
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

constant uint J2K_IDWT97_2D_TILE_W = 56u;
constant uint J2K_IDWT97_2D_TILE_H = 56u;
constant uint J2K_IDWT97_2D_STRIDE = J2K_IDWT97_2D_TILE_W + 2u * J2K_IDWT97_HALO + 1u;

// One 9/7 synthesis level in a single pass, as `j2k_idwt_reversible53_fused_2d`
// does for 5/3: load the tile and a four-sample halo with the horizontal scale,
// run the four horizontal lifting steps, scale the tile's columns vertically,
// run the four vertical steps, and write the tile. Each sample sees the same
// operations in the same order as the fused horizontal and vertical kernels.
// Needs width and height of at least 2.
template<typename Params, typename Writer>
inline void idwt97_fused_2d_core(
    device const float *ll,
    device const float *hl,
    device const float *lh,
    device const float *hh,
    Writer writer,
    constant Params &params,
    constant J2kIdwt97LiftSteps &horizontal,
    constant J2kIdwt97LiftSteps &vertical,
    uint2 group,
    uint2 local,
    threadgroup float *tile
) {
    const uint width = params.width;
    const uint height = params.height;
    const uint tx0 = group.x * J2K_IDWT97_2D_TILE_W;
    const uint ty0 = group.y * J2K_IDWT97_2D_TILE_H;
    if (tx0 >= width || ty0 >= height) {
        return;
    }
    const uint tx1 = min(tx0 + J2K_IDWT97_2D_TILE_W, width);
    const uint ty1 = min(ty0 + J2K_IDWT97_2D_TILE_H, height);
    const uint lx0 = tx0 > J2K_IDWT97_HALO ? tx0 - J2K_IDWT97_HALO : 0u;
    const uint ly0 = ty0 > J2K_IDWT97_HALO ? ty0 - J2K_IDWT97_HALO : 0u;
    const uint lx1 = min(tx1 + J2K_IDWT97_HALO, width);
    const uint ly1 = min(ty1 + J2K_IDWT97_HALO, height);
    const uint origin_x = params.x0 + params.output_x;
    const uint origin_y = params.y0 + params.output_y;
    const float KAPPA = CODEC_MATH_DWT97_KAPPA;
    const float high_pass = as_type<float>(horizontal.high_pass_bits);

    const uint low_x = lx0 + ((origin_x + lx0) & 1u);
    const uint high_x = lx0 + ((origin_x + lx0 + 1u) & 1u);
    const uint low_band_x = low_index(origin_x + low_x, params.x0);
    const uint high_band_x = high_index(origin_x + high_x, params.x0);
    // Each SIMD group owns a row and keeps a low/high pair per lane in
    // registers. The tile transfers the horizontal result to the columns.
    const uint xl = low_x + 2u * local.x;
    const uint xh = high_x + 2u * local.x;
    const bool low_first = low_x < high_x;
    for (uint y = ly0 + local.y; y < ly1; y += J2K_IDWT53_2D_ROWS) {
        const uint global_y = origin_y + y;
        const bool low_y = (global_y & 1u) == 0u;
        const uint band_y = low_y ? low_index(global_y, params.y0) : high_index(global_y, params.y0);
        float low = 0.0f;
        float high = 0.0f;
        if (xl < lx1) {
            low = low_y
                ? idwt53_band_sample(ll, params.ll_width, params.ll_height,
                    low_band_x + local.x - params.ll_x, band_y - params.ll_y)
                : idwt53_band_sample(lh, params.lh_width, params.lh_height,
                    low_band_x + local.x - params.lh_x, band_y - params.lh_y);
            low *= KAPPA;
        }
        if (xh < lx1) {
            high = low_y
                ? idwt53_band_sample(hl, params.hl_width, params.hl_height,
                    high_band_x + local.x - params.hl_x, band_y - params.hl_y)
                : idwt53_band_sample(hh, params.hh_width, params.hh_height,
                    high_band_x + local.x - params.hh_x, band_y - params.hh_y);
            high *= high_pass;
        }
        // The four-sample halo contains every stored sample's dependency
        // cone. Extra lifts outside the shrinking valid range cannot reach
        // that output, so every lane can execute without per-step bounds.
        for (uint step = 0u; step < 4u; ++step) {
            if ((step & 1u) == 0u) {
                const float adjacent = low_first
                    ? simd_shuffle_up(high, 1u) : simd_shuffle_down(high, 1u);
                const float left = xl == 0u ? high : (low_first ? adjacent : high);
                const float right = xl + 1u == width ? left : (low_first ? high : adjacent);
                low = fma(left + right, horizontal.coefficients[step], low);
            } else {
                const float adjacent = low_first
                    ? simd_shuffle_down(low, 1u) : simd_shuffle_up(low, 1u);
                const float left = xh == 0u ? low : (low_first ? low : adjacent);
                const float right = xh + 1u == width ? left : (low_first ? adjacent : low);
                high = fma(left + right, horizontal.coefficients[step], high);
            }
        }
        const float scale = low_y ? KAPPA : high_pass;
        threadgroup float *row = tile + (y - ly0) * J2K_IDWT97_2D_STRIDE;
        if (xl >= tx0 && xl < tx1) {
            row[xl - lx0] = low * scale;
        }
        if (xh >= tx0 && xh < tx1) {
            row[xh - lx0] = high * scale;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Transpose the work assignment for the vertical lifts. Padding the
    // tile avoids a power-of-two stride for these column reads.
    const uint low_y0 = ly0 + ((origin_y + ly0) & 1u);
    const uint high_y0 = ly0 + ((origin_y + ly0 + 1u) & 1u);
    const uint yl = low_y0 + 2u * local.x;
    const uint yh = high_y0 + 2u * local.x;
    const bool low_y_first = low_y0 < high_y0;
    for (uint x = tx0 + local.y; x < tx1; x += J2K_IDWT53_2D_ROWS) {
        const uint col = x - lx0;
        float low = yl < ly1 ? tile[(yl - ly0) * J2K_IDWT97_2D_STRIDE + col] : 0.0f;
        float high = yh < ly1 ? tile[(yh - ly0) * J2K_IDWT97_2D_STRIDE + col] : 0.0f;
        for (uint step = 0u; step < 4u; ++step) {
            if ((step & 1u) == 0u) {
                const float adjacent = low_y_first
                    ? simd_shuffle_up(high, 1u) : simd_shuffle_down(high, 1u);
                const float left = yl == 0u ? high : (low_y_first ? adjacent : high);
                const float right = yl + 1u == height ? left : (low_y_first ? high : adjacent);
                low = fma(left + right, vertical.coefficients[step], low);
            } else {
                const float adjacent = low_y_first
                    ? simd_shuffle_down(low, 1u) : simd_shuffle_up(low, 1u);
                const float left = yh == 0u ? low : (low_y_first ? low : adjacent);
                const float right = yh + 1u == height ? left : (low_y_first ? adjacent : low);
                high = fma(left + right, vertical.coefficients[step], high);
            }
        }
        if (yl >= ty0 && yl < ty1) {
            tile[(yl - ly0) * J2K_IDWT97_2D_STRIDE + col] = low;
        }
        if (yh >= ty0 && yh < ty1) {
            tile[(yh - ly0) * J2K_IDWT97_2D_STRIDE + col] = high;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint y = ty0 + local.y; y < ty1; y += J2K_IDWT53_2D_ROWS) {
        threadgroup const float *row = tile + (y - ly0) * J2K_IDWT97_2D_STRIDE;
        const uint x = tx0 + 2u * local.x;
        if (x < tx1) {
            writer.write(x, y, row[x - lx0]);
        }
        if (x + 1u < tx1) {
            writer.write(x + 1u, y, row[x + 1u - lx0]);
        }
    }
}

kernel void j2k_idwt_irreversible97_fused_2d(
    device const float *ll [[buffer(0)]],
    device const float *hl [[buffer(1)]],
    device const float *lh [[buffer(2)]],
    device const float *hh [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant J2kIdwtSingleDecompositionParams &params [[buffer(5)]],
    constant J2kIdwt97LiftSteps &horizontal [[buffer(6)]],
    constant J2kIdwt97LiftSteps &vertical [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]]
) {
    threadgroup float tile[(J2K_IDWT97_2D_TILE_H + 2u * J2K_IDWT97_HALO) * J2K_IDWT97_2D_STRIDE];
    idwt97_fused_2d_core(
        ll, hl, lh, hh, J2kIdwtPlaneWriter{out, params.width}, params, horizontal, vertical,
        group, local, tile
    );
}

kernel void j2k_idwt_irreversible97_fused_2d_batched(
    device const float *ll [[buffer(0)]],
    device const float *hl [[buffer(1)]],
    device const float *lh [[buffer(2)]],
    device const float *hh [[buffer(3)]],
    device float *out [[buffer(4)]],
    constant J2kRepeatedIdwtSingleDecompositionParams &params [[buffer(5)]],
    constant J2kIdwt97LiftSteps &horizontal [[buffer(6)]],
    constant J2kIdwt97LiftSteps &vertical [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 local [[thread_position_in_threadgroup]]
) {
    threadgroup float tile[(J2K_IDWT97_2D_TILE_H + 2u * J2K_IDWT97_HALO) * J2K_IDWT97_2D_STRIDE];
    if (group.z >= params.batch_count) {
        return;
    }
    idwt97_fused_2d_core(
        ll + group.z * params.ll_instance_stride,
        hl + group.z * params.hl_instance_stride,
        lh + group.z * params.lh_instance_stride,
        hh + group.z * params.hh_instance_stride,
        J2kIdwtPlaneWriter{out + ulong(group.z) * params.width * params.height, params.width},
        params,
        horizontal, vertical, group.xy,
        local.xy,
        tile
    );
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
