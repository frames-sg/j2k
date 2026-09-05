// SPDX-License-Identifier: MIT OR Apache-2.0

struct J2kLosslessDeinterleaveParams {
    uint src_width;
    uint src_height;
    uint src_stride;
    uint dst_width;
    uint dst_height;
    uint components;
    uint bytes_per_sample;
    uint bit_depth;
    uint sample_offset;
    uint signed_samples;
};

inline uint j2k_lossless_precision_mask(uint bit_depth) {
    return (1u << bit_depth) - 1u;
}

inline float j2k_lossless_load_sample(
    device const uchar *src,
    uint base,
    uint component,
    uint components,
    uint bytes_per_sample,
    uint bit_depth,
    uint sample_offset,
    uint signed_samples,
    bool inside_src
) {
    if (!inside_src) {
        return signed_samples == 0u ? -float(int(sample_offset)) : 0.0f;
    }
    if (bytes_per_sample == 1u) {
        const uint raw = uint(src[base + component]) & j2k_lossless_precision_mask(bit_depth);
        if (signed_samples != 0u) {
            const uint sign_bit = 1u << (bit_depth - 1u);
            const int signed_raw = (raw & sign_bit) != 0u
                ? int(raw) - int(1u << bit_depth)
                : int(raw);
            return float(signed_raw);
        }
        return float(int(raw) - int(sample_offset));
    }
    const uint byte_offset = base + component * 2u;
    const uint raw = (
        uint(src[byte_offset]) | (uint(src[byte_offset + 1u]) << 8u)
    ) & j2k_lossless_precision_mask(bit_depth);
    if (signed_samples != 0u) {
        const uint sign_bit = 1u << (bit_depth - 1u);
        const int signed_raw = (raw & sign_bit) != 0u
            ? int(raw) - int(1u << bit_depth)
            : int(raw);
        return float(signed_raw);
    }
    return float(int(raw) - int(sample_offset));
}

kernel void j2k_lossless_deinterleave_to_planes(
    device const uchar *src [[buffer(0)]],
    device float *plane0 [[buffer(1)]],
    device float *plane1 [[buffer(2)]],
    device float *plane2 [[buffer(3)]],
    constant J2kLosslessDeinterleaveParams &params [[buffer(4)]],
    device float *plane3 [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.dst_width || gid.y >= params.dst_height) {
        return;
    }

    const bool inside_src = gid.x < params.src_width && gid.y < params.src_height;
    const uint src_base = gid.y * params.src_stride
        + gid.x * params.components * params.bytes_per_sample;
    const uint dst_idx = gid.y * params.dst_width + gid.x;
    plane0[dst_idx] = j2k_lossless_load_sample(
        src,
        src_base,
        0u,
        params.components,
        params.bytes_per_sample,
        params.bit_depth,
        params.sample_offset,
        params.signed_samples,
        inside_src
    );
    if (params.components >= 2u) {
        plane1[dst_idx] = j2k_lossless_load_sample(
            src,
            src_base,
            1u,
            params.components,
            params.bytes_per_sample,
            params.bit_depth,
            params.sample_offset,
            params.signed_samples,
            inside_src
        );
    }
    if (params.components >= 3u) {
        plane2[dst_idx] = j2k_lossless_load_sample(
            src,
            src_base,
            2u,
            params.components,
            params.bytes_per_sample,
            params.bit_depth,
            params.sample_offset,
            params.signed_samples,
            inside_src
        );
    }
    if (params.components >= 4u) {
        plane3[dst_idx] = j2k_lossless_load_sample(
            src,
            src_base,
            3u,
            params.components,
            params.bytes_per_sample,
            params.bit_depth,
            params.sample_offset,
            params.signed_samples,
            inside_src
        );
    }
}

struct J2kLosslessCoefficientJob {
    uint coefficient_offset;
    uint component;
    uint subband_x;
    uint subband_y;
    uint block_x;
    uint block_y;
    uint block_width;
    uint block_height;
    uint full_width;
};

kernel void j2k_lossless_extract_coefficients(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device int *coefficients [[buffer(3)]],
    constant J2kLosslessCoefficientJob *jobs [[buffer(4)]],
    constant uint &job_count [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.z >= job_count) {
        return;
    }
    constant J2kLosslessCoefficientJob &job = jobs[gid.z];
    if (gid.x >= job.block_width || gid.y >= job.block_height) {
        return;
    }

    device const float *plane = plane0;
    if (job.component == 1u) {
        plane = plane1;
    } else if (job.component == 2u) {
        plane = plane2;
    }
    const uint src_x = job.subband_x + job.block_x + gid.x;
    const uint src_y = job.subband_y + job.block_y + gid.y;
    const uint src_idx = src_y * job.full_width + src_x;
    const uint dst_idx = job.coefficient_offset + gid.y * job.block_width + gid.x;
    coefficients[dst_idx] = int(round(plane[src_idx]));
}
