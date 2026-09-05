// SPDX-License-Identifier: MIT OR Apache-2.0

kernel void j2k_forward_rct(
    device float *plane0 [[buffer(0)]],
    device float *plane1 [[buffer(1)]],
    device float *plane2 [[buffer(2)]],
    constant J2kForwardRctParams &params [[buffer(3)]],
    device J2kMctStatus *status [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.len) {
        return;
    }

    const float r = plane0[gid];
    const float g = plane1[gid];
    const float b = plane2[gid];

    plane0[gid] = floor((r + 2.0f * g + b) * 0.25f);
    plane1[gid] = b - g;
    plane2[gid] = r - g;

    if (gid == 0) {
        status->code = J2K_MCT_STATUS_OK;
        status->detail = 0;
    }
}

kernel void j2k_forward_ict(
    device float *plane0 [[buffer(0)]],
    device float *plane1 [[buffer(1)]],
    device float *plane2 [[buffer(2)]],
    constant J2kForwardIctParams &params [[buffer(3)]],
    device J2kMctStatus *status [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
    if (gid >= params.len) {
        return;
    }

    const float r = plane0[gid];
    const float g = plane1[gid];
    const float b = plane2[gid];

    // Match the CPU transform's target-independent nested fused rounding.
    plane0[gid] = fma(0.114f, b, fma(0.299f, r, 0.587f * g));
    plane1[gid] = fma(0.5f, b, fma(-0.16875f, r, -0.33126f * g));
    plane2[gid] = fma(-0.08131f, b, fma(0.5f, r, -0.41869f * g));

    if (gid == 0) {
        status->code = J2K_MCT_STATUS_OK;
        status->detail = 0;
    }
}

kernel void j2k_encode_deinterleave_mct(
    device const uchar *src [[buffer(0)]],
    device float *plane0 [[buffer(1)]],
    device float *plane1 [[buffer(2)]],
    device float *plane2 [[buffer(3)]],
    constant J2kFusedInputMctParams &params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
    if (gid >= params.len) {
        return;
    }

    const uint src_base = gid * 3u * params.bytes_per_sample;
    const float r = j2k_lossless_load_sample(
        src, src_base, 0u, 3u, params.bytes_per_sample, params.bit_depth,
        params.sample_offset, params.signed_samples, true
    );
    const float g = j2k_lossless_load_sample(
        src, src_base, 1u, 3u, params.bytes_per_sample, params.bit_depth,
        params.sample_offset, params.signed_samples, true
    );
    const float b = j2k_lossless_load_sample(
        src, src_base, 2u, 3u, params.bytes_per_sample, params.bit_depth,
        params.sample_offset, params.signed_samples, true
    );

    if (params.reversible != 0u) {
        plane0[gid] = floor((r + 2.0f * g + b) * 0.25f);
        plane1[gid] = b - g;
        plane2[gid] = r - g;
        return;
    }

    // Match the CPU ICT's target-independent nested fused rounding.
    plane0[gid] = fma(0.114f, b, fma(0.299f, r, 0.587f * g));
    plane1[gid] = fma(0.5f, b, fma(-0.16875f, r, -0.33126f * g));
    plane2[gid] = fma(-0.08131f, b, fma(0.5f, r, -0.41869f * g));
}

kernel void j2k_lossless_deinterleave_rct_rgb8_to_planes(
    device const uchar *src [[buffer(0)]],
    device float *plane0 [[buffer(1)]],
    device float *plane1 [[buffer(2)]],
    device float *plane2 [[buffer(3)]],
    constant J2kLosslessDeinterleaveParams &params [[buffer(4)]],
    device J2kMctStatus *status [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.dst_width || gid.y >= params.dst_height) {
        return;
    }

    const bool inside_src = gid.x < params.src_width && gid.y < params.src_height;
    const uint src_base = gid.y * params.src_stride + gid.x * 3u;
    const uint dst_idx = gid.y * params.dst_width + gid.x;
    const float r = j2k_lossless_load_sample(
        src,
        src_base,
        0u,
        3u,
        1u,
        params.bit_depth,
        params.sample_offset,
        0u,
        inside_src
    );
    const float g = j2k_lossless_load_sample(
        src,
        src_base,
        1u,
        3u,
        1u,
        params.bit_depth,
        params.sample_offset,
        0u,
        inside_src
    );
    const float b = j2k_lossless_load_sample(
        src,
        src_base,
        2u,
        3u,
        1u,
        params.bit_depth,
        params.sample_offset,
        0u,
        inside_src
    );

    plane0[dst_idx] = floor((r + 2.0f * g + b) * 0.25f);
    plane1[dst_idx] = b - g;
    plane2[dst_idx] = r - g;

    if (gid.x == 0u && gid.y == 0u) {
        status->code = J2K_MCT_STATUS_OK;
        status->detail = 0;
    }
}
