// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

struct J2kInverseMctParams {
    uint len;
    uint transform;
    float addend0;
    float addend1;
    float addend2;
};

struct J2kMctStatus {
    uint code;
    uint detail;
    uint reserved0;
    uint reserved1;
};

struct J2kForwardRctParams {
    uint len;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

constant uint J2K_MCT_TRANSFORM_REVERSIBLE53 = 0;
constant uint J2K_MCT_TRANSFORM_IRREVERSIBLE97 = 1;
constant uint J2K_MCT_STATUS_OK = 0;
constant uint J2K_MCT_STATUS_FAIL = 1;

kernel void j2k_inverse_mct(
    device float *plane0 [[buffer(0)]],
    device float *plane1 [[buffer(1)]],
    device float *plane2 [[buffer(2)]],
    constant J2kInverseMctParams &params [[buffer(3)]],
    device J2kMctStatus *status [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.len) {
        return;
    }

    const float y0 = plane0[gid];
    const float y1 = plane1[gid];
    const float y2 = plane2[gid];

    if (params.transform == J2K_MCT_TRANSFORM_REVERSIBLE53) {
        const float i1 = y0 - floor((y2 + y1) * 0.25f);
        plane0[gid] = y2 + i1 + params.addend0;
        plane1[gid] = i1 + params.addend1;
        plane2[gid] = y1 + i1 + params.addend2;
        return;
    }

    if (params.transform == J2K_MCT_TRANSFORM_IRREVERSIBLE97) {
        plane0[gid] = y2 * 1.402f + y0 + params.addend0;
        plane1[gid] = y2 * -0.71414f + y1 * -0.34413f + y0 + params.addend1;
        plane2[gid] = y1 * 1.772f + y0 + params.addend2;
        return;
    }

    if (gid == 0) {
        status->code = J2K_MCT_STATUS_FAIL;
        status->detail = params.transform;
    }
}

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
        params.sample_offset,
        inside_src
    );
    const float g = j2k_lossless_load_sample(
        src,
        src_base,
        1u,
        3u,
        1u,
        params.sample_offset,
        inside_src
    );
    const float b = j2k_lossless_load_sample(
        src,
        src_base,
        2u,
        3u,
        1u,
        params.sample_offset,
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
