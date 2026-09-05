// SPDX-License-Identifier: MIT OR Apache-2.0

struct J2kPackParams {
    uint width;
    uint height;
    uint out_stride;
    uint output_channels;
    uint opaque_alpha;
    float max_values[4];
    float u8_scales[4];
    float u16_scales[4];
};

struct J2kMctRgb8PackParams {
    uint width;
    uint height;
    uint out_stride;
    uint transform;
    float addends[3];
    float max_values[3];
    float u8_scales[3];
};

struct J2kBatchedMctRgb8PackParams {
    uint width;
    uint height;
    uint out_stride;
    uint transform;
    uint batch_count;
    uint plane_stride;
    uint output_stride;
    float addends[3];
    float max_values[3];
    float u8_scales[3];
};

inline uchar scale_to_u8(float sample, float max_value, float scale) {
    const float clamped = clamp(sample, 0.0f, max_value);
    return uchar(min(floor(clamped * scale + 0.5f), 255.0f));
}

inline ushort pack_to_u16(float sample, float max_value, float scale) {
    const float clamped = clamp(sample, 0.0f, max_value);
    return ushort(min(floor(clamped * scale + 0.5f), 65535.0f));
}

kernel void j2k_pack_gray8(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device const float *plane3 [[buffer(3)]],
    device uchar *out [[buffer(4)]],
    constant J2kPackParams &params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint idx = gid.y * params.width + gid.x;
    const uint out_idx = gid.y * params.out_stride + gid.x;
    out[out_idx] = scale_to_u8(plane0[idx], params.max_values[0], params.u8_scales[0]);
}

kernel void j2k_pack_rgb8(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device const float *plane3 [[buffer(3)]],
    device uchar *out [[buffer(4)]],
    constant J2kPackParams &params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint idx = gid.y * params.width + gid.x;
    const uint out_idx = gid.y * params.out_stride + gid.x * 3u;
    out[out_idx] = scale_to_u8(plane0[idx], params.max_values[0], params.u8_scales[0]);
    out[out_idx + 1] = scale_to_u8(plane1[idx], params.max_values[1], params.u8_scales[1]);
    out[out_idx + 2] = scale_to_u8(plane2[idx], params.max_values[2], params.u8_scales[2]);
}

kernel void j2k_pack_mct_rgb8(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant J2kMctRgb8PackParams &params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint idx = gid.y * params.width + gid.x;
    const float y0 = plane0[idx];
    const float y1 = plane1[idx];
    const float y2 = plane2[idx];
    float rgb0;
    float rgb1;
    float rgb2;

    if (params.transform == 0u) {
        const float i1 = y0 - floor((y2 + y1) * 0.25f);
        rgb0 = y2 + i1 + params.addends[0];
        rgb1 = i1 + params.addends[1];
        rgb2 = y1 + i1 + params.addends[2];
    } else {
        rgb0 = y2 * 1.402f + y0 + params.addends[0];
        rgb1 = y2 * -0.71414f + y1 * -0.34413f + y0 + params.addends[1];
        rgb2 = y1 * 1.772f + y0 + params.addends[2];
    }

    const uint out_idx = gid.y * params.out_stride + gid.x * 3u;
    out[out_idx] = scale_to_u8(rgb0, params.max_values[0], params.u8_scales[0]);
    out[out_idx + 1] = scale_to_u8(rgb1, params.max_values[1], params.u8_scales[1]);
    out[out_idx + 2] = scale_to_u8(rgb2, params.max_values[2], params.u8_scales[2]);
}

kernel void j2k_pack_mct_rgb8_batched(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device uchar *out [[buffer(3)]],
    constant J2kBatchedMctRgb8PackParams &params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.batch_count) {
        return;
    }

    const uint plane_base = gid.z * params.plane_stride;
    const uint idx = plane_base + gid.y * params.width + gid.x;
    const float y0 = plane0[idx];
    const float y1 = plane1[idx];
    const float y2 = plane2[idx];
    float rgb0;
    float rgb1;
    float rgb2;

    if (params.transform == 0u) {
        const float i1 = y0 - floor((y2 + y1) * 0.25f);
        rgb0 = y2 + i1 + params.addends[0];
        rgb1 = i1 + params.addends[1];
        rgb2 = y1 + i1 + params.addends[2];
    } else {
        rgb0 = y2 * 1.402f + y0 + params.addends[0];
        rgb1 = y2 * -0.71414f + y1 * -0.34413f + y0 + params.addends[1];
        rgb2 = y1 * 1.772f + y0 + params.addends[2];
    }

    const uint out_idx = gid.z * params.output_stride + gid.y * params.out_stride + gid.x * 3u;
    out[out_idx] = scale_to_u8(rgb0, params.max_values[0], params.u8_scales[0]);
    out[out_idx + 1] = scale_to_u8(rgb1, params.max_values[1], params.u8_scales[1]);
    out[out_idx + 2] = scale_to_u8(rgb2, params.max_values[2], params.u8_scales[2]);
}

kernel void j2k_pack_rgb_opaque_rgba8(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device const float *plane3 [[buffer(3)]],
    device uchar *out [[buffer(4)]],
    constant J2kPackParams &params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint idx = gid.y * params.width + gid.x;
    const uint out_idx = gid.y * params.out_stride + gid.x * 4u;
    out[out_idx] = scale_to_u8(plane0[idx], params.max_values[0], params.u8_scales[0]);
    out[out_idx + 1] = scale_to_u8(plane1[idx], params.max_values[1], params.u8_scales[1]);
    out[out_idx + 2] = scale_to_u8(plane2[idx], params.max_values[2], params.u8_scales[2]);
    out[out_idx + 3] = uchar(255);
}

kernel void j2k_pack_rgba8(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device const float *plane3 [[buffer(3)]],
    device uchar *out [[buffer(4)]],
    constant J2kPackParams &params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint idx = gid.y * params.width + gid.x;
    const uint out_idx = gid.y * params.out_stride + gid.x * 4u;
    out[out_idx] = scale_to_u8(plane0[idx], params.max_values[0], params.u8_scales[0]);
    out[out_idx + 1] = scale_to_u8(plane1[idx], params.max_values[1], params.u8_scales[1]);
    out[out_idx + 2] = scale_to_u8(plane2[idx], params.max_values[2], params.u8_scales[2]);
    out[out_idx + 3] = scale_to_u8(plane3[idx], params.max_values[3], params.u8_scales[3]);
}

kernel void j2k_pack_gray16(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device const float *plane3 [[buffer(3)]],
    device ushort *out [[buffer(4)]],
    constant J2kPackParams &params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint idx = gid.y * params.width + gid.x;
    const uint out_idx = (gid.y * params.out_stride) / 2u + gid.x;
    out[out_idx] = pack_to_u16(plane0[idx], params.max_values[0], params.u16_scales[0]);
}

kernel void j2k_pack_rgb16(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device const float *plane3 [[buffer(3)]],
    device ushort *out [[buffer(4)]],
    constant J2kPackParams &params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) {
        return;
    }

    const uint idx = gid.y * params.width + gid.x;
    const uint out_idx = (gid.y * params.out_stride) / 2u + gid.x * 3u;
    out[out_idx] = pack_to_u16(plane0[idx], params.max_values[0], params.u16_scales[0]);
    out[out_idx + 1] = pack_to_u16(plane1[idx], params.max_values[1], params.u16_scales[1]);
    out[out_idx + 2] = pack_to_u16(plane2[idx], params.max_values[2], params.u16_scales[2]);
}

struct J2kRepeatedGrayPackParams {
    uint width;
    uint height;
    uint out_stride;
    uint batch_count;
    float max_value;
    float u8_scale;
    float u16_scale;
};

kernel void j2k_pack_u8_repeated_gray(
    device const float *plane0 [[buffer(0)]],
    device uchar *out [[buffer(1)]],
    constant J2kRepeatedGrayPackParams &params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.batch_count) {
        return;
    }

    const uint plane_base = gid.z * params.width * params.height;
    const uint out_base = gid.z * params.out_stride * params.height;
    const uint plane_idx = plane_base + gid.y * params.width + gid.x;
    const uint out_idx = out_base + gid.y * params.out_stride + gid.x;
    out[out_idx] = scale_to_u8(plane0[plane_idx], params.max_value, params.u8_scale);
}

kernel void j2k_pack_u16_repeated_gray(
    device const float *plane0 [[buffer(0)]],
    device ushort *out [[buffer(1)]],
    constant J2kRepeatedGrayPackParams &params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.batch_count) {
        return;
    }

    const uint plane_base = gid.z * params.width * params.height;
    const uint out_base = (gid.z * params.out_stride * params.height) / 2u;
    const uint plane_idx = plane_base + gid.y * params.width + gid.x;
    const uint out_idx = out_base + gid.y * (params.out_stride / 2u) + gid.x;
    out[out_idx] = pack_to_u16(plane0[plane_idx], params.max_value, params.u16_scale);
}
