// SPDX-License-Identifier: Apache-2.0

#[cfg(target_os = "macos")]
pub(super) const SHADER_SOURCE: &str = concat!(
    r#"
#include <metal_stdlib>
using namespace metal;

kernel void j2k_zero_u32_buffer(
    device uint *buffer [[buffer(0)]],
    constant uint &word_count [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= word_count) {
        return;
    }

    buffer[gid] = 0u;
}

struct J2kValidateBytesParams {
    uint byte_len;
};

struct J2kValidateBytesStatus {
    uint code;
    uint index;
    uint expected;
    uint actual;
};

kernel void j2k_validate_bytes_equal(
    device const uchar *actual [[buffer(0)]],
    device const uchar *expected [[buffer(1)]],
    device J2kValidateBytesStatus *status [[buffer(2)]],
    constant J2kValidateBytesParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) {
        return;
    }

    status[0].code = 0u;
    status[0].index = 0u;
    status[0].expected = 0u;
    status[0].actual = 0u;

    for (uint i = 0u; i < params.byte_len; ++i) {
        const uchar actual_byte = actual[i];
        const uchar expected_byte = expected[i];
        if (actual_byte != expected_byte) {
            status[0].code = 1u;
            status[0].index = i;
            status[0].expected = uint(expected_byte);
            status[0].actual = uint(actual_byte);
            return;
        }
    }
}

struct J2kCopyInterleavedParams {
    uint src_width;
    uint src_height;
    uint src_stride;
    uint dst_width;
    uint dst_height;
    uint dst_stride;
    uint bytes_per_pixel;
};

kernel void j2k_copy_interleaved_padded(
    device const uchar *src [[buffer(0)]],
    device uchar *dst [[buffer(1)]],
    constant J2kCopyInterleavedParams &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.dst_width || gid.y >= params.dst_height) {
        return;
    }

    const uint dst_idx = gid.y * params.dst_stride + gid.x * params.bytes_per_pixel;
    const bool inside_src = gid.x < params.src_width && gid.y < params.src_height;
    const uint src_idx = gid.y * params.src_stride + gid.x * params.bytes_per_pixel;
    for (uint byte_idx = 0u; byte_idx < params.bytes_per_pixel; ++byte_idx) {
        dst[dst_idx + byte_idx] = inside_src ? src[src_idx + byte_idx] : uchar(0);
    }
}

struct J2kLosslessDeinterleaveParams {
    uint src_width;
    uint src_height;
    uint src_stride;
    uint dst_width;
    uint dst_height;
    uint components;
    uint bytes_per_sample;
    uint sample_offset;
    uint signed_samples;
};

inline float j2k_lossless_load_sample(
    device const uchar *src,
    uint base,
    uint component,
    uint components,
    uint bytes_per_sample,
    uint sample_offset,
    uint signed_samples,
    bool inside_src
) {
    if (!inside_src) {
        return signed_samples == 0u ? -float(int(sample_offset)) : 0.0f;
    }
    if (bytes_per_sample == 1u) {
        const uint raw = uint(src[base + component]);
        if (signed_samples != 0u) {
            return float(raw >= 128u ? int(raw) - 256 : int(raw));
        }
        return float(int(raw) - int(sample_offset));
    }
    const uint byte_offset = base + component * 2u;
    const uint raw = uint(src[byte_offset]) | (uint(src[byte_offset + 1u]) << 8u);
    if (signed_samples != 0u) {
        return float(raw >= 32768u ? int(raw) - 65536 : int(raw));
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
"#,
    "\n",
    include_str!("../classic.metal"),
    "\n",
    include_str!("../encode_bitstream.metal"),
    "\n",
    include_str!("../idwt.metal"),
    "\n",
    include_str!("../fdwt.metal"),
    "\n",
    include_str!("../mct.metal"),
    "\n",
    include_str!("../store.metal"),
    "\n",
    include_str!("../ht_cleanup.metal"),
);
