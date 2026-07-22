// SPDX-License-Identifier: MIT OR Apache-2.0

#include <metal_stdlib>
using namespace metal;

struct J2kStoreParams {
    uint input_width;
    uint source_x;
    uint source_y;
    uint copy_width;
    uint copy_height;
    uint output_width;
    uint output_x;
    uint output_y;
    float addend;
};

struct J2kRepeatedStoreParams {
    uint input_width;
    uint input_height;
    uint input_instance_stride;
    uint source_x;
    uint source_y;
    uint copy_width;
    uint copy_height;
    uint output_width;
    uint output_height;
    uint output_x;
    uint output_y;
    float addend;
    uint batch_count;
};

struct J2kRepeatedGrayStoreParams {
    uint input_width;
    uint input_height;
    uint source_x;
    uint source_y;
    uint copy_width;
    uint copy_height;
    uint output_width;
    uint output_height;
    uint output_x;
    uint output_y;
    float addend;
    uint batch_count;
    float max_value;
    float u8_scale;
    float u16_scale;
};

struct J2kGrayStoreParams {
    uint input_width;
    uint source_x;
    uint source_y;
    uint copy_width;
    uint copy_height;
    uint output_width;
    uint output_stride;
    uint output_item_offset;
    uint output_x;
    uint output_y;
    float addend;
    float max_value;
    float u8_scale;
    float u16_scale;
};


constant uint J2K_BATCH_LAYOUT_NCHW = 0;
constant uint J2K_BATCH_LAYOUT_NHWC = 1;
constant uint J2K_NATIVE_RGB_NO_MCT = 0;

inline float j2k_unsigned_native_sample(float value, uint bit_depth) {
    if (isnan(value)) {
        return 0.0f;
    }
    const float max_value = float((1u << bit_depth) - 1u);
    return floor(clamp(value, 0.0f, max_value) + 0.5f);
}

inline float3 j2k_native_color_samples(
    float value0,
    float value1,
    float value2,
    uint mct,
    uint transform,
    uint is_signed,
    uint bit_depth0,
    uint bit_depth1,
    uint bit_depth2
) {
    if (mct == J2K_NATIVE_RGB_NO_MCT) {
        return float3(value0, value1, value2);
    }

    const float3 addends = is_signed != 0u
        ? float3(0.0f)
        : float3(
            float(1u << (bit_depth0 - 1u)),
            float(1u << (bit_depth1 - 1u)),
            float(1u << (bit_depth2 - 1u))
        );
    if (transform == J2K_MCT_TRANSFORM_REVERSIBLE53) {
        const float green = value0 - floor((value2 + value1) * 0.25f);
        return float3(value2 + green, green, value1 + green) + addends;
    }
    return float3(
        value2 * 1.402f + value0,
        value2 * -0.71414f + value1 * -0.34413f + value0,
        value1 * 1.772f + value0
    ) + addends;
}

inline uint j2k_native_color_output_index(
    uint width,
    uint height,
    uint output_row_stride,
    uint output_item_offset,
    uint layout,
    uint2 gid,
    uint channels,
    uint channel
) {
    if (layout == J2K_BATCH_LAYOUT_NCHW) {
        const uint plane_len = width * height;
        return output_item_offset + channel * plane_len + gid.y * width + gid.x;
    }
    return output_item_offset + gid.y * output_row_stride + gid.x * channels + channel;
}

struct J2kStoreWindowIndices {
    uint src_idx;
    uint dst_idx;
};

inline short j2k_pack_native_i16(float value, float positive_max) {
    const float negative_min = -positive_max - 1.0f;
    return short(rint(clamp(value, negative_min, positive_max)));
}

inline J2kStoreWindowIndices j2k_store_window_indices(
    uint input_width,
    uint output_width,
    uint source_x,
    uint source_y,
    uint output_x,
    uint output_y,
    uint2 gid,
    uint input_offset,
    uint output_offset
) {
    const uint src_x = source_x + gid.x;
    const uint src_y = source_y + gid.y;
    const uint dst_x = output_x + gid.x;
    const uint dst_y = output_y + gid.y;

    return {
        input_offset + src_y * input_width + src_x,
        output_offset + dst_y * output_width + dst_x,
    };
}

kernel void j2k_store_component(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant J2kStoreParams &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.copy_width || gid.y >= params.copy_height) {
        return;
    }

    const J2kStoreWindowIndices indices = j2k_store_window_indices(
        params.input_width,
        params.output_width,
        params.source_x,
        params.source_y,
        params.output_x,
        params.output_y,
        gid,
        0u,
        0u
    );
    output[indices.dst_idx] = input[indices.src_idx] + params.addend;
}

kernel void j2k_store_component_repeated(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant J2kRepeatedStoreParams &params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.copy_width || gid.y >= params.copy_height || gid.z >= params.batch_count) {
        return;
    }

    const uint output_plane_len = params.output_width * params.output_height;
    const J2kStoreWindowIndices indices = j2k_store_window_indices(
        params.input_width,
        params.output_width,
        params.source_x,
        params.source_y,
        params.output_x,
        params.output_y,
        gid.xy,
        gid.z * params.input_instance_stride,
        gid.z * output_plane_len
    );
    output[indices.dst_idx] = input[indices.src_idx] + params.addend;
}

kernel void j2k_store_component_repeated_gray_u8(
    device const float *input [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    constant J2kRepeatedGrayStoreParams &params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.copy_width || gid.y >= params.copy_height || gid.z >= params.batch_count) {
        return;
    }

    const uint input_plane_len = params.input_width * params.input_height;
    const uint output_plane_len = params.output_width * params.output_height;
    const J2kStoreWindowIndices indices = j2k_store_window_indices(
        params.input_width,
        params.output_width,
        params.source_x,
        params.source_y,
        params.output_x,
        params.output_y,
        gid.xy,
        gid.z * input_plane_len,
        gid.z * output_plane_len
    );
    output[indices.dst_idx] = scale_to_u8(input[indices.src_idx] + params.addend, params.max_value, params.u8_scale);
}

kernel void j2k_store_component_repeated_gray_u16(
    device const float *input [[buffer(0)]],
    device ushort *output [[buffer(1)]],
    constant J2kRepeatedGrayStoreParams &params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.copy_width || gid.y >= params.copy_height || gid.z >= params.batch_count) {
        return;
    }

    const uint input_plane_len = params.input_width * params.input_height;
    const uint output_plane_len = params.output_width * params.output_height;
    const J2kStoreWindowIndices indices = j2k_store_window_indices(
        params.input_width,
        params.output_width,
        params.source_x,
        params.source_y,
        params.output_x,
        params.output_y,
        gid.xy,
        gid.z * input_plane_len,
        gid.z * output_plane_len
    );
    output[indices.dst_idx] = pack_to_u16(input[indices.src_idx] + params.addend, params.max_value, params.u16_scale);
}

kernel void j2k_store_component_repeated_gray_i16(
    device const float *input [[buffer(0)]],
    device short *output [[buffer(1)]],
    constant J2kRepeatedGrayStoreParams &params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.copy_width || gid.y >= params.copy_height || gid.z >= params.batch_count) {
        return;
    }

    const uint input_plane_len = params.input_width * params.input_height;
    const uint output_plane_len = params.output_width * params.output_height;
    const J2kStoreWindowIndices indices = j2k_store_window_indices(
        params.input_width,
        params.output_width,
        params.source_x,
        params.source_y,
        params.output_x,
        params.output_y,
        gid.xy,
        gid.z * input_plane_len,
        gid.z * output_plane_len
    );
    output[indices.dst_idx] = j2k_pack_native_i16(
        input[indices.src_idx] + params.addend,
        params.max_value
    );
}

kernel void j2k_store_component_repeated_gray_u8_contiguous(
    device const float *input [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    constant J2kRepeatedGrayStoreParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint plane_len = params.input_width * params.input_height;
    const uint total_len = plane_len * params.batch_count;
    if (gid >= total_len) {
        return;
    }

    output[gid] = scale_to_u8(input[gid] + params.addend, params.max_value, params.u8_scale);
}

kernel void j2k_store_component_repeated_gray_u16_contiguous(
    device const float *input [[buffer(0)]],
    device ushort *output [[buffer(1)]],
    constant J2kRepeatedGrayStoreParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint plane_len = params.input_width * params.input_height;
    const uint total_len = plane_len * params.batch_count;
    if (gid >= total_len) {
        return;
    }

    output[gid] = pack_to_u16(input[gid] + params.addend, params.max_value, params.u16_scale);
}

kernel void j2k_store_component_gray_u8(
    device const float *input [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    constant J2kGrayStoreParams &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.copy_width || gid.y >= params.copy_height) {
        return;
    }

    const J2kStoreWindowIndices indices = j2k_store_window_indices(
        params.input_width,
        params.output_stride,
        params.source_x,
        params.source_y,
        params.output_x,
        params.output_y,
        gid,
        0u,
        params.output_item_offset
    );
    output[indices.dst_idx] = scale_to_u8(input[indices.src_idx] + params.addend, params.max_value, params.u8_scale);
}

kernel void j2k_store_component_gray_u16(
    device const float *input [[buffer(0)]],
    device ushort *output [[buffer(1)]],
    constant J2kGrayStoreParams &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.copy_width || gid.y >= params.copy_height) {
        return;
    }

    const J2kStoreWindowIndices indices = j2k_store_window_indices(
        params.input_width,
        params.output_stride,
        params.source_x,
        params.source_y,
        params.output_x,
        params.output_y,
        gid,
        0u,
        params.output_item_offset
    );
    output[indices.dst_idx] = pack_to_u16(input[indices.src_idx] + params.addend, params.max_value, params.u16_scale);
}

kernel void j2k_store_component_gray_i16(
    device const float *input [[buffer(0)]],
    device short *output [[buffer(1)]],
    constant J2kGrayStoreParams &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.copy_width || gid.y >= params.copy_height) {
        return;
    }

    const J2kStoreWindowIndices indices = j2k_store_window_indices(
        params.input_width,
        params.output_stride,
        params.source_x,
        params.source_y,
        params.output_x,
        params.output_y,
        gid,
        0u,
        params.output_item_offset
    );
    output[indices.dst_idx] = j2k_pack_native_i16(
        input[indices.src_idx] + params.addend,
        params.max_value
    );
}
