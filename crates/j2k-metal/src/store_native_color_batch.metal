// SPDX-License-Identifier: MIT OR Apache-2.0

struct J2kNativeColorBatchStoreParams {
    uint width;
    uint height;
    uint plane_stride;
    uint output_row_stride;
    uint output_item_stride;
    uint batch_count;
    uint layout;
    uint mct;
    uint transform;
    uint is_signed;
    uint bit_depths[4];
};

kernel void j2k_store_native_rgb_batch_u8(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device uchar *output [[buffer(3)]],
    constant J2kNativeColorBatchStoreParams &params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.batch_count) {
        return;
    }
    const uint plane_idx = gid.z * params.plane_stride + gid.y * params.width + gid.x;
    const float3 rgb = j2k_native_color_samples(
        plane0[plane_idx],
        plane1[plane_idx],
        plane2[plane_idx],
        params.mct,
        params.transform,
        params.is_signed,
        params.bit_depths[0],
        params.bit_depths[1],
        params.bit_depths[2]
    );
    output[j2k_native_color_output_index(params.width, params.height, params.output_row_stride, gid.z * params.output_item_stride, params.layout, uint2(gid.x, gid.y), 3u, 0u)] =
        uchar(j2k_unsigned_native_sample(rgb[0], params.bit_depths[0]));
    output[j2k_native_color_output_index(params.width, params.height, params.output_row_stride, gid.z * params.output_item_stride, params.layout, uint2(gid.x, gid.y), 3u, 1u)] =
        uchar(j2k_unsigned_native_sample(rgb[1], params.bit_depths[1]));
    output[j2k_native_color_output_index(params.width, params.height, params.output_row_stride, gid.z * params.output_item_stride, params.layout, uint2(gid.x, gid.y), 3u, 2u)] =
        uchar(j2k_unsigned_native_sample(rgb[2], params.bit_depths[2]));
}

kernel void j2k_store_native_rgb_batch_u16(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device ushort *output [[buffer(3)]],
    constant J2kNativeColorBatchStoreParams &params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.batch_count) {
        return;
    }
    const uint plane_idx = gid.z * params.plane_stride + gid.y * params.width + gid.x;
    const float3 rgb = j2k_native_color_samples(
        plane0[plane_idx],
        plane1[plane_idx],
        plane2[plane_idx],
        params.mct,
        params.transform,
        params.is_signed,
        params.bit_depths[0],
        params.bit_depths[1],
        params.bit_depths[2]
    );
    output[j2k_native_color_output_index(params.width, params.height, params.output_row_stride, gid.z * params.output_item_stride, params.layout, uint2(gid.x, gid.y), 3u, 0u)] =
        ushort(j2k_unsigned_native_sample(rgb[0], params.bit_depths[0]));
    output[j2k_native_color_output_index(params.width, params.height, params.output_row_stride, gid.z * params.output_item_stride, params.layout, uint2(gid.x, gid.y), 3u, 1u)] =
        ushort(j2k_unsigned_native_sample(rgb[1], params.bit_depths[1]));
    output[j2k_native_color_output_index(params.width, params.height, params.output_row_stride, gid.z * params.output_item_stride, params.layout, uint2(gid.x, gid.y), 3u, 2u)] =
        ushort(j2k_unsigned_native_sample(rgb[2], params.bit_depths[2]));
}

kernel void j2k_store_native_rgb_batch_i16(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device short *output [[buffer(3)]],
    constant J2kNativeColorBatchStoreParams &params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.batch_count) {
        return;
    }
    const uint plane_idx = gid.z * params.plane_stride + gid.y * params.width + gid.x;
    const float3 rgb = j2k_native_color_samples(
        plane0[plane_idx],
        plane1[plane_idx],
        plane2[plane_idx],
        params.mct,
        params.transform,
        params.is_signed,
        params.bit_depths[0],
        params.bit_depths[1],
        params.bit_depths[2]
    );
    output[j2k_native_color_output_index(params.width, params.height, params.output_row_stride, gid.z * params.output_item_stride, params.layout, uint2(gid.x, gid.y), 3u, 0u)] =
        j2k_pack_native_i16(rgb[0], float((1u << (params.bit_depths[0] - 1u)) - 1u));
    output[j2k_native_color_output_index(params.width, params.height, params.output_row_stride, gid.z * params.output_item_stride, params.layout, uint2(gid.x, gid.y), 3u, 1u)] =
        j2k_pack_native_i16(rgb[1], float((1u << (params.bit_depths[1] - 1u)) - 1u));
    output[j2k_native_color_output_index(params.width, params.height, params.output_row_stride, gid.z * params.output_item_stride, params.layout, uint2(gid.x, gid.y), 3u, 2u)] =
        j2k_pack_native_i16(rgb[2], float((1u << (params.bit_depths[2] - 1u)) - 1u));
}

kernel void j2k_store_native_rgba_batch_u8(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device const float *plane3 [[buffer(3)]],
    device uchar *output [[buffer(4)]],
    constant J2kNativeColorBatchStoreParams &params [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.batch_count) {
        return;
    }
    const uint plane_idx = gid.z * params.plane_stride + gid.y * params.width + gid.x;
    const float3 rgb = j2k_native_color_samples(
        plane0[plane_idx],
        plane1[plane_idx],
        plane2[plane_idx],
        params.mct,
        params.transform,
        params.is_signed,
        params.bit_depths[0],
        params.bit_depths[1],
        params.bit_depths[2]
    );
    output[j2k_native_color_output_index(params.width, params.height, params.output_row_stride, gid.z * params.output_item_stride, params.layout, uint2(gid.x, gid.y), 4u, 0u)] =
        uchar(j2k_unsigned_native_sample(rgb[0], params.bit_depths[0]));
    output[j2k_native_color_output_index(params.width, params.height, params.output_row_stride, gid.z * params.output_item_stride, params.layout, uint2(gid.x, gid.y), 4u, 1u)] =
        uchar(j2k_unsigned_native_sample(rgb[1], params.bit_depths[1]));
    output[j2k_native_color_output_index(params.width, params.height, params.output_row_stride, gid.z * params.output_item_stride, params.layout, uint2(gid.x, gid.y), 4u, 2u)] =
        uchar(j2k_unsigned_native_sample(rgb[2], params.bit_depths[2]));
    output[j2k_native_color_output_index(params.width, params.height, params.output_row_stride, gid.z * params.output_item_stride, params.layout, uint2(gid.x, gid.y), 4u, 3u)] =
        uchar(j2k_unsigned_native_sample(plane3[plane_idx], params.bit_depths[3]));
}

kernel void j2k_store_native_rgba_batch_u16(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device const float *plane3 [[buffer(3)]],
    device ushort *output [[buffer(4)]],
    constant J2kNativeColorBatchStoreParams &params [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.batch_count) {
        return;
    }
    const uint plane_idx = gid.z * params.plane_stride + gid.y * params.width + gid.x;
    const float3 rgb = j2k_native_color_samples(
        plane0[plane_idx],
        plane1[plane_idx],
        plane2[plane_idx],
        params.mct,
        params.transform,
        params.is_signed,
        params.bit_depths[0],
        params.bit_depths[1],
        params.bit_depths[2]
    );
    output[j2k_native_color_output_index(params.width, params.height, params.output_row_stride, gid.z * params.output_item_stride, params.layout, uint2(gid.x, gid.y), 4u, 0u)] =
        ushort(j2k_unsigned_native_sample(rgb[0], params.bit_depths[0]));
    output[j2k_native_color_output_index(params.width, params.height, params.output_row_stride, gid.z * params.output_item_stride, params.layout, uint2(gid.x, gid.y), 4u, 1u)] =
        ushort(j2k_unsigned_native_sample(rgb[1], params.bit_depths[1]));
    output[j2k_native_color_output_index(params.width, params.height, params.output_row_stride, gid.z * params.output_item_stride, params.layout, uint2(gid.x, gid.y), 4u, 2u)] =
        ushort(j2k_unsigned_native_sample(rgb[2], params.bit_depths[2]));
    output[j2k_native_color_output_index(params.width, params.height, params.output_row_stride, gid.z * params.output_item_stride, params.layout, uint2(gid.x, gid.y), 4u, 3u)] =
        ushort(j2k_unsigned_native_sample(plane3[plane_idx], params.bit_depths[3]));
}

kernel void j2k_store_native_rgba_batch_i16(
    device const float *plane0 [[buffer(0)]],
    device const float *plane1 [[buffer(1)]],
    device const float *plane2 [[buffer(2)]],
    device const float *plane3 [[buffer(3)]],
    device short *output [[buffer(4)]],
    constant J2kNativeColorBatchStoreParams &params [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height || gid.z >= params.batch_count) {
        return;
    }
    const uint plane_idx = gid.z * params.plane_stride + gid.y * params.width + gid.x;
    const float3 rgb = j2k_native_color_samples(
        plane0[plane_idx],
        plane1[plane_idx],
        plane2[plane_idx],
        params.mct,
        params.transform,
        params.is_signed,
        params.bit_depths[0],
        params.bit_depths[1],
        params.bit_depths[2]
    );
    output[j2k_native_color_output_index(params.width, params.height, params.output_row_stride, gid.z * params.output_item_stride, params.layout, uint2(gid.x, gid.y), 4u, 0u)] =
        j2k_pack_native_i16(rgb[0], float((1u << (params.bit_depths[0] - 1u)) - 1u));
    output[j2k_native_color_output_index(params.width, params.height, params.output_row_stride, gid.z * params.output_item_stride, params.layout, uint2(gid.x, gid.y), 4u, 1u)] =
        j2k_pack_native_i16(rgb[1], float((1u << (params.bit_depths[1] - 1u)) - 1u));
    output[j2k_native_color_output_index(params.width, params.height, params.output_row_stride, gid.z * params.output_item_stride, params.layout, uint2(gid.x, gid.y), 4u, 2u)] =
        j2k_pack_native_i16(rgb[2], float((1u << (params.bit_depths[2] - 1u)) - 1u));
    output[j2k_native_color_output_index(params.width, params.height, params.output_row_stride, gid.z * params.output_item_stride, params.layout, uint2(gid.x, gid.y), 4u, 3u)] =
        j2k_pack_native_i16(
            plane3[plane_idx],
            float((1u << (params.bit_depths[3] - 1u)) - 1u)
        );
}
