// SPDX-License-Identifier: MIT OR Apache-2.0

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
