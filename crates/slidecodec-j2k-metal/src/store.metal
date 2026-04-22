// SPDX-License-Identifier: Apache-2.0

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

kernel void j2k_store_component(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant J2kStoreParams &params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= params.copy_width || gid.y >= params.copy_height) {
        return;
    }

    const uint src_x = params.source_x + gid.x;
    const uint src_y = params.source_y + gid.y;
    const uint dst_x = params.output_x + gid.x;
    const uint dst_y = params.output_y + gid.y;

    const uint src_idx = src_y * params.input_width + src_x;
    const uint dst_idx = dst_y * params.output_width + dst_x;
    output[dst_idx] = input[src_idx] + params.addend;
}
