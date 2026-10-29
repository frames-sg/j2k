// SPDX-License-Identifier: MIT OR Apache-2.0

// Full origin-zero component expansion, matching the native decoder's store.
// params: output width, output height, horizontal sampling, vertical sampling.
// The grid's z index selects one of consecutive stacked planes; a 2D dispatch
// expands the single plane at z = 0.
kernel void j2k_expand_sampled_plane(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint *params [[buffer(2)]],
    uint3 pos [[thread_position_in_grid]]
) {
    if (pos.x >= params[0] || pos.y >= params[1]) return;
    const uint input_width = params[0] / params[2] + (params[0] % params[2] != 0u);
    const uint input_height = params[1] / params[3] + (params[1] % params[3] != 0u);
    const ulong input_plane = ulong(pos.z) * input_width * input_height;
    const ulong output_plane = ulong(pos.z) * params[0] * params[1];
    output[output_plane + ulong(pos.y) * params[0] + pos.x] =
        input[input_plane + ulong(pos.y / params[3]) * input_width + pos.x / params[2]];
}
