// SPDX-License-Identifier: MIT OR Apache-2.0

kernel void j2k_inverse_mct(
    device float *plane0 [[buffer(0)]],
    device float *plane1 [[buffer(1)]],
    device float *plane2 [[buffer(2)]],
    constant J2kInverseMctParams &params [[buffer(3)]],
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
        const float3 centered = j2k_inverse_ict_centered(y0, y1, y2);
        const bool round_centered = params.round_centered != 0u;
        plane0[gid] = j2k_shift_centered_sample(centered[0], params.addend0, round_centered);
        plane1[gid] = j2k_shift_centered_sample(centered[1], params.addend1, round_centered);
        plane2[gid] = j2k_shift_centered_sample(centered[2], params.addend2, round_centered);
        return;
    }

}
