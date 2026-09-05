kernel void j2k_encode_classic_code_block(
    device const int *coefficients [[buffer(0)]],
    device uchar *out [[buffer(1)]],
    constant J2kClassicEncodeParams &params [[buffer(2)]],
    device J2kClassicEncodeStatus *status [[buffer(3)]],
    device J2kClassicSegment *segments [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) {
        return;
    }
   j2k_encode_classic_code_block_impl(coefficients, out, params, status, segments);
}

constant uint J2K_CLASSIC_CODE_BLOCK_MODE_DEFAULT = 0u;
constant uint J2K_CLASSIC_CODE_BLOCK_MODE_STYLE0 = 1u;
constant uint J2K_CLASSIC_CODE_BLOCK_MODE_32 = 2u;
constant uint J2K_CLASSIC_CODE_BLOCK_MODE_BYPASS_32 = 3u;
constant uint J2K_CLASSIC_CODE_BLOCK_MODE_BYPASS_U16_32 = 4u;
constant uint J2K_CLASSIC_CODE_BLOCK_MODE_STYLE0_32 = 5u;

inline void j2k_encode_classic_code_blocks_dispatch(
    device const int *coefficients,
    device uchar *out,
    device const J2kClassicEncodeBatchJob *jobs,
    device J2kClassicEncodeStatus *statuses,
    device J2kClassicSegment *segments,
    uint job_count,
    uint gid,
    uint mode
) {
    if (gid >= job_count) {
        return;
    }
    const J2kClassicEncodeBatchJob job = jobs[gid];
    J2kClassicEncodeParams params;
    params.width = job.width;
    params.height = job.height;
    params.sub_band_type = job.sub_band_type;
    params.total_bitplanes = job.total_bitplanes;
    params.style_flags = job.style_flags;
    params.output_capacity = job.output_capacity;
    params.segment_capacity = job.segment_capacity;

    device const int *job_coefficients = coefficients + job.coefficient_offset;
    device uchar *job_out = out + job.output_offset;
    device J2kClassicEncodeStatus *job_status = statuses + gid;
    device J2kClassicSegment *job_segments = segments + job.segment_offset;

    switch (mode) {
    case J2K_CLASSIC_CODE_BLOCK_MODE_STYLE0:
        params.style_flags = 0u;
        j2k_encode_classic_code_block_impl_style0(
            job_coefficients,
            job_out,
            params,
            job_status,
            job_segments
        );
        break;
    case J2K_CLASSIC_CODE_BLOCK_MODE_32:
        j2k_encode_classic_code_block_impl_32(
            job_coefficients,
            job_out,
            params,
            job_status,
            job_segments
        );
        break;
    case J2K_CLASSIC_CODE_BLOCK_MODE_BYPASS_32:
        params.style_flags = J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS;
        j2k_encode_classic_code_block_impl_bypass_32(
            job_coefficients,
            job_out,
            params,
            job_status,
            job_segments
        );
        break;
    case J2K_CLASSIC_CODE_BLOCK_MODE_BYPASS_U16_32:
        params.style_flags = J2K_CLASSIC_STYLE_SELECTIVE_ARITHMETIC_CODING_BYPASS;
        j2k_encode_classic_code_block_impl_bypass_u16_32(
            job_coefficients,
            job_out,
            params,
            job_status,
            job_segments
        );
        break;
    case J2K_CLASSIC_CODE_BLOCK_MODE_STYLE0_32:
        params.style_flags = 0u;
        j2k_encode_classic_code_block_impl_style0_32(
            job_coefficients,
            job_out,
            params,
            job_status,
            job_segments
        );
        break;
    default:
        j2k_encode_classic_code_block_impl(
            job_coefficients,
            job_out,
            params,
            job_status,
            job_segments
        );
        break;
    }
}

kernel void j2k_encode_classic_code_blocks(
    device const int *coefficients [[buffer(0)]],
    device uchar *out [[buffer(1)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(2)]],
    device J2kClassicEncodeStatus *statuses [[buffer(3)]],
    device J2kClassicSegment *segments [[buffer(4)]],
    constant uint &job_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    j2k_encode_classic_code_blocks_dispatch(
        coefficients,
        out,
        jobs,
        statuses,
        segments,
        job_count,
        gid,
        J2K_CLASSIC_CODE_BLOCK_MODE_DEFAULT
    );
}

kernel void j2k_encode_classic_code_blocks_style0(
    device const int *coefficients [[buffer(0)]],
    device uchar *out [[buffer(1)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(2)]],
    device J2kClassicEncodeStatus *statuses [[buffer(3)]],
    device J2kClassicSegment *segments [[buffer(4)]],
    constant uint &job_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    j2k_encode_classic_code_blocks_dispatch(
        coefficients,
        out,
        jobs,
        statuses,
        segments,
        job_count,
        gid,
        J2K_CLASSIC_CODE_BLOCK_MODE_STYLE0
    );
}

kernel void j2k_encode_classic_code_blocks_32(
    device const int *coefficients [[buffer(0)]],
    device uchar *out [[buffer(1)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(2)]],
    device J2kClassicEncodeStatus *statuses [[buffer(3)]],
    device J2kClassicSegment *segments [[buffer(4)]],
    constant uint &job_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    j2k_encode_classic_code_blocks_dispatch(
        coefficients,
        out,
        jobs,
        statuses,
        segments,
        job_count,
        gid,
        J2K_CLASSIC_CODE_BLOCK_MODE_32
    );
}

kernel void j2k_encode_classic_code_blocks_bypass_32(
    device const int *coefficients [[buffer(0)]],
    device uchar *out [[buffer(1)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(2)]],
    device J2kClassicEncodeStatus *statuses [[buffer(3)]],
    device J2kClassicSegment *segments [[buffer(4)]],
    constant uint &job_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    j2k_encode_classic_code_blocks_dispatch(
        coefficients,
        out,
        jobs,
        statuses,
        segments,
        job_count,
        gid,
        J2K_CLASSIC_CODE_BLOCK_MODE_BYPASS_32
    );
}

kernel void j2k_encode_classic_code_blocks_bypass_u16_32(
    device const int *coefficients [[buffer(0)]],
    device uchar *out [[buffer(1)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(2)]],
    device J2kClassicEncodeStatus *statuses [[buffer(3)]],
    device J2kClassicSegment *segments [[buffer(4)]],
    constant uint &job_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    j2k_encode_classic_code_blocks_dispatch(
        coefficients,
        out,
        jobs,
        statuses,
        segments,
        job_count,
        gid,
        J2K_CLASSIC_CODE_BLOCK_MODE_BYPASS_U16_32
    );
}

kernel void j2k_encode_classic_code_blocks_style0_32(
    device const int *coefficients [[buffer(0)]],
    device uchar *out [[buffer(1)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(2)]],
    device J2kClassicEncodeStatus *statuses [[buffer(3)]],
    device J2kClassicSegment *segments [[buffer(4)]],
    constant uint &job_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    j2k_encode_classic_code_blocks_dispatch(
        coefficients,
        out,
        jobs,
        statuses,
        segments,
        job_count,
        gid,
        J2K_CLASSIC_CODE_BLOCK_MODE_STYLE0_32
    );
}
