kernel void j2k_profile_classic_tier1_density_bypass_u16_32(
    device const int *coefficients [[buffer(0)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(1)]],
    device J2kClassicTier1DensityCounters *counters [[buffer(2)]],
    constant uint &job_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
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
   j2k_profile_classic_tier1_density_bypass_u16_32_impl(
        coefficients + job.coefficient_offset,
        params,
        counters + gid
    );
}

kernel void j2k_plan_classic_tier1_symbols_bypass_u16_32(
    device const int *coefficients [[buffer(0)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(1)]],
    device J2kClassicTier1SymbolPlanCounters *counters [[buffer(2)]],
    constant uint &job_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
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
   j2k_plan_classic_tier1_symbols_bypass_u16_32_impl(
        coefficients + job.coefficient_offset,
        params,
        counters + gid
    );
}

kernel void j2k_plan_classic_tier1_passes_bypass_u16_32(
    device const int *coefficients [[buffer(0)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(1)]],
    device J2kClassicTier1PassPlanCounters *counters [[buffer(2)]],
    constant uint &job_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
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
   j2k_plan_classic_tier1_passes_bypass_u16_32_impl(
        coefficients + job.coefficient_offset,
        params,
        counters + gid
    );
}

kernel void j2k_emit_classic_tier1_tokens_bypass_u16_32(
    device const int *coefficients [[buffer(0)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(1)]],
    device J2kClassicTier1SymbolPlanCounters *counters [[buffer(2)]],
    device uchar *token_data [[buffer(3)]],
    device J2kClassicTier1TokenSegment *token_segments [[buffer(4)]],
    constant uint &token_stride_bytes [[buffer(5)]],
    constant uint &token_segment_stride [[buffer(6)]],
    constant uint &job_count [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
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
   j2k_emit_classic_tier1_tokens_bypass_u16_32_impl(
        coefficients + job.coefficient_offset,
        params,
        counters + gid,
        token_data + (gid * token_stride_bytes),
        token_segments + (gid * token_segment_stride),
        token_stride_bytes,
        token_segment_stride
    );
}

kernel void j2k_emit_classic_tier1_split_tokens_bypass_u16_32(
    device const int *coefficients [[buffer(0)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(1)]],
    device J2kClassicTier1SymbolPlanCounters *counters [[buffer(2)]],
    device uchar *mq_token_data [[buffer(3)]],
    device uchar *raw_token_data [[buffer(4)]],
    device J2kClassicTier1TokenSegment *token_segments [[buffer(5)]],
    constant uint &mq_token_stride_bytes [[buffer(6)]],
    constant uint &raw_token_stride_bytes [[buffer(7)]],
    constant uint &token_segment_stride [[buffer(8)]],
    constant uint &job_count [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
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
   j2k_emit_classic_tier1_split_tokens_bypass_u16_32_impl(
        coefficients + job.coefficient_offset,
        params,
        counters + gid,
        mq_token_data + (gid * mq_token_stride_bytes),
        raw_token_data + (gid * raw_token_stride_bytes),
        token_segments + (gid * token_segment_stride),
        mq_token_stride_bytes,
        raw_token_stride_bytes,
        token_segment_stride
    );
}

kernel void j2k_emit_classic_tier1_split_mq_byte_raw_tokens_bypass_u16_32(
    device const int *coefficients [[buffer(0)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(1)]],
    device J2kClassicTier1SymbolPlanCounters *counters [[buffer(2)]],
    device uchar *mq_token_data [[buffer(3)]],
    device uchar *raw_token_data [[buffer(4)]],
    device J2kClassicTier1TokenSegment *token_segments [[buffer(5)]],
    constant uint &mq_token_stride_bytes [[buffer(6)]],
    constant uint &raw_token_stride_bytes [[buffer(7)]],
    constant uint &token_segment_stride [[buffer(8)]],
    constant uint &job_count [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
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
   j2k_emit_classic_tier1_split_mq_byte_raw_tokens_bypass_u16_32_impl(
        coefficients + job.coefficient_offset,
        params,
        counters + gid,
        mq_token_data + (gid * mq_token_stride_bytes),
        raw_token_data + (gid * raw_token_stride_bytes),
        token_segments + (gid * token_segment_stride),
        mq_token_stride_bytes,
        raw_token_stride_bytes,
        token_segment_stride
    );
}

kernel void j2k_pack_classic_tier1_tokens_bypass_u16_32(
    device const J2kClassicEncodeBatchJob *jobs [[buffer(0)]],
    device const J2kClassicTier1SymbolPlanCounters *counters [[buffer(1)]],
    device const uchar *token_data [[buffer(2)]],
    device const J2kClassicTier1TokenSegment *token_segments [[buffer(3)]],
    device uchar *out [[buffer(4)]],
    device J2kClassicEncodeStatus *statuses [[buffer(5)]],
    device J2kClassicSegment *segments [[buffer(6)]],
    constant uint &token_stride_bytes [[buffer(7)]],
    constant uint &token_segment_stride [[buffer(8)]],
    constant uint &job_count [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
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
   j2k_pack_classic_tier1_tokens_bypass_u16_32_impl(
        params,
        counters[gid],
        token_data + (gid * token_stride_bytes),
        token_segments + (gid * token_segment_stride),
        token_stride_bytes,
        token_segment_stride,
        out + job.output_offset,
        statuses + gid,
        segments + job.segment_offset
    );
}

kernel void j2k_pack_classic_tier1_split_tokens_bypass_u16_32(
    device const J2kClassicEncodeBatchJob *jobs [[buffer(0)]],
    device const J2kClassicTier1SymbolPlanCounters *counters [[buffer(1)]],
    device const uchar *mq_token_data [[buffer(2)]],
    device const uchar *raw_token_data [[buffer(3)]],
    device const J2kClassicTier1TokenSegment *token_segments [[buffer(4)]],
    device uchar *out [[buffer(5)]],
    device J2kClassicEncodeStatus *statuses [[buffer(6)]],
    device J2kClassicSegment *segments [[buffer(7)]],
    constant uint &mq_token_stride_bytes [[buffer(8)]],
    constant uint &raw_token_stride_bytes [[buffer(9)]],
    constant uint &token_segment_stride [[buffer(10)]],
    constant uint &job_count [[buffer(11)]],
    uint gid [[thread_position_in_grid]]
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
   j2k_pack_classic_tier1_split_tokens_bypass_u16_32_impl(
        params,
        counters[gid],
        mq_token_data + gid * mq_token_stride_bytes,
        raw_token_data + gid * raw_token_stride_bytes,
        token_segments + gid * token_segment_stride,
        mq_token_stride_bytes,
        raw_token_stride_bytes,
        token_segment_stride,
        out + job.output_offset,
        statuses + gid,
        segments + job.segment_offset
    );
}

kernel void j2k_profile_classic_tier1_raw_pack_bypass_u16_32(
    device const int *coefficients [[buffer(0)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(1)]],
    device uchar *out [[buffer(2)]],
    constant uint &job_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
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
   j2k_profile_classic_tier1_raw_pack_bypass_u16_32_impl(
        coefficients + job.coefficient_offset,
        out + job.output_offset,
        params
    );
}

kernel void j2k_profile_classic_tier1_arithmetic_pack_bypass_u16_32(
    device const int *coefficients [[buffer(0)]],
    device const J2kClassicEncodeBatchJob *jobs [[buffer(1)]],
    device uchar *out [[buffer(2)]],
    constant uint &job_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
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
   j2k_profile_classic_tier1_arithmetic_pack_bypass_u16_32_impl(
        coefficients + job.coefficient_offset,
        out + job.output_offset,
        params
    );
}
