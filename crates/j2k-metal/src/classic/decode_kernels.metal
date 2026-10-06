kernel void j2k_decode_classic_cleanup_batched(
    device const uchar *coded_data [[buffer(0)]],
    device float *output [[buffer(1)]],
    device const J2kClassicCleanupBatchJob *jobs [[buffer(2)]],
    device const J2kClassicSegment *segments [[buffer(3)]],
    device J2kClassicStatus *statuses [[buffer(4)]],
    device uint *coefficients_scratch [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    device J2kClassicStatus *status = statuses + gid;
    set_classic_status(status, J2K_CLASSIC_STATUS_OK, 0u);
        if (!decode_classic_job(
                jobs[gid],
                coded_data,
                segments,
                coefficients_scratch,
                gid * J2K_CLASSIC_MAX_COEFF_COUNT,
                output,
                true,
                status
            ) &&
        status->code == J2K_CLASSIC_STATUS_OK) {
        set_classic_status(status, J2K_CLASSIC_STATUS_FAIL, 0u);
    }
}

kernel void j2k_decode_classic_cleanup_plain_batched(
    device const uchar *coded_data [[buffer(0)]],
    device float *output [[buffer(1)]],
    device const J2kClassicCleanupBatchJob *jobs [[buffer(2)]],
    device const J2kClassicSegment *segments [[buffer(3)]],
    device J2kClassicStatus *statuses [[buffer(4)]],
    device uint *coefficients_scratch [[buffer(5)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    threadgroup uint shared_flags[J2K_CLASSIC_MAX_FLAG_WORDS];
    device J2kClassicStatus *status = statuses + gid;
    const J2kClassicCleanupBatchJob job = jobs[gid];
    const uint padded_width = job.width + J2K_CLASSIC_PADDING * 2u;
    const uint coeff_count = padded_width * (job.height + J2K_CLASSIC_PADDING * 2u);
    const uint flag_count = (job.width + 2u) * ((job.height + 3u) / 4u);
    device uint *coefficients = coefficients_scratch + gid * J2K_CLASSIC_MAX_COEFF_COUNT;

    for (uint idx = lane; idx < coeff_count; idx += 32u) {
        coefficients[idx] = 0u;
    }
    for (uint idx = lane; idx < flag_count; idx += 32u) {
        shared_flags[idx] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    if (lane == 0u) {
        set_classic_status(status, J2K_CLASSIC_STATUS_OK, 0u);
        if (!decode_classic_job_plain(
                job,
                coded_data,
                segments,
                coefficients_scratch,
                gid * J2K_CLASSIC_MAX_COEFF_COUNT,
                shared_flags,
                output,
                status
            ) &&
            status->code == J2K_CLASSIC_STATUS_OK) {
            set_classic_status(status, J2K_CLASSIC_STATUS_FAIL, 0u);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (status->code == J2K_CLASSIC_STATUS_OK) {
        store_classic_job_plain_output_tg(
            job,
            coefficients_scratch,
            gid * J2K_CLASSIC_MAX_COEFF_COUNT,
            output,
            lane
        );
    }
}

kernel void j2k_decode_classic_cleanup_plain_dense_batched(
    device const uchar *coded_data [[buffer(0)]],
    device float *output [[buffer(1)]],
    device const J2kClassicCleanupBatchJob *jobs [[buffer(2)]],
    device const J2kClassicSegment *segments [[buffer(3)]],
    device J2kClassicStatus *statuses [[buffer(4)]],
    device uint *coefficients_scratch [[buffer(5)]],
    device uint *flags_scratch [[buffer(6)]],
    constant uint &job_pitch [[buffer(7)]],
    device const uint2 *lane_groups [[buffer(8)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    threadgroup uint shared_contexts[19u * 32u];
    // Each SIMD group owns a contiguous run of 1..32 jobs (first job, job count).
    // Long code-blocks get fewer lanes so divergence does not stretch the
    // longest serial MQ stream; short ones keep all 32 lanes busy.
    const uint2 lane_group = lane_groups[gid];
    const uint job_index = lane_group.x + lane;
    if (lane >= lane_group.y || job_index >= job_pitch) {
        return;
    }
    const J2kClassicCleanupBatchJob job = jobs[job_index];
    // Keep each block's hot state contiguous as lanes diverge through MQ decoding.
    // A lane owns these slices and its context column; no inter-lane barrier is needed.
    device uint *coefficients = coefficients_scratch
        + ulong(job_index) * J2K_CLASSIC_MAX_COEFF_COUNT;
    device uint *flags = flags_scratch
        + ulong(job_index) * J2K_CLASSIC_MAX_FLAG_WORDS;
    J2kPlainThreadgroupSoaContexts contexts = { shared_contexts, lane };
    const uint padded_width = job.width + J2K_CLASSIC_PADDING * 2u;
    const uint coeff_count = padded_width * (job.height + J2K_CLASSIC_PADDING * 2u);
    const uint flag_count = (job.width + 2u) * ((job.height + 3u) / 4u);
    for (uint idx = 0u; idx < coeff_count; ++idx) {
        coefficients[idx] = 0u;
    }
    for (uint idx = 0u; idx < flag_count; ++idx) {
        flags[idx] = 0u;
    }
    device J2kClassicStatus *status = statuses + job_index;
    set_classic_status(status, J2K_CLASSIC_STATUS_OK, 0u);
    if (!decode_classic_job_plain_dense(
            job,
            coded_data,
            segments,
            coefficients,
            flags,
            contexts,
            output,
            status
        ) &&
        status->code == J2K_CLASSIC_STATUS_OK) {
        set_classic_status(status, J2K_CLASSIC_STATUS_FAIL, 0u);
    }
    if (status->code == J2K_CLASSIC_STATUS_OK) {
        const uint sample_count = job.width * job.height;
        for (uint sample_idx = 0u; sample_idx < sample_count; ++sample_idx) {
            const uint x = sample_idx % job.width;
            const uint y = sample_idx / job.width;
            const uint coeff_idx =
                coeff_index(padded_width, x + J2K_CLASSIC_PADDING, y + J2K_CLASSIC_PADDING);
            const uint coeff = coefficients[coeff_idx];
            output[job.output_offset + y * job.output_stride + x] =
                reconstructed_classic_sample(coeff, job) * job.dequantization_step;
        }
    }
}

kernel void j2k_decode_classic_cleanup_repeated_batched(
    device const uchar *coded_data [[buffer(0)]],
    device float *output [[buffer(1)]],
    device const J2kClassicCleanupBatchJob *jobs [[buffer(2)]],
    device const J2kClassicSegment *segments [[buffer(3)]],
    device J2kClassicStatus *statuses [[buffer(4)]],
    device uint *coefficients_scratch [[buffer(5)]],
    constant J2kClassicRepeatedBatchParams &repeated [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= repeated.job_count || gid.y >= repeated.batch_count) {
        return;
    }
    const uint linear_idx = gid.y * repeated.job_count + gid.x;
    device J2kClassicStatus *status = statuses + linear_idx;
    J2kClassicCleanupBatchJob job = jobs[gid.x];
    job.output_offset += gid.y * repeated.output_plane_len;
    set_classic_status(status, J2K_CLASSIC_STATUS_OK, 0u);
        if (!decode_classic_job(
                job,
                coded_data,
                segments,
                coefficients_scratch,
                linear_idx * J2K_CLASSIC_MAX_COEFF_COUNT,
                output,
                false,
                status
            ) &&
        status->code == J2K_CLASSIC_STATUS_OK) {
        set_classic_status(status, J2K_CLASSIC_STATUS_FAIL, 0u);
    }
}

kernel void j2k_store_classic_repeated_batched(
    device float *output [[buffer(0)]],
    device const J2kClassicCleanupBatchJob *jobs [[buffer(1)]],
    device const uint *coefficients_scratch [[buffer(2)]],
    constant J2kClassicRepeatedBatchParams &repeated [[buffer(3)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (gid.x >= repeated.job_count || gid.y >= repeated.batch_count) {
        return;
    }
    J2kClassicCleanupBatchJob job = jobs[gid.x];
    job.output_offset += gid.y * repeated.output_plane_len;
    const uint padded_width = job.width + J2K_CLASSIC_PADDING * 2u;
    const uint linear_idx = gid.y * repeated.job_count + gid.x;
    device const uint *coefficients =
        coefficients_scratch + linear_idx * J2K_CLASSIC_MAX_COEFF_COUNT;
    const uint sample_count = job.width * job.height;
    for (uint sample_idx = lane; sample_idx < sample_count; sample_idx += 32u) {
        const uint x = sample_idx % job.width;
        const uint y = sample_idx / job.width;
        const uint coeff =
            coefficients[coeff_index(padded_width, x + J2K_CLASSIC_PADDING, y + J2K_CLASSIC_PADDING)];
        output[job.output_offset + y * job.output_stride + x] =
            reconstructed_classic_sample(coeff, job) * job.dequantization_step;
    }
}

kernel void j2k_decode_classic_cleanup_plain_repeated_batched(
    device const uchar *coded_data [[buffer(0)]],
    device float *output [[buffer(1)]],
    device const J2kClassicCleanupBatchJob *jobs [[buffer(2)]],
    device const J2kClassicSegment *segments [[buffer(3)]],
    device J2kClassicStatus *statuses [[buffer(4)]],
    device uint *coefficients_scratch [[buffer(5)]],
    constant J2kClassicRepeatedBatchParams &repeated [[buffer(6)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (gid.x >= repeated.job_count || gid.y >= repeated.batch_count) {
        return;
    }
    threadgroup uint shared_flags[J2K_CLASSIC_MAX_FLAG_WORDS];
    const uint linear_idx = gid.y * repeated.job_count + gid.x;
    device J2kClassicStatus *status = statuses + linear_idx;
    J2kClassicCleanupBatchJob job = jobs[gid.x];
    job.output_offset += gid.y * repeated.output_plane_len;
    const uint padded_width = job.width + J2K_CLASSIC_PADDING * 2u;
    const uint coeff_count = padded_width * (job.height + J2K_CLASSIC_PADDING * 2u);
    const uint flag_count = (job.width + 2u) * ((job.height + 3u) / 4u);
    device uint *coefficients = coefficients_scratch + linear_idx * J2K_CLASSIC_MAX_COEFF_COUNT;

    for (uint idx = lane; idx < coeff_count; idx += 32u) {
        coefficients[idx] = 0u;
    }
    for (uint idx = lane; idx < flag_count; idx += 32u) {
        shared_flags[idx] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    if (lane == 0u) {
        set_classic_status(status, J2K_CLASSIC_STATUS_OK, 0u);
        if (!decode_classic_job_plain(
                job,
                coded_data,
                segments,
                coefficients_scratch,
                linear_idx * J2K_CLASSIC_MAX_COEFF_COUNT,
                shared_flags,
                output,
                status
            ) &&
            status->code == J2K_CLASSIC_STATUS_OK) {
            set_classic_status(status, J2K_CLASSIC_STATUS_FAIL, 0u);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (status->code == J2K_CLASSIC_STATUS_OK) {
        store_classic_job_plain_output_tg(
            job,
            coefficients_scratch,
            linear_idx * J2K_CLASSIC_MAX_COEFF_COUNT,
            output,
            lane
        );
    }
}

kernel void j2k_decode_classic_cleanup_plain_dev_repeated_batched(
    device const uchar *coded_data [[buffer(0)]],
    device float *output [[buffer(1)]],
    device const J2kClassicCleanupBatchJob *jobs [[buffer(2)]],
    device const J2kClassicSegment *segments [[buffer(3)]],
    device J2kClassicStatus *statuses [[buffer(4)]],
    device uint *coefficients_scratch [[buffer(5)]],
    device uint *flags_scratch [[buffer(6)]],
    constant J2kClassicRepeatedBatchParams &repeated [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= repeated.job_count || gid.y >= repeated.batch_count) {
        return;
    }
    const uint linear_idx = gid.y * repeated.job_count + gid.x;
    device J2kClassicStatus *status = statuses + linear_idx;
    J2kClassicCleanupBatchJob job = jobs[gid.x];
    job.output_offset += gid.y * repeated.output_plane_len;
    set_classic_status(status, J2K_CLASSIC_STATUS_OK, 0u);
    if (!decode_classic_job_plain_dev(
            job,
            coded_data,
            segments,
            coefficients_scratch,
            linear_idx * J2K_CLASSIC_MAX_COEFF_COUNT,
            flags_scratch,
            output,
            false,
            status
        ) &&
        status->code == J2K_CLASSIC_STATUS_OK) {
        set_classic_status(status, J2K_CLASSIC_STATUS_FAIL, 0u);
    }
}
