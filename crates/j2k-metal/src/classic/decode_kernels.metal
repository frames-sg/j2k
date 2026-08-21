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
    threadgroup uchar shared_states[J2K_CLASSIC_MAX_COEFF_COUNT];
    device J2kClassicStatus *status = statuses + gid;
    const J2kClassicCleanupBatchJob job = jobs[gid];
    const uint padded_width = job.width + J2K_CLASSIC_PADDING * 2u;
    const uint coeff_count = padded_width * (job.height + J2K_CLASSIC_PADDING * 2u);
    device uint *coefficients = coefficients_scratch + gid * J2K_CLASSIC_MAX_COEFF_COUNT;

    for (uint idx = lane; idx < coeff_count; idx += 32u) {
        coefficients[idx] = 0u;
        shared_states[idx] = uchar(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    set_classic_status(status, J2K_CLASSIC_STATUS_OK, 0u);
    if (lane == 0u) {
        if (!decode_classic_job_plain(
                job,
                coded_data,
                segments,
                coefficients_scratch,
                gid * J2K_CLASSIC_MAX_COEFF_COUNT,
                shared_states,
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
            shared_states,
            output,
            lane
        );
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
    threadgroup uchar shared_states[J2K_CLASSIC_MAX_COEFF_COUNT];
    const uint linear_idx = gid.y * repeated.job_count + gid.x;
    device J2kClassicStatus *status = statuses + linear_idx;
    J2kClassicCleanupBatchJob job = jobs[gid.x];
    job.output_offset += gid.y * repeated.output_plane_len;
    const uint padded_width = job.width + J2K_CLASSIC_PADDING * 2u;
    const uint coeff_count = padded_width * (job.height + J2K_CLASSIC_PADDING * 2u);
    device uint *coefficients = coefficients_scratch + linear_idx * J2K_CLASSIC_MAX_COEFF_COUNT;

    for (uint idx = lane; idx < coeff_count; idx += 32u) {
        coefficients[idx] = 0u;
        shared_states[idx] = uchar(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    set_classic_status(status, J2K_CLASSIC_STATUS_OK, 0u);
    if (lane == 0u) {
        if (!decode_classic_job_plain(
                job,
                coded_data,
                segments,
                coefficients_scratch,
                linear_idx * J2K_CLASSIC_MAX_COEFF_COUNT,
                shared_states,
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
            shared_states,
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
    device uchar *states_scratch [[buffer(6)]],
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
            states_scratch,
            output,
            false,
            status
        ) &&
        status->code == J2K_CLASSIC_STATUS_OK) {
        set_classic_status(status, J2K_CLASSIC_STATUS_FAIL, 0u);
    }
}
