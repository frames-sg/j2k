// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use super::test_counters;
use super::{
    accumulate_classic_tier1_scan_estimates, checked_buffer_slice, classic_tier1_pass_class_counts,
    compare_classic_tier1_symbol_plan_and_pass_plan_counters,
    compare_classic_tier1_symbol_plan_and_split_token_emit_counters,
    compare_classic_tier1_symbol_plan_and_token_emit_counters,
    completed_command_buffers_gpu_duration_and_elapsed_window, duration_share, encode_status_error,
    hybrid_stage_signpost, metal_profile_stages_enabled, packet_encode_status_error,
    profile_classic_tier1_token_pack, record_classic_tier1_density_counters,
    record_classic_tier1_pass_plan_counters, record_classic_tier1_symbol_plan_counters,
    record_classic_tier1_token_emit_counters, record_completed_resident_encode_gpu_stages,
    recycle_private_buffers, recycle_shared_buffers,
    validate_classic_tier1_split_token_emit_counters, wait_for_completion_metal, CommandBufferRef,
    Error, Instant, J2kClassicEncodeStatus, J2kCodestreamAssemblyStatus, J2kHtEncodeStatus,
    J2kPacketEncodeStatus, J2kPendingResidentLosslessCodestreamBatch, J2kResidentEncodeStageStats,
    J2kResidentLosslessCodestream, J2kResidentLosslessCodestreamBatchResult,
    J2kResidentTier1StatusKind, J2kResidentTier1StatusReadback, J2K_ENCODE_STATUS_OK,
    PACKET_PAYLOAD_COPY_STRIPES_PER_JOB, SIGNPOST_ENCODE_HYBRID_COMMAND_WAIT,
    SIGNPOST_ENCODE_HYBRID_RESULT_HARVEST,
};

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "output validation is an exhaustive status-kind mapping"
)]
pub(in crate::compute) fn record_resident_tier1_output_usage(
    stage_stats: &mut J2kResidentEncodeStageStats,
    readback: &J2kResidentTier1StatusReadback,
    classic_gpu_token_pack_used: bool,
) -> Result<(), Error> {
    match readback.kind {
        J2kResidentTier1StatusKind::Classic => {
            let classic_jobs =
                readback
                    .classic_jobs
                    .as_ref()
                    .ok_or_else(|| Error::MetalKernel {
                        message:
                            "J2K Metal classic Tier-1 profile readback is missing job metadata"
                                .to_string(),
                    })?;
            let statuses = checked_buffer_slice::<J2kClassicEncodeStatus>(
                &readback.buffer,
                readback.count,
                "resident classic Tier-1 statuses",
            )?;
            if classic_jobs.len() != statuses.len() {
                return Err(Error::MetalKernel {
                    message: "J2K Metal classic Tier-1 profile readback job/status count mismatch"
                        .to_string(),
                });
            }
            for (status, job) in statuses.iter().zip(classic_jobs) {
                if status.code != J2K_ENCODE_STATUS_OK {
                    return Err(encode_status_error(
                        "classic Tier-1",
                        status.code,
                        status.detail,
                    ));
                }
                let data_len =
                    usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
                        message: "J2K Metal classic Tier-1 output length exceeds usize".to_string(),
                    })?;
                stage_stats.tier1_output_used_bytes_total = stage_stats
                    .tier1_output_used_bytes_total
                    .saturating_add(data_len);
                stage_stats.max_tier1_output_used_bytes =
                    stage_stats.max_tier1_output_used_bytes.max(data_len);
                if classic_gpu_token_pack_used {
                    stage_stats.tier1_token_pack_output_bytes_total = stage_stats
                        .tier1_token_pack_output_bytes_total
                        .saturating_add(data_len);
                    stage_stats.max_tier1_token_pack_output_bytes_per_block = stage_stats
                        .max_tier1_token_pack_output_bytes_per_block
                        .max(data_len);
                }
                let coding_passes =
                    usize::try_from(status.number_of_coding_passes).map_err(|_| {
                        Error::MetalKernel {
                            message: "J2K Metal classic Tier-1 coding-pass count exceeds usize"
                                .to_string(),
                        }
                    })?;
                stage_stats.tier1_coding_pass_count_total = stage_stats
                    .tier1_coding_pass_count_total
                    .saturating_add(coding_passes);
                stage_stats.max_tier1_coding_passes_per_block = stage_stats
                    .max_tier1_coding_passes_per_block
                    .max(coding_passes);
                let pass_counts =
                    classic_tier1_pass_class_counts(coding_passes, readback.classic_style_flags);
                let coeff_count = usize::try_from(job.width)
                    .and_then(|width| {
                        usize::try_from(job.height).map(|height| width.saturating_mul(height))
                    })
                    .map_err(|_| Error::MetalKernel {
                        message: "J2K Metal classic Tier-1 code-block dimensions exceed usize"
                            .to_string(),
                    })?;
                accumulate_classic_tier1_scan_estimates(stage_stats, pass_counts, coeff_count);
                stage_stats.tier1_arithmetic_pass_count_total = stage_stats
                    .tier1_arithmetic_pass_count_total
                    .saturating_add(pass_counts.arithmetic);
                stage_stats.tier1_raw_pass_count_total = stage_stats
                    .tier1_raw_pass_count_total
                    .saturating_add(pass_counts.raw);
                stage_stats.tier1_cleanup_pass_count_total = stage_stats
                    .tier1_cleanup_pass_count_total
                    .saturating_add(pass_counts.cleanup);
                stage_stats.tier1_sigprop_pass_count_total = stage_stats
                    .tier1_sigprop_pass_count_total
                    .saturating_add(pass_counts.sigprop);
                stage_stats.tier1_magref_pass_count_total = stage_stats
                    .tier1_magref_pass_count_total
                    .saturating_add(pass_counts.magref);
                stage_stats.tier1_arithmetic_cleanup_pass_count_total = stage_stats
                    .tier1_arithmetic_cleanup_pass_count_total
                    .saturating_add(pass_counts.arithmetic_cleanup);
                stage_stats.tier1_arithmetic_sigprop_pass_count_total = stage_stats
                    .tier1_arithmetic_sigprop_pass_count_total
                    .saturating_add(pass_counts.arithmetic_sigprop);
                stage_stats.tier1_arithmetic_magref_pass_count_total = stage_stats
                    .tier1_arithmetic_magref_pass_count_total
                    .saturating_add(pass_counts.arithmetic_magref);
                stage_stats.tier1_raw_sigprop_pass_count_total = stage_stats
                    .tier1_raw_sigprop_pass_count_total
                    .saturating_add(pass_counts.raw_sigprop);
                stage_stats.tier1_raw_magref_pass_count_total = stage_stats
                    .tier1_raw_magref_pass_count_total
                    .saturating_add(pass_counts.raw_magref);
                if coding_passes == 0 {
                    stage_stats.tier1_zero_block_count_total =
                        stage_stats.tier1_zero_block_count_total.saturating_add(1);
                } else {
                    stage_stats.tier1_nonzero_block_count_total = stage_stats
                        .tier1_nonzero_block_count_total
                        .saturating_add(1);
                }
                let missing_bitplanes =
                    usize::try_from(status.missing_bit_planes).map_err(|_| Error::MetalKernel {
                        message: "J2K Metal classic Tier-1 missing-bitplane count exceeds usize"
                            .to_string(),
                    })?;
                stage_stats.tier1_missing_bitplane_count_total = stage_stats
                    .tier1_missing_bitplane_count_total
                    .saturating_add(missing_bitplanes);
                stage_stats.max_tier1_missing_bitplanes_per_block = stage_stats
                    .max_tier1_missing_bitplanes_per_block
                    .max(missing_bitplanes);
                let segment_count =
                    usize::try_from(status.segment_count).map_err(|_| Error::MetalKernel {
                        message: "J2K Metal classic Tier-1 segment count exceeds usize".to_string(),
                    })?;
                stage_stats.tier1_segment_count_total = stage_stats
                    .tier1_segment_count_total
                    .saturating_add(segment_count);
                stage_stats.max_tier1_segments_per_block =
                    stage_stats.max_tier1_segments_per_block.max(segment_count);
            }
        }
        J2kResidentTier1StatusKind::HighThroughput => {
            let statuses = checked_buffer_slice::<J2kHtEncodeStatus>(
                &readback.buffer,
                readback.count,
                "resident HT Tier-1 statuses",
            )?;
            for status in statuses {
                if status.code != J2K_ENCODE_STATUS_OK {
                    return Err(encode_status_error(
                        "HTJ2K Tier-1",
                        status.code,
                        status.detail,
                    ));
                }
                let data_len =
                    usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
                        message: "HTJ2K Metal Tier-1 output length exceeds usize".to_string(),
                    })?;
                stage_stats.tier1_output_used_bytes_total = stage_stats
                    .tier1_output_used_bytes_total
                    .saturating_add(data_len);
                stage_stats.max_tier1_output_used_bytes =
                    stage_stats.max_tier1_output_used_bytes.max(data_len);
                let coding_passes =
                    usize::try_from(status.num_coding_passes).map_err(|_| Error::MetalKernel {
                        message: "HTJ2K Metal Tier-1 coding-pass count exceeds usize".to_string(),
                    })?;
                stage_stats.tier1_coding_pass_count_total = stage_stats
                    .tier1_coding_pass_count_total
                    .saturating_add(coding_passes);
                stage_stats.max_tier1_coding_passes_per_block = stage_stats
                    .max_tier1_coding_passes_per_block
                    .max(coding_passes);
                if coding_passes == 0 {
                    stage_stats.tier1_zero_block_count_total =
                        stage_stats.tier1_zero_block_count_total.saturating_add(1);
                } else {
                    stage_stats.tier1_nonzero_block_count_total = stage_stats
                        .tier1_nonzero_block_count_total
                        .saturating_add(1);
                }
                let missing_bitplanes =
                    usize::try_from(status.num_zero_bitplanes).map_err(|_| Error::MetalKernel {
                        message: "HTJ2K Metal Tier-1 missing-bitplane count exceeds usize"
                            .to_string(),
                    })?;
                stage_stats.tier1_missing_bitplane_count_total = stage_stats
                    .tier1_missing_bitplane_count_total
                    .saturating_add(missing_bitplanes);
                stage_stats.max_tier1_missing_bitplanes_per_block = stage_stats
                    .max_tier1_missing_bitplanes_per_block
                    .max(missing_bitplanes);
            }
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn wait_resident_codestream_command_buffer(
    command_buffer: &CommandBufferRef,
) -> Result<(), Error> {
    #[cfg(test)]
    test_counters::record_resident_codestream_command_buffer_wait();
    let _signpost = hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_COMMAND_WAIT);
    wait_for_completion_metal(command_buffer)
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "result harvesting keeps buffer validation and timing ownership atomic"
)]
pub(in crate::compute) fn finish_completed_resident_lossless_codestream_batch(
    pending: J2kPendingResidentLosslessCodestreamBatch,
) -> Result<J2kResidentLosslessCodestreamBatchResult, Error> {
    let _signpost = hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_RESULT_HARVEST);
    let profile_stages = metal_profile_stages_enabled();
    let result_harvest_started = profile_stages.then(Instant::now);
    let gpu_timings = completed_command_buffers_gpu_duration_and_elapsed_window(
        &pending.retained_command_buffers,
        &pending.command_buffer,
    );
    let gpu_duration = gpu_timings.map(|timings| timings.0);
    let gpu_elapsed_wall_duration = gpu_timings.map(|timings| timings.1);
    let mut stage_stats = pending.stage_stats;
    if let Some(duration) = gpu_elapsed_wall_duration {
        stage_stats.gpu_elapsed_wall_duration = duration;
    }
    if profile_stages {
        record_completed_resident_encode_gpu_stages(
            &mut stage_stats,
            &pending.gpu_stage_command_buffers,
        );
    }
    if let Some(readback) = pending.tier1_status_readback.as_ref() {
        record_resident_tier1_output_usage(
            &mut stage_stats,
            readback,
            pending.classic_gpu_token_pack_used,
        )?;
    }
    if let Some(readback) = pending.classic_tier1_density_readback.as_ref() {
        record_classic_tier1_density_counters(&mut stage_stats, readback)?;
    }
    if let Some(readback) = pending.classic_tier1_symbol_plan_readback.as_ref() {
        record_classic_tier1_symbol_plan_counters(&mut stage_stats, readback)?;
    }
    if let (Some(symbol_plan), Some(pass_plan)) = (
        pending.classic_tier1_symbol_plan_readback.as_ref(),
        pending.classic_tier1_pass_plan_readback.as_ref(),
    ) {
        compare_classic_tier1_symbol_plan_and_pass_plan_counters(symbol_plan, pass_plan)?;
    }
    if let Some(readback) = pending.classic_tier1_pass_plan_readback.as_ref() {
        record_classic_tier1_pass_plan_counters(&mut stage_stats, readback)?;
    }
    if let (Some(symbol_plan), Some(token_emit)) = (
        pending.classic_tier1_symbol_plan_readback.as_ref(),
        pending.classic_tier1_token_emit_readback.as_ref(),
    ) {
        compare_classic_tier1_symbol_plan_and_token_emit_counters(symbol_plan, token_emit)?;
    }
    if let Some(readback) = pending.classic_tier1_token_emit_readback.as_ref() {
        record_classic_tier1_token_emit_counters(&mut stage_stats, readback)?;
        profile_classic_tier1_token_pack(&mut stage_stats, readback)?;
    }
    if let Some(readback) = pending.classic_tier1_split_token_emit_readback.as_ref() {
        validate_classic_tier1_split_token_emit_counters(readback)?;
    }
    if let (Some(symbol_plan), Some(split_emit)) = (
        pending.classic_tier1_symbol_plan_readback.as_ref(),
        pending.classic_tier1_split_token_emit_readback.as_ref(),
    ) {
        compare_classic_tier1_symbol_plan_and_split_token_emit_counters(symbol_plan, split_emit)?;
    }
    let runtime = pending.runtime.clone();
    let recyclable_private_buffers = pending.recyclable_private_buffers;
    let private_recycle_started = profile_stages.then(Instant::now);
    recycle_private_buffers(&runtime, recyclable_private_buffers)?;
    if let Some(started) = private_recycle_started {
        stage_stats.result_private_recycle_duration = started.elapsed();
    }
    let gpu_duration_share =
        gpu_duration.map(|duration| duration_share(duration, pending.capacities.len()));
    let status_copy_started = profile_stages.then(Instant::now);
    let statuses = checked_buffer_slice::<J2kCodestreamAssemblyStatus>(
        &pending.status_buffer,
        pending.capacities.len(),
        "resident codestream assembly statuses",
    )?;
    let packet_statuses = checked_buffer_slice::<J2kPacketEncodeStatus>(
        &pending.packet_status_buffer,
        pending.capacities.len(),
        "resident packet encode statuses",
    )?;
    if let Some(started) = status_copy_started {
        stage_stats.result_status_copy_duration = started.elapsed();
    }
    let recyclable_shared_buffers = pending.recyclable_shared_buffers;
    let shared_recycle_started = profile_stages.then(Instant::now);
    recycle_shared_buffers(&runtime, recyclable_shared_buffers)?;
    if let Some(started) = shared_recycle_started {
        stage_stats.result_shared_recycle_duration = started.elapsed();
    }
    let codestream_collect_started = profile_stages.then(Instant::now);
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K Metal resident codestream result collection",
    );
    let mut codestreams = budget.try_vec(
        pending.capacities.len(),
        "J2K Metal resident codestream results",
    )?;
    for (index, status) in statuses.into_iter().enumerate() {
        let packet_status = packet_statuses
            .get(index)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal packetization status missing for resident batch tile"
                    .to_string(),
            })?;
        if packet_status.code != J2K_ENCODE_STATUS_OK {
            return Err(packet_encode_status_error(*packet_status));
        }
        if status.code != J2K_ENCODE_STATUS_OK {
            return Err(encode_status_error(
                pending.status_stage,
                status.code,
                status.detail,
            ));
        }
        let data_len = usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
            message: pending.length_error.to_string(),
        })?;
        let capacity = pending.capacities[index];
        if data_len > capacity {
            return Err(Error::MetalKernel {
                message: pending.capacity_error.to_string(),
            });
        }
        let packet_output_used =
            usize::try_from(packet_status.data_len).map_err(|_| Error::MetalKernel {
                message: "J2K Metal packet output length exceeds usize".to_string(),
            })?;
        let packet_payload_copy_jobs =
            usize::try_from(packet_status.detail).map_err(|_| Error::MetalKernel {
                message: "J2K Metal packet payload-copy count exceeds usize".to_string(),
            })?;
        let packet_payload_copy_bytes =
            usize::try_from(packet_status.payload_copy_bytes).map_err(|_| Error::MetalKernel {
                message: "J2K Metal packet payload-copy byte count exceeds usize".to_string(),
            })?;
        let packet_payload_copy_small_jobs = usize::try_from(packet_status.payload_copy_small_jobs)
            .map_err(|_| Error::MetalKernel {
                message: "J2K Metal small packet payload-copy count exceeds usize".to_string(),
            })?;
        let packet_payload_copy_medium_jobs =
            usize::try_from(packet_status.payload_copy_medium_jobs).map_err(|_| {
                Error::MetalKernel {
                    message: "J2K Metal medium packet payload-copy count exceeds usize".to_string(),
                }
            })?;
        let packet_payload_copy_large_jobs = usize::try_from(packet_status.payload_copy_large_jobs)
            .map_err(|_| Error::MetalKernel {
                message: "J2K Metal large packet payload-copy count exceeds usize".to_string(),
            })?;
        let packet_payload_copy_active_stripes =
            packet_payload_copy_jobs.saturating_mul(PACKET_PAYLOAD_COPY_STRIPES_PER_JOB as usize);
        stage_stats.packet_output_used_bytes_total = stage_stats
            .packet_output_used_bytes_total
            .saturating_add(packet_output_used);
        stage_stats.max_packet_output_used_bytes = stage_stats
            .max_packet_output_used_bytes
            .max(packet_output_used);
        stage_stats.packet_payload_copy_job_count_total = stage_stats
            .packet_payload_copy_job_count_total
            .saturating_add(packet_payload_copy_jobs);
        stage_stats.max_packet_payload_copy_jobs_used_per_tile = stage_stats
            .max_packet_payload_copy_jobs_used_per_tile
            .max(packet_payload_copy_jobs);
        stage_stats.packet_payload_copy_bytes_total = stage_stats
            .packet_payload_copy_bytes_total
            .saturating_add(packet_payload_copy_bytes);
        stage_stats.max_packet_payload_copy_bytes_per_tile = stage_stats
            .max_packet_payload_copy_bytes_per_tile
            .max(packet_payload_copy_bytes);
        stage_stats.packet_payload_copy_small_job_count_total = stage_stats
            .packet_payload_copy_small_job_count_total
            .saturating_add(packet_payload_copy_small_jobs);
        stage_stats.packet_payload_copy_medium_job_count_total = stage_stats
            .packet_payload_copy_medium_job_count_total
            .saturating_add(packet_payload_copy_medium_jobs);
        stage_stats.packet_payload_copy_large_job_count_total = stage_stats
            .packet_payload_copy_large_job_count_total
            .saturating_add(packet_payload_copy_large_jobs);
        stage_stats.packet_payload_copy_active_stripe_count_total = stage_stats
            .packet_payload_copy_active_stripe_count_total
            .saturating_add(packet_payload_copy_active_stripes);
        if pending.codestream_payload_copy_dispatched {
            stage_stats.codestream_payload_copy_bytes_total = stage_stats
                .codestream_payload_copy_bytes_total
                .saturating_add(packet_output_used);
        }
        codestreams.push(J2kResidentLosslessCodestream {
            buffer: pending.buffer.clone(),
            byte_offset: pending.byte_offsets[index],
            byte_len: data_len,
            capacity,
            gpu_duration: gpu_duration_share,
        });
    }
    if let Some(started) = codestream_collect_started {
        stage_stats.result_codestream_collect_duration = started.elapsed();
    }
    if let Some(started) = result_harvest_started {
        stage_stats.result_harvest_duration = started.elapsed();
    }
    Ok(J2kResidentLosslessCodestreamBatchResult {
        codestreams,
        stage_stats,
    })
}
