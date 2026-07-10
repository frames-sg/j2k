// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::resident_packet_plan::PreparedLosslessBatchTile;
use super::super::resident_tier1::J2kResidentTier1StatusReadback;
use super::{
    build_resident_batch_packet_plan, classic_cod_block_style_from_flags,
    classic_packet_output_capacity, collect_prepared_batch_retention, copied_slice_buffer,
    dispatch_batched_packet_payload_copy, finish_resident_encode_split_command_buffer_timed,
    hybrid_stage_signpost, label_command_buffer, label_compute_encoder,
    metal_profile_stages_enabled, prepare_classic_tier1, prepared_lossless_batch_tiles,
    schedule_resident_tier1_status_readback, size_of, take_recyclable_private_buffer,
    with_runtime_for_session, zeroed_recyclable_shared_buffer, zeroed_shared_buffer, Buffer,
    ClassicTier1Prepared, Duration, Error, Instant, J2kBatchedPacketPayloadCopyDispatch,
    J2kClassicEncodeOutputCapacityMode, J2kCodestreamAssemblyStatus, J2kPacketEncodeStatus,
    J2kPacketPayloadCopyJob, J2kPendingResidentLosslessCodestreamBatch, J2kResidentBatchEncodeItem,
    J2kResidentEncodeGpuStage, J2kResidentEncodeGpuStageCommandBuffer, J2kResidentEncodeStageStats,
    J2kResidentPacketBlockParams, MTLResourceOptions, MTLSize, MetalRuntime,
    ResidentBatchPacketPlan, ResidentBatchPacketPlanParams, ResidentTier1StatusReadbackRequest,
    PACKET_PAYLOAD_COPY_STRIPES_PER_JOB,
    SIGNPOST_ENCODE_HYBRID_CLASSIC_CODESTREAM_ASSEMBLY_COMMAND_ENCODE,
    SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKETIZATION_COMMAND_ENCODE,
    SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_BUFFER_SETUP, SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_PLAN,
    SIGNPOST_ENCODE_HYBRID_CLASSIC_PAYLOAD_COPY_COMMAND_ENCODE,
};

struct ClassicBatchSubmission {
    tier1: ClassicTier1Prepared,
    codestream_buffer: Buffer,
    codestream_offsets: Vec<usize>,
    codestream_capacities: Vec<usize>,
    codestream_status_buffer: Buffer,
    packet_status_buffer: Buffer,
    tier1_status_readback: Option<J2kResidentTier1StatusReadback>,
    packet_buffers: ClassicPacketBuffers,
    stage_stats: J2kResidentEncodeStageStats,
    codestream_payload_copy_dispatched: bool,
}

struct ClassicPacketBuffers {
    packet_resolution: Buffer,
    packet_subband: Buffer,
    resident_block: Buffer,
    packet_descriptor: Buffer,
    state_block: Buffer,
    packet_payload_copy_job: Buffer,
    header: Buffer,
    scratch: Buffer,
    packet_job: Buffer,
    codestream_job: Buffer,
}

impl ClassicPacketBuffers {
    fn retain_in(self, retained: &mut Vec<Buffer>, packet_status: &Buffer) {
        retained.push(self.packet_resolution);
        retained.push(self.packet_subband);
        retained.push(self.resident_block);
        retained.push(self.packet_descriptor);
        retained.push(self.state_block);
        retained.push(self.packet_payload_copy_job);
        retained.push(self.header);
        retained.push(self.scratch);
        retained.push(self.packet_job);
        retained.push(packet_status.clone());
        retained.push(self.codestream_job);
    }
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "resident packet submission preserves fixed command and resource order"
)]
fn submit_classic_packet_stages(
    runtime: &MetalRuntime,
    prepared_tiles: &[PreparedLosslessBatchTile],
    output_capacity_mode: J2kClassicEncodeOutputCapacityMode,
    profile_stages: bool,
) -> Result<ClassicBatchSubmission, Error> {
    // Commit classic stages independently so the long Tier-1 kernel can run
    // while CPU packet metadata for the following stages is built.
    let split_command_buffers = true;
    let mut classic_packet_plan_duration = Duration::ZERO;
    let mut classic_packet_buffer_setup_duration = Duration::ZERO;
    let packet_block_prep_duration = Duration::ZERO;
    let mut packetization_duration = Duration::ZERO;
    let mut codestream_assembly_duration = Duration::ZERO;
    let ClassicTier1Prepared {
        mut command_buffer,
        coefficient_buffer,
        tier1_jobs,
        tier1_job_count,
        tier1_job_buffer,
        tier1_output_buffer,
        tier1_status_buffer,
        tier1_segment_buffer,
        tile_tier1_job_bases,
        tile_tier1_output_capacities,
        tier1_output_capacity_total,
        max_tier1_output_capacity,
        tier1_segment_capacity_total,
        mut recyclable_private_buffers,
        mut recyclable_shared_buffers,
        mut gpu_stage_command_buffers,
        classic_tier1_density_readback,
        classic_tier1_raw_pack_buffer,
        classic_tier1_arithmetic_pack_buffer,
        classic_tier1_symbol_plan_readback,
        classic_tier1_pass_plan_readback,
        classic_tier1_token_emit_readback,
        classic_tier1_split_token_emit_readback,
        classic_gpu_token_pack_used,
        classic_resident_style_flags,
        classic_tier1_setup_duration,
        classic_block_encode_duration,
        mut classic_command_buffer_commit_duration,
    } = prepare_classic_tier1(
        runtime,
        prepared_tiles,
        output_capacity_mode,
        profile_stages,
    )?;
    let classic_packet_plan_started = profile_stages.then(Instant::now);
    let classic_packet_plan_signpost =
        hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_PLAN);
    let ResidentBatchPacketPlan {
        packet_resolutions,
        packet_subbands,
        resident_blocks,
        packet_descriptors,
        state_blocks,
        packet_jobs,
        assembly_jobs,
        packet_output_capacity_total,
        packet_payload_copy_job_capacity_total,
        max_payload_copy_jobs_per_tile,
        header_capacity_total,
        scratch_words_total,
        codestream_capacity_total,
        codestream_offsets,
        codestream_capacities,
    } = build_resident_batch_packet_plan(
        prepared_tiles,
        &tile_tier1_job_bases,
        ResidentBatchPacketPlanParams {
            family_name: "J2K",
            block_coding_mode: 0,
            high_throughput: 0,
            code_block_style: classic_cod_block_style_from_flags(classic_resident_style_flags),
        },
        |tile_index, tile, header_capacity| {
            classic_packet_output_capacity(
                tile_tier1_output_capacities[tile_index],
                header_capacity,
                tile.packet_descriptors.len().max(tile.resolutions.len()),
                tile.codestream,
            )
        },
    )?;
    drop(classic_packet_plan_signpost);
    if let Some(started) = classic_packet_plan_started {
        classic_packet_plan_duration = started.elapsed();
    }

    let classic_packet_buffer_setup_started = profile_stages.then(Instant::now);
    let classic_packet_buffer_setup_signpost =
        hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKET_BUFFER_SETUP);
    let packet_resolution_buffer = copied_slice_buffer(&runtime.device, &packet_resolutions);
    let packet_subband_buffer = copied_slice_buffer(&runtime.device, &packet_subbands);
    let resident_block_buffer = copied_slice_buffer(&runtime.device, &resident_blocks);
    let packet_descriptor_buffer = copied_slice_buffer(&runtime.device, &packet_descriptors);
    let state_block_buffer = copied_slice_buffer(&runtime.device, &state_blocks);
    let packet_payload_copy_job_buffer = take_recyclable_private_buffer(
        runtime,
        packet_payload_copy_job_capacity_total
            .max(1)
            .checked_mul(size_of::<J2kPacketPayloadCopyJob>())
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal batch packet payload-copy buffer size overflow".to_string(),
            })?,
        &mut recyclable_private_buffers,
    )?;
    let header_buffer = take_recyclable_private_buffer(
        runtime,
        header_capacity_total.max(1),
        &mut recyclable_private_buffers,
    )?;
    let scratch_buffer = take_recyclable_private_buffer(
        runtime,
        scratch_words_total.max(1) * size_of::<u32>(),
        &mut recyclable_private_buffers,
    )?;
    let packet_job_buffer = copied_slice_buffer(&runtime.device, &packet_jobs);
    let packet_status_buffer = zeroed_recyclable_shared_buffer(
        runtime,
        packet_jobs.len().max(1) * size_of::<J2kPacketEncodeStatus>(),
        &mut recyclable_shared_buffers,
    )?;
    let codestream_job_buffer = copied_slice_buffer(&runtime.device, &assembly_jobs);
    let codestream_buffer = runtime.device.new_buffer(
        codestream_capacity_total.max(1) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let codestream_status_buffer = zeroed_shared_buffer(
        &runtime.device,
        assembly_jobs.len() * size_of::<J2kCodestreamAssemblyStatus>(),
    );
    drop(classic_packet_buffer_setup_signpost);
    if let Some(started) = classic_packet_buffer_setup_started {
        classic_packet_buffer_setup_duration = started.elapsed();
    }

    let resident_block_params = J2kResidentPacketBlockParams {
        block_count: u32::try_from(resident_blocks.len()).map_err(|_| Error::MetalKernel {
            message: "J2K Metal batch resident block count exceeds u32".to_string(),
        })?,
        tier1_job_count,
    };

    let tile_count = u64::try_from(packet_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal batch tile count exceeds u64".to_string(),
    })?;
    let command_encode_started = profile_stages.then(Instant::now);
    let signpost =
        hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_CLASSIC_PACKETIZATION_COMMAND_ENCODE);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K packetization");
    encoder.set_compute_pipeline_state(&runtime.packet_encode_resident_classic_batched);
    encoder.set_buffer(0, Some(&packet_resolution_buffer), 0);
    encoder.set_buffer(1, Some(&packet_subband_buffer), 0);
    encoder.set_buffer(2, Some(&resident_block_buffer), 0);
    encoder.set_buffer(3, Some(&tier1_output_buffer), 0);
    encoder.set_buffer(4, Some(&codestream_buffer), 0);
    encoder.set_buffer(5, Some(&header_buffer), 0);
    encoder.set_buffer(6, Some(&scratch_buffer), 0);
    encoder.set_buffer(7, Some(&packet_job_buffer), 0);
    encoder.set_buffer(8, Some(&packet_status_buffer), 0);
    encoder.set_buffer(9, Some(&packet_descriptor_buffer), 0);
    encoder.set_buffer(10, Some(&state_block_buffer), 0);
    encoder.set_buffer(11, Some(&packet_payload_copy_job_buffer), 0);
    encoder.set_buffer(12, Some(&tier1_job_buffer), 0);
    encoder.set_buffer(13, Some(&tier1_status_buffer), 0);
    encoder.set_buffer(14, Some(&tier1_segment_buffer), 0);
    encoder.set_bytes(
        15,
        size_of::<J2kResidentPacketBlockParams>() as u64,
        (&raw const resident_block_params).cast(),
    );
    encoder.dispatch_threads(
        MTLSize {
            width: tile_count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: runtime
                .packet_encode_resident_classic_batched
                .thread_execution_width()
                .max(1),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    drop(signpost);
    if let Some(started) = command_encode_started {
        packetization_duration = packetization_duration.saturating_add(started.elapsed());
    }
    if split_command_buffers {
        command_buffer = finish_resident_encode_split_command_buffer_timed(
            command_buffer,
            runtime,
            J2kResidentEncodeGpuStage::Packetization,
            "j2k classic resident packet payload copy",
            &mut gpu_stage_command_buffers,
            profile_stages,
            &mut classic_command_buffer_commit_duration,
        );
    }
    let packet_payload_copy_dispatched = dispatch_batched_packet_payload_copy(
        runtime,
        &command_buffer,
        J2kBatchedPacketPayloadCopyDispatch {
            payload_buffer: &tier1_output_buffer,
            packet_output_buffer: &codestream_buffer,
            packet_job_buffer: &packet_job_buffer,
            packet_status_buffer: &packet_status_buffer,
            packet_payload_copy_job_buffer: &packet_payload_copy_job_buffer,
            tile_count,
            max_payload_copy_jobs_per_tile: max_payload_copy_jobs_per_tile as u64,
            label: "J2K packetization payload copy",
            signpost_name: SIGNPOST_ENCODE_HYBRID_CLASSIC_PAYLOAD_COPY_COMMAND_ENCODE,
        },
    );
    if split_command_buffers {
        if packet_payload_copy_dispatched {
            command_buffer = finish_resident_encode_split_command_buffer_timed(
                command_buffer,
                runtime,
                J2kResidentEncodeGpuStage::PacketPayloadCopy,
                "j2k classic resident codestream assembly",
                &mut gpu_stage_command_buffers,
                profile_stages,
                &mut classic_command_buffer_commit_duration,
            );
        } else {
            label_command_buffer(&command_buffer, "j2k classic resident codestream assembly");
        }
    }

    let max_packet_output_capacity = packet_jobs
        .iter()
        .map(|job| job.output_capacity)
        .max()
        .unwrap_or(0);
    let max_packet_output_capacity =
        usize::try_from(max_packet_output_capacity).map_err(|_| Error::MetalKernel {
            message: "J2K Metal batch max packet output capacity exceeds usize".to_string(),
        })?;
    let command_encode_started = profile_stages.then(Instant::now);
    let signpost =
        hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_CLASSIC_CODESTREAM_ASSEMBLY_COMMAND_ENCODE);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K codestream assembly");
    encoder.set_compute_pipeline_state(&runtime.lossless_codestream_assemble_batched);
    encoder.set_buffer(0, Some(&codestream_buffer), 0);
    encoder.set_buffer(1, Some(&packet_status_buffer), 0);
    encoder.set_buffer(2, Some(&codestream_buffer), 0);
    encoder.set_buffer(3, Some(&codestream_job_buffer), 0);
    encoder.set_buffer(4, Some(&codestream_status_buffer), 0);
    encoder.dispatch_threads(
        MTLSize {
            width: tile_count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: runtime
                .lossless_codestream_assemble_batched
                .thread_execution_width()
                .max(1),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    drop(signpost);
    if split_command_buffers {
        command_buffer = finish_resident_encode_split_command_buffer_timed(
            command_buffer,
            runtime,
            J2kResidentEncodeGpuStage::CodestreamAssembly,
            "j2k classic resident result readback",
            &mut gpu_stage_command_buffers,
            profile_stages,
            &mut classic_command_buffer_commit_duration,
        );
    }
    let codestream_payload_copy_dispatched = false;
    if let Some(started) = command_encode_started {
        codestream_assembly_duration =
            codestream_assembly_duration.saturating_add(started.elapsed());
    }
    let tier1_status_readback =
        schedule_resident_tier1_status_readback(ResidentTier1StatusReadbackRequest::classic(
            runtime,
            &command_buffer,
            &tier1_status_buffer,
            classic_resident_style_flags,
            &tier1_jobs,
            profile_stages,
        ))?;
    let final_commit_started = profile_stages.then(Instant::now);
    command_buffer.commit();
    if let Some(started) = final_commit_started {
        classic_command_buffer_commit_duration =
            classic_command_buffer_commit_duration.saturating_add(started.elapsed());
    }
    if split_command_buffers && codestream_payload_copy_dispatched {
        gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
            stage: J2kResidentEncodeGpuStage::CodestreamPayloadCopy,
            command_buffer: command_buffer.clone(),
        });
    }

    let packet_job_count = packet_jobs.len();
    let packet_buffers = ClassicPacketBuffers {
        packet_resolution: packet_resolution_buffer,
        packet_subband: packet_subband_buffer,
        resident_block: resident_block_buffer,
        packet_descriptor: packet_descriptor_buffer,
        state_block: state_block_buffer,
        packet_payload_copy_job: packet_payload_copy_job_buffer,
        header: header_buffer,
        scratch: scratch_buffer,
        packet_job: packet_job_buffer,
        codestream_job: codestream_job_buffer,
    };

    let stage_stats = J2kResidentEncodeStageStats {
        classic_tier1_setup_duration,
        classic_block_encode_duration,
        classic_packet_plan_duration,
        classic_packet_buffer_setup_duration,
        classic_command_buffer_commit_duration,
        packet_block_prep_duration,
        packetization_duration,
        codestream_assembly_duration,
        packet_payload_copy_job_capacity_total,
        max_packet_payload_copy_jobs_per_tile: max_payload_copy_jobs_per_tile,
        packet_payload_copy_launched_stripe_count_total: packet_job_count
            .saturating_mul(max_payload_copy_jobs_per_tile)
            .saturating_mul(PACKET_PAYLOAD_COPY_STRIPES_PER_JOB as usize),
        tier1_output_capacity_total,
        max_tier1_output_capacity,
        tier1_segment_capacity_total,
        max_tier1_segment_capacity_per_block: tier1_jobs
            .iter()
            .map(|job| job.segment_capacity as usize)
            .max()
            .unwrap_or(0),
        packet_output_capacity_total,
        max_packet_output_capacity,
        codestream_payload_copy_launched_thread_count_total: 0,
        code_block_count: tier1_jobs.len(),
        ..J2kResidentEncodeStageStats::default()
    };

    Ok(ClassicBatchSubmission {
        tier1: ClassicTier1Prepared {
            command_buffer,
            coefficient_buffer,
            tier1_jobs,
            tier1_job_count,
            tier1_job_buffer,
            tier1_output_buffer,
            tier1_status_buffer,
            tier1_segment_buffer,
            tile_tier1_job_bases,
            tile_tier1_output_capacities,
            tier1_output_capacity_total,
            max_tier1_output_capacity,
            tier1_segment_capacity_total,
            recyclable_private_buffers,
            recyclable_shared_buffers,
            gpu_stage_command_buffers,
            classic_tier1_density_readback,
            classic_tier1_raw_pack_buffer,
            classic_tier1_arithmetic_pack_buffer,
            classic_tier1_symbol_plan_readback,
            classic_tier1_pass_plan_readback,
            classic_tier1_token_emit_readback,
            classic_tier1_split_token_emit_readback,
            classic_gpu_token_pack_used,
            classic_resident_style_flags,
            classic_tier1_setup_duration,
            classic_block_encode_duration,
            classic_command_buffer_commit_duration,
        },
        codestream_buffer,
        codestream_offsets,
        codestream_capacities,
        codestream_status_buffer,
        packet_status_buffer,
        tier1_status_readback,
        packet_buffers,
        stage_stats,
        codestream_payload_copy_dispatched,
    })
}

#[cfg(target_os = "macos")]
fn finish_classic_batch(
    session: &crate::MetalBackendSession,
    prepared_tiles: Vec<PreparedLosslessBatchTile>,
    profile_stages: bool,
    submitted: ClassicBatchSubmission,
) -> Result<J2kPendingResidentLosslessCodestreamBatch, Error> {
    let ClassicBatchSubmission {
        tier1,
        codestream_buffer,
        codestream_offsets,
        codestream_capacities,
        codestream_status_buffer,
        packet_status_buffer,
        tier1_status_readback,
        packet_buffers,
        stage_stats,
        codestream_payload_copy_dispatched,
    } = submitted;
    let ClassicTier1Prepared {
        command_buffer,
        coefficient_buffer,
        tier1_jobs: _,
        tier1_job_count: _,
        tier1_job_buffer,
        tier1_output_buffer,
        tier1_status_buffer,
        tier1_segment_buffer,
        tile_tier1_job_bases: _,
        tile_tier1_output_capacities: _,
        tier1_output_capacity_total: _,
        max_tier1_output_capacity: _,
        tier1_segment_capacity_total: _,
        mut recyclable_private_buffers,
        recyclable_shared_buffers,
        mut gpu_stage_command_buffers,
        classic_tier1_density_readback,
        classic_tier1_raw_pack_buffer,
        classic_tier1_arithmetic_pack_buffer,
        classic_tier1_symbol_plan_readback,
        classic_tier1_pass_plan_readback,
        classic_tier1_token_emit_readback,
        classic_tier1_split_token_emit_readback,
        classic_gpu_token_pack_used,
        classic_resident_style_flags: _,
        classic_tier1_setup_duration: _,
        classic_block_encode_duration: _,
        classic_command_buffer_commit_duration: _,
    } = tier1;
    let (retained_command_buffers, mut retained_buffers) = collect_prepared_batch_retention(
        profile_stages,
        prepared_tiles,
        &mut gpu_stage_command_buffers,
        &mut recyclable_private_buffers,
    );
    retained_buffers.push(coefficient_buffer);
    retained_buffers.push(tier1_job_buffer);
    retained_buffers.push(tier1_output_buffer);
    retained_buffers.push(tier1_status_buffer);
    retained_buffers.push(tier1_segment_buffer);
    if let Some(buffer) = classic_tier1_raw_pack_buffer {
        retained_buffers.push(buffer);
    }
    if let Some(buffer) = classic_tier1_arithmetic_pack_buffer {
        retained_buffers.push(buffer);
    }
    packet_buffers.retain_in(&mut retained_buffers, &packet_status_buffer);

    Ok(J2kPendingResidentLosslessCodestreamBatch {
        runtime: session.runtime()?,
        buffer: codestream_buffer,
        byte_offsets: codestream_offsets,
        capacities: codestream_capacities,
        status_buffer: codestream_status_buffer,
        packet_status_buffer,
        tier1_status_readback,
        classic_tier1_density_readback,
        classic_tier1_symbol_plan_readback,
        classic_tier1_pass_plan_readback,
        classic_tier1_token_emit_readback,
        classic_tier1_split_token_emit_readback,
        classic_gpu_token_pack_used,
        command_buffer: command_buffer.clone(),
        retained_command_buffers,
        _retained_buffers: retained_buffers,
        recyclable_private_buffers,
        recyclable_shared_buffers,
        gpu_stage_command_buffers,
        stage_stats,
        codestream_payload_copy_dispatched,
        status_stage: "J2K batched codestream assembly",
        length_error: "J2K Metal batched codestream output length exceeds usize",
        capacity_error: "J2K Metal batched codestream output length exceeds buffer",
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn submit_lossless_codestream_buffers_from_prepared_classic_batch(
    session: &crate::MetalBackendSession,
    items: Vec<J2kResidentBatchEncodeItem>,
    output_capacity_mode: J2kClassicEncodeOutputCapacityMode,
) -> Result<J2kPendingResidentLosslessCodestreamBatch, Error> {
    if items.is_empty() {
        return Err(Error::MetalKernel {
            message: "J2K Metal resident batch encode requires at least one tile".to_string(),
        });
    }

    let prepared_tiles = prepared_lossless_batch_tiles(items);
    with_runtime_for_session(session, |runtime| {
        let profile_stages = metal_profile_stages_enabled();
        let submitted = submit_classic_packet_stages(
            runtime,
            &prepared_tiles,
            output_capacity_mode,
            profile_stages,
        )?;
        finish_classic_batch(session, prepared_tiles, profile_stages, submitted)
    })
}
