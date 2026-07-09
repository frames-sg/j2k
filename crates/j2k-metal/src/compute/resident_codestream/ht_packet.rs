// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::resident_packet_plan::PreparedLosslessBatchTile;
use super::super::resident_tier1::J2kResidentTier1StatusReadback;
use super::{
    build_resident_batch_packet_plan, collect_prepared_batch_retention,
    copied_recyclable_shared_slice_buffer, dispatch_batched_packet_payload_copy,
    finish_resident_encode_split_command_buffer, ht_packet_output_capacity_for_mode,
    hybrid_stage_signpost, label_command_buffer, label_compute_encoder,
    metal_profile_stages_enabled, prepare_ht_tier1, prepared_lossless_batch_tiles,
    schedule_resident_tier1_status_readback, size_of, take_recyclable_private_buffer,
    with_runtime_for_session, zeroed_recyclable_shared_buffer, Buffer, Duration, Error,
    HtTier1Prepared, Instant, J2kBatchedPacketPayloadCopyDispatch, J2kCodestreamAssemblyStatus,
    J2kHtPacketOutputCapacityMode, J2kPacketBlock, J2kPacketEncodeStatus, J2kPacketPayloadCopyJob,
    J2kPendingResidentLosslessCodestreamBatch, J2kResidentBatchEncodeItem,
    J2kResidentEncodeGpuStage, J2kResidentEncodeGpuStageCommandBuffer, J2kResidentEncodeStageStats,
    J2kResidentPacketBlockParams, MTLResourceOptions, MTLSize, MetalRuntime,
    ResidentBatchPacketPlan, ResidentBatchPacketPlanParams, ResidentTier1StatusReadbackRequest,
    PACKET_PAYLOAD_COPY_STRIPES_PER_JOB,
    SIGNPOST_ENCODE_HYBRID_HT_CODESTREAM_ASSEMBLY_COMMAND_ENCODE,
    SIGNPOST_ENCODE_HYBRID_HT_PACKETIZATION_COMMAND_ENCODE,
    SIGNPOST_ENCODE_HYBRID_HT_PACKET_BLOCK_PREP_COMMAND_ENCODE,
    SIGNPOST_ENCODE_HYBRID_HT_PACKET_BUFFER_SETUP, SIGNPOST_ENCODE_HYBRID_HT_PACKET_PLAN,
    SIGNPOST_ENCODE_HYBRID_HT_PAYLOAD_COPY_COMMAND_ENCODE,
};

struct HtBatchSubmission {
    tier1: HtTier1Prepared,
    codestream_buffer: Buffer,
    codestream_offsets: Vec<usize>,
    codestream_capacities: Vec<usize>,
    codestream_status_buffer: Buffer,
    packet_status_buffer: Buffer,
    tier1_status_readback: Option<J2kResidentTier1StatusReadback>,
    packet_buffers: HtPacketBuffers,
    stage_stats: J2kResidentEncodeStageStats,
    codestream_payload_copy_dispatched: bool,
}

struct HtPacketBuffers {
    packet_resolution: Buffer,
    packet_subband: Buffer,
    resident_block: Buffer,
    packet_block: Buffer,
    packet_descriptor: Buffer,
    state_block: Buffer,
    packet_payload_copy_job: Buffer,
    header: Buffer,
    scratch: Buffer,
    packet_job: Buffer,
    codestream_job: Buffer,
}

impl HtPacketBuffers {
    fn retain_in(self, retained: &mut Vec<Buffer>, packet_status: &Buffer) {
        retained.push(self.packet_resolution);
        retained.push(self.packet_subband);
        retained.push(self.resident_block);
        retained.push(self.packet_block);
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
fn submit_ht_packet_stages(
    runtime: &MetalRuntime,
    prepared_tiles: &[PreparedLosslessBatchTile],
    packet_capacity_mode: J2kHtPacketOutputCapacityMode,
    profile_stages: bool,
) -> Result<HtBatchSubmission, Error> {
    let split_profile_commands = true;
    let mut packet_block_prep_duration = Duration::ZERO;
    let mut packetization_duration = Duration::ZERO;
    let mut codestream_assembly_duration = Duration::ZERO;
    let mut ht_buffer_allocation_duration = Duration::ZERO;
    let HtTier1Prepared {
        mut command_buffer,
        coefficient_buffer,
        tier1_jobs,
        tier1_job_count,
        tier1_job_buffer,
        tier1_output_buffer,
        tier1_status_buffer,
        tile_tier1_job_bases,
        tier1_output_capacity_total,
        max_tier1_output_capacity,
        mut recyclable_private_buffers,
        mut recyclable_shared_buffers,
        mut gpu_stage_command_buffers,
        mut ht_table_build_duration,
        ht_block_encode_duration,
    } = prepare_ht_tier1(runtime, prepared_tiles, profile_stages)?;
    let mut ht_table_build_started = profile_stages.then(Instant::now);
    let ht_packet_plan_signpost = hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_HT_PACKET_PLAN);
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
            family_name: "HTJ2K",
            block_coding_mode: 1,
            high_throughput: 1,
            code_block_style: 0x40,
        },
        |_tile_index, tile, header_capacity| {
            ht_packet_output_capacity_for_mode(
                tile.code_blocks.len(),
                header_capacity,
                tile.packet_descriptors.len().max(tile.resolutions.len()),
                tile.codestream,
                packet_capacity_mode,
            )
        },
    )?;

    drop(ht_packet_plan_signpost);
    if let Some(started) = ht_table_build_started.take() {
        ht_table_build_duration = ht_table_build_duration.saturating_add(started.elapsed());
    }
    let ht_buffer_allocation_started = profile_stages.then(Instant::now);
    let ht_packet_buffer_setup_signpost =
        hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_HT_PACKET_BUFFER_SETUP);
    let packet_resolution_buffer = copied_recyclable_shared_slice_buffer(
        runtime,
        &packet_resolutions,
        &mut recyclable_shared_buffers,
    )?;
    let packet_subband_buffer = copied_recyclable_shared_slice_buffer(
        runtime,
        &packet_subbands,
        &mut recyclable_shared_buffers,
    )?;
    let resident_block_buffer = copied_recyclable_shared_slice_buffer(
        runtime,
        &resident_blocks,
        &mut recyclable_shared_buffers,
    )?;
    let packet_block_buffer = take_recyclable_private_buffer(
        runtime,
        resident_blocks.len().max(1) * size_of::<J2kPacketBlock>(),
        &mut recyclable_private_buffers,
    )?;
    let packet_descriptor_buffer = copied_recyclable_shared_slice_buffer(
        runtime,
        &packet_descriptors,
        &mut recyclable_shared_buffers,
    )?;
    let state_block_buffer = copied_recyclable_shared_slice_buffer(
        runtime,
        &state_blocks,
        &mut recyclable_shared_buffers,
    )?;
    let packet_payload_copy_job_buffer = take_recyclable_private_buffer(
        runtime,
        packet_payload_copy_job_capacity_total
            .max(1)
            .checked_mul(size_of::<J2kPacketPayloadCopyJob>())
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Metal batch packet payload-copy buffer size overflow".to_string(),
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
    let packet_job_buffer = copied_recyclable_shared_slice_buffer(
        runtime,
        &packet_jobs,
        &mut recyclable_shared_buffers,
    )?;
    let packet_status_buffer = zeroed_recyclable_shared_buffer(
        runtime,
        packet_jobs.len().max(1) * size_of::<J2kPacketEncodeStatus>(),
        &mut recyclable_shared_buffers,
    )?;
    let codestream_job_buffer = copied_recyclable_shared_slice_buffer(
        runtime,
        &assembly_jobs,
        &mut recyclable_shared_buffers,
    )?;
    let codestream_buffer = runtime.device.new_buffer(
        codestream_capacity_total.max(1) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let codestream_status_buffer = zeroed_recyclable_shared_buffer(
        runtime,
        assembly_jobs.len() * size_of::<J2kCodestreamAssemblyStatus>(),
        &mut recyclable_shared_buffers,
    )?;
    drop(ht_packet_buffer_setup_signpost);
    if let Some(started) = ht_buffer_allocation_started {
        ht_buffer_allocation_duration = started.elapsed();
    }

    let resident_block_params = J2kResidentPacketBlockParams {
        block_count: u32::try_from(resident_blocks.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K Metal batch resident block count exceeds u32".to_string(),
        })?,
        tier1_job_count,
    };

    let tile_count = u64::try_from(packet_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "HTJ2K Metal batch tile count exceeds u64".to_string(),
    })?;
    if !resident_blocks.is_empty() {
        let command_encode_started = profile_stages.then(Instant::now);
        let signpost =
            hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_HT_PACKET_BLOCK_PREP_COMMAND_ENCODE);
        let encoder = command_buffer.new_compute_command_encoder();
        label_compute_encoder(encoder, "HTJ2K packet block prep");
        encoder.set_compute_pipeline_state(&runtime.packet_block_prepare_resident_ht);
        encoder.set_buffer(0, Some(&resident_block_buffer), 0);
        encoder.set_buffer(1, Some(&tier1_job_buffer), 0);
        encoder.set_buffer(2, Some(&tier1_status_buffer), 0);
        encoder.set_buffer(3, Some(&packet_block_buffer), 0);
        encoder.set_bytes(
            4,
            size_of::<J2kResidentPacketBlockParams>() as u64,
            (&raw const resident_block_params).cast(),
        );
        encoder.dispatch_threads(
            MTLSize {
                width: resident_blocks.len() as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: runtime
                    .packet_block_prepare_resident_ht
                    .thread_execution_width()
                    .max(1),
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        drop(signpost);
        if let Some(started) = command_encode_started {
            packet_block_prep_duration =
                packet_block_prep_duration.saturating_add(started.elapsed());
        }
        if split_profile_commands {
            command_buffer = finish_resident_encode_split_command_buffer(
                command_buffer,
                runtime,
                J2kResidentEncodeGpuStage::PacketBlockPrep,
                "j2k htj2k resident packetization",
                &mut gpu_stage_command_buffers,
            );
        }
    } else if split_profile_commands {
        label_command_buffer(&command_buffer, "j2k htj2k resident packetization");
    }
    let command_encode_started = profile_stages.then(Instant::now);
    let signpost = hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_HT_PACKETIZATION_COMMAND_ENCODE);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "HTJ2K packetization");
    encoder.set_compute_pipeline_state(&runtime.packet_encode_batched);
    encoder.set_buffer(0, Some(&packet_resolution_buffer), 0);
    encoder.set_buffer(1, Some(&packet_subband_buffer), 0);
    encoder.set_buffer(2, Some(&packet_block_buffer), 0);
    encoder.set_buffer(3, Some(&tier1_output_buffer), 0);
    encoder.set_buffer(4, Some(&codestream_buffer), 0);
    encoder.set_buffer(5, Some(&header_buffer), 0);
    encoder.set_buffer(6, Some(&scratch_buffer), 0);
    encoder.set_buffer(7, Some(&packet_job_buffer), 0);
    encoder.set_buffer(8, Some(&packet_status_buffer), 0);
    encoder.set_buffer(9, Some(&packet_descriptor_buffer), 0);
    encoder.set_buffer(10, Some(&state_block_buffer), 0);
    encoder.set_buffer(11, Some(&packet_payload_copy_job_buffer), 0);
    encoder.dispatch_threads(
        MTLSize {
            width: tile_count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: runtime
                .packet_encode_batched
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
    if split_profile_commands {
        command_buffer = finish_resident_encode_split_command_buffer(
            command_buffer,
            runtime,
            J2kResidentEncodeGpuStage::Packetization,
            "j2k htj2k resident packet payload copy",
            &mut gpu_stage_command_buffers,
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
            label: "HTJ2K packetization payload copy",
            signpost_name: SIGNPOST_ENCODE_HYBRID_HT_PAYLOAD_COPY_COMMAND_ENCODE,
        },
    );
    if split_profile_commands {
        if packet_payload_copy_dispatched {
            command_buffer = finish_resident_encode_split_command_buffer(
                command_buffer,
                runtime,
                J2kResidentEncodeGpuStage::PacketPayloadCopy,
                "j2k htj2k resident codestream assembly",
                &mut gpu_stage_command_buffers,
            );
        } else {
            label_command_buffer(&command_buffer, "j2k htj2k resident codestream assembly");
        }
    }

    let command_encode_started = profile_stages.then(Instant::now);
    let signpost =
        hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_HT_CODESTREAM_ASSEMBLY_COMMAND_ENCODE);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "HTJ2K codestream assembly");
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
    let max_packet_output_capacity = packet_jobs
        .iter()
        .map(|job| job.output_capacity)
        .max()
        .unwrap_or(0);
    let max_packet_output_capacity =
        usize::try_from(max_packet_output_capacity).map_err(|_| Error::MetalKernel {
            message: "HTJ2K Metal batch max packet output capacity exceeds usize".to_string(),
        })?;
    if split_profile_commands {
        command_buffer = finish_resident_encode_split_command_buffer(
            command_buffer,
            runtime,
            J2kResidentEncodeGpuStage::CodestreamAssembly,
            "j2k htj2k resident result readback",
            &mut gpu_stage_command_buffers,
        );
    }
    let codestream_payload_copy_dispatched = false;
    if let Some(started) = command_encode_started {
        codestream_assembly_duration =
            codestream_assembly_duration.saturating_add(started.elapsed());
    }
    let tier1_status_readback = schedule_resident_tier1_status_readback(
        ResidentTier1StatusReadbackRequest::high_throughput(
            runtime,
            &command_buffer,
            &tier1_status_buffer,
            tier1_jobs.len(),
            profile_stages,
        ),
    )?;
    command_buffer.commit();
    if split_profile_commands && codestream_payload_copy_dispatched {
        gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
            stage: J2kResidentEncodeGpuStage::CodestreamPayloadCopy,
            command_buffer: command_buffer.clone(),
        });
    }

    let packet_job_count = packet_jobs.len();
    let packet_buffers = HtPacketBuffers {
        packet_resolution: packet_resolution_buffer,
        packet_subband: packet_subband_buffer,
        resident_block: resident_block_buffer,
        packet_block: packet_block_buffer,
        packet_descriptor: packet_descriptor_buffer,
        state_block: state_block_buffer,
        packet_payload_copy_job: packet_payload_copy_job_buffer,
        header: header_buffer,
        scratch: scratch_buffer,
        packet_job: packet_job_buffer,
        codestream_job: codestream_job_buffer,
    };

    let stage_stats = J2kResidentEncodeStageStats {
        ht_buffer_allocation_duration,
        ht_table_build_duration,
        ht_block_encode_duration,
        packet_block_prep_duration,
        packetization_duration,
        codestream_assembly_duration,
        ht_command_encode_duration: ht_block_encode_duration
            .saturating_add(packet_block_prep_duration)
            .saturating_add(packetization_duration)
            .saturating_add(codestream_assembly_duration),
        packet_payload_copy_job_capacity_total,
        max_packet_payload_copy_jobs_per_tile: max_payload_copy_jobs_per_tile,
        packet_payload_copy_launched_stripe_count_total: packet_job_count
            .saturating_mul(max_payload_copy_jobs_per_tile)
            .saturating_mul(PACKET_PAYLOAD_COPY_STRIPES_PER_JOB as usize),
        tier1_output_capacity_total,
        max_tier1_output_capacity,
        packet_output_capacity_total,
        max_packet_output_capacity,
        codestream_payload_copy_launched_thread_count_total: 0,
        code_block_count: tier1_jobs.len(),
        ..J2kResidentEncodeStageStats::default()
    };

    Ok(HtBatchSubmission {
        tier1: HtTier1Prepared {
            command_buffer,
            coefficient_buffer,
            tier1_jobs,
            tier1_job_count,
            tier1_job_buffer,
            tier1_output_buffer,
            tier1_status_buffer,
            tile_tier1_job_bases,
            tier1_output_capacity_total,
            max_tier1_output_capacity,
            recyclable_private_buffers,
            recyclable_shared_buffers,
            gpu_stage_command_buffers,
            ht_table_build_duration,
            ht_block_encode_duration,
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
fn finish_ht_batch(
    session: &crate::MetalBackendSession,
    prepared_tiles: Vec<PreparedLosslessBatchTile>,
    profile_stages: bool,
    submitted: HtBatchSubmission,
) -> Result<J2kPendingResidentLosslessCodestreamBatch, Error> {
    let HtBatchSubmission {
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
    let HtTier1Prepared {
        command_buffer,
        coefficient_buffer,
        tier1_jobs: _,
        tier1_job_count: _,
        tier1_job_buffer,
        tier1_output_buffer,
        tier1_status_buffer,
        tile_tier1_job_bases: _,
        tier1_output_capacity_total: _,
        max_tier1_output_capacity: _,
        mut recyclable_private_buffers,
        recyclable_shared_buffers,
        mut gpu_stage_command_buffers,
        ht_table_build_duration: _,
        ht_block_encode_duration: _,
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
    packet_buffers.retain_in(&mut retained_buffers, &packet_status_buffer);

    Ok(J2kPendingResidentLosslessCodestreamBatch {
        runtime: session.runtime()?,
        buffer: codestream_buffer,
        byte_offsets: codestream_offsets,
        capacities: codestream_capacities,
        status_buffer: codestream_status_buffer,
        packet_status_buffer,
        tier1_status_readback,
        classic_tier1_density_readback: None,
        classic_tier1_symbol_plan_readback: None,
        classic_tier1_pass_plan_readback: None,
        classic_tier1_token_emit_readback: None,
        classic_tier1_split_token_emit_readback: None,
        classic_gpu_token_pack_used: false,
        command_buffer,
        retained_command_buffers,
        _retained_buffers: retained_buffers,
        recyclable_private_buffers,
        recyclable_shared_buffers,
        gpu_stage_command_buffers,
        stage_stats,
        codestream_payload_copy_dispatched,
        status_stage: "HTJ2K batched codestream assembly",
        length_error: "HTJ2K Metal batched codestream output length exceeds usize",
        capacity_error: "HTJ2K Metal batched codestream output length exceeds buffer",
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn submit_lossless_codestream_buffers_from_prepared_ht_batch(
    session: &crate::MetalBackendSession,
    items: Vec<J2kResidentBatchEncodeItem>,
    packet_capacity_mode: J2kHtPacketOutputCapacityMode,
) -> Result<J2kPendingResidentLosslessCodestreamBatch, Error> {
    if items.is_empty() {
        return Err(Error::MetalKernel {
            message: "HTJ2K Metal resident batch encode requires at least one tile".to_string(),
        });
    }

    let prepared_tiles = prepared_lossless_batch_tiles(items);
    with_runtime_for_session(session, |runtime| {
        let profile_stages = metal_profile_stages_enabled();
        let submitted = submit_ht_packet_stages(
            runtime,
            &prepared_tiles,
            packet_capacity_mode,
            profile_stages,
        )?;
        finish_ht_batch(session, prepared_tiles, profile_stages, submitted)
    })
}
