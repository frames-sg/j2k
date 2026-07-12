// SPDX-License-Identifier: MIT OR Apache-2.0

use metal::CommandBuffer;

#[cfg(test)]
use super::super::test_counters;

use super::super::resident_packet_plan::PreparedLosslessBatchTile;
use super::{
    copied_slice_buffer, dispatch_1d_pipeline, finish_resident_encode_split_command_buffer,
    ht_encode_output_capacity, hybrid_stage_signpost, label_command_buffer, label_compute_encoder,
    metal_profile_stages_enabled, new_blit_command_encoder, new_compute_command_encoder,
    new_resident_encode_command_buffer, size_of, take_recyclable_private_buffer, Buffer, Duration,
    Error, ForeignType, Instant, J2kHtEncodeBatchJob, J2kHtEncodeStatus, J2kResidentEncodeGpuStage,
    J2kResidentEncodeGpuStageCommandBuffer, MetalRuntime,
    SIGNPOST_ENCODE_HYBRID_HT_TIER1_COMMAND_ENCODE, SIGNPOST_ENCODE_HYBRID_HT_TIER1_SETUP,
};

pub(super) struct HtTier1Prepared {
    pub(super) command_buffer: CommandBuffer,
    pub(super) coefficient_buffer: Buffer,
    pub(super) tier1_jobs: Vec<J2kHtEncodeBatchJob>,
    pub(super) tier1_job_count: u32,
    pub(super) tier1_job_buffer: Buffer,
    pub(super) tier1_output_buffer: Buffer,
    pub(super) tier1_status_buffer: Buffer,
    pub(super) tile_tier1_job_bases: Vec<usize>,
    pub(super) tier1_output_capacity_total: usize,
    pub(super) max_tier1_output_capacity: usize,
    pub(super) recyclable_private_buffers: Vec<(usize, Buffer)>,
    pub(super) recyclable_shared_buffers: Vec<(usize, Buffer)>,
    pub(super) gpu_stage_command_buffers: Vec<J2kResidentEncodeGpuStageCommandBuffer>,
    pub(super) ht_table_build_duration: Duration,
    pub(super) ht_block_encode_duration: Duration,
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "HT Tier-1 planning keeps capacities, buffers, and jobs synchronized"
)]
pub(super) fn prepare_ht_tier1(
    runtime: &MetalRuntime,
    prepared_tiles: &[PreparedLosslessBatchTile],
    profile_stages: bool,
) -> Result<HtTier1Prepared, Error> {
    let mut ht_table_build_duration = Duration::ZERO;
    let mut ht_block_encode_duration = Duration::ZERO;
    let mut ht_table_build_started = profile_stages.then(Instant::now);
    let ht_tier1_setup_signpost = hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_HT_TIER1_SETUP);
    let split_profile_commands = true;
    let mut gpu_stage_command_buffers = Vec::new();
    let mut recyclable_private_buffers = Vec::<(usize, Buffer)>::new();
    let recyclable_shared_buffers = Vec::<(usize, Buffer)>::new();
    let shared_coefficient_buffer = prepared_tiles.first().and_then(|first| {
        let ptr = first.coefficient_buffer.as_ptr();
        prepared_tiles
            .iter()
            .all(|tile| {
                tile.coefficient_buffer_is_batch_shared && tile.coefficient_buffer.as_ptr() == ptr
            })
            .then(|| first.coefficient_buffer.clone())
    });
    let needs_coefficient_copy = shared_coefficient_buffer.is_none();
    let initial_command_buffer_label = if split_profile_commands && needs_coefficient_copy {
        "j2k htj2k resident coefficient copy"
    } else if split_profile_commands {
        "j2k htj2k resident tier1 encode"
    } else {
        "j2k htj2k resident encode batch"
    };
    let mut command_buffer =
        new_resident_encode_command_buffer(runtime, initial_command_buffer_label)?;
    let (coefficient_buffer, coefficient_offsets) =
        if let Some(coefficient_buffer) = shared_coefficient_buffer {
            let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
                "J2K Metal resident HT coefficient offsets",
            );
            let mut coefficient_offsets = budget.try_vec(
                prepared_tiles.len(),
                "J2K Metal resident HT coefficient offsets",
            )?;
            coefficient_offsets.extend(
                prepared_tiles
                    .iter()
                    .map(|tile| tile.coefficient_byte_offset),
            );
            (coefficient_buffer, coefficient_offsets)
        } else {
            let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
                "J2K Metal resident HT coefficient offsets",
            );
            let mut coefficient_offsets = budget.try_vec(
                prepared_tiles.len(),
                "J2K Metal resident HT coefficient offsets",
            )?;
            let mut total_coefficient_bytes = 0usize;
            for tile in prepared_tiles {
                coefficient_offsets.push(total_coefficient_bytes);
                total_coefficient_bytes = total_coefficient_bytes
                    .checked_add(tile.coefficient_byte_len)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K Metal batch coefficient buffer size overflow".to_string(),
                    })?;
            }
            let coefficient_buffer = take_recyclable_private_buffer(
                runtime,
                total_coefficient_bytes.max(1),
                &mut recyclable_private_buffers,
            )?;
            let blit = new_blit_command_encoder(&command_buffer)?;
            if metal_profile_stages_enabled() {
                blit.set_label("HTJ2K coefficient prep");
            }
            for (tile, &dst_offset) in prepared_tiles.iter().zip(coefficient_offsets.iter()) {
                if tile.coefficient_byte_len > 0 {
                    #[cfg(test)]
                    test_counters::record_ht_batch_coefficient_copy_blit();
                    blit.copy_from_buffer(
                        &tile.coefficient_buffer,
                        tile.coefficient_byte_offset as u64,
                        &coefficient_buffer,
                        dst_offset as u64,
                        tile.coefficient_byte_len as u64,
                    );
                }
            }
            blit.end_encoding();
            if split_profile_commands {
                command_buffer = finish_resident_encode_split_command_buffer(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::CoefficientCopy,
                    "j2k htj2k resident tier1 encode",
                    &mut gpu_stage_command_buffers,
                )?;
            }
            (coefficient_buffer, coefficient_offsets)
        };

    let tier1_job_count = crate::batch_allocation::checked_count_sum(
        prepared_tiles.iter().map(|tile| tile.code_blocks.len()),
        "J2K Metal resident HT Tier-1 jobs",
    )?;
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal resident HT Tier-1 job plan");
    budget.preflight(&[
        crate::batch_allocation::BatchMetadataRequest::of::<J2kHtEncodeBatchJob>(tier1_job_count),
        crate::batch_allocation::BatchMetadataRequest::of::<usize>(prepared_tiles.len()),
    ])?;
    let mut tier1_jobs = budget.try_vec(tier1_job_count, "J2K Metal resident HT Tier-1 jobs")?;
    let mut tier1_output_capacity_total = 0usize;
    let mut max_tier1_output_capacity = 0usize;
    let mut tile_tier1_job_bases =
        budget.try_vec(prepared_tiles.len(), "J2K Metal resident HT tile job bases")?;
    for (tile, &coefficient_byte_offset) in prepared_tiles.iter().zip(coefficient_offsets.iter()) {
        tile_tier1_job_bases.push(tier1_jobs.len());
        let coefficient_word_offset = coefficient_byte_offset
            .checked_div(size_of::<i32>())
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Metal batch coefficient offset division failed".to_string(),
            })?;
        let coefficient_word_offset_u32 =
            u32::try_from(coefficient_word_offset).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal batch coefficient offset exceeds u32".to_string(),
            })?;
        for block in &tile.code_blocks {
            let output_capacity_per_job = ht_encode_output_capacity(block.width, block.height)?;
            max_tier1_output_capacity = max_tier1_output_capacity.max(output_capacity_per_job);
            let output_capacity_per_job_u32 =
                u32::try_from(output_capacity_per_job).map_err(|_| Error::MetalKernel {
                    message: "HTJ2K Metal batch output capacity exceeds u32".to_string(),
                })?;
            let output_offset =
                u32::try_from(tier1_output_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "HTJ2K Metal batch Tier-1 output offset exceeds u32".to_string(),
                })?;
            let coefficient_offset = block
                .coefficient_offset
                .checked_add(coefficient_word_offset_u32)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal batch coefficient offset overflow".to_string(),
                })?;
            tier1_jobs.push(J2kHtEncodeBatchJob {
                coefficient_offset,
                output_offset,
                width: block.width,
                height: block.height,
                total_bitplanes: u32::from(block.total_bitplanes),
                output_capacity: output_capacity_per_job_u32,
            });
            tier1_output_capacity_total = tier1_output_capacity_total
                .checked_add(output_capacity_per_job)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal batch Tier-1 output buffer overflow".to_string(),
                })?;
        }
    }

    let tier1_job_buffer = copied_slice_buffer(&runtime.device, &tier1_jobs)?;
    let tier1_output_buffer = take_recyclable_private_buffer(
        runtime,
        tier1_output_capacity_total.max(1),
        &mut recyclable_private_buffers,
    )?;
    let tier1_status_buffer = take_recyclable_private_buffer(
        runtime,
        tier1_jobs.len().max(1) * size_of::<J2kHtEncodeStatus>(),
        &mut recyclable_private_buffers,
    )?;
    let tier1_job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "HTJ2K Metal batch Tier-1 job count exceeds u32".to_string(),
    })?;
    drop(ht_tier1_setup_signpost);
    if let Some(started) = ht_table_build_started.take() {
        ht_table_build_duration = ht_table_build_duration.saturating_add(started.elapsed());
    }
    if tier1_job_count > 0 {
        let command_encode_started = profile_stages.then(Instant::now);
        let signpost = hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_HT_TIER1_COMMAND_ENCODE);
        let encoder = new_compute_command_encoder(&command_buffer)?;
        label_compute_encoder(&encoder, "HTJ2K Tier-1 encode");
        let pipeline = &runtime.ht_encode_code_blocks;
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&coefficient_buffer), 0);
        encoder.set_buffer(1, Some(&tier1_output_buffer), 0);
        encoder.set_buffer(2, Some(&tier1_job_buffer), 0);
        encoder.set_buffer(3, Some(&runtime.ht_vlc_encode_table0), 0);
        encoder.set_buffer(4, Some(&runtime.ht_vlc_encode_table1), 0);
        encoder.set_buffer(5, Some(&runtime.ht_uvlc_encode_table), 0);
        encoder.set_buffer(6, Some(&tier1_status_buffer), 0);
        encoder.set_bytes(
            7,
            size_of::<u32>() as u64,
            (&raw const tier1_job_count).cast(),
        );
        dispatch_1d_pipeline(&encoder, pipeline, u64::from(tier1_job_count));
        encoder.end_encoding();
        drop(signpost);
        if let Some(started) = command_encode_started {
            ht_block_encode_duration = ht_block_encode_duration.saturating_add(started.elapsed());
        }
        if split_profile_commands {
            command_buffer = finish_resident_encode_split_command_buffer(
                command_buffer,
                runtime,
                J2kResidentEncodeGpuStage::HtBlock,
                "j2k htj2k resident packetization",
                &mut gpu_stage_command_buffers,
            )?;
        }
    } else if split_profile_commands {
        label_command_buffer(&command_buffer, "j2k htj2k resident packetization");
    }

    Ok(HtTier1Prepared {
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
    })
}
