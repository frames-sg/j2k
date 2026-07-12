// SPDX-License-Identifier: MIT OR Apache-2.0

use metal::CommandBuffer;

use super::super::resident_packet_plan::PreparedLosslessBatchTile;
use super::super::resident_tier1::{
    J2kResidentClassicTier1DensityReadback, J2kResidentClassicTier1PassPlanReadback,
    J2kResidentClassicTier1SplitTokenBuffers, J2kResidentClassicTier1SymbolPlanReadback,
    J2kResidentClassicTier1TokenEmitReadback,
};
use super::{
    classic_encode_code_blocks_pipeline, classic_encode_output_capacity_for_mode,
    classic_encode_segment_capacity, classic_encode_sub_band_code, classic_profile_stages_from_env,
    classic_resident_style_flags_from_env, classic_tier1_gpu_token_pack_requested,
    classic_tier1_gpu_token_pack_supported, classic_tier1_split_gpu_token_pack_requested,
    classic_tier1_split_mq_byte_gpu_token_pack_disabled,
    classic_tier1_split_mq_byte_gpu_token_pack_requested, copied_slice_buffer,
    dispatch_1d_pipeline, dispatch_classic_tier1_profiles,
    dispatch_classic_tier1_split_token_emit_for_gpu_pack,
    dispatch_classic_tier1_split_token_pack_from_gpu_tokens,
    dispatch_classic_tier1_token_emit_for_gpu_pack,
    dispatch_classic_tier1_token_pack_from_gpu_tokens,
    finish_resident_encode_split_command_buffer_timed, hybrid_stage_signpost, label_command_buffer,
    label_compute_encoder, new_blit_command_encoder, new_compute_command_encoder,
    new_private_buffer, new_resident_encode_command_buffer,
    schedule_classic_tier1_gpu_token_pack_readback, size_of, take_recyclable_private_buffer,
    Buffer, ClassicTier1ProfileRequest, ClassicTier1ProfileResult, Duration, Error, ForeignType,
    Instant, J2kClassicEncodeBatchJob, J2kClassicEncodeOutputCapacityMode, J2kClassicEncodeStatus,
    J2kClassicSegment, J2kResidentEncodeGpuStage, J2kResidentEncodeGpuStageCommandBuffer,
    MetalRuntime, SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_COMMAND_ENCODE,
    SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_SETUP,
};

pub(super) struct ClassicTier1Prepared {
    pub(super) command_buffer: CommandBuffer,
    pub(super) coefficient_buffer: Buffer,
    pub(super) tier1_jobs: Vec<J2kClassicEncodeBatchJob>,
    pub(super) tier1_job_count: u32,
    pub(super) tier1_job_buffer: Buffer,
    pub(super) tier1_output_buffer: Buffer,
    pub(super) tier1_status_buffer: Buffer,
    pub(super) tier1_segment_buffer: Buffer,
    pub(super) tile_tier1_job_bases: Vec<usize>,
    pub(super) tile_tier1_output_capacities: Vec<usize>,
    pub(super) tier1_output_capacity_total: usize,
    pub(super) max_tier1_output_capacity: usize,
    pub(super) tier1_segment_capacity_total: usize,
    pub(super) recyclable_private_buffers: Vec<crate::buffer_pool::PooledBuffer>,
    pub(super) recyclable_shared_buffers: Vec<crate::buffer_pool::PooledBuffer>,
    pub(super) gpu_stage_command_buffers: Vec<J2kResidentEncodeGpuStageCommandBuffer>,
    pub(super) classic_tier1_density_readback: Option<J2kResidentClassicTier1DensityReadback>,
    pub(super) classic_tier1_raw_pack_buffer: Option<Buffer>,
    pub(super) classic_tier1_arithmetic_pack_buffer: Option<Buffer>,
    pub(super) classic_tier1_symbol_plan_readback:
        Option<J2kResidentClassicTier1SymbolPlanReadback>,
    pub(super) classic_tier1_pass_plan_readback: Option<J2kResidentClassicTier1PassPlanReadback>,
    pub(super) classic_tier1_token_emit_readback: Option<J2kResidentClassicTier1TokenEmitReadback>,
    pub(super) classic_tier1_split_token_emit_readback:
        Option<J2kResidentClassicTier1SplitTokenBuffers>,
    pub(super) classic_gpu_token_pack_used: bool,
    pub(super) classic_resident_style_flags: u32,
    pub(super) classic_tier1_setup_duration: Duration,
    pub(super) classic_block_encode_duration: Duration,
    pub(super) classic_command_buffer_commit_duration: Duration,
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "classic Tier-1 planning keeps capacities, buffers, and jobs synchronized"
)]
pub(super) fn prepare_classic_tier1(
    runtime: &MetalRuntime,
    prepared_tiles: &[PreparedLosslessBatchTile],
    output_capacity_mode: J2kClassicEncodeOutputCapacityMode,
    profile_stages: bool,
) -> Result<ClassicTier1Prepared, Error> {
    let mut classic_tier1_setup_duration = Duration::ZERO;
    let mut classic_block_encode_duration = Duration::ZERO;
    let mut classic_command_buffer_commit_duration = Duration::ZERO;
    let mut gpu_stage_command_buffers = Vec::new();
    let classic_profile_stages = classic_profile_stages_from_env();
    let ClassicCoefficientPreparation {
        command_buffer,
        coefficient_buffer,
        coefficient_offsets,
    } = prepare_classic_coefficients(
        runtime,
        prepared_tiles,
        profile_stages,
        &mut gpu_stage_command_buffers,
        &mut classic_command_buffer_commit_duration,
    )?;
    let classic_tier1_setup_started = profile_stages.then(Instant::now);
    let classic_tier1_setup_signpost =
        hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_SETUP);
    let classic_resident_style_flags = classic_resident_style_flags_from_env();
    let ClassicTier1JobPlan {
        tier1_jobs,
        tile_tier1_job_bases,
        tile_tier1_output_capacities,
        tier1_output_capacity_total,
        max_tier1_output_capacity,
        tier1_segment_capacity_total,
    } = plan_classic_tier1_jobs(
        prepared_tiles,
        &coefficient_offsets,
        output_capacity_mode,
        classic_resident_style_flags,
    )?;
    let ClassicTier1Buffers {
        tier1_job_count,
        tier1_job_buffer,
        tier1_output_buffer,
        tier1_status_buffer,
        tier1_segment_buffer,
        mut recyclable_private_buffers,
        recyclable_shared_buffers,
    } = allocate_classic_tier1_buffers(
        runtime,
        &tier1_jobs,
        tier1_output_capacity_total,
        tier1_segment_capacity_total,
    )?;
    drop(classic_tier1_setup_signpost);
    if let Some(started) = classic_tier1_setup_started {
        classic_tier1_setup_duration = started.elapsed();
    }
    let classic_tier1_route = select_classic_tier1_route(tier1_job_count, &tier1_jobs)?;
    let ClassicTier1DispatchResult {
        command_buffer,
        classic_gpu_token_pack_readback,
    } = dispatch_classic_tier1(ClassicTier1DispatchRequest {
        runtime,
        command_buffer,
        coefficient_buffer: &coefficient_buffer,
        tier1_job_buffer: &tier1_job_buffer,
        tier1_jobs: &tier1_jobs,
        tier1_job_count,
        tier1_output_buffer: &tier1_output_buffer,
        tier1_status_buffer: &tier1_status_buffer,
        tier1_segment_buffer: &tier1_segment_buffer,
        route: classic_tier1_route,
        token_pack_next_label: classic_profile_stages.token_pack_next_label,
        profile_stages,
        recyclable_private_buffers: &mut recyclable_private_buffers,
        gpu_stage_command_buffers: &mut gpu_stage_command_buffers,
        classic_block_encode_duration: &mut classic_block_encode_duration,
        classic_command_buffer_commit_duration: &mut classic_command_buffer_commit_duration,
    })?;

    let ClassicTier1ProfileResult {
        command_buffer,
        classic_tier1_density_readback,
        classic_tier1_raw_pack_buffer,
        classic_tier1_arithmetic_pack_buffer,
        classic_tier1_symbol_plan_readback,
        classic_tier1_pass_plan_readback,
        classic_tier1_token_emit_readback,
        classic_tier1_split_token_emit_readback,
    } = dispatch_classic_tier1_profiles(ClassicTier1ProfileRequest {
        runtime,
        command_buffer,
        coefficient_buffer: &coefficient_buffer,
        tier1_job_buffer: &tier1_job_buffer,
        tier1_jobs: &tier1_jobs,
        tier1_job_count,
        tier1_output_capacity_total,
        classic_gpu_token_pack_readback,
        profile_stages,
        stages: classic_profile_stages,
        gpu_stage_command_buffers: &mut gpu_stage_command_buffers,
        classic_command_buffer_commit_duration: &mut classic_command_buffer_commit_duration,
    })?;

    Ok(ClassicTier1Prepared {
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
        classic_gpu_token_pack_used: classic_tier1_route.use_classic_gpu_token_pack
            || classic_tier1_route.use_classic_split_gpu_token_pack
            || classic_tier1_route.use_classic_split_mq_byte_gpu_token_pack,
        classic_resident_style_flags,
        classic_tier1_setup_duration,
        classic_block_encode_duration,
        classic_command_buffer_commit_duration,
    })
}

struct ClassicCoefficientPreparation {
    command_buffer: CommandBuffer,
    coefficient_buffer: Buffer,
    coefficient_offsets: Vec<usize>,
}

#[cfg(target_os = "macos")]
fn prepare_classic_coefficients(
    runtime: &MetalRuntime,
    prepared_tiles: &[PreparedLosslessBatchTile],
    profile_stages: bool,
    gpu_stage_command_buffers: &mut Vec<J2kResidentEncodeGpuStageCommandBuffer>,
    classic_command_buffer_commit_duration: &mut Duration,
) -> Result<ClassicCoefficientPreparation, Error> {
    let split_command_buffers = true;
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
    let initial_command_buffer_label = if split_command_buffers && needs_coefficient_copy {
        "j2k classic resident coefficient copy"
    } else if split_command_buffers {
        "j2k classic resident Tier-1 encode"
    } else {
        "j2k classic resident encode batch"
    };
    let mut command_buffer =
        new_resident_encode_command_buffer(runtime, initial_command_buffer_label)?;
    let (coefficient_buffer, coefficient_offsets) =
        if let Some(coefficient_buffer) = shared_coefficient_buffer {
            let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
                "J2K Metal resident classic coefficient offsets",
            );
            let mut coefficient_offsets = budget.try_vec(
                prepared_tiles.len(),
                "J2K Metal resident classic coefficient offsets",
            )?;
            coefficient_offsets.extend(
                prepared_tiles
                    .iter()
                    .map(|tile| tile.coefficient_byte_offset),
            );
            (coefficient_buffer, coefficient_offsets)
        } else {
            let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
                "J2K Metal resident classic coefficient offsets",
            );
            let mut coefficient_offsets = budget.try_vec(
                prepared_tiles.len(),
                "J2K Metal resident classic coefficient offsets",
            )?;
            let mut total_coefficient_bytes = 0usize;
            for tile in prepared_tiles {
                coefficient_offsets.push(total_coefficient_bytes);
                total_coefficient_bytes = total_coefficient_bytes
                    .checked_add(tile.coefficient_byte_len)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "J2K Metal batch coefficient buffer size overflow".to_string(),
                    })?;
            }
            let coefficient_buffer =
                new_private_buffer(&runtime.device, total_coefficient_bytes.max(1))?;
            let blit = new_blit_command_encoder(&command_buffer)?;
            if profile_stages {
                blit.set_label("J2K coefficient prep");
            }
            for (tile, &dst_offset) in prepared_tiles.iter().zip(coefficient_offsets.iter()) {
                if tile.coefficient_byte_len > 0 {
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
            if split_command_buffers {
                command_buffer = finish_resident_encode_split_command_buffer_timed(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::CoefficientCopy,
                    "j2k classic resident Tier-1 encode",
                    &mut *gpu_stage_command_buffers,
                    profile_stages,
                    &mut *classic_command_buffer_commit_duration,
                )?;
            }
            (coefficient_buffer, coefficient_offsets)
        };

    Ok(ClassicCoefficientPreparation {
        command_buffer,
        coefficient_buffer,
        coefficient_offsets,
    })
}

struct ClassicTier1JobPlan {
    tier1_jobs: Vec<J2kClassicEncodeBatchJob>,
    tile_tier1_job_bases: Vec<usize>,
    tile_tier1_output_capacities: Vec<usize>,
    tier1_output_capacity_total: usize,
    max_tier1_output_capacity: usize,
    tier1_segment_capacity_total: usize,
}

struct ClassicTier1JobOwners {
    tier1_jobs: Vec<J2kClassicEncodeBatchJob>,
    tile_job_bases: Vec<usize>,
    tile_output_capacities: Vec<usize>,
}

#[cfg(target_os = "macos")]
fn allocate_classic_tier1_job_owners(
    prepared_tiles: &[PreparedLosslessBatchTile],
) -> Result<ClassicTier1JobOwners, Error> {
    let job_count = crate::batch_allocation::checked_count_sum(
        prepared_tiles.iter().map(|tile| tile.code_blocks.len()),
        "J2K Metal resident classic Tier-1 jobs",
    )?;
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K Metal resident classic Tier-1 job plan",
    );
    budget.preflight(&[
        crate::batch_allocation::BatchMetadataRequest::of::<J2kClassicEncodeBatchJob>(job_count),
        crate::batch_allocation::BatchMetadataRequest::of::<usize>(prepared_tiles.len()),
        crate::batch_allocation::BatchMetadataRequest::of::<usize>(prepared_tiles.len()),
    ])?;
    Ok(ClassicTier1JobOwners {
        tier1_jobs: budget.try_vec(job_count, "J2K Metal resident classic Tier-1 jobs")?,
        tile_job_bases: budget.try_vec(
            prepared_tiles.len(),
            "J2K Metal resident classic tile job bases",
        )?,
        tile_output_capacities: budget.try_vec(
            prepared_tiles.len(),
            "J2K Metal resident classic tile output capacities",
        )?,
    })
}

#[cfg(target_os = "macos")]
fn plan_classic_tier1_jobs(
    prepared_tiles: &[PreparedLosslessBatchTile],
    coefficient_offsets: &[usize],
    output_capacity_mode: J2kClassicEncodeOutputCapacityMode,
    classic_resident_style_flags: u32,
) -> Result<ClassicTier1JobPlan, Error> {
    let ClassicTier1JobOwners {
        mut tier1_jobs,
        tile_job_bases: mut tile_tier1_job_bases,
        tile_output_capacities: mut tile_tier1_output_capacities,
    } = allocate_classic_tier1_job_owners(prepared_tiles)?;
    let mut tier1_output_capacity_total = 0usize;
    let mut max_tier1_output_capacity = 0usize;
    let mut tier1_segment_capacity_total = 0usize;
    for (tile, &coefficient_byte_offset) in prepared_tiles.iter().zip(coefficient_offsets.iter()) {
        tile_tier1_job_bases.push(tier1_jobs.len());
        let tile_output_start = tier1_output_capacity_total;
        let coefficient_word_offset = coefficient_byte_offset
            .checked_div(size_of::<i32>())
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal batch coefficient offset division failed".to_string(),
            })?;
        let coefficient_word_offset_u32 =
            u32::try_from(coefficient_word_offset).map_err(|_| Error::MetalKernel {
                message: "J2K Metal batch coefficient offset exceeds u32".to_string(),
            })?;
        for block in &tile.code_blocks {
            let output_capacity = classic_encode_output_capacity_for_mode(
                block.width,
                block.height,
                block.total_bitplanes,
                output_capacity_mode,
            )?;
            max_tier1_output_capacity = max_tier1_output_capacity.max(output_capacity);
            let output_offset =
                u32::try_from(tier1_output_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "J2K Metal batch Tier-1 output offset exceeds u32".to_string(),
                })?;
            let segment_offset =
                u32::try_from(tier1_segment_capacity_total).map_err(|_| Error::MetalKernel {
                    message: "J2K Metal batch Tier-1 segment offset exceeds u32".to_string(),
                })?;
            let coefficient_offset = block
                .coefficient_offset
                .checked_add(coefficient_word_offset_u32)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal batch coefficient offset overflow".to_string(),
                })?;
            let segment_capacity = classic_encode_segment_capacity(
                classic_resident_style_flags,
                block.total_bitplanes,
            );
            tier1_jobs.push(J2kClassicEncodeBatchJob {
                coefficient_offset,
                output_offset,
                segment_offset,
                width: block.width,
                height: block.height,
                sub_band_type: classic_encode_sub_band_code(block.sub_band_type),
                total_bitplanes: u32::from(block.total_bitplanes),
                style_flags: classic_resident_style_flags,
                output_capacity: u32::try_from(output_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal batch Tier-1 output capacity exceeds u32".to_string(),
                    }
                })?,
                segment_capacity: u32::try_from(segment_capacity).map_err(|_| {
                    Error::MetalKernel {
                        message: "J2K Metal batch Tier-1 segment capacity exceeds u32".to_string(),
                    }
                })?,
            });
            tier1_output_capacity_total = tier1_output_capacity_total
                .checked_add(output_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal batch Tier-1 output buffer overflow".to_string(),
                })?;
            tier1_segment_capacity_total = tier1_segment_capacity_total
                .checked_add(segment_capacity)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal batch Tier-1 segment buffer overflow".to_string(),
                })?;
        }
        tile_tier1_output_capacities.push(
            tier1_output_capacity_total
                .checked_sub(tile_output_start)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal batch tile Tier-1 capacity underflow".to_string(),
                })?,
        );
    }

    Ok(ClassicTier1JobPlan {
        tier1_jobs,
        tile_tier1_job_bases,
        tile_tier1_output_capacities,
        tier1_output_capacity_total,
        max_tier1_output_capacity,
        tier1_segment_capacity_total,
    })
}

struct ClassicTier1Buffers {
    tier1_job_count: u32,
    tier1_job_buffer: Buffer,
    tier1_output_buffer: Buffer,
    tier1_status_buffer: Buffer,
    tier1_segment_buffer: Buffer,
    recyclable_private_buffers: Vec<crate::buffer_pool::PooledBuffer>,
    recyclable_shared_buffers: Vec<crate::buffer_pool::PooledBuffer>,
}

#[cfg(target_os = "macos")]
fn allocate_classic_tier1_buffers(
    runtime: &MetalRuntime,
    tier1_jobs: &[J2kClassicEncodeBatchJob],
    tier1_output_capacity_total: usize,
    tier1_segment_capacity_total: usize,
) -> Result<ClassicTier1Buffers, Error> {
    let tier1_job_count = u32::try_from(tier1_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal batch Tier-1 job count exceeds u32".to_string(),
    })?;
    let tier1_job_buffer = copied_slice_buffer(&runtime.device, tier1_jobs)?;
    let mut recyclable_private_buffers = Vec::new();
    let recyclable_shared_buffers = Vec::new();
    let tier1_output_buffer = take_recyclable_private_buffer(
        runtime,
        tier1_output_capacity_total.max(1),
        &mut recyclable_private_buffers,
    )?;
    let tier1_status_buffer = take_recyclable_private_buffer(
        runtime,
        tier1_jobs.len().max(1) * size_of::<J2kClassicEncodeStatus>(),
        &mut recyclable_private_buffers,
    )?;
    let tier1_segment_buffer = take_recyclable_private_buffer(
        runtime,
        tier1_segment_capacity_total.max(1) * size_of::<J2kClassicSegment>(),
        &mut recyclable_private_buffers,
    )?;

    Ok(ClassicTier1Buffers {
        tier1_job_count,
        tier1_job_buffer,
        tier1_output_buffer,
        tier1_status_buffer,
        tier1_segment_buffer,
        recyclable_private_buffers,
        recyclable_shared_buffers,
    })
}

#[derive(Clone, Copy)]
struct ClassicTier1Route {
    use_classic_split_mq_byte_gpu_token_pack: bool,
    use_classic_split_gpu_token_pack: bool,
    use_classic_gpu_token_pack: bool,
}

#[cfg(target_os = "macos")]
fn select_classic_tier1_route(
    tier1_job_count: u32,
    tier1_jobs: &[J2kClassicEncodeBatchJob],
) -> Result<ClassicTier1Route, Error> {
    let classic_split_mq_byte_gpu_token_pack_requested =
        classic_tier1_split_mq_byte_gpu_token_pack_requested();
    let classic_split_mq_byte_gpu_token_pack_disabled =
        classic_tier1_split_mq_byte_gpu_token_pack_disabled();
    let classic_split_gpu_token_pack_requested = classic_tier1_split_gpu_token_pack_requested();
    let classic_gpu_token_pack_requested = classic_tier1_gpu_token_pack_requested();
    let use_classic_split_mq_byte_gpu_token_pack = if tier1_job_count > 0 {
        if classic_split_mq_byte_gpu_token_pack_requested {
            if !classic_tier1_gpu_token_pack_supported(tier1_jobs) {
                return Err(Error::MetalKernel {
                    message: "J2K Metal classic split MQ-byte GPU token-pack route currently supports only bypass_u16_32 resident jobs".to_string(),
                });
            }
            true
        } else {
            !classic_split_mq_byte_gpu_token_pack_disabled
                && !classic_split_gpu_token_pack_requested
                && !classic_gpu_token_pack_requested
                && classic_tier1_gpu_token_pack_supported(tier1_jobs)
        }
    } else {
        false
    };
    let use_classic_split_gpu_token_pack = if classic_split_gpu_token_pack_requested
        && !use_classic_split_mq_byte_gpu_token_pack
        && tier1_job_count > 0
    {
        if !classic_tier1_gpu_token_pack_supported(tier1_jobs) {
            return Err(Error::MetalKernel {
                message: "J2K Metal classic split GPU token-pack route currently supports only bypass_u16_32 resident jobs".to_string(),
            });
        }
        true
    } else {
        false
    };
    let use_classic_gpu_token_pack = if !use_classic_split_mq_byte_gpu_token_pack
        && !use_classic_split_gpu_token_pack
        && classic_gpu_token_pack_requested
        && tier1_job_count > 0
    {
        if !classic_tier1_gpu_token_pack_supported(tier1_jobs) {
            return Err(Error::MetalKernel {
                message: "J2K Metal classic GPU token-pack route currently supports only bypass_u16_32 resident jobs".to_string(),
            });
        }
        true
    } else {
        false
    };

    Ok(ClassicTier1Route {
        use_classic_split_mq_byte_gpu_token_pack,
        use_classic_split_gpu_token_pack,
        use_classic_gpu_token_pack,
    })
}

struct ClassicTier1DispatchRequest<'a> {
    runtime: &'a MetalRuntime,
    command_buffer: CommandBuffer,
    coefficient_buffer: &'a Buffer,
    tier1_job_buffer: &'a Buffer,
    tier1_jobs: &'a [J2kClassicEncodeBatchJob],
    tier1_job_count: u32,
    tier1_output_buffer: &'a Buffer,
    tier1_status_buffer: &'a Buffer,
    tier1_segment_buffer: &'a Buffer,
    route: ClassicTier1Route,
    token_pack_next_label: &'static str,
    profile_stages: bool,
    recyclable_private_buffers: &'a mut Vec<crate::buffer_pool::PooledBuffer>,
    gpu_stage_command_buffers: &'a mut Vec<J2kResidentEncodeGpuStageCommandBuffer>,
    classic_block_encode_duration: &'a mut Duration,
    classic_command_buffer_commit_duration: &'a mut Duration,
}

struct ClassicTier1DispatchResult {
    command_buffer: CommandBuffer,
    classic_gpu_token_pack_readback: Option<J2kResidentClassicTier1TokenEmitReadback>,
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "classic Tier-1 dispatch preserves fixed command and resource order"
)]
fn dispatch_classic_tier1(
    request: ClassicTier1DispatchRequest<'_>,
) -> Result<ClassicTier1DispatchResult, Error> {
    let ClassicTier1DispatchRequest {
        runtime,
        mut command_buffer,
        coefficient_buffer,
        tier1_job_buffer,
        tier1_jobs,
        tier1_job_count,
        tier1_output_buffer,
        tier1_status_buffer,
        tier1_segment_buffer,
        route:
            ClassicTier1Route {
                use_classic_split_mq_byte_gpu_token_pack,
                use_classic_split_gpu_token_pack,
                use_classic_gpu_token_pack,
            },
        token_pack_next_label: classic_token_pack_next_label,
        profile_stages,
        recyclable_private_buffers,
        gpu_stage_command_buffers,
        classic_block_encode_duration,
        classic_command_buffer_commit_duration,
    } = request;
    let split_command_buffers = true;
    let mut classic_gpu_token_pack_readback = None;
    if tier1_job_count > 0 {
        let command_encode_started = profile_stages.then(Instant::now);
        let signpost = hybrid_stage_signpost(SIGNPOST_ENCODE_HYBRID_CLASSIC_TIER1_COMMAND_ENCODE);
        if use_classic_split_mq_byte_gpu_token_pack || use_classic_split_gpu_token_pack {
            let token_buffers = dispatch_classic_tier1_split_token_emit_for_gpu_pack(
                runtime,
                &command_buffer,
                coefficient_buffer,
                tier1_job_buffer,
                tier1_jobs,
                &mut *recyclable_private_buffers,
                use_classic_split_mq_byte_gpu_token_pack,
            )?;
            if split_command_buffers {
                command_buffer = finish_resident_encode_split_command_buffer_timed(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::ClassicTier1SplitTokenEmit,
                    "j2k classic resident Tier-1 split token pack",
                    &mut *gpu_stage_command_buffers,
                    profile_stages,
                    &mut *classic_command_buffer_commit_duration,
                )?;
            }
            dispatch_classic_tier1_split_token_pack_from_gpu_tokens(
                runtime,
                &command_buffer,
                tier1_job_buffer,
                &token_buffers,
                tier1_output_buffer,
                tier1_status_buffer,
                tier1_segment_buffer,
            )?;
            drop(signpost);
            if let Some(started) = command_encode_started {
                *classic_block_encode_duration =
                    classic_block_encode_duration.saturating_add(started.elapsed());
            }
            if split_command_buffers {
                let next_label = classic_token_pack_next_label;
                command_buffer = finish_resident_encode_split_command_buffer_timed(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::ClassicTier1TokenPack,
                    next_label,
                    &mut *gpu_stage_command_buffers,
                    profile_stages,
                    &mut *classic_command_buffer_commit_duration,
                )?;
            }
        } else if use_classic_gpu_token_pack {
            let token_buffers = dispatch_classic_tier1_token_emit_for_gpu_pack(
                runtime,
                &command_buffer,
                coefficient_buffer,
                tier1_job_buffer,
                tier1_jobs,
                &mut *recyclable_private_buffers,
            )?;
            if split_command_buffers {
                command_buffer = finish_resident_encode_split_command_buffer_timed(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::ClassicTier1TokenEmit,
                    "j2k classic resident Tier-1 token pack",
                    &mut *gpu_stage_command_buffers,
                    profile_stages,
                    &mut *classic_command_buffer_commit_duration,
                )?;
            }
            dispatch_classic_tier1_token_pack_from_gpu_tokens(
                runtime,
                &command_buffer,
                tier1_job_buffer,
                &token_buffers,
                tier1_output_buffer,
                tier1_status_buffer,
                tier1_segment_buffer,
            )?;
            classic_gpu_token_pack_readback = schedule_classic_tier1_gpu_token_pack_readback(
                runtime,
                &command_buffer,
                &token_buffers,
                profile_stages,
            )?;
            drop(signpost);
            if let Some(started) = command_encode_started {
                *classic_block_encode_duration =
                    classic_block_encode_duration.saturating_add(started.elapsed());
            }
            if split_command_buffers {
                let next_label = classic_token_pack_next_label;
                command_buffer = finish_resident_encode_split_command_buffer_timed(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::ClassicTier1TokenPack,
                    next_label,
                    &mut *gpu_stage_command_buffers,
                    profile_stages,
                    &mut *classic_command_buffer_commit_duration,
                )?;
            }
        } else {
            let encoder = new_compute_command_encoder(&command_buffer)?;
            label_compute_encoder(&encoder, "J2K Tier-1 encode");
            let classic_encode_pipeline = classic_encode_code_blocks_pipeline(runtime, tier1_jobs);
            encoder.set_compute_pipeline_state(classic_encode_pipeline);
            encoder.set_buffer(0, Some(coefficient_buffer), 0);
            encoder.set_buffer(1, Some(tier1_output_buffer), 0);
            encoder.set_buffer(2, Some(tier1_job_buffer), 0);
            encoder.set_buffer(3, Some(tier1_status_buffer), 0);
            encoder.set_buffer(4, Some(tier1_segment_buffer), 0);
            encoder.set_bytes(
                5,
                size_of::<u32>() as u64,
                (&raw const tier1_job_count).cast(),
            );
            dispatch_1d_pipeline(
                &encoder,
                classic_encode_pipeline,
                u64::from(tier1_job_count),
            );
            encoder.end_encoding();
            drop(signpost);
            if let Some(started) = command_encode_started {
                *classic_block_encode_duration =
                    classic_block_encode_duration.saturating_add(started.elapsed());
            }
            if split_command_buffers {
                let next_label = classic_token_pack_next_label;
                command_buffer = finish_resident_encode_split_command_buffer_timed(
                    command_buffer,
                    runtime,
                    J2kResidentEncodeGpuStage::ClassicBlock,
                    next_label,
                    &mut *gpu_stage_command_buffers,
                    profile_stages,
                    &mut *classic_command_buffer_commit_duration,
                )?;
            }
        }
    } else if split_command_buffers {
        label_command_buffer(&command_buffer, "j2k classic resident packetization");
    }

    Ok(ClassicTier1DispatchResult {
        command_buffer,
        classic_gpu_token_pack_readback,
    })
}
