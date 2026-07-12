// SPDX-License-Identifier: MIT OR Apache-2.0

use metal::{Buffer, CommandBuffer};

use super::super::resident_packet_plan::PreparedLosslessBatchTile;
use super::{ForeignType, J2kResidentEncodeGpuStage, J2kResidentEncodeGpuStageCommandBuffer};

fn prepared_command_buffer_count(tile: &PreparedLosslessBatchTile) -> Result<usize, crate::Error> {
    crate::batch_allocation::checked_count_sum(
        [
            1,
            usize::from(tile.prepare_deinterleave_rct_command_buffer.is_some()),
            usize::from(tile.prepare_dwt53_command_buffer.is_some()),
            tile.prepare_dwt53_vertical_command_buffers.len(),
            tile.prepare_dwt53_horizontal_command_buffers.len(),
            usize::from(tile.prepare_coefficient_extract_command_buffer.is_some()),
        ],
        "J2K Metal resident command-buffer retention",
    )
    .map_err(crate::Error::from)
}

fn record_profiled_prepare_command_buffers(
    prepared_tiles: &[PreparedLosslessBatchTile],
    gpu_stage_command_buffers: &mut Vec<J2kResidentEncodeGpuStageCommandBuffer>,
) -> Result<(), crate::Error> {
    let prepare_count = prepared_tiles.iter().try_fold(0usize, |total, tile| {
        crate::batch_allocation::checked_count_sum(
            [total, prepared_command_buffer_count(tile)?],
            "J2K Metal resident profiled command buffers",
        )
        .map_err(crate::Error::from)
    })?;
    let target = crate::batch_allocation::checked_count_sum(
        [gpu_stage_command_buffers.len(), prepare_count],
        "J2K Metal resident profiled command buffers",
    )?;
    crate::batch_allocation::try_reserve_to(
        gpu_stage_command_buffers,
        target,
        "J2K Metal resident profiled command buffers",
    )?;
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K Metal resident profile command-buffer deduplication",
    );
    budget.account_capacity::<J2kResidentEncodeGpuStageCommandBuffer>(
        gpu_stage_command_buffers.capacity(),
    )?;
    let mut seen = budget.try_vec(prepare_count, "J2K Metal resident command-buffer pointers")?;
    for tile in prepared_tiles {
        let mut pushed_split_prepare = false;
        for (stage, command_buffer) in [
            (
                J2kResidentEncodeGpuStage::CoefficientDeinterleaveRct,
                tile.prepare_deinterleave_rct_command_buffer.as_ref(),
            ),
            (
                J2kResidentEncodeGpuStage::CoefficientDwt53,
                tile.prepare_dwt53_command_buffer.as_ref(),
            ),
            (
                J2kResidentEncodeGpuStage::CoefficientExtract,
                tile.prepare_coefficient_extract_command_buffer.as_ref(),
            ),
        ] {
            if let Some(command_buffer) = command_buffer {
                let ptr = command_buffer.as_ptr();
                if seen.contains(&ptr) {
                    continue;
                }
                seen.push(ptr);
                pushed_split_prepare = true;
                gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
                    stage,
                    command_buffer: command_buffer.clone(),
                });
            }
        }
        for (stage, command_buffers) in [
            (
                J2kResidentEncodeGpuStage::CoefficientDwt53Vertical,
                &tile.prepare_dwt53_vertical_command_buffers,
            ),
            (
                J2kResidentEncodeGpuStage::CoefficientDwt53Horizontal,
                &tile.prepare_dwt53_horizontal_command_buffers,
            ),
        ] {
            for command_buffer in command_buffers {
                let ptr = command_buffer.as_ptr();
                if seen.contains(&ptr) {
                    continue;
                }
                seen.push(ptr);
                pushed_split_prepare = true;
                gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
                    stage,
                    command_buffer: command_buffer.clone(),
                });
            }
        }
        if !pushed_split_prepare {
            let command_buffer = &tile.prepare_command_buffer;
            let ptr = command_buffer.as_ptr();
            if !seen.contains(&ptr) {
                seen.push(ptr);
                gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
                    stage: J2kResidentEncodeGpuStage::CoefficientPrep,
                    command_buffer: command_buffer.clone(),
                });
            }
        }
    }
    Ok(())
}

pub(super) fn collect_prepared_batch_retention(
    profile_stages: bool,
    prepared_tiles: Vec<PreparedLosslessBatchTile>,
    gpu_stage_command_buffers: &mut Vec<J2kResidentEncodeGpuStageCommandBuffer>,
    recyclable_private_buffers: &mut Vec<(usize, Buffer)>,
) -> Result<(Vec<CommandBuffer>, Vec<Buffer>), crate::Error> {
    if profile_stages {
        record_profiled_prepare_command_buffers(&prepared_tiles, gpu_stage_command_buffers)?;
    }

    let retained_command_buffer_count =
        prepared_tiles
            .iter()
            .try_fold(gpu_stage_command_buffers.len(), |total, tile| {
                crate::batch_allocation::checked_count_sum(
                    [total, prepared_command_buffer_count(tile)?],
                    "J2K Metal resident command-buffer retention",
                )
                .map_err(crate::Error::from)
            })?;
    let retained_buffer_count = prepared_tiles.iter().try_fold(0usize, |total, tile| {
        total
            .checked_add(3)
            .and_then(|count| count.checked_add(tile.plane_buffers.len()))
            .and_then(|count| count.checked_add(tile.scratch_buffers.len()))
            .ok_or(j2k_core::BatchInfrastructureError::AllocationTooLarge {
                what: "J2K Metal resident buffer retention",
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            })
    })?;
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K Metal resident command-buffer retention",
    );
    budget.preflight(&[
        crate::batch_allocation::BatchMetadataRequest::of::<CommandBuffer>(
            retained_command_buffer_count,
        ),
        crate::batch_allocation::BatchMetadataRequest::of::<Buffer>(retained_buffer_count),
    ])?;
    let mut retained_command_buffers = budget.try_vec(
        retained_command_buffer_count,
        "J2K Metal retained resident command buffers",
    )?;
    retained_command_buffers.extend(
        gpu_stage_command_buffers
            .iter()
            .map(|stage_command_buffer| stage_command_buffer.command_buffer.clone()),
    );
    let mut retained_buffers =
        budget.try_vec(retained_buffer_count, "J2K Metal retained resident buffers")?;
    for tile in prepared_tiles {
        if let Some(command_buffer) = tile.prepare_deinterleave_rct_command_buffer {
            retained_command_buffers.push(command_buffer);
        }
        if let Some(command_buffer) = tile.prepare_dwt53_command_buffer {
            retained_command_buffers.push(command_buffer);
        }
        retained_command_buffers.extend(tile.prepare_dwt53_vertical_command_buffers);
        retained_command_buffers.extend(tile.prepare_dwt53_horizontal_command_buffers);
        if let Some(command_buffer) = tile.prepare_coefficient_extract_command_buffer {
            retained_command_buffers.push(command_buffer);
        }
        retained_command_buffers.push(tile.prepare_command_buffer);
        retained_buffers.push(tile.coefficient_buffer);
        retained_buffers.push(tile.deinterleave_status_buffer);
        retained_buffers.extend(tile.plane_buffers);
        retained_buffers.extend(tile.scratch_buffers);
        retained_buffers.push(tile.coefficient_job_buffer);
        recyclable_private_buffers.extend(tile.recyclable_private_buffers);
    }
    Ok((retained_command_buffers, retained_buffers))
}
