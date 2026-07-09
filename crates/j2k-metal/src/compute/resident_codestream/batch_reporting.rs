// SPDX-License-Identifier: MIT OR Apache-2.0

use metal::{Buffer, CommandBuffer};

use super::super::resident_packet_plan::PreparedLosslessBatchTile;
use super::{ForeignType, J2kResidentEncodeGpuStage, J2kResidentEncodeGpuStageCommandBuffer};

pub(super) fn collect_prepared_batch_retention(
    profile_stages: bool,
    prepared_tiles: Vec<PreparedLosslessBatchTile>,
    gpu_stage_command_buffers: &mut Vec<J2kResidentEncodeGpuStageCommandBuffer>,
    recyclable_private_buffers: &mut Vec<(usize, Buffer)>,
) -> (Vec<CommandBuffer>, Vec<Buffer>) {
    if profile_stages {
        let mut prepare_command_buffer_ptrs = Vec::new();
        for tile in &prepared_tiles {
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
                    if prepare_command_buffer_ptrs.contains(&ptr) {
                        continue;
                    }
                    prepare_command_buffer_ptrs.push(ptr);
                    pushed_split_prepare = true;
                    gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
                        stage,
                        command_buffer: command_buffer.clone(),
                    });
                }
            }
            for command_buffer in &tile.prepare_dwt53_vertical_command_buffers {
                let ptr = command_buffer.as_ptr();
                if prepare_command_buffer_ptrs.contains(&ptr) {
                    continue;
                }
                prepare_command_buffer_ptrs.push(ptr);
                pushed_split_prepare = true;
                gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
                    stage: J2kResidentEncodeGpuStage::CoefficientDwt53Vertical,
                    command_buffer: command_buffer.clone(),
                });
            }
            for command_buffer in &tile.prepare_dwt53_horizontal_command_buffers {
                let ptr = command_buffer.as_ptr();
                if prepare_command_buffer_ptrs.contains(&ptr) {
                    continue;
                }
                prepare_command_buffer_ptrs.push(ptr);
                pushed_split_prepare = true;
                gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
                    stage: J2kResidentEncodeGpuStage::CoefficientDwt53Horizontal,
                    command_buffer: command_buffer.clone(),
                });
            }
            if pushed_split_prepare {
                continue;
            }
            let ptr = tile.prepare_command_buffer.as_ptr();
            if prepare_command_buffer_ptrs.contains(&ptr) {
                continue;
            }
            prepare_command_buffer_ptrs.push(ptr);
            gpu_stage_command_buffers.push(J2kResidentEncodeGpuStageCommandBuffer {
                stage: J2kResidentEncodeGpuStage::CoefficientPrep,
                command_buffer: tile.prepare_command_buffer.clone(),
            });
        }
    }

    let mut retained_command_buffers = Vec::with_capacity(prepared_tiles.len());
    retained_command_buffers.extend(
        gpu_stage_command_buffers
            .iter()
            .map(|stage_command_buffer| stage_command_buffer.command_buffer.clone()),
    );
    let mut retained_buffers = Vec::<Buffer>::new();
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
    (retained_command_buffers, retained_buffers)
}
