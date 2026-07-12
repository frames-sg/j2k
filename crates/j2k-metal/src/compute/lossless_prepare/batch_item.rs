// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    dispatch_forward_dwt53_components_on_buffers, dispatch_forward_dwt53_components_split_profile,
    dispatch_forward_dwt53_on_buffers, dispatch_forward_dwt53_on_buffers_split_profile,
    dispatch_forward_rct_on_buffers, dispatch_lossless_deinterleave,
    dispatch_lossless_deinterleave_rct_rgb8, dispatch_lossless_extract_coefficients,
    lossless_deinterleave_rct_rgb8_supported, new_resident_encode_command_buffer, size_of,
    take_recyclable_private_buffer, zeroed_shared_buffer, Buffer, CommandBuffer, CommandBufferRef,
    Error, ForwardDwt53ComponentsDispatch, ForwardDwt53SplitProfile, J2kLosslessCoefficientJob,
    J2kLosslessDeviceBatchPrepareItem, J2kLosslessDevicePrepareJob, J2kLosslessPrepareSizes,
    J2kMctStatus, J2kPreparedLosslessDeviceCodeBlocks, MetalRuntime,
};

pub(in crate::compute) struct BatchPrepareItemRequest<'a, 'job> {
    pub(in crate::compute) runtime: &'a MetalRuntime,
    pub(in crate::compute) item: J2kLosslessDeviceBatchPrepareItem<'job>,
    pub(in crate::compute) item_sizes: J2kLosslessPrepareSizes,
    pub(in crate::compute) coefficient_buffer: &'a Buffer,
    pub(in crate::compute) coefficient_byte_offset: usize,
    pub(in crate::compute) split_prepare_command_buffers: bool,
    pub(in crate::compute) shared_command_buffer: &'a Option<CommandBuffer>,
    pub(in crate::compute) shared_recyclable_private_buffers: &'a mut Vec<(usize, Buffer)>,
}

struct BatchPrepareCommandStrategy<'a> {
    runtime: &'a MetalRuntime,
    split: bool,
    shared: &'a Option<CommandBuffer>,
}

impl BatchPrepareCommandStrategy<'_> {
    fn command_buffer(&self, label: &'static str) -> Result<CommandBuffer, Error> {
        if self.split {
            new_resident_encode_command_buffer(self.runtime, label)
        } else {
            self.shared.clone().ok_or_else(|| Error::MetalKernel {
                message: "shared coefficient prep command buffer is missing".to_string(),
            })
        }
    }

    fn finish_split_stage(&self, command_buffer: CommandBuffer) -> Option<CommandBuffer> {
        if self.split {
            command_buffer.commit();
            Some(command_buffer)
        } else {
            None
        }
    }

    fn shared_command_buffer(&self) -> Result<&CommandBufferRef, Error> {
        self.shared.as_deref().ok_or_else(|| Error::MetalKernel {
            message: "shared coefficient prep command buffer is missing".to_string(),
        })
    }
}

struct BatchDwtPreparation {
    active_planes: Vec<Buffer>,
    vertical_command_buffers: Vec<CommandBuffer>,
    horizontal_command_buffers: Vec<CommandBuffer>,
}

fn prepare_batch_item_dwt(
    strategy: &BatchPrepareCommandStrategy<'_>,
    job: J2kLosslessDevicePrepareJob<'_>,
    plane_buffers: &[Buffer],
    scratch_buffers: &[Buffer],
) -> Result<BatchDwtPreparation, Error> {
    let component_count = usize::from(job.component_count);
    let mut active_planes = Vec::with_capacity(component_count);
    let mut vertical_command_buffers = Vec::new();
    let mut horizontal_command_buffers = Vec::new();

    if job.num_decomposition_levels == 0 {
        active_planes.extend(plane_buffers.iter().take(component_count).cloned());
    } else if strategy.split {
        if component_count > 1 {
            let ForwardDwt53SplitProfile {
                active: mut component_active_planes,
                vertical_command_buffers: mut component_vertical_command_buffers,
                horizontal_command_buffers: mut component_horizontal_command_buffers,
            } = dispatch_forward_dwt53_components_split_profile(
                strategy.runtime,
                plane_buffers,
                scratch_buffers,
                job.output_width,
                job.output_height,
                job.num_decomposition_levels,
                component_count,
            )?;
            active_planes.append(&mut component_active_planes);
            vertical_command_buffers.append(&mut component_vertical_command_buffers);
            horizontal_command_buffers.append(&mut component_horizontal_command_buffers);
        } else {
            for component in 0..component_count {
                let ForwardDwt53SplitProfile {
                    active: active_plane,
                    vertical_command_buffers: mut component_vertical_command_buffers,
                    horizontal_command_buffers: mut component_horizontal_command_buffers,
                } = dispatch_forward_dwt53_on_buffers_split_profile(
                    strategy.runtime,
                    &plane_buffers[component],
                    &scratch_buffers[component],
                    job.output_width,
                    job.output_height,
                    job.num_decomposition_levels,
                )?;
                active_planes.push(active_plane);
                vertical_command_buffers.append(&mut component_vertical_command_buffers);
                horizontal_command_buffers.append(&mut component_horizontal_command_buffers);
            }
        }
    } else if component_count > 1 {
        active_planes =
            dispatch_forward_dwt53_components_on_buffers(ForwardDwt53ComponentsDispatch {
                runtime: strategy.runtime,
                command_buffer: strategy.shared_command_buffer()?,
                plane_buffers,
                scratch_buffers,
                width: job.output_width,
                height: job.output_height,
                num_levels: job.num_decomposition_levels,
                component_count,
            })?;
    } else {
        for component in 0..component_count {
            active_planes.push(dispatch_forward_dwt53_on_buffers(
                strategy.runtime,
                strategy.shared_command_buffer()?,
                &plane_buffers[component],
                &scratch_buffers[component],
                job.output_width,
                job.output_height,
                job.num_decomposition_levels,
            )?);
        }
    }

    while active_planes.len() < 3 {
        active_planes.push(active_planes[0].clone());
    }
    Ok(BatchDwtPreparation {
        active_planes,
        vertical_command_buffers,
        horizontal_command_buffers,
    })
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "batch preparation keeps validated offsets and retained resources together"
)]
pub(in crate::compute) fn prepare_lossless_batch_item(
    request: BatchPrepareItemRequest<'_, '_>,
) -> Result<J2kPreparedLosslessDeviceCodeBlocks, Error> {
    let BatchPrepareItemRequest {
        runtime,
        item,
        item_sizes,
        coefficient_buffer,
        coefficient_byte_offset,
        split_prepare_command_buffers,
        shared_command_buffer,
        shared_recyclable_private_buffers,
    } = request;
    let command_strategy = BatchPrepareCommandStrategy {
        runtime,
        split: split_prepare_command_buffers,
        shared: shared_command_buffer,
    };
    let job = item.job;
    let mut recyclable_private_buffers = Vec::new();
    if !shared_recyclable_private_buffers.is_empty() {
        recyclable_private_buffers.append(shared_recyclable_private_buffers);
    }
    let mut plane_buffers = Vec::with_capacity(3);
    let mut scratch_buffers = Vec::with_capacity(usize::from(job.component_count));
    for _ in 0..3 {
        plane_buffers.push(take_recyclable_private_buffer(
            runtime,
            item_sizes.plane_bytes,
            &mut recyclable_private_buffers,
        )?);
    }
    for _ in 0..job.component_count {
        scratch_buffers.push(take_recyclable_private_buffer(
            runtime,
            item_sizes.plane_bytes,
            &mut recyclable_private_buffers,
        )?);
    }

    let status_buffer = zeroed_shared_buffer(&runtime.device, size_of::<J2kMctStatus>())?;

    let prepare_dwt53_command_buffer = None;
    let deinterleave_command_buffer =
        command_strategy.command_buffer("j2k coefficient prep deinterleave rct")?;
    if lossless_deinterleave_rct_rgb8_supported(job) {
        dispatch_lossless_deinterleave_rct_rgb8(
            runtime,
            &deinterleave_command_buffer,
            job,
            &plane_buffers[0],
            &plane_buffers[1],
            &plane_buffers[2],
            &status_buffer,
        )
    } else {
        dispatch_lossless_deinterleave(
            runtime,
            &deinterleave_command_buffer,
            job,
            &plane_buffers[0],
            &plane_buffers[1],
            &plane_buffers[2],
        )
    }
    .map_err(|err| Error::MetalKernel {
        message: format!(
            "J2K Metal resident batch coefficient prep failed at tile {}: {err}",
            item.tile_index
        ),
    })?;
    if job.component_count == 3 && !lossless_deinterleave_rct_rgb8_supported(job) {
        dispatch_forward_rct_on_buffers(
            runtime,
            &deinterleave_command_buffer,
            &plane_buffers[0],
            &plane_buffers[1],
            &plane_buffers[2],
            item_sizes.plane_len,
            &status_buffer,
        )
        .map_err(|err| Error::MetalKernel {
            message: format!(
                "J2K Metal resident batch coefficient prep failed at tile {}: {err}",
                item.tile_index
            ),
        })?;
    }
    let prepare_deinterleave_rct_command_buffer =
        command_strategy.finish_split_stage(deinterleave_command_buffer);
    let dwt_preparation =
        prepare_batch_item_dwt(&command_strategy, job, &plane_buffers, &scratch_buffers)?;

    let coefficient_word_offset = coefficient_byte_offset
        .checked_div(size_of::<i32>())
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal resident batch coefficient offset division failed".to_string(),
        })?;
    let coefficient_word_offset_u32 =
        u32::try_from(coefficient_word_offset).map_err(|_| Error::MetalKernel {
            message: format!(
                "J2K Metal resident batch coefficient offset exceeds u32 at tile {}",
                item.tile_index
            ),
        })?;
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "J2K Metal resident batch coefficient job metadata",
    );
    let mut coefficient_jobs = budget.try_vec(
        item.code_blocks.len(),
        "J2K Metal resident batch coefficient jobs",
    )?;
    for block in &item.code_blocks {
        let coefficient_offset = block
            .coefficient_offset
            .checked_add(coefficient_word_offset_u32)
            .ok_or_else(|| Error::MetalKernel {
                message: format!(
                    "J2K Metal resident batch coefficient offset overflow at tile {}",
                    item.tile_index
                ),
            })?;
        coefficient_jobs.push(J2kLosslessCoefficientJob {
            coefficient_offset,
            component: block.component,
            subband_x: block.subband_x,
            subband_y: block.subband_y,
            block_x: block.block_x,
            block_y: block.block_y,
            block_width: block.width,
            block_height: block.height,
            full_width: job.output_width,
        });
    }
    let extract_command_buffer = command_strategy.command_buffer("j2k coefficient prep extract")?;
    let coefficient_job_buffer = dispatch_lossless_extract_coefficients(
        runtime,
        &extract_command_buffer,
        &dwt_preparation.active_planes,
        coefficient_buffer,
        &coefficient_jobs,
        job.output_width,
    )
    .map_err(|err| Error::MetalKernel {
        message: format!(
            "J2K Metal resident batch coefficient prep failed at tile {}: {err}",
            item.tile_index
        ),
    })?;
    let prepare_command_buffer = extract_command_buffer.clone();
    let prepare_coefficient_extract_command_buffer =
        command_strategy.finish_split_stage(extract_command_buffer);

    Ok(J2kPreparedLosslessDeviceCodeBlocks {
        coefficient_buffer: coefficient_buffer.clone(),
        coefficient_byte_offset,
        coefficient_byte_len: item_sizes.coefficient_bytes,
        coefficient_buffer_is_batch_shared: true,
        code_blocks: item.code_blocks,
        recyclable_private_buffers,
        _prepare_command_buffer: prepare_command_buffer,
        _prepare_deinterleave_rct_command_buffer: prepare_deinterleave_rct_command_buffer,
        _prepare_dwt53_command_buffer: prepare_dwt53_command_buffer,
        _prepare_dwt53_vertical_command_buffers: dwt_preparation.vertical_command_buffers,
        _prepare_dwt53_horizontal_command_buffers: dwt_preparation.horizontal_command_buffers,
        _prepare_coefficient_extract_command_buffer: prepare_coefficient_extract_command_buffer,
        _deinterleave_status_buffer: status_buffer,
        _plane_buffers: plane_buffers,
        _scratch_buffers: scratch_buffers,
        _coefficient_job_buffer: coefficient_job_buffer,
    })
}
