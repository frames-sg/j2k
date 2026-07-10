// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    dispatch_forward_dwt53_on_buffers, dispatch_forward_rct_on_buffers,
    dispatch_lossless_deinterleave, dispatch_lossless_deinterleave_rct_rgb8,
    dispatch_lossless_extract_coefficients, lossless_deinterleave_rct_rgb8_supported,
    lossless_prepare_sizes, size_of, with_runtime_for_session, zeroed_shared_buffer, Error,
    J2kLosslessCoefficientJob, J2kLosslessDeviceCodeBlock, J2kLosslessDevicePrepareJob,
    J2kMctStatus, J2kPreparedLosslessDeviceCodeBlocks, MTLResourceOptions,
};

#[cfg(target_os = "macos")]
pub(crate) fn prepare_lossless_device_code_blocks(
    session: &crate::MetalBackendSession,
    job: J2kLosslessDevicePrepareJob<'_>,
    code_blocks: Vec<J2kLosslessDeviceCodeBlock>,
) -> Result<J2kPreparedLosslessDeviceCodeBlocks, Error> {
    let sizes = lossless_prepare_sizes(job)?;

    with_runtime_for_session(session, |runtime| {
        let mut plane_buffers = Vec::with_capacity(3);
        let mut scratch_buffers = Vec::with_capacity(usize::from(job.component_count));
        for _ in 0..3 {
            plane_buffers.push(runtime.device.new_buffer(
                sizes.plane_bytes as u64,
                MTLResourceOptions::StorageModePrivate,
            ));
        }
        for _ in 0..job.component_count {
            scratch_buffers.push(runtime.device.new_buffer(
                sizes.plane_bytes as u64,
                MTLResourceOptions::StorageModePrivate,
            ));
        }
        let coefficient_buffer = runtime.device.new_buffer(
            sizes.coefficient_bytes as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let status_buffer = zeroed_shared_buffer(&runtime.device, size_of::<J2kMctStatus>());
        let command_buffer = runtime.queue.new_command_buffer();

        if lossless_deinterleave_rct_rgb8_supported(job) {
            dispatch_lossless_deinterleave_rct_rgb8(
                runtime,
                command_buffer,
                job,
                &plane_buffers[0],
                &plane_buffers[1],
                &plane_buffers[2],
                &status_buffer,
            )?;
        } else {
            dispatch_lossless_deinterleave(
                runtime,
                command_buffer,
                job,
                &plane_buffers[0],
                &plane_buffers[1],
                &plane_buffers[2],
            )?;
        }
        if job.component_count == 3 && !lossless_deinterleave_rct_rgb8_supported(job) {
            dispatch_forward_rct_on_buffers(
                runtime,
                command_buffer,
                &plane_buffers[0],
                &plane_buffers[1],
                &plane_buffers[2],
                sizes.plane_len,
                &status_buffer,
            )?;
        }

        let mut active_planes = Vec::with_capacity(usize::from(job.component_count));
        for component in 0..usize::from(job.component_count) {
            if job.num_decomposition_levels == 0 {
                active_planes.push(plane_buffers[component].clone());
            } else {
                active_planes.push(dispatch_forward_dwt53_on_buffers(
                    runtime,
                    command_buffer,
                    &plane_buffers[component],
                    &scratch_buffers[component],
                    job.output_width,
                    job.output_height,
                    job.num_decomposition_levels,
                ));
            }
        }
        while active_planes.len() < 3 {
            active_planes.push(active_planes[0].clone());
        }

        let coefficient_jobs = code_blocks
            .iter()
            .map(|block| J2kLosslessCoefficientJob {
                coefficient_offset: block.coefficient_offset,
                component: block.component,
                subband_x: block.subband_x,
                subband_y: block.subband_y,
                block_x: block.block_x,
                block_y: block.block_y,
                block_width: block.width,
                block_height: block.height,
                full_width: job.output_width,
            })
            .collect::<Vec<_>>();
        let coefficient_job_buffer = dispatch_lossless_extract_coefficients(
            runtime,
            command_buffer,
            &active_planes,
            &coefficient_buffer,
            &coefficient_jobs,
            job.output_width,
        )?;

        command_buffer.commit();
        Ok(J2kPreparedLosslessDeviceCodeBlocks {
            coefficient_buffer,
            coefficient_byte_offset: 0,
            coefficient_byte_len: sizes.coefficient_bytes,
            coefficient_buffer_is_batch_shared: false,
            code_blocks,
            recyclable_private_buffers: Vec::new(),
            _prepare_command_buffer: command_buffer.to_owned(),
            _prepare_deinterleave_rct_command_buffer: None,
            _prepare_dwt53_command_buffer: None,
            _prepare_dwt53_vertical_command_buffers: Vec::new(),
            _prepare_dwt53_horizontal_command_buffers: Vec::new(),
            _prepare_coefficient_extract_command_buffer: None,
            _deinterleave_status_buffer: status_buffer,
            _plane_buffers: plane_buffers,
            _scratch_buffers: scratch_buffers,
            _coefficient_job_buffer: coefficient_job_buffer,
        })
    })
}
