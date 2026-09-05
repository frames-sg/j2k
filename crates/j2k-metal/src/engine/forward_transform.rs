// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

use std::mem::{size_of, size_of_val};

use j2k_metal_support::{dispatch_2d_pipeline, dispatch_3d_pipeline};

use crate::profile_env::{label_command_buffer, label_compute_encoder};

use super::abi::{
    J2kForwardDwt53BatchedParams, J2kForwardDwt53Params, J2kFusedInputMctParams,
    J2kLosslessDeinterleaveParams,
};
use super::{
    checked_buffer_slice, commit_and_wait_metal, copied_slice_buffer, new_blit_command_encoder,
    new_command_buffer, new_compute_command_encoder, new_shared_buffer, with_runtime, Buffer,
    CommandBufferRef, ComputePipelineState, Error, J2kDeinterleaveMctToF32Job,
    J2kDeinterleaveToF32Job, J2kForwardDwt53Level, J2kForwardDwt53Output, J2kForwardDwt97Level,
    J2kForwardDwt97Output,
};

#[cfg(target_os = "macos")]
mod commands97;
#[cfg(target_os = "macos")]
pub(super) use commands97::encode_forward_dwt97_commands;

#[cfg(target_os = "macos")]
pub(crate) fn encode_deinterleave_mct_to_f32(
    job: J2kDeinterleaveMctToF32Job<'_>,
) -> Result<Vec<Vec<f32>>, Error> {
    validate_encode_deinterleave_to_f32_job(J2kDeinterleaveToF32Job {
        pixels: job.pixels,
        num_pixels: job.num_pixels,
        num_components: 3,
        bit_depth: job.bit_depth,
        signed: job.signed,
    })?;
    let len = u32::try_from(job.num_pixels).map_err(|_| Error::MetalKernel {
        message: "J2K Metal combined input MCT pixel count exceeds u32".to_string(),
    })?;
    let bytes_per_sample = encode_deinterleave_bytes_per_sample(job.bit_depth);
    let expected_len = job
        .num_pixels
        .checked_mul(3)
        .and_then(|samples| samples.checked_mul(bytes_per_sample))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal combined input MCT length overflow".to_string(),
        })?;
    if job.pixels.len() != expected_len {
        return Err(Error::MetalKernel {
            message: format!(
                "J2K Metal combined input MCT length mismatch: expected {expected_len} bytes, got {}",
                job.pixels.len()
            ),
        });
    }
    let plane_bytes = job
        .num_pixels
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal combined input MCT output length overflow".to_string(),
        })?;

    with_runtime(|runtime| {
        let input_buffer = copied_slice_buffer(&runtime.device, job.pixels)?;
        let mut plane_budget = crate::batch_allocation::BatchMetadataBudget::new(
            "J2K Metal combined input MCT plane buffers",
        );
        let mut plane_buffers =
            plane_budget.try_vec(3, "J2K Metal combined input MCT plane buffer handles")?;
        for _ in 0..3 {
            plane_buffers.push(new_shared_buffer(&runtime.device, plane_bytes)?);
        }
        let params = J2kFusedInputMctParams {
            len,
            bytes_per_sample: u32::try_from(bytes_per_sample).map_err(|_| Error::MetalKernel {
                message: "J2K Metal combined input MCT sample width exceeds u32".to_string(),
            })?,
            bit_depth: u32::from(job.bit_depth),
            sample_offset: encode_deinterleave_sample_offset(job.bit_depth, job.signed),
            signed_samples: u32::from(job.signed),
            reversible: u32::from(job.reversible),
        };
        let command_buffer = new_command_buffer(&runtime.queue)?;
        label_command_buffer(&command_buffer, "j2k combined encode input MCT");
        let encoder = new_compute_command_encoder(&command_buffer)?;
        label_compute_encoder(&encoder, "J2K combined encode input MCT");
        encoder.setComputePipelineState(&runtime.encode()?.encode_deinterleave_mct);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&plane_buffers[0]), 0);
        encoder.set_buffer(2, Some(&plane_buffers[1]), 0);
        encoder.set_buffer(3, Some(&plane_buffers[2]), 0);
        encoder.set_bytes::<J2kFusedInputMctParams>(4, &params);
        dispatch_2d_pipeline(
            &encoder,
            &runtime.encode()?.encode_deinterleave_mct,
            (len, 1),
        );
        encoder.endEncoding();
        commit_and_wait_metal(&command_buffer)?;

        plane_buffers
            .iter()
            .map(|buffer| {
                checked_buffer_slice::<f32>(buffer, job.num_pixels, "combined input MCT plane")
            })
            .collect()
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_forward_dwt53(
    samples: &[f32],
    width: u32,
    height: u32,
    num_levels: u8,
) -> Result<J2kForwardDwt53Output, Error> {
    if width == 0 || height == 0 {
        return Err(Error::MetalKernel {
            message: "J2K Metal forward DWT dimensions must be non-zero".to_string(),
        });
    }
    let expected_len = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal forward DWT dimensions overflow".to_string(),
        })?;
    if samples.len() != expected_len {
        return Err(Error::MetalKernel {
            message: "J2K Metal forward DWT sample length mismatch".to_string(),
        });
    }

    with_runtime(|runtime| {
        let bytes = size_of_val(samples);
        let buffer_a = copied_slice_buffer(&runtime.device, samples)?;
        let buffer_b = new_shared_buffer(&runtime.device, bytes)?;
        let command_buffer = new_command_buffer(&runtime.queue)?;

        let mut current_width = width;
        let mut current_height = height;
        let mut shapes = Vec::new();
        let mut levels_run = 0u8;
        let mut active_is_a = true;

        while levels_run < num_levels && (current_width >= 2 || current_height >= 2) {
            let low_width = current_width.div_ceil(2);
            let low_height = current_height.div_ceil(2);
            let params = J2kForwardDwt53Params {
                full_width: width,
                current_width,
                current_height,
                low_width,
                low_height,
            };

            if current_height >= 2 {
                let (input, output) =
                    active_forward_dwt53_buffers(&buffer_a, &buffer_b, active_is_a);
                dispatch_forward_dwt53_pass(
                    &runtime.encode()?.fdwt53_vertical,
                    &command_buffer,
                    input,
                    output,
                    params,
                    "J2K forward DWT 5/3 vertical",
                )?;
                active_is_a = !active_is_a;
            }
            if current_width >= 2 {
                let (input, output) =
                    active_forward_dwt53_buffers(&buffer_a, &buffer_b, active_is_a);
                dispatch_forward_dwt53_pass(
                    &runtime.encode()?.fdwt53_horizontal,
                    &command_buffer,
                    input,
                    output,
                    params,
                    "J2K forward DWT 5/3 horizontal",
                )?;
                active_is_a = !active_is_a;
            }

            shapes.push(J2kForwardDwt53Level {
                hl: Vec::new(),
                lh: Vec::new(),
                hh: Vec::new(),
                width: current_width,
                height: current_height,
                low_width,
                low_height,
                high_width: current_width / 2,
                high_height: current_height / 2,
            });
            current_width = low_width;
            current_height = low_height;
            levels_run = levels_run.saturating_add(1);
        }

        commit_and_wait_metal(&command_buffer)?;

        let active_buffer = if active_is_a { &buffer_a } else { &buffer_b };
        let transformed = checked_buffer_slice::<f32>(active_buffer, samples.len(), "DWT 5/3")?;
        let output = extract_forward_dwt53_output(
            &transformed,
            width,
            current_width,
            current_height,
            shapes,
        )?;
        Ok(output)
    })
}

#[cfg(target_os = "macos")]
pub(super) const FDWT97_ALPHA: f32 = j2k_codec_math::dwt::DWT97_ALPHA_F32;
#[cfg(target_os = "macos")]
pub(super) const FDWT97_BETA: f32 = j2k_codec_math::dwt::DWT97_BETA_F32;
#[cfg(target_os = "macos")]
pub(super) const FDWT97_GAMMA: f32 = j2k_codec_math::dwt::DWT97_GAMMA_F32;
#[cfg(target_os = "macos")]
pub(super) const FDWT97_DELTA: f32 = j2k_codec_math::dwt::DWT97_DELTA_F32;
#[cfg(target_os = "macos")]
pub(super) const FDWT97_HIGH_PASS: u32 = 1;
#[cfg(target_os = "macos")]
pub(super) const FDWT97_LOW_PASS: u32 = 0;

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct J2kForwardDwt97Params {
    pub(super) full_width: u32,
    pub(super) current_width: u32,
    pub(super) current_height: u32,
    pub(super) low_width: u32,
    pub(super) low_height: u32,
    pub(super) parity: u32,
    pub(super) coefficient: f32,
    pub(super) _reserved: u32,
}

const _: [(); 0] = [(); core::mem::offset_of!(J2kForwardDwt97Params, full_width)];
const _: [(); 4] = [(); core::mem::offset_of!(J2kForwardDwt97Params, current_width)];
const _: [(); 8] = [(); core::mem::offset_of!(J2kForwardDwt97Params, current_height)];
const _: [(); 12] = [(); core::mem::offset_of!(J2kForwardDwt97Params, low_width)];
const _: [(); 16] = [(); core::mem::offset_of!(J2kForwardDwt97Params, low_height)];
const _: [(); 20] = [(); core::mem::offset_of!(J2kForwardDwt97Params, parity)];
const _: [(); 24] = [(); core::mem::offset_of!(J2kForwardDwt97Params, coefficient)];
const _: [(); 28] = [(); core::mem::offset_of!(J2kForwardDwt97Params, _reserved)];
const _: [(); 32] = [(); core::mem::size_of::<J2kForwardDwt97Params>()];

// SAFETY: The compile-time field walk proves the repr(C) value is exactly
// eight initialized four-byte scalar fields without internal or tail padding.
// Every u32/f32 bit pattern is valid and the Metal declaration has the same
// field order and widths.
unsafe impl j2k_core::accelerator::GpuAbi for J2kForwardDwt97Params {
    const NAME: &'static str = "J2kForwardDwt97Params";
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "forward transform dispatch preserves scratch-buffer and command ordering"
)]
pub(crate) fn encode_forward_dwt97(
    samples: &[f32],
    width: u32,
    height: u32,
    num_levels: u8,
) -> Result<J2kForwardDwt97Output, Error> {
    if width == 0 || height == 0 {
        return Err(Error::MetalKernel {
            message: "J2K Metal forward DWT dimensions must be non-zero".to_string(),
        });
    }
    let width_usize = usize::try_from(width).map_err(|_| Error::MetalKernel {
        message: "J2K Metal forward DWT width does not fit usize".to_string(),
    })?;
    let height_usize = usize::try_from(height).map_err(|_| Error::MetalKernel {
        message: "J2K Metal forward DWT height does not fit usize".to_string(),
    })?;
    let expected_len = width_usize
        .checked_mul(height_usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal forward DWT dimensions overflow".to_string(),
        })?;
    if samples.len() != expected_len {
        return Err(Error::MetalKernel {
            message: "J2K Metal forward DWT sample length mismatch".to_string(),
        });
    }
    let bytes = size_of_val(samples);
    with_runtime(|runtime| {
        let buffer_a = copied_slice_buffer(&runtime.device, samples)?;
        let buffer_b = new_shared_buffer(&runtime.device, bytes)?;
        let command_buffer = new_command_buffer(&runtime.queue)?;

        let layout = encode_forward_dwt97_commands(
            runtime,
            &command_buffer,
            (&buffer_a, &buffer_b),
            (width, height),
            num_levels,
        )?;
        commit_and_wait_metal(&command_buffer)?;

        let active_buffer = if layout.active_is_a {
            &buffer_a
        } else {
            &buffer_b
        };
        let transformed = checked_buffer_slice::<f32>(active_buffer, samples.len(), "DWT 9/7")?;
        extract_forward_dwt97_output(
            &transformed,
            width,
            layout.ll_width,
            layout.ll_height,
            layout.levels,
        )
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_deinterleave_to_f32(
    job: J2kDeinterleaveToF32Job<'_>,
) -> Result<Option<Vec<Vec<f32>>>, Error> {
    validate_encode_deinterleave_to_f32_job(job)?;
    let pixel_count = u32::try_from(job.num_pixels).map_err(|_| Error::MetalKernel {
        message: "J2K Metal encode deinterleave pixel count exceeds u32".to_string(),
    })?;
    let bytes_per_sample = encode_deinterleave_bytes_per_sample(job.bit_depth);
    let sample_count = job
        .num_pixels
        .checked_mul(usize::from(job.num_components))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal encode deinterleave sample count overflow".to_string(),
        })?;
    let expected_len = job
        .num_pixels
        .checked_mul(usize::from(job.num_components))
        .and_then(|samples| samples.checked_mul(bytes_per_sample))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal encode deinterleave input length overflow".to_string(),
        })?;
    if job.pixels.len() != expected_len {
        return Err(Error::MetalKernel {
            message: format!(
                "J2K Metal encode deinterleave input length mismatch: expected {expected_len} bytes, got {}",
                job.pixels.len()
            ),
        });
    }
    let src_stride = u32::try_from(expected_len).map_err(|_| Error::MetalKernel {
        message: "J2K Metal encode deinterleave row stride exceeds u32".to_string(),
    })?;
    let plane_bytes = job
        .num_pixels
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal encode deinterleave output length overflow".to_string(),
        })?;

    with_runtime(|runtime| {
        let input_buffer = copied_slice_buffer(&runtime.device, job.pixels)?;
        let mut plane_budget = crate::batch_allocation::BatchMetadataBudget::new(
            "J2K Metal encode deinterleave plane buffers",
        );
        let mut plane_buffers =
            plane_budget.try_vec(4, "J2K Metal deinterleave plane buffer handles")?;
        for _ in 0..4 {
            plane_buffers.push(new_shared_buffer(&runtime.device, plane_bytes)?);
        }
        let params = J2kLosslessDeinterleaveParams {
            src_width: pixel_count,
            src_height: 1,
            src_stride,
            dst_width: pixel_count,
            dst_height: 1,
            components: u32::from(job.num_components),
            bytes_per_sample: u32::try_from(bytes_per_sample).map_err(|_| Error::MetalKernel {
                message: "J2K Metal encode deinterleave sample width exceeds u32".to_string(),
            })?,
            bit_depth: u32::from(job.bit_depth),
            sample_offset: encode_deinterleave_sample_offset(job.bit_depth, job.signed),
            signed_samples: u32::from(job.signed),
        };

        let command_buffer = new_command_buffer(&runtime.queue)?;
        label_command_buffer(&command_buffer, "j2k encode-stage deinterleave");
        let encoder = new_compute_command_encoder(&command_buffer)?;
        label_compute_encoder(&encoder, "J2K encode-stage deinterleave");
        encoder.setComputePipelineState(&runtime.encode()?.lossless_deinterleave_to_planes);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&plane_buffers[0]), 0);
        encoder.set_buffer(2, Some(&plane_buffers[1]), 0);
        encoder.set_buffer(3, Some(&plane_buffers[2]), 0);
        encoder.set_bytes::<J2kLosslessDeinterleaveParams>(4, &params);
        encoder.set_buffer(5, Some(&plane_buffers[3]), 0);
        dispatch_2d_pipeline(
            &encoder,
            &runtime.encode()?.lossless_deinterleave_to_planes,
            (pixel_count, 1),
        );
        encoder.endEncoding();
        commit_and_wait_metal(&command_buffer)?;

        let planes = plane_buffers
            .iter()
            .take(usize::from(job.num_components))
            .map(|buffer| {
                checked_buffer_slice::<f32>(buffer, job.num_pixels, "deinterleaved plane")
            })
            .collect::<Result<Vec<_>, Error>>()?;
        debug_assert_eq!(
            sample_count.checked_mul(bytes_per_sample),
            Some(expected_len)
        );
        Ok(Some(planes))
    })
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_forward_dwt97_lift_steps(
    pipeline: &ComputePipelineState,
    command_buffer: &CommandBufferRef,
    buffer_a: &Buffer,
    buffer_b: &Buffer,
    active_is_a: bool,
    base_params: J2kForwardDwt97Params,
    label_prefix: &str,
) -> Result<(), Error> {
    let active_buffer = if active_is_a { buffer_a } else { buffer_b };
    for (parity, coefficient) in [
        (FDWT97_HIGH_PASS, FDWT97_ALPHA),
        (FDWT97_LOW_PASS, FDWT97_BETA),
        (FDWT97_HIGH_PASS, FDWT97_GAMMA),
        (FDWT97_LOW_PASS, FDWT97_DELTA),
    ] {
        let params = J2kForwardDwt97Params {
            parity,
            coefficient,
            ..base_params
        };
        dispatch_forward_dwt97_pass(
            pipeline,
            command_buffer,
            active_buffer,
            active_buffer,
            params,
            label_prefix,
        )?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_forward_dwt97_pass(
    pipeline: &ComputePipelineState,
    command_buffer: &CommandBufferRef,
    input: &Buffer,
    output: &Buffer,
    params: J2kForwardDwt97Params,
    label: &str,
) -> Result<(), Error> {
    let encoder = new_compute_command_encoder(command_buffer)?;
    label_compute_encoder(&encoder, label);
    encoder.setComputePipelineState(pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(output), 0);
    encoder.set_bytes::<J2kForwardDwt97Params>(2, &params);
    dispatch_2d_pipeline(
        &encoder,
        pipeline,
        (params.current_width, params.current_height),
    );
    encoder.endEncoding();
    Ok(())
}

#[cfg(target_os = "macos")]
pub(super) fn validate_encode_deinterleave_to_f32_job(
    job: J2kDeinterleaveToF32Job<'_>,
) -> Result<(), Error> {
    if job.num_pixels == 0 {
        return Err(Error::capability_rejected(
            j2k_core::CapabilityRejection::unsupported_operation(
                "J2K Metal encode deinterleave requires at least one pixel",
            ),
        ));
    }
    if !(1..=4).contains(&job.num_components) {
        return Err(Error::capability_rejected(
            j2k_core::CapabilityRejection::unsupported_format(
                "J2K Metal encode deinterleave supports 1-4 component samples",
            ),
        ));
    }
    if job.bit_depth == 0 || job.bit_depth > 16 {
        return Err(Error::capability_rejected(
            j2k_core::CapabilityRejection::unsupported_operation(
                "J2K Metal encode deinterleave supports 1-16 bits per sample",
            ),
        ));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
pub(super) fn encode_deinterleave_bytes_per_sample(bit_depth: u8) -> usize {
    if bit_depth <= 8 {
        1
    } else {
        2
    }
}

#[cfg(target_os = "macos")]
pub(super) fn encode_deinterleave_sample_offset(bit_depth: u8, signed: bool) -> u32 {
    if signed {
        0
    } else {
        1u32 << (u32::from(bit_depth) - 1)
    }
}

#[cfg(target_os = "macos")]
pub(super) fn active_forward_dwt53_buffers<'a>(
    buffer_a: &'a Buffer,
    buffer_b: &'a Buffer,
    active_is_a: bool,
) -> (&'a Buffer, &'a Buffer) {
    if active_is_a {
        (buffer_a, buffer_b)
    } else {
        (buffer_b, buffer_a)
    }
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_forward_dwt53_pass(
    pipeline: &ComputePipelineState,
    command_buffer: &CommandBufferRef,
    input: &Buffer,
    output: &Buffer,
    params: J2kForwardDwt53Params,
    label: &str,
) -> Result<(), Error> {
    let encoder = new_compute_command_encoder(command_buffer)?;
    label_compute_encoder(&encoder, label);
    encoder.setComputePipelineState(pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(output), 0);
    encoder.set_bytes::<J2kForwardDwt53Params>(2, &params);
    dispatch_2d_pipeline(
        &encoder,
        pipeline,
        (params.current_width, params.current_height),
    );
    encoder.endEncoding();
    Ok(())
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_forward_dwt53_batched_pass(
    pipeline: &ComputePipelineState,
    command_buffer: &CommandBufferRef,
    inputs: &[Buffer],
    outputs: &[Buffer],
    params: J2kForwardDwt53BatchedParams,
    label: &str,
) -> Result<(), Error> {
    debug_assert!(!inputs.is_empty());
    debug_assert!(!outputs.is_empty());
    debug_assert!(params.component_count >= 1 && params.component_count <= 3);
    let first_input_buffer = &inputs[0];
    let second_input_buffer = inputs.get(1).unwrap_or(first_input_buffer);
    let third_input_buffer = inputs.get(2).unwrap_or(first_input_buffer);
    let first_output_buffer = &outputs[0];
    let second_output_buffer = outputs.get(1).unwrap_or(first_output_buffer);
    let third_output_buffer = outputs.get(2).unwrap_or(first_output_buffer);

    let encoder = new_compute_command_encoder(command_buffer)?;
    label_compute_encoder(&encoder, label);
    encoder.setComputePipelineState(pipeline);
    encoder.set_buffer(0, Some(first_input_buffer), 0);
    encoder.set_buffer(1, Some(second_input_buffer), 0);
    encoder.set_buffer(2, Some(third_input_buffer), 0);
    encoder.set_buffer(3, Some(first_output_buffer), 0);
    encoder.set_buffer(4, Some(second_output_buffer), 0);
    encoder.set_buffer(5, Some(third_output_buffer), 0);
    encoder.set_bytes::<J2kForwardDwt53BatchedParams>(6, &params);
    dispatch_3d_pipeline(
        &encoder,
        pipeline,
        (
            params.current_width,
            params.current_height,
            params.component_count,
        ),
    );
    encoder.endEncoding();
    Ok(())
}

#[cfg(target_os = "macos")]
pub(super) fn extract_forward_dwt53_output(
    transformed: &[f32],
    full_width: u32,
    ll_width: u32,
    ll_height: u32,
    mut shapes: Vec<J2kForwardDwt53Level>,
) -> Result<J2kForwardDwt53Output, Error> {
    let full_width_usize = full_width as usize;
    let ll_len = crate::batch_allocation::checked_count_product(
        ll_width as usize,
        ll_height as usize,
        "J2K Metal forward DWT 5/3 LL coefficients",
    )?;
    let mut ll =
        crate::batch_allocation::try_vec(ll_len, "J2K Metal forward DWT 5/3 LL coefficients")?;
    for y in 0..ll_height as usize {
        let row_start = y
            .checked_mul(full_width_usize)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal forward DWT LL row offset overflow".to_string(),
            })?;
        ll.extend_from_slice(&transformed[row_start..row_start + ll_width as usize]);
    }

    for shape in &mut shapes {
        shape.hl = extract_subband(
            transformed,
            full_width_usize,
            shape.low_width,
            0,
            shape.high_width,
            shape.low_height,
        )?;
        shape.lh = extract_subband(
            transformed,
            full_width_usize,
            0,
            shape.low_height,
            shape.low_width,
            shape.high_height,
        )?;
        shape.hh = extract_subband(
            transformed,
            full_width_usize,
            shape.low_width,
            shape.low_height,
            shape.high_width,
            shape.high_height,
        )?;
    }
    shapes.reverse();

    Ok(J2kForwardDwt53Output {
        ll,
        ll_width,
        ll_height,
        levels: shapes,
    })
}

#[cfg(target_os = "macos")]
pub(super) fn extract_forward_dwt97_output(
    transformed: &[f32],
    full_width: u32,
    ll_width: u32,
    ll_height: u32,
    mut shapes: Vec<J2kForwardDwt97Level>,
) -> Result<J2kForwardDwt97Output, Error> {
    let full_width_usize = usize::try_from(full_width).map_err(|_| Error::MetalKernel {
        message: "J2K Metal forward DWT full width does not fit usize".to_string(),
    })?;
    let ll_width_usize = usize::try_from(ll_width).map_err(|_| Error::MetalKernel {
        message: "J2K Metal forward DWT LL width does not fit usize".to_string(),
    })?;
    let ll_height_usize = usize::try_from(ll_height).map_err(|_| Error::MetalKernel {
        message: "J2K Metal forward DWT LL height does not fit usize".to_string(),
    })?;
    let ll_len = ll_width_usize
        .checked_mul(ll_height_usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal forward DWT LL dimensions overflow".to_string(),
        })?;
    let mut ll =
        crate::batch_allocation::try_vec(ll_len, "J2K Metal forward DWT 9/7 LL coefficients")?;
    for y in 0..ll_height_usize {
        let row_start = y
            .checked_mul(full_width_usize)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal forward DWT LL row offset overflow".to_string(),
            })?;
        let row_end = row_start
            .checked_add(ll_width_usize)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal forward DWT LL row end overflow".to_string(),
            })?;
        let row = transformed
            .get(row_start..row_end)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal forward DWT LL row out of bounds".to_string(),
            })?;
        ll.extend_from_slice(row);
    }

    for shape in &mut shapes {
        shape.hl = extract_subband(
            transformed,
            full_width_usize,
            shape.low_width,
            0,
            shape.high_width,
            shape.low_height,
        )?;
        shape.lh = extract_subband(
            transformed,
            full_width_usize,
            0,
            shape.low_height,
            shape.low_width,
            shape.high_height,
        )?;
        shape.hh = extract_subband(
            transformed,
            full_width_usize,
            shape.low_width,
            shape.low_height,
            shape.high_width,
            shape.high_height,
        )?;
    }
    shapes.reverse();

    Ok(J2kForwardDwt97Output {
        ll,
        ll_width,
        ll_height,
        levels: shapes,
    })
}

#[cfg(target_os = "macos")]
pub(super) fn extract_subband(
    transformed: &[f32],
    full_width: usize,
    x0: u32,
    y0: u32,
    width: u32,
    height: u32,
) -> Result<Vec<f32>, Error> {
    let output_len = crate::batch_allocation::checked_count_product(
        width as usize,
        height as usize,
        "J2K Metal forward DWT subband coefficients",
    )?;
    let mut out =
        crate::batch_allocation::try_vec(output_len, "J2K Metal forward DWT subband coefficients")?;
    for y in 0..height as usize {
        let row_start = (y0 as usize)
            .checked_add(y)
            .and_then(|row| row.checked_mul(full_width))
            .and_then(|row| row.checked_add(x0 as usize))
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal forward DWT subband offset overflow".to_string(),
            })?;
        out.extend_from_slice(&transformed[row_start..row_start + width as usize]);
    }
    Ok(out)
}
