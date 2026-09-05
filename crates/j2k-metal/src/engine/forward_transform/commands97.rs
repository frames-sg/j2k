// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    active_forward_dwt53_buffers, dispatch_forward_dwt97_lift_steps, dispatch_forward_dwt97_pass,
    new_blit_command_encoder, Buffer, CommandBufferRef, Error, J2kForwardDwt97Level,
    J2kForwardDwt97Params, FDWT97_HIGH_PASS,
};
use crate::engine::runtime::MetalRuntime;
use crate::metal_types::prelude::*;

pub(in crate::engine) struct ForwardDwt97Layout {
    pub(in crate::engine) active_is_a: bool,
    pub(in crate::engine) ll_width: u32,
    pub(in crate::engine) ll_height: u32,
    pub(in crate::engine) levels: Vec<J2kForwardDwt97Level>,
}

pub(in crate::engine) fn encode_forward_dwt97_commands(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    buffers: (&Buffer, &Buffer),
    dimensions: (u32, u32),
    num_levels: u8,
) -> Result<ForwardDwt97Layout, Error> {
    let (buffer_a, buffer_b) = buffers;
    let (width, height) = dimensions;
    let bytes = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(size_of::<f32>()))
        .ok_or_else(|| Error::MetalKernel {
            message: "Metal resident DWT97 dimensions overflow".to_owned(),
        })?;
    let mut current_width = width;
    let mut current_height = height;
    let mut shapes = Vec::new();
    let mut levels_run = 0u8;
    let mut active_is_a = true;

    while levels_run < num_levels && (current_width >= 2 || current_height >= 2) {
        let low_width = current_width.div_ceil(2);
        let low_height = current_height.div_ceil(2);
        // With only one active axis, this level ends in the other buffer.
        // Carry the already completed high bands across that swap.
        if levels_run > 0 && ((current_width >= 2) != (current_height >= 2)) {
            let (input, output) = active_forward_dwt53_buffers(buffer_a, buffer_b, active_is_a);
            let blit = new_blit_command_encoder(command_buffer)?;
            blit.copy_from_buffer(input, 0, output, 0, bytes as u64)?;
            blit.endEncoding();
        }
        let base_params = J2kForwardDwt97Params {
            full_width: width,
            current_width,
            current_height,
            low_width,
            low_height,
            parity: FDWT97_HIGH_PASS,
            coefficient: 0.0,
            _reserved: 0,
        };

        if current_height >= 2 {
            dispatch_forward_dwt97_lift_steps(
                &runtime.encode()?.fdwt97_lift_vertical,
                command_buffer,
                buffer_a,
                buffer_b,
                active_is_a,
                base_params,
                "J2K forward DWT 9/7 vertical",
            )?;
            let (input, output) = active_forward_dwt53_buffers(buffer_a, buffer_b, active_is_a);
            dispatch_forward_dwt97_pass(
                &runtime.encode()?.fdwt97_deinterleave_vertical,
                command_buffer,
                input,
                output,
                base_params,
                "J2K forward DWT 9/7 vertical deinterleave",
            )?;
            active_is_a = !active_is_a;
        }
        if current_width >= 2 {
            dispatch_forward_dwt97_lift_steps(
                &runtime.encode()?.fdwt97_lift_horizontal,
                command_buffer,
                buffer_a,
                buffer_b,
                active_is_a,
                base_params,
                "J2K forward DWT 9/7 horizontal",
            )?;
            let (input, output) = active_forward_dwt53_buffers(buffer_a, buffer_b, active_is_a);
            dispatch_forward_dwt97_pass(
                &runtime.encode()?.fdwt97_deinterleave_horizontal,
                command_buffer,
                input,
                output,
                base_params,
                "J2K forward DWT 9/7 horizontal deinterleave",
            )?;
            active_is_a = !active_is_a;
        }

        shapes.push(J2kForwardDwt97Level {
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
    Ok(ForwardDwt97Layout {
        active_is_a,
        ll_width: current_width,
        ll_height: current_height,
        levels: shapes,
    })
}
