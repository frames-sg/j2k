// SPDX-License-Identifier: MIT OR Apache-2.0

//! Progressive coefficient rendering, upsampling, and row output.

use alloc::vec::Vec;

use crate::allocation::{checked_allocation_len, try_reserve_for_len_with_live_budget};
use crate::backend::Backend;
use crate::color::upsample::{upsample_h2v1_fancy_row, upsample_h2v2_fancy_row};
use crate::entropy::block::clamp_i16;
use crate::entropy::ZIGZAG;
use crate::error::JpegError;
use crate::info::ColorSpace;
use crate::output::OutputWriter;

use super::allocation::{allocate_component_images, checked_phase_capacity, ComponentImage};
use super::model::{PreparedProgressiveComponentPlan, PreparedProgressivePlan};

pub(super) fn render_component_images(
    plan: &PreparedProgressivePlan,
    backend: Backend,
    coeffs: &[Vec<[i32; 64]>],
    coefficient_live_bytes: usize,
) -> Result<Vec<ComponentImage>, JpegError> {
    if coeffs.len() != plan.components.len() {
        return Err(JpegError::InternalInvariant {
            reason: "progressive coefficient/component count mismatch",
        });
    }
    let mut images = allocate_component_images(plan, coefficient_live_bytes)?;
    for ((component, component_coeffs), image) in plan
        .components
        .iter()
        .zip(coeffs.iter())
        .zip(images.iter_mut())
    {
        let mut dequant = [0i16; 64];
        let mut pixels = [0u8; 64];
        for by in 0..component.block_rows as usize {
            for bx in 0..component.block_cols as usize {
                let block_index = by * component.block_cols as usize + bx;
                dequantize_block(
                    &component_coeffs[block_index],
                    &component.quant,
                    &mut dequant,
                );
                backend.idct(&dequant, &mut pixels);
                deposit_block(&mut image.plane, image.stride, bx * 8, by * 8, &pixels);
            }
        }
    }
    Ok(images)
}

fn dequantize_block(coeffs: &[i32; 64], quant: &[u16; 64], out: &mut [i16; 64]) {
    out.fill(0);
    for k in 0..64 {
        let natural_idx = usize::from(ZIGZAG[k]);
        let value = coeffs[natural_idx].wrapping_mul(i32::from(quant[k]));
        out[natural_idx] = clamp_i16(value);
    }
}

fn deposit_block(plane: &mut [u8], stride: usize, x: usize, y: usize, block: &[u8; 64]) {
    for row in 0..8 {
        let dst = (y + row) * stride + x;
        let src = row * 8;
        plane[dst..dst + 8].copy_from_slice(&block[src..src + 8]);
    }
}

pub(super) fn emit_component_images<W: OutputWriter>(
    plan: &PreparedProgressivePlan,
    images: &[ComponentImage],
    image_live_bytes: usize,
    writer: &mut W,
) -> Result<(), JpegError> {
    let (width, height) = plan.dimensions;
    let width_usize = width as usize;
    if plan.components.len() == 1 {
        checked_phase_capacity(image_live_bytes, width_usize, plan.scratch_bytes)?;
        let image = images.first().ok_or(JpegError::InternalInvariant {
            reason: "progressive grayscale render has no component image",
        })?;
        let mut live_bytes = image_live_bytes;
        let mut gray = Vec::new();
        try_reserve_for_len_with_live_budget(
            &mut gray,
            width_usize,
            &mut live_bytes,
            plan.scratch_bytes,
        )?;
        gray.resize(width_usize, 0u8);
        for y in 0..height {
            upsample_component_row(plan, 0, image, y, &mut gray);
            writer.write_gray_row(y, &gray)?;
        }
        return Ok(());
    }

    let first = component_slot(plan, 0)?;
    let second = component_slot(plan, 1)?;
    let third = component_slot(plan, 2)?;
    let row_bytes = checked_allocation_len::<u8>(width_usize, 3)?;
    checked_phase_capacity(image_live_bytes, row_bytes, plan.scratch_bytes)?;
    let mut live_bytes = image_live_bytes;
    let mut a = Vec::new();
    try_reserve_for_len_with_live_budget(&mut a, width_usize, &mut live_bytes, plan.scratch_bytes)?;
    a.resize(width_usize, 0u8);
    let mut b = Vec::new();
    try_reserve_for_len_with_live_budget(&mut b, width_usize, &mut live_bytes, plan.scratch_bytes)?;
    b.resize(width_usize, 0u8);
    let mut c = Vec::new();
    try_reserve_for_len_with_live_budget(&mut c, width_usize, &mut live_bytes, plan.scratch_bytes)?;
    c.resize(width_usize, 0u8);
    for y in 0..height {
        upsample_component_row(plan, first, &images[first], y, &mut a);
        upsample_component_row(plan, second, &images[second], y, &mut b);
        upsample_component_row(plan, third, &images[third], y, &mut c);
        match plan.color_space {
            ColorSpace::YCbCr => writer.write_ycbcr_row(y, &a, &b, &c)?,
            ColorSpace::Rgb => writer.write_rgb_row(y, &a, &b, &c)?,
            ColorSpace::Grayscale => writer.write_gray_row(y, &a)?,
            ColorSpace::Cmyk | ColorSpace::Ycck => {
                return Err(JpegError::UnsupportedColorSpace {
                    color_space: plan.color_space,
                });
            }
        }
    }

    Ok(())
}

fn component_slot(plan: &PreparedProgressivePlan, output_index: usize) -> Result<usize, JpegError> {
    plan.components
        .iter()
        .position(|component| component.output_index == output_index)
        .ok_or(JpegError::UnsupportedColorSpace {
            color_space: plan.color_space,
        })
}

fn upsample_component_row(
    plan: &PreparedProgressivePlan,
    component_index: usize,
    image: &ComponentImage,
    y: u32,
    out: &mut [u8],
) {
    let component = &plan.components[component_index];
    let h_ratio = plan.sampling.max_h / component.h;
    let v_ratio = plan.sampling.max_v / component.v;
    if h_ratio == 1 && v_ratio == 1 {
        let sample_y = (y as usize).min(component.sample_height.saturating_sub(1) as usize);
        let row = component_row(component, image, sample_y);
        out.copy_from_slice(&row[..out.len()]);
    } else if h_ratio == 2 && v_ratio == 1 {
        let sample_y = (y as usize).min(component.sample_height.saturating_sub(1) as usize);
        let row = component_row(component, image, sample_y);
        upsample_h2v1_fancy_row(row, out.len(), out);
    } else if h_ratio == 2 && v_ratio == 2 {
        let sample_y = ((y / 2) as usize).min(component.sample_height.saturating_sub(1) as usize);
        let prev_y = sample_y.saturating_sub(1);
        let next_y = (sample_y + 1).min(component.sample_height.saturating_sub(1) as usize);
        let prev = component_row(component, image, prev_y);
        let curr = component_row(component, image, sample_y);
        let next = component_row(component, image, next_y);
        upsample_h2v2_fancy_row(prev, curr, next, out.len(), y % 2 == 1, out);
    } else {
        upsample_nearest(plan, component, image, y, out);
    }
}

fn component_row<'a>(
    component: &PreparedProgressiveComponentPlan,
    image: &'a ComponentImage,
    y: usize,
) -> &'a [u8] {
    let width = component.sample_width as usize;
    let row_start = y * image.stride;
    &image.plane[row_start..row_start + width]
}

fn upsample_nearest(
    plan: &PreparedProgressivePlan,
    component: &PreparedProgressiveComponentPlan,
    image: &ComponentImage,
    y: u32,
    out: &mut [u8],
) {
    let sample_y = ((y as usize) * usize::from(component.v) / usize::from(plan.sampling.max_v))
        .min(component.sample_height.saturating_sub(1) as usize);
    let row = component_row(component, image, sample_y);
    for (x, dst) in out.iter_mut().enumerate() {
        let sample_x = (x * usize::from(component.h) / usize::from(plan.sampling.max_h))
            .min(row.len().saturating_sub(1));
        *dst = row[sample_x];
    }
}
