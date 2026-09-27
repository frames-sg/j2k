// SPDX-License-Identifier: MIT OR Apache-2.0

//! Progressive coefficient rendering, upsampling, and row output.

use alloc::vec::Vec;

use crate::allocation::{checked_allocation_len, try_reserve_for_len_with_live_budget};
use crate::backend::Backend;
use crate::color::scaled_sampling::{ScaledComponent, ScaledSampling, Upsample};
use crate::color::upsample::{upsample_h1v2_fancy_row, upsample_h2v1_fancy_row};
use crate::entropy::block::clamp_i16;
use crate::entropy::ZIGZAG;
use crate::error::JpegError;
use crate::idct::downscale;
use crate::info::ColorSpace;
use crate::output::OutputWriter;

use super::allocation::{allocate_component_images, checked_phase_capacity, ComponentImage};
use super::model::PreparedProgressivePlan;

/// IDCTs every block into its component image at the component's reduced
/// size (8, 4, 2 or 1 samples per block side, as libjpeg-turbo chooses).
pub(super) fn render_component_images(
    plan: &PreparedProgressivePlan,
    backend: Backend,
    scaled: &ScaledSampling,
    coeffs: &[Vec<[i32; 64]>],
    coefficient_live_bytes: usize,
) -> Result<Vec<ComponentImage>, JpegError> {
    if coeffs.len() != plan.components.len() {
        return Err(JpegError::InternalInvariant {
            reason: "progressive coefficient/component count mismatch",
        });
    }
    let mut images = allocate_component_images(plan, scaled, coefficient_live_bytes)?;
    for (((index, component), component_coeffs), image) in plan
        .components
        .iter()
        .enumerate()
        .zip(coeffs.iter())
        .zip(images.iter_mut())
    {
        let idct_size = scaled.component(index).idct_size as usize;
        let mut dequant = [0i16; 64];
        let mut pixels = [0u8; 64];
        let mut pixels_4x4 = [0u8; 16];
        let mut pixels_2x2 = [0u8; 4];
        let natural_quant = natural_order_quant(&component.quant);
        for by in 0..component.block_rows as usize {
            for bx in 0..component.block_cols as usize {
                let block_index = by * component.block_cols as usize + bx;
                dequantize_block(&component_coeffs[block_index], &natural_quant, &mut dequant);
                let block: &[u8] = match idct_size {
                    8 => {
                        backend.idct(&dequant, &mut pixels);
                        &pixels
                    }
                    4 => {
                        downscale::idct_islow_4x4(&dequant, &mut pixels_4x4);
                        &pixels_4x4
                    }
                    2 => {
                        downscale::idct_islow_2x2(&dequant, &mut pixels_2x2);
                        &pixels_2x2
                    }
                    _ => {
                        pixels[0] = downscale::idct_islow_1x1(&dequant);
                        &pixels[..1]
                    }
                };
                deposit_block(
                    &mut image.plane,
                    image.stride,
                    bx * idct_size,
                    by * idct_size,
                    idct_size,
                    block,
                );
            }
        }
    }
    Ok(images)
}

/// Quantization table (stored in zigzag order, T.81 B.2.4.1) permuted to the
/// natural order the coefficients are kept in.
fn natural_order_quant(quant: &[u16; 64]) -> [u16; 64] {
    let mut natural = [0u16; 64];
    for (&q, &natural_idx) in quant.iter().zip(&ZIGZAG) {
        natural[usize::from(natural_idx)] = q;
    }
    natural
}

/// Dequantize a natural-order block with a natural-order quant table; a
/// straight lane-wise loop the compiler vectorizes.
fn dequantize_block(coeffs: &[i32; 64], natural_quant: &[u16; 64], out: &mut [i16; 64]) {
    for ((out, &coeff), &q) in out.iter_mut().zip(coeffs).zip(natural_quant) {
        *out = clamp_i16(coeff.wrapping_mul(i32::from(q)));
    }
}

fn deposit_block(plane: &mut [u8], stride: usize, x: usize, y: usize, size: usize, block: &[u8]) {
    for row in 0..size {
        let dst = (y + row) * stride + x;
        let src = row * size;
        plane[dst..dst + size].copy_from_slice(&block[src..src + size]);
    }
}

/// Upsamples the component images to the (scaled) output grid row by row.
pub(super) fn emit_component_images<W: OutputWriter>(
    plan: &PreparedProgressivePlan,
    backend: Backend,
    scaled: &ScaledSampling,
    output_dimensions: (u32, u32),
    images: &[ComponentImage],
    image_live_bytes: usize,
    writer: &mut W,
) -> Result<(), JpegError> {
    let (width, height) = output_dimensions;
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
            upsample_component_row(backend, scaled.component(0), image, y, &mut gray);
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
        upsample_component_row(backend, scaled.component(first), &images[first], y, &mut a);
        upsample_component_row(
            backend,
            scaled.component(second),
            &images[second],
            y,
            &mut b,
        );
        upsample_component_row(backend, scaled.component(third), &images[third], y, &mut c);
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

/// Emits output row `y` of one component with libjpeg-turbo's upsampler for
/// it; rows past the component's last sample row replicate that row.
fn upsample_component_row(
    backend: Backend,
    component: ScaledComponent,
    image: &ComponentImage,
    y: u32,
    out: &mut [u8],
) {
    let last_row = component.height.saturating_sub(1) as usize;
    let row = |sample_y: usize| component_row(component, image, sample_y.min(last_row));
    let y = y as usize;
    match component.upsample {
        Upsample::None => out.copy_from_slice(&row(y)[..out.len()]),
        Upsample::FancyH2V1 => upsample_h2v1_fancy_row(row(y), out.len(), out),
        Upsample::FancyH2V2 => {
            let sample_y = y / 2;
            let rows = [
                row(sample_y.saturating_sub(1)),
                row(sample_y),
                row(sample_y + 1),
            ];
            backend.upsample_h2v2_fancy_row(rows, out.len(), y % 2 == 1, out);
        }
        Upsample::FancyH1V2 => {
            let sample_y = y / 2;
            upsample_h1v2_fancy_row(
                row(sample_y.saturating_sub(1)),
                row(sample_y),
                row(sample_y + 1),
                out.len(),
                y % 2 == 1,
                out,
            );
        }
        Upsample::Replicate => {
            let samples = row(y / component.v_ratio as usize);
            let h_ratio = component.h_ratio as usize;
            for (x, dst) in out.iter_mut().enumerate() {
                *dst = samples[(x / h_ratio).min(samples.len() - 1)];
            }
        }
    }
}

fn component_row(component: ScaledComponent, image: &ComponentImage, y: usize) -> &[u8] {
    let row_start = y * image.stride;
    &image.plane[row_start..row_start + component.width as usize]
}
