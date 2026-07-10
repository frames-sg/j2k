// SPDX-License-Identifier: MIT OR Apache-2.0

//! Extended-precision component-plane construction and progressive rendering.

use super::super::{
    checked_scratch_len, finish_scan, BitReader, CoefficientBlock, JpegError, PreparedDecodePlan,
    PreparedProgressivePlan, SofKind, Warning, ZIGZAG,
};
use super::state::{decode_extended12_block_pixels, Extended12RestartTracker};
use alloc::vec;
use alloc::vec::Vec;

pub(super) struct Extended12Plane {
    pub(super) pixels: Vec<u16>,
    pub(super) stride: usize,
    pub(super) width: usize,
}

pub(super) fn decode_extended12_color_planes(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    sof: SofKind,
) -> Result<([Extended12Plane; 3], Vec<Warning>), JpegError> {
    if plan.components.len() != 3 {
        return Err(JpegError::NotImplemented { sof });
    }
    let mut planes = extended12_planes_for_sequential_plan(plan, sof)?;
    let mut br = BitReader::new(scan_bytes);
    let mut prev_dc = [0i32; 3];
    let mut coeffs: [CoefficientBlock; 3] = core::array::from_fn(|_| CoefficientBlock::default());
    let mut pixels = [[0u16; 64]; 3];
    let mcu_cols = plan
        .dimensions
        .0
        .div_ceil(u32::from(plan.sampling.max_h) * 8);
    let mcu_rows = plan
        .dimensions
        .1
        .div_ceil(u32::from(plan.sampling.max_v) * 8);
    let total_mcus = mcu_cols * mcu_rows;
    let mut restart_tracker = Extended12RestartTracker::new(plan.restart_interval, total_mcus);

    for mcu_y in 0..mcu_rows {
        for mcu_x in 0..mcu_cols {
            let current_mcu = mcu_y * mcu_cols + mcu_x;
            if restart_tracker.begin_mcu(&mut br, current_mcu)? {
                prev_dc.fill(0);
            }
            for component in &plan.components {
                let output_index = component.output_index;
                if output_index > 2 {
                    return Err(JpegError::NotImplemented { sof });
                }
                for by in 0..u32::from(component.v) {
                    for bx in 0..u32::from(component.h) {
                        decode_extended12_block_pixels(
                            &mut br,
                            component,
                            &mut prev_dc[output_index],
                            &mut coeffs[output_index],
                            &mut pixels[output_index],
                        )?;
                        deposit_extended12_block(
                            &mut planes[output_index],
                            (mcu_x * u32::from(component.h) + bx) as usize * 8,
                            (mcu_y * u32::from(component.v) + by) as usize * 8,
                            &pixels[output_index],
                        );
                    }
                }
            }
            restart_tracker.finish_mcu();
        }
    }

    let scan_warnings = finish_scan(&mut br, true)?;
    Ok((planes, scan_warnings))
}

pub(super) fn decode_extended12_four_component_planes(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    sof: SofKind,
) -> Result<([Extended12Plane; 4], Vec<Warning>), JpegError> {
    if plan.components.len() != 4 {
        return Err(JpegError::NotImplemented { sof });
    }
    let mut planes = extended12_four_component_planes_for_sequential_plan(plan, sof)?;
    let mut br = BitReader::new(scan_bytes);
    let mut prev_dc = [0i32; 4];
    let mut coeffs: [CoefficientBlock; 4] = core::array::from_fn(|_| CoefficientBlock::default());
    let mut pixels = [[0u16; 64]; 4];
    let mcu_cols = plan
        .dimensions
        .0
        .div_ceil(u32::from(plan.sampling.max_h) * 8);
    let mcu_rows = plan
        .dimensions
        .1
        .div_ceil(u32::from(plan.sampling.max_v) * 8);
    let total_mcus = mcu_cols * mcu_rows;
    let mut restart_tracker = Extended12RestartTracker::new(plan.restart_interval, total_mcus);

    for mcu_y in 0..mcu_rows {
        for mcu_x in 0..mcu_cols {
            let current_mcu = mcu_y * mcu_cols + mcu_x;
            if restart_tracker.begin_mcu(&mut br, current_mcu)? {
                prev_dc.fill(0);
            }
            for component in &plan.components {
                let output_index = component.output_index;
                if output_index > 3 {
                    return Err(JpegError::NotImplemented { sof });
                }
                for by in 0..u32::from(component.v) {
                    for bx in 0..u32::from(component.h) {
                        decode_extended12_block_pixels(
                            &mut br,
                            component,
                            &mut prev_dc[output_index],
                            &mut coeffs[output_index],
                            &mut pixels[output_index],
                        )?;
                        deposit_extended12_block(
                            &mut planes[output_index],
                            (mcu_x * u32::from(component.h) + bx) as usize * 8,
                            (mcu_y * u32::from(component.v) + by) as usize * 8,
                            &pixels[output_index],
                        );
                    }
                }
            }
            restart_tracker.finish_mcu();
        }
    }

    let scan_warnings = finish_scan(&mut br, true)?;
    Ok((planes, scan_warnings))
}

pub(super) fn extended12_planes_for_sequential_plan(
    plan: &PreparedDecodePlan,
    sof: SofKind,
) -> Result<[Extended12Plane; 3], JpegError> {
    let mcu_cols = plan
        .dimensions
        .0
        .div_ceil(u32::from(plan.sampling.max_h) * 8);
    let mcu_rows = plan
        .dimensions
        .1
        .div_ceil(u32::from(plan.sampling.max_v) * 8);
    let mut widths = [0usize; 3];
    let mut strides = [0usize; 3];
    let mut heights = [0usize; 3];
    let mut lens = [0usize; 3];
    for component in &plan.components {
        if component.output_index > 2 {
            return Err(JpegError::NotImplemented { sof });
        }
        widths[component.output_index] =
            plan.dimensions
                .0
                .saturating_mul(u32::from(component.h))
                .div_ceil(u32::from(plan.sampling.max_h)) as usize;
        strides[component.output_index] =
            checked_scratch_len(&[mcu_cols as usize, usize::from(component.h), 8])?;
        heights[component.output_index] =
            checked_scratch_len(&[mcu_rows as usize, usize::from(component.v), 8])?;
        lens[component.output_index] = checked_scratch_len(&[
            strides[component.output_index],
            heights[component.output_index],
            core::mem::size_of::<u16>(),
        ])? / core::mem::size_of::<u16>();
    }
    Ok(core::array::from_fn(|index| Extended12Plane {
        pixels: vec![0u16; lens[index]],
        stride: strides[index],
        width: widths[index],
    }))
}

pub(super) fn extended12_four_component_planes_for_sequential_plan(
    plan: &PreparedDecodePlan,
    sof: SofKind,
) -> Result<[Extended12Plane; 4], JpegError> {
    let mcu_cols = plan
        .dimensions
        .0
        .div_ceil(u32::from(plan.sampling.max_h) * 8);
    let mcu_rows = plan
        .dimensions
        .1
        .div_ceil(u32::from(plan.sampling.max_v) * 8);
    let mut widths = [0usize; 4];
    let mut strides = [0usize; 4];
    let mut heights = [0usize; 4];
    let mut lens = [0usize; 4];
    for component in &plan.components {
        if component.output_index > 3 {
            return Err(JpegError::NotImplemented { sof });
        }
        widths[component.output_index] =
            plan.dimensions
                .0
                .saturating_mul(u32::from(component.h))
                .div_ceil(u32::from(plan.sampling.max_h)) as usize;
        strides[component.output_index] =
            checked_scratch_len(&[mcu_cols as usize, usize::from(component.h), 8])?;
        heights[component.output_index] =
            checked_scratch_len(&[mcu_rows as usize, usize::from(component.v), 8])?;
        lens[component.output_index] = checked_scratch_len(&[
            strides[component.output_index],
            heights[component.output_index],
            core::mem::size_of::<u16>(),
        ])? / core::mem::size_of::<u16>();
    }
    Ok(core::array::from_fn(|index| Extended12Plane {
        pixels: vec![0u16; lens[index]],
        stride: strides[index],
        width: widths[index],
    }))
}

pub(super) fn render_progressive12_color_planes(
    plan: &PreparedProgressivePlan,
    coeffs: &[Vec<[i32; 64]>],
) -> Result<[Extended12Plane; 3], JpegError> {
    let mut planes = progressive12_color_planes(plan)?;
    let mut dequant = [0i16; 64];
    let mut pixels = [0u16; 64];
    for (component_index, component) in plan.components.iter().enumerate() {
        let output_index = component.output_index;
        if output_index > 2 {
            return Err(JpegError::NotImplemented {
                sof: SofKind::Progressive12,
            });
        }
        for by in 0..component.block_rows as usize {
            for bx in 0..component.block_cols as usize {
                let block_index = by * component.block_cols as usize + bx;
                dequantize_progressive12_block(
                    &coeffs[component_index][block_index],
                    &component.quant,
                    &mut dequant,
                );
                if dequant[1..].iter().all(|&coeff| coeff == 0) {
                    pixels.fill(crate::idct::idct_islow_12bit_dc_only_sample(dequant[0]));
                } else {
                    crate::idct::idct_islow_12bit(&dequant, &mut pixels);
                }
                deposit_extended12_block(&mut planes[output_index], bx * 8, by * 8, &pixels);
            }
        }
    }
    Ok(planes)
}

pub(super) fn render_progressive12_four_component_planes(
    plan: &PreparedProgressivePlan,
    coeffs: &[Vec<[i32; 64]>],
) -> Result<[Extended12Plane; 4], JpegError> {
    let mut planes = progressive12_four_component_planes(plan)?;
    let mut dequant = [0i16; 64];
    let mut pixels = [0u16; 64];
    for (component_index, component) in plan.components.iter().enumerate() {
        let output_index = component.output_index;
        if output_index > 3 {
            return Err(JpegError::NotImplemented {
                sof: SofKind::Progressive12,
            });
        }
        for by in 0..component.block_rows as usize {
            for bx in 0..component.block_cols as usize {
                let block_index = by * component.block_cols as usize + bx;
                dequantize_progressive12_block(
                    &coeffs[component_index][block_index],
                    &component.quant,
                    &mut dequant,
                );
                if dequant[1..].iter().all(|&coeff| coeff == 0) {
                    pixels.fill(crate::idct::idct_islow_12bit_dc_only_sample(dequant[0]));
                } else {
                    crate::idct::idct_islow_12bit(&dequant, &mut pixels);
                }
                deposit_extended12_block(&mut planes[output_index], bx * 8, by * 8, &pixels);
            }
        }
    }
    Ok(planes)
}

pub(super) fn progressive12_color_planes(
    plan: &PreparedProgressivePlan,
) -> Result<[Extended12Plane; 3], JpegError> {
    let mut widths = [0usize; 3];
    let mut strides = [0usize; 3];
    let mut heights = [0usize; 3];
    for component in &plan.components {
        if component.output_index > 2 {
            return Err(JpegError::NotImplemented {
                sof: SofKind::Progressive12,
            });
        }
        widths[component.output_index] = component.sample_width as usize;
        strides[component.output_index] = component.block_cols as usize * 8;
        heights[component.output_index] = component.block_rows as usize * 8;
    }
    Ok(core::array::from_fn(|index| Extended12Plane {
        pixels: vec![0u16; strides[index] * heights[index]],
        stride: strides[index],
        width: widths[index],
    }))
}

pub(super) fn progressive12_four_component_planes(
    plan: &PreparedProgressivePlan,
) -> Result<[Extended12Plane; 4], JpegError> {
    let mut widths = [0usize; 4];
    let mut strides = [0usize; 4];
    let mut heights = [0usize; 4];
    for component in &plan.components {
        if component.output_index > 3 {
            return Err(JpegError::NotImplemented {
                sof: SofKind::Progressive12,
            });
        }
        widths[component.output_index] = component.sample_width as usize;
        strides[component.output_index] = component.block_cols as usize * 8;
        heights[component.output_index] = component.block_rows as usize * 8;
    }
    Ok(core::array::from_fn(|index| Extended12Plane {
        pixels: vec![0u16; strides[index] * heights[index]],
        stride: strides[index],
        width: widths[index],
    }))
}

pub(super) fn deposit_extended12_block(
    plane: &mut Extended12Plane,
    x: usize,
    y: usize,
    block: &[u16; 64],
) {
    for row in 0..8 {
        let dst_start = (y + row) * plane.stride + x;
        let src_start = row * 8;
        plane.pixels[dst_start..dst_start + 8].copy_from_slice(&block[src_start..src_start + 8]);
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "12-bit progressive coefficients are clamped to the i16 storage range before conversion"
)]
pub(super) fn dequantize_progressive12_block(
    coeffs: &[i32; 64],
    quant: &[u16; 64],
    out: &mut [i16; 64],
) {
    out.fill(0);
    for k in 0..64 {
        let natural_idx = usize::from(ZIGZAG[k]);
        let value = coeffs[natural_idx].wrapping_mul(i32::from(quant[k]));
        out[natural_idx] = value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
    }
}
