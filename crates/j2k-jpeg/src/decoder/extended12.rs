// SPDX-License-Identifier: MIT OR Apache-2.0

use super::lossless_helpers::{upsample_h2v1_u16_at, Extended12RestartTracker};
use super::{
    checked_scratch_len, decode_block_with_activity, finish_scan, BitReader, BlockActivity,
    CoefficientBlock, ColorSpace, DownscaleFactor, Info, JpegError, LosslessColorSampling,
    PreparedComponentPlan, PreparedDecodePlan, PreparedProgressivePlan, Rect, SofKind, Vec,
    Warning, ZIGZAG,
};

#[derive(Debug, Clone, Copy)]
pub(super) enum Extended12Output {
    Gray16,
    Rgb16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Extended12ColorSampling {
    S444,
    S422,
    S420,
}

pub(super) fn lossless_color_sampling(info: &Info) -> Option<LosslessColorSampling> {
    if info.sampling.len() != 3 {
        return None;
    }
    match (
        info.sampling.max_h,
        info.sampling.max_v,
        info.sampling.components(),
    ) {
        (1, 1, &[(1, 1), (1, 1), (1, 1)]) => Some(LosslessColorSampling::S444),
        (2, 1, &[(2, 1), (1, 1), (1, 1)])
            if matches!(info.bit_depth, 8 | 16) && info.dimensions.0.is_multiple_of(2) =>
        {
            Some(LosslessColorSampling::S422)
        }
        (2, 2, &[(2, 2), (1, 1), (1, 1)])
            if matches!(info.bit_depth, 8 | 16)
                && info.dimensions.0.is_multiple_of(2)
                && info.dimensions.1.is_multiple_of(2) =>
        {
            Some(LosslessColorSampling::S420)
        }
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum Extended12RgbProjection {
    Identity,
    YCbCr,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Extended12WriteRegion {
    pub(super) output_rect: Rect,
    pub(super) dimensions: (u32, u32),
    pub(super) downscale: DownscaleFactor,
    pub(super) output: Extended12Output,
}

pub(super) struct Extended12Plane {
    pub(super) pixels: Vec<u16>,
    pub(super) stride: usize,
    pub(super) width: usize,
}

pub(super) fn decode_extended12_block_pixels(
    br: &mut BitReader<'_>,
    component: &PreparedComponentPlan,
    prev_dc: &mut i32,
    coeff: &mut CoefficientBlock,
    pixels: &mut [u16; 64],
) -> Result<(), JpegError> {
    let activity = decode_block_with_activity(
        br,
        &component.dc_table,
        &component.ac_table,
        prev_dc,
        component.quant.as_ref(),
        coeff,
    )?;
    match activity {
        BlockActivity::DcOnly => {
            pixels.fill(crate::idct::idct_islow_12bit_dc_only_sample(
                coeff.dc_coeff(),
            ));
        }
        BlockActivity::BottomHalfZero | BlockActivity::General => {
            crate::idct::idct_islow_12bit(coeff.coefficients(), pixels);
        }
    }
    Ok(())
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

pub(super) fn validate_extended12_color444_plan(
    plan: &PreparedDecodePlan,
    sof: SofKind,
) -> Result<(), JpegError> {
    if plan.components.len() != 3 || plan.sampling.max_h != 1 || plan.sampling.max_v != 1 {
        return Err(JpegError::NotImplemented { sof });
    }
    let mut seen = [false; 3];
    for component in &plan.components {
        if component.h != 1 || component.v != 1 || component.output_index > 2 {
            return Err(JpegError::NotImplemented { sof });
        }
        if seen[component.output_index] {
            return Err(JpegError::NotImplemented { sof });
        }
        seen[component.output_index] = true;
    }
    if seen.iter().any(|&present| !present) {
        return Err(JpegError::NotImplemented { sof });
    }
    Ok(())
}

pub(super) fn validate_extended12_four_component444_plan(
    plan: &PreparedDecodePlan,
    sof: SofKind,
) -> Result<(), JpegError> {
    if extended12_four_component_sampling(plan, sof)? != Extended12ColorSampling::S444 {
        return Err(JpegError::NotImplemented { sof });
    }
    Ok(())
}

pub(super) fn extended12_color_sampling(
    plan: &PreparedDecodePlan,
    sof: SofKind,
) -> Result<Extended12ColorSampling, JpegError> {
    if plan.components.len() != 3 {
        return Err(JpegError::NotImplemented { sof });
    }
    let components = color_component_sampling_from_sequential(plan, sof)?;
    color_sampling_from_components(plan.sampling.max_h, plan.sampling.max_v, components, sof)
}

pub(super) fn extended12_four_component_sampling(
    plan: &PreparedDecodePlan,
    sof: SofKind,
) -> Result<Extended12ColorSampling, JpegError> {
    if plan.components.len() != 4 {
        return Err(JpegError::NotImplemented { sof });
    }
    let components = four_component_sampling_from_sequential(plan, sof)?;
    four_component_sampling_from_components(
        plan.sampling.max_h,
        plan.sampling.max_v,
        components,
        sof,
    )
}

pub(super) fn color_component_sampling_from_sequential(
    plan: &PreparedDecodePlan,
    sof: SofKind,
) -> Result<[(u8, u8); 3], JpegError> {
    let mut components = [(0u8, 0u8); 3];
    let mut seen = [false; 3];
    for component in &plan.components {
        if component.output_index > 2 || seen[component.output_index] {
            return Err(JpegError::NotImplemented { sof });
        }
        seen[component.output_index] = true;
        components[component.output_index] = (component.h, component.v);
    }
    if seen.iter().any(|&present| !present) {
        return Err(JpegError::NotImplemented { sof });
    }
    Ok(components)
}

pub(super) fn four_component_sampling_from_sequential(
    plan: &PreparedDecodePlan,
    sof: SofKind,
) -> Result<[(u8, u8); 4], JpegError> {
    let mut components = [(0u8, 0u8); 4];
    let mut seen = [false; 4];
    for component in &plan.components {
        if component.output_index > 3 || seen[component.output_index] {
            return Err(JpegError::NotImplemented { sof });
        }
        seen[component.output_index] = true;
        components[component.output_index] = (component.h, component.v);
    }
    if seen.iter().any(|&present| !present) {
        return Err(JpegError::NotImplemented { sof });
    }
    Ok(components)
}

pub(super) fn progressive_color_sampling(
    plan: &PreparedProgressivePlan,
    sof: SofKind,
) -> Result<Extended12ColorSampling, JpegError> {
    if plan.components.len() != 3 {
        return Err(JpegError::NotImplemented { sof });
    }
    let components = color_component_sampling_from_progressive(plan, sof)?;
    color_sampling_from_components(plan.sampling.max_h, plan.sampling.max_v, components, sof)
}

pub(super) fn progressive_four_component_sampling(
    plan: &PreparedProgressivePlan,
    sof: SofKind,
) -> Result<Extended12ColorSampling, JpegError> {
    if plan.components.len() != 4 {
        return Err(JpegError::NotImplemented { sof });
    }
    let components = four_component_sampling_from_progressive(plan, sof)?;
    four_component_sampling_from_components(
        plan.sampling.max_h,
        plan.sampling.max_v,
        components,
        sof,
    )
}

pub(super) fn color_component_sampling_from_progressive(
    plan: &PreparedProgressivePlan,
    sof: SofKind,
) -> Result<[(u8, u8); 3], JpegError> {
    let mut components = [(0u8, 0u8); 3];
    let mut seen = [false; 3];
    for component in &plan.components {
        if component.output_index > 2 || seen[component.output_index] {
            return Err(JpegError::NotImplemented { sof });
        }
        seen[component.output_index] = true;
        components[component.output_index] = (component.h, component.v);
    }
    if seen.iter().any(|&present| !present) {
        return Err(JpegError::NotImplemented { sof });
    }
    Ok(components)
}

pub(super) fn four_component_sampling_from_progressive(
    plan: &PreparedProgressivePlan,
    sof: SofKind,
) -> Result<[(u8, u8); 4], JpegError> {
    let mut components = [(0u8, 0u8); 4];
    let mut seen = [false; 4];
    for component in &plan.components {
        if component.output_index > 3 || seen[component.output_index] {
            return Err(JpegError::NotImplemented { sof });
        }
        seen[component.output_index] = true;
        components[component.output_index] = (component.h, component.v);
    }
    if seen.iter().any(|&present| !present) {
        return Err(JpegError::NotImplemented { sof });
    }
    Ok(components)
}

pub(super) fn color_sampling_from_components(
    max_h: u8,
    max_v: u8,
    components: [(u8, u8); 3],
    sof: SofKind,
) -> Result<Extended12ColorSampling, JpegError> {
    match (max_h, max_v, components) {
        (1, 1, [(1, 1), (1, 1), (1, 1)]) => Ok(Extended12ColorSampling::S444),
        (2, 1, [(2, 1), (1, 1), (1, 1)]) => Ok(Extended12ColorSampling::S422),
        (2, 2, [(2, 2), (1, 1), (1, 1)]) => Ok(Extended12ColorSampling::S420),
        _ => Err(JpegError::NotImplemented { sof }),
    }
}

pub(super) fn four_component_sampling_from_components(
    max_h: u8,
    max_v: u8,
    components: [(u8, u8); 4],
    sof: SofKind,
) -> Result<Extended12ColorSampling, JpegError> {
    match (max_h, max_v, components) {
        (1, 1, [(1, 1), (1, 1), (1, 1), (1, 1)]) => Ok(Extended12ColorSampling::S444),
        (2, 1, [(2, 1), (1, 1), (1, 1), (1, 1)]) => Ok(Extended12ColorSampling::S422),
        (2, 2, [(2, 2), (1, 1), (1, 1), (1, 1)]) => Ok(Extended12ColorSampling::S420),
        _ => Err(JpegError::NotImplemented { sof }),
    }
}

pub(super) fn progressive_color_component_indices(
    plan: &PreparedProgressivePlan,
) -> Result<[usize; 3], JpegError> {
    let mut indices = [usize::MAX; 3];
    for (component_index, component) in plan.components.iter().enumerate() {
        if component.output_index < 3 {
            if indices[component.output_index] != usize::MAX {
                return Err(JpegError::NotImplemented {
                    sof: SofKind::Progressive12,
                });
            }
            indices[component.output_index] = component_index;
        }
    }
    if indices.contains(&usize::MAX) {
        return Err(JpegError::NotImplemented {
            sof: SofKind::Progressive12,
        });
    }
    Ok(indices)
}

pub(super) fn dequantize_progressive12_block(
    coeffs: &[i32; 64],
    quant: &[u16; 64],
    out: &mut [i16; 64],
) {
    out.fill(0);
    for k in 0..64 {
        let natural_idx = usize::from(ZIGZAG[k]);
        let value = coeffs[natural_idx].wrapping_mul(i32::from(quant[k]));
        out[natural_idx] = value.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
}

pub(super) fn write_extended12_rgb_block_region(
    out: &mut [u8],
    stride: usize,
    region: Extended12WriteRegion,
    projection: Extended12RgbProjection,
    block_origin: (u32, u32),
    pixels: &[[u16; 64]; 3],
) {
    let (width, height) = region.dimensions;
    let (x0, y0) = block_origin;
    let block_x1 = (x0 + 8).min(width);
    let block_y1 = (y0 + 8).min(height);
    let denom = region.downscale.denominator();
    let output_rect = region.output_rect;
    for output_y in output_rect.y..output_rect.y + output_rect.h {
        let source_y = output_y.saturating_mul(denom).min(height - 1);
        if source_y < y0 || source_y >= block_y1 {
            continue;
        }
        let src_row = (source_y - y0) as usize;
        let dst_row = (output_y - output_rect.y) as usize;
        for output_x in output_rect.x..output_rect.x + output_rect.w {
            let source_x = output_x.saturating_mul(denom).min(width - 1);
            if source_x < x0 || source_x >= block_x1 {
                continue;
            }
            let src_col = (source_x - x0) as usize;
            let src_index = src_row * 8 + src_col;
            let dst_col = (output_x - output_rect.x) as usize;
            let dst_start = dst_row * stride + dst_col * 6;
            let dst = &mut out[dst_start..dst_start + 6];
            let (r, g, b) = match projection {
                Extended12RgbProjection::Identity => (
                    pixels[0][src_index],
                    pixels[1][src_index],
                    pixels[2][src_index],
                ),
                Extended12RgbProjection::YCbCr => crate::color::ycbcr::ycbcr12_to_rgb16(
                    pixels[0][src_index],
                    pixels[1][src_index],
                    pixels[2][src_index],
                ),
            };
            dst[0..2].copy_from_slice(&r.to_le_bytes());
            dst[2..4].copy_from_slice(&g.to_le_bytes());
            dst[4..6].copy_from_slice(&b.to_le_bytes());
        }
    }
}

pub(super) fn write_extended12_four_component_block_region(
    out: &mut [u8],
    stride: usize,
    region: Extended12WriteRegion,
    color_space: ColorSpace,
    block_origin: (u32, u32),
    pixels: &[[u16; 64]; 4],
) {
    let (width, height) = region.dimensions;
    let (x0, y0) = block_origin;
    let block_x1 = (x0 + 8).min(width);
    let block_y1 = (y0 + 8).min(height);
    let denom = region.downscale.denominator();
    let output_rect = region.output_rect;
    for output_y in output_rect.y..output_rect.y + output_rect.h {
        let source_y = output_y.saturating_mul(denom).min(height - 1);
        if source_y < y0 || source_y >= block_y1 {
            continue;
        }
        let src_row = (source_y - y0) as usize;
        let dst_row = (output_y - output_rect.y) as usize;
        for output_x in output_rect.x..output_rect.x + output_rect.w {
            let source_x = output_x.saturating_mul(denom).min(width - 1);
            if source_x < x0 || source_x >= block_x1 {
                continue;
            }
            let src_col = (source_x - x0) as usize;
            let src_index = src_row * 8 + src_col;
            let (r, g, b) = match color_space {
                ColorSpace::Cmyk => crate::color::cmyk::inverted_cmyk12_to_rgb16(
                    pixels[0][src_index],
                    pixels[1][src_index],
                    pixels[2][src_index],
                    pixels[3][src_index],
                ),
                ColorSpace::Ycck => crate::color::cmyk::ycck12_to_rgb16(
                    pixels[0][src_index],
                    pixels[1][src_index],
                    pixels[2][src_index],
                    pixels[3][src_index],
                ),
                _ => unreachable!("12-bit four-component path only accepts CMYK/YCCK"),
            };
            let dst_col = (output_x - output_rect.x) as usize;
            let dst_start = dst_row * stride + dst_col * 6;
            let dst = &mut out[dst_start..dst_start + 6];
            dst[0..2].copy_from_slice(&r.to_le_bytes());
            dst[2..4].copy_from_slice(&g.to_le_bytes());
            dst[4..6].copy_from_slice(&b.to_le_bytes());
        }
    }
}

pub(super) fn write_extended12_color422_planes_region(
    out: &mut [u8],
    stride: usize,
    region: Extended12WriteRegion,
    projection: Extended12RgbProjection,
    planes: &[Extended12Plane; 3],
) {
    let (width, height) = region.dimensions;
    let denom = region.downscale.denominator();
    let output_rect = region.output_rect;
    for output_y in output_rect.y..output_rect.y + output_rect.h {
        let source_y = output_y.saturating_mul(denom).min(height - 1) as usize;
        let dst_row = (output_y - output_rect.y) as usize;
        for output_x in output_rect.x..output_rect.x + output_rect.w {
            let source_x = output_x.saturating_mul(denom).min(width - 1) as usize;
            let y = planes[0].pixels[source_y * planes[0].stride + source_x];
            let chroma_y = source_y.min(planes[1].pixels.len() / planes[1].stride - 1);
            let cb_row = &planes[1].pixels
                [chroma_y * planes[1].stride..chroma_y * planes[1].stride + planes[1].width];
            let cr_row = &planes[2].pixels
                [chroma_y * planes[2].stride..chroma_y * planes[2].stride + planes[2].width];
            let c1 = upsample_h2v1_u16_at(cb_row, source_x);
            let c2 = upsample_h2v1_u16_at(cr_row, source_x);
            let (r, g, b) = match projection {
                Extended12RgbProjection::Identity => (y, c1, c2),
                Extended12RgbProjection::YCbCr => crate::color::ycbcr::ycbcr12_to_rgb16(y, c1, c2),
            };
            let dst_col = (output_x - output_rect.x) as usize;
            let dst_start = dst_row * stride + dst_col * 6;
            let dst = &mut out[dst_start..dst_start + 6];
            dst[0..2].copy_from_slice(&r.to_le_bytes());
            dst[2..4].copy_from_slice(&g.to_le_bytes());
            dst[4..6].copy_from_slice(&b.to_le_bytes());
        }
    }
}

pub(super) fn write_extended12_color420_planes_region(
    out: &mut [u8],
    stride: usize,
    region: Extended12WriteRegion,
    projection: Extended12RgbProjection,
    planes: &[Extended12Plane; 3],
) {
    let (width, height) = region.dimensions;
    let denom = region.downscale.denominator();
    let output_rect = region.output_rect;
    for output_y in output_rect.y..output_rect.y + output_rect.h {
        let source_y = output_y.saturating_mul(denom).min(height - 1) as usize;
        let dst_row = (output_y - output_rect.y) as usize;
        for output_x in output_rect.x..output_rect.x + output_rect.w {
            let source_x = output_x.saturating_mul(denom).min(width - 1) as usize;
            let y = planes[0].pixels[source_y * planes[0].stride + source_x];
            let chroma_height = planes[1].pixels.len() / planes[1].stride;
            let chroma_y = (source_y / 2).min(chroma_height - 1);
            let prev_y = chroma_y.saturating_sub(1);
            let next_y = (chroma_y + 1).min(chroma_height - 1);
            let c1 = upsample_h2v2_u16_rows_at(
                extended12_plane_row(&planes[1], prev_y),
                extended12_plane_row(&planes[1], chroma_y),
                extended12_plane_row(&planes[1], next_y),
                source_x,
                !source_y.is_multiple_of(2),
            );
            let c2 = upsample_h2v2_u16_rows_at(
                extended12_plane_row(&planes[2], prev_y),
                extended12_plane_row(&planes[2], chroma_y),
                extended12_plane_row(&planes[2], next_y),
                source_x,
                !source_y.is_multiple_of(2),
            );
            let (r, g, b) = match projection {
                Extended12RgbProjection::Identity => (y, c1, c2),
                Extended12RgbProjection::YCbCr => crate::color::ycbcr::ycbcr12_to_rgb16(y, c1, c2),
            };
            let dst_col = (output_x - output_rect.x) as usize;
            let dst_start = dst_row * stride + dst_col * 6;
            let dst = &mut out[dst_start..dst_start + 6];
            dst[0..2].copy_from_slice(&r.to_le_bytes());
            dst[2..4].copy_from_slice(&g.to_le_bytes());
            dst[4..6].copy_from_slice(&b.to_le_bytes());
        }
    }
}

pub(super) fn write_extended12_four_component_planes_region(
    out: &mut [u8],
    stride: usize,
    region: Extended12WriteRegion,
    color_space: ColorSpace,
    sampling: Extended12ColorSampling,
    planes: &[Extended12Plane; 4],
) {
    let (width, height) = region.dimensions;
    let denom = region.downscale.denominator();
    let output_rect = region.output_rect;
    for output_y in output_rect.y..output_rect.y + output_rect.h {
        let source_y = output_y.saturating_mul(denom).min(height - 1) as usize;
        let dst_row = (output_y - output_rect.y) as usize;
        for output_x in output_rect.x..output_rect.x + output_rect.w {
            let source_x = output_x.saturating_mul(denom).min(width - 1) as usize;
            let c0 = planes[0].pixels[source_y * planes[0].stride + source_x];
            let (c1, c2, c3) = match sampling {
                Extended12ColorSampling::S444 => (
                    sample_extended12_plane_at(&planes[1], source_x, source_y),
                    sample_extended12_plane_at(&planes[2], source_x, source_y),
                    sample_extended12_plane_at(&planes[3], source_x, source_y),
                ),
                Extended12ColorSampling::S422 => (
                    upsample_extended12_plane_h2v1_at(&planes[1], source_x, source_y),
                    upsample_extended12_plane_h2v1_at(&planes[2], source_x, source_y),
                    upsample_extended12_plane_h2v1_at(&planes[3], source_x, source_y),
                ),
                Extended12ColorSampling::S420 => (
                    upsample_extended12_plane_h2v2_at(&planes[1], source_x, source_y),
                    upsample_extended12_plane_h2v2_at(&planes[2], source_x, source_y),
                    upsample_extended12_plane_h2v2_at(&planes[3], source_x, source_y),
                ),
            };
            let (r, g, b) = match color_space {
                ColorSpace::Cmyk => crate::color::cmyk::inverted_cmyk12_to_rgb16(c0, c1, c2, c3),
                ColorSpace::Ycck => crate::color::cmyk::ycck12_to_rgb16(c0, c1, c2, c3),
                _ => unreachable!("12-bit four-component plane path only accepts CMYK/YCCK"),
            };
            let dst_col = (output_x - output_rect.x) as usize;
            let dst_start = dst_row * stride + dst_col * 6;
            let dst = &mut out[dst_start..dst_start + 6];
            dst[0..2].copy_from_slice(&r.to_le_bytes());
            dst[2..4].copy_from_slice(&g.to_le_bytes());
            dst[4..6].copy_from_slice(&b.to_le_bytes());
        }
    }
}

pub(super) fn sample_extended12_plane_at(
    plane: &Extended12Plane,
    source_x: usize,
    source_y: usize,
) -> u16 {
    let height = plane.pixels.len() / plane.stride;
    let y = source_y.min(height - 1);
    let x = source_x.min(plane.width - 1);
    plane.pixels[y * plane.stride + x]
}

pub(super) fn upsample_extended12_plane_h2v1_at(
    plane: &Extended12Plane,
    source_x: usize,
    source_y: usize,
) -> u16 {
    let height = plane.pixels.len() / plane.stride;
    let y = source_y.min(height - 1);
    upsample_h2v1_u16_at(extended12_plane_row(plane, y), source_x)
}

pub(super) fn upsample_extended12_plane_h2v2_at(
    plane: &Extended12Plane,
    source_x: usize,
    source_y: usize,
) -> u16 {
    let height = plane.pixels.len() / plane.stride;
    let chroma_y = (source_y / 2).min(height - 1);
    let prev_y = chroma_y.saturating_sub(1);
    let next_y = (chroma_y + 1).min(height - 1);
    upsample_h2v2_u16_rows_at(
        extended12_plane_row(plane, prev_y),
        extended12_plane_row(plane, chroma_y),
        extended12_plane_row(plane, next_y),
        source_x,
        !source_y.is_multiple_of(2),
    )
}

pub(super) fn extended12_plane_row(plane: &Extended12Plane, y: usize) -> &[u16] {
    let row_start = y * plane.stride;
    &plane.pixels[row_start..row_start + plane.width]
}

pub(super) trait UpsampleSample: Copy {
    fn to_u32(self) -> u32;
    fn from_u32(value: u32) -> Self;
}

impl UpsampleSample for u8 {
    fn to_u32(self) -> u32 {
        u32::from(self)
    }

    fn from_u32(value: u32) -> Self {
        value as u8
    }
}

impl UpsampleSample for u16 {
    fn to_u32(self) -> u32 {
        u32::from(self)
    }

    fn from_u32(value: u32) -> Self {
        value as u16
    }
}

pub(super) fn upsample_h2v1_sample_at<S: UpsampleSample>(row: &[S], output_x: usize) -> S {
    debug_assert!(!row.is_empty());
    if row.len() == 1 {
        return row[0];
    }
    let sample = output_x / 2;
    if output_x == 0 {
        row[0]
    } else if output_x == row.len() * 2 - 1 {
        row[row.len() - 1]
    } else if output_x.is_multiple_of(2) {
        S::from_u32((3 * row[sample].to_u32() + row[sample - 1].to_u32() + 2) / 4)
    } else {
        S::from_u32((3 * row[sample].to_u32() + row[sample + 1].to_u32() + 2) / 4)
    }
}

pub(super) fn upsample_h2v2_rows_at<S: UpsampleSample>(
    curr: &[S],
    near: &[S],
    output_width: usize,
    output_x: usize,
) -> S {
    debug_assert!(!curr.is_empty());
    debug_assert_eq!(near.len(), curr.len());
    let colsum = |index: usize| 3 * curr[index].to_u32() + near[index].to_u32();
    if curr.len() == 1 {
        return S::from_u32((4 * colsum(0) + 8) >> 4);
    }

    let sample = output_x / 2;
    let this = colsum(sample);
    // Match IJG/libjpeg fancy h2v2 upsampling: left/even samples round with
    // +8, right/odd samples with +7 before >> 4 to preserve bit-identical
    // interpolation at mirrored sample positions.
    match output_x {
        0 => S::from_u32((this * 4 + 8) >> 4),
        _ if output_x == output_width - 1 => S::from_u32((this * 4 + 7) >> 4),
        _ if output_x.is_multiple_of(2) => {
            let last = colsum(sample - 1);
            S::from_u32((this * 3 + last + 8) >> 4)
        }
        _ => {
            let next = colsum(sample + 1);
            S::from_u32((this * 3 + next + 7) >> 4)
        }
    }
}

pub(super) fn upsample_h2v2_u16_rows_at(
    prev: &[u16],
    curr: &[u16],
    next: &[u16],
    output_x: usize,
    output_is_bottom: bool,
) -> u16 {
    debug_assert_eq!(prev.len(), curr.len());
    debug_assert_eq!(next.len(), curr.len());
    let near = if output_is_bottom { next } else { prev };
    upsample_h2v2_rows_at(curr, near, curr.len() * 2, output_x)
}

pub(super) fn write_extended12_block_region(
    out: &mut [u8],
    stride: usize,
    region: Extended12WriteRegion,
    block_origin: (u32, u32),
    pixels: &[u16; 64],
) {
    let (width, height) = region.dimensions;
    let (x0, y0) = block_origin;
    let block_x1 = (x0 + 8).min(width);
    let block_y1 = (y0 + 8).min(height);
    let denom = region.downscale.denominator();
    let bytes_per_pixel = match region.output {
        Extended12Output::Gray16 => 2,
        Extended12Output::Rgb16 => 6,
    };
    let output_rect = region.output_rect;
    for output_y in output_rect.y..output_rect.y + output_rect.h {
        let source_y = output_y.saturating_mul(denom).min(height - 1);
        if source_y < y0 || source_y >= block_y1 {
            continue;
        }
        let src_row = (source_y - y0) as usize;
        let dst_row = (output_y - output_rect.y) as usize;
        for output_x in output_rect.x..output_rect.x + output_rect.w {
            let source_x = output_x.saturating_mul(denom).min(width - 1);
            if source_x < x0 || source_x >= block_x1 {
                continue;
            }
            let src_col = (source_x - x0) as usize;
            let sample = pixels[src_row * 8 + src_col].to_le_bytes();
            let dst_col = (output_x - output_rect.x) as usize;
            let dst_start = dst_row * stride + dst_col * bytes_per_pixel;
            let dst = &mut out[dst_start..dst_start + bytes_per_pixel];
            match region.output {
                Extended12Output::Gray16 => {
                    dst.copy_from_slice(&sample);
                }
                Extended12Output::Rgb16 => {
                    dst[0..2].copy_from_slice(&sample);
                    dst[2..4].copy_from_slice(&sample);
                    dst[4..6].copy_from_slice(&sample);
                }
            }
        }
    }
}
