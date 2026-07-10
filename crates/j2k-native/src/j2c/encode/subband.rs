// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    copy_code_block_coefficients, copy_code_block_coefficients_i64, quantize, vec, BlockCodingMode,
    ComponentRoiEncodeRegion, J2kEncodeStageAccelerator, J2kHtSubbandEncodeJob,
    J2kQuantizeSubbandJob, PreparedEncodeCodeBlock, PreparedEncodeSubband, QuantStepSize,
    SubBandType, Vec,
};

fn apply_roi_maxshift_encode(
    coefficients: &mut [i32],
    width: u32,
    height: u32,
    roi_shift: u8,
    roi_regions: &[ComponentRoiEncodeRegion],
    roi_scale: u32,
) -> Result<(), &'static str> {
    if roi_shift == 0 {
        return Ok(());
    }
    let expected_len = (width as usize)
        .checked_mul(height as usize)
        .ok_or("ROI subband dimensions overflow")?;
    if coefficients.len() != expected_len {
        return Err("ROI subband coefficient length mismatch");
    }
    if roi_regions.is_empty() {
        for coefficient in coefficients {
            shift_roi_coefficient(coefficient, roi_shift)?;
        }
        return Ok(());
    }

    let mut selected = vec![false; coefficients.len()];
    for region in roi_regions {
        let Some((x0, y0, x1, y1)) = roi_region_subband_window(*region, width, height, roi_scale)
        else {
            continue;
        };
        for y in y0..y1 {
            for x in x0..x1 {
                let idx = (y as usize)
                    .checked_mul(width as usize)
                    .and_then(|row| row.checked_add(x as usize))
                    .ok_or("ROI subband index overflow")?;
                if selected[idx] {
                    continue;
                }
                selected[idx] = true;
                shift_roi_coefficient(&mut coefficients[idx], roi_shift)?;
            }
        }
    }
    Ok(())
}

fn apply_roi_maxshift_encode_i64(
    coefficients: &mut [i64],
    width: u32,
    height: u32,
    roi_shift: u8,
    roi_regions: &[ComponentRoiEncodeRegion],
    roi_scale: u32,
) -> Result<(), &'static str> {
    if roi_shift == 0 {
        return Ok(());
    }
    let expected_len = (width as usize)
        .checked_mul(height as usize)
        .ok_or("ROI subband dimensions overflow")?;
    if coefficients.len() != expected_len {
        return Err("ROI subband coefficient length mismatch");
    }
    if roi_regions.is_empty() {
        for coefficient in coefficients {
            shift_roi_coefficient_i64(coefficient, roi_shift)?;
        }
        return Ok(());
    }

    let mut selected = vec![false; coefficients.len()];
    for region in roi_regions {
        let Some((x0, y0, x1, y1)) = roi_region_subband_window(*region, width, height, roi_scale)
        else {
            continue;
        };
        for y in y0..y1 {
            for x in x0..x1 {
                let idx = (y as usize)
                    .checked_mul(width as usize)
                    .and_then(|row| row.checked_add(x as usize))
                    .ok_or("ROI subband index overflow")?;
                if selected[idx] {
                    continue;
                }
                selected[idx] = true;
                shift_roi_coefficient_i64(&mut coefficients[idx], roi_shift)?;
            }
        }
    }
    Ok(())
}

fn shift_roi_coefficient(coefficient: &mut i32, roi_shift: u8) -> Result<(), &'static str> {
    *coefficient = coefficient
        .checked_shl(u32::from(roi_shift))
        .ok_or("ROI maxshift coefficient overflow")?;
    Ok(())
}

fn shift_roi_coefficient_i64(coefficient: &mut i64, roi_shift: u8) -> Result<(), &'static str> {
    let factor = 1_i64
        .checked_shl(u32::from(roi_shift))
        .ok_or("ROI maxshift coefficient overflow")?;
    *coefficient = coefficient
        .checked_mul(factor)
        .ok_or("ROI maxshift coefficient overflow")?;
    Ok(())
}

fn roi_region_subband_window(
    region: ComponentRoiEncodeRegion,
    width: u32,
    height: u32,
    roi_scale: u32,
) -> Option<(u32, u32, u32, u32)> {
    if width == 0 || height == 0 || roi_scale == 0 {
        return None;
    }
    let x1 = region.x.saturating_add(region.width);
    let y1 = region.y.saturating_add(region.height);
    let x0 = (region.x / roi_scale).min(width);
    let y0 = (region.y / roi_scale).min(height);
    let x1 = x1.div_ceil(roi_scale).min(width);
    let y1 = y1.div_ceil(roi_scale).min(height);
    if x0 >= x1 || y0 >= y1 {
        None
    } else {
        Some((x0, y0, x1, y1))
    }
}

pub(super) fn prepare_subband(
    coefficients: &[f32],
    width: u32,
    height: u32,
    step_size: &QuantStepSize,
    bit_depth: u8,
    guard_bits: u8,
    reversible: bool,
    block_coding_mode: BlockCodingMode,
    cb_width: u32,
    cb_height: u32,
    sub_band_type: SubBandType,
    roi_shift: u8,
    roi_regions: &[ComponentRoiEncodeRegion],
    roi_scale: u32,
    ht_target_coding_passes: u8,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<PreparedEncodeSubband, &'static str> {
    if width == 0 || height == 0 {
        return Ok(PreparedEncodeSubband {
            code_blocks: Vec::new(),
            preencoded_ht_code_blocks: None,
            num_cbs_x: 0,
            num_cbs_y: 0,
            code_block_width: cb_width,
            code_block_height: cb_height,
            width,
            height,
            sub_band_type,
            total_bitplanes: 0,
            block_coding_mode,
            ht_target_coding_passes,
        });
    }

    let range_bits = subband_range_bits(bit_depth, sub_band_type);
    debug_assert!(step_size.exponent <= u16::from(u8::MAX));
    let base_total_bitplanes = guard_bits
        .saturating_add(step_size.exponent as u8)
        .saturating_sub(1);
    let total_bitplanes = base_total_bitplanes
        .checked_add(roi_shift)
        .ok_or("ROI maxshift exceeds supported coded bitplane count")?;
    let num_cbs_x = width.div_ceil(cb_width);
    let num_cbs_y = height.div_ceil(cb_height);

    if block_coding_mode == BlockCodingMode::HighThroughput
        && roi_shift == 0
        && ht_target_coding_passes == 1
    {
        if let Some(encoded) = accelerator.encode_ht_subband(J2kHtSubbandEncodeJob {
            coefficients,
            width,
            height,
            step_exponent: step_size.exponent,
            step_mantissa: step_size.mantissa,
            range_bits,
            reversible,
            code_block_width: cb_width,
            code_block_height: cb_height,
            total_bitplanes,
        })? {
            let expected_code_blocks = (num_cbs_x as usize)
                .checked_mul(num_cbs_y as usize)
                .ok_or("code-block count overflow")?;
            if encoded.len() != expected_code_blocks {
                return Err("accelerated HT subband code-block count mismatch");
            }
            return Ok(PreparedEncodeSubband {
                code_blocks: code_block_shapes(width, height, cb_width, cb_height)?,
                preencoded_ht_code_blocks: Some(encoded),
                num_cbs_x,
                num_cbs_y,
                code_block_width: cb_width,
                code_block_height: cb_height,
                width,
                height,
                sub_band_type,
                total_bitplanes,
                block_coding_mode,
                ht_target_coding_passes,
            });
        }
    }

    let mut quantized = match accelerator.encode_quantize_subband(J2kQuantizeSubbandJob {
        coefficients,
        step_exponent: step_size.exponent,
        step_mantissa: step_size.mantissa,
        range_bits,
        reversible,
    })? {
        Some(quantized) => {
            if quantized.len() != coefficients.len() {
                return Err("accelerated quantized subband length mismatch");
            }
            quantized
        }
        None => quantize::quantize_subband(coefficients, step_size, range_bits, reversible),
    };
    apply_roi_maxshift_encode(
        &mut quantized,
        width,
        height,
        roi_shift,
        roi_regions,
        roi_scale,
    )?;

    let mut code_blocks = Vec::with_capacity((num_cbs_x * num_cbs_y) as usize);

    for cby in 0..num_cbs_y {
        for cbx in 0..num_cbs_x {
            let x0 = cbx * cb_width;
            let y0 = cby * cb_height;
            let x1 = (x0 + cb_width).min(width);
            let y1 = (y0 + cb_height).min(height);
            let cbw = x1 - x0;
            let cbh = y1 - y0;

            let cb_coeffs = copy_code_block_coefficients(
                &quantized,
                width as usize,
                x0 as usize,
                y0 as usize,
                cbw as usize,
                cbh as usize,
            )
            .into_iter()
            .map(i64::from)
            .collect();

            code_blocks.push(PreparedEncodeCodeBlock {
                coefficients: cb_coeffs,
                width: cbw,
                height: cbh,
            });
        }
    }

    Ok(PreparedEncodeSubband {
        code_blocks,
        preencoded_ht_code_blocks: None,
        num_cbs_x,
        num_cbs_y,
        code_block_width: cb_width,
        code_block_height: cb_height,
        width,
        height,
        sub_band_type,
        total_bitplanes,
        block_coding_mode,
        ht_target_coding_passes,
    })
}

#[derive(Clone, Copy)]
pub(super) struct I64SubbandEncodeSettings<'a> {
    pub(super) guard_bits: u8,
    pub(super) cb_width: u32,
    pub(super) cb_height: u32,
    pub(super) roi_shift: u8,
    pub(super) roi_regions: &'a [ComponentRoiEncodeRegion],
    pub(super) roi_scale: u32,
    pub(super) block_coding_mode: BlockCodingMode,
    pub(super) ht_target_coding_passes: u8,
}

pub(super) fn prepare_subband_i64(
    coefficients: &[i64],
    width: u32,
    height: u32,
    step_size: &QuantStepSize,
    sub_band_type: SubBandType,
    settings: I64SubbandEncodeSettings<'_>,
) -> Result<PreparedEncodeSubband, &'static str> {
    let I64SubbandEncodeSettings {
        guard_bits,
        cb_width,
        cb_height,
        roi_shift,
        roi_regions,
        roi_scale,
        block_coding_mode,
        ht_target_coding_passes,
    } = settings;
    if width == 0 || height == 0 {
        return Ok(PreparedEncodeSubband {
            code_blocks: Vec::new(),
            preencoded_ht_code_blocks: None,
            num_cbs_x: 0,
            num_cbs_y: 0,
            code_block_width: cb_width,
            code_block_height: cb_height,
            width,
            height,
            sub_band_type,
            total_bitplanes: 0,
            block_coding_mode,
            ht_target_coding_passes,
        });
    }

    debug_assert!(step_size.exponent <= u16::from(u8::MAX));
    let base_total_bitplanes = guard_bits
        .saturating_add(step_size.exponent as u8)
        .saturating_sub(1);
    let total_bitplanes = base_total_bitplanes
        .checked_add(roi_shift)
        .ok_or("ROI maxshift exceeds supported coded bitplane count")?;
    let num_cbs_x = width.div_ceil(cb_width);
    let num_cbs_y = height.div_ceil(cb_height);
    let mut quantized = coefficients.to_vec();
    apply_roi_maxshift_encode_i64(
        &mut quantized,
        width,
        height,
        roi_shift,
        roi_regions,
        roi_scale,
    )?;

    let mut code_blocks = Vec::with_capacity((num_cbs_x * num_cbs_y) as usize);
    for cby in 0..num_cbs_y {
        for cbx in 0..num_cbs_x {
            let x0 = cbx * cb_width;
            let y0 = cby * cb_height;
            let x1 = (x0 + cb_width).min(width);
            let y1 = (y0 + cb_height).min(height);
            let cbw = x1 - x0;
            let cbh = y1 - y0;

            let cb_coeffs = copy_code_block_coefficients_i64(
                &quantized,
                width as usize,
                x0 as usize,
                y0 as usize,
                cbw as usize,
                cbh as usize,
            );

            code_blocks.push(PreparedEncodeCodeBlock {
                coefficients: cb_coeffs,
                width: cbw,
                height: cbh,
            });
        }
    }

    Ok(PreparedEncodeSubband {
        code_blocks,
        preencoded_ht_code_blocks: None,
        num_cbs_x,
        num_cbs_y,
        code_block_width: cb_width,
        code_block_height: cb_height,
        width,
        height,
        sub_band_type,
        total_bitplanes,
        block_coding_mode,
        ht_target_coding_passes,
    })
}

pub(super) fn prepare_subband_cpu_quantized(
    coefficients: &[f32],
    width: u32,
    height: u32,
    step_size: &QuantStepSize,
    bit_depth: u8,
    guard_bits: u8,
    reversible: bool,
    block_coding_mode: BlockCodingMode,
    cb_width: u32,
    cb_height: u32,
    sub_band_type: SubBandType,
) -> Result<PreparedEncodeSubband, &'static str> {
    if width == 0 || height == 0 {
        return Ok(PreparedEncodeSubband {
            code_blocks: Vec::new(),
            preencoded_ht_code_blocks: None,
            num_cbs_x: 0,
            num_cbs_y: 0,
            code_block_width: cb_width,
            code_block_height: cb_height,
            width,
            height,
            sub_band_type,
            total_bitplanes: 0,
            block_coding_mode,
            ht_target_coding_passes: 1,
        });
    }

    let range_bits = subband_range_bits(bit_depth, sub_band_type);
    debug_assert!(step_size.exponent <= u16::from(u8::MAX));
    let total_bitplanes = guard_bits
        .saturating_add(step_size.exponent as u8)
        .saturating_sub(1);
    let num_cbs_x = width.div_ceil(cb_width);
    let num_cbs_y = height.div_ceil(cb_height);
    let quantized = quantize::quantize_subband(coefficients, step_size, range_bits, reversible);
    let mut code_blocks = Vec::with_capacity((num_cbs_x * num_cbs_y) as usize);

    for cby in 0..num_cbs_y {
        for cbx in 0..num_cbs_x {
            let x0 = cbx * cb_width;
            let y0 = cby * cb_height;
            let x1 = (x0 + cb_width).min(width);
            let y1 = (y0 + cb_height).min(height);
            let cbw = x1 - x0;
            let cbh = y1 - y0;
            let cb_coeffs = copy_code_block_coefficients(
                &quantized,
                width as usize,
                x0 as usize,
                y0 as usize,
                cbw as usize,
                cbh as usize,
            )
            .into_iter()
            .map(i64::from)
            .collect();

            code_blocks.push(PreparedEncodeCodeBlock {
                coefficients: cb_coeffs,
                width: cbw,
                height: cbh,
            });
        }
    }

    Ok(PreparedEncodeSubband {
        code_blocks,
        preencoded_ht_code_blocks: None,
        num_cbs_x,
        num_cbs_y,
        code_block_width: cb_width,
        code_block_height: cb_height,
        width,
        height,
        sub_band_type,
        total_bitplanes,
        block_coding_mode,
        ht_target_coding_passes: 1,
    })
}

fn code_block_shapes(
    width: u32,
    height: u32,
    cb_width: u32,
    cb_height: u32,
) -> Result<Vec<PreparedEncodeCodeBlock>, &'static str> {
    let num_cbs_x = width.div_ceil(cb_width);
    let num_cbs_y = height.div_ceil(cb_height);
    let count = (num_cbs_x as usize)
        .checked_mul(num_cbs_y as usize)
        .ok_or("code-block count overflow")?;
    let mut code_blocks = Vec::with_capacity(count);
    for cby in 0..num_cbs_y {
        for cbx in 0..num_cbs_x {
            let x0 = cbx * cb_width;
            let y0 = cby * cb_height;
            let x1 = (x0 + cb_width).min(width);
            let y1 = (y0 + cb_height).min(height);
            code_blocks.push(PreparedEncodeCodeBlock {
                coefficients: Vec::new(),
                width: x1 - x0,
                height: y1 - y0,
            });
        }
    }
    Ok(code_blocks)
}

fn subband_range_bits(bit_depth: u8, sub_band_type: SubBandType) -> u8 {
    let log_gain = match sub_band_type {
        SubBandType::LowLow => 0,
        SubBandType::LowHigh | SubBandType::HighLow => 1,
        SubBandType::HighHigh => 2,
    };

    bit_depth.saturating_add(log_gain)
}
