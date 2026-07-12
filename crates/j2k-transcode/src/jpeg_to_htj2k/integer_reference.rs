// SPDX-License-Identifier: MIT OR Apache-2.0

use super::integer_storage::{
    checked_product, checked_sum, idct_component_samples_i32, validate_band_len,
};
use super::{
    idct_islow_block, integer_dct_job_for_component, record_accelerator_attempt,
    record_accelerator_dispatch, record_cpu_fallback, reversible_lift_53_high_at_fallible,
    reversible_lift_53_i32, reversible_lift_53_low_at_fallible, DctToWaveletStageAccelerator,
    Instant, JpegDctComponent, JpegToHtj2kError, JpegToHtj2kScratch, ReversibleDwt53FirstLevel,
    TranscodeTimingReport,
};
use crate::allocation::{try_vec_reserve_len, try_vec_resize_with, try_vec_with_capacity};

pub(super) struct IntegerWaveletLevel {
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) low_width: usize,
    pub(super) low_height: usize,
    pub(super) high_width: usize,
    pub(super) high_height: usize,
    pub(super) hl: Vec<i32>,
    pub(super) lh: Vec<i32>,
    pub(super) hh: Vec<i32>,
}

pub(super) struct IntegerWavelet {
    pub(super) final_ll: Vec<i32>,
    pub(super) final_ll_width: usize,
    pub(super) final_ll_height: usize,
    pub(super) levels: Vec<IntegerWaveletLevel>,
}

pub(super) fn integer_direct_wavelet_from_component(
    component: &JpegDctComponent,
    decomposition_levels: u8,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut impl DctToWaveletStageAccelerator,
    timings: &mut TranscodeTimingReport,
) -> Result<IntegerWavelet, JpegToHtj2kError> {
    let job = integer_dct_job_for_component(component)?;
    timings.component_count = timings.component_count.saturating_add(1);
    record_accelerator_attempt(timings, 1);
    let accelerator_start = Instant::now();
    let accelerated_first_level = accelerator
        .dct_grid_to_reversible_dwt53(job)
        .map_err(JpegToHtj2kError::Accelerator)?;
    timings.dct_to_wavelet_accelerator_us = timings
        .dct_to_wavelet_accelerator_us
        .saturating_add(accelerator_start.elapsed().as_micros());
    if let Some(first_level) = accelerated_first_level {
        record_accelerator_dispatch(timings, 1);
        let decompose_start = Instant::now();
        let wavelet = integer_wavelet_from_first_level(first_level, decomposition_levels)?;
        timings.dwt_decompose_us = timings
            .dwt_decompose_us
            .saturating_add(decompose_start.elapsed().as_micros());
        return Ok(wavelet);
    }

    scratch.integer_idct_blocks.clear();
    try_vec_resize_with(
        &mut scratch.integer_idct_blocks,
        component.dequantized_blocks.len(),
        || None,
    )?;
    record_cpu_fallback(timings, 1);
    let fallback_start = Instant::now();
    let (final_ll, final_ll_width, final_ll_height, first_level) =
        integer_direct_first_level_from_component(
            component,
            &mut scratch.integer_idct_blocks,
            &mut scratch.integer_row,
        )?;
    timings.dct_to_wavelet_cpu_fallback_us = timings
        .dct_to_wavelet_cpu_fallback_us
        .saturating_add(fallback_start.elapsed().as_micros());
    let decompose_start = Instant::now();
    let wavelet = integer_wavelet_from_first_parts(
        final_ll,
        final_ll_width,
        final_ll_height,
        first_level,
        decomposition_levels,
    )?;
    timings.dwt_decompose_us = timings
        .dwt_decompose_us
        .saturating_add(decompose_start.elapsed().as_micros());
    Ok(wavelet)
}

pub(super) fn integer_wavelet_from_first_level(
    first_level: ReversibleDwt53FirstLevel,
    decomposition_levels: u8,
) -> Result<IntegerWavelet, JpegToHtj2kError> {
    let (final_ll, final_ll_width, final_ll_height, first_level) =
        integer_wavelet_first_level_from_accelerated(first_level)?;
    integer_wavelet_from_first_parts(
        final_ll,
        final_ll_width,
        final_ll_height,
        first_level,
        decomposition_levels,
    )
}

pub(super) fn integer_wavelet_from_first_parts(
    mut final_ll: Vec<i32>,
    mut final_ll_width: usize,
    mut final_ll_height: usize,
    first_level: IntegerWaveletLevel,
    decomposition_levels: u8,
) -> Result<IntegerWavelet, JpegToHtj2kError> {
    let mut levels = try_vec_with_capacity(usize::from(decomposition_levels.max(1)))?;
    levels.push(first_level);

    let remaining_levels = usize::from(decomposition_levels.saturating_sub(1));
    if remaining_levels > 0 {
        let tail =
            reversible_dwt53_i32(final_ll, final_ll_width, final_ll_height, remaining_levels)?;
        final_ll = tail.final_ll;
        final_ll_width = tail.final_ll_width;
        final_ll_height = tail.final_ll_height;
        levels.extend(tail.levels);
    }

    Ok(IntegerWavelet {
        final_ll,
        final_ll_width,
        final_ll_height,
        levels,
    })
}

pub(super) fn integer_wavelet_first_level_from_accelerated(
    first_level: ReversibleDwt53FirstLevel,
) -> Result<(Vec<i32>, usize, usize, IntegerWaveletLevel), JpegToHtj2kError> {
    let width = checked_sum(first_level.low_width, first_level.high_width)?;
    let height = checked_sum(first_level.low_height, first_level.high_height)?;
    validate_band_len(
        &first_level.ll,
        first_level.low_width,
        first_level.low_height,
    )?;
    validate_band_len(
        &first_level.hl,
        first_level.high_width,
        first_level.low_height,
    )?;
    validate_band_len(
        &first_level.lh,
        first_level.low_width,
        first_level.high_height,
    )?;
    validate_band_len(
        &first_level.hh,
        first_level.high_width,
        first_level.high_height,
    )?;
    let level = IntegerWaveletLevel {
        width,
        height,
        low_width: first_level.low_width,
        low_height: first_level.low_height,
        high_width: first_level.high_width,
        high_height: first_level.high_height,
        hl: first_level.hl,
        lh: first_level.lh,
        hh: first_level.hh,
    };
    Ok((
        first_level.ll,
        first_level.low_width,
        first_level.low_height,
        level,
    ))
}

pub(super) fn integer_direct_first_level_from_component(
    component: &JpegDctComponent,
    idct_blocks: &mut [Option<[i32; 64]>],
    row: &mut Vec<i32>,
) -> Result<(Vec<i32>, usize, usize, IntegerWaveletLevel), JpegToHtj2kError> {
    let width = component.width as usize;
    let height = component.height as usize;
    let low_width = width.div_ceil(2);
    let low_height = height.div_ceil(2);
    let high_width = width / 2;
    let high_height = height / 2;

    let mut ll = try_vec_with_capacity(checked_product(low_width, low_height)?)?;
    let mut hl = try_vec_with_capacity(checked_product(high_width, low_height)?)?;
    let mut lh = try_vec_with_capacity(checked_product(low_width, high_height)?)?;
    let mut hh = try_vec_with_capacity(checked_product(high_width, high_height)?)?;
    row.clear();
    try_vec_reserve_len(row, width)?;

    for output_y in 0..low_height {
        row.clear();
        for x in 0..width {
            row.push(vertical_53_i32_at(
                component,
                idct_blocks,
                x,
                output_y,
                true,
            )?);
        }
        reversible_lift_53_i32(row);
        ll.extend(row.iter().step_by(2).copied());
        hl.extend(row.iter().skip(1).step_by(2).copied());
    }

    for output_y in 0..high_height {
        row.clear();
        for x in 0..width {
            row.push(vertical_53_i32_at(
                component,
                idct_blocks,
                x,
                output_y,
                false,
            )?);
        }
        reversible_lift_53_i32(row);
        lh.extend(row.iter().step_by(2).copied());
        hh.extend(row.iter().skip(1).step_by(2).copied());
    }

    let level = IntegerWaveletLevel {
        width,
        height,
        low_width,
        low_height,
        high_width,
        high_height,
        hl,
        lh,
        hh,
    };

    Ok((ll, low_width, low_height, level))
}

pub(super) fn vertical_53_i32_at(
    component: &JpegDctComponent,
    idct_blocks: &mut [Option<[i32; 64]>],
    x: usize,
    output_y: usize,
    low_pass: bool,
) -> Result<i32, JpegToHtj2kError> {
    if low_pass {
        vertical_low_53_i32_at(component, idct_blocks, x, output_y)
    } else {
        vertical_high_53_i32_at(component, idct_blocks, x, output_y)
    }
}

pub(super) fn vertical_low_53_i32_at(
    component: &JpegDctComponent,
    idct_blocks: &mut [Option<[i32; 64]>],
    x: usize,
    low_idx: usize,
) -> Result<i32, JpegToHtj2kError> {
    let height = component.height as usize;
    reversible_lift_53_low_at_fallible(height, low_idx, |y| {
        component_sample_i32(component, idct_blocks, x, y)
    })
}

pub(super) fn vertical_high_53_i32_at(
    component: &JpegDctComponent,
    idct_blocks: &mut [Option<[i32; 64]>],
    x: usize,
    high_idx: usize,
) -> Result<i32, JpegToHtj2kError> {
    let height = component.height as usize;
    reversible_lift_53_high_at_fallible(height, high_idx, |y| {
        component_sample_i32(component, idct_blocks, x, y)
    })
}

pub(super) fn component_sample_i32(
    component: &JpegDctComponent,
    idct_blocks: &mut [Option<[i32; 64]>],
    x: usize,
    y: usize,
) -> Result<i32, JpegToHtj2kError> {
    if x >= component.width as usize || y >= component.height as usize {
        return Err(JpegToHtj2kError::Validation(
            "component sample coordinate exceeds dimensions",
        ));
    }
    let block_cols = component.block_cols as usize;
    let block_x = x / 8;
    let block_y = y / 8;
    let block_idx = block_y * block_cols + block_x;
    let block = component
        .dequantized_blocks
        .get(block_idx)
        .ok_or(JpegToHtj2kError::Validation(
            "component block grid does not cover requested sample",
        ))?;
    let cached = idct_blocks
        .get_mut(block_idx)
        .ok_or(JpegToHtj2kError::Validation(
            "integer IDCT cache does not cover requested block",
        ))?;
    let block_samples = cached.get_or_insert_with(|| {
        let decoded = idct_islow_block(block);
        decoded.map(|sample| i32::from(sample) - 128)
    });
    let local_idx = (y % 8) * 8 + (x % 8);
    Ok(block_samples[local_idx])
}

pub(super) fn integer_reference_coefficients(
    component: &JpegDctComponent,
    decomposition_levels: u8,
) -> Result<Vec<i32>, JpegToHtj2kError> {
    let samples = idct_component_samples_i32(component)?;
    let wavelet = reversible_dwt53_i32(
        samples,
        component.width as usize,
        component.height as usize,
        usize::from(decomposition_levels),
    )?;
    flatten_integer_wavelet(&wavelet)
}

pub(super) fn reversible_dwt53_i32(
    mut buffer: Vec<i32>,
    width: usize,
    height: usize,
    decomposition_levels: usize,
) -> Result<IntegerWavelet, JpegToHtj2kError> {
    let sample_count = checked_product(width, height)?;
    if buffer.len() != sample_count {
        return Err(JpegToHtj2kError::Validation(
            "integer DWT sample count does not match dimensions",
        ));
    }
    let mut current_width = width;
    let mut current_height = height;
    let mut levels = try_vec_with_capacity(decomposition_levels)?;
    let mut column = try_vec_with_capacity(height)?;
    let mut row = try_vec_with_capacity(width)?;

    for _ in 0..decomposition_levels {
        for x in 0..current_width {
            column.clear();
            for y in 0..current_height {
                column.push(buffer[y * width + x]);
            }
            reversible_lift_53_i32(&mut column);
            let low_len = current_height.div_ceil(2);
            for (idx, value) in column.iter().step_by(2).copied().enumerate() {
                buffer[idx * width + x] = value;
            }
            for (idx, value) in column.iter().skip(1).step_by(2).copied().enumerate() {
                buffer[(low_len + idx) * width + x] = value;
            }
        }

        for y in 0..current_height {
            let row_start = y * width;
            row.clear();
            row.extend_from_slice(&buffer[row_start..row_start + current_width]);
            reversible_lift_53_i32(&mut row);
            let low_len = current_width.div_ceil(2);
            for (idx, value) in row.iter().step_by(2).copied().enumerate() {
                buffer[row_start + idx] = value;
            }
            for (idx, value) in row.iter().skip(1).step_by(2).copied().enumerate() {
                buffer[row_start + low_len + idx] = value;
            }
        }

        let low_width = current_width.div_ceil(2);
        let low_height = current_height.div_ceil(2);
        let high_width = current_width / 2;
        let high_height = current_height / 2;
        let mut hl = try_vec_with_capacity(checked_product(high_width, low_height)?)?;
        let mut lh = try_vec_with_capacity(checked_product(low_width, high_height)?)?;
        let mut hh = try_vec_with_capacity(checked_product(high_width, high_height)?)?;

        for y in 0..low_height {
            for x in 0..high_width {
                hl.push(buffer[y * width + low_width + x]);
            }
        }
        for y in 0..high_height {
            for x in 0..low_width {
                lh.push(buffer[(low_height + y) * width + x]);
            }
        }
        for y in 0..high_height {
            for x in 0..high_width {
                hh.push(buffer[(low_height + y) * width + low_width + x]);
            }
        }

        levels.push(IntegerWaveletLevel {
            width: current_width,
            height: current_height,
            low_width,
            low_height,
            high_width,
            high_height,
            hl,
            lh,
            hh,
        });
        current_width = low_width;
        current_height = low_height;
    }

    let mut final_ll = try_vec_with_capacity(checked_product(current_width, current_height)?)?;
    for y in 0..current_height {
        for x in 0..current_width {
            final_ll.push(buffer[y * width + x]);
        }
    }

    Ok(IntegerWavelet {
        final_ll,
        final_ll_width: current_width,
        final_ll_height: current_height,
        levels,
    })
}

pub(super) fn flatten_integer_wavelet(
    wavelet: &IntegerWavelet,
) -> Result<Vec<i32>, JpegToHtj2kError> {
    let mut coefficient_count = wavelet.final_ll.len();
    for level in &wavelet.levels {
        coefficient_count = checked_sum(coefficient_count, level.hl.len())?;
        coefficient_count = checked_sum(coefficient_count, level.lh.len())?;
        coefficient_count = checked_sum(coefficient_count, level.hh.len())?;
    }
    let mut output = try_vec_with_capacity(coefficient_count)?;
    output.extend_from_slice(&wavelet.final_ll);
    for level in wavelet.levels.iter().rev() {
        output.extend_from_slice(&level.hl);
        output.extend_from_slice(&level.lh);
        output.extend_from_slice(&level.hh);
    }
    Ok(output)
}
