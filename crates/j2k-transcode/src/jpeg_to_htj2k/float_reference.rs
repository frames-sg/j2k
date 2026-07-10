// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    dct53_grid_error, dct8x8_blocks_then_dwt53_float, dct8x8_blocks_then_dwt97_float,
    dct8x8_blocks_then_dwt97_float_with_scratch, dct8x8_blocks_to_dwt53_float_linear_with_scratch,
    dct97_grid_error, linearized_53_2d_from_plane, linearized_97_2d_from_plane_with_scratch,
    record_accelerator_attempt, record_accelerator_dispatch, record_cpu_fallback, Dct97GridScratch,
    DctGridToDwt53Job, DctGridToDwt97Job, DctToWaveletStageAccelerator, Dwt53TwoDimensional,
    Dwt97TwoDimensional, Instant, IntegerWavelet, J2kForwardDwt53Level, J2kForwardDwt53Output,
    J2kForwardDwt97Level, J2kForwardDwt97Output, JpegDctComponent, JpegToHtj2kError,
    JpegToHtj2kScratch, TranscodeTimingReport,
};

pub(super) struct ComponentWavelet {
    pub(super) final_ll: Vec<f64>,
    pub(super) final_ll_width: usize,
    pub(super) final_ll_height: usize,
    pub(super) levels: Vec<Dwt53TwoDimensional<f64>>,
}

pub(super) struct ComponentWavelet97 {
    pub(super) final_ll: Vec<f64>,
    pub(super) final_ll_width: usize,
    pub(super) final_ll_height: usize,
    pub(super) levels: Vec<Dwt97TwoDimensional<f64>>,
}

pub(super) fn float_direct_wavelet_from_component(
    component: &JpegDctComponent,
    decomposition_levels: u8,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut impl DctToWaveletStageAccelerator,
    timings: &mut TranscodeTimingReport,
) -> Result<ComponentWavelet, JpegToHtj2kError> {
    timings.component_count = timings.component_count.saturating_add(1);
    let repack_start = Instant::now();
    dct_blocks_to_8x8_f64_into(&component.dequantized_blocks, &mut scratch.dct_blocks_f64);
    timings.jpeg_dct_repack_us = timings
        .jpeg_dct_repack_us
        .saturating_add(repack_start.elapsed().as_micros());
    let blocks = &scratch.dct_blocks_f64;
    let job = DctGridToDwt53Job {
        blocks,
        block_cols: component.block_cols as usize,
        block_rows: component.block_rows as usize,
        width: component.width as usize,
        height: component.height as usize,
    };
    record_accelerator_attempt(timings, 1);
    let accelerator_start = Instant::now();
    let accelerated = accelerator
        .dct_grid_to_dwt53(job)
        .map_err(JpegToHtj2kError::Accelerator)?;
    timings.dct_to_wavelet_accelerator_us = timings
        .dct_to_wavelet_accelerator_us
        .saturating_add(accelerator_start.elapsed().as_micros());
    let bands = if let Some(bands) = accelerated {
        record_accelerator_dispatch(timings, 1);
        bands
    } else {
        record_cpu_fallback(timings, 1);
        let fallback_start = Instant::now();
        let bands = dct8x8_blocks_to_dwt53_float_linear_with_scratch(
            blocks,
            component.block_cols as usize,
            component.block_rows as usize,
            component.width as usize,
            component.height as usize,
            &mut scratch.dct53_grid,
        )
        .map_err(dct53_grid_error)?;
        timings.dct_to_wavelet_cpu_fallback_us = timings
            .dct_to_wavelet_cpu_fallback_us
            .saturating_add(fallback_start.elapsed().as_micros());
        bands
    };
    let decompose_start = Instant::now();
    let wavelet = decompose_from_first_level(bands, usize::from(decomposition_levels));
    timings.dwt_decompose_us = timings
        .dwt_decompose_us
        .saturating_add(decompose_start.elapsed().as_micros());
    Ok(wavelet)
}

pub(super) fn float_direct_97_wavelet_from_component(
    component: &JpegDctComponent,
    decomposition_levels: u8,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut impl DctToWaveletStageAccelerator,
    timings: &mut TranscodeTimingReport,
) -> Result<ComponentWavelet97, JpegToHtj2kError> {
    timings.component_count = timings.component_count.saturating_add(1);
    let repack_start = Instant::now();
    dct_blocks_to_8x8_f64_into(&component.dequantized_blocks, &mut scratch.dct_blocks_f64);
    timings.jpeg_dct_repack_us = timings
        .jpeg_dct_repack_us
        .saturating_add(repack_start.elapsed().as_micros());
    let blocks = &scratch.dct_blocks_f64;
    let job = DctGridToDwt97Job {
        blocks,
        block_cols: component.block_cols as usize,
        block_rows: component.block_rows as usize,
        width: component.width as usize,
        height: component.height as usize,
    };
    record_accelerator_attempt(timings, 1);
    let accelerator_start = Instant::now();
    let accelerated = accelerator
        .dct_grid_to_dwt97(job)
        .map_err(JpegToHtj2kError::Accelerator)?;
    timings.dct_to_wavelet_accelerator_us = timings
        .dct_to_wavelet_accelerator_us
        .saturating_add(accelerator_start.elapsed().as_micros());
    let bands = if let Some(bands) = accelerated {
        record_accelerator_dispatch(timings, 1);
        bands
    } else {
        record_cpu_fallback(timings, 1);
        let fallback_start = Instant::now();
        let bands = dct8x8_blocks_then_dwt97_float_with_scratch(
            blocks,
            component.block_cols as usize,
            component.block_rows as usize,
            component.width as usize,
            component.height as usize,
            &mut scratch.dct97_grid,
        )
        .map_err(dct97_grid_error)?;
        timings.dct_to_wavelet_cpu_fallback_us = timings
            .dct_to_wavelet_cpu_fallback_us
            .saturating_add(fallback_start.elapsed().as_micros());
        bands
    };
    let decompose_start = Instant::now();
    let wavelet = decompose_97_from_first_level_with_scratch(
        bands,
        usize::from(decomposition_levels),
        &mut scratch.dct97_grid,
    );
    timings.dwt_decompose_us = timings
        .dwt_decompose_us
        .saturating_add(decompose_start.elapsed().as_micros());
    Ok(wavelet)
}

pub(super) fn float_reference_coefficients(
    component: &JpegDctComponent,
    decomposition_levels: u8,
    scratch: &mut JpegToHtj2kScratch,
) -> Result<Vec<i32>, JpegToHtj2kError> {
    dct_blocks_to_8x8_f64_into(&component.dequantized_blocks, &mut scratch.dct_blocks_f64);
    let blocks = &scratch.dct_blocks_f64;
    let first_reference_level = dct8x8_blocks_then_dwt53_float(
        blocks,
        component.block_cols as usize,
        component.block_rows as usize,
        component.width as usize,
        component.height as usize,
    )
    .map_err(dct53_grid_error)?;
    let reference =
        decompose_from_first_level(first_reference_level, usize::from(decomposition_levels));
    rounded_wavelet_i32(&reference)
}

pub(super) fn float97_reference_coefficients(
    component: &JpegDctComponent,
    decomposition_levels: u8,
    scratch: &mut JpegToHtj2kScratch,
) -> Result<Vec<i32>, JpegToHtj2kError> {
    dct_blocks_to_8x8_f64_into(&component.dequantized_blocks, &mut scratch.dct_blocks_f64);
    let blocks = &scratch.dct_blocks_f64;
    let first_reference_level = dct8x8_blocks_then_dwt97_float(
        blocks,
        component.block_cols as usize,
        component.block_rows as usize,
        component.width as usize,
        component.height as usize,
    )
    .map_err(dct97_grid_error)?;
    let reference =
        decompose_97_from_first_level(first_reference_level, usize::from(decomposition_levels));
    rounded_wavelet97_i32(&reference)
}

pub(super) fn decompose_from_first_level(
    first_level: Dwt53TwoDimensional<f64>,
    decomposition_levels: usize,
) -> ComponentWavelet {
    let mut wavelet = ComponentWavelet {
        final_ll: first_level.ll.clone(),
        final_ll_width: first_level.low_width,
        final_ll_height: first_level.low_height,
        levels: vec![first_level],
    };

    while wavelet.levels.len() < decomposition_levels {
        let next = linearized_53_2d_from_plane(
            &wavelet.final_ll,
            wavelet.final_ll_width,
            wavelet.final_ll_height,
        );
        wavelet.final_ll.clone_from(&next.ll);
        wavelet.final_ll_width = next.low_width;
        wavelet.final_ll_height = next.low_height;
        wavelet.levels.push(next);
    }

    wavelet
}

pub(super) fn decompose_97_from_first_level(
    first_level: Dwt97TwoDimensional<f64>,
    decomposition_levels: usize,
) -> ComponentWavelet97 {
    let mut scratch = Dct97GridScratch::default();
    decompose_97_from_first_level_with_scratch(first_level, decomposition_levels, &mut scratch)
}

pub(super) fn decompose_97_from_first_level_with_scratch(
    first_level: Dwt97TwoDimensional<f64>,
    decomposition_levels: usize,
    scratch: &mut Dct97GridScratch,
) -> ComponentWavelet97 {
    let mut wavelet = ComponentWavelet97 {
        final_ll: first_level.ll.clone(),
        final_ll_width: first_level.low_width,
        final_ll_height: first_level.low_height,
        levels: vec![first_level],
    };

    while wavelet.levels.len() < decomposition_levels {
        let next = linearized_97_2d_from_plane_with_scratch(
            &wavelet.final_ll,
            wavelet.final_ll_width,
            wavelet.final_ll_height,
            scratch,
        );
        wavelet.final_ll.clone_from(&next.ll);
        wavelet.final_ll_width = next.low_width;
        wavelet.final_ll_height = next.low_height;
        wavelet.levels.push(next);
    }

    wavelet
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "validated JPEG component geometry fits u32 and the public coefficient ABI stores f32"
)]
pub(super) fn j2k_dwt_from_wavelet(
    wavelet: &ComponentWavelet,
    width: usize,
    height: usize,
) -> J2kForwardDwt53Output {
    let mut current_width = width;
    let mut current_height = height;
    let mut levels = Vec::with_capacity(wavelet.levels.len());

    for level in &wavelet.levels {
        levels.push(J2kForwardDwt53Level {
            hl: level.hl.iter().map(|&value| value as f32).collect(),
            lh: level.lh.iter().map(|&value| value as f32).collect(),
            hh: level.hh.iter().map(|&value| value as f32).collect(),
            width: current_width as u32,
            height: current_height as u32,
            low_width: level.low_width as u32,
            low_height: level.low_height as u32,
            high_width: level.high_width as u32,
            high_height: level.high_height as u32,
        });
        current_width = level.low_width;
        current_height = level.low_height;
    }
    levels.reverse();

    J2kForwardDwt53Output {
        ll: wavelet.final_ll.iter().map(|&value| value as f32).collect(),
        ll_width: wavelet.final_ll_width as u32,
        ll_height: wavelet.final_ll_height as u32,
        levels,
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "validated JPEG component geometry fits u32 and the public coefficient ABI stores f32"
)]
pub(super) fn j2k_dwt97_from_wavelet(
    wavelet: &ComponentWavelet97,
    width: usize,
    height: usize,
) -> J2kForwardDwt97Output {
    let mut current_width = width;
    let mut current_height = height;
    let mut levels = Vec::with_capacity(wavelet.levels.len());

    for level in &wavelet.levels {
        levels.push(J2kForwardDwt97Level {
            hl: level.hl.iter().map(|&value| value as f32).collect(),
            lh: level.lh.iter().map(|&value| value as f32).collect(),
            hh: level.hh.iter().map(|&value| value as f32).collect(),
            width: current_width as u32,
            height: current_height as u32,
            low_width: level.low_width as u32,
            low_height: level.low_height as u32,
            high_width: level.high_width as u32,
            high_height: level.high_height as u32,
        });
        current_width = level.low_width;
        current_height = level.low_height;
    }
    levels.reverse();

    J2kForwardDwt97Output {
        ll: wavelet.final_ll.iter().map(|&value| value as f32).collect(),
        ll_width: wavelet.final_ll_width as u32,
        ll_height: wavelet.final_ll_height as u32,
        levels,
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "validated JPEG geometry fits u32 and the public coefficient ABI intentionally stores f32"
)]
pub(super) fn j2k_dwt_from_integer_wavelet(wavelet: &IntegerWavelet) -> J2kForwardDwt53Output {
    let mut levels = Vec::with_capacity(wavelet.levels.len());
    for level in &wavelet.levels {
        levels.push(J2kForwardDwt53Level {
            hl: level.hl.iter().map(|&value| value as f32).collect(),
            lh: level.lh.iter().map(|&value| value as f32).collect(),
            hh: level.hh.iter().map(|&value| value as f32).collect(),
            width: level.width as u32,
            height: level.height as u32,
            low_width: level.low_width as u32,
            low_height: level.low_height as u32,
            high_width: level.high_width as u32,
            high_height: level.high_height as u32,
        });
    }
    levels.reverse();

    J2kForwardDwt53Output {
        ll: wavelet.final_ll.iter().map(|&value| value as f32).collect(),
        ll_width: wavelet.final_ll_width as u32,
        ll_height: wavelet.final_ll_height as u32,
        levels,
    }
}

pub(super) fn rounded_wavelet_i32(
    wavelet: &ComponentWavelet,
) -> Result<Vec<i32>, JpegToHtj2kError> {
    let coefficient_count = wavelet.final_ll.len()
        + wavelet
            .levels
            .iter()
            .map(|level| level.hl.len() + level.lh.len() + level.hh.len())
            .sum::<usize>();
    let mut output = Vec::with_capacity(coefficient_count);
    append_rounded_i32(&wavelet.final_ll, &mut output)?;
    for level in wavelet.levels.iter().rev() {
        append_rounded_i32(&level.hl, &mut output)?;
        append_rounded_i32(&level.lh, &mut output)?;
        append_rounded_i32(&level.hh, &mut output)?;
    }
    Ok(output)
}

pub(super) fn rounded_wavelet97_i32(
    wavelet: &ComponentWavelet97,
) -> Result<Vec<i32>, JpegToHtj2kError> {
    let coefficient_count = wavelet.final_ll.len()
        + wavelet
            .levels
            .iter()
            .map(|level| level.hl.len() + level.lh.len() + level.hh.len())
            .sum::<usize>();
    let mut output = Vec::with_capacity(coefficient_count);
    append_rounded_i32(&wavelet.final_ll, &mut output)?;
    for level in wavelet.levels.iter().rev() {
        append_rounded_i32(&level.hl, &mut output)?;
        append_rounded_i32(&level.lh, &mut output)?;
        append_rounded_i32(&level.hh, &mut output)?;
    }
    Ok(output)
}

pub(super) fn append_rounded_i32(
    values: &[f64],
    output: &mut Vec<i32>,
) -> Result<(), JpegToHtj2kError> {
    for &value in values {
        output.push(round_f64_to_i32(value)?);
    }
    Ok(())
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the finite rounded coefficient is explicitly checked against the complete i32 range"
)]
pub(super) fn round_f64_to_i32(value: f64) -> Result<i32, JpegToHtj2kError> {
    let rounded = value.round();
    if !rounded.is_finite() {
        return Err(JpegToHtj2kError::Validation(
            "float reference coefficient is not finite",
        ));
    }
    if rounded < f64::from(i32::MIN) || rounded > f64::from(i32::MAX) {
        return Err(JpegToHtj2kError::Validation(
            "float reference coefficient exceeds i32 range",
        ));
    }
    Ok(rounded as i32)
}

pub(super) fn dct_blocks_to_8x8_f64_into(blocks: &[[i16; 64]], output: &mut Vec<[[f64; 8]; 8]>) {
    output.clear();
    output.reserve(blocks.len());
    for block in blocks {
        let mut converted = [[0.0; 8]; 8];
        for (idx, &coefficient) in block.iter().enumerate() {
            converted[idx / 8][idx % 8] = f64::from(coefficient);
        }
        output.push(converted);
    }
}

pub(super) fn dct_blocks_to_8x8_f64(blocks: &[[i16; 64]]) -> Vec<[[f64; 8]; 8]> {
    let mut output = Vec::with_capacity(blocks.len());
    dct_blocks_to_8x8_f64_into(blocks, &mut output);
    output
}
