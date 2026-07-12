// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    dct53_transform_error, dct8x8_blocks_then_dwt53_float, dct8x8_blocks_then_dwt97_float,
    dct8x8_blocks_then_dwt97_float_with_scratch, dct8x8_blocks_to_dwt53_float_linear_with_scratch,
    dct97_transform_error, linearized_53_2d_from_plane, linearized_97_2d_from_plane_with_scratch,
    record_accelerator_attempt, record_accelerator_dispatch, record_cpu_fallback,
    rounded_wavelet97_i32, rounded_wavelet_i32, Dct97GridScratch, DctGridToDwt53Job,
    DctGridToDwt97Job, DctToWaveletStageAccelerator, Dwt53TwoDimensional, Dwt97TwoDimensional,
    Instant, JpegDctComponent, JpegToHtj2kError, JpegToHtj2kScratch, TranscodeTimingReport,
};
use crate::allocation::{try_vec_from_slice, try_vec_reserve_len, try_vec_with_capacity};

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
    dct_blocks_to_8x8_f64_into(&component.dequantized_blocks, &mut scratch.dct_blocks_f64)?;
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
        .map_err(dct53_transform_error)?;
        timings.dct_to_wavelet_cpu_fallback_us = timings
            .dct_to_wavelet_cpu_fallback_us
            .saturating_add(fallback_start.elapsed().as_micros());
        bands
    };
    let decompose_start = Instant::now();
    let wavelet = decompose_from_first_level(bands, usize::from(decomposition_levels))?;
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
    dct_blocks_to_8x8_f64_into(&component.dequantized_blocks, &mut scratch.dct_blocks_f64)?;
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
        .map_err(dct97_transform_error)?;
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
    )?;
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
    dct_blocks_to_8x8_f64_into(&component.dequantized_blocks, &mut scratch.dct_blocks_f64)?;
    let blocks = &scratch.dct_blocks_f64;
    let first_reference_level = dct8x8_blocks_then_dwt53_float(
        blocks,
        component.block_cols as usize,
        component.block_rows as usize,
        component.width as usize,
        component.height as usize,
    )
    .map_err(dct53_transform_error)?;
    let reference =
        decompose_from_first_level(first_reference_level, usize::from(decomposition_levels))?;
    rounded_wavelet_i32(&reference)
}

pub(super) fn float97_reference_coefficients(
    component: &JpegDctComponent,
    decomposition_levels: u8,
    scratch: &mut JpegToHtj2kScratch,
) -> Result<Vec<i32>, JpegToHtj2kError> {
    dct_blocks_to_8x8_f64_into(&component.dequantized_blocks, &mut scratch.dct_blocks_f64)?;
    let blocks = &scratch.dct_blocks_f64;
    let first_reference_level = dct8x8_blocks_then_dwt97_float(
        blocks,
        component.block_cols as usize,
        component.block_rows as usize,
        component.width as usize,
        component.height as usize,
    )
    .map_err(dct97_transform_error)?;
    let reference =
        decompose_97_from_first_level(first_reference_level, usize::from(decomposition_levels))?;
    rounded_wavelet97_i32(&reference)
}

pub(super) fn decompose_from_first_level(
    first_level: Dwt53TwoDimensional<f64>,
    decomposition_levels: usize,
) -> Result<ComponentWavelet, JpegToHtj2kError> {
    let final_ll = try_vec_from_slice(&first_level.ll)?;
    let mut levels = try_vec_with_capacity(decomposition_levels.max(1))?;
    levels.push(first_level);
    let mut wavelet = ComponentWavelet {
        final_ll,
        final_ll_width: levels[0].low_width,
        final_ll_height: levels[0].low_height,
        levels,
    };

    while wavelet.levels.len() < decomposition_levels {
        let next = linearized_53_2d_from_plane(
            &wavelet.final_ll,
            wavelet.final_ll_width,
            wavelet.final_ll_height,
        )
        .map_err(dct53_transform_error)?;
        wavelet.final_ll.clear();
        try_vec_reserve_len(&mut wavelet.final_ll, next.ll.len())?;
        wavelet.final_ll.extend_from_slice(&next.ll);
        wavelet.final_ll_width = next.low_width;
        wavelet.final_ll_height = next.low_height;
        wavelet.levels.push(next);
    }

    Ok(wavelet)
}

pub(super) fn decompose_97_from_first_level(
    first_level: Dwt97TwoDimensional<f64>,
    decomposition_levels: usize,
) -> Result<ComponentWavelet97, JpegToHtj2kError> {
    let mut scratch = Dct97GridScratch::default();
    decompose_97_from_first_level_with_scratch(first_level, decomposition_levels, &mut scratch)
}

pub(super) fn decompose_97_from_first_level_with_scratch(
    first_level: Dwt97TwoDimensional<f64>,
    decomposition_levels: usize,
    scratch: &mut Dct97GridScratch,
) -> Result<ComponentWavelet97, JpegToHtj2kError> {
    let final_ll = try_vec_from_slice(&first_level.ll)?;
    let mut levels = try_vec_with_capacity(decomposition_levels.max(1))?;
    levels.push(first_level);
    let mut wavelet = ComponentWavelet97 {
        final_ll,
        final_ll_width: levels[0].low_width,
        final_ll_height: levels[0].low_height,
        levels,
    };

    while wavelet.levels.len() < decomposition_levels {
        let next = linearized_97_2d_from_plane_with_scratch(
            &wavelet.final_ll,
            wavelet.final_ll_width,
            wavelet.final_ll_height,
            scratch,
        )
        .map_err(dct97_transform_error)?;
        wavelet.final_ll.clear();
        try_vec_reserve_len(&mut wavelet.final_ll, next.ll.len())?;
        wavelet.final_ll.extend_from_slice(&next.ll);
        wavelet.final_ll_width = next.low_width;
        wavelet.final_ll_height = next.low_height;
        wavelet.levels.push(next);
    }

    Ok(wavelet)
}

pub(super) fn dct_blocks_to_8x8_f64_into(
    blocks: &[[i16; 64]],
    output: &mut Vec<[[f64; 8]; 8]>,
) -> Result<(), JpegToHtj2kError> {
    output.clear();
    try_vec_reserve_len(output, blocks.len())?;
    for block in blocks {
        let mut converted = [[0.0; 8]; 8];
        for (idx, &coefficient) in block.iter().enumerate() {
            converted[idx / 8][idx % 8] = f64::from(coefficient);
        }
        output.push(converted);
    }
    Ok(())
}

pub(super) fn dct_blocks_to_8x8_f64(
    blocks: &[[i16; 64]],
) -> Result<Vec<[[f64; 8]; 8]>, JpegToHtj2kError> {
    let mut output = try_vec_with_capacity(blocks.len())?;
    dct_blocks_to_8x8_f64_into(blocks, &mut output)?;
    Ok(output)
}
