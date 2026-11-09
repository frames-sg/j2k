// SPDX-License-Identifier: MIT OR Apache-2.0

//! Default-precision (canonical binary32) 9/7 coefficient production.
//!
//! The first decomposition level comes from an accelerator or from the CPU
//! canonical transform; later levels, quantization, and code-block layout are
//! always computed here, so every backend hands the native encoder identical
//! quantized coefficients.

use j2k_codec_math::transcode97::{quantize97, quantize97_inv_delta};

use super::batch::htj2k97_codeblock_options;
use super::component_plan::PrecomputedComponent;
use super::float_output::round_f64_to_i32;
use super::{
    dct97_transform_error, record_accelerator_attempt, record_accelerator_dispatch,
    record_cpu_fallback, DctGridToDwt97Job, DctToWaveletStageAccelerator, Dwt97TwoDimensional,
    Instant, J2kForwardDwt97Level, J2kForwardDwt97Output, JpegDctComponent,
    JpegToHtj2kEncodeOptions, JpegToHtj2kError, JpegToHtj2kOptions, JpegToHtj2kScratch,
    PrecomputedHtj2k97Component, TranscodeTimingReport,
};
use crate::allocation::{try_vec_from_slice, try_vec_with_capacity};
use crate::dct97_f32::{dct_blocks_then_dwt97_f32, lift97_plane_f32, HostTranscode97Arithmetic};
use crate::htj2k97_codeblock_oracle::{htj2k97_level_delta, htj2k97_level_total_bitplanes};
use crate::{
    Htj2k97CodeBlockOptions, J2kSubBandType, PrequantizedHtj2k97CodeBlock,
    PrequantizedHtj2k97Component, PrequantizedHtj2k97Resolution, PrequantizedHtj2k97Subband,
};

/// Canonical binary32 wavelet of one component; `levels[0]` is the finest.
pub(super) struct ComponentWavelet97F32 {
    pub(super) final_ll: Vec<f32>,
    pub(super) final_ll_width: usize,
    pub(super) final_ll_height: usize,
    pub(super) levels: Vec<Dwt97TwoDimensional<f32>>,
}

/// Whether the native prequantized route accepts these encode options. Other
/// options hand the canonical bands to the native precomputed encoder.
pub(super) fn canonical_prequantized_route(options: &JpegToHtj2kEncodeOptions) -> bool {
    options.num_layers == 1
        && options.quality_layer_byte_targets.is_empty()
        && options.tile_size.is_none()
}

/// Encoder input for one canonical component: canonical quantization into
/// code blocks when the prequantized route applies, otherwise the canonical
/// bands for the native precomputed encoder.
pub(super) fn canonical_encode_component(
    wavelet: &ComponentWavelet97F32,
    component: &JpegDctComponent,
    x_rsiz: u8,
    y_rsiz: u8,
    options: &JpegToHtj2kOptions,
) -> Result<PrecomputedComponent, JpegToHtj2kError> {
    if canonical_prequantized_route(&options.encode_options) {
        return Ok(PrecomputedComponent::Prequantized97(
            prequantized_component_from_wavelet_f32(
                wavelet,
                x_rsiz,
                y_rsiz,
                htj2k97_codeblock_options(&options.encode_options),
            )?,
        ));
    }
    Ok(PrecomputedComponent::Dwt97(PrecomputedHtj2k97Component {
        x_rsiz,
        y_rsiz,
        dwt: j2k_dwt97_from_wavelet_f32(
            wavelet,
            component.width as usize,
            component.height as usize,
        )?,
    }))
}

/// Canonical coefficients rounded to integers in validation order: final LL,
/// then HL, LH and HH from the deepest level to the finest.
pub(super) fn rounded_wavelet97_f32_i32(
    wavelet: &ComponentWavelet97F32,
) -> Result<Vec<i32>, JpegToHtj2kError> {
    let mut count = wavelet.final_ll.len();
    for level in &wavelet.levels {
        for len in [level.hl.len(), level.lh.len(), level.hh.len()] {
            count = count
                .checked_add(len)
                .ok_or(JpegToHtj2kError::MemoryCapExceeded {
                    requested: usize::MAX,
                    cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                })?;
        }
    }
    let mut output = try_vec_with_capacity(count)?;
    let mut append = |values: &[f32]| -> Result<(), JpegToHtj2kError> {
        for &value in values {
            output.push(round_f64_to_i32(f64::from(value))?);
        }
        Ok(())
    };
    append(&wavelet.final_ll)?;
    for level in wavelet.levels.iter().rev() {
        append(&level.hl)?;
        append(&level.lh)?;
        append(&level.hh)?;
    }
    Ok(output)
}

/// Canonical first level for one component: offered to the accelerator as a
/// float-band job, otherwise computed by the CPU canonical transform.
pub(super) fn canonical_97_wavelet_from_component(
    component: &JpegDctComponent,
    decomposition_levels: u8,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut impl DctToWaveletStageAccelerator,
    timings: &mut TranscodeTimingReport,
) -> Result<ComponentWavelet97F32, JpegToHtj2kError> {
    timings.component_count = timings.component_count.saturating_add(1);
    let repack_start = Instant::now();
    super::float_reference::dct_blocks_to_8x8_f64_into(
        &component.dequantized_blocks,
        &mut scratch.dct_blocks_f64,
    )?;
    timings.jpeg_dct_repack_us = timings
        .jpeg_dct_repack_us
        .saturating_add(repack_start.elapsed().as_micros());
    let job = DctGridToDwt97Job {
        blocks: &scratch.dct_blocks_f64,
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
    let first_level = if let Some(bands) = accelerated {
        record_accelerator_dispatch(timings, 1);
        canonical_bands_from_accelerator(&bands)?
    } else {
        record_cpu_fallback(timings, 1);
        let fallback_start = Instant::now();
        let bands = dct_blocks_then_dwt97_f32(
            &component.dequantized_blocks,
            component.block_cols as usize,
            component.block_rows as usize,
            component.width as usize,
            component.height as usize,
            &mut scratch.dct97_f32,
        )
        .map_err(dct97_transform_error)?;
        timings.dct_to_wavelet_cpu_fallback_us = timings
            .dct_to_wavelet_cpu_fallback_us
            .saturating_add(fallback_start.elapsed().as_micros());
        bands
    };
    let decompose_start = Instant::now();
    let wavelet = decompose_97_f32_from_first_level(
        first_level,
        usize::from(decomposition_levels),
        &mut scratch.dct97_f32,
    )?;
    timings.dwt_decompose_us = timings
        .dwt_decompose_us
        .saturating_add(decompose_start.elapsed().as_micros());
    Ok(wavelet)
}

/// Accelerators return canonical binary32 bands widened to `f64`; narrowing
/// them back is exact.
#[expect(
    clippy::cast_possible_truncation,
    reason = "accelerator bands carry binary32 values widened losslessly to f64"
)]
pub(super) fn canonical_bands_from_accelerator(
    bands: &Dwt97TwoDimensional<f64>,
) -> Result<Dwt97TwoDimensional<f32>, JpegToHtj2kError> {
    let narrow = |values: &[f64]| -> Result<Vec<f32>, JpegToHtj2kError> {
        let mut output = try_vec_with_capacity(values.len())?;
        output.extend(values.iter().map(|&value| value as f32));
        Ok(output)
    };
    Ok(Dwt97TwoDimensional {
        ll: narrow(&bands.ll)?,
        hl: narrow(&bands.hl)?,
        lh: narrow(&bands.lh)?,
        hh: narrow(&bands.hh)?,
        low_width: bands.low_width,
        low_height: bands.low_height,
        high_width: bands.high_width,
        high_height: bands.high_height,
    })
}

/// Canonical later decomposition levels over the previous LL band.
pub(super) fn decompose_97_f32_from_first_level(
    first_level: Dwt97TwoDimensional<f32>,
    decomposition_levels: usize,
    scratch: &mut crate::dct97_f32::Dct97F32Scratch,
) -> Result<ComponentWavelet97F32, JpegToHtj2kError> {
    let final_ll = try_vec_from_slice(&first_level.ll)?;
    let mut levels = try_vec_with_capacity(decomposition_levels.max(1))?;
    let (final_ll_width, final_ll_height) = (first_level.low_width, first_level.low_height);
    levels.push(first_level);
    let mut wavelet = ComponentWavelet97F32 {
        final_ll,
        final_ll_width,
        final_ll_height,
        levels,
    };
    while wavelet.levels.len() < decomposition_levels {
        let next = lift97_plane_f32(
            &wavelet.final_ll,
            wavelet.final_ll_width,
            wavelet.final_ll_height,
            scratch,
        )
        .map_err(dct97_transform_error)?;
        wavelet.final_ll = try_vec_from_slice(&next.ll)?;
        wavelet.final_ll_width = next.low_width;
        wavelet.final_ll_height = next.low_height;
        wavelet.levels.push(next);
    }
    Ok(wavelet)
}

/// Canonical bands in the native precomputed coefficient ABI.
#[expect(
    clippy::cast_possible_truncation,
    reason = "validated JPEG component geometry fits u32"
)]
pub(super) fn j2k_dwt97_from_wavelet_f32(
    wavelet: &ComponentWavelet97F32,
    width: usize,
    height: usize,
) -> Result<J2kForwardDwt97Output, JpegToHtj2kError> {
    let mut current_width = width;
    let mut current_height = height;
    let mut levels = try_vec_with_capacity(wavelet.levels.len())?;
    for level in &wavelet.levels {
        levels.push(J2kForwardDwt97Level {
            hl: try_vec_from_slice(&level.hl)?,
            lh: try_vec_from_slice(&level.lh)?,
            hh: try_vec_from_slice(&level.hh)?,
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
    Ok(J2kForwardDwt97Output {
        ll: try_vec_from_slice(&wavelet.final_ll)?,
        ll_width: wavelet.final_ll_width as u32,
        ll_height: wavelet.final_ll_height as u32,
        levels,
    })
}

/// Quantize a canonical wavelet into native prequantized code-block layout.
///
/// Resolution 0 holds the deepest LL band; resolution `r` holds HL, LH and HH
/// of decomposition level `levels - r + 1`.
pub(super) fn prequantized_component_from_wavelet_f32(
    wavelet: &ComponentWavelet97F32,
    x_rsiz: u8,
    y_rsiz: u8,
    options: Htj2k97CodeBlockOptions,
) -> Result<PrequantizedHtj2k97Component, JpegToHtj2kError> {
    let levels = u8::try_from(wavelet.levels.len())
        .map_err(|_| JpegToHtj2kError::Unsupported("9/7 decomposition level count exceeds u8"))?;
    let mut resolutions = try_vec_with_capacity(wavelet.levels.len() + 1)?;
    let mut low_low = try_vec_with_capacity(1)?;
    low_low.push(quantize_band(
        BandRef {
            values: &wavelet.final_ll,
            width: wavelet.final_ll_width,
            height: wavelet.final_ll_height,
            sub_band_type: J2kSubBandType::LowLow,
            level: levels,
        },
        levels,
        options,
    )?);
    resolutions.push(PrequantizedHtj2k97Resolution { subbands: low_low });
    for (level_number, level) in (1..=levels).rev().zip(wavelet.levels.iter().rev()) {
        let mut subbands = try_vec_with_capacity(3)?;
        for (values, width, height, sub_band_type) in [
            (
                &level.hl,
                level.high_width,
                level.low_height,
                J2kSubBandType::HighLow,
            ),
            (
                &level.lh,
                level.low_width,
                level.high_height,
                J2kSubBandType::LowHigh,
            ),
            (
                &level.hh,
                level.high_width,
                level.high_height,
                J2kSubBandType::HighHigh,
            ),
        ] {
            subbands.push(quantize_band(
                BandRef {
                    values,
                    width,
                    height,
                    sub_band_type,
                    level: level_number,
                },
                levels,
                options,
            )?);
        }
        resolutions.push(PrequantizedHtj2k97Resolution { subbands });
    }
    Ok(PrequantizedHtj2k97Component {
        x_rsiz,
        y_rsiz,
        resolutions,
    })
}

#[derive(Clone, Copy)]
struct BandRef<'a> {
    values: &'a [f32],
    width: usize,
    height: usize,
    sub_band_type: J2kSubBandType,
    level: u8,
}

fn quantize_band(
    band: BandRef<'_>,
    levels: u8,
    options: Htj2k97CodeBlockOptions,
) -> Result<PrequantizedHtj2k97Subband, JpegToHtj2kError> {
    let BandRef {
        values,
        width,
        height,
        sub_band_type,
        level,
    } = band;
    let cb_width = code_block_dim(options.code_block_width_exp)?;
    let cb_height = code_block_dim(options.code_block_height_exp)?;
    let num_cbs_x = width.div_ceil(cb_width);
    let num_cbs_y = height.div_ceil(cb_height);
    let block_count = num_cbs_x
        .checked_mul(num_cbs_y)
        .ok_or(JpegToHtj2kError::Validation(
            "9/7 code-block count overflow",
        ))?;
    if values.len() != width * height {
        return Err(JpegToHtj2kError::Validation(
            "canonical 9/7 band length does not match its geometry",
        ));
    }
    let total_bitplanes = if block_count == 0 {
        0
    } else {
        htj2k97_level_total_bitplanes(options, sub_band_type, level, levels)
    };
    let inv_delta =
        quantize97_inv_delta(htj2k97_level_delta(options, sub_band_type, level, levels));
    let mut code_blocks = try_vec_with_capacity(block_count)?;
    for cby in 0..num_cbs_y {
        for cbx in 0..num_cbs_x {
            let x0 = cbx * cb_width;
            let y0 = cby * cb_height;
            let block_width = (width - x0).min(cb_width);
            let block_height = (height - y0).min(cb_height);
            let mut coefficients = try_vec_with_capacity(block_width * block_height)?;
            for y in y0..y0 + block_height {
                let row = &values[y * width + x0..y * width + x0 + block_width];
                coefficients.extend(
                    row.iter()
                        .map(|&value| quantize97::<HostTranscode97Arithmetic>(value, inv_delta)),
                );
            }
            code_blocks.push(PrequantizedHtj2k97CodeBlock {
                coefficients,
                width: dimension_u32(block_width)?,
                height: dimension_u32(block_height)?,
            });
        }
    }
    Ok(PrequantizedHtj2k97Subband {
        sub_band_type,
        num_cbs_x: dimension_u32(num_cbs_x)?,
        num_cbs_y: dimension_u32(num_cbs_y)?,
        total_bitplanes,
        code_blocks,
    })
}

fn code_block_dim(exp_minus_two: u8) -> Result<usize, JpegToHtj2kError> {
    1usize
        .checked_shl(u32::from(exp_minus_two) + 2)
        .ok_or(JpegToHtj2kError::Unsupported(
            "9/7 code-block dimension exponent is unsupported",
        ))
}

fn dimension_u32(value: usize) -> Result<u32, JpegToHtj2kError> {
    u32::try_from(value)
        .map_err(|_| JpegToHtj2kError::Validation("9/7 code-block geometry exceeds u32"))
}
