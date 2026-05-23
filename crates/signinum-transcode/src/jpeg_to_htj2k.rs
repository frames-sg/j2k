// SPDX-License-Identifier: Apache-2.0

//! Experimental JPEG DCT to HTJ2K codestream transcode entry point.

use core::fmt;
use std::time::Instant;

use signinum_j2k_native::{
    encode_precomputed_htj2k_53, EncodeOptions, J2kForwardDwt53Level, J2kForwardDwt53Output,
    PrecomputedHtj2k53Component, PrecomputedHtj2k53Image,
};
use signinum_jpeg::transcode::{
    extract_dct_blocks, idct_islow_block, DctExtractOptions, JpegDctComponent,
};

use crate::dct53_2d::{
    dct8x8_blocks_then_dwt53_float, dct8x8_blocks_to_dwt53_float_linear,
    linearized_53_2d_from_plane, Dct53GridError, Dwt53TwoDimensional,
};
use crate::metrics::{error_metrics_i32, ErrorMetrics, MetricsLengthError};

/// Options for the experimental JPEG-to-HTJ2K path.
#[derive(Debug, Clone)]
pub struct JpegToHtj2kOptions {
    /// Native HTJ2K encode options used after wavelet bands are produced.
    pub encode_options: EncodeOptions,
    /// Materialize the float IDCT-then-DWT oracle and report rounded
    /// coefficient differences. This is intended for validation and tests, not
    /// the production direct path.
    pub validate_against_float_reference: bool,
    /// Materialize signinum-jpeg scalar ISLOW samples and report reversible
    /// integer 5/3 coefficient differences against the rounded direct path.
    /// This is intended for validation and tests, not the production direct
    /// path.
    pub validate_against_integer_reference: bool,
}

impl Default for JpegToHtj2kOptions {
    fn default() -> Self {
        Self {
            encode_options: EncodeOptions {
                num_decomposition_levels: 1,
                reversible: true,
                use_ht_block_coding: true,
                use_mct: false,
                validate_high_throughput_codestream: false,
                ..EncodeOptions::default()
            },
            validate_against_float_reference: false,
            validate_against_integer_reference: false,
        }
    }
}

/// Reusable experimental JPEG-to-HTJ2K transcoder state.
///
/// Create one value per worker thread and reuse it across many tiles to keep
/// scratch buffers allocated between calls. The scalar math and output are the
/// same as [`jpeg_to_htj2k`].
#[derive(Debug, Default)]
pub struct JpegToHtj2kTranscoder {
    scratch: JpegToHtj2kScratch,
}

impl JpegToHtj2kTranscoder {
    /// Transcode a constrained baseline JPEG tile into HTJ2K using this
    /// instance's reusable scratch buffers.
    pub fn transcode(
        &mut self,
        bytes: &[u8],
        options: &JpegToHtj2kOptions,
    ) -> Result<EncodedTranscode, JpegToHtj2kError> {
        jpeg_to_htj2k_with_scratch(bytes, options, &mut self.scratch)
    }

    /// Current capacity of the reusable DCT block conversion scratch.
    ///
    /// This is exposed for benchmark and validation harnesses while the API is
    /// experimental.
    #[must_use]
    pub fn dct_block_scratch_capacity(&self) -> usize {
        self.scratch.dct_blocks_f64.capacity()
    }
}

#[derive(Debug, Default)]
struct JpegToHtj2kScratch {
    dct_blocks_f64: Vec<[[f64; 8]; 8]>,
}

/// Encoded transcode output and validation/report metadata.
#[derive(Debug, Clone)]
pub struct EncodedTranscode {
    /// HTJ2K codestream bytes.
    pub codestream: Vec<u8>,
    /// Summary of the experimental path used.
    pub report: TranscodeReport,
}

/// Per-component transcode geometry preserved in the generated codestream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscodeComponentReport {
    /// Component index in JPEG SOF order.
    pub component_index: usize,
    /// Native component width in samples before HTJ2K SIZ expansion.
    pub width: u32,
    /// Native component height in samples before HTJ2K SIZ expansion.
    pub height: u32,
    /// Number of DCT blocks per component row, including padded edge blocks.
    pub block_cols: u32,
    /// Number of DCT block rows, including padded edge blocks.
    pub block_rows: u32,
    /// HTJ2K SIZ horizontal sampling factor.
    pub x_rsiz: u8,
    /// HTJ2K SIZ vertical sampling factor.
    pub y_rsiz: u8,
}

/// Error metrics from an optional validation oracle.
pub type TranscodeValidationMetrics = ErrorMetrics;

/// Transcode summary for validation and benchmarking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscodeReport {
    /// Source reference-grid width.
    pub width: u32,
    /// Source reference-grid height.
    pub height: u32,
    /// Number of transformed components.
    pub component_count: usize,
    /// Native transformed component geometry and SIZ sampling.
    pub components: Vec<TranscodeComponentReport>,
    /// Rounded coefficient metrics against the optional float IDCT-then-DWT
    /// oracle.
    pub float_reference_metrics: Option<TranscodeValidationMetrics>,
    /// Rounded direct coefficients compared with signinum-jpeg scalar
    /// ISLOW-IDCT-then-reversible-5/3 coefficients.
    pub integer_reference_metrics: Option<TranscodeValidationMetrics>,
    /// Number of reversible 5/3 decomposition levels encoded.
    pub decomposition_levels: u8,
    /// Name of the experimental path used.
    pub path: &'static str,
    /// Wall-clock extraction time in microseconds.
    pub extract_us: u128,
    /// Wall-clock DCT-to-wavelet time in microseconds.
    pub transform_us: u128,
    /// Wall-clock HTJ2K encode time in microseconds.
    pub encode_us: u128,
}

/// Error returned by the experimental transcode path.
#[derive(Debug)]
pub enum JpegToHtj2kError {
    /// JPEG parse or entropy decode failed.
    Jpeg(signinum_jpeg::JpegError),
    /// Input is outside the currently implemented experimental slice.
    Unsupported(&'static str),
    /// DCT block grid metadata did not cover the component dimensions.
    Grid(Dct53GridError),
    /// Validation metric inputs were inconsistent.
    Metrics(MetricsLengthError),
    /// Validation encountered an out-of-range or non-finite coefficient.
    Validation(&'static str),
    /// Native HTJ2K encode failed.
    Encode(&'static str),
}

impl fmt::Display for JpegToHtj2kError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Jpeg(err) => write!(f, "JPEG extraction failed: {err}"),
            Self::Unsupported(reason) => write!(f, "unsupported transcode input: {reason}"),
            Self::Grid(err) => write!(f, "DCT grid transform failed: {err}"),
            Self::Metrics(err) => write!(f, "validation metrics failed: {err}"),
            Self::Validation(reason) => write!(f, "validation failed: {reason}"),
            Self::Encode(reason) => write!(f, "HTJ2K encode failed: {reason}"),
        }
    }
}

impl std::error::Error for JpegToHtj2kError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Jpeg(err) => Some(err),
            Self::Grid(err) => Some(err),
            Self::Metrics(err) => Some(err),
            Self::Unsupported(_) | Self::Validation(_) | Self::Encode(_) => None,
        }
    }
}

impl From<signinum_jpeg::JpegError> for JpegToHtj2kError {
    fn from(value: signinum_jpeg::JpegError) -> Self {
        Self::Jpeg(value)
    }
}

impl From<Dct53GridError> for JpegToHtj2kError {
    fn from(value: Dct53GridError) -> Self {
        Self::Grid(value)
    }
}

impl From<MetricsLengthError> for JpegToHtj2kError {
    fn from(value: MetricsLengthError) -> Self {
        Self::Metrics(value)
    }
}

/// Transcode a constrained baseline grayscale JPEG tile into an HTJ2K
/// codestream using direct DCT-domain 5/3 wavelet coefficients.
///
/// Current implementation scope is baseline JPEG with one or more components
/// at native JPEG component resolution, and one reversible 5/3 decomposition
/// level. Component subsampling is preserved through SIZ `XRsiz`/`YRsiz`
/// instead of chroma upsampling.
pub fn jpeg_to_htj2k(
    bytes: &[u8],
    options: &JpegToHtj2kOptions,
) -> Result<EncodedTranscode, JpegToHtj2kError> {
    JpegToHtj2kTranscoder::default().transcode(bytes, options)
}

fn jpeg_to_htj2k_with_scratch(
    bytes: &[u8],
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
) -> Result<EncodedTranscode, JpegToHtj2kError> {
    let extract_start = Instant::now();
    let jpeg = extract_dct_blocks(bytes, DctExtractOptions::default())?;
    let extract_us = extract_start.elapsed().as_micros();

    if jpeg.components.is_empty() || jpeg.components.len() > 4 {
        return Err(JpegToHtj2kError::Unsupported(
            "unsupported JPEG component count for jpeg_to_htj2k",
        ));
    }
    let component_sampling =
        component_sampling_for_jpeg(&jpeg.components, jpeg.width, jpeg.height)?;
    let decomposition_levels = decomposition_levels_for_components(
        &jpeg.components,
        options.encode_options.num_decomposition_levels,
    )?;
    let all_unit_sampled = component_sampling
        .iter()
        .all(|&(x_rsiz, y_rsiz)| x_rsiz == 1 && y_rsiz == 1);
    let component_reports = jpeg
        .components
        .iter()
        .zip(component_sampling.iter().copied())
        .map(|(component, (x_rsiz, y_rsiz))| TranscodeComponentReport {
            component_index: component.component_index,
            width: component.width,
            height: component.height,
            block_cols: component.block_cols,
            block_rows: component.block_rows,
            x_rsiz,
            y_rsiz,
        })
        .collect();

    let transform_start = Instant::now();
    let component_batch = transcode_component_batch(
        &jpeg.components,
        &component_sampling,
        decomposition_levels,
        options,
        scratch,
    )?;
    let transform_us = transform_start.elapsed().as_micros();

    let precomputed = PrecomputedHtj2k53Image {
        width: jpeg.width,
        height: jpeg.height,
        bit_depth: 8,
        signed: false,
        components: component_batch.precomputed_components,
    };

    let encode_start = Instant::now();
    let codestream = encode_precomputed_htj2k_53(&precomputed, &options.encode_options)
        .map_err(JpegToHtj2kError::Encode)?;
    let encode_us = encode_start.elapsed().as_micros();

    Ok(EncodedTranscode {
        codestream,
        report: TranscodeReport {
            width: jpeg.width,
            height: jpeg.height,
            component_count: jpeg.components.len(),
            components: component_reports,
            float_reference_metrics: component_batch.float_reference_metrics,
            integer_reference_metrics: component_batch.integer_reference_metrics,
            decomposition_levels,
            path: if all_unit_sampled {
                "full_resolution_components_float_direct_53"
            } else {
                "native_component_sampling_float_direct_53"
            },
            extract_us,
            transform_us,
            encode_us,
        },
    })
}

struct ComponentTranscodeBatch {
    precomputed_components: Vec<PrecomputedHtj2k53Component>,
    float_reference_metrics: Option<TranscodeValidationMetrics>,
    integer_reference_metrics: Option<TranscodeValidationMetrics>,
}

struct ComponentTranscodeResult {
    precomputed: PrecomputedHtj2k53Component,
    float_validation_coefficients: Option<(Vec<i32>, Vec<i32>)>,
    integer_validation_coefficients: Option<(Vec<i32>, Vec<i32>)>,
}

struct ComponentWavelet {
    final_ll: Vec<f64>,
    final_ll_width: usize,
    final_ll_height: usize,
    levels: Vec<Dwt53TwoDimensional<f64>>,
}

struct IntegerWaveletLevel {
    hl: Vec<i32>,
    lh: Vec<i32>,
    hh: Vec<i32>,
}

struct IntegerWavelet {
    final_ll: Vec<i32>,
    levels: Vec<IntegerWaveletLevel>,
}

fn transcode_component_batch(
    components: &[JpegDctComponent],
    component_sampling: &[(u8, u8)],
    decomposition_levels: u8,
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
) -> Result<ComponentTranscodeBatch, JpegToHtj2kError> {
    let mut precomputed_components = Vec::with_capacity(components.len());
    let mut float_validation_actual = Vec::new();
    let mut float_validation_expected = Vec::new();
    let mut integer_validation_actual = Vec::new();
    let mut integer_validation_expected = Vec::new();

    for (component, (x_rsiz, y_rsiz)) in components.iter().zip(component_sampling.iter().copied()) {
        let component_result = component_to_precomputed_htj2k(
            component,
            x_rsiz,
            y_rsiz,
            decomposition_levels,
            options.validate_against_float_reference,
            options.validate_against_integer_reference,
            scratch,
        )?;
        precomputed_components.push(component_result.precomputed);
        if let Some((actual, expected)) = component_result.float_validation_coefficients {
            float_validation_actual.extend(actual);
            float_validation_expected.extend(expected);
        }
        if let Some((actual, expected)) = component_result.integer_validation_coefficients {
            integer_validation_actual.extend(actual);
            integer_validation_expected.extend(expected);
        }
    }

    let float_reference_metrics = if options.validate_against_float_reference {
        Some(error_metrics_i32(
            &float_validation_actual,
            &float_validation_expected,
        )?)
    } else {
        None
    };
    let integer_reference_metrics = if options.validate_against_integer_reference {
        Some(error_metrics_i32(
            &integer_validation_actual,
            &integer_validation_expected,
        )?)
    } else {
        None
    };

    Ok(ComponentTranscodeBatch {
        precomputed_components,
        float_reference_metrics,
        integer_reference_metrics,
    })
}

fn component_to_precomputed_htj2k(
    component: &JpegDctComponent,
    x_rsiz: u8,
    y_rsiz: u8,
    decomposition_levels: u8,
    validate_against_float_reference: bool,
    validate_against_integer_reference: bool,
    scratch: &mut JpegToHtj2kScratch,
) -> Result<ComponentTranscodeResult, JpegToHtj2kError> {
    dct_blocks_to_8x8_f64_into(&component.dequantized_blocks, &mut scratch.dct_blocks_f64);
    let blocks = &scratch.dct_blocks_f64;
    let bands = dct8x8_blocks_to_dwt53_float_linear(
        blocks,
        component.block_cols as usize,
        component.block_rows as usize,
        component.width as usize,
        component.height as usize,
    )?;
    let wavelet = decompose_from_first_level(bands, usize::from(decomposition_levels));
    let float_validation_coefficients = if validate_against_float_reference {
        let first_reference_level = dct8x8_blocks_then_dwt53_float(
            blocks,
            component.block_cols as usize,
            component.block_rows as usize,
            component.width as usize,
            component.height as usize,
        )?;
        let reference =
            decompose_from_first_level(first_reference_level, usize::from(decomposition_levels));
        Some((
            rounded_wavelet_i32(&wavelet)?,
            rounded_wavelet_i32(&reference)?,
        ))
    } else {
        None
    };
    let integer_validation_coefficients = if validate_against_integer_reference {
        let expected = integer_reference_coefficients(component, decomposition_levels)?;
        Some((rounded_wavelet_i32(&wavelet)?, expected))
    } else {
        None
    };
    let dwt = j2k_dwt_from_wavelet(
        &wavelet,
        component.width as usize,
        component.height as usize,
    );

    Ok(ComponentTranscodeResult {
        precomputed: PrecomputedHtj2k53Component {
            x_rsiz,
            y_rsiz,
            dwt,
        },
        float_validation_coefficients,
        integer_validation_coefficients,
    })
}

fn decompose_from_first_level(
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

fn j2k_dwt_from_wavelet(
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

fn rounded_wavelet_i32(wavelet: &ComponentWavelet) -> Result<Vec<i32>, JpegToHtj2kError> {
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

fn integer_reference_coefficients(
    component: &JpegDctComponent,
    decomposition_levels: u8,
) -> Result<Vec<i32>, JpegToHtj2kError> {
    let samples = idct_component_samples_i32(component)?;
    let wavelet = reversible_dwt53_i32(
        samples,
        component.width as usize,
        component.height as usize,
        usize::from(decomposition_levels),
    );
    Ok(flatten_integer_wavelet(&wavelet))
}

fn idct_component_samples_i32(component: &JpegDctComponent) -> Result<Vec<i32>, JpegToHtj2kError> {
    let width = component.width as usize;
    let height = component.height as usize;
    let block_cols = component.block_cols as usize;
    let block_rows = component.block_rows as usize;
    let expected_blocks =
        block_cols
            .checked_mul(block_rows)
            .ok_or(JpegToHtj2kError::Validation(
                "component block grid overflow",
            ))?;
    if component.dequantized_blocks.len() != expected_blocks {
        return Err(JpegToHtj2kError::Validation(
            "component block count does not match block grid",
        ));
    }

    let mut samples = vec![0; width * height];
    for block_y in 0..block_rows {
        for block_x in 0..block_cols {
            let block = &component.dequantized_blocks[block_y * block_cols + block_x];
            let block_samples = idct_islow_block(block);
            for local_y in 0..8 {
                let y = block_y * 8 + local_y;
                if y >= height {
                    continue;
                }
                for local_x in 0..8 {
                    let x = block_x * 8 + local_x;
                    if x >= width {
                        continue;
                    }
                    samples[y * width + x] = i32::from(block_samples[local_y * 8 + local_x]) - 128;
                }
            }
        }
    }

    Ok(samples)
}

fn reversible_dwt53_i32(
    mut buffer: Vec<i32>,
    width: usize,
    height: usize,
    decomposition_levels: usize,
) -> IntegerWavelet {
    let mut current_width = width;
    let mut current_height = height;
    let mut levels = Vec::with_capacity(decomposition_levels);

    for _ in 0..decomposition_levels {
        for x in 0..current_width {
            let mut column = Vec::with_capacity(current_height);
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
            let mut row = buffer[row_start..row_start + current_width].to_vec();
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
        let mut hl = Vec::with_capacity(high_width * low_height);
        let mut lh = Vec::with_capacity(low_width * high_height);
        let mut hh = Vec::with_capacity(high_width * high_height);

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

        levels.push(IntegerWaveletLevel { hl, lh, hh });
        current_width = low_width;
        current_height = low_height;
    }

    let mut final_ll = Vec::with_capacity(current_width * current_height);
    for y in 0..current_height {
        for x in 0..current_width {
            final_ll.push(buffer[y * width + x]);
        }
    }

    IntegerWavelet { final_ll, levels }
}

fn reversible_lift_53_i32(values: &mut [i32]) {
    let n = values.len();
    if n < 2 {
        return;
    }

    if n.is_multiple_of(2) {
        for i in (1..n - 1).step_by(2) {
            values[i] -= floor_div_i32(values[i - 1] + values[i + 1], 2);
        }
        values[n - 1] -= values[n - 2];

        values[0] += floor_div_i32(values[1] + 1, 2);
        for i in (2..n).step_by(2) {
            values[i] += floor_div_i32(values[i - 1] + values[i + 1] + 2, 4);
        }
        return;
    }

    let last_even = n - 1;
    for i in (1..n).step_by(2) {
        let right = values.get(i + 1).copied().unwrap_or(values[last_even]);
        values[i] -= floor_div_i32(values[i - 1] + right, 2);
    }
    for i in (0..n).step_by(2) {
        let left = if i > 0 { values[i - 1] } else { values[1] };
        let right = values.get(i + 1).copied().unwrap_or(left);
        values[i] += floor_div_i32(left + right + 2, 4);
    }
}

fn floor_div_i32(numerator: i32, denominator: i32) -> i32 {
    numerator.div_euclid(denominator)
}

fn flatten_integer_wavelet(wavelet: &IntegerWavelet) -> Vec<i32> {
    let coefficient_count = wavelet.final_ll.len()
        + wavelet
            .levels
            .iter()
            .map(|level| level.hl.len() + level.lh.len() + level.hh.len())
            .sum::<usize>();
    let mut output = Vec::with_capacity(coefficient_count);
    output.extend_from_slice(&wavelet.final_ll);
    for level in wavelet.levels.iter().rev() {
        output.extend_from_slice(&level.hl);
        output.extend_from_slice(&level.lh);
        output.extend_from_slice(&level.hh);
    }
    output
}

fn append_rounded_i32(values: &[f64], output: &mut Vec<i32>) -> Result<(), JpegToHtj2kError> {
    for &value in values {
        output.push(round_f64_to_i32(value)?);
    }
    Ok(())
}

fn round_f64_to_i32(value: f64) -> Result<i32, JpegToHtj2kError> {
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

fn decomposition_levels_for_components(
    components: &[JpegDctComponent],
    requested_levels: u8,
) -> Result<u8, JpegToHtj2kError> {
    if requested_levels == 0 {
        return Err(JpegToHtj2kError::Unsupported(
            "jpeg_to_htj2k requires at least one decomposition level",
        ));
    }

    let available_levels = components
        .iter()
        .map(|component| available_decomposition_levels(component.width, component.height))
        .min()
        .ok_or(JpegToHtj2kError::Unsupported("missing JPEG components"))?;
    let decomposition_levels = requested_levels.min(available_levels);
    if decomposition_levels == 0 {
        return Err(JpegToHtj2kError::Unsupported(
            "component dimensions are too small for a 5/3 decomposition",
        ));
    }

    Ok(decomposition_levels)
}

fn available_decomposition_levels(width: u32, height: u32) -> u8 {
    let min_dim = width.min(height);
    if min_dim <= 1 {
        0
    } else {
        min_dim.ilog2() as u8
    }
}

fn component_sampling_for_jpeg(
    components: &[JpegDctComponent],
    reference_width: u32,
    reference_height: u32,
) -> Result<Vec<(u8, u8)>, JpegToHtj2kError> {
    let max_h = components
        .iter()
        .map(|component| component.h_samp)
        .max()
        .ok_or(JpegToHtj2kError::Unsupported("missing JPEG components"))?;
    let max_v = components
        .iter()
        .map(|component| component.v_samp)
        .max()
        .ok_or(JpegToHtj2kError::Unsupported("missing JPEG components"))?;

    components
        .iter()
        .map(|component| {
            if component.h_samp == 0 || component.v_samp == 0 {
                return Err(JpegToHtj2kError::Unsupported(
                    "JPEG component sampling factors must be non-zero",
                ));
            }
            if max_h % component.h_samp != 0 || max_v % component.v_samp != 0 {
                return Err(JpegToHtj2kError::Unsupported(
                    "fractional JPEG component sampling is not supported",
                ));
            }

            let x_rsiz = max_h / component.h_samp;
            let y_rsiz = max_v / component.v_samp;
            let expected_width = reference_width.div_ceil(u32::from(x_rsiz));
            let expected_height = reference_height.div_ceil(u32::from(y_rsiz));
            if component.width != expected_width || component.height != expected_height {
                return Err(JpegToHtj2kError::Unsupported(
                    "JPEG component dimensions do not match derived SIZ sampling",
                ));
            }

            Ok((x_rsiz, y_rsiz))
        })
        .collect()
}

fn dct_blocks_to_8x8_f64_into(blocks: &[[i16; 64]], output: &mut Vec<[[f64; 8]; 8]>) {
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
