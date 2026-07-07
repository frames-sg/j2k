// SPDX-License-Identifier: MIT OR Apache-2.0

//! Experimental JPEG DCT to HTJ2K codestream transcode entry point.

use core::fmt;
use std::time::Instant;

use j2k::{
    CpuOnlyJ2kEncodeStageAccelerator, J2kEncodeDispatchReport, J2kEncodeStageAccelerator,
    J2kForwardDwt53Level, J2kForwardDwt53Output, J2kForwardDwt97Level, J2kForwardDwt97Output,
    PrecomputedHtj2k53Component, PrecomputedHtj2k53Image, PrecomputedHtj2k97Component,
    PrecomputedHtj2k97Image, PreencodedHtj2k97CompactComponent, PreencodedHtj2k97CompactImage,
    PreencodedHtj2k97Component, PreencodedHtj2k97Image, PrequantizedHtj2k97Component,
    PrequantizedHtj2k97Image,
};
use j2k_jpeg::transcode::{
    extract_dct_blocks, idct_islow_block, DctExtractOptions, JpegDctComponent, JpegDctImage,
};
use j2k_native::{
    encode_precomputed_htj2k_53_with_accelerator,
    encode_precomputed_htj2k_97_batch_with_accelerator,
    encode_precomputed_htj2k_97_with_accelerator,
    encode_preencoded_htj2k_97_compact_owned_with_accelerator,
    encode_preencoded_htj2k_97_owned_with_accelerator,
    encode_prequantized_htj2k_97_with_accelerator,
};
use rayon::prelude::*;

use crate::dct53_2d::{
    dct8x8_blocks_then_dwt53_float, dct8x8_blocks_to_dwt53_float_linear_with_scratch,
    linearized_53_2d_from_plane, Dct53GridScratch,
};
use crate::dct97_2d::{
    dct8x8_blocks_then_dwt97_float, dct8x8_blocks_then_dwt97_float_with_scratch,
    linearized_97_2d_from_plane_with_scratch, Dct97GridScratch,
};
use crate::metrics::{error_metrics_i32, ErrorMetrics, MetricsLengthError};
use crate::reversible53::{
    reversible_lift_53_high_at_fallible, reversible_lift_53_i32, reversible_lift_53_low_at_fallible,
};
use crate::{
    CpuOnlyDctToWaveletStageAccelerator, DctGridError, DctGridI16ToHtj2k97CodeBlockBatch,
    DctGridI16ToHtj2k97CodeBlockJob, DctGridToDwt53Job, DctGridToDwt97Job,
    DctGridToHtj2k97CodeBlockJob, DctGridToReversibleDwt53Job, DctToWaveletStageAccelerator,
    Dwt53TwoDimensional, Dwt97BatchStageTimings, Dwt97TwoDimensional, Htj2k97CodeBlockOptions,
    ReversibleDwt53FirstLevel, TranscodeStageError,
};

mod options;
pub use self::options::{
    JpegToHtj2kCoefficientPath, JpegToHtj2kEncodeOptions, JpegToHtj2kOptions,
    JPEG_TO_HTJ2K_LOSSY_97_QUANTIZATION_SCALE,
};
mod report;
pub use self::report::*;
mod error;
pub use self::error::*;
mod batch;
pub use self::batch::*;

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
        let mut accelerator = CpuOnlyDctToWaveletStageAccelerator;
        self.transcode_with_accelerator(bytes, options, &mut accelerator)
    }

    /// Transcode with an optional stage accelerator.
    ///
    /// Accelerators may handle direct DCT-grid projection stages and return
    /// `None` for scalar fallback. Integer-direct 5/3 is offered in
    /// same-geometry batches before falling back to per-component work.
    pub fn transcode_with_accelerator<A: DctToWaveletStageAccelerator>(
        &mut self,
        bytes: &[u8],
        options: &JpegToHtj2kOptions,
        accelerator: &mut A,
    ) -> Result<EncodedTranscode, JpegToHtj2kError> {
        let mut encode_accelerator = CpuOnlyJ2kEncodeStageAccelerator;
        self.transcode_with_accelerators(bytes, options, accelerator, &mut encode_accelerator)
    }

    /// Transcode with separate transform-stage and HTJ2K encode-stage
    /// accelerators.
    pub fn transcode_with_accelerators<
        A: DctToWaveletStageAccelerator,
        E: J2kEncodeStageAccelerator,
    >(
        &mut self,
        bytes: &[u8],
        options: &JpegToHtj2kOptions,
        transform_accelerator: &mut A,
        encode_accelerator: &mut E,
    ) -> Result<EncodedTranscode, JpegToHtj2kError> {
        jpeg_to_htj2k_with_scratch(
            bytes,
            options,
            &mut self.scratch,
            transform_accelerator,
            encode_accelerator,
        )
    }

    /// Transcode many JPEG tiles, preserving per-tile failures in the returned
    /// batch. Integer-direct 5/3 groups same-geometry components across tiles
    /// before calling the accelerator.
    pub fn transcode_batch(
        &mut self,
        tiles: &[JpegTileBatchInput<'_>],
        options: &JpegToHtj2kOptions,
    ) -> Result<EncodedTranscodeBatch, JpegToHtj2kError> {
        let mut accelerator = CpuOnlyDctToWaveletStageAccelerator;
        self.transcode_batch_with_accelerator(tiles, options, &mut accelerator)
    }

    /// Transcode many JPEG tiles with an optional stage accelerator.
    pub fn transcode_batch_with_accelerator<A: DctToWaveletStageAccelerator>(
        &mut self,
        tiles: &[JpegTileBatchInput<'_>],
        options: &JpegToHtj2kOptions,
        accelerator: &mut A,
    ) -> Result<EncodedTranscodeBatch, JpegToHtj2kError> {
        let mut encode_accelerator = CpuOnlyJ2kEncodeStageAccelerator;
        self.transcode_batch_with_accelerators(tiles, options, accelerator, &mut encode_accelerator)
    }

    /// Transcode many JPEG tiles with separate transform-stage and HTJ2K
    /// encode-stage accelerators.
    pub fn transcode_batch_with_accelerators<
        A: DctToWaveletStageAccelerator,
        E: J2kEncodeStageAccelerator,
    >(
        &mut self,
        tiles: &[JpegTileBatchInput<'_>],
        options: &JpegToHtj2kOptions,
        transform_accelerator: &mut A,
        encode_accelerator: &mut E,
    ) -> Result<EncodedTranscodeBatch, JpegToHtj2kError> {
        jpeg_tile_batch_to_htj2k_with_scratch(
            tiles,
            options,
            &mut self.scratch,
            transform_accelerator,
            encode_accelerator,
        )
    }
}

#[derive(Debug, Default)]
struct JpegToHtj2kScratch {
    dct_blocks_f64: Vec<[[f64; 8]; 8]>,
    dct53_grid: Dct53GridScratch,
    dct97_grid: Dct97GridScratch,
    integer_idct_blocks: Vec<Option<[i32; 64]>>,
    integer_row: Vec<i32>,
}

/// Encoded transcode output and validation/report metadata.
#[derive(Debug, Clone)]
pub struct EncodedTranscode {
    /// HTJ2K codestream bytes.
    pub codestream: Vec<u8>,
    /// Summary of the experimental path used.
    pub report: TranscodeReport,
}

/// One JPEG tile input for batch transcode.
#[derive(Debug, Clone, Copy)]
pub struct JpegTileBatchInput<'a> {
    /// JPEG codestream bytes for one tile.
    pub bytes: &'a [u8],
}

/// Batch transcode output. Tile-level parse/encode failures are preserved so a
/// WSI ingest queue can continue past isolated bad tiles.
#[derive(Debug)]
pub struct EncodedTranscodeBatch {
    /// Per-input tile result in input order.
    pub tiles: Vec<Result<EncodedTranscode, JpegToHtj2kError>>,
    /// Aggregate batch report.
    pub report: BatchTranscodeReport,
}

/// Transcode a constrained baseline grayscale JPEG tile into an HTJ2K
/// codestream using direct DCT-domain wavelet coefficients.
///
/// Current implementation scope is baseline JPEG with one or more components
/// at native JPEG component resolution. Component subsampling is preserved
/// through SIZ `XRsiz`/`YRsiz` instead of chroma upsampling.
pub fn jpeg_to_htj2k(
    bytes: &[u8],
    options: &JpegToHtj2kOptions,
) -> Result<EncodedTranscode, JpegToHtj2kError> {
    JpegToHtj2kTranscoder::default().transcode(bytes, options)
}

#[allow(clippy::too_many_lines)]
fn jpeg_to_htj2k_with_scratch<A: DctToWaveletStageAccelerator, E: J2kEncodeStageAccelerator>(
    bytes: &[u8],
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut A,
    encode_accelerator: &mut E,
) -> Result<EncodedTranscode, JpegToHtj2kError> {
    validate_transcode_options(options)?;
    let mut timings = TranscodeTimingReport {
        tile_count: 1,
        ..TranscodeTimingReport::default()
    };

    let extract_start = Instant::now();
    let jpeg = extract_dct_blocks(bytes, DctExtractOptions::default())?;
    let extract_us = extract_start.elapsed().as_micros();
    timings.jpeg_dct_extract_us = extract_us;

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
        accelerator,
        &mut timings,
    )?;
    let transform_us = transform_start.elapsed().as_micros();
    timings.dct_to_wavelet_total_us = transform_us;

    let encode_start = Instant::now();
    let encode_dispatch_before = encode_accelerator.dispatch_report();
    let native_encode_options = options.encode_options.to_native();
    let codestream = match component_batch.precomputed_components {
        PrecomputedComponentBatch::Dwt53(components) => {
            let precomputed = PrecomputedHtj2k53Image {
                width: jpeg.width,
                height: jpeg.height,
                bit_depth: 8,
                signed: false,
                components,
            };
            let native_precomputed = precomputed;
            encode_precomputed_htj2k_53_with_accelerator(
                &native_precomputed,
                &native_encode_options,
                encode_accelerator,
            )
            .map_err(JpegToHtj2kError::Encode)?
        }
        PrecomputedComponentBatch::Dwt97(components) => {
            let precomputed = PrecomputedHtj2k97Image {
                width: jpeg.width,
                height: jpeg.height,
                bit_depth: 8,
                signed: false,
                components,
            };
            let native_precomputed = precomputed;
            encode_precomputed_htj2k_97_with_accelerator(
                &native_precomputed,
                &native_encode_options,
                encode_accelerator,
            )
            .map_err(JpegToHtj2kError::Encode)?
        }
    };
    record_encode_dispatch_delta(
        &mut timings,
        encode_dispatch_before,
        encode_accelerator.dispatch_report(),
    );
    let encode_us = encode_start.elapsed().as_micros();
    timings.htj2k_encode_us = encode_us;

    Ok(EncodedTranscode {
        codestream,
        report: TranscodeReport {
            width: jpeg.width,
            height: jpeg.height,
            component_count: jpeg.components.len(),
            components: component_reports,
            float_reference_classification: component_batch
                .float_reference_metrics
                .as_ref()
                .map(TranscodeValidationClassification::classify_metrics),
            float_reference_metrics: component_batch.float_reference_metrics,
            integer_reference_classification: component_batch
                .integer_reference_metrics
                .as_ref()
                .map(TranscodeValidationClassification::classify_metrics),
            integer_reference_metrics: component_batch.integer_reference_metrics,
            decomposition_levels,
            coefficient_path: options.coefficient_path,
            path: transcode_path_name(all_unit_sampled, options.coefficient_path),
            extract_us,
            transform_us,
            encode_us,
            timings,
        },
    })
}

fn validate_transcode_options(options: &JpegToHtj2kOptions) -> Result<(), JpegToHtj2kError> {
    if !options.encode_options.use_ht_block_coding {
        return Err(JpegToHtj2kError::Unsupported(
            "jpeg_to_htj2k requires HT block coding",
        ));
    }
    if options.encode_options.use_mct {
        return Err(JpegToHtj2kError::Unsupported(
            "jpeg_to_htj2k requires use_mct=false because JPEG components stay in native color space",
        ));
    }

    match (options.coefficient_path, options.encode_options.reversible) {
        (
            JpegToHtj2kCoefficientPath::IntegerDirect53
            | JpegToHtj2kCoefficientPath::FloatDirectLinear53,
            true,
        )
        | (JpegToHtj2kCoefficientPath::FloatDirectLinear97, false) => Ok(()),
        (
            JpegToHtj2kCoefficientPath::IntegerDirect53
            | JpegToHtj2kCoefficientPath::FloatDirectLinear53,
            false,
        ) => Err(JpegToHtj2kError::Unsupported(
            "5/3 coefficient path requires reversible HTJ2K encode",
        )),
        (JpegToHtj2kCoefficientPath::FloatDirectLinear97, true) => {
            Err(JpegToHtj2kError::Unsupported(
                "9/7 coefficient path requires irreversible HTJ2K encode",
            ))
        }
    }
}

struct ComponentTranscodeBatch {
    precomputed_components: PrecomputedComponentBatch,
    float_reference_metrics: Option<TranscodeValidationMetrics>,
    integer_reference_metrics: Option<TranscodeValidationMetrics>,
}

enum PrecomputedComponentBatch {
    Dwt53(Vec<PrecomputedHtj2k53Component>),
    Dwt97(Vec<PrecomputedHtj2k97Component>),
}

struct ComponentTranscodeResult {
    precomputed: PrecomputedComponent,
    float_validation_coefficients: Option<(Vec<i32>, Vec<i32>)>,
    integer_validation_coefficients: Option<(Vec<i32>, Vec<i32>)>,
}

enum PrecomputedComponent {
    Dwt53(PrecomputedHtj2k53Component),
    Dwt97(PrecomputedHtj2k97Component),
}

struct ComponentWavelet {
    final_ll: Vec<f64>,
    final_ll_width: usize,
    final_ll_height: usize,
    levels: Vec<Dwt53TwoDimensional<f64>>,
}

struct ComponentWavelet97 {
    final_ll: Vec<f64>,
    final_ll_width: usize,
    final_ll_height: usize,
    levels: Vec<Dwt97TwoDimensional<f64>>,
}

struct IntegerWaveletLevel {
    width: usize,
    height: usize,
    low_width: usize,
    low_height: usize,
    high_width: usize,
    high_height: usize,
    hl: Vec<i32>,
    lh: Vec<i32>,
    hh: Vec<i32>,
}

struct IntegerWavelet {
    final_ll: Vec<i32>,
    final_ll_width: usize,
    final_ll_height: usize,
    levels: Vec<IntegerWaveletLevel>,
}

fn transcode_component_batch(
    components: &[JpegDctComponent],
    component_sampling: &[(u8, u8)],
    decomposition_levels: u8,
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut impl DctToWaveletStageAccelerator,
    timings: &mut TranscodeTimingReport,
) -> Result<ComponentTranscodeBatch, JpegToHtj2kError> {
    if matches!(
        options.coefficient_path,
        JpegToHtj2kCoefficientPath::FloatDirectLinear97
    ) && options.validate_against_integer_reference
    {
        return Err(JpegToHtj2kError::Unsupported(
            "integer reversible validation is only defined for 5/3 coefficient paths",
        ));
    }

    if matches!(
        options.coefficient_path,
        JpegToHtj2kCoefficientPath::IntegerDirect53
    ) {
        return transcode_integer_component_batch(
            components,
            component_sampling,
            decomposition_levels,
            options,
            scratch,
            accelerator,
            timings,
        );
    }

    let mut precomputed_53 = Vec::with_capacity(components.len());
    let mut precomputed_97 = Vec::with_capacity(components.len());
    let mut float_validation_actual = Vec::new();
    let mut float_validation_expected = Vec::new();
    let mut integer_validation_actual = Vec::new();
    let mut integer_validation_expected = Vec::new();

    for (component, (x_rsiz, y_rsiz)) in components.iter().zip(component_sampling.iter().copied()) {
        let component_result = component_to_precomputed_htj2k(
            ComponentTranscodePlan {
                component,
                x_rsiz,
                y_rsiz,
                decomposition_levels,
                options,
            },
            scratch,
            accelerator,
            timings,
        )?;
        match component_result.precomputed {
            PrecomputedComponent::Dwt53(precomputed) => precomputed_53.push(precomputed),
            PrecomputedComponent::Dwt97(precomputed) => precomputed_97.push(precomputed),
        }
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

    let precomputed_components = if matches!(
        options.coefficient_path,
        JpegToHtj2kCoefficientPath::FloatDirectLinear97
    ) {
        PrecomputedComponentBatch::Dwt97(precomputed_97)
    } else {
        PrecomputedComponentBatch::Dwt53(precomputed_53)
    };

    Ok(ComponentTranscodeBatch {
        precomputed_components,
        float_reference_metrics,
        integer_reference_metrics,
    })
}

fn transcode_integer_component_batch(
    components: &[JpegDctComponent],
    component_sampling: &[(u8, u8)],
    decomposition_levels: u8,
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut impl DctToWaveletStageAccelerator,
    timings: &mut TranscodeTimingReport,
) -> Result<ComponentTranscodeBatch, JpegToHtj2kError> {
    let mut precomputed_53: Vec<Option<PrecomputedHtj2k53Component>> =
        (0..components.len()).map(|_| None).collect();
    let mut float_validation_actual = Vec::new();
    let mut float_validation_expected = Vec::new();
    let mut integer_validation_actual = Vec::new();
    let mut integer_validation_expected = Vec::new();

    for group in same_geometry_component_groups(components) {
        let group_wavelets = integer_wavelets_for_component_group(
            &group,
            components,
            decomposition_levels,
            scratch,
            accelerator,
            timings,
        )?;
        for (component_index, wavelet) in group.into_iter().zip(group_wavelets) {
            let component = &components[component_index];
            let (x_rsiz, y_rsiz) = component_sampling[component_index];
            let actual_coefficients = flatten_integer_wavelet(&wavelet);
            precomputed_53[component_index] = Some(PrecomputedHtj2k53Component {
                x_rsiz,
                y_rsiz,
                dwt: j2k_dwt_from_integer_wavelet(&wavelet),
            });

            if options.validate_against_float_reference {
                float_validation_actual.extend(actual_coefficients.clone());
                float_validation_expected.extend(float_reference_coefficients(
                    component,
                    decomposition_levels,
                    scratch,
                )?);
            }
            if options.validate_against_integer_reference {
                integer_validation_actual.extend(actual_coefficients);
                integer_validation_expected.extend(integer_reference_coefficients(
                    component,
                    decomposition_levels,
                )?);
            }
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
    let precomputed_components = precomputed_53
        .into_iter()
        .map(|component| {
            component.ok_or(JpegToHtj2kError::Validation(
                "integer transcode did not produce all components",
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(ComponentTranscodeBatch {
        precomputed_components: PrecomputedComponentBatch::Dwt53(precomputed_components),
        float_reference_metrics,
        integer_reference_metrics,
    })
}

fn integer_wavelets_for_component_group(
    group: &[usize],
    components: &[JpegDctComponent],
    decomposition_levels: u8,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut impl DctToWaveletStageAccelerator,
    timings: &mut TranscodeTimingReport,
) -> Result<Vec<IntegerWavelet>, JpegToHtj2kError> {
    let jobs = group
        .iter()
        .map(|&component_index| integer_dct_job_for_component(&components[component_index]))
        .collect::<Result<Vec<_>, _>>()?;
    record_batch_attempt(timings, group.len());
    let accelerator_start = Instant::now();
    let accelerated_first_levels = accelerator
        .dct_grid_to_reversible_dwt53_batch(&jobs)
        .map_err(JpegToHtj2kError::Accelerator)?;
    timings.dct_to_wavelet_accelerator_us = timings
        .dct_to_wavelet_accelerator_us
        .saturating_add(accelerator_start.elapsed().as_micros());

    if let Some(first_levels) = accelerated_first_levels {
        if first_levels.len() != group.len() {
            return Err(JpegToHtj2kError::Validation(
                "reversible 5/3 batch accelerator returned wrong component count",
            ));
        }
        timings.component_count = timings.component_count.saturating_add(group.len());
        record_accelerator_dispatch(timings, group.len());
        let decompose_start = Instant::now();
        let wavelets = first_levels
            .into_iter()
            .map(|first_level| integer_wavelet_from_first_level(first_level, decomposition_levels))
            .collect();
        timings.dwt_decompose_us = timings
            .dwt_decompose_us
            .saturating_add(decompose_start.elapsed().as_micros());
        return Ok(wavelets);
    }

    group
        .iter()
        .map(|&component_index| {
            integer_direct_wavelet_from_component(
                &components[component_index],
                decomposition_levels,
                scratch,
                accelerator,
                timings,
            )
        })
        .collect()
}

fn same_geometry_component_groups(components: &[JpegDctComponent]) -> Vec<Vec<usize>> {
    let mut assigned = vec![false; components.len()];
    let mut groups = Vec::new();

    for component_index in 0..components.len() {
        if assigned[component_index] {
            continue;
        }
        assigned[component_index] = true;
        let mut group = vec![component_index];
        for candidate_index in component_index + 1..components.len() {
            if !assigned[candidate_index]
                && same_component_geometry(
                    &components[component_index],
                    &components[candidate_index],
                )
            {
                assigned[candidate_index] = true;
                group.push(candidate_index);
            }
        }
        groups.push(group);
    }

    groups
}

fn same_component_geometry(left: &JpegDctComponent, right: &JpegDctComponent) -> bool {
    left.width == right.width
        && left.height == right.height
        && left.block_cols == right.block_cols
        && left.block_rows == right.block_rows
}

fn integer_dct_job_for_component(
    component: &JpegDctComponent,
) -> Result<DctGridToReversibleDwt53Job<'_>, JpegToHtj2kError> {
    validate_component_block_grid(component)?;
    Ok(DctGridToReversibleDwt53Job {
        dequantized_blocks: &component.dequantized_blocks,
        block_cols: component.block_cols as usize,
        block_rows: component.block_rows as usize,
        width: component.width as usize,
        height: component.height as usize,
    })
}

#[derive(Clone, Copy)]
struct ComponentTranscodePlan<'a> {
    component: &'a JpegDctComponent,
    x_rsiz: u8,
    y_rsiz: u8,
    decomposition_levels: u8,
    options: &'a JpegToHtj2kOptions,
}

fn component_to_precomputed_htj2k(
    plan: ComponentTranscodePlan<'_>,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut impl DctToWaveletStageAccelerator,
    timings: &mut TranscodeTimingReport,
) -> Result<ComponentTranscodeResult, JpegToHtj2kError> {
    let ComponentTranscodePlan {
        component,
        x_rsiz,
        y_rsiz,
        decomposition_levels,
        options,
    } = plan;
    let (dwt, actual_coefficients) = match options.coefficient_path {
        JpegToHtj2kCoefficientPath::IntegerDirect53 => {
            let wavelet = integer_direct_wavelet_from_component(
                component,
                decomposition_levels,
                scratch,
                accelerator,
                timings,
            )?;
            (
                PrecomputedComponent::Dwt53(PrecomputedHtj2k53Component {
                    x_rsiz,
                    y_rsiz,
                    dwt: j2k_dwt_from_integer_wavelet(&wavelet),
                }),
                flatten_integer_wavelet(&wavelet),
            )
        }
        JpegToHtj2kCoefficientPath::FloatDirectLinear53 => {
            let wavelet = float_direct_wavelet_from_component(
                component,
                decomposition_levels,
                scratch,
                accelerator,
                timings,
            )?;
            (
                PrecomputedComponent::Dwt53(PrecomputedHtj2k53Component {
                    x_rsiz,
                    y_rsiz,
                    dwt: j2k_dwt_from_wavelet(
                        &wavelet,
                        component.width as usize,
                        component.height as usize,
                    ),
                }),
                rounded_wavelet_i32(&wavelet)?,
            )
        }
        JpegToHtj2kCoefficientPath::FloatDirectLinear97 => {
            let wavelet = float_direct_97_wavelet_from_component(
                component,
                decomposition_levels,
                scratch,
                accelerator,
                timings,
            )?;
            (
                PrecomputedComponent::Dwt97(PrecomputedHtj2k97Component {
                    x_rsiz,
                    y_rsiz,
                    dwt: j2k_dwt97_from_wavelet(
                        &wavelet,
                        component.width as usize,
                        component.height as usize,
                    ),
                }),
                rounded_wavelet97_i32(&wavelet)?,
            )
        }
    };
    let float_validation_coefficients = if options.validate_against_float_reference {
        let expected = match options.coefficient_path {
            JpegToHtj2kCoefficientPath::FloatDirectLinear97 => {
                float97_reference_coefficients(component, decomposition_levels, scratch)?
            }
            JpegToHtj2kCoefficientPath::IntegerDirect53
            | JpegToHtj2kCoefficientPath::FloatDirectLinear53 => {
                float_reference_coefficients(component, decomposition_levels, scratch)?
            }
        };
        Some((actual_coefficients.clone(), expected))
    } else {
        None
    };
    let integer_validation_coefficients = if options.validate_against_integer_reference {
        let expected = integer_reference_coefficients(component, decomposition_levels)?;
        Some((actual_coefficients, expected))
    } else {
        None
    };

    Ok(ComponentTranscodeResult {
        precomputed: dwt,
        float_validation_coefficients,
        integer_validation_coefficients,
    })
}

fn transcode_path_name(
    all_unit_sampled: bool,
    coefficient_path: JpegToHtj2kCoefficientPath,
) -> &'static str {
    match (all_unit_sampled, coefficient_path) {
        (true, JpegToHtj2kCoefficientPath::IntegerDirect53) => {
            "full_resolution_components_integer_direct_53"
        }
        (false, JpegToHtj2kCoefficientPath::IntegerDirect53) => {
            "native_component_sampling_integer_direct_53"
        }
        (true, JpegToHtj2kCoefficientPath::FloatDirectLinear53) => {
            "full_resolution_components_float_direct_53"
        }
        (false, JpegToHtj2kCoefficientPath::FloatDirectLinear53) => {
            "native_component_sampling_float_direct_53"
        }
        (true, JpegToHtj2kCoefficientPath::FloatDirectLinear97) => {
            "full_resolution_components_float_direct_97"
        }
        (false, JpegToHtj2kCoefficientPath::FloatDirectLinear97) => {
            "native_component_sampling_float_direct_97"
        }
    }
}

fn float_direct_wavelet_from_component(
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

fn float_direct_97_wavelet_from_component(
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

fn float_reference_coefficients(
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

fn float97_reference_coefficients(
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

fn decompose_97_from_first_level(
    first_level: Dwt97TwoDimensional<f64>,
    decomposition_levels: usize,
) -> ComponentWavelet97 {
    let mut scratch = Dct97GridScratch::default();
    decompose_97_from_first_level_with_scratch(first_level, decomposition_levels, &mut scratch)
}

fn decompose_97_from_first_level_with_scratch(
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

fn j2k_dwt97_from_wavelet(
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

fn j2k_dwt_from_integer_wavelet(wavelet: &IntegerWavelet) -> J2kForwardDwt53Output {
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

fn rounded_wavelet97_i32(wavelet: &ComponentWavelet97) -> Result<Vec<i32>, JpegToHtj2kError> {
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

fn integer_direct_wavelet_from_component(
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
        let wavelet = integer_wavelet_from_first_level(first_level, decomposition_levels);
        timings.dwt_decompose_us = timings
            .dwt_decompose_us
            .saturating_add(decompose_start.elapsed().as_micros());
        return Ok(wavelet);
    }

    scratch.integer_idct_blocks.clear();
    scratch
        .integer_idct_blocks
        .resize_with(component.dequantized_blocks.len(), || None);
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
    );
    timings.dwt_decompose_us = timings
        .dwt_decompose_us
        .saturating_add(decompose_start.elapsed().as_micros());
    Ok(wavelet)
}

fn integer_wavelet_from_first_level(
    first_level: ReversibleDwt53FirstLevel,
    decomposition_levels: u8,
) -> IntegerWavelet {
    let (final_ll, final_ll_width, final_ll_height, first_level) =
        integer_wavelet_first_level_from_accelerated(first_level);
    integer_wavelet_from_first_parts(
        final_ll,
        final_ll_width,
        final_ll_height,
        first_level,
        decomposition_levels,
    )
}

fn integer_wavelet_from_first_parts(
    mut final_ll: Vec<i32>,
    mut final_ll_width: usize,
    mut final_ll_height: usize,
    first_level: IntegerWaveletLevel,
    decomposition_levels: u8,
) -> IntegerWavelet {
    let mut levels = vec![first_level];

    let remaining_levels = usize::from(decomposition_levels.saturating_sub(1));
    if remaining_levels > 0 {
        let tail =
            reversible_dwt53_i32(final_ll, final_ll_width, final_ll_height, remaining_levels);
        final_ll = tail.final_ll;
        final_ll_width = tail.final_ll_width;
        final_ll_height = tail.final_ll_height;
        levels.extend(tail.levels);
    }

    IntegerWavelet {
        final_ll,
        final_ll_width,
        final_ll_height,
        levels,
    }
}

fn integer_wavelet_first_level_from_accelerated(
    first_level: ReversibleDwt53FirstLevel,
) -> (Vec<i32>, usize, usize, IntegerWaveletLevel) {
    let level = IntegerWaveletLevel {
        width: first_level.low_width + first_level.high_width,
        height: first_level.low_height + first_level.high_height,
        low_width: first_level.low_width,
        low_height: first_level.low_height,
        high_width: first_level.high_width,
        high_height: first_level.high_height,
        hl: first_level.hl,
        lh: first_level.lh,
        hh: first_level.hh,
    };
    (
        first_level.ll,
        first_level.low_width,
        first_level.low_height,
        level,
    )
}

fn integer_direct_first_level_from_component(
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

    let mut ll = Vec::with_capacity(low_width * low_height);
    let mut hl = Vec::with_capacity(high_width * low_height);
    let mut lh = Vec::with_capacity(low_width * high_height);
    let mut hh = Vec::with_capacity(high_width * high_height);
    row.clear();
    if row.capacity() < width {
        row.reserve(width - row.capacity());
    }

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

fn vertical_53_i32_at(
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

fn vertical_low_53_i32_at(
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

fn vertical_high_53_i32_at(
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

fn component_sample_i32(
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
    validate_component_block_grid(component)?;

    let width = component.width as usize;
    let height = component.height as usize;
    let block_cols = component.block_cols as usize;
    let block_rows = component.block_rows as usize;
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

fn validate_component_block_grid(component: &JpegDctComponent) -> Result<(), JpegToHtj2kError> {
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

    Ok(())
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

    let mut final_ll = Vec::with_capacity(current_width * current_height);
    for y in 0..current_height {
        for x in 0..current_width {
            final_ll.push(buffer[y * width + x]);
        }
    }

    IntegerWavelet {
        final_ll,
        final_ll_width: current_width,
        final_ll_height: current_height,
        levels,
    }
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
            "component dimensions are too small for a DWT decomposition",
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

fn dct_blocks_to_8x8_f64(blocks: &[[i16; 64]]) -> Vec<[[f64; 8]; 8]> {
    let mut output = Vec::with_capacity(blocks.len());
    dct_blocks_to_8x8_f64_into(blocks, &mut output);
    output
}

#[cfg(test)]
mod tests;
