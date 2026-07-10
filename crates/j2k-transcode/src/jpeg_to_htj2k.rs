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
pub use self::report::{
    BatchTranscodeReport, TranscodeBatchProfileRequest, TranscodeBatchProfileRow,
    TranscodeComponentReport, TranscodeReport, TranscodeTimingReport,
    TranscodeValidationClassification, TranscodeValidationMetrics,
};
mod error;
pub use self::error::JpegToHtj2kError;
use self::error::{dct53_grid_error, dct97_grid_error};
mod validation;
use self::validation::{
    component_sampling_for_jpeg, decomposition_levels_for_components,
    validate_component_block_grid, validate_transcode_options,
};
mod component_plan;
use self::component_plan::{
    integer_dct_job_for_component, transcode_component_batch, transcode_path_name,
    PrecomputedComponentBatch,
};
mod float_reference;
use self::float_reference::{
    dct_blocks_to_8x8_f64, decompose_97_from_first_level, float97_reference_coefficients,
    float_direct_97_wavelet_from_component, float_direct_wavelet_from_component,
    float_reference_coefficients, j2k_dwt97_from_wavelet, j2k_dwt_from_integer_wavelet,
    j2k_dwt_from_wavelet, rounded_wavelet97_i32, rounded_wavelet_i32, ComponentWavelet97,
};
mod integer_reference;
use self::integer_reference::{
    flatten_integer_wavelet, integer_direct_wavelet_from_component, integer_reference_coefficients,
    integer_wavelet_from_first_level, IntegerWavelet,
};
mod single_tile_encode;
use self::single_tile_encode::encode_component_batch;
mod batch;
pub use self::batch::jpeg_to_htj2k_batch;
#[cfg(test)]
use self::batch::{
    encode_float97_prepared_tiles, store_compact_preencoded_component,
    transform_float97_batch_tiles, Float97BatchTile,
};
use self::batch::{
    jpeg_tile_batch_to_htj2k_with_scratch, record_accelerator_attempt, record_accelerator_dispatch,
    record_batch_attempt, record_cpu_fallback, record_encode_dispatch_delta,
};

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

    let (codestream, encode_us) = encode_component_batch(
        jpeg.width,
        jpeg.height,
        component_batch.precomputed_components,
        options,
        encode_accelerator,
        &mut timings,
    )?;

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

#[cfg(test)]
mod tests;
