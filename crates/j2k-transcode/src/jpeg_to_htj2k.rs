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
    encode_precomputed_htj2k_53_with_accelerator_and_max_host_bytes,
    encode_precomputed_htj2k_97_batch_owned_with_accelerator_and_max_host_bytes,
    encode_precomputed_htj2k_97_with_accelerator_and_max_host_bytes,
    encode_preencoded_htj2k_97_compact_owned_with_accelerator_and_max_host_bytes,
    encode_preencoded_htj2k_97_owned_with_accelerator_and_max_host_bytes,
    encode_prequantized_htj2k_97_with_accelerator_and_max_host_bytes,
};
use rayon::prelude::*;

use crate::allocation::try_vec_with_capacity;
use crate::dct53_2d::{
    dct8x8_blocks_then_dwt53_float, dct8x8_blocks_to_dwt53_float_linear_with_scratch,
    linearized_53_2d_from_plane, Dct53GridScratch,
};
use crate::dct97_2d::{
    dct8x8_blocks_then_dwt97_float, dct8x8_blocks_then_dwt97_float_with_scratch,
    linearized_97_2d_from_plane_with_scratch, Dct97GridScratch,
};
use crate::metrics::{error_metrics_i32_with_live_budget, ErrorMetrics, MetricsError};
use crate::reversible53::{
    reversible_lift_53_high_at_fallible, reversible_lift_53_i32, reversible_lift_53_low_at_fallible,
};
use crate::{
    CpuOnlyDctToWaveletStageAccelerator, DctGridI16ToHtj2k97CodeBlockBatch,
    DctGridI16ToHtj2k97CodeBlockJob, DctGridToDwt53Job, DctGridToDwt97Job,
    DctGridToHtj2k97CodeBlockJob, DctGridToReversibleDwt53Job, DctToWaveletStageAccelerator,
    DctTransformError, Dwt53TwoDimensional, Dwt97BatchStageTimings, Dwt97TwoDimensional,
    Htj2k97CodeBlockOptions, ReversibleDwt53FirstLevel, TranscodeStageError,
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
use self::error::{dct53_transform_error, dct97_transform_error, map_encode_error};
pub use self::error::{Htj2kEncodeError, Htj2kEncodeErrorKind, JpegToHtj2kError};
mod validation;
use self::validation::{
    component_sampling_for_jpeg, decomposition_levels_for_components,
    validate_component_block_grid, validate_transcode_options,
};
mod workspace;
use self::workspace::validate_jpeg_transcode_workspace;
mod scratch;
use self::scratch::JpegToHtj2kScratch;
mod output;
pub use self::output::{EncodedTranscode, EncodedTranscodeBatch, JpegTileBatchInput};
mod live_budget;
use self::live_budget::{
    encoded_transcode_retained_bytes, precomputed_batch_retained_bytes,
    validation_metrics_retained_bytes, HostLiveBudget,
};
mod component_plan;
use self::component_plan::{
    integer_dct_job_for_component, transcode_component_batch, transcode_path_name,
    ComponentBatchRequest, ComponentTranscodeBatch, PrecomputedComponentBatch,
};
mod component_groups;
use self::component_groups::same_geometry_component_groups;
mod float_reference;
use self::float_reference::{
    dct_blocks_to_8x8_f64, decompose_97_from_first_level, float97_reference_coefficients,
    float_direct_97_wavelet_from_component, float_direct_wavelet_from_component,
    float_reference_coefficients, ComponentWavelet, ComponentWavelet97,
};
mod float_output;
use self::float_output::{
    j2k_dwt97_from_wavelet, j2k_dwt_from_integer_wavelet, j2k_dwt_from_wavelet,
    rounded_wavelet97_i32, rounded_wavelet_i32,
};
mod integer_reference;
use self::integer_reference::{
    flatten_integer_wavelet, integer_direct_wavelet_from_component, integer_reference_coefficients,
    integer_wavelet_from_first_level, IntegerWavelet,
};
mod integer_storage;
mod single;
use self::single::jpeg_to_htj2k_with_scratch;
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
            0,
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

#[cfg(test)]
mod tests;
