// SPDX-License-Identifier: Apache-2.0

//! Experimental JPEG DCT to HTJ2K codestream transcode entry point.

use core::fmt;
use std::time::Instant;

use rayon::prelude::*;
use signinum_j2k_native::{
    encode_precomputed_htj2k_53_with_accelerator, encode_precomputed_htj2k_97_with_accelerator,
    encode_preencoded_htj2k_97_with_accelerator, encode_prequantized_htj2k_97_with_accelerator,
    CpuOnlyJ2kEncodeStageAccelerator, EncodeOptions, J2kEncodeDispatchReport,
    J2kEncodeStageAccelerator, J2kForwardDwt53Level, J2kForwardDwt53Output, J2kForwardDwt97Level,
    J2kForwardDwt97Output, PrecomputedHtj2k53Component, PrecomputedHtj2k53Image,
    PrecomputedHtj2k97Component, PrecomputedHtj2k97Image, PreencodedHtj2k97Component,
    PreencodedHtj2k97Image, PrequantizedHtj2k97Component, PrequantizedHtj2k97Image,
};
use signinum_jpeg::transcode::{
    extract_dct_blocks, idct_islow_block, DctExtractOptions, JpegDctComponent, JpegDctImage,
};

use crate::accelerator::{
    CpuOnlyDctToWaveletStageAccelerator, DctGridToDwt53Job, DctGridToDwt97Job,
    DctGridToHtj2k97CodeBlockJob, DctGridToReversibleDwt53Job, DctToWaveletStageAccelerator,
    Dwt97BatchStageTimings, Htj2k97CodeBlockOptions, ReversibleDwt53FirstLevel,
};
use crate::dct53_2d::{
    dct8x8_blocks_then_dwt53_float, dct8x8_blocks_to_dwt53_float_linear_with_scratch,
    linearized_53_2d_from_plane, Dct53GridError, Dct53GridScratch, Dwt53TwoDimensional,
};
use crate::dct97_2d::{
    dct8x8_blocks_then_dwt97_float, dct8x8_blocks_then_dwt97_float_with_scratch,
    linearized_97_2d_from_plane_with_scratch, Dct97GridError, Dct97GridScratch,
    Dwt97TwoDimensional,
};
use crate::metrics::{error_metrics_i32, ErrorMetrics, MetricsLengthError};

/// Default irreversible quantization multiplier for JPEG direct 9/7 HTJ2K.
///
/// Empirically rate-match the explicit lossy comparison profile near the
/// nvJPEG2000 default output size on the bundled WSI tiles. Lower values
/// produce larger/higher-quality codestreams; `1.0` matches the native encoder
/// default but overshoots the NVIDIA baseline size for this transcode path.
pub const JPEG_TO_HTJ2K_LOSSY_97_QUANTIZATION_SCALE: f32 = 1.9;

/// Options for the experimental JPEG-to-HTJ2K path.
#[derive(Debug, Clone)]
pub struct JpegToHtj2kOptions {
    /// Native HTJ2K encode options used after wavelet bands are produced.
    pub encode_options: EncodeOptions,
    /// Coefficient production path used for HTJ2K precomputed bands.
    pub coefficient_path: JpegToHtj2kCoefficientPath,
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
        Self::lossless_53()
    }
}

impl JpegToHtj2kOptions {
    /// Options for the default reversible 5/3 HTJ2K coefficient path.
    #[must_use]
    pub fn lossless_53() -> Self {
        Self {
            encode_options: transcode_encode_options(true),
            coefficient_path: JpegToHtj2kCoefficientPath::IntegerDirect53,
            validate_against_float_reference: false,
            validate_against_integer_reference: false,
        }
    }

    /// Options for the irreversible 9/7 HTJ2K float-linear coefficient path.
    #[must_use]
    pub fn lossy_97() -> Self {
        let mut encode_options = transcode_encode_options(false);
        encode_options.irreversible_quantization_scale = JPEG_TO_HTJ2K_LOSSY_97_QUANTIZATION_SCALE;
        Self {
            encode_options,
            coefficient_path: JpegToHtj2kCoefficientPath::FloatDirectLinear97,
            validate_against_float_reference: false,
            validate_against_integer_reference: false,
        }
    }
}

fn transcode_encode_options(reversible: bool) -> EncodeOptions {
    EncodeOptions {
        num_decomposition_levels: 1,
        reversible,
        use_ht_block_coding: true,
        use_mct: false,
        validate_high_throughput_codestream: false,
        ..EncodeOptions::default()
    }
}

/// Experimental production path used to generate HTJ2K wavelet coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegToHtj2kCoefficientPath {
    /// Exact reversible 5/3 coefficients relative to `signinum-jpeg` scalar
    /// ISLOW block decode semantics. The first 5/3 level is computed from DCT
    /// blocks without materializing a full spatial image plane; later levels
    /// recurse conventionally over the LL coefficient band.
    IntegerDirect53,
    /// Floating-point linear composition of IDCT and 5/3 analysis. This is the
    /// linear math oracle path and remains useful for validating the direct
    /// matrix composition, but it is not the integer reversible production
    /// default.
    FloatDirectLinear53,
    /// Floating-point linear composition of IDCT and irreversible 9/7
    /// analysis. This is a lossy experimental path and must be paired with an
    /// irreversible HTJ2K encode.
    FloatDirectLinear97,
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

    /// Current capacity of the reusable DCT block conversion scratch.
    ///
    /// This is exposed for benchmark and validation harnesses while the API is
    /// experimental.
    #[must_use]
    pub fn dct_block_scratch_capacity(&self) -> usize {
        self.scratch.dct_blocks_f64.capacity()
    }

    /// Current capacity of the reusable integer block-local IDCT sample cache.
    ///
    /// This cache stores level-shifted 8x8 block samples for the integer-direct
    /// path. It is block-local scratch, not a full spatial image plane.
    #[must_use]
    pub fn integer_idct_block_scratch_capacity(&self) -> usize {
        self.scratch.integer_idct_blocks.capacity()
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

/// Aggregate report for multi-tile transcode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchTranscodeReport {
    /// Number of input tiles.
    pub tile_count: usize,
    /// Number of successfully encoded output tiles.
    pub successful_tiles: usize,
    /// Number of tile-local failures.
    pub failed_tiles: usize,
    /// Number of transformed components across successful extracted tiles.
    pub transformed_components: usize,
    /// Number of same-geometry reversible 5/3 batches submitted.
    pub reversible_dwt53_batches: usize,
    /// Number of reversible 5/3 component jobs in submitted batches.
    pub reversible_dwt53_batch_jobs: usize,
    /// Batch extraction time in microseconds.
    pub extract_us: u128,
    /// Batch DCT-to-wavelet time in microseconds.
    pub transform_us: u128,
    /// Batch HTJ2K encode time in microseconds.
    pub encode_us: u128,
    /// Detailed stage timings for the batch. Batch-accelerated 5/3 transform
    /// timings stay here instead of being copied into every tile report.
    pub timings: TranscodeTimingReport,
    /// Coefficient path used by the batch.
    pub coefficient_path: JpegToHtj2kCoefficientPath,
}

/// Detailed timing and dispatch counters for JPEG-to-HTJ2K transcode.
///
/// Durations are wall-clock microseconds measured around the current Rust API
/// boundaries. Accelerator time includes backend submission and wait overhead
/// visible to this crate; backend-specific hardware counters are not exposed
/// here.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TranscodeTimingReport {
    /// Raw compressed-tile probe/read time before JPEG DCT extraction.
    pub source_raw_probe_us: u128,
    /// Source region decode time for strip/retile workflows.
    pub read_region_decode_us: u128,
    /// Region compose/pad time for generated regular tiles.
    pub compose_pad_us: u128,
    /// JPEG encode time when the workflow generates regular JPEG tiles.
    pub generated_jpeg_encode_us: u128,
    /// JPEG DCT extraction time in microseconds.
    pub jpeg_dct_extract_us: u128,
    /// Time spent repacking integer DCT coefficients into float block grids.
    pub jpeg_dct_repack_us: u128,
    /// Total wall time spent producing DWT bands from JPEG DCT coefficients.
    pub dct_to_wavelet_total_us: u128,
    /// Wall time spent inside accelerator hook calls.
    pub dct_to_wavelet_accelerator_us: u128,
    /// Wall time spent in scalar CPU fallback transforms.
    pub dct_to_wavelet_cpu_fallback_us: u128,
    /// Time spent decomposing first-level DWT output into requested levels.
    pub dwt_decompose_us: u128,
    /// Backend 9/7 batch host pack/upload time in microseconds.
    pub dwt97_batch_pack_upload_us: u128,
    /// Backend 9/7 batch IDCT plus horizontal row-lift time in microseconds.
    pub dwt97_batch_idct_row_lift_us: u128,
    /// Backend 9/7 batch vertical column-lift time in microseconds.
    pub dwt97_batch_column_lift_us: u128,
    /// Backend 9/7 batch quantize/code-block layout time in microseconds.
    pub dwt97_batch_quantize_codeblock_us: u128,
    /// Backend 9/7 resident HT code-block encode time in microseconds.
    pub dwt97_batch_ht_encode_us: u128,
    /// Backend 9/7 resident HT code-block encode dispatches.
    pub dwt97_batch_ht_codeblock_dispatches: usize,
    /// Backend 9/7 batch output readback/unpack time in microseconds.
    pub dwt97_batch_readback_us: u128,
    /// HTJ2K encode time in microseconds.
    pub htj2k_encode_us: u128,
    /// Encode-stage accelerator dispatches during HTJ2K encode.
    pub htj2k_encode_accelerator_dispatches: usize,
    /// HT cleanup code-block accelerator dispatches during HTJ2K encode.
    pub htj2k_encode_ht_code_block_dispatches: usize,
    /// Packetization accelerator dispatches during HTJ2K encode.
    pub htj2k_encode_packetization_dispatches: usize,
    /// Time spent writing compressed frames to a DICOM `PixelData` spool.
    pub dicom_spool_write_us: u128,
    /// Time spent writing final DICOM instances.
    pub dicom_final_write_us: u128,
    /// Number of source tiles represented by this timing report.
    pub tile_count: usize,
    /// Number of components transformed into wavelet bands.
    pub component_count: usize,
    /// Number of same-geometry transform batches offered to the accelerator.
    pub batch_count: usize,
    /// Number of component jobs in same-geometry transform batches.
    pub batch_jobs: usize,
    /// Number of accelerator hook calls.
    pub accelerator_attempts: usize,
    /// Number of component jobs offered through accelerator hook calls.
    pub accelerator_jobs: usize,
    /// Number of accelerator hook calls that returned an accelerated result.
    pub accelerator_dispatches: usize,
    /// Number of component jobs completed by accelerated results.
    pub accelerator_dispatched_jobs: usize,
    /// Number of component jobs completed by scalar CPU fallback transforms.
    pub cpu_fallback_jobs: usize,
}

impl TranscodeTimingReport {
    fn add_assign(&mut self, other: Self) {
        self.source_raw_probe_us = self
            .source_raw_probe_us
            .saturating_add(other.source_raw_probe_us);
        self.read_region_decode_us = self
            .read_region_decode_us
            .saturating_add(other.read_region_decode_us);
        self.compose_pad_us = self.compose_pad_us.saturating_add(other.compose_pad_us);
        self.generated_jpeg_encode_us = self
            .generated_jpeg_encode_us
            .saturating_add(other.generated_jpeg_encode_us);
        self.jpeg_dct_extract_us = self
            .jpeg_dct_extract_us
            .saturating_add(other.jpeg_dct_extract_us);
        self.jpeg_dct_repack_us = self
            .jpeg_dct_repack_us
            .saturating_add(other.jpeg_dct_repack_us);
        self.dct_to_wavelet_total_us = self
            .dct_to_wavelet_total_us
            .saturating_add(other.dct_to_wavelet_total_us);
        self.dct_to_wavelet_accelerator_us = self
            .dct_to_wavelet_accelerator_us
            .saturating_add(other.dct_to_wavelet_accelerator_us);
        self.dct_to_wavelet_cpu_fallback_us = self
            .dct_to_wavelet_cpu_fallback_us
            .saturating_add(other.dct_to_wavelet_cpu_fallback_us);
        self.dwt_decompose_us = self.dwt_decompose_us.saturating_add(other.dwt_decompose_us);
        self.dwt97_batch_pack_upload_us = self
            .dwt97_batch_pack_upload_us
            .saturating_add(other.dwt97_batch_pack_upload_us);
        self.dwt97_batch_idct_row_lift_us = self
            .dwt97_batch_idct_row_lift_us
            .saturating_add(other.dwt97_batch_idct_row_lift_us);
        self.dwt97_batch_column_lift_us = self
            .dwt97_batch_column_lift_us
            .saturating_add(other.dwt97_batch_column_lift_us);
        self.dwt97_batch_quantize_codeblock_us = self
            .dwt97_batch_quantize_codeblock_us
            .saturating_add(other.dwt97_batch_quantize_codeblock_us);
        self.dwt97_batch_ht_encode_us = self
            .dwt97_batch_ht_encode_us
            .saturating_add(other.dwt97_batch_ht_encode_us);
        self.dwt97_batch_ht_codeblock_dispatches = self
            .dwt97_batch_ht_codeblock_dispatches
            .saturating_add(other.dwt97_batch_ht_codeblock_dispatches);
        self.dwt97_batch_readback_us = self
            .dwt97_batch_readback_us
            .saturating_add(other.dwt97_batch_readback_us);
        self.htj2k_encode_us = self.htj2k_encode_us.saturating_add(other.htj2k_encode_us);
        self.htj2k_encode_accelerator_dispatches = self
            .htj2k_encode_accelerator_dispatches
            .saturating_add(other.htj2k_encode_accelerator_dispatches);
        self.htj2k_encode_ht_code_block_dispatches = self
            .htj2k_encode_ht_code_block_dispatches
            .saturating_add(other.htj2k_encode_ht_code_block_dispatches);
        self.htj2k_encode_packetization_dispatches = self
            .htj2k_encode_packetization_dispatches
            .saturating_add(other.htj2k_encode_packetization_dispatches);
        self.dicom_spool_write_us = self
            .dicom_spool_write_us
            .saturating_add(other.dicom_spool_write_us);
        self.dicom_final_write_us = self
            .dicom_final_write_us
            .saturating_add(other.dicom_final_write_us);
        self.tile_count = self.tile_count.saturating_add(other.tile_count);
        self.component_count = self.component_count.saturating_add(other.component_count);
        self.batch_count = self.batch_count.saturating_add(other.batch_count);
        self.batch_jobs = self.batch_jobs.saturating_add(other.batch_jobs);
        self.accelerator_attempts = self
            .accelerator_attempts
            .saturating_add(other.accelerator_attempts);
        self.accelerator_jobs = self.accelerator_jobs.saturating_add(other.accelerator_jobs);
        self.accelerator_dispatches = self
            .accelerator_dispatches
            .saturating_add(other.accelerator_dispatches);
        self.accelerator_dispatched_jobs = self
            .accelerator_dispatched_jobs
            .saturating_add(other.accelerator_dispatched_jobs);
        self.cpu_fallback_jobs = self
            .cpu_fallback_jobs
            .saturating_add(other.cpu_fallback_jobs);
    }
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

/// Classification for optional coefficient-validation metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscodeValidationClassification {
    /// All compared coefficients match the selected oracle exactly.
    Exact,
    /// Coefficients satisfy the experimental one-LSB-bounded threshold:
    /// maximum absolute error is at most one LSB and at least 99.9% of
    /// coefficients match exactly.
    OneLsbBounded,
    /// Coefficients do not satisfy the exact or one-LSB-bounded thresholds.
    OutsideThreshold,
}

impl TranscodeValidationClassification {
    /// Classify validation metrics using the experimental acceptance
    /// thresholds documented for this coefficient-domain path.
    #[must_use]
    pub fn classify_metrics(metrics: &TranscodeValidationMetrics) -> Self {
        if metrics.exact_matches == metrics.total && metrics.max_abs_error == 0 {
            Self::Exact
        } else if metrics.is_one_lsb_bounded(0.999) {
            Self::OneLsbBounded
        } else {
            Self::OutsideThreshold
        }
    }
}

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
    /// Threshold classification for `float_reference_metrics`.
    pub float_reference_classification: Option<TranscodeValidationClassification>,
    /// Rounded direct coefficients compared with signinum-jpeg scalar
    /// ISLOW-IDCT-then-reversible-5/3 coefficients.
    pub integer_reference_metrics: Option<TranscodeValidationMetrics>,
    /// Threshold classification for `integer_reference_metrics`.
    pub integer_reference_classification: Option<TranscodeValidationClassification>,
    /// Number of DWT decomposition levels encoded.
    pub decomposition_levels: u8,
    /// Coefficient path used to generate the HTJ2K bands.
    pub coefficient_path: JpegToHtj2kCoefficientPath,
    /// Name of the experimental path used.
    pub path: &'static str,
    /// Wall-clock extraction time in microseconds.
    pub extract_us: u128,
    /// Wall-clock DCT-to-wavelet time in microseconds.
    pub transform_us: u128,
    /// Wall-clock HTJ2K encode time in microseconds.
    pub encode_us: u128,
    /// Detailed stage timings and accelerator/fallback counters.
    pub timings: TranscodeTimingReport,
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
    /// DCT block grid metadata did not cover the component dimensions for the
    /// 9/7 path.
    Grid97(Dct97GridError),
    /// Optional transform acceleration failed.
    Accelerator(&'static str),
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
            Self::Grid97(err) => write!(f, "DCT grid transform failed: {err}"),
            Self::Accelerator(reason) => write!(f, "transform accelerator failed: {reason}"),
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
            Self::Grid97(err) => Some(err),
            Self::Metrics(err) => Some(err),
            Self::Unsupported(_) | Self::Accelerator(_) | Self::Validation(_) | Self::Encode(_) => {
                None
            }
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

impl From<Dct97GridError> for JpegToHtj2kError {
    fn from(value: Dct97GridError) -> Self {
        Self::Grid97(value)
    }
}

impl From<MetricsLengthError> for JpegToHtj2kError {
    fn from(value: MetricsLengthError) -> Self {
        Self::Metrics(value)
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

/// Transcode many JPEG tiles into HTJ2K codestreams.
pub fn jpeg_to_htj2k_batch(
    tiles: &[JpegTileBatchInput<'_>],
    options: &JpegToHtj2kOptions,
) -> Result<EncodedTranscodeBatch, JpegToHtj2kError> {
    JpegToHtj2kTranscoder::default().transcode_batch(tiles, options)
}

fn jpeg_tile_batch_to_htj2k_with_scratch<
    A: DctToWaveletStageAccelerator,
    E: J2kEncodeStageAccelerator,
>(
    tiles: &[JpegTileBatchInput<'_>],
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut A,
    encode_accelerator: &mut E,
) -> Result<EncodedTranscodeBatch, JpegToHtj2kError> {
    validate_transcode_options(options)?;
    match options.coefficient_path {
        JpegToHtj2kCoefficientPath::IntegerDirect53 => {}
        JpegToHtj2kCoefficientPath::FloatDirectLinear97
            if accelerator.supports_dwt97_batch()
                || accelerator.supports_htj2k97_codeblock_batch() =>
        {
            return jpeg_float97_tile_batch_to_htj2k_with_scratch(
                tiles,
                options,
                scratch,
                accelerator,
                encode_accelerator,
            );
        }
        JpegToHtj2kCoefficientPath::FloatDirectLinear53
        | JpegToHtj2kCoefficientPath::FloatDirectLinear97 => {
            return Ok(transcode_tile_batch_individually(
                tiles,
                options,
                scratch,
                accelerator,
                encode_accelerator,
            ));
        }
    }

    let extract_start = Instant::now();
    let prepared_results = tiles
        .par_iter()
        .enumerate()
        .map(|(tile_index, tile)| {
            (
                tile_index,
                prepare_integer_batch_tile(tile_index, tile.bytes, options),
            )
        })
        .collect::<Vec<_>>();
    let extract_us = extract_start.elapsed().as_micros();
    let mut tile_results: Vec<Option<Result<EncodedTranscode, JpegToHtj2kError>>> =
        (0..tiles.len()).map(|_| None).collect();
    let mut prepared_tiles = Vec::new();
    for (tile_index, result) in prepared_results {
        match result {
            Ok(prepared) => prepared_tiles.push(prepared),
            Err(error) => tile_results[tile_index] = Some(Err(error)),
        }
    }

    let transform_start = Instant::now();
    let mut timings = TranscodeTimingReport::default();
    let (reversible_dwt53_batches, reversible_dwt53_batch_jobs) = transform_integer_batch_tiles(
        &mut prepared_tiles,
        options,
        scratch,
        accelerator,
        &mut timings,
    )?;
    let transform_us = transform_start.elapsed().as_micros();
    timings.jpeg_dct_extract_us = extract_us;
    timings.dct_to_wavelet_total_us = transform_us;
    timings.tile_count = prepared_tiles.len();

    let encode_start = Instant::now();
    let encoded_tiles = encode_integer_prepared_tiles(prepared_tiles, options, encode_accelerator);
    for (tile_index, encoded) in encoded_tiles {
        add_encode_timing_counters_from_result(&mut timings, &encoded);
        tile_results[tile_index] = Some(encoded);
    }
    let encode_us = encode_start.elapsed().as_micros();
    timings.htj2k_encode_us = encode_us;

    let output_tiles = tile_results
        .into_iter()
        .map(|tile| {
            tile.unwrap_or(Err(JpegToHtj2kError::Validation(
                "batch transcode did not produce a tile result",
            )))
        })
        .collect::<Vec<_>>();
    Ok(batch_output(
        output_tiles,
        BatchTranscodeReport {
            tile_count: tiles.len(),
            successful_tiles: 0,
            failed_tiles: 0,
            transformed_components: reversible_dwt53_batch_jobs,
            reversible_dwt53_batches,
            reversible_dwt53_batch_jobs,
            extract_us,
            transform_us,
            encode_us,
            timings,
            coefficient_path: options.coefficient_path,
        },
    ))
}

fn jpeg_float97_tile_batch_to_htj2k_with_scratch<
    A: DctToWaveletStageAccelerator,
    E: J2kEncodeStageAccelerator,
>(
    tiles: &[JpegTileBatchInput<'_>],
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut A,
    encode_accelerator: &mut E,
) -> Result<EncodedTranscodeBatch, JpegToHtj2kError> {
    let extract_start = Instant::now();
    let prepared_results = tiles
        .par_iter()
        .enumerate()
        .map(|(tile_index, tile)| {
            (
                tile_index,
                prepare_float97_batch_tile(tile_index, tile.bytes, options),
            )
        })
        .collect::<Vec<_>>();
    let extract_us = extract_start.elapsed().as_micros();
    let mut tile_results: Vec<Option<Result<EncodedTranscode, JpegToHtj2kError>>> =
        (0..tiles.len()).map(|_| None).collect();
    let mut prepared_tiles = Vec::new();
    for (tile_index, result) in prepared_results {
        match result {
            Ok(prepared) => prepared_tiles.push(prepared),
            Err(error) => tile_results[tile_index] = Some(Err(error)),
        }
    }

    let transform_start = Instant::now();
    let mut timings = TranscodeTimingReport::default();
    let (_dwt97_batches, dwt97_batch_jobs) = transform_float97_batch_tiles(
        &mut prepared_tiles,
        options,
        scratch,
        accelerator,
        &mut timings,
    )?;
    let transform_us = transform_start.elapsed().as_micros();
    timings.jpeg_dct_extract_us = extract_us;
    timings.dct_to_wavelet_total_us = transform_us;
    timings.tile_count = prepared_tiles.len();

    let encode_start = Instant::now();
    let encoded_tiles = encode_float97_prepared_tiles(prepared_tiles, options, encode_accelerator);
    for (tile_index, encoded) in encoded_tiles {
        add_encode_timing_counters_from_result(&mut timings, &encoded);
        tile_results[tile_index] = Some(encoded);
    }
    let encode_us = encode_start.elapsed().as_micros();
    timings.htj2k_encode_us = encode_us;

    let output_tiles = tile_results
        .into_iter()
        .map(|tile| {
            tile.unwrap_or(Err(JpegToHtj2kError::Validation(
                "9/7 batch transcode did not produce a tile result",
            )))
        })
        .collect::<Vec<_>>();
    Ok(batch_output(
        output_tiles,
        BatchTranscodeReport {
            tile_count: tiles.len(),
            successful_tiles: 0,
            failed_tiles: 0,
            transformed_components: dwt97_batch_jobs,
            reversible_dwt53_batches: 0,
            reversible_dwt53_batch_jobs: 0,
            extract_us,
            transform_us,
            encode_us,
            timings,
            coefficient_path: options.coefficient_path,
        },
    ))
}

fn transcode_tile_batch_individually<
    A: DctToWaveletStageAccelerator,
    E: J2kEncodeStageAccelerator,
>(
    tiles: &[JpegTileBatchInput<'_>],
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut A,
    encode_accelerator: &mut E,
) -> EncodedTranscodeBatch {
    let start = Instant::now();
    let output_tiles = tiles
        .iter()
        .map(|tile| {
            jpeg_to_htj2k_with_scratch(
                tile.bytes,
                options,
                scratch,
                accelerator,
                encode_accelerator,
            )
        })
        .collect::<Vec<_>>();
    let mut timings = aggregate_tile_timings(&output_tiles);
    timings.tile_count = output_tiles.iter().filter(|tile| tile.is_ok()).count();
    let elapsed_us = start.elapsed().as_micros();
    if timings.dct_to_wavelet_total_us == 0 {
        timings.dct_to_wavelet_total_us = elapsed_us
            .saturating_sub(timings.jpeg_dct_extract_us)
            .saturating_sub(timings.htj2k_encode_us);
    }
    batch_output(
        output_tiles,
        BatchTranscodeReport {
            tile_count: tiles.len(),
            successful_tiles: 0,
            failed_tiles: 0,
            transformed_components: timings.component_count,
            reversible_dwt53_batches: 0,
            reversible_dwt53_batch_jobs: 0,
            extract_us: timings.jpeg_dct_extract_us,
            transform_us: timings.dct_to_wavelet_total_us,
            encode_us: timings.htj2k_encode_us,
            timings,
            coefficient_path: options.coefficient_path,
        },
    )
}

fn aggregate_tile_timings(
    tiles: &[Result<EncodedTranscode, JpegToHtj2kError>],
) -> TranscodeTimingReport {
    let mut timings = TranscodeTimingReport::default();
    for tile in tiles.iter().filter_map(|tile| tile.as_ref().ok()) {
        timings.add_assign(tile.report.timings);
    }
    timings
}

fn batch_output(
    tiles: Vec<Result<EncodedTranscode, JpegToHtj2kError>>,
    mut report: BatchTranscodeReport,
) -> EncodedTranscodeBatch {
    report.successful_tiles = tiles.iter().filter(|tile| tile.is_ok()).count();
    report.failed_tiles = tiles.len().saturating_sub(report.successful_tiles);
    EncodedTranscodeBatch { tiles, report }
}

struct IntegerBatchTile {
    tile_index: usize,
    jpeg: JpegDctImage,
    component_sampling: Vec<(u8, u8)>,
    decomposition_levels: u8,
    all_unit_sampled: bool,
    component_reports: Vec<TranscodeComponentReport>,
    precomputed_components: Vec<Option<PrecomputedHtj2k53Component>>,
    float_validation_actual: Vec<i32>,
    float_validation_expected: Vec<i32>,
    integer_validation_actual: Vec<i32>,
    integer_validation_expected: Vec<i32>,
    timings: TranscodeTimingReport,
}

struct Float97BatchTile {
    tile_index: usize,
    jpeg: JpegDctImage,
    component_sampling: Vec<(u8, u8)>,
    decomposition_levels: u8,
    all_unit_sampled: bool,
    component_reports: Vec<TranscodeComponentReport>,
    precomputed_components: Vec<Option<PrecomputedHtj2k97Component>>,
    preencoded_components: Vec<Option<PreencodedHtj2k97Component>>,
    prequantized_components: Vec<Option<PrequantizedHtj2k97Component>>,
    float_validation_actual: Vec<i32>,
    float_validation_expected: Vec<i32>,
    timings: TranscodeTimingReport,
}

#[derive(Clone, Copy)]
struct BatchComponentRef {
    tile_index: usize,
    component_index: usize,
}

fn prepare_integer_batch_tile(
    tile_index: usize,
    bytes: &[u8],
    options: &JpegToHtj2kOptions,
) -> Result<IntegerBatchTile, JpegToHtj2kError> {
    let extract_start = Instant::now();
    let jpeg = extract_dct_blocks(bytes, DctExtractOptions::default())?;
    let timings = TranscodeTimingReport {
        jpeg_dct_extract_us: extract_start.elapsed().as_micros(),
        tile_count: 1,
        ..TranscodeTimingReport::default()
    };
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
        .collect::<Vec<_>>();
    let precomputed_components = (0..jpeg.components.len()).map(|_| None).collect();

    Ok(IntegerBatchTile {
        tile_index,
        jpeg,
        component_sampling,
        decomposition_levels,
        all_unit_sampled,
        component_reports,
        precomputed_components,
        float_validation_actual: Vec::new(),
        float_validation_expected: Vec::new(),
        integer_validation_actual: Vec::new(),
        integer_validation_expected: Vec::new(),
        timings,
    })
}

fn prepare_float97_batch_tile(
    tile_index: usize,
    bytes: &[u8],
    options: &JpegToHtj2kOptions,
) -> Result<Float97BatchTile, JpegToHtj2kError> {
    let extract_start = Instant::now();
    let jpeg = extract_dct_blocks(bytes, DctExtractOptions::default())?;
    let timings = TranscodeTimingReport {
        jpeg_dct_extract_us: extract_start.elapsed().as_micros(),
        tile_count: 1,
        ..TranscodeTimingReport::default()
    };
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
        .collect::<Vec<_>>();
    let precomputed_components = (0..jpeg.components.len()).map(|_| None).collect();
    let preencoded_components = (0..jpeg.components.len()).map(|_| None).collect();
    let prequantized_components = (0..jpeg.components.len()).map(|_| None).collect();

    Ok(Float97BatchTile {
        tile_index,
        jpeg,
        component_sampling,
        decomposition_levels,
        all_unit_sampled,
        component_reports,
        precomputed_components,
        preencoded_components,
        prequantized_components,
        float_validation_actual: Vec::new(),
        float_validation_expected: Vec::new(),
        timings,
    })
}

fn transform_integer_batch_tiles<A: DctToWaveletStageAccelerator>(
    tiles: &mut [IntegerBatchTile],
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut A,
    timings: &mut TranscodeTimingReport,
) -> Result<(usize, usize), JpegToHtj2kError> {
    let groups = batch_component_groups(tiles);
    let mut batch_count = 0usize;
    let mut job_count = 0usize;

    for group in groups {
        batch_count = batch_count.saturating_add(1);
        job_count = job_count.saturating_add(group.len());
        let wavelets =
            integer_wavelets_for_batch_group(&group, tiles, scratch, accelerator, timings)?;
        for (component_ref, wavelet) in group.into_iter().zip(wavelets) {
            store_integer_batch_wavelet(component_ref, &wavelet, tiles, options, scratch)?;
        }
    }

    Ok((batch_count, job_count))
}

fn transform_float97_batch_tiles<A: DctToWaveletStageAccelerator>(
    tiles: &mut [Float97BatchTile],
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut A,
    timings: &mut TranscodeTimingReport,
) -> Result<(usize, usize), JpegToHtj2kError> {
    let groups = float97_batch_component_groups(tiles);
    let mut batch_count = 0usize;
    let mut job_count = 0usize;

    for group in groups {
        batch_count = batch_count.saturating_add(1);
        job_count = job_count.saturating_add(group.len());
        if try_store_prequantized_float97_batch_group(&group, tiles, options, accelerator, timings)?
        {
            continue;
        }
        let wavelets =
            float97_wavelets_for_batch_group(&group, tiles, scratch, accelerator, timings)?;
        for (component_ref, wavelet) in group.into_iter().zip(wavelets) {
            store_float97_batch_wavelet(component_ref, &wavelet, tiles, options, scratch)?;
        }
    }

    Ok((batch_count, job_count))
}

fn batch_component_groups(tiles: &[IntegerBatchTile]) -> Vec<Vec<BatchComponentRef>> {
    let mut groups: Vec<Vec<BatchComponentRef>> = Vec::new();

    for (tile_index, tile) in tiles.iter().enumerate() {
        for (component_index, component) in tile.jpeg.components.iter().enumerate() {
            let component_ref = BatchComponentRef {
                tile_index,
                component_index,
            };
            if let Some(group) = groups.iter_mut().find(|group| {
                let first = group[0];
                same_batch_component_key(
                    &tiles[first.tile_index],
                    first.component_index,
                    tile,
                    component_index,
                )
            }) {
                group.push(component_ref);
            } else {
                let _ = component;
                groups.push(vec![component_ref]);
            }
        }
    }

    groups
}

fn float97_batch_component_groups(tiles: &[Float97BatchTile]) -> Vec<Vec<BatchComponentRef>> {
    let mut groups: Vec<Vec<BatchComponentRef>> = Vec::new();

    for (tile_index, tile) in tiles.iter().enumerate() {
        for component_index in 0..tile.jpeg.components.len() {
            let component_ref = BatchComponentRef {
                tile_index,
                component_index,
            };
            if let Some(group) = groups.iter_mut().find(|group| {
                let first = group[0];
                same_float97_batch_component_key(
                    &tiles[first.tile_index],
                    first.component_index,
                    tile,
                    component_index,
                )
            }) {
                group.push(component_ref);
            } else {
                groups.push(vec![component_ref]);
            }
        }
    }

    groups
}

fn same_batch_component_key(
    left_tile: &IntegerBatchTile,
    left_component_index: usize,
    right_tile: &IntegerBatchTile,
    right_component_index: usize,
) -> bool {
    let left = &left_tile.jpeg.components[left_component_index];
    let right = &right_tile.jpeg.components[right_component_index];
    left.component_index == right.component_index
        && left.width == right.width
        && left.height == right.height
        && left.block_cols == right.block_cols
        && left.block_rows == right.block_rows
        && left_tile.component_sampling[left_component_index]
            == right_tile.component_sampling[right_component_index]
}

fn same_float97_batch_component_key(
    left_tile: &Float97BatchTile,
    left_component_index: usize,
    right_tile: &Float97BatchTile,
    right_component_index: usize,
) -> bool {
    let left = &left_tile.jpeg.components[left_component_index];
    let right = &right_tile.jpeg.components[right_component_index];
    left.width == right.width
        && left.height == right.height
        && left.block_cols == right.block_cols
        && left.block_rows == right.block_rows
        && left_tile.component_sampling[left_component_index]
            == right_tile.component_sampling[right_component_index]
}

fn integer_wavelets_for_batch_group<A: DctToWaveletStageAccelerator>(
    group: &[BatchComponentRef],
    tiles: &[IntegerBatchTile],
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut A,
    timings: &mut TranscodeTimingReport,
) -> Result<Vec<IntegerWavelet>, JpegToHtj2kError> {
    let jobs = group
        .iter()
        .map(|component_ref| {
            integer_dct_job_for_component(
                &tiles[component_ref.tile_index].jpeg.components[component_ref.component_index],
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    timings.batch_count = timings.batch_count.saturating_add(1);
    timings.batch_jobs = timings.batch_jobs.saturating_add(group.len());
    timings.accelerator_attempts = timings.accelerator_attempts.saturating_add(1);
    timings.accelerator_jobs = timings.accelerator_jobs.saturating_add(group.len());
    let accelerator_start = Instant::now();
    let accelerated = accelerator
        .dct_grid_to_reversible_dwt53_batch(&jobs)
        .map_err(JpegToHtj2kError::Accelerator)?;
    timings.dct_to_wavelet_accelerator_us = timings
        .dct_to_wavelet_accelerator_us
        .saturating_add(accelerator_start.elapsed().as_micros());

    if let Some(first_levels) = accelerated {
        if first_levels.len() != group.len() {
            return Err(JpegToHtj2kError::Validation(
                "reversible 5/3 batch accelerator returned wrong component count",
            ));
        }
        timings.component_count = timings.component_count.saturating_add(group.len());
        timings.accelerator_dispatches = timings.accelerator_dispatches.saturating_add(1);
        timings.accelerator_dispatched_jobs = timings
            .accelerator_dispatched_jobs
            .saturating_add(group.len());
        let decompose_start = Instant::now();
        let wavelets = first_levels
            .into_iter()
            .zip(group.iter().copied())
            .map(|(first_level, component_ref)| {
                integer_wavelet_from_first_level(
                    first_level,
                    tiles[component_ref.tile_index].decomposition_levels,
                )
            })
            .collect();
        timings.dwt_decompose_us = timings
            .dwt_decompose_us
            .saturating_add(decompose_start.elapsed().as_micros());
        return Ok(wavelets);
    }

    group
        .iter()
        .map(|component_ref| {
            integer_direct_wavelet_from_component(
                &tiles[component_ref.tile_index].jpeg.components[component_ref.component_index],
                tiles[component_ref.tile_index].decomposition_levels,
                scratch,
                accelerator,
                timings,
            )
        })
        .collect()
}

#[allow(clippy::too_many_lines)]
fn try_store_prequantized_float97_batch_group<A: DctToWaveletStageAccelerator>(
    group: &[BatchComponentRef],
    tiles: &mut [Float97BatchTile],
    options: &JpegToHtj2kOptions,
    accelerator: &mut A,
    timings: &mut TranscodeTimingReport,
) -> Result<bool, JpegToHtj2kError> {
    if !accelerator.supports_htj2k97_codeblock_batch()
        || options.validate_against_float_reference
        || group
            .iter()
            .any(|component_ref| tiles[component_ref.tile_index].decomposition_levels != 1)
    {
        return Ok(false);
    }

    let repack_start = Instant::now();
    let block_storage = group
        .par_iter()
        .map(|component_ref| {
            dct_blocks_to_8x8_f64(
                &tiles[component_ref.tile_index].jpeg.components[component_ref.component_index]
                    .dequantized_blocks,
            )
        })
        .collect::<Vec<_>>();
    timings.jpeg_dct_repack_us = timings
        .jpeg_dct_repack_us
        .saturating_add(repack_start.elapsed().as_micros());

    let jobs = group
        .iter()
        .zip(block_storage.iter())
        .map(|(component_ref, blocks)| {
            let tile = &tiles[component_ref.tile_index];
            let component = &tile.jpeg.components[component_ref.component_index];
            let (x_rsiz, y_rsiz) = tile.component_sampling[component_ref.component_index];
            validate_component_block_grid(component)?;
            Ok(DctGridToHtj2k97CodeBlockJob {
                blocks,
                block_cols: component.block_cols as usize,
                block_rows: component.block_rows as usize,
                width: component.width as usize,
                height: component.height as usize,
                x_rsiz,
                y_rsiz,
            })
        })
        .collect::<Result<Vec<_>, JpegToHtj2kError>>()?;

    timings.accelerator_attempts = timings.accelerator_attempts.saturating_add(1);
    timings.accelerator_jobs = timings.accelerator_jobs.saturating_add(group.len());
    let accelerator_start = Instant::now();
    let codeblock_options = htj2k97_codeblock_options(&options.encode_options);
    let preencoded_components = accelerator
        .dct_grid_to_htj2k97_preencoded_batch(&jobs, codeblock_options)
        .map_err(JpegToHtj2kError::Accelerator)?;
    if let Some(components) = preencoded_components {
        if let Some(stage_timings) = accelerator.last_dwt97_batch_stage_timings() {
            add_dwt97_batch_stage_timings(timings, stage_timings);
        }
        timings.dct_to_wavelet_accelerator_us = timings
            .dct_to_wavelet_accelerator_us
            .saturating_add(accelerator_start.elapsed().as_micros());
        if components.len() != group.len() {
            return Err(JpegToHtj2kError::Validation(
                "9/7 preencoded accelerator returned wrong component count",
            ));
        }

        timings.batch_count = timings.batch_count.saturating_add(1);
        timings.batch_jobs = timings.batch_jobs.saturating_add(group.len());
        timings.component_count = timings.component_count.saturating_add(group.len());
        timings.accelerator_dispatches = timings.accelerator_dispatches.saturating_add(1);
        timings.accelerator_dispatched_jobs = timings
            .accelerator_dispatched_jobs
            .saturating_add(group.len());
        for (component_ref, component) in group.iter().copied().zip(components) {
            tiles[component_ref.tile_index].preencoded_components[component_ref.component_index] =
                Some(component);
        }

        return Ok(true);
    }

    let accelerated_components = accelerator
        .dct_grid_to_htj2k97_codeblock_batch(&jobs, codeblock_options)
        .map_err(JpegToHtj2kError::Accelerator)?;
    if let Some(stage_timings) = accelerator.last_dwt97_batch_stage_timings() {
        add_dwt97_batch_stage_timings(timings, stage_timings);
    }
    timings.dct_to_wavelet_accelerator_us = timings
        .dct_to_wavelet_accelerator_us
        .saturating_add(accelerator_start.elapsed().as_micros());

    let Some(components) = accelerated_components else {
        return Ok(false);
    };
    if components.len() != group.len() {
        return Err(JpegToHtj2kError::Validation(
            "9/7 code-block accelerator returned wrong component count",
        ));
    }

    timings.batch_count = timings.batch_count.saturating_add(1);
    timings.batch_jobs = timings.batch_jobs.saturating_add(group.len());
    timings.component_count = timings.component_count.saturating_add(group.len());
    timings.accelerator_dispatches = timings.accelerator_dispatches.saturating_add(1);
    timings.accelerator_dispatched_jobs = timings
        .accelerator_dispatched_jobs
        .saturating_add(group.len());
    for (component_ref, component) in group.iter().copied().zip(components) {
        tiles[component_ref.tile_index].prequantized_components[component_ref.component_index] =
            Some(component);
    }

    Ok(true)
}

fn htj2k97_codeblock_options(options: &EncodeOptions) -> Htj2k97CodeBlockOptions {
    Htj2k97CodeBlockOptions {
        bit_depth: 8,
        guard_bits: options.guard_bits.max(2),
        code_block_width_exp: options.code_block_width_exp,
        code_block_height_exp: options.code_block_height_exp,
        irreversible_quantization_scale: options.irreversible_quantization_scale,
        irreversible_quantization_subband_scales: options.irreversible_quantization_subband_scales,
    }
}

fn float97_wavelets_for_batch_group<A: DctToWaveletStageAccelerator>(
    group: &[BatchComponentRef],
    tiles: &[Float97BatchTile],
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut A,
    timings: &mut TranscodeTimingReport,
) -> Result<Vec<ComponentWavelet97>, JpegToHtj2kError> {
    let repack_start = Instant::now();
    let block_storage = group
        .iter()
        .map(|component_ref| {
            dct_blocks_to_8x8_f64(
                &tiles[component_ref.tile_index].jpeg.components[component_ref.component_index]
                    .dequantized_blocks,
            )
        })
        .collect::<Vec<_>>();
    timings.jpeg_dct_repack_us = timings
        .jpeg_dct_repack_us
        .saturating_add(repack_start.elapsed().as_micros());

    let jobs = group
        .iter()
        .zip(block_storage.iter())
        .map(|(component_ref, blocks)| {
            let component =
                &tiles[component_ref.tile_index].jpeg.components[component_ref.component_index];
            validate_component_block_grid(component)?;
            Ok(DctGridToDwt97Job {
                blocks,
                block_cols: component.block_cols as usize,
                block_rows: component.block_rows as usize,
                width: component.width as usize,
                height: component.height as usize,
            })
        })
        .collect::<Result<Vec<_>, JpegToHtj2kError>>()?;

    timings.batch_count = timings.batch_count.saturating_add(1);
    timings.batch_jobs = timings.batch_jobs.saturating_add(group.len());
    timings.accelerator_attempts = timings.accelerator_attempts.saturating_add(1);
    timings.accelerator_jobs = timings.accelerator_jobs.saturating_add(group.len());
    let accelerator_start = Instant::now();
    let accelerated_first_levels = accelerator
        .dct_grid_to_dwt97_batch(&jobs)
        .map_err(JpegToHtj2kError::Accelerator)?;
    if let Some(stage_timings) = accelerator.last_dwt97_batch_stage_timings() {
        add_dwt97_batch_stage_timings(timings, stage_timings);
    }
    timings.dct_to_wavelet_accelerator_us = timings
        .dct_to_wavelet_accelerator_us
        .saturating_add(accelerator_start.elapsed().as_micros());

    if let Some(first_levels) = accelerated_first_levels {
        if first_levels.len() != group.len() {
            return Err(JpegToHtj2kError::Validation(
                "9/7 batch accelerator returned wrong component count",
            ));
        }
        timings.component_count = timings.component_count.saturating_add(group.len());
        timings.accelerator_dispatches = timings.accelerator_dispatches.saturating_add(1);
        timings.accelerator_dispatched_jobs = timings
            .accelerator_dispatched_jobs
            .saturating_add(group.len());
        let decompose_start = Instant::now();
        let mut wavelets = Vec::with_capacity(first_levels.len());
        for (first_level, component_ref) in first_levels.into_iter().zip(group.iter().copied()) {
            wavelets.push(decompose_97_from_first_level_with_scratch(
                first_level,
                usize::from(tiles[component_ref.tile_index].decomposition_levels),
                &mut scratch.dct97_grid,
            ));
        }
        timings.dwt_decompose_us = timings
            .dwt_decompose_us
            .saturating_add(decompose_start.elapsed().as_micros());
        return Ok(wavelets);
    }

    group
        .iter()
        .map(|component_ref| {
            float_direct_97_wavelet_from_component(
                &tiles[component_ref.tile_index].jpeg.components[component_ref.component_index],
                tiles[component_ref.tile_index].decomposition_levels,
                scratch,
                accelerator,
                timings,
            )
        })
        .collect()
}

fn add_dwt97_batch_stage_timings(
    timings: &mut TranscodeTimingReport,
    stage_timings: Dwt97BatchStageTimings,
) {
    timings.dwt97_batch_pack_upload_us = timings
        .dwt97_batch_pack_upload_us
        .saturating_add(stage_timings.pack_upload_us);
    timings.dwt97_batch_idct_row_lift_us = timings
        .dwt97_batch_idct_row_lift_us
        .saturating_add(stage_timings.idct_row_lift_us);
    timings.dwt97_batch_column_lift_us = timings
        .dwt97_batch_column_lift_us
        .saturating_add(stage_timings.column_lift_us);
    timings.dwt97_batch_quantize_codeblock_us = timings
        .dwt97_batch_quantize_codeblock_us
        .saturating_add(stage_timings.quantize_codeblock_us);
    timings.dwt97_batch_ht_encode_us = timings
        .dwt97_batch_ht_encode_us
        .saturating_add(stage_timings.ht_encode_us);
    timings.dwt97_batch_ht_codeblock_dispatches = timings
        .dwt97_batch_ht_codeblock_dispatches
        .saturating_add(stage_timings.ht_codeblock_dispatches);
    timings.dwt97_batch_readback_us = timings
        .dwt97_batch_readback_us
        .saturating_add(stage_timings.readback_us);
}

fn store_integer_batch_wavelet(
    component_ref: BatchComponentRef,
    wavelet: &IntegerWavelet,
    tiles: &mut [IntegerBatchTile],
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
) -> Result<(), JpegToHtj2kError> {
    let tile = &mut tiles[component_ref.tile_index];
    let component = &tile.jpeg.components[component_ref.component_index];
    let (x_rsiz, y_rsiz) = tile.component_sampling[component_ref.component_index];
    let actual_coefficients = flatten_integer_wavelet(wavelet);
    tile.precomputed_components[component_ref.component_index] =
        Some(PrecomputedHtj2k53Component {
            x_rsiz,
            y_rsiz,
            dwt: j2k_dwt_from_integer_wavelet(wavelet),
        });

    if options.validate_against_float_reference {
        tile.float_validation_actual
            .extend(actual_coefficients.clone());
        tile.float_validation_expected
            .extend(float_reference_coefficients(
                component,
                tile.decomposition_levels,
                scratch,
            )?);
    }
    if options.validate_against_integer_reference {
        tile.integer_validation_actual.extend(actual_coefficients);
        tile.integer_validation_expected
            .extend(integer_reference_coefficients(
                component,
                tile.decomposition_levels,
            )?);
    }

    Ok(())
}

fn store_float97_batch_wavelet(
    component_ref: BatchComponentRef,
    wavelet: &ComponentWavelet97,
    tiles: &mut [Float97BatchTile],
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
) -> Result<(), JpegToHtj2kError> {
    let tile = &mut tiles[component_ref.tile_index];
    let component = &tile.jpeg.components[component_ref.component_index];
    let (x_rsiz, y_rsiz) = tile.component_sampling[component_ref.component_index];
    let actual_coefficients = rounded_wavelet97_i32(wavelet)?;
    tile.precomputed_components[component_ref.component_index] =
        Some(PrecomputedHtj2k97Component {
            x_rsiz,
            y_rsiz,
            dwt: j2k_dwt97_from_wavelet(
                wavelet,
                component.width as usize,
                component.height as usize,
            ),
        });

    if options.validate_against_float_reference {
        tile.float_validation_actual
            .extend(actual_coefficients.clone());
        tile.float_validation_expected
            .extend(float97_reference_coefficients(
                component,
                tile.decomposition_levels,
                scratch,
            )?);
    }

    Ok(())
}

fn record_encode_dispatch_delta(
    timings: &mut TranscodeTimingReport,
    before: J2kEncodeDispatchReport,
    after: J2kEncodeDispatchReport,
) {
    let delta = after.saturating_delta(before);
    timings.htj2k_encode_accelerator_dispatches = timings
        .htj2k_encode_accelerator_dispatches
        .saturating_add(delta.total());
    timings.htj2k_encode_ht_code_block_dispatches = timings
        .htj2k_encode_ht_code_block_dispatches
        .saturating_add(delta.ht_code_block);
    timings.htj2k_encode_packetization_dispatches = timings
        .htj2k_encode_packetization_dispatches
        .saturating_add(delta.packetization);
}

fn add_encode_timing_counters_from_result(
    timings: &mut TranscodeTimingReport,
    tile: &Result<EncodedTranscode, JpegToHtj2kError>,
) {
    let Ok(tile) = tile else {
        return;
    };
    timings.htj2k_encode_accelerator_dispatches = timings
        .htj2k_encode_accelerator_dispatches
        .saturating_add(tile.report.timings.htj2k_encode_accelerator_dispatches);
    timings.htj2k_encode_ht_code_block_dispatches = timings
        .htj2k_encode_ht_code_block_dispatches
        .saturating_add(tile.report.timings.htj2k_encode_ht_code_block_dispatches);
    timings.htj2k_encode_packetization_dispatches = timings
        .htj2k_encode_packetization_dispatches
        .saturating_add(tile.report.timings.htj2k_encode_packetization_dispatches);
}

fn encode_integer_prepared_tiles<E: J2kEncodeStageAccelerator>(
    prepared_tiles: Vec<IntegerBatchTile>,
    options: &JpegToHtj2kOptions,
    encode_accelerator: &mut E,
) -> Vec<(usize, Result<EncodedTranscode, JpegToHtj2kError>)> {
    if encode_accelerator.prefer_parallel_cpu_tile_encode() {
        return prepared_tiles
            .into_par_iter()
            .map(|prepared| {
                let tile_index = prepared.tile_index;
                let mut cpu_accelerator = CpuOnlyJ2kEncodeStageAccelerator;
                (
                    tile_index,
                    encode_integer_batch_tile(prepared, options, &mut cpu_accelerator),
                )
            })
            .collect();
    }

    prepared_tiles
        .into_iter()
        .map(|prepared| {
            let tile_index = prepared.tile_index;
            (
                tile_index,
                encode_integer_batch_tile(prepared, options, encode_accelerator),
            )
        })
        .collect()
}

fn encode_float97_prepared_tiles<E: J2kEncodeStageAccelerator>(
    prepared_tiles: Vec<Float97BatchTile>,
    options: &JpegToHtj2kOptions,
    encode_accelerator: &mut E,
) -> Vec<(usize, Result<EncodedTranscode, JpegToHtj2kError>)> {
    if encode_accelerator.prefer_parallel_cpu_tile_encode() {
        return prepared_tiles
            .into_par_iter()
            .map(|prepared| {
                let tile_index = prepared.tile_index;
                let mut cpu_accelerator = CpuOnlyJ2kEncodeStageAccelerator;
                (
                    tile_index,
                    encode_float97_batch_tile(prepared, options, &mut cpu_accelerator),
                )
            })
            .collect();
    }

    prepared_tiles
        .into_iter()
        .map(|prepared| {
            let tile_index = prepared.tile_index;
            (
                tile_index,
                encode_float97_batch_tile(prepared, options, encode_accelerator),
            )
        })
        .collect()
}

fn encode_integer_batch_tile<E: J2kEncodeStageAccelerator>(
    tile: IntegerBatchTile,
    options: &JpegToHtj2kOptions,
    encode_accelerator: &mut E,
) -> Result<EncodedTranscode, JpegToHtj2kError> {
    let mut timings = tile.timings;
    let components = tile
        .precomputed_components
        .into_iter()
        .map(|component| {
            component.ok_or(JpegToHtj2kError::Validation(
                "integer batch transcode did not produce all components",
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let encode_start = Instant::now();
    let precomputed = PrecomputedHtj2k53Image {
        width: tile.jpeg.width,
        height: tile.jpeg.height,
        bit_depth: 8,
        signed: false,
        components,
    };
    let encode_dispatch_before = encode_accelerator.dispatch_report();
    let codestream = encode_precomputed_htj2k_53_with_accelerator(
        &precomputed,
        &options.encode_options,
        encode_accelerator,
    )
    .map_err(JpegToHtj2kError::Encode)?;
    record_encode_dispatch_delta(
        &mut timings,
        encode_dispatch_before,
        encode_accelerator.dispatch_report(),
    );
    let encode_us = encode_start.elapsed().as_micros();
    timings.htj2k_encode_us = encode_us;
    let integer_reference_metrics = if options.validate_against_integer_reference {
        Some(error_metrics_i32(
            &tile.integer_validation_actual,
            &tile.integer_validation_expected,
        )?)
    } else {
        None
    };
    let float_reference_metrics = if options.validate_against_float_reference {
        Some(error_metrics_i32(
            &tile.float_validation_actual,
            &tile.float_validation_expected,
        )?)
    } else {
        None
    };

    Ok(EncodedTranscode {
        codestream,
        report: TranscodeReport {
            width: tile.jpeg.width,
            height: tile.jpeg.height,
            component_count: tile.jpeg.components.len(),
            components: tile.component_reports,
            float_reference_classification: float_reference_metrics
                .as_ref()
                .map(TranscodeValidationClassification::classify_metrics),
            float_reference_metrics,
            integer_reference_classification: integer_reference_metrics
                .as_ref()
                .map(TranscodeValidationClassification::classify_metrics),
            integer_reference_metrics,
            decomposition_levels: tile.decomposition_levels,
            coefficient_path: options.coefficient_path,
            path: transcode_path_name(tile.all_unit_sampled, options.coefficient_path),
            extract_us: timings.jpeg_dct_extract_us,
            transform_us: 0,
            encode_us,
            timings,
        },
    })
}

#[allow(clippy::too_many_lines)]
fn encode_float97_batch_tile<E: J2kEncodeStageAccelerator>(
    tile: Float97BatchTile,
    options: &JpegToHtj2kOptions,
    encode_accelerator: &mut E,
) -> Result<EncodedTranscode, JpegToHtj2kError> {
    let Float97BatchTile {
        jpeg,
        decomposition_levels,
        all_unit_sampled,
        component_reports,
        precomputed_components,
        preencoded_components,
        prequantized_components,
        float_validation_actual,
        float_validation_expected,
        mut timings,
        ..
    } = tile;

    let encode_start = Instant::now();
    let encode_dispatch_before = encode_accelerator.dispatch_report();
    let codestream = if preencoded_components.iter().any(Option::is_some) {
        let components = preencoded_components
            .into_iter()
            .map(|component| {
                component.ok_or(JpegToHtj2kError::Validation(
                    "9/7 preencoded batch transcode did not produce all components",
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let preencoded = PreencodedHtj2k97Image {
            width: jpeg.width,
            height: jpeg.height,
            bit_depth: 8,
            signed: false,
            components,
        };
        encode_preencoded_htj2k_97_with_accelerator(
            &preencoded,
            &options.encode_options,
            encode_accelerator,
        )
        .map_err(JpegToHtj2kError::Encode)?
    } else if prequantized_components.iter().any(Option::is_some) {
        let components = prequantized_components
            .into_iter()
            .map(|component| {
                component.ok_or(JpegToHtj2kError::Validation(
                    "9/7 code-block batch transcode did not produce all components",
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let prequantized = PrequantizedHtj2k97Image {
            width: jpeg.width,
            height: jpeg.height,
            bit_depth: 8,
            signed: false,
            components,
        };
        encode_prequantized_htj2k_97_with_accelerator(
            &prequantized,
            &options.encode_options,
            encode_accelerator,
        )
        .map_err(JpegToHtj2kError::Encode)?
    } else {
        let components = precomputed_components
            .into_iter()
            .map(|component| {
                component.ok_or(JpegToHtj2kError::Validation(
                    "9/7 batch transcode did not produce all components",
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let precomputed = PrecomputedHtj2k97Image {
            width: jpeg.width,
            height: jpeg.height,
            bit_depth: 8,
            signed: false,
            components,
        };
        encode_precomputed_htj2k_97_with_accelerator(
            &precomputed,
            &options.encode_options,
            encode_accelerator,
        )
        .map_err(JpegToHtj2kError::Encode)?
    };
    record_encode_dispatch_delta(
        &mut timings,
        encode_dispatch_before,
        encode_accelerator.dispatch_report(),
    );
    let encode_us = encode_start.elapsed().as_micros();
    timings.htj2k_encode_us = encode_us;
    let float_reference_metrics = if options.validate_against_float_reference {
        Some(error_metrics_i32(
            &float_validation_actual,
            &float_validation_expected,
        )?)
    } else {
        None
    };

    Ok(EncodedTranscode {
        codestream,
        report: TranscodeReport {
            width: jpeg.width,
            height: jpeg.height,
            component_count: jpeg.components.len(),
            components: component_reports,
            float_reference_classification: float_reference_metrics
                .as_ref()
                .map(TranscodeValidationClassification::classify_metrics),
            float_reference_metrics,
            integer_reference_classification: None,
            integer_reference_metrics: None,
            decomposition_levels,
            coefficient_path: options.coefficient_path,
            path: transcode_path_name(all_unit_sampled, options.coefficient_path),
            extract_us: timings.jpeg_dct_extract_us,
            transform_us: 0,
            encode_us,
            timings,
        },
    })
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
    let codestream = match component_batch.precomputed_components {
        PrecomputedComponentBatch::Dwt53(components) => {
            let precomputed = PrecomputedHtj2k53Image {
                width: jpeg.width,
                height: jpeg.height,
                bit_depth: 8,
                signed: false,
                components,
            };
            encode_precomputed_htj2k_53_with_accelerator(
                &precomputed,
                &options.encode_options,
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
            encode_precomputed_htj2k_97_with_accelerator(
                &precomputed,
                &options.encode_options,
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
            component,
            x_rsiz,
            y_rsiz,
            decomposition_levels,
            options,
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
    timings.batch_count = timings.batch_count.saturating_add(1);
    timings.batch_jobs = timings.batch_jobs.saturating_add(group.len());
    timings.accelerator_attempts = timings.accelerator_attempts.saturating_add(1);
    timings.accelerator_jobs = timings.accelerator_jobs.saturating_add(group.len());
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
        timings.accelerator_dispatches = timings.accelerator_dispatches.saturating_add(1);
        timings.accelerator_dispatched_jobs = timings
            .accelerator_dispatched_jobs
            .saturating_add(group.len());
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

#[allow(clippy::too_many_arguments)]
fn component_to_precomputed_htj2k(
    component: &JpegDctComponent,
    x_rsiz: u8,
    y_rsiz: u8,
    decomposition_levels: u8,
    options: &JpegToHtj2kOptions,
    scratch: &mut JpegToHtj2kScratch,
    accelerator: &mut impl DctToWaveletStageAccelerator,
    timings: &mut TranscodeTimingReport,
) -> Result<ComponentTranscodeResult, JpegToHtj2kError> {
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
    timings.accelerator_attempts = timings.accelerator_attempts.saturating_add(1);
    timings.accelerator_jobs = timings.accelerator_jobs.saturating_add(1);
    let accelerator_start = Instant::now();
    let accelerated = accelerator
        .dct_grid_to_dwt53(job)
        .map_err(JpegToHtj2kError::Accelerator)?;
    timings.dct_to_wavelet_accelerator_us = timings
        .dct_to_wavelet_accelerator_us
        .saturating_add(accelerator_start.elapsed().as_micros());
    let bands = if let Some(bands) = accelerated {
        timings.accelerator_dispatches = timings.accelerator_dispatches.saturating_add(1);
        timings.accelerator_dispatched_jobs = timings.accelerator_dispatched_jobs.saturating_add(1);
        bands
    } else {
        timings.cpu_fallback_jobs = timings.cpu_fallback_jobs.saturating_add(1);
        let fallback_start = Instant::now();
        let bands = dct8x8_blocks_to_dwt53_float_linear_with_scratch(
            blocks,
            component.block_cols as usize,
            component.block_rows as usize,
            component.width as usize,
            component.height as usize,
            &mut scratch.dct53_grid,
        )?;
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
    timings.accelerator_attempts = timings.accelerator_attempts.saturating_add(1);
    timings.accelerator_jobs = timings.accelerator_jobs.saturating_add(1);
    let accelerator_start = Instant::now();
    let accelerated = accelerator
        .dct_grid_to_dwt97(job)
        .map_err(JpegToHtj2kError::Accelerator)?;
    timings.dct_to_wavelet_accelerator_us = timings
        .dct_to_wavelet_accelerator_us
        .saturating_add(accelerator_start.elapsed().as_micros());
    let bands = if let Some(bands) = accelerated {
        timings.accelerator_dispatches = timings.accelerator_dispatches.saturating_add(1);
        timings.accelerator_dispatched_jobs = timings.accelerator_dispatched_jobs.saturating_add(1);
        bands
    } else {
        timings.cpu_fallback_jobs = timings.cpu_fallback_jobs.saturating_add(1);
        let fallback_start = Instant::now();
        let bands = dct8x8_blocks_then_dwt97_float_with_scratch(
            blocks,
            component.block_cols as usize,
            component.block_rows as usize,
            component.width as usize,
            component.height as usize,
            &mut scratch.dct97_grid,
        )?;
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
    )?;
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
    )?;
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
    timings.accelerator_attempts = timings.accelerator_attempts.saturating_add(1);
    timings.accelerator_jobs = timings.accelerator_jobs.saturating_add(1);
    let accelerator_start = Instant::now();
    let accelerated_first_level = accelerator
        .dct_grid_to_reversible_dwt53(job)
        .map_err(JpegToHtj2kError::Accelerator)?;
    timings.dct_to_wavelet_accelerator_us = timings
        .dct_to_wavelet_accelerator_us
        .saturating_add(accelerator_start.elapsed().as_micros());
    if let Some(first_level) = accelerated_first_level {
        timings.accelerator_dispatches = timings.accelerator_dispatches.saturating_add(1);
        timings.accelerator_dispatched_jobs = timings.accelerator_dispatched_jobs.saturating_add(1);
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
    timings.cpu_fallback_jobs = timings.cpu_fallback_jobs.saturating_add(1);
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
    let even_idx = low_idx * 2;
    let current = component_sample_i32(component, idct_blocks, x, even_idx)?;
    if height < 2 {
        return Ok(current);
    }

    if height.is_multiple_of(2) {
        let right = vertical_high_53_i32_at(component, idct_blocks, x, low_idx)?;
        if low_idx == 0 {
            return Ok(current + floor_div_i32(right + 1, 2));
        }
        let left = vertical_high_53_i32_at(component, idct_blocks, x, low_idx - 1)?;
        return Ok(current + floor_div_i32(left + right + 2, 4));
    }

    let high_len = height / 2;
    if high_len == 0 {
        return Ok(current);
    }
    let left = if low_idx > 0 {
        vertical_high_53_i32_at(component, idct_blocks, x, low_idx - 1)?
    } else {
        vertical_high_53_i32_at(component, idct_blocks, x, 0)?
    };
    let right = if low_idx < high_len {
        vertical_high_53_i32_at(component, idct_blocks, x, low_idx)?
    } else {
        left
    };
    Ok(current + floor_div_i32(left + right + 2, 4))
}

fn vertical_high_53_i32_at(
    component: &JpegDctComponent,
    idct_blocks: &mut [Option<[i32; 64]>],
    x: usize,
    high_idx: usize,
) -> Result<i32, JpegToHtj2kError> {
    let height = component.height as usize;
    let odd_idx = high_idx * 2 + 1;
    let current = component_sample_i32(component, idct_blocks, x, odd_idx)?;
    let left = component_sample_i32(component, idct_blocks, x, odd_idx - 1)?;
    if height.is_multiple_of(2) && odd_idx + 1 == height {
        return Ok(current - left);
    }

    let right_idx = if odd_idx + 1 < height {
        odd_idx + 1
    } else {
        height - 1
    };
    let right = component_sample_i32(component, idct_blocks, x, right_idx)?;
    Ok(current - floor_div_i32(left + right, 2))
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
