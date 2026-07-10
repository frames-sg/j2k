// SPDX-License-Identifier: MIT OR Apache-2.0

//! JPEG-to-HTJ2K coefficient-domain transcode workflow.

#[doc(hidden)]
pub mod accelerator;
pub use self::accelerator::{
    idct_blocks_to_signed_samples_rayon, CpuOnlyDctToWaveletStageAccelerator,
    DctGridI16ToHtj2k97CodeBlockBatch, DctGridI16ToHtj2k97CodeBlockJob, DctGridToDwt53Job,
    DctGridToDwt97Job, DctGridToHtj2k97CodeBlockJob, DctGridToReversibleDwt53Job,
    DctToWaveletStageAccelerator, DctToWaveletStageCounterEvent, DctToWaveletStageCounters,
    Dwt97BatchStageTimings, EncodedHtJ2kCodeBlock, Htj2k97CodeBlockOptions,
    IrreversibleQuantizationSubbandScales, J2kSubBandType, PreencodedHtj2k97CodeBlock,
    PreencodedHtj2k97CompactBatch, PreencodedHtj2k97CompactBatchGroups,
    PreencodedHtj2k97CompactCodeBlock, PreencodedHtj2k97CompactComponent,
    PreencodedHtj2k97CompactImage, PreencodedHtj2k97CompactResolution,
    PreencodedHtj2k97CompactSubband, PreencodedHtj2k97Component, PreencodedHtj2k97Resolution,
    PreencodedHtj2k97Subband, PrequantizedHtj2k97CodeBlock, PrequantizedHtj2k97Component,
    PrequantizedHtj2k97Image, PrequantizedHtj2k97Resolution, PrequantizedHtj2k97Subband,
    RayonReversibleDwt53Accelerator, ReversibleDwt53FirstLevel, TranscodeStageDispatchMode,
    TranscodeStageError,
};

mod dct53_2d;
mod dct97_2d;
mod dct_grid;
#[cfg(feature = "dev-support")]
#[doc(hidden)]
pub mod dev_support;
mod htj2k97_codeblock_oracle;
#[doc(hidden)]
mod jpeg_to_htj2k;
#[doc(hidden)]
pub mod metrics;
mod pipeline_map;
mod resident;
mod reversible53;

pub use j2k::J2kProgressionOrder as EncodeProgressionOrder;

/// One separable single-level 2D 5/3 transform result.
#[derive(Debug, Clone, PartialEq)]
pub struct Dwt53TwoDimensional<T> {
    /// Low-horizontal, low-vertical band.
    pub ll: Vec<T>,
    /// High-horizontal, low-vertical band.
    pub hl: Vec<T>,
    /// Low-horizontal, high-vertical band.
    pub lh: Vec<T>,
    /// High-horizontal, high-vertical band.
    pub hh: Vec<T>,
    /// Width of horizontally low-pass bands.
    pub low_width: usize,
    /// Height of vertically low-pass bands.
    pub low_height: usize,
    /// Width of horizontally high-pass bands.
    pub high_width: usize,
    /// Height of vertically high-pass bands.
    pub high_height: usize,
}

/// One separable single-level 2D 9/7 transform result.
#[derive(Debug, Clone, PartialEq)]
pub struct Dwt97TwoDimensional<T> {
    /// Low-horizontal, low-vertical band.
    pub ll: Vec<T>,
    /// High-horizontal, low-vertical band.
    pub hl: Vec<T>,
    /// Low-horizontal, high-vertical band.
    pub lh: Vec<T>,
    /// High-horizontal, high-vertical band.
    pub hh: Vec<T>,
    /// Width of horizontally low-pass bands.
    pub low_width: usize,
    /// Height of vertically low-pass bands.
    pub low_height: usize,
    /// Width of horizontally high-pass bands.
    pub high_width: usize,
    /// Height of vertically high-pass bands.
    pub high_height: usize,
}

pub use dct53_2d::{dct8x8_blocks_then_dwt53_float, dct8x8_blocks_to_dwt53_float_linear};
pub use dct97_2d::dct8x8_blocks_then_dwt97_float;
pub use dct_grid::DctGridError;
pub use htj2k97_codeblock_oracle::{
    htj2k97_subband_delta, htj2k97_subband_total_bitplanes, validate_htj2k97_codeblock_options,
};
pub use jpeg_to_htj2k::{
    jpeg_to_htj2k, jpeg_to_htj2k_batch, BatchTranscodeReport, EncodedTranscode,
    EncodedTranscodeBatch, JpegTileBatchInput, JpegToHtj2kCoefficientPath,
    JpegToHtj2kEncodeOptions, JpegToHtj2kError, JpegToHtj2kOptions, JpegToHtj2kTranscoder,
    TranscodeBatchProfileRequest, TranscodeBatchProfileRow, TranscodeComponentReport,
    TranscodeReport, TranscodeTimingReport, TranscodeValidationClassification,
    TranscodeValidationMetrics, JPEG_TO_HTJ2K_LOSSY_97_QUANTIZATION_SCALE,
};
pub use pipeline_map::{
    TranscodePipelineMap, TranscodePipelineStageKind, TranscodePipelineStageReport,
    TranscodeResidentStageRecommendation, TranscodeStageProcessor,
};
pub use resident::{
    ResidentBufferRef, ResidentCodestreamBuffer, ResidentColorModel, ResidentComponentGeometry,
    ResidentDctCoefficientOrder, ResidentDctGridLayout, ResidentDwtSubband, ResidentDwtSubbandKind,
    ResidentDwtSubbandLayout, ResidentHandoffError, ResidentJpegDctGrid, ResidentSampleInfo,
    ResidentSampling,
};
