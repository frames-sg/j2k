// SPDX-License-Identifier: MIT OR Apache-2.0

//! JPEG-to-HTJ2K coefficient-domain transcode workflow.

mod accelerator_contracts;
mod allocation;
pub use self::accelerator_contracts::{
    idct_blocks_to_signed_samples_rayon, CpuOnlyDctToWaveletStageAccelerator,
    DctGridI16ToHtj2k97CodeBlockBatch, DctGridI16ToHtj2k97CodeBlockJob, DctGridToDwt53Job,
    DctGridToDwt97Job, DctGridToHtj2k97CodeBlockJob, DctToWaveletStageAccelerator,
    DctToWaveletStageCounterEvent, DctToWaveletStageCounters, EncodedHtJ2kCodeBlock,
    Htj2k97CodeBlockOptions, IrreversibleQuantizationSubbandScales, J2kSubBandType,
    PreencodedHtj2k97CodeBlock, PreencodedHtj2k97CompactBatch, PreencodedHtj2k97CompactBatchGroups,
    PreencodedHtj2k97CompactCodeBlock, PreencodedHtj2k97CompactComponent,
    PreencodedHtj2k97CompactImage, PreencodedHtj2k97CompactResolution,
    PreencodedHtj2k97CompactSubband, PreencodedHtj2k97Component, PreencodedHtj2k97Resolution,
    PreencodedHtj2k97Subband, PrequantizedHtj2k97CodeBlock, PrequantizedHtj2k97Component,
    PrequantizedHtj2k97Image, PrequantizedHtj2k97Resolution, PrequantizedHtj2k97Subband,
    RayonReversibleDwt53Accelerator, TranscodeStageDispatchMode,
};

/// Compatibility namespace for the accelerator contracts exported at the
/// crate root.
#[doc(hidden)]
pub mod accelerator {
    pub use crate::{
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
}

// These contracts occur in downstream crates' public signatures. Defining
// them at the crate root preserves their canonical semver paths while their
// implementations remain in the focused accelerator-contracts module.

/// Direct DCT-grid to one-level reversible integer 5/3 projection job.
#[derive(Debug, Clone, Copy)]
pub struct DctGridToReversibleDwt53Job<'a> {
    /// Natural-order, dequantized 8x8 DCT blocks.
    pub dequantized_blocks: &'a [[i16; 64]],
    /// Number of DCT block columns in `dequantized_blocks`.
    pub block_cols: usize,
    /// Number of DCT block rows in `dequantized_blocks`.
    pub block_rows: usize,
    /// Logical component width in samples.
    pub width: usize,
    /// Logical component height in samples.
    pub height: usize,
}

/// One separable single-level reversible integer 5/3 transform result.
#[derive(Debug, PartialEq, Eq)]
pub struct ReversibleDwt53FirstLevel {
    /// Low-horizontal, low-vertical band.
    pub ll: Vec<i32>,
    /// High-horizontal, low-vertical band.
    pub hl: Vec<i32>,
    /// Low-horizontal, high-vertical band.
    pub lh: Vec<i32>,
    /// High-horizontal, high-vertical band.
    pub hh: Vec<i32>,
    /// Width of horizontally low-pass bands.
    pub low_width: usize,
    /// Height of vertically low-pass bands.
    pub low_height: usize,
    /// Width of horizontally high-pass bands.
    pub high_width: usize,
    /// Height of vertically high-pass bands.
    pub high_height: usize,
}

/// Backend-specific timing breakdown for a same-geometry 9/7 batch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Dwt97BatchStageTimings {
    /// Host packing, buffer allocation, and upload time in microseconds.
    pub pack_upload_us: u128,
    /// Logical host-to-device transfers included in [`Self::pack_upload_us`].
    pub pack_upload_transfers: usize,
    /// Host-to-device bytes included in [`Self::pack_upload_us`].
    pub pack_upload_bytes: u64,
    /// Resident JPEG DCT-grid descriptors validated for this batch.
    pub resident_dct_handoff_count: usize,
    /// Time spent in the IDCT plus horizontal 9/7 row-lift stage.
    pub idct_row_lift_us: u128,
    /// Time spent in the vertical 9/7 column-lift stage.
    pub column_lift_us: u128,
    /// Resident DWT subband descriptors validated for this batch.
    pub resident_dwt_handoff_count: usize,
    /// Time spent quantizing 9/7 bands into HTJ2K code-block layout.
    pub quantize_codeblock_us: u128,
    /// Time spent HT-encoding resident code-block coefficients.
    pub ht_encode_us: u128,
    /// Resident HT cleanup-pass encode kernel time in microseconds.
    pub ht_kernel_us: u128,
    /// Resident HT status-buffer device-to-host readback time in microseconds.
    pub ht_status_readback_us: u128,
    /// Logical device-to-host status readbacks included in [`Self::ht_status_readback_us`].
    pub ht_status_readback_transfers: usize,
    /// Device-to-host status bytes included in [`Self::ht_status_readback_us`].
    pub ht_status_readback_bytes: u64,
    /// Resident HT encoded-byte compaction kernel time in microseconds.
    pub ht_compact_us: u128,
    /// Resident HT compacted encoded-byte device-to-host readback time in microseconds.
    pub ht_output_readback_us: u128,
    /// Logical device-to-host output readbacks included in [`Self::ht_output_readback_us`].
    pub ht_output_readback_transfers: usize,
    /// Device-to-host output bytes included in [`Self::ht_output_readback_us`].
    pub ht_output_readback_bytes: u64,
    /// Number of HT code-block encode kernel dispatches in this batch.
    pub ht_codeblock_dispatches: usize,
    /// Time spent reading and unpacking Metal band buffers into host outputs.
    pub readback_us: u128,
    /// Logical device-to-host transfers included in [`Self::readback_us`].
    pub readback_transfers: usize,
    /// Device-to-host bytes included in [`Self::readback_us`].
    pub readback_bytes: u64,
}

mod dct53_2d;
mod dct97_2d;
mod dct_grid;
#[cfg(feature = "dev-support")]
#[doc(hidden)]
pub mod dev_support;
mod htj2k97_codeblock_error;
mod htj2k97_codeblock_oracle;
#[doc(hidden)]
mod jpeg_to_htj2k;
#[doc(hidden)]
pub mod metrics;
mod pipeline_map;
mod resident;
mod reversible53;
mod transcode_stage_error;

pub use j2k::J2kProgressionOrder as EncodeProgressionOrder;

/// One separable single-level 2D 5/3 transform result.
#[derive(Debug, PartialEq)]
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
#[derive(Debug, PartialEq)]
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
pub use dct97_2d::{dct8x8_blocks_then_dwt97_float_with_scratch, Dct97GridScratch};
pub use dct_grid::{DctGridError, DctTransformError};
pub use htj2k97_codeblock_error::{Htj2k97CodeBlockAxis, Htj2k97CodeBlockOptionsError};
pub use htj2k97_codeblock_oracle::{
    htj2k97_subband_delta, htj2k97_subband_total_bitplanes, validate_htj2k97_codeblock_options,
};
pub use jpeg_to_htj2k::{
    jpeg_to_htj2k, jpeg_to_htj2k_batch, BatchTranscodeReport, EncodedTranscode,
    EncodedTranscodeBatch, Htj2kEncodeError, Htj2kEncodeErrorKind, JpegTileBatchInput,
    JpegToHtj2kCoefficientPath, JpegToHtj2kEncodeOptions, JpegToHtj2kError, JpegToHtj2kOptions,
    JpegToHtj2kTranscoder, TranscodeBatchProfileRequest, TranscodeBatchProfileRow,
    TranscodeComponentReport, TranscodeReport, TranscodeTimingReport,
    TranscodeValidationClassification, TranscodeValidationMetrics,
    JPEG_TO_HTJ2K_LOSSY_97_QUANTIZATION_SCALE,
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
pub use transcode_stage_error::TranscodeStageError;
