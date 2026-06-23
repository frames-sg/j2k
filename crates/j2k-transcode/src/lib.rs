// SPDX-License-Identifier: MIT OR Apache-2.0

//! JPEG-to-HTJ2K coefficient-domain transcode workflow.

pub mod accelerator;

#[doc(hidden)]
pub mod corpus_validation;
#[doc(hidden)]
pub mod dct53_1d;
pub mod dct53_2d;
#[doc(hidden)]
pub mod dct53_multilevel;
pub mod dct97_2d;
mod dct_grid;
pub mod htj2k97_codeblock_oracle;
#[doc(hidden)]
mod jpeg_to_htj2k;
#[doc(hidden)]
pub mod metrics;
mod pipeline_map;
mod reversible53;

pub use j2k::J2kProgressionOrder as EncodeProgressionOrder;

pub use dct_grid::DctGridError;
pub use jpeg_to_htj2k::{
    jpeg_to_htj2k, jpeg_to_htj2k_batch, BatchTranscodeReport, EncodedTranscode,
    EncodedTranscodeBatch, JpegTileBatchInput, JpegToHtj2kCoefficientPath,
    JpegToHtj2kEncodeOptions, JpegToHtj2kError, JpegToHtj2kOptions, JpegToHtj2kTranscoder,
    TranscodeComponentReport, TranscodeReport, TranscodeTimingReport,
    TranscodeValidationClassification, TranscodeValidationMetrics,
    JPEG_TO_HTJ2K_LOSSY_97_QUANTIZATION_SCALE,
};
pub use pipeline_map::{
    TranscodePipelineMap, TranscodePipelineStageKind, TranscodePipelineStageReport,
    TranscodeResidentStageRecommendation, TranscodeStageProcessor,
};
