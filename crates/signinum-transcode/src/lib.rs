// SPDX-License-Identifier: Apache-2.0

//! Coefficient-domain transcode primitives.
//!
//! The stable API covers the CPU-first JPEG-to-HTJ2K transcode pipeline, its
//! validation reports, and the accelerator hooks used by device adapters.

#![deny(missing_docs)]

pub mod accelerator;
pub mod corpus_validation;
pub mod dct53_1d;
pub mod dct53_2d;
pub mod dct53_multilevel;
pub mod dct97_2d;
pub mod htj2k_wavelet;
mod jpeg_to_htj2k;
pub mod metrics;

pub use signinum_j2k_native::EncodeProgressionOrder;

pub use jpeg_to_htj2k::{
    jpeg_to_htj2k, jpeg_to_htj2k_batch, BatchTranscodeReport, EncodedTranscode,
    EncodedTranscodeBatch, JpegTileBatchInput, JpegToHtj2kCoefficientPath, JpegToHtj2kError,
    JpegToHtj2kOptions, JpegToHtj2kTranscoder, TranscodeComponentReport, TranscodeReport,
    TranscodeTimingReport, TranscodeValidationClassification, TranscodeValidationMetrics,
    JPEG_TO_HTJ2K_LOSSY_97_QUANTIZATION_SCALE,
};
