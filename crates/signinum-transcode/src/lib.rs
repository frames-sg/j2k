// SPDX-License-Identifier: Apache-2.0

//! Experimental coefficient-domain transcode primitives.
//!
//! The first target is a constrained synthetic proof for mapping one 8-sample
//! DCT block into one level of the reversible 5/3 wavelet transform.

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
