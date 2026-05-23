// SPDX-License-Identifier: Apache-2.0

//! Experimental coefficient-domain transcode primitives.
//!
//! The first target is a constrained synthetic proof for mapping one 8-sample
//! DCT block into one level of the reversible 5/3 wavelet transform.

pub mod dct53_1d;
pub mod dct53_2d;
pub mod dct53_multilevel;
pub mod htj2k_wavelet;
mod jpeg_to_htj2k;
pub mod metrics;

pub use jpeg_to_htj2k::{
    jpeg_to_htj2k, EncodedTranscode, JpegToHtj2kError, JpegToHtj2kOptions,
    TranscodeComponentReport, TranscodeReport, TranscodeValidationMetrics,
};
