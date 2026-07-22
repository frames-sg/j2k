// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::sync::Arc;
use core::fmt;

use crate::{DecodeSettings, J2kError};

/// Stage at which an indexed image failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BatchErrorStage {
    /// Header parsing, request planning, or representability validation.
    Prepare,
    /// Pixel reconstruction or output packing.
    Decode,
}

impl fmt::Display for BatchErrorStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Prepare => f.write_str("prepare"),
            Self::Decode => f.write_str("decode"),
        }
    }
}

/// Reason an otherwise valid image cannot be represented by the fast batch API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NonRepresentableReason {
    /// Output requires a component count other than Gray, RGB, or RGBA.
    UnsupportedComponentCount,
    /// Components use different significant precisions.
    MixedPrecision,
    /// Components mix signed and unsigned sample domains.
    MixedSignedness,
    /// At least one component is subsampled.
    ComponentSubsampling,
    /// At least one component uses more than sixteen significant bits.
    PrecisionAboveSixteen,
    /// Tile or component coding-style overrides select different wavelet transforms.
    MixedWaveletTransform,
    /// Parsed color metadata is not representable as Gray, RGB, or RGBA.
    UnsupportedColor,
}

impl fmt::Display for NonRepresentableReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedComponentCount => f.write_str("unsupported component count"),
            Self::MixedPrecision => f.write_str("mixed component precision"),
            Self::MixedSignedness => f.write_str("mixed component signedness"),
            Self::ComponentSubsampling => f.write_str("component subsampling"),
            Self::PrecisionAboveSixteen => f.write_str("component precision above sixteen bits"),
            Self::MixedWaveletTransform => {
                f.write_str("mixed tile or component wavelet transforms")
            }
            Self::UnsupportedColor => f.write_str("unsupported color interpretation"),
        }
    }
}

/// Structured error for one submitted image.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum BatchItemError {
    /// Codec parsing or decoding failed.
    #[error("batch image {stage} failed: {source}")]
    Codec {
        /// Operation phase.
        stage: BatchErrorStage,
        /// Source-preserving codec error.
        #[source]
        source: Arc<J2kError>,
    },
    /// Valid codec output cannot be represented by one dense native-width batch.
    #[error("non-representable batch output: {reason}")]
    NonRepresentableBatchOutput {
        /// Stable representability class.
        reason: NonRepresentableReason,
    },
    /// A prepared image was submitted under a different codec validation policy.
    #[error(
        "prepared image decode settings {prepared:?} do not match requested settings {requested:?}"
    )]
    PreparedDecodeSettingsMismatch {
        /// Validation policy used to parse and build the retained execution plan.
        prepared: DecodeSettings,
        /// Validation policy requested for the new batch.
        requested: DecodeSettings,
    },
}

/// One batch item error retaining the caller's input index.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("image {index}: {source}")]
pub struct IndexedBatchError {
    /// Index in the original submitted input slice.
    pub index: usize,
    /// Structured preparation or decode failure.
    #[source]
    pub source: BatchItemError,
}
