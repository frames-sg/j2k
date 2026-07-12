// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed validation failures for public HT packet-segment length math.

use core::fmt;

/// Failure returned while validating HTJ2K packet contribution segment lengths.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum HtSegmentLengthError {
    /// A zero-pass contribution carried payload or segment metadata.
    EmptyContributionHasSegments,
    /// The contribution payload cannot be represented by JPEG 2000's u32 lengths.
    ContributionLengthExceedsU32 {
        /// Actual contribution payload length.
        data_len: usize,
    },
    /// A refinement-only contribution length does not cover its complete payload.
    RefinementOnlyLengthMismatch {
        /// Complete contribution payload length.
        data_len: u32,
        /// Declared refinement segment length.
        refinement_length: u32,
    },
    /// A refinement segment cannot be represented by HTJ2K packet-header signaling.
    RefinementLengthOutOfRange {
        /// Rejected refinement segment length.
        refinement_length: u32,
    },
    /// A single-pass contribution incorrectly declared refinement bytes.
    SinglePassHasRefinement {
        /// Rejected refinement segment length.
        refinement_length: u32,
    },
    /// A single-pass cleanup length does not cover its complete payload.
    SinglePassLengthMismatch {
        /// Complete contribution payload length.
        data_len: u32,
        /// Declared cleanup segment length.
        cleanup_length: u32,
    },
    /// A multi-pass contribution omitted either its cleanup or refinement length.
    MultiPassRequiresSegments {
        /// Declared cleanup segment length.
        cleanup_length: u32,
        /// Declared refinement segment length.
        refinement_length: u32,
    },
    /// Adding the multi-pass cleanup and refinement lengths overflowed.
    MultiPassLengthOverflow {
        /// Declared cleanup segment length.
        cleanup_length: u32,
        /// Declared refinement segment length.
        refinement_length: u32,
    },
    /// Multi-pass cleanup and refinement lengths do not cover the complete payload.
    MultiPassLengthMismatch {
        /// Complete contribution payload length.
        data_len: u32,
        /// Declared cleanup segment length.
        cleanup_length: u32,
        /// Declared refinement segment length.
        refinement_length: u32,
    },
    /// A cleanup segment cannot be represented by HTJ2K packet-header signaling.
    CleanupLengthOutOfRange {
        /// Rejected cleanup segment length.
        cleanup_length: u32,
    },
}

impl HtSegmentLengthError {
    /// Returns allocation-free presentation text for legacy diagnostics.
    #[must_use]
    pub const fn reason(self) -> &'static str {
        match self {
            Self::EmptyContributionHasSegments => {
                "empty HTJ2K packet contribution must not carry segment bytes"
            }
            Self::ContributionLengthExceedsU32 { .. } => {
                "HTJ2K packet contribution exceeds u32 length"
            }
            Self::RefinementOnlyLengthMismatch { .. } => {
                "refinement-only HTJ2K packet contribution length mismatch"
            }
            Self::RefinementLengthOutOfRange { .. } => {
                "HTJ2K refinement segment length is out of range"
            }
            Self::SinglePassHasRefinement { .. } => {
                "single-pass HTJ2K packet contribution must not carry refinement bytes"
            }
            Self::SinglePassLengthMismatch { .. } => {
                "single-pass HTJ2K packet contribution length mismatch"
            }
            Self::MultiPassRequiresSegments { .. } => {
                "multi-pass HTJ2K packet contribution requires cleanup/refinement lengths"
            }
            Self::MultiPassLengthOverflow { .. } => {
                "multi-pass HTJ2K packet contribution length overflow"
            }
            Self::MultiPassLengthMismatch { .. } => {
                "multi-pass HTJ2K packet contribution length mismatch"
            }
            Self::CleanupLengthOutOfRange { .. } => "HTJ2K cleanup segment length is out of range",
        }
    }
}

impl fmt::Display for HtSegmentLengthError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::EmptyContributionHasSegments => formatter.write_str(self.reason()),
            Self::ContributionLengthExceedsU32 { data_len } => {
                write!(formatter, "{}: {data_len} bytes", self.reason())
            }
            Self::RefinementOnlyLengthMismatch {
                data_len,
                refinement_length,
            } => write!(
                formatter,
                "{}: payload {data_len}, refinement {refinement_length}",
                self.reason()
            ),
            Self::RefinementLengthOutOfRange { refinement_length }
            | Self::SinglePassHasRefinement { refinement_length } => {
                write!(formatter, "{}: {refinement_length}", self.reason())
            }
            Self::SinglePassLengthMismatch {
                data_len,
                cleanup_length,
            } => write!(
                formatter,
                "{}: payload {data_len}, cleanup {cleanup_length}",
                self.reason()
            ),
            Self::MultiPassRequiresSegments {
                cleanup_length,
                refinement_length,
            }
            | Self::MultiPassLengthOverflow {
                cleanup_length,
                refinement_length,
            } => write!(
                formatter,
                "{}: cleanup {cleanup_length}, refinement {refinement_length}",
                self.reason()
            ),
            Self::MultiPassLengthMismatch {
                data_len,
                cleanup_length,
                refinement_length,
            } => write!(
                formatter,
                "{}: payload {data_len}, cleanup {cleanup_length}, refinement {refinement_length}",
                self.reason()
            ),
            Self::CleanupLengthOutOfRange { cleanup_length } => {
                write!(formatter, "{}: {cleanup_length}", self.reason())
            }
        }
    }
}

impl core::error::Error for HtSegmentLengthError {}
