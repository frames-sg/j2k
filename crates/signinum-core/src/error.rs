// SPDX-License-Identifier: Apache-2.0

use crate::{pixel::PixelFormat, sample::SampleType};

/// Error returned when caller-owned input or output buffers are invalid.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum BufferError {
    /// Output buffer does not contain enough bytes.
    #[error("output buffer too small: required {required} bytes, have {have}")]
    OutputTooSmall {
        /// Required byte length.
        required: usize,
        /// Available byte length.
        have: usize,
    },
    /// Output exceeded the caller-provided capacity before the full decoded
    /// size was known.
    #[error("output exceeds capacity: observed at least {lower_bound} bytes, have {have}")]
    OutputExceedsCapacity {
        /// Lower bound observed before stopping the decode.
        lower_bound: usize,
        /// Available byte length.
        have: usize,
    },
    /// Input buffer does not contain enough bytes.
    #[error("input buffer too small: required {required} bytes, have {have}")]
    InputTooSmall {
        /// Required byte length.
        required: usize,
        /// Available byte length.
        have: usize,
    },
    /// Size computation overflowed.
    #[error("buffer size overflow while computing {what}")]
    SizeOverflow {
        /// Name of the size being computed.
        what: &'static str,
    },
    /// Destination stride is smaller than one row.
    #[error("stride {stride} is smaller than row width {row_bytes}")]
    StrideTooSmall {
        /// Required bytes per row.
        row_bytes: usize,
        /// Caller-provided stride.
        stride: usize,
    },
    /// Destination stride does not meet an alignment requirement.
    #[error("stride {stride} is not aligned to {align}")]
    StrideNotAligned {
        /// Caller-provided stride.
        stride: usize,
        /// Required alignment in bytes.
        align: usize,
    },
    /// Pixel format and typed sample buffer do not agree.
    #[error("pixel format {fmt:?} does not match sample type {sample_type:?}")]
    SampleTypeMismatch {
        /// Requested pixel format.
        fmt: PixelFormat,
        /// Sample type accepted by the target buffer.
        sample_type: SampleType,
    },
}

/// Error returned while reading structured input bytes.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum InputError {
    /// Input ended before the minimum required length.
    #[error("input too short: need {need} bytes, have {have}")]
    TooShort {
        /// Required byte length.
        need: usize,
        /// Available byte length.
        have: usize,
    },
    /// Input ended while reading a named segment.
    #[error("input truncated at offset {offset} while reading {segment}")]
    TruncatedAt {
        /// Byte offset where parsing failed.
        offset: usize,
        /// Segment or marker being read.
        segment: &'static str,
    },
}

/// Marker error for functionality that is planned but not implemented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("not yet implemented: {what}")]
pub struct NotImplemented {
    /// Feature or operation that is not implemented.
    pub what: &'static str,
}

/// Marker error for inputs or options outside the supported codec surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("unsupported: {what}")]
pub struct Unsupported {
    /// Unsupported feature or operation.
    pub what: &'static str,
}

/// Shared classification methods implemented by codec-specific error enums.
pub trait CodecError: core::error::Error + Send + Sync + 'static {
    /// Return true when the error was caused by truncated input.
    fn is_truncated(&self) -> bool;
    /// Return true when the error reports an intentionally missing feature.
    fn is_not_implemented(&self) -> bool;
    /// Return true when the error reports unsupported input or options.
    fn is_unsupported(&self) -> bool;
    /// Return true when the error was caused by caller-owned buffer shape.
    fn is_buffer_error(&self) -> bool;
}
