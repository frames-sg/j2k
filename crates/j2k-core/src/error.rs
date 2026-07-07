// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{pixel::PixelFormat, sample::SampleType};

/// Buffer validation and copy errors.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BufferError {
    /// Destination buffer is too small for the requested output.
    #[error("output buffer too small: required {required} bytes, have {have}")]
    OutputTooSmall {
        /// Required destination length in bytes.
        required: usize,
        /// Actual destination length in bytes.
        have: usize,
    },
    /// Source buffer is too small for the requested copy/decode.
    #[error("input buffer too small: required {required} bytes, have {have}")]
    InputTooSmall {
        /// Required source length in bytes.
        required: usize,
        /// Actual source length in bytes.
        have: usize,
    },
    /// A byte-count computation overflowed.
    #[error("buffer size overflow while computing {what}")]
    SizeOverflow {
        /// Name of the size being computed.
        what: &'static str,
    },
    /// A requested allocation exceeds the configured safety cap.
    #[error("{what} allocation too large: requested {requested} bytes, cap {cap}")]
    AllocationTooLarge {
        /// Requested byte count.
        requested: usize,
        /// Configured byte cap.
        cap: usize,
        /// Name of the allocation being checked.
        what: &'static str,
    },
    /// Output stride cannot hold one decoded row.
    #[error("stride {stride} is smaller than row width {row_bytes}")]
    StrideTooSmall {
        /// Required row width in bytes.
        row_bytes: usize,
        /// Supplied stride in bytes.
        stride: usize,
    },
    /// Output stride does not meet backend alignment requirements.
    #[error("stride {stride} is not aligned to {align}")]
    StrideNotAligned {
        /// Supplied stride in bytes.
        stride: usize,
        /// Required byte alignment.
        align: usize,
    },
    /// Requested pixel format uses a different sample width than the row type.
    #[error("pixel format {fmt:?} does not match sample type {sample_type:?}")]
    SampleTypeMismatch {
        /// Requested output pixel format.
        fmt: PixelFormat,
        /// Sample type accepted by the current sink or buffer.
        sample_type: SampleType,
    },
}

/// Generic malformed or truncated input errors.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InputError {
    /// Input ended before a required fixed-size read.
    #[error("input too short: need {need} bytes, have {have}")]
    TooShort {
        /// Required byte count.
        need: usize,
        /// Available byte count.
        have: usize,
    },
    /// Input ended while reading a named segment.
    #[error("input truncated at offset {offset} while reading {segment}")]
    TruncatedAt {
        /// Byte offset where truncation was detected.
        offset: usize,
        /// Name of the segment being read.
        segment: &'static str,
    },
}

/// Error for a valid request that is not implemented yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("not yet implemented: {what}")]
pub struct NotImplemented {
    /// Feature or path that is not implemented.
    pub what: &'static str,
}

/// Error for input or options unsupported by the current codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("unsupported: {what}")]
pub struct Unsupported {
    /// Unsupported feature or option.
    pub what: &'static str,
}

/// Shared error classification used by facade traits.
pub trait CodecError: core::error::Error + Send + Sync + 'static {
    /// True when the error indicates truncated input.
    fn is_truncated(&self) -> bool;
    /// True when the error indicates an unimplemented supported surface.
    fn is_not_implemented(&self) -> bool;
    /// True when the error indicates unsupported input or options.
    fn is_unsupported(&self) -> bool;
    /// True when the error indicates caller buffer sizing or layout problems.
    fn is_buffer_error(&self) -> bool;
}

/// Backend-adapter-local error classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AdapterErrorKind {
    /// Error is not a shared adapter classification.
    Other,
    /// Error is a caller buffer sizing or layout problem.
    Buffer,
    /// Error is an unsupported backend, request, input, or host capability.
    Unsupported,
}

/// Variant mapping supplied by GPU adapter error enums.
pub trait AdapterErrorParts {
    /// Return the wrapped codec/fallback error, when this adapter error has one.
    fn source_codec_error(&self) -> Option<&dyn CodecError>;

    /// Return the adapter-local classification for non-codec variants.
    fn adapter_error_kind(&self) -> AdapterErrorKind;
}

/// Shared truncated-input classification for adapter errors.
#[doc(hidden)]
pub fn adapter_error_is_truncated(error: &impl AdapterErrorParts) -> bool {
    error
        .source_codec_error()
        .is_some_and(CodecError::is_truncated)
}

/// Shared not-implemented classification for adapter errors.
#[doc(hidden)]
pub fn adapter_error_is_not_implemented(error: &impl AdapterErrorParts) -> bool {
    error
        .source_codec_error()
        .is_some_and(CodecError::is_not_implemented)
}

/// Shared unsupported classification for adapter errors.
#[doc(hidden)]
pub fn adapter_error_is_unsupported(error: &impl AdapterErrorParts) -> bool {
    error.adapter_error_kind() == AdapterErrorKind::Unsupported
        || error
            .source_codec_error()
            .is_some_and(CodecError::is_unsupported)
}

/// Shared buffer classification for adapter errors.
#[doc(hidden)]
pub fn adapter_error_is_buffer_error(error: &impl AdapterErrorParts) -> bool {
    error.adapter_error_kind() == AdapterErrorKind::Buffer
        || error
            .source_codec_error()
            .is_some_and(CodecError::is_buffer_error)
}
