// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{BufferError, CodecError, InputError, NotImplemented, Unsupported};
use j2k_native::{DecodeError as NativeDecodeError, DecodeErrorClass as NativeDecodeErrorClass};

/// Machine-readable class for backend failures surfaced by the facade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BackendErrorKind {
    /// Backend failure has no shared codec classification.
    Other,
    /// Backend failure indicates truncated input.
    Truncated,
    /// Backend failure indicates a planned but unimplemented feature.
    NotImplemented,
    /// Backend failure indicates unsupported input, options, or host capability.
    Unsupported,
    /// Backend failure indicates caller buffer sizing or layout problems.
    Buffer,
    /// Backend failure happened while validating a backend-produced codestream.
    Validation,
    /// Failure indicates an internal facade/cache invariant violation.
    InternalInvariant,
}

impl core::fmt::Display for BackendErrorKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Other => f.write_str("other"),
            Self::Truncated => f.write_str("truncated"),
            Self::NotImplemented => f.write_str("not-implemented"),
            Self::Unsupported => f.write_str("unsupported"),
            Self::Buffer => f.write_str("buffer"),
            Self::Validation => f.write_str("validation"),
            Self::InternalInvariant => f.write_str("internal-invariant"),
        }
    }
}

/// Structured backend failure details.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("backend failed ({kind}): {message}")]
pub struct BackendError {
    kind: BackendErrorKind,
    message: String,
}

impl BackendError {
    /// Construct a backend failure with an explicit class.
    pub fn new(kind: BackendErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Construct a native-backend failure.
    pub(crate) fn native(message: impl Into<String>) -> Self {
        Self::new(BackendErrorKind::Other, message)
    }

    /// Construct a truncated backend failure.
    pub fn truncated(message: impl Into<String>) -> Self {
        Self::new(BackendErrorKind::Truncated, message)
    }

    /// Construct an unimplemented backend failure.
    pub fn not_implemented(message: impl Into<String>) -> Self {
        Self::new(BackendErrorKind::NotImplemented, message)
    }

    /// Construct an unsupported backend failure.
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::new(BackendErrorKind::Unsupported, message)
    }

    /// Construct a backend buffer/layout failure.
    pub fn buffer(message: impl Into<String>) -> Self {
        Self::new(BackendErrorKind::Buffer, message)
    }

    /// Construct a backend-output validation failure.
    pub(crate) fn validation(message: impl Into<String>) -> Self {
        Self::new(BackendErrorKind::Validation, message)
    }

    /// Construct an internal invariant failure.
    pub(crate) fn internal_invariant(message: impl Into<String>) -> Self {
        Self::new(BackendErrorKind::InternalInvariant, message)
    }

    /// Return the machine-readable failure class.
    pub const fn kind(&self) -> BackendErrorKind {
        self.kind
    }

    /// Return the human-readable backend failure message.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl From<String> for BackendError {
    fn from(message: String) -> Self {
        Self::native(message)
    }
}

impl From<&str> for BackendError {
    fn from(message: &str) -> Self {
        Self::native(message)
    }
}

/// Error returned by JPEG 2000 inspect, decode, encode, and recode APIs.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum J2kError {
    /// Caller-owned buffers were too small or malformed.
    #[error(transparent)]
    Buffer(#[from] BufferError),

    /// Input was too short or truncated while parsing.
    #[error(transparent)]
    Input(#[from] InputError),

    /// Requested feature is planned but not implemented.
    #[error(transparent)]
    NotImplemented(#[from] NotImplemented),

    /// Requested input feature or option is unsupported.
    #[error(transparent)]
    Unsupported(#[from] Unsupported),

    /// Native backend, accelerator adapter, or backend-output validation failed.
    #[error(transparent)]
    Backend(#[from] BackendError),

    /// Caller-provided encode samples were malformed.
    #[error("invalid JPEG 2000 samples: {what}")]
    InvalidSamples {
        /// Description of the invalid sample condition.
        what: String,
    },

    /// Lossy rate-control search could not satisfy the requested target.
    #[error("JPEG 2000 lossy rate target unreachable: {target}, best {best}")]
    RateTargetUnreachable {
        /// Requested rate target.
        target: String,
        /// Best achievable result observed by the search.
        best: String,
    },

    /// Requested region lies outside image bounds.
    #[error("region ({x},{y} {w}x{h}) is outside image bounds {image_w}x{image_h}")]
    InvalidRegion {
        /// Region left coordinate.
        x: u32,
        /// Region top coordinate.
        y: u32,
        /// Region width.
        w: u32,
        /// Region height.
        h: u32,
        /// Image width.
        image_w: u32,
        /// Image height.
        image_h: u32,
    },

    /// JP2 box structure was invalid.
    #[error("invalid JP2 box at offset {offset}: {what}")]
    InvalidBox {
        /// Byte offset of the invalid box.
        offset: usize,
        /// Description of the invalid box condition.
        what: &'static str,
    },

    /// Required JP2 box was absent.
    #[error("missing required JP2 box {box_type}")]
    MissingRequiredBox {
        /// Missing box type.
        box_type: &'static str,
    },

    /// Codestream marker was invalid.
    #[error("invalid codestream marker FF{marker:02X} at offset {offset}")]
    InvalidMarker {
        /// Byte offset of the invalid marker.
        offset: usize,
        /// Marker byte following the `0xFF` prefix.
        marker: u8,
    },

    /// Required codestream marker was absent.
    #[error("missing required codestream marker {marker}")]
    MissingRequiredMarker {
        /// Missing marker name.
        marker: &'static str,
    },

    /// SIZ segment was invalid.
    #[error("invalid SIZ segment: {what}")]
    InvalidSiz {
        /// Description of the invalid SIZ condition.
        what: &'static str,
    },

    /// COD segment was invalid.
    #[error("invalid COD segment: {what}")]
    InvalidCod {
        /// Description of the invalid COD condition.
        what: &'static str,
    },

    /// Image dimensions overflowed a size computation.
    #[error("dimension overflow: {width}x{height}")]
    DimensionOverflow {
        /// Image width.
        width: u32,
        /// Image height.
        height: u32,
    },
}

impl J2kError {
    /// Construct a native-backend failure.
    pub(crate) fn backend(message: impl Into<String>) -> Self {
        Self::Backend(BackendError::native(message))
    }

    /// Construct a backend-output validation failure.
    pub(crate) fn validation_backend(message: impl Into<String>) -> Self {
        Self::Backend(BackendError::validation(message))
    }

    /// Construct an internal facade/cache invariant failure.
    pub(crate) fn internal_backend(message: impl Into<String>) -> Self {
        Self::Backend(BackendError::internal_invariant(message))
    }

    pub(crate) fn from_native_decode_error(error: NativeDecodeError) -> Self {
        Self::from_native_decode_error_with_context(error, "native JPEG 2000 backend failed")
    }

    pub(crate) fn from_native_decode_error_with_context(
        error: NativeDecodeError,
        context: &'static str,
    ) -> Self {
        match error.classify() {
            NativeDecodeErrorClass::InputTooShort { need, have } => {
                Self::Input(InputError::TooShort { need, have })
            }
            NativeDecodeErrorClass::InputTruncatedAt { offset, segment } => {
                Self::Input(InputError::TruncatedAt { offset, segment })
            }
            NativeDecodeErrorClass::Unsupported { what } => Self::Unsupported(Unsupported { what }),
            _ => Self::Backend(BackendError::native(format!("{context}: {error}"))),
        }
    }
}

#[doc(hidden)]
impl CodecError for J2kError {
    fn is_truncated(&self) -> bool {
        matches!(
            self,
            Self::Input(InputError::TooShort { .. } | InputError::TruncatedAt { .. })
        ) || matches!(
            self,
            Self::Backend(error) if error.kind() == BackendErrorKind::Truncated
        )
    }

    fn is_not_implemented(&self) -> bool {
        matches!(self, Self::NotImplemented(_))
            || matches!(
                self,
                Self::Backend(error) if error.kind() == BackendErrorKind::NotImplemented
            )
    }

    fn is_unsupported(&self) -> bool {
        matches!(self, Self::Unsupported(_))
            || matches!(
                self,
                Self::Backend(error) if error.kind() == BackendErrorKind::Unsupported
            )
    }

    fn is_buffer_error(&self) -> bool {
        matches!(self, Self::Buffer(_))
            || matches!(
                self,
                Self::Backend(error) if error.kind() == BackendErrorKind::Buffer
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use j2k_native::{DecodeError as NativeDecodeError, DecodingError as NativeDecodingError};

    #[test]
    fn native_unsupported_errors_keep_codec_classification() {
        let error = J2kError::from_native_decode_error(NativeDecodeError::Decoding(
            NativeDecodingError::UnsupportedFeature("test feature"),
        ));

        assert!(error.is_unsupported());
        assert!(!error.is_truncated());
        assert!(!error.is_not_implemented());
    }

    #[test]
    fn native_truncated_errors_keep_codec_classification() {
        let error = J2kError::from_native_decode_error(NativeDecodeError::Decoding(
            NativeDecodingError::UnexpectedEof,
        ));

        assert!(error.is_truncated());
        assert!(!error.is_unsupported());
    }

    #[test]
    fn validation_backend_errors_are_not_reclassified_as_unsupported() {
        let error = J2kError::validation_backend("roundtrip validation failed");

        assert!(!error.is_unsupported());
        assert!(!error.is_truncated());
        assert!(!error.is_not_implemented());
    }

    #[test]
    fn backend_error_kind_drives_codec_classification() {
        let unsupported = J2kError::Backend(BackendError::unsupported("device format"));
        let truncated = J2kError::Backend(BackendError::truncated("entropy data"));
        let not_implemented = J2kError::Backend(BackendError::not_implemented("JPX path"));
        let buffer = J2kError::Backend(BackendError::buffer("output pitch"));
        let other = J2kError::backend("kernel launch failed");

        assert!(unsupported.is_unsupported());
        assert!(truncated.is_truncated());
        assert!(not_implemented.is_not_implemented());
        assert!(buffer.is_buffer_error());
        assert!(!other.is_unsupported());
        assert!(!other.is_truncated());
        assert!(!other.is_not_implemented());
        assert!(!other.is_buffer_error());
    }
}
