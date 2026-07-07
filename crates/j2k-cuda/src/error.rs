// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{BackendError, BackendErrorKind, J2kError};
use j2k_core::{
    adapter_error_is_buffer_error, adapter_error_is_not_implemented, adapter_error_is_truncated,
    adapter_error_is_unsupported, AdapterErrorKind, AdapterErrorParts, BackendRequest, BufferError,
    CodecError, InputError, Unsupported,
};
use j2k_native::{
    DecodeError as NativeDecodeError, DecodingError as NativeDecodingError,
    DirectPlanUnsupportedReason as NativeDirectPlanUnsupportedReason,
    FormatError as NativeFormatError, MarkerError as NativeMarkerError,
};

/// Error returned by the CUDA JPEG 2000 adapter.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// CPU JPEG 2000 decode failed.
    #[error(transparent)]
    Decode(#[from] J2kError),
    /// Caller-owned output buffers were invalid.
    #[error(transparent)]
    Buffer(#[from] BufferError),
    /// Backend request is unsupported by this adapter.
    #[error("backend request {request:?} is not supported by j2k-cuda")]
    UnsupportedBackend {
        /// Requested backend.
        request: BackendRequest,
    },
    /// CUDA request is unsupported by the strict CUDA adapter contract.
    #[error("unsupported CUDA request: {reason}")]
    UnsupportedCudaRequest {
        /// Human-readable rejection reason.
        reason: &'static str,
    },
    /// CUDA runtime or device is unavailable.
    #[error("CUDA is unavailable on this host")]
    CudaUnavailable,
    #[cfg(feature = "cuda-runtime")]
    /// CUDA runtime returned an error.
    #[error("CUDA runtime error: {message}")]
    CudaRuntime {
        /// Runtime error message.
        message: String,
    },
}

#[doc(hidden)]
impl AdapterErrorParts for Error {
    fn source_codec_error(&self) -> Option<&dyn CodecError> {
        match self {
            Self::Decode(inner) => Some(inner),
            _ => None,
        }
    }

    fn adapter_error_kind(&self) -> AdapterErrorKind {
        match self {
            Self::Buffer(_) => AdapterErrorKind::Buffer,
            Self::UnsupportedBackend { .. }
            | Self::UnsupportedCudaRequest { .. }
            | Self::CudaUnavailable => AdapterErrorKind::Unsupported,
            Self::Decode(_) => AdapterErrorKind::Other,
            #[cfg(feature = "cuda-runtime")]
            Self::CudaRuntime { .. } => AdapterErrorKind::Other,
        }
    }
}

#[doc(hidden)]
impl CodecError for Error {
    fn is_truncated(&self) -> bool {
        adapter_error_is_truncated(self)
    }

    fn is_not_implemented(&self) -> bool {
        adapter_error_is_not_implemented(self)
    }

    fn is_unsupported(&self) -> bool {
        adapter_error_is_unsupported(self)
    }

    fn is_buffer_error(&self) -> bool {
        adapter_error_is_buffer_error(self)
    }
}

pub(crate) fn native_decode_error(error: NativeDecodeError) -> Error {
    Error::Decode(native_decode_j2k_error(error))
}

fn adapter_backend_error(message: impl Into<String>) -> J2kError {
    J2kError::Backend(BackendError::new(BackendErrorKind::Other, message))
}

fn native_decode_j2k_error(error: NativeDecodeError) -> J2kError {
    match error {
        NativeDecodeError::Format(NativeFormatError::TooShort { need, have }) => {
            J2kError::Input(InputError::TooShort { need, have })
        }
        NativeDecodeError::Format(NativeFormatError::TruncatedAt { offset, segment }) => {
            J2kError::Input(InputError::TruncatedAt { offset, segment })
        }
        NativeDecodeError::Format(NativeFormatError::Unsupported) => {
            J2kError::Unsupported(Unsupported {
                what: "JP2 image format",
            })
        }
        NativeDecodeError::Marker(NativeMarkerError::Unsupported) => {
            J2kError::Unsupported(Unsupported {
                what: "JPEG 2000 marker",
            })
        }
        NativeDecodeError::Decoding(NativeDecodingError::DirectPlanUnsupported(reason)) => {
            J2kError::Unsupported(Unsupported {
                what: native_direct_plan_unsupported_what(reason),
            })
        }
        NativeDecodeError::Decoding(NativeDecodingError::UnsupportedFeature(what)) => {
            J2kError::Unsupported(Unsupported { what })
        }
        NativeDecodeError::Decoding(NativeDecodingError::UnexpectedEof) => {
            J2kError::Input(InputError::TruncatedAt {
                offset: 0,
                segment: "JPEG 2000 entropy data",
            })
        }
        error => adapter_backend_error(error.to_string()),
    }
}

fn native_direct_plan_unsupported_what(reason: NativeDirectPlanUnsupportedReason) -> &'static str {
    match reason {
        NativeDirectPlanUnsupportedReason::GrayscaleImageWithoutAlpha => {
            "direct grayscale plan only supports grayscale images without alpha"
        }
        NativeDirectPlanUnsupportedReason::GrayscaleSingleTileCodestream => {
            "direct grayscale plan only supports single-tile codestreams"
        }
        NativeDirectPlanUnsupportedReason::GrayscaleSingleComponentCodestream => {
            "direct grayscale plan only supports single-component codestreams"
        }
        NativeDirectPlanUnsupportedReason::ColorRgbImageWithoutAlpha => {
            "direct color plan only supports RGB images without alpha"
        }
        NativeDirectPlanUnsupportedReason::ColorSingleTileCodestream => {
            "direct color plan only supports single-tile codestreams"
        }
        NativeDirectPlanUnsupportedReason::ColorThreeComponentRgbCodestream => {
            "direct color plan only supports three-component RGB codestreams"
        }
        NativeDirectPlanUnsupportedReason::ComponentIndexOutOfRange => {
            "direct component plan index is out of range"
        }
        NativeDirectPlanUnsupportedReason::ComponentUnitSampled => {
            "direct component plan only supports unit-sampled components"
        }
        NativeDirectPlanUnsupportedReason::ComponentDecompositionIndexOutOfRange => {
            "direct component decomposition index is out of range"
        }
        _ => "direct JPEG 2000 plan is unsupported",
    }
}

#[cfg(test)]
mod tests {
    use j2k_core::CodecError;
    use j2k_native::{
        DecodeError as NativeDecodeError, DecodingError as NativeDecodingError,
        DirectPlanUnsupportedReason as NativeDirectPlanUnsupportedReason,
    };

    use super::native_decode_error;

    #[test]
    fn native_decode_unsupported_error_keeps_codec_classification() {
        let error = native_decode_error(NativeDecodeError::Decoding(
            NativeDecodingError::UnsupportedFeature("test feature"),
        ));

        assert!(error.is_unsupported());
        assert!(!error.is_truncated());
        assert!(!error.is_not_implemented());
    }

    #[test]
    fn native_decode_direct_plan_error_keeps_codec_classification() {
        let error = native_decode_error(NativeDecodeError::Decoding(
            NativeDecodingError::DirectPlanUnsupported(
                NativeDirectPlanUnsupportedReason::ColorSingleTileCodestream,
            ),
        ));

        assert!(error.is_unsupported());
        assert!(!error.is_truncated());
        assert!(!error.is_not_implemented());
    }

    #[test]
    fn native_decode_unexpected_eof_keeps_codec_classification() {
        let error = native_decode_error(NativeDecodeError::Decoding(
            NativeDecodingError::UnexpectedEof,
        ));

        assert!(error.is_truncated());
        assert!(!error.is_unsupported());
    }
}
