// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{BackendError, BackendErrorKind, J2kError};
use j2k_core::{
    adapter_error_is_buffer_error, adapter_error_is_not_implemented, adapter_error_is_truncated,
    adapter_error_is_unsupported, AdapterErrorKind, AdapterErrorParts, BackendRequest, BufferError,
    CodecError, InputError, Unsupported,
};
use j2k_native::{DecodeError as NativeDecodeError, DecodeErrorClass as NativeDecodeErrorClass};

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

#[cfg_attr(not(any(test, feature = "cuda-runtime")), allow(dead_code))]
pub(crate) fn native_decode_error(error: NativeDecodeError) -> Error {
    Error::Decode(native_decode_j2k_error(error))
}

#[cfg_attr(not(any(test, feature = "cuda-runtime")), allow(dead_code))]
fn adapter_backend_error(message: impl Into<String>) -> J2kError {
    J2kError::Backend(BackendError::new(BackendErrorKind::Other, message))
}

#[cfg_attr(not(any(test, feature = "cuda-runtime")), allow(dead_code))]
fn native_decode_j2k_error(error: NativeDecodeError) -> J2kError {
    match error.classify() {
        NativeDecodeErrorClass::InputTooShort { need, have } => {
            J2kError::Input(InputError::TooShort { need, have })
        }
        NativeDecodeErrorClass::InputTruncatedAt { offset, segment } => {
            J2kError::Input(InputError::TruncatedAt { offset, segment })
        }
        NativeDecodeErrorClass::Unsupported { what } => J2kError::Unsupported(Unsupported { what }),
        _ => adapter_backend_error(error.to_string()),
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
