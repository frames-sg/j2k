// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{
    adapter_error_is_buffer_error, adapter_error_is_not_implemented, adapter_error_is_truncated,
    adapter_error_is_unsupported, AdapterErrorKind, AdapterErrorParts, BackendRequest, BufferError,
    CodecError,
};
use j2k_jpeg::{JpegEncodeError, JpegError};

#[derive(Debug, thiserror::Error)]
/// Errors returned by the CUDA JPEG adapter.
pub enum Error {
    /// Error returned by the CPU JPEG parser or fallback decoder.
    #[error(transparent)]
    Decode(#[from] JpegError),
    /// Error returned by JPEG baseline encode setup or frame assembly.
    #[error(transparent)]
    Encode(#[from] JpegEncodeError),
    /// Output buffer validation failed.
    #[error(transparent)]
    Buffer(#[from] BufferError),
    /// The requested backend is unsupported by this crate.
    #[error("backend request {request:?} is not supported by j2k-jpeg-cuda")]
    UnsupportedBackend {
        /// Backend requested by the caller.
        request: BackendRequest,
    },
    /// CUDA request is unsupported by the strict CUDA adapter contract.
    #[error("unsupported CUDA request: {reason}")]
    UnsupportedCudaRequest {
        /// Human-readable rejection reason.
        reason: &'static str,
    },
    /// CUDA is not available on the current host.
    #[error("CUDA is unavailable on this host")]
    CudaUnavailable,
    #[cfg(feature = "cuda-runtime")]
    /// CUDA runtime operation failed.
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
            Self::UnsupportedBackend { .. }
            | Self::UnsupportedCudaRequest { .. }
            | Self::CudaUnavailable
            | Self::Encode(
                JpegEncodeError::UnsupportedBackend { .. }
                | JpegEncodeError::IncompatibleSubsampling { .. },
            ) => AdapterErrorKind::Unsupported,
            Self::Buffer(_)
            | Self::Encode(
                JpegEncodeError::SampleLength { .. }
                | JpegEncodeError::EmptyDimensions
                | JpegEncodeError::DimensionsTooLarge { .. },
            ) => AdapterErrorKind::Buffer,
            Self::Decode(_) | Self::Encode(_) => AdapterErrorKind::Other,
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
