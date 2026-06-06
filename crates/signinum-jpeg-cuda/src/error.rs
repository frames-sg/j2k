// SPDX-License-Identifier: Apache-2.0

use signinum_core::{BackendRequest, BufferError, CodecError};
use signinum_jpeg::JpegError;

#[derive(Debug, thiserror::Error)]
/// Errors returned by the CUDA JPEG adapter.
pub enum Error {
    /// Error returned by the CPU JPEG parser or fallback decoder.
    #[error(transparent)]
    Decode(#[from] JpegError),
    /// Output buffer validation failed.
    #[error(transparent)]
    Buffer(#[from] BufferError),
    /// The requested backend is unsupported by this crate.
    #[error("backend request {request:?} is not supported by signinum-jpeg-cuda")]
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

impl CodecError for Error {
    fn is_truncated(&self) -> bool {
        matches!(self, Self::Decode(inner) if inner.is_truncated())
    }

    fn is_not_implemented(&self) -> bool {
        matches!(self, Self::Decode(inner) if inner.is_not_implemented())
    }

    fn is_unsupported(&self) -> bool {
        matches!(
            self,
            Self::UnsupportedBackend { .. }
                | Self::UnsupportedCudaRequest { .. }
                | Self::CudaUnavailable
        ) || matches!(self, Self::Decode(inner) if inner.is_unsupported())
    }

    fn is_buffer_error(&self) -> bool {
        matches!(self, Self::Buffer(_))
            || matches!(self, Self::Decode(inner) if inner.is_buffer_error())
    }
}
