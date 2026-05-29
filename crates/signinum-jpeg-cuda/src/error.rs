// SPDX-License-Identifier: Apache-2.0

use signinum_core::{BackendRequest, BufferError, CodecError};
use signinum_jpeg::JpegError;

/// Error returned by the CUDA JPEG adapter.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// CPU JPEG decode failed.
    #[error(transparent)]
    Decode(#[from] JpegError),
    /// Caller-owned output buffers were invalid.
    #[error(transparent)]
    Buffer(#[from] BufferError),
    /// Backend request is unsupported by this adapter.
    #[error("backend request {request:?} is not supported by signinum-jpeg-cuda")]
    UnsupportedBackend {
        /// Requested backend.
        request: BackendRequest,
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
            Self::UnsupportedBackend { .. } | Self::CudaUnavailable
        ) || matches!(self, Self::Decode(inner) if inner.is_unsupported())
    }

    fn is_buffer_error(&self) -> bool {
        matches!(self, Self::Buffer(_))
            || matches!(self, Self::Decode(inner) if inner.is_buffer_error())
    }
}
