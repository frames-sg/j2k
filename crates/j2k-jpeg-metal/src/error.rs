// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{
    adapter_error_is_buffer_error, adapter_error_is_not_implemented, adapter_error_is_truncated,
    adapter_error_is_unsupported, AdapterErrorKind, AdapterErrorParts, BackendRequest, BufferError,
    CodecError,
};
use j2k_jpeg::{JpegEncodeError, JpegError};

#[derive(Debug, thiserror::Error)]
/// Errors returned by the Metal JPEG backend.
pub enum Error {
    /// Error returned by the CPU JPEG parser or fallback decoder.
    #[error(transparent)]
    Decode(#[from] JpegError),
    /// Error returned while assembling a baseline JPEG encode result.
    #[error(transparent)]
    Encode(#[from] JpegEncodeError),
    /// Output buffer validation failed.
    #[error(transparent)]
    Buffer(#[from] BufferError),
    /// The requested backend is not supported by this crate.
    #[error("backend request {request:?} is not supported by j2k-jpeg-metal")]
    UnsupportedBackend {
        /// Backend requested by the caller.
        request: BackendRequest,
    },
    /// A Metal-specific request is structurally unsupported.
    #[error("unsupported JPEG Metal request: {reason}")]
    UnsupportedMetalRequest {
        /// Static reason describing the rejected request.
        reason: &'static str,
    },
    /// Metal is not available on the current host.
    #[error("Metal is unavailable on this host")]
    MetalUnavailable,
    /// Metal runtime creation or device setup failed.
    #[error("Metal runtime error: {message}")]
    MetalRuntime {
        /// Runtime error message.
        message: String,
    },
    /// Metal kernel launch, validation, or completion failed.
    #[error("Metal kernel error: {message}")]
    MetalKernel {
        /// Kernel error message.
        message: String,
    },
    /// Shared Metal backend state was poisoned by a prior panic.
    #[error("Metal state `{state}` is poisoned")]
    MetalStatePoisoned {
        /// Name of the poisoned state.
        state: &'static str,
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
            | Self::UnsupportedMetalRequest { .. }
            | Self::MetalUnavailable
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
            Self::Decode(_)
            | Self::Encode(_)
            | Self::MetalRuntime { .. }
            | Self::MetalKernel { .. }
            | Self::MetalStatePoisoned { .. } => AdapterErrorKind::Other,
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
