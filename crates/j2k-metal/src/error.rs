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

#[derive(Debug, thiserror::Error)]
/// Errors returned by the Metal J2K backend.
pub enum Error {
    /// Error returned by the CPU or native J2K decoder.
    #[error(transparent)]
    Decode(#[from] J2kError),
    /// Output buffer validation failed.
    #[error(transparent)]
    Buffer(#[from] BufferError),
    /// The requested backend is unsupported by this crate.
    #[error("backend request {request:?} is not supported by j2k-metal")]
    UnsupportedBackend {
        /// Backend requested by the caller.
        request: BackendRequest,
    },
    /// A Metal-specific request is structurally unsupported.
    #[error("unsupported J2K Metal request: {reason}")]
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
    /// Metal kernel failure with structured retry classification.
    #[error("Metal kernel error: {message}")]
    MetalKernelRetryable {
        /// Kernel error message.
        message: String,
        /// Retry class assigned at the error construction site.
        retry_class: MetalKernelRetryClass,
    },
    /// Metal direct decode path could not handle a request and should fall back.
    #[error("Metal kernel error: {message}")]
    MetalDirectFallback {
        /// User-visible fallback message.
        message: String,
        /// Structured fallback reason assigned at the error construction site.
        reason: MetalDirectFallbackReason,
    },
    /// Shared Metal backend state was poisoned by a prior panic.
    #[error("Metal state `{state}` is poisoned")]
    MetalStatePoisoned {
        /// Name of the poisoned state.
        state: &'static str,
    },
}

/// Structured fallback class for Metal direct decode routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MetalDirectFallbackReason {
    /// Native direct-plan construction rejected the codestream or image shape.
    UnsupportedPlan,
    /// A prepared direct plan cannot be executed by the Metal runtime path.
    UnsupportedRuntimeInput,
}

/// Conservative retry class for Metal kernel failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MetalKernelRetryClass {
    /// Retry a resident classic J2K batch with a conservative capacity path.
    ResidentClassicBatch,
    /// Retry a resident HTJ2K batch with a conservative capacity path.
    ResidentHtBatch,
    /// Retry either resident classic or HTJ2K batches.
    ResidentClassicOrHtBatch,
}

impl MetalKernelRetryClass {
    fn applies_to(self, requested: Self) -> bool {
        self == requested
            || matches!(
                (self, requested),
                (
                    Self::ResidentClassicOrHtBatch,
                    Self::ResidentClassicBatch | Self::ResidentHtBatch
                )
            )
    }
}

impl Error {
    pub(crate) fn is_conservative_retry_candidate(&self, requested: MetalKernelRetryClass) -> bool {
        match self {
            Self::MetalKernelRetryable { retry_class, .. } => retry_class.applies_to(requested),
            _ => false,
        }
    }

    pub(crate) fn is_direct_fallback(&self) -> bool {
        matches!(self, Self::MetalDirectFallback { .. })
    }
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
            | Self::UnsupportedMetalRequest { .. }
            | Self::MetalUnavailable
            | Self::MetalDirectFallback { .. } => AdapterErrorKind::Unsupported,
            Self::Decode(_)
            | Self::MetalRuntime { .. }
            | Self::MetalKernel { .. }
            | Self::MetalKernelRetryable { .. }
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

pub(crate) fn native_decode_error(error: NativeDecodeError) -> Error {
    Error::Decode(native_decode_j2k_error(error))
}

pub(crate) fn adapter_backend_error(message: impl Into<String>) -> J2kError {
    J2kError::Backend(BackendError::new(BackendErrorKind::Other, message))
}

pub(crate) fn native_decode_j2k_error(error: NativeDecodeError) -> J2kError {
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
