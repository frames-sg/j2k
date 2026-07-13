// SPDX-License-Identifier: MIT OR Apache-2.0

mod native_source;

use j2k::J2kError;
#[cfg(target_os = "macos")]
use j2k::{BackendError, BackendErrorKind};
use j2k_core::{
    adapter_error_is_buffer_error, adapter_error_is_not_implemented, adapter_error_is_truncated,
    adapter_error_is_unsupported, AdapterErrorKind, AdapterErrorParts, BackendRequest,
    BatchInfrastructureError, BufferError, CodecError,
};
use j2k_metal_support::MetalSupportError;
#[cfg(any(test, target_os = "macos"))]
use j2k_native::{DecodeError as NativeDecodeError, EncodeError as NativeEncodeError};

pub use native_source::NativeBackendError;

#[derive(Debug, thiserror::Error)]
/// Errors returned by the Metal J2K backend.
pub enum Error {
    /// Error returned by the CPU or native J2K decoder.
    #[error(transparent)]
    Decode(#[from] J2kError),
    /// Native decoder failure produced while preparing Metal-resident work.
    #[error("{context}: {source}")]
    NativeDecode {
        /// Stable adapter operation context.
        context: &'static str,
        /// Concrete native decoder failure.
        #[source]
        source: NativeBackendError,
    },
    /// Native J2K encode helper failed after the Metal path produced inputs.
    #[error("native J2K encode error during {operation}: {source}")]
    NativeEncode {
        /// Stable Metal adapter operation that crossed into native encoding.
        operation: &'static str,
        /// Concrete native encode failure.
        #[source]
        source: NativeBackendError,
    },
    /// Output buffer validation failed.
    #[error(transparent)]
    Buffer(#[from] BufferError),
    /// CPU batch allocation, scheduling, or result collection failed independently of one tile.
    #[error("{0}")]
    BatchInfrastructure(
        #[from]
        #[source]
        BatchInfrastructureError,
    ),
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
    /// A shared Metal support operation failed with its typed source preserved.
    #[error("{message}")]
    MetalSupport {
        /// Existing adapter diagnostic, including its operation context.
        message: String,
        /// Typed shared Metal support failure.
        #[source]
        source: MetalSupportError,
    },
    /// Prepared-plan cache storage could not reserve required host memory.
    #[error("Metal kernel error: {context}: allocation failed: {source}")]
    PreparedPlanCacheAllocation {
        /// Cache operation that failed.
        context: &'static str,
        /// Original host reservation failure.
        #[source]
        source: std::collections::TryReserveError,
    },
    /// Prepared-plan cache bookkeeping violated an internal invariant.
    #[error("Metal kernel error: {context}: cache invariant failed: {reason}")]
    PreparedPlanCacheInvariant {
        /// Cache operation that failed.
        context: &'static str,
        /// Static invariant diagnostic from the cache owner.
        reason: &'static str,
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
    /// Internal Metal state contradicted its checked ownership/accounting ledger.
    #[error("Metal state `{state}` invariant failed: {reason}")]
    MetalStateInvariant {
        /// Name of the affected state owner.
        state: &'static str,
        /// Static invariant that was violated.
        reason: &'static str,
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
    #[cfg(target_os = "macos")]
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
    #[cfg(target_os = "macos")]
    pub(crate) fn is_conservative_retry_candidate(&self, requested: MetalKernelRetryClass) -> bool {
        match self {
            Self::MetalKernelRetryable { retry_class, .. } => retry_class.applies_to(requested),
            _ => false,
        }
    }

    #[cfg(target_os = "macos")]
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
            | Self::NativeDecode { .. }
            | Self::NativeEncode { .. }
            | Self::BatchInfrastructure(_)
            | Self::MetalRuntime { .. }
            | Self::MetalSupport { .. }
            | Self::PreparedPlanCacheAllocation { .. }
            | Self::PreparedPlanCacheInvariant { .. }
            | Self::MetalKernel { .. }
            | Self::MetalKernelRetryable { .. }
            | Self::MetalStatePoisoned { .. }
            | Self::MetalStateInvariant { .. } => AdapterErrorKind::Other,
        }
    }
}

#[doc(hidden)]
impl CodecError for Error {
    fn is_truncated(&self) -> bool {
        matches!(
            self,
            Self::NativeDecode { source, .. } if source.is_decode_truncated()
        ) || adapter_error_is_truncated(self)
    }

    fn is_not_implemented(&self) -> bool {
        adapter_error_is_not_implemented(self)
    }

    fn is_unsupported(&self) -> bool {
        matches!(
            self,
            Self::NativeDecode { source, .. } if source.is_unsupported()
        ) || matches!(
            self,
            Self::NativeEncode { source, .. } if source.is_unsupported()
        ) || adapter_error_is_unsupported(self)
    }

    fn is_buffer_error(&self) -> bool {
        adapter_error_is_buffer_error(self)
    }
}

#[cfg(any(test, target_os = "macos"))]
pub(crate) fn native_decode_error(error: NativeDecodeError) -> Error {
    Error::NativeDecode {
        context: "native JPEG 2000 backend failed",
        source: NativeBackendError::decode(error),
    }
}

#[cfg(any(test, target_os = "macos"))]
pub(crate) fn native_encode_error(operation: &'static str, source: NativeEncodeError) -> Error {
    Error::NativeEncode {
        operation,
        source: NativeBackendError::encode(source),
    }
}

#[cfg(any(test, target_os = "macos"))]
pub(crate) fn metal_kernel_support_error(
    message: impl Into<String>,
    source: MetalSupportError,
) -> Error {
    Error::MetalSupport {
        message: format!("Metal kernel error: {}", message.into()),
        source,
    }
}

#[cfg(any(test, target_os = "macos"))]
pub(crate) fn metal_runtime_support_error(source: &MetalSupportError) -> Error {
    if source.is_unavailable() {
        Error::MetalUnavailable
    } else {
        Error::MetalSupport {
            message: format!("Metal runtime error: {source}"),
            source: source.clone(),
        }
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn adapter_backend_error(message: impl Into<String>) -> J2kError {
    J2kError::Backend(BackendError::new(BackendErrorKind::Other, message))
}

#[cfg(test)]
mod tests {
    use j2k_core::CodecError;
    use j2k_metal_support::MetalSupportError;
    use j2k_native::{
        DecodeError as NativeDecodeError, DecodingError as NativeDecodingError,
        DirectPlanUnsupportedReason as NativeDirectPlanUnsupportedReason,
        EncodeError as NativeEncodeError,
    };

    use super::{
        metal_kernel_support_error, metal_runtime_support_error, native_decode_error,
        native_encode_error, Error, NativeBackendError,
    };

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
    fn native_encode_crossing_preserves_operation_and_concrete_source() {
        let error = native_encode_error(
            "classic Tier-1 token pack",
            NativeEncodeError::ArithmeticOverflow {
                what: "test token length",
            },
        );

        assert!(error.to_string().contains("classic Tier-1 token pack"));
        assert!(matches!(
            &error,
            Error::NativeEncode { operation, source }
                if *operation == "classic Tier-1 token pack"
                    && source == &NativeBackendError::encode(
                        NativeEncodeError::ArithmeticOverflow {
                            what: "test token length",
                        }
                    )
        ));
        let opaque = std::error::Error::source(&error).expect("opaque adapter source");
        assert!(opaque.downcast_ref::<NativeBackendError>().is_some());
        let concrete = opaque.source().expect("concrete native encode source");
        assert!(matches!(
            concrete.downcast_ref::<NativeEncodeError>(),
            Some(NativeEncodeError::ArithmeticOverflow {
                what: "test token length"
            })
        ));
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

    #[test]
    fn native_decode_resource_errors_preserve_typed_sources() {
        let sources = [
            NativeDecodeError::AllocationTooLarge {
                what: "Metal decode fixture",
                requested: 9,
                cap: 8,
            },
            NativeDecodeError::HostAllocationFailed {
                what: "Metal decode fixture",
                bytes: 7,
            },
        ];

        for source in sources {
            let error = native_decode_error(source);
            assert!(matches!(
                &error,
                Error::NativeDecode {
                    context: "native JPEG 2000 backend failed",
                    source: stored,
                } if stored == &NativeBackendError::decode(source)
            ));
            let opaque = core::error::Error::source(&error).expect("opaque adapter source");
            assert!(opaque.downcast_ref::<NativeBackendError>().is_some());
            let concrete = opaque.source().expect("concrete native decode source");
            assert_eq!(concrete.downcast_ref::<NativeDecodeError>(), Some(&source));
            assert!(!error.is_buffer_error());
            assert!(!error.is_unsupported());
        }
    }

    #[test]
    fn metal_support_error_keeps_display_source_and_other_classification() {
        let source = MetalSupportError::BufferAlignment {
            offset_bytes: 1,
            align: 4,
        };
        let error = metal_kernel_support_error(
            format!("J2K Metal status readback buffer access invalid: {source}"),
            source.clone(),
        );

        assert_eq!(
            error.to_string(),
            "Metal kernel error: J2K Metal status readback buffer access invalid: Metal buffer range offset 1 is not aligned to 4 bytes"
        );
        assert!(matches!(
            &error,
            Error::MetalSupport { source: stored, .. } if stored == &source
        ));
        let chained = std::error::Error::source(&error).expect("typed Metal support source");
        assert!(chained.downcast_ref::<MetalSupportError>().is_some());
        assert!(!error.is_unsupported());
        assert!(!error.is_buffer_error());
    }

    #[test]
    fn runtime_unavailability_keeps_existing_unsupported_route() {
        let error = metal_runtime_support_error(&MetalSupportError::MetalUnavailable);

        assert!(matches!(error, Error::MetalUnavailable));
        assert!(error.is_unsupported());
    }
}
