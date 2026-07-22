// SPDX-License-Identifier: MIT OR Apache-2.0

mod native_source;

use j2k::J2kError;
use j2k_core::{
    adapter_error_is_buffer_error, adapter_error_is_not_implemented, adapter_error_is_truncated,
    adapter_error_is_unsupported, AdapterErrorKind, AdapterErrorParts, BackendRequest, BufferError,
    CodecError,
};
use j2k_native::DecodeError as NativeDecodeError;

pub use native_source::NativeBackendError;

/// Error returned by the CUDA JPEG 2000 adapter.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// CPU JPEG 2000 decode failed.
    #[error(transparent)]
    Decode(#[from] J2kError),
    /// Native decoder failure produced while preparing CUDA-resident work.
    #[error("{context}: {source}")]
    NativeDecode {
        /// Stable adapter operation context.
        context: &'static str,
        /// Concrete native decoder failure.
        #[source]
        source: NativeBackendError,
    },
    /// Caller-owned output buffers were invalid.
    #[error(transparent)]
    Buffer(#[from] BufferError),
    /// A host-side allocation needed by the adapter could not be reserved.
    #[error("host allocation failed for {what}: {bytes} bytes")]
    HostAllocationFailed {
        /// Requested allocation size in bytes.
        bytes: usize,
        /// Logical allocation purpose.
        what: &'static str,
    },
    /// Allocator-reported host capacity exceeds the codec phase budget.
    #[error(
        "host allocation capacity for {what} is too large: requested {requested} bytes, cap {cap} bytes"
    )]
    HostAllocationTooLarge {
        /// Aggregate allocator-reported byte capacity.
        requested: usize,
        /// Maximum permitted simultaneously live host bytes.
        cap: usize,
        /// Logical phase or owner graph.
        what: &'static str,
    },
    /// Bounded HTJ2K GPU job planning rejected one source job.
    #[error(transparent)]
    HtJobChunkPlan(#[from] j2k_core::HtGpuJobChunkPlanError),
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
    #[error("CUDA runtime error: {source}")]
    CudaRuntime {
        /// Typed runtime failure, including nested completion or resource-release errors.
        #[source]
        source: j2k_cuda_runtime::CudaError,
    },
    #[cfg(feature = "cuda-runtime")]
    /// A classic JPEG 2000 or HTJ2K tier-1 descriptor failed during GPU execution.
    #[error(
        "CUDA tier-1 source {source_index} job {original_job_index} failed during device execution: {source}"
    )]
    CudaTier1JobFailed {
        /// Original caller input index that owns the failed job.
        source_index: usize,
        /// Stable job index before pass-bucket ordering and chunk splitting.
        original_job_index: usize,
        /// Indexed CUDA kernel failure.
        #[source]
        source: j2k_cuda_runtime::CudaError,
    },
    #[cfg(feature = "cuda-runtime")]
    /// A CUDA operation failed and the required resource cleanup also failed.
    #[error("CUDA operation failed ({primary}); CUDA cleanup also failed ({cleanup})")]
    CudaCleanupFailed {
        /// Error returned by the original operation.
        primary: Box<Error>,
        /// Error returned while synchronizing or releasing queued resources.
        cleanup: Box<Error>,
    },
}

impl Error {
    /// Whether this error can leave submitted CUDA work referencing an
    /// external allocation whose completion was not established.
    #[cfg(feature = "cuda-runtime")]
    #[doc(hidden)]
    pub fn completion_is_uncertain(&self) -> bool {
        match self {
            Self::CudaRuntime { source } | Self::CudaTier1JobFailed { source, .. } => {
                source.completion_is_uncertain()
            }
            Self::CudaCleanupFailed { primary, cleanup } => {
                primary.completion_is_uncertain() || cleanup.completion_is_uncertain()
            }
            _ => false,
        }
    }

    /// Whether the persistent CUDA session cannot safely execute later groups.
    #[doc(hidden)]
    #[must_use]
    pub fn session_is_unusable(&self) -> bool {
        match self {
            Self::CudaUnavailable => true,
            #[cfg(feature = "cuda-runtime")]
            Self::CudaRuntime { source } | Self::CudaTier1JobFailed { source, .. } => {
                source.session_is_unusable()
            }
            #[cfg(feature = "cuda-runtime")]
            Self::CudaCleanupFailed { primary, cleanup } => {
                primary.session_is_unusable() || cleanup.session_is_unusable()
            }
            _ => false,
        }
    }
}

#[cfg(feature = "cuda-runtime")]
pub(crate) fn combine_cuda_cleanup_errors(primary_error: Error, cleanup_error: Error) -> Error {
    Error::CudaCleanupFailed {
        primary: Box::new(primary_error),
        cleanup: Box::new(cleanup_error),
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
            | Self::UnsupportedCudaRequest { .. }
            | Self::CudaUnavailable => AdapterErrorKind::Unsupported,
            Self::Decode(_)
            | Self::NativeDecode { .. }
            | Self::HostAllocationFailed { .. }
            | Self::HostAllocationTooLarge { .. }
            | Self::HtJobChunkPlan(_) => AdapterErrorKind::Other,
            #[cfg(feature = "cuda-runtime")]
            Self::CudaRuntime { .. }
            | Self::CudaTier1JobFailed { .. }
            | Self::CudaCleanupFailed { .. } => AdapterErrorKind::Other,
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
        ) || adapter_error_is_unsupported(self)
    }

    fn is_buffer_error(&self) -> bool {
        adapter_error_is_buffer_error(self)
    }
}

#[cfg_attr(
    not(any(test, feature = "cuda-runtime")),
    expect(
        dead_code,
        reason = "native decode translation is used by CUDA decode and its tests"
    )
)]
pub(crate) fn native_decode_error(error: NativeDecodeError) -> Error {
    Error::NativeDecode {
        context: "native JPEG 2000 backend failed",
        source: NativeBackendError::decode(error),
    }
}

#[cfg(test)]
mod tests {
    use j2k_core::CodecError;
    use j2k_native::{
        DecodeError as NativeDecodeError, DecodingError as NativeDecodingError,
        DirectPlanUnsupportedReason as NativeDirectPlanUnsupportedReason,
    };

    #[cfg(feature = "cuda-runtime")]
    use super::combine_cuda_cleanup_errors;
    use super::{native_decode_error, Error, NativeBackendError};
    #[cfg(feature = "cuda-runtime")]
    use j2k_cuda_runtime::CudaError;

    #[cfg(feature = "cuda-runtime")]
    fn runtime_error(message: &str) -> Error {
        Error::CudaRuntime {
            source: CudaError::StatePoisoned {
                message: message.to_string(),
            },
        }
    }

    #[test]
    fn host_allocation_failure_is_an_operational_error_not_buffer_misuse() {
        let error = Error::HostAllocationFailed {
            bytes: 4096,
            what: "test staging",
        };
        assert!(!error.is_buffer_error());
        assert!(!error.is_unsupported());
        assert!(error.to_string().contains("4096"));
    }

    #[test]
    fn host_capacity_failure_preserves_actual_phase_budget() {
        let error = Error::HostAllocationTooLarge {
            requested: 17,
            cap: 16,
            what: "test phase",
        };
        assert!(!error.is_buffer_error());
        assert!(!error.is_unsupported());
        assert!(error.to_string().contains("17"));
        assert!(error.to_string().contains("16"));
    }

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

    #[test]
    fn native_decode_resource_errors_preserve_typed_sources() {
        let sources = [
            NativeDecodeError::AllocationTooLarge {
                what: "CUDA decode fixture",
                requested: 9,
                cap: 8,
            },
            NativeDecodeError::HostAllocationFailed {
                what: "CUDA decode fixture",
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

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_cleanup_failure_preserves_both_runtime_diagnostics() {
        let combined = combine_cuda_cleanup_errors(
            runtime_error("primary launch failure"),
            runtime_error("follow-up synchronization failure"),
        );

        assert!(matches!(&combined, Error::CudaCleanupFailed { .. }));
        let rendered = combined.to_string();
        assert!(rendered.contains("primary launch failure"));
        assert!(rendered.contains("follow-up synchronization failure"));
        assert!(!combined.is_unsupported());
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn cuda_cleanup_failure_preserves_non_runtime_primary_and_blocks_fallback() {
        let combined = combine_cuda_cleanup_errors(
            Error::UnsupportedCudaRequest {
                reason: "invalid prepared store",
            },
            runtime_error("synchronization failure"),
        );

        assert!(matches!(&combined, Error::CudaCleanupFailed { .. }));
        let rendered = combined.to_string();
        assert!(rendered.contains("invalid prepared store"));
        assert!(rendered.contains("synchronization failure"));
        assert!(!combined.is_unsupported());
    }
}
