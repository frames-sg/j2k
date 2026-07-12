// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{
    adapter_error_is_buffer_error, adapter_error_is_not_implemented, adapter_error_is_truncated,
    adapter_error_is_unsupported, AdapterErrorKind, AdapterErrorParts, BackendRequest, BufferError,
    CodecError,
};
use j2k_jpeg::adapter::{FastPacketError, JpegPlanCacheError};
use j2k_jpeg::{JpegEncodeError, JpegError};

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
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
    /// Backend-neutral JPEG fast-packet construction failed.
    #[error("backend-neutral JPEG fast-packet construction failed: {0}")]
    FastPacket(
        #[from]
        #[source]
        FastPacketError,
    ),
    /// Backend-neutral JPEG plan-cache ownership or allocation failed.
    #[error("backend-neutral JPEG plan-cache operation failed: {0}")]
    JpegPlanCache(
        #[from]
        #[source]
        JpegPlanCacheError,
    ),
    /// The shared CUDA JPEG packet-plan cache mutex was poisoned.
    #[error("shared CUDA JPEG packet-plan cache mutex is poisoned")]
    OwnedPacketCachePoisoned,
    /// The clone-shared synchronous owned-JPEG operation gate was poisoned.
    #[error("shared CUDA JPEG-host operation gate is poisoned")]
    JpegHostOperationPoisoned,
    /// The clone-shared lazy CUDA context or output-pool state was poisoned.
    #[error("shared CUDA JPEG runtime state is poisoned")]
    CudaSessionRuntimePoisoned,
    /// In-flight host-owner accounting can no longer prove an exact total.
    #[error("shared CUDA JPEG in-flight host-owner ledger is poisoned")]
    InFlightHostLedgerPoisoned,
    /// A CUDA operation and the following host-accounting verification both failed.
    #[error("CUDA operation failed ({primary}); host-accounting verification also failed ({accounting})")]
    OperationAndHostAccountingFailed {
        /// Primary operation failure.
        primary: Box<Error>,
        /// Host-accounting verification failure.
        accounting: Box<Error>,
    },
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
    /// A fixed-capacity, session-accounted batch received more results than reserved.
    #[error("batch capacity exceeded for {what}: reserved {capacity} results")]
    BatchCapacityExceeded {
        /// Number of result slots reserved before batch execution.
        capacity: usize,
        /// Logical batch result owner.
        what: &'static str,
    },
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
    #[error("CUDA runtime error: {source}")]
    CudaRuntime {
        /// Typed runtime failure, including nested completion or resource-release errors.
        #[source]
        source: j2k_cuda_runtime::CudaError,
    },
}

#[doc(hidden)]
impl AdapterErrorParts for Error {
    fn source_codec_error(&self) -> Option<&dyn CodecError> {
        match self {
            Self::Decode(inner) => Some(inner),
            Self::FastPacket(inner) => Some(inner),
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
            Self::Decode(_)
            | Self::FastPacket(_)
            | Self::JpegPlanCache(_)
            | Self::OwnedPacketCachePoisoned
            | Self::JpegHostOperationPoisoned
            | Self::CudaSessionRuntimePoisoned
            | Self::InFlightHostLedgerPoisoned
            | Self::OperationAndHostAccountingFailed { .. }
            | Self::BatchCapacityExceeded { .. }
            | Self::Encode(_)
            | Self::HostAllocationFailed { .. }
            | Self::HostAllocationTooLarge { .. } => AdapterErrorKind::Other,
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

#[cfg(test)]
mod tests {
    use super::Error;
    use j2k_core::CodecError;
    use j2k_jpeg::adapter::{FastPacketError, JpegPlanCacheError};
    use std::error::Error as _;

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
    fn fixed_batch_capacity_failure_is_not_an_allocator_or_buffer_error() {
        let error = Error::BatchCapacityExceeded {
            capacity: 4,
            what: "test batch",
        };
        assert!(!error.is_buffer_error());
        assert!(!error.is_unsupported());
        assert!(error.to_string().contains("reserved 4 results"));
    }

    #[test]
    fn fast_packet_error_preserves_source_and_codec_classification() {
        let unsupported = Error::FastPacket(FastPacketError::UnsupportedSampling);
        assert!(unsupported.is_unsupported());
        assert!(unsupported.source().is_some());

        let hard = Error::FastPacket(FastPacketError::MissingQuantTable { slot: 3 });
        assert!(!hard.is_unsupported());
        assert!(hard.source().is_some());
    }

    #[test]
    fn plan_cache_and_poison_errors_remain_distinct_operational_categories() {
        let cache = Error::JpegPlanCache(JpegPlanCacheError::Limit {
            what: "test input",
            requested: 17,
            cap: 16,
        });
        assert!(!cache.is_unsupported());
        assert!(cache.source().is_some());

        let poisoned = Error::OwnedPacketCachePoisoned;
        assert!(!poisoned.is_unsupported());
        assert!(poisoned.source().is_none());
    }
}
