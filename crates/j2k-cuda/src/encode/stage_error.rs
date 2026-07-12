// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{J2kEncodeStageError, J2kEncodeStageResult};

/// Result used throughout the CUDA implementation of the shared encode-stage SPI.
pub(super) type CudaStageResult<T> = J2kEncodeStageResult<T>;

pub(super) const fn arithmetic_overflow(what: &'static str) -> J2kEncodeStageError {
    J2kEncodeStageError::arithmetic_overflow(what)
}

#[cfg(any(feature = "cuda-runtime", test))]
pub(super) const fn internal_invariant(what: &'static str) -> J2kEncodeStageError {
    J2kEncodeStageError::internal_invariant(what)
}

pub(super) fn adapter_error(operation: &'static str, error: crate::Error) -> J2kEncodeStageError {
    match error {
        crate::Error::HostAllocationFailed { bytes, what } => {
            J2kEncodeStageError::host_allocation_failed(what, bytes)
        }
        crate::Error::HostAllocationTooLarge {
            requested,
            cap,
            what,
        } => J2kEncodeStageError::memory_cap_exceeded(what, requested, cap),
        source => J2kEncodeStageError::backend("cuda", operation, source),
    }
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn runtime_error(
    operation: &'static str,
    error: j2k_cuda_runtime::CudaError,
) -> J2kEncodeStageError {
    match error {
        j2k_cuda_runtime::CudaError::HostAllocationFailed { bytes } => {
            J2kEncodeStageError::host_allocation_failed(operation, bytes)
        }
        j2k_cuda_runtime::CudaError::HostAllocationTooLarge {
            requested,
            cap,
            what,
        } => J2kEncodeStageError::memory_cap_exceeded(what, requested, cap),
        source => J2kEncodeStageError::backend("cuda", operation, source),
    }
}

#[cfg(test)]
mod tests {
    use core::error::Error as _;

    use j2k::{J2kEncodeStageError, J2kEncodeStageErrorKind};

    use super::{adapter_error, arithmetic_overflow, internal_invariant};

    #[test]
    fn adapter_allocation_preserves_typed_allocation_details() {
        let error = adapter_error(
            "test allocation",
            crate::Error::HostAllocationFailed {
                bytes: 4096,
                what: "CUDA test staging",
            },
        );

        assert_eq!(error.kind(), J2kEncodeStageErrorKind::HostAllocationFailed);
        assert!(matches!(
            error,
            J2kEncodeStageError::HostAllocationFailed {
                bytes: 4096,
                what: "CUDA test staging"
            }
        ));
    }

    #[test]
    fn adapter_cap_failure_preserves_phase_budget_details() {
        let error = adapter_error(
            "test allocation cap",
            crate::Error::HostAllocationTooLarge {
                requested: 17,
                cap: 16,
                what: "CUDA adapter test phase",
            },
        );

        assert!(matches!(
            error,
            J2kEncodeStageError::MemoryCapExceeded {
                requested: 17,
                cap: 16,
                what: "CUDA adapter test phase"
            }
        ));
    }

    #[test]
    fn adapter_backend_failure_keeps_concrete_source() {
        let error = adapter_error("test backend", crate::Error::CudaUnavailable);

        assert_eq!(error.kind(), J2kEncodeStageErrorKind::Backend);
        assert!(error
            .source()
            .and_then(|source| source.downcast_ref::<crate::Error>())
            .is_some());
        assert!(matches!(
            error,
            J2kEncodeStageError::Backend {
                backend: "cuda",
                operation: "test backend",
                ..
            }
        ));
    }

    #[test]
    fn local_categories_are_explicit() {
        assert_eq!(
            arithmetic_overflow("test overflow").kind(),
            J2kEncodeStageErrorKind::ArithmeticOverflow
        );
        assert_eq!(
            internal_invariant("test invariant").kind(),
            J2kEncodeStageErrorKind::InternalInvariant
        );
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn runtime_backend_failure_keeps_concrete_source() {
        let error = super::runtime_error(
            "cuTest",
            j2k_cuda_runtime::CudaError::InternalInvariant {
                what: "test runtime invariant",
            },
        );

        assert_eq!(error.kind(), J2kEncodeStageErrorKind::Backend);
        let source = error.source().expect("CUDA source must remain reachable");
        assert!(source
            .downcast_ref::<j2k_cuda_runtime::CudaError>()
            .is_some());
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn runtime_cap_failure_keeps_budget_details() {
        let error = super::runtime_error(
            "test staging",
            j2k_cuda_runtime::CudaError::HostAllocationTooLarge {
                requested: 17,
                cap: 16,
                what: "CUDA test phase",
            },
        );

        assert!(matches!(
            error,
            J2kEncodeStageError::MemoryCapExceeded {
                requested: 17,
                cap: 16,
                what: "CUDA test phase"
            }
        ));
    }
}
