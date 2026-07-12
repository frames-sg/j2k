// SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;

use j2k_transcode::TranscodeStageError;

/// Stable message returned when the CUDA runtime is unavailable (feature not
/// compiled, no device, or the transcode kernels were not built).
pub const CUDA_UNAVAILABLE: &str = "CUDA is unavailable on this host";

/// Error returned by the CUDA transcode accelerator.
#[derive(Debug)]
pub enum CudaTranscodeError {
    /// CUDA is unavailable on this host or the kernels were not built.
    CudaUnavailable,
    /// The request is outside the current CUDA implementation.
    UnsupportedJob(&'static str),
    /// A validated CUDA result violated an internal kernel contract.
    Kernel(&'static str),
    /// A codec-owned host allocation exceeds the transcode safety limit.
    HostAllocationTooLarge {
        /// Requested byte count, saturated when size arithmetic overflowed.
        requested: usize,
        /// Maximum permitted byte count.
        cap: usize,
        /// Logical allocation purpose.
        what: &'static str,
    },
    /// A bounded codec-owned host allocation could not be reserved.
    HostAllocationFailed {
        /// Requested byte count.
        requested: usize,
        /// Logical allocation purpose.
        what: &'static str,
    },
    /// CUDA runtime or kernel execution failed with the backend diagnostic
    /// retained as an error source.
    Runtime(CudaRuntimeFailure),
}

impl CudaTranscodeError {
    /// Whether Auto mode may recover from this error by using the scalar
    /// fallback (`Ok(None)`). Hard kernel and allocation failures propagate.
    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn is_recoverable(&self) -> bool {
        matches!(self, Self::CudaUnavailable | Self::UnsupportedJob(_))
            || matches!(self, Self::Runtime(failure) if failure.is_unavailable())
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn runtime(operation: &'static str, source: j2k_cuda_runtime::CudaError) -> Self {
        let unavailable = source.is_unavailable();
        Self::Runtime(CudaRuntimeFailure::new(operation, unavailable, source))
    }
}

/// Diagnostic retained when the CUDA runtime rejects a transcode operation.
///
/// The runtime dependency is optional, so this feature-independent wrapper
/// retains a boxed concrete source without changing the public error shape
/// between feature configurations.
#[derive(Debug)]
pub struct CudaRuntimeFailure {
    operation: &'static str,
    unavailable: bool,
    source: Box<dyn std::error::Error + Send + Sync + 'static>,
}

impl CudaRuntimeFailure {
    #[cfg(any(feature = "cuda-runtime", test))]
    fn new(
        operation: &'static str,
        unavailable: bool,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self {
            operation,
            unavailable,
            source: Box::new(source),
        }
    }

    /// Logical operation that failed.
    #[must_use]
    pub const fn operation(&self) -> &'static str {
        self.operation
    }

    /// Whether the CUDA runtime classified this failure as device/driver
    /// unavailability rather than an execution failure.
    #[must_use]
    pub const fn is_unavailable(&self) -> bool {
        self.unavailable
    }
}

impl fmt::Display for CudaRuntimeFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.operation, self.source)
    }
}

impl fmt::Display for CudaTranscodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CudaUnavailable => f.write_str(CUDA_UNAVAILABLE),
            Self::UnsupportedJob(reason) | Self::Kernel(reason) => f.write_str(reason),
            Self::HostAllocationTooLarge {
                requested,
                cap,
                what,
            } => write!(
                f,
                "CUDA transcode host allocation for {what} is too large: requested {requested} bytes, cap {cap} bytes"
            ),
            Self::HostAllocationFailed { requested, what } => write!(
                f,
                "CUDA transcode host allocation failed for {what}: {requested} bytes"
            ),
            Self::Runtime(failure) => failure.fmt(f),
        }
    }
}

impl From<CudaTranscodeError> for TranscodeStageError {
    fn from(error: CudaTranscodeError) -> Self {
        match error {
            CudaTranscodeError::CudaUnavailable => Self::DeviceUnavailable,
            CudaTranscodeError::UnsupportedJob(reason) => Self::Unsupported(reason),
            CudaTranscodeError::Kernel(reason) => Self::backend(
                "cuda",
                "validate CUDA transcode result",
                CudaTranscodeError::Kernel(reason),
            ),
            CudaTranscodeError::HostAllocationTooLarge { requested, cap, .. } => {
                Self::MemoryCapExceeded { requested, cap }
            }
            CudaTranscodeError::HostAllocationFailed { requested, .. } => {
                Self::HostAllocationFailed { bytes: requested }
            }
            CudaTranscodeError::Runtime(failure) => {
                let operation = failure.operation();
                Self::backend("cuda", operation, CudaTranscodeError::Runtime(failure))
            }
        }
    }
}

impl std::error::Error for CudaRuntimeFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.source.as_ref())
    }
}

impl std::error::Error for CudaTranscodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Runtime(failure) => Some(failure),
            Self::CudaUnavailable
            | Self::UnsupportedJob(_)
            | Self::Kernel(_)
            | Self::HostAllocationTooLarge { .. }
            | Self::HostAllocationFailed { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CudaRuntimeFailure, CudaTranscodeError};
    use core::fmt;
    use j2k_transcode::TranscodeStageError;

    #[derive(Debug)]
    struct TestRuntimeError;

    impl fmt::Display for TestRuntimeError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("driver rejected launch")
        }
    }

    impl std::error::Error for TestRuntimeError {}

    #[test]
    fn runtime_failure_retains_operation_detail_and_error_source() {
        let error = CudaTranscodeError::Runtime(CudaRuntimeFailure::new(
            "CUDA test dispatch",
            false,
            TestRuntimeError,
        ));
        assert_eq!(
            error.to_string(),
            "CUDA test dispatch: driver rejected launch"
        );
        let stage = TranscodeStageError::from(error);
        let TranscodeStageError::Backend {
            backend,
            operation,
            source,
        } = &stage
        else {
            panic!("runtime execution failures must remain backend failures");
        };
        assert_eq!(*backend, "cuda");
        assert_eq!(*operation, "CUDA test dispatch");
        let cuda_error = source
            .downcast_ref::<CudaTranscodeError>()
            .expect("stage source must retain the CUDA adapter error");
        let runtime_failure = std::error::Error::source(cuda_error)
            .expect("CUDA adapter error must retain its runtime wrapper");
        let concrete = std::error::Error::source(runtime_failure)
            .expect("runtime wrapper must retain the concrete runtime source");
        assert!(concrete.downcast_ref::<TestRuntimeError>().is_some());
    }

    #[test]
    fn allocation_failures_preserve_typed_stage_classification() {
        assert!(matches!(
            TranscodeStageError::from(CudaTranscodeError::HostAllocationTooLarge {
                requested: 17,
                cap: 16,
                what: "test",
            }),
            TranscodeStageError::MemoryCapExceeded {
                requested: 17,
                cap: 16,
            }
        ));
        assert!(matches!(
            TranscodeStageError::from(CudaTranscodeError::HostAllocationFailed {
                requested: 32,
                what: "test",
            }),
            TranscodeStageError::HostAllocationFailed { bytes: 32 }
        ));
    }

    #[test]
    fn kernel_contract_failure_is_a_concrete_backend_source() {
        let stage = TranscodeStageError::from(CudaTranscodeError::Kernel("bad kernel output"));
        let TranscodeStageError::Backend {
            backend,
            operation,
            source,
        } = &stage
        else {
            panic!("kernel contract failures must remain backend failures");
        };
        assert_eq!(*backend, "cuda");
        assert_eq!(*operation, "validate CUDA transcode result");
        assert!(matches!(
            source.downcast_ref::<CudaTranscodeError>(),
            Some(CudaTranscodeError::Kernel("bad kernel output"))
        ));
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn unavailable_runtime_failure_retains_diagnostic_and_allows_auto_recovery() {
        let error = CudaTranscodeError::runtime(
            "CUDA context initialization",
            j2k_cuda_runtime::CudaError::Unavailable {
                message: "no compatible device".to_string(),
            },
        );

        assert!(error.is_recoverable());
        let CudaTranscodeError::Runtime(failure) = error else {
            panic!("runtime constructor must return the runtime variant");
        };
        assert!(failure.is_unavailable());
        assert!(matches!(
            std::error::Error::source(&failure)
                .and_then(|source| source.downcast_ref::<j2k_cuda_runtime::CudaError>()),
            Some(j2k_cuda_runtime::CudaError::Unavailable { message })
                if message == "no compatible device"
        ));
    }
}
