// SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;
use j2k_metal_support::MetalSupportError;
use j2k_transcode::TranscodeStageError;

use crate::{weights::SparseWeightRowsError, METAL_UNAVAILABLE};

/// Error returned by the Metal transcode accelerator.
#[derive(Debug)]
pub enum MetalTranscodeError {
    /// Metal is unavailable on this host or target.
    MetalUnavailable,
    /// The request is outside the current Metal implementation.
    UnsupportedJob(&'static str),
    /// Metal runtime creation or device execution failed with its diagnostic retained.
    Runtime(MetalRuntimeFailure),
    /// A shared Metal resource operation failed with its typed source retained.
    MetalSupport {
        /// Logical Metal operation that failed.
        operation: &'static str,
        /// Typed shared-support failure.
        source: MetalSupportError,
    },
    /// A validated Metal result violated an internal kernel contract.
    Kernel(&'static str),
    /// A codec-owned host allocation exceeds the transcode safety limit.
    HostAllocationTooLarge {
        /// Requested byte count, saturated on overflow.
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
    /// A codec-owned Metal allocation exceeds the shared transcode safety limit.
    DeviceAllocationTooLarge {
        /// Requested byte count, saturated on overflow.
        requested: usize,
        /// Maximum permitted byte count.
        cap: usize,
        /// Logical allocation purpose.
        what: &'static str,
    },
    /// Metal returned no buffer for a bounded device allocation.
    DeviceAllocationFailed {
        /// Requested byte count.
        requested: usize,
        /// Logical allocation purpose.
        what: &'static str,
    },
}

impl MetalTranscodeError {
    /// Whether Auto mode may recover from this error by using scalar fallback.
    pub(crate) const fn is_recoverable(&self) -> bool {
        matches!(self, Self::MetalUnavailable | Self::UnsupportedJob(_))
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn runtime(
        operation: &'static str,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Runtime(MetalRuntimeFailure {
            operation,
            source: Box::new(source),
        })
    }

    pub(crate) const fn support(operation: &'static str, source: MetalSupportError) -> Self {
        Self::MetalSupport { operation, source }
    }
}

/// Diagnostic retained when the Metal runtime rejects a transcode operation.
#[derive(Debug)]
pub struct MetalRuntimeFailure {
    operation: &'static str,
    source: Box<dyn std::error::Error + Send + Sync + 'static>,
}

impl MetalRuntimeFailure {
    /// Logical Metal operation that failed.
    #[must_use]
    pub const fn operation(&self) -> &'static str {
        self.operation
    }

    /// Concrete runtime source retained by this adapter boundary.
    #[must_use]
    pub fn source_error(&self) -> &(dyn std::error::Error + Send + Sync + 'static) {
        self.source.as_ref()
    }
}

impl fmt::Display for MetalRuntimeFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.operation, self.source)
    }
}

impl fmt::Display for MetalTranscodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MetalUnavailable => f.write_str(METAL_UNAVAILABLE),
            Self::UnsupportedJob(reason) | Self::Kernel(reason) => f.write_str(reason),
            Self::Runtime(failure) => failure.fmt(f),
            Self::MetalSupport { operation, source } => {
                write!(f, "{operation}: {source}")
            }
            Self::HostAllocationTooLarge {
                requested,
                cap,
                what,
            } => write!(
                f,
                "Metal transcode host allocation for {what} is too large: requested {requested} bytes, cap {cap} bytes"
            ),
            Self::HostAllocationFailed { requested, what } => write!(
                f,
                "Metal transcode host allocation failed for {what}: {requested} bytes"
            ),
            Self::DeviceAllocationTooLarge {
                requested,
                cap,
                what,
            } => write!(
                f,
                "Metal transcode device allocation for {what} is too large: requested {requested} bytes, cap {cap} bytes"
            ),
            Self::DeviceAllocationFailed { requested, what } => write!(
                f,
                "Metal transcode device allocation failed for {what}: {requested} bytes"
            ),
        }
    }
}

impl From<MetalTranscodeError> for TranscodeStageError {
    fn from(error: MetalTranscodeError) -> Self {
        match error {
            MetalTranscodeError::MetalUnavailable => Self::DeviceUnavailable,
            MetalTranscodeError::UnsupportedJob(reason) => Self::Unsupported(reason),
            MetalTranscodeError::Runtime(failure) => {
                let operation = failure.operation();
                Self::backend("metal", operation, failure)
            }
            MetalTranscodeError::MetalSupport { operation, source } => {
                Self::backend("metal", operation, source)
            }
            MetalTranscodeError::Kernel(reason) => Self::backend(
                "metal",
                "kernel execution",
                MetalTranscodeError::Kernel(reason),
            ),
            MetalTranscodeError::HostAllocationTooLarge { requested, cap, .. } => {
                Self::MemoryCapExceeded { requested, cap }
            }
            MetalTranscodeError::HostAllocationFailed { requested, .. } => {
                Self::HostAllocationFailed { bytes: requested }
            }
            MetalTranscodeError::DeviceAllocationTooLarge {
                requested,
                cap,
                what,
            } => Self::DeviceMemoryCapExceeded {
                backend: "metal",
                what,
                requested,
                cap,
            },
            MetalTranscodeError::DeviceAllocationFailed { requested, what } => {
                Self::DeviceAllocationFailed {
                    backend: "metal",
                    what,
                    requested,
                }
            }
        }
    }
}

impl From<SparseWeightRowsError> for MetalTranscodeError {
    fn from(error: SparseWeightRowsError) -> Self {
        match error {
            SparseWeightRowsError::SizeOverflow => Self::HostAllocationTooLarge {
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                what: "projection weights",
            },
            SparseWeightRowsError::AllocationTooLarge { requested, cap } => {
                Self::HostAllocationTooLarge {
                    requested,
                    cap,
                    what: "projection weights",
                }
            }
            SparseWeightRowsError::HostAllocationFailed { requested } => {
                Self::HostAllocationFailed {
                    requested,
                    what: "projection weights",
                }
            }
        }
    }
}

impl std::error::Error for MetalRuntimeFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.source.as_ref())
    }
}

impl std::error::Error for MetalTranscodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Runtime(failure) => Some(failure),
            Self::MetalSupport { source, .. } => Some(source),
            Self::MetalUnavailable
            | Self::UnsupportedJob(_)
            | Self::Kernel(_)
            | Self::HostAllocationTooLarge { .. }
            | Self::HostAllocationFailed { .. }
            | Self::DeviceAllocationTooLarge { .. }
            | Self::DeviceAllocationFailed { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MetalTranscodeError, TranscodeStageError};
    use j2k_metal_support::MetalSupportError;

    #[test]
    fn host_cap_failure_preserves_typed_stage_error() {
        let stage = TranscodeStageError::from(MetalTranscodeError::HostAllocationTooLarge {
            requested: 65,
            cap: 64,
            what: "test host output",
        });
        assert!(matches!(
            stage,
            TranscodeStageError::MemoryCapExceeded {
                requested: 65,
                cap: 64,
            }
        ));
    }

    #[test]
    fn host_allocator_failure_preserves_typed_stage_error() {
        let stage = TranscodeStageError::from(MetalTranscodeError::HostAllocationFailed {
            requested: 4096,
            what: "test host output",
        });
        assert!(matches!(
            stage,
            TranscodeStageError::HostAllocationFailed { bytes: 4096 }
        ));
    }

    #[test]
    fn device_allocation_failures_preserve_stage_resource_categories() {
        let cap = TranscodeStageError::from(MetalTranscodeError::DeviceAllocationTooLarge {
            requested: 65,
            cap: 64,
            what: "test device output",
        });
        assert!(matches!(
            cap,
            TranscodeStageError::DeviceMemoryCapExceeded {
                backend: "metal",
                what: "test device output",
                requested: 65,
                cap: 64,
            }
        ));

        let allocation = TranscodeStageError::from(MetalTranscodeError::DeviceAllocationFailed {
            requested: 4096,
            what: "test device output",
        });
        assert!(matches!(
            allocation,
            TranscodeStageError::DeviceAllocationFailed {
                backend: "metal",
                what: "test device output",
                requested: 4096,
            }
        ));
    }

    #[test]
    fn metal_support_failure_keeps_typed_source_and_is_not_recoverable() {
        let source = MetalSupportError::CommandBufferUnavailable;
        let error = MetalTranscodeError::support("test command buffer", source.clone());

        assert!(!error.is_recoverable());
        assert!(matches!(
            &error,
            MetalTranscodeError::MetalSupport {
                operation: "test command buffer",
                source: stored,
            } if stored == &source
        ));
        let chained = std::error::Error::source(&error).expect("typed Metal support source");
        assert!(chained.downcast_ref::<MetalSupportError>().is_some());

        let stage = TranscodeStageError::from(error);
        assert!(matches!(
            &stage,
            TranscodeStageError::Backend {
                backend: "metal",
                operation: "test command buffer",
                ..
            }
        ));
        let chained = std::error::Error::source(&stage).expect("typed stage backend source");
        assert_eq!(chained.downcast_ref::<MetalSupportError>(), Some(&source));
    }
}
