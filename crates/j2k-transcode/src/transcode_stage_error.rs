// SPDX-License-Identifier: MIT OR Apache-2.0

//! Source-preserving failures for optional transcode-stage accelerators.

use core::fmt;
use std::error::Error;

/// Error returned by accelerated transcode stage backends.
#[derive(Debug)]
#[non_exhaustive]
pub enum TranscodeStageError {
    /// The job shape, options, or environment are outside what this backend
    /// supports.
    Unsupported(&'static str),
    /// A backend accepted work and failed while executing it.
    Backend {
        /// Stable backend name.
        backend: &'static str,
        /// Stable backend operation name.
        operation: &'static str,
        /// Concrete adapter, runtime, or kernel source.
        source: Box<dyn Error + Send + Sync + 'static>,
    },
    /// The requested host workspace exceeds the shared process safety cap.
    MemoryCapExceeded {
        /// Requested host bytes.
        requested: usize,
        /// Maximum permitted host bytes.
        cap: usize,
    },
    /// The host allocator could not reserve the requested workspace.
    HostAllocationFailed {
        /// Requested host bytes.
        bytes: usize,
    },
    /// A device allocation exceeds the backend's checked safety cap.
    DeviceMemoryCapExceeded {
        /// Stable backend name.
        backend: &'static str,
        /// Logical device allocation purpose.
        what: &'static str,
        /// Requested device bytes.
        requested: usize,
        /// Maximum permitted device bytes.
        cap: usize,
    },
    /// A cap-valid device allocation failed.
    DeviceAllocationFailed {
        /// Stable backend name.
        backend: &'static str,
        /// Logical device allocation purpose.
        what: &'static str,
        /// Requested device bytes.
        requested: usize,
    },
    /// The device or runtime backing this accelerator is unavailable.
    DeviceUnavailable,
}

impl TranscodeStageError {
    /// Retains a concrete backend, runtime, or kernel error source.
    pub fn backend(
        backend: &'static str,
        operation: &'static str,
        source: impl Error + Send + Sync + 'static,
    ) -> Self {
        Self::Backend {
            backend,
            operation,
            source: Box::new(source),
        }
    }
}

impl fmt::Display for TranscodeStageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(reason) => formatter.write_str(reason),
            Self::Backend {
                backend,
                operation,
                source,
            } => write!(formatter, "{backend} {operation} failed: {source}"),
            Self::MemoryCapExceeded { requested, cap } => write!(
                formatter,
                "transcode stage host workspace requires {requested} bytes, exceeding the {cap}-byte cap"
            ),
            Self::HostAllocationFailed { bytes } => write!(
                formatter,
                "transcode stage host allocation failed for {bytes} bytes"
            ),
            Self::DeviceMemoryCapExceeded {
                backend,
                what,
                requested,
                cap,
            } => write!(
                formatter,
                "{backend} device allocation for {what} requires {requested} bytes, exceeding the {cap}-byte cap"
            ),
            Self::DeviceAllocationFailed {
                backend,
                what,
                requested,
            } => write!(
                formatter,
                "{backend} device allocation failed for {what}: {requested} bytes"
            ),
            Self::DeviceUnavailable => formatter.write_str("accelerator device is unavailable"),
        }
    }
}

impl Error for TranscodeStageError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Backend { source, .. } => Some(source.as_ref()),
            Self::Unsupported(_)
            | Self::MemoryCapExceeded { .. }
            | Self::HostAllocationFailed { .. }
            | Self::DeviceMemoryCapExceeded { .. }
            | Self::DeviceAllocationFailed { .. }
            | Self::DeviceUnavailable => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TranscodeStageError;
    use core::fmt;
    use std::error::Error;

    #[derive(Debug)]
    struct TestBackendError {
        code: u32,
    }

    impl fmt::Display for TestBackendError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "backend fixture {}", self.code)
        }
    }

    impl Error for TestBackendError {}

    #[test]
    fn backend_failure_retains_operation_and_concrete_source() {
        let error =
            TranscodeStageError::backend("fixture", "test dispatch", TestBackendError { code: 17 });
        let TranscodeStageError::Backend {
            backend,
            operation,
            source,
        } = &error
        else {
            panic!("expected backend failure");
        };
        assert_eq!(*backend, "fixture");
        assert_eq!(*operation, "test dispatch");
        assert_eq!(
            source
                .downcast_ref::<TestBackendError>()
                .map(|source| source.code),
            Some(17)
        );
        assert!(Error::source(&error).is_some());
    }

    #[test]
    fn device_resource_failures_remain_typed_and_allocation_free() {
        let cap = TranscodeStageError::DeviceMemoryCapExceeded {
            backend: "fixture",
            what: "output",
            requested: 17,
            cap: 16,
        };
        let allocation = TranscodeStageError::DeviceAllocationFailed {
            backend: "fixture",
            what: "output",
            requested: 16,
        };

        assert!(matches!(
            cap,
            TranscodeStageError::DeviceMemoryCapExceeded {
                requested: 17,
                cap: 16,
                ..
            }
        ));
        assert!(matches!(
            allocation,
            TranscodeStageError::DeviceAllocationFailed { requested: 16, .. }
        ));
    }
}
