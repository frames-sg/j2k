// SPDX-License-Identifier: MIT OR Apache-2.0

//! Source-preserving failures for the shared encode-stage accelerator SPI.

use alloc::boxed::Box;
use core::error::Error;
use core::fmt;

/// Result returned by fallible encode-stage accelerator hooks.
pub type J2kEncodeStageResult<T> = Result<T, J2kEncodeStageError>;

/// Stable category for an encode-stage accelerator failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum J2kEncodeStageErrorKind {
    /// The submitted stage request was malformed.
    InvalidRequest,
    /// The stage shape or capability is unsupported.
    Unsupported,
    /// Checked stage size arithmetic overflowed.
    ArithmeticOverflow,
    /// Simultaneously live host allocations would exceed the shared cap.
    MemoryCapExceeded,
    /// A cap-valid host reservation failed.
    HostAllocationFailed,
    /// A backend or runtime accepted work and failed it.
    Backend,
    /// Backend or adapter state violated an invariant.
    InternalInvariant,
}

/// Failure returned by a shared encode-stage accelerator hook.
///
/// Equality for [`Self::Backend`] is intentionally source-identity based:
/// independently constructed dynamic backend errors are not assumed to have
/// semantic equality merely because their rendered text matches.
#[derive(Debug)]
#[non_exhaustive]
pub enum J2kEncodeStageError {
    /// The submitted stage request was malformed.
    InvalidRequest {
        /// Stable description of the invalid request.
        what: &'static str,
    },
    /// The requested stage shape or capability is unsupported.
    Unsupported {
        /// Stable description of the unsupported request.
        what: &'static str,
    },
    /// Checked size arithmetic overflowed.
    ArithmeticOverflow {
        /// Stage value or phase whose arithmetic overflowed.
        what: &'static str,
    },
    /// Simultaneously live host allocations would exceed the shared cap.
    MemoryCapExceeded {
        /// Stage allocation or phase being checked.
        what: &'static str,
        /// Checked requested live bytes.
        requested: usize,
        /// Maximum permitted live bytes.
        cap: usize,
    },
    /// A cap-valid host allocation failed.
    HostAllocationFailed {
        /// Stage allocation that failed.
        what: &'static str,
        /// Requested allocation bytes.
        bytes: usize,
    },
    /// A backend or runtime accepted work and failed it.
    Backend {
        /// Stable backend name.
        backend: &'static str,
        /// Stable backend operation name.
        operation: &'static str,
        /// Concrete backend or runtime source.
        source: Box<dyn Error + Send + Sync + 'static>,
    },
    /// Backend or adapter state violated an invariant.
    InternalInvariant {
        /// Stable description of the violated invariant.
        what: &'static str,
    },
}

impl J2kEncodeStageError {
    /// Constructs an invalid-request failure.
    pub const fn invalid_request(what: &'static str) -> Self {
        Self::InvalidRequest { what }
    }

    /// Constructs an unsupported-capability failure.
    pub const fn unsupported(what: &'static str) -> Self {
        Self::Unsupported { what }
    }

    /// Constructs an arithmetic-overflow failure.
    pub const fn arithmetic_overflow(what: &'static str) -> Self {
        Self::ArithmeticOverflow { what }
    }

    /// Constructs a shared-cap failure.
    pub const fn memory_cap_exceeded(what: &'static str, requested: usize, cap: usize) -> Self {
        Self::MemoryCapExceeded {
            what,
            requested,
            cap,
        }
    }

    /// Constructs a host-allocation failure without allocating a message.
    pub const fn host_allocation_failed(what: &'static str, bytes: usize) -> Self {
        Self::HostAllocationFailed { what, bytes }
    }

    /// Retains a concrete backend or runtime source.
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

    /// Constructs an internal-invariant failure.
    pub const fn internal_invariant(what: &'static str) -> Self {
        Self::InternalInvariant { what }
    }

    /// Returns the stable failure category.
    pub const fn kind(&self) -> J2kEncodeStageErrorKind {
        match self {
            Self::InvalidRequest { .. } => J2kEncodeStageErrorKind::InvalidRequest,
            Self::Unsupported { .. } => J2kEncodeStageErrorKind::Unsupported,
            Self::ArithmeticOverflow { .. } => J2kEncodeStageErrorKind::ArithmeticOverflow,
            Self::MemoryCapExceeded { .. } => J2kEncodeStageErrorKind::MemoryCapExceeded,
            Self::HostAllocationFailed { .. } => J2kEncodeStageErrorKind::HostAllocationFailed,
            Self::Backend { .. } => J2kEncodeStageErrorKind::Backend,
            Self::InternalInvariant { .. } => J2kEncodeStageErrorKind::InternalInvariant,
        }
    }

    /// Returns stable presentation text for policy logs and legacy diagnostics.
    pub const fn reason(&self) -> &'static str {
        match self {
            Self::InvalidRequest { what }
            | Self::Unsupported { what }
            | Self::ArithmeticOverflow { what }
            | Self::MemoryCapExceeded { what, .. }
            | Self::HostAllocationFailed { what, .. }
            | Self::InternalInvariant { what } => what,
            Self::Backend { operation, .. } => operation,
        }
    }
}

impl fmt::Display for J2kEncodeStageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest { what } => write!(formatter, "invalid stage request: {what}"),
            Self::Unsupported { what } => write!(formatter, "unsupported stage request: {what}"),
            Self::ArithmeticOverflow { what } => {
                write!(formatter, "stage arithmetic overflow: {what}")
            }
            Self::MemoryCapExceeded {
                what,
                requested,
                cap,
            } => write!(
                formatter,
                "{what} requires {requested} live host bytes, exceeding the {cap}-byte cap"
            ),
            Self::HostAllocationFailed { what, bytes } => {
                write!(
                    formatter,
                    "host allocation failed for {bytes} bytes while allocating {what}"
                )
            }
            Self::Backend {
                backend,
                operation,
                source,
            } => write!(formatter, "{backend} {operation} failed: {source}"),
            Self::InternalInvariant { what } => {
                write!(formatter, "encode-stage invariant failed: {what}")
            }
        }
    }
}

impl Error for J2kEncodeStageError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Backend { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

impl PartialEq for J2kEncodeStageError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::InvalidRequest { what: left }, Self::InvalidRequest { what: right })
            | (Self::Unsupported { what: left }, Self::Unsupported { what: right })
            | (Self::ArithmeticOverflow { what: left }, Self::ArithmeticOverflow { what: right })
            | (Self::InternalInvariant { what: left }, Self::InternalInvariant { what: right }) => {
                left == right
            }
            (
                Self::MemoryCapExceeded {
                    what: left_what,
                    requested: left_requested,
                    cap: left_cap,
                },
                Self::MemoryCapExceeded {
                    what: right_what,
                    requested: right_requested,
                    cap: right_cap,
                },
            ) => {
                left_what == right_what
                    && left_requested == right_requested
                    && left_cap == right_cap
            }
            (
                Self::HostAllocationFailed {
                    what: left_what,
                    bytes: left_bytes,
                },
                Self::HostAllocationFailed {
                    what: right_what,
                    bytes: right_bytes,
                },
            ) => left_what == right_what && left_bytes == right_bytes,
            (
                Self::Backend {
                    backend: left_backend,
                    operation: left_operation,
                    source: left_source,
                },
                Self::Backend {
                    backend: right_backend,
                    operation: right_operation,
                    source: right_source,
                },
            ) => {
                left_backend == right_backend
                    && left_operation == right_operation
                    && core::ptr::eq(left_source, right_source)
            }
            _ => false,
        }
    }
}

impl Eq for J2kEncodeStageError {}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;
    use core::{error::Error, fmt};

    use super::{J2kEncodeStageError, J2kEncodeStageErrorKind};

    #[derive(Debug)]
    struct TestBackendError;

    impl fmt::Display for TestBackendError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("backend fixture")
        }
    }

    impl Error for TestBackendError {}

    #[test]
    fn every_stage_category_has_stable_kind_and_reason() {
        let cases = [
            (
                J2kEncodeStageError::invalid_request("invalid"),
                J2kEncodeStageErrorKind::InvalidRequest,
                "invalid",
            ),
            (
                J2kEncodeStageError::unsupported("unsupported"),
                J2kEncodeStageErrorKind::Unsupported,
                "unsupported",
            ),
            (
                J2kEncodeStageError::arithmetic_overflow("overflow"),
                J2kEncodeStageErrorKind::ArithmeticOverflow,
                "overflow",
            ),
            (
                J2kEncodeStageError::memory_cap_exceeded("cap", 9, 8),
                J2kEncodeStageErrorKind::MemoryCapExceeded,
                "cap",
            ),
            (
                J2kEncodeStageError::host_allocation_failed("allocation", 9),
                J2kEncodeStageErrorKind::HostAllocationFailed,
                "allocation",
            ),
            (
                J2kEncodeStageError::internal_invariant("invariant"),
                J2kEncodeStageErrorKind::InternalInvariant,
                "invariant",
            ),
        ];
        for (error, kind, reason) in cases {
            assert_eq!(error.kind(), kind);
            assert_eq!(error.reason(), reason);
            assert!(Error::source(&error).is_none());
        }
    }

    #[test]
    fn backend_failure_retains_concrete_source() {
        let error = J2kEncodeStageError::backend("test", "dispatch", TestBackendError);
        assert_eq!(error.kind(), J2kEncodeStageErrorKind::Backend);
        assert_eq!(error.reason(), "dispatch");
        assert!(Error::source(&error)
            .and_then(|source| source.downcast_ref::<TestBackendError>())
            .is_some());
    }

    #[test]
    fn every_stage_category_has_stable_display_text() {
        let cases = [
            (
                J2kEncodeStageError::invalid_request("dimensions"),
                "invalid stage request: dimensions",
            ),
            (
                J2kEncodeStageError::unsupported("sampling"),
                "unsupported stage request: sampling",
            ),
            (
                J2kEncodeStageError::arithmetic_overflow("coefficient count"),
                "stage arithmetic overflow: coefficient count",
            ),
            (
                J2kEncodeStageError::memory_cap_exceeded("workspace", 9, 8),
                "workspace requires 9 live host bytes, exceeding the 8-byte cap",
            ),
            (
                J2kEncodeStageError::host_allocation_failed("workspace", 9),
                "host allocation failed for 9 bytes while allocating workspace",
            ),
            (
                J2kEncodeStageError::internal_invariant("result count"),
                "encode-stage invariant failed: result count",
            ),
        ];

        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected);
        }

        let backend = J2kEncodeStageError::backend("cuda", "dispatch", TestBackendError);
        assert_eq!(backend.to_string(), "cuda dispatch failed: backend fixture");
    }

    #[test]
    fn equality_compares_typed_payloads_and_backend_source_identity() {
        assert_eq!(
            J2kEncodeStageError::invalid_request("dimensions"),
            J2kEncodeStageError::invalid_request("dimensions")
        );
        assert_ne!(
            J2kEncodeStageError::invalid_request("dimensions"),
            J2kEncodeStageError::unsupported("dimensions")
        );
        assert_eq!(
            J2kEncodeStageError::memory_cap_exceeded("workspace", 9, 8),
            J2kEncodeStageError::memory_cap_exceeded("workspace", 9, 8)
        );
        assert_ne!(
            J2kEncodeStageError::memory_cap_exceeded("workspace", 9, 8),
            J2kEncodeStageError::memory_cap_exceeded("workspace", 10, 8)
        );
        assert_eq!(
            J2kEncodeStageError::host_allocation_failed("workspace", 9),
            J2kEncodeStageError::host_allocation_failed("workspace", 9)
        );
        assert_ne!(
            J2kEncodeStageError::host_allocation_failed("workspace", 9),
            J2kEncodeStageError::host_allocation_failed("workspace", 10)
        );

        let backend = J2kEncodeStageError::backend("cuda", "dispatch", TestBackendError);
        assert_eq!(backend, backend);
        assert_ne!(
            backend,
            J2kEncodeStageError::backend("cuda", "dispatch", TestBackendError)
        );
    }
}
