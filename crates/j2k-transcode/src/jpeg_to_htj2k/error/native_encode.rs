// SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;

use j2k_native::EncodeError;

/// Machine-readable class for a native HTJ2K encode failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Htj2kEncodeErrorKind {
    /// Caller-provided samples, geometry, metadata, or options were invalid.
    InvalidInput,
    /// The requested encode feature or shape is not implemented.
    Unsupported,
    /// Checked arithmetic overflowed while planning an encode phase.
    ArithmeticOverflow,
    /// A host memory limit or allocation rejected the encode workspace.
    HostResource,
    /// An optional encode-stage accelerator failed.
    Accelerator,
    /// A generated codestream failed validation.
    CodestreamValidation,
    /// Native encode state violated an internal invariant.
    InternalInvariant,
    /// A newer native failure has no more specific shared classification.
    Other,
}

/// Opaque, transcode-owned source for a native HTJ2K encode failure.
///
/// The concrete native error remains available through
/// [`core::error::Error::source`] without making `j2k-native` part of this
/// crate's public type signatures.
#[derive(Debug, PartialEq, Eq)]
pub struct Htj2kEncodeError {
    source: EncodeError,
}

impl Htj2kEncodeError {
    pub(super) const fn new(source: EncodeError) -> Self {
        Self { source }
    }

    /// Return the machine-readable failure class.
    #[must_use]
    pub const fn kind(&self) -> Htj2kEncodeErrorKind {
        match self.source {
            EncodeError::InvalidInput { .. } => Htj2kEncodeErrorKind::InvalidInput,
            EncodeError::Unsupported { .. } => Htj2kEncodeErrorKind::Unsupported,
            EncodeError::ArithmeticOverflow { .. } => Htj2kEncodeErrorKind::ArithmeticOverflow,
            EncodeError::AllocationTooLarge { .. } | EncodeError::HostAllocationFailed { .. } => {
                Htj2kEncodeErrorKind::HostResource
            }
            EncodeError::Accelerator { .. } => Htj2kEncodeErrorKind::Accelerator,
            EncodeError::CodestreamValidation { .. } => Htj2kEncodeErrorKind::CodestreamValidation,
            EncodeError::InternalInvariant { .. } => Htj2kEncodeErrorKind::InternalInvariant,
            _ => Htj2kEncodeErrorKind::Other,
        }
    }
}

impl fmt::Display for Htj2kEncodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.source.fmt(formatter)
    }
}

impl core::error::Error for Htj2kEncodeError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        Some(&self.source)
    }
}

#[cfg(test)]
mod tests {
    use super::{EncodeError, Htj2kEncodeError, Htj2kEncodeErrorKind};

    #[test]
    fn native_encode_kinds_remain_structured_behind_the_opaque_boundary() {
        let cases = [
            (
                EncodeError::InvalidInput { what: "invalid" },
                Htj2kEncodeErrorKind::InvalidInput,
            ),
            (
                EncodeError::Unsupported {
                    what: "unsupported",
                },
                Htj2kEncodeErrorKind::Unsupported,
            ),
            (
                EncodeError::ArithmeticOverflow { what: "overflow" },
                Htj2kEncodeErrorKind::ArithmeticOverflow,
            ),
            (
                EncodeError::AllocationTooLarge {
                    what: "workspace",
                    requested: 17,
                    cap: 16,
                },
                Htj2kEncodeErrorKind::HostResource,
            ),
            (
                EncodeError::HostAllocationFailed {
                    what: "workspace",
                    bytes: 16,
                },
                Htj2kEncodeErrorKind::HostResource,
            ),
            (
                EncodeError::Accelerator {
                    operation: "dispatch",
                    source: j2k::J2kEncodeStageError::internal_invariant("failed"),
                },
                Htj2kEncodeErrorKind::Accelerator,
            ),
            (
                EncodeError::CodestreamValidation { detail: "invalid" },
                Htj2kEncodeErrorKind::CodestreamValidation,
            ),
            (
                EncodeError::InternalInvariant { what: "invalid" },
                Htj2kEncodeErrorKind::InternalInvariant,
            ),
        ];

        for (source, expected) in cases {
            let error = Htj2kEncodeError::new(source);
            assert_eq!(error.kind(), expected);
            assert!(core::error::Error::source(&error)
                .and_then(|source| source.downcast_ref::<EncodeError>())
                .is_some());
        }
    }

    #[test]
    fn opaque_native_encode_error_retains_structural_equality() {
        assert_eq!(
            Htj2kEncodeError::new(EncodeError::InvalidInput { what: "fixture" }),
            Htj2kEncodeError::new(EncodeError::InvalidInput { what: "fixture" })
        );
    }
}
