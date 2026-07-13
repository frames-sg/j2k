// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{fmt, DctTransformError, MetricsError, TranscodeStageError};

mod native_encode;
pub use native_encode::{Htj2kEncodeError, Htj2kEncodeErrorKind};

/// Error returned by the experimental transcode path.
#[derive(Debug)]
pub enum JpegToHtj2kError {
    /// JPEG parse or entropy decode failed.
    Jpeg(j2k_jpeg::JpegError),
    /// Input is outside the currently implemented experimental slice.
    Unsupported(&'static str),
    /// The 5/3 DCT-grid transform could not execute.
    Dct53(DctTransformError),
    /// The 9/7 DCT-grid transform could not execute.
    Dct97(DctTransformError),
    /// Optional transform acceleration failed.
    Accelerator(TranscodeStageError),
    /// The requested transcode workspace exceeds the shared process safety cap.
    MemoryCapExceeded {
        /// Requested host bytes.
        requested: usize,
        /// Maximum permitted host bytes.
        cap: usize,
    },
    /// The host allocator could not reserve the requested transcode workspace.
    HostAllocationFailed {
        /// Requested host bytes.
        bytes: usize,
    },
    /// Validation metric construction failed.
    Metrics(MetricsError),
    /// Validation encountered an out-of-range or non-finite coefficient.
    Validation(&'static str),
    /// Internal transcode or batch-scheduler state violated an invariant.
    InternalInvariant {
        /// Stable description of the violated invariant.
        what: &'static str,
    },
    /// Native HTJ2K encode failed.
    Encode(Htj2kEncodeError),
}

impl fmt::Display for JpegToHtj2kError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Jpeg(err) => write!(f, "JPEG extraction failed: {err}"),
            Self::Unsupported(reason) => write!(f, "unsupported transcode input: {reason}"),
            Self::Dct53(reason) => write!(f, "5/3 DCT transform failed: {reason}"),
            Self::Dct97(reason) => write!(f, "9/7 DCT transform failed: {reason}"),
            Self::Accelerator(reason) => write!(f, "transform accelerator failed: {reason}"),
            Self::MemoryCapExceeded { requested, cap } => write!(
                f,
                "transcode host workspace requires {requested} bytes, exceeding the {cap}-byte cap"
            ),
            Self::HostAllocationFailed { bytes } => {
                write!(f, "transcode host allocation failed for {bytes} bytes")
            }
            Self::Metrics(reason) => write!(f, "validation metrics failed: {reason}"),
            Self::Validation(reason) => write!(f, "validation failed: {reason}"),
            Self::InternalInvariant { what } => {
                write!(f, "internal transcode invariant failed: {what}")
            }
            Self::Encode(reason) => write!(f, "HTJ2K encode failed: {reason}"),
        }
    }
}

impl std::error::Error for JpegToHtj2kError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Jpeg(err) => Some(err),
            Self::Dct53(err) | Self::Dct97(err) => Some(err),
            Self::Accelerator(err) => Some(err),
            Self::Metrics(err) => Some(err),
            Self::Encode(err) => Some(err),
            Self::Unsupported(_)
            | Self::MemoryCapExceeded { .. }
            | Self::HostAllocationFailed { .. }
            | Self::Validation(_)
            | Self::InternalInvariant { .. } => None,
        }
    }
}

impl From<j2k_jpeg::JpegError> for JpegToHtj2kError {
    fn from(value: j2k_jpeg::JpegError) -> Self {
        Self::Jpeg(value)
    }
}

pub(super) fn dct53_transform_error(value: DctTransformError) -> JpegToHtj2kError {
    map_transform_error(value, JpegToHtj2kError::Dct53)
}

pub(super) fn dct97_transform_error(value: DctTransformError) -> JpegToHtj2kError {
    map_transform_error(value, JpegToHtj2kError::Dct97)
}

pub(super) fn map_encode_error(value: j2k_native::EncodeError) -> JpegToHtj2kError {
    match value {
        j2k_native::EncodeError::AllocationTooLarge { requested, cap, .. } => {
            JpegToHtj2kError::MemoryCapExceeded { requested, cap }
        }
        j2k_native::EncodeError::HostAllocationFailed { bytes, .. } => {
            JpegToHtj2kError::HostAllocationFailed { bytes }
        }
        error => JpegToHtj2kError::Encode(Htj2kEncodeError::new(error)),
    }
}

fn map_transform_error(
    value: DctTransformError,
    transform: fn(DctTransformError) -> JpegToHtj2kError,
) -> JpegToHtj2kError {
    match value {
        DctTransformError::MemoryCapExceeded { requested, cap } => {
            JpegToHtj2kError::MemoryCapExceeded { requested, cap }
        }
        DctTransformError::HostAllocationFailed { bytes } => {
            JpegToHtj2kError::HostAllocationFailed { bytes }
        }
        error => transform(error),
    }
}

impl From<MetricsError> for JpegToHtj2kError {
    fn from(value: MetricsError) -> Self {
        Self::Metrics(value)
    }
}

#[cfg(test)]
mod transform_mapping_tests;

#[cfg(test)]
mod tests {
    use super::{
        map_encode_error, Htj2kEncodeError, Htj2kEncodeErrorKind, JpegToHtj2kError, MetricsError,
    };
    use j2k_native::EncodeError;

    #[test]
    fn native_encode_resource_errors_lift_into_transcode_resource_categories() {
        assert!(matches!(
            map_encode_error(EncodeError::AllocationTooLarge {
                what: "test encode workspace",
                requested: 17,
                cap: 16,
            }),
            JpegToHtj2kError::MemoryCapExceeded {
                requested: 17,
                cap: 16
            }
        ));
        assert!(matches!(
            map_encode_error(EncodeError::HostAllocationFailed {
                what: "test encode workspace",
                bytes: 17,
            }),
            JpegToHtj2kError::HostAllocationFailed { bytes: 17 }
        ));
    }

    #[test]
    fn native_encode_semantic_errors_remain_typed_and_are_error_sources() {
        let invalid = map_encode_error(EncodeError::InvalidInput {
            what: "invalid test image",
        });
        let JpegToHtj2kError::Encode(invalid_source) = &invalid else {
            panic!("expected native encode error");
        };
        assert_eq!(invalid_source.kind(), Htj2kEncodeErrorKind::InvalidInput);
        assert_eq!(
            invalid.to_string(),
            "HTJ2K encode failed: invalid encode input: invalid test image"
        );
        assert_eq!(
            invalid_source.to_string(),
            "invalid encode input: invalid test image"
        );
        assert_eq!(
            std::error::Error::source(&invalid)
                .and_then(|source| source.downcast_ref::<Htj2kEncodeError>()),
            Some(invalid_source)
        );
        assert!(matches!(
            std::error::Error::source(invalid_source)
                .and_then(|source| source.downcast_ref::<EncodeError>()),
            Some(EncodeError::InvalidInput {
                what: "invalid test image"
            })
        ));

        let accelerator = map_encode_error(EncodeError::Accelerator {
            operation: "test accelerator operation",
            source: j2k::J2kEncodeStageError::internal_invariant("test backend failure"),
        });
        let JpegToHtj2kError::Encode(accelerator_source) = &accelerator else {
            panic!("expected native accelerator encode error");
        };
        assert_eq!(accelerator_source.kind(), Htj2kEncodeErrorKind::Accelerator);
        let native_source = std::error::Error::source(accelerator_source)
            .and_then(|source| source.downcast_ref::<EncodeError>())
            .expect("concrete native encode source");
        let EncodeError::Accelerator { operation, source } = native_source else {
            panic!("expected native accelerator source");
        };
        assert_eq!(*operation, "test accelerator operation");
        assert_eq!(source.reason(), "test backend failure");
        assert_eq!(
            std::error::Error::source(native_source)
                .and_then(|source| source.downcast_ref::<j2k::J2kEncodeStageError>()),
            Some(source)
        );
    }

    #[test]
    fn metrics_failure_retains_its_typed_error_source() {
        let mismatch = JpegToHtj2kError::from(MetricsError::LengthMismatch {
            actual: 2,
            expected: 1,
        });
        let mismatch_source = std::error::Error::source(&mismatch)
            .and_then(|source| source.downcast_ref::<MetricsError>());
        assert!(matches!(
            mismatch_source,
            Some(MetricsError::LengthMismatch {
                actual: 2,
                expected: 1,
            })
        ));

        let cap = JpegToHtj2kError::from(MetricsError::MemoryCapExceeded {
            requested: 65,
            cap: 64,
        });
        let cap_source = std::error::Error::source(&cap)
            .and_then(|source| source.downcast_ref::<MetricsError>());
        assert!(matches!(
            cap_source,
            Some(MetricsError::MemoryCapExceeded {
                requested: 65,
                cap: 64,
            })
        ));

        let allocation = JpegToHtj2kError::from(MetricsError::HostAllocationFailed { bytes: 4096 });
        let allocation_source = std::error::Error::source(&allocation)
            .and_then(|source| source.downcast_ref::<MetricsError>());
        assert!(matches!(
            allocation_source,
            Some(MetricsError::HostAllocationFailed { bytes: 4096 })
        ));
    }
}
