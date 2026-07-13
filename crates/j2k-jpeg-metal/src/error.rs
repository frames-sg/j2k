// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{
    adapter_error_is_buffer_error, adapter_error_is_not_implemented, adapter_error_is_truncated,
    adapter_error_is_unsupported, AdapterErrorKind, AdapterErrorParts, BackendRequest,
    BatchInfrastructureError, BufferError, CodecError,
};
use j2k_jpeg::{
    adapter::{FastPacketError, JpegCachedPlanBuildError, JpegPlanCacheError},
    JpegEncodeError, JpegError,
};
use j2k_metal_support::MetalSupportError;

#[derive(Clone, Debug, thiserror::Error)]
/// Errors returned by the Metal JPEG backend.
pub enum Error {
    /// Error returned by the CPU JPEG parser or fallback decoder.
    #[error(transparent)]
    Decode(#[from] JpegError),
    /// Error returned while assembling a baseline JPEG encode result.
    #[error(transparent)]
    Encode(#[from] JpegEncodeError),
    /// Fast-packet construction failed after the input matched a supported
    /// sampling family.
    #[error("JPEG fast-packet construction failed: {source}")]
    FastPacket {
        /// Concrete malformed-input, resource, or invariant failure.
        #[source]
        source: FastPacketError,
    },
    /// Shared JPEG plan construction or cache retention failed.
    #[error(transparent)]
    JpegPlanCache(#[from] JpegPlanCacheError),
    /// Output buffer validation failed.
    #[error(transparent)]
    Buffer(#[from] BufferError),
    /// Batch metadata allocation, scheduling, or result collection failed independently of one tile.
    #[error("{0}")]
    BatchInfrastructure(
        #[from]
        #[source]
        BatchInfrastructureError,
    ),
    /// The requested backend is not supported by this crate.
    #[error("backend request {request:?} is not supported by j2k-jpeg-metal")]
    UnsupportedBackend {
        /// Backend requested by the caller.
        request: BackendRequest,
    },
    /// A Metal-specific request is structurally unsupported.
    #[error("unsupported JPEG Metal request: {reason}")]
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
    /// Metal kernel launch, validation, or completion failed.
    #[error("Metal kernel error: {message}")]
    MetalKernel {
        /// Kernel error message.
        message: String,
    },
    /// Shared Metal backend state was poisoned by a prior panic.
    #[error("Metal state `{state}` is poisoned")]
    MetalStatePoisoned {
        /// Name of the poisoned state.
        state: &'static str,
    },
}

#[doc(hidden)]
impl AdapterErrorParts for Error {
    fn source_codec_error(&self) -> Option<&dyn CodecError> {
        match self {
            Self::Decode(inner) => Some(inner),
            Self::FastPacket { source } => Some(source),
            _ => None,
        }
    }

    fn adapter_error_kind(&self) -> AdapterErrorKind {
        match self {
            Self::UnsupportedBackend { .. }
            | Self::UnsupportedMetalRequest { .. }
            | Self::MetalUnavailable
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
            | Self::Encode(_)
            | Self::FastPacket { .. }
            | Self::JpegPlanCache(_)
            | Self::BatchInfrastructure(_)
            | Self::MetalRuntime { .. }
            | Self::MetalSupport { .. }
            | Self::MetalKernel { .. }
            | Self::MetalStatePoisoned { .. } => AdapterErrorKind::Other,
        }
    }
}

impl From<JpegCachedPlanBuildError> for Error {
    fn from(error: JpegCachedPlanBuildError) -> Self {
        match error {
            JpegCachedPlanBuildError::Decode(source) => Self::Decode(source),
            JpegCachedPlanBuildError::FastPacket(source) => Self::FastPacket { source },
            JpegCachedPlanBuildError::Cache(source) => Self::JpegPlanCache(source),
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

#[cfg(any(target_os = "macos", test))]
pub(crate) fn metal_kernel_support_error(
    message: impl Into<String>,
    source: MetalSupportError,
) -> Error {
    Error::MetalSupport {
        message: format!("Metal kernel error: {}", message.into()),
        source,
    }
}

#[cfg(any(target_os = "macos", test))]
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

#[cfg(test)]
mod tests {
    use std::error::Error as _;

    use j2k_core::CodecError;
    use j2k_jpeg::{
        adapter::{FastPacketError, JpegCachedPlanBuildError, JpegPlanCacheError, TableKind},
        JpegError,
    };
    use j2k_metal_support::MetalSupportError;

    use super::{metal_kernel_support_error, metal_runtime_support_error, Error};

    #[test]
    fn cloned_metal_support_error_keeps_display_source_and_classification() {
        let source = MetalSupportError::BufferAlignment {
            offset_bytes: 1,
            align: 4,
        };
        let error = metal_kernel_support_error(
            format!("JPEG Metal status readback buffer access invalid: {source}"),
            source.clone(),
        );
        let cloned = error.clone();

        assert_eq!(
            cloned.to_string(),
            "Metal kernel error: JPEG Metal status readback buffer access invalid: Metal buffer range offset 1 is not aligned to 4 bytes"
        );
        assert!(matches!(
            &cloned,
            Error::MetalSupport { source: stored, .. } if stored == &source
        ));
        let chained = std::error::Error::source(&cloned).expect("typed Metal support source");
        assert!(chained.downcast_ref::<MetalSupportError>().is_some());
        assert!(!cloned.is_unsupported());
        assert!(!cloned.is_buffer_error());
    }

    #[test]
    fn runtime_unavailability_keeps_existing_unsupported_route() {
        let error = metal_runtime_support_error(&MetalSupportError::MetalUnavailable);

        assert!(matches!(error, Error::MetalUnavailable));
        assert!(error.is_unsupported());
    }

    #[test]
    fn cached_plan_build_errors_keep_their_existing_metal_categories() {
        let decode = Error::from(JpegCachedPlanBuildError::Decode(JpegError::UnexpectedEoi {
            mcu_at: 1,
            mcu_total: 2,
        }));
        let packet = Error::from(JpegCachedPlanBuildError::FastPacket(
            FastPacketError::MissingHuffmanTable {
                kind: TableKind::Ac,
                slot: 1,
            },
        ));
        let cache = Error::from(JpegCachedPlanBuildError::Cache(
            JpegPlanCacheError::Invariant("test cached-plan invariant"),
        ));

        assert!(matches!(decode, Error::Decode(_)));
        assert!(matches!(packet, Error::FastPacket { .. }));
        assert!(matches!(cache, Error::JpegPlanCache(_)));
    }

    #[test]
    fn cloned_plan_cache_allocation_error_preserves_source_and_classification() {
        let mut impossible = Vec::<u8>::new();
        let source = impossible
            .try_reserve(usize::MAX)
            .expect_err("impossible reservation must fail");
        let error = Error::JpegPlanCache(JpegPlanCacheError::Allocation {
            what: "test JPEG plan cache metadata",
            bytes: usize::MAX,
            source,
        });
        let cloned = error.clone();

        assert!(matches!(
            &cloned,
            Error::JpegPlanCache(JpegPlanCacheError::Allocation {
                what: "test JPEG plan cache metadata",
                bytes: usize::MAX,
                ..
            })
        ));
        assert!(cloned.source().is_some());
        assert!(!cloned.is_truncated());
        assert!(!cloned.is_not_implemented());
        assert!(!cloned.is_unsupported());
        assert!(!cloned.is_buffer_error());
    }

    #[test]
    fn plan_cache_over_limit_error_remains_typed_and_non_codec() {
        let error = Error::JpegPlanCache(JpegPlanCacheError::Limit {
            what: "shared JPEG input bytes",
            requested: 65,
            cap: 64,
        });
        let cloned = error.clone();

        assert!(matches!(
            cloned,
            Error::JpegPlanCache(JpegPlanCacheError::Limit {
                what: "shared JPEG input bytes",
                requested: 65,
                cap: 64,
            })
        ));
        assert!(!error.is_truncated());
        assert!(!error.is_not_implemented());
        assert!(!error.is_unsupported());
        assert!(!error.is_buffer_error());
    }
}
