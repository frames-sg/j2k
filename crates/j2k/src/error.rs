// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{BufferError, CodecError, InputError, NotImplemented, Unsupported};
use j2k_native::{
    DecodeError as NativeDecodeError, DecodeErrorClass as NativeDecodeErrorClass,
    EncodeError as NativeEncodeError,
};

mod native_source;
pub use native_source::NativeBackendError;

/// Machine-readable class for backend failures surfaced by the facade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BackendErrorKind {
    /// Backend failure has no shared codec classification.
    Other,
    /// Backend failure indicates truncated input.
    Truncated,
    /// Backend failure indicates a planned but unimplemented feature.
    NotImplemented,
    /// Backend failure indicates unsupported input, options, or host capability.
    Unsupported,
    /// Backend failure indicates caller buffer sizing or layout problems.
    Buffer,
    /// Backend failure happened while validating a backend-produced codestream.
    Validation,
    /// Failure indicates an internal facade/cache invariant violation.
    InternalInvariant,
}

impl core::fmt::Display for BackendErrorKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Other => f.write_str("other"),
            Self::Truncated => f.write_str("truncated"),
            Self::NotImplemented => f.write_str("not-implemented"),
            Self::Unsupported => f.write_str("unsupported"),
            Self::Buffer => f.write_str("buffer"),
            Self::Validation => f.write_str("validation"),
            Self::InternalInvariant => f.write_str("internal-invariant"),
        }
    }
}

/// Structured backend failure details.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("backend failed ({kind}): {message}")]
pub struct BackendError {
    kind: BackendErrorKind,
    message: String,
}

impl BackendError {
    /// Construct a backend failure with an explicit class.
    pub fn new(kind: BackendErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Construct a truncated backend failure.
    pub fn truncated(message: impl Into<String>) -> Self {
        Self::new(BackendErrorKind::Truncated, message)
    }

    /// Construct an unimplemented backend failure.
    pub fn not_implemented(message: impl Into<String>) -> Self {
        Self::new(BackendErrorKind::NotImplemented, message)
    }

    /// Construct an unsupported backend failure.
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::new(BackendErrorKind::Unsupported, message)
    }

    /// Construct a backend buffer/layout failure.
    pub fn buffer(message: impl Into<String>) -> Self {
        Self::new(BackendErrorKind::Buffer, message)
    }

    /// Return the machine-readable failure class.
    pub const fn kind(&self) -> BackendErrorKind {
        self.kind
    }

    /// Return the human-readable backend failure message.
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// Error returned by JPEG 2000 inspect, decode, encode, and recode APIs.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum J2kError {
    /// Caller-owned buffers were too small or malformed.
    #[error(transparent)]
    Buffer(#[from] BufferError),

    /// Input was too short or truncated while parsing.
    #[error(transparent)]
    Input(#[from] InputError),

    /// Requested feature is planned but not implemented.
    #[error(transparent)]
    NotImplemented(#[from] NotImplemented),

    /// Requested input feature or option is unsupported.
    #[error(transparent)]
    Unsupported(#[from] Unsupported),

    /// Native backend, accelerator adapter, or backend-output validation failed.
    #[error(transparent)]
    Backend(#[from] BackendError),

    /// Heap-free native decoder failure with stable operation context.
    #[error("{context}: {source}")]
    NativeDecode {
        /// Decode operation that failed.
        context: &'static str,
        /// Structured native decoder failure.
        #[source]
        source: NativeBackendError,
    },

    /// Native decoder failure while validating generated encoder output.
    ///
    /// The structured source is retained for diagnostics, but this phase is
    /// deliberately not classified as truncated caller input or an
    /// unsupported caller request.
    #[error("{context}: {source}")]
    NativeValidation {
        /// Validation operation that failed.
        context: &'static str,
        /// Structured native decoder failure.
        #[source]
        source: NativeBackendError,
    },

    /// Typed native encoder failure with stable operation context.
    #[error("{context}: {source}")]
    NativeEncode {
        /// Encode operation that failed.
        context: &'static str,
        /// Structured native encoder failure.
        #[source]
        source: NativeBackendError,
    },

    /// Native resident-encode failure not represented by the shared encoder taxonomy.
    ///
    /// Current native variants map to the narrower facade categories above.
    /// This source-preserving fallback keeps future variants of the
    /// non-exhaustive resident boundary from being reduced to display text.
    #[error("{context}: {source}")]
    NativeResidentEncode {
        /// Resident encode operation that failed.
        context: &'static str,
        /// Structured resident-boundary failure.
        #[source]
        source: NativeBackendError,
    },

    /// Heap-free codestream-header inspection failure.
    #[error("{context}: {source}")]
    CodestreamHeader {
        /// Header operation that failed.
        context: &'static str,
        /// Structured header parser failure.
        #[source]
        source: NativeBackendError,
    },

    /// A native component plane was shorter than its declared image geometry.
    #[error(
        "backend component plane {component} has {samples} samples, expected at least {expected}"
    )]
    BackendComponentPlaneTooShort {
        /// Zero-based component index.
        component: usize,
        /// Samples actually returned by the backend.
        samples: usize,
        /// Minimum samples required by the output geometry.
        expected: usize,
    },

    /// A facade/cache invariant failed with a static diagnostic.
    #[error("internal JPEG 2000 invariant failed: {what}")]
    InternalInvariant {
        /// Stable description of the violated invariant.
        what: &'static str,
    },

    /// Caller-provided encode samples were malformed.
    #[error("invalid JPEG 2000 samples: {what}")]
    InvalidSamples {
        /// Description of the invalid sample condition.
        what: String,
    },

    /// Lossy rate-control search could not satisfy the requested target.
    #[error("JPEG 2000 lossy rate target unreachable: {target}, best {best}")]
    RateTargetUnreachable {
        /// Requested rate target.
        target: String,
        /// Best achievable result observed by the search.
        best: String,
    },

    /// Requested region lies outside image bounds.
    #[error("region ({x},{y} {w}x{h}) is outside image bounds {image_w}x{image_h}")]
    InvalidRegion {
        /// Region left coordinate.
        x: u32,
        /// Region top coordinate.
        y: u32,
        /// Region width.
        w: u32,
        /// Region height.
        h: u32,
        /// Image width.
        image_w: u32,
        /// Image height.
        image_h: u32,
    },

    /// JP2 box structure was invalid.
    #[error("invalid JP2 box at offset {offset}: {what}")]
    InvalidBox {
        /// Byte offset of the invalid box.
        offset: usize,
        /// Description of the invalid box condition.
        what: &'static str,
    },

    /// Required JP2 box was absent.
    #[error("missing required JP2 box {box_type}")]
    MissingRequiredBox {
        /// Missing box type.
        box_type: &'static str,
    },

    /// Codestream marker was invalid.
    #[error("invalid codestream marker FF{marker:02X} at offset {offset}")]
    InvalidMarker {
        /// Byte offset of the invalid marker.
        offset: usize,
        /// Marker byte following the `0xFF` prefix.
        marker: u8,
    },

    /// Required codestream marker was absent.
    #[error("missing required codestream marker {marker}")]
    MissingRequiredMarker {
        /// Missing marker name.
        marker: &'static str,
    },

    /// SIZ segment was invalid.
    #[error("invalid SIZ segment: {what}")]
    InvalidSiz {
        /// Description of the invalid SIZ condition.
        what: &'static str,
    },

    /// COD segment was invalid.
    #[error("invalid COD segment: {what}")]
    InvalidCod {
        /// Description of the invalid COD condition.
        what: &'static str,
    },

    /// Image dimensions overflowed a size computation.
    #[error("dimension overflow: {width}x{height}")]
    DimensionOverflow {
        /// Image width.
        width: u32,
        /// Image height.
        height: u32,
    },
}

impl J2kError {
    /// Construct an internal facade/cache invariant failure.
    pub(crate) const fn internal_backend(what: &'static str) -> Self {
        Self::InternalInvariant { what }
    }

    /// Translate a native decoder failure into the facade error taxonomy.
    ///
    /// This hidden SPI exists for sibling accelerator adapters. Applications
    /// should receive [`J2kError`] through the public codec APIs instead.
    pub(crate) fn from_native_decode_error(error: NativeDecodeError) -> Self {
        Self::from_native_decode_error_with_context(error, "native JPEG 2000 backend failed")
    }

    pub(crate) fn from_native_decode_error_with_context(
        error: NativeDecodeError,
        context: &'static str,
    ) -> Self {
        match error.classify() {
            NativeDecodeErrorClass::InputTooShort { need, have } => {
                Self::Input(InputError::TooShort { need, have })
            }
            NativeDecodeErrorClass::InputTruncatedAt { offset, segment } => {
                Self::Input(InputError::TruncatedAt { offset, segment })
            }
            NativeDecodeErrorClass::Unsupported { what } => Self::Unsupported(Unsupported { what }),
            _ => Self::NativeDecode {
                context,
                source: NativeBackendError::decode(error),
            },
        }
    }

    pub(crate) const fn from_native_encode_error_with_context(
        source: NativeEncodeError,
        context: &'static str,
    ) -> Self {
        Self::NativeEncode {
            context,
            source: NativeBackendError::encode(source),
        }
    }

    /// Normalize a CPU batch decode failure to the heap-free error subset.
    ///
    /// Encode-only string-bearing variants cannot arise from the batch worker
    /// methods. Converting an unexpected legacy backend variant to a static
    /// invariant prevents one overlooked path from retaining unbounded error
    /// strings across a caller-sized batch.
    pub(crate) fn into_heap_free_batch_decode_error(self) -> Self {
        if self.is_heap_free_batch_decode_error() {
            self
        } else {
            Self::InternalInvariant {
                what: "batch decode produced a heap-owning error variant",
            }
        }
    }

    pub(crate) const fn is_heap_free_batch_decode_error(&self) -> bool {
        !matches!(
            self,
            Self::Backend(_) | Self::InvalidSamples { .. } | Self::RateTargetUnreachable { .. }
        )
    }
}

#[doc(hidden)]
impl CodecError for J2kError {
    fn is_truncated(&self) -> bool {
        matches!(
            self,
            Self::Input(InputError::TooShort { .. } | InputError::TruncatedAt { .. })
        ) || matches!(
            self,
            Self::Backend(error) if error.kind() == BackendErrorKind::Truncated
        ) || matches!(
            self,
            Self::NativeDecode { source, .. } if source.is_decode_truncated()
        ) || matches!(
            self,
            Self::CodestreamHeader { source, .. } if source.is_codestream_header_truncated()
        )
    }

    fn is_not_implemented(&self) -> bool {
        matches!(self, Self::NotImplemented(_))
            || matches!(
                self,
                Self::Backend(error) if error.kind() == BackendErrorKind::NotImplemented
            )
    }

    fn is_unsupported(&self) -> bool {
        matches!(self, Self::Unsupported(_))
            || matches!(
                self,
                Self::Backend(error) if error.kind() == BackendErrorKind::Unsupported
            )
            || matches!(
                self,
                Self::NativeDecode { source, .. }
                    | Self::NativeEncode { source, .. }
                    | Self::CodestreamHeader { source, .. }
                    if source.is_unsupported()
            )
    }

    fn is_buffer_error(&self) -> bool {
        matches!(self, Self::Buffer(_))
            || matches!(
                self,
                Self::Backend(error) if error.kind() == BackendErrorKind::Buffer
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use j2k_native::{
        DecodeError as NativeDecodeError, DecodingError as NativeDecodingError,
        EncodeError as NativeEncodeError, J2kCodestreamHeaderError,
        ResidentHtj2kEncodeError as NativeResidentHtj2kEncodeError,
    };

    #[test]
    fn native_encode_categories_remain_structured_and_preserve_diagnostics() {
        let sources = [
            NativeEncodeError::InvalidInput {
                what: "invalid fixture",
            },
            NativeEncodeError::Unsupported {
                what: "unsupported fixture",
            },
            NativeEncodeError::ArithmeticOverflow {
                what: "overflow fixture",
            },
            NativeEncodeError::AllocationTooLarge {
                what: "cap fixture",
                requested: 9,
                cap: 8,
            },
            NativeEncodeError::HostAllocationFailed {
                what: "allocation fixture",
                bytes: 17,
            },
            NativeEncodeError::Accelerator {
                operation: "accelerator fixture",
                source: crate::J2kEncodeStageError::internal_invariant("backend detail"),
            },
            NativeEncodeError::CodestreamValidation {
                detail: "validation fixture",
            },
            NativeEncodeError::InternalInvariant {
                what: "invariant fixture",
            },
        ];

        for source in sources {
            let expected = source.to_string();
            let error = J2kError::from_native_encode_error_with_context(
                source,
                "native encode fixture failed",
            );
            let J2kError::NativeEncode { context, source } = &error else {
                panic!("expected native encode failure");
            };
            assert_eq!(*context, "native encode fixture failed");
            assert_eq!(source.to_string(), expected);
            assert!(error.is_heap_free_batch_decode_error());
            let concrete_source = core::error::Error::source(&error)
                .and_then(core::error::Error::source)
                .and_then(|source| source.downcast_ref::<NativeEncodeError>())
                .expect("native encode source chain");
            assert_eq!(concrete_source.to_string(), expected);
        }
    }

    #[test]
    fn native_encode_unsupported_keeps_codec_classification() {
        let error = J2kError::from_native_encode_error_with_context(
            NativeEncodeError::Unsupported {
                what: "unsupported encode fixture",
            },
            "native encode failed",
        );

        assert!(error.is_unsupported());
        assert!(!error.is_truncated());
        assert!(!error.is_not_implemented());
    }

    #[test]
    fn native_unsupported_errors_keep_codec_classification() {
        let error = J2kError::from_native_decode_error(NativeDecodeError::Decoding(
            NativeDecodingError::UnsupportedFeature("test feature"),
        ));

        assert!(error.is_unsupported());
        assert!(!error.is_truncated());
        assert!(!error.is_not_implemented());
    }

    #[test]
    fn native_truncated_errors_keep_codec_classification() {
        let error = J2kError::from_native_decode_error(NativeDecodeError::Decoding(
            NativeDecodingError::UnexpectedEof,
        ));

        assert!(error.is_truncated());
        assert!(!error.is_unsupported());
    }

    #[test]
    fn native_backend_errors_remain_typed_and_heap_free() {
        let source = NativeDecodeError::Decoding(NativeDecodingError::CodeBlockDecodeFailure);
        let error = J2kError::from_native_decode_error(source);

        assert!(matches!(
            &error,
            J2kError::NativeDecode {
                context: "native JPEG 2000 backend failed",
                source: stored,
            } if stored == &NativeBackendError::decode(source)
        ));
        assert!(error.is_heap_free_batch_decode_error());
        assert_native_source_chain(&error, &source);
    }

    #[test]
    fn native_decode_resource_errors_preserve_context_and_source() {
        let sources = [
            NativeDecodeError::AllocationTooLarge {
                what: "validation fixture",
                requested: 9,
                cap: 8,
            },
            NativeDecodeError::HostAllocationFailed {
                what: "validation fixture",
                bytes: 7,
            },
        ];

        for source in sources {
            let error = J2kError::from_native_decode_error_with_context(
                source,
                "encode round-trip validation failed",
            );
            assert!(matches!(
                &error,
                J2kError::NativeDecode {
                    context: "encode round-trip validation failed",
                    source: stored,
                } if stored == &NativeBackendError::decode(source)
            ));
            assert!(error.is_heap_free_batch_decode_error());
            assert_native_source_chain(&error, &source);
        }
    }

    #[test]
    fn generated_output_validation_errors_do_not_masquerade_as_caller_failures() {
        for source in [
            NativeDecodeError::Decoding(NativeDecodingError::UnexpectedEof),
            NativeDecodeError::Decoding(NativeDecodingError::UnsupportedFeature(
                "generated fixture feature",
            )),
        ] {
            let error = J2kError::NativeValidation {
                context: "generated JPEG 2000 validation failed",
                source: NativeBackendError::decode(source),
            };
            assert!(!error.is_truncated());
            assert!(!error.is_unsupported());
            assert!(!error.is_not_implemented());
            assert!(error.is_heap_free_batch_decode_error());
            assert_native_source_chain(&error, &source);
        }
    }

    #[test]
    fn codestream_header_source_chain_and_classification_survive_the_facade_boundary() {
        let source = J2kCodestreamHeaderError::TruncatedAt {
            offset: 11,
            segment: "fixture header",
        };
        let error = J2kError::CodestreamHeader {
            context: "header fixture failed",
            source: NativeBackendError::codestream_header(source),
        };

        assert!(error.is_truncated());
        assert!(!error.is_unsupported());
        assert_native_source_chain(&error, &source);
    }

    #[test]
    fn batch_error_normalization_drops_heap_owning_legacy_details() {
        let error = J2kError::Backend(BackendError::new(
            BackendErrorKind::Other,
            "legacy heap detail",
        ));
        let normalized = error.into_heap_free_batch_decode_error();

        assert!(matches!(
            normalized,
            J2kError::InternalInvariant {
                what: "batch decode produced a heap-owning error variant"
            }
        ));
        assert!(normalized.is_heap_free_batch_decode_error());
    }

    #[test]
    fn validation_backend_errors_are_not_reclassified_as_unsupported() {
        let error = J2kError::Backend(BackendError::new(
            BackendErrorKind::Validation,
            "roundtrip validation failed",
        ));

        assert!(!error.is_unsupported());
        assert!(!error.is_truncated());
        assert!(!error.is_not_implemented());
    }

    #[test]
    fn backend_error_kind_drives_codec_classification() {
        let unsupported = J2kError::Backend(BackendError::unsupported("device format"));
        let truncated = J2kError::Backend(BackendError::truncated("entropy data"));
        let not_implemented = J2kError::Backend(BackendError::not_implemented("JPX path"));
        let buffer = J2kError::Backend(BackendError::buffer("output pitch"));
        let other = J2kError::Backend(BackendError::new(
            BackendErrorKind::Other,
            "kernel launch failed",
        ));

        assert!(unsupported.is_unsupported());
        assert!(truncated.is_truncated());
        assert!(not_implemented.is_not_implemented());
        assert!(buffer.is_buffer_error());
        assert!(!other.is_unsupported());
        assert!(!other.is_truncated());
        assert!(!other.is_not_implemented());
        assert!(!other.is_buffer_error());
    }

    #[test]
    fn future_resident_encode_fallback_retains_its_typed_source() {
        let source = NativeResidentHtj2kEncodeError::Declined;
        let error = J2kError::NativeResidentEncode {
            context: "native resident encode failed",
            source: NativeBackendError::resident_encode(source),
        };

        assert!(matches!(
            &error,
            J2kError::NativeResidentEncode {
                context: "native resident encode failed",
                source,
            }
            if source == &NativeBackendError::resident_encode(
                NativeResidentHtj2kEncodeError::Declined,
            )
        ));
        assert_native_source_chain(&error, &NativeResidentHtj2kEncodeError::Declined);
    }

    fn assert_native_source_chain<E>(error: &J2kError, expected: &E)
    where
        E: core::error::Error + PartialEq + 'static,
    {
        let facade_source = core::error::Error::source(error)
            .expect("facade error must expose its facade-owned source");
        assert!(facade_source.downcast_ref::<NativeBackendError>().is_some());
        let concrete_source = facade_source
            .source()
            .expect("facade-owned source must retain the concrete native error");
        assert_eq!(concrete_source.downcast_ref::<E>(), Some(expected));
    }
}
