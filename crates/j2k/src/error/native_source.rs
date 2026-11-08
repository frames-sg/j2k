// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::sync::Arc;

use j2k_native::{
    DecodeError, DecodeErrorClass, EncodeError, J2kCodestreamHeaderError, ResidentHtj2kEncodeError,
};

/// Opaque, facade-owned source for failures raised by the native codec.
///
/// The concrete implementation error remains available through
/// [`core::error::Error::source`] without becoming part of this crate's public
/// type signatures. Callers should normally classify the enclosing
/// [`crate::J2kError`] through [`crate::CodecError`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeBackendError {
    source: NativeBackendErrorSource,
}

// Decode/header errors stay inline. Encode errors may carry non-cloneable dynamic
// accelerator sources; sharing only those preserves their original source chain.
#[derive(Debug, Clone, PartialEq, Eq)]
enum NativeBackendErrorSource {
    Decode(DecodeError),
    Encode(Arc<EncodeError>),
    ResidentEncode(Arc<ResidentHtj2kEncodeError>),
    CodestreamHeader(J2kCodestreamHeaderError),
}

impl NativeBackendError {
    pub(crate) const fn decode(source: DecodeError) -> Self {
        Self {
            source: NativeBackendErrorSource::Decode(source),
        }
    }

    pub(crate) fn encode(source: EncodeError) -> Self {
        Self {
            source: NativeBackendErrorSource::Encode(Arc::new(source)),
        }
    }

    pub(crate) fn resident_encode(source: ResidentHtj2kEncodeError) -> Self {
        Self {
            source: NativeBackendErrorSource::ResidentEncode(Arc::new(source)),
        }
    }

    pub(crate) const fn codestream_header(source: J2kCodestreamHeaderError) -> Self {
        Self {
            source: NativeBackendErrorSource::CodestreamHeader(source),
        }
    }

    pub(super) fn is_decode_truncated(&self) -> bool {
        matches!(
            &self.source,
            NativeBackendErrorSource::Decode(source)
                if matches!(
                    source.classify(),
                    DecodeErrorClass::InputTooShort { .. }
                        | DecodeErrorClass::InputTruncatedAt { .. }
                )
        )
    }

    pub(super) fn is_unsupported(&self) -> bool {
        match &self.source {
            NativeBackendErrorSource::Decode(source) => {
                matches!(source.classify(), DecodeErrorClass::Unsupported { .. })
            }
            NativeBackendErrorSource::Encode(source) => {
                matches!(source.as_ref(), EncodeError::Unsupported { .. })
            }
            NativeBackendErrorSource::CodestreamHeader(J2kCodestreamHeaderError::Unsupported {
                ..
            }) => true,
            NativeBackendErrorSource::ResidentEncode(_)
            | NativeBackendErrorSource::CodestreamHeader(_) => false,
        }
    }

    pub(super) const fn is_codestream_header_truncated(&self) -> bool {
        matches!(
            self.source,
            NativeBackendErrorSource::CodestreamHeader(
                J2kCodestreamHeaderError::TooShort { .. }
                    | J2kCodestreamHeaderError::TruncatedAt { .. }
            )
        )
    }
}

impl core::fmt::Display for NativeBackendError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match &self.source {
            NativeBackendErrorSource::Decode(source) => source.fmt(f),
            NativeBackendErrorSource::Encode(source) => source.fmt(f),
            NativeBackendErrorSource::ResidentEncode(source) => source.fmt(f),
            NativeBackendErrorSource::CodestreamHeader(source) => source.fmt(f),
        }
    }
}

impl core::error::Error for NativeBackendError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        Some(match &self.source {
            NativeBackendErrorSource::Decode(source) => source,
            NativeBackendErrorSource::Encode(source) => source.as_ref(),
            NativeBackendErrorSource::ResidentEncode(source) => source.as_ref(),
            NativeBackendErrorSource::CodestreamHeader(source) => source,
        })
    }
}
