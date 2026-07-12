// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::{
    DecodeError, DecodeErrorClass, EncodeError, J2kCodestreamHeaderError, ResidentHtj2kEncodeError,
};

/// Opaque, facade-owned source for failures raised by the native codec.
///
/// The concrete implementation error remains available through
/// [`core::error::Error::source`] without becoming part of this crate's public
/// type signatures. Callers should normally classify the enclosing
/// [`crate::J2kError`] through [`crate::CodecError`].
#[derive(Debug, PartialEq, Eq)]
pub struct NativeBackendError {
    source: NativeBackendErrorSource,
}

#[derive(Debug, PartialEq, Eq)]
enum NativeBackendErrorSource {
    Decode(DecodeError),
    Encode(EncodeError),
    ResidentEncode(ResidentHtj2kEncodeError),
    CodestreamHeader(J2kCodestreamHeaderError),
}

impl NativeBackendError {
    pub(crate) const fn decode(source: DecodeError) -> Self {
        Self {
            source: NativeBackendErrorSource::Decode(source),
        }
    }

    pub(crate) const fn encode(source: EncodeError) -> Self {
        Self {
            source: NativeBackendErrorSource::Encode(source),
        }
    }

    pub(crate) const fn resident_encode(source: ResidentHtj2kEncodeError) -> Self {
        Self {
            source: NativeBackendErrorSource::ResidentEncode(source),
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
            NativeBackendErrorSource::Encode(EncodeError::Unsupported { .. })
            | NativeBackendErrorSource::CodestreamHeader(J2kCodestreamHeaderError::Unsupported {
                ..
            }) => true,
            NativeBackendErrorSource::Encode(_)
            | NativeBackendErrorSource::ResidentEncode(_)
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
            NativeBackendErrorSource::Encode(source) => source,
            NativeBackendErrorSource::ResidentEncode(source) => source,
            NativeBackendErrorSource::CodestreamHeader(source) => source,
        })
    }
}
