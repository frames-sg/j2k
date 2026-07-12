// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::{DecodeError, DecodeErrorClass, EncodeError};

/// Opaque, Metal-adapter-owned source for native codec failures.
///
/// The concrete native error remains available through
/// [`core::error::Error::source`] without becoming part of this crate's public
/// type signatures. Classify the enclosing [`crate::Error`] through
/// [`j2k_core::CodecError`].
#[derive(Debug, PartialEq, Eq)]
pub struct NativeBackendError {
    source: NativeBackendErrorSource,
}

#[derive(Debug, PartialEq, Eq)]
enum NativeBackendErrorSource {
    Decode(DecodeError),
    Encode(EncodeError),
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

    pub(crate) fn is_decode_truncated(&self) -> bool {
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

    pub(crate) fn is_unsupported(&self) -> bool {
        match &self.source {
            NativeBackendErrorSource::Decode(source) => {
                matches!(source.classify(), DecodeErrorClass::Unsupported { .. })
            }
            NativeBackendErrorSource::Encode(EncodeError::Unsupported { .. }) => true,
            NativeBackendErrorSource::Encode(_) => false,
        }
    }
}

impl core::fmt::Display for NativeBackendError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match &self.source {
            NativeBackendErrorSource::Decode(source) => source.fmt(f),
            NativeBackendErrorSource::Encode(source) => source.fmt(f),
        }
    }
}

impl core::error::Error for NativeBackendError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        Some(match &self.source {
            NativeBackendErrorSource::Decode(source) => source,
            NativeBackendErrorSource::Encode(source) => source,
        })
    }
}
