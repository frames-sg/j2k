// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::{DecodeError, DecodeErrorClass};

/// Opaque, CUDA-adapter-owned source for native decoder failures.
///
/// The concrete native error remains available through
/// [`core::error::Error::source`] without becoming part of this crate's public
/// type signatures. Classify the enclosing [`crate::Error`] through
/// [`j2k_core::CodecError`].
#[derive(Debug, PartialEq, Eq)]
pub struct NativeBackendError {
    source: DecodeError,
}

impl NativeBackendError {
    pub(crate) const fn decode(source: DecodeError) -> Self {
        Self { source }
    }

    pub(crate) fn is_decode_truncated(&self) -> bool {
        matches!(
            self.source.classify(),
            DecodeErrorClass::InputTooShort { .. } | DecodeErrorClass::InputTruncatedAt { .. }
        )
    }

    pub(crate) fn is_unsupported(&self) -> bool {
        matches!(self.source.classify(), DecodeErrorClass::Unsupported { .. })
    }
}

impl core::fmt::Display for NativeBackendError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.source.fmt(f)
    }
}

impl core::error::Error for NativeBackendError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        Some(&self.source)
    }
}
