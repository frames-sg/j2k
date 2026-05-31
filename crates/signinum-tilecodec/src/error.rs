// SPDX-License-Identifier: Apache-2.0

use signinum_core::{BufferError, CodecError, InputError, Unsupported};

#[derive(Debug, thiserror::Error)]
/// Error returned by tile decompression codecs.
pub enum TileCodecError {
    #[error(transparent)]
    /// Output buffer or allocation limit error.
    Buffer(#[from] BufferError),
    #[error(transparent)]
    /// Input payload is truncated or malformed.
    Input(#[from] InputError),
    #[error(transparent)]
    /// Compression type or feature is unsupported.
    Unsupported(#[from] Unsupported),
    #[error("{0}")]
    /// Backend library reported a decode failure.
    Backend(String),
}

impl CodecError for TileCodecError {
    fn is_truncated(&self) -> bool {
        matches!(self, Self::Input(_))
    }

    fn is_not_implemented(&self) -> bool {
        false
    }

    fn is_unsupported(&self) -> bool {
        matches!(self, Self::Unsupported(_))
    }

    fn is_buffer_error(&self) -> bool {
        matches!(self, Self::Buffer(_))
    }
}
