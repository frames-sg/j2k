// SPDX-License-Identifier: Apache-2.0

use signinum_core::{BufferError, CodecError, InputError, Unsupported};

/// Error returned by tile decompression codecs.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TileCodecError {
    /// Caller-owned buffers were too small or malformed.
    #[error(transparent)]
    Buffer(#[from] BufferError),
    /// Compressed input was truncated or structurally invalid.
    #[error(transparent)]
    Input(#[from] InputError),
    /// The requested tile compression feature is unsupported.
    #[error(transparent)]
    Unsupported(#[from] Unsupported),
    /// Backend library or algorithm-specific failure.
    #[error("{0}")]
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
