// SPDX-License-Identifier: Apache-2.0

use j2k_core::{BufferError, CodecError, InputError, Unsupported};
use std::io::ErrorKind;

#[derive(Debug, thiserror::Error)]
/// Error returned by tile decompression codecs.
pub enum TileCodecError {
    #[error(transparent)]
    /// Output buffer or allocation limit error.
    Buffer(#[from] BufferError),
    #[error(transparent)]
    /// Input payload is truncated.
    Input(#[from] InputError),
    #[error("{context}: malformed input: {message}")]
    /// Input payload is present but invalid for the selected codec.
    Malformed {
        /// Codec operation that rejected the payload.
        context: &'static str,
        /// Backend error or decoder status.
        message: String,
    },
    #[error(transparent)]
    /// Compression type or feature is unsupported.
    Unsupported(#[from] Unsupported),
    #[error("{0}")]
    /// Backend library reported a decode failure.
    Backend(String),
}

pub(crate) fn truncated_input(segment: &'static str) -> TileCodecError {
    TileCodecError::Input(InputError::TruncatedAt { offset: 0, segment })
}

pub(crate) fn malformed_input(context: &'static str, message: impl Into<String>) -> TileCodecError {
    TileCodecError::Malformed {
        context,
        message: message.into(),
    }
}

pub(crate) fn malformed_io_error(error: &std::io::Error, context: &'static str) -> TileCodecError {
    if error.kind() == ErrorKind::UnexpectedEof {
        return truncated_input(context);
    }
    malformed_input(context, error.to_string())
}

pub(crate) fn input_or_backend_io_error(
    error: &std::io::Error,
    context: &'static str,
) -> TileCodecError {
    match error.kind() {
        ErrorKind::UnexpectedEof => truncated_input(context),
        ErrorKind::InvalidData | ErrorKind::InvalidInput => {
            malformed_input(context, error.to_string())
        }
        _ => TileCodecError::Backend(format!("{context}: {error}")),
    }
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
