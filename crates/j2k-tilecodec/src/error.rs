// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{BufferError, CodecError, InputError, Unsupported};
use std::io::ErrorKind;

/// Error returned by tile decompression codecs.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TileCodecError {
    #[error(transparent)]
    /// Output buffer or allocation limit error.
    Buffer(#[from] BufferError),
    #[error(transparent)]
    /// Input payload is truncated.
    Input(#[from] InputError),
    #[error("{context}: malformed input: {source}")]
    /// Input payload is present but invalid for the selected codec.
    Malformed {
        /// Codec operation that rejected the payload.
        context: &'static str,
        /// Original decoder I/O error, or an `InvalidData` error for a
        /// decoder status that did not provide a typed error.
        #[source]
        source: std::io::Error,
    },
    #[error(transparent)]
    /// Compression type or feature is unsupported.
    Unsupported(#[from] Unsupported),
    #[error("{context}: {source}")]
    /// Decoder or encoder I/O failed outside malformed/truncated input.
    Io {
        /// Codec operation that encountered the I/O failure.
        context: &'static str,
        /// Original I/O error.
        #[source]
        source: std::io::Error,
    },
}

pub(crate) fn truncated_input(segment: &'static str) -> TileCodecError {
    TileCodecError::Input(InputError::TruncatedAt { offset: 0, segment })
}

pub(crate) fn malformed_input(context: &'static str, message: impl Into<String>) -> TileCodecError {
    let message = message.into();
    TileCodecError::Malformed {
        context,
        source: std::io::Error::new(ErrorKind::InvalidData, message),
    }
}

pub(crate) fn malformed_io_error(error: std::io::Error, context: &'static str) -> TileCodecError {
    if error.kind() == ErrorKind::UnexpectedEof {
        return truncated_input(context);
    }
    TileCodecError::Malformed {
        context,
        source: error,
    }
}

pub(crate) fn input_or_backend_io_error(
    error: std::io::Error,
    context: &'static str,
) -> TileCodecError {
    match error.kind() {
        ErrorKind::UnexpectedEof => truncated_input(context),
        ErrorKind::InvalidData | ErrorKind::InvalidInput => TileCodecError::Malformed {
            context,
            source: error,
        },
        _ => TileCodecError::Io {
            context,
            source: error,
        },
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

#[cfg(test)]
mod tests {
    use super::*;
    use j2k_core::CodecError;
    use std::{error::Error as _, fmt, io::Error};

    #[derive(Debug)]
    struct OriginalCause(&'static str);

    impl fmt::Display for OriginalCause {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str(self.0)
        }
    }

    impl std::error::Error for OriginalCause {}

    fn io_source(error: &TileCodecError) -> &Error {
        error
            .source()
            .and_then(|source| source.downcast_ref::<Error>())
            .expect("tilecodec error must retain its I/O source")
    }

    #[test]
    fn operational_io_error_preserves_source_kind_context_and_display() {
        let error = input_or_backend_io_error(
            Error::new(ErrorKind::BrokenPipe, OriginalCause("decoder pipe closed")),
            "bounded decode",
        );

        assert!(matches!(
            &error,
            TileCodecError::Io {
                context: "bounded decode",
                source,
            } if source.kind() == ErrorKind::BrokenPipe
        ));
        assert_eq!(error.to_string(), "bounded decode: decoder pipe closed");
        let source = io_source(&error);
        assert_eq!(source.kind(), ErrorKind::BrokenPipe);
        assert!(
            source
                .get_ref()
                .and_then(|cause| cause.downcast_ref::<OriginalCause>())
                .is_some(),
            "the original concrete I/O cause must remain inspectable"
        );
        assert!(!error.is_truncated());
        assert!(!error.is_unsupported());
        assert!(!error.is_buffer_error());
    }

    #[test]
    fn malformed_io_errors_preserve_original_kinds_and_sources() {
        for kind in [ErrorKind::InvalidData, ErrorKind::InvalidInput] {
            let error = input_or_backend_io_error(
                Error::new(kind, OriginalCause("invalid compressed frame")),
                "deflate decode failed",
            );

            assert!(matches!(
                &error,
                TileCodecError::Malformed {
                    context: "deflate decode failed",
                    source,
                } if source.kind() == kind
            ));
            assert_eq!(
                error.to_string(),
                "deflate decode failed: malformed input: invalid compressed frame"
            );
            let source = io_source(&error);
            assert_eq!(source.kind(), kind);
            assert!(
                source
                    .get_ref()
                    .and_then(|cause| cause.downcast_ref::<OriginalCause>())
                    .is_some(),
                "the original malformed I/O cause must remain inspectable"
            );
            assert!(!error.is_truncated());
        }
    }

    #[test]
    fn malformed_only_helper_preserves_non_eof_io_source() {
        let error = malformed_io_error(
            Error::other(OriginalCause("zstd frame error")),
            "zstd decode failed",
        );

        assert!(matches!(
            &error,
            TileCodecError::Malformed {
                context: "zstd decode failed",
                source,
            } if source.kind() == ErrorKind::Other
        ));
        assert_eq!(
            error.to_string(),
            "zstd decode failed: malformed input: zstd frame error"
        );
        assert_eq!(io_source(&error).kind(), ErrorKind::Other);
    }

    #[test]
    fn unexpected_eof_remains_a_truncated_input_without_io_source() {
        let error = input_or_backend_io_error(
            Error::new(ErrorKind::UnexpectedEof, "short frame"),
            "bounded decode",
        );

        assert!(matches!(&error, TileCodecError::Input(_)));
        assert!(error.is_truncated());
        assert!(error.source().is_none());
        assert!(io_source_if_present(&error).is_none());
        assert_eq!(
            error.to_string(),
            "input truncated at offset 0 while reading bounded decode"
        );
    }

    fn io_source_if_present(error: &TileCodecError) -> Option<&Error> {
        error
            .source()
            .and_then(|source| source.downcast_ref::<Error>())
    }

    #[test]
    fn decoder_status_messages_use_an_invalid_data_source() {
        let error = malformed_input("lzw decode", "no progress");

        assert!(matches!(
            &error,
            TileCodecError::Malformed {
                context: "lzw decode",
                source,
            } if source.kind() == ErrorKind::InvalidData
        ));
        assert_eq!(
            error.to_string(),
            "lzw decode: malformed input: no progress"
        );
        assert_eq!(io_source(&error).kind(), ErrorKind::InvalidData);
    }
}
