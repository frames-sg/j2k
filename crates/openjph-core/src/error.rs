//! Error types for the OpenJPH-RS codec — port of the C++ exception model.
//!
//! The C++ library uses three severity levels (INFO / WARN / ERROR) with
//! integer error codes.  This module maps that model onto a single
//! [`OjphError`] enum backed by [`thiserror`].

use thiserror::Error;

/// Top-level error type returned by all fallible codec operations.
#[derive(Error, Debug)]
pub enum OjphError {
    /// A codec-level error with a numeric code (matches the C++ error codes).
    #[error("codec error (0x{code:08x}): {message}")]
    Codec {
        /// Numeric error code.
        code: u32,
        /// Human-readable description.
        message: String,
    },

    /// Wraps a [`std::io::Error`].
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// An invalid parameter was supplied.
    #[error("invalid parameter: {0}")]
    InvalidParam(String),

    /// The requested feature is not (yet) supported.
    #[error("unsupported feature: {0}")]
    Unsupported(String),

    /// A memory allocation failed.
    #[error("memory allocation failed")]
    AllocationFailed,
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, OjphError>;
