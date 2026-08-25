// SPDX-License-Identifier: MIT OR Apache-2.0

/// Failure at the codec-to-MPSGraph boundary.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The current target is not Apple Silicon macOS.
    #[error("MPSGraph integration requires Apple Silicon macOS 11 or newer")]
    UnsupportedPlatform,
    /// A requested or derived tensor contract cannot be represented by this integration.
    #[error("invalid MPSGraph tensor contract: {reason}")]
    InvalidTensorContract {
        /// Human-readable contract violation.
        reason: &'static str,
    },
    /// Tensor shape or byte-length arithmetic overflowed `usize`.
    #[error("MPSGraph tensor shape arithmetic overflow")]
    TensorShapeOverflow,
    /// `MPSGraph` reported an asynchronous execution error.
    #[error("MPSGraph execution failed ({domain}, code {code}): {description}")]
    GraphExecution {
        /// Foundation error domain.
        domain: String,
        /// Foundation error code.
        code: isize,
        /// Owned localized error description.
        description: String,
    },
    /// `MPSGraph` completed without returning one of the requested target tensors.
    #[error("MPSGraph did not return target result {index}")]
    MissingGraphOutput {
        /// Zero-based target index.
        index: usize,
    },
    /// A fallible host allocation could not reserve the required capacity.
    #[error("failed to reserve {requested} entries for {what}: {source}")]
    Allocation {
        /// Collection or operation being allocated.
        what: &'static str,
        /// Requested element capacity.
        requested: usize,
        /// Allocation failure returned by the standard library.
        #[source]
        source: std::collections::TryReserveError,
    },
    /// Failure from the shared Metal runtime support layer.
    #[error("Metal runtime operation failed: {0}")]
    MetalRuntime(#[from] j2k_metal_support::MetalSupportError),
    /// Failure from the Metal codec layer.
    #[error("Metal codec operation failed: {0}")]
    Metal(#[from] j2k_metal::Error),
}
