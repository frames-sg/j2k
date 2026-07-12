// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed failures for bounded profile ownership and formatting.

use core::fmt;

/// Result type for fallible profile parsing, aggregation, and formatting.
pub type ProfileResult<T> = Result<T, ProfileError>;

/// Failure returned by bounded profiling helpers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ProfileError {
    /// A configured limit set is internally inconsistent.
    InvalidLimits {
        /// Limit relationship that is invalid.
        what: &'static str,
    },
    /// Caller-provided profile text cannot be represented unambiguously.
    InvalidInput {
        /// Input rule that was violated.
        what: &'static str,
    },
    /// Checked size, count, counter, or numeric aggregation overflowed.
    SizeOverflow {
        /// Operation whose size arithmetic overflowed.
        what: &'static str,
    },
    /// A configured profile limit was exceeded.
    LimitExceeded {
        /// Limited profile owner or output.
        what: &'static str,
        /// Requested count or byte size.
        requested: usize,
        /// Configured maximum.
        limit: usize,
    },
    /// The host allocator rejected a bounded reservation.
    AllocationFailed {
        /// Profile owner whose reservation failed.
        what: &'static str,
        /// Requested element count or byte size.
        requested: usize,
    },
}

impl fmt::Display for ProfileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimits { what } => write!(formatter, "invalid profile limits: {what}"),
            Self::InvalidInput { what } => write!(formatter, "invalid profile input: {what}"),
            Self::SizeOverflow { what } => write!(formatter, "profile size overflow: {what}"),
            Self::LimitExceeded {
                what,
                requested,
                limit,
            } => write!(
                formatter,
                "profile {what} exceeds limit: requested {requested}, limit {limit}"
            ),
            Self::AllocationFailed { what, requested } => {
                write!(
                    formatter,
                    "profile {what} allocation failed for {requested}"
                )
            }
        }
    }
}

impl core::error::Error for ProfileError {}
