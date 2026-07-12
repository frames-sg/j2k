// SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;

/// Error returned when sparse projection weights cannot be built within the
/// codec-owned host-allocation budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseWeightRowsError {
    /// Size arithmetic overflowed the host address space.
    SizeOverflow,
    /// The complete sparse rows and transient workspace exceed the safety cap.
    AllocationTooLarge {
        /// Requested byte count, saturated on overflow.
        requested: usize,
        /// Maximum permitted byte count.
        cap: usize,
    },
    /// A bounded allocation could not be reserved.
    HostAllocationFailed {
        /// Requested byte count.
        requested: usize,
    },
}

impl fmt::Display for SparseWeightRowsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SizeOverflow => f.write_str("sparse Metal weight-row size overflow"),
            Self::AllocationTooLarge { requested, cap } => write!(
                f,
                "sparse Metal weight rows require {requested} bytes, cap {cap} bytes"
            ),
            Self::HostAllocationFailed { requested } => write!(
                f,
                "sparse Metal weight-row host allocation failed for {requested} bytes"
            ),
        }
    }
}

impl std::error::Error for SparseWeightRowsError {}

pub(super) fn allocation_error(error: j2k_core::HostAllocationError) -> SparseWeightRowsError {
    SparseWeightRowsError::HostAllocationFailed {
        requested: error.requested_bytes(),
    }
}
