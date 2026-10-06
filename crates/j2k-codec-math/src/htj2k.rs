// SPDX-License-Identifier: MIT OR Apache-2.0

//! HTJ2K decode limits shared by host launch code and device kernels.

/// Maximum `u16` symbol-scratch entries the cleanup decoder uses for one code
/// block. Host scratch allocations and device scratch views must agree on it.
pub const HT_MAX_SCRATCH: usize = 3096;
