// SPDX-License-Identifier: MIT OR Apache-2.0

//! HTJ2K decode limits shared by host launch code and device kernels.

/// Maximum `u16` symbol-scratch entries the cleanup decoder uses for one code
/// block. Host scratch allocations and device scratch views must agree on it.
pub const HT_MAX_SCRATCH: usize = 3096;

/// The MEL coder's run-length exponent `E_MEL[k]` for each state `k`. The
/// encoder and decoder step `k` through these thresholds as runs complete.
pub const MEL_EXPONENTS: [u8; 13] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5];
