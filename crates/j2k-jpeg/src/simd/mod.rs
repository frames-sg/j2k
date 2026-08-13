// SPDX-License-Identifier: MIT OR Apache-2.0

//! Private capability and memory boundaries for CPU SIMD kernels.

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon_memory;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_memory;
