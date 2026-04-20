//! SIMD-accelerated coding routines.
//!
//! Provides SIMD helper functions for the HTJ2K block encoder/decoder.

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86;
