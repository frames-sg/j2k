//! SIMD-accelerated transform routines.
//!
//! Platform-specific modules are gated behind `cfg(target_arch)` so the crate
//! compiles on any target while only pulling in intrinsics for the host.

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon_colour;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86;

#[cfg(target_arch = "x86_64")]
pub(crate) mod x86_colour;
