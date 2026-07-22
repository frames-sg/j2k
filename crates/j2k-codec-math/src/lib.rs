//! Shared codec constants, generated fragments, and helper algorithms for the
//! `j2k` workspace.
//!
//! This crate is deliberately small, allocation-free, and `no_std`. Its
//! constants and pure helper algorithms must stay equivalent across CPU,
//! CUDA-Oxide, and Metal backends, but it does not own backend dispatch or
//! kernel-launch policy.

#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// JPEG 2000 classic Tier-1 probability and context tables.
pub mod classic;
/// JPEG 2000 DWT constants.
pub mod dwt;
/// Generated backend source fragments derived from codec constants.
pub mod generated {
    /// Metal source fragment defining JPEG 2000 DWT 9/7 constants.
    pub const DWT97_CONSTANTS_METAL: &str = include_str!("../generated/dwt97_constants.metal");
}
/// Baseline JPEG constants.
pub mod jpeg;
/// JPEG 2000 multi-component transform constants.
pub mod mct;
