//! Shared codec constants and pure tables for the `j2k` workspace.
//!
//! This crate is deliberately small and `no_std`: it owns constants and pure
//! math tables that must stay equivalent across CPU, CUDA-Oxide, and Metal
//! backends, but it does not own backend dispatch or kernel launch policy.

#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// JPEG 2000 DWT constants.
pub mod dwt;
/// Baseline JPEG constants.
pub mod jpeg;
/// JPEG 2000 multi-component transform constants.
pub mod mct;
