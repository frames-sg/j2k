// SPDX-License-Identifier: MIT OR Apache-2.0

//! Direct `MPSGraph` integration for Metal-resident JPEG 2000 and HTJ2K batches.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(unreachable_pub)]

#[cfg(any(test, all(target_arch = "aarch64", target_os = "macos")))]
mod allocation;
mod contract;
mod error;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
mod platform;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
mod program;
#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
mod unsupported;

pub use self::contract::{MpsGraphElementType, MpsGraphTensorSpec};
pub use self::error::Error;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub use self::platform::{MpsGraphBatchDecode, MpsGraphBatchDecoder, MpsGraphInputGroup};
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub use self::program::{MpsGraphProgram, MpsGraphRunOutput, SubmittedMpsGraphRun};
#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
pub use self::unsupported::{
    MpsGraphBatchDecode, MpsGraphBatchDecoder, MpsGraphInputGroup, MpsGraphProgram,
    MpsGraphRunOutput, SubmittedMpsGraphRun,
};
