// SPDX-License-Identifier: MIT OR Apache-2.0

//! Owned asynchronous `MPSGraph` submission shared by Metal codec adapters.
//! The adapter retains responsibility for tensor contracts and codec completion.

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
mod submission;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub use submission::{GraphExecutionError, MpsGraphSubmission};
