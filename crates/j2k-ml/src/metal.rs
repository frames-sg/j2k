// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal codec output written into Burn-owned allocations.

mod batch;
#[cfg(target_os = "macos")]
mod interop;

pub use batch::{MetalBurnDecoder, SubmittedMetalBurnBatch};
