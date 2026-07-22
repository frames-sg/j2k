// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA codec output written into Burn-owned allocations.

mod batch;
mod interop;

pub use batch::{CudaBurnDecoder, SubmittedCudaBurnBatch};
