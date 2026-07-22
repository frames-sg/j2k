// SPDX-License-Identifier: MIT OR Apache-2.0

//! Portable materialization of native codec batch groups into Burn tensors.

mod batch;

pub use batch::CpuBurnDecoder;
