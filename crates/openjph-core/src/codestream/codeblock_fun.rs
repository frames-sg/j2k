//! Code-block function dispatch.
//!
//! Port of `ojph_codeblock_fun.h/cpp`. Provides function pointer tables
//! for block encoding/decoding that can be swapped for SIMD variants.

#![allow(dead_code)]

/// Function signature for block decoder (32-bit path).
pub type BlockDecodeFn = fn(
    coded_data: &[u8],
    coded_data_size: u32,
    decoded_data: &mut [i32],
    missing_msbs: u32,
    num_passes: u32,
    lengths1: u32,
    lengths2: u32,
    width: u32,
    height: u32,
    stride: u32,
    stripe_causal: bool,
) -> bool;

/// Function signature for block encoder.
pub type BlockEncodeFn = fn(
    decoded_data: &[i32],
    missing_msbs: u32,
    num_passes: &mut u32,
    lengths1: &mut u32,
    lengths2: &mut u32,
    coded_data: &mut Vec<u8>,
    width: u32,
    height: u32,
    stride: u32,
    stripe_causal: bool,
) -> bool;

/// Runtime-dispatched codeblock function table.
#[derive(Clone, Default)]
pub struct CodeblockFuns {
    pub decode: Option<BlockDecodeFn>,
    pub encode: Option<BlockEncodeFn>,
}

impl CodeblockFuns {
    /// Initialize with generic (non-SIMD) functions.
    /// Actual implementations will be plugged in during Phase 3/5.
    pub fn init_generic() -> Self {
        Self::default()
    }
}
