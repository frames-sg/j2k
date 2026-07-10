//! Classic JPEG 2000 Tier-1 bitplane decoding.

mod arithmetic;
mod bypass;
mod context;
mod facade;
mod observer;
mod schedule;
mod state;

pub(crate) use facade::{
    decode, decode_code_block_segments_validated, decode_code_block_segments_validated_profiled,
};
pub(crate) use observer::J2kBlockDecodeStats;
pub(crate) use state::{
    BitPlaneDecodeBuffers, BitPlaneDecodeContext, Coefficient, CoefficientState, BITPLANE_BIT_SIZE,
};

#[cfg(test)]
mod golden_tests;
#[cfg(test)]
mod tests;
