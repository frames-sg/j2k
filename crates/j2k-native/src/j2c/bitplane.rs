//! Classic JPEG 2000 Tier-1 bitplane decoding.

mod arithmetic;
mod bypass;
mod context;
mod facade;
mod flag_passes;
pub(super) mod flags;
mod observer;
mod reconstruction;
mod scan;
mod schedule;
mod state;

pub(crate) use facade::{
    decode, decode_code_block_segments_validated, decode_code_block_segments_validated_profiled,
};
pub(crate) use observer::J2kBlockDecodeStats;
#[cfg(test)]
pub(crate) use state::FORCE_PER_COEFFICIENT;
pub(crate) use state::{
    classic_decode_workspace_bytes, BitPlaneDecodeBuffers, BitPlaneDecodeContext, Coefficient,
    CoefficientState, BITPLANE_BIT_SIZE,
};

#[cfg(test)]
mod golden_tests;
#[cfg(test)]
mod tests;
