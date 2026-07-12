//! Scalar HTJ2K block decoding.

mod benchmark;
mod cleanup;
mod facade;
mod magnitude;
mod pipeline;
mod readers;
mod refinement;
mod segments;
mod significance;
mod state;
mod validation;

pub(crate) use benchmark::{
    decode_sigprop_benchmark_state, prepare_sigprop_benchmark_state, HtSigPropBenchmarkState,
};
pub(crate) use facade::{coefficient_to_i32, decode_with_stats};
pub(crate) use pipeline::{
    ht_decode_workspace_bytes, PHASE_LIMIT_CLEANUP, PHASE_LIMIT_MAGREF, PHASE_LIMIT_SIGPROP,
};
pub(crate) use segments::{
    collect_code_block_data, collect_code_block_segments, CombinedCodeBlockData,
    HtCodeBlockSegments,
};
pub(crate) use significance::sigma_stride;
pub(crate) use state::{HtBlockDecodeContext, HtBlockDecodeScratch, HtBlockDecodeStats};
pub(crate) use validation::decode_segments_validated_with_scratch_for_phase;
#[cfg(test)]
pub(crate) use validation::{
    decode_combined_validated, decode_segments_validated, decode_segments_validated_for_phase,
};

#[cfg(test)]
mod tests;
