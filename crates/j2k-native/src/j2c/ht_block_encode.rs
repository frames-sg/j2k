//! Scalar HTJ2K block encoding.
mod allocation;
mod cleanup;
mod distortion;
mod distribution;
mod emit;
mod facade;
mod quad;
mod refinement;
mod workspace;
mod writers;
pub(crate) use allocation::ht_worker_allocation;
pub(crate) use distribution::collect_encode_distribution;
pub(crate) use facade::try_encode_code_block_with_passes_in_workspace;
pub(crate) use facade::{
    candidate_cleanup_bitplanes, code_block_set_distortion_deltas, effective_coding_passes,
    select_tile_code_block_candidates, tile_candidate_selection_workspace_bytes,
    truncate_code_block_candidate, try_encode_code_block_candidate_sets_with_workspace,
    HtCandidateRange, HtCandidateSelection,
};
#[cfg(test)]
pub(crate) use facade::{
    encode_code_block, encode_code_block_with_passes, try_encode_code_block_set_with_workspace,
};
pub(crate) use facade::{try_encode_code_block, try_encode_code_block_with_passes};
pub(crate) use workspace::HtEncodeWorkspace;
#[cfg(test)]
mod golden_tests;
#[cfg(test)]
mod tests;
