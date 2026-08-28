// SPDX-License-Identifier: MIT OR Apache-2.0

//! Validated HT cleanup/refinement layout shared by contribution builders.

use super::super::super::super::{
    bitplane_encode, NativeEncodePipelineError, NativeEncodePipelineResult,
};
use super::super::ht_segment_count;

#[derive(Debug, Clone, Copy)]
pub(super) struct HtContributionLayout {
    pub(super) layer_count: usize,
    pub(super) cleanup_len: usize,
    pub(super) sigprop_len: usize,
    pub(super) magref_len: usize,
    pub(super) sigprop_end: usize,
    pub(super) refinement_end: usize,
    pub(super) split_refinement: bool,
}

pub(super) fn ht_contribution_layout(
    encoded: &bitplane_encode::EncodedCodeBlock,
    num_layers: u8,
    segment_layers: &[usize],
) -> NativeEncodePipelineResult<HtContributionLayout> {
    let layer_count = usize::from(num_layers);
    if segment_layers.len() != ht_segment_count(encoded) {
        return Err(NativeEncodePipelineError::internal_invariant(
            "HTJ2K segment assignment count mismatch",
        ));
    }
    if segment_layers.iter().any(|&layer| layer >= layer_count) {
        return Err(NativeEncodePipelineError::internal_invariant(
            "HTJ2K segment layer exceeds layer count",
        ));
    }
    if segment_layers
        .windows(2)
        .any(|layers| layers[1] < layers[0])
    {
        return Err(NativeEncodePipelineError::internal_invariant(
            "HTJ2K segment layers must be monotonic",
        ));
    }

    let cleanup_len = usize::try_from(encoded.ht_cleanup_length).map_err(|_| {
        NativeEncodePipelineError::arithmetic_overflow("HTJ2K cleanup segment length overflow")
    })?;
    let refinement_len = usize::try_from(encoded.ht_refinement_length).map_err(|_| {
        NativeEncodePipelineError::arithmetic_overflow("HTJ2K refinement segment length overflow")
    })?;
    let sigprop_len = usize::try_from(encoded.ht_sigprop_length).map_err(|_| {
        NativeEncodePipelineError::arithmetic_overflow("HTJ2K SigProp pass length overflow")
    })?;
    let magref_len = usize::try_from(encoded.ht_magref_length).map_err(|_| {
        NativeEncodePipelineError::arithmetic_overflow("HTJ2K MagRef pass length overflow")
    })?;
    if sigprop_len.checked_add(magref_len) != Some(refinement_len) {
        return Err(NativeEncodePipelineError::internal_invariant(
            "HTJ2K refinement pass lengths do not match the segment length",
        ));
    }
    let split_refinement = ht_segment_count(encoded) == 3;
    let sigprop_end = cleanup_len.checked_add(sigprop_len).ok_or_else(|| {
        NativeEncodePipelineError::arithmetic_overflow("HTJ2K SigProp pass range overflow")
    })?;
    let refinement_end = cleanup_len.checked_add(refinement_len).ok_or_else(|| {
        NativeEncodePipelineError::arithmetic_overflow("HTJ2K refinement segment range overflow")
    })?;
    if encoded.num_coding_passes > 0 && cleanup_len == 0 {
        return Err(NativeEncodePipelineError::internal_invariant(
            "HTJ2K cleanup segment is missing",
        ));
    }
    if encoded.num_coding_passes > 1 && refinement_len == 0 {
        return Err(NativeEncodePipelineError::internal_invariant(
            "HTJ2K refinement segment is missing",
        ));
    }
    if refinement_end > encoded.data.len() {
        return Err(NativeEncodePipelineError::internal_invariant(
            "HTJ2K segment range invalid",
        ));
    }
    Ok(HtContributionLayout {
        layer_count,
        cleanup_len,
        sigprop_len,
        magref_len,
        sigprop_end,
        refinement_end,
        split_refinement,
    })
}
