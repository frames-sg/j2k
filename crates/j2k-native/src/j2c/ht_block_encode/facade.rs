// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::bitplane_encode::EncodedCodeBlock;
use super::workspace::HtEncodeWorkspace;
use crate::{j2c::coefficient_view::CoefficientBlockView, EncodeResult};

mod candidates;
mod core;
mod tile_candidates;
pub(crate) use candidates::{
    candidate_cleanup_bitplanes, code_block_set_distortion_deltas, truncate_code_block_candidate,
    try_encode_code_block_candidate_sets_with_workspace,
};
pub(crate) use tile_candidates::{
    select_tile_code_block_candidates, tile_candidate_selection_workspace_bytes, HtCandidateRange,
    HtCandidateSelection,
};

pub(super) const MAX_HT_BITPLANES: u8 = 31;

fn validate_set_request(
    total_bitplanes: u8,
    cleanup_bitplane: u8,
    target_coding_passes: u8,
) -> crate::EncodeResult<()> {
    let what = if !(1..=MAX_HT_BITPLANES).contains(&total_bitplanes) {
        "HTJ2K scalar encoder currently supports 1..=31 bitplanes"
    } else if !(1..=3).contains(&target_coding_passes) {
        "HTJ2K scalar encoder currently supports cleanup, sigprop, and one magref refinement pass"
    } else if cleanup_bitplane >= total_bitplanes {
        "HTJ2K cleanup bitplane must be below the configured bitplane count"
    } else if cleanup_bitplane == 0 && target_coding_passes > 1 {
        "HTJ2K cleanup bitplane zero cannot carry refinement passes"
    } else {
        return Ok(());
    };
    Err(crate::EncodeError::InvalidInput { what })
}

pub(crate) const fn effective_coding_passes(total_bitplanes: u8, requested: u8) -> u8 {
    if requested >= 2 && total_bitplanes > 1 {
        requested
    } else {
        1
    }
}

pub(crate) fn try_encode_code_block(
    coefficients: &[i32],
    width: u32,
    height: u32,
    total_bitplanes: u8,
) -> EncodeResult<EncodedCodeBlock> {
    try_encode_code_block_with_passes(coefficients, width, height, total_bitplanes, 1)
}

pub(crate) fn try_encode_code_block_with_passes(
    coefficients: &[i32],
    width: u32,
    height: u32,
    total_bitplanes: u8,
    target_coding_passes: u8,
) -> EncodeResult<EncodedCodeBlock> {
    let mut workspace = HtEncodeWorkspace::try_new()?;
    try_encode_code_block_with_passes_in_workspace(
        coefficients,
        width,
        height,
        total_bitplanes,
        target_coding_passes,
        &mut workspace,
    )
}

pub(crate) fn try_encode_code_block_with_passes_in_workspace(
    coefficients: &[i32],
    width: u32,
    height: u32,
    total_bitplanes: u8,
    target_coding_passes: u8,
    workspace: &mut HtEncodeWorkspace,
) -> EncodeResult<EncodedCodeBlock> {
    let coefficients =
        CoefficientBlockView::try_contiguous(coefficients, width as usize, height as usize)?;
    try_encode_code_block_view_in_workspace(
        coefficients,
        total_bitplanes,
        target_coding_passes,
        workspace,
    )
}

pub(crate) fn try_encode_code_block_set_with_workspace(
    coefficients: &[i32],
    width: u32,
    height: u32,
    total_bitplanes: u8,
    cleanup_bitplane: u8,
    target_coding_passes: u8,
    workspace: &mut HtEncodeWorkspace,
) -> EncodeResult<EncodedCodeBlock> {
    let coefficients =
        CoefficientBlockView::try_contiguous(coefficients, width as usize, height as usize)?;
    core::try_encode_code_block_set_view_in_workspace(
        coefficients,
        total_bitplanes,
        cleanup_bitplane,
        target_coding_passes,
        true,
        workspace,
    )
}

#[cfg(test)]
pub(crate) fn try_encode_code_block_view(
    coefficients: CoefficientBlockView<'_, i32>,
    total_bitplanes: u8,
    target_coding_passes: u8,
) -> EncodeResult<EncodedCodeBlock> {
    let mut workspace = HtEncodeWorkspace::try_new()?;
    try_encode_code_block_view_in_workspace(
        coefficients,
        total_bitplanes,
        target_coding_passes,
        &mut workspace,
    )
}

fn try_encode_code_block_view_in_workspace(
    coefficients: CoefficientBlockView<'_, i32>,
    total_bitplanes: u8,
    target_coding_passes: u8,
    workspace: &mut HtEncodeWorkspace,
) -> EncodeResult<EncodedCodeBlock> {
    let cleanup_bitplane = u8::from(target_coding_passes >= 2 && total_bitplanes > 1);
    core::try_encode_code_block_set_view_in_workspace(
        coefficients,
        total_bitplanes,
        cleanup_bitplane,
        target_coding_passes,
        false,
        workspace,
    )
}
#[cfg(test)]
mod legacy;
#[cfg(test)]
pub(crate) use legacy::{encode_code_block, encode_code_block_view, encode_code_block_with_passes};
