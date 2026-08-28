// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded consecutive-set generation for HT post-compression selection.

use alloc::vec::Vec;

use super::try_encode_code_block_set_with_workspace;
use crate::j2c::bitplane_encode::EncodedCodeBlock;
use crate::j2c::coefficient_view::CoefficientBlockView;
use crate::j2c::ht_block_encode::distortion::pass_distortion_deltas;
use crate::j2c::ht_block_encode::workspace::HtEncodeWorkspace;
use crate::{EncodeError, EncodeResult};

pub(crate) fn candidate_cleanup_bitplanes(total_bitplanes: u8) -> EncodeResult<&'static [u8]> {
    match total_bitplanes {
        0 => Err(EncodeError::InvalidInput {
            what: "HTJ2K scalar encoder currently supports 1..=31 bitplanes",
        }),
        1 => Ok(&[0]),
        2 => Ok(&[1]),
        _ => Ok(&[2, 1]),
    }
}

pub(crate) fn code_block_set_distortion_deltas(
    coefficients: &[i32],
    width: u32,
    height: u32,
    cleanup_bitplane: u8,
    num_coding_passes: u8,
) -> EncodeResult<[f64; 3]> {
    let coefficients =
        CoefficientBlockView::try_contiguous(coefficients, width as usize, height as usize)?;
    Ok(pass_distortion_deltas(
        coefficients,
        cleanup_bitplane,
        num_coding_passes,
    ))
}

pub(crate) fn try_encode_code_block_candidate_sets_with_workspace(
    coefficients: &[i32],
    width: u32,
    height: u32,
    total_bitplanes: u8,
    workspace: &mut HtEncodeWorkspace,
) -> EncodeResult<Vec<EncodedCodeBlock>> {
    let cleanup_bitplanes = candidate_cleanup_bitplanes(total_bitplanes)?;
    let mut candidates = Vec::new();
    candidates
        .try_reserve_exact(cleanup_bitplanes.len())
        .map_err(|_| EncodeError::HostAllocationFailed {
            what: "HTJ2K candidate set owners",
            bytes: cleanup_bitplanes
                .len()
                .saturating_mul(core::mem::size_of::<EncodedCodeBlock>()),
        })?;
    for &cleanup_bitplane in cleanup_bitplanes {
        candidates.push(try_encode_code_block_set_with_workspace(
            coefficients,
            width,
            height,
            total_bitplanes,
            cleanup_bitplane,
            if cleanup_bitplane == 0 { 1 } else { 3 },
            workspace,
        )?);
    }
    Ok(candidates)
}

pub(crate) fn truncate_code_block_candidate(
    mut candidate: EncodedCodeBlock,
    num_coding_passes: u8,
) -> EncodeResult<EncodedCodeBlock> {
    if num_coding_passes == 0 || num_coding_passes > candidate.num_coding_passes {
        return Err(EncodeError::InternalInvariant {
            what: "HTJ2K selected candidate pass count is invalid",
        });
    }
    let refinement_length = match num_coding_passes {
        1 => 0,
        2 => candidate.ht_sigprop_length,
        _ => candidate.ht_refinement_length,
    };
    let data_length = candidate
        .ht_cleanup_length
        .checked_add(refinement_length)
        .and_then(|length| usize::try_from(length).ok())
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "HTJ2K selected candidate length",
        })?;
    candidate.data.truncate(data_length);
    candidate.num_coding_passes = num_coding_passes;
    candidate.ht_refinement_length = refinement_length;
    if num_coding_passes < 3 {
        candidate.ht_magref_length = 0;
        candidate.ht_distortion_deltas[2] = 0.0;
    }
    if num_coding_passes < 2 {
        candidate.ht_sigprop_length = 0;
        candidate.ht_distortion_deltas[1] = 0.0;
    }
    Ok(candidate)
}
