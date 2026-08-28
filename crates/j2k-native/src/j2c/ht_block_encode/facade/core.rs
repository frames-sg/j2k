// SPDX-License-Identifier: MIT OR Apache-2.0
use alloc::vec::Vec;

use super::validate_set_request;
use crate::j2c::bitplane_encode::EncodedCodeBlock;
use crate::j2c::coefficient_view::CoefficientBlockView;
use crate::j2c::encode::allocation::try_untracked_vec;
use crate::j2c::ht_block_encode::allocation::ht_worker_allocation;
use crate::j2c::ht_block_encode::cleanup::{
    max_nonzero_magnitude_view, try_encode_cleanup_segment_from_view_in_workspace,
};
use crate::j2c::ht_block_encode::distortion::pass_distortion_deltas;
use crate::j2c::ht_block_encode::refinement::{
    try_encode_refinement_segment_view, EncodedRefinementSegment,
};
use crate::j2c::ht_block_encode::workspace::HtEncodeWorkspace;
use crate::{EncodeError, EncodeResult};

pub(super) fn try_encode_code_block_set_view_in_workspace(
    coefficients: CoefficientBlockView<'_, i32>,
    total_bitplanes: u8,
    cleanup_bitplane: u8,
    target_coding_passes: u8,
    collect_distortion: bool,
    workspace: &mut HtEncodeWorkspace,
) -> EncodeResult<EncodedCodeBlock> {
    validate_set_request(total_bitplanes, cleanup_bitplane, target_coding_passes)?;
    let allocation = ht_worker_allocation(
        coefficients.width(),
        coefficients.height(),
        target_coding_passes,
    )?;

    let Some(max_magnitude) = max_nonzero_magnitude_view(coefficients) else {
        return Ok(EncodedCodeBlock {
            data: Vec::new(),
            num_coding_passes: 0,
            num_zero_bitplanes: total_bitplanes,
            ht_cleanup_length: 0,
            ht_refinement_length: 0,
            ht_sigprop_length: 0,
            ht_magref_length: 0,
            ht_distortion_deltas: [0.0; 3],
        });
    };

    let block_bitplanes = crate::math::bit_width_u32(max_magnitude);
    if block_bitplanes > total_bitplanes {
        return Err(EncodeError::InvalidInput {
            what: "HTJ2K block magnitude exceeds configured bitplane count",
        });
    }

    let effective_coding_passes = if cleanup_bitplane == 0 {
        1
    } else {
        target_coding_passes
    };
    let missing_msbs = total_bitplanes
        .checked_sub(cleanup_bitplane)
        .and_then(|value| value.checked_sub(1))
        .ok_or(EncodeError::InvalidInput {
            what: "HTJ2K cleanup bitplane is outside the configured bitplane range",
        })?;
    let cleanup = try_encode_cleanup_segment_from_view_in_workspace(
        coefficients,
        missing_msbs,
        total_bitplanes,
        workspace,
    )?;
    if cleanup.len() > allocation.cleanup_bytes {
        return Err(EncodeError::InternalInvariant {
            what: "HTJ2K cleanup segment exceeded its checked bound",
        });
    }
    let ht_cleanup_length =
        u32::try_from(cleanup.len()).map_err(|_| EncodeError::InternalInvariant {
            what: "HTJ2K cleanup segment exceeds u32 length",
        })?;
    let refinement = if effective_coding_passes > 1 {
        try_encode_refinement_segment_view(
            coefficients,
            cleanup_bitplane,
            effective_coding_passes,
            allocation,
        )?
    } else {
        EncodedRefinementSegment::default()
    };
    let ht_refinement_length =
        u32::try_from(refinement.data.len()).map_err(|_| EncodeError::InternalInvariant {
            what: "HTJ2K refinement segment exceeds u32 length",
        })?;
    let combined_len = cleanup.len().checked_add(refinement.data.len()).ok_or(
        EncodeError::ArithmeticOverflow {
            what: "HTJ2K combined block payload",
        },
    )?;
    if combined_len > allocation.output_bytes {
        return Err(EncodeError::InternalInvariant {
            what: "HTJ2K block output exceeded its checked bound",
        });
    }
    let mut data = try_untracked_vec(combined_len, "HTJ2K block output")?;
    data.extend_from_slice(&cleanup);
    data.extend_from_slice(&refinement.data);

    Ok(EncodedCodeBlock {
        data,
        num_coding_passes: effective_coding_passes,
        num_zero_bitplanes: missing_msbs,
        ht_cleanup_length,
        ht_refinement_length,
        ht_sigprop_length: refinement.sigprop_length,
        ht_magref_length: refinement.magref_length,
        ht_distortion_deltas: if collect_distortion {
            pass_distortion_deltas(coefficients, cleanup_bitplane, effective_coding_passes)
        } else {
            [0.0; 3]
        },
    })
}
