// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared layer arithmetic with separate classic and HT contribution owners.

use super::super::{bitplane_encode, NativeEncodePipelineError, NativeEncodePipelineResult};

mod classic;
pub(in crate::j2c::encode) use classic::{
    classic_layer_contributions_accounted, classic_unbudgeted_segment_layers_accounted,
};
mod ht;
#[cfg(test)]
pub(in crate::j2c::encode) use ht::ht_layer_contributions;
pub(in crate::j2c::encode) use ht::{
    ht_layer_contributions_accounted, ht_unbudgeted_segment_layers_accounted,
};

pub(in crate::j2c::encode) fn ht_segment_count(
    encoded: &bitplane_encode::EncodedCodeBlock,
) -> usize {
    match encoded.num_coding_passes {
        0 => 0,
        1 => 1,
        _ => 2,
    }
}

pub(in crate::j2c::encode) fn ht_segment_rate(
    encoded: &bitplane_encode::EncodedCodeBlock,
    segment_idx: usize,
) -> NativeEncodePipelineResult<u64> {
    match segment_idx {
        0 if encoded.num_coding_passes > 0 => Ok(u64::from(encoded.ht_cleanup_length)),
        1 if encoded.num_coding_passes > 1 => Ok(u64::from(encoded.ht_refinement_length)),
        _ => Err(NativeEncodePipelineError::internal_invariant(
            "HTJ2K segment index out of range",
        )),
    }
}

fn layer_pass_count(
    num_coding_passes: u8,
    layer_count: usize,
    num_layers: u8,
) -> NativeEncodePipelineResult<u8> {
    if num_layers == 0 {
        return Err(NativeEncodePipelineError::invalid_input(
            "quality layer count must be non-zero",
        ));
    }
    let numerator = u32::from(num_coding_passes)
        .checked_mul(
            u32::try_from(layer_count).map_err(|_| {
                NativeEncodePipelineError::arithmetic_overflow("layer index overflow")
            })?,
        )
        .ok_or_else(|| {
            NativeEncodePipelineError::arithmetic_overflow("quality layer pass allocation overflow")
        })?;
    numerator
        .div_ceil(u32::from(num_layers))
        .try_into()
        .map_err(|_| {
            NativeEncodePipelineError::arithmetic_overflow("quality layer pass allocation overflow")
        })
}

fn previous_layer_pass_count(
    num_coding_passes: u8,
    layer_idx: usize,
    num_layers: u8,
) -> NativeEncodePipelineResult<u8> {
    if layer_idx == 0 {
        Ok(0)
    } else {
        layer_pass_count(num_coding_passes, layer_idx, num_layers)
    }
}

fn ht_target_layer(
    block_idx: usize,
    block_count: usize,
    layer_count: usize,
) -> NativeEncodePipelineResult<usize> {
    if block_count == 0 {
        return Err(NativeEncodePipelineError::internal_invariant(
            "HTJ2K layer allocation requires at least one code block",
        ));
    }
    if layer_count == 0 {
        return Err(NativeEncodePipelineError::invalid_input(
            "HTJ2K layer allocation requires at least one quality layer",
        ));
    }
    Ok(block_idx.checked_mul(layer_count).ok_or_else(|| {
        NativeEncodePipelineError::arithmetic_overflow("HTJ2K layer allocation overflow")
    })? / block_count)
}
