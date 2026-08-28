// SPDX-License-Identifier: MIT OR Apache-2.0

//! HTJ2K segment metadata and fallible per-layer payload construction.

use super::super::super::allocation::{checked_add_bytes, checked_element_bytes};
use super::super::super::tier1_allocation::Tier1PhaseTracker;
use super::super::super::{
    bitplane_encode, BlockCodingMode, CodeBlockPacketData, NativeEncodePipelineError,
    NativeEncodePipelineResult, Vec,
};
use super::{ht_segment_count, ht_target_layer};

mod layout;
use layout::{ht_contribution_layout, HtContributionLayout};

#[derive(Clone, Copy)]
struct HtLayerSelection {
    has_cleanup: bool,
    has_sigprop: bool,
    has_magref: bool,
    payload_len: usize,
    refinement_len: usize,
}

fn ht_layer_selection(
    encoded: &bitplane_encode::EncodedCodeBlock,
    layout: HtContributionLayout,
    segment_layers: &[usize],
    layer_idx: usize,
) -> NativeEncodePipelineResult<HtLayerSelection> {
    // Emit one selected HT set atomically in the layer containing its final
    // selected pass. This keeps the cleanup boundary unambiguous for external
    // decoders while different code-blocks can still enter different layers.
    let set_layer = segment_layers.last().copied();
    let has_cleanup = set_layer == Some(layer_idx);
    let has_sigprop = encoded.num_coding_passes > 1 && set_layer == Some(layer_idx);
    let has_magref = layout.split_refinement && set_layer == Some(layer_idx);
    let refinement_len = usize::from(has_sigprop)
        .checked_mul(layout.sigprop_len)
        .and_then(|bytes| bytes.checked_add(usize::from(has_magref) * layout.magref_len))
        .ok_or_else(|| {
            NativeEncodePipelineError::arithmetic_overflow("HTJ2K layer refinement length overflow")
        })?;
    let payload_len = usize::from(has_cleanup)
        .checked_mul(layout.cleanup_len)
        .and_then(|bytes| bytes.checked_add(refinement_len))
        .ok_or_else(|| {
            NativeEncodePipelineError::arithmetic_overflow(
                "HTJ2K layer contribution payload overflow",
            )
        })?;
    Ok(HtLayerSelection {
        has_cleanup,
        has_sigprop,
        has_magref,
        payload_len,
        refinement_len,
    })
}

fn append_ht_layer_payload(
    encoded: &bitplane_encode::EncodedCodeBlock,
    layout: HtContributionLayout,
    selection: HtLayerSelection,
    data: &mut Vec<u8>,
) -> NativeEncodePipelineResult<u8> {
    let mut passes = 0u8;
    if selection.has_cleanup {
        data.extend_from_slice(encoded.data.get(..layout.cleanup_len).ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant("HTJ2K cleanup segment range invalid")
        })?);
        passes = 1;
    }
    if selection.has_sigprop {
        data.extend_from_slice(
            encoded
                .data
                .get(layout.cleanup_len..layout.sigprop_end)
                .ok_or_else(|| {
                    NativeEncodePipelineError::internal_invariant(
                        "HTJ2K SigProp pass range invalid",
                    )
                })?,
        );
        passes = passes
            .checked_add(if layout.split_refinement {
                1
            } else {
                encoded.num_coding_passes - 1
            })
            .ok_or_else(|| {
                NativeEncodePipelineError::arithmetic_overflow(
                    "HTJ2K packet contribution pass count overflow",
                )
            })?;
    }
    if selection.has_magref {
        data.extend_from_slice(
            encoded
                .data
                .get(layout.sigprop_end..layout.refinement_end)
                .ok_or_else(|| {
                    NativeEncodePipelineError::internal_invariant("HTJ2K MagRef pass range invalid")
                })?,
        );
        passes = passes.checked_add(1).ok_or_else(|| {
            NativeEncodePipelineError::arithmetic_overflow(
                "HTJ2K packet contribution pass count overflow",
            )
        })?;
    }
    Ok(passes)
}

fn ht_packet_contribution(
    encoded: &bitplane_encode::EncodedCodeBlock,
    selection: HtLayerSelection,
    data: Vec<u8>,
    num_coding_passes: u8,
) -> NativeEncodePipelineResult<CodeBlockPacketData> {
    Ok(CodeBlockPacketData {
        data,
        ht_cleanup_length: if selection.has_cleanup {
            encoded.ht_cleanup_length
        } else {
            0
        },
        ht_refinement_length: u32::try_from(selection.refinement_len).map_err(|_| {
            NativeEncodePipelineError::arithmetic_overflow("HTJ2K layer refinement length overflow")
        })?,
        ht_sigprop_length: 0,
        ht_magref_length: 0,
        ht_distortion_deltas: [0.0; 3],
        num_coding_passes,
        classic_segment_lengths: Vec::new(),
        num_zero_bitplanes: encoded.num_zero_bitplanes,
        previously_included: false,
        l_block: 3,
        block_coding_mode: BlockCodingMode::HighThroughput,
    })
}

#[cfg(test)]
pub(in crate::j2c::encode) fn ht_layer_contributions(
    encoded: &bitplane_encode::EncodedCodeBlock,
    num_layers: u8,
    segment_layers: &[usize],
) -> NativeEncodePipelineResult<Vec<CodeBlockPacketData>> {
    let layout = ht_contribution_layout(encoded, num_layers, segment_layers)?;
    let mut contributions = Vec::new();
    contributions
        .try_reserve_exact(layout.layer_count)
        .map_err(|_| crate::EncodeError::HostAllocationFailed {
            what: "HTJ2K layer contribution owners",
            bytes: layout
                .layer_count
                .saturating_mul(core::mem::size_of::<CodeBlockPacketData>()),
        })?;
    for layer_idx in 0..layout.layer_count {
        let selection = ht_layer_selection(encoded, layout, segment_layers, layer_idx)?;
        let mut data = Vec::new();
        data.try_reserve_exact(selection.payload_len).map_err(|_| {
            crate::EncodeError::HostAllocationFailed {
                what: "HTJ2K layer contribution payload",
                bytes: selection.payload_len,
            }
        })?;
        let passes = append_ht_layer_payload(encoded, layout, selection, &mut data)?;
        contributions.push(ht_packet_contribution(encoded, selection, data, passes)?);
    }
    Ok(contributions)
}

pub(in crate::j2c::encode) fn ht_unbudgeted_segment_layers_accounted(
    encoded: &bitplane_encode::EncodedCodeBlock,
    num_layers: u8,
    block_idx: usize,
    block_count: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    retained_live_bytes: usize,
) -> NativeEncodePipelineResult<Vec<usize>> {
    let segment_count = ht_segment_count(encoded);
    let encoded_bytes = encoded.data.capacity();
    let (mut segment_layers, _) = tracker.try_vec::<usize>(
        segment_count,
        [retained_live_bytes, encoded_bytes],
        "HT segment-layer metadata",
    )?;
    if segment_count == 0 {
        return Ok(segment_layers);
    }
    let layer_count = usize::from(num_layers);
    if layer_count == 0 {
        return Err(NativeEncodePipelineError::invalid_input(
            "HTJ2K layer allocation requires at least one quality layer",
        ));
    }
    let target_layer = ht_target_layer(block_idx, block_count, layer_count)?;
    for _ in 0..segment_count {
        segment_layers.push(target_layer);
    }
    Ok(segment_layers)
}

pub(in crate::j2c::encode) fn ht_layer_contributions_accounted(
    encoded: &bitplane_encode::EncodedCodeBlock,
    num_layers: u8,
    segment_layers: &[usize],
    segment_layer_capacity: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
    retained_live_bytes: usize,
) -> NativeEncodePipelineResult<Vec<CodeBlockPacketData>> {
    if segment_layer_capacity < segment_layers.len() {
        return Err(crate::EncodeError::InternalInvariant {
            what: "HT segment-layer capacity is smaller than its length",
        }
        .into());
    }
    let layout = ht_contribution_layout(encoded, num_layers, segment_layers)?;
    let encoded_bytes = encoded.data.capacity();
    let layer_metadata_bytes =
        checked_element_bytes::<usize>(segment_layer_capacity, "HT segment-layer metadata")?;
    let (mut contributions, contribution_owner_bytes) = tracker.try_vec::<CodeBlockPacketData>(
        layout.layer_count,
        [retained_live_bytes, encoded_bytes, layer_metadata_bytes],
        "HT layer contribution owners",
    )?;
    let mut contribution_payload_bytes = 0usize;
    for layer_idx in 0..layout.layer_count {
        let selection = ht_layer_selection(encoded, layout, segment_layers, layer_idx)?;
        let (mut data, data_bytes) = tracker.try_vec::<u8>(
            selection.payload_len,
            [
                retained_live_bytes,
                encoded_bytes,
                layer_metadata_bytes,
                contribution_owner_bytes,
                contribution_payload_bytes,
            ],
            "HT layer contribution payload",
        )?;
        let passes = append_ht_layer_payload(encoded, layout, selection, &mut data)?;
        contribution_payload_bytes = checked_add_bytes(
            contribution_payload_bytes,
            data_bytes,
            "HT layer contribution payload graph",
        )?;
        contributions.push(ht_packet_contribution(encoded, selection, data, passes)?);
    }
    Ok(contributions)
}
