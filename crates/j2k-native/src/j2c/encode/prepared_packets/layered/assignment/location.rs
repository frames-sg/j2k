// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checked lookup of layered packet segment-assignment destinations.

use super::super::super::super::{
    ClassicSegmentLocation, HtSegmentLocation, LayeredPreparedBlock, LayeredPreparedPacket,
    NativeEncodePipelineError, NativeEncodePipelineResult,
};

pub(super) fn classic_segment_layer_mut<'a>(
    layered_packets: &'a mut [LayeredPreparedPacket],
    location: &ClassicSegmentLocation,
) -> NativeEncodePipelineResult<&'a mut usize> {
    let block = layered_packets
        .get_mut(location.packet_idx)
        .ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant("classic PCRD packet index mismatch")
        })?
        .subbands
        .get_mut(location.subband_idx)
        .ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant("classic PCRD subband index mismatch")
        })?
        .blocks
        .get_mut(location.block_idx)
        .ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant("classic PCRD block index mismatch")
        })?;
    let LayeredPreparedBlock::Classic { segment_layers, .. } = block else {
        return Err(NativeEncodePipelineError::internal_invariant(
            "classic PCRD assignment referenced HT block",
        ));
    };
    segment_layers.get_mut(location.segment_idx).ok_or_else(|| {
        NativeEncodePipelineError::internal_invariant("classic PCRD segment index mismatch")
    })
}

pub(super) fn ht_segment_layer_mut<'a>(
    layered_packets: &'a mut [LayeredPreparedPacket],
    location: &HtSegmentLocation,
) -> NativeEncodePipelineResult<&'a mut usize> {
    let block = layered_packets
        .get_mut(location.packet_idx)
        .ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant("HTJ2K packet index mismatch")
        })?
        .subbands
        .get_mut(location.subband_idx)
        .ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant("HTJ2K subband index mismatch")
        })?
        .blocks
        .get_mut(location.block_idx)
        .ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant("HTJ2K block index mismatch")
        })?;
    let LayeredPreparedBlock::HighThroughput { segment_layers, .. } = block else {
        return Err(NativeEncodePipelineError::internal_invariant(
            "HTJ2K segment assignment referenced classic block",
        ));
    };
    segment_layers.get_mut(location.segment_idx).ok_or_else(|| {
        NativeEncodePipelineError::internal_invariant("HTJ2K segment index mismatch")
    })
}
