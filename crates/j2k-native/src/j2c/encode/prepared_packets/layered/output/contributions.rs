// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-block classic and HT contribution ownership handoff.

use super::super::super::super::allocation::{checked_add_bytes, checked_element_bytes};
use super::super::super::super::tier1_allocation::{segmented_block_ownership, Tier1PhaseTracker};
use super::super::super::super::{
    classic_layer_contributions_accounted, ht_layer_contributions_accounted, CodeBlockPacketData,
    LayeredPreparedBlock, NativeEncodePipelineResult, Vec,
};
use super::super::ownership::checked_sum;

#[derive(Debug, Clone, Copy)]
pub(super) struct ContributionOwners {
    pub(super) fixed: usize,
    pub(super) current_output_bytes: usize,
    pub(super) local_subband_bytes: usize,
    pub(super) local_layer_packet_bytes: usize,
}

pub(super) fn build_block_contributions(
    block: LayeredPreparedBlock,
    num_layers: u8,
    owners: ContributionOwners,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
) -> NativeEncodePipelineResult<Vec<CodeBlockPacketData>> {
    match block {
        LayeredPreparedBlock::Classic {
            encoded,
            segment_layers,
        } => {
            let current_block_bytes = checked_add_bytes(
                segmented_block_ownership(&encoded)?,
                checked_element_bytes::<usize>(
                    segment_layers.capacity(),
                    "classic segment-layer metadata",
                )?,
                "classic layered block",
            )?;
            classic_layer_contributions_accounted(
                &encoded,
                num_layers,
                &segment_layers,
                segment_layers.capacity(),
                tracker,
                contribution_retained_bytes(
                    owners,
                    current_block_bytes,
                    "classic contribution retained owners",
                    "classic layered block ownership underflowed",
                )?,
            )
        }
        LayeredPreparedBlock::HighThroughput {
            encoded,
            segment_layers,
        } => {
            let current_block_bytes = checked_add_bytes(
                encoded.data.capacity(),
                checked_element_bytes::<usize>(
                    segment_layers.capacity(),
                    "HT segment-layer metadata",
                )?,
                "HT layered block",
            )?;
            ht_layer_contributions_accounted(
                &encoded,
                num_layers,
                &segment_layers,
                segment_layers.capacity(),
                tracker,
                contribution_retained_bytes(
                    owners,
                    current_block_bytes,
                    "HT contribution retained owners",
                    "HT layered block ownership underflowed",
                )?,
            )
        }
    }
}

fn contribution_retained_bytes(
    owners: ContributionOwners,
    current_block_bytes: usize,
    what: &'static str,
    underflow: &'static str,
) -> NativeEncodePipelineResult<usize> {
    Ok(checked_sum(
        [
            owners
                .fixed
                .checked_sub(current_block_bytes)
                .ok_or(crate::EncodeError::InternalInvariant { what: underflow })?,
            owners.current_output_bytes,
            owners.local_subband_bytes,
            owners.local_layer_packet_bytes,
        ],
        what,
    )?)
}
