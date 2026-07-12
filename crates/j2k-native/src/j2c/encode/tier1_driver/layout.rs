// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tier-1 coding-mode and packet-owner layout planning.

use super::super::allocation::checked_add_bytes;
use super::super::tier1_allocation::Tier1PhaseTracker;
use super::super::{
    BlockCodingMode, CodeBlockPacketData, NativeEncodePipelineResult, PreparedEncodeSubband,
    SubbandPrecinct, Vec,
};

pub(super) fn consistent_block_coding_mode(
    prepared_subbands: &[PreparedEncodeSubband],
) -> Result<Option<BlockCodingMode>, &'static str> {
    let mut mode = None;
    for subband in prepared_subbands
        .iter()
        .filter(|subband| !subband.code_blocks.is_empty())
    {
        if mode.is_some_and(|existing| existing != subband.block_coding_mode) {
            return Err("mixed classic and HT Tier-1 subbands are unsupported");
        }
        mode = Some(subband.block_coding_mode);
    }
    Ok(mode)
}

pub(super) fn try_packet_shells(
    prepared_subbands: &[PreparedEncodeSubband],
    prepared_bytes: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
) -> NativeEncodePipelineResult<(Vec<SubbandPrecinct>, usize)> {
    let (mut precincts, outer_bytes) = tracker.try_vec::<SubbandPrecinct>(
        prepared_subbands.len(),
        [prepared_bytes],
        "Tier-1 packet subband owners",
    )?;
    let mut structural_bytes = outer_bytes;
    for subband in prepared_subbands {
        let (code_blocks, code_block_bytes) = tracker.try_vec::<CodeBlockPacketData>(
            subband.code_blocks.len(),
            [prepared_bytes, structural_bytes],
            "Tier-1 packet code-block owners",
        )?;
        structural_bytes = checked_add_bytes(
            structural_bytes,
            code_block_bytes,
            "Tier-1 packet structural owners",
        )?;
        precincts.push(SubbandPrecinct {
            code_blocks,
            num_cbs_x: subband.num_cbs_x,
            num_cbs_y: subband.num_cbs_y,
        });
    }
    Ok((precincts, structural_bytes))
}
