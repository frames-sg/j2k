// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible borrowed metadata graph for the packetization accelerator boundary.

use super::super::allocation::checked_add_bytes;
use super::super::tier1_allocation::Tier1PhaseTracker;
use super::super::{
    J2kPacketizationCodeBlock, J2kPacketizationResolution, J2kPacketizationSubband,
    NativeEncodePipelineResult, NativeEncodeSession, ResolutionPacket, Vec,
};
use super::public_packetization_block_coding_mode;

pub(super) fn try_public_packetization_resolutions<'a>(
    resolution_packets: &'a [ResolutionPacket],
    session: &NativeEncodeSession<'_>,
    owned_packet_bytes: usize,
) -> NativeEncodePipelineResult<Vec<J2kPacketizationResolution<'a>>> {
    let mut tracker = Tier1PhaseTracker::new(session, owned_packet_bytes);
    let (mut resolutions, resolution_owner_bytes) = tracker
        .try_vec::<J2kPacketizationResolution<'a>>(
            resolution_packets.len(),
            [],
            "packet accelerator resolution metadata",
        )?;
    let mut metadata_bytes = resolution_owner_bytes;

    for resolution in resolution_packets {
        let (mut subbands, subband_owner_bytes) = tracker.try_vec::<J2kPacketizationSubband<'a>>(
            resolution.subbands.len(),
            [metadata_bytes],
            "packet accelerator subband metadata",
        )?;
        let mut code_block_owner_bytes = 0usize;
        for subband in &resolution.subbands {
            let (mut code_blocks, owner_bytes) = tracker.try_vec::<J2kPacketizationCodeBlock<'a>>(
                subband.code_blocks.len(),
                [metadata_bytes, subband_owner_bytes, code_block_owner_bytes],
                "packet accelerator code-block metadata",
            )?;
            for code_block in &subband.code_blocks {
                code_blocks.push(J2kPacketizationCodeBlock {
                    data: &code_block.data,
                    ht_cleanup_length: code_block.ht_cleanup_length,
                    ht_refinement_length: code_block.ht_refinement_length,
                    num_coding_passes: code_block.num_coding_passes,
                    num_zero_bitplanes: code_block.num_zero_bitplanes,
                    previously_included: code_block.previously_included,
                    l_block: code_block.l_block,
                    block_coding_mode: public_packetization_block_coding_mode(
                        code_block.block_coding_mode,
                    ),
                });
            }
            code_block_owner_bytes = checked_add_bytes(
                code_block_owner_bytes,
                owner_bytes,
                "packet accelerator code-block metadata graph",
            )?;
            subbands.push(J2kPacketizationSubband {
                code_blocks,
                num_cbs_x: subband.num_cbs_x,
                num_cbs_y: subband.num_cbs_y,
            });
        }
        metadata_bytes = checked_add_bytes(
            metadata_bytes,
            checked_add_bytes(
                subband_owner_bytes,
                code_block_owner_bytes,
                "packet accelerator resolution metadata",
            )?,
            "packet accelerator metadata graph",
        )?;
        resolutions.push(J2kPacketizationResolution { subbands });
    }
    tracker.check([metadata_bytes], "completed packet accelerator metadata")?;
    Ok(resolutions)
}

#[cfg(test)]
mod tests;
