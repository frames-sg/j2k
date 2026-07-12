// SPDX-License-Identifier: MIT OR Apache-2.0

//! Serial classic Tier-1 handoff for prepared i64 coefficient owners.

use super::super::super::allocation::checked_add_bytes;
use super::super::super::tier1_allocation::Tier1PhaseTracker;
use super::super::super::{
    bitplane_encode, NativeEncodePipelineError, NativeEncodePipelineResult,
    PreparedCodeBlockCoefficients, PreparedEncodeSubband, SubbandPrecinct,
};
use super::super::output::push_packet_block;

pub(in crate::j2c::encode::tier1_driver) fn encode_classic_i64_direct(
    prepared_subbands: &[PreparedEncodeSubband],
    precincts: &mut [SubbandPrecinct],
    prepared_bytes: usize,
    packet_structural_bytes: usize,
    tracker: &mut Tier1PhaseTracker<'_, '_>,
) -> NativeEncodePipelineResult<()> {
    let mut packet_payload_bytes = 0usize;
    for (subband, precinct) in prepared_subbands.iter().zip(precincts) {
        for block in &subband.code_blocks {
            let worker = bitplane_encode::classic_worker_allocation(
                block.width as usize,
                block.height as usize,
                subband.total_bitplanes,
            )?;
            tracker.check(
                [
                    prepared_bytes,
                    packet_structural_bytes,
                    packet_payload_bytes,
                    worker.output_bytes,
                    worker.scratch_bytes,
                ],
                "classic i64 Tier-1 worker wave",
            )?;
            let encoded = match &block.coefficients {
                PreparedCodeBlockCoefficients::I32(values) => {
                    bitplane_encode::try_encode_code_block(
                        values,
                        block.width,
                        block.height,
                        subband.sub_band_type,
                        subband.total_bitplanes,
                    )?
                }
                PreparedCodeBlockCoefficients::I64(values) => {
                    bitplane_encode::try_encode_code_block_i64(
                        values,
                        block.width,
                        block.height,
                        subband.sub_band_type,
                        subband.total_bitplanes,
                    )?
                }
                PreparedCodeBlockCoefficients::Empty => {
                    return Err(NativeEncodePipelineError::internal_invariant(
                        "classic Tier-1 coefficient storage is missing",
                    ));
                }
            };
            packet_payload_bytes = checked_add_bytes(
                packet_payload_bytes,
                encoded.data.capacity(),
                "classic i64 Tier-1 packet payload",
            )?;
            tracker.check(
                [
                    prepared_bytes,
                    packet_structural_bytes,
                    packet_payload_bytes,
                ],
                "classic i64 Tier-1 output",
            )?;
            push_packet_block(precinct, encoded, subband.block_coding_mode)?;
        }
    }
    Ok(())
}
