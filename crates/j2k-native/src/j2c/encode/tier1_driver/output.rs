// SPDX-License-Identifier: MIT OR Apache-2.0

//! Move-only adaptation from Tier-1 results into packet-owned code blocks.

use super::super::{
    bitplane_encode, BlockCodingMode, CodeBlockPacketData, EncodedJ2kCodeBlock,
    NativeEncodePipelineError, NativeEncodePipelineResult, PreparedEncodeSubband, SubbandPrecinct,
    Vec,
};

mod validation;
pub(super) use validation::{
    validate_classic_batch_outputs, validate_ht_batch_outputs, validated_classic_output,
    validated_ht_output,
};

pub(super) fn move_public_ht_outputs(
    encoded: Vec<crate::EncodedHtJ2kCodeBlock>,
    prepared_subbands: &[PreparedEncodeSubband],
    precincts: &mut [SubbandPrecinct],
) -> NativeEncodePipelineResult<()> {
    move_native_output_iter(
        encoded
            .into_iter()
            .map(ht_encoded_code_block_from_accelerator),
        prepared_subbands,
        precincts,
        BlockCodingMode::HighThroughput,
    )
}

pub(super) fn move_public_classic_outputs(
    encoded: Vec<EncodedJ2kCodeBlock>,
    prepared_subbands: &[PreparedEncodeSubband],
    precincts: &mut [SubbandPrecinct],
) -> NativeEncodePipelineResult<()> {
    move_native_output_iter(
        encoded.into_iter().map(encoded_code_block_from_accelerator),
        prepared_subbands,
        precincts,
        BlockCodingMode::Classic,
    )
}

fn move_native_output_iter(
    mut encoded: impl Iterator<Item = bitplane_encode::EncodedCodeBlock>,
    prepared_subbands: &[PreparedEncodeSubband],
    precincts: &mut [SubbandPrecinct],
    expected_mode: BlockCodingMode,
) -> NativeEncodePipelineResult<()> {
    for (subband, precinct) in prepared_subbands.iter().zip(precincts) {
        if !subband.code_blocks.is_empty() && subband.block_coding_mode != expected_mode {
            return Err(NativeEncodePipelineError::internal_invariant(
                "Tier-1 output coding mode mismatch",
            ));
        }
        for _ in &subband.code_blocks {
            push_packet_block(
                precinct,
                encoded.next().ok_or_else(|| {
                    NativeEncodePipelineError::internal_invariant(
                        "encoded code-block count mismatch",
                    )
                })?,
                subband.block_coding_mode,
            )?;
        }
    }
    if encoded.next().is_some() {
        return Err(NativeEncodePipelineError::internal_invariant(
            "encoded code-block count mismatch",
        ));
    }
    Ok(())
}

pub(super) fn move_native_result_iter(
    encoded: impl Iterator<Item = crate::EncodeResult<bitplane_encode::EncodedCodeBlock>>,
    prepared_subbands: &[PreparedEncodeSubband],
    precincts: &mut [SubbandPrecinct],
    expected_mode: BlockCodingMode,
) -> NativeEncodePipelineResult<()> {
    let mut encoded = encoded;
    for (subband, precinct) in prepared_subbands.iter().zip(precincts) {
        if !subband.code_blocks.is_empty() && subband.block_coding_mode != expected_mode {
            return Err(NativeEncodePipelineError::internal_invariant(
                "Tier-1 output coding mode mismatch",
            ));
        }
        for _ in &subband.code_blocks {
            push_packet_block(
                precinct,
                encoded.next().ok_or_else(|| {
                    NativeEncodePipelineError::internal_invariant(
                        "encoded code-block count mismatch",
                    )
                })??,
                subband.block_coding_mode,
            )?;
        }
    }
    if encoded.next().is_some() {
        return Err(NativeEncodePipelineError::internal_invariant(
            "encoded code-block count mismatch",
        ));
    }
    Ok(())
}

pub(super) fn push_packet_block(
    precinct: &mut SubbandPrecinct,
    encoded: bitplane_encode::EncodedCodeBlock,
    block_coding_mode: BlockCodingMode,
) -> NativeEncodePipelineResult<()> {
    if precinct.code_blocks.len() == precinct.code_blocks.capacity() {
        return Err(crate::EncodeError::InternalInvariant {
            what: "Tier-1 packet code-block owner exceeded its planned capacity",
        }
        .into());
    }
    precinct.code_blocks.push(CodeBlockPacketData {
        data: encoded.data,
        ht_cleanup_length: if block_coding_mode == BlockCodingMode::HighThroughput {
            encoded.ht_cleanup_length
        } else {
            0
        },
        ht_refinement_length: if block_coding_mode == BlockCodingMode::HighThroughput {
            encoded.ht_refinement_length
        } else {
            0
        },
        num_coding_passes: encoded.num_coding_passes,
        classic_segment_lengths: Vec::new(),
        num_zero_bitplanes: encoded.num_zero_bitplanes,
        previously_included: false,
        l_block: 3,
        block_coding_mode,
    });
    Ok(())
}

pub(super) fn ht_encoded_code_block_from_accelerator(
    encoded: crate::EncodedHtJ2kCodeBlock,
) -> bitplane_encode::EncodedCodeBlock {
    bitplane_encode::EncodedCodeBlock {
        data: encoded.data,
        num_coding_passes: encoded.num_coding_passes,
        num_zero_bitplanes: encoded.num_zero_bitplanes,
        ht_cleanup_length: encoded.cleanup_length,
        ht_refinement_length: encoded.refinement_length,
    }
}

pub(super) fn encoded_code_block_from_accelerator(
    encoded: EncodedJ2kCodeBlock,
) -> bitplane_encode::EncodedCodeBlock {
    bitplane_encode::EncodedCodeBlock {
        data: encoded.data,
        num_coding_passes: encoded.number_of_coding_passes,
        num_zero_bitplanes: encoded.missing_bit_planes,
        ht_cleanup_length: 0,
        ht_refinement_length: 0,
    }
}
