// SPDX-License-Identifier: MIT OR Apache-2.0

//! Move-only subband and code-block distribution into precinct owners.

use alloc::vec::Vec;

use super::geometry::{
    precinct_index_for_block, precinct_subband_geometry, subband_precinct_grid,
    PreparedSubbandShape, ResolutionPrecinctGrid, SubbandPrecinctGrid,
};
use super::ownership::{try_destination_vec, try_push_planned, PrecinctSplitAccounting};
use crate::j2c::encode::{
    NativeEncodePipelineError, NativeEncodePipelineResult, PreparedEncodeCodeBlock,
    PreparedEncodeSubband, PreparedResolutionPacket,
};

pub(super) fn distribute_owned_subband(
    subband: PreparedEncodeSubband,
    resolution: u32,
    horizontal_exponent: u8,
    vertical_exponent: u8,
    resolution_grid: ResolutionPrecinctGrid,
    packets: &mut [PreparedResolutionPacket],
    accounting: &mut PrecinctSplitAccounting<'_, '_>,
) -> NativeEncodePipelineResult<()> {
    let shape = PreparedSubbandShape::from(&subband);
    let PreparedEncodeSubband {
        code_blocks,
        preencoded_ht_code_blocks,
        ..
    } = subband;
    let source_code_block_capacity = code_blocks.capacity();
    let source_preencoded_capacity = preencoded_ht_code_blocks.as_ref().map_or(0, Vec::capacity);
    let subband_grid = subband_precinct_grid(
        shape,
        resolution,
        horizontal_exponent,
        vertical_exponent,
        resolution_grid,
    )?;
    let subband_index = packets.first().map_or(0, |packet| packet.subbands.len());
    let distributed_block_count = append_split_subbands(
        packets,
        shape,
        subband_grid,
        resolution_grid,
        preencoded_ht_code_blocks.is_some(),
        accounting,
    )?;

    if distributed_block_count != code_blocks.len() {
        return Err(NativeEncodePipelineError::internal_invariant(
            "precinct code-block grid count mismatch",
        ));
    }
    if preencoded_ht_code_blocks
        .as_ref()
        .is_some_and(|blocks| blocks.len() != code_blocks.len())
    {
        return Err(NativeEncodePipelineError::internal_invariant(
            "preencoded HT subband code-block count mismatch",
        ));
    }

    move_code_blocks_to_precincts(code_blocks, shape, subband_grid, subband_index, packets)?;
    accounting.release_source_capacity::<PreparedEncodeCodeBlock>(
        source_code_block_capacity,
        "source code-block owners",
    )?;

    if let Some(preencoded_blocks) = preencoded_ht_code_blocks {
        move_preencoded_blocks_to_precincts(
            preencoded_blocks,
            shape,
            subband_grid,
            subband_index,
            packets,
        )?;
        accounting.release_source_capacity::<crate::EncodedHtJ2kCodeBlock>(
            source_preencoded_capacity,
            "source preencoded code-block owners",
        )?;
    }
    Ok(())
}

fn append_split_subbands(
    packets: &mut [PreparedResolutionPacket],
    shape: PreparedSubbandShape,
    subband_grid: SubbandPrecinctGrid,
    resolution_grid: ResolutionPrecinctGrid,
    include_preencoded: bool,
    accounting: &mut PrecinctSplitAccounting<'_, '_>,
) -> NativeEncodePipelineResult<usize> {
    let mut distributed_block_count = 0usize;
    for (packet_index, packet) in packets.iter_mut().enumerate() {
        let packet_index = u32::try_from(packet_index).map_err(|_| {
            NativeEncodePipelineError::arithmetic_overflow("precinct packet index exceeds u32")
        })?;
        let precinct_column = packet_index % resolution_grid.columns;
        let precinct_row = packet_index / resolution_grid.columns;
        let geometry =
            precinct_subband_geometry(shape, subband_grid, precinct_column, precinct_row)?;
        let block_count = usize::try_from(
            u64::from(geometry.block_columns)
                .checked_mul(u64::from(geometry.block_rows))
                .ok_or_else(|| {
                    NativeEncodePipelineError::arithmetic_overflow(
                        "precinct code-block count overflow",
                    )
                })?,
        )
        .map_err(|_| {
            NativeEncodePipelineError::arithmetic_overflow(
                "precinct code-block count exceeds usize",
            )
        })?;
        distributed_block_count = distributed_block_count
            .checked_add(block_count)
            .ok_or_else(|| {
                NativeEncodePipelineError::arithmetic_overflow("precinct code-block count overflow")
            })?;
        let code_blocks = try_destination_vec(block_count, accounting, "split code-block owners")?;
        let preencoded_ht_code_blocks = include_preencoded
            .then(|| {
                try_destination_vec(
                    block_count,
                    accounting,
                    "split preencoded code-block owners",
                )
            })
            .transpose()?;
        try_push_planned(
            &mut packet.subbands,
            PreparedEncodeSubband {
                code_blocks,
                preencoded_ht_code_blocks,
                num_cbs_x: geometry.block_columns,
                num_cbs_y: geometry.block_rows,
                code_block_width: shape.code_block_horizontal_span,
                code_block_height: shape.code_block_vertical_span,
                width: geometry.horizontal_span,
                height: geometry.vertical_span,
                sub_band_type: shape.sub_band_type,
                total_bitplanes: shape.total_bitplanes,
                block_coding_mode: shape.block_coding_mode,
                ht_target_coding_passes: shape.ht_target_coding_passes,
            },
        )?;
    }
    Ok(distributed_block_count)
}

fn move_code_blocks_to_precincts(
    code_blocks: Vec<PreparedEncodeCodeBlock>,
    shape: PreparedSubbandShape,
    subband_grid: SubbandPrecinctGrid,
    subband_index: usize,
    packets: &mut [PreparedResolutionPacket],
) -> NativeEncodePipelineResult<()> {
    for (block_index, block) in code_blocks.into_iter().enumerate() {
        let packet_index = precinct_index_for_block(block_index, shape, subband_grid)?;
        let destination = packets
            .get_mut(packet_index)
            .and_then(|packet| packet.subbands.get_mut(subband_index))
            .ok_or_else(|| {
                NativeEncodePipelineError::internal_invariant(
                    "precinct code-block destination missing",
                )
            })?;
        try_push_planned(&mut destination.code_blocks, block)?;
    }
    Ok(())
}

fn move_preencoded_blocks_to_precincts(
    code_blocks: Vec<crate::EncodedHtJ2kCodeBlock>,
    shape: PreparedSubbandShape,
    subband_grid: SubbandPrecinctGrid,
    subband_index: usize,
    packets: &mut [PreparedResolutionPacket],
) -> NativeEncodePipelineResult<()> {
    for (block_index, block) in code_blocks.into_iter().enumerate() {
        let packet_index = precinct_index_for_block(block_index, shape, subband_grid)?;
        let destination = packets
            .get_mut(packet_index)
            .and_then(|packet| packet.subbands.get_mut(subband_index))
            .and_then(|subband| subband.preencoded_ht_code_blocks.as_mut())
            .ok_or_else(|| {
                NativeEncodePipelineError::internal_invariant(
                    "precinct preencoded code-block destination missing",
                )
            })?;
        try_push_planned(destination, block)?;
    }
    Ok(())
}
