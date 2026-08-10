// SPDX-License-Identifier: MIT OR Apache-2.0

//! Original JPEG 2000 code-block packet contributions.

use super::{
    decode_num_coding_passes, push_segment_or_record_error, read_lblock_increment,
    read_segment_length, resolve_code_block_inclusion, PacketResult, MAX_BITPLANE_COUNT,
};
use crate::error::DecodeError;
use crate::j2c::build::{CodeBlock, CodeBlockCoding, Segment};
use crate::j2c::codestream::{CodeBlockStyle, ComponentInfo};
use crate::j2c::decode::DecompositionStorage;
use crate::j2c::progression::ProgressionData;
use crate::reader::BitReader;

pub(super) fn resolve_segments(
    sub_band_idx: usize,
    progression_data: &ProgressionData,
    reader: &mut BitReader<'_>,
    storage: &mut DecompositionStorage<'_>,
    component_info: &ComponentInfo,
) -> PacketResult {
    let sub_band = &storage.sub_bands[sub_band_idx];
    let precincts = &mut storage.precincts[sub_band.precincts.clone()];
    let precinct_index = usize::try_from(progression_data.precinct)
        .map_err(|_| "classic packet precinct index does not fit the platform")?;
    let Some(precinct) = precincts.get_mut(precinct_index) else {
        lwarn!("progression data yielded invalid precinct index");
        return Err("classic packet references a missing precinct");
    };
    let code_blocks = &mut storage.code_blocks[precinct.code_blocks.clone()];

    for code_block in code_blocks {
        let inclusion = resolve_code_block_inclusion(
            code_block,
            precinct,
            progression_data,
            reader,
            &mut storage.tag_tree_nodes,
        )
        .ok_or("classic code-block inclusion or zero-bitplane tree is truncated")?;
        if !inclusion.included {
            continue;
        }

        let layer = storage.layers[code_block.layers.clone()]
            .get_mut(usize::from(progression_data.layer_num))
            .ok_or("classic packet references a missing quality layer")?;
        let added_coding_passes = decode_num_coding_passes(reader)
            .ok_or("classic coding-pass count is truncated or invalid")?;
        code_block.l_block = code_block
            .l_block
            .checked_add(
                read_lblock_increment(reader).ok_or("classic Lblock increment is truncated")?,
            )
            .ok_or("classic Lblock increment overflows")?;

        let start = storage.segments.len();
        parse_contribution_lengths(
            reader,
            added_coding_passes,
            None,
            code_block,
            component_info.code_block_style(),
            &mut storage.segments,
            storage.structural_workspace_bytes,
            &mut storage.packet_workspace_error,
        )?;
        layer.segments = Some(start..storage.segments.len());
        code_block.coding = Some(CodeBlockCoding::Classic);
        code_block.non_empty_layer_count = code_block
            .non_empty_layer_count
            .checked_add(1)
            .ok_or("classic non-empty layer count overflows")?;
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "packet parsing keeps the code-block state, bounded segment owner, and input cursor explicit"
)]
pub(super) fn parse_contribution_lengths(
    reader: &mut BitReader<'_>,
    added_coding_passes: u8,
    mut first_length: Option<u32>,
    code_block: &mut CodeBlock,
    style: CodeBlockStyle,
    segments: &mut alloc::vec::Vec<Segment<'_>>,
    structural_workspace_bytes: usize,
    packet_workspace_error: &mut Option<DecodeError>,
) -> PacketResult {
    const MAX_CODING_PASSES: u8 = 1 + 3 * (MAX_BITPLANE_COUNT - 1);

    let previous_passes = code_block.number_of_coding_passes;
    let cumulative_passes = previous_passes
        .checked_add(added_coding_passes)
        .ok_or("classic cumulative coding-pass count overflows")?;
    if cumulative_passes > MAX_CODING_PASSES {
        return Err("classic cumulative coding-pass count exceeds the supported maximum");
    }

    let segment_idx = |pass_idx: u8| {
        if style.termination_on_each_pass {
            pass_idx
        } else if style.selective_arithmetic_coding_bypass {
            segment_idx_for_bypass(pass_idx)
        } else {
            code_block.non_empty_layer_count
        }
    };
    let mut push_segment = |idx: u8, coding_passes: u8| -> PacketResult {
        let length = if let Some(length) = first_length.take() {
            length
        } else {
            read_segment_length(reader, code_block.l_block, coding_passes)?
        };
        push_segment_or_record_error(
            segments,
            structural_workspace_bytes,
            packet_workspace_error,
            Segment {
                idx,
                data_length: length,
                coding_pases: coding_passes,
                data: &[],
            },
        )
        .ok_or("classic segment metadata allocation failed")
    };

    let mut last_segment = segment_idx(previous_passes);
    let mut passes_in_segment = 0;
    for coding_pass in previous_passes..cumulative_passes {
        let segment = segment_idx(coding_pass);
        if segment == last_segment {
            passes_in_segment += 1;
        } else {
            push_segment(last_segment, passes_in_segment)?;
            last_segment = segment;
            passes_in_segment = 1;
        }
    }
    if passes_in_segment != 0 {
        push_segment(last_segment, passes_in_segment)?;
    }
    if first_length.is_some() {
        return Err("classic contribution did not consume its prefetched length");
    }
    code_block.number_of_coding_passes = cumulative_passes;
    Ok(())
}

fn segment_idx_for_bypass(pass_idx: u8) -> u8 {
    if pass_idx < 10 {
        0
    } else {
        1 + (2 * ((pass_idx - 10) / 3)) + u8::from(((pass_idx - 10) % 3) == 2)
    }
}
