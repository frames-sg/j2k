// SPDX-License-Identifier: MIT OR Apache-2.0

//! Original JPEG 2000 code-block packet contributions.

use super::{
    code_block_is_read, decode_num_coding_passes, read_lblock_increment, read_segment_length,
    resolve_code_block_inclusion, PacketResult, PacketSkips, SegmentSink, MAX_BITPLANE_COUNT,
};
use crate::j2c::build::{CodeBlock, CodeBlockCoding, Segment};
use crate::j2c::codestream::{CodeBlockStyle, ComponentInfo};
use crate::j2c::decode::DecompositionStorage;
use crate::j2c::progression::ProgressionData;
use crate::reader::BitReader;

#[expect(
    clippy::too_many_arguments,
    reason = "the packet, its precinct's offset among the records, and the parser state it fills stay explicit"
)]
pub(super) fn resolve_segments(
    sub_band_idx: usize,
    progression_data: &ProgressionData,
    precinct_offset: usize,
    reader: &mut BitReader<'_>,
    storage: &mut DecompositionStorage<'_>,
    component_info: &ComponentInfo,
    num_layers: u8,
    skips: &mut PacketSkips,
) -> PacketResult {
    let sub_band = &storage.sub_bands[sub_band_idx];
    let precinct_range = sub_band.precincts.clone();
    let Some(code_block_range) = storage.precincts[precinct_range.clone()]
        .get(precinct_offset)
        .map(|precinct| precinct.code_blocks.clone())
    else {
        lwarn!("progression data yielded invalid precinct index");
        return Err("classic packet references a missing precinct");
    };

    for code_block_idx in code_block_range {
        let precinct = &mut storage.precincts[precinct_range.clone()][precinct_offset];
        let code_block = &mut storage.code_blocks[code_block_idx];
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

        // Only code-blocks the decode reads have layer records; the others'
        // contributions are checked the same way and then skipped.
        let read = code_block_is_read(storage, sub_band_idx, &storage.code_blocks[code_block_idx]);
        #[cfg(feature = "std")]
        let read = read && !skips.discard_contributions;
        let layer_num = usize::from(progression_data.layer_num);
        let code_block = &mut storage.code_blocks[code_block_idx];
        let layer = if read {
            Some(
                storage.layers[code_block.layers.clone()]
                    .get_mut(layer_num)
                    .ok_or("classic packet references a missing quality layer")?,
            )
        } else if layer_num < usize::from(num_layers) {
            None
        } else {
            return Err("classic packet references a missing quality layer");
        };
        let added_coding_passes = decode_num_coding_passes(reader)
            .ok_or("classic coding-pass count is truncated or invalid")?;
        code_block.l_block = code_block
            .l_block
            .checked_add(
                read_lblock_increment(reader).ok_or("classic Lblock increment is truncated")?,
            )
            .ok_or("classic Lblock increment overflows")?;

        if let Some(layer) = layer {
            let start = storage.segments.len();
            parse_contribution_lengths(
                reader,
                added_coding_passes,
                None,
                code_block,
                component_info.code_block_style(),
                &mut SegmentSink::Record {
                    segments: &mut storage.segments,
                    structural_workspace_bytes: storage.structural_workspace_bytes,
                    packet_workspace_error: &mut storage.packet_workspace_error,
                },
            )?;
            layer.segments = Some(start..storage.segments.len());
        } else {
            let mut bytes = 0;
            parse_contribution_lengths(
                reader,
                added_coding_passes,
                None,
                code_block,
                component_info.code_block_style(),
                &mut SegmentSink::Skip(&mut bytes),
            )?;
            skips.push(code_block_idx, bytes)?;
        }
        code_block.coding = Some(CodeBlockCoding::Classic);
        code_block.non_empty_layer_count = code_block
            .non_empty_layer_count
            .checked_add(1)
            .ok_or("classic non-empty layer count overflows")?;
    }
    Ok(())
}

pub(super) fn parse_contribution_lengths(
    reader: &mut BitReader<'_>,
    added_coding_passes: u8,
    mut first_length: Option<u32>,
    code_block: &mut CodeBlock,
    style: CodeBlockStyle,
    sink: &mut SegmentSink<'_, '_>,
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
        sink.push(Segment {
            idx,
            data_length: length,
            coding_pases: coding_passes,
            data: &[],
        })
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
