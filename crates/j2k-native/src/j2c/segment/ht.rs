// SPDX-License-Identifier: MIT OR Apache-2.0

//! HTJ2K packet contributions and codeword-segment lengths.

use super::{
    classic, decode_num_coding_passes, push_segment_or_record_error, read_lblock_increment,
    read_segment_length, resolve_code_block_inclusion, PacketResult, MAX_BITPLANE_COUNT,
};
use crate::error::DecodeError;
use crate::j2c::build::{CodeBlock, CodeBlockCoding, Segment};
use crate::j2c::codestream::{CodeBlockStyle, ComponentInfo};
use crate::j2c::decode::DecompositionStorage;
use crate::j2c::progression::ProgressionData;
use crate::reader::BitReader;

pub(super) fn resolve_segments(
    sub_band_dx: usize,
    progression_data: &ProgressionData,
    reader: &mut BitReader<'_>,
    storage: &mut DecompositionStorage<'_>,
    component_info: &ComponentInfo,
) -> PacketResult {
    const MAX_CODING_PASSES: u8 = 1 + 3 * (MAX_BITPLANE_COUNT - 1);

    let sub_band = &storage.sub_bands[sub_band_dx];
    let precincts = &mut storage.precincts[sub_band.precincts.clone()];
    let precinct_index = usize::try_from(progression_data.precinct)
        .map_err(|_| "HT packet precinct index does not fit the platform")?;
    let Some(precinct) = precincts.get_mut(precinct_index) else {
        lwarn!("progression data yielded invalid precinct index");

        return Err("HT packet references a missing precinct");
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
        .ok_or("HT code-block inclusion or zero-bitplane tree is truncated")?;

        if !inclusion.included {
            continue;
        }

        let layer = storage.layers[code_block.layers.clone()]
            .get_mut(usize::from(progression_data.layer_num))
            .ok_or("HT packet references a missing quality layer")?;

        if layer.segments.is_some() {
            return Err("HT quality layer already owns segment metadata");
        }

        let raw_num_passes = decode_num_coding_passes(reader)
            .ok_or("HT coding-pass count is truncated or invalid")?;

        if raw_num_passes > MAX_CODING_PASSES {
            return Err("HT coding-pass count exceeds the supported maximum");
        }

        ltrace!("HT raw number of coding passes: {}", raw_num_passes);

        let start = storage.segments.len();
        if inclusion.included_first_time
            && (code_block.number_of_coding_passes != 0
                || code_block.ht_total_coding_passes != 0
                || code_block.ht_first_cleanup_pass.is_some()
                || code_block.ht_selected_set.is_some()
                || code_block.coding.is_some()
                || code_block.non_empty_layer_count != 0)
        {
            return Err("first HT inclusion has stale coding-pass state");
        }

        parse_segment_lengths(
            reader,
            raw_num_passes,
            component_info.code_block_style(),
            code_block,
            &mut storage.segments,
            storage.structural_workspace_bytes,
            &mut storage.packet_workspace_error,
        )?;

        let end = storage.segments.len();
        layer.segments = Some(start..end);
        code_block.non_empty_layer_count = code_block
            .non_empty_layer_count
            .checked_add(1)
            .ok_or("HT non-empty layer count overflows")?;
    }

    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "the ordered HT packet contribution state machine stays cohesive across placeholder, cleanup, and refinement passes"
)]
fn parse_segment_lengths(
    reader: &mut BitReader<'_>,
    raw_num_passes: u8,
    style: CodeBlockStyle,
    code_block: &mut CodeBlock,
    segments: &mut alloc::vec::Vec<Segment<'_>>,
    structural_workspace_bytes: usize,
    packet_workspace_error: &mut Option<DecodeError>,
) -> PacketResult {
    const MAX_CODING_PASSES: u8 = 1 + 3 * (MAX_BITPLANE_COUNT - 1);

    let total_after = code_block
        .ht_total_coding_passes
        .checked_add(raw_num_passes)
        .ok_or("HT cumulative coding-pass count overflows")?;
    if total_after > MAX_CODING_PASSES {
        return Err("HT cumulative coding-pass count exceeds the supported maximum");
    }

    code_block.l_block = code_block
        .l_block
        .checked_add(read_lblock_increment(reader).ok_or("HT Lblock increment is truncated")?)
        .ok_or("HT Lblock increment overflows")?;

    if code_block.coding == Some(CodeBlockCoding::Classic) {
        return classic::parse_contribution_lengths(
            reader,
            raw_num_passes,
            None,
            code_block,
            style,
            segments,
            structural_workspace_bytes,
            packet_workspace_error,
        );
    }

    let mut push_segment = |idx: u8, coding_passes: u8, data_length: u32| -> PacketResult {
        push_segment_or_record_error(
            segments,
            structural_workspace_bytes,
            packet_workspace_error,
            Segment {
                idx,
                coding_pases: coding_passes,
                data_length,
                data: &[],
            },
        )
        .ok_or("HT segment metadata allocation failed")
    };

    if code_block.ht_first_cleanup_pass.is_none() {
        let refinement_passes = total_after.saturating_sub(1) % 3;
        let possible_cleanup_passes = raw_num_passes.saturating_sub(refinement_passes);
        let first_contribution_passes = if possible_cleanup_passes == 0 {
            raw_num_passes
        } else {
            possible_cleanup_passes
        };
        let mut cleanup_length =
            read_segment_length(reader, code_block.l_block, first_contribution_passes)?;

        if style.allows_mixed_block_coding() && code_block.coding.is_none() {
            let candidate_bits = code_block
                .l_block
                .checked_add(first_contribution_passes.ilog2())
                .ok_or("mixed segment-length field width overflows")?;
            let classic_bits = code_block
                .l_block
                .checked_add(raw_num_passes.ilog2())
                .ok_or("mixed classic segment-length field width overflows")?;
            let mut bits_read = candidate_bits;
            let possible_ht_cleanup = possible_cleanup_passes != 0;
            let ht_discriminator = possible_ht_cleanup
                && code_block.l_block > 3
                && cleanup_length > 1
                && candidate_bits <= 32
                && cleanup_length & (1_u32 << (candidate_bits - 1)) == 0;

            while !ht_discriminator && bits_read < classic_bits {
                cleanup_length = cleanup_length
                    .checked_shl(1)
                    .ok_or("mixed classic segment length overflows")?
                    | reader
                        .read_bits_with_stuffing(1)
                        .ok_or("mixed classic segment length is truncated")?;
                bits_read += 1;
            }

            if cleanup_length != 0 && !ht_discriminator {
                code_block.coding = Some(CodeBlockCoding::Classic);
                code_block.number_of_coding_passes = code_block.ht_total_coding_passes;
                return classic::parse_contribution_lengths(
                    reader,
                    raw_num_passes,
                    Some(cleanup_length),
                    code_block,
                    style,
                    segments,
                    structural_workspace_bytes,
                    packet_workspace_error,
                );
            }
            if ht_discriminator {
                code_block.coding = Some(CodeBlockCoding::HighThroughput);
            }
        } else {
            code_block.coding = Some(CodeBlockCoding::HighThroughput);
        }

        if cleanup_length == 0 {
            if !style.allows_mixed_block_coding() && first_contribution_passes < raw_num_passes {
                let extra_bits = raw_num_passes
                    .ilog2()
                    .checked_sub(first_contribution_passes.ilog2())
                    .ok_or("HT placeholder length width is inconsistent")?;
                if extra_bits != 0 {
                    cleanup_length = reader
                        .read_bits_with_stuffing(
                            u8::try_from(extra_bits)
                                .map_err(|_| "HT placeholder length field is too wide")?,
                        )
                        .ok_or("HT placeholder length is truncated")?;
                }
            }
            if cleanup_length != 0 {
                return Err("HT placeholder contribution has a non-zero length");
            }
            code_block.ht_total_coding_passes = total_after;
            return Ok(());
        }

        if !(2..65_535).contains(&cleanup_length) {
            return Err("HT cleanup segment length is invalid");
        }

        let cleanup_pass = code_block
            .ht_total_coding_passes
            .checked_add(possible_cleanup_passes)
            .and_then(|value| value.checked_sub(1))
            .ok_or("HT first cleanup pass index overflows")?;
        if cleanup_pass % 3 != 0 {
            return Err("HT first cleanup pass is not aligned to an HT set");
        }
        let placeholder_bitplanes = cleanup_pass / 3;
        code_block.missing_bit_planes = code_block
            .missing_bit_planes
            .checked_add(placeholder_bitplanes)
            .ok_or("HT placeholder bit-plane count overflows")?;
        code_block.ht_first_cleanup_pass = Some(cleanup_pass);
        code_block.ht_selected_set = Some(0);
        code_block.number_of_coding_passes = 1;
        push_segment(0, possible_cleanup_passes, cleanup_length)?;
        ltrace!("HT cleanup length {}", cleanup_length);

        if refinement_passes != 0 {
            let refinement_length =
                read_segment_length(reader, code_block.l_block, refinement_passes)?;
            if refinement_length >= 2_047 {
                return Err("HT refinement segment length is invalid");
            }
            push_segment(1, refinement_passes, refinement_length)?;
            code_block.number_of_coding_passes = 1 + refinement_passes;
            ltrace!("HT refinement length {}", refinement_length);
        }

        code_block.ht_total_coding_passes = total_after;
        return Ok(());
    }

    let first_cleanup_pass = code_block
        .ht_first_cleanup_pass
        .ok_or("HT cleanup state is inconsistent")?;
    let mut next_pass = code_block.ht_total_coding_passes;
    let mut remaining = raw_num_passes;
    while remaining != 0 {
        let relative_pass = next_pass
            .checked_sub(first_cleanup_pass)
            .ok_or("HT pass precedes the first cleanup pass")?;
        let pass_in_set = relative_pass % 3;
        let set_index = relative_pass / 3;
        let is_cleanup = pass_in_set == 0;
        let available_in_segment = if pass_in_set == 1 { 2 } else { 1 };
        let contribution_passes = remaining.min(available_in_segment);
        let length = read_segment_length(reader, code_block.l_block, contribution_passes)?;
        let segment_index = set_index
            .checked_mul(2)
            .and_then(|value| value.checked_add(u8::from(!is_cleanup)))
            .ok_or("HT segment index overflows")?;

        if is_cleanup {
            if length == 1 || length >= 65_535 {
                return Err("HT cleanup segment length is invalid");
            }
            if length != 0 {
                let previous_set = code_block
                    .ht_selected_set
                    .ok_or("HT selected-set state is inconsistent")?;
                code_block.missing_bit_planes = code_block
                    .missing_bit_planes
                    .checked_add(
                        set_index
                            .checked_sub(previous_set)
                            .ok_or("HT selected-set order is invalid")?,
                    )
                    .ok_or("HT selected bit-plane count overflows")?;
                code_block.ht_selected_set = Some(set_index);
                code_block.number_of_coding_passes = 1;
                push_segment(segment_index, contribution_passes, length)?;
                ltrace!("HT cleanup length {}", length);
            }
        } else {
            if length >= 2_047 {
                return Err("HT refinement segment length is invalid");
            }
            if code_block.ht_selected_set == Some(set_index) {
                push_segment(segment_index, contribution_passes, length)?;
                code_block.number_of_coding_passes = pass_in_set + contribution_passes;
                ltrace!("HT refinement length {}", length);
            } else if length != 0 {
                return Err("empty HT set has a non-zero refinement length");
            }
        }

        next_pass = next_pass
            .checked_add(contribution_passes)
            .ok_or("HT cumulative coding-pass count overflows")?;
        remaining -= contribution_passes;
    }

    code_block.ht_total_coding_passes = total_after;
    Ok(())
}

#[cfg(test)]
mod tests;
