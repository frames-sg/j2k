//! Parsing of layers and their segments, as specified in Annex B.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::mem::size_of;

use super::build::{CodeBlock, Precinct, Segment};
use super::codestream::markers::{EPH, SOP};
use super::codestream::{ComponentInfo, Header};
use super::decode::DecompositionStorage;
use super::progression::ProgressionData;
use super::tag_tree::TagNode;
use super::tile::{Tile, TilePartCursor};
use crate::error::{bail, DecodeError, DecodingError, Result, ValidationError};
use crate::reader::BitReader;
use crate::{try_reserve_decode_elements, DEFAULT_MAX_DECODE_BYTES};

mod classic;
mod ht;

pub(crate) const MAX_BITPLANE_COUNT: u8 = 63;
type PacketResult<T = ()> = core::result::Result<T, &'static str>;

pub(crate) fn parse<'a, 'b>(
    tile: &'b Tile<'a>,
    mut progression_iterator: Box<dyn Iterator<Item = ProgressionData> + '_>,
    header: &Header<'_>,
    storage: &mut DecompositionStorage<'a>,
) -> Result<()> {
    for tile_part in &tile.tile_parts {
        let parsed = tile_part
            .cursor()
            .ok_or("tile-part cursor is inconsistent")
            .and_then(|cursor| {
                parse_inner(
                    cursor,
                    &mut progression_iterator,
                    &tile.component_infos,
                    storage,
                )
            });
        if let Some(error) = storage.packet_workspace_error.take() {
            return Err(error);
        }
        if let Err(context) = parsed {
            if header.strict {
                bail!(DecodingError::PacketParseFailure(context));
            }
        }
    }

    Ok(())
}

fn parse_inner<'a>(
    mut tile_part: TilePartCursor<'_, 'a>,
    progression_iterator: &mut dyn Iterator<Item = ProgressionData>,
    component_infos: &[ComponentInfo],
    storage: &mut DecompositionStorage<'a>,
) -> PacketResult {
    while !tile_part.header().at_end() {
        let progression_data = progression_iterator
            .next()
            .ok_or("packet header outlives the progression iterator")?;
        let resolution = progression_data.resolution;
        let precinct_index = usize::try_from(progression_data.precinct)
            .map_err(|_| "packet precinct index does not fit the platform")?;
        let component_index = usize::from(progression_data.component);
        let component_info = component_infos
            .get(component_index)
            .ok_or("packet progression references a missing component")?;
        let tile_decompositions = storage
            .tile_decompositions
            .get_mut(component_index)
            .ok_or("packet storage has no matching component")?;
        let sub_band_iter = tile_decompositions.sub_band_iter(resolution, &storage.decompositions);

        let packet_start = tile_part.packet_start_offset();
        let body_reader = tile_part.body();

        if component_info.coding_style.flags.may_use_sop_markers()
            && body_reader.peek_marker() == Some(SOP)
        {
            body_reader
                .read_marker()
                .map_err(|_| "SOP marker is truncated")?;
            body_reader
                .skip_bytes(4)
                .ok_or("SOP marker segment is truncated")?;
        }

        let header_reader = tile_part.header();

        let zero_length = header_reader
            .read_bits_with_stuffing(1)
            .ok_or("packet empty/non-empty flag is truncated")?
            == 0;

        // B.10.3 Zero length packet
        // "The first bit in the packet header denotes whether the packet has a length of zero
        // (empty packet). The value 0 indicates a zero length; no code-blocks are included in this
        // case. The value 1 indicates a non-zero length."
        if !zero_length {
            for sub_band in sub_band_iter.clone() {
                resolve_segments(
                    sub_band,
                    &progression_data,
                    header_reader,
                    storage,
                    component_info,
                )?;
            }
        }

        header_reader.align();

        if component_info.coding_style.flags.uses_eph_marker()
            && header_reader
                .read_marker()
                .map_err(|_| "EPH marker is truncated")?
                != EPH
        {
            return Err("EPH marker is missing");
        }

        // Now read the packet body.
        let body_reader = tile_part.body();

        if !zero_length {
            for sub_band_idx in sub_band_iter {
                let sub_band = &mut storage.sub_bands[sub_band_idx];
                let precinct = &mut storage.precincts[sub_band.precincts.clone()][precinct_index];
                let code_blocks = &mut storage.code_blocks[precinct.code_blocks.clone()];

                for code_block in code_blocks {
                    let required = storage
                        .roi_plan
                        .as_ref()
                        .is_none_or(|plan| plan.code_block_required(sub_band_idx, code_block.rect));
                    let layer = &mut storage.layers[code_block.layers.clone()]
                        [progression_data.layer_num as usize];

                    if let Some(segments) = layer.segments.clone() {
                        let segments = &mut storage.segments[segments.clone()];

                        for segment in segments {
                            if required {
                                segment.data = body_reader
                                    .read_bytes(segment.data_length as usize)
                                    .ok_or("packet body is shorter than its segment lengths")?;
                            } else {
                                body_reader.skip_bytes(segment.data_length as usize).ok_or(
                                    "skipped packet body is shorter than its segment lengths",
                                )?;
                            }
                        }
                    }
                }
            }
        }

        tile_part
            .validate_packet_length(packet_start)
            .ok_or("packet length marker disagrees with consumed bytes")?;
    }

    tile_part
        .validate_all_packet_lengths_consumed()
        .ok_or("packet length markers contain unused entries")?;

    Ok(())
}

fn try_push_segment_with_budget<'a>(
    segments: &mut Vec<Segment<'a>>,
    structural_workspace_bytes: usize,
    segment: Segment<'a>,
) -> Result<()> {
    let available_bytes = DEFAULT_MAX_DECODE_BYTES
        .checked_sub(structural_workspace_bytes)
        .ok_or(ValidationError::ImageTooLarge)?;
    let max_segment_count = available_bytes / size_of::<Segment<'_>>();
    let required_count = segments
        .len()
        .checked_add(1)
        .ok_or(ValidationError::ImageTooLarge)?;
    if required_count > max_segment_count {
        return Err(ValidationError::ImageTooLarge.into());
    }

    if required_count > segments.capacity() {
        let old_capacity = segments.capacity();
        let doubled_capacity = old_capacity.checked_mul(2).unwrap_or(max_segment_count);
        let target_capacity = doubled_capacity
            .max(4)
            .max(required_count)
            .min(max_segment_count);
        let target_capacity = segment_growth_capacity(
            structural_workspace_bytes,
            old_capacity,
            required_count,
            target_capacity,
        )?;
        try_reserve_decode_elements(segments, target_capacity)?;
        validate_segment_reallocation_peak(
            structural_workspace_bytes,
            old_capacity,
            segments.capacity(),
        )?;
    }
    segments.push(segment);
    Ok(())
}

fn segment_growth_capacity(
    structural_workspace_bytes: usize,
    old_capacity: usize,
    required_count: usize,
    preferred_capacity: usize,
) -> Result<usize> {
    if validate_segment_reallocation_peak(
        structural_workspace_bytes,
        old_capacity,
        preferred_capacity,
    )
    .is_ok()
    {
        return Ok(preferred_capacity);
    }
    validate_segment_reallocation_peak(structural_workspace_bytes, old_capacity, required_count)?;
    Ok(required_count)
}

fn validate_segment_reallocation_peak(
    structural_workspace_bytes: usize,
    old_capacity: usize,
    new_capacity: usize,
) -> Result<()> {
    let simultaneous_capacity = old_capacity
        .checked_add(new_capacity)
        .ok_or(ValidationError::ImageTooLarge)?;
    let segment_bytes = simultaneous_capacity
        .checked_mul(size_of::<Segment<'_>>())
        .ok_or(ValidationError::ImageTooLarge)?;
    let peak = structural_workspace_bytes
        .checked_add(segment_bytes)
        .ok_or(ValidationError::ImageTooLarge)?;
    if peak > DEFAULT_MAX_DECODE_BYTES {
        return Err(ValidationError::ImageTooLarge.into());
    }
    Ok(())
}

fn push_segment_or_record_error<'a>(
    segments: &mut Vec<Segment<'a>>,
    structural_workspace_bytes: usize,
    packet_workspace_error: &mut Option<DecodeError>,
    segment: Segment<'a>,
) -> Option<()> {
    match try_push_segment_with_budget(segments, structural_workspace_bytes, segment) {
        Ok(()) => Some(()),
        Err(error) => {
            *packet_workspace_error = Some(error);
            None
        }
    }
}

fn resolve_segments(
    sub_band_dx: usize,
    progression_data: &ProgressionData,
    reader: &mut BitReader<'_>,
    storage: &mut DecompositionStorage<'_>,
    component_info: &ComponentInfo,
) -> PacketResult {
    if component_info
        .coding_style
        .parameters
        .code_block_style
        .uses_high_throughput_block_coding()
    {
        ht::resolve_segments(
            sub_band_dx,
            progression_data,
            reader,
            storage,
            component_info,
        )
    } else {
        classic::resolve_segments(
            sub_band_dx,
            progression_data,
            reader,
            storage,
            component_info,
        )
    }
}

#[derive(Clone, Copy)]
struct CodeBlockInclusion {
    included: bool,
    included_first_time: bool,
}

fn resolve_code_block_inclusion(
    code_block: &mut CodeBlock,
    precinct: &mut Precinct,
    progression_data: &ProgressionData,
    reader: &mut BitReader<'_>,
    tag_tree_nodes: &mut [TagNode],
) -> Option<CodeBlockInclusion> {
    // B.10.4 Code-block inclusion
    let included_first_time = !code_block.has_been_included;
    let is_included = if code_block.has_been_included {
        // "For code-blocks that have been included in a previous packet,
        // a single bit is used to represent the information, where a 1
        // means that the code-block is included in this layer and a 0 means
        // that it is not."
        reader.read_bits_with_stuffing(1)? == 1
    } else {
        // "For code-blocks that have not been previously included in any packet,
        // this information is signalled with a separate tag tree code for each precinct
        // as confined to a sub-band. The values in this tag tree are the number of the
        // layer in which the current code-block is first included. Although the exact
        // sequence of bits that represent the inclusion tag tree appears in the bit
        // stream, only the bits needed for determining whether the code-block is
        // included are placed in the packet header. If some of the tag tree is already
        // known from previous code-blocks or previous layers, it is not repeated.
        // Likewise, only as much of the tag tree as is needed to determine inclusion in
        // the current layer is included. If a code-block is not included until a later
        // layer, then only a partial tag tree is included at that point in the bit
        // stream."
        precinct.code_inclusion_tree.read(
            code_block.x_idx,
            code_block.y_idx,
            reader,
            u32::from(progression_data.layer_num) + 1,
            tag_tree_nodes,
        )? <= u32::from(progression_data.layer_num)
    };

    ltrace!("code-block inclusion: {}", is_included);

    if !is_included {
        return Some(CodeBlockInclusion {
            included: false,
            included_first_time: false,
        });
    }

    // B.10.5 Zero bit-plane information
    // "If a code-block is included for the first time, the packet header contains
    // information identifying the actual number of bit-planes used to represent
    // coefficients from the code-block."
    if included_first_time {
        code_block.missing_bit_planes = u8::try_from(precinct.zero_bitplane_tree.read(
            code_block.x_idx,
            code_block.y_idx,
            reader,
            u32::MAX,
            tag_tree_nodes,
        )?)
        .ok()?;
        ltrace!(
            "zero bit-plane information: {}",
            code_block.missing_bit_planes
        );
    }

    code_block.has_been_included = true;

    Some(CodeBlockInclusion {
        included: true,
        included_first_time,
    })
}

fn decode_num_coding_passes(reader: &mut BitReader<'_>) -> Option<u8> {
    if reader.peak_bits_with_stuffing(9) == Some(0x1ff) {
        reader.read_bits_with_stuffing(9)?;
        u8::try_from(reader.read_bits_with_stuffing(7)? + 37).ok()
    } else if reader.peak_bits_with_stuffing(4) == Some(0x0f) {
        reader.read_bits_with_stuffing(4)?;
        u8::try_from(reader.read_bits_with_stuffing(5)? + 6).ok()
    } else if reader.peak_bits_with_stuffing(4) == Some(0b1110) {
        reader.read_bits_with_stuffing(4)?;
        Some(5)
    } else if reader.peak_bits_with_stuffing(4) == Some(0b1101) {
        reader.read_bits_with_stuffing(4)?;
        Some(4)
    } else if reader.peak_bits_with_stuffing(4) == Some(0b1100) {
        reader.read_bits_with_stuffing(4)?;
        Some(3)
    } else if reader.peak_bits_with_stuffing(2) == Some(0b10) {
        reader.read_bits_with_stuffing(2)?;
        Some(2)
    } else if reader.peak_bits_with_stuffing(1) == Some(0) {
        reader.read_bits_with_stuffing(1)?;
        Some(1)
    } else {
        None
    }
}

fn read_lblock_increment(reader: &mut BitReader<'_>) -> Option<u32> {
    let mut increment = 0;

    while reader.read_bits_with_stuffing(1)? == 1 {
        increment += 1;
    }

    Some(increment)
}

fn read_segment_length(
    reader: &mut BitReader<'_>,
    l_block: u32,
    coding_passes: u8,
) -> PacketResult<u32> {
    let bits = l_block
        .checked_add(coding_passes.ilog2())
        .ok_or("code-block segment-length field width overflows")?;
    let bits = u8::try_from(bits).map_err(|_| "code-block segment-length field is too wide")?;
    reader
        .read_bits_with_stuffing(bits)
        .ok_or("code-block segment length is truncated")
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn packet_segment_growth_respects_remaining_workspace_budget() {
        let segment_bytes = size_of::<Segment<'_>>();
        let structural_bytes = DEFAULT_MAX_DECODE_BYTES - segment_bytes;
        let mut segments = Vec::new();

        try_push_segment_with_budget(
            &mut segments,
            structural_bytes,
            Segment {
                idx: 0,
                coding_pases: 1,
                data_length: 0,
                data: &[],
            },
        )
        .unwrap();
        let error = try_push_segment_with_budget(
            &mut segments,
            structural_bytes,
            Segment {
                idx: 1,
                coding_pases: 1,
                data_length: 0,
                data: &[],
            },
        )
        .unwrap_err();

        assert_eq!(
            error,
            DecodeError::Validation(ValidationError::ImageTooLarge)
        );
        assert_eq!(segments.len(), 1);
    }

    #[test]
    fn packet_segment_growth_falls_back_to_the_exact_transient_boundary() {
        let mut segments = Vec::new();
        segments
            .try_reserve_exact(4)
            .expect("small retained segment buffer");
        let old_capacity = segments.capacity();
        for idx in 0..old_capacity {
            segments.push(Segment {
                idx: u8::try_from(idx).unwrap_or(u8::MAX),
                coding_pases: 1,
                data_length: 0,
                data: &[],
            });
        }
        let required_count = old_capacity + 1;
        let transient_count = old_capacity + required_count;
        let structural_bytes =
            DEFAULT_MAX_DECODE_BYTES - transient_count * size_of::<Segment<'_>>();

        try_push_segment_with_budget(
            &mut segments,
            structural_bytes,
            Segment {
                idx: u8::MAX,
                coding_pases: 1,
                data_length: 0,
                data: &[],
            },
        )
        .expect("exact old-plus-required reallocation peak fits");

        assert_eq!(segments.len(), required_count);
    }

    #[test]
    fn packet_segment_growth_rejects_old_plus_new_transient_over_cap() {
        let segment_bytes = size_of::<Segment<'_>>();
        let structural_bytes = DEFAULT_MAX_DECODE_BYTES - 8 * segment_bytes;
        assert_eq!(
            segment_growth_capacity(structural_bytes, 4, 5, 8),
            Err(DecodeError::Validation(ValidationError::ImageTooLarge))
        );
    }
}
