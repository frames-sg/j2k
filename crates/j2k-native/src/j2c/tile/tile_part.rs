// SPDX-License-Identifier: MIT OR Apache-2.0

//! Parsing of tile-part marker segments and their packet data readers.

use alloc::vec::Vec;

use super::metadata::{
    try_clone_coding_parameters, try_clone_quantization_info, TileMetadataBudget,
    TileMetadataTransaction,
};
use super::{
    ComponentTile, MergedTilePart, PacketLengthMetadata, ResolutionTile, SeparatedTilePart, Tile,
    TilePart,
};
use crate::error::{bail, err, DecodingError, MarkerError, Result, TileError, ValidationError};
use crate::j2c::codestream::{self, markers, skip_marker_segment, Header, PacketLengthMarker};
use crate::reader::BitReader;

#[expect(
    clippy::too_many_lines,
    reason = "the ordered JPEG 2000 state machine stays cohesive to preserve marker, packet, pass, and sample order"
)]
pub(super) fn parse_tile_part<'a>(
    reader: &mut BitReader<'a>,
    main_header: &Header<'a>,
    tiles: &mut [Tile<'a>],
    ppm_packet_idx: &mut usize,
    metadata_budget: &mut TileMetadataBudget,
) -> Result<()> {
    let mut allocations = metadata_budget.transaction();
    if reader.read_marker()? != markers::SOT {
        bail!(MarkerError::Expected("SOT"));
    }

    let tile_part_header = sot_marker(reader).ok_or(MarkerError::ParseFailure("SOT"))?;

    if u32::from(tile_part_header.tile_index) >= main_header.size_data.num_tiles() {
        bail!(TileError::InvalidIndex);
    }

    let data_len = if tile_part_header.tile_part_length == 0 {
        reader.tail().map_or(0, <[u8]>::len)
    } else {
        // Subtract 12 to account for the marker length.
        (tile_part_header.tile_part_length as usize)
            .checked_sub(12)
            .ok_or(TileError::Invalid)?
    };

    let start = reader.offset();

    let tile = &mut tiles[tile_part_header.tile_index as usize];
    let num_components =
        u16::try_from(tile.component_infos.len()).map_err(|_| ValidationError::TooManyChannels)?;

    let mut packet_length_markers = Vec::new();
    let mut packet_lengths_present = false;
    let mut ppt_headers = Vec::new();

    loop {
        let Some(marker) = reader.peek_marker() else {
            return if main_header.strict {
                err!(MarkerError::Invalid)
            } else {
                Ok(())
            };
        };

        match marker {
            markers::SOD => {
                reader.read_marker()?;
                break;
            }
            // COD, COC, QCD and QCC should only be used in the _first_
            // tile-part header, if they appear at all.
            markers::COD => {
                reader.read_marker()?;
                let cod = codestream::cod_marker(reader)?;
                allocations
                    .track_temporary_vec(&cod.component_parameters.parameters.precinct_exponents)?;

                tile.mct = cod.mct;
                tile.num_layers = cod.num_layers;
                tile.progression_order = cod.progression_order;

                let component_count = tile.component_infos.len();
                if component_count == 0 {
                    return Err(ValidationError::InvalidComponentMetadata.into());
                }
                let flags = cod.component_parameters.flags;
                let mut source = Some(cod.component_parameters.parameters);
                for (component_idx, component) in tile.component_infos.iter_mut().enumerate() {
                    let replacement = if component_idx + 1 == component_count {
                        source
                            .take()
                            .ok_or(ValidationError::InvalidComponentMetadata)?
                    } else {
                        try_clone_coding_parameters(
                            source
                                .as_ref()
                                .ok_or(ValidationError::InvalidComponentMetadata)?,
                            &mut allocations,
                        )?
                    };
                    allocations.replace_coding_parameters(
                        &mut component.coding_style.parameters,
                        replacement,
                    )?;
                    component.coding_style.flags.raw |= flags.raw;
                }
            }
            markers::COC => {
                reader.read_marker()?;

                let (component_index, coc) = codestream::coc_marker(reader, num_components)?;
                allocations.track_temporary_vec(&coc.parameters.precinct_exponents)?;

                let component_index = component_index as usize;
                let component = tile
                    .component_infos
                    .get_mut(component_index)
                    .ok_or(ValidationError::InvalidComponentMetadata)?;
                allocations.replace_coding_parameters(
                    &mut component.coding_style.parameters,
                    coc.parameters,
                )?;
                component.coding_style.flags.raw |= coc.flags.raw;
            }
            markers::QCD => {
                reader.read_marker()?;
                let qcd = codestream::qcd_marker(reader)?;
                allocations.track_temporary_vec(&qcd.step_sizes)?;

                let component_count = tile.component_infos.len();
                if component_count == 0 {
                    return Err(ValidationError::InvalidComponentMetadata.into());
                }
                let mut source = Some(qcd);
                for (component_idx, component) in tile.component_infos.iter_mut().enumerate() {
                    let replacement = if component_idx + 1 == component_count {
                        source
                            .take()
                            .ok_or(ValidationError::InvalidComponentMetadata)?
                    } else {
                        try_clone_quantization_info(
                            source
                                .as_ref()
                                .ok_or(ValidationError::InvalidComponentMetadata)?,
                            &mut allocations,
                        )?
                    };
                    allocations
                        .replace_quantization(&mut component.quantization_info, replacement)?;
                }
            }
            markers::QCC => {
                reader.read_marker()?;
                let (component_index, qcc) = codestream::qcc_marker(reader, num_components)?;
                allocations.track_temporary_vec(&qcc.step_sizes)?;

                let component_index = component_index as usize;
                let component = tile
                    .component_infos
                    .get_mut(component_index)
                    .ok_or(ValidationError::InvalidComponentMetadata)?;
                allocations.replace_quantization(&mut component.quantization_info, qcc)?;
            }
            markers::POC => {
                reader.read_marker()?;
                let progression_changes = codestream::poc_marker(
                    reader,
                    num_components,
                    tile.num_layers,
                    allocations.remaining_bytes() / 2,
                )?;
                allocations.track_temporary_vec(&progression_changes)?;
                allocations.append_temporary(&mut tile.progression_changes, progression_changes)?;
            }
            markers::RGN => {
                reader.read_marker()?;
                let rgn = codestream::rgn_marker(reader, num_components)
                    .ok_or(MarkerError::ParseFailure("RGN"))?;
                if rgn.style != 0 {
                    bail!(DecodingError::UnsupportedFeature("explicit ROI coding"));
                }
                tile.component_infos
                    .get_mut(rgn.component_index as usize)
                    .ok_or(ValidationError::InvalidComponentMetadata)?
                    .roi_shift = rgn.shift;
            }
            markers::EOC => break,
            markers::PPT => {
                if !main_header.ppm_packets.is_empty() {
                    bail!(TileError::PpmPptConflict);
                }

                reader.read_marker()?;
                let target_len = ppt_headers
                    .len()
                    .checked_add(1)
                    .ok_or(ValidationError::ImageTooLarge)?;
                allocations.try_reserve_temporary(&mut ppt_headers, target_len)?;
                ppt_headers.push(ppt_marker(reader).ok_or(MarkerError::ParseFailure("PPT"))?);
            }
            markers::PLT => {
                reader.read_marker()?;
                packet_lengths_present = true;
                let target_len = packet_length_markers
                    .len()
                    .checked_add(1)
                    .ok_or(ValidationError::ImageTooLarge)?;
                allocations.try_reserve_temporary(&mut packet_length_markers, target_len)?;
                let marker = codestream::plt_marker(reader, allocations.remaining_bytes())?;
                allocations.track_temporary_vec(&marker.packet_lengths)?;
                packet_length_markers.push(marker);
            }
            markers::COM => {
                reader.read_marker()?;
                skip_marker_segment(reader).ok_or(MarkerError::ParseFailure("COM"))?;
            }
            (0x30..=0x3F) => {
                // Reserved marker codes without marker segment parameters.
                reader.read_marker()?;
            }
            _ => {
                bail!(MarkerError::Unsupported);
            }
        }
    }

    let Some(remaining_bytes) = data_len.checked_sub(reader.offset() - start) else {
        return if main_header.strict {
            err!(TileError::Invalid)
        } else {
            Ok(())
        };
    };

    let packet_length_marker_capacity = packet_length_markers.capacity();
    let temporary_packet_length_capacity =
        packet_length_markers
            .iter()
            .try_fold(0_usize, |total, marker| -> Result<usize> {
                total
                    .checked_add(marker.packet_lengths.capacity())
                    .ok_or(ValidationError::ImageTooLarge.into())
            })?;
    let temporary_packet_length_count =
        packet_length_markers
            .iter()
            .try_fold(0_usize, |total, marker| -> Result<usize> {
                total
                    .checked_add(marker.packet_lengths.len())
                    .ok_or(ValidationError::ImageTooLarge.into())
            })?;
    let ppt_header_capacity = ppt_headers.capacity();

    ppt_headers.sort_by_key(|ppt_header| ppt_header.sequence_idx);
    let ppm_header_count = ppm_header_count(
        tile,
        packet_lengths_present.then_some(temporary_packet_length_count),
        &tile_part_header,
        main_header,
        *ppm_packet_idx,
    )?;
    let header_count = ppt_headers
        .len()
        .checked_add(ppm_header_count)
        .ok_or(ValidationError::ImageTooLarge)?;
    let mut headers = Vec::new();
    allocations.try_reserve_temporary(&mut headers, header_count)?;
    headers.extend(
        ppt_headers
            .iter()
            .map(|ppt_header| BitReader::new(ppt_header.data)),
    );
    packet_length_markers.sort_by_key(|marker| marker.sequence_idx);
    let use_main_header_packet_lengths = !packet_lengths_present
        && !main_header.plm_packet_lengths.is_empty()
        && main_header.size_data.num_tiles() == 1
        && tile_part_header.tile_part_index == 0
        && tile_part_header.num_tile_parts == 1;
    let packet_lengths = if use_main_header_packet_lengths {
        PacketLengthMetadata::new(
            true,
            allocations.try_copy_temporary(&main_header.plm_packet_lengths)?,
        )
    } else {
        let mut packet_lengths = Vec::new();
        allocations.try_reserve_temporary(&mut packet_lengths, temporary_packet_length_count)?;
        for marker in &mut packet_length_markers {
            packet_lengths.append(&mut marker.packet_lengths);
        }
        PacketLengthMetadata::new(packet_lengths_present, packet_lengths)
    };

    let ppm_packet_end = ppm_packet_idx
        .checked_add(ppm_header_count)
        .ok_or(ValidationError::ImageTooLarge)?;
    let ppm_packets = main_header
        .ppm_packets
        .get(*ppm_packet_idx..ppm_packet_end)
        .ok_or(TileError::Invalid)?;
    headers.extend(
        ppm_packets
            .iter()
            .map(|ppm_packet| BitReader::new(ppm_packet.data)),
    );
    *ppm_packet_idx = ppm_packet_end;

    drop(packet_length_markers);
    allocations.release_temporary_capacity::<PacketLengthMarker>(packet_length_marker_capacity)?;
    allocations.release_temporary_capacity::<u32>(temporary_packet_length_capacity)?;
    drop(ppt_headers);
    allocations.release_temporary_capacity::<PptMarkerData<'_>>(ppt_header_capacity)?;

    let data = reader
        .read_bytes(remaining_bytes)
        .ok_or(TileError::Invalid)?;

    let tile_part = if headers.is_empty() {
        TilePart::Merged(MergedTilePart {
            data: BitReader::new(data),
            packet_lengths,
        })
    } else {
        TilePart::Separated(SeparatedTilePart {
            headers,
            body: BitReader::new(data),
            packet_lengths,
        })
    };

    let tile_part_count = tile
        .tile_parts
        .len()
        .checked_add(1)
        .ok_or(ValidationError::ImageTooLarge)?;
    allocations.try_reserve_retained(&mut tile.tile_parts, tile_part_count)?;
    retain_tile_part_metadata(&mut allocations, &tile_part)?;
    tile.tile_parts.push(tile_part);

    Ok(())
}

fn retain_tile_part_metadata(
    allocations: &mut TileMetadataTransaction<'_>,
    tile_part: &TilePart<'_>,
) -> Result<()> {
    match tile_part {
        TilePart::Merged(part) => {
            allocations.retain_temporary_vec(&part.packet_lengths.lengths)?;
        }
        TilePart::Separated(part) => {
            allocations.retain_temporary_vec(&part.headers)?;
            allocations.retain_temporary_vec(&part.packet_lengths.lengths)?;
        }
    }
    Ok(())
}

fn ppm_header_count(
    tile: &Tile<'_>,
    packet_length_count: Option<usize>,
    tile_part_header: &TilePartHeader,
    main_header: &Header<'_>,
    ppm_packet_idx: usize,
) -> Result<usize> {
    if main_header.ppm_packets.is_empty() {
        return Ok(0);
    }
    if let Some(packet_length_count) = packet_length_count {
        return Ok(packet_length_count);
    }
    if tile_part_header.num_tile_parts == 1 {
        return tile_packet_count(tile);
    }

    // Without PLT lengths, this legacy PPM representation has no serialized
    // boundary for a multi-part tile. Preserve the former one-entry fallback
    // until the parser supports cross-marker Nppm tile-part accumulation.
    Ok(usize::from(
        main_header.ppm_packets.get(ppm_packet_idx).is_some(),
    ))
}

fn tile_packet_count(tile: &Tile<'_>) -> Result<usize> {
    if tile.progression_changes.is_empty() {
        let component_end = u16::try_from(tile.component_infos.len())
            .map_err(|_| ValidationError::TooManyChannels)?;
        let resolution_end = tile
            .component_infos
            .iter()
            .map(|component| component.coding_style.parameters.num_resolution_levels)
            .max()
            .ok_or(ValidationError::InvalidComponentMetadata)?;
        return packet_count_for_bounds(tile, 0, resolution_end, tile.num_layers, 0, component_end);
    }

    tile.progression_changes
        .iter()
        .try_fold(0_usize, |total, change| {
            let count = packet_count_for_bounds(
                tile,
                change.resolution_start,
                change.resolution_end,
                change.layer_end.min(tile.num_layers),
                change.component_start,
                change.component_end,
            )?;
            total
                .checked_add(count)
                .ok_or(ValidationError::ImageTooLarge.into())
        })
}

fn packet_count_for_bounds(
    tile: &Tile<'_>,
    resolution_start: u8,
    resolution_end: u8,
    layer_end: u8,
    component_start: u16,
    component_end: u16,
) -> Result<usize> {
    let component_len =
        u16::try_from(tile.component_infos.len()).map_err(|_| ValidationError::TooManyChannels)?;
    let component_end = component_end.min(component_len);
    let total_resolution_end = tile
        .component_infos
        .iter()
        .map(|component| component.coding_style.parameters.num_resolution_levels)
        .max()
        .ok_or(ValidationError::InvalidComponentMetadata)?;
    let resolution_end = resolution_end.min(total_resolution_end);
    if resolution_start >= resolution_end || layer_end == 0 || component_start >= component_end {
        return Err(DecodingError::InvalidProgressionIterator.into());
    }

    let mut packet_count = 0_usize;
    for component_idx in component_start..component_end {
        let component = tile
            .component_infos
            .get(usize::from(component_idx))
            .ok_or(ValidationError::InvalidComponentMetadata)?;
        let component_tile = ComponentTile::new(tile, component);
        let component_resolution_end = resolution_end.min(component.num_resolution_levels());
        for resolution in resolution_start..component_resolution_end {
            let precinct_count =
                usize::try_from(ResolutionTile::new(component_tile, resolution).num_precincts())
                    .map_err(|_| ValidationError::ImageTooLarge)?;
            let layer_packets = precinct_count
                .checked_mul(usize::from(layer_end))
                .ok_or(ValidationError::ImageTooLarge)?;
            packet_count = packet_count
                .checked_add(layer_packets)
                .ok_or(ValidationError::ImageTooLarge)?;
        }
    }
    Ok(packet_count)
}

struct TilePartHeader {
    tile_index: u16,
    tile_part_length: u32,
    tile_part_index: u8,
    num_tile_parts: u8,
}

struct PptMarkerData<'a> {
    data: &'a [u8],
    sequence_idx: u8,
}

fn ppt_marker<'a>(reader: &mut BitReader<'a>) -> Option<PptMarkerData<'a>> {
    let length = reader.read_u16()?.checked_sub(2)?;
    let header_len = length.checked_sub(1)?;
    let sequence_idx = reader.read_byte()?;
    Some(PptMarkerData {
        data: reader.read_bytes(header_len as usize)?,
        sequence_idx,
    })
}

fn sot_marker(reader: &mut BitReader<'_>) -> Option<TilePartHeader> {
    // Length.
    let _ = reader.read_u16()?;

    let tile_index = reader.read_u16()?;
    let tile_part_length = reader.read_u32()?;

    // We infer those ourselves.
    let tile_part_index = reader.read_byte()?;
    let num_tile_parts = reader.read_byte()?;

    Some(TilePartHeader {
        tile_index,
        tile_part_length,
        tile_part_index,
        num_tile_parts,
    })
}

#[cfg(test)]
mod tests;
