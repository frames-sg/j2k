//! Parsing of layers and their segments, as specified in Annex B.

use alloc::vec::Vec;
use core::mem::size_of;

use super::build::{CodeBlock, ParseScratch, Precinct, PrecinctSelection, PrecinctUse, Segment};
use super::codestream::markers::{EPH, SOP};
use super::codestream::{ComponentInfo, Header};
use super::decode::{DecodeAllocationBudget, DecompositionStorage};
use super::progression::{progression_iterator, ProgressionData, ProgressionIterator};
use super::tag_tree::TagNode;
use super::tile::{Tile, TilePartCursor};
use crate::error::{bail, DecodeError, DecodingError, Result, ValidationError};
use crate::reader::BitReader;
use crate::{try_reserve_decode_elements, DEFAULT_MAX_DECODE_BYTES};

mod classic;
#[cfg(feature = "std")]
pub(crate) mod directory;
mod ht;

#[derive(Default)]
pub(crate) struct PacketParseStats {
    #[cfg(test)]
    pub(crate) parsed_headers: usize,
    #[cfg(test)]
    pub(crate) skipped_packets: usize,
}

#[derive(Default)]
struct PacketWalk<'d> {
    stats: PacketParseStats,
    #[cfg(feature = "std")]
    recording: Option<&'d mut Vec<u64>>,
    #[cfg(feature = "std")]
    indexed: Option<&'d directory::PacketDirectory>,
    next_packet: usize,
    marker: core::marker::PhantomData<&'d ()>,
}

pub(crate) const MAX_BITPLANE_COUNT: u8 = 63;
type PacketResult<T = ()> = core::result::Result<T, &'static str>;

/// Where packet-header parsing puts a code-block contribution's segments.
pub(super) enum SegmentSink<'s, 'a> {
    /// Record each segment, for a code-block the decode reads.
    Record {
        segments: &'s mut Vec<Segment<'a>>,
        structural_workspace_bytes: usize,
        packet_workspace_error: &'s mut Option<DecodeError>,
    },
    /// Add up the segment lengths, for a code-block whose body is skipped.
    Skip(&'s mut usize),
}

impl<'a> SegmentSink<'_, 'a> {
    fn push(&mut self, segment: Segment<'a>) -> Option<()> {
        match self {
            Self::Record {
                segments,
                structural_workspace_bytes,
                packet_workspace_error,
            } => push_segment_or_record_error(
                segments,
                *structural_workspace_bytes,
                packet_workspace_error,
                segment,
            ),
            Self::Skip(bytes) => {
                **bytes = bytes.checked_add(usize::try_from(segment.data_length).ok()?)?;
                Some(())
            }
        }
    }
}

/// The body bytes of one packet that belong to code-blocks the decode does
/// not read, in body order, so the body can be skipped without recording
/// their segments.
pub(super) struct PacketSkips {
    /// (code-block index, body bytes) for each contribution to skip.
    entries: Vec<(usize, usize)>,
    /// The entries' bytes added up, saturating: more than any body holds.
    bytes: usize,
    #[cfg(feature = "std")]
    discard_contributions: bool,
}

impl PacketSkips {
    /// Room for every code-block of one packet: a packet holds one precinct
    /// of each of a resolution's (at most three) sub-bands, built or borrowing
    /// scratch. Code-blocks are
    /// only skipped this way when an ROI plan leaves some out.
    fn new(storage: &DecompositionStorage<'_>, live_bytes: usize) -> Result<Self> {
        let capacity = if storage.roi_plan.is_some() || storage.parse_scratch.is_some() {
            storage
                .precincts
                .iter()
                .map(|precinct| precinct.code_blocks.len())
                .chain(storage.parse_scratch.map(|scratch| scratch.code_blocks()))
                .max()
                .unwrap_or(0)
                .checked_mul(3)
                .ok_or(ValidationError::ImageTooLarge)?
        } else {
            0
        };
        let mut entries = Vec::new();
        DecodeAllocationBudget::from_live_bytes(live_bytes)?.reserve_new(&mut entries, capacity)?;
        Ok(Self {
            entries,
            bytes: 0,
            #[cfg(feature = "std")]
            discard_contributions: false,
        })
    }

    fn owned_bytes(&self) -> usize {
        self.entries.capacity() * size_of::<(usize, usize)>()
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.bytes = 0;
    }

    pub(super) fn push(&mut self, code_block: usize, bytes: usize) -> PacketResult {
        if bytes == 0 {
            return Ok(());
        }
        if self.entries.len() == self.entries.capacity() {
            return Err("packet has more skipped contributions than code-blocks");
        }
        self.entries.push((code_block, bytes));
        self.bytes = self.bytes.saturating_add(bytes);
        Ok(())
    }
}

pub(crate) fn parse<'a>(
    tile: &Tile<'a>,
    header: &Header<'_>,
    storage: &mut DecompositionStorage<'a>,
) -> Result<()> {
    parse_selected(tile, header, storage, None).map(|_| ())
}

/// Parse a tile's packets. With a region decode's `selection`, packets of
/// unused precincts are skipped or parsed only to reach later packets, and
/// parsing stops once every required packet has been read.
pub(crate) fn parse_selected<'a>(
    tile: &Tile<'a>,
    header: &Header<'_>,
    storage: &mut DecompositionStorage<'a>,
    selection: Option<&PrecinctSelection>,
) -> Result<PacketParseStats> {
    let mut walk = PacketWalk {
        #[cfg(feature = "std")]
        indexed: tile.packet_directory.as_deref(),
        ..PacketWalk::default()
    };
    parse_with_walk(tile, header, storage, selection, &mut walk)?;
    Ok(walk.stats)
}

fn parse_with_walk<'a>(
    tile: &Tile<'a>,
    header: &Header<'_>,
    storage: &mut DecompositionStorage<'a>,
    selection: Option<&PrecinctSelection>,
    walk: &mut PacketWalk<'_>,
) -> Result<()> {
    if selection.is_some_and(|selection| selection.required_packets() == 0) {
        return Ok(());
    }
    let retained_live_bytes = DecodeAllocationBudget::for_storage(storage)?.live_bytes();
    let mut progression_iterator = progression_iterator(tile, retained_live_bytes)?;
    let mut skips = PacketSkips::new(
        storage,
        retained_live_bytes
            .checked_add(progression_iterator.owned_bytes())
            .ok_or(ValidationError::ImageTooLarge)?,
    )?;
    #[cfg(feature = "std")]
    {
        skips.discard_contributions = walk.recording.is_some();
    }
    let parse_bytes = progression_iterator
        .owned_bytes()
        .checked_add(skips.owned_bytes())
        .ok_or(ValidationError::ImageTooLarge)?;
    storage.structural_workspace_bytes = storage
        .structural_workspace_bytes
        .checked_add(parse_bytes)
        .ok_or(ValidationError::ImageTooLarge)?;
    let result = parse_selected_inner(
        tile,
        &mut progression_iterator,
        &mut skips,
        header,
        storage,
        selection,
        walk,
    );
    #[cfg(feature = "std")]
    let result = result.and_then(|()| {
        if walk.recording.is_some() && progression_iterator.next().is_some() {
            return Err(DecodingError::PacketParseFailure(
                "packet directory has incomplete progression coverage",
            )
            .into());
        }
        Ok(())
    });
    storage.structural_workspace_bytes = storage
        .structural_workspace_bytes
        .checked_sub(parse_bytes)
        .ok_or(ValidationError::ImageTooLarge)?;
    result
}

fn parse_selected_inner<'a>(
    tile: &Tile<'a>,
    progression_iterator: &mut ProgressionIterator<'_>,
    skips: &mut PacketSkips,
    header: &Header<'_>,
    storage: &mut DecompositionStorage<'a>,
    selection: Option<&PrecinctSelection>,
    walk: &mut PacketWalk<'_>,
) -> Result<()> {
    let mut relevance = selection.map(|selection| PacketRelevance::new(selection, storage));
    let result = parse_tile_parts(
        tile,
        progression_iterator,
        skips,
        header,
        storage,
        relevance.as_mut(),
        walk,
    );
    if let Some(mut scratch) = relevance.and_then(|relevance| relevance.scratch) {
        scratch.give_all_back(&mut storage.precincts);
    }
    result
}

fn parse_tile_parts<'a>(
    tile: &Tile<'a>,
    progression_iterator: &mut ProgressionIterator<'_>,
    skips: &mut PacketSkips,
    header: &Header<'_>,
    storage: &mut DecompositionStorage<'a>,
    mut relevance: Option<&mut PacketRelevance<'_>>,
    walk: &mut PacketWalk<'_>,
) -> Result<()> {
    #[cfg(feature = "std")]
    let strict = header.strict || walk.recording.is_some();
    #[cfg(not(feature = "std"))]
    let strict = header.strict;
    for tile_part in &tile.tile_parts {
        if relevance
            .as_ref()
            .is_some_and(|relevance| relevance.remaining_packets == 0)
        {
            // Every packet the ROI plan needs has been read; the rest of the
            // tile can only hold code-blocks the region does not use.
            break;
        }
        let parsed = tile_part
            .cursor()
            .ok_or("tile-part cursor is inconsistent")
            .and_then(|cursor| {
                parse_inner(
                    cursor,
                    progression_iterator,
                    &tile.component_infos,
                    tile.num_layers,
                    storage,
                    skips,
                    relevance.as_deref_mut(),
                    walk,
                    strict,
                )
            });
        if let Some(mut cursor) = tile_part.cursor() {
            cursor.header().check_cancelled()?;
        }
        if let Some(error) = storage.packet_workspace_error.take() {
            return Err(error);
        }
        if let Err(context) = parsed {
            if strict {
                bail!(DecodingError::PacketParseFailure(context));
            }
        }
    }

    Ok(())
}

/// Packets a region decode still needs, read from its precinct selection.
///
/// A packet carries one precinct of one component and resolution for one
/// layer. It matters only if one of that precinct's code-blocks is required by
/// the ROI plan, independent of the layer, so an unneeded precinct's packets
/// can be skipped without ever updating its tag trees.
struct PacketRelevance<'s> {
    selection: &'s PrecinctSelection,
    remaining_packets: usize,
    /// The scratch precincts parsed-only precincts borrow, taken from the
    /// storage while parsing.
    scratch: Option<ParseScratch>,
}

impl<'s> PacketRelevance<'s> {
    fn new(selection: &'s PrecinctSelection, storage: &mut DecompositionStorage<'_>) -> Self {
        Self {
            selection,
            remaining_packets: selection.required_packets(),
            scratch: storage.parse_scratch.take(),
        }
    }

    /// Lend a parsed-only packet's precincts their scratch precincts, one per
    /// sub-band, before its header is parsed. Returns whether it did: the
    /// packet's code-blocks then have no layer records, so the header skips
    /// every contribution.
    fn lend_scratch(
        relevance: Option<&mut Self>,
        packet: &ProgressionData,
        precinct_offset: usize,
        sub_bands: impl Iterator<Item = usize>,
        storage: &mut DecompositionStorage<'_>,
    ) -> PacketResult<bool> {
        let Some(relevance) = relevance else {
            return Ok(false);
        };
        let Some(scratch) = relevance.scratch.as_mut() else {
            return Ok(false);
        };
        if relevance.selection.packet_use(packet) != Some(PrecinctUse::Parsed) {
            return Ok(false);
        }
        for (slot, sub_band_idx) in sub_bands.enumerate() {
            let precinct = storage.sub_bands[sub_band_idx]
                .precincts
                .start
                .checked_add(precinct_offset)
                .ok_or("packet precinct index overflows")?;
            scratch
                .lend(
                    slot,
                    precinct,
                    packet.layer_num,
                    &mut storage.precincts,
                    &mut storage.code_blocks,
                    &mut storage.tag_tree_nodes,
                )
                .ok_or("parsed-only precinct cannot borrow its scratch precinct")?;
        }
        Ok(true)
    }

    /// Whether a packet must be parsed: always without a selection. A packet
    /// for a precinct the tile lacks counts as required so the parser reports
    /// it.
    fn requires(relevance: Option<&Self>, packet: &ProgressionData) -> bool {
        relevance.is_none_or(|relevance| {
            relevance.selection.packet_use(packet) != Some(PrecinctUse::Unused)
        })
    }

    /// Count one parsed packet. Returns whether it was the last required one.
    fn record_parsed(relevance: Option<&mut Self>, packet: &ProgressionData) -> bool {
        relevance.is_some_and(|relevance| {
            if relevance.selection.packet_use(packet) == Some(PrecinctUse::Required) {
                relevance.remaining_packets = relevance.remaining_packets.saturating_sub(1);
            }
            relevance.remaining_packets == 0
        })
    }
}

/// Parse packets until the tile-part or the tile's progression ends, or
/// until `relevance` has seen every packet the ROI plan needs, which returns
/// `Ok(true)`.
#[expect(
    clippy::too_many_lines,
    reason = "one ordered pass keeps packet skipping, header parsing, body reads, and length checks together"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "the packet walk shares parser state and optional boundary observation"
)]
fn parse_inner<'a>(
    mut tile_part: TilePartCursor<'_, 'a>,
    progression_iterator: &mut ProgressionIterator<'_>,
    component_infos: &[ComponentInfo],
    num_layers: u8,
    storage: &mut DecompositionStorage<'a>,
    skips: &mut PacketSkips,
    mut relevance: Option<&mut PacketRelevance<'_>>,
    walk: &mut PacketWalk<'_>,
    strict: bool,
) -> PacketResult<bool> {
    // The last precinct's place among its sub-bands' records, by (component,
    // resolution, precinct): position-major progressions read all of a
    // precinct's layers in a row, so it is looked up once for them.
    let mut last_record: Option<((u16, u8, u64), usize)> = None;
    while !tile_part.header().at_end() {
        // The outer parser restores the typed cancellation error before any
        // lenient packet recovery; this internal error only unwinds the walk.
        tile_part
            .header()
            .check_cancelled()
            .map_err(|_| "decode cancelled")?;
        let Some(progression_data) = progression_iterator.next() else {
            // Every packet of the tile has been read. Bytes the tile-part
            // holds after them belong to no packet and are left unread, as
            // a region decode leaves the packets after its last required one.
            lwarn!("tile-part data continues after the tile's last packet");
            return Ok(false);
        };
        let ordinal = walk.next_packet;
        walk.next_packet = ordinal.checked_add(1).ok_or("packet ordinal overflows")?;
        #[cfg(feature = "std")]
        {
            if let Some(directory) = walk.indexed {
                if relevance.as_ref().is_some_and(|relevance| {
                    matches!(
                        relevance.selection.packet_use(&progression_data),
                        Some(PrecinctUse::Parsed | PrecinctUse::Unused)
                    )
                }) {
                    let (start, end) = directory
                        .range(ordinal)
                        .ok_or("derived packet range is missing")?;
                    tile_part
                        .skip_derived_packet(start, end)
                        .ok_or("derived packet range disagrees with the cursor")?;
                    #[cfg(test)]
                    {
                        walk.stats.skipped_packets += 1;
                    }
                    continue;
                }
            }
        }
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
        let packet_required = PacketRelevance::requires(relevance.as_deref(), &progression_data);
        // When PLT lengths cover every packet of the tile, an unneeded packet
        // is skipped without parsing its header. Otherwise it is parsed to
        // find where the next one starts. Skipping is decided for the whole
        // tile: once one packet of a precinct is skipped, its tag trees and
        // Lblock state are stale, so none of its later packets can be parsed.
        if !packet_required
            && relevance
                .as_ref()
                .is_some_and(|relevance| relevance.selection.skip_by_length())
        {
            // Need is decided per precinct, so the precinct's remaining
            // layers, which a position-major progression reads next, are
            // skipped with this one, up to the end of the tile-part.
            #[cfg(feature = "std")]
            let layers = if walk.recording.is_none() {
                progression_iterator.remaining_precinct_layers()
            } else {
                0
            };
            #[cfg(not(feature = "std"))]
            let layers = progression_iterator.remaining_precinct_layers();
            // A packet header takes a byte at least, and its EPH marker two
            // more where the coding style requires one.
            let min_length = if component_info.coding_style.flags.uses_eph_marker() {
                3
            } else {
                1
            };
            let (skipped, complete) = tile_part.skip_packets(1 + layers, min_length)?;
            // The packets a one-by-one walk would have taken from the
            // progression after this one: each skipped, plus a failing one.
            let further = if complete {
                skipped.saturating_sub(1)
            } else {
                skipped
            };
            progression_iterator.skip_precinct_layers(further);
            walk.next_packet = walk
                .next_packet
                .checked_add(further)
                .ok_or("packet ordinal overflows")?;
            if !complete {
                return Err("PLT packet length runs past the tile-part");
            }
            continue;
        }

        // Where the packet's precinct lies among each of its sub-bands'
        // precinct records. A region decode keeps none for unused
        // precincts. A packet for a precinct without a record gets an offset
        // past them all, so it fails below as one for a missing precinct.
        let key = (
            progression_data.component,
            progression_data.resolution,
            progression_data.precinct,
        );
        let precinct_offset = match last_record {
            Some((last, offset)) if last == key => offset,
            _ => {
                let offset = relevance
                    .as_deref()
                    .map_or(Some(precinct_index), |relevance| {
                        relevance.selection.record_offset(&progression_data)
                    })
                    .unwrap_or(usize::MAX);
                last_record = Some((key, offset));
                offset
            }
        };

        let packet_start = tile_part.packet_start_offset();
        let body_reader = tile_part.body();

        if component_info.coding_style.flags.may_use_sop_markers()
            && body_reader.peek_marker() == Some(SOP)
        {
            parse_sop_marker(body_reader, ordinal)?;
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
        skips.clear();
        // Even an empty packet starts its precinct: a precinct's first packet
        // is the one that borrows its scratch.
        let in_scratch = PacketRelevance::lend_scratch(
            relevance.as_deref_mut(),
            &progression_data,
            precinct_offset,
            sub_band_iter.clone(),
            storage,
        )?;
        if !zero_length {
            for sub_band in sub_band_iter.clone() {
                resolve_segments(
                    sub_band,
                    &progression_data,
                    precinct_offset,
                    header_reader,
                    storage,
                    component_info,
                    num_layers,
                    skips,
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

        if !zero_length && in_scratch {
            // Every contribution was skipped: the body is just their bytes.
            body_reader
                .skip_bytes(skips.bytes)
                .ok_or("skipped packet body is shorter than its segment lengths")?;
        } else if !zero_length {
            let mut skipped = skips.entries.iter().peekable();
            for sub_band_idx in sub_band_iter {
                let sub_band = &storage.sub_bands[sub_band_idx];
                let precinct = &storage.precincts[sub_band.precincts.clone()][precinct_offset];

                for code_block_idx in precinct.code_blocks.clone() {
                    // A code-block the decode doesn't read has no layers; its
                    // contribution, if any, is skipped as one run.
                    if let Some(&(_, bytes)) =
                        skipped.next_if(|&&(block, _)| block == code_block_idx)
                    {
                        body_reader
                            .skip_bytes(bytes)
                            .ok_or("skipped packet body is shorter than its segment lengths")?;
                        continue;
                    }
                    let code_block = &storage.code_blocks[code_block_idx];
                    if code_block.layers.is_empty() {
                        continue;
                    }
                    let required = code_block_is_read(storage, sub_band_idx, code_block);
                    let layer = &mut storage.layers[code_block.layers.clone()]
                        [progression_data.layer_num as usize];

                    if let Some(segments) = layer.segments() {
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

        #[cfg(test)]
        {
            walk.stats.parsed_headers += 1;
        }
        #[cfg(feature = "std")]
        {
            if let Some(boundaries) = walk.recording.as_mut() {
                let end = tile_part
                    .inline_offset()
                    .ok_or("packet headers are separated")?;
                let end = u64::try_from(end).map_err(|_| "packet offset overflows")?;
                if boundaries.len() == boundaries.capacity()
                    || boundaries.last().is_none_or(|&start| start >= end)
                {
                    return Err("packet boundary capacity or order is invalid");
                }
                boundaries.push(end);
            }
            if let Some(directory) = walk.indexed {
                let (_, end) = directory
                    .range(ordinal)
                    .ok_or("derived packet range is missing")?;
                if tile_part.inline_offset() != Some(end) {
                    return Err("parsed packet disagrees with its derived boundary");
                }
            }
        }

        if PacketRelevance::record_parsed(relevance.as_deref_mut(), &progression_data) {
            return Ok(true);
        }
    }

    // PLT entries left once the tile's last packet has been read describe no
    // packet, like bytes after it. Only a walk that stops at an error looks
    // ahead: one that goes on needs the next packet for the next tile-part.
    if tile_part.validate_all_packet_lengths_consumed().is_none()
        && (!strict || progression_iterator.next().is_some())
    {
        return Err("packet length markers contain unused entries");
    }

    Ok(false)
}

fn parse_sop_marker(body_reader: &mut BitReader<'_>, packet_ordinal: usize) -> PacketResult {
    if body_reader
        .read_marker()
        .map_err(|_| "SOP marker is truncated")?
        != SOP
    {
        return Err("packet does not start with an SOP marker");
    }
    let length = body_reader
        .read_u16()
        .ok_or("SOP marker segment is truncated")?;
    if length != 4 {
        return Err("SOP marker segment length is not 4");
    }
    let sequence = body_reader
        .read_u16()
        .ok_or("SOP marker segment is truncated")?;
    let modulus = usize::from(u16::MAX) + 1;
    let expected = u16::try_from(packet_ordinal % modulus)
        .map_err(|_| "SOP packet sequence does not fit u16")?;
    if sequence != expected {
        return Err("SOP marker sequence number disagrees with the packet ordinal");
    }
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
        // Grow into a fresh allocation, not by `realloc`: macOS 26 never
        // reuses a freed block that realloc grew, so every decode's last
        // segment block would stay dirty in its large-allocation cache.
        let mut grown = Vec::new();
        try_reserve_decode_elements(&mut grown, target_capacity)?;
        grown.append(segments);
        *segments = grown;
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

#[expect(
    clippy::too_many_arguments,
    reason = "the packet, its precinct's offset among the records, and the parser state it fills stay explicit"
)]
fn resolve_segments(
    sub_band_dx: usize,
    progression_data: &ProgressionData,
    precinct_offset: usize,
    reader: &mut BitReader<'_>,
    storage: &mut DecompositionStorage<'_>,
    component_info: &ComponentInfo,
    num_layers: u8,
    skips: &mut PacketSkips,
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
            precinct_offset,
            reader,
            storage,
            component_info,
            num_layers,
            skips,
        )
    } else {
        classic::resolve_segments(
            sub_band_dx,
            progression_data,
            precinct_offset,
            reader,
            storage,
            component_info,
            num_layers,
            skips,
        )
    }
}

/// Whether the decode reads `code_block` of `sub_band_idx`: always without an
/// ROI plan. Only these code-blocks have layer records.
fn code_block_is_read(
    storage: &DecompositionStorage<'_>,
    sub_band_idx: usize,
    code_block: &CodeBlock,
) -> bool {
    storage
        .roi_plan
        .as_ref()
        .is_none_or(|plan| plan.code_block_required(sub_band_idx, code_block.rect))
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
    // Table B.4, read a prefix at a time: 0 -> 1, 10 -> 2, 11xx -> 3 + xx
    // (xx < 3), 1111 xxxxx -> 6 + xxxxx (xxxxx < 31) and 1111 11111 xxxxxxx
    // -> 37 + xxxxxxx. Each codeword's bits are read once, in order.
    if reader.read_bits_with_stuffing(1)? == 0 {
        return Some(1);
    }
    if reader.read_bits_with_stuffing(1)? == 0 {
        return Some(2);
    }
    let two = reader.read_bits_with_stuffing(2)?;
    if two != 0b11 {
        return u8::try_from(two + 3).ok();
    }
    let five = reader.read_bits_with_stuffing(5)?;
    if five != 0b1_1111 {
        return u8::try_from(five + 6).ok();
    }
    u8::try_from(reader.read_bits_with_stuffing(7)? + 37).ok()
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
    fn sop_sequence_number_wraps_after_u16_max() {
        for (ordinal, sequence) in [
            (0, 0_u16),
            (usize::from(u16::MAX), u16::MAX),
            (usize::from(u16::MAX) + 1, 0),
            (usize::from(u16::MAX) + 2, 1),
        ] {
            let sequence = sequence.to_be_bytes();
            let bytes = [0xff, SOP, 0, 4, sequence[0], sequence[1]];
            let mut reader = BitReader::new(&bytes);
            assert_eq!(parse_sop_marker(&mut reader, ordinal), Ok(()));
            assert!(reader.at_end());
        }
    }

    /// The original chain of separate peeks, as the reference for
    /// `decode_num_coding_passes`.
    fn coding_passes_by_peeks(reader: &mut BitReader<'_>) -> Option<u8> {
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

    #[test]
    fn coding_pass_counts_match_separate_prefix_peeks() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..3000 {
            let len = usize::try_from(next() % 6).expect("small length");
            let data: Vec<u8> = (0..len)
                .map(|_| match next() % 3 {
                    0 => 0xff,
                    _ => u8::try_from(next() >> 56).expect("top byte"),
                })
                .collect();
            let skip = u8::try_from(next() % 8).expect("bit offset");
            let mut fast = BitReader::new(&data);
            if fast.read_bits_with_stuffing(skip).is_none() {
                continue;
            }
            let mut reference = fast.clone();
            loop {
                let (a, b) = (
                    decode_num_coding_passes(&mut fast),
                    coding_passes_by_peeks(&mut reference),
                );
                assert_eq!(a, b, "{data:02x?} skip {skip}");
                if a.is_none() {
                    break;
                }
                assert_eq!(fast.offset(), reference.offset());
                assert_eq!(
                    fast.clone().read_bits_with_stuffing(1),
                    reference.clone().read_bits_with_stuffing(1)
                );
            }
        }
    }

    /// The segments a non-strict walk records for `codestream`'s tile.
    fn non_strict_segments(codestream: &[u8]) -> Vec<(u8, u8, u32)> {
        use crate::j2c::{build, tile};

        let mut parsed = crate::j2c::parse_raw(codestream, &crate::DecodeSettings::default())
            .expect("parse codestream");
        parsed.header.strict = false;
        let tiles =
            tile::parse(&mut BitReader::new(parsed.data), &parsed.header, 0).expect("parse tiles");
        let tile = tiles.iter().next().expect("one tile");
        let mut storage = DecompositionStorage::default();
        build::build(
            tile,
            &mut storage,
            tiles.structural_workspace_bytes(),
            false,
            build::BuildWorkspace::MetadataOnly,
            None,
        )
        .expect("plan code-blocks");
        parse(tile, &parsed.header, &mut storage).expect("non-strict walk");
        storage
            .segments
            .iter()
            .map(|segment| (segment.idx, segment.coding_pases, segment.data_length))
            .collect()
    }

    /// A walk that goes on past a tile-part's error, as a non-strict one
    /// does, parses the next tile-parts from where the progression stands: a
    /// PLT length left after the first tile-part's packet takes no packet
    /// from it.
    #[test]
    fn non_strict_walk_keeps_the_progression_past_surplus_packet_lengths() {
        let mut state = 0x2545_f491_u32;
        let pixels: Vec<u8> = (0..16 * 16)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state.to_be_bytes()[0]
            })
            .collect();
        let options = crate::EncodeOptions {
            num_decomposition_levels: 1,
            num_layers: 2,
            tile_part_packet_limit: Some(1),
            write_plt: true,
            ..crate::EncodeOptions::default()
        };
        let clean = crate::encode(&pixels, 16, 16, 1, 8, false, &options).expect("encode");
        let sot = clean
            .windows(2)
            .position(|window| window == [0xff, 0x90])
            .expect("SOT marker");
        let plt = sot
            + clean[sot..]
                .windows(2)
                .position(|window| window == [0xff, 0x58])
                .expect("PLT marker");
        let mut edited = clean.clone();
        let plt_length = u16::from_be_bytes([edited[plt + 2], edited[plt + 3]]);
        edited.insert(plt + 2 + usize::from(plt_length), 5);
        edited[plt + 2..plt + 4].copy_from_slice(&(plt_length + 1).to_be_bytes());
        let psot = u32::from_be_bytes(edited[sot + 6..sot + 10].try_into().expect("Psot"));
        edited[sot + 6..sot + 10].copy_from_slice(&(psot + 1).to_be_bytes());

        let segments = non_strict_segments(&clean);
        assert!(segments.len() > 1, "packets after the first tile-part");
        assert_eq!(non_strict_segments(&edited), segments);
    }

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
