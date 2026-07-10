//! Creating tiles and parsing their constituent tile parts.

use alloc::vec;
use alloc::vec::Vec;
use core::mem::size_of;

use super::build::{PrecinctData, SubBandType};
use super::codestream::{
    markers, skip_marker_segment, ComponentInfo, Header, ProgressionChange, ProgressionOrder,
};
use super::rect::IntRect;
use crate::error::{bail, err, DecodingError, MarkerError, Result, TileError, ValidationError};
use crate::j2c::codestream;
use crate::reader::BitReader;
use crate::DEFAULT_MAX_DECODE_BYTES;

fn ceil_div_by_power_of_two(value: u32, exponent: u8) -> u32 {
    if exponent == 0 {
        value
    } else if u32::from(exponent) >= u32::BITS {
        u32::from(value != 0)
    } else {
        value.div_ceil(1_u32 << exponent)
    }
}

fn subband_coordinate(value: u32, decomposition_level: u8, high_pass: bool) -> u32 {
    let adjusted = if !high_pass || decomposition_level == 0 {
        value
    } else if u32::from(decomposition_level) > u32::BITS {
        0
    } else {
        let offset = 1_u32 << (decomposition_level - 1);
        value.saturating_sub(offset)
    };
    ceil_div_by_power_of_two(adjusted, decomposition_level)
}

/// A single tile in the image.
#[derive(Clone, Debug)]
#[expect(
    clippy::struct_field_names,
    reason = "tile_parts is the JPEG 2000 specification term for a tile's ordered parts"
)]
pub(crate) struct Tile<'a> {
    /// The index of the tile, in row-major order.
    pub(crate) idx: u32,
    /// The concatenated tile parts that contain all the information for all
    /// constituent codeblocks.
    pub(crate) tile_parts: Vec<TilePart<'a>>,
    /// Parameters for each component. In most cases, those are directly
    /// inherited from the main header. But in some cases, individual tiles
    /// might override them.
    pub(crate) component_infos: Vec<ComponentInfo>,
    /// The rectangle making up the area of the tile. `x1` and `y1` are
    /// exclusive.
    pub(crate) rect: IntRect,
    pub(crate) progression_order: ProgressionOrder,
    pub(crate) progression_changes: Vec<ProgressionChange>,
    pub(crate) num_layers: u8,
    pub(crate) mct: bool,
}

/// A tile part where packet headers and packet data are interleaved.
#[derive(Clone, Debug)]
pub(crate) struct MergedTilePart<'a> {
    pub(crate) data: BitReader<'a>,
    packet_lengths: PacketLengthMetadata,
}

/// A tile part where packet headers and packet data are separated.
#[derive(Clone, Debug)]
pub(crate) struct SeparatedTilePart<'a> {
    pub(crate) headers: Vec<BitReader<'a>>,
    pub(crate) active_header_reader: usize,
    pub(crate) body: BitReader<'a>,
    packet_lengths: PacketLengthMetadata,
}

#[derive(Clone, Debug)]
pub(crate) enum TilePart<'a> {
    Merged(MergedTilePart<'a>),
    Separated(SeparatedTilePart<'a>),
}

#[derive(Clone, Debug, Default)]
struct PacketLengthMetadata {
    present: bool,
    lengths: Vec<u32>,
    next: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PacketLengthExpectation {
    NotTracked,
    Length(u32),
}

impl PacketLengthMetadata {
    fn new(present: bool, lengths: Vec<u32>) -> Self {
        Self {
            present,
            lengths,
            next: 0,
        }
    }

    fn is_present(&self) -> bool {
        self.present
    }

    fn next(&mut self) -> Option<PacketLengthExpectation> {
        if !self.present {
            return Some(PacketLengthExpectation::NotTracked);
        }

        let packet_length = self.lengths.get(self.next).copied()?;
        self.next += 1;
        Some(PacketLengthExpectation::Length(packet_length))
    }

    fn fully_consumed(&self) -> bool {
        !self.present || self.next == self.lengths.len()
    }
}

impl<'a> TilePart<'a> {
    pub(crate) fn header(&mut self) -> &mut BitReader<'a> {
        match self {
            TilePart::Merged(m) => &mut m.data,
            TilePart::Separated(s) => {
                if s.headers[s.active_header_reader].at_end()
                    && s.headers.len() - 1 > s.active_header_reader
                {
                    s.active_header_reader += 1;
                }

                &mut s.headers[s.active_header_reader]
            }
        }
    }

    pub(crate) fn body(&mut self) -> &mut BitReader<'a> {
        match self {
            TilePart::Merged(m) => &mut m.data,
            TilePart::Separated(s) => &mut s.body,
        }
    }

    pub(crate) fn packet_start_offset(&self) -> Option<usize> {
        match self {
            TilePart::Merged(m) if m.packet_lengths.is_present() => Some(m.data.offset()),
            TilePart::Separated(_) | TilePart::Merged(_) => None,
        }
    }

    pub(crate) fn validate_packet_length(&mut self, packet_start: Option<usize>) -> Option<()> {
        let expected = match self {
            TilePart::Merged(m) => m.packet_lengths.next()?,
            TilePart::Separated(s) => s.packet_lengths.next()?,
        };

        let expected = match expected {
            PacketLengthExpectation::NotTracked => return Some(()),
            PacketLengthExpectation::Length(expected) => expected,
        };

        let packet_start = packet_start?;
        let actual = match self {
            TilePart::Merged(m) => m.data.offset().checked_sub(packet_start)?,
            TilePart::Separated(_) => return Some(()),
        };

        if actual != expected as usize {
            return None;
        }
        Some(())
    }

    pub(crate) fn validate_all_packet_lengths_consumed(&self) -> Option<()> {
        match self {
            TilePart::Merged(m) => m.packet_lengths.fully_consumed().then_some(()),
            TilePart::Separated(s) => s.packet_lengths.fully_consumed().then_some(()),
        }
    }
}

impl Tile<'_> {
    fn new(idx: u32, header: &Header<'_>) -> Self {
        let rect = {
            let size_data = &header.size_data;

            let x_coord = size_data.tile_x_coord(idx);
            let y_coord = size_data.tile_y_coord(idx);

            // See B-7, B-8, B-9 and B-10. Saturating arithmetic: the results
            // are clamped against the reference grid anyway, and a saturated
            // intermediate already exceeds every in-grid bound, so the clamp
            // still produces the mathematically correct edge for crafted
            // headers whose products overflow u32.
            let x0 = u32::max(
                size_data
                    .tile_x_offset
                    .saturating_add(x_coord.saturating_mul(size_data.tile_width)),
                size_data.image_area_x_offset,
            );
            let y0 = u32::max(
                size_data
                    .tile_y_offset
                    .saturating_add(y_coord.saturating_mul(size_data.tile_height)),
                size_data.image_area_y_offset,
            );

            // Note that `x1` and `y1` are exclusive.
            let x1 = u32::min(
                size_data
                    .tile_x_offset
                    .saturating_add((x_coord + 1).saturating_mul(size_data.tile_width)),
                size_data.reference_grid_width,
            );
            let y1 = u32::min(
                size_data
                    .tile_y_offset
                    .saturating_add((y_coord + 1).saturating_mul(size_data.tile_height)),
                size_data.reference_grid_height,
            );

            IntRect::from_ltrb(x0, y0, x1, y1)
        };

        Tile {
            idx,
            // Will be filled once we start parsing.
            tile_parts: vec![],
            rect,
            // By default, each tile inherits the settings from the main
            // header. When parsing the tile parts, some of these settings
            // might be overridden.
            component_infos: header.component_infos.clone(),
            progression_order: header.global_coding_style.progression_order,
            progression_changes: header.progression_changes.clone(),
            mct: header.global_coding_style.mct,
            num_layers: header.global_coding_style.num_layers,
        }
    }

    pub(crate) fn component_tiles(&self) -> impl Iterator<Item = ComponentTile<'_>> {
        self.component_infos
            .iter()
            .map(|i| ComponentTile::new(self, i))
    }
}

/// Create the tiles and parse their constituent tile parts.
pub(crate) fn parse<'a>(
    reader: &mut BitReader<'a>,
    main_header: &Header<'a>,
) -> Result<Vec<Tile<'a>>> {
    validate_tile_structural_budget(main_header)?;

    let mut tiles = (0..main_header.size_data.num_tiles())
        .map(|idx| Tile::new(idx, main_header))
        .collect::<Vec<_>>();

    let mut tile_part_idx = 0;

    parse_tile_part(reader, main_header, &mut tiles, tile_part_idx)?;
    tile_part_idx += 1;

    while reader.peek_marker() == Some(markers::SOT) {
        parse_tile_part(reader, main_header, &mut tiles, tile_part_idx)?;
        tile_part_idx += 1;
    }

    if main_header.strict && reader.read_marker()? != markers::EOC {
        bail!(MarkerError::Expected("EOC"));
    }

    Ok(tiles)
}

fn validate_tile_structural_budget(main_header: &Header<'_>) -> Result<()> {
    let num_tiles = usize::try_from(main_header.size_data.num_tiles())
        .map_err(|_| ValidationError::ImageTooLarge)?;
    let component_count = main_header.component_infos.len();
    let progression_change_count = main_header.progression_changes.len();

    let per_tile_components = size_of::<ComponentInfo>()
        .checked_mul(component_count)
        .ok_or(ValidationError::ImageTooLarge)?;
    let per_tile_progression_changes = size_of::<ProgressionChange>()
        .checked_mul(progression_change_count)
        .ok_or(ValidationError::ImageTooLarge)?;
    let per_tile_bytes = size_of::<Tile<'static>>()
        .checked_add(per_tile_components)
        .and_then(|bytes| bytes.checked_add(per_tile_progression_changes))
        .ok_or(ValidationError::ImageTooLarge)?;
    let total_bytes = per_tile_bytes
        .checked_mul(num_tiles)
        .ok_or(ValidationError::ImageTooLarge)?;

    if total_bytes > DEFAULT_MAX_DECODE_BYTES {
        bail!(ValidationError::ImageTooLarge);
    }
    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "the ordered JPEG 2000 state machine stays cohesive to preserve marker, packet, pass, and sample order"
)]
fn parse_tile_part<'a>(
    reader: &mut BitReader<'a>,
    main_header: &Header<'a>,
    tiles: &mut [Tile<'a>],
    tile_part_idx: usize,
) -> Result<()> {
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

    let mut packet_length_markers = vec![];
    let mut packet_lengths_present = false;
    let mut ppt_headers = vec![];

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
                let cod = codestream::cod_marker(reader).ok_or(MarkerError::ParseFailure("COD"))?;

                tile.mct = cod.mct;
                tile.num_layers = cod.num_layers;
                tile.progression_order = cod.progression_order;

                for component in &mut tile.component_infos {
                    component.coding_style.flags.raw |= cod.component_parameters.flags.raw;
                    component.coding_style.parameters = cod.component_parameters.clone().parameters;
                }
            }
            markers::COC => {
                reader.read_marker()?;

                let (component_index, coc) = codestream::coc_marker(reader, num_components)
                    .ok_or(MarkerError::ParseFailure("COC"))?;

                let old = tile
                    .component_infos
                    .get_mut(component_index as usize)
                    .ok_or(ValidationError::InvalidComponentMetadata)?;

                old.coding_style.parameters = coc.parameters;
                old.coding_style.flags.raw |= coc.flags.raw;
            }
            markers::QCD => {
                reader.read_marker()?;
                let qcd = codestream::qcd_marker(reader).ok_or(MarkerError::ParseFailure("QCD"))?;

                for component_info in &mut tile.component_infos {
                    component_info.quantization_info = qcd.clone();
                }
            }
            markers::QCC => {
                reader.read_marker()?;
                let (component_index, qcc) = codestream::qcc_marker(reader, num_components)
                    .ok_or(MarkerError::ParseFailure("QCC"))?;

                tile.component_infos
                    .get_mut(component_index as usize)
                    .ok_or(ValidationError::InvalidComponentMetadata)?
                    .quantization_info = qcc.clone();
            }
            markers::POC => {
                reader.read_marker()?;
                tile.progression_changes.extend(
                    codestream::poc_marker(reader, num_components, tile.num_layers)
                        .ok_or(MarkerError::ParseFailure("POC"))?,
                );
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
                ppt_headers.push(ppt_marker(reader).ok_or(MarkerError::ParseFailure("PPT"))?);
            }
            markers::PLT => {
                reader.read_marker()?;
                packet_lengths_present = true;
                packet_length_markers
                    .push(codestream::plt_marker(reader).ok_or(MarkerError::ParseFailure("PLT"))?);
            }
            markers::COM => {
                reader.read_marker()?;
                skip_marker_segment(reader).ok_or(MarkerError::ParseFailure("COM"))?;
            }
            (0x30..=0x3F) => {
                // "All markers with the marker code between 0xFF30 and 0xFF3F
                // have no marker segment parameters. They shall be skipped by
                // the decoder."
                reader.read_marker()?;
                // skip_marker_segment(reader);
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

    ppt_headers.sort_by_key(|ppt_header| ppt_header.sequence_idx);
    let mut headers: Vec<_> = ppt_headers.iter().map(|i| BitReader::new(i.data)).collect();
    packet_length_markers.sort_by_key(|marker| marker.sequence_idx);
    let use_main_header_packet_lengths = !packet_lengths_present
        && !main_header.plm_packet_lengths.is_empty()
        && main_header.size_data.num_tiles() == 1
        && tile_part_header.tile_part_index == 0
        && tile_part_header.num_tile_parts == 1;
    let packet_lengths = if use_main_header_packet_lengths {
        PacketLengthMetadata::new(true, main_header.plm_packet_lengths.clone())
    } else {
        PacketLengthMetadata::new(
            packet_lengths_present,
            packet_length_markers
                .into_iter()
                .flat_map(|marker| marker.packet_lengths)
                .collect(),
        )
    };

    if let Some(ppm_marker) = main_header.ppm_packets.get(tile_part_idx) {
        headers.push(BitReader::new(ppm_marker.data));
    }

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
            active_header_reader: 0,
            body: BitReader::new(data),
            packet_lengths,
        })
    };

    tile.tile_parts.push(tile_part);

    Ok(())
}

/// A tile, instantiated to a specific component.
#[derive(Debug, Copy, Clone)]
pub(crate) struct ComponentTile<'a> {
    pub(crate) tile: &'a Tile<'a>,
    /// The information of the component of the tile.
    pub(crate) component_info: &'a ComponentInfo,
    /// The rectangle of the component tile.
    pub(crate) rect: IntRect,
}

impl<'a> ComponentTile<'a> {
    #[expect(
        clippy::similar_names,
        reason = "paired axis, subband, and marker names follow JPEG 2000 specification notation"
    )]
    pub(crate) fn new(tile: &'a Tile<'a>, component_info: &'a ComponentInfo) -> Self {
        let tile_rect = tile.rect;

        let rect = if component_info.size_info.horizontal_resolution == 1
            && component_info.size_info.vertical_resolution == 1
        {
            tile_rect
        } else {
            // As described in B-12.
            let t_x0 = tile_rect
                .x0
                .div_ceil(u32::from(component_info.size_info.horizontal_resolution));
            let t_y0 = tile_rect
                .y0
                .div_ceil(u32::from(component_info.size_info.vertical_resolution));
            let t_x1 = tile_rect
                .x1
                .div_ceil(u32::from(component_info.size_info.horizontal_resolution));
            let t_y1 = tile_rect
                .y1
                .div_ceil(u32::from(component_info.size_info.vertical_resolution));

            IntRect::from_ltrb(t_x0, t_y0, t_x1, t_y1)
        };

        ComponentTile {
            tile,
            component_info,
            rect,
        }
    }

    pub(crate) fn resolution_tiles(&self) -> impl Iterator<Item = ResolutionTile<'_>> {
        (0..self
            .component_info
            .coding_style
            .parameters
            .num_resolution_levels)
            .map(|r| ResolutionTile::new(*self, r))
    }
}

/// A tile instantiated to a specific resolution of a component tile.
pub(crate) struct ResolutionTile<'a> {
    /// The resolution of the tile.
    pub(crate) resolution: u8,
    /// The decomposition level of the tile.
    pub(crate) decomposition_level: u8,
    /// The underlying component tile.
    pub(crate) component_tile: ComponentTile<'a>,
    /// The rectangle of the resolution tile.
    pub(crate) rect: IntRect,
}

impl<'a> ResolutionTile<'a> {
    pub(crate) fn new(component_tile: ComponentTile<'a>, resolution: u8) -> Self {
        assert!(
            component_tile
                .component_info
                .coding_style
                .parameters
                .num_resolution_levels
                > resolution
        );

        let rect = {
            // See formula B-14.
            let n_l = component_tile
                .component_info
                .coding_style
                .parameters
                .num_decomposition_levels;

            let scale = n_l - resolution;
            let tx0 = ceil_div_by_power_of_two(component_tile.rect.x0, scale);
            let ty0 = ceil_div_by_power_of_two(component_tile.rect.y0, scale);
            let tx1 = ceil_div_by_power_of_two(component_tile.rect.x1, scale);
            let ty1 = ceil_div_by_power_of_two(component_tile.rect.y1, scale);

            IntRect::from_ltrb(tx0, ty0, tx1, ty1)
        };

        // Decomposition level and resolution level are inversely related
        // to each other. In addition to that, there is always one more
        // resolution than decomposition levels (resolution level 0 only
        // include the LL subband of the N_L decomposition, resolution level
        // 1 includes the HL, LH and HH subbands of the N_L decomposition.
        let decomposition_level = {
            if resolution == 0 {
                component_tile
                    .component_info
                    .coding_style
                    .parameters
                    .num_decomposition_levels
            } else {
                component_tile
                    .component_info
                    .coding_style
                    .parameters
                    .num_decomposition_levels
                    - (resolution - 1)
            }
        };

        ResolutionTile {
            resolution,
            decomposition_level,
            component_tile,
            rect,
        }
    }

    #[expect(
        clippy::similar_names,
        reason = "paired axis, subband, and marker names follow JPEG 2000 specification notation"
    )]
    pub(crate) fn sub_band_rect(&self, sub_band_type: SubBandType) -> IntRect {
        // This is the only permissible sub-band type for the given resolution.
        if self.resolution == 0 {
            assert_eq!(sub_band_type, SubBandType::LowLow);
        }

        // Formula B-15.

        let high_pass_x = matches!(sub_band_type, SubBandType::HighLow | SubBandType::HighHigh);
        let high_pass_y = matches!(sub_band_type, SubBandType::LowHigh | SubBandType::HighHigh);

        let tbx_0 = subband_coordinate(
            self.component_tile.rect.x0,
            self.decomposition_level,
            high_pass_x,
        );
        let tbx_1 = subband_coordinate(
            self.component_tile.rect.x1,
            self.decomposition_level,
            high_pass_x,
        );
        let tby_0 = subband_coordinate(
            self.component_tile.rect.y0,
            self.decomposition_level,
            high_pass_y,
        );
        let tby_1 = subband_coordinate(
            self.component_tile.rect.y1,
            self.decomposition_level,
            high_pass_y,
        );

        IntRect::from_ltrb(tbx_0, tby_0, tbx_1, tby_1)
    }

    /// The exponent for determining the horizontal size of a precinct.
    ///
    /// `PPx` in the specification.
    fn precinct_exponent_x(&self) -> u8 {
        self.component_tile
            .component_info
            .coding_style
            .parameters
            .precinct_exponents[self.resolution as usize]
            .0
    }

    /// The exponent for determining the vertical size of a precinct.
    ///
    /// `PPx` in the specification.
    fn precinct_exponent_y(&self) -> u8 {
        self.component_tile
            .component_info
            .coding_style
            .parameters
            .precinct_exponents[self.resolution as usize]
            .1
    }

    fn num_precincts_x(&self) -> u32 {
        // See B-16.
        let IntRect { x0, x1, .. } = self.rect;

        if x0 == x1 {
            0
        } else {
            x1.div_ceil(2_u32.pow(u32::from(self.precinct_exponent_x())))
                - x0 / 2_u32.pow(u32::from(self.precinct_exponent_x()))
        }
    }

    fn num_precincts_y(&self) -> u32 {
        // See B-16.
        let IntRect { y0, y1, .. } = self.rect;

        if y0 == y1 {
            0
        } else {
            y1.div_ceil(2_u32.pow(u32::from(self.precinct_exponent_y())))
                - y0 / 2_u32.pow(u32::from(self.precinct_exponent_y()))
        }
    }

    pub(crate) fn num_precincts(&self) -> u64 {
        u64::from(self.num_precincts_x()) * u64::from(self.num_precincts_y())
    }

    /// Return an iterator over the data of the precincts in this resolution
    /// tile.
    #[expect(
        clippy::similar_names,
        reason = "paired axis, subband, and marker names follow JPEG 2000 specification notation"
    )]
    pub(crate) fn precincts(&self) -> Option<impl Iterator<Item = PrecinctData>> {
        let num_precincts_y = self.num_precincts_y();
        let num_precincts_x = self.num_precincts_x();

        let mut ppx = self.precinct_exponent_x();
        let mut ppy = self.precinct_exponent_y();

        let mut y_start = (self.rect.y0 / (1 << ppy)) * (1 << ppy);
        let mut x_start = (self.rect.x0 / (1 << ppx)) * (1 << ppx);

        // It is unclear why this is necessary, but it is. The spec only
        // mentions that ppx/ppy must be decreased when calculating codeblock
        // dimensions, but not that it's necessary for precincts as well.
        if self.resolution > 0 {
            ppx = ppx.checked_sub(1)?;
            ppy = ppy.checked_sub(1)?;

            x_start /= 2;
            y_start /= 2;
        }

        let ppx_pow2 = 1_u32 << ppx;
        let ppy_pow2 = 1_u32 << ppy;

        let nl_minus_r = self
            .component_tile
            .component_info
            .num_decomposition_levels()
            - self.resolution;

        let x_stride = 1_u32.checked_shl(u32::from(
            self.precinct_exponent_x().checked_add(nl_minus_r)?,
        ))?;
        let y_stride = 1_u32.checked_shl(u32::from(
            self.precinct_exponent_y().checked_add(nl_minus_r)?,
        ))?;

        let precinct_x_step = u32::from(
            self.component_tile
                .component_info
                .size_info
                .horizontal_resolution,
        )
        .checked_mul(x_stride)?;

        let precinct_y_step = u32::from(
            self.component_tile
                .component_info
                .size_info
                .vertical_resolution,
        )
        .checked_mul(y_stride)?;

        // These variables are used to map the start coordinates of each
        // precinct _on the reference grid_. Remember that the first
        // precinct in each row/column is at the start position of the tile
        // which might not be a multiple of precinct exponent, but all subsequent
        // precincts are at a multiple of the exponent.
        let mut r_x = self.component_tile.tile.rect.x0;
        let mut r_y = self.component_tile.tile.rect.y0;

        // The second part of the condition in the formula in B.12.1.3. If it
        // is divisible, then we can't take the x/y position of the tile
        // as the start of the precinct, but instead have to advance to the
        // next multiple.
        if !r_x.is_multiple_of(precinct_x_step)
            && (self.rect.x0 * (1 << nl_minus_r)).is_multiple_of(precinct_x_step)
        {
            r_x = r_x.checked_next_multiple_of(precinct_x_step)?;
        }

        // Same as above.
        if !r_y.is_multiple_of(precinct_y_step)
            && (self.rect.y0 * (1 << nl_minus_r)).is_multiple_of(precinct_y_step)
        {
            r_y = r_y.checked_next_multiple_of(precinct_y_step)?;
        }

        let iter = (0..num_precincts_y).flat_map(move |y| {
            let y0 = y * ppy_pow2 + y_start;
            let mut r_x = r_x;

            let res = (0..num_precincts_x).map(move |x| {
                let x0 = x * ppx_pow2 + x_start;

                let data = PrecinctData {
                    r_x,
                    r_y,
                    rect: IntRect::from_xywh(x0, y0, ppx_pow2, ppy_pow2),
                    idx: u64::from(num_precincts_x) * u64::from(y) + u64::from(x),
                };

                // If r_x is already aligned, we simply step by `precinct_x_step`.
                // Otherwise (can only be the case for precincts in the first
                // row or column), align to the next multiple.
                r_x = (r_x + 1).next_multiple_of(precinct_x_step);

                data
            });

            // Same as for r_x.
            r_y = (r_y + 1).next_multiple_of(precinct_y_step);

            res
        });

        Some(iter)
    }

    pub(crate) fn code_block_width(&self) -> u32 {
        // See B-17.
        let xcb = self
            .component_tile
            .component_info
            .coding_style
            .parameters
            .code_block_width;

        let xcb = if self.resolution > 0 {
            u8::min(xcb, self.precinct_exponent_x() - 1)
        } else {
            u8::min(xcb, self.precinct_exponent_x())
        };

        2_u32.pow(u32::from(xcb))
    }

    pub(crate) fn code_block_height(&self) -> u32 {
        // See B-18.
        let ycb = self
            .component_tile
            .component_info
            .coding_style
            .parameters
            .code_block_height;

        let ycb = if self.resolution > 0 {
            u8::min(ycb, self.precinct_exponent_y() - 1)
        } else {
            u8::min(ycb, self.precinct_exponent_y())
        };

        2_u32.pow(u32::from(ycb))
    }
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

/// PPT marker (A.7.5).
fn ppt_marker<'a>(reader: &mut BitReader<'a>) -> Option<PptMarkerData<'a>> {
    let length = reader.read_u16()?.checked_sub(2)?;
    let header_len = length.checked_sub(1)?;
    let sequence_idx = reader.read_byte()?;
    Some(PptMarkerData {
        data: reader.read_bytes(header_len as usize)?,
        sequence_idx,
    })
}

/// SOT marker (A.4.2).
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
mod tests {
    use super::*;
    use crate::j2c::codestream::{
        CodeBlockStyle, CodingStyleComponent, CodingStyleDefault, CodingStyleFlags,
        CodingStyleParameters, ComponentSizeInfo, QuantizationInfo, QuantizationStyle, SizeData,
        WaveletTransform,
    };

    #[test]
    fn power_of_two_geometry_preserves_u32_boundaries() {
        assert_eq!(ceil_div_by_power_of_two(u32::MAX, 0), u32::MAX);
        assert_eq!(ceil_div_by_power_of_two(u32::MAX, 31), 2);
        assert_eq!(ceil_div_by_power_of_two(u32::MAX, 32), 1);
        assert_eq!(ceil_div_by_power_of_two(0, u8::MAX), 0);
        assert_eq!(subband_coordinate(u32::MAX, 32, true), 1);
        assert_eq!(subband_coordinate(u32::MAX, 33, true), 0);
    }

    /// Test case for the example in B.4.
    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "the complete JPEG 2000 B.4 fixture stays together so its geometry remains reviewable"
    )]
    fn test_jpeg2000_standard_example_b4() {
        let component_size_info_0 = ComponentSizeInfo {
            precision: 8,
            signed: false,
            horizontal_resolution: 1,
            vertical_resolution: 1,
        };

        let dummy_component_coding_style = CodingStyleComponent {
            flags: CodingStyleFlags::default(),
            parameters: CodingStyleParameters {
                num_decomposition_levels: 0,
                num_resolution_levels: 0,
                code_block_width: 0,
                code_block_height: 0,
                code_block_style: CodeBlockStyle::default(),
                transformation: WaveletTransform::Irreversible97,
                precinct_exponents: vec![],
            },
        };

        let dummy_quantization_info = QuantizationInfo {
            quantization_style: QuantizationStyle::NoQuantization,
            guard_bits: 0,
            step_sizes: vec![],
        };

        let component_info_0 = ComponentInfo {
            size_info: component_size_info_0,
            coding_style: dummy_component_coding_style.clone(),
            quantization_info: dummy_quantization_info.clone(),
            roi_shift: 0,
        };

        let component_size_info_1 = ComponentSizeInfo {
            precision: 8,
            signed: false,
            horizontal_resolution: 2,
            vertical_resolution: 2,
        };

        let component_info_1 = ComponentInfo {
            size_info: component_size_info_1,
            coding_style: dummy_component_coding_style.clone(),
            quantization_info: dummy_quantization_info.clone(),
            roi_shift: 0,
        };

        let size_data = SizeData {
            reference_grid_width: 1432,
            reference_grid_height: 954,
            image_area_x_offset: 152,
            image_area_y_offset: 234,
            tile_width: 396,
            tile_height: 297,
            tile_x_offset: 0,
            tile_y_offset: 0,
            component_sizes: vec![component_size_info_0, component_size_info_1],
            x_shrink_factor: 1,
            y_shrink_factor: 1,
            x_resolution_shrink_factor: 1,
            y_resolution_shrink_factor: 1,
        };

        assert_eq!(size_data.image_width(), 1280);
        assert_eq!(size_data.image_height(), 720);

        assert_eq!(size_data.num_x_tiles(), 4);
        assert_eq!(size_data.num_y_tiles(), 4);
        assert_eq!(size_data.num_tiles(), 16);

        let header = Header {
            size_data,
            // Just dummy values.
            global_coding_style: CodingStyleDefault {
                progression_order: ProgressionOrder::LayerResolutionComponentPosition,
                num_layers: 0,
                mct: false,
                component_parameters: CodingStyleComponent {
                    flags: CodingStyleFlags::default(),
                    parameters: CodingStyleParameters {
                        num_decomposition_levels: 0,
                        num_resolution_levels: 0,
                        code_block_width: 0,
                        code_block_height: 0,
                        code_block_style: CodeBlockStyle::default(),
                        transformation: WaveletTransform::Irreversible97,
                        precinct_exponents: vec![],
                    },
                },
            },
            component_infos: vec![],
            progression_changes: vec![],
            plm_packet_lengths: vec![],
            ppm_packets: vec![],
            skipped_resolution_levels: 0,
            strict: false,
        };

        let tile_0_0 = Tile::new(0, &header);
        let coords_0_0 = ComponentTile::new(&tile_0_0, &component_info_0).rect;
        assert_eq!(coords_0_0.x0, 152);
        assert_eq!(coords_0_0.y0, 234);
        assert_eq!(coords_0_0.x1, 396);
        assert_eq!(coords_0_0.y1, 297);
        assert_eq!(coords_0_0.width(), 244);
        assert_eq!(coords_0_0.height(), 63);

        let tile_1_0 = Tile::new(1, &header);
        let coords_1_0 = ComponentTile::new(&tile_1_0, &component_info_0).rect;
        assert_eq!(coords_1_0.x0, 396);
        assert_eq!(coords_1_0.y0, 234);
        assert_eq!(coords_1_0.x1, 792);
        assert_eq!(coords_1_0.y1, 297);
        assert_eq!(coords_1_0.width(), 396);
        assert_eq!(coords_1_0.height(), 63);

        let tile_0_1 = Tile::new(4, &header);
        let coords_0_1 = ComponentTile::new(&tile_0_1, &component_info_0).rect;
        assert_eq!(coords_0_1.x0, 152);
        assert_eq!(coords_0_1.y0, 297);
        assert_eq!(coords_0_1.x1, 396);
        assert_eq!(coords_0_1.y1, 594);
        assert_eq!(coords_0_1.width(), 244);
        assert_eq!(coords_0_1.height(), 297);

        let tile_1_1 = Tile::new(5, &header);
        let coords_1_1 = ComponentTile::new(&tile_1_1, &component_info_0).rect;
        assert_eq!(coords_1_1.x0, 396);
        assert_eq!(coords_1_1.y0, 297);
        assert_eq!(coords_1_1.x1, 792);
        assert_eq!(coords_1_1.y1, 594);
        assert_eq!(coords_1_1.width(), 396);
        assert_eq!(coords_1_1.height(), 297);

        let tile_3_3 = Tile::new(15, &header);
        let coords_3_3 = ComponentTile::new(&tile_3_3, &component_info_0).rect;
        assert_eq!(coords_3_3.x0, 1188);
        assert_eq!(coords_3_3.y0, 891);
        assert_eq!(coords_3_3.x1, 1432);
        assert_eq!(coords_3_3.y1, 954);
        assert_eq!(coords_3_3.width(), 244);
        assert_eq!(coords_3_3.height(), 63);

        let tile_0_0_comp1 = ComponentTile::new(&tile_0_0, &component_info_1).rect;
        assert_eq!(tile_0_0_comp1.x0, 76);
        assert_eq!(tile_0_0_comp1.y0, 117);
        assert_eq!(tile_0_0_comp1.x1, 198);
        assert_eq!(tile_0_0_comp1.y1, 149);
        assert_eq!(tile_0_0_comp1.width(), 122);
        assert_eq!(tile_0_0_comp1.height(), 32);

        let tile_1_0_comp1 = ComponentTile::new(&tile_1_0, &component_info_1).rect;
        assert_eq!(tile_1_0_comp1.x0, 198);
        assert_eq!(tile_1_0_comp1.y0, 117);
        assert_eq!(tile_1_0_comp1.x1, 396);
        assert_eq!(tile_1_0_comp1.y1, 149);
        assert_eq!(tile_1_0_comp1.width(), 198);
        assert_eq!(tile_1_0_comp1.height(), 32);

        let tile_0_1_comp1 = ComponentTile::new(&tile_0_1, &component_info_1).rect;
        assert_eq!(tile_0_1_comp1.x0, 76);
        assert_eq!(tile_0_1_comp1.y0, 149);
        assert_eq!(tile_0_1_comp1.x1, 198);
        assert_eq!(tile_0_1_comp1.y1, 297);
        assert_eq!(tile_0_1_comp1.width(), 122);
        assert_eq!(tile_0_1_comp1.height(), 148);

        let tile_1_1_comp1 = ComponentTile::new(&tile_1_1, &component_info_1).rect;
        assert_eq!(tile_1_1_comp1.x0, 198);
        assert_eq!(tile_1_1_comp1.y0, 149);
        assert_eq!(tile_1_1_comp1.x1, 396);
        assert_eq!(tile_1_1_comp1.y1, 297);
        assert_eq!(tile_1_1_comp1.width(), 198);
        assert_eq!(tile_1_1_comp1.height(), 148);

        let tile_2_1 = Tile::new(6, &header);
        let tile_2_1_comp1 = ComponentTile::new(&tile_2_1, &component_info_1).rect;
        assert_eq!(tile_2_1_comp1.x0, 396);
        assert_eq!(tile_2_1_comp1.y0, 149);
        assert_eq!(tile_2_1_comp1.x1, 594);
        assert_eq!(tile_2_1_comp1.y1, 297);
        assert_eq!(tile_2_1_comp1.width(), 198);
        assert_eq!(tile_2_1_comp1.height(), 148);

        assert_eq!(tile_1_1_comp1.width(), tile_2_1_comp1.width());
        assert_eq!(tile_1_1_comp1.height(), tile_2_1_comp1.height());
    }
}
