//! Creating tiles and parsing their constituent tile parts.

use alloc::vec::Vec;

use super::build::{PrecinctData, SubBandType};
use super::codestream::{markers, ComponentInfo, Header, ProgressionChange, ProgressionOrder};
use super::rect::IntRect;
use crate::error::{bail, MarkerError, Result, ValidationError};
use crate::reader::BitReader;

mod cursor;
mod metadata;
mod parsed;
mod tile_part;

pub(crate) use cursor::TilePartCursor;
use metadata::{inherit_tile_metadata, TileMetadataBudget};
pub(crate) use parsed::ParsedTiles;
use tile_part::parse_tile_part;

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
#[derive(Debug)]
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
#[derive(Debug)]
pub(crate) struct MergedTilePart<'a> {
    pub(crate) data: BitReader<'a>,
    packet_lengths: PacketLengthMetadata,
}

/// A tile part where packet headers and packet data are separated.
#[derive(Debug)]
pub(crate) struct SeparatedTilePart<'a> {
    pub(crate) headers: Vec<BitReader<'a>>,
    pub(crate) body: BitReader<'a>,
    packet_lengths: PacketLengthMetadata,
}

#[derive(Debug)]
pub(crate) enum TilePart<'a> {
    Merged(MergedTilePart<'a>),
    Separated(SeparatedTilePart<'a>),
}

#[derive(Debug, Default)]
struct PacketLengthMetadata {
    present: bool,
    lengths: Vec<u32>,
}

impl PacketLengthMetadata {
    fn new(present: bool, lengths: Vec<u32>) -> Self {
        Self { present, lengths }
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
            tile_parts: Vec::new(),
            rect,
            // By default, each tile inherits the settings from the main
            // header. When parsing the tile parts, some of these settings
            // might be overridden.
            component_infos: Vec::new(),
            progression_order: header.global_coding_style.progression_order,
            progression_changes: Vec::new(),
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
    retained_image_bytes: usize,
) -> Result<ParsedTiles<'a>> {
    let mut metadata_budget = TileMetadataBudget::for_image(main_header, retained_image_bytes)?;
    let num_tiles = usize::try_from(main_header.size_data.num_tiles())
        .map_err(|_| ValidationError::ImageTooLarge)?;

    let mut tiles = Vec::new();
    metadata_budget.try_reserve_retained(&mut tiles, num_tiles)?;
    for idx in 0..main_header.size_data.num_tiles() {
        tiles.push(Tile::new(idx, main_header));
        let tile = tiles.last_mut().ok_or(ValidationError::ImageTooLarge)?;
        inherit_tile_metadata(tile, main_header, &mut metadata_budget)?;
    }

    let mut ppm_packet_idx = 0;

    parse_tile_part(
        reader,
        main_header,
        &mut tiles,
        &mut ppm_packet_idx,
        &mut metadata_budget,
    )?;

    while reader.peek_marker() == Some(markers::SOT) {
        parse_tile_part(
            reader,
            main_header,
            &mut tiles,
            &mut ppm_packet_idx,
            &mut metadata_budget,
        )?;
    }

    if main_header.strict && reader.read_marker()? != markers::EOC {
        bail!(MarkerError::Expected("EOC"));
    }

    metadata_budget.validate_owner_graph(&tiles)?;
    Ok(ParsedTiles::new(tiles, metadata_budget.retained_bytes()))
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

#[cfg(test)]
mod tests {
    use alloc::vec;

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

        let dummy_component_coding_style = || CodingStyleComponent {
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

        let dummy_quantization_info = || QuantizationInfo {
            quantization_style: QuantizationStyle::NoQuantization,
            guard_bits: 0,
            step_sizes: vec![],
        };

        let component_info_0 = ComponentInfo {
            size_info: component_size_info_0,
            coding_style: dummy_component_coding_style(),
            quantization_info: dummy_quantization_info(),
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
            coding_style: dummy_component_coding_style(),
            quantization_info: dummy_quantization_info(),
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
