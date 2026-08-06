//! Progression iterators, defined in Section B.12.
//!
//! A progression iterator essentially yields tuples of
//! (`layer_num`, resolution, component, precinct) in a specific order that
//! determines in which order the data appears in the codestream.

use alloc::vec::Vec;

use super::codestream::ComponentInfo;
use super::tile::{ComponentTile, ResolutionTile, Tile};
use crate::error::{DecodingError, Result};
use crate::{try_resize_decode_elements, ValidationError, DEFAULT_MAX_DECODE_BYTES};
use alloc::boxed::Box;
use core::cmp::Ordering;
use core::iter;
use core::mem::size_of;

const PACKETS_PER_INCLUSION_WORD: usize = u64::BITS as usize;

#[derive(Default, Copy, Clone, Debug, PartialEq, Hash, Eq)]
pub(crate) struct ProgressionData {
    pub(crate) layer_num: u8,
    pub(crate) resolution: u8,
    pub(crate) component: u16,
    pub(crate) precinct: u64,
}

pub(crate) struct IteratorInput<'a> {
    layers: (u8, u8),
    tile: &'a Tile<'a>,
    resolutions: (u8, u8),
    components: (u16, u16),
}

struct PacketInclusionMap {
    resolution_offsets: Vec<usize>,
    words: Vec<u64>,
    max_resolution: usize,
    layers: usize,
}

impl PacketInclusionMap {
    fn new(tile: &Tile<'_>) -> Result<Self> {
        let max_resolution = tile
            .component_infos
            .iter()
            .map(ComponentInfo::num_resolution_levels)
            .max()
            .map(usize::from)
            .ok_or(DecodingError::InvalidProgressionIterator)?;
        let layers = usize::from(tile.num_layers);
        let slot_count = tile
            .component_infos
            .len()
            .checked_mul(max_resolution)
            .ok_or(ValidationError::ImageTooLarge)?;
        let mut resolution_offsets = Vec::new();
        try_resize_decode_elements(&mut resolution_offsets, slot_count, usize::MAX)?;

        let mut packet_count = 0_usize;
        for (component_index, component) in tile.component_tiles().enumerate() {
            for resolution in component.resolution_tiles() {
                let slot = component_index
                    .checked_mul(max_resolution)
                    .and_then(|base| base.checked_add(usize::from(resolution.resolution)))
                    .ok_or(ValidationError::ImageTooLarge)?;
                resolution_offsets[slot] = packet_count;
                let precinct_count = usize::try_from(resolution.num_precincts())
                    .map_err(|_| ValidationError::ImageTooLarge)?;
                let resolution_packets = precinct_count
                    .checked_mul(layers)
                    .ok_or(ValidationError::ImageTooLarge)?;
                packet_count = packet_count
                    .checked_add(resolution_packets)
                    .ok_or(ValidationError::ImageTooLarge)?;
            }
        }

        let word_count = packet_count.div_ceil(PACKETS_PER_INCLUSION_WORD);
        let retained_bytes = resolution_offsets
            .capacity()
            .checked_mul(size_of::<usize>())
            .and_then(|bytes| {
                word_count
                    .checked_mul(size_of::<u64>())
                    .and_then(|word_bytes| bytes.checked_add(word_bytes))
            })
            .ok_or(ValidationError::ImageTooLarge)?;
        if retained_bytes > DEFAULT_MAX_DECODE_BYTES {
            return Err(ValidationError::ImageTooLarge.into());
        }
        let mut words = Vec::new();
        try_resize_decode_elements(&mut words, word_count, 0_u64)?;
        Ok(Self {
            resolution_offsets,
            words,
            max_resolution,
            layers,
        })
    }

    fn insert(&mut self, packet: ProgressionData) -> bool {
        let Some(slot) = usize::from(packet.component)
            .checked_mul(self.max_resolution)
            .and_then(|base| base.checked_add(usize::from(packet.resolution)))
        else {
            return false;
        };
        let Some(&resolution_offset) = self.resolution_offsets.get(slot) else {
            return false;
        };
        let Some(packet_offset) = usize::try_from(packet.precinct)
            .ok()
            .and_then(|precinct| precinct.checked_mul(self.layers))
            .and_then(|base| base.checked_add(usize::from(packet.layer_num)))
            .and_then(|offset| resolution_offset.checked_add(offset))
        else {
            return false;
        };
        let word_index = packet_offset / PACKETS_PER_INCLUSION_WORD;
        let mask = 1_u64 << (packet_offset % PACKETS_PER_INCLUSION_WORD);
        let Some(word) = self.words.get_mut(word_index) else {
            return false;
        };
        let is_new = *word & mask == 0;
        *word |= mask;
        is_new
    }
}

impl<'a> IteratorInput<'a> {
    pub(crate) fn new(tile: &'a Tile<'a>) -> Option<Self> {
        Self::try_new_with_custom_bounds(
            tile,
            // Will be clamped automatically.
            (0, u8::MAX),
            (0, u8::MAX),
            (0, u16::MAX),
        )
    }

    pub(crate) fn try_new_with_custom_bounds(
        tile: &'a Tile<'a>,
        mut resolutions: (u8, u8),
        mut layers: (u8, u8),
        mut components: (u16, u16),
    ) -> Option<Self> {
        let max_resolution = tile
            .component_infos
            .iter()
            .map(|c| c.coding_style.parameters.num_resolution_levels)
            .max()
            .unwrap_or(0);
        let max_layer = tile.num_layers;
        let max_component = u16::try_from(tile.component_infos.len()).ok()?;

        // Make sure we don't exceed what's actually possible
        resolutions.1 = resolutions.1.min(max_resolution);
        layers.1 = layers.1.min(max_layer);
        components.1 = components.1.min(max_component);

        if resolutions.1 <= resolutions.0 || layers.1 <= layers.0 || components.1 <= components.0 {
            return None;
        }

        Some(Self {
            layers,
            tile,
            resolutions,
            components,
        })
    }

    fn min_layer(&self) -> u8 {
        self.layers.0
    }

    fn max_layer(&self) -> u8 {
        self.layers.1
    }

    fn min_resolution(&self) -> u8 {
        self.resolutions.0
    }

    fn total_max_resolution(&self) -> u8 {
        self.resolutions.1
    }

    fn max_resolution(&self, component_idx: u16) -> u8 {
        self.total_max_resolution()
            // It's possible that the different component tiles have different resolution levels
            // (self.resolutions.1 stores the maximum across all component tiles), so
            // take the minimum of both.
            .min(self.tile.component_infos[component_idx as usize].num_resolution_levels())
    }

    fn min_comp(&self) -> u16 {
        self.components.0
    }

    fn max_comp(&self) -> u16 {
        self.components.1
    }

    fn component_tiles(&self) -> Vec<ComponentTile<'a>> {
        self.tile
            .component_infos
            .iter()
            .map(|c| ComponentTile::new(self.tile, c))
            .collect::<Vec<_>>()
    }
}

pub(crate) fn progression_iterator<'a>(
    tile: &'a Tile<'a>,
) -> Result<Box<dyn Iterator<Item = ProgressionData> + 'a>> {
    if tile.progression_changes.is_empty() {
        let iter_input =
            IteratorInput::new(tile).ok_or(DecodingError::InvalidProgressionIterator)?;
        return progression_iterator_for_order(tile.progression_order, iter_input);
    }

    let mut iterators = Vec::new();
    crate::try_reserve_decode_elements(&mut iterators, tile.progression_changes.len())?;
    for change in &tile.progression_changes {
        let iter_input = IteratorInput::try_new_with_custom_bounds(
            tile,
            (change.resolution_start, change.resolution_end),
            (0, change.layer_end),
            (change.component_start, change.component_end),
        )
        .ok_or(DecodingError::InvalidProgressionIterator)?;
        iterators.push(progression_iterator_for_order(
            change.progression_order,
            iter_input,
        )?);
    }

    let mut inclusion = PacketInclusionMap::new(tile)?;

    Ok(Box::new(
        iterators
            .into_iter()
            .flatten()
            .filter(move |packet| inclusion.insert(*packet)),
    ))
}

fn progression_iterator_for_order<'a>(
    progression_order: super::codestream::ProgressionOrder,
    iter_input: IteratorInput<'a>,
) -> Result<Box<dyn Iterator<Item = ProgressionData> + 'a>> {
    let iterator: Box<dyn Iterator<Item = ProgressionData>> = match progression_order {
        super::codestream::ProgressionOrder::LayerResolutionComponentPosition => {
            Box::new(layer_resolution_component_position_progression(iter_input))
        }
        super::codestream::ProgressionOrder::ResolutionLayerComponentPosition => {
            Box::new(resolution_layer_component_position_progression(iter_input))
        }
        super::codestream::ProgressionOrder::ResolutionPositionComponentLayer => Box::new(
            resolution_position_component_layer_progression(iter_input)
                .ok_or(DecodingError::InvalidProgressionIterator)?,
        ),
        super::codestream::ProgressionOrder::PositionComponentResolutionLayer => Box::new(
            position_component_resolution_layer_progression(iter_input)
                .ok_or(DecodingError::InvalidProgressionIterator)?,
        ),
        super::codestream::ProgressionOrder::ComponentPositionResolutionLayer => Box::new(
            component_position_resolution_layer_progression(iter_input)
                .ok_or(DecodingError::InvalidProgressionIterator)?,
        ),
    };
    Ok(iterator)
}

/// B.12.1.1 Layer-resolution level-component-position progression.
pub(crate) fn layer_resolution_component_position_progression(
    input: IteratorInput<'_>,
) -> impl Iterator<Item = ProgressionData> + '_ {
    let component_tiles = input.component_tiles();

    let mut layer = input.min_layer();
    let mut resolution = input.min_resolution();
    let mut component_idx = input.min_comp();

    let mut resolution_tile = ResolutionTile::new(component_tiles[0], resolution);
    let mut precinct = 0;

    iter::from_fn(move || {
        if layer == input.max_layer() || resolution == input.total_max_resolution() {
            return None;
        }

        if precinct == resolution_tile.num_precincts() {
            loop {
                precinct = 0;
                component_idx += 1;

                if component_idx == input.max_comp() {
                    component_idx = input.min_comp();

                    resolution += 1;

                    if resolution == input.max_resolution(component_idx) {
                        resolution = input.min_resolution();
                        layer += 1;

                        if layer == input.max_layer() {
                            return None;
                        }
                    }
                }

                resolution_tile =
                    ResolutionTile::new(component_tiles[component_idx as usize], resolution);

                // Only yield if the resolution tile has precincts, otherwise
                // we need to keep advancing.
                if resolution_tile.num_precincts() != 0 {
                    break;
                }
            }
        }

        let data = ProgressionData {
            layer_num: layer,
            resolution,
            component: component_idx,
            precinct,
        };

        precinct += 1;

        Some(data)
    })
}

/// B.12.1.2 Resolution level-layer-component-position progression.
pub(crate) fn resolution_layer_component_position_progression(
    input: IteratorInput<'_>,
) -> impl Iterator<Item = ProgressionData> + '_ {
    let component_tiles = input.component_tiles();

    let mut layer = input.min_layer();
    let mut resolution = input.min_resolution();
    let mut component_idx = input.min_comp();
    let mut resolution_tile =
        ResolutionTile::new(component_tiles[component_idx as usize], resolution);
    let mut precinct = 0;

    iter::from_fn(move || {
        if layer == input.max_layer() || resolution == input.total_max_resolution() {
            return None;
        }

        if precinct == resolution_tile.num_precincts() {
            loop {
                precinct = 0;
                component_idx += 1;

                if component_idx == input.max_comp() {
                    component_idx = input.min_comp();
                    layer += 1;

                    if layer == input.max_layer() {
                        layer = input.min_layer();
                        resolution += 1;

                        if resolution == input.total_max_resolution() {
                            return None;
                        }
                    }
                }

                // If the given resolution level doesn't exist for the current
                // component, continue.
                if resolution >= input.max_resolution(component_idx) {
                    continue;
                }

                resolution_tile =
                    ResolutionTile::new(component_tiles[component_idx as usize], resolution);

                // Only yield if the resolution tile has precincts, otherwise
                // we need to keep advancing.
                if resolution_tile.num_precincts() != 0 {
                    break;
                }
            }
        }

        let data = ProgressionData {
            layer_num: layer,
            resolution,
            component: component_idx,
            precinct,
        };

        precinct += 1;

        Some(data)
    })
}

// The formula for the remaining three progressions looks very intimidating.
// But really, all they boil down to is that we need to determine all precinct
// indices for each component/resolution combination and sort them by ascending
// y/x coordinate on the reference grid. Other than that, they can be treated
// exactly the same, except that the sort order precedence of the fields change.

// Note that the order of fields here is important!
struct PrecinctStore {
    resolution: u8,
    precinct_y: u32,
    precinct_x: u32,
    component_idx: u16,
    precinct_idx: u64,
}

fn position_progression_common(
    input: IteratorInput<'_>,
    sort: impl FnMut(&PrecinctStore, &PrecinctStore) -> Ordering,
) -> Option<impl Iterator<Item = ProgressionData> + '_> {
    let mut elements = Vec::new();

    for (component_idx, component) in input
        .tile
        .component_tiles()
        .enumerate()
        .skip(input.min_comp() as usize)
        .take(input.max_comp() as usize - input.min_comp() as usize)
    {
        let component_idx = u16::try_from(component_idx).ok()?;
        for (resolution, resolution_tile) in component
            .resolution_tiles()
            .enumerate()
            .skip(input.min_resolution() as usize)
            .take(input.total_max_resolution() as usize - input.min_resolution() as usize)
        {
            let resolution = u8::try_from(resolution).ok()?;
            elements.extend(resolution_tile.precincts()?.map(|d| PrecinctStore {
                precinct_y: d.r_y,
                precinct_x: d.r_x,
                component_idx,
                resolution,
                precinct_idx: d.idx,
            }));
        }
    }

    elements.sort_by(sort);

    Some(elements.into_iter().flat_map(move |e| {
        (input.min_layer()..input.max_layer()).map(move |layer| ProgressionData {
            layer_num: layer,
            resolution: e.resolution,
            component: e.component_idx,
            precinct: e.precinct_idx,
        })
    }))
}

/// B.12.1.3 Resolution level-position-component-layer progression.
pub(crate) fn resolution_position_component_layer_progression(
    input: IteratorInput<'_>,
) -> Option<impl Iterator<Item = ProgressionData> + '_> {
    position_progression_common(input, |p, s| {
        p.resolution
            .cmp(&s.resolution)
            .then_with(|| p.precinct_y.cmp(&s.precinct_y))
            .then_with(|| p.precinct_x.cmp(&s.precinct_x))
            .then_with(|| p.component_idx.cmp(&s.component_idx))
            .then_with(|| p.precinct_idx.cmp(&s.precinct_idx))
    })
}

/// B.12.1.4 Position-component-resolution level-layer progression.
pub(crate) fn position_component_resolution_layer_progression(
    input: IteratorInput<'_>,
) -> Option<impl Iterator<Item = ProgressionData> + '_> {
    position_progression_common(input, |p, s| {
        p.precinct_y
            .cmp(&s.precinct_y)
            .then_with(|| p.precinct_x.cmp(&s.precinct_x))
            .then_with(|| p.component_idx.cmp(&s.component_idx))
            .then_with(|| p.resolution.cmp(&s.resolution))
            .then_with(|| p.precinct_idx.cmp(&s.precinct_idx))
    })
}

/// B.12.1.5 Component-position-resolution level-layer progression.
pub(crate) fn component_position_resolution_layer_progression(
    input: IteratorInput<'_>,
) -> Option<impl Iterator<Item = ProgressionData> + '_> {
    position_progression_common(input, |p, s| {
        p.component_idx
            .cmp(&s.component_idx)
            .then_with(|| p.precinct_y.cmp(&s.precinct_y))
            .then_with(|| p.precinct_x.cmp(&s.precinct_x))
            .then_with(|| p.resolution.cmp(&s.resolution))
            .then_with(|| p.precinct_idx.cmp(&s.precinct_idx))
    })
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::j2c::codestream::{
        CodeBlockStyle, CodingStyleComponent, CodingStyleFlags, CodingStyleParameters,
        ComponentInfo, ComponentSizeInfo, ProgressionChange, ProgressionOrder, QuantizationInfo,
        QuantizationStyle, WaveletTransform,
    };
    use crate::j2c::rect::IntRect;

    #[test]
    fn overlapping_progression_changes_emit_each_packet_once() {
        let tile = Tile {
            idx: 0,
            tile_parts: vec![],
            component_infos: vec![ComponentInfo {
                size_info: ComponentSizeInfo {
                    precision: 8,
                    signed: false,
                    horizontal_resolution: 1,
                    vertical_resolution: 1,
                },
                coding_style: CodingStyleComponent {
                    flags: CodingStyleFlags::default(),
                    parameters: CodingStyleParameters {
                        num_decomposition_levels: 1,
                        num_resolution_levels: 2,
                        code_block_width: 6,
                        code_block_height: 6,
                        code_block_style: CodeBlockStyle::default(),
                        transformation: WaveletTransform::Reversible53,
                        precinct_exponents: vec![(15, 15), (15, 15)],
                    },
                },
                quantization_info: QuantizationInfo {
                    quantization_style: QuantizationStyle::NoQuantization,
                    guard_bits: 2,
                    step_sizes: vec![],
                },
                roi_shift: 0,
            }],
            rect: IntRect::from_ltrb(0, 0, 8, 8),
            progression_order: ProgressionOrder::LayerResolutionComponentPosition,
            progression_changes: vec![
                ProgressionChange {
                    resolution_start: 0,
                    component_start: 0,
                    layer_end: 2,
                    resolution_end: 1,
                    component_end: 1,
                    progression_order: ProgressionOrder::LayerResolutionComponentPosition,
                },
                ProgressionChange {
                    resolution_start: 0,
                    component_start: 0,
                    layer_end: 2,
                    resolution_end: 2,
                    component_end: 1,
                    progression_order: ProgressionOrder::LayerResolutionComponentPosition,
                },
            ],
            num_layers: 2,
            mct: false,
        };

        let packets = progression_iterator(&tile)
            .expect("valid progression")
            .collect::<Vec<_>>();

        assert_eq!(
            packets,
            [
                ProgressionData {
                    layer_num: 0,
                    resolution: 0,
                    component: 0,
                    precinct: 0,
                },
                ProgressionData {
                    layer_num: 1,
                    resolution: 0,
                    component: 0,
                    precinct: 0,
                },
                ProgressionData {
                    layer_num: 0,
                    resolution: 1,
                    component: 0,
                    precinct: 0,
                },
                ProgressionData {
                    layer_num: 1,
                    resolution: 1,
                    component: 0,
                    precinct: 0,
                },
            ]
        );
    }

    #[test]
    fn empty_component_set_is_an_invalid_progression_iterator() {
        let tile = Tile {
            idx: 0,
            tile_parts: vec![],
            component_infos: vec![],
            rect: IntRect::from_ltrb(0, 0, 1, 1),
            progression_order: ProgressionOrder::LayerResolutionComponentPosition,
            progression_changes: vec![],
            num_layers: 1,
            mct: false,
        };

        assert!(matches!(
            progression_iterator(&tile),
            Err(crate::error::DecodeError::Decoding(
                DecodingError::InvalidProgressionIterator
            ))
        ));
    }
}
