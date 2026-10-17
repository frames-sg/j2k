//! Progression iterators, defined in Section B.12.
//!
//! A progression iterator essentially yields tuples of
//! (`layer_num`, resolution, component, precinct) in a specific order that
//! determines in which order the data appears in the codestream.

use alloc::vec::Vec;

use super::codestream::ComponentInfo;
use super::tile::{ComponentTile, ResolutionTile, Tile};
use crate::error::{DecodingError, Result};
use crate::j2c::decode::DecodeAllocationBudget;
use crate::ValidationError;
use core::cmp::Ordering;

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
    fn new(tile: &Tile<'_>, budget: &mut DecodeAllocationBudget) -> Result<Self> {
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
        budget.resize_new(&mut resolution_offsets, slot_count, usize::MAX)?;

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
        let mut words = Vec::new();
        budget.resize_new(&mut words, word_count, 0_u64)?;
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
}

pub(crate) struct ProgressionIterator<'a> {
    inner: ProgressionInner<'a>,
    owned_bytes: usize,
}

enum ProgressionInner<'a> {
    Single(ProgressionOrderIterator<'a>),
    Changes {
        iterators: Vec<ProgressionOrderIterator<'a>>,
        current: usize,
        inclusion: PacketInclusionMap,
    },
}

impl ProgressionIterator<'_> {
    pub(crate) const fn owned_bytes(&self) -> usize {
        self.owned_bytes
    }
}

impl Iterator for ProgressionIterator<'_> {
    type Item = ProgressionData;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            ProgressionInner::Single(iterator) => iterator.next(),
            ProgressionInner::Changes {
                iterators,
                current,
                inclusion,
            } => loop {
                let iterator = iterators.get_mut(*current)?;
                match iterator.next() {
                    Some(packet) if inclusion.insert(packet) => return Some(packet),
                    Some(_) => {}
                    None => *current += 1,
                }
            },
        }
    }
}

pub(crate) fn progression_iterator<'a>(
    tile: &'a Tile<'a>,
    retained_live_bytes: usize,
) -> Result<ProgressionIterator<'a>> {
    let mut budget = DecodeAllocationBudget::from_live_bytes(retained_live_bytes)?;
    if tile.progression_changes.is_empty() {
        let iter_input =
            IteratorInput::new(tile).ok_or(DecodingError::InvalidProgressionIterator)?;
        let iterator =
            progression_iterator_for_order(tile.progression_order, iter_input, &mut budget)?;
        return Ok(ProgressionIterator {
            inner: ProgressionInner::Single(iterator),
            owned_bytes: budget
                .live_bytes()
                .checked_sub(retained_live_bytes)
                .ok_or(ValidationError::ImageTooLarge)?,
        });
    }

    let mut iterators = Vec::new();
    budget.reserve_new(&mut iterators, tile.progression_changes.len())?;
    for change in &tile.progression_changes {
        let iter_input = IteratorInput::try_new_with_custom_bounds(
            tile,
            (change.resolution_start, change.resolution_end),
            (0, change.layer_end),
            (change.component_start, change.component_end),
        )
        .ok_or(DecodingError::InvalidProgressionIterator)?;
        if iterators.len() == iterators.capacity() {
            return Err(DecodingError::HostAllocationFailed.into());
        }
        iterators.push(progression_iterator_for_order(
            change.progression_order,
            iter_input,
            &mut budget,
        )?);
    }

    let inclusion = PacketInclusionMap::new(tile, &mut budget)?;
    Ok(ProgressionIterator {
        inner: ProgressionInner::Changes {
            iterators,
            current: 0,
            inclusion,
        },
        owned_bytes: budget
            .live_bytes()
            .checked_sub(retained_live_bytes)
            .ok_or(ValidationError::ImageTooLarge)?,
    })
}

fn progression_iterator_for_order<'a>(
    progression_order: super::codestream::ProgressionOrder,
    iter_input: IteratorInput<'a>,
    budget: &mut DecodeAllocationBudget,
) -> Result<ProgressionOrderIterator<'a>> {
    let iterator = match progression_order {
        super::codestream::ProgressionOrder::LayerResolutionComponentPosition => {
            ProgressionOrderIterator::Lrcp(layer_resolution_component_position_progression(
                iter_input,
            ))
        }
        super::codestream::ProgressionOrder::ResolutionLayerComponentPosition => {
            ProgressionOrderIterator::Rlcp(resolution_layer_component_position_progression(
                iter_input,
            ))
        }
        super::codestream::ProgressionOrder::ResolutionPositionComponentLayer => {
            ProgressionOrderIterator::Position(resolution_position_component_layer_progression(
                &iter_input,
                budget,
            )?)
        }
        super::codestream::ProgressionOrder::PositionComponentResolutionLayer => {
            ProgressionOrderIterator::Position(position_component_resolution_layer_progression(
                &iter_input,
                budget,
            )?)
        }
        super::codestream::ProgressionOrder::ComponentPositionResolutionLayer => {
            ProgressionOrderIterator::Position(component_position_resolution_layer_progression(
                &iter_input,
                budget,
            )?)
        }
    };
    Ok(iterator)
}

/// B.12.1.1 Layer-resolution level-component-position progression.
fn layer_resolution_component_position_progression(
    input: IteratorInput<'_>,
) -> LayerResolutionComponentPosition<'_> {
    LayerResolutionComponentPosition::new(input)
}

struct LayerResolutionComponentPosition<'a> {
    input: IteratorInput<'a>,
    layer: u8,
    resolution: u8,
    component_idx: u16,
    resolution_tile: Option<ResolutionTile<'a>>,
    precinct: u64,
}

impl<'a> LayerResolutionComponentPosition<'a> {
    fn new(input: IteratorInput<'a>) -> Self {
        let layer = input.min_layer();
        let resolution = input.min_resolution();
        let component_idx = input.min_comp();
        let resolution_tile = first_resolution_tile(&input);
        Self {
            input,
            layer,
            resolution,
            component_idx,
            resolution_tile,
            precinct: 0,
        }
    }
}

impl Iterator for LayerResolutionComponentPosition<'_> {
    type Item = ProgressionData;

    fn next(&mut self) -> Option<Self::Item> {
        if self.layer == self.input.max_layer()
            || self.resolution == self.input.total_max_resolution()
        {
            return None;
        }

        if self
            .resolution_tile
            .as_ref()
            .is_none_or(|tile| self.precinct == tile.num_precincts())
        {
            loop {
                self.precinct = 0;
                self.component_idx += 1;

                if self.component_idx == self.input.max_comp() {
                    self.component_idx = self.input.min_comp();

                    self.resolution += 1;

                    if self.resolution == self.input.total_max_resolution() {
                        self.resolution = self.input.min_resolution();
                        self.layer += 1;

                        if self.layer == self.input.max_layer() {
                            return None;
                        }
                    }
                }

                // If the given resolution level doesn't exist for the current
                // component, continue.
                if self.resolution >= self.input.max_resolution(self.component_idx) {
                    continue;
                }

                let tile = ResolutionTile::new(
                    ComponentTile::new(
                        self.input.tile,
                        &self.input.tile.component_infos[self.component_idx as usize],
                    ),
                    self.resolution,
                );
                let has_precincts = tile.num_precincts() != 0;
                self.resolution_tile = Some(tile);

                // Only yield if the resolution tile has precincts, otherwise
                // we need to keep advancing.
                if has_precincts {
                    break;
                }
            }
        }

        let data = ProgressionData {
            layer_num: self.layer,
            resolution: self.resolution,
            component: self.component_idx,
            precinct: self.precinct,
        };

        self.precinct += 1;

        Some(data)
    }
}

/// B.12.1.2 Resolution level-layer-component-position progression.
fn resolution_layer_component_position_progression(
    input: IteratorInput<'_>,
) -> ResolutionLayerComponentPosition<'_> {
    ResolutionLayerComponentPosition::new(input)
}

struct ResolutionLayerComponentPosition<'a> {
    input: IteratorInput<'a>,
    layer: u8,
    resolution: u8,
    component_idx: u16,
    resolution_tile: Option<ResolutionTile<'a>>,
    precinct: u64,
}

impl<'a> ResolutionLayerComponentPosition<'a> {
    fn new(input: IteratorInput<'a>) -> Self {
        let layer = input.min_layer();
        let resolution = input.min_resolution();
        let component_idx = input.min_comp();
        let resolution_tile = first_resolution_tile(&input);
        Self {
            input,
            layer,
            resolution,
            component_idx,
            resolution_tile,
            precinct: 0,
        }
    }
}

impl Iterator for ResolutionLayerComponentPosition<'_> {
    type Item = ProgressionData;

    fn next(&mut self) -> Option<Self::Item> {
        if self.layer == self.input.max_layer()
            || self.resolution == self.input.total_max_resolution()
        {
            return None;
        }

        if self
            .resolution_tile
            .as_ref()
            .is_none_or(|tile| self.precinct == tile.num_precincts())
        {
            loop {
                self.precinct = 0;
                self.component_idx += 1;

                if self.component_idx == self.input.max_comp() {
                    self.component_idx = self.input.min_comp();
                    self.layer += 1;

                    if self.layer == self.input.max_layer() {
                        self.layer = self.input.min_layer();
                        self.resolution += 1;

                        if self.resolution == self.input.total_max_resolution() {
                            return None;
                        }
                    }
                }

                // If the given resolution level doesn't exist for the current
                // component, continue.
                if self.resolution >= self.input.max_resolution(self.component_idx) {
                    continue;
                }

                let tile = ResolutionTile::new(
                    ComponentTile::new(
                        self.input.tile,
                        &self.input.tile.component_infos[self.component_idx as usize],
                    ),
                    self.resolution,
                );
                let has_precincts = tile.num_precincts() != 0;
                self.resolution_tile = Some(tile);

                // Only yield if the resolution tile has precincts, otherwise
                // we need to keep advancing.
                if has_precincts {
                    break;
                }
            }
        }

        let data = ProgressionData {
            layer_num: self.layer,
            resolution: self.resolution,
            component: self.component_idx,
            precinct: self.precinct,
        };

        self.precinct += 1;

        Some(data)
    }
}

enum ProgressionOrderIterator<'a> {
    Lrcp(LayerResolutionComponentPosition<'a>),
    Rlcp(ResolutionLayerComponentPosition<'a>),
    Position(PositionProgression),
}

impl Iterator for ProgressionOrderIterator<'_> {
    type Item = ProgressionData;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Lrcp(iterator) => iterator.next(),
            Self::Rlcp(iterator) => iterator.next(),
            Self::Position(iterator) => iterator.next(),
        }
    }
}

/// The tile a layer-major progression starts from, or `None` when its first
/// component lacks the first resolution level, so the iterator advances to the
/// next component that has it.
fn first_resolution_tile<'a>(input: &IteratorInput<'a>) -> Option<ResolutionTile<'a>> {
    let component_idx = input.min_comp();
    if input.min_resolution() >= input.max_resolution(component_idx) {
        return None;
    }
    Some(ResolutionTile::new(
        ComponentTile::new(
            input.tile,
            &input.tile.component_infos[usize::from(component_idx)],
        ),
        input.min_resolution(),
    ))
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

struct PositionProgression {
    elements: alloc::vec::IntoIter<PrecinctStore>,
    current: Option<PrecinctStore>,
    min_layer: u8,
    max_layer: u8,
    next_layer: u8,
}

impl Iterator for PositionProgression {
    type Item = ProgressionData;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(current) = self.current.as_ref() {
                if self.next_layer < self.max_layer {
                    let packet = ProgressionData {
                        layer_num: self.next_layer,
                        resolution: current.resolution,
                        component: current.component_idx,
                        precinct: current.precinct_idx,
                    };
                    self.next_layer += 1;
                    return Some(packet);
                }
            }
            self.current = self.elements.next();
            self.next_layer = self.min_layer;
            self.current.as_ref()?;
        }
    }
}

fn position_progression_common(
    input: &IteratorInput<'_>,
    budget: &mut DecodeAllocationBudget,
    sort: impl FnMut(&PrecinctStore, &PrecinctStore) -> Ordering,
) -> Result<PositionProgression> {
    let mut element_count = 0usize;
    for component in input
        .tile
        .component_tiles()
        .skip(input.min_comp() as usize)
        .take(input.max_comp() as usize - input.min_comp() as usize)
    {
        for resolution_tile in component
            .resolution_tiles()
            .skip(input.min_resolution() as usize)
            .take(input.total_max_resolution() as usize - input.min_resolution() as usize)
        {
            let precincts = usize::try_from(resolution_tile.num_precincts())
                .map_err(|_| ValidationError::ImageTooLarge)?;
            element_count = element_count
                .checked_add(precincts)
                .ok_or(ValidationError::ImageTooLarge)?;
        }
    }
    let mut elements = Vec::new();
    budget.reserve_new(&mut elements, element_count)?;

    for (component_idx, component) in input
        .tile
        .component_tiles()
        .enumerate()
        .skip(input.min_comp() as usize)
        .take(input.max_comp() as usize - input.min_comp() as usize)
    {
        let component_idx =
            u16::try_from(component_idx).map_err(|_| ValidationError::ImageTooLarge)?;
        for (resolution, resolution_tile) in component
            .resolution_tiles()
            .enumerate()
            .skip(input.min_resolution() as usize)
            .take(input.total_max_resolution() as usize - input.min_resolution() as usize)
        {
            let resolution =
                u8::try_from(resolution).map_err(|_| ValidationError::ImageTooLarge)?;
            for precinct in resolution_tile
                .precincts()
                .ok_or(DecodingError::InvalidProgressionIterator)?
            {
                if elements.len() == elements.capacity() {
                    return Err(DecodingError::HostAllocationFailed.into());
                }
                elements.push(PrecinctStore {
                    precinct_y: precinct.r_y,
                    precinct_x: precinct.r_x,
                    component_idx,
                    resolution,
                    precinct_idx: precinct.idx,
                });
            }
        }
    }

    if elements.len() != element_count {
        return Err(DecodingError::InvalidProgressionIterator.into());
    }

    elements.sort_unstable_by(sort);

    Ok(PositionProgression {
        elements: elements.into_iter(),
        current: None,
        min_layer: input.min_layer(),
        max_layer: input.max_layer(),
        next_layer: input.min_layer(),
    })
}

/// B.12.1.3 Resolution level-position-component-layer progression.
fn resolution_position_component_layer_progression(
    input: &IteratorInput<'_>,
    budget: &mut DecodeAllocationBudget,
) -> Result<PositionProgression> {
    position_progression_common(input, budget, |p, s| {
        p.resolution
            .cmp(&s.resolution)
            .then_with(|| p.precinct_y.cmp(&s.precinct_y))
            .then_with(|| p.precinct_x.cmp(&s.precinct_x))
            .then_with(|| p.component_idx.cmp(&s.component_idx))
            .then_with(|| p.precinct_idx.cmp(&s.precinct_idx))
    })
}

/// B.12.1.4 Position-component-resolution level-layer progression.
fn position_component_resolution_layer_progression(
    input: &IteratorInput<'_>,
    budget: &mut DecodeAllocationBudget,
) -> Result<PositionProgression> {
    position_progression_common(input, budget, |p, s| {
        p.precinct_y
            .cmp(&s.precinct_y)
            .then_with(|| p.precinct_x.cmp(&s.precinct_x))
            .then_with(|| p.component_idx.cmp(&s.component_idx))
            .then_with(|| p.resolution.cmp(&s.resolution))
            .then_with(|| p.precinct_idx.cmp(&s.precinct_idx))
    })
}

/// B.12.1.5 Component-position-resolution level-layer progression.
fn component_position_resolution_layer_progression(
    input: &IteratorInput<'_>,
    budget: &mut DecodeAllocationBudget,
) -> Result<PositionProgression> {
    position_progression_common(input, budget, |p, s| {
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

    fn progression_tile(
        progression_order: ProgressionOrder,
        progression_changes: Vec<ProgressionChange>,
    ) -> Tile<'static> {
        Tile {
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
            progression_order,
            progression_changes,
            num_layers: 2,
            mct: false,
        }
    }

    #[test]
    fn overlapping_progression_changes_emit_each_packet_once() {
        let tile = progression_tile(
            ProgressionOrder::LayerResolutionComponentPosition,
            vec![
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
        );

        let packets = progression_iterator(&tile, 0)
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
    fn position_major_workspace_is_rejected_against_the_live_decode_baseline() {
        let tile = progression_tile(ProgressionOrder::ResolutionPositionComponentLayer, vec![]);

        let Err(error) = progression_iterator(&tile, crate::DEFAULT_MAX_DECODE_BYTES) else {
            panic!("position-major workspace has no headroom");
        };

        assert_eq!(
            error,
            crate::DecodeError::Validation(ValidationError::ImageTooLarge)
        );
    }

    #[test]
    fn poc_iterator_owners_are_rejected_against_the_live_decode_baseline() {
        let tile = progression_tile(
            ProgressionOrder::LayerResolutionComponentPosition,
            vec![ProgressionChange {
                resolution_start: 0,
                component_start: 0,
                layer_end: 2,
                resolution_end: 2,
                component_end: 1,
                progression_order: ProgressionOrder::LayerResolutionComponentPosition,
            }],
        );

        let Err(error) = progression_iterator(&tile, crate::DEFAULT_MAX_DECODE_BYTES) else {
            panic!("POC iterator ownership has no headroom");
        };

        assert_eq!(
            error,
            crate::DecodeError::Validation(ValidationError::ImageTooLarge)
        );
    }

    #[test]
    fn position_major_poc_uses_the_budgeted_inline_iterator() {
        let tile = progression_tile(
            ProgressionOrder::LayerResolutionComponentPosition,
            vec![ProgressionChange {
                resolution_start: 0,
                component_start: 0,
                layer_end: 2,
                resolution_end: 2,
                component_end: 1,
                progression_order: ProgressionOrder::ResolutionPositionComponentLayer,
            }],
        );

        let iterator = progression_iterator(&tile, 0).expect("position-major POC");
        assert!(iterator.owned_bytes() > 0, "POC workspace is charged");
        assert_eq!(iterator.collect::<Vec<_>>().len(), 4);
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
            progression_iterator(&tile, 0),
            Err(crate::error::DecodeError::Decoding(
                DecodingError::InvalidProgressionIterator
            ))
        ));
    }

    /// A component with its own number of decomposition levels, as a COC
    /// marker gives it, and one precinct per resolution level.
    fn component_with_levels(levels: u8) -> ComponentInfo {
        ComponentInfo {
            size_info: ComponentSizeInfo {
                precision: 8,
                signed: false,
                horizontal_resolution: 1,
                vertical_resolution: 1,
            },
            coding_style: CodingStyleComponent {
                flags: CodingStyleFlags::default(),
                parameters: CodingStyleParameters {
                    num_decomposition_levels: levels,
                    num_resolution_levels: levels + 1,
                    code_block_width: 6,
                    code_block_height: 6,
                    code_block_style: CodeBlockStyle::default(),
                    transformation: WaveletTransform::Reversible53,
                    precinct_exponents: vec![(15, 15); usize::from(levels) + 1],
                },
            },
            quantization_info: QuantizationInfo {
                quantization_style: QuantizationStyle::NoQuantization,
                guard_bits: 2,
                step_sizes: vec![],
            },
            roi_shift: 0,
        }
    }

    /// The (resolution, component) of each packet, for a one-layer tile.
    fn packet_order(
        levels: &[u8],
        order: ProgressionOrder,
        progression_changes: Vec<ProgressionChange>,
    ) -> Vec<(u8, u16)> {
        let tile = Tile {
            idx: 0,
            tile_parts: vec![],
            component_infos: levels.iter().map(|&l| component_with_levels(l)).collect(),
            rect: IntRect::from_ltrb(0, 0, 16, 16),
            progression_order: order,
            progression_changes,
            num_layers: 1,
            mct: false,
        };
        progression_iterator(&tile, 0)
            .expect("valid progression")
            .map(|packet| {
                assert_eq!((packet.layer_num, packet.precinct), (0, 0));
                (packet.resolution, packet.component)
            })
            .collect()
    }

    #[test]
    fn lrcp_skips_resolution_levels_a_component_does_not_have() {
        // B.12.1.1: a component without resolution level r has no packets
        // there. Component 1 has 2 resolution levels, component 0 has 4.
        assert_eq!(
            packet_order(
                &[3, 1],
                ProgressionOrder::LayerResolutionComponentPosition,
                vec![]
            ),
            [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (3, 0)]
        );
        // With the levels swapped, the iterator must not stop after
        // component 0's last resolution level.
        assert_eq!(
            packet_order(
                &[1, 3],
                ProgressionOrder::LayerResolutionComponentPosition,
                vec![]
            ),
            [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (3, 1)]
        );
    }

    #[test]
    fn layer_major_progression_changes_respect_component_and_resolution_starts() {
        // Begin at the selected component, or skip it when it lacks the
        // starting resolution level. An excluded component must not affect this.
        for order in [
            ProgressionOrder::LayerResolutionComponentPosition,
            ProgressionOrder::ResolutionLayerComponentPosition,
        ] {
            for (levels, component_start, expected_component) in [
                (&[1, 3][..], 0, 1),
                (&[1, 3, 1][..], 1, 1),
                (&[3, 1, 3][..], 1, 2),
            ] {
                let change = ProgressionChange {
                    resolution_start: 2,
                    component_start,
                    layer_end: 1,
                    resolution_end: 4,
                    component_end: component_start + 2,
                    progression_order: order,
                };
                assert_eq!(
                    packet_order(levels, order, vec![change]),
                    [(2, expected_component), (3, expected_component)],
                    "{order:?}, component_start={component_start}"
                );
            }
        }
    }
}
