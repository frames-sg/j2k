use alloc::vec;
use alloc::vec::Vec;

use super::build::Decomposition;
use super::codestream::{Header, WaveletTransform};
use super::decode::{DecompositionStorage, OutputRegion};
use super::rect::IntRect;
use super::tile::{ComponentTile, ResolutionTile, Tile};

#[derive(Clone, Debug)]
pub(crate) struct RoiPlan {
    sub_band_windows: Vec<Option<IntRect>>,
    idwt_windows: Vec<Option<IntRect>>,
    final_windows: Vec<Option<IntRect>>,
}

impl RoiPlan {
    pub(crate) fn build(
        tile: &Tile<'_>,
        header: &Header<'_>,
        storage: &DecompositionStorage<'_>,
        output_region: OutputRegion,
    ) -> Option<Self> {
        if tile.component_infos.iter().any(|component_info| {
            component_info.size_info.horizontal_resolution != 1
                || component_info.size_info.vertical_resolution != 1
        }) {
            return None;
        }

        let mut plan = Self {
            sub_band_windows: vec![None; storage.sub_bands.len()],
            idwt_windows: vec![None; storage.decompositions.len()],
            final_windows: vec![None; tile.component_infos.len()],
        };

        for (component_idx, component_info) in tile.component_infos.iter().enumerate() {
            let component_tile = ComponentTile::new(tile, component_info);
            let resolution_tile = ResolutionTile::new(
                component_tile,
                component_info.num_resolution_levels() - 1 - header.skipped_resolution_levels,
            );

            let x_offset = header
                .size_data
                .image_area_x_offset
                .div_ceil(header.size_data.x_shrink_factor);
            let y_offset = header
                .size_data
                .image_area_y_offset
                .div_ceil(header.size_data.y_shrink_factor);
            let region_x1 = output_region.x.saturating_add(output_region.width);
            let region_y1 = output_region.y.saturating_add(output_region.height);
            let region = IntRect::from_ltrb(
                output_region.x.saturating_add(x_offset),
                output_region.y.saturating_add(y_offset),
                region_x1.saturating_add(x_offset),
                region_y1.saturating_add(y_offset),
            );
            let final_window = resolution_tile.rect.intersect(region);
            if final_window.is_empty() {
                continue;
            }
            if final_window.x1 == resolution_tile.rect.x1
                || final_window.y1 == resolution_tile.rect.y1
            {
                return None;
            }
            plan.final_windows[component_idx] = Some(final_window);

            let tile_decompositions = &storage.tile_decompositions[component_idx];
            let decompositions =
                &storage.decompositions[tile_decompositions.decompositions.clone()];
            let active_len = decompositions
                .len()
                .saturating_sub(header.skipped_resolution_levels as usize);

            if active_len == 0 {
                plan.add_sub_band_window(tile_decompositions.first_ll_sub_band, final_window);
                continue;
            }

            let mut required_output = final_window;
            for local_decomposition_idx in (0..active_len).rev() {
                let decomposition_idx =
                    tile_decompositions.decompositions.start + local_decomposition_idx;
                let decomposition = &decompositions[local_decomposition_idx];
                let expanded = required_output.expanded_within(
                    idwt_required_output_margin(component_info.wavelet_transform()),
                    decomposition.rect,
                );
                plan.add_idwt_window(decomposition_idx, expanded);

                let ll_window = plan.add_idwt_input_windows(decomposition, expanded, storage);
                if local_decomposition_idx == 0 {
                    plan.add_sub_band_window(tile_decompositions.first_ll_sub_band, ll_window);
                } else {
                    required_output = ll_window;
                }
            }
        }

        Some(plan)
    }

    pub(crate) fn code_block_required(&self, sub_band_idx: usize, rect: IntRect) -> bool {
        self.sub_band_windows
            .get(sub_band_idx)
            .and_then(|window| *window)
            .is_some_and(|window| window.intersects(rect))
    }

    pub(crate) fn sub_band_window(&self, sub_band_idx: usize) -> Option<IntRect> {
        self.sub_band_windows
            .get(sub_band_idx)
            .and_then(|window| *window)
    }

    pub(crate) fn idwt_window(&self, decomposition_idx: usize) -> Option<IntRect> {
        self.idwt_windows
            .get(decomposition_idx)
            .and_then(|window| *window)
    }

    pub(crate) fn final_window(&self, component_idx: usize) -> Option<IntRect> {
        self.final_windows
            .get(component_idx)
            .and_then(|window| *window)
    }

    fn add_sub_band_window(&mut self, sub_band_idx: usize, window: IntRect) {
        add_window(&mut self.sub_band_windows[sub_band_idx], window);
    }

    fn add_idwt_window(&mut self, decomposition_idx: usize, window: IntRect) {
        add_window(&mut self.idwt_windows[decomposition_idx], window);
    }

    fn add_idwt_input_windows(
        &mut self,
        decomposition: &Decomposition,
        output_window: IntRect,
        storage: &DecompositionStorage<'_>,
    ) -> IntRect {
        let hl = &storage.sub_bands[decomposition.sub_bands[0]];
        let lh = &storage.sub_bands[decomposition.sub_bands[1]];
        let hh = &storage.sub_bands[decomposition.sub_bands[2]];

        let ll_window = idwt_input_required_region(
            output_window,
            decomposition.rect,
            low_band_rect(decomposition.rect),
            true,
            true,
        );
        self.add_sub_band_window(
            decomposition.sub_bands[0],
            idwt_input_required_region(output_window, decomposition.rect, hl.rect, false, true),
        );
        self.add_sub_band_window(
            decomposition.sub_bands[1],
            idwt_input_required_region(output_window, decomposition.rect, lh.rect, true, false),
        );
        self.add_sub_band_window(
            decomposition.sub_bands[2],
            idwt_input_required_region(output_window, decomposition.rect, hh.rect, false, false),
        );

        ll_window
    }
}

fn add_window(slot: &mut Option<IntRect>, window: IntRect) {
    if window.is_empty() {
        return;
    }
    *slot = Some(slot.map_or(window, |existing| existing.union(window)));
}

fn low_band_rect(output_rect: IntRect) -> IntRect {
    IntRect::from_ltrb(
        output_rect.x0.div_ceil(2),
        output_rect.y0.div_ceil(2),
        output_rect.x1.div_ceil(2),
        output_rect.y1.div_ceil(2),
    )
}

fn idwt_required_output_margin(transform: WaveletTransform) -> u32 {
    match transform {
        WaveletTransform::Reversible53 => 16,
        WaveletTransform::Irreversible97 => 40,
    }
}

fn idwt_input_required_region(
    output_window: IntRect,
    output_rect: IntRect,
    band_rect: IntRect,
    low_x: bool,
    low_y: bool,
) -> IntRect {
    if output_window.is_empty() {
        return IntRect::from_xywh(0, 0, 0, 0);
    }

    let x0 = band_rect.x0.saturating_add(idwt_band_index(
        output_rect.x0,
        output_window.x0.saturating_sub(output_rect.x0),
        low_x,
    ));
    let x1 = band_rect.x0.saturating_add(
        idwt_band_index(
            output_rect.x0,
            output_window
                .x1
                .saturating_sub(1)
                .saturating_sub(output_rect.x0),
            low_x,
        )
        .saturating_add(1),
    );
    let y0 = band_rect.y0.saturating_add(idwt_band_index(
        output_rect.y0,
        output_window.y0.saturating_sub(output_rect.y0),
        low_y,
    ));
    let y1 = band_rect.y0.saturating_add(
        idwt_band_index(
            output_rect.y0,
            output_window
                .y1
                .saturating_sub(1)
                .saturating_sub(output_rect.y0),
            low_y,
        )
        .saturating_add(1),
    );

    IntRect::from_ltrb(
        x0.min(band_rect.x1),
        y0.min(band_rect.y1),
        x1.min(band_rect.x1),
        y1.min(band_rect.y1),
    )
}

pub(crate) fn idwt_band_coord(
    output_origin: u32,
    output_coord: u32,
    band_origin: u32,
    low: bool,
) -> u32 {
    band_origin.saturating_add(idwt_band_index(
        output_origin,
        output_coord.saturating_sub(output_origin),
        low,
    ))
}

fn idwt_band_index(origin: u32, local_coord: u32, low: bool) -> u32 {
    let global = u64::from(origin) + u64::from(local_coord);
    let origin = u64::from(origin);
    let index = if low {
        global.div_ceil(2).saturating_sub(origin.div_ceil(2))
    } else {
        (global / 2).saturating_sub(origin / 2)
    };
    u32::try_from(index).unwrap_or(u32::MAX)
}
