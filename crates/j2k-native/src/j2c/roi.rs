use alloc::vec::Vec;

use super::build::Decomposition;
use super::codestream::{Header, SizeData, WaveletTransform};
use super::decode::{DecompositionStorage, OutputRegion};
use super::rect::IntRect;
use super::tile::{ComponentTile, ResolutionTile, Tile};
use crate::{
    idwt_required_input_window_for_rects, try_resize_decode_elements, J2kRequiredBandRegion, Result,
};

/// Whether a reference-grid tile can contribute to a decoded output region.
///
/// The output region is expressed in final image coordinates. Tile bounds are
/// projected into the same component- and resolution-shrunk grid with outward
/// rounding, which may retain a boundary tile but never drops a contributor.
pub(crate) fn tile_intersects_output_region(
    tile_rect: IntRect,
    size_data: &SizeData,
    output_region: OutputRegion,
) -> bool {
    let x_shrink = size_data
        .x_shrink_factor
        .saturating_mul(size_data.x_resolution_shrink_factor)
        .max(1);
    let y_shrink = size_data
        .y_shrink_factor
        .saturating_mul(size_data.y_resolution_shrink_factor)
        .max(1);
    let region = output_region_rect(size_data, output_region);
    let tile_rect = IntRect::from_ltrb(
        tile_rect.x0 / x_shrink,
        tile_rect.y0 / y_shrink,
        tile_rect.x1.div_ceil(x_shrink),
        tile_rect.y1.div_ceil(y_shrink),
    );
    tile_rect.intersects(region)
}

pub(crate) fn output_region_rect(size_data: &SizeData, output_region: OutputRegion) -> IntRect {
    let (x_offset, y_offset) = output_grid_offset(size_data);
    let x0 = output_region.x.saturating_add(x_offset);
    let y0 = output_region.y.saturating_add(y_offset);
    IntRect::from_ltrb(
        x0,
        y0,
        x0.saturating_add(output_region.width),
        y0.saturating_add(output_region.height),
    )
}

/// Return the image-area origin in the final component- and
/// resolution-shrunk output grid.
pub(crate) fn output_grid_offset(size_data: &SizeData) -> (u32, u32) {
    let x_shrink_factor = size_data
        .checked_x_shrink_factor()
        .expect("validated JPEG 2000 horizontal shrink factors");
    let y_shrink_factor = size_data
        .checked_y_shrink_factor()
        .expect("validated JPEG 2000 vertical shrink factors");
    (
        size_data.image_area_x_offset.div_ceil(x_shrink_factor),
        size_data.image_area_y_offset.div_ceil(y_shrink_factor),
    )
}

#[derive(Debug)]
#[expect(
    clippy::struct_field_names,
    reason = "the repeated _windows suffix distinguishes the three ROI planning stages"
)]
pub(crate) struct RoiPlan {
    sub_band_windows: Vec<Option<IntRect>>,
    idwt_windows: Vec<Option<IntRect>>,
    final_windows: Vec<Option<IntRect>>,
}

crate::move_only::assert_move_only!(RoiPlan);

impl RoiPlan {
    pub(crate) fn build(
        tile: &Tile<'_>,
        header: &Header<'_>,
        storage: &DecompositionStorage<'_>,
        output_region: OutputRegion,
    ) -> Result<Option<Self>> {
        if tile.component_infos.iter().any(|component_info| {
            component_info.size_info.horizontal_resolution != 1
                || component_info.size_info.vertical_resolution != 1
        }) {
            return Ok(None);
        }

        let mut sub_band_windows = Vec::new();
        try_resize_decode_elements(&mut sub_band_windows, storage.sub_bands.len(), None)?;
        let mut idwt_windows = Vec::new();
        try_resize_decode_elements(&mut idwt_windows, storage.decompositions.len(), None)?;
        let mut final_windows = Vec::new();
        try_resize_decode_elements(&mut final_windows, tile.component_infos.len(), None)?;
        let mut plan = Self {
            sub_band_windows,
            idwt_windows,
            final_windows,
        };

        for (component_idx, component_info) in tile.component_infos.iter().enumerate() {
            let component_tile = ComponentTile::new(tile, component_info);
            let resolution_tile = ResolutionTile::new(
                component_tile,
                component_info.num_resolution_levels() - 1 - header.skipped_resolution_levels,
            );

            let region = output_region_rect(&header.size_data, output_region);
            let final_window = resolution_tile.rect.intersect(region);
            if final_window.is_empty() {
                continue;
            }
            if final_window.x1 == resolution_tile.rect.x1
                || final_window.y1 == resolution_tile.rect.y1
            {
                return Ok(None);
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
                    roi_required_output_margin(component_info.wavelet_transform()),
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

        Ok(Some(plan))
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

        let ll_window = int_rect_from_required_region(idwt_required_input_window_for_rects(
            required_region_from_int_rect(output_window),
            decomposition.rect.into(),
            low_band_rect(decomposition.rect).into(),
            true,
            true,
        ));
        self.add_sub_band_window(
            decomposition.sub_bands[0],
            int_rect_from_required_region(idwt_required_input_window_for_rects(
                required_region_from_int_rect(output_window),
                decomposition.rect.into(),
                hl.rect.into(),
                false,
                true,
            )),
        );
        self.add_sub_band_window(
            decomposition.sub_bands[1],
            int_rect_from_required_region(idwt_required_input_window_for_rects(
                required_region_from_int_rect(output_window),
                decomposition.rect.into(),
                lh.rect.into(),
                true,
                false,
            )),
        );
        self.add_sub_band_window(
            decomposition.sub_bands[2],
            int_rect_from_required_region(idwt_required_input_window_for_rects(
                required_region_from_int_rect(output_window),
                decomposition.rect.into(),
                hh.rect.into(),
                false,
                false,
            )),
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

fn native_wavelet_transform(transform: WaveletTransform) -> crate::J2kWaveletTransform {
    match transform {
        WaveletTransform::Reversible53 => crate::J2kWaveletTransform::Reversible53,
        WaveletTransform::Irreversible97 => crate::J2kWaveletTransform::Irreversible97,
    }
}

fn roi_required_output_margin(transform: WaveletTransform) -> u32 {
    crate::idwt_required_output_margin(native_wavelet_transform(transform))
}

fn required_region_from_int_rect(rect: IntRect) -> J2kRequiredBandRegion {
    J2kRequiredBandRegion {
        x0: rect.x0,
        y0: rect.y0,
        x1: rect.x1,
        y1: rect.y1,
    }
}

fn int_rect_from_required_region(region: J2kRequiredBandRegion) -> IntRect {
    IntRect::from_ltrb(region.x0, region.y0, region.x1, region.y1)
}

#[cfg(test)]
mod tests {
    use super::{tile_intersects_output_region, IntRect, OutputRegion, SizeData};
    use crate::j2c::codestream::ComponentSizeInfo;

    fn size_data(
        image_offset: (u32, u32),
        component_shrink: (u32, u32),
        resolution_shrink: (u32, u32),
    ) -> SizeData {
        SizeData {
            decoder_capabilities: 0,
            reference_grid_width: 515,
            reference_grid_height: 389,
            image_area_x_offset: image_offset.0,
            image_area_y_offset: image_offset.1,
            tile_width: 128,
            tile_height: 128,
            tile_x_offset: 1,
            tile_y_offset: 1,
            component_sizes: vec![ComponentSizeInfo {
                precision: 8,
                signed: false,
                horizontal_resolution: 1,
                vertical_resolution: 1,
            }],
            x_shrink_factor: component_shrink.0,
            y_shrink_factor: component_shrink.1,
            x_resolution_shrink_factor: resolution_shrink.0,
            y_resolution_shrink_factor: resolution_shrink.1,
        }
    }

    #[test]
    fn tile_intersection_handles_nonzero_image_and_tile_origins() {
        let size_data = size_data((3, 5), (1, 1), (4, 2));
        let first_tile = IntRect::from_ltrb(3, 5, 129, 129);
        let next_tile = IntRect::from_ltrb(129, 5, 257, 129);
        let top_left = OutputRegion {
            x: 0,
            y: 0,
            width: 8,
            height: 8,
        };

        assert!(tile_intersects_output_region(
            first_tile, &size_data, top_left
        ));
        assert!(!tile_intersects_output_region(
            next_tile, &size_data, top_left
        ));
    }

    #[test]
    fn tile_intersection_combines_component_and_resolution_shrink() {
        let size_data = size_data((3, 5), (2, 2), (2, 4));
        let tile = IntRect::from_ltrb(129, 129, 257, 257);

        assert!(tile_intersects_output_region(
            tile,
            &size_data,
            OutputRegion {
                x: 31,
                y: 15,
                width: 2,
                height: 2,
            }
        ));
        assert!(!tile_intersects_output_region(
            tile,
            &size_data,
            OutputRegion {
                x: 0,
                y: 0,
                width: 8,
                height: 8,
            }
        ));
    }
}
