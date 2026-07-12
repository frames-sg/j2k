// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::super::build::{Decomposition, SubBand};
use super::super::codestream::WaveletTransform;
use super::super::decode::{DecompositionStorage, TileDecodeContext};
use super::super::rect::IntRect;
use super::super::roi;
use super::horizontal::filter_horizontal;
use super::model::{CoefficientSource, IDWTOutput};
use super::vertical::filter_vertical;
use crate::error::DecodingError;
use crate::j2c::Header;
use crate::{checked_decode_usize_product2, try_resize_decode_elements, Result};

pub(super) fn apply_roi(
    storage: &DecompositionStorage<'_>,
    tile_ctx: &mut TileDecodeContext,
    component_idx: usize,
    header: &Header<'_>,
    transform: WaveletTransform,
) -> Result<()> {
    let roi_plan = storage
        .roi_plan
        .as_ref()
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    let tile_decompositions = &storage.tile_decompositions[component_idx];
    let output = &mut tile_ctx.idwt_output;
    let ll_sub_band = &storage.sub_bands[tile_decompositions.first_ll_sub_band];

    let decompositions = &storage.decompositions[tile_decompositions.decompositions.clone()];
    let active_len = decompositions
        .len()
        .saturating_sub(header.skipped_resolution_levels as usize);

    if active_len == 0 {
        let Some(window) = roi_plan
            .sub_band_window(tile_decompositions.first_ll_sub_band)
            .or_else(|| roi_plan.final_window(component_idx))
        else {
            output.coefficients.clear();
            output.rect = IntRect::from_xywh(0, 0, 0, 0);
            return Ok(());
        };
        copy_sub_band_window_to_output(ll_sub_band, storage, window, output)?;
        return Ok(());
    }

    let mut current_coefficients = Vec::new();
    let mut current_rect = IntRect::from_xywh(0, 0, 0, 0);
    let mut have_current = false;

    for (local_idx, decomposition) in decompositions.iter().take(active_len).enumerate() {
        let decomposition_idx = tile_decompositions.decompositions.start + local_idx;
        let Some(output_window) = roi_plan.idwt_window(decomposition_idx) else {
            output.coefficients.clear();
            output.rect = IntRect::from_xywh(0, 0, 0, 0);
            return Ok(());
        };

        let ll_input = if have_current {
            CoefficientSource::new(&current_coefficients, current_rect, current_rect.width())
        } else {
            CoefficientSource::from_sub_band(ll_sub_band, storage)
        };

        let mut next = Vec::new();
        apply_level_roi(
            ll_input,
            &mut next,
            output_window,
            decomposition,
            transform,
            storage,
            &mut tile_ctx.debug_counters.idwt_output_samples,
        )?;
        current_coefficients = next;
        current_rect = output_window;
        have_current = true;
    }

    output.coefficients = current_coefficients;
    output.rect = current_rect;
    Ok(())
}

fn copy_sub_band_window_to_output(
    sub_band: &SubBand,
    storage: &DecompositionStorage<'_>,
    window: IntRect,
    output: &mut IDWTOutput,
) -> Result<()> {
    output.coefficients.clear();
    let required_len =
        checked_decode_usize_product2(window.width() as usize, window.height() as usize)?;
    try_resize_decode_elements(&mut output.coefficients, required_len, 0.0)?;
    let source = CoefficientSource::from_sub_band(sub_band, storage);
    for y in window.y0..window.y1 {
        for x in window.x0..window.x1 {
            let dst = (y - window.y0) as usize * window.width() as usize + (x - window.x0) as usize;
            output.coefficients[dst] = source.get(x, y);
        }
    }
    output.rect = window;
    Ok(())
}

fn apply_level_roi(
    ll: CoefficientSource<'_>,
    target: &mut Vec<f32>,
    output_window: IntRect,
    decomposition: &Decomposition,
    transform: WaveletTransform,
    storage: &DecompositionStorage<'_>,
    idwt_output_samples: &mut usize,
) -> Result<()> {
    let hl =
        CoefficientSource::from_sub_band(&storage.sub_bands[decomposition.sub_bands[0]], storage);
    let lh =
        CoefficientSource::from_sub_band(&storage.sub_bands[decomposition.sub_bands[1]], storage);
    let hh =
        CoefficientSource::from_sub_band(&storage.sub_bands[decomposition.sub_bands[2]], storage);

    target.clear();
    let required_len = checked_decode_usize_product2(
        output_window.width() as usize,
        output_window.height() as usize,
    )?;
    try_resize_decode_elements(target, required_len, 0.0)?;
    *idwt_output_samples = idwt_output_samples.saturating_add(required_len);

    interleave_samples_roi(ll, hl, lh, hh, target, output_window, decomposition.rect);
    if output_window.width() > 0 && output_window.height() > 0 {
        filter_horizontal(target, output_window, transform);
        filter_vertical(target, output_window, transform);
    }
    Ok(())
}

pub(super) fn interleave_samples_roi(
    ll: CoefficientSource<'_>,
    hl: CoefficientSource<'_>,
    lh: CoefficientSource<'_>,
    hh: CoefficientSource<'_>,
    output: &mut [f32],
    output_window: IntRect,
    decomposition_rect: IntRect,
) {
    let width = output_window.width() as usize;
    for y in output_window.y0..output_window.y1 {
        let low_y = y % 2 == 0;
        for x in output_window.x0..output_window.x1 {
            let low_x = x % 2 == 0;
            let (source, band_x, band_y) = match (low_x, low_y) {
                (true, true) => (
                    ll,
                    roi::idwt_band_coord(decomposition_rect.x0, x, ll.rect.x0, true),
                    roi::idwt_band_coord(decomposition_rect.y0, y, ll.rect.y0, true),
                ),
                (false, true) => (
                    hl,
                    roi::idwt_band_coord(decomposition_rect.x0, x, hl.rect.x0, false),
                    roi::idwt_band_coord(decomposition_rect.y0, y, hl.rect.y0, true),
                ),
                (true, false) => (
                    lh,
                    roi::idwt_band_coord(decomposition_rect.x0, x, lh.rect.x0, true),
                    roi::idwt_band_coord(decomposition_rect.y0, y, lh.rect.y0, false),
                ),
                (false, false) => (
                    hh,
                    roi::idwt_band_coord(decomposition_rect.x0, x, hh.rect.x0, false),
                    roi::idwt_band_coord(decomposition_rect.y0, y, hh.rect.y0, false),
                ),
            };
            let dst = (y - output_window.y0) as usize * width + (x - output_window.x0) as usize;
            output[dst] = source.get(band_x, band_y);
        }
    }
}
