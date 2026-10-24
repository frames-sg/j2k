// SPDX-License-Identifier: MIT OR Apache-2.0

//! Region plans reconstruct only the windows their output needs: each Tier-1
//! sub-band is laid out over its retained code blocks and required region,
//! and each IDWT step over its required output window. The IDWT kernels read
//! bands by absolute band coordinate, so every input rectangle is the extent
//! of the buffer that holds the band.

use j2k_native::J2kRequiredBandRegion as RequiredBandRegion;

use super::{
    required_regions::RequiredBandRegions, shared::CudaPlanOwners, CudaHtj2kBandId, CudaHtj2kRect,
    Error, PLAN_OUTPUT_RECT_MISMATCH,
};
use crate::allocation::HostPhaseBudget;

const WINDOW_EXTENTS: &str = "CUDA region plan band windows";

pub(super) fn window_region_plan(
    owners: &mut CudaPlanOwners,
    required: &RequiredBandRegions,
    host_budget: &mut HostPhaseBudget,
) -> Result<(), Error> {
    let band_count =
        owners.subbands.len() + owners.classic_subbands.len() + owners.idwt_steps.len();
    // Full absolute rectangle of every band, before any window is applied.
    let mut full = host_budget.try_vec_with_capacity_named(band_count * 4 + 1, WINDOW_EXTENTS)?;
    for step in &owners.idwt_steps {
        full.extend([
            (step.ll_band_id, step.ll_rect),
            (step.hl_band_id, step.hl_rect),
            (step.lh_band_id, step.lh_rect),
            (step.hh_band_id, step.hh_rect),
        ]);
    }
    if let [store] = owners.store_steps.as_slice() {
        full.push((store.input_band_id, store.input_rect));
    }
    let mut extents = host_budget.try_vec_with_capacity_named(band_count, WINDOW_EXTENTS)?;
    window_tier1_bands(owners, required, &full, &mut extents);

    for step in &mut owners.idwt_steps {
        if let Some(window) = required.idwt_window(step.output_band_id) {
            step.rect = absolute(step.rect, window);
        }
        for (band_id, rect) in [
            (step.ll_band_id, &mut step.ll_rect),
            (step.hl_band_id, &mut step.hl_rect),
            (step.lh_band_id, &mut step.lh_rect),
            (step.hh_band_id, &mut step.hh_rect),
        ] {
            if let Some(extent) = find_rect(&extents, band_id) {
                *rect = extent;
            }
        }
        extents.push((step.output_band_id, step.rect));
    }

    for store in &mut owners.store_steps {
        let Some(extent) = find_rect(&extents, store.input_band_id) else {
            continue;
        };
        store.source_x = store
            .source_x
            .checked_sub(extent.x0 - store.input_rect.x0)
            .ok_or_else(rect_mismatch)?;
        store.source_y = store
            .source_y
            .checked_sub(extent.y0 - store.input_rect.y0)
            .ok_or_else(rect_mismatch)?;
        if store.source_x + store.copy_width > extent.x1 - extent.x0
            || store.source_y + store.copy_height > extent.y1 - extent.y0
        {
            return Err(rect_mismatch());
        }
        store.input_rect = extent;
    }
    Ok(())
}

/// Lays every HT and classic sub-band out over its window and records the
/// window's absolute rectangle in `extents`.
fn window_tier1_bands(
    owners: &mut CudaPlanOwners,
    required: &RequiredBandRegions,
    full: &[(CudaHtj2kBandId, CudaHtj2kRect)],
    extents: &mut Vec<(CudaHtj2kBandId, CudaHtj2kRect)>,
) {
    for subband in &mut owners.subbands {
        let (Some(rect), Some(blocks)) = (
            find_rect(full, subband.band_id),
            block_range(
                &mut owners.code_blocks,
                subband.code_block_start,
                subband.code_block_count,
            ),
        ) else {
            continue;
        };
        let Some(window) = window_blocks(
            required.get(subband.band_id).copied(),
            blocks,
            (subband.width, subband.height),
            |block| {
                (
                    &mut block.output_x,
                    &mut block.output_y,
                    &mut block.output_stride,
                    block.width,
                    block.height,
                )
            },
        ) else {
            continue;
        };
        let extent = absolute(rect, window);
        (subband.x0, subband.y0, subband.x1, subband.y1) =
            (extent.x0, extent.y0, extent.x1, extent.y1);
        (subband.width, subband.height) = (window.width(), window.height());
        extents.push((subband.band_id, extent));
    }
    for subband in &mut owners.classic_subbands {
        let (Some(rect), Some(blocks)) = (
            find_rect(full, subband.band_id),
            block_range(
                &mut owners.classic_code_blocks,
                subband.code_block_start,
                subband.code_block_count,
            ),
        ) else {
            continue;
        };
        let Some(window) = window_blocks(
            required.get(subband.band_id).copied(),
            blocks,
            (subband.width, subband.height),
            |block| {
                (
                    &mut block.output_x,
                    &mut block.output_y,
                    &mut block.output_stride,
                    block.width,
                    block.height,
                )
            },
        ) else {
            continue;
        };
        (subband.width, subband.height) = (window.width(), window.height());
        extents.push((subband.band_id, absolute(rect, window)));
    }
}

fn block_range<B>(blocks: &mut [B], start: u32, count: u32) -> Option<&mut [B]> {
    let start = start as usize;
    blocks.get_mut(start..start.checked_add(count as usize)?)
}

/// Band-local window covering `required` and every retained block, clamped to
/// the band; the blocks are rebased into it.
fn window_blocks<B>(
    required: Option<RequiredBandRegion>,
    blocks: &mut [B],
    band: (u32, u32),
    fields: fn(&mut B) -> (&mut u32, &mut u32, &mut u32, u32, u32),
) -> Option<RequiredBandRegion> {
    let mut covered = required;
    for block in blocks.iter_mut() {
        let (x, y, _, width, height) = fields(block);
        let Some(block) = RequiredBandRegion::new(*x, *y, *x + width, *y + height) else {
            continue;
        };
        covered = Some(covered.map_or(block, |window| window.union(block)));
    }
    let covered = covered?;
    let window = RequiredBandRegion::new(
        covered.x0,
        covered.y0,
        covered.x1.min(band.0),
        covered.y1.min(band.1),
    )?;
    for block in blocks.iter_mut() {
        let (x, y, stride, _, _) = fields(block);
        *x -= window.x0;
        *y -= window.y0;
        *stride = window.width();
    }
    Some(window)
}

fn absolute(rect: CudaHtj2kRect, window: RequiredBandRegion) -> CudaHtj2kRect {
    CudaHtj2kRect {
        x0: rect.x0 + window.x0,
        y0: rect.y0 + window.y0,
        x1: rect.x0 + window.x1,
        y1: rect.y0 + window.y1,
    }
}

fn find_rect(
    rects: &[(CudaHtj2kBandId, CudaHtj2kRect)],
    band_id: CudaHtj2kBandId,
) -> Option<CudaHtj2kRect> {
    rects
        .iter()
        .find(|(existing, _)| *existing == band_id)
        .map(|(_, rect)| *rect)
}

fn rect_mismatch() -> Error {
    Error::capability_rejected(j2k_core::CapabilityRejection::geometry_mismatch(
        PLAN_OUTPUT_RECT_MISMATCH,
    ))
}
