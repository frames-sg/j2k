// SPDX-License-Identifier: MIT OR Apache-2.0

//! Region-sized copies of referenced direct plans for the CPU executors.
//!
//! A region's referenced plan holds only the code-blocks the region needs,
//! but its sub-bands and IDWT levels span whole resolution levels so that
//! every backend reads the same geometry. The CPU executors honour each
//! band's rectangle, so they can run a copy whose bands hold only what the
//! stores read: the IDWT windows native region decode uses, with each
//! sub-band grown to the whole code-blocks it decodes. Execution memory then
//! follows the region instead of the level.

#[cfg(test)]
mod tests;

use alloc::vec::Vec;

use super::{
    idwt_required_input_window_for_rects, idwt_required_output_margin, J2kRequiredBandRegion,
};
use crate::j2c::codestream::allocation::try_copy_vec;
use crate::{
    try_reserve_decode_elements, DecodingError, HtOwnedCodeBlockBatchJob, HtOwnedSubBandPlan,
    J2kDirectBandId, J2kDirectColorPlan, J2kDirectGrayscalePlan, J2kDirectGrayscaleStep,
    J2kDirectRgbaPlan, J2kDirectStoreStep, J2kOwnedCodeBlockBatchJob, J2kOwnedSubBandPlan, J2kRect,
    J2kReferencedClassicPlan, J2kReferencedHtj2kPlan, J2kReferencedTileGeometry,
    J2kReferencedTilePlan, Result,
};

/// Copy `plan` with every band sized to what its stores read, or return
/// `None` when that would not pay off: always for full and reduced decodes,
/// and for regions covering most of a small image.
#[doc(hidden)]
pub fn window_referenced_htj2k_plan_to_stores(
    plan: &J2kReferencedHtj2kPlan,
) -> Result<Option<J2kReferencedHtj2kPlan>> {
    let Some(tiles) = windowed_tiles(plan.tiles())? else {
        return Ok(None);
    };
    Ok(Some(match plan {
        J2kReferencedHtj2kPlan::Grayscale {
            full_dimensions,
            output_rect,
            payloads,
            ..
        } => J2kReferencedHtj2kPlan::grayscale(
            tiles,
            *full_dimensions,
            *output_rect,
            try_copy_vec(payloads)?,
        ),
        J2kReferencedHtj2kPlan::Color {
            full_dimensions,
            output_rect,
            payloads,
            ..
        } => J2kReferencedHtj2kPlan::color(
            tiles,
            *full_dimensions,
            *output_rect,
            try_copy_vec(payloads)?,
        ),
        J2kReferencedHtj2kPlan::Rgba {
            full_dimensions,
            output_rect,
            payloads,
            ..
        } => J2kReferencedHtj2kPlan::rgba(
            tiles,
            *full_dimensions,
            *output_rect,
            try_copy_vec(payloads)?,
        ),
    }))
}

/// Copy `plan` with every band sized to what its stores read, or return
/// `None` when that would not pay off: always for full and reduced decodes,
/// and for regions covering most of a small image.
#[doc(hidden)]
pub fn window_referenced_classic_plan_to_stores(
    plan: &J2kReferencedClassicPlan,
) -> Result<Option<J2kReferencedClassicPlan>> {
    let Some(tiles) = windowed_tiles(plan.tiles())? else {
        return Ok(None);
    };
    Ok(Some(match plan {
        J2kReferencedClassicPlan::Grayscale {
            full_dimensions,
            output_rect,
            payloads,
            ranges,
            ..
        } => J2kReferencedClassicPlan::grayscale(
            tiles,
            *full_dimensions,
            *output_rect,
            try_copy_vec(payloads)?,
            try_copy_vec(ranges)?,
        ),
        J2kReferencedClassicPlan::Color {
            full_dimensions,
            output_rect,
            payloads,
            ranges,
            ..
        } => J2kReferencedClassicPlan::color(
            tiles,
            *full_dimensions,
            *output_rect,
            try_copy_vec(payloads)?,
            try_copy_vec(ranges)?,
        ),
        J2kReferencedClassicPlan::Rgba {
            full_dimensions,
            output_rect,
            payloads,
            ranges,
            ..
        } => J2kReferencedClassicPlan::rgba(
            tiles,
            *full_dimensions,
            *output_rect,
            try_copy_vec(payloads)?,
            try_copy_vec(ranges)?,
        ),
    }))
}

/// A windowed copy trades the banded whole-level IDWT for the slower
/// windowed one, so it pays off once it at least halves the coefficients a
/// plan holds, or once the whole-level plan is large enough that its memory
/// matters more: 16 Mi coefficients, 64 MiB as `f32`.
const ALWAYS_WINDOW_COEFFICIENTS: u64 = 1 << 24;

/// Windowed copies of `tiles`, or `None` when windowing would not pay off.
fn windowed_tiles(tiles: &[J2kReferencedTilePlan]) -> Result<Option<Vec<J2kReferencedTilePlan>>> {
    let mut tile_windows = Vec::new();
    try_reserve_decode_elements(&mut tile_windows, tiles.len())?;
    let (mut whole_area, mut windowed_area) = (0_u64, 0_u64);
    for tile in tiles {
        let windows = component_plans(tile)
            .map(band_windows)
            .collect::<Result<Vec<_>>>()?;
        for (plan, windows) in component_plans(tile).zip(&windows) {
            let whole = plan_area(plan);
            whole_area = whole_area.saturating_add(whole);
            windowed_area =
                windowed_area.saturating_add(windows.as_ref().map_or(whole, BandWindows::area));
        }
        tile_windows.push(windows);
    }
    let pays_off =
        windowed_area.saturating_mul(2) <= whole_area || whole_area >= ALWAYS_WINDOW_COEFFICIENTS;
    if windowed_area >= whole_area || !pays_off {
        return Ok(None);
    }

    let mut windowed = Vec::new();
    try_reserve_decode_elements(&mut windowed, tiles.len())?;
    for (tile, windows) in tiles.iter().zip(&tile_windows) {
        let mut geometry = tile_geometry(tile)?;
        for (plan, windows) in geometry_component_plans(&mut geometry).zip(windows) {
            if let Some(windows) = windows {
                apply_band_windows(plan, windows)?;
            }
        }
        windowed.push(J2kReferencedTilePlan::new(
            tile.tile_index(),
            tile.decoded_rect(),
            tile.destination_rect(),
            tile.payload_records(),
            try_copy_vec(tile.classic_payloads())?,
            try_copy_vec(tile.classic_ranges())?,
            tile.wavelet_transform(),
            geometry,
        ));
    }
    Ok(Some(windowed))
}

fn component_plans(tile: &J2kReferencedTilePlan) -> impl Iterator<Item = &J2kDirectGrayscalePlan> {
    let plans: &[J2kDirectGrayscalePlan] = tile
        .grayscale_geometry()
        .map(core::slice::from_ref)
        .or_else(|| {
            tile.color_geometry()
                .map(|plan| plan.component_plans.as_slice())
        })
        .or_else(|| {
            tile.rgba_geometry()
                .map(|plan| plan.component_plans.as_slice())
        })
        .unwrap_or_default();
    plans.iter()
}

fn tile_geometry(tile: &J2kReferencedTilePlan) -> Result<J2kReferencedTileGeometry> {
    if let Some(plan) = tile.grayscale_geometry() {
        return Ok(J2kReferencedTileGeometry::Grayscale(try_copy_plan(plan)?));
    }
    if let Some(plan) = tile.color_geometry() {
        return Ok(J2kReferencedTileGeometry::Color(J2kDirectColorPlan {
            component_plans: try_copy_plans(&plan.component_plans)?,
            ..*plan
        }));
    }
    let plan = tile
        .rgba_geometry()
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    Ok(J2kReferencedTileGeometry::Rgba(J2kDirectRgbaPlan {
        component_plans: try_copy_plans(&plan.component_plans)?,
        ..*plan
    }))
}

fn try_copy_plans(plans: &[J2kDirectGrayscalePlan]) -> Result<Vec<J2kDirectGrayscalePlan>> {
    let mut copies = Vec::new();
    try_reserve_decode_elements(&mut copies, plans.len())?;
    for plan in plans {
        copies.push(try_copy_plan(plan)?);
    }
    Ok(copies)
}

/// Copy one component plan through fallible allocations, as the plan
/// builders allocate it.
fn try_copy_plan(plan: &J2kDirectGrayscalePlan) -> Result<J2kDirectGrayscalePlan> {
    let mut steps = Vec::new();
    try_reserve_decode_elements(&mut steps, plan.steps.len())?;
    for step in &plan.steps {
        steps.push(match step {
            J2kDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                let mut jobs = Vec::new();
                try_reserve_decode_elements(&mut jobs, sub_band.jobs.len())?;
                for job in &sub_band.jobs {
                    jobs.push(J2kOwnedCodeBlockBatchJob {
                        data: try_copy_vec(&job.data)?,
                        segments: try_copy_vec(&job.segments)?,
                        ..*job
                    });
                }
                J2kDirectGrayscaleStep::ClassicSubBand(J2kOwnedSubBandPlan { jobs, ..*sub_band })
            }
            J2kDirectGrayscaleStep::HtSubBand(sub_band) => {
                let mut jobs = Vec::new();
                try_reserve_decode_elements(&mut jobs, sub_band.jobs.len())?;
                for job in &sub_band.jobs {
                    jobs.push(HtOwnedCodeBlockBatchJob {
                        data: try_copy_vec(&job.data)?,
                        ..*job
                    });
                }
                J2kDirectGrayscaleStep::HtSubBand(HtOwnedSubBandPlan { jobs, ..*sub_band })
            }
            J2kDirectGrayscaleStep::Idwt(idwt) => J2kDirectGrayscaleStep::Idwt(*idwt),
            J2kDirectGrayscaleStep::Store(store) => J2kDirectGrayscaleStep::Store(*store),
        });
    }
    Ok(J2kDirectGrayscalePlan { steps, ..*plan })
}

fn geometry_component_plans(
    geometry: &mut J2kReferencedTileGeometry,
) -> impl Iterator<Item = &mut J2kDirectGrayscalePlan> {
    match geometry {
        J2kReferencedTileGeometry::Grayscale(plan) => core::slice::from_mut(plan).iter_mut(),
        J2kReferencedTileGeometry::Color(plan) => plan.component_plans.iter_mut(),
        J2kReferencedTileGeometry::Rgba(plan) => plan.component_plans.iter_mut(),
    }
}

/// The rectangle each band of one component plan keeps.
struct BandWindows {
    rects: Vec<(J2kDirectBandId, J2kRect)>,
}

impl BandWindows {
    /// Coefficients the windowed bands hold.
    fn area(&self) -> u64 {
        self.rects
            .iter()
            .fold(0, |area, (_, rect)| area.saturating_add(rect_area(*rect)))
    }

    fn get(&self, band_id: J2kDirectBandId) -> Result<J2kRect> {
        self.rects
            .iter()
            .find(|(id, _)| *id == band_id)
            .map(|(_, rect)| *rect)
            .ok_or_else(|| DecodingError::CodeBlockDecodeFailure.into())
    }
}

/// Windows for one component plan, computed from its whole-level geometry,
/// or `None` when every store reads its whole input band.
fn band_windows(plan: &J2kDirectGrayscalePlan) -> Result<Option<BandWindows>> {
    let mut required: Vec<(J2kDirectBandId, J2kRequiredBandRegion)> = Vec::new();
    let mut whole = true;
    for step in &plan.steps {
        if let J2kDirectGrayscaleStep::Store(store) = step {
            let copy = store_copy_region(store)?;
            whole &= copy.is_some_and(|copy| copy.to_rect() == store.input_rect);
            if let Some(copy) = copy {
                add_required(&mut required, store.input_band_id, copy)?;
            }
        }
    }
    if whole {
        return Ok(None);
    }

    let mut rects = Vec::new();
    // Native region decode expands each level's required output by the
    // synthesis margin before deriving the next level down, so the windowed
    // IDWT reproduces whole-level samples everywhere a store reads.
    for step in plan.steps.iter().rev() {
        let J2kDirectGrayscaleStep::Idwt(idwt) = step else {
            continue;
        };
        let window = lookup(&required, idwt.output_band_id)
            .map(|output| {
                output.expanded_within_rect(idwt_required_output_margin(idwt.transform), idwt.rect)
            })
            .filter(|window| window.width() > 0 && window.height() > 0);
        let Some(window) = window else {
            push_rect(&mut rects, idwt.output_band_id, empty_at(idwt.rect))?;
            continue;
        };
        push_rect(&mut rects, idwt.output_band_id, window.to_rect())?;
        for (band_id, band_rect, low_x, low_y) in [
            (idwt.ll_band_id, idwt.ll, true, true),
            (idwt.hl_band_id, idwt.hl, false, true),
            (idwt.lh_band_id, idwt.lh, true, false),
            (idwt.hh_band_id, idwt.hh, false, false),
        ] {
            let input =
                idwt_required_input_window_for_rects(window, idwt.rect, band_rect, low_x, low_y);
            if input.width() > 0 && input.height() > 0 {
                add_required(&mut required, band_id, input)?;
            }
        }
    }

    // Each code-block decodes whole into its band, so a sub-band keeps the
    // window the IDWT reads plus every job the plan decodes. A band split
    // across classic and HT steps shares one rectangle.
    let mut sub_bands: Vec<(J2kDirectBandId, J2kRect, Option<J2kRequiredBandRegion>)> = Vec::new();
    for step in &plan.steps {
        let (band_id, rect, jobs) = match step {
            J2kDirectGrayscaleStep::ClassicSubBand(sub_band) => (
                sub_band.band_id,
                sub_band.rect,
                job_bounds(
                    sub_band.rect,
                    sub_band
                        .jobs
                        .iter()
                        .map(|job| (job.output_x, job.output_y, job.width, job.height)),
                )?,
            ),
            J2kDirectGrayscaleStep::HtSubBand(sub_band) => (
                sub_band.band_id,
                sub_band.rect,
                job_bounds(
                    sub_band.rect,
                    sub_band
                        .jobs
                        .iter()
                        .map(|job| (job.output_x, job.output_y, job.width, job.height)),
                )?,
            ),
            J2kDirectGrayscaleStep::Idwt(_) | J2kDirectGrayscaleStep::Store(_) => continue,
        };
        if let Some(entry) = sub_bands.iter_mut().find(|(id, _, _)| *id == band_id) {
            entry.2 = union(entry.2, jobs);
        } else {
            let len = sub_bands.len() + 1;
            try_reserve_decode_elements(&mut sub_bands, len)?;
            sub_bands.push((band_id, rect, union(lookup(&required, band_id), jobs)));
        }
    }
    for (band_id, rect, stored) in sub_bands {
        let stored = stored.map_or_else(|| empty_at(rect), |stored| clip(stored.to_rect(), rect));
        push_rect(&mut rects, band_id, stored)?;
    }
    Ok(Some(BandWindows { rects }))
}

/// Rewrite one component plan's bands, jobs, IDWT levels and stores to the
/// rectangles in `windows`.
fn apply_band_windows(plan: &mut J2kDirectGrayscalePlan, windows: &BandWindows) -> Result<()> {
    for step in &mut plan.steps {
        match step {
            J2kDirectGrayscaleStep::ClassicSubBand(sub_band) => {
                let stored = windows.get(sub_band.band_id)?;
                for job in &mut sub_band.jobs {
                    rebase_job(sub_band.rect, stored, &mut job.output_x, &mut job.output_y)?;
                    job.output_stride = stored.width() as usize;
                }
                (sub_band.rect, sub_band.width, sub_band.height) =
                    (stored, stored.width(), stored.height());
            }
            J2kDirectGrayscaleStep::HtSubBand(sub_band) => {
                let stored = windows.get(sub_band.band_id)?;
                for job in &mut sub_band.jobs {
                    rebase_job(sub_band.rect, stored, &mut job.output_x, &mut job.output_y)?;
                    job.output_stride = stored.width() as usize;
                }
                (sub_band.rect, sub_band.width, sub_band.height) =
                    (stored, stored.width(), stored.height());
            }
            J2kDirectGrayscaleStep::Idwt(idwt) => {
                idwt.rect = windows.get(idwt.output_band_id)?;
                idwt.ll = windows.get(idwt.ll_band_id)?;
                idwt.hl = windows.get(idwt.hl_band_id)?;
                idwt.lh = windows.get(idwt.lh_band_id)?;
                idwt.hh = windows.get(idwt.hh_band_id)?;
            }
            J2kDirectGrayscaleStep::Store(store) => {
                let input = windows.get(store.input_band_id)?;
                if let Some(copy) = store_copy_region(store)? {
                    if copy.x0 < input.x0
                        || copy.y0 < input.y0
                        || copy.x1 > input.x1
                        || copy.y1 > input.y1
                    {
                        return Err(DecodingError::CodeBlockDecodeFailure.into());
                    }
                    store.source_x = copy.x0 - input.x0;
                    store.source_y = copy.y0 - input.y0;
                }
                store.input_rect = input;
            }
        }
    }
    Ok(())
}

/// Coefficients a whole-level component plan holds across its bands. A band
/// split across classic and HT steps appears as adjacent steps and counts
/// once.
fn plan_area(plan: &J2kDirectGrayscalePlan) -> u64 {
    let mut previous_sub_band = None;
    plan.steps.iter().fold(0_u64, |area, step| {
        let (band_id, rect) = match step {
            J2kDirectGrayscaleStep::ClassicSubBand(sub_band) => (sub_band.band_id, sub_band.rect),
            J2kDirectGrayscaleStep::HtSubBand(sub_band) => (sub_band.band_id, sub_band.rect),
            J2kDirectGrayscaleStep::Idwt(idwt) => return area.saturating_add(rect_area(idwt.rect)),
            J2kDirectGrayscaleStep::Store(_) => return area,
        };
        if previous_sub_band.replace(band_id) == Some(band_id) {
            area
        } else {
            area.saturating_add(rect_area(rect))
        }
    })
}

fn rect_area(rect: J2kRect) -> u64 {
    u64::from(rect.width()) * u64::from(rect.height())
}

/// The absolute input-band region one store copies, if any.
fn store_copy_region(store: &J2kDirectStoreStep) -> Result<Option<J2kRequiredBandRegion>> {
    let x0 = checked_add(store.input_rect.x0, store.source_x)?;
    let y0 = checked_add(store.input_rect.y0, store.source_y)?;
    Ok(J2kRequiredBandRegion::new(
        x0,
        y0,
        checked_add(x0, store.copy_width)?,
        checked_add(y0, store.copy_height)?,
    ))
}

/// The absolute bounds of the jobs a sub-band decodes, from band-relative
/// `(x, y, width, height)` placements.
fn job_bounds(
    rect: J2kRect,
    jobs: impl Iterator<Item = (u32, u32, u32, u32)>,
) -> Result<Option<J2kRequiredBandRegion>> {
    let mut bounds = None;
    for (x, y, width, height) in jobs {
        let x0 = checked_add(rect.x0, x)?;
        let y0 = checked_add(rect.y0, y)?;
        let job =
            J2kRequiredBandRegion::new(x0, y0, checked_add(x0, width)?, checked_add(y0, height)?);
        bounds = union(bounds, job);
    }
    Ok(bounds)
}

fn rebase_job(from: J2kRect, to: J2kRect, output_x: &mut u32, output_y: &mut u32) -> Result<()> {
    *output_x = checked_add(from.x0, *output_x)?
        .checked_sub(to.x0)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    *output_y = checked_add(from.y0, *output_y)?
        .checked_sub(to.y0)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    Ok(())
}

fn add_required(
    required: &mut Vec<(J2kDirectBandId, J2kRequiredBandRegion)>,
    band_id: J2kDirectBandId,
    region: J2kRequiredBandRegion,
) -> Result<()> {
    if let Some(entry) = required.iter_mut().find(|(id, _)| *id == band_id) {
        entry.1 = entry.1.union(region);
    } else {
        let len = required.len() + 1;
        try_reserve_decode_elements(required, len)?;
        required.push((band_id, region));
    }
    Ok(())
}

fn push_rect(
    rects: &mut Vec<(J2kDirectBandId, J2kRect)>,
    band_id: J2kDirectBandId,
    rect: J2kRect,
) -> Result<()> {
    let len = rects.len() + 1;
    try_reserve_decode_elements(rects, len)?;
    rects.push((band_id, rect));
    Ok(())
}

fn lookup(
    required: &[(J2kDirectBandId, J2kRequiredBandRegion)],
    band_id: J2kDirectBandId,
) -> Option<J2kRequiredBandRegion> {
    required
        .iter()
        .find(|(id, _)| *id == band_id)
        .map(|(_, region)| *region)
}

fn union(
    a: Option<J2kRequiredBandRegion>,
    b: Option<J2kRequiredBandRegion>,
) -> Option<J2kRequiredBandRegion> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.union(b)),
        (a, b) => a.or(b),
    }
}

fn clip(rect: J2kRect, within: J2kRect) -> J2kRect {
    let x0 = rect.x0.max(within.x0).min(within.x1);
    let y0 = rect.y0.max(within.y0).min(within.y1);
    J2kRect {
        x0,
        y0,
        x1: rect.x1.min(within.x1).max(x0),
        y1: rect.y1.min(within.y1).max(y0),
    }
}

const fn empty_at(rect: J2kRect) -> J2kRect {
    J2kRect {
        x0: rect.x0,
        y0: rect.y0,
        x1: rect.x0,
        y1: rect.y0,
    }
}

fn checked_add(a: u32, b: u32) -> Result<u32> {
    a.checked_add(b)
        .ok_or_else(|| DecodingError::CodeBlockDecodeFailure.into())
}
