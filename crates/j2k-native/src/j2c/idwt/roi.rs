// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;
use core::mem::size_of;

use super::super::build::{Decomposition, SubBand};
use super::super::codestream::WaveletTransform;
use super::super::decode::{DecompositionStorage, TileDecodeContext, WorkspaceBytes};
use super::super::rect::IntRect;
use super::horizontal::filter_horizontal_codestream;
use super::lines::{reconstruct_rows, ring_samples, LineReconstruction, StoredRows, SubBandRows};
use super::model::{CoefficientSource, IDWTOutput, IdwtPlacement};
use super::vertical::filter_vertical_codestream;
use crate::error::{DecodingError, ValidationError};
use crate::j2c::roi::output_region_rect;
use crate::j2c::Header;
use crate::math::{SimdBuffer, SIMD_WIDTH};
use crate::{checked_decode_usize_product2, try_resize_decode_elements, Result};

/// The most f32 samples [`apply_roi`] holds at once for any one component.
///
/// Each level reads the previous level's window and writes a new one, so both
/// are live together. With no active level, the copied sub-band window is the
/// only buffer. Components run one after another, and [`apply_roi`] releases
/// the previous component's output before it allocates, so the peak is the
/// largest per-component peak rather than their sum. A last level written
/// into a `channel_data` plane uses padding the plane already holds. The
/// row-streamed IDWT instead holds every level's ring of rows at once, charged
/// at the 9/7 ring depth whatever the component's transform. With no active
/// level it holds nothing, since it copies the LL window straight into the
/// plane.
pub(crate) fn roi_idwt_peak_elements(
    storage: &DecompositionStorage<'_>,
    skipped_resolution_levels: u8,
) -> Result<usize> {
    let Some(roi_plan) = storage.roi_plan.as_ref() else {
        return Ok(0);
    };
    let mut peak = 0usize;
    for (component_idx, tile_decompositions) in storage.tile_decompositions.iter().enumerate() {
        let active_len = tile_decompositions
            .decompositions
            .len()
            .saturating_sub(usize::from(skipped_resolution_levels));
        if active_len == 0 {
            if storage.line_region_idwt {
                continue;
            }
            if let Some(window) = roi_plan
                .sub_band_window(tile_decompositions.first_ll_sub_band)
                .or_else(|| roi_plan.final_window(component_idx))
            {
                peak = peak.max(window_elements(window)?);
            }
            continue;
        }
        if storage.line_region_idwt {
            let windows = tile_decompositions
                .decompositions
                .clone()
                .take(active_len)
                .map_while(|decomposition_idx| roi_plan.idwt_window(decomposition_idx));
            peak = peak.max(ring_samples(windows, WaveletTransform::Irreversible97)?);
            continue;
        }
        let mut previous = 0usize;
        for (level, decomposition_idx) in tile_decompositions
            .decompositions
            .clone()
            .take(active_len)
            .enumerate()
        {
            let Some(window) = roi_plan.idwt_window(decomposition_idx) else {
                break;
            };
            let current = window_elements(window)?;
            let in_channel = storage.last_idwt_level_in_channels && level + 1 == active_len;
            let live = if in_channel {
                previous
            } else {
                previous
                    .checked_add(current)
                    .ok_or(ValidationError::ImageTooLarge)?
            };
            peak = peak.max(live);
            previous = current;
        }
    }
    Ok(peak)
}

/// Whether every component of the tile can write its last IDWT level into
/// the padding of its `channel_data` plane: the decode reserved that padding,
/// each component has a level to apply and needs the whole output region,
/// and each last level's window fits its plane's allocation.
pub(crate) fn last_level_fits_channels(
    storage: &DecompositionStorage<'_>,
    tile_ctx: &TileDecodeContext,
    header: &Header<'_>,
) -> Result<bool> {
    let (Some(roi_plan), Some(_), Some(output_region)) = (
        storage.roi_plan.as_ref(),
        tile_ctx.last_idwt_level_padding,
        tile_ctx.output_region,
    ) else {
        return Ok(false);
    };
    if storage.exact_integer_decode || storage.tile_decompositions.is_empty() {
        return Ok(false);
    }
    let region = output_region_rect(&header.size_data, output_region);
    for (component_idx, tile_decompositions) in storage.tile_decompositions.iter().enumerate() {
        let active_len = tile_decompositions
            .decompositions
            .len()
            .saturating_sub(usize::from(header.skipped_resolution_levels));
        let last_window = active_len
            .checked_sub(1)
            .and_then(|last| roi_plan.idwt_window(tile_decompositions.decompositions.start + last));
        let (Some(last_window), Some(channel)) =
            (last_window, tile_ctx.channel_data.get(component_idx))
        else {
            return Ok(false);
        };
        if roi_plan.final_window(component_idx) != Some(region)
            || window_elements(last_window)? > channel.container.capacity()
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn window_elements(window: IntRect) -> Result<usize> {
    checked_decode_usize_product2(window.width() as usize, window.height() as usize)
}

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
        return apply_roi_without_levels(storage, tile_ctx, component_idx, header);
    }

    // The previous component's output is never reused here, so release it
    // before allocating this component's windows.
    output.coefficients = Vec::new();
    output.rect = IntRect::from_xywh(0, 0, 0, 0);
    if storage.line_region_idwt {
        return apply_roi_lines(
            storage,
            tile_ctx,
            component_idx,
            &decompositions[..active_len],
            header,
            transform,
        );
    }
    // Everything else the decode holds stays the same while the level
    // windows below come and go, so sample it once.
    let workspace = WorkspaceBytes::sample(tile_ctx, storage)?;
    let output = &mut tile_ctx.idwt_output;

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

        if storage.last_idwt_level_in_channels && local_idx + 1 == active_len {
            // The plane reserved this window's padding; `store` compacts it.
            let target = tile_ctx.channel_data[component_idx]
                .container
                .padded_scratch(window_elements(output_window)?)
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            fill_level_roi(
                ll_input,
                target,
                output_window,
                decomposition,
                transform,
                storage,
                &mut tile_ctx.debug_counters.idwt_output_samples,
            );
            let window_bytes = current_coefficients.capacity() * size_of::<f32>();
            tile_ctx
                .debug_counters
                .workspace_peak
                .record(workspace.with_idwt(window_bytes));
            output.coefficients = Vec::new();
            output.rect = output_window;
            output.placement = IdwtPlacement::ChannelPadding;
            return Ok(());
        }

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
        let window_bytes = (current_coefficients.capacity() + next.capacity()) * size_of::<f32>();
        tile_ctx
            .debug_counters
            .workspace_peak
            .record(workspace.with_idwt(window_bytes));
        current_coefficients = next;
        current_rect = output_window;
        have_current = true;
    }

    output.coefficients = current_coefficients;
    output.rect = current_rect;
    Ok(())
}

/// Stream the component's active levels a row at a time and write the last
/// level's rows inside the tile and output region into its `channel_data`
/// plane, leaving the level shift to `store`.
fn apply_roi_lines(
    storage: &DecompositionStorage<'_>,
    tile_ctx: &mut TileDecodeContext,
    component_idx: usize,
    decompositions: &[Decomposition],
    header: &Header<'_>,
    transform: WaveletTransform,
) -> Result<()> {
    let Some(mut lines) = ComponentLines::prepare(
        storage,
        tile_ctx,
        component_idx,
        decompositions.len(),
        header,
        transform,
    )?
    else {
        return Ok(());
    };
    let result = lines.run(&mut StoredRows { storage });
    lines.finish(tile_ctx, result)
}

/// A component's streamed IDWT, ready to run: its levels' windows and
/// sub-bands, and the rings and output plane it writes, taken from the
/// decode context until [`Self::finish`] returns them. While it runs, the
/// rows it reads may be decoded on demand.
pub(crate) struct ComponentLines {
    component_idx: usize,
    plan: LinePlan,
    ring: Vec<f32>,
    ring_len: usize,
    plane: SimdBuffer<SIMD_WIDTH>,
    output_samples: usize,
}

/// The geometry of one component's streamed reconstruction.
pub(crate) struct LinePlan {
    windows: Vec<IntRect>,
    sub_bands: Vec<[usize; 3]>,
    ll_sub_band: usize,
    transform: WaveletTransform,
    last_window: IntRect,
    /// The last level's rows the output receives.
    pub(crate) copy: IntRect,
    /// The output region, in the same coordinates as `copy`.
    pub(crate) region: IntRect,
}

impl LinePlan {
    /// Plan the first `active_levels` levels of the component. Returns
    /// `None` when a level has no window, so there is nothing to
    /// reconstruct, as in the whole-window path.
    pub(crate) fn new(
        storage: &DecompositionStorage<'_>,
        tile_ctx: &TileDecodeContext,
        component_idx: usize,
        active_levels: usize,
        header: &Header<'_>,
        transform: WaveletTransform,
    ) -> Result<Option<Self>> {
        let roi_plan = storage
            .roi_plan
            .as_ref()
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        let tile_decompositions = &storage.tile_decompositions[component_idx];
        let mut windows = Vec::new();
        windows
            .try_reserve_exact(active_levels)
            .map_err(|_| DecodingError::HostAllocationFailed)?;
        let mut sub_bands = Vec::new();
        sub_bands
            .try_reserve_exact(active_levels)
            .map_err(|_| DecodingError::HostAllocationFailed)?;
        for decomposition_idx in tile_decompositions
            .decompositions
            .clone()
            .take(active_levels)
        {
            // Without a window the level has nothing to reconstruct, as in the
            // whole-window path.
            let Some(window) = roi_plan.idwt_window(decomposition_idx) else {
                return Ok(None);
            };
            windows.push(window);
            sub_bands.push(storage.decompositions[decomposition_idx].sub_bands);
        }
        let (Some(last_window), Some(final_window), Some(output_region)) = (
            windows.last().copied(),
            roi_plan.final_window(component_idx),
            tile_ctx.output_region,
        ) else {
            return Ok(None);
        };
        Ok(Some(Self {
            windows,
            sub_bands,
            ll_sub_band: tile_decompositions.first_ll_sub_band,
            transform,
            last_window,
            copy: last_window.intersect(final_window),
            region: output_region_rect(&header.size_data, output_region),
        }))
    }

    /// Samples the levels' rings hold together.
    pub(crate) fn ring_len(&self) -> Result<usize> {
        ring_samples(self.windows.iter().copied(), self.transform)
    }

    /// Start the reconstruction, with `ring` holding at least
    /// [`Self::ring_len`] samples.
    pub(crate) fn reconstruction<'r>(&self, ring: &'r mut [f32]) -> Result<LineReconstruction<'r>> {
        LineReconstruction::new(
            &self.sub_bands,
            &self.windows,
            self.ll_sub_band,
            self.transform,
            ring,
            self.copy,
        )
    }
}

impl ComponentLines {
    /// Prepare the first `active_levels` levels of the component, as
    /// [`LinePlan::new`] plans them, with the plane that receives them.
    pub(crate) fn prepare(
        storage: &DecompositionStorage<'_>,
        tile_ctx: &mut TileDecodeContext,
        component_idx: usize,
        active_levels: usize,
        header: &Header<'_>,
        transform: WaveletTransform,
    ) -> Result<Option<Self>> {
        let Some(plan) = LinePlan::new(
            storage,
            tile_ctx,
            component_idx,
            active_levels,
            header,
            transform,
        )?
        else {
            return Ok(None);
        };
        let mut ring = core::mem::take(&mut tile_ctx.idwt_line_rings);
        let ring_len = plan.ring_len()?;
        if ring.len() < ring_len {
            try_resize_decode_elements(&mut ring, ring_len, 0.0)?;
        }
        tile_ctx.idwt_line_rings = ring;
        record_workspace_now(tile_ctx, storage)?;
        let ring = core::mem::take(&mut tile_ctx.idwt_line_rings);
        let plane = core::mem::replace(
            &mut tile_ctx.channel_data[component_idx].container,
            SimdBuffer::empty(),
        );
        Ok(Some(Self {
            component_idx,
            plan,
            ring,
            ring_len,
            plane,
            output_samples: 0,
        }))
    }

    /// Reconstruct the levels, reading their sub-bands through `rows`, and
    /// write the last level's rows into the plane.
    pub(crate) fn run(&mut self, rows: &mut dyn SubBandRows) -> Result<()> {
        let plan = &self.plan;
        let (copy, region) = (plan.copy, plan.region);
        let plane = &mut self.plane;
        let plane_width = region.width() as usize;
        let mut write_row = |y: u32, samples: &[f32]| -> Result<()> {
            let start = (y - region.y0) as usize * plane_width + (copy.x0 - region.x0) as usize;
            plane
                .get_mut(start..start + samples.len())
                .ok_or(DecodingError::CodeBlockDecodeFailure)?
                .copy_from_slice(samples);
            Ok(())
        };
        reconstruct_rows(
            rows,
            &plan.sub_bands,
            &plan.windows,
            plan.ll_sub_band,
            plan.transform,
            &mut self.ring[..self.ring_len],
            copy,
            &mut write_row,
            &mut self.output_samples,
        )
    }

    /// Return the rings and plane to the context and, when `result` is a
    /// success, record that the plane holds the last level.
    pub(crate) fn finish(self, tile_ctx: &mut TileDecodeContext, result: Result<()>) -> Result<()> {
        tile_ctx.idwt_line_rings = self.ring;
        tile_ctx.channel_data[self.component_idx].container = self.plane;
        tile_ctx.debug_counters.idwt_output_samples = tile_ctx
            .debug_counters
            .idwt_output_samples
            .saturating_add(self.output_samples);
        result?;
        tile_ctx.idwt_output.rect = self.plan.last_window;
        tile_ctx.idwt_output.placement = IdwtPlacement::ChannelPlane;
        Ok(())
    }
}

/// With no level left to reconstruct, the output is the LL sub-band's window.
fn apply_roi_without_levels(
    storage: &DecompositionStorage<'_>,
    tile_ctx: &mut TileDecodeContext,
    component_idx: usize,
    header: &Header<'_>,
) -> Result<()> {
    let roi_plan = storage
        .roi_plan
        .as_ref()
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    let first_ll_sub_band = storage.tile_decompositions[component_idx].first_ll_sub_band;
    let output = &mut tile_ctx.idwt_output;
    let Some(window) = roi_plan
        .sub_band_window(first_ll_sub_band)
        .or_else(|| roi_plan.final_window(component_idx))
    else {
        output.coefficients.clear();
        output.rect = IntRect::from_xywh(0, 0, 0, 0);
        return Ok(());
    };
    if storage.line_region_idwt {
        return copy_ll_window_into_plane(storage, tile_ctx, component_idx, window, header);
    }
    copy_sub_band_window_to_output(
        &storage.sub_bands[first_ll_sub_band],
        storage,
        window,
        output,
    )
}

/// With no level left to reconstruct, write the LL sub-band's overlap with
/// the component's final window straight into its `channel_data` plane, as
/// the streamed IDWT writes its last level, instead of copying the window into
/// the IDWT output for `store` to copy again.
fn copy_ll_window_into_plane(
    storage: &DecompositionStorage<'_>,
    tile_ctx: &mut TileDecodeContext,
    component_idx: usize,
    window: IntRect,
    header: &Header<'_>,
) -> Result<()> {
    let roi_plan = storage
        .roi_plan
        .as_ref()
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    let (Some(final_window), Some(output_region)) =
        (roi_plan.final_window(component_idx), tile_ctx.output_region)
    else {
        return Ok(());
    };
    let copy = window.intersect(final_window);
    let region = output_region_rect(&header.size_data, output_region);
    let ll_sub_band =
        &storage.sub_bands[storage.tile_decompositions[component_idx].first_ll_sub_band];
    let source = CoefficientSource::from_sub_band(ll_sub_band, storage);
    let plane = &mut tile_ctx.channel_data[component_idx].container;
    let plane_width = region.width() as usize;
    for y in copy.y0..copy.y1 {
        let start = (y - region.y0) as usize * plane_width + (copy.x0 - region.x0) as usize;
        let row = plane
            .get_mut(start..start + copy.width() as usize)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        for (sample, x) in row.iter_mut().zip(copy.x0..copy.x1) {
            *sample = source.get(x, y);
        }
    }
    let output = &mut tile_ctx.idwt_output;
    output.coefficients = Vec::new();
    output.rect = window;
    output.placement = IdwtPlacement::ChannelPlane;
    Ok(())
}

fn record_workspace_now(
    tile_ctx: &mut TileDecodeContext,
    storage: &DecompositionStorage<'_>,
) -> Result<()> {
    let workspace = WorkspaceBytes::sample(tile_ctx, storage)?;
    tile_ctx.debug_counters.workspace_peak.record(workspace);
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
    target.clear();
    let required_len = checked_decode_usize_product2(
        output_window.width() as usize,
        output_window.height() as usize,
    )?;
    try_resize_decode_elements(target, required_len, 0.0)?;
    fill_level_roi(
        ll,
        target,
        output_window,
        decomposition,
        transform,
        storage,
        idwt_output_samples,
    );
    Ok(())
}

/// Apply one level into `target`, which holds exactly `output_window`.
fn fill_level_roi(
    ll: CoefficientSource<'_>,
    target: &mut [f32],
    output_window: IntRect,
    decomposition: &Decomposition,
    transform: WaveletTransform,
    storage: &DecompositionStorage<'_>,
    idwt_output_samples: &mut usize,
) {
    let hl =
        CoefficientSource::from_sub_band(&storage.sub_bands[decomposition.sub_bands[0]], storage);
    let lh =
        CoefficientSource::from_sub_band(&storage.sub_bands[decomposition.sub_bands[1]], storage);
    let hh =
        CoefficientSource::from_sub_band(&storage.sub_bands[decomposition.sub_bands[2]], storage);
    *idwt_output_samples = idwt_output_samples.saturating_add(target.len());

    interleave_samples_roi(ll, hl, lh, hh, target, output_window);
    if output_window.width() > 0 && output_window.height() > 0 {
        filter_horizontal_codestream(target, output_window, transform);
        filter_vertical_codestream(target, output_window, transform);
    }
}

pub(super) fn interleave_samples_roi(
    ll: CoefficientSource<'_>,
    hl: CoefficientSource<'_>,
    lh: CoefficientSource<'_>,
    hh: CoefficientSource<'_>,
    output: &mut [f32],
    output_window: IntRect,
) {
    // Interleaved output coordinate u reads band coordinate u / 2 on each
    // axis: even u selects the low band (ceil(u / 2) == u / 2) and odd u the
    // high band (floor(u / 2) == u / 2). Band coordinates are absolute
    // (Formula B-15), and every source resolves them against its own rect in
    // `get` — including windowed intermediate outputs whose rect origin lies
    // inside the band plane, which must not be mistaken for a band origin.
    let width = output_window.width() as usize;
    for y in output_window.y0..output_window.y1 {
        let low_y = y % 2 == 0;
        let band_y = y / 2;
        for x in output_window.x0..output_window.x1 {
            let low_x = x % 2 == 0;
            let band_x = x / 2;
            let source = match (low_x, low_y) {
                (true, true) => ll,
                (false, true) => hl,
                (true, false) => lh,
                (false, false) => hh,
            };
            let dst = (y - output_window.y0) as usize * width + (x - output_window.x0) as usize;
            output[dst] = source.get(band_x, band_y);
        }
    }
}
