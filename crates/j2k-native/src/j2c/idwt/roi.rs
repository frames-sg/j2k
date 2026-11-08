// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;
use core::mem::size_of;

use super::super::build::{Decomposition, SubBand};
use super::super::codestream::WaveletTransform;
use super::super::decode::{
    DecodeAllocationBudget, DecodeDebugCounters, DecompositionStorage, TileDecodeContext,
    WorkspaceBytes,
};
use super::super::rect::IntRect;
use super::horizontal::{filter_horizontal_codestream, filter_horizontal_i64};
use super::lines::{reconstruct_rows, ring_samples, LineReconstruction, StoredRows, SubBandRows};
use super::model::{CoefficientSource, IdwtPlacement};
use super::vertical::{filter_vertical_codestream, filter_vertical_i64};
use crate::error::{DecodingError, ValidationError};
use crate::j2c::roi::output_region_rect;
use crate::j2c::Header;
use crate::math::{SimdBuffer, SIMD_WIDTH};
use crate::{
    checked_decode_usize_product2, try_resize_decode_elements, try_zero_fill_decode_elements,
    Result,
};

/// Samples of each component's rows a tile streaming from pool bands
/// reconstructs at most at once, before they are transformed and handed over.
const STREAMED_BATCH_SAMPLES: usize = 2 * 1024 * 1024;

/// Batches a tile streaming from pool bands splits its rows into at least,
/// so the two batches it holds at once stay at most half of the rows'
/// samples.
const STREAMED_MIN_BATCHES: usize = 4;

/// Rows a batch of a tile holding one batch keeps at least when it is
/// halved: halving the 116-row batches of an 18000-wide tile cost 2 to 5%.
const STREAMED_DEEP_MIN_ROWS: usize = 128;

/// Rows a batch of a tile holding two batches keeps at most. HTJ2K tiles
/// pipeline two: on 10 to 18 threads, 256-row batches of a 4096-wide tile
/// decode as fast as 512-row ones (0.99-1.01 of their time) and hold two
/// thirds of the memory (48 vs 72 MiB); 128-row ones hold half (36 MiB) but
/// take 3-10% longer, and 1024 x 1024 regions 6-15% longer.
const STREAMED_PIPELINED_MAX_ROWS: usize = 256;

/// Rows of each component a tile streaming from pool bands through a reader
/// per component reconstructs at once, for `height` rows of `width` samples.
/// A tile that holds `one_batch` (a classic tile; HTJ2K tiles pipeline two)
/// transforms it before the next one reconstructs, so its batches take half
/// the rows while that leaves at least `STREAMED_DEEP_MIN_ROWS`: as fast,
/// with half the rows held. Two batches keep to
/// `STREAMED_PIPELINED_MAX_ROWS` each. A tile with one reader holds no
/// batch: it finishes each row in the IDWT's ring.
pub(crate) fn streamed_batch_rows(width: usize, height: usize, one_batch: bool) -> usize {
    let rows = STREAMED_BATCH_SAMPLES / width.max(1);
    let rows = if !one_batch {
        rows.min(STREAMED_PIPELINED_MAX_ROWS)
    } else if rows / 2 >= STREAMED_DEEP_MIN_ROWS {
        rows / 2
    } else {
        rows
    };
    rows.min(height / STREAMED_MIN_BATCHES).clamp(1, 512)
}

/// The most coefficient samples [`apply_roi`] holds at once for one component.
/// Whole-window storage uses the selected sample width; streamed rings use f32.
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
/// plane. When the tile streams its rows to a sink, every component's rings
/// are live at once, the rows finishing in place in them, with one or two
/// batches of rows as wide as each final window from pool bands (see
/// [`streamed_batch_rows`]), so the peak is their sum.
pub(crate) fn roi_idwt_peak_elements(
    storage: &DecompositionStorage<'_>,
    skipped_resolution_levels: u8,
) -> Result<usize> {
    let Some(roi_plan) = storage.roi_plan.as_ref() else {
        return Ok(0);
    };
    let mut peak = 0usize;
    let mut lockstep = 0usize;
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
                .skip_while(|decomposition_idx| {
                    roi_plan.idwt_window(*decomposition_idx).is_none()
                        && storage.decompositions[*decomposition_idx].rect.is_empty()
                })
                .map_while(|decomposition_idx| roi_plan.idwt_window(decomposition_idx));
            let rings = ring_samples(windows, WaveletTransform::Irreversible97)?;
            if storage.lockstep_lines {
                let batch_rows = roi_plan.final_window(component_idx).map_or(0, |window| {
                    let (width, height) = (window.width() as usize, window.height() as usize);
                    if storage.parallel_bands && !storage.one_synthesis_reader {
                        // Two batches, when each component reconstructs
                        // on its own reader, unless the band rings hold
                        // more bands instead; one reader holds none.
                        let batches = if storage.deep_band_rings { 1 } else { 2 };
                        width
                            .saturating_mul(streamed_batch_rows(
                                width,
                                height,
                                storage.deep_band_rings,
                            ))
                            .saturating_mul(batches)
                    } else {
                        0
                    }
                });
                lockstep = lockstep
                    .checked_add(rings)
                    .and_then(|total| total.checked_add(batch_rows))
                    .ok_or(ValidationError::ImageTooLarge)?;
            } else {
                peak = peak.max(rings);
            }
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
    Ok(peak.max(lockstep))
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
    let decompositions = &storage.decompositions[tile_decompositions.decompositions.clone()];
    let active_len = decompositions
        .len()
        .saturating_sub(usize::from(header.skipped_resolution_levels));
    if active_len == 0 {
        return apply_roi_without_levels(storage, tile_ctx, component_idx, header);
    }

    // The previous component's output is no longer used. Drop it before
    // sampling owners and allocating this component's level windows.
    tile_ctx.idwt_output.rect = IntRect::from_xywh(0, 0, 0, 0);
    if storage.exact_integer_decode {
        tile_ctx.idwt_output.coefficients_i64 = Vec::new();
        let workspace = WorkspaceBytes::sample(tile_ctx, storage)?;
        let (coefficients, rect, placement) = reconstruct_window_levels(
            storage,
            &storage.coefficients_i64,
            (component_idx, active_len),
            None,
            (
                &tile_ctx.operation_control,
                &mut tile_ctx.debug_counters,
                workspace,
            ),
            |target, window| {
                filter_horizontal_i64(target, window);
                filter_vertical_i64(target, window);
            },
        )?;
        tile_ctx.idwt_output.coefficients_i64 = coefficients;
        tile_ctx.idwt_output.rect = rect;
        tile_ctx.idwt_output.placement = placement;
    } else {
        tile_ctx.idwt_output.coefficients = Vec::new();
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
        let workspace = WorkspaceBytes::sample(tile_ctx, storage)?;
        let last_window = storage
            .last_idwt_level_in_channels
            .then(|| {
                roi_plan.idwt_window(tile_decompositions.decompositions.start + active_len - 1)
            })
            .flatten();
        let final_target = match last_window {
            Some(window) => Some(
                tile_ctx.channel_data[component_idx]
                    .container
                    .padded_scratch(window_elements(window)?)
                    .ok_or(DecodingError::CodeBlockDecodeFailure)?,
            ),
            None => None,
        };
        let (coefficients, rect, placement) = reconstruct_window_levels(
            storage,
            &storage.coefficients,
            (component_idx, active_len),
            final_target,
            (
                &tile_ctx.operation_control,
                &mut tile_ctx.debug_counters,
                workspace,
            ),
            |target, window| {
                filter_horizontal_codestream(target, window, transform);
                filter_vertical_codestream(target, window, transform);
            },
        )?;
        tile_ctx.idwt_output.coefficients = coefficients;
        tile_ctx.idwt_output.rect = rect;
        tile_ctx.idwt_output.placement = placement;
    }
    Ok(())
}

/// Reconstruct the same planned windows for either sample representation.
/// Only the coefficient storage and existing lifting filters are specialized.
fn reconstruct_window_levels<T: Copy + Default>(
    storage: &DecompositionStorage<'_>,
    coefficients: &[T],
    (component_idx, active_len): (usize, usize),
    mut final_target: Option<&mut [T]>,
    (control, counters, workspace): (
        &crate::DecodeOperationControl,
        &mut DecodeDebugCounters,
        WorkspaceBytes,
    ),
    mut filter: impl FnMut(&mut [T], IntRect),
) -> Result<(Vec<T>, IntRect, IdwtPlacement)> {
    let roi_plan = storage
        .roi_plan
        .as_ref()
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    let tile_decompositions = &storage.tile_decompositions[component_idx];
    let ll_sub_band = &storage.sub_bands[tile_decompositions.first_ll_sub_band];
    let decompositions = &storage.decompositions[tile_decompositions.decompositions.clone()];
    let mut current_coefficients = Vec::new();
    let mut current_rect = IntRect::from_xywh(0, 0, 0, 0);
    let mut have_current = false;
    for (local_idx, decomposition) in decompositions.iter().take(active_len).enumerate() {
        control.check()?;
        let decomposition_idx = tile_decompositions.decompositions.start + local_idx;
        let Some(output_window) = roi_plan.idwt_window(decomposition_idx) else {
            if !have_current && decomposition.rect.is_empty() {
                continue;
            }
            return Ok((
                Vec::new(),
                IntRect::from_xywh(0, 0, 0, 0),
                IdwtPlacement::Coefficients,
            ));
        };
        let ll_input = if have_current {
            CoefficientSource::new(&current_coefficients, current_rect, current_rect.width())
        } else {
            CoefficientSource::from_sub_band(ll_sub_band, coefficients)
        };
        if local_idx + 1 == active_len {
            if let Some(target) = final_target.take() {
                fill_level_roi(
                    ll_input,
                    target,
                    output_window,
                    decomposition,
                    (storage, coefficients),
                    &mut counters.idwt_output_samples,
                    &mut filter,
                );
                counters
                    .workspace_peak
                    .record(workspace.with_idwt(current_coefficients.capacity() * size_of::<T>()));
                return Ok((Vec::new(), output_window, IdwtPlacement::ChannelPadding));
            }
        }
        let mut next = Vec::new();
        apply_level_roi(
            ll_input,
            (&mut next, current_coefficients.capacity()),
            output_window,
            decomposition,
            (storage, coefficients),
            &mut counters.idwt_output_samples,
            &mut filter,
        )?;
        let window_bytes = (current_coefficients.capacity() + next.capacity()) * size_of::<T>();
        counters
            .workspace_peak
            .record(workspace.with_idwt(window_bytes));
        current_coefficients = next;
        current_rect = output_window;
        have_current = true;
    }
    Ok((
        current_coefficients,
        current_rect,
        IdwtPlacement::Coefficients,
    ))
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
    control: crate::DecodeOperationControl,
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
                if windows.is_empty() && storage.decompositions[decomposition_idx].rect.is_empty() {
                    // A coarse level with no samples: the next level's LL input is empty.
                    continue;
                }
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
            try_zero_fill_decode_elements(&mut ring, ring_len)?;
        }
        tile_ctx.idwt_line_rings = ring;
        record_workspace_now(tile_ctx, storage)?;
        let ring = core::mem::take(&mut tile_ctx.idwt_line_rings);
        let plane = core::mem::replace(
            &mut tile_ctx.channel_data[component_idx].container,
            SimdBuffer::empty(),
        );
        Ok(Some(Self {
            control: tile_ctx.operation_control.clone(),
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
            self.control.check()?;
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
        if storage.exact_integer_decode {
            output.coefficients_i64.clear();
        } else {
            output.coefficients.clear();
        }
        output.rect = IntRect::from_xywh(0, 0, 0, 0);
        return Ok(());
    };
    if !storage.exact_integer_decode && storage.line_region_idwt {
        return copy_ll_window_into_plane(storage, tile_ctx, component_idx, window, header);
    }
    let sub_band = &storage.sub_bands[first_ll_sub_band];
    if storage.exact_integer_decode {
        copy_sub_band_window_to_output(
            sub_band,
            (storage, &storage.coefficients_i64),
            window,
            &mut output.coefficients_i64,
            &tile_ctx.operation_control,
        )?;
    } else {
        copy_sub_band_window_to_output(
            sub_band,
            (storage, &storage.coefficients),
            window,
            &mut output.coefficients,
            &tile_ctx.operation_control,
        )?;
    }
    output.rect = window;
    Ok(())
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
    let source = CoefficientSource::from_sub_band(ll_sub_band, &storage.coefficients);
    let plane = &mut tile_ctx.channel_data[component_idx].container;
    let plane_width = region.width() as usize;
    for y in copy.y0..copy.y1 {
        tile_ctx.operation_control.check()?;
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

fn copy_sub_band_window_to_output<T: Copy + Default>(
    sub_band: &SubBand,
    (storage, coefficients): (&DecompositionStorage<'_>, &[T]),
    window: IntRect,
    output: &mut Vec<T>,
    control: &crate::DecodeOperationControl,
) -> Result<()> {
    output.clear();
    let required_len =
        checked_decode_usize_product2(window.width() as usize, window.height() as usize)?;
    let grows = required_len > output.capacity();
    if grows {
        // Previous component samples are dead; drop them before replacement.
        *output = Vec::new();
    }
    let mut budget = DecodeAllocationBudget::for_storage(storage)?;
    try_resize_decode_elements(output, required_len, T::default())?;
    if grows {
        budget
            .include_capacity_overage::<T>(storage.roi_idwt_charge_elements, output.capacity())?;
    }
    let source = CoefficientSource::from_sub_band(sub_band, coefficients);
    for y in window.y0..window.y1 {
        control.check()?;
        for x in window.x0..window.x1 {
            let dst = (y - window.y0) as usize * window.width() as usize + (x - window.x0) as usize;
            output[dst] = source.get(x, y);
        }
    }
    Ok(())
}

fn apply_level_roi<T: Copy + Default>(
    ll: CoefficientSource<'_, T>,
    (target, previous_capacity): (&mut Vec<T>, usize),
    output_window: IntRect,
    decomposition: &Decomposition,
    sources: (&DecompositionStorage<'_>, &[T]),
    idwt_output_samples: &mut usize,
    filter: &mut impl FnMut(&mut [T], IntRect),
) -> Result<()> {
    target.clear();
    let required_len = checked_decode_usize_product2(
        output_window.width() as usize,
        output_window.height() as usize,
    )?;
    let budget = DecodeAllocationBudget::for_storage(sources.0)?;
    let mut planned = budget;
    planned.include_capacity_overage::<T>(
        sources.0.roi_idwt_charge_elements,
        previous_capacity
            .checked_add(required_len)
            .ok_or(ValidationError::ImageTooLarge)?,
    )?;
    try_resize_decode_elements(target, required_len, T::default())?;
    let mut actual = budget;
    actual.include_capacity_overage::<T>(
        sources.0.roi_idwt_charge_elements,
        previous_capacity
            .checked_add(target.capacity())
            .ok_or(ValidationError::ImageTooLarge)?,
    )?;
    fill_level_roi(
        ll,
        target,
        output_window,
        decomposition,
        sources,
        idwt_output_samples,
        filter,
    );
    Ok(())
}

/// Apply one level into `target`, which holds exactly `output_window`.
fn fill_level_roi<T: Copy + Default>(
    ll: CoefficientSource<'_, T>,
    target: &mut [T],
    output_window: IntRect,
    decomposition: &Decomposition,
    (storage, coefficients): (&DecompositionStorage<'_>, &[T]),
    idwt_output_samples: &mut usize,
    filter: &mut impl FnMut(&mut [T], IntRect),
) {
    let hl = CoefficientSource::from_sub_band(
        &storage.sub_bands[decomposition.sub_bands[0]],
        coefficients,
    );
    let lh = CoefficientSource::from_sub_band(
        &storage.sub_bands[decomposition.sub_bands[1]],
        coefficients,
    );
    let hh = CoefficientSource::from_sub_band(
        &storage.sub_bands[decomposition.sub_bands[2]],
        coefficients,
    );
    *idwt_output_samples = idwt_output_samples.saturating_add(target.len());

    interleave_samples_roi(ll, hl, lh, hh, target, output_window);
    if output_window.width() > 0 && output_window.height() > 0 {
        filter(target, output_window);
    }
}

pub(super) fn interleave_samples_roi<T: Copy + Default>(
    ll: CoefficientSource<'_, T>,
    hl: CoefficientSource<'_, T>,
    lh: CoefficientSource<'_, T>,
    hh: CoefficientSource<'_, T>,
    output: &mut [T],
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
