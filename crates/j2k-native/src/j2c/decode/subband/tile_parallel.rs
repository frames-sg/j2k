// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tile-wide HTJ2K code-block decode with in-place output.
//!
//! The per-sub-band path joins once per sub-band, copies every code-block
//! payload, decodes into a slab, and scatters the slab serially. Here every
//! row of code blocks in every sub-band of the tile is one task. A row of code
//! blocks owns a contiguous range of its sub-band's coefficients, so tasks
//! write their output in place through disjoint slices and Rayon balances
//! them with work stealing. Cleanup-only payloads are borrowed from the
//! codestream; only payloads that span several segments are gathered.

use alloc::vec::Vec;
use core::ops::Range;
use core::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use rayon::prelude::*;

use super::{
    add_roi_shift_to_bitplanes, code_block_required_by_index, ht_code_block_has_decodable_passes,
    sub_band_decode_parameters, CpuDecodeParallelism, DecodeAllocationBudget, DecompositionStorage,
    Header, HtCodeBlockDecodeJob, HtCodeBlockDecoder, Result, Tile, TileDecodeContext,
};
use crate::error::{DecodingError, ValidationError};
use crate::j2c::build::{CodeBlock, CodeBlockCoding};
use crate::j2c::decode::workspace::{ht_task_workspace_bytes, HtTaskWorkspace};
use crate::j2c::ht_block_decode::{ht_decode_workspace_bytes, selected_code_block_segment_lengths};
use crate::scalar::decode_ht_code_block_scalar_with_workspace_midpoint;
use crate::{decode_ht_code_block_scalar_with_workspace, HtCodeBlockDecodeWorkspace};

/// Below this many coded bytes per tile (about 45 us of serial entropy decode
/// on an Apple M4 Pro core) a Rayon fork-join costs more than it saves, so
/// the tile decodes serially. Unit tests use tiny fixtures and take the
/// parallel route unconditionally; `tile_route_threshold` covers the cutoff.
#[cfg(not(test))]
const MIN_PARALLEL_CODED_BYTES: usize = 8 * 1024;
#[cfg(test)]
const MIN_PARALLEL_CODED_BYTES: usize = 0;

/// Whether a tile with `strips` rows of code blocks and `coded_bytes` of
/// payload is worth decoding in parallel.
fn worth_parallel_decode(strips: usize, coded_bytes: usize, min_coded_bytes: usize) -> bool {
    strips >= 2 && coded_bytes >= min_coded_bytes
}

#[derive(Clone, Copy)]
struct BandParameters {
    num_bitplanes: u8,
    roi_shift: u8,
    stripe_causal: bool,
    dequantization_step: f32,
    irreversible_midpoint: bool,
}

#[derive(Clone, Copy)]
enum Payload<'a> {
    /// A single cleanup segment borrowed from the codestream.
    Borrowed(&'a [u8]),
    /// Gathered segments starting at this offset of the payload arena.
    Gathered(usize),
}

struct BlockJob<'a> {
    payload: Payload<'a>,
    code_block: usize,
    cleanup_length: u32,
    refinement_length: u32,
    x_offset: u32,
    y_offset: u32,
    width: u32,
    height: u32,
    missing_bit_planes: u8,
    number_of_coding_passes: u8,
    band: usize,
}

struct Strip {
    /// First coefficient of the strip in tile coefficient storage.
    start: usize,
    len: usize,
    stride: usize,
    jobs: Range<usize>,
    work: usize,
}

struct TilePlan<'a> {
    bands: Vec<BandParameters>,
    jobs: Vec<BlockJob<'a>>,
    strips: Vec<Strip>,
    arena: Vec<u8>,
    skipped: usize,
    coded_bytes: usize,
    max_width: u32,
    max_height: u32,
    strict: bool,
}

/// How [`plan_tile_ht_route`] wants the tile's code blocks decoded.
pub(in crate::j2c::decode) enum TileHtRoute {
    /// Every HT code block of the tile was decoded in place.
    Decoded,
    /// Too little work to fork: decode the tile serially.
    Serial,
    /// The tile needs the general per-sub-band path.
    PerSubBand,
}

/// Decode every HT code block of the tile in parallel when all of its
/// components use HT block coding only and the tile carries enough coded
/// data to amortize a fork-join. Returns without side effects for the
/// non-`Decoded` routes.
#[expect(
    clippy::too_many_arguments,
    reason = "routing sees the tile, component range, state buffers, adapter, and options explicitly"
)]
pub(in crate::j2c::decode) fn plan_tile_ht_route<'a>(
    tile: &Tile<'a>,
    components: Range<usize>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    header: &Header<'_>,
    ht_decoder: &Option<&mut dyn HtCodeBlockDecoder>,
    cpu_decode_parallelism: CpuDecodeParallelism,
    profile_enabled: bool,
) -> Result<TileHtRoute> {
    #[cfg(test)]
    if tile_ctx.force_per_sub_band_ht {
        return Ok(TileHtRoute::PerSubBand);
    }
    // One worker cannot overlap entropy jobs; the serial path keeps its own
    // context and allocates no parallel workspace bank.
    if cpu_decode_parallelism != CpuDecodeParallelism::Auto
        || rayon::current_num_threads() < 2
        || profile_enabled
        || ht_decoder.is_some()
        || storage.exact_integer_decode
        || !components_use_ht_only(tile, components.clone())
    {
        return Ok(TileHtRoute::PerSubBand);
    }

    let mut budget = DecodeAllocationBudget::for_storage(storage)?;
    let plan = match plan_tile(tile, components, storage, header, &mut budget) {
        Ok(Some(plan)) => plan,
        Ok(None) => return Ok(TileHtRoute::PerSubBand),
        // Planning metadata did not fit the decode budget; the per-sub-band
        // path needs less and reports its own limits.
        Err(error) if crate::j2c::decode::reuse::is_capacity_error(&error) => {
            return Ok(TileHtRoute::PerSubBand);
        }
        Err(error) => return Err(error),
    };
    if !worth_parallel_decode(
        plan.strips.len(),
        plan.coded_bytes,
        MIN_PARALLEL_CODED_BYTES,
    ) {
        return Ok(TileHtRoute::Serial);
    }

    let workers = rayon::current_num_threads().min(plan.strips.len());
    let Ok(growths) = prepare_uniform_task_workspaces(
        &mut tile_ctx.ht_task_workspaces,
        workers,
        plan.max_width,
        plan.max_height,
        &mut storage.structural_workspace_bytes,
        &mut budget,
    ) else {
        // The per-sub-band path has its own smaller-footprint fallbacks.
        return Ok(TileHtRoute::PerSubBand);
    };

    let mut coefficients = core::mem::take(&mut storage.coefficients);
    let result = decode_strips(&plan, &mut coefficients, tile_ctx, &mut budget);
    storage.coefficients = coefficients;
    result?;

    let counters = &mut tile_ctx.debug_counters;
    counters.decoded_code_blocks += plan.jobs.len();
    counters.skipped_code_blocks += plan.skipped;
    let stats = &mut counters.ht_phase_stats;
    for job in &plan.jobs {
        stats.blocks += 1;
        stats.cleanup_bytes += u128::from(job.cleanup_length);
        if job.refinement_length > 0 {
            stats.refinement_blocks += 1;
            stats.refinement_bytes += u128::from(job.refinement_length);
        }
    }
    #[cfg(test)]
    {
        counters.ht_parallel_tasks = counters.ht_parallel_tasks.max(workers);
        counters.ht_task_workspace_growths += growths;
    }
    #[cfg(not(test))]
    let _ = growths;
    Ok(TileHtRoute::Decoded)
}

fn components_use_ht_only(tile: &Tile<'_>, components: Range<usize>) -> bool {
    let Some(component_infos) = tile.component_infos.get(components) else {
        return false;
    };
    component_infos.iter().all(|component_info| {
        let style = component_info.coding_style.parameters.code_block_style;
        style.uses_high_throughput_block_coding() && !style.allows_mixed_block_coding()
    })
}

#[expect(
    clippy::too_many_lines,
    reason = "planning walks components, resolutions, sub-bands, and code blocks in one ordered pass"
)]
fn plan_tile<'a>(
    tile: &Tile<'a>,
    components: Range<usize>,
    storage: &DecompositionStorage<'a>,
    header: &Header<'_>,
    budget: &mut DecodeAllocationBudget,
) -> Result<Option<TilePlan<'a>>> {
    let mut plan = TilePlan {
        bands: Vec::new(),
        jobs: Vec::new(),
        strips: Vec::new(),
        arena: Vec::new(),
        skipped: 0,
        coded_bytes: 0,
        max_width: 0,
        max_height: 0,
        strict: header.strict,
    };
    budget.reserve_new(&mut plan.jobs, storage.code_blocks.len())?;
    budget.reserve_new(&mut plan.strips, storage.code_blocks.len())?;
    budget.reserve_new(&mut plan.bands, storage.sub_bands.len())?;
    let mut arena_len = 0usize;

    let component_infos = tile
        .component_infos
        .get(components.clone())
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    for (component_idx, component_info) in components.zip(component_infos) {
        let stripe_causal = component_info
            .coding_style
            .parameters
            .code_block_style
            .vertically_causal_context;
        let resolutions = component_info
            .num_resolution_levels()
            .checked_sub(header.skipped_resolution_levels)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        for resolution in 0..resolutions {
            let tile_composition = &storage.tile_decompositions[component_idx];
            for sub_band_idx in tile_composition.sub_band_iter(resolution, &storage.decompositions)
            {
                let sub_band = &storage.sub_bands[sub_band_idx];
                let parameters = sub_band_decode_parameters(sub_band, resolution, component_info)?;
                let coded_bitplanes = add_roi_shift_to_bitplanes(
                    parameters.num_bitplanes,
                    component_info.roi_shift,
                    31,
                )?;
                let band = plan.bands.len();
                plan.bands.push(BandParameters {
                    num_bitplanes: parameters.num_bitplanes,
                    roi_shift: component_info.roi_shift,
                    stripe_causal,
                    dequantization_step: parameters.dequantization_step,
                    irreversible_midpoint: parameters.irreversible_midpoint,
                });

                let first_job = plan.jobs.len();
                for precinct in sub_band
                    .precincts
                    .clone()
                    .map(|idx| &storage.precincts[idx])
                {
                    for code_block_idx in precinct.code_blocks.clone() {
                        let code_block = &storage.code_blocks[code_block_idx];
                        if code_block.coding != Some(CodeBlockCoding::HighThroughput) {
                            continue;
                        }
                        if !code_block_required_by_index(storage, sub_band_idx, code_block) {
                            plan.skipped += 1;
                            continue;
                        }
                        if !ht_code_block_has_decodable_passes(
                            code_block,
                            coded_bitplanes,
                            header.strict,
                        )? {
                            continue;
                        }
                        let job = plan_block(
                            code_block,
                            code_block_idx,
                            sub_band.storage_rect.x0,
                            sub_band.storage_rect.y0,
                            band,
                            storage,
                            &mut arena_len,
                        )?;
                        plan.coded_bytes = plan
                            .coded_bytes
                            .saturating_add(job.cleanup_length as usize)
                            .saturating_add(job.refinement_length as usize);
                        plan.max_width = plan.max_width.max(job.width);
                        plan.max_height = plan.max_height.max(job.height);
                        if plan.jobs.len() == plan.jobs.capacity() {
                            return Err(DecodingError::CodeBlockDecodeFailure.into());
                        }
                        plan.jobs.push(job);
                    }
                }
                if !plan_band_strips(
                    &mut plan,
                    first_job,
                    sub_band.coefficients.clone(),
                    sub_band.storage_rect.width(),
                    sub_band.storage_rect.height(),
                )? {
                    return Ok(None);
                }
            }
        }
    }

    if !strips_are_disjoint(&mut plan.strips, storage.coefficients.len()) {
        return Ok(None);
    }
    gather_payloads(&mut plan, storage, arena_len, budget)?;
    budget.include_elements::<(&mut [f32], &Strip)>(plan.strips.len())?;
    Ok(Some(plan))
}

#[expect(
    clippy::similar_names,
    reason = "x0 and y0 name the sub-band origin coordinates"
)]
fn plan_block<'a>(
    code_block: &CodeBlock,
    code_block_idx: usize,
    sub_band_x0: u32,
    sub_band_y0: u32,
    band: usize,
    storage: &DecompositionStorage<'a>,
    arena_len: &mut usize,
) -> Result<BlockJob<'a>> {
    let (cleanup_length, refinement_length) =
        selected_code_block_segment_lengths(code_block, storage)?;
    let payload = match single_cleanup_segment(code_block, storage)? {
        Some(cleanup) if refinement_length == 0 => Payload::Borrowed(cleanup),
        _ => {
            let start = *arena_len;
            *arena_len = start
                .checked_add(cleanup_length)
                .and_then(|end| end.checked_add(refinement_length))
                .ok_or(ValidationError::ImageTooLarge)?;
            Payload::Gathered(start)
        }
    };
    Ok(BlockJob {
        payload,
        code_block: code_block_idx,
        cleanup_length: u32::try_from(cleanup_length)
            .map_err(|_| DecodingError::CodeBlockDecodeFailure)?,
        refinement_length: u32::try_from(refinement_length)
            .map_err(|_| DecodingError::CodeBlockDecodeFailure)?,
        x_offset: code_block.rect.x0 - sub_band_x0,
        y_offset: code_block.rect.y0 - sub_band_y0,
        width: code_block.rect.width(),
        height: code_block.rect.height(),
        missing_bit_planes: code_block.missing_bit_planes,
        number_of_coding_passes: code_block.number_of_coding_passes,
        band,
    })
}

/// The block's cleanup segment when it is the only selected segment.
fn single_cleanup_segment<'a>(
    code_block: &CodeBlock,
    storage: &DecompositionStorage<'a>,
) -> Result<Option<&'a [u8]>> {
    let selected_set = code_block
        .ht_selected_set
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    let cleanup_index = selected_set
        .checked_mul(2)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    let mut cleanup = None;
    for layer in &storage.layers[code_block.layers.clone()] {
        let Some(range) = layer.segments.clone() else {
            continue;
        };
        for segment in &storage.segments[range] {
            if segment.idx < cleanup_index {
                continue;
            }
            if segment.idx != cleanup_index || cleanup.is_some() {
                return Ok(None);
            }
            cleanup = Some(segment.data);
        }
    }
    Ok(cleanup)
}

/// Group one sub-band's jobs into rows of code blocks. Returns `false` when
/// the blocks do not tile the sub-band in rows of equal height.
fn plan_band_strips(
    plan: &mut TilePlan<'_>,
    first_job: usize,
    coefficients: Range<usize>,
    band_width: u32,
    band_height: u32,
) -> Result<bool> {
    let band_jobs = &mut plan.jobs[first_job..];
    band_jobs.sort_unstable_by_key(|job| (job.y_offset, job.x_offset));
    let stride = band_width as usize;
    let mut index = first_job;
    while index < plan.jobs.len() {
        let first = &plan.jobs[index];
        let (y_offset, height) = (first.y_offset, first.height);
        let mut end = index;
        let mut work = 0usize;
        let mut next_x = 0u32;
        while end < plan.jobs.len() && plan.jobs[end].y_offset == y_offset {
            let job = &plan.jobs[end];
            let fits = job.height == height
                && job.x_offset >= next_x
                && job
                    .x_offset
                    .checked_add(job.width)
                    .is_some_and(|right| right <= band_width);
            if !fits {
                return Ok(false);
            }
            next_x = job.x_offset + job.width;
            work = work
                .saturating_add(job.cleanup_length as usize)
                .saturating_add(job.refinement_length as usize)
                .saturating_add((job.width as usize * job.height as usize) / 8);
            end += 1;
        }
        if y_offset
            .checked_add(height)
            .is_none_or(|bottom| bottom > band_height)
        {
            return Ok(false);
        }
        let start = (y_offset as usize)
            .checked_mul(stride)
            .and_then(|offset| offset.checked_add(coefficients.start))
            .ok_or(ValidationError::ImageTooLarge)?;
        let len = (height as usize)
            .checked_mul(stride)
            .ok_or(ValidationError::ImageTooLarge)?;
        if start
            .checked_add(len)
            .is_none_or(|strip_end| strip_end > coefficients.end)
        {
            return Ok(false);
        }
        if plan.strips.len() == plan.strips.capacity() {
            return Err(DecodingError::CodeBlockDecodeFailure.into());
        }
        plan.strips.push(Strip {
            start,
            len,
            stride,
            jobs: index..end,
            work,
        });
        index = end;
    }
    Ok(true)
}

fn strips_are_disjoint(strips: &mut [Strip], coefficient_len: usize) -> bool {
    strips.sort_unstable_by_key(|strip| strip.start);
    let mut previous_end = 0usize;
    for strip in strips.iter() {
        if strip.start < previous_end {
            return false;
        }
        previous_end = strip.start + strip.len;
    }
    previous_end <= coefficient_len
}

fn gather_payloads<'a>(
    plan: &mut TilePlan<'a>,
    storage: &DecompositionStorage<'a>,
    arena_len: usize,
    budget: &mut DecodeAllocationBudget,
) -> Result<()> {
    if arena_len == 0 {
        return Ok(());
    }
    budget.reserve_new(&mut plan.arena, arena_len)?;
    for job in &mut plan.jobs {
        let Payload::Gathered(_) = job.payload else {
            continue;
        };
        // Row sorting can change precinct traversal order. Assign the offset
        // when the payload is appended in its final execution order.
        let start = plan.arena.len();
        job.payload = Payload::Gathered(start);
        let code_block = &storage.code_blocks[job.code_block];
        let selected_set = code_block
            .ht_selected_set
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        let cleanup_index = selected_set
            .checked_mul(2)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        for layer in &storage.layers[code_block.layers.clone()] {
            let Some(range) = layer.segments.clone() else {
                continue;
            };
            for segment in &storage.segments[range] {
                if segment.idx == cleanup_index || segment.idx == cleanup_index + 1 {
                    plan.arena.extend_from_slice(segment.data);
                }
            }
        }
        let expected = job.cleanup_length as usize + job.refinement_length as usize;
        if plan.arena.len() - start != expected {
            return Err(DecodingError::CodeBlockDecodeFailure.into());
        }
    }
    Ok(())
}

fn charge_actual_workspace(
    budget: &mut DecodeAllocationBudget,
    planned: usize,
    actual: usize,
) -> Result<()> {
    if actual > planned {
        budget.include_bytes(actual - planned)
    } else {
        budget.release_bytes(planned - actual)
    }
}

/// Make the first `count` retained workspaces fit `width` x `height` blocks,
/// replacing the bank in one budgeted transaction when any slot is short.
fn prepare_uniform_task_workspaces(
    workspaces: &mut Vec<HtTaskWorkspace>,
    count: usize,
    width: u32,
    height: u32,
    structural_workspace_bytes: &mut usize,
    budget: &mut DecodeAllocationBudget,
) -> Result<usize> {
    if workspaces.len() >= count
        && workspaces[..count]
            .iter()
            .all(|slot| slot.prepared_width >= width && slot.prepared_height >= height)
    {
        return Ok(0);
    }
    let old_bank_bytes = ht_task_workspace_bytes(workspaces)?;
    let mut trial = *budget;
    let mut replacement = Vec::new();
    let count = count.max(workspaces.len());
    trial.reserve_new(&mut replacement, count)?;
    let mut growths = 0usize;
    for index in 0..count {
        let (slot_width, slot_height) = workspaces.get(index).map_or((width, height), |old| {
            (
                width.max(old.prepared_width),
                height.max(old.prepared_height),
            )
        });
        let planned = ht_decode_workspace_bytes(slot_width, slot_height)?;
        trial.include_bytes(planned)?;
        let mut workspace = HtCodeBlockDecodeWorkspace::default();
        workspace.reserve(slot_width, slot_height)?;
        charge_actual_workspace(&mut trial, planned, workspace.allocated_bytes()?)?;
        let grew = workspaces
            .get(index)
            .map_or(Ok(true), |old| -> Result<bool> {
                Ok(workspace.allocated_bytes()? > old.workspace.allocated_bytes()?)
            })?;
        growths += usize::from(grew);
        replacement.push(HtTaskWorkspace {
            workspace,
            prepared_width: slot_width,
            prepared_height: slot_height,
        });
    }
    let new_bank_bytes = ht_task_workspace_bytes(&replacement)?;
    trial.release_bytes(old_bank_bytes)?;
    let structural = structural_workspace_bytes
        .checked_sub(old_bank_bytes)
        .and_then(|bytes| bytes.checked_add(new_bank_bytes))
        .ok_or(ValidationError::ImageTooLarge)?;
    *workspaces = replacement;
    *structural_workspace_bytes = structural;
    *budget = trial;
    Ok(growths)
}

fn decode_strips(
    plan: &TilePlan<'_>,
    coefficients: &mut [f32],
    tile_ctx: &mut TileDecodeContext,
    budget: &mut DecodeAllocationBudget,
) -> Result<()> {
    // Strips are sorted by start offset, so carving them front to back yields
    // disjoint mutable outputs.
    let mut tasks = Vec::new();
    tasks
        .try_reserve_exact(plan.strips.len())
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    budget.include_capacity_overage::<(&mut [f32], &Strip)>(plan.strips.len(), tasks.capacity())?;
    let mut rest = coefficients;
    let mut consumed = 0usize;
    for strip in &plan.strips {
        let tail = core::mem::take(&mut rest);
        let (_, tail) = tail.split_at_mut(strip.start - consumed);
        let (output, tail) = tail.split_at_mut(strip.len);
        rest = tail;
        consumed = strip.start + strip.len;
        tasks.push((output, strip));
    }

    // Largest rows first so stragglers are small.
    tasks.sort_unstable_by_key(|(_, strip)| core::cmp::Reverse(strip.work));
    let workers = rayon::current_num_threads().min(tasks.len());
    let queue = Mutex::new(tasks.into_iter());
    let failed = AtomicBool::new(false);
    // Each worker owns one already-budgeted workspace for the whole tile.
    // A single queue pop per strip keeps work balanced across unequal cores
    // without forking a Rayon job and borrowing/returning scratch for each row.
    tile_ctx.ht_task_workspaces[..workers]
        .par_iter_mut()
        .with_max_len(1)
        .try_for_each(|slot| -> Result<()> {
            while !failed.load(Ordering::Relaxed) {
                let next = queue
                    .lock()
                    .map_err(|_| DecodingError::CodeBlockDecodeFailure)?
                    .next();
                let Some((output, strip)) = next else {
                    break;
                };
                if let Err(error) = decode_strip(plan, strip, output, &mut slot.workspace) {
                    failed.store(true, Ordering::Relaxed);
                    return Err(error);
                }
            }
            Ok(())
        })
}

fn decode_strip(
    plan: &TilePlan<'_>,
    strip: &Strip,
    output: &mut [f32],
    workspace: &mut HtCodeBlockDecodeWorkspace,
) -> Result<()> {
    for job in &plan.jobs[strip.jobs.clone()] {
        let band = plan.bands[job.band];
        let total = job.cleanup_length as usize + job.refinement_length as usize;
        let data = match job.payload {
            Payload::Borrowed(data) => data,
            Payload::Gathered(start) => plan
                .arena
                .get(start..start + total)
                .ok_or(DecodingError::CodeBlockDecodeFailure)?,
        };
        let decode_job = HtCodeBlockDecodeJob {
            data,
            cleanup_length: job.cleanup_length,
            refinement_length: job.refinement_length,
            width: job.width,
            height: job.height,
            output_stride: strip.stride,
            missing_bit_planes: job.missing_bit_planes,
            number_of_coding_passes: job.number_of_coding_passes,
            num_bitplanes: band.num_bitplanes,
            roi_shift: band.roi_shift,
            stripe_causal: band.stripe_causal,
            strict: plan.strict,
            dequantization_step: band.dequantization_step,
        };
        let output = &mut output[job.x_offset as usize..];
        #[cfg(feature = "simd")]
        if crate::scalar::try_decode_ht_code_block_to_f32(
            decode_job,
            output,
            workspace,
            band.irreversible_midpoint,
        )? {
            continue;
        }
        if band.irreversible_midpoint {
            decode_ht_code_block_scalar_with_workspace_midpoint(decode_job, output, workspace)?;
        } else {
            decode_ht_code_block_scalar_with_workspace(decode_job, output, workspace)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::worth_parallel_decode;

    #[test]
    fn tile_route_threshold() {
        assert!(!worth_parallel_decode(1, usize::MAX, 8192));
        assert!(!worth_parallel_decode(16, 8191, 8192));
        assert!(worth_parallel_decode(2, 8192, 8192));
    }
}
