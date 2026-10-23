// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tile-wide code-block decode with in-place output.
//!
//! The per-sub-band path joins once per sub-band, copies every code-block
//! payload, decodes into a slab, and scatters the slab serially. Here every
//! row of code blocks in every sub-band of the tile is one task. A row of code
//! blocks owns a contiguous range of its sub-band's coefficients, so tasks
//! write their output in place through disjoint slices and Rayon balances
//! them with work stealing. HT cleanup-only and single-segment classic
//! payloads are borrowed from the codestream; only payloads that span several
//! segments are gathered. Classic blocks take this route when each decodes as
//! one arithmetic segment: no selective bypass or per-pass termination.

use alloc::vec::Vec;
use core::ops::Range;
use core::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use rayon::prelude::*;

use super::{
    add_roi_shift_to_bitplanes, classic_decode_job_parameters, code_block_required_by_index,
    ht_code_block_has_decodable_passes, stored_placement, sub_band_decode_parameters,
    BlockPlacement, ClassicJobParameters, CpuDecodeParallelism, DecodeAllocationBudget,
    DecompositionStorage, Header, HtCodeBlockDecodeJob, HtCodeBlockDecoder,
    J2kCodeBlockDecodeWorkspace, Result, Tile, TileDecodeContext, MAX_BITPLANE_COUNT,
};
use crate::error::{DecodingError, ValidationError};
use crate::j2c::build::{CodeBlock, CodeBlockCoding};
use crate::j2c::codestream::CodeBlockStyle;
use crate::j2c::decode::direct_plan::classic_allocation_counts;
use crate::j2c::decode::peak::WorkspaceBytes;
use crate::j2c::decode::workspace::{task_workspace_bytes, TaskScratch, TaskWorkspace};
use crate::j2c::ht_block_decode::selected_code_block_segment_lengths;
use crate::j2c::tile::{ComponentTile, ResolutionTile};
use crate::scalar::decode_ht_code_block_scalar_with_workspace_midpoint;
use crate::{
    decode_ht_code_block_scalar_with_workspace, try_resize_decode_elements,
    HtCodeBlockDecodeWorkspace, J2kCodeBlockSegment,
};

/// Below this many coded bytes per tile (about 45 us of serial entropy decode
/// on an Apple M4 Pro core) a Rayon fork-join costs more than it saves, so
/// the tile decodes serially. Classic MQ decoding costs far more per byte, so
/// classic tiles fork from an eighth of this: on an M4 at 4 threads, routed
/// classic tiles of 1.6 to 8 KiB decoded 4% to 50% faster than serially, and
/// tiles under 1 KiB per component broke even. Unit tests use tiny fixtures
/// and take the parallel route unconditionally; `tile_route` covers the cutoffs.
#[cfg(not(test))]
const MIN_PARALLEL_CODED_BYTES: usize = 8 * 1024;
#[cfg(test)]
const MIN_PARALLEL_CODED_BYTES: usize = 0;

/// Where a tile of `coding` blocks with `strips` rows of code blocks and
/// `coded_bytes` of payload decodes on `threads` workers: on the route
/// (`None`), serially, or per sub-band. A classic tile with fewer rows than
/// workers leaves some idle, while the per-sub-band path splits each row's
/// blocks among them.
fn route_fallback(
    coding: CodeBlockCoding,
    strips: usize,
    coded_bytes: usize,
    threads: usize,
    min_coded_bytes: usize,
) -> Option<TileRoute> {
    let classic = coding == CodeBlockCoding::Classic;
    if too_few_coded_bytes(coding, coded_bytes, min_coded_bytes) {
        Some(TileRoute::Serial)
    } else if classic && strips < threads {
        Some(TileRoute::PerSubBand)
    } else if strips < 2 {
        Some(TileRoute::Serial)
    } else {
        None
    }
}

/// Whether `coded_bytes` of `coding` blocks are too few to fork for.
fn too_few_coded_bytes(
    coding: CodeBlockCoding,
    coded_bytes: usize,
    min_coded_bytes: usize,
) -> bool {
    match coding {
        CodeBlockCoding::Classic => coded_bytes < min_coded_bytes / 8,
        CodeBlockCoding::HighThroughput => coded_bytes < min_coded_bytes,
    }
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
    /// A single segment borrowed from the codestream.
    Borrowed(&'a [u8]),
    /// Gathered segments starting at this offset of the payload arena.
    Gathered(usize),
}

struct BlockJob<'a> {
    payload: Payload<'a>,
    code_block: usize,
    /// Bytes of the cleanup and refinement segments of an HT block, or of the
    /// arithmetic segment of a classic one.
    payload_length: u32,
    /// Bytes at the end of the payload that are an HT refinement segment.
    refinement_length: u32,
    /// The samples of the block that its sub-band stores.
    placement: BlockPlacement,
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
    coding: CodeBlockCoding,
    bands: Vec<BandParameters>,
    classic_bands: Vec<ClassicJobParameters>,
    strict: bool,
    jobs: Vec<BlockJob<'a>>,
    strips: Vec<Strip>,
    arena: Vec<u8>,
    /// Bytes of the payloads to gather into `arena` once the route is taken.
    arena_len: usize,
    skipped: usize,
    coded_bytes: usize,
    max_width: u32,
    max_height: u32,
    /// Samples in the largest block its sub-band stores only part of. Each
    /// worker decodes such blocks into scratch of this size.
    max_clipped_samples: usize,
}

/// How [`plan_tile_route`] wants the tile's code blocks decoded.
#[cfg_attr(test, derive(Debug, PartialEq, Eq))]
pub(in crate::j2c::decode) enum TileRoute {
    /// Every code block of the tile was decoded in place.
    Decoded,
    /// Too little work to fork: decode the tile serially.
    Serial,
    /// The tile needs the general per-sub-band path.
    PerSubBand,
}

/// Decode every code block of the tile in parallel when all of its
/// components use HT block coding only, or classic coding with one arithmetic
/// segment per block, and the tile carries enough coded data to amortize a
/// fork-join. Returns without side effects for the non-`Decoded` routes.
pub(in crate::j2c::decode) fn plan_tile_route<'a>(
    tile: &Tile<'a>,
    components: Range<usize>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    header: &Header<'_>,
    ht_decoder: &Option<&mut dyn HtCodeBlockDecoder>,
    cpu_decode_parallelism: CpuDecodeParallelism,
) -> Result<TileRoute> {
    #[cfg(test)]
    if tile_ctx.force_per_sub_band {
        return Ok(TileRoute::PerSubBand);
    }
    // One worker cannot overlap entropy jobs; the serial path keeps its own
    // context and allocates no parallel workspace bank.
    if cpu_decode_parallelism != CpuDecodeParallelism::Auto
        || rayon::current_num_threads() < 2
        || ht_decoder.is_some()
        || storage.exact_integer_decode
    {
        return Ok(TileRoute::PerSubBand);
    }
    let Some(coding) = tile_route_coding(tile, components.clone()) else {
        return Ok(TileRoute::PerSubBand);
    };

    let has_later_components = components.end < tile.component_infos.len();
    // Segment bytes bound the coded bytes, so a tile below the cutoff decodes
    // serially before anything is planned.
    let (required, segment_bytes, rows) =
        required_code_blocks(tile, components.clone(), storage, header, coding)?;
    if let Some(route) = route_fallback(
        coding,
        rows,
        segment_bytes,
        rayon::current_num_threads(),
        MIN_PARALLEL_CODED_BYTES,
    ) {
        return Ok(route);
    }
    let mut budget = DecodeAllocationBudget::for_storage(storage)?;
    let mut plan = match plan_tile(
        tile,
        components,
        storage,
        header,
        &mut budget,
        coding,
        required,
    ) {
        Ok(Some(plan)) => plan,
        Ok(None) => return Ok(TileRoute::PerSubBand),
        // Planning metadata did not fit the decode budget; the per-sub-band
        // path needs less and reports its own limits.
        Err(error) if crate::j2c::decode::reuse::is_capacity_error(&error) => {
            return Ok(TileRoute::PerSubBand);
        }
        Err(error) => return Err(error),
    };
    if let Some(route) = route_fallback(
        coding,
        plan.strips.len(),
        plan.coded_bytes,
        rayon::current_num_threads(),
        MIN_PARALLEL_CODED_BYTES,
    ) {
        return Ok(route);
    }

    let workers = rayon::current_num_threads().min(plan.strips.len());
    let mut tasks: Vec<(&mut [f32], &Strip)> = Vec::new();
    let route_budget = (|| -> Result<()> {
        // Payloads are gathered only once the tile takes the route.
        gather_payloads(&mut plan, storage, &mut budget)?;
        tasks
            .try_reserve_exact(plan.strips.len())
            .map_err(|_| DecodingError::HostAllocationFailed)?;
        budget.include_capacity_overage::<(&mut [f32], &Strip)>(
            plan.strips.len(),
            tasks.capacity(),
        )?;
        // Each worker allocates its clipped-block scratch on first use.
        budget.include_elements::<f32>(
            workers
                .checked_mul(plan.max_clipped_samples)
                .ok_or(ValidationError::ImageTooLarge)?,
        )?;
        if has_later_components {
            // Component-at-a-time decoding retains this serial context while
            // preparing the next component. Do not commit a parallel bank
            // that leaves too little headroom for that replacement peak.
            budget.include_bytes(tile_ctx.serial_tier1_capacity_bytes()?)?;
        }
        Ok(())
    })();
    if let Err(error) = route_budget {
        if crate::j2c::decode::reuse::is_capacity_error(&error) {
            return Ok(TileRoute::PerSubBand);
        }
        return Err(error);
    }
    let Some(growths) =
        decode_with_retained_bank(&plan, workers, tasks, tile_ctx, storage, &mut budget)?
    else {
        // The per-sub-band path has its own smaller-footprint fallbacks.
        return Ok(TileRoute::PerSubBand);
    };
    #[cfg(test)]
    if let Some(post_decode) = tile_ctx.tile_route_post_decode.take() {
        post_decode(storage)?;
    }
    record_route_peak(tile_ctx, storage, &plan, workers)?;
    count_decoded_blocks(tile_ctx, &plan, workers, growths);
    Ok(TileRoute::Decoded)
}

/// Decode the plan with the retained bank of its block coding. Returns the
/// bank's growths, or `None` when the bank does not fit the budget.
fn decode_with_retained_bank<'a>(
    plan: &TilePlan<'a>,
    workers: usize,
    tasks: Vec<(&mut [f32], &Strip)>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    budget: &mut DecodeAllocationBudget,
) -> Result<Option<usize>> {
    match plan.coding {
        CodeBlockCoding::Classic => decode_with_bank(
            plan,
            workers,
            tasks,
            &mut tile_ctx.classic_task_workspaces,
            storage,
            budget,
            |job, output, stride, workspace| {
                decode_classic_strip_block(plan, job, output, stride, workspace)
            },
        ),
        CodeBlockCoding::HighThroughput => decode_with_bank(
            plan,
            workers,
            tasks,
            &mut tile_ctx.ht_task_workspaces,
            storage,
            budget,
            |job, output, stride, workspace| {
                decode_strip_block(plan, job, output, stride, workspace)
            },
        ),
    }
}

/// Count the decoded plan's blocks and, for HT, its phase statistics.
fn count_decoded_blocks(
    tile_ctx: &mut TileDecodeContext,
    plan: &TilePlan<'_>,
    workers: usize,
    growths: usize,
) {
    let counters = &mut tile_ctx.debug_counters;
    counters.decoded_code_blocks += plan.jobs.len();
    counters.skipped_code_blocks += plan.skipped;
    if plan.coding == CodeBlockCoding::HighThroughput {
        let stats = &mut counters.ht_phase_stats;
        for job in &plan.jobs {
            stats.blocks += 1;
            stats.cleanup_bytes += u128::from(job.payload_length - job.refinement_length);
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
    }
    #[cfg(test)]
    if plan.coding == CodeBlockCoding::Classic {
        counters.classic_task_workspace_growths += growths;
    }
    #[cfg(not(test))]
    let _ = (workers, growths);
}

/// Record the decode's workspace peak while the plan, its tasks and each
/// worker's clipped-block scratch, as budgeted, are live next to the retained
/// workspace bank.
fn record_route_peak(
    tile_ctx: &mut TileDecodeContext,
    storage: &DecompositionStorage<'_>,
    plan: &TilePlan<'_>,
    workers: usize,
) -> Result<()> {
    let route_bytes = [
        plan.bands.capacity() * size_of::<BandParameters>(),
        plan.classic_bands.capacity() * size_of::<ClassicJobParameters>(),
        plan.jobs.capacity() * size_of::<BlockJob<'_>>(),
        plan.strips.capacity() * size_of::<Strip>(),
        plan.arena.capacity(),
        plan.strips.len() * size_of::<(&mut [f32], &Strip)>(),
        workers
            .saturating_mul(plan.max_clipped_samples)
            .saturating_mul(size_of::<f32>()),
    ]
    .into_iter()
    .fold(0, usize::saturating_add);
    let sample = WorkspaceBytes::sample(tile_ctx, storage)?.with_tier1(route_bytes);
    tile_ctx.debug_counters.workspace_peak.record(sample);
    Ok(())
}

/// Prepare one workspace per worker in the retained `bank`, then decode the
/// plan's strips into `storage`'s coefficients in place through
/// `decode_block`. Returns the bank's growths, or `None` when the bank does
/// not fit the budget.
fn decode_with_bank<'a, W: TaskScratch>(
    plan: &TilePlan<'a>,
    workers: usize,
    tasks: Vec<(&mut [f32], &Strip)>,
    bank: &mut Vec<TaskWorkspace<W>>,
    storage: &mut DecompositionStorage<'a>,
    budget: &mut DecodeAllocationBudget,
    decode_block: impl Fn(&BlockJob<'_>, &mut [f32], usize, &mut W) -> Result<()> + Sync,
) -> Result<Option<usize>> {
    let Ok(growths) = prepare_uniform_task_workspaces(
        bank,
        workers,
        plan.max_width,
        plan.max_height,
        &mut storage.structural_workspace_bytes,
        budget,
    ) else {
        return Ok(None);
    };
    let mut coefficients = core::mem::take(&mut storage.coefficients);
    let result = decode_strips(
        plan,
        &mut coefficients,
        &mut bank[..workers],
        tasks,
        |job, output, stride, slot| decode_block(job, output, stride, &mut slot.workspace),
    );
    storage.coefficients = coefficients;
    result.map(|()| Some(growths))
}

/// The block coding of every component in `components` when the route can
/// decode it: HT only, or classic with one arithmetic segment per block.
fn tile_route_coding(tile: &Tile<'_>, components: Range<usize>) -> Option<CodeBlockCoding> {
    let mut styles = tile
        .component_infos
        .get(components)?
        .iter()
        .map(|component_info| component_info.coding_style.parameters.code_block_style);
    if styles.clone().all(|style| {
        style.uses_high_throughput_block_coding() && !style.allows_mixed_block_coding()
    }) {
        return Some(CodeBlockCoding::HighThroughput);
    }
    styles
        .all(|style| {
            !style.uses_high_throughput_block_coding()
                && !style.allows_mixed_block_coding()
                && style.single_codeword_segment()
        })
        .then_some(CodeBlockCoding::Classic)
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
    coding: CodeBlockCoding,
    required: usize,
) -> Result<Option<TilePlan<'a>>> {
    let mut plan = TilePlan {
        coding,
        bands: Vec::new(),
        classic_bands: Vec::new(),
        strict: header.strict,
        jobs: Vec::new(),
        strips: Vec::new(),
        arena: Vec::new(),
        arena_len: 0,
        skipped: 0,
        coded_bytes: 0,
        max_width: 0,
        max_height: 0,
        max_clipped_samples: 0,
    };
    // A region decode may need only a few of the tile's code blocks.
    budget.reserve_new(&mut plan.jobs, required)?;
    budget.reserve_new(&mut plan.strips, required)?;
    // Classic job metadata must not enlarge the HT plan's band entries.
    if coding == CodeBlockCoding::Classic {
        budget.reserve_new(&mut plan.classic_bands, storage.sub_bands.len())?;
    } else {
        budget.reserve_new(&mut plan.bands, storage.sub_bands.len())?;
    }
    let mut arena_len = 0usize;

    let component_infos = tile
        .component_infos
        .get(components.clone())
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    for (component_idx, component_info) in components.zip(component_infos) {
        let style = component_info.coding_style.parameters.code_block_style;
        let classic_style = (coding == CodeBlockCoding::Classic).then_some(style);
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
                    if classic_style.is_some() {
                        MAX_BITPLANE_COUNT
                    } else {
                        31
                    },
                )?;
                let band = if classic_style.is_some() {
                    let (sub_band_type, job_style) =
                        classic_decode_job_parameters(sub_band.sub_band_type, component_info);
                    let band = plan.classic_bands.len();
                    plan.classic_bands.push(ClassicJobParameters {
                        sub_band_type,
                        style: job_style,
                        strict: header.strict,
                        total_bitplanes: parameters.num_bitplanes,
                        roi_shift: component_info.roi_shift,
                        dequantization_step: parameters.dequantization_step,
                        irreversible_midpoint: parameters.irreversible_midpoint,
                    });
                    band
                } else {
                    let band = plan.bands.len();
                    plan.bands.push(BandParameters {
                        num_bitplanes: parameters.num_bitplanes,
                        roi_shift: component_info.roi_shift,
                        stripe_causal: style.vertically_causal_context,
                        dequantization_step: parameters.dequantization_step,
                        irreversible_midpoint: parameters.irreversible_midpoint,
                    });
                    band
                };

                let first_job = plan.jobs.len();
                for precinct in sub_band
                    .precincts
                    .clone()
                    .map(|idx| &storage.precincts[idx])
                {
                    for code_block_idx in precinct.code_blocks.clone() {
                        let code_block = &storage.code_blocks[code_block_idx];
                        if code_block.coding != Some(coding) {
                            continue;
                        }
                        if !code_block_required_by_index(storage, sub_band_idx, code_block) {
                            plan.skipped += 1;
                            continue;
                        }
                        if classic_style.is_none()
                            && !ht_code_block_has_decodable_passes(
                                code_block,
                                coded_bitplanes,
                                header.strict,
                            )?
                        {
                            continue;
                        }
                        let job = plan_block(
                            code_block,
                            code_block_idx,
                            stored_placement(code_block, sub_band),
                            band,
                            storage,
                            &mut arena_len,
                            classic_style,
                        )?;
                        if !job.placement.is_whole(job.width, job.height) {
                            let samples = (job.width as usize)
                                .checked_mul(job.height as usize)
                                .ok_or(ValidationError::ImageTooLarge)?;
                            plan.max_clipped_samples = plan.max_clipped_samples.max(samples);
                        }
                        plan.coded_bytes =
                            plan.coded_bytes.saturating_add(job.payload_length as usize);
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
    plan.arena_len = arena_len;
    budget.include_elements::<(&mut [f32], &Strip)>(plan.strips.len())?;
    Ok(Some(plan))
}

/// The code blocks of `components` with `coding` that the decode needs,
/// which bounds the plan's jobs and strips, and the bytes of all their
/// segments. The stored block grid bounds the number of strips without
/// allocating or sorting jobs, so skinny classic tiles can decline early.
fn required_code_blocks(
    tile: &Tile<'_>,
    components: Range<usize>,
    storage: &DecompositionStorage<'_>,
    header: &Header<'_>,
    coding: CodeBlockCoding,
) -> Result<(usize, usize, usize)> {
    let mut count = 0usize;
    let mut segment_bytes = 0usize;
    // HT retains its exact strip check after planning.
    let mut rows = if coding == CodeBlockCoding::Classic {
        0
    } else {
        usize::MAX
    };
    for component_idx in components {
        let component_info = tile
            .component_infos
            .get(component_idx)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        let resolutions = component_info
            .num_resolution_levels()
            .checked_sub(header.skipped_resolution_levels)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        let tile_composition = &storage.tile_decompositions[component_idx];
        for resolution in 0..resolutions {
            let block_height = (coding == CodeBlockCoding::Classic).then(|| {
                ResolutionTile::new(ComponentTile::new(tile, component_info), resolution)
                    .code_block_height()
            });
            for sub_band_idx in tile_composition.sub_band_iter(resolution, &storage.decompositions)
            {
                let sub_band = &storage.sub_bands[sub_band_idx];
                if let Some(height) = block_height {
                    let rect = sub_band.storage_rect;
                    if !rect.is_empty() {
                        rows += (rect.y1.div_ceil(height) - rect.y0 / height) as usize;
                    }
                }
                for precinct in sub_band.precincts.clone() {
                    for code_block in storage.precincts[precinct]
                        .code_blocks
                        .clone()
                        .map(|idx| &storage.code_blocks[idx])
                        .filter(|code_block| {
                            code_block.coding == Some(coding)
                                && code_block_required_by_index(storage, sub_band_idx, code_block)
                        })
                    {
                        count += 1;
                        segment_bytes = storage.layers[code_block.layers.clone()]
                            .iter()
                            .filter_map(|layer| layer.segments.clone())
                            .flat_map(|range| &storage.segments[range])
                            .fold(segment_bytes, |bytes, segment| {
                                bytes.saturating_add(segment.data.len())
                            });
                    }
                }
            }
        }
    }
    Ok((count, segment_bytes, rows))
}

/// A block's job. A payload in one codestream segment is borrowed; others
/// get room in the arena.
fn plan_block<'a>(
    code_block: &CodeBlock,
    code_block_idx: usize,
    placement: BlockPlacement,
    band: usize,
    storage: &DecompositionStorage<'a>,
    arena_len: &mut usize,
    classic_style: Option<CodeBlockStyle>,
) -> Result<BlockJob<'a>> {
    let (payload_length, refinement_length, single) = match classic_style {
        Some(style) => classic_payload(code_block, style, storage)?,
        None => ht_payload(code_block, storage)?,
    };
    let payload = if let Some(data) = single {
        Payload::Borrowed(data)
    } else {
        let start = *arena_len;
        *arena_len = start
            .checked_add(payload_length)
            .ok_or(ValidationError::ImageTooLarge)?;
        Payload::Gathered(start)
    };
    Ok(BlockJob {
        payload,
        code_block: code_block_idx,
        payload_length: u32::try_from(payload_length)
            .map_err(|_| DecodingError::CodeBlockDecodeFailure)?,
        refinement_length: u32::try_from(refinement_length)
            .map_err(|_| DecodingError::CodeBlockDecodeFailure)?,
        placement,
        width: code_block.rect.width(),
        height: code_block.rect.height(),
        missing_bit_planes: code_block.missing_bit_planes,
        number_of_coding_passes: code_block.number_of_coding_passes,
        band,
    })
}

/// An HT block's payload and refinement lengths, and its cleanup segment when
/// that is the only selected segment.
fn ht_payload<'a>(
    code_block: &CodeBlock,
    storage: &DecompositionStorage<'a>,
) -> Result<(usize, usize, Option<&'a [u8]>)> {
    let (cleanup_length, refinement_length) =
        selected_code_block_segment_lengths(code_block, storage)?;
    let single = single_cleanup_segment(code_block, storage)?.filter(|_| refinement_length == 0);
    let payload_length = cleanup_length
        .checked_add(refinement_length)
        .ok_or(ValidationError::ImageTooLarge)?;
    Ok((payload_length, refinement_length, single))
}

/// A classic block's payload length, checked as
/// `collect_classic_code_block_data` checks it, and its segment data when at
/// most one segment holds any.
fn classic_payload<'a>(
    code_block: &CodeBlock,
    style: CodeBlockStyle,
    storage: &DecompositionStorage<'a>,
) -> Result<(usize, usize, Option<&'a [u8]>)> {
    let counts = classic_allocation_counts(code_block, style, storage)?;
    if counts.coding_passes != code_block.number_of_coding_passes {
        return Err(DecodingError::CodeBlockDecodeFailure.into());
    }
    let single = (counts.fragments <= 1).then(|| {
        storage.layers[code_block.layers.clone()]
            .iter()
            .filter_map(|layer| layer.segments.clone())
            .flat_map(|range| &storage.segments[range])
            .map(|segment| segment.data)
            .find(|data| !data.is_empty())
            .unwrap_or_default()
    });
    Ok((counts.data_bytes, 0, single))
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

/// Group one sub-band's jobs into rows of code blocks. A strip covers the
/// stored rows of its blocks. Returns `false` when the stored parts of the
/// blocks do not tile the sub-band in rows of equal height.
fn plan_band_strips(
    plan: &mut TilePlan<'_>,
    first_job: usize,
    coefficients: Range<usize>,
    band_width: u32,
    band_height: u32,
) -> Result<bool> {
    let band_jobs = &mut plan.jobs[first_job..];
    band_jobs.sort_unstable_by_key(|job| (job.placement.target_y, job.placement.target_x));
    let stride = band_width as usize;
    let mut index = first_job;
    while index < plan.jobs.len() {
        let first = plan.jobs[index].placement;
        let (y_offset, height) = (first.target_y, first.height);
        let mut end = index;
        let mut work = 0usize;
        let mut next_x = 0u32;
        while end < plan.jobs.len() && plan.jobs[end].placement.target_y == y_offset {
            let job = &plan.jobs[end];
            let placement = job.placement;
            let fits = placement.height == height
                && placement.source_y == first.source_y
                && placement.target_x >= next_x
                && placement
                    .validate(job.width, job.height, band_width, band_height)
                    .is_ok();
            if !fits {
                return Ok(false);
            }
            next_x = placement.target_x + placement.width;
            work = work
                .saturating_add(job.payload_length as usize)
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
    budget: &mut DecodeAllocationBudget,
) -> Result<()> {
    if plan.arena_len == 0 {
        return Ok(());
    }
    budget.reserve_new(&mut plan.arena, plan.arena_len)?;
    for job in &mut plan.jobs {
        let Payload::Gathered(_) = job.payload else {
            continue;
        };
        // Row sorting can change precinct traversal order. Assign the offset
        // when the payload is appended in its final execution order.
        let start = plan.arena.len();
        job.payload = Payload::Gathered(start);
        let code_block = &storage.code_blocks[job.code_block];
        // A classic payload is every segment; an HT payload the selected
        // set's cleanup and refinement segments.
        let cleanup_index = if plan.coding == CodeBlockCoding::Classic {
            None
        } else {
            let selected_set = code_block
                .ht_selected_set
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            Some(
                selected_set
                    .checked_mul(2)
                    .ok_or(DecodingError::CodeBlockDecodeFailure)?,
            )
        };
        for layer in &storage.layers[code_block.layers.clone()] {
            let Some(range) = layer.segments.clone() else {
                continue;
            };
            for segment in &storage.segments[range] {
                if cleanup_index
                    .is_none_or(|index| segment.idx == index || segment.idx == index + 1)
                {
                    plan.arena.extend_from_slice(segment.data);
                }
            }
        }
        let expected = job.payload_length as usize;
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
fn prepare_uniform_task_workspaces<W: TaskScratch>(
    workspaces: &mut Vec<TaskWorkspace<W>>,
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
    let old_bank_bytes = task_workspace_bytes(workspaces)?;
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
        let planned = W::planned_bytes(slot_width, slot_height)?;
        trial.include_bytes(planned)?;
        let mut workspace = W::default();
        workspace.prepare_for(slot_width, slot_height)?;
        charge_actual_workspace(&mut trial, planned, workspace.owned_bytes()?)?;
        let grew = workspaces
            .get(index)
            .map_or(Ok(true), |old| -> Result<bool> {
                Ok(workspace.owned_bytes()? > old.workspace.owned_bytes()?)
            })?;
        growths += usize::from(grew);
        replacement.push(TaskWorkspace {
            workspace,
            prepared_width: slot_width,
            prepared_height: slot_height,
        });
    }
    let new_bank_bytes = task_workspace_bytes(&replacement)?;
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

/// Decode the strips on one worker per workspace, each block through
/// `decode_block`.
fn decode_strips<'plan, 'coefficients, W: Send>(
    plan: &'plan TilePlan<'_>,
    coefficients: &'coefficients mut [f32],
    workspaces: &mut [W],
    mut tasks: Vec<(&'coefficients mut [f32], &'plan Strip)>,
    decode_block: impl Fn(&BlockJob<'_>, &mut [f32], usize, &mut W) -> Result<()> + Sync,
) -> Result<()> {
    // Strips are sorted by start offset, so carving them front to back yields
    // disjoint mutable outputs.
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
    let queue = Mutex::new(tasks.into_iter());
    let failed = AtomicBool::new(false);
    // Each worker owns one already-budgeted workspace for the whole tile.
    // A single queue pop per strip keeps work balanced across unequal cores
    // without forking a Rayon job and borrowing/returning scratch for each row.
    workspaces
        .par_iter_mut()
        .with_max_len(1)
        .try_for_each(|workspace| -> Result<()> {
            let mut clipped_scratch = Vec::new();
            while !failed.load(Ordering::Relaxed) {
                let next = queue
                    .lock()
                    .map_err(|_| DecodingError::CodeBlockDecodeFailure)?
                    .next();
                let Some((output, strip)) = next else {
                    break;
                };
                if let Err(error) = decode_strip(
                    plan,
                    strip,
                    output,
                    workspace,
                    &mut clipped_scratch,
                    &decode_block,
                ) {
                    failed.store(true, Ordering::Relaxed);
                    return Err(error);
                }
            }
            Ok(())
        })
}

fn decode_strip<W>(
    plan: &TilePlan<'_>,
    strip: &Strip,
    output: &mut [f32],
    workspace: &mut W,
    clipped_scratch: &mut Vec<f32>,
    decode_block: &impl Fn(&BlockJob<'_>, &mut [f32], usize, &mut W) -> Result<()>,
) -> Result<()> {
    for job in &plan.jobs[strip.jobs.clone()] {
        // The strip starts at the block row's first stored row.
        let placement = BlockPlacement {
            target_y: 0,
            ..job.placement
        };
        let whole = placement.is_whole(job.width, job.height);
        // The decoder writes every row and column of the block, so a block
        // the sub-band stores only part of goes through scratch.
        let (block_output, output_stride) = if whole {
            (&mut output[placement.target_x as usize..], strip.stride)
        } else {
            if clipped_scratch.is_empty() {
                try_resize_decode_elements(clipped_scratch, plan.max_clipped_samples, 0.0)?;
            }
            let block = clipped_scratch
                .get_mut(..job.width as usize * job.height as usize)
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            block.fill(0.0);
            (block, job.width as usize)
        };
        decode_block(job, block_output, output_stride, workspace)?;
        if !whole {
            placement.copy_from_block(clipped_scratch, job.width, output, strip.stride)?;
        }
    }
    Ok(())
}

fn decode_strip_block(
    plan: &TilePlan<'_>,
    job: &BlockJob<'_>,
    output: &mut [f32],
    output_stride: usize,
    workspace: &mut HtCodeBlockDecodeWorkspace,
) -> Result<()> {
    let band = plan.bands[job.band];
    let decode_job = HtCodeBlockDecodeJob {
        data: job_payload(plan, job)?,
        cleanup_length: job.payload_length - job.refinement_length,
        refinement_length: job.refinement_length,
        width: job.width,
        height: job.height,
        output_stride,
        missing_bit_planes: job.missing_bit_planes,
        number_of_coding_passes: job.number_of_coding_passes,
        num_bitplanes: band.num_bitplanes,
        roi_shift: band.roi_shift,
        stripe_causal: band.stripe_causal,
        strict: plan.strict,
        dequantization_step: band.dequantization_step,
    };
    #[cfg(feature = "simd")]
    if crate::scalar::try_decode_ht_code_block_to_f32(
        decode_job,
        output,
        workspace,
        band.irreversible_midpoint,
    )? {
        return Ok(());
    }
    if band.irreversible_midpoint {
        decode_ht_code_block_scalar_with_workspace_midpoint(decode_job, output, workspace)?;
    } else {
        decode_ht_code_block_scalar_with_workspace(decode_job, output, workspace)?;
    }
    Ok(())
}

fn decode_classic_strip_block(
    plan: &TilePlan<'_>,
    job: &BlockJob<'_>,
    output: &mut [f32],
    output_stride: usize,
    workspace: &mut J2kCodeBlockDecodeWorkspace,
) -> Result<()> {
    let band = plan.classic_bands[job.band];
    let segments = [J2kCodeBlockSegment {
        data_offset: 0,
        data_length: job.payload_length,
        start_coding_pass: 0,
        end_coding_pass: job.number_of_coding_passes,
        use_arithmetic: true,
    }];
    let decode_job = band.job(
        job_payload(plan, job)?,
        &segments,
        job.width,
        job.height,
        output_stride,
        job.missing_bit_planes,
        job.number_of_coding_passes,
    );
    band.decode(decode_job, output, workspace)
}

/// The job's payload, borrowed from the codestream or from the arena.
fn job_payload<'p>(plan: &'p TilePlan<'_>, job: &BlockJob<'p>) -> Result<&'p [u8]> {
    match job.payload {
        Payload::Borrowed(data) => Ok(data),
        Payload::Gathered(start) => plan
            .arena
            .get(start..start + job.payload_length as usize)
            .ok_or_else(|| DecodingError::CodeBlockDecodeFailure.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::{route_fallback, CodeBlockCoding, TileRoute};

    #[test]
    fn tile_route() {
        let (ht, classic) = (CodeBlockCoding::HighThroughput, CodeBlockCoding::Classic);
        // (coding, strips, coded bytes, threads, expected fallback) with an 8 KiB cutoff.
        for (coding, strips, coded_bytes, threads, expected) in [
            (ht, 1, usize::MAX, 4, Some(TileRoute::Serial)),
            (ht, 16, 8191, 4, Some(TileRoute::Serial)),
            (ht, 2, 8192, 4, None),
            (classic, 16, 1023, 4, Some(TileRoute::Serial)),
            (classic, 16, 1024, 4, None),
            (classic, 3, 1024, 4, Some(TileRoute::PerSubBand)),
            (classic, 1, usize::MAX, 2, Some(TileRoute::PerSubBand)),
        ] {
            assert_eq!(
                route_fallback(coding, strips, coded_bytes, threads, 8192),
                expected,
                "{coding:?} {strips} strips, {coded_bytes} bytes, {threads} threads"
            );
        }
    }
}
