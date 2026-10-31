// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tile-wide code-block decode with in-place output.
//!
//! The per-sub-band path joins once per sub-band, copies every code-block
//! payload, decodes into a slab, and scatters the slab serially. Here every
//! row of code blocks in every sub-band of the tile is one task. A row of code
//! blocks owns a contiguous range of its sub-band's coefficients, so tasks
//! write their output in place through disjoint slices and Rayon balances
//! them with work stealing. HT cleanup-only and single-segment classic
//! payloads are borrowed from the codestream; a payload that spans several
//! segments is gathered from the block's segment records into its worker's
//! scratch just before the block decodes. Classic blocks take this route when
//! each decodes as one arithmetic segment: no selective bypass or per-pass
//! termination.

use alloc::boxed::Box;
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
use crate::j2c::build::{CodeBlock, CodeBlockCoding, Layer, Segment};
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

/// Below this average work per band, as the route weighs strips (payload
/// bytes plus an eighth of the samples), scheduling each band on the pool
/// costs more than overlapping Tier-1 with the IDWT saves, and a component's
/// bands decode over whole windows on the route instead. A tall region of
/// 16x4 HT blocks, at 88 per band, decoded 3.6 times slower in bands than
/// over whole windows at 4 threads; 1024 px regions of the benchmark images
/// average 2,000 to 15,000.
pub(crate) const MIN_BAND_WORK: usize = 1024;

/// Where a component of `coding` blocks with `strips` rows of code blocks
/// and `coded_bytes` of payload decodes on the band route with `threads`
/// workers: on the route (`None`), serially, or per sub-band. A classic
/// component with fewer rows than workers leaves some idle there, while the
/// per-sub-band path splits each row's blocks among them.
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

/// Where a tile of `coding` blocks with `strips` rows of code blocks and
/// `coded_bytes` of payload decodes: on the tile route (`None`) or serially.
/// Unlike the band route, a classic tile with fewer rows than workers stays on
/// the tile route, which splits its long rows into parts of about equal work;
/// the per-sub-band path would decode its sub-bands one after another.
fn tile_route_fallback(
    coding: CodeBlockCoding,
    strips: usize,
    coded_bytes: usize,
    min_coded_bytes: usize,
) -> Option<TileRoute> {
    (too_few_coded_bytes(coding, coded_bytes, min_coded_bytes) || strips < 2)
        .then_some(TileRoute::Serial)
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
    /// Segments gathered from the block's records into the decoding
    /// worker's scratch.
    Gathered,
}

/// The records a gathered payload is read from while blocks decode: every
/// code-block, layer and segment of the tile, borrowed from its storage.
#[derive(Clone, Copy)]
pub(in crate::j2c::decode) struct PayloadSource<'s, 'a> {
    code_blocks: &'s [CodeBlock],
    layers: &'s [Layer],
    segments: &'s [Segment<'a>],
}

impl<'s, 'a> PayloadSource<'s, 'a> {
    pub(in crate::j2c::decode) fn new(storage: &'s DecompositionStorage<'a>) -> Self {
        Self {
            code_blocks: &storage.code_blocks,
            layers: &storage.layers,
            segments: &storage.segments,
        }
    }
}

/// A worker's scratch for the blocks it decodes: a block its sub-band stores
/// only part of, and a payload gathered from several segments.
#[derive(Default)]
pub(in crate::j2c::decode) struct BlockScratch {
    clipped: Vec<f32>,
    payload: Vec<u8>,
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
    /// Bytes of the largest payload gathered from several segments. Each
    /// worker gathers such payloads into scratch of this size.
    max_gathered_len: usize,
    skipped: usize,
    coded_bytes: usize,
    max_width: u32,
    max_height: u32,
    /// Samples in the largest block its sub-band stores only part of. Each
    /// worker decodes such blocks into scratch of this size.
    max_clipped_samples: usize,
    /// The sub-band of each band, in the order of `bands` or `classic_bands`.
    band_sub_bands: Vec<usize>,
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
    if let Some(route) = tile_route_fallback(coding, rows, segment_bytes, MIN_PARALLEL_CODED_BYTES)
    {
        return Ok(route);
    }
    let mut budget = DecodeAllocationBudget::for_storage(storage)?;
    let plan = match plan_tile(
        tile,
        components,
        storage,
        header,
        &mut budget,
        coding,
        required,
        false,
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
    if let Some(route) = tile_route_fallback(
        coding,
        plan.strips.len(),
        plan.coded_bytes,
        MIN_PARALLEL_CODED_BYTES,
    ) {
        return Ok(route);
    }

    let workers = rayon::current_num_threads().min(plan.strips.len());
    let counts = TaskCounts::new(&plan, workers, storage.stale_coefficients)?;
    let route_budget = (|| -> Result<()> {
        // Each worker gathers payloads into scratch of its own as it decodes.
        include_payload_scratch(&mut budget, &plan, workers)?;
        // `plan_tile` charged one task per strip.
        budget.include_elements::<StripTask<'_, '_, '_>>(counts.tasks - plan.strips.len())?;
        budget.include_elements::<Mutex<&mut [f32]>>(counts.shared)?;
        // Each worker allocates its block scratch on first use.
        budget.include_elements::<f32>(
            workers
                .checked_mul(counts.scratch_samples)
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
        decode_with_retained_bank(&plan, workers, counts, tile_ctx, storage, &mut budget)?
    else {
        // The per-sub-band path has its own smaller-footprint fallbacks.
        return Ok(TileRoute::PerSubBand);
    };
    #[cfg(test)]
    if let Some(post_decode) = tile_ctx.tile_route_post_decode.take() {
        post_decode(storage)?;
    }
    record_route_peak(tile_ctx, storage, &plan, workers, counts)?;
    count_decoded_blocks(tile_ctx, &plan, workers, growths);
    Ok(TileRoute::Decoded)
}

/// Decode the plan with the retained bank of its block coding. Returns the
/// bank's growths, or `None` when the bank does not fit the budget.
fn decode_with_retained_bank<'a>(
    plan: &TilePlan<'a>,
    workers: usize,
    counts: TaskCounts,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    budget: &mut DecodeAllocationBudget,
) -> Result<Option<usize>> {
    match plan.coding {
        CodeBlockCoding::Classic => decode_with_bank(
            plan,
            workers,
            counts,
            &mut tile_ctx.classic_task_workspaces,
            storage,
            budget,
            |job, payload, output, stride, workspace| {
                decode_classic_strip_block(plan, job, payload, output, stride, workspace)
            },
        ),
        CodeBlockCoding::HighThroughput => decode_with_bank(
            plan,
            workers,
            counts,
            &mut tile_ctx.ht_task_workspaces,
            storage,
            budget,
            |job, payload, output, stride, workspace| {
                decode_strip_block(plan, job, payload, output, stride, workspace)
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
    counts: TaskCounts,
) -> Result<()> {
    let route_bytes = [
        plan.bands.capacity() * size_of::<BandParameters>(),
        plan.classic_bands.capacity() * size_of::<ClassicJobParameters>(),
        plan.jobs.capacity() * size_of::<BlockJob<'_>>(),
        plan.strips.capacity() * size_of::<Strip>(),
        plan.band_sub_bands.capacity() * size_of::<usize>(),
        workers.saturating_mul(plan.max_gathered_len),
        counts.tasks * size_of::<StripTask<'_, '_, '_>>(),
        counts.shared * size_of::<Mutex<&mut [f32]>>(),
        workers
            .saturating_mul(counts.scratch_samples)
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
/// `decode_block`, which leaves a stale buffer current. Returns the bank's
/// growths, or `None` when the bank does not fit the budget.
fn decode_with_bank<'a, W: TaskScratch>(
    plan: &TilePlan<'a>,
    workers: usize,
    counts: TaskCounts,
    bank: &mut Vec<TaskWorkspace<W>>,
    storage: &mut DecompositionStorage<'a>,
    budget: &mut DecodeAllocationBudget,
    decode_block: impl Fn(&BlockJob<'_>, &[u8], &mut [f32], usize, &mut W) -> Result<()> + Sync,
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
    let stale = storage.stale_coefficients;
    let result = decode_strips(
        plan,
        &mut coefficients,
        stale,
        &mut bank[..workers],
        counts,
        PayloadSource::new(storage),
        |job, payload, output, stride, slot| {
            decode_block(job, payload, output, stride, &mut slot.workspace)
        },
    );
    storage.coefficients = coefficients;
    result?;
    storage.stale_coefficients = false;
    Ok(Some(growths))
}

/// Whether the route can decode the code-blocks of every component of
/// `tile`.
pub(in crate::j2c::decode) fn routes_tile(tile: &Tile<'_>) -> bool {
    tile_route_coding(tile, 0..tile.component_infos.len()).is_some()
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
    clippy::too_many_arguments,
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
    banded: bool,
) -> Result<Option<TilePlan<'a>>> {
    let mut plan = TilePlan {
        coding,
        bands: Vec::new(),
        classic_bands: Vec::new(),
        strict: header.strict,
        jobs: Vec::new(),
        strips: Vec::new(),
        max_gathered_len: 0,
        skipped: 0,
        coded_bytes: 0,
        max_width: 0,
        max_height: 0,
        max_clipped_samples: 0,
        band_sub_bands: Vec::new(),
    };
    // A region decode may need only a few of the tile's code blocks.
    budget.reserve_new(&mut plan.jobs, required)?;
    budget.reserve_new(&mut plan.strips, required)?;
    budget.reserve_new(&mut plan.band_sub_bands, storage.sub_bands.len())?;
    // Classic job metadata must not enlarge the HT plan's band entries.
    if coding == CodeBlockCoding::Classic {
        budget.reserve_new(&mut plan.classic_bands, storage.sub_bands.len())?;
    } else {
        budget.reserve_new(&mut plan.bands, storage.sub_bands.len())?;
    }
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

                plan.band_sub_bands.push(sub_band_idx);
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
                            classic_style,
                        )?;
                        if matches!(job.payload, Payload::Gathered) {
                            plan.max_gathered_len =
                                plan.max_gathered_len.max(job.payload_length as usize);
                        }
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
                let (width, height) = (
                    sub_band.storage_rect.width(),
                    sub_band.storage_rect.height(),
                );
                // Banded strips address rows of the sub-band's stored
                // rectangle; the executor gives each band its own slot.
                let coefficients = if banded {
                    0..(width as usize)
                        .checked_mul(height as usize)
                        .ok_or(ValidationError::ImageTooLarge)?
                } else {
                    sub_band.coefficients.clone()
                };
                if !plan_band_strips(&mut plan, first_job, coefficients, width, height)? {
                    return Ok(None);
                }
            }
        }
    }

    if !banded && !strips_are_disjoint(&mut plan.strips, storage.coefficients.len()) {
        return Ok(None);
    }
    budget.include_elements::<StripTask<'_, '_, '_>>(plan.strips.len())?;
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
/// are gathered when the block decodes.
fn plan_block<'a>(
    code_block: &CodeBlock,
    code_block_idx: usize,
    placement: BlockPlacement,
    band: usize,
    storage: &DecompositionStorage<'a>,
    classic_style: Option<CodeBlockStyle>,
) -> Result<BlockJob<'a>> {
    let (payload_length, refinement_length, single) = match classic_style {
        Some(style) => classic_payload(code_block, style, storage)?,
        None => ht_payload(code_block, storage)?,
    };
    let payload = single.map_or(Payload::Gathered, Payload::Borrowed);
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
            work = work.saturating_add(job_work(job));
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

/// A block's estimated decode cost, as the route weighs strips and parts.
fn job_work(job: &BlockJob<'_>) -> usize {
    (job.payload_length as usize).saturating_add((job.width as usize * job.height as usize) / 8)
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

/// Charge the payload scratch each of `threads` threads grows to as it
/// gathers the plan's payloads.
fn include_payload_scratch(
    budget: &mut DecodeAllocationBudget,
    plan: &TilePlan<'_>,
    threads: usize,
) -> Result<()> {
    budget.include_bytes(
        threads
            .checked_mul(plan.max_gathered_len)
            .ok_or(ValidationError::ImageTooLarge)?,
    )
}

/// Gather the payload of `job`, a block whose payload spans several
/// segments, into `scratch`: for a classic block every segment, for an HT
/// block the selected set's cleanup and refinement segments.
fn gather_payload<'s>(
    coding: CodeBlockCoding,
    job: &BlockJob<'_>,
    source: PayloadSource<'_, '_>,
    scratch: &'s mut Vec<u8>,
) -> Result<&'s [u8]> {
    let expected = job.payload_length as usize;
    scratch.clear();
    scratch
        .try_reserve(expected)
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    let code_block = source
        .code_blocks
        .get(job.code_block)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    let cleanup_index = if coding == CodeBlockCoding::Classic {
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
    let layers = source
        .layers
        .get(code_block.layers.clone())
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    for layer in layers {
        let Some(range) = layer.segments.clone() else {
            continue;
        };
        let segments = source
            .segments
            .get(range)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        for segment in segments {
            if cleanup_index.is_none_or(|index| segment.idx == index || segment.idx == index + 1) {
                if scratch.len() + segment.data.len() > expected {
                    return Err(DecodingError::CodeBlockDecodeFailure.into());
                }
                scratch.extend_from_slice(segment.data);
            }
        }
    }
    if scratch.len() != expected {
        return Err(DecodingError::CodeBlockDecodeFailure.into());
    }
    Ok(scratch)
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

/// Parts per worker that a classic tile's code-block rows are split into,
/// so that the queue ends in short tasks: a worker that finds it empty waits
/// for the others' last tasks only, and a strip of a whole row can take a
/// couple of milliseconds.
const CLASSIC_PARTS_PER_WORKER: usize = 8;

/// One entry of the route's queue.
enum StripTask<'coefficients, 'plan, 'shared> {
    /// A range no strip decodes, which a stale buffer must zero.
    Zero(&'coefficients mut [f32]),
    /// A whole strip, decoded in place into its output.
    Strip(&'coefficients mut [f32], &'plan Strip),
    /// Jobs of a strip whose output its parts share, with their work.
    Part(
        &'shared Mutex<&'coefficients mut [f32]>,
        &'plan Strip,
        Range<usize>,
        usize,
    ),
}

impl StripTask<'_, '_, '_> {
    fn work(&self) -> usize {
        match self {
            Self::Zero(_) => 0,
            Self::Strip(_, strip) => strip.work,
            Self::Part(_, _, _, work) => *work,
        }
    }
}

/// The sizes of the route's queue, as budgeted before the decode.
#[derive(Clone, Copy)]
struct TaskCounts {
    /// Queue entries, at most.
    tasks: usize,
    /// Strips split into parts, which share their output.
    shared: usize,
    /// Samples of each worker's block scratch.
    scratch_samples: usize,
    /// The work a part of a split strip aims at; `None` keeps whole strips.
    part_work: Option<usize>,
}

impl TaskCounts {
    fn new(plan: &TilePlan<'_>, workers: usize, stale: bool) -> Result<Self> {
        // HT rows decode quickly enough whole.
        let part_work = (plan.coding == CodeBlockCoding::Classic).then(|| {
            let total = plan
                .strips
                .iter()
                .fold(0usize, |total, strip| total.saturating_add(strip.work));
            (total / workers.saturating_mul(CLASSIC_PARTS_PER_WORKER).max(1)).max(1)
        });
        let mut counts = Self {
            tasks: 0,
            shared: 0,
            scratch_samples: plan.max_clipped_samples,
            part_work,
        };
        for strip in &plan.strips {
            let parts = counts.parts(strip);
            counts.tasks = counts.tasks.saturating_add(parts);
            counts.shared += usize::from(parts > 1);
        }
        if stale {
            // A zeroing task for each gap around the strips.
            counts.tasks = counts.tasks.saturating_add(plan.strips.len() + 1);
        }
        if counts.shared > 0 {
            let block = (plan.max_width as usize)
                .checked_mul(plan.max_height as usize)
                .ok_or(ValidationError::ImageTooLarge)?;
            counts.scratch_samples = counts.scratch_samples.max(block);
        }
        Ok(counts)
    }

    /// How many parts `strip` is split into.
    fn parts(&self, strip: &Strip) -> usize {
        self.part_work.map_or(1, |part| {
            strip.work.div_ceil(part).clamp(1, strip.jobs.len().max(1))
        })
    }
}

/// Split `strip` into `parts` runs of consecutive jobs of about equal work,
/// each passed to `emit` with its work.
fn split_strip(
    plan: &TilePlan<'_>,
    strip: &Strip,
    parts: usize,
    emit: impl FnMut(Range<usize>, usize) -> Result<()>,
) -> Result<()> {
    let works = strip.jobs.clone().map(|index| {
        plan.jobs
            .get(index)
            .map(job_work)
            .ok_or_else(|| DecodingError::CodeBlockDecodeFailure.into())
    });
    split_runs(strip.jobs.start, works, strip.work, parts, emit)
}

/// Split consecutive jobs, numbered from `first` and with `works` summing to
/// `total`, into at most `parts` runs of about equal work: a run ends at the
/// first job that takes the work done to its share of `total`. Each run is
/// passed to `emit` with its work.
fn split_runs(
    first: usize,
    works: impl Iterator<Item = Result<usize>>,
    total: usize,
    parts: usize,
    mut emit: impl FnMut(Range<usize>, usize) -> Result<()>,
) -> Result<()> {
    let mut works = works.peekable();
    let mut start = first;
    let mut end = first;
    let mut done = 0usize;
    let mut part_work = 0usize;
    let mut cuts = 1usize;
    while let Some(work) = works.next() {
        let work = work?;
        end += 1;
        done = done.saturating_add(work);
        part_work = part_work.saturating_add(work);
        let last = works.peek().is_none();
        if last || (cuts < parts && done.saturating_mul(parts) >= cuts.saturating_mul(total)) {
            emit(start..end, part_work)?;
            start = end;
            part_work = 0;
            cuts += 1;
        }
    }
    Ok(())
}

/// Queue `task`. The budget covers [`TaskCounts`], so the queue never grows
/// past the capacity reserved from it.
fn push_task<T>(tasks: &mut Vec<T>, task: T) -> Result<()> {
    if tasks.len() == tasks.capacity() {
        return Err(DecodingError::CodeBlockDecodeFailure.into());
    }
    tasks.push(task);
    Ok(())
}

/// Decode the strips on one worker per workspace, each block through
/// `decode_block`.
fn decode_strips<'plan, 'coefficients, W: Send>(
    plan: &'plan TilePlan<'_>,
    coefficients: &'coefficients mut [f32],
    stale: bool,
    workspaces: &mut [W],
    counts: TaskCounts,
    source: PayloadSource<'_, '_>,
    decode_block: impl Fn(&BlockJob<'_>, &[u8], &mut [f32], usize, &mut W) -> Result<()> + Sync,
) -> Result<()> {
    let mut shared: Vec<Mutex<&'coefficients mut [f32]>> = Vec::new();
    shared
        .try_reserve_exact(counts.shared)
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    let mut tasks: Vec<StripTask<'coefficients, 'plan, '_>> = Vec::new();
    tasks
        .try_reserve_exact(counts.tasks)
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    // Strips are sorted by start offset, so carving them front to back yields
    // disjoint mutable outputs. In a stale buffer each strip zeroes its own
    // rows on its worker, where the decode then finds them in cache; the
    // ranges around the strips are zeroed by tasks of their own.
    let mut rest = coefficients;
    let mut consumed = 0usize;
    for strip in &plan.strips {
        let tail = core::mem::take(&mut rest);
        let (gap, tail) = tail.split_at_mut(strip.start - consumed);
        let (output, tail) = tail.split_at_mut(strip.len);
        rest = tail;
        consumed = strip.start + strip.len;
        if stale && !gap.is_empty() {
            push_task(&mut tasks, StripTask::Zero(gap))?;
        }
        if counts.parts(strip) > 1 {
            if shared.len() == shared.capacity() {
                return Err(DecodingError::CodeBlockDecodeFailure.into());
            }
            shared.push(Mutex::new(output));
        } else {
            push_task(&mut tasks, StripTask::Strip(output, strip))?;
        }
    }
    if stale && !rest.is_empty() {
        push_task(&mut tasks, StripTask::Zero(rest))?;
    }
    let mut outputs = shared.iter();
    for strip in &plan.strips {
        let parts = counts.parts(strip);
        if parts > 1 {
            let output = outputs
                .next()
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            split_strip(plan, strip, parts, |jobs, work| {
                push_task(&mut tasks, StripTask::Part(output, strip, jobs, work))
            })?;
        }
    }

    // Largest tasks first so stragglers are small; zeroing tasks last.
    tasks.sort_unstable_by_key(|task| core::cmp::Reverse(task.work()));
    let queue = Mutex::new(tasks.into_iter());
    let failed = AtomicBool::new(false);
    // Each worker owns one already-budgeted workspace for the whole tile.
    // A single queue pop per task keeps work balanced across unequal cores
    // without forking a Rayon job and borrowing/returning scratch for each row.
    workspaces
        .par_iter_mut()
        .with_max_len(1)
        .try_for_each(|workspace| -> Result<()> {
            let mut scratch = BlockScratch::default();
            while !failed.load(Ordering::Relaxed) {
                let next = queue
                    .lock()
                    .map_err(|_| DecodingError::CodeBlockDecodeFailure)?
                    .next();
                let Some(task) = next else {
                    break;
                };
                let decoded = match task {
                    StripTask::Zero(output) => {
                        output.fill(0.0);
                        Ok(())
                    }
                    StripTask::Strip(output, strip) => {
                        if stale {
                            output.fill(0.0);
                        }
                        decode_strip(
                            plan,
                            strip,
                            output,
                            workspace,
                            (&mut scratch, source),
                            &decode_block,
                        )
                    }
                    StripTask::Part(output, strip, jobs, _) => decode_part(
                        plan,
                        strip,
                        jobs,
                        output,
                        stale,
                        counts.scratch_samples,
                        workspace,
                        (&mut scratch, source),
                        &decode_block,
                    ),
                };
                if let Err(error) = decoded {
                    failed.store(true, Ordering::Relaxed);
                    return Err(error);
                }
            }
            Ok(())
        })
}

/// Decode jobs `jobs` of `strip`, whose output its parts share: each block
/// into scratch, then into the output under its lock. In a stale buffer the
/// part first zeroes its columns, from its first block to the next part's.
#[expect(
    clippy::too_many_arguments,
    reason = "a part carries its strip's shared output and the worker's scratch explicitly"
)]
fn decode_part<W>(
    plan: &TilePlan<'_>,
    strip: &Strip,
    jobs: Range<usize>,
    output: &Mutex<&mut [f32]>,
    stale: bool,
    scratch_samples: usize,
    workspace: &mut W,
    (scratch, source): (&mut BlockScratch, PayloadSource<'_, '_>),
    decode_block: &impl Fn(&BlockJob<'_>, &[u8], &mut [f32], usize, &mut W) -> Result<()>,
) -> Result<()> {
    let BlockScratch {
        clipped: scratch,
        payload: payload_scratch,
    } = scratch;
    let lock = || {
        output
            .lock()
            .map_err(|_| DecodingError::CodeBlockDecodeFailure)
    };
    if stale {
        let column = |index: usize| {
            if index == strip.jobs.start {
                Some(0)
            } else if index == strip.jobs.end {
                Some(strip.stride)
            } else {
                plan.jobs
                    .get(index)
                    .map(|job| job.placement.target_x as usize)
            }
        };
        let (Some(start), Some(end)) = (column(jobs.start), column(jobs.end)) else {
            return Err(DecodingError::CodeBlockDecodeFailure.into());
        };
        for row in lock()?.chunks_mut(strip.stride) {
            row.get_mut(start..end)
                .ok_or(DecodingError::CodeBlockDecodeFailure)?
                .fill(0.0);
        }
    }
    if scratch.len() < scratch_samples {
        try_resize_decode_elements(scratch, scratch_samples, 0.0)?;
    }
    for job in plan
        .jobs
        .get(jobs)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?
    {
        // The strip starts at the block row's first stored row.
        let placement = BlockPlacement {
            target_y: 0,
            ..job.placement
        };
        let block = scratch
            .get_mut(..job.width as usize * job.height as usize)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        if !placement.is_whole(job.width, job.height) {
            block.fill(0.0);
        }
        let payload = job_payload(plan, job, source, payload_scratch)?;
        decode_block(job, payload, block, job.width as usize, workspace)?;
        placement.copy_from_block(block, job.width, &mut lock()?, strip.stride)?;
    }
    Ok(())
}

fn decode_strip<W>(
    plan: &TilePlan<'_>,
    strip: &Strip,
    output: &mut [f32],
    workspace: &mut W,
    (scratch, source): (&mut BlockScratch, PayloadSource<'_, '_>),
    decode_block: &impl Fn(&BlockJob<'_>, &[u8], &mut [f32], usize, &mut W) -> Result<()>,
) -> Result<()> {
    let BlockScratch {
        clipped: clipped_scratch,
        payload: payload_scratch,
    } = scratch;
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
            // One worker's scratch serves the strips of every component's
            // plan, so it grows to the largest clipped block of each.
            if clipped_scratch.len() < plan.max_clipped_samples {
                try_resize_decode_elements(clipped_scratch, plan.max_clipped_samples, 0.0)?;
            }
            let block = clipped_scratch
                .get_mut(..job.width as usize * job.height as usize)
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            block.fill(0.0);
            (block, job.width as usize)
        };
        let payload = job_payload(plan, job, source, payload_scratch)?;
        decode_block(job, payload, block_output, output_stride, workspace)?;
        if !whole {
            placement.copy_from_block(clipped_scratch, job.width, output, strip.stride)?;
        }
    }
    Ok(())
}

fn decode_strip_block(
    plan: &TilePlan<'_>,
    job: &BlockJob<'_>,
    payload: &[u8],
    output: &mut [f32],
    output_stride: usize,
    workspace: &mut HtCodeBlockDecodeWorkspace,
) -> Result<()> {
    let band = plan.bands[job.band];
    let decode_job = HtCodeBlockDecodeJob {
        data: payload,
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
    payload: &[u8],
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
        payload,
        &segments,
        job.width,
        job.height,
        output_stride,
        job.missing_bit_planes,
        job.number_of_coding_passes,
    );
    band.decode(decode_job, output, workspace)
}

/// The job's payload, borrowed from the codestream or gathered into
/// `scratch`.
fn job_payload<'p>(
    plan: &TilePlan<'_>,
    job: &BlockJob<'p>,
    source: PayloadSource<'_, '_>,
    scratch: &'p mut Vec<u8>,
) -> Result<&'p [u8]> {
    match job.payload {
        Payload::Borrowed(data) => Ok(data),
        Payload::Gathered => gather_payload(plan.coding, job, source, scratch),
    }
}

/// Where a band decodes: into its slot, or as run `part` of its blocks split
/// for `parts` parts (see [`BandRoute::strip_parts`]), into the slot its
/// parts share. A part zeroes its own columns of the slot, from its first
/// block to the next part's, and copies each block in under the lock.
pub(in crate::j2c::decode) enum BandOutput<'o, 'c> {
    Whole(&'o mut [f32]),
    Part {
        part: usize,
        parts: usize,
        slot: &'o Mutex<&'c mut [f32]>,
    },
}

/// One code-block row of a sub-band, as [`plan_band_route`] plans it: the
/// band's rows, counted from the top of the sub-band's stored rectangle.
#[derive(Clone, Copy)]
pub(in crate::j2c::decode) struct BandStrip {
    pub(in crate::j2c::decode) sub_band: usize,
    pub(in crate::j2c::decode) first_row: usize,
    pub(in crate::j2c::decode) rows: usize,
    /// The strip's estimated decode cost, as the route weighs it.
    pub(in crate::j2c::decode) work: usize,
}

/// A component's code-block rows planned for decoding band by band on the
/// worker pool, each into a slot of its own, with a workspace per worker in
/// the retained bank of the plan's block coding.
pub(in crate::j2c::decode) struct BandRoute<'a> {
    plan: TilePlan<'a>,
    workers: usize,
    growths: usize,
    /// The decode's budget with the plan, every thread's payload and block
    /// scratch and the bank's growth charged, which stay live while the
    /// bands decode, beside whatever the budget the route was planned in
    /// held already.
    budget: DecodeAllocationBudget,
}

/// How [`plan_band_route`] wants a component's bands decoded.
pub(in crate::j2c::decode) enum BandRouting<'a> {
    /// On the worker pool, band by band.
    Route(Box<BandRoute<'a>>),
    /// Too little work to fork: decode the bands serially.
    Serial,
    /// Bands the route can't decode, or too few rows for the workers:
    /// decode the whole windows the old way.
    WholeWindows,
}

/// Plan component `component_idx` of a tile whose sub-bands hold rings of
/// bands, with the route's checks and cutoffs: its block coding, its coded
/// bytes and its rows of code-blocks. With a plan, the coding's retained bank
/// holds a workspace for each of `workers` pool threads and one more for the
/// calling thread, charged to the structural workspace.
pub(in crate::j2c::decode) fn plan_band_route<'a>(
    tile: &Tile<'a>,
    component_idx: usize,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    header: &Header<'_>,
) -> Result<BandRouting<'a>> {
    plan_band_route_beside(tile, component_idx, tile_ctx, storage, header, None)
}

/// [`plan_band_route`], charging the route to `live` when other routes stay
/// live beside it: the budget of the route planned last for the same
/// executor, which holds every earlier route's charges. Without it, the
/// route is charged to the decode's storage budget.
#[expect(
    clippy::too_many_lines,
    reason = "the route's coding, tile route, plan and cutoffs decide one band routing in order"
)]
pub(in crate::j2c::decode) fn plan_band_route_beside<'a>(
    tile: &Tile<'a>,
    component_idx: usize,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    header: &Header<'_>,
    live: Option<DecodeAllocationBudget>,
) -> Result<BandRouting<'a>> {
    let components = component_idx..component_idx + 1;
    let Some(coding) = tile_route_coding(tile, components.clone()) else {
        return Ok(BandRouting::WholeWindows);
    };
    let threads = rayon::current_num_threads();
    let declined = |route| match route {
        TileRoute::Serial => BandRouting::Serial,
        _ => BandRouting::WholeWindows,
    };
    let (required, segment_bytes, rows) =
        required_code_blocks(tile, components.clone(), storage, header, coding)?;
    if let Some(route) = route_fallback(
        coding,
        rows,
        segment_bytes,
        threads,
        MIN_PARALLEL_CODED_BYTES,
    ) {
        return Ok(declined(route));
    }
    let mut budget = match live {
        Some(live) => live,
        None => DecodeAllocationBudget::for_storage(storage)?,
    };
    let plan = match plan_tile(
        tile,
        components,
        storage,
        header,
        &mut budget,
        coding,
        required,
        true,
    ) {
        Ok(Some(plan)) => plan,
        Ok(None) => return Ok(BandRouting::WholeWindows),
        // The serial bands need less and report their own limits.
        Err(error) if crate::j2c::decode::reuse::is_capacity_error(&error) => {
            return Ok(BandRouting::Serial);
        }
        Err(error) => return Err(error),
    };
    if let Some(route) = route_fallback(
        coding,
        plan.strips.len(),
        plan.coded_bytes,
        threads,
        MIN_PARALLEL_CODED_BYTES,
    ) {
        return Ok(declined(route));
    }
    let work = plan
        .strips
        .iter()
        .fold(0usize, |work, strip| work.saturating_add(strip.work));
    // Unit tests use tiny fixtures and take the pool unless a test sets the
    // cutoff on the decode's context.
    #[cfg(test)]
    let min_band_work = tile_ctx.band_work_cutoff;
    #[cfg(not(test))]
    let min_band_work = MIN_BAND_WORK;
    if work / plan.strips.len().max(1) < min_band_work {
        return Ok(BandRouting::WholeWindows);
    }
    let workers = threads.min(plan.strips.len());
    let prepared = (|| -> Result<usize> {
        // Each thread gathers payloads into scratch of its own as it
        // decodes, and allocates its block scratch on first use: a clipped
        // block, or any block of a band it decodes a part of.
        include_payload_scratch(&mut budget, &plan, workers + 1)?;
        budget.include_elements::<f32>(
            (workers + 1)
                .checked_mul(band_scratch_samples(&plan)?)
                .ok_or(ValidationError::ImageTooLarge)?,
        )?;
        let (width, height) = (plan.max_width, plan.max_height);
        let structural = &mut storage.structural_workspace_bytes;
        match coding {
            CodeBlockCoding::Classic => prepare_uniform_task_workspaces(
                &mut tile_ctx.classic_task_workspaces,
                workers + 1,
                width,
                height,
                structural,
                &mut budget,
            ),
            CodeBlockCoding::HighThroughput => prepare_uniform_task_workspaces(
                &mut tile_ctx.ht_task_workspaces,
                workers + 1,
                width,
                height,
                structural,
                &mut budget,
            ),
        }
    })();
    match prepared {
        Ok(growths) => Ok(BandRouting::Route(Box::new(BandRoute {
            plan,
            workers,
            growths,
            budget,
        }))),
        Err(error) if crate::j2c::decode::reuse::is_capacity_error(&error) => {
            Ok(BandRouting::Serial)
        }
        Err(error) => Err(error),
    }
}

/// The block scratch each thread of a band route holds: as large as any
/// block, since a part decodes every block of its run through it.
fn band_scratch_samples(plan: &TilePlan<'_>) -> Result<usize> {
    let block = (plan.max_width as usize)
        .checked_mul(plan.max_height as usize)
        .ok_or(ValidationError::ImageTooLarge)?;
    Ok(block.max(plan.max_clipped_samples))
}

impl BandRoute<'_> {
    pub(in crate::j2c::decode) fn coding(&self) -> CodeBlockCoding {
        self.plan.coding
    }

    /// The decode's budget with the route's own allocations charged; the
    /// executor charges what it adds to a copy.
    pub(in crate::j2c::decode) fn budget(&self) -> DecodeAllocationBudget {
        self.budget
    }

    /// Pool threads to decode on, besides the calling thread.
    pub(in crate::j2c::decode) fn workers(&self) -> usize {
        self.workers
    }

    pub(in crate::j2c::decode) fn strips(&self) -> impl Iterator<Item = BandStrip> + '_ {
        self.plan.strips.iter().map(|strip| {
            let band = self.plan.jobs[strip.jobs.start].band;
            BandStrip {
                sub_band: self.plan.band_sub_bands[band],
                first_row: strip.start / strip.stride,
                rows: strip.len / strip.stride,
                work: strip.work,
            }
        })
    }

    /// The runs of blocks strip `index` splits into when it aims at
    /// `parts` parts of about equal work: `parts` at most, fewer when a few
    /// blocks hold most of its work.
    pub(in crate::j2c::decode) fn strip_parts(&self, index: usize, parts: usize) -> Result<usize> {
        let strip = self
            .plan
            .strips
            .get(index)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        let mut runs = 0usize;
        split_strip(&self.plan, strip, parts, |_, _| {
            runs += 1;
            Ok(())
        })?;
        Ok(runs.max(1))
    }

    /// Decode strip `index` of a classic plan into `output`, whose slot
    /// holds the band's rows from its first.
    pub(in crate::j2c::decode) fn decode_classic(
        &self,
        index: usize,
        output: BandOutput<'_, '_>,
        workspace: &mut J2kCodeBlockDecodeWorkspace,
        scratch: (&mut BlockScratch, PayloadSource<'_, '_>),
    ) -> Result<()> {
        let plan = &self.plan;
        self.decode_band(
            index,
            output,
            workspace,
            scratch,
            &|job, payload, output, stride, workspace| {
                decode_classic_strip_block(plan, job, payload, output, stride, workspace)
            },
        )
    }

    /// Decode strip `index` of an HT plan into `output`, whose slot holds
    /// the band's rows from its first.
    pub(in crate::j2c::decode) fn decode_ht(
        &self,
        index: usize,
        output: BandOutput<'_, '_>,
        workspace: &mut HtCodeBlockDecodeWorkspace,
        scratch: (&mut BlockScratch, PayloadSource<'_, '_>),
    ) -> Result<()> {
        let plan = &self.plan;
        self.decode_band(
            index,
            output,
            workspace,
            scratch,
            &|job, payload, output, stride, workspace| {
                decode_strip_block(plan, job, payload, output, stride, workspace)
            },
        )
    }

    fn decode_band<W>(
        &self,
        index: usize,
        output: BandOutput<'_, '_>,
        workspace: &mut W,
        scratch: (&mut BlockScratch, PayloadSource<'_, '_>),
        decode_block: &impl Fn(&BlockJob<'_>, &[u8], &mut [f32], usize, &mut W) -> Result<()>,
    ) -> Result<()> {
        let plan = &self.plan;
        let strip = plan
            .strips
            .get(index)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        match output {
            BandOutput::Whole(slot) => {
                decode_strip(plan, strip, slot, workspace, scratch, decode_block)
            }
            BandOutput::Part { part, parts, slot } => {
                let mut jobs = None;
                let mut run = 0usize;
                split_strip(plan, strip, parts, |range, _| {
                    if run == part {
                        jobs = Some(range);
                    }
                    run += 1;
                    Ok(())
                })?;
                let jobs = jobs.ok_or(DecodingError::CodeBlockDecodeFailure)?;
                decode_part(
                    plan,
                    strip,
                    jobs,
                    slot,
                    true,
                    band_scratch_samples(plan)?,
                    workspace,
                    scratch,
                    decode_block,
                )
            }
        }
    }

    /// Count the plan's blocks and record the decode's peak with the plan
    /// and every thread's clipped-block scratch live.
    pub(in crate::j2c::decode) fn record(
        &self,
        tile_ctx: &mut TileDecodeContext,
        storage: &DecompositionStorage<'_>,
    ) -> Result<()> {
        // Band slots, not route tasks, hold this plan's outputs.
        let counts = TaskCounts {
            tasks: 0,
            shared: 0,
            scratch_samples: band_scratch_samples(&self.plan)?,
            part_work: None,
        };
        record_route_peak(tile_ctx, storage, &self.plan, self.workers + 1, counts)?;
        count_decoded_blocks(tile_ctx, &self.plan, self.workers + 1, self.growths);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::{
        route_fallback, split_runs, tile_route_fallback, CodeBlockCoding, Strip, TaskCounts,
        TileRoute,
    };

    #[test]
    fn split_runs_cover_the_strip_in_balanced_parts() {
        let ramp: Vec<usize> = (1..=40).collect();
        for (works, parts) in [
            (&[5_usize; 32][..], 8),
            (&[1, 1, 1, 100, 1, 1, 1][..], 3),
            (&[10, 20, 30, 40][..], 8),
            (&[7][..], 4),
            (&[3, 0, 0, 3][..], 2),
            (&[0, 0, 0][..], 2),
            (&ramp[..], 5),
        ] {
            let total: usize = works.iter().sum();
            let largest = works.iter().copied().max().unwrap_or(0);
            let mut runs = Vec::new();
            split_runs(
                10,
                works.iter().map(|&work| Ok(work)),
                total,
                parts,
                |jobs, work| {
                    runs.push((jobs, work));
                    Ok(())
                },
            )
            .unwrap();
            assert!(runs.len() <= parts, "{works:?} into {parts}: {runs:?}");
            let mut next = 10;
            for (jobs, work) in &runs {
                assert_eq!(jobs.start, next, "{works:?} into {parts}: {runs:?}");
                assert!(jobs.end > jobs.start, "{works:?} into {parts}: {runs:?}");
                assert_eq!(
                    *work,
                    works[jobs.start - 10..jobs.end - 10].iter().sum::<usize>()
                );
                // A run passes its share by less than its last job.
                assert!(
                    *work <= total.div_ceil(parts) + largest,
                    "{works:?}: {runs:?}"
                );
                next = jobs.end;
            }
            assert_eq!(next, 10 + works.len(), "{works:?} into {parts}: {runs:?}");
        }

        // Equal jobs split into equal runs.
        let mut runs = Vec::new();
        split_runs(
            0,
            [5_usize; 32].into_iter().map(Ok),
            160,
            8,
            |jobs, work| {
                runs.push((jobs, work));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(runs.len(), 8);
        assert!(runs
            .iter()
            .all(|(jobs, work)| jobs.len() == 4 && *work == 20));
    }

    #[test]
    fn strips_split_into_at_most_one_part_per_job() {
        let strip = |jobs, work| Strip {
            start: 0,
            len: 0,
            stride: 0,
            jobs,
            work,
        };
        let counts = |part_work| TaskCounts {
            tasks: 0,
            shared: 0,
            scratch_samples: 0,
            part_work,
        };
        assert_eq!(counts(None).parts(&strip(0..32, 1_000_000)), 1);
        assert_eq!(counts(Some(100)).parts(&strip(0..32, 1000)), 10);
        assert_eq!(counts(Some(100)).parts(&strip(0..32, 1001)), 11);
        assert_eq!(counts(Some(10)).parts(&strip(0..3, 1000)), 3);
        assert_eq!(counts(Some(100)).parts(&strip(0..32, 0)), 1);
        assert_eq!(counts(Some(100)).parts(&strip(5..5, 1000)), 1);
    }

    #[test]
    fn tile_route() {
        let (ht, classic) = (CodeBlockCoding::HighThroughput, CodeBlockCoding::Classic);
        // (coding, strips, coded bytes, expected fallback) with an 8 KiB cutoff:
        // a classic tile with fewer rows than workers stays on the route.
        for (coding, strips, coded_bytes, expected) in [
            (ht, 1, usize::MAX, Some(TileRoute::Serial)),
            (ht, 16, 8191, Some(TileRoute::Serial)),
            (ht, 2, 8192, None),
            (classic, 16, 1023, Some(TileRoute::Serial)),
            (classic, 16, 1024, None),
            (classic, 3, 1024, None),
            (classic, 1, usize::MAX, Some(TileRoute::Serial)),
        ] {
            assert_eq!(
                tile_route_fallback(coding, strips, coded_bytes, 8192),
                expected,
                "{coding:?} {strips} strips, {coded_bytes} bytes"
            );
        }
    }

    #[test]
    fn band_route() {
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
