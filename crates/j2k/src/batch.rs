// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded, fallible CPU batch facade for independent J2K/HTJ2K tiles.

use alloc::vec::Vec;
use std::sync::Arc;

pub use j2k_core::TileBatchOptions;
use j2k_core::{
    BatchResultSlot, DecodeOutcome, DecoderContext, PixelFormat, Rect, TileBatchDecode,
};

use crate::{J2kCodec, J2kContext, J2kDecodeWarning, J2kError, J2kScratchPool};

mod admission;
mod allocation;
mod direct;
mod planning;
mod scheduler;
mod worker;

use admission::BatchAllocationBudget;
use direct::build_repeated_direct_color_region_plan;

/// One full-tile decode request for [`decode_tiles_into`].
pub type TileDecodeJob<'i, 'o> = j2k_core::TileDecodeJob<'i, 'o>;

/// One ROI tile decode request for [`decode_tiles_region_into`].
pub type TileRegionDecodeJob<'i, 'o> = j2k_core::TileRegionDecodeJob<'i, 'o>;

/// One scaled tile decode request for [`decode_tiles_scaled_into`].
pub type TileScaledDecodeJob<'i, 'o> = j2k_core::TileScaledDecodeJob<'i, 'o>;

/// One ROI+scaled tile decode request for [`decode_tiles_region_scaled_into`].
pub type TileRegionScaledDecodeJob<'i, 'o> = j2k_core::TileRegionScaledDecodeJob<'i, 'o>;

/// Caller-owned output target for one context-reused J2K/HTJ2K tile decode helper.
pub struct TileDecodeOutput<'o> {
    /// Caller-owned output buffer.
    pub out: &'o mut [u8],
    /// Distance in bytes between output rows.
    pub stride: usize,
    /// Requested output pixel format.
    pub fmt: PixelFormat,
}

/// Error returned by J2K CPU tile batches.
///
/// Tile-specific codec failures retain their caller input index. Allocation,
/// spawning, worker panic, and result-integrity failures are reported as typed
/// infrastructure errors without inventing a tile-zero decode failure.
pub type TileBatchError = j2k_core::BatchDecodeError<J2kError>;

type BatchOutcome = DecodeOutcome<J2kDecodeWarning>;
type J2kBatchResultSlot = BatchResultSlot<BatchOutcome, J2kError>;

/// One-shot parse-plus-decode of an independent J2K/HTJ2K tile into the
/// caller's buffer, reusing both caller-owned [`DecoderContext`] and
/// caller-owned [`J2kScratchPool`].
#[doc(hidden)]
pub fn decode_tile_into_in_context(
    bytes: &[u8],
    ctx: &mut DecoderContext<J2kContext>,
    pool: &mut J2kScratchPool,
    output: TileDecodeOutput<'_>,
) -> Result<BatchOutcome, J2kError> {
    let TileDecodeOutput { out, stride, fmt } = output;
    <J2kCodec as TileBatchDecode>::decode_tile(ctx, pool, bytes, out, stride, fmt)
}

/// One-shot parse-plus-ROI-decode of an independent J2K/HTJ2K tile into the
/// caller's buffer, reusing both caller-owned [`DecoderContext`] and
/// caller-owned [`J2kScratchPool`].
#[doc(hidden)]
pub fn decode_tile_region_into_in_context(
    bytes: &[u8],
    ctx: &mut DecoderContext<J2kContext>,
    pool: &mut J2kScratchPool,
    output: TileDecodeOutput<'_>,
    roi: Rect,
) -> Result<BatchOutcome, J2kError> {
    let TileDecodeOutput { out, stride, fmt } = output;
    <J2kCodec as TileBatchDecode>::decode_tile_region(ctx, pool, bytes, out, stride, fmt, roi)
}

/// One-shot parse-plus-scaled-decode of an independent J2K/HTJ2K tile into the
/// caller's buffer, reusing both caller-owned [`DecoderContext`] and
/// caller-owned [`J2kScratchPool`].
#[doc(hidden)]
pub fn decode_tile_scaled_into_in_context(
    bytes: &[u8],
    ctx: &mut DecoderContext<J2kContext>,
    pool: &mut J2kScratchPool,
    output: TileDecodeOutput<'_>,
    scale: j2k_core::Downscale,
) -> Result<BatchOutcome, J2kError> {
    let TileDecodeOutput { out, stride, fmt } = output;
    <J2kCodec as TileBatchDecode>::decode_tile_scaled(ctx, pool, bytes, out, stride, fmt, scale)
}

/// One-shot parse-plus-ROI-scaled-decode of an independent J2K/HTJ2K tile
/// into the caller's buffer, reusing both caller-owned [`DecoderContext`] and
/// caller-owned [`J2kScratchPool`].
#[doc(hidden)]
pub fn decode_tile_region_scaled_into_in_context(
    bytes: &[u8],
    ctx: &mut DecoderContext<J2kContext>,
    pool: &mut J2kScratchPool,
    output: TileDecodeOutput<'_>,
    roi: Rect,
    scale: j2k_core::Downscale,
) -> Result<BatchOutcome, J2kError> {
    let TileDecodeOutput { out, stride, fmt } = output;
    <J2kCodec as TileBatchDecode>::decode_tile_region_scaled(
        ctx,
        pool,
        fmt,
        TileRegionScaledDecodeJob {
            input: bytes,
            out,
            stride,
            roi,
            scale,
        },
    )
}

/// Decode independent J2K/HTJ2K tiles into caller-owned output buffers.
///
/// Outcomes preserve caller input order. Generic native decoding is limited to
/// four concurrent workers by the aggregate memory policy; requesting more
/// workers reduces concurrency before spawning rather than weakening the
/// native decoder's per-operation bound.
pub fn decode_tiles_into(
    jobs: &mut [TileDecodeJob<'_, '_>],
    fmt: PixelFormat,
    options: TileBatchOptions,
) -> Result<Vec<BatchOutcome>, TileBatchError> {
    scheduler::decode_batch(jobs, options, |worker, chunk, results| {
        worker.decode_tile_jobs(chunk, results, fmt)
    })
}

/// Decode independent J2K/HTJ2K tile regions into caller-owned output buffers.
pub fn decode_tiles_region_into(
    jobs: &mut [TileRegionDecodeJob<'_, '_>],
    fmt: PixelFormat,
    options: TileBatchOptions,
) -> Result<Vec<BatchOutcome>, TileBatchError> {
    scheduler::decode_batch(jobs, options, |worker, chunk, results| {
        worker.decode_tile_region_jobs(chunk, results, fmt)
    })
}

/// Decode independent J2K/HTJ2K tiles at reduced resolution into caller-owned
/// output buffers.
pub fn decode_tiles_scaled_into(
    jobs: &mut [TileScaledDecodeJob<'_, '_>],
    fmt: PixelFormat,
    options: TileBatchOptions,
) -> Result<Vec<BatchOutcome>, TileBatchError> {
    scheduler::decode_batch(jobs, options, |worker, chunk, results| {
        worker.decode_tile_scaled_jobs(chunk, results, fmt)
    })
}

/// Decode independent J2K/HTJ2K tile regions at reduced resolution into
/// caller-owned output buffers.
pub fn decode_tiles_region_scaled_into(
    jobs: &mut [TileRegionScaledDecodeJob<'_, '_>],
    fmt: PixelFormat,
    options: TileBatchOptions,
) -> Result<Vec<BatchOutcome>, TileBatchError> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }

    let shared_direct_plan = build_repeated_direct_color_region_plan(jobs, fmt)
        .map_err(|source| TileBatchError::Tile(j2k_core::TileBatchError { index: 0, source }))?;
    let plan = scheduler::plan_direct_batch(jobs.len(), options)?;
    let shared_plan_bytes = shared_direct_plan
        .as_ref()
        .map_or(
            Ok(0),
            direct::DirectColorRegionCache::retained_allocation_bytes,
        )
        .map_err(|source| TileBatchError::Tile(j2k_core::TileBatchError { index: 0, source }))?;
    let allocation_budget = BatchAllocationBudget::with_baseline(shared_plan_bytes)?;
    let results = scheduler::run_chunks_scoped(
        jobs,
        plan,
        Some(Arc::clone(&allocation_budget)),
        |worker, chunk, results| {
            worker.decode_tile_region_scaled_jobs(chunk, results, fmt, shared_direct_plan.as_ref())
        },
    )?;

    // The shared native plan must not overlap ordered-result allocation. All
    // thread-owned contexts and scratch were already dropped by the scheduler.
    drop(shared_direct_plan);
    drop(allocation_budget);
    scheduler::collect_results(results)
}
