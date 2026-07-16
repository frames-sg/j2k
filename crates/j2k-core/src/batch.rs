// SPDX-License-Identifier: MIT OR Apache-2.0

use core::num::NonZeroUsize;

use crate::{backend::BackendRequest, pixel::PixelFormat, scale::Downscale, types::Rect};

mod allocation;
mod collection;

#[doc(hidden)]
pub use allocation::{
    checked_batch_count_product, checked_batch_count_sum, try_batch_reserve_for_push,
    try_batch_reserve_to, BatchAllocationBudget, BatchAllocationRequest,
};
pub use collection::{
    try_collect_indexed_batch_results, try_collect_ordered_batch_results_with_limits,
};

/// Worker configuration for CPU tile batches.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TileBatchOptions {
    /// Worker count. `None` asks the codec crate to use available parallelism.
    pub workers: Option<NonZeroUsize>,
}

impl TileBatchOptions {
    /// Construct tile-batch options with an optional fixed worker count.
    #[must_use]
    pub const fn new(workers: Option<NonZeroUsize>) -> Self {
        Self { workers }
    }
}

/// Indexed result produced by one tile-batch worker.
#[doc(hidden)]
pub type IndexedBatchResult<T, E> = (usize, Result<T, E>);

/// One ordered batch result slot written by exactly one worker.
#[doc(hidden)]
pub type BatchResultSlot<T, E> = Option<Result<T, E>>;

/// One full-tile decode request.
pub struct TileDecodeJob<'i, 'o> {
    /// Compressed tile bytes.
    pub input: &'i [u8],
    /// Caller-owned output buffer for this tile.
    pub out: &'o mut [u8],
    /// Distance in bytes between output rows.
    pub stride: usize,
}

/// One region tile decode request.
pub struct TileRegionDecodeJob<'i, 'o> {
    /// Compressed tile bytes.
    pub input: &'i [u8],
    /// Caller-owned output buffer for this tile.
    pub out: &'o mut [u8],
    /// Distance in bytes between output rows.
    pub stride: usize,
    /// Region of interest in source-image coordinates.
    pub roi: Rect,
}

/// One scaled tile decode request.
pub struct TileScaledDecodeJob<'i, 'o> {
    /// Compressed tile bytes.
    pub input: &'i [u8],
    /// Caller-owned output buffer for this tile.
    pub out: &'o mut [u8],
    /// Distance in bytes between output rows.
    pub stride: usize,
    /// Downscale factor applied to the full-tile decode.
    pub scale: Downscale,
}

/// One region+scaled tile decode request.
pub struct TileRegionScaledDecodeJob<'i, 'o> {
    /// Compressed tile bytes.
    pub input: &'i [u8],
    /// Caller-owned output buffer for this tile.
    pub out: &'o mut [u8],
    /// Distance in bytes between output rows.
    pub stride: usize,
    /// Region of interest in source-image coordinates.
    pub roi: Rect,
    /// Downscale factor applied to the region decode.
    pub scale: Downscale,
}

/// One region+scaled tile device decode request.
pub struct TileRegionScaledDeviceDecodeRequest<'i> {
    /// Compressed tile bytes.
    pub input: &'i [u8],
    /// Pixel format requested for the decoded surface.
    pub fmt: PixelFormat,
    /// Region of interest in source-image coordinates.
    pub roi: Rect,
    /// Downscale factor applied to the region decode.
    pub scale: Downscale,
    /// Backend requested for the returned surface.
    pub backend: BackendRequest,
}

/// Error returned by tile batches, annotated with the failing input index.
#[derive(Debug)]
pub struct TileBatchError<E> {
    /// Index of the first failing tile in input order.
    pub index: usize,
    /// Decode error reported for that tile.
    pub source: E,
}

impl<E: core::fmt::Display> core::fmt::Display for TileBatchError<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "tile {} decode failed: {}", self.index, self.source)
    }
}

impl<E: core::error::Error + 'static> core::error::Error for TileBatchError<E> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Failure in batch scheduling, allocation, or worker-result collection.
///
/// These failures are deliberately separate from [`TileBatchError`]: no tile
/// index exists for an allocator failure or an internal scheduler invariant.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum BatchInfrastructureError {
    /// An internal planning boundary was invoked without any submitted jobs.
    #[error("batch plan requires at least one job")]
    EmptyBatchPlan,
    /// Batch-owned metadata would exceed the operation's host-memory budget.
    #[error("{what} is too large: requested {requested} bytes, cap {cap}")]
    AllocationTooLarge {
        /// Name of the batch-owned allocation or live set.
        what: &'static str,
        /// Requested live byte count, saturated to `usize::MAX` on overflow.
        requested: usize,
        /// Maximum permitted live byte count.
        cap: usize,
    },
    /// The host allocator rejected an otherwise cap-valid reservation.
    #[error("host allocation failed for {bytes} bytes while allocating {what}")]
    HostAllocationFailed {
        /// Name of the batch-owned allocation.
        what: &'static str,
        /// Requested host byte count.
        bytes: usize,
    },
    /// The host could not create a requested scoped worker.
    #[error("failed to spawn batch worker {worker}")]
    WorkerSpawnFailed {
        /// Zero-based worker index in the planned batch.
        worker: usize,
    },
    /// A worker unwound before completing its assigned jobs.
    #[error("batch worker {worker} panicked")]
    WorkerPanicked {
        /// Zero-based worker index in the planned batch.
        worker: usize,
    },
    /// Planned work referenced a worker slot that does not exist.
    #[error("batch worker slot {worker} is outside retained slot count {available}")]
    WorkerSlotMissing {
        /// Missing zero-based worker slot.
        worker: usize,
        /// Number of worker slots available to the scheduler.
        available: usize,
    },
    /// A worker managed by a shared parallel runtime unwound.
    #[error("parallel batch worker panicked")]
    ParallelWorkerPanicked,
    /// Shared batch state was poisoned by an earlier unwind.
    #[error("batch scheduler state was poisoned")]
    SchedulerPoisoned,
    /// A worker reported an index outside the submitted job range.
    #[error("batch result index {index} is outside job count {job_count}")]
    ResultIndexOutOfBounds {
        /// Invalid worker-reported index.
        index: usize,
        /// Number of submitted jobs.
        job_count: usize,
    },
    /// More than one worker result claimed the same job index.
    #[error("batch result index {index} was reported more than once")]
    DuplicateResult {
        /// Duplicated job index.
        index: usize,
    },
    /// No worker result was produced for a submitted job.
    #[error("batch worker result missing for job {index}")]
    MissingResult {
        /// Missing job index.
        index: usize,
    },
    /// Collector state contradicted a result kind it had just inspected.
    #[error("batch result {index} changed kind during ordered collection")]
    ResultKindMismatch {
        /// Job index whose result kind contradicted the inspected state.
        index: usize,
    },
}

/// Error returned by a fallible batch boundary.
///
/// `Tile` identifies an input-specific codec failure. `Infrastructure`
/// identifies failures for which assigning a tile index would be misleading.
#[derive(Debug)]
#[non_exhaustive]
pub enum BatchDecodeError<E> {
    /// The first codec failure in caller input order.
    Tile(TileBatchError<E>),
    /// Allocation, scheduling, or collection failed independently of a tile.
    Infrastructure(BatchInfrastructureError),
}

impl<E> BatchDecodeError<E> {
    /// Return the indexed codec failure when this is a tile-specific error.
    #[must_use]
    pub const fn tile_error(&self) -> Option<&TileBatchError<E>> {
        match self {
            Self::Tile(error) => Some(error),
            Self::Infrastructure(_) => None,
        }
    }

    /// Return the infrastructure failure when no tile index applies.
    #[must_use]
    pub const fn infrastructure_error(&self) -> Option<&BatchInfrastructureError> {
        match self {
            Self::Tile(_) => None,
            Self::Infrastructure(error) => Some(error),
        }
    }
}

impl<E: core::fmt::Display> core::fmt::Display for BatchDecodeError<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Tile(error) => error.fmt(f),
            Self::Infrastructure(error) => error.fmt(f),
        }
    }
}

impl<E: core::error::Error + 'static> core::error::Error for BatchDecodeError<E> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Tile(error) => Some(error),
            Self::Infrastructure(error) => Some(error),
        }
    }
}

impl<E> From<BatchInfrastructureError> for BatchDecodeError<E> {
    fn from(error: BatchInfrastructureError) -> Self {
        Self::Infrastructure(error)
    }
}

impl<E> From<TileBatchError<E>> for BatchDecodeError<E> {
    fn from(error: TileBatchError<E>) -> Self {
        Self::Tile(error)
    }
}

/// Resolve the number of CPU workers for a tile batch.
///
/// `available_workers` should be the host's available parallelism. Passing
/// `0` is accepted and treated as one available worker.
#[doc(hidden)]
pub fn tile_batch_worker_count(
    batch_size: usize,
    options: TileBatchOptions,
    available_workers: usize,
) -> usize {
    if batch_size <= 1 {
        return 1;
    }
    let workers = options.workers.map_or(available_workers, NonZeroUsize::get);
    workers.max(1).min(batch_size)
}
