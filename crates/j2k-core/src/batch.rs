// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;
use core::num::NonZeroUsize;

use crate::{scale::Downscale, types::Rect};

/// Worker configuration for CPU tile batches.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TileBatchOptions {
    /// Worker count. `None` asks the codec crate to use available parallelism.
    pub workers: Option<NonZeroUsize>,
}

impl TileBatchOptions {
    /// Construct tile-batch options with an optional fixed worker count.
    pub const fn new(workers: Option<NonZeroUsize>) -> Self {
        Self { workers }
    }
}

/// Indexed result produced by one tile-batch worker.
pub type IndexedBatchResult<T, E> = (usize, Result<T, E>);

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

/// Resolve the number of CPU workers for a tile batch.
///
/// `available_workers` should be the host's available parallelism. Passing
/// `0` is accepted and treated as one available worker.
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

/// Restore successful indexed worker results to caller input order.
///
/// If any worker result failed, returns the error produced by `make_error`
/// for the lowest failing input index.
///
/// # Panics
///
/// Panics if a successful batch is missing an index or if a result index is
/// outside `0..job_count`.
pub fn collect_indexed_batch_results<T, E, B, F>(
    job_count: usize,
    results: Vec<IndexedBatchResult<T, E>>,
    make_error: F,
) -> Result<Vec<T>, B>
where
    F: FnOnce(usize, E) -> B,
{
    let mut outcomes = Vec::with_capacity(job_count);
    outcomes.resize_with(job_count, || None);
    let mut first_error = None::<(usize, E)>;
    for (index, result) in results {
        assert!(
            index < job_count,
            "indexed batch result index {index} outside job count {job_count}"
        );
        match result {
            Ok(outcome) => outcomes[index] = Some(outcome),
            Err(source) => {
                if first_error
                    .as_ref()
                    .is_none_or(|(current, _)| index < *current)
                {
                    first_error = Some((index, source));
                }
            }
        }
    }

    if let Some((index, source)) = first_error {
        return Err(make_error(index, source));
    }

    Ok(outcomes
        .into_iter()
        .map(|outcome| outcome.expect("successful batch stores one outcome per tile"))
        .collect())
}
