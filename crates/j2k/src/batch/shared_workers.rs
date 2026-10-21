// SPDX-License-Identifier: MIT OR Apache-2.0

//! Worker workspaces the free full-tile batch API keeps between calls.
//!
//! Each decode allocates several image-sized planes. Freeing them after every
//! call lets the allocator return the memory to the kernel, so the next call
//! page-faults and zero-fills it again; with many workers that churn dominates.
//! Keeping one bounded set of workers avoids it without an API change.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, TryLockError};

use j2k_core::PixelFormat;

use super::allocation::try_vec_with_capacity;
use super::scheduler::{decode_batch, decode_retained_tiles, plan_batch};
use super::worker::BatchWorker;
use super::{BatchOutcome, TileBatchError, TileBatchOptions, TileDecodeJob};
use crate::DecodeSettings;

/// Workers whose decoded-component, Tier-1, and IDWT allocations
/// [`decode_tiles_into`](super::decode_tiles_into) keeps between calls.
static SHARED_TILE_WORKERS: SharedWorkers = SharedWorkers::new();

/// Workspace bytes the shared workers may keep after a call. A larger set,
/// such as after decoding unusually large tiles, is released instead.
const SHARED_WORKSPACE_CAP_BYTES: usize = 256 * 1024 * 1024;

pub(super) fn decode_tiles_with_shared_workers(
    jobs: &mut [TileDecodeJob<'_, '_>],
    fmt: PixelFormat,
    options: TileBatchOptions,
) -> Result<Vec<BatchOutcome>, TileBatchError> {
    SHARED_TILE_WORKERS.decode(jobs, fmt, options, SHARED_WORKSPACE_CAP_BYTES)
}

pub(super) fn shared_tile_workspace_bytes() -> usize {
    SHARED_TILE_WORKERS.retained_bytes()
}

pub(super) fn release_shared_tile_workers() {
    SHARED_TILE_WORKERS.release();
}

/// One bounded set of retained workers. Only a decode ever holds the lock;
/// reading the retained size and requesting a release never wait for it.
struct SharedWorkers {
    workers: Mutex<Vec<BatchWorker>>,
    /// Workspace bytes retained after the most recent call.
    retained_bytes: AtomicUsize,
    /// Set by `release`; whichever call next holds the workers drops them.
    release_requested: AtomicBool,
}

impl SharedWorkers {
    const fn new() -> Self {
        Self {
            workers: Mutex::new(Vec::new()),
            retained_bytes: AtomicUsize::new(0),
            release_requested: AtomicBool::new(false),
        }
    }

    fn retained_bytes(&self) -> usize {
        self.retained_bytes.load(Ordering::Relaxed)
    }

    /// Free the retained workers now, or as soon as the decode using them ends.
    fn release(&self) {
        self.release_requested.store(true, Ordering::Release);
        if let Some(mut workers) = self.try_hold() {
            self.drop_if_requested(&mut workers);
        }
    }

    fn decode(
        &self,
        jobs: &mut [TileDecodeJob<'_, '_>],
        fmt: PixelFormat,
        options: TileBatchOptions,
        retention_cap_bytes: usize,
    ) -> Result<Vec<BatchOutcome>, TileBatchError> {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }
        let Some(mut workers) = self.try_hold() else {
            // Another call is using the shared workers. Decode with per-call
            // workers rather than wait for it.
            return decode_batch(jobs, options, |worker, chunk, results| {
                worker.decode_tile_jobs(chunk, results, fmt)
            });
        };
        self.drop_if_requested(&mut workers);
        let result = decode_on(&mut workers, jobs, fmt, options);
        if retained_workspace_bytes(&workers) > retention_cap_bytes {
            *workers = Vec::new();
        }
        self.drop_if_requested(&mut workers);
        self.retained_bytes
            .store(retained_workspace_bytes(&workers), Ordering::Relaxed);
        drop(workers);
        // A release requested after the check above still frees the workers
        // now instead of waiting for the next call.
        if self.release_requested.load(Ordering::Acquire) {
            self.release();
        }
        result
    }

    /// Hold the workers unless a decode already does. Workers left by a
    /// caller that panicked mid-decode may hold a workspace that was taken out
    /// and never returned, so they are discarded.
    fn try_hold(&self) -> Option<MutexGuard<'_, Vec<BatchWorker>>> {
        match self.workers.try_lock() {
            Ok(workers) => Some(workers),
            Err(TryLockError::Poisoned(poisoned)) => {
                let mut workers = poisoned.into_inner();
                *workers = Vec::new();
                self.retained_bytes.store(0, Ordering::Relaxed);
                self.workers.clear_poison();
                Some(workers)
            }
            Err(TryLockError::WouldBlock) => None,
        }
    }

    fn drop_if_requested(&self, workers: &mut Vec<BatchWorker>) {
        if self.release_requested.swap(false, Ordering::AcqRel) {
            *workers = Vec::new();
            self.retained_bytes.store(0, Ordering::Relaxed);
        }
    }
}

/// Grow the held workers to the planned count and decode on them.
fn decode_on(
    workers: &mut Vec<BatchWorker>,
    jobs: &mut [TileDecodeJob<'_, '_>],
    fmt: PixelFormat,
    options: TileBatchOptions,
) -> Result<Vec<BatchOutcome>, TileBatchError> {
    let worker_count = plan_batch(jobs.len(), options)?.worker_count;
    if workers.len() < worker_count {
        let mut grown = try_vec_with_capacity(worker_count, "J2K shared tile workers")?;
        grown.append(workers);
        grown.resize_with(worker_count, || BatchWorker::new_owned(worker_count.max(2)));
        *workers = grown;
    }
    decode_retained_tiles(workers, jobs, fmt, options, DecodeSettings::default())
}

/// Decoded-component, Tier-1, and IDWT bytes the workers keep allocated.
fn retained_workspace_bytes(workers: &[BatchWorker]) -> usize {
    workers.iter().fold(0_usize, |total, worker| {
        let stats = worker.native_workspace_stats();
        total
            .saturating_add(stats.retained_component_bytes())
            .saturating_add(stats.retained_tier1_bytes())
            .saturating_add(stats.retained_idwt_bytes())
    })
}

#[cfg(test)]
mod tests;
