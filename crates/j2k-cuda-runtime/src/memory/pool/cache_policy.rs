// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    size_buckets::CudaBufferPoolSizeBuckets, CudaBufferPool, CudaBufferPoolFree,
    CudaBufferPoolInner, CudaBufferPoolState,
};
use crate::{
    allocation::host_allocation_error, context::CudaContext, error::select_resource_release_error,
    memory::CudaDeviceBuffer, CudaError,
};
use std::sync::{Arc, Mutex};

const DEFAULT_MAX_CACHED_BYTES: usize = 512 * 1024 * 1024;
const DEFAULT_MAX_CACHED_BUFFERS: usize = 128;
const DEFAULT_MAX_SIZE_BUCKETS: usize = 64;

#[doc(hidden)]
/// Retention limits shared by first-fit and best-fit CUDA buffer pools.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaBufferPoolLimits {
    /// Maximum total byte length of completed device allocations retained for reuse.
    pub max_cached_bytes: usize,
    /// Maximum number of completed device allocations retained for reuse.
    pub max_cached_buffers: usize,
    /// Maximum number of distinct allocation sizes retained by a best-fit pool.
    pub max_size_buckets: usize,
}

impl Default for CudaBufferPoolLimits {
    fn default() -> Self {
        Self {
            max_cached_bytes: DEFAULT_MAX_CACHED_BYTES,
            max_cached_buffers: DEFAULT_MAX_CACHED_BUFFERS,
            max_size_buckets: DEFAULT_MAX_SIZE_BUCKETS,
        }
    }
}

#[doc(hidden)]
/// Shared retention and high-water diagnostics for a CUDA buffer pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaBufferPoolDiagnostics {
    /// Retention policy used by this pool.
    pub limits: CudaBufferPoolLimits,
    /// Completed buffers currently available for reuse.
    pub cached_buffers: usize,
    /// Actual device-allocation bytes currently available for reuse.
    pub cached_bytes: usize,
    /// Distinct allocation sizes currently retained by a best-fit pool.
    pub cached_size_buckets: usize,
    /// Buffers retained until queued work establishes completion.
    pub deferred_buffers: usize,
    /// Actual device-allocation bytes retained until completion.
    pub deferred_bytes: usize,
    /// Active guards preventing deferred buffers from becoming reusable.
    pub reuse_holds: usize,
    /// Highest completed-buffer count observed by this pool.
    pub peak_cached_buffers: usize,
    /// Highest completed allocation-byte total observed by this pool.
    pub peak_cached_bytes: usize,
    /// Highest best-fit bucket count observed by this pool.
    pub peak_cached_size_buckets: usize,
    /// Highest deferred-buffer count observed by this pool.
    pub peak_deferred_buffers: usize,
    /// Highest deferred allocation-byte total observed by this pool.
    pub peak_deferred_bytes: usize,
    /// Completed buffers evicted to admit a newer completed buffer.
    pub evicted_buffers: usize,
    /// Completed buffers rejected because one allocation cannot fit the policy.
    pub rejected_buffers: usize,
    /// Completed buffers not retained after host cache-metadata allocation failed.
    pub metadata_failures: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct CudaBufferPoolMetrics {
    pub(super) peak_cached_buffers: usize,
    pub(super) peak_cached_bytes: usize,
    pub(super) peak_cached_size_buckets: usize,
    pub(super) peak_deferred_buffers: usize,
    pub(super) peak_deferred_bytes: usize,
    pub(super) evicted_buffers: usize,
    pub(super) rejected_buffers: usize,
    pub(super) metadata_failures: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CacheInventory {
    buffers: usize,
    bytes: usize,
    size_buckets: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CacheAdmissionDecision {
    Admit,
    Evict,
    Reject,
}

impl CudaBufferPool {
    /// Create a new first-fit pool for `context` with bounded default retention.
    pub fn new(context: CudaContext) -> Self {
        Self::with_limits(context, CudaBufferPoolLimits::default())
    }

    #[doc(hidden)]
    /// Create a first-fit pool with explicit retention limits.
    pub fn with_limits(context: CudaContext, limits: CudaBufferPoolLimits) -> Self {
        Self::with_free(context, limits, CudaBufferPoolFree::FirstFit(Vec::new()))
    }

    pub(in super::super) fn new_size_buckets(context: CudaContext) -> Self {
        Self::best_fit_with_limits(context, CudaBufferPoolLimits::default())
    }

    #[doc(hidden)]
    /// Create a best-fit pool with explicit retention and size-bucket limits.
    pub fn best_fit_with_limits(context: CudaContext, limits: CudaBufferPoolLimits) -> Self {
        Self::with_free(
            context,
            limits,
            CudaBufferPoolFree::SizeBuckets(CudaBufferPoolSizeBuckets::new()),
        )
    }

    fn with_free(
        context: CudaContext,
        limits: CudaBufferPoolLimits,
        free: CudaBufferPoolFree,
    ) -> Self {
        Self {
            inner: Arc::new(CudaBufferPoolInner {
                context,
                limits,
                state: Mutex::new(CudaBufferPoolState {
                    free,
                    deferred: Vec::new(),
                    deferred_bytes: 0,
                    reuse_holds: 0,
                    metrics: CudaBufferPoolMetrics::default(),
                }),
            }),
        }
    }

    #[doc(hidden)]
    /// Snapshot diagnostics shared by every clone of this pool.
    pub fn diagnostics(&self) -> Result<CudaBufferPoolDiagnostics, CudaError> {
        self.inner
            .context
            .inner
            .ensure_resource_lifetime_available()?;
        let state = self
            .inner
            .state
            .lock()
            .map_err(|error| CudaError::StatePoisoned {
                message: error.to_string(),
            })?;
        let inventory = state.free.inventory();
        Ok(CudaBufferPoolDiagnostics {
            limits: self.inner.limits,
            cached_buffers: inventory.buffers,
            cached_bytes: inventory.bytes,
            cached_size_buckets: inventory.size_buckets,
            deferred_buffers: state.deferred.len(),
            deferred_bytes: state.deferred_bytes,
            reuse_holds: state.reuse_holds,
            peak_cached_buffers: state.metrics.peak_cached_buffers,
            peak_cached_bytes: state.metrics.peak_cached_bytes,
            peak_cached_size_buckets: state.metrics.peak_cached_size_buckets,
            peak_deferred_buffers: state.metrics.peak_deferred_buffers,
            peak_deferred_bytes: state.metrics.peak_deferred_bytes,
            evicted_buffers: state.metrics.evicted_buffers,
            rejected_buffers: state.metrics.rejected_buffers,
            metadata_failures: state.metrics.metadata_failures,
        })
    }
}

impl CudaBufferPoolInner {
    pub(super) fn recycle_completed_buffer(
        &self,
        buffer: CudaDeviceBuffer,
    ) -> Result<(), CudaError> {
        let mut candidate = Some(buffer);
        loop {
            let Some(candidate_buffer) = candidate.as_ref() else {
                return Err(CudaError::InternalInvariant {
                    what: "completed CUDA cache candidate ownership was lost",
                });
            };
            let candidate_bytes = candidate_buffer.byte_len();
            let mut state = match self.state.lock() {
                Ok(state) => state,
                Err(error) => {
                    // The cache state is unknown after poisoning. Conservatively
                    // retain the allocation token instead of risking an unsafe free.
                    std::mem::forget(candidate.take());
                    return Err(CudaError::StatePoisoned {
                        message: error.to_string(),
                    });
                }
            };
            let inventory = state.free.inventory();
            let adds_size_bucket = state.free.candidate_adds_size_bucket(candidate_bytes);
            match cache_admission_decision(
                self.limits,
                inventory,
                candidate_bytes,
                adds_size_bucket,
            ) {
                CacheAdmissionDecision::Reject => {
                    state.metrics.rejected_buffers =
                        state.metrics.rejected_buffers.saturating_add(1);
                    let rejected = candidate.take();
                    drop(state);
                    // This path is reached only after completion is known. Drop
                    // outside the pool lock so CUDA release cannot block peers.
                    drop(rejected);
                    return self.context.inner.ensure_resource_lifetime_available();
                }
                CacheAdmissionDecision::Evict => {
                    let Some(evicted) = state.free.evict_deterministic() else {
                        let unretained = candidate.take();
                        drop(state);
                        drop(unretained);
                        return Err(CudaError::InternalInvariant {
                            what: "CUDA buffer cache selected eviction without a victim",
                        });
                    };
                    state.metrics.evicted_buffers = state.metrics.evicted_buffers.saturating_add(1);
                    drop(state);
                    // Eviction is legal only for completed buffers. Release the
                    // driver allocation without holding the shared cache mutex.
                    drop(evicted);
                    if let Err(error) = self.context.inner.ensure_resource_lifetime_available() {
                        // A failed device release quarantines the context. Keep
                        // the still-owned candidate from attempting another free.
                        std::mem::forget(candidate.take());
                        return Err(error);
                    }
                }
                CacheAdmissionDecision::Admit => {
                    let Some(admitted) = candidate.take() else {
                        drop(state);
                        return Err(CudaError::InternalInvariant {
                            what: "CUDA buffer cache admitted a missing candidate",
                        });
                    };
                    if let Err((error, unretained)) = state.free.try_recycle(admitted) {
                        state.metrics.metadata_failures =
                            state.metrics.metadata_failures.saturating_add(1);
                        drop(state);
                        drop(unretained);
                        if let Err(release_error) =
                            self.context.inner.ensure_resource_lifetime_available()
                        {
                            return Err(select_resource_release_error(error, release_error));
                        }
                        return Err(error);
                    }
                    observe_cache_high_water(&mut state);
                    return Ok(());
                }
            }
        }
    }
}

impl CudaBufferPoolFree {
    fn inventory(&self) -> CacheInventory {
        match self {
            Self::FirstFit(buffers) => CacheInventory {
                buffers: buffers.len(),
                bytes: buffers.iter().fold(0usize, |total, buffer| {
                    total.saturating_add(buffer.byte_len())
                }),
                size_buckets: 0,
            },
            Self::SizeBuckets(buckets) => CacheInventory {
                buffers: buckets.cached_count(),
                bytes: buckets.cached_bytes(),
                size_buckets: buckets.bucket_count(),
            },
        }
    }

    fn candidate_adds_size_bucket(&self, candidate_bytes: usize) -> bool {
        match self {
            Self::FirstFit(_) => false,
            Self::SizeBuckets(buckets) => !buckets.contains_size(candidate_bytes),
        }
    }

    fn evict_deterministic(&mut self) -> Option<CudaDeviceBuffer> {
        match self {
            Self::FirstFit(buffers) => (!buffers.is_empty()).then(|| buffers.remove(0)),
            Self::SizeBuckets(buckets) => buckets.evict_largest_oldest(),
        }
    }

    fn try_recycle(
        &mut self,
        buffer: CudaDeviceBuffer,
    ) -> Result<(), (CudaError, CudaDeviceBuffer)> {
        match self {
            Self::FirstFit(buffers) => {
                if buffers.try_reserve(1).is_err() {
                    let error =
                        host_allocation_error::<CudaDeviceBuffer>(buffers.len().saturating_add(1));
                    return Err((error, buffer));
                }
                buffers.push(buffer);
                Ok(())
            }
            Self::SizeBuckets(buckets) => buckets.try_recycle(buffer),
        }
    }
}

fn cache_admission_decision(
    limits: CudaBufferPoolLimits,
    inventory: CacheInventory,
    candidate_bytes: usize,
    adds_size_bucket: bool,
) -> CacheAdmissionDecision {
    if limits.max_cached_buffers == 0
        || candidate_bytes > limits.max_cached_bytes
        || (adds_size_bucket && limits.max_size_buckets == 0)
    {
        return CacheAdmissionDecision::Reject;
    }

    let next_buffers = inventory.buffers.checked_add(1);
    let next_bytes = inventory.bytes.checked_add(candidate_bytes);
    let next_buckets = inventory
        .size_buckets
        .checked_add(usize::from(adds_size_bucket));
    let fits = next_buffers.is_some_and(|count| count <= limits.max_cached_buffers)
        && next_bytes.is_some_and(|bytes| bytes <= limits.max_cached_bytes)
        && next_buckets.is_some_and(|count| count <= limits.max_size_buckets);
    if fits {
        CacheAdmissionDecision::Admit
    } else if inventory.buffers == 0 {
        CacheAdmissionDecision::Reject
    } else {
        CacheAdmissionDecision::Evict
    }
}

pub(super) fn checked_deferred_bytes(current: usize, added: usize) -> Result<usize, CudaError> {
    current
        .checked_add(added)
        .ok_or(CudaError::InternalInvariant {
            what: "CUDA deferred buffer byte accounting overflow",
        })
}

pub(super) fn observe_deferred_high_water(state: &mut CudaBufferPoolState) {
    state.metrics.peak_deferred_buffers = state
        .metrics
        .peak_deferred_buffers
        .max(state.deferred.len());
    state.metrics.peak_deferred_bytes = state.metrics.peak_deferred_bytes.max(state.deferred_bytes);
}

fn observe_cache_high_water(state: &mut CudaBufferPoolState) {
    let inventory = state.free.inventory();
    state.metrics.peak_cached_buffers = state.metrics.peak_cached_buffers.max(inventory.buffers);
    state.metrics.peak_cached_bytes = state.metrics.peak_cached_bytes.max(inventory.bytes);
    state.metrics.peak_cached_size_buckets = state
        .metrics
        .peak_cached_size_buckets
        .max(inventory.size_buckets);
}

#[cfg(test)]
mod tests;
