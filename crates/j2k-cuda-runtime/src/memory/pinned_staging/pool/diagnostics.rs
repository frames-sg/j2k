// SPDX-License-Identifier: MIT OR Apache-2.0

const DEFAULT_MAX_CACHED_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_MAX_CACHED_BUFFERS: usize = 8;

#[doc(hidden)]
/// Retention limits for a CUDA context's reusable page-locked upload staging.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaPinnedUploadStagingPoolLimits {
    /// Maximum page-locked allocation bytes retained after an upload completes.
    pub max_cached_bytes: usize,
    /// Maximum number of page-locked allocations retained after upload completion.
    pub max_cached_buffers: usize,
}

impl Default for CudaPinnedUploadStagingPoolLimits {
    fn default() -> Self {
        Self {
            max_cached_bytes: DEFAULT_MAX_CACHED_BYTES,
            max_cached_buffers: DEFAULT_MAX_CACHED_BUFFERS,
        }
    }
}

#[doc(hidden)]
/// Retention diagnostics shared by every clone of one CUDA context.
///
/// Byte totals count CUDA page-locked allocations. They intentionally exclude
/// the small Rust `Vec` metadata used to index those allocations; ordinary
/// cache metadata is bounded by `limits.max_cached_buffers`. At least two fallible
/// quarantine slots are reserved before checkout; reservation scales to cover
/// every live checkout plus one eviction victim.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaPinnedUploadStagingPoolDiagnostics {
    /// Retention policy used by this context.
    pub limits: CudaPinnedUploadStagingPoolLimits,
    /// Completed page-locked allocations currently available for reuse.
    pub cached_buffers: usize,
    /// Actual page-locked allocation bytes currently available for reuse.
    pub cached_bytes: usize,
    /// Allocations quarantined because a CUDA free did not establish release.
    pub uncertain_buffers: usize,
    /// Page-locked bytes quarantined after an uncertain CUDA release.
    pub uncertain_bytes: usize,
    /// Page-locked allocations currently checked out by an upload transaction.
    pub active_buffers: usize,
    /// Actual page-locked bytes currently checked out by an upload transaction.
    pub active_bytes: usize,
    /// All page-locked allocation wrappers retained by this context.
    pub retained_buffers: usize,
    /// All page-locked allocation bytes retained by this context.
    pub retained_bytes: usize,
    /// Highest completed allocation count observed by this context.
    pub peak_cached_buffers: usize,
    /// Highest completed page-locked byte total observed by this context.
    pub peak_cached_bytes: usize,
    /// Highest uncertain-release allocation count observed by this context.
    pub peak_uncertain_buffers: usize,
    /// Highest uncertain-release page-locked byte total observed by this context.
    pub peak_uncertain_bytes: usize,
    /// Highest checked-out staging allocation count observed by this context.
    pub peak_active_buffers: usize,
    /// Highest checked-out page-locked byte total observed by this context.
    pub peak_active_bytes: usize,
    /// Highest total retained allocation count observed by this context.
    pub peak_retained_buffers: usize,
    /// Highest total retained page-locked byte count observed by this context.
    pub peak_retained_bytes: usize,
    /// Completed allocations evicted to admit newer completed staging.
    pub evicted_buffers: usize,
    /// Completed allocations not retained because one allocation exceeded the policy.
    pub rejected_buffers: usize,
    /// Completed allocations not retained after host cache-metadata allocation failed.
    pub metadata_failures: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct PinnedUploadStagingPoolMetrics {
    pub(super) peak_cached_buffers: usize,
    pub(super) peak_cached_bytes: usize,
    pub(super) peak_uncertain_buffers: usize,
    pub(super) peak_uncertain_bytes: usize,
    pub(super) peak_active_buffers: usize,
    pub(super) peak_active_bytes: usize,
    pub(super) peak_retained_buffers: usize,
    pub(super) peak_retained_bytes: usize,
    pub(super) evicted_buffers: usize,
    pub(super) rejected_buffers: usize,
    pub(super) metadata_failures: usize,
}
