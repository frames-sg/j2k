// SPDX-License-Identifier: MIT OR Apache-2.0

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[doc(hidden)]
/// Observable retention, high-water, and admission state for a JPEG plan cache.
pub struct JpegPlanCacheDiagnostics {
    /// Current retained entries.
    pub entries: usize,
    /// Current retained bytes, including entry-vector capacity and owner graphs.
    pub retained_bytes: usize,
    /// Allocator-reported byte capacity of the flat entry vector.
    pub metadata_capacity_bytes: usize,
    /// Highest retained-byte count observed by this cache.
    pub peak_bytes: usize,
    /// Highest retained-entry count observed by this cache.
    pub peak_entries: usize,
    /// Successful full-input lookups.
    pub hits: u64,
    /// Failed full-input lookups.
    pub misses: u64,
    /// Entries removed by deterministic LRU admission.
    pub evictions: u64,
    /// Plans rejected because their retained graph could not fit by itself.
    pub oversized_rejections: u64,
    /// Plans rejected because the cache entry limit is zero.
    pub disabled_rejections: u64,
    /// Failed fallible entry-metadata reservations.
    pub metadata_allocation_failures: u64,
    /// Configured maximum number of entries.
    pub entry_limit: usize,
    /// Configured maximum retained host bytes.
    pub host_byte_limit: usize,
}

impl JpegPlanCacheDiagnostics {
    pub(super) const fn new(entry_limit: usize, host_byte_limit: usize) -> Self {
        Self {
            entries: 0,
            retained_bytes: 0,
            metadata_capacity_bytes: 0,
            peak_bytes: 0,
            peak_entries: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
            oversized_rejections: 0,
            disabled_rejections: 0,
            metadata_allocation_failures: 0,
            entry_limit,
            host_byte_limit,
        }
    }
}
