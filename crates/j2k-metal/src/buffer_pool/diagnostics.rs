// SPDX-License-Identifier: MIT OR Apache-2.0

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
/// Retention and high-water counters for one Metal scratch-buffer pool.
pub struct MetalBufferPoolDiagnostics {
    /// Bytes currently retained in completed reusable buffers.
    pub cached_bytes: usize,
    /// Completed reusable buffers currently retained.
    pub cached_buffers: usize,
    /// Allocator-reported capacity of the flat host metadata owner.
    pub metadata_capacity: usize,
    /// Highest completed-buffer byte count retained by this pool.
    pub peak_cached_bytes: usize,
    /// Highest completed-buffer count retained by this pool.
    pub peak_cached_buffers: usize,
    /// Oldest completed buffers evicted to admit more useful entries.
    pub evictions: usize,
    /// Completed buffers deliberately declined instead of retained.
    pub rejections: usize,
    /// Metadata reservations rejected by the host allocator.
    pub metadata_failures: usize,
    /// Caller size records that disagreed with the Metal allocation length.
    pub size_mismatches: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
/// Separate diagnostics for private and shared Metal scratch retention.
pub struct MetalBufferPoolsDiagnostics {
    /// Private-storage scratch pool counters.
    pub private: MetalBufferPoolDiagnostics,
    /// Shared-storage scratch pool counters.
    pub shared: MetalBufferPoolDiagnostics,
}
