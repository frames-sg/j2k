// SPDX-License-Identifier: MIT OR Apache-2.0

/// Cache hit/miss counters reported by codec contexts.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    /// Number of cache lookups that reused existing state.
    pub hits: u64,
    /// Number of cache lookups that had to build new state.
    pub misses: u64,
    /// Number of currently occupied cache slots.
    pub occupied_slots: u64,
    /// Number of cache entries evicted by later insertions.
    pub evictions: u64,
}

impl CacheStats {
    /// Construct cache statistics from explicit counters.
    #[must_use]
    pub const fn new(hits: u64, misses: u64) -> Self {
        Self {
            hits,
            misses,
            occupied_slots: 0,
            evictions: 0,
        }
    }

    /// Construct cache statistics from full counters.
    #[must_use]
    pub const fn with_slots(hits: u64, misses: u64, occupied_slots: u64, evictions: u64) -> Self {
        Self {
            hits,
            misses,
            occupied_slots,
            evictions,
        }
    }
}

/// Reusable codec state cached across decode calls.
pub trait CodecContext: Default + Send {
    /// Drop cached state while keeping the context reusable.
    fn clear(&mut self);

    /// Return current cache counters, when the codec tracks them.
    fn cache_stats(&self) -> CacheStats {
        CacheStats::default()
    }
}

#[cfg(test)]
mod tests;
