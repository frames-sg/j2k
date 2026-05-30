// SPDX-License-Identifier: Apache-2.0

/// Cache hit/miss counters reported by codec contexts.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    /// Number of cache lookups that reused existing state.
    pub hits: u64,
    /// Number of cache lookups that had to build new state.
    pub misses: u64,
}

impl CacheStats {
    /// Construct cache statistics from explicit counters.
    pub const fn new(hits: u64, misses: u64) -> Self {
        Self { hits, misses }
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

/// Wrapper that owns codec context state for repeated decode calls.
#[derive(Debug, Default)]
pub struct DecoderContext<C: CodecContext> {
    codec: C,
}

impl<C: CodecContext> DecoderContext<C> {
    /// Construct an empty decoder context.
    pub fn new() -> Self {
        Self {
            codec: C::default(),
        }
    }

    /// Borrow the codec-specific context.
    pub fn codec(&self) -> &C {
        &self.codec
    }

    /// Mutably borrow the codec-specific context.
    pub fn codec_mut(&mut self) -> &mut C {
        &mut self.codec
    }

    /// Clear cached codec state.
    pub fn clear(&mut self) {
        self.codec.clear();
    }

    /// Return cache counters from the codec-specific context.
    pub fn cache_stats(&self) -> CacheStats {
        self.codec.cache_stats()
    }

    /// Consume the wrapper and return the codec-specific context.
    pub fn into_inner(self) -> C {
        self.codec
    }
}
