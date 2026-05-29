// SPDX-License-Identifier: Apache-2.0

/// Cache hit/miss counters reported by reusable codec contexts.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct CacheStats {
    /// Number of cache lookups served from cached state.
    pub hits: u64,
    /// Number of cache lookups that required recomputation.
    pub misses: u64,
}

impl CacheStats {
    /// Create cache statistics from hit and miss counters.
    pub const fn new(hits: u64, misses: u64) -> Self {
        Self { hits, misses }
    }
}

/// Codec-owned state that can be reused across independent decodes.
pub trait CodecContext: Default + Send {
    /// Clear reusable state while keeping allocated storage available.
    fn clear(&mut self);

    /// Return cache counters for this context.
    fn cache_stats(&self) -> CacheStats {
        CacheStats::default()
    }
}

/// Generic decoder context wrapper used by tile-batch APIs.
#[derive(Debug, Default)]
pub struct DecoderContext<C: CodecContext> {
    codec: C,
}

impl<C: CodecContext> DecoderContext<C> {
    /// Create a context using the codec context default value.
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

    /// Clear codec-specific reusable state.
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
