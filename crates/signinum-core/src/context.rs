// SPDX-License-Identifier: Apache-2.0

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
}

impl CacheStats {
    pub const fn new(hits: u64, misses: u64) -> Self {
        Self { hits, misses }
    }
}

pub trait CodecContext: Default + Send {
    fn clear(&mut self);

    fn cache_stats(&self) -> CacheStats {
        CacheStats::default()
    }
}

#[derive(Debug, Default)]
pub struct DecoderContext<C: CodecContext> {
    codec: C,
}

impl<C: CodecContext> DecoderContext<C> {
    pub fn new() -> Self {
        Self {
            codec: C::default(),
        }
    }

    pub fn codec(&self) -> &C {
        &self.codec
    }

    pub fn codec_mut(&mut self) -> &mut C {
        &mut self.codec
    }

    pub fn clear(&mut self) {
        self.codec.clear();
    }

    pub fn cache_stats(&self) -> CacheStats {
        self.codec.cache_stats()
    }

    pub fn into_inner(self) -> C {
        self.codec
    }
}
