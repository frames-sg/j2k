// SPDX-License-Identifier: Apache-2.0

use signinum_core::ScratchPool;

#[derive(Debug, Default)]
/// Scratch storage reused by DEFLATE decoding.
pub struct DeflatePool {
    pub(crate) scratch: Vec<u8>,
}

#[derive(Debug, Default)]
/// Scratch storage reused by Zstandard decoding.
pub struct ZstdPool {
    pub(crate) scratch: Vec<u8>,
}

#[derive(Debug, Default)]
/// Scratch storage reused by LZW decoding.
pub struct LzwPool {
    pub(crate) scratch: Vec<u8>,
}

#[derive(Debug, Default, Clone, Copy)]
/// Zero-sized scratch pool for codecs that do not allocate.
pub struct NoPool;

impl DeflatePool {
    #[must_use]
    /// Create an empty DEFLATE scratch pool.
    pub fn new() -> Self {
        Self::default()
    }
}

impl ZstdPool {
    #[must_use]
    /// Create an empty Zstandard scratch pool.
    pub fn new() -> Self {
        Self::default()
    }
}

impl LzwPool {
    #[must_use]
    /// Create an empty LZW scratch pool.
    pub fn new() -> Self {
        Self::default()
    }
}

impl ScratchPool for DeflatePool {
    fn bytes_allocated(&self) -> usize {
        self.scratch.capacity()
    }

    fn reset(&mut self) {
        self.scratch.clear();
    }
}

impl ScratchPool for ZstdPool {
    fn bytes_allocated(&self) -> usize {
        self.scratch.capacity()
    }

    fn reset(&mut self) {
        self.scratch.clear();
    }
}

impl ScratchPool for LzwPool {
    fn bytes_allocated(&self) -> usize {
        self.scratch.capacity()
    }

    fn reset(&mut self) {
        self.scratch.clear();
    }
}

impl ScratchPool for NoPool {
    fn bytes_allocated(&self) -> usize {
        0
    }

    fn reset(&mut self) {}
}
