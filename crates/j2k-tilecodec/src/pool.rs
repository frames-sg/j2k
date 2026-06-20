// SPDX-License-Identifier: Apache-2.0

use j2k_core::ScratchPool;

macro_rules! vec_scratch_pool {
    ($(#[$meta:meta])* $name:ident, $new_doc:literal) => {
        $(#[$meta])*
        #[derive(Debug, Default)]
        pub struct $name {
            pub(crate) scratch: Vec<u8>,
        }

        impl $name {
            #[must_use]
            #[doc = $new_doc]
            pub fn new() -> Self {
                Self::default()
            }
        }

        impl ScratchPool for $name {
            fn bytes_allocated(&self) -> usize {
                self.scratch.capacity()
            }

            fn reset(&mut self) {
                self.scratch.clear();
            }
        }
    };
}

vec_scratch_pool!(
    /// Scratch storage reused by DEFLATE decoding.
    DeflatePool,
    "Create an empty DEFLATE scratch pool."
);

vec_scratch_pool!(
    /// Scratch storage reused by Zstandard decoding.
    ZstdPool,
    "Create an empty Zstandard scratch pool."
);

vec_scratch_pool!(
    /// Scratch storage reused by LZW decoding.
    LzwPool,
    "Create an empty LZW scratch pool."
);

#[derive(Debug, Default, Clone, Copy)]
/// Zero-sized scratch pool for codecs that do not allocate.
pub struct NoPool;

impl ScratchPool for NoPool {
    fn bytes_allocated(&self) -> usize {
        0
    }

    fn reset(&mut self) {}
}
