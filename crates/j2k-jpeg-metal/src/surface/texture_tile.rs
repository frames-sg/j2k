// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::{Arc, Mutex};

use j2k_core::PixelFormat;
use metal::{Texture, TextureRef};

/// One decoded JPEG tile resident in a caller-owned Metal texture.
pub struct MetalTextureTile {
    pub(super) texture: Texture,
    pub(super) access_gate: Arc<Mutex<()>>,
    dimensions: (u32, u32),
    fmt: PixelFormat,
}

impl MetalTextureTile {
    pub(crate) fn new(
        texture: Texture,
        access_gate: Arc<Mutex<()>>,
        dimensions: (u32, u32),
        fmt: PixelFormat,
    ) -> Self {
        Self {
            texture,
            access_gate,
            dimensions,
            fmt,
        }
    }

    /// Return the raw Metal texture containing the decoded tile.
    ///
    /// # Safety
    ///
    /// The caller must synchronize every CPU and GPU access made through the
    /// returned texture or any handle cloned from it. The safe decode gate
    /// shared with the originating [`crate::MetalBatchTextureOutput`] cannot observe
    /// work submitted through raw handles. No raw access may overlap a safe
    /// decode through that output, one of its clones or subsets, or another
    /// tile derived from the same allocation.
    pub unsafe fn texture(&self) -> &TextureRef {
        self.texture_trusted()
    }

    pub(crate) fn texture_trusted(&self) -> &TextureRef {
        self.texture.as_ref()
    }

    /// Decoded tile dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    /// Decoded tile pixel format.
    pub fn pixel_format(&self) -> PixelFormat {
        self.fmt
    }
}

impl Clone for MetalTextureTile {
    fn clone(&self) -> Self {
        Self {
            texture: self.texture.clone(),
            access_gate: Arc::clone(&self.access_gate),
            dimensions: self.dimensions,
            fmt: self.fmt,
        }
    }
}
