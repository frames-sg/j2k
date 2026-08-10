// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::{Arc, Mutex, MutexGuard};

use j2k_core::{Downscale, PixelFormat, Rect};
use objc2_metal::{
    MTLDevice, MTLPixelFormat, MTLResource, MTLStorageMode, MTLTextureDescriptor, MTLTextureUsage,
};

use crate::metal_types::{Texture, TextureRef};

#[cfg(test)]
use super::MetalTextureTile;
use crate::error::metal_kernel_support_error;
use crate::{
    report_required_output_dimensions, scaled_dims, Error, JpegMetalResidentBatchReport,
    MetalBackendSession,
};

#[derive(Clone)]
/// Reusable caller-owned Metal textures for full-tile JPEG batch output.
pub struct MetalBatchTextureOutput {
    set: Arc<MetalBatchTextureSet>,
}

// SAFETY: Metal textures support cross-thread ownership, and every safe decode
// or access path is serialized by the allocation set's shared mutex. Raw
// texture access is unsafe and documents the remaining synchronization duty.
unsafe impl Send for MetalBatchTextureOutput {}
// SAFETY: Safe shared references expose immutable metadata or lock the shared
// access gate; no safe API mutates a texture without that serialization.
unsafe impl Sync for MetalBatchTextureOutput {}

struct MetalBatchTextureSet {
    textures: Vec<Texture>,
    access_gate: Arc<Mutex<()>>,
    dimensions: (u32, u32),
    fmt: PixelFormat,
    metal_fmt: MTLPixelFormat,
}

// SAFETY: Metal textures support cross-thread ownership. Every safe operation
// that can submit writes or expose readback holds `access_gate`, and raw
// texture access is unsafe with an explicit external synchronization contract.
unsafe impl Send for MetalBatchTextureSet {}
// SAFETY: Shared references expose immutable metadata or texture handles whose
// safe access is serialized by `access_gate`; no unsynchronized safe mutation
// is available through this allocation set.
unsafe impl Sync for MetalBatchTextureSet {}

impl MetalBatchTextureOutput {
    /// Allocate reusable private RGBA8 textures for `tile_capacity` full-size tiles.
    pub fn new_rgba8_tiles(
        session: &MetalBackendSession,
        dimensions: (u32, u32),
        tile_capacity: usize,
    ) -> Result<Self, Error> {
        if dimensions.0 == 0 || dimensions.1 == 0 || tile_capacity == 0 {
            return Err(Error::UnsupportedMetalRequest {
                reason:
                    "JPEG Metal batch texture output requires nonzero dimensions and tile capacity",
            });
        }

        // SAFETY: Dimensions were checked nonzero above, are representable as
        // `usize` on supported 64-bit macOS, and RGBA8 is valid for a 2D,
        // single-sample, non-mipmapped descriptor.
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::RGBA8Unorm,
                dimensions.0 as usize,
                dimensions.1 as usize,
                false,
            )
        };
        descriptor.setStorageMode(MTLStorageMode::Private);
        descriptor.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);

        let pixels = crate::batch_allocation::checked_count_product(
            dimensions.0 as usize,
            dimensions.1 as usize,
            "JPEG Metal batch texture pixels",
        )?;
        let tile_bytes = crate::batch_allocation::checked_count_product(
            pixels,
            PixelFormat::Rgba8.bytes_per_pixel(),
            "JPEG Metal batch texture bytes",
        )?;
        let heap_texture_bytes = session
            .device()
            .heapTextureSizeAndAlignWithDescriptor(&descriptor)
            .size;
        let planned_texture_bytes = crate::batch_allocation::checked_count_product(
            tile_bytes.max(heap_texture_bytes),
            tile_capacity,
            "JPEG Metal batch texture planned allocation",
        )?;
        let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
            "JPEG Metal batch texture collection",
        );
        budget.preflight(&[
            crate::batch_allocation::BatchMetadataRequest::of::<Texture>(tile_capacity),
            crate::batch_allocation::BatchMetadataRequest::of::<u8>(planned_texture_bytes),
        ])?;
        let mut textures = budget.try_vec(tile_capacity, "JPEG Metal batch texture handles")?;
        for _ in 0..tile_capacity {
            let texture = j2k_metal_support::checked_texture(session.device(), &descriptor)
                .map_err(|source| {
                    metal_kernel_support_error("JPEG Metal batch texture allocation", source)
                })?;
            let texture_bytes = texture.allocatedSize();
            budget.account_capacity::<u8>(texture_bytes)?;
            textures.push(texture);
        }

        Ok(Self {
            set: Arc::new(MetalBatchTextureSet {
                textures,
                access_gate: Arc::new(Mutex::new(())),
                dimensions,
                fmt: PixelFormat::Rgba8,
                metal_fmt: MTLPixelFormat::RGBA8Unorm,
            }),
        })
    }

    /// Ensure this output set can hold `tile_capacity` RGBA8 textures with `dimensions`.
    ///
    /// Existing textures are retained when they already have the requested
    /// layout and at least the requested capacity. Otherwise the texture set is
    /// replaced with new private RGBA8 textures.
    pub fn ensure_rgba8_tiles(
        &mut self,
        session: &MetalBackendSession,
        dimensions: (u32, u32),
        tile_capacity: usize,
    ) -> Result<(), Error> {
        if self.set.dimensions == dimensions
            && self.set.fmt == PixelFormat::Rgba8
            && self.set.metal_fmt == MTLPixelFormat::RGBA8Unorm
            && self.tile_capacity() >= tile_capacity
        {
            return Ok(());
        }

        let replacement = Self::new_rgba8_tiles(session, dimensions, tile_capacity)?;
        self.set = replacement.set;
        Ok(())
    }

    /// Ensure this output set fits a full-image scaled RGBA8 texture batch.
    pub fn ensure_rgba8_scaled_tiles(
        &mut self,
        session: &MetalBackendSession,
        full_dimensions: (u32, u32),
        scale: Downscale,
        tile_capacity: usize,
    ) -> Result<(), Error> {
        self.ensure_rgba8_tiles(session, scaled_dims(full_dimensions, scale), tile_capacity)
    }

    /// Ensure this output set fits a region-scaled RGBA8 texture batch.
    pub fn ensure_rgba8_region_scaled_tiles(
        &mut self,
        session: &MetalBackendSession,
        roi: Rect,
        scale: Downscale,
        tile_capacity: usize,
    ) -> Result<(), Error> {
        let scaled = roi.scaled_covering(scale);
        self.ensure_rgba8_tiles(session, (scaled.w, scaled.h), tile_capacity)
    }

    /// Ensure this texture set fits a preflighted RGB8 Metal resident batch.
    ///
    /// Ineligible reports return an error without replacing the existing
    /// textures. Eligible empty reports are a no-op.
    #[doc(hidden)]
    pub fn ensure_rgba8_batch_report(
        &mut self,
        session: &MetalBackendSession,
        report: &JpegMetalResidentBatchReport,
    ) -> Result<(), Error> {
        let Some(dimensions) = report_required_output_dimensions(report)? else {
            return Ok(());
        };
        self.ensure_rgba8_tiles(session, dimensions, report.required_tile_capacity())
    }

    /// Tile dimensions for this output allocation.
    pub fn dimensions(&self) -> (u32, u32) {
        self.set.dimensions
    }

    /// Pixel format for this output allocation.
    pub fn pixel_format(&self) -> PixelFormat {
        self.set.fmt
    }

    /// Metal pixel format for each backing texture.
    pub fn metal_pixel_format(&self) -> MTLPixelFormat {
        self.set.metal_fmt
    }

    /// Number of reusable tile texture slots.
    pub fn tile_capacity(&self) -> usize {
        self.set.textures.len()
    }

    /// Return a raw reusable output texture by tile slot.
    ///
    /// # Safety
    ///
    /// The caller must synchronize every CPU and GPU access made through the
    /// returned texture or any handle cloned from it. The internal safe-access
    /// gate cannot observe work submitted through raw handles. No such access
    /// may overlap a safe decode into this output, any clone or subset that
    /// shares its allocation gate, or access through a derived
    /// [`crate::MetalTextureTile`].
    pub unsafe fn texture(
        &self,
        index: usize,
    ) -> Option<&objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture>> {
        self.texture_trusted(index)
    }

    pub(crate) fn texture_trusted(&self, index: usize) -> Option<&TextureRef> {
        self.set
            .textures
            .get(index)
            .map(std::convert::AsRef::as_ref)
    }

    pub(crate) fn clone_texture_trusted(&self, index: usize) -> Option<Texture> {
        self.set.textures.get(index).cloned()
    }

    pub(crate) fn clone_access_gate(&self) -> Arc<Mutex<()>> {
        Arc::clone(&self.set.access_gate)
    }

    pub(crate) fn lock_for_safe_access(&self) -> Result<MutexGuard<'_, ()>, Error> {
        self.set.access_gate.lock().map_err(|_| Error::MetalKernel {
            message: "JPEG Metal batch texture output access gate was poisoned".to_string(),
        })
    }

    pub(crate) fn clone_slots(&self, indices: &[usize]) -> Result<Self, Error> {
        let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
            "JPEG Metal cloned texture slot collection",
        );
        let mut textures = budget.try_vec(indices.len(), "JPEG Metal cloned texture handles")?;
        for &index in indices {
            textures.push(
                self.clone_texture_trusted(index)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "JPEG Metal batch texture output slot was missing".to_string(),
                    })?,
            );
        }
        Ok(Self {
            set: Arc::new(MetalBatchTextureSet {
                textures,
                access_gate: Arc::clone(&self.set.access_gate),
                dimensions: self.set.dimensions,
                fmt: self.set.fmt,
                metal_fmt: self.set.metal_fmt,
            }),
        })
    }

    #[cfg(test)]
    pub(crate) fn shares_allocation_set_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.set, &other.set)
    }

    #[cfg(test)]
    pub(crate) fn shares_access_gate_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.set.access_gate, &other.set.access_gate)
    }

    #[cfg(test)]
    pub(crate) fn shares_access_gate_with_tile(&self, tile: &MetalTextureTile) -> bool {
        Arc::ptr_eq(&self.set.access_gate, &tile.access_gate)
    }
}
