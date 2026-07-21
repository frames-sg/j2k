// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared Metal batch result and metadata contracts.

use super::{
    BatchDecodeOptions, BatchGroupInfo, BatchLayout, Buffer, Error, IndexedBatchError,
    J2kDecodeWarning, PixelFormat, PreparedBatchGroup, Rect, ResidentMetalImage, Surface,
};

pub(super) fn validate_group_contract(info: &BatchGroupInfo) -> Result<PixelFormat, Error> {
    if !matches!(info.layout, BatchLayout::Nchw | BatchLayout::Nhwc) {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal batch received an unknown output layout",
        });
    }
    info.native_pixel_format()
        .ok_or(Error::UnsupportedMetalRequest {
            reason: "J2K Metal batch metadata contains an unsupported color/sample combination",
        })
}

/// Codec metadata released only after a caller-owned Metal group destination
/// has completed and its device status has been validated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetalBatchGroupCompletion {
    pub(super) decoded_rects: Vec<Rect>,
    pub(super) warnings: Vec<Vec<J2kDecodeWarning>>,
}

impl MetalBatchGroupCompletion {
    pub(super) fn from_prepared(group: &PreparedBatchGroup, options: BatchDecodeOptions) -> Self {
        let decoded_rects = group
            .images()
            .iter()
            .map(|image| image.plan().output_rect())
            .collect();
        let warnings = group
            .images()
            .iter()
            .map(|_| {
                if options.settings.lenient_tolerance_enabled() {
                    vec![J2kDecodeWarning::LenientDecodeMode]
                } else {
                    Vec::new()
                }
            })
            .collect();
        Self {
            decoded_rects,
            warnings,
        }
    }

    /// Actual decoded source rectangle for every completed destination item.
    #[must_use]
    pub fn decoded_rects(&self) -> &[Rect] {
        &self.decoded_rects
    }

    /// Non-fatal codec warnings for every completed destination item.
    #[must_use]
    pub fn warnings(&self) -> &[Vec<J2kDecodeWarning>] {
        &self.warnings
    }

    /// Consume the completion into rectangles and warnings in batch order.
    #[must_use]
    pub fn into_parts(self) -> (Vec<Rect>, Vec<Vec<J2kDecodeWarning>>) {
        (self.decoded_rects, self.warnings)
    }
}

/// One completed dense codec-owned Metal batch allocation.
///
/// The byte ordering is described by the owning [`MetalBatchGroup`]'s
/// [`BatchGroupInfo`], including its NCHW or NHWC layout. The allocation is
/// logically immutable after codec completion; raw Metal access therefore
/// remains an explicit unsafe interop boundary.
#[cfg(target_os = "macos")]
#[derive(Clone)]
pub struct MetalResidentBatch {
    pub(super) storage: ResidentMetalImage,
}

#[cfg(target_os = "macos")]
impl core::fmt::Debug for MetalResidentBatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetalResidentBatch")
            .field("device_registry_id", &self.device_registry_id())
            .field("byte_offset", &self.byte_offset())
            .field("byte_len", &self.byte_len())
            .field("image_count", &self.image_count())
            .field("image_stride_bytes", &self.image_stride_bytes())
            .finish_non_exhaustive()
    }
}

#[cfg(target_os = "macos")]
impl MetalResidentBatch {
    /// Byte offset of the dense group inside the retained allocation.
    #[must_use]
    pub const fn byte_offset(&self) -> usize {
        self.storage.layout().byte_offset()
    }

    /// Number of bytes occupied by the complete dense group.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.storage.layout().byte_len()
    }

    /// Number of images in the dense batch dimension.
    #[must_use]
    pub const fn image_count(&self) -> usize {
        self.storage.layout().image_count()
    }

    /// Byte distance between consecutive images in the dense batch.
    #[must_use]
    pub const fn image_stride_bytes(&self) -> usize {
        self.storage.layout().image_stride_bytes()
    }

    /// Registry identifier of the Metal device that owns the allocation.
    #[must_use]
    pub fn device_registry_id(&self) -> u64 {
        self.storage.device_registry_id()
    }

    /// Borrow the completed Metal allocation for read-only GPU interop.
    ///
    /// # Safety
    ///
    /// The caller may bind the handle only for reads and must retain this
    /// value (or a clone) until that GPU work completes. No CPU or GPU writer
    /// may access the allocation while any resident-batch owner exists.
    #[must_use]
    pub unsafe fn metal_buffer(&self) -> &Buffer {
        // SAFETY: the caller accepts the immutable raw-handle contract above.
        unsafe { self.storage.raw_buffer() }
    }
}

/// One successfully decoded homogeneous Metal-resident output group.
pub struct MetalBatchGroup {
    pub(super) info: BatchGroupInfo,
    pub(super) source_indices: Vec<usize>,
    pub(super) decoded_rects: Vec<Rect>,
    pub(super) warnings: Vec<Vec<J2kDecodeWarning>>,
    pub(super) surfaces: Vec<Surface>,
    #[cfg(target_os = "macos")]
    pub(super) resident_batch: MetalResidentBatch,
}

/// Owned parts returned when consuming one homogeneous Metal batch group.
pub type MetalBatchGroupParts = (
    BatchGroupInfo,
    Vec<usize>,
    Vec<Rect>,
    Vec<Vec<J2kDecodeWarning>>,
    Vec<Surface>,
);

/// Failure while executing one homogeneous Metal group.
///
/// No partially written output from the affected group is exposed. Other
/// prepared groups may still succeed when the retained Metal session remains
/// usable.
#[derive(Debug, thiserror::Error)]
#[error("Metal batch group containing source indices {source_indices:?} failed: {source}")]
pub struct MetalBatchGroupError {
    pub(super) source_indices: Vec<usize>,
    #[source]
    pub(super) source: Box<Error>,
}

impl MetalBatchGroupError {
    pub(super) fn new(group: &PreparedBatchGroup, source: Error) -> Self {
        Self {
            source_indices: group.source_indices().to_vec(),
            source: Box::new(source),
        }
    }

    /// Original input indices whose dense group output was discarded.
    #[must_use]
    pub fn source_indices(&self) -> &[usize] {
        &self.source_indices
    }

    /// Strict Metal adapter or runtime failure for this group.
    #[must_use]
    pub fn source(&self) -> &Error {
        &self.source
    }

    /// Consume the group failure into affected indices and its source.
    #[must_use]
    pub fn into_parts(self) -> (Vec<usize>, Error) {
        (self.source_indices, *self.source)
    }
}

impl core::fmt::Debug for MetalBatchGroup {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut debug = f.debug_struct("MetalBatchGroup");
        debug
            .field("info", &self.info)
            .field("source_indices", &self.source_indices)
            .field("decoded_rects", &self.decoded_rects)
            .field("warnings", &self.warnings)
            .field("surface_count", &self.surfaces.len());
        #[cfg(target_os = "macos")]
        debug.field("resident_batch", &self.resident_batch);
        debug.finish()
    }
}

impl MetalBatchGroup {
    /// Shared native-width output metadata.
    pub fn info(&self) -> &BatchGroupInfo {
        &self.info
    }

    /// Original input indices in the resident batch dimension.
    pub fn source_indices(&self) -> &[usize] {
        &self.source_indices
    }

    /// Actual decoded rectangle for every resident image.
    pub fn decoded_rects(&self) -> &[Rect] {
        &self.decoded_rects
    }

    /// Non-fatal warnings for every resident image.
    pub fn warnings(&self) -> &[Vec<J2kDecodeWarning>] {
        &self.warnings
    }

    /// Interleaved Metal-resident image views in group order.
    ///
    /// Gray groups and NHWC color groups expose one convenience [`Surface`]
    /// per image. NCHW color groups return an empty slice because a [`Surface`]
    /// promises interleaved pixel semantics; use [`Self::resident_batch`] for
    /// their dense planar allocation.
    pub fn surfaces(&self) -> &[Surface] {
        &self.surfaces
    }

    /// Completed dense codec-owned Metal storage for this group.
    ///
    /// Its byte ordering is exactly `self.info().layout`; no decoded host
    /// transfer or final device copy is performed to construct this owner.
    #[cfg(target_os = "macos")]
    #[must_use]
    pub fn resident_batch(&self) -> Option<&MetalResidentBatch> {
        Some(&self.resident_batch)
    }

    /// Consume this group and retain its dense codec-owned Metal storage.
    #[cfg(target_os = "macos")]
    #[must_use]
    pub fn into_resident_batch(self) -> Option<MetalResidentBatch> {
        Some(self.resident_batch)
    }

    /// Consume this group into metadata, indices, rectangles, warnings, and surfaces.
    pub fn into_parts(self) -> MetalBatchGroupParts {
        (
            self.info,
            self.source_indices,
            self.decoded_rects,
            self.warnings,
            self.surfaces,
        )
    }
}

/// Successful resident groups plus indexed input preparation failures.
#[derive(Debug)]
pub struct MetalBatchDecodeResult {
    pub(super) groups: Vec<MetalBatchGroup>,
    pub(super) errors: Vec<IndexedBatchError>,
    pub(super) group_errors: Vec<MetalBatchGroupError>,
}

impl MetalBatchDecodeResult {
    /// Successfully decoded homogeneous groups.
    pub fn groups(&self) -> &[MetalBatchGroup] {
        &self.groups
    }

    /// Indexed parsing, planning, and representability failures.
    pub fn errors(&self) -> &[IndexedBatchError] {
        &self.errors
    }

    /// Homogeneous groups that failed during Metal execution.
    #[must_use]
    pub fn group_errors(&self) -> &[MetalBatchGroupError] {
        &self.group_errors
    }

    /// Consume the result into resident groups, indexed preparation failures,
    /// and homogeneous execution failures.
    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        Vec<MetalBatchGroup>,
        Vec<IndexedBatchError>,
        Vec<MetalBatchGroupError>,
    ) {
        (self.groups, self.errors, self.group_errors)
    }
}
