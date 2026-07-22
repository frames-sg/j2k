// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{sync::Arc, vec::Vec};

use j2k_core::{BatchInfrastructureError, Downscale, TileLayout};

use super::{
    prepare_batch, prepare_batch_from_images, BatchDecodeOptions, BatchGroupInfo, DecodeRequest,
    EncodedImage, IndexedBatchError, J2kCodestreamRange, PreparationDepth, PreparedClassicPlan,
    PreparedHtj2kPlan,
};
use crate::{DecodeSettings, DeviceDecodePlan, J2kSupportInfo};

/// Cheaply cloneable preparation result for one image.
#[derive(Debug, Clone)]
pub struct PreparedImage {
    pub(super) inner: Arc<PreparedImageInner>,
}

#[derive(Debug)]
pub(super) enum PreparedCodecPlan {
    MetadataOnly,
    Htj2k(PreparedHtj2kPlan),
    Classic(PreparedClassicPlan),
}

impl PreparedCodecPlan {
    pub(super) const fn preparation_depth(&self) -> PreparationDepth {
        match self {
            Self::MetadataOnly => PreparationDepth::MetadataOnly,
            Self::Htj2k(_) => PreparationDepth::Htj2kOffsetPlan,
            Self::Classic(_) => PreparationDepth::ClassicOffsetPlan,
        }
    }
}

#[derive(Debug)]
pub(super) struct PreparedImageInner {
    pub(super) bytes: Arc<[u8]>,
    pub(super) request: DecodeRequest,
    pub(super) source_index: usize,
    pub(super) decode_settings: DecodeSettings,
    pub(super) support: Arc<J2kSupportInfo>,
    pub(super) plan: DeviceDecodePlan,
    pub(super) codestream_range: J2kCodestreamRange,
    pub(super) codec_plan: PreparedCodecPlan,
}

impl PreparedImage {
    /// Original encoded bytes.
    #[must_use]
    pub fn bytes(&self) -> &Arc<[u8]> {
        &self.inner.bytes
    }

    /// Caller decode request.
    #[must_use]
    pub fn request(&self) -> DecodeRequest {
        self.inner.request
    }

    /// Original caller input index.
    #[must_use]
    pub fn source_index(&self) -> usize {
        self.inner.source_index
    }

    /// Validation policy used to parse and build this retained execution plan.
    #[must_use]
    pub fn decode_settings(&self) -> DecodeSettings {
        self.inner.decode_settings
    }

    /// Parsed codestream and wrapper metadata.
    #[must_use]
    pub fn support(&self) -> &J2kSupportInfo {
        &self.inner.support
    }

    /// Normalized source and output geometry.
    #[must_use]
    pub fn plan(&self) -> DeviceDecodePlan {
        self.inner.plan
    }

    /// Raw codestream range inside [`Self::bytes`].
    #[must_use]
    pub fn codestream_range(&self) -> J2kCodestreamRange {
        self.inner.codestream_range
    }

    /// Whether this image retains metadata only or a parse-free codec offset plan.
    #[must_use]
    pub fn preparation_depth(&self) -> PreparationDepth {
        self.inner.codec_plan.preparation_depth()
    }

    /// Reusable HTJ2K geometry and payload references, when supported.
    /// Payload offsets are absolute byte ranges inside [`Self::bytes`],
    /// including any JP2/JPH container prefix.
    #[must_use]
    pub fn htj2k_plan(&self) -> Option<&PreparedHtj2kPlan> {
        match &self.inner.codec_plan {
            PreparedCodecPlan::Htj2k(plan) => Some(plan),
            PreparedCodecPlan::MetadataOnly | PreparedCodecPlan::Classic(_) => None,
        }
    }

    /// Reusable classic JPEG 2000 geometry and payload-fragment references,
    /// when supported. Fragment offsets are absolute ranges inside
    /// [`Self::bytes`], including any JP2 container prefix.
    #[must_use]
    pub fn classic_plan(&self) -> Option<&PreparedClassicPlan> {
        match &self.inner.codec_plan {
            PreparedCodecPlan::Classic(plan) => Some(plan),
            PreparedCodecPlan::MetadataOnly | PreparedCodecPlan::Htj2k(_) => None,
        }
    }

    pub(super) fn codec_plan(&self) -> &PreparedCodecPlan {
        &self.inner.codec_plan
    }
}

/// One homogeneous set of prepared images.
#[derive(Debug)]
pub struct PreparedBatchGroup {
    pub(super) info: BatchGroupInfo,
    pub(super) options: BatchDecodeOptions,
    pub(super) execution_shape: BatchExecutionShape,
    pub(super) images: Vec<PreparedImage>,
    pub(super) source_indices: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BatchExecutionShape {
    pub(super) source_dimensions: (u32, u32),
    pub(super) source_rect_dimensions: (u32, u32),
    pub(super) scale: Downscale,
    pub(super) tile_layout: Option<TileLayout>,
    pub(super) resolution_levels: u8,
    pub(super) preparation_depth: PreparationDepth,
}

impl PreparedBatchGroup {
    /// Shared output metadata and grouping key.
    #[must_use]
    pub fn info(&self) -> &BatchGroupInfo {
        &self.info
    }

    /// Decode policy captured when every image in this group was prepared.
    ///
    /// Backend sessions use this value instead of their current defaults so a
    /// reusable group cannot silently drift between strict and lenient decode.
    #[must_use]
    pub const fn options(&self) -> BatchDecodeOptions {
        self.options
    }

    /// Prepared images in caller order.
    #[must_use]
    pub fn images(&self) -> &[PreparedImage] {
        &self.images
    }

    /// Original caller indices in group order.
    #[must_use]
    pub fn source_indices(&self) -> &[usize] {
        &self.source_indices
    }
}

/// Cheaply cloneable parsed and grouped batch reusable across decode calls.
#[derive(Debug, Clone)]
pub struct PreparedBatch {
    pub(super) groups: Arc<[PreparedBatchGroup]>,
    pub(super) errors: Arc<[IndexedBatchError]>,
    pub(super) options: BatchDecodeOptions,
}

/// Common synchronous boundary implemented by persistent codec batch sessions.
///
/// GPU implementations may choose a resident or pending associated output;
/// synchronization and submission details remain backend-specific.
pub trait BatchDecoder {
    /// Backend-specific successful output.
    type Output;
    /// Backend-specific infrastructure or execution error.
    type Error: From<BatchInfrastructureError>;

    /// Preparation and output policy retained by this persistent session.
    fn options(&self) -> BatchDecodeOptions;

    /// Inspect and group owned inputs without consuming their encoded byte owners.
    fn prepare_batch(&self, inputs: Vec<EncodedImage>) -> Result<PreparedBatch, Self::Error> {
        prepare_batch(inputs, self.options()).map_err(Self::Error::from)
    }

    /// Regroup caller-supplied prepared images without reparsing their encoded bytes.
    fn prepare_prepared_images(
        &self,
        images: Vec<PreparedImage>,
    ) -> Result<PreparedBatch, Self::Error> {
        prepare_batch_from_images(images, self.options()).map_err(Self::Error::from)
    }

    /// Prepare and decode one owned batch through the common session boundary.
    fn decode_batch(&mut self, inputs: Vec<EncodedImage>) -> Result<Self::Output, Self::Error> {
        let prepared = self.prepare_batch(inputs)?;
        self.decode_prepared(&prepared)
    }

    /// Regroup and decode caller-supplied prepared images without reparsing them.
    fn decode_prepared_images(
        &mut self,
        images: Vec<PreparedImage>,
    ) -> Result<Self::Output, Self::Error> {
        let prepared = self.prepare_prepared_images(images)?;
        self.decode_prepared(&prepared)
    }

    /// Decode a reusable prepared batch without consuming its encoded inputs or plans.
    fn decode_prepared(&mut self, prepared: &PreparedBatch) -> Result<Self::Output, Self::Error>;
}

impl PreparedBatch {
    /// Homogeneous prepared groups in first-occurrence order.
    #[must_use]
    pub fn groups(&self) -> &[PreparedBatchGroup] {
        &self.groups
    }

    /// Indexed preflight failures.
    #[must_use]
    pub fn errors(&self) -> &[IndexedBatchError] {
        &self.errors
    }

    /// Options captured when this batch was prepared.
    #[must_use]
    pub const fn options(&self) -> BatchDecodeOptions {
        self.options
    }

    /// Consume this handle into its shared group and error owners.
    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        Arc<[PreparedBatchGroup]>,
        Arc<[IndexedBatchError]>,
        BatchDecodeOptions,
    ) {
        (self.groups, self.errors, self.options)
    }
}
