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
    /// Receiving-session policy; immutable parsed facts remain shared in `inner`.
    pub(super) cpu_execution: CpuExecution,
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

    /// Capacity the retained offset plan holds; none without a plan.
    pub(super) fn retained_allocation_bytes(
        &self,
    ) -> Result<usize, j2k_types::DecodePlanAllocationError> {
        match self {
            Self::MetadataOnly => Ok(0),
            Self::Classic(plan) => plan.retained_allocation_bytes(),
            Self::Htj2k(plan) => plan.retained_allocation_bytes(),
        }
    }
}

/// CPU execution derived under the receiving session's working-memory bound.
/// Device decoders ignore it: the retained codec plan is unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CpuExecution {
    /// Execute the retained offset plan; `bytes` is its planned peak.
    OffsetPlan { bytes: usize },
    /// Reparse and decode whole component planes; no offset plan exists.
    ComponentPlanes,
    /// The planned offset-plan execution exceeds the per-decode memory
    /// bound and the output has no streamed route, so the CPU decoder rejects
    /// the image before any decode work.
    ExceedsDecodeBound,
    /// A native Gray/RGB image whose offset plan exceeds the per-decode bound
    /// streams rows from the full-output region decode, whose working memory
    /// does not grow with the image.
    StreamedRows,
    /// A Gray8/RGB8 image whose offset plan fits the bound (`bytes` is its
    /// parallel peak) but exceeds the streaming threshold. It streams rows
    /// like [`Self::StreamedRows`], except as the only image of a Gray8
    /// group: alone in flight, its plan spreads one component's code-block
    /// rows over every core, which the streamed route does not match. If a
    /// preferred stream exceeds its budget, this admitted plan may retry.
    StreamedRowsOrPlan { bytes: usize },
}

impl CpuExecution {
    /// Conservative bytes for staged-window sizing: the planned peak of an
    /// offset plan, or one worst-case native decode.
    pub(super) const fn claim_bytes(self) -> usize {
        match self {
            Self::OffsetPlan { bytes } => bytes,
            // The streamed route has no planned peak yet; size it as one
            // worst-case native decode (measured: 95 MiB for a 5661 x 8287
            // RGB8 scan at default threads).
            // An optional plan fits the bound, so its peak is within this
            // amount too.
            Self::ComponentPlanes | Self::StreamedRows | Self::StreamedRowsOrPlan { .. } => {
                crate::batch::allocation::GENERIC_WORKER_CLAIM_BYTES
            }
            Self::ExceedsDecodeBound => 0,
        }
    }

    /// The planned peak of the offset plan this image may decode, if any:
    /// what the staged route, which decodes plans only, budgets for it.
    pub(super) const fn plan_bytes(self) -> Option<usize> {
        match self {
            Self::OffsetPlan { bytes } | Self::StreamedRowsOrPlan { bytes } => Some(bytes),
            Self::ComponentPlanes | Self::ExceedsDecodeBound | Self::StreamedRows => None,
        }
    }

    /// Whether the CPU decoder may stream this image's rows; a group holding
    /// such an image never takes the staged route.
    pub(super) const fn streams(self) -> bool {
        matches!(self, Self::StreamedRows | Self::StreamedRowsOrPlan { .. })
    }

    /// The error the offset-plan execution would report, raised up front.
    pub(super) fn decode_bound_error() -> crate::J2kError {
        crate::J2kError::from_native_decode_error(j2k_native::ValidationError::ImageTooLarge.into())
    }
}

#[derive(Debug)]
pub(super) struct PreparedImageInner {
    pub(super) bytes: Arc<[u8]>,
    pub(super) request: DecodeRequest,
    pub(super) source_index: usize,
    pub(super) decode_settings: DecodeSettings,
    pub(super) used_lenient_metadata_recovery: bool,
    pub(super) support: Arc<J2kSupportInfo>,
    pub(super) reference_grid: crate::J2kReferenceGrid,
    pub(super) plan: DeviceDecodePlan,
    pub(super) codestream_range: J2kCodestreamRange,
    pub(super) codec_plan: PreparedCodecPlan,
    pub(super) region_streams_rows: bool,
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

    /// Whether preparation used an explicitly documented lenient metadata recovery.
    #[doc(hidden)]
    #[must_use]
    pub fn used_lenient_metadata_recovery(&self) -> bool {
        self.inner.used_lenient_metadata_recovery
    }

    /// Parsed codestream and wrapper metadata.
    #[must_use]
    pub fn support(&self) -> &J2kSupportInfo {
        &self.inner.support
    }

    /// Original full-resolution image and tile coordinates.
    #[must_use]
    pub fn reference_grid(&self) -> crate::J2kReferenceGrid {
        self.inner.reference_grid
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

    pub(super) fn cpu_execution(&self) -> CpuExecution {
        self.cpu_execution
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

/// Borrowed images and source indices of one homogeneous prepared group.
#[derive(Clone, Copy)]
pub(super) struct PreparedBatchGroupRef<'a> {
    pub(super) info: &'a BatchGroupInfo,
    pub(super) images: &'a [super::CpuPreparedImage],
    pub(super) source_indices: &'a [usize],
    /// The decoding session's [`BatchDecodeOptions::max_working_memory_bytes`].
    pub(super) working_memory_limit: Option<core::num::NonZeroUsize>,
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
    /// Backend sessions preserve these validation and output settings so a
    /// reusable group cannot silently drift between strict and lenient decode.
    /// CPU admission and scheduling use the receiving session's working-memory
    /// limit and worker policy.
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
