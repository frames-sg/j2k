// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    EncodedImage, IndexedBatchError, J2kDecodeWarning, PreparedBatch, PreparedImageOperation, Rect,
};
use j2k_core::{BatchInfrastructureError, PixelFormat};

use super::{
    decode_warnings, grayscale_inputs, group_pixel_format, native_color_group_storage,
    native_color_inputs, validate_layout, CudaBatchDecodeResult, CudaBatchDecoder, CudaBatchError,
    CudaBatchGroup, CudaBatchGroupError, CudaGroupView, CudaResidentBatchBuffer, Error, Surface,
};
use crate::CudaHtj2kProfileReport;

struct ResidentGroupMetadata {
    info: j2k::BatchGroupInfo,
    source_indices: Vec<usize>,
    decoded_rects: Vec<Rect>,
    warnings: Vec<Vec<J2kDecodeWarning>>,
}

impl ResidentGroupMetadata {
    fn from_view(group: CudaGroupView<'_>) -> Result<Self, Error> {
        let mut budget =
            crate::allocation::HostPhaseBudget::new("CUDA resident batch result metadata");
        Ok(Self {
            info: group.info.clone(),
            source_indices: budget.try_clone_slice(group.source_indices)?,
            decoded_rects: budget
                .try_collect_exact(group.images().map(super::input::GroupImage::output_rect))?,
            warnings: decode_warnings(group)?,
        })
    }

    fn finish(
        self,
        surfaces: Vec<Surface>,
        dense_output: CudaResidentBatchBuffer,
        profile_report: CudaHtj2kProfileReport,
    ) -> CudaBatchGroup {
        CudaBatchGroup {
            info: self.info,
            source_indices: self.source_indices,
            decoded_rects: self.decoded_rects,
            warnings: self.warnings,
            surfaces,
            profile_report,
            dense_output,
        }
    }
}

enum SubmittedResidentCodecGroup {
    Grayscale(crate::decoder::grayscale_batch::SubmittedGrayscaleResidentBatch),
    Color(crate::decoder::SubmittedNativeColorResidentBatch),
}

impl SubmittedResidentCodecGroup {
    fn is_complete(&self) -> Result<bool, Error> {
        match self {
            Self::Grayscale(pending) => pending.is_complete(),
            Self::Color(pending) => pending.is_complete(),
        }
    }

    fn finish(
        self,
        info: &j2k::BatchGroupInfo,
    ) -> Result<
        (
            Vec<Surface>,
            CudaResidentBatchBuffer,
            CudaHtj2kProfileReport,
        ),
        Error,
    > {
        match self {
            Self::Grayscale(pending) => {
                let (output, report) = pending.finish()?;
                Ok((
                    output.surfaces,
                    CudaResidentBatchBuffer {
                        buffer: output.buffer,
                        ranges: output.ranges,
                    },
                    report,
                ))
            }
            Self::Color(pending) => {
                let (output, report) = pending.finish()?;
                let fmt = group_pixel_format(info)?;
                let (surfaces, dense_output) = native_color_group_storage(info, fmt, output);
                Ok((surfaces, dense_output, report))
            }
        }
    }
}

struct SubmittedResidentGroup {
    metadata: ResidentGroupMetadata,
    pending: SubmittedResidentCodecGroup,
}

/// Submitted codec-owned CUDA resident batch whose final work may be pending.
///
/// Every homogeneous group owns its final CUDA allocation before submission.
/// [`Self::wait`] exposes those allocations only after final-store completion
/// and entropy-status validation. Dropping this guard safely retires all work.
#[must_use = "submitted CUDA resident decode must be retained or waited"]
pub struct SubmittedCudaResidentBatch {
    pending: Vec<SubmittedResidentGroup>,
    errors: Vec<IndexedBatchError>,
    group_errors: Vec<CudaBatchGroupError>,
}

impl core::fmt::Debug for SubmittedCudaResidentBatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SubmittedCudaResidentBatch")
            .field("pending_groups", &self.pending.len())
            .field("errors", &self.errors)
            .field("group_errors", &self.group_errors)
            .finish()
    }
}

impl SubmittedCudaResidentBatch {
    /// Number of successfully submitted homogeneous groups still retained.
    #[must_use]
    pub fn pending_group_count(&self) -> usize {
        self.pending.len()
    }

    /// Query whether every successfully submitted group has completed.
    pub fn is_complete(&self) -> Result<bool, CudaBatchError> {
        for group in &self.pending {
            let source_indices = group.metadata.source_indices.clone();
            if !group
                .pending
                .is_complete()
                .map_err(|source| CudaBatchError::GroupExecution {
                    source_indices,
                    source: Box::new(source),
                })?
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Wait once for all group final stores and return codec-owned outputs.
    pub fn wait(mut self) -> Result<CudaBatchDecodeResult, CudaBatchError> {
        let mut groups = Vec::new();
        groups.try_reserve_exact(self.pending.len()).map_err(|_| {
            BatchInfrastructureError::HostAllocationFailed {
                what: "completed CUDA resident batch groups",
                bytes: self
                    .pending
                    .len()
                    .saturating_mul(core::mem::size_of::<CudaBatchGroup>()),
            }
        })?;
        let mut fatal = None;
        for submitted in self.pending.drain(..) {
            let SubmittedResidentGroup { metadata, pending } = submitted;
            let source_indices = metadata.source_indices.clone();
            match pending.finish(&metadata.info) {
                Ok((surfaces, dense_output, profile_report)) => {
                    groups.push(metadata.finish(surfaces, dense_output, profile_report));
                }
                Err(source) if source.session_is_unusable() => {
                    if fatal.is_none() {
                        fatal = Some(CudaBatchError::GroupExecution {
                            source_indices,
                            source: Box::new(source),
                        });
                    }
                }
                Err(source) => self
                    .group_errors
                    .push(CudaBatchGroupError::from_parts(source_indices, source)),
            }
        }
        if let Some(error) = fatal {
            return Err(error);
        }
        Ok(CudaBatchDecodeResult {
            groups,
            errors: core::mem::take(&mut self.errors),
            group_errors: core::mem::take(&mut self.group_errors),
        })
    }
}

impl CudaBatchDecoder {
    /// Prepare and submit a codec-owned CUDA resident batch.
    ///
    /// Waits on the host like [`Self::submit_prepared`].
    pub fn submit_batch(
        &mut self,
        inputs: Vec<EncodedImage>,
    ) -> Result<SubmittedCudaResidentBatch, CudaBatchError> {
        let prepared = self.prepare(inputs)?;
        self.submit_prepared(&prepared)
    }

    /// Submit reusable prepared inputs to codec-owned CUDA output, leaving each
    /// group's final store pending.
    ///
    /// Recoverable execution failures remain group-local and do not suppress
    /// later groups. No decoded host staging or final device copy is used.
    ///
    /// RGB and RGBA groups larger than 32 images decode in successive 32-image
    /// chunks into one dense allocation, so decode scratch stays within the
    /// session cache. Submission blocks the host until every chunk except the
    /// last has completed.
    pub fn submit_prepared(
        &mut self,
        prepared: &PreparedBatch,
    ) -> Result<SubmittedCudaResidentBatch, CudaBatchError> {
        let groups = prepared.groups();
        self.submit_group_views(
            groups.len(),
            groups.iter().map(CudaGroupView::from),
            prepared.errors(),
        )
    }

    /// Submit exact operations to codec-owned CUDA output, leaving each
    /// group's final store pending. See [`Self::decode_operations`].
    ///
    /// # Errors
    /// As [`Self::decode_operations`].
    pub fn submit_operations(
        &mut self,
        operations: &[PreparedImageOperation<'_>],
    ) -> Result<SubmittedCudaResidentBatch, CudaBatchError> {
        let prepared = j2k::prepare_device_operations(operations, self.options)?;
        let groups = prepared.groups();
        self.submit_group_views(
            groups.len(),
            groups.iter().map(CudaGroupView::from),
            prepared.errors(),
        )
    }

    /// Submit homogeneous groups in order, keeping indexed preparation errors
    /// and recoverable per-group failures.
    pub(super) fn submit_group_views<'g>(
        &mut self,
        group_count: usize,
        groups: impl Iterator<Item = CudaGroupView<'g>>,
        preparation_errors: &[IndexedBatchError],
    ) -> Result<SubmittedCudaResidentBatch, CudaBatchError> {
        let mut pending = Vec::new();
        pending.try_reserve_exact(group_count).map_err(|_| {
            BatchInfrastructureError::HostAllocationFailed {
                what: "submitted CUDA resident groups",
                bytes: group_count.saturating_mul(core::mem::size_of::<SubmittedResidentGroup>()),
            }
        })?;
        let mut group_errors = Vec::new();
        group_errors.try_reserve_exact(group_count).map_err(|_| {
            BatchInfrastructureError::HostAllocationFailed {
                what: "submitted CUDA resident group failures",
                bytes: group_count.saturating_mul(core::mem::size_of::<CudaBatchGroupError>()),
            }
        })?;
        let mut errors = Vec::new();
        errors
            .try_reserve_exact(preparation_errors.len())
            .map_err(|_| BatchInfrastructureError::HostAllocationFailed {
                what: "submitted CUDA resident indexed errors",
                bytes: preparation_errors
                    .len()
                    .saturating_mul(core::mem::size_of::<IndexedBatchError>()),
            })?;
        errors.extend_from_slice(preparation_errors);
        for group in groups {
            match group
                .check_cancelled()
                .and_then(|()| self.submit_resident_group(group))
            {
                Ok(submitted) => pending.push(submitted),
                Err(source) if source.session_is_unusable() => {
                    return Err(CudaBatchError::group(group.source_indices, source));
                }
                Err(source) => {
                    group_errors.push(CudaBatchGroupError::new(group.source_indices, source));
                }
            }
        }
        Ok(SubmittedCudaResidentBatch {
            pending,
            errors,
            group_errors,
        })
    }

    fn submit_resident_group(
        &mut self,
        group: CudaGroupView<'_>,
    ) -> Result<SubmittedResidentGroup, Error> {
        let fmt = group_pixel_format(group.info)?;
        validate_layout(group.info)?;
        let pending = if matches!(
            fmt,
            PixelFormat::Rgb8
                | PixelFormat::Rgb16
                | PixelFormat::RgbI16
                | PixelFormat::Rgba8
                | PixelFormat::Rgba16
                | PixelFormat::RgbaI16
        ) {
            let inputs = native_color_inputs(group)?;
            SubmittedResidentCodecGroup::Color(
                crate::decoder::submit_native_color_resident_prepared_batch(
                    &inputs,
                    &mut self.session,
                    fmt,
                    group.info.layout,
                )?,
            )
        } else if matches!(
            fmt,
            PixelFormat::Gray8 | PixelFormat::Gray16 | PixelFormat::GrayI16
        ) {
            let inputs = grayscale_inputs(group);
            SubmittedResidentCodecGroup::Grayscale(
                crate::decoder::grayscale_batch::submit_grayscale_cuda_resident_prepared_batch(
                    &inputs,
                    group.fallback_settings(),
                    &mut self.session,
                    fmt,
                )?,
            )
        } else {
            return Err(Error::capability_rejected(
                j2k_core::CapabilityRejection::unsupported_format(
                    "resident CUDA batch submission requires exact Gray, RGB, or RGBA output",
                ),
            ));
        };
        Ok(SubmittedResidentGroup {
            metadata: ResidentGroupMetadata::from_view(group)?,
            pending,
        })
    }
}
