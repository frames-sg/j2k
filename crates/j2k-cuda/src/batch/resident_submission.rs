// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    EncodedImage, IndexedBatchError, J2kDecodeWarning, PreparedBatch, PreparedBatchGroup, Rect,
};
use j2k_core::{BatchInfrastructureError, PixelFormat};

use super::{
    decode_warnings, group_pixel_format, native_color_group_storage, native_color_inputs,
    native_decode_settings, native_referenced_classic_plan, native_referenced_htj2k_plan,
    validate_layout, CudaBatchDecodeResult, CudaBatchDecoder, CudaBatchError, CudaBatchGroup,
    CudaBatchGroupError, CudaResidentBatchBuffer, Error, Surface,
};

struct ResidentGroupMetadata {
    info: j2k::BatchGroupInfo,
    source_indices: Vec<usize>,
    decoded_rects: Vec<Rect>,
    warnings: Vec<Vec<J2kDecodeWarning>>,
}

impl ResidentGroupMetadata {
    fn from_prepared(group: &PreparedBatchGroup) -> Self {
        Self {
            info: group.info().clone(),
            source_indices: group.source_indices().to_vec(),
            decoded_rects: group
                .images()
                .iter()
                .map(|image| image.plan().output_rect())
                .collect(),
            warnings: decode_warnings(group.options(), group.images().len()),
        }
    }

    fn finish(
        self,
        surfaces: Vec<Surface>,
        dense_output: CudaResidentBatchBuffer,
    ) -> CudaBatchGroup {
        CudaBatchGroup {
            info: self.info,
            source_indices: self.source_indices,
            decoded_rects: self.decoded_rects,
            warnings: self.warnings,
            surfaces,
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
    ) -> Result<(Vec<Surface>, CudaResidentBatchBuffer), Error> {
        match self {
            Self::Grayscale(pending) => {
                let (output, _report) = pending.finish()?;
                Ok((
                    output.surfaces,
                    CudaResidentBatchBuffer {
                        buffer: output.buffer,
                        ranges: output.ranges,
                    },
                ))
            }
            Self::Color(pending) => {
                let (output, _report) = pending.finish()?;
                let fmt = group_pixel_format(info)?;
                Ok(native_color_group_storage(info, fmt, output))
            }
        }
    }
}

struct SubmittedResidentGroup {
    metadata: ResidentGroupMetadata,
    pending: SubmittedResidentCodecGroup,
}

/// Asynchronously submitted codec-owned CUDA resident batch.
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
                Ok((surfaces, dense_output)) => {
                    groups.push(metadata.finish(surfaces, dense_output));
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
    /// Prepare and asynchronously submit a codec-owned CUDA resident batch.
    pub fn submit_batch(
        &mut self,
        inputs: Vec<EncodedImage>,
    ) -> Result<SubmittedCudaResidentBatch, CudaBatchError> {
        let prepared = self.prepare(inputs)?;
        self.submit_prepared(&prepared)
    }

    /// Asynchronously submit reusable prepared inputs to codec-owned CUDA output.
    ///
    /// Recoverable execution failures remain group-local and do not suppress
    /// later groups. No decoded host staging or final device copy is used.
    pub fn submit_prepared(
        &mut self,
        prepared: &PreparedBatch,
    ) -> Result<SubmittedCudaResidentBatch, CudaBatchError> {
        let mut pending = Vec::new();
        pending
            .try_reserve_exact(prepared.groups().len())
            .map_err(|_| BatchInfrastructureError::HostAllocationFailed {
                what: "submitted CUDA resident groups",
                bytes: prepared
                    .groups()
                    .len()
                    .saturating_mul(core::mem::size_of::<SubmittedResidentGroup>()),
            })?;
        let mut group_errors = Vec::new();
        group_errors
            .try_reserve_exact(prepared.groups().len())
            .map_err(|_| BatchInfrastructureError::HostAllocationFailed {
                what: "submitted CUDA resident group failures",
                bytes: prepared
                    .groups()
                    .len()
                    .saturating_mul(core::mem::size_of::<CudaBatchGroupError>()),
            })?;
        for group in prepared.groups() {
            match self.submit_resident_group(group) {
                Ok(submitted) => pending.push(submitted),
                Err(source) if source.session_is_unusable() => {
                    return Err(CudaBatchError::group(group, source));
                }
                Err(source) => group_errors.push(CudaBatchGroupError::new(group, source)),
            }
        }
        Ok(SubmittedCudaResidentBatch {
            pending,
            errors: prepared.errors().to_vec(),
            group_errors,
        })
    }

    fn submit_resident_group(
        &mut self,
        group: &PreparedBatchGroup,
    ) -> Result<SubmittedResidentGroup, Error> {
        let fmt = group_pixel_format(group.info())?;
        validate_layout(group.info())?;
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
                    group.info().layout,
                )?,
            )
        } else if matches!(
            fmt,
            PixelFormat::Gray8 | PixelFormat::Gray16 | PixelFormat::GrayI16
        ) {
            let inputs = resident_grayscale_inputs(group)?;
            SubmittedResidentCodecGroup::Grayscale(
                crate::decoder::grayscale_batch::submit_grayscale_cuda_resident_prepared_batch(
                    &inputs,
                    native_decode_settings(group.options().settings),
                    &mut self.session,
                    fmt,
                )?,
            )
        } else {
            return Err(Error::UnsupportedCudaRequest {
                reason: "resident CUDA batch submission requires exact Gray, RGB, or RGBA output",
            });
        };
        Ok(SubmittedResidentGroup {
            metadata: ResidentGroupMetadata::from_prepared(group),
            pending,
        })
    }
}

fn resident_grayscale_inputs(
    group: &PreparedBatchGroup,
) -> Result<Vec<crate::decoder::grayscale_batch::GrayscaleBatchInput<'_>>, Error> {
    group
        .images()
        .iter()
        .zip(group.source_indices().iter().copied())
        .map(|(image, source_index)| {
            let referenced_plan = image
                .htj2k_plan()
                .map(native_referenced_htj2k_plan)
                .transpose()?;
            let referenced_classic_plan = image
                .classic_plan()
                .map(native_referenced_classic_plan)
                .transpose()?;
            Ok(crate::decoder::grayscale_batch::GrayscaleBatchInput {
                source_index,
                bytes: image.bytes().as_ref(),
                device_plan: Some(image.plan()),
                referenced_plan,
                referenced_classic_plan,
            })
        })
        .collect()
}
