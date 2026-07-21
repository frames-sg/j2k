// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal submission ownership, allocation, and completion.

use super::{
    validate_group_contract, BatchColor, BatchDecodeOptions, BatchGroupInfo, BatchLayout, Buffer,
    DeviceRef, DeviceSubmission, Error, IndexedBatchError, J2kDecodeWarning,
    MetalBatchDecodeResult, MetalBatchGroup, MetalBatchGroupCompletion, MetalBatchGroupError,
    MetalImageDestination, MetalImageLayout, MetalResidentBatch, PixelFormat, PreparedBatchGroup,
    Rect, ResidentMetalImage, Surface,
};

/// Pending direct decode of one prepared group into caller-owned Metal storage.
///
/// This guard retains the exclusive destination and all codec scratch owners
/// until completion. [`Self::wait`] reports command or codec-status failures.
/// Dropping it safely retires the committed work before releasing the
/// destination, so the decoder session remains reusable.
#[cfg(target_os = "macos")]
pub struct SubmittedMetalGroupDecodeInto {
    pub(super) submission: crate::compute::SubmittedDirectDestination,
    pub(super) destination: MetalImageDestination,
    pub(super) completion: MetalBatchGroupCompletion,
}

#[cfg(target_os = "macos")]
impl core::fmt::Debug for SubmittedMetalGroupDecodeInto {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SubmittedMetalGroupDecodeInto")
            .field("pending", &true)
            .finish_non_exhaustive()
    }
}

#[cfg(target_os = "macos")]
impl SubmittedMetalGroupDecodeInto {
    /// Enqueue a GPU-side wait on a same-device consumer command queue.
    ///
    /// The wait command is committed before this method returns, so commands
    /// subsequently submitted to `consumer_queue` cannot observe the decoded
    /// destination before the codec producer signals completion. This method
    /// performs no CPU wait.
    pub fn enqueue_consumer_wait(
        &mut self,
        consumer_queue: &metal::CommandQueueRef,
    ) -> Result<(), Error> {
        self.submission.enqueue_consumer_wait(consumer_queue)
    }

    /// Wait for GPU completion, validate group status, and release exclusive
    /// access to the destination.
    pub fn wait(self) -> Result<MetalBatchGroupCompletion, Error> {
        let Self {
            submission,
            destination,
            completion,
        } = self;
        let result = submission.wait();
        drop(destination);
        result?;
        Ok(completion)
    }
}

#[cfg(target_os = "macos")]
impl DeviceSubmission for SubmittedMetalGroupDecodeInto {
    type Output = MetalBatchGroupCompletion;
    type Error = Error;

    fn wait(self) -> Result<Self::Output, Self::Error> {
        Self::wait(self)
    }
}

/// Pending shared-contract decode into codec-owned Metal-resident groups.
///
/// Every group is committed before this guard is returned. [`Self::wait`]
/// retires all groups, preserves indexed preparation failures and non-fatal
/// group failures, then exposes completed resident surfaces. Dropping the
/// guard also retires every committed command buffer before releasing its
/// output allocation.
#[cfg(target_os = "macos")]
pub struct SubmittedMetalPreparedBatch {
    pub(super) pending_groups: Vec<SubmittedMetalResidentGroup>,
    pub(super) errors: Vec<IndexedBatchError>,
    pub(super) group_errors: Vec<MetalBatchGroupError>,
}

#[cfg(target_os = "macos")]
impl core::fmt::Debug for SubmittedMetalPreparedBatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SubmittedMetalPreparedBatch")
            .field("pending_groups", &self.pending_groups.len())
            .field("errors", &self.errors.len())
            .field("group_errors", &self.group_errors.len())
            .finish_non_exhaustive()
    }
}

#[cfg(target_os = "macos")]
impl SubmittedMetalPreparedBatch {
    /// Number of successfully committed resident groups awaiting completion.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pending_groups.len()
    }

    /// Whether no resident group was committed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pending_groups.is_empty()
    }

    /// Retire every group and collect the shared codec batch result.
    pub fn wait(mut self) -> Result<MetalBatchDecodeResult, Error> {
        let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
            "J2K submitted prepared Metal batch completion",
        );
        let mut groups = budget.try_vec(
            self.pending_groups.len(),
            "J2K submitted prepared Metal output groups",
        )?;
        let mut first_fatal = None;
        for pending in self.pending_groups.drain(..) {
            match pending.wait() {
                Ok(group) => groups.push(group),
                Err((_, source)) if source.session_is_unusable() => {
                    if first_fatal.is_none() {
                        first_fatal = Some(source);
                    }
                }
                Err((source_indices, source)) => self.group_errors.push(MetalBatchGroupError {
                    source_indices,
                    source,
                }),
            }
        }
        if let Some(source) = first_fatal {
            return Err(*source);
        }
        Ok(MetalBatchDecodeResult {
            groups,
            errors: core::mem::take(&mut self.errors),
            group_errors: core::mem::take(&mut self.group_errors),
        })
    }
}

#[cfg(target_os = "macos")]
impl DeviceSubmission for SubmittedMetalPreparedBatch {
    type Output = MetalBatchDecodeResult;
    type Error = Error;

    fn wait(self) -> Result<Self::Output, Self::Error> {
        Self::wait(self)
    }
}

#[cfg(target_os = "macos")]
pub(super) struct SubmittedMetalResidentGroup {
    pub(super) metadata: MetalResidentGroupMetadata,
    pub(super) submission: crate::compute::SubmittedDirectDestination,
    pub(super) destination: MetalImageDestination,
    pub(super) output: Buffer,
    pub(super) layout: MetalImageLayout,
}

#[cfg(target_os = "macos")]
pub(super) struct MetalResidentGroupMetadata {
    info: BatchGroupInfo,
    source_indices: Vec<usize>,
    decoded_rects: Vec<Rect>,
    warnings: Vec<Vec<J2kDecodeWarning>>,
}

#[cfg(target_os = "macos")]
impl MetalResidentGroupMetadata {
    pub(super) fn from_prepared(group: &PreparedBatchGroup, options: BatchDecodeOptions) -> Self {
        let MetalBatchGroupCompletion {
            decoded_rects,
            warnings,
        } = MetalBatchGroupCompletion::from_prepared(group, options);
        Self {
            info: group.info().clone(),
            source_indices: group.source_indices().to_vec(),
            decoded_rects,
            warnings,
        }
    }
}

#[cfg(target_os = "macos")]
pub(super) struct CodecOwnedMetalGroupDestination {
    pub(super) destination: MetalImageDestination,
    pub(super) output: Buffer,
    pub(super) layout: MetalImageLayout,
}

#[cfg(target_os = "macos")]
pub(super) fn validate_codec_owned_resident_group(
    group: &PreparedBatchGroup,
) -> Result<PixelFormat, Error> {
    let format = validate_group_contract(group.info())?;
    if group.images().is_empty() {
        return Err(Error::MetalStateInvariant {
            state: "J2K submitted codec-owned Metal group",
            reason: "prepared homogeneous group contains no images",
        });
    }
    Ok(format)
}

#[cfg(target_os = "macos")]
pub(super) fn allocate_codec_owned_group_destination(
    device: &DeviceRef,
    group: &PreparedBatchGroup,
    fmt: PixelFormat,
) -> Result<CodecOwnedMetalGroupDestination, Error> {
    let dimensions = group.info().dimensions;
    let row_bytes = usize::try_from(dimensions.0)
        .ok()
        .and_then(|width| width.checked_mul(fmt.bytes_per_pixel()))
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K submitted codec-owned Metal row size overflow".to_string(),
        })?;
    let image_bytes = row_bytes
        .checked_mul(dimensions.1 as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K submitted codec-owned Metal image size overflow".to_string(),
        })?;
    let total_bytes = image_bytes
        .checked_mul(group.images().len())
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K submitted codec-owned Metal group size overflow".to_string(),
        })?;
    let output =
        j2k_metal_support::checked_shared_buffer(device, total_bytes).map_err(|source| {
            crate::error::metal_kernel_support_error(
                "J2K submitted codec-owned Metal group allocation failed",
                source,
            )
        })?;
    let layout = MetalImageLayout::new_batch(
        0,
        dimensions,
        row_bytes,
        fmt,
        group.images().len(),
        image_bytes,
    )
    .map_err(|source| {
        crate::error::metal_kernel_support_error(
            "J2K submitted codec-owned Metal group layout failed",
            source,
        )
    })?;
    // SAFETY: `output` is a fresh codec-owned allocation and remains
    // exclusively retained by the pending group until GPU completion.
    let destination =
        unsafe { MetalImageDestination::from_exclusive_buffer(output.clone(), layout) }.map_err(
            |source| {
                crate::error::metal_kernel_support_error(
                    "J2K submitted codec-owned Metal destination failed",
                    source,
                )
            },
        )?;
    Ok(CodecOwnedMetalGroupDestination {
        destination,
        output,
        layout,
    })
}

#[cfg(target_os = "macos")]
fn completed_codec_owned_resident_batch(
    output: Buffer,
    layout: MetalImageLayout,
    expose_surface_views: bool,
) -> Result<(MetalResidentBatch, Vec<Surface>), Error> {
    // SAFETY: the producer submission completed successfully, its exclusive
    // destination was dropped, and `output` is the last raw writable owner.
    let storage =
        unsafe { ResidentMetalImage::from_completed_buffer(output, layout) }.map_err(|source| {
            crate::error::metal_kernel_support_error(
                "J2K submitted codec-owned resident batch",
                source,
            )
        })?;
    let mut surfaces = Vec::new();
    if expose_surface_views {
        surfaces
            .try_reserve_exact(layout.image_count())
            .map_err(|source| Error::PreparedPlanCacheAllocation {
                context: "J2K submitted codec-owned Metal surface views",
                source,
            })?;
        for index in 0..layout.image_count() {
            let relative_offset =
                layout
                    .image_offset_bytes(index)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "J2K submitted codec-owned Metal surface offset overflow"
                            .to_string(),
                    })?;
            let offset = layout
                .byte_offset()
                .checked_add(relative_offset)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K submitted codec-owned Metal surface offset overflow".to_string(),
                })?;
            let view_layout = MetalImageLayout::new(
                offset,
                layout.dimensions(),
                layout.pitch_bytes(),
                layout.pixel_format(),
            )
            .map_err(|source| {
                crate::error::metal_kernel_support_error(
                    "J2K submitted codec-owned Metal surface layout",
                    source,
                )
            })?;
            let view = storage.view(view_layout).map_err(|source| {
                crate::error::metal_kernel_support_error(
                    "J2K submitted codec-owned Metal surface view",
                    source,
                )
            })?;
            surfaces.push(Surface::from_resident_metal_image(view));
        }
    }
    Ok((MetalResidentBatch { storage }, surfaces))
}

#[cfg(target_os = "macos")]
impl SubmittedMetalResidentGroup {
    pub(super) fn wait(self) -> Result<MetalBatchGroup, (Vec<usize>, Box<Error>)> {
        let Self {
            mut metadata,
            submission,
            destination,
            output,
            layout,
        } = self;
        if let Err(source) = submission.wait() {
            return Err((metadata.source_indices, Box::new(source)));
        }
        drop(destination);
        let expose_surface_views =
            metadata.info.color == BatchColor::Gray || metadata.info.layout == BatchLayout::Nhwc;
        let (resident_batch, surfaces) =
            completed_codec_owned_resident_batch(output, layout, expose_surface_views).map_err(
                |source| {
                    (
                        core::mem::take(&mut metadata.source_indices),
                        Box::new(source),
                    )
                },
            )?;
        Ok(MetalBatchGroup {
            info: metadata.info,
            source_indices: metadata.source_indices,
            decoded_rects: metadata.decoded_rects,
            warnings: metadata.warnings,
            surfaces,
            resident_batch,
        })
    }
}
