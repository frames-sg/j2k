// SPDX-License-Identifier: MIT OR Apache-2.0

//! Codec-owned resident Metal group submission.

#[cfg(target_os = "macos")]
use std::sync::Arc;

#[cfg(target_os = "macos")]
use j2k::{BatchGroupInfo, PreparedDeviceOperation, PreparedDeviceOperationGroup};

#[cfg(target_os = "macos")]
use super::{
    allocate_codec_owned_group_destination, submission::CodecOwnedMetalGroupDestination,
    validate_codec_owned_resident_group, BatchColor, DecodeRequest, MetalBatchGroupError,
    MetalResidentGroupMetadata, PixelFormat, PreparedImage, SubmittedMetalResidentGroup,
};
use super::{BatchDecodeOptions, Error, MetalBatchDecoder, MetalBatchGroup, PreparedBatchGroup};

/// One homogeneous output group as resident submission sees it: the images of
/// a prepared batch group, or exact operations prepared for the device.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(super) struct ResidentGroupView<'g> {
    pub(super) info: &'g BatchGroupInfo,
    pub(super) source_indices: &'g [usize],
    images: GroupImages<'g>,
    /// The exact-operation group whose shared cancellation token applies.
    operations: Option<&'g PreparedDeviceOperationGroup>,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
enum GroupImages<'g> {
    Prepared(&'g [PreparedImage]),
    Operations(&'g [PreparedDeviceOperation]),
}

/// One image of a [`ResidentGroupView`].
#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(super) enum GroupImage<'g> {
    Prepared(&'g PreparedImage),
    Operation(&'g PreparedDeviceOperation),
}

#[cfg(target_os = "macos")]
impl<'g> From<&'g PreparedBatchGroup> for ResidentGroupView<'g> {
    fn from(group: &'g PreparedBatchGroup) -> Self {
        Self {
            info: group.info(),
            source_indices: group.source_indices(),
            images: GroupImages::Prepared(group.images()),
            operations: None,
        }
    }
}

#[cfg(target_os = "macos")]
impl<'g> From<&'g PreparedDeviceOperationGroup> for ResidentGroupView<'g> {
    fn from(group: &'g PreparedDeviceOperationGroup) -> Self {
        Self {
            info: group.info(),
            source_indices: group.source_indices(),
            images: GroupImages::Operations(group.operations()),
            operations: Some(group),
        }
    }
}

#[cfg(target_os = "macos")]
impl<'g> ResidentGroupView<'g> {
    /// Stop before allocating or committing a cancelled operation group.
    /// Prepared batches carry no cancellation.
    pub(super) fn check_cancelled(self) -> Result<(), Error> {
        self.operations
            .map_or(Ok(()), PreparedDeviceOperationGroup::check_cancelled)
            .map_err(Error::Decode)
    }

    pub(super) const fn len(self) -> usize {
        match self.images {
            GroupImages::Prepared(images) => images.len(),
            GroupImages::Operations(operations) => operations.len(),
        }
    }

    pub(super) const fn is_empty(self) -> bool {
        self.len() == 0
    }

    pub(super) fn images(self) -> impl Iterator<Item = GroupImage<'g>> {
        let (prepared, operations) = match self.images {
            GroupImages::Prepared(images) => (images, &[][..]),
            GroupImages::Operations(operations) => (&[][..], operations),
        };
        prepared
            .iter()
            .map(GroupImage::Prepared)
            .chain(operations.iter().map(GroupImage::Operation))
    }
}

#[cfg(target_os = "macos")]
impl<'g> GroupImage<'g> {
    pub(super) fn bytes(self) -> &'g Arc<[u8]> {
        match self {
            Self::Prepared(image) => image.bytes(),
            Self::Operation(operation) => operation.bytes(),
        }
    }

    pub(super) fn htj2k_plan(self) -> Option<&'g j2k::PreparedHtj2kPlan> {
        match self {
            Self::Prepared(image) => image.htj2k_plan(),
            Self::Operation(operation) => operation.htj2k_plan(),
        }
    }

    pub(super) fn classic_plan(self) -> Option<&'g j2k::PreparedClassicPlan> {
        match self {
            Self::Prepared(image) => image.classic_plan(),
            Self::Operation(operation) => operation.classic_plan(),
        }
    }

    /// Decoded rectangle in the coordinates of the decoded level.
    pub(super) fn output_rect(self) -> j2k_core::Rect {
        match self {
            Self::Prepared(image) => image.plan().output_rect(),
            Self::Operation(operation) => operation.output_rect(),
        }
    }

    pub(super) fn used_lenient_metadata_recovery(self) -> bool {
        match self {
            Self::Prepared(image) => image.used_lenient_metadata_recovery(),
            Self::Operation(operation) => operation.used_lenient_metadata_recovery(),
        }
    }

    /// Whether the image stores only part of its decoded level. Referenced
    /// plans then keep whole-level coefficient and IDWT windows unless pruned
    /// to their stores; at full resolution on large images those exceed
    /// Metal's buffer cap.
    pub(super) fn stores_a_region(self) -> bool {
        match self {
            Self::Prepared(image) => matches!(
                image.request(),
                DecodeRequest::Region { .. } | DecodeRequest::RegionReduced { .. }
            ),
            Self::Operation(operation) => !operation.covers_reduced_image(),
        }
    }

    /// Whether the image is the whole source at full resolution.
    pub(super) fn is_full_resolution_image(self) -> bool {
        match self {
            Self::Prepared(image) => image.request() == DecodeRequest::Full,
            Self::Operation(operation) => {
                operation.covers_reduced_image() && operation.reduction_levels() == 0
            }
        }
    }

    pub(super) fn has_component_subsampling(self) -> bool {
        match self {
            Self::Prepared(image) => image.support().has_component_subsampling(),
            Self::Operation(operation) => operation.support().has_component_subsampling(),
        }
    }

    pub(super) fn decode_settings(self) -> j2k::DecodeSettings {
        match self {
            Self::Prepared(image) => image.decode_settings(),
            Self::Operation(operation) => operation.decode_settings(),
        }
    }

    /// Persistent plan-cache identity: exact operations never share a key with
    /// their source's own prepared plan.
    pub(super) fn plan_cache_key(
        self,
        fmt: PixelFormat,
        color: bool,
    ) -> crate::session::PreparedPlanCacheKey<'g> {
        match self {
            Self::Prepared(image) if color => crate::session::PreparedPlanCacheKey::prepared_color(
                image.bytes(),
                image.request(),
                fmt,
            ),
            Self::Prepared(image) => crate::session::PreparedPlanCacheKey::prepared_gray(
                image.bytes(),
                image.request(),
                fmt,
            ),
            Self::Operation(operation) => crate::session::PreparedPlanCacheKey::prepared_operation(
                operation.bytes(),
                operation.source_region(),
                operation.reduction_levels(),
                fmt,
                color,
            ),
        }
    }
}

/// Direct plans of one prepared group.
#[cfg(target_os = "macos")]
pub(super) enum GroupPlans {
    Gray(Vec<Arc<crate::engine::PreparedDirectGrayscalePlan>>),
    Color(Vec<Arc<crate::engine::PreparedDirectColorPlan>>),
}

/// A resident group whose output is allocated and plans resolved.
#[cfg(target_os = "macos")]
pub(super) struct PreparedResidentGroup<'g> {
    pub(super) group: ResidentGroupView<'g>,
    fmt: PixelFormat,
    plans: GroupPlans,
    allocation: CodecOwnedMetalGroupDestination,
    metadata: MetalResidentGroupMetadata,
    /// CPU workers for automatic Tier-1 on this group, from the options it
    /// was prepared with.
    automatic_cpu_workers: Option<std::num::NonZeroUsize>,
}

/// Submitted resident groups in input order, and the results of the extra
/// color submissions that shared their Tier-1 dispatch.
#[cfg(target_os = "macos")]
pub(super) type ResidentSubmissions = (
    Vec<Result<SubmittedMetalResidentGroup, MetalBatchGroupError>>,
    Vec<Result<crate::engine::SubmittedDirectDestination, Error>>,
);

impl MetalBatchDecoder {
    pub(super) fn decode_prepared_group_with_options(
        &mut self,
        group: &PreparedBatchGroup,
        options: BatchDecodeOptions,
    ) -> Result<MetalBatchGroup, Error> {
        #[cfg(target_os = "macos")]
        {
            let pending = self.submit_prepared_resident_group(group, options)?;
            pending.wait().map_err(|(_, source)| *source)
        }

        #[cfg(not(target_os = "macos"))]
        {
            let _ = self;
            let _ = group;
            let _ = options;
            Err(Error::MetalUnavailable)
        }
    }

    #[cfg(target_os = "macos")]
    pub(super) fn submit_prepared_resident_group(
        &mut self,
        group: &PreparedBatchGroup,
        options: BatchDecodeOptions,
    ) -> Result<SubmittedMetalResidentGroup, Error> {
        let prepared = self.prepare_resident_group(group, options)?;
        let prepared = crate::batch_allocation::try_vec_from_array(
            [prepared],
            "J2K submitted codec-owned Metal group",
        )?;
        let (mut submitted, _) = self.submit_resident_groups(prepared, Vec::new(), true)?;
        submitted
            .pop()
            .ok_or(Error::MetalStateInvariant {
                state: "J2K submitted codec-owned Metal group",
                reason: "single-group submission returned no result",
            })?
            .map_err(|error| error.into_parts().1)
    }

    #[cfg(target_os = "macos")]
    pub(super) fn prepare_resident_group<'g>(
        &mut self,
        group: &'g PreparedBatchGroup,
        options: BatchDecodeOptions,
    ) -> Result<PreparedResidentGroup<'g>, Error> {
        self.prepare_resident_group_view(group.into(), options)
    }

    #[cfg(target_os = "macos")]
    pub(super) fn prepare_resident_group_view<'g>(
        &mut self,
        group: ResidentGroupView<'g>,
        options: BatchDecodeOptions,
    ) -> Result<PreparedResidentGroup<'g>, Error> {
        let fmt = validate_codec_owned_resident_group(group)?;
        let allocation =
            allocate_codec_owned_group_destination(self.backend_session(), group, fmt)?;
        let plans = self.prepared_group_plans(group, fmt)?;
        Ok(PreparedResidentGroup {
            group,
            fmt,
            plans,
            allocation,
            metadata: MetalResidentGroupMetadata::from_view(group, options)?,
            automatic_cpu_workers: super::automatic_cpu_tier1_workers(options),
        })
    }

    #[cfg(target_os = "macos")]
    pub(super) fn prepared_group_plans(
        &mut self,
        group: ResidentGroupView<'_>,
        fmt: PixelFormat,
    ) -> Result<GroupPlans, Error> {
        match group.info.color {
            BatchColor::Gray => Ok(GroupPlans::Gray(
                self.prepared_gray_group_plans(group, fmt, true)?,
            )),
            BatchColor::Rgb | BatchColor::Rgba => Ok(GroupPlans::Color(
                self.prepared_color_group_plans(group, fmt)?,
            )),
            _ => Err(Error::capability_rejected(
                j2k_core::CapabilityRejection::unsupported_format(
                    "J2K Metal prepared group received an unknown color contract",
                ),
            )),
        }
    }

    /// Submits resident groups together with `extra` color work. Compatible
    /// color groups, including the extra ones, decode their classic Tier-1 in
    /// one shared dispatch. A session-fatal failure returns `Err` after
    /// retiring every committed group.
    #[cfg(target_os = "macos")]
    pub(super) fn submit_resident_groups(
        &mut self,
        groups: Vec<PreparedResidentGroup<'_>>,
        extra: Vec<crate::engine::ColorGroupSubmission<'_>>,
        allow_automatic_cpu_tier1: bool,
    ) -> Result<ResidentSubmissions, Error> {
        let runtime = self.backend_session().runtime()?;
        let extra_count = extra.len();
        let mut budget =
            crate::batch_allocation::BatchMetadataBudget::new("J2K submitted resident groups");
        let mut color = budget.try_vec(
            extra_count.saturating_add(groups.len()),
            "J2K submitted resident color groups",
        )?;
        color.extend(extra);
        for prepared in &groups {
            if let GroupPlans::Color(plans) = &prepared.plans {
                color.push(crate::engine::ColorGroupSubmission {
                    plans,
                    fmt: prepared.fmt,
                    layout: prepared.group.info.layout,
                    destination: &prepared.allocation.destination,
                    source_indices: Some(prepared.group.source_indices),
                    consumer_ordering:
                        crate::engine::DirectDestinationConsumerOrdering::HostCompletionOnly,
                    automatic_cpu_workers: prepared
                        .automatic_cpu_workers
                        .filter(|_| allow_automatic_cpu_tier1),
                });
            }
        }
        let mut color_results =
            crate::engine::submit_prepared_direct_color_plan_batches_into_groups(&runtime, color)?
                .into_iter();
        let mut extra_results = budget.try_vec(extra_count, "J2K submitted extra color groups")?;
        extra_results.extend(color_results.by_ref().take(extra_count));

        let mut submitted = budget.try_vec(groups.len(), "J2K submitted resident group results")?;
        for prepared in groups {
            let submission = match &prepared.plans {
                GroupPlans::Color(_) => {
                    color_results
                        .next()
                        .unwrap_or(Err(Error::MetalStateInvariant {
                            state: "J2K submitted resident color groups",
                            reason: "multi-group submission returned fewer results than groups",
                        }))
                }
                GroupPlans::Gray(plans) => {
                    crate::engine::submit_prepared_direct_grayscale_plan_batch_into_group(
                        runtime.clone(),
                        plans,
                        prepared.fmt,
                        &prepared.allocation.destination,
                        Some(prepared.group.source_indices),
                        crate::engine::DirectDestinationConsumerOrdering::HostCompletionOnly,
                    )
                }
            };
            match submission {
                Ok(submission) => {
                    self.record_submission();
                    submitted.push(Ok(SubmittedMetalResidentGroup {
                        metadata: prepared.metadata,
                        submission,
                        destination: prepared.allocation.destination,
                        output: prepared.allocation.output,
                        layout: prepared.allocation.layout,
                    }));
                }
                Err(source) if source.session_is_unusable() => return Err(source),
                Err(source) => {
                    submitted.push(Err(MetalBatchGroupError::for_sources(
                        prepared.group.source_indices,
                        source,
                    )));
                }
            }
        }
        Ok((submitted, extra_results))
    }
}
