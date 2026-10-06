// SPDX-License-Identifier: MIT OR Apache-2.0

//! Codec-owned resident Metal group submission.

#[cfg(target_os = "macos")]
use std::sync::Arc;

#[cfg(target_os = "macos")]
use super::{
    allocate_codec_owned_group_destination, submission::CodecOwnedMetalGroupDestination,
    validate_codec_owned_resident_group, BatchColor, MetalBatchGroupError,
    MetalResidentGroupMetadata, PixelFormat, SubmittedMetalResidentGroup,
};
use super::{BatchDecodeOptions, Error, MetalBatchDecoder, MetalBatchGroup, PreparedBatchGroup};

/// Direct plans of one prepared group.
#[cfg(target_os = "macos")]
pub(super) enum GroupPlans {
    Gray(Vec<Arc<crate::engine::PreparedDirectGrayscalePlan>>),
    Color(Vec<Arc<crate::engine::PreparedDirectColorPlan>>),
}

/// A resident group whose output is allocated and plans resolved.
#[cfg(target_os = "macos")]
pub(super) struct PreparedResidentGroup<'g> {
    group: &'g PreparedBatchGroup,
    fmt: PixelFormat,
    plans: GroupPlans,
    allocation: CodecOwnedMetalGroupDestination,
    metadata: MetalResidentGroupMetadata,
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
        let (mut submitted, _) = self.submit_resident_groups(prepared, Vec::new())?;
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
        let fmt = validate_codec_owned_resident_group(group)?;
        let allocation =
            allocate_codec_owned_group_destination(self.backend_session().device(), group, fmt)?;
        let plans = self.prepared_group_plans(group, fmt)?;
        Ok(PreparedResidentGroup {
            group,
            fmt,
            plans,
            allocation,
            metadata: MetalResidentGroupMetadata::from_prepared(group, options)?,
        })
    }

    #[cfg(target_os = "macos")]
    pub(super) fn prepared_group_plans(
        &mut self,
        group: &PreparedBatchGroup,
        fmt: PixelFormat,
    ) -> Result<GroupPlans, Error> {
        match group.info().color {
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
                    layout: prepared.group.info().layout,
                    destination: &prepared.allocation.destination,
                    source_indices: Some(prepared.group.source_indices()),
                    consumer_ordering:
                        crate::engine::DirectDestinationConsumerOrdering::HostCompletionOnly,
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
                        Some(prepared.group.source_indices()),
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
                    submitted.push(Err(MetalBatchGroupError::new(prepared.group, source)));
                }
            }
        }
        Ok((submitted, extra_results))
    }
}
