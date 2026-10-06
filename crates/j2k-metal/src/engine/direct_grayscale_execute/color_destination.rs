// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

use std::sync::Arc;

use j2k::BatchLayout;
use j2k_metal_support::{dispatch_3d_pipeline, MetalImageDestination};

use super::{
    allocation::{
        allocate_direct_execution_metadata, direct_ht_job_count, DirectExecutionMetadata,
    },
    destination::{commit_direct_destination, DirectDestinationConsumerOrdering},
    encode_prepared_direct_component_plane_in_encoder, encode_stacked_direct_component_plane_batch,
    encode_stacked_direct_component_plane_batch_with_shared_tier1, new_command_buffer,
    new_compute_command_encoder, prepared_direct_color_plan_supports_runtime,
    supports_stacked_direct_component_plane_batch, Buffer, DirectColorBatchCommandBuffers,
    DirectComponentPlaneRequest, DirectHybridStageTimings, DirectTier1Mode, Error, MetalRuntime,
    PixelFormat, PreparedDirectColorPlan, PreparedDirectGrayscalePlan,
    StackedDirectComponentPlaneBatchRequest, SubmittedDirectDestination,
};

mod encoder;
mod store;
mod validation;

use encoder::{color_component_plan_refs, ColorGroupEncoder};
use store::encode_exact_native_color_batch_store_in_encoder;
use validation::validate_color_group;

enum ColorTier1Execution {
    Metal(DirectDestinationConsumerOrdering),
    Cpu(std::num::NonZeroUsize),
}

#[cfg(target_os = "macos")]
pub(crate) fn submit_prepared_direct_color_plan_batch_into_group(
    runtime: Arc<MetalRuntime>,
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
    layout: BatchLayout,
    destination: &MetalImageDestination,
    source_indices: Option<&[usize]>,
    consumer_ordering: DirectDestinationConsumerOrdering,
) -> Result<SubmittedDirectDestination, Error> {
    submit_color_group(
        runtime,
        plans,
        fmt,
        layout,
        destination,
        source_indices,
        ColorTier1Execution::Metal(consumer_ordering),
        None,
    )
}

/// One prepared color group of a multi-group submission.
pub(crate) struct ColorGroupSubmission<'a> {
    pub(crate) plans: &'a [Arc<PreparedDirectColorPlan>],
    pub(crate) fmt: PixelFormat,
    pub(crate) layout: BatchLayout,
    pub(crate) destination: &'a MetalImageDestination,
    pub(crate) source_indices: Option<&'a [usize]>,
    pub(crate) consumer_ordering: DirectDestinationConsumerOrdering,
}

/// Submits several color groups, returning one result per group in order.
///
/// When at least two groups decode only classic code blocks through the
/// stacked RGB route, all their Tier-1 work is committed first as one shared
/// dispatch. If that dispatch cannot be built, every group falls back to its
/// own Tier-1 dispatch, which reports any error against that group. The outer
/// error reports a host allocation failure before anything is committed, or a
/// session-fatal failure after committed work has been retired.
#[cfg(target_os = "macos")]
pub(crate) fn submit_prepared_direct_color_plan_batches_into_groups(
    runtime: &Arc<MetalRuntime>,
    groups: Vec<ColorGroupSubmission<'_>>,
) -> Result<Vec<Result<SubmittedDirectDestination, Error>>, Error> {
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal multi-group color submission");
    let mut shared = budget.try_vec(groups.len(), "J2K Metal shared Tier-1 group slots")?;
    shared.extend(groups.iter().map(|_| None));
    let mut eligible = budget.try_vec(groups.len(), "J2K Metal shared Tier-1 eligible groups")?;
    let mut eligible_plans =
        budget.try_vec(groups.len(), "J2K Metal shared Tier-1 component plans")?;
    for (index, group) in groups.iter().enumerate() {
        if validate_color_group(
            runtime,
            group.plans,
            group.fmt,
            group.layout,
            group.destination,
        )
        .is_err()
        {
            // Reported by the group's own submission below.
            continue;
        }
        if let Some(plans) = shared_tier1_component_plans(group.plans, group.fmt)? {
            eligible.push(index);
            eligible_plans.push(plans);
        }
    }
    if eligible.len() >= 2 {
        match super::super::encode_shared_classic_tier1(runtime, &eligible_plans) {
            Ok(Some(parts)) => {
                for (index, part) in eligible.iter().zip(parts) {
                    shared[*index] = Some(part);
                }
            }
            Err(error) if error.session_is_unusable() => return Err(error),
            // The combined batch can exceed 32-bit coefficient offsets or
            // allocation limits that each group fits within. Each group then
            // decodes its own Tier-1 and reports any failure with its own
            // source attribution.
            Ok(None) | Err(_) => {}
        }
    }
    drop(eligible_plans);

    let mut results = budget.try_vec(groups.len(), "J2K Metal multi-group color results")?;
    for (group, shared) in groups.into_iter().zip(shared) {
        let result = submit_color_group(
            runtime.clone(),
            group.plans,
            group.fmt,
            group.layout,
            group.destination,
            group.source_indices,
            ColorTier1Execution::Metal(group.consumer_ordering),
            shared,
        );
        match result {
            Err(error) if error.session_is_unusable() => return Err(error),
            result => results.push(result),
        }
    }
    Ok(results)
}

/// Component plans of a group in the order of its stacked RGB batch (every
/// red plane, then green, then blue), when the group takes that route and
/// decodes only classic code blocks.
fn shared_tier1_component_plans(
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
) -> Result<Option<Vec<&PreparedDirectGrayscalePlan>>, Error> {
    // Mirrors the route `submit_color_group` takes for distinct plans:
    // coalesced, not repeated, and stacked across all three components.
    let distinct = plans
        .iter()
        .enumerate()
        .all(|(index, plan)| !plans[..index].iter().any(|seen| Arc::ptr_eq(seen, plan)));
    if plans.len() < 2
        || !distinct
        || fmt.channels() != 3
        || plans.iter().any(|plan| plan.component_plans.len() != 3)
    {
        return Ok(None);
    }
    let component_plan_refs = color_component_plan_refs(plans)?;
    if !component_plan_refs
        .iter()
        .all(|refs| supports_stacked_direct_component_plane_batch(refs))
    {
        return Ok(None);
    }
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal shared Tier-1 eligibility");
    let mut combined = budget.try_vec(
        plans.len().saturating_mul(3),
        "J2K Metal shared Tier-1 stacked component plans",
    )?;
    combined.extend(component_plan_refs.iter().flatten().copied());
    let eligible = combined
        .iter()
        .all(|component| component.tier1_prepare_mode == DirectTier1Mode::Metal)
        && supports_stacked_direct_component_plane_batch(&combined)
        && super::super::stacked_plans_use_classic_tier1_only(&combined);
    Ok(eligible.then_some(combined))
}

pub(crate) fn submit_cooperative_cpu_color_group(
    runtime: Arc<MetalRuntime>,
    plans: &[Arc<PreparedDirectColorPlan>],
    layout: BatchLayout,
    destination: &MetalImageDestination,
    source_indices: &[usize],
    max_workers: std::num::NonZeroUsize,
) -> Result<SubmittedDirectDestination, Error> {
    submit_color_group(
        runtime,
        plans,
        PixelFormat::Rgb8,
        layout,
        destination,
        Some(source_indices),
        ColorTier1Execution::Cpu(max_workers),
        None,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "the group's plans, output contract, attribution, and Tier-1 source are all independent"
)]
fn submit_color_group(
    runtime: Arc<MetalRuntime>,
    plans: &[Arc<PreparedDirectColorPlan>],
    fmt: PixelFormat,
    layout: BatchLayout,
    destination: &MetalImageDestination,
    source_indices: Option<&[usize]>,
    execution: ColorTier1Execution,
    shared_tier1: Option<super::super::SharedClassicTier1Group>,
) -> Result<SubmittedDirectDestination, Error> {
    let (consumer_ordering, cpu_workers) = match execution {
        ColorTier1Execution::Metal(ordering) => (ordering, None),
        ColorTier1Execution::Cpu(workers) => (
            DirectDestinationConsumerOrdering::HostCompletionOnly,
            Some(workers),
        ),
    };
    let Some(_) = plans.first() else {
        return Err(Error::capability_rejected(
            j2k_core::CapabilityRejection::contract_violation(
                "J2K Metal exact RGB destination requires at least one image",
            ),
        ));
    };
    validate_color_group(&runtime, plans, fmt, layout, destination)?;
    if source_indices.is_some_and(|indices| indices.len() != plans.len()) {
        return Err(Error::MetalStateInvariant {
            state: "J2K submitted RGB source attribution",
            reason: "source index count does not match prepared plan count",
        });
    }

    let step_count = crate::batch_allocation::checked_count_sum(
        plans
            .iter()
            .flat_map(|plan| plan.component_plans.iter())
            .map(|component| component.steps.len()),
        "J2K Metal exact RGB destination batch step metadata",
    )?;
    let mut metadata = allocate_direct_execution_metadata(
        step_count,
        direct_ht_job_count(
            plans.iter().flat_map(|plan| plan.component_plans.iter()),
            "J2K Metal exact RGB destination HT jobs",
        )?,
        crate::batch_allocation::BatchMetadataBudget::new(
            "J2K Metal exact RGB destination batch execution resources",
        ),
    )?;
    let component_plan_refs = color_component_plan_refs(plans)?;
    let use_stacked = plans.len() > 1
        && component_plan_refs
            .iter()
            .all(|refs| supports_stacked_direct_component_plane_batch(refs));
    let mut stage_timings = DirectHybridStageTimings::default();
    let cpu_cache = if let Some(max_workers) = cpu_workers {
        if cooperative_color_costs(plans)?.is_none() || !use_stacked {
            return Err(Error::MetalStateInvariant {
                state: "J2K cooperative CPU color group",
                reason: "CPU cohort must contain compatible distinct classic RGB8 plans",
            });
        }
        Some(super::super::build_flattened_cpu_tier1_cache(
            &runtime,
            plans,
            &mut stage_timings,
            &mut metadata.retained_buffers,
            Some(max_workers),
        )?)
    } else {
        None
    };
    let command_buffer = new_command_buffer(&runtime.queue)?;
    let compute_encoder = new_compute_command_encoder(&command_buffer)?;
    let mut encoder = ColorGroupEncoder {
        runtime: &runtime,
        command_buffer: &command_buffer,
        compute_encoder: &compute_encoder,
        plans,
        fmt,
        layout,
        destination,
        source_indices,
        metadata: &mut metadata,
        stage_timings,
        cpu_cache: cpu_cache.as_ref(),
        shared_tier1,
    };
    let result = if use_stacked {
        encoder.encode_coalesced(&component_plan_refs)
    } else {
        encoder.encode_individually()
    };
    let shared_tier1 = encoder.shared_tier1.take();
    compute_encoder.endEncoding();
    result?;
    let shared_pass = shared_tier1
        .map(super::super::SharedClassicTier1Group::finish)
        .transpose()?;

    metadata.record_tier1_dispatches();
    if cpu_workers.is_some() {
        metadata.dispatch_report.cpu_tier1_images = plans.len();
    }

    let mut submitted =
        commit_direct_destination(runtime, command_buffer, metadata, consumer_ordering)?;
    if let Some(pass) = shared_pass {
        submitted.depend_on_shared_tier1(pass);
    }
    Ok(submitted)
}

/// Conservative eligibility and entropy work estimates; no pixel decoding or caching.
pub(crate) fn cooperative_color_costs(
    plans: &[Arc<PreparedDirectColorPlan>],
) -> Result<Option<Vec<u64>>, Error> {
    if plans.len() < 2
        || plans.iter().enumerate().any(|(index, plan)| {
            plan.component_plans.len() != 3
                || plan.signed
                || plan.alpha_bit_depth.is_some()
                || !plan.mct
                || plan.transform != super::super::J2kWaveletTransform::Reversible53
                || plan.bit_depths != [8; 3]
                || plans[..index].iter().any(|other| Arc::ptr_eq(plan, other))
                || plan.component_plans.iter().any(|component| {
                    component.steps.iter().any(|step| {
                        matches!(
                            step,
                            super::super::PreparedDirectGrayscaleStep::HtSubBand(_)
                        )
                    })
                })
        })
    {
        return Ok(None);
    }
    if color_component_plan_refs(plans)?
        .iter()
        .any(|refs| !supports_stacked_direct_component_plane_batch(refs))
    {
        return Ok(None);
    }
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K cooperative work estimates");
    let mut costs = budget.try_vec(plans.len(), "J2K cooperative image costs")?;
    for plan in plans {
        let mut cost = 0u64;
        for component in &plan.component_plans {
            for step in &component.steps {
                if let super::super::PreparedDirectGrayscaleStep::ClassicSubBand(band) = step {
                    for job in &band.jobs {
                        // Coefficient-pass visits dominate MQ work. Include payload bytes
                        // and one unit per block to distinguish sparse/empty block sets.
                        cost = cost
                            .saturating_add(
                                u64::from(job.width)
                                    .saturating_mul(u64::from(job.height))
                                    .saturating_mul(u64::from(job.number_of_coding_passes)),
                            )
                            .saturating_add(u64::from(job.coded_len))
                            .saturating_add(1);
                    }
                }
            }
        }
        costs.push(cost);
    }
    Ok(Some(costs))
}
