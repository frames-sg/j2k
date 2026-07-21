// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{mem::size_of, sync::Arc};

use j2k::BatchLayout;
use j2k_metal_support::{dispatch_3d_pipeline, MetalImageDestination};

use super::{
    allocation::{
        allocate_direct_execution_metadata, direct_ht_job_count, DirectExecutionMetadata,
    },
    destination::{commit_direct_destination, DirectDestinationConsumerOrdering},
    encode_prepared_direct_component_plane_in_encoder, encode_stacked_direct_component_plane_batch,
    new_command_buffer, new_compute_command_encoder, prepared_direct_color_plan_supports_runtime,
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
    let Some(_) = plans.first() else {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal exact RGB destination requires at least one image",
        });
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
    let command_buffer = new_command_buffer(&runtime.queue)?;
    let compute_encoder = new_compute_command_encoder(&command_buffer)?;
    let component_plan_refs = color_component_plan_refs(plans)?;
    let use_stacked = plans.len() > 1
        && component_plan_refs
            .iter()
            .all(|refs| supports_stacked_direct_component_plane_batch(refs));
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
        stage_timings: DirectHybridStageTimings::default(),
    };
    let result = if use_stacked {
        encoder.encode_coalesced(&component_plan_refs)
    } else {
        encoder.encode_individually()
    };
    compute_encoder.end_encoding();
    result?;

    commit_direct_destination(runtime, command_buffer, metadata, consumer_ordering)
}
