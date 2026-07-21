// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    allocate_direct_execution_metadata, new_command_buffer, new_compute_command_encoder, Arc,
    Error, MetalRuntime, PixelFormat, PreparedDirectGrayscalePlan,
};
use j2k_metal_support::MetalImageDestination;

mod group_encode;
mod submission;

use self::group_encode::GrayscaleGroupEncoder;
pub(super) use self::submission::commit_direct_destination;
pub(crate) use self::submission::{DirectDestinationConsumerOrdering, SubmittedDirectDestination};

pub(crate) fn submit_prepared_direct_grayscale_plan_batch_into_group(
    runtime: Arc<MetalRuntime>,
    plans: &[Arc<PreparedDirectGrayscalePlan>],
    fmt: PixelFormat,
    destination: &MetalImageDestination,
    source_indices: Option<&[usize]>,
    consumer_ordering: DirectDestinationConsumerOrdering,
) -> Result<SubmittedDirectDestination, Error> {
    let Some(first) = plans.first() else {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal external group destination requires at least one image",
        });
    };
    destination
        .validate_device(&runtime.device)
        .and_then(|()| destination.validate_batch(first.dimensions, fmt, plans.len()))
        .map_err(|source| {
            crate::error::metal_kernel_support_error(
                "J2K Metal direct grayscale group destination validation failed",
                source,
            )
        })?;
    if plans.iter().any(|plan| plan.dimensions != first.dimensions) {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal external group destination requires homogeneous image dimensions",
        });
    }
    if source_indices.is_some_and(|indices| indices.len() != plans.len()) {
        return Err(Error::MetalStateInvariant {
            state: "J2K submitted grayscale source attribution",
            reason: "source index count does not match prepared plan count",
        });
    }

    let step_count = crate::batch_allocation::checked_count_sum(
        plans.iter().map(|plan| plan.steps.len()),
        "J2K Metal submitted direct grayscale destination batch step metadata",
    )?;
    let mut metadata = allocate_direct_execution_metadata(
        step_count,
        0,
        crate::batch_allocation::BatchMetadataBudget::new(
            "J2K Metal submitted direct grayscale destination batch execution resources",
        ),
    )?;
    let command_buffer = new_command_buffer(&runtime.queue)?;
    let compute_encoder = new_compute_command_encoder(&command_buffer)?;
    let result = GrayscaleGroupEncoder {
        runtime: &runtime,
        command_buffer: &command_buffer,
        compute_encoder: &compute_encoder,
        plans,
        fmt,
        destination,
        source_indices,
        metadata: &mut metadata,
    }
    .encode();
    compute_encoder.end_encoding();
    result?;
    commit_direct_destination(runtime, command_buffer, metadata, consumer_ordering)
}
