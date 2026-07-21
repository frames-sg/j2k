// SPDX-License-Identifier: MIT OR Apache-2.0

//! Direct decode into caller-owned Metal group storage.

use super::{
    validate_group_contract, BatchColor, Error, MetalBatchDecoder, MetalBatchGroupCompletion,
    MetalImageDestination, PreparedBatchGroup, SubmittedMetalGroupDecodeInto,
};

pub(super) fn validate_consumer_registry_ids(
    producer_registry_id: u64,
    consumer_registry_id: u64,
) -> Result<(), Error> {
    if producer_registry_id == consumer_registry_id {
        return Ok(());
    }
    Err(crate::error::metal_kernel_support_error(
        "J2K Metal consumer queue belongs to a different device",
        j2k_metal_support::MetalSupportError::MetalImageDeviceMismatch {
            image_registry_id: producer_registry_id,
            requested_registry_id: consumer_registry_id,
        },
    ))
}

impl MetalBatchDecoder {
    /// Decode one prepared homogeneous Gray, RGB, or RGBA group with native U8, U16, or I16 samples
    /// directly into one caller-owned Metal allocation.
    ///
    /// The group allocation is bound once at its validated base. Per-image
    /// offsets are applied by the final-store kernel, so tightly packed Gray8
    /// images do not need independently aligned byte offsets.
    #[cfg(target_os = "macos")]
    pub fn decode_prepared_group_into(
        &mut self,
        group: &PreparedBatchGroup,
        destination: &MetalImageDestination,
    ) -> Result<(), Error> {
        let fmt = validate_group_contract(group.info())?;
        destination
            .validate_device(self.backend_session().device())
            .and_then(|()| {
                destination.validate_batch(group.info().dimensions, fmt, group.images().len())
            })
            .map_err(|source| {
                crate::error::metal_kernel_support_error(
                    "J2K Metal prepared group destination validation failed",
                    source,
                )
            })?;
        match group.info().color {
            BatchColor::Gray => {
                let plans = self.prepared_gray_group_plans(group, fmt, false)?;
                let runtime = self.backend_session().runtime()?;
                crate::compute::submit_prepared_direct_grayscale_plan_batch_into_group(
                    runtime,
                    &plans,
                    fmt,
                    destination,
                    Some(group.source_indices()),
                    crate::compute::DirectDestinationConsumerOrdering::HostCompletionOnly,
                )?
                .wait()?;
            }
            BatchColor::Rgb | BatchColor::Rgba => {
                let plans = self.prepared_color_group_plans(group, fmt)?;
                let runtime = self.backend_session().runtime()?;
                crate::compute::submit_prepared_direct_color_plan_batch_into_group(
                    runtime,
                    &plans,
                    fmt,
                    group.info().layout,
                    destination,
                    Some(group.source_indices()),
                    crate::compute::DirectDestinationConsumerOrdering::HostCompletionOnly,
                )?
                .wait()?;
            }
            _ => {
                return Err(Error::UnsupportedMetalRequest {
                    reason:
                        "J2K Metal exact external final-store received an unknown color contract",
                })
            }
        }
        self.record_submission();
        Ok(())
    }

    /// Submit one prepared homogeneous Gray, RGB, or RGBA group directly
    /// into one caller-owned Metal allocation without waiting on the CPU.
    ///
    /// The returned guard retains exclusive destination access, the committed
    /// command buffer, status buffers, and scratch resources. Call
    /// [`SubmittedMetalGroupDecodeInto::wait`] to surface execution failures.
    /// Dropping the guard also retires the work safely.
    #[cfg(target_os = "macos")]
    pub fn submit_prepared_group_into(
        &mut self,
        group: &PreparedBatchGroup,
        destination: MetalImageDestination,
    ) -> Result<SubmittedMetalGroupDecodeInto, Error> {
        self.submit_prepared_group_into_with_ordering(
            group,
            destination,
            crate::compute::DirectDestinationConsumerOrdering::Deferred,
        )
    }

    /// Submit one prepared group into caller-owned storage while registering
    /// its dependency on a consumer queue known before producer commit.
    ///
    /// The exact producer queue needs no event bridge. A different queue on
    /// the same device receives a GPU-side `MTLEvent` wait before this method
    /// returns. Queues from another device are rejected before codec work is
    /// committed.
    #[cfg(target_os = "macos")]
    pub fn submit_prepared_group_into_for_consumer_queue(
        &mut self,
        group: &PreparedBatchGroup,
        destination: MetalImageDestination,
        consumer_queue: &metal::CommandQueueRef,
    ) -> Result<SubmittedMetalGroupDecodeInto, Error> {
        let producer_registry_id = self.backend_session().device().registry_id();
        let consumer_registry_id = consumer_queue.device().registry_id();
        validate_consumer_registry_ids(producer_registry_id, consumer_registry_id)?;
        self.submit_prepared_group_into_with_ordering(
            group,
            destination,
            crate::compute::DirectDestinationConsumerOrdering::Known {
                consumer_queue: consumer_queue.to_owned(),
                timeline: self.backend_session().consumer_event_timeline(),
            },
        )
    }

    fn submit_prepared_group_into_with_ordering(
        &mut self,
        group: &PreparedBatchGroup,
        destination: MetalImageDestination,
        consumer_ordering: crate::compute::DirectDestinationConsumerOrdering,
    ) -> Result<SubmittedMetalGroupDecodeInto, Error> {
        let fmt = validate_group_contract(group.info())?;
        destination
            .validate_device(self.backend_session().device())
            .and_then(|()| {
                destination.validate_batch(group.info().dimensions, fmt, group.images().len())
            })
            .map_err(|source| {
                crate::error::metal_kernel_support_error(
                    "J2K Metal submitted prepared group destination validation failed",
                    source,
                )
            })?;
        let runtime = self.backend_session().runtime()?;
        let submission =
            match group.info().color {
                BatchColor::Gray => {
                    let plans = self.prepared_gray_group_plans(group, fmt, true)?;
                    crate::compute::submit_prepared_direct_grayscale_plan_batch_into_group(
                        runtime,
                        &plans,
                        fmt,
                        &destination,
                        Some(group.source_indices()),
                        consumer_ordering,
                    )?
                }
                BatchColor::Rgb | BatchColor::Rgba => {
                    let plans = self.prepared_color_group_plans(group, fmt)?;
                    crate::compute::submit_prepared_direct_color_plan_batch_into_group(
                        runtime,
                        &plans,
                        fmt,
                        group.info().layout,
                        &destination,
                        Some(group.source_indices()),
                        consumer_ordering,
                    )?
                }
                _ => return Err(Error::UnsupportedMetalRequest {
                    reason:
                        "J2K Metal exact external final-store received an unknown color contract",
                }),
            };
        self.record_submission();
        Ok(SubmittedMetalGroupDecodeInto {
            submission,
            destination,
            completion: MetalBatchGroupCompletion::from_prepared(group, group.options()),
        })
    }
}
