// SPDX-License-Identifier: MIT OR Apache-2.0

//! Direct decode into caller-owned Metal group storage.

use objc2::Message as _;

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

use super::resident::GroupPlans;
use super::{
    validate_group_contract, BatchColor, Error, MetalBatchDecoder, MetalBatchGroupCompletion,
    MetalBatchGroupError, MetalImageDestination, PreparedBatchGroup, SubmittedMetalGroupDecodeInto,
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
                crate::engine::submit_prepared_direct_grayscale_plan_batch_into_group(
                    runtime,
                    &plans,
                    fmt,
                    destination,
                    Some(group.source_indices()),
                    crate::engine::DirectDestinationConsumerOrdering::HostCompletionOnly,
                )?
                .wait()?;
            }
            BatchColor::Rgb | BatchColor::Rgba => {
                let plans = self.prepared_color_group_plans(group, fmt)?;
                let runtime = self.backend_session().runtime()?;
                crate::engine::submit_prepared_direct_color_plan_batch_into_group(
                    runtime,
                    &plans,
                    fmt,
                    group.info().layout,
                    destination,
                    Some(group.source_indices()),
                    crate::engine::DirectDestinationConsumerOrdering::HostCompletionOnly,
                )?
                .wait()?;
            }
            _ => {
                return Err(Error::capability_rejected(
                    j2k_core::CapabilityRejection::unsupported_format(
                        "J2K Metal exact external final-store received an unknown color contract",
                    ),
                ))
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
            crate::engine::DirectDestinationConsumerOrdering::Deferred,
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
        consumer_queue: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    ) -> Result<SubmittedMetalGroupDecodeInto, Error> {
        let producer_registry_id = self.backend_session().device().registryID();
        let consumer_registry_id = consumer_queue.device().registryID();
        validate_consumer_registry_ids(producer_registry_id, consumer_registry_id)?;
        self.submit_prepared_group_into_with_ordering(
            group,
            destination,
            crate::engine::DirectDestinationConsumerOrdering::Known {
                consumer_queue: consumer_queue.retain(),
                timeline: self.backend_session().consumer_event_timeline(),
            },
        )
    }

    /// Submit several prepared groups into caller-owned Metal allocations
    /// without waiting on the CPU.
    ///
    /// Each group behaves as if passed to [`Self::submit_prepared_group_into`],
    /// except that compatible classic RGB groups decode their Tier-1 code
    /// blocks in one shared dispatch before each group's transforms and store.
    /// Submitting groups one call at a time instead runs their Tier-1
    /// dispatches concurrently, which is slower on Apple GPUs.
    ///
    /// Results follow the input order. A failed group releases its
    /// destination and is reported with its source indices without affecting
    /// the others. A session-fatal failure returns `Err` after retiring every
    /// committed group.
    #[cfg(target_os = "macos")]
    pub fn submit_prepared_groups_into<'g>(
        &mut self,
        groups: impl IntoIterator<Item = (&'g PreparedBatchGroup, MetalImageDestination)>,
    ) -> Result<Vec<Result<SubmittedMetalGroupDecodeInto, MetalBatchGroupError>>, Error> {
        let groups = groups.into_iter();
        let mut budget =
            crate::batch_allocation::BatchMetadataBudget::new("J2K submitted external groups");
        let mut prepared = budget.try_vec(groups.size_hint().0, "J2K prepared external groups")?;
        for (group, destination) in groups {
            crate::batch_allocation::try_reserve_for_push(
                &mut prepared,
                "J2K prepared external groups",
            )?;
            prepared.push(match self.prepare_external_group(group, destination) {
                Ok(external) => Ok(external),
                Err(source) if source.session_is_unusable() => return Err(source),
                Err(source) => Err(MetalBatchGroupError::new(group, source)),
            });
        }

        let runtime = self.backend_session().runtime()?;
        let mut color = budget.try_vec(prepared.len(), "J2K submitted external color groups")?;
        for external in prepared.iter().flatten() {
            if let GroupPlans::Color(plans) = &external.plans {
                color.push(crate::engine::ColorGroupSubmission {
                    plans,
                    fmt: external.fmt,
                    layout: external.group.info().layout,
                    destination: &external.destination,
                    source_indices: Some(external.group.source_indices()),
                    consumer_ordering: crate::engine::DirectDestinationConsumerOrdering::Deferred,
                });
            }
        }
        let mut color_results =
            crate::engine::submit_prepared_direct_color_plan_batches_into_groups(&runtime, color)?
                .into_iter();

        let mut results = budget.try_vec(prepared.len(), "J2K submitted external group results")?;
        for external in prepared {
            let external = match external {
                Ok(external) => external,
                Err(error) => {
                    results.push(Err(error));
                    continue;
                }
            };
            let submission = match &external.plans {
                GroupPlans::Color(_) => {
                    color_results
                        .next()
                        .unwrap_or(Err(Error::MetalStateInvariant {
                            state: "J2K submitted external color groups",
                            reason: "multi-group submission returned fewer results than groups",
                        }))
                }
                GroupPlans::Gray(plans) => {
                    crate::engine::submit_prepared_direct_grayscale_plan_batch_into_group(
                        runtime.clone(),
                        plans,
                        external.fmt,
                        &external.destination,
                        Some(external.group.source_indices()),
                        crate::engine::DirectDestinationConsumerOrdering::Deferred,
                    )
                }
            };
            match submission {
                Ok(submission) => {
                    self.record_submission();
                    results.push(Ok(SubmittedMetalGroupDecodeInto {
                        submission,
                        destination: external.destination,
                        completion: external.completion,
                    }));
                }
                Err(source) if source.session_is_unusable() => return Err(source),
                Err(source) => results.push(Err(MetalBatchGroupError::new(external.group, source))),
            }
        }
        Ok(results)
    }

    fn submit_prepared_group_into_with_ordering(
        &mut self,
        group: &PreparedBatchGroup,
        destination: MetalImageDestination,
        consumer_ordering: crate::engine::DirectDestinationConsumerOrdering,
    ) -> Result<SubmittedMetalGroupDecodeInto, Error> {
        let external = self.prepare_external_group(group, destination)?;
        let runtime = self.backend_session().runtime()?;
        let submission = match &external.plans {
            GroupPlans::Gray(plans) => {
                crate::engine::submit_prepared_direct_grayscale_plan_batch_into_group(
                    runtime,
                    plans,
                    external.fmt,
                    &external.destination,
                    Some(group.source_indices()),
                    consumer_ordering,
                )?
            }
            GroupPlans::Color(plans) => {
                crate::engine::submit_prepared_direct_color_plan_batch_into_group(
                    runtime,
                    plans,
                    external.fmt,
                    group.info().layout,
                    &external.destination,
                    Some(group.source_indices()),
                    consumer_ordering,
                )?
            }
        };
        self.record_submission();
        Ok(SubmittedMetalGroupDecodeInto {
            submission,
            destination: external.destination,
            completion: external.completion,
        })
    }

    fn prepare_external_group<'g>(
        &mut self,
        group: &'g PreparedBatchGroup,
        destination: MetalImageDestination,
    ) -> Result<ExternalGroup<'g>, Error> {
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
        let plans = self.prepared_group_plans(group, fmt)?;
        Ok(ExternalGroup {
            group,
            fmt,
            plans,
            destination,
            completion: MetalBatchGroupCompletion::from_prepared(group)?,
        })
    }
}

/// An external group whose destination is validated and plans resolved.
#[cfg(target_os = "macos")]
struct ExternalGroup<'g> {
    group: &'g PreparedBatchGroup,
    fmt: j2k_core::PixelFormat,
    plans: GroupPlans,
    destination: MetalImageDestination,
    completion: MetalBatchGroupCompletion,
}
