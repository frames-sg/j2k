// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use super::super::DirectStatusCheck;
use super::super::{
    new_command_buffer, recycle_scratch_buffers, retire_direct_status_checks,
    wait_for_completion_metal, Arc, CommandBuffer, DirectExecutionMetadata,
    DirectStatusRetirementMode, Error, MetalRuntime,
};
use metal::{foreign_types::ForeignType, CommandQueue, CommandQueueRef, Event, SharedEvent};

pub(crate) enum DirectDestinationConsumerOrdering {
    Deferred,
    HostCompletionOnly,
    Known {
        consumer_queue: CommandQueue,
        timeline: Arc<std::sync::Mutex<crate::session::MetalConsumerEventTimeline>>,
    },
}

enum DirectDestinationCompletionDependency {
    Deferred(SharedEvent),
    Known { _event: Event },
}

pub(crate) struct SubmittedDirectDestination {
    pub(in crate::compute::direct_grayscale_execute) runtime: Arc<MetalRuntime>,
    pub(in crate::compute::direct_grayscale_execute) command_buffer: Option<CommandBuffer>,
    pub(in crate::compute::direct_grayscale_execute) metadata: Option<DirectExecutionMetadata>,
    completion_dependency: Option<DirectDestinationCompletionDependency>,
    pub(in crate::compute::direct_grayscale_execute) consumer_waits: Vec<CommandBuffer>,
    #[cfg(test)]
    known_consumer_event_ptr: Option<usize>,
    #[cfg(test)]
    known_consumer_value: Option<u64>,
}

impl SubmittedDirectDestination {
    pub(crate) fn enqueue_consumer_wait(
        &mut self,
        consumer_queue: &CommandQueueRef,
    ) -> Result<(), Error> {
        let producer_registry_id = self.runtime.device.registry_id();
        let consumer_registry_id = consumer_queue.device().registry_id();
        if producer_registry_id != consumer_registry_id {
            return Err(crate::error::metal_kernel_support_error(
                "J2K Metal consumer queue belongs to a different device",
                j2k_metal_support::MetalSupportError::MetalImageDeviceMismatch {
                    image_registry_id: producer_registry_id,
                    requested_registry_id: consumer_registry_id,
                },
            ));
        }
        crate::batch_allocation::try_reserve_for_push(
            &mut self.consumer_waits,
            "J2K Metal consumer queue completion waits",
        )?;
        let DirectDestinationCompletionDependency::Deferred(completion_event) = self
            .completion_dependency
            .as_ref()
            .ok_or(Error::MetalStateInvariant {
                state: "J2K Metal direct destination consumer ordering",
                reason: "known-queue submission has no deferred consumer event bridge",
            })?
        else {
            return Err(Error::MetalStateInvariant {
                state: "J2K Metal direct destination consumer ordering",
                reason: "known consumer dependency was already registered at submission",
            });
        };
        let wait_command = new_command_buffer(consumer_queue)?;
        wait_command.encode_wait_for_event(completion_event, 1);
        wait_command.commit();
        #[cfg(test)]
        crate::compute::test_counters::record_direct_destination_event_wait();
        self.consumer_waits.push(wait_command);
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn ordering_diagnostics_for_test(&self) -> (bool, bool, usize) {
        let has_event = self.completion_dependency.is_some();
        let has_signal = matches!(
            self.completion_dependency,
            Some(
                DirectDestinationCompletionDependency::Deferred(_)
                    | DirectDestinationCompletionDependency::Known { .. }
            )
        );
        (has_event, has_signal, self.consumer_waits.len())
    }

    #[cfg(test)]
    pub(crate) fn known_consumer_timeline_for_test(&self) -> Option<(usize, u64)> {
        self.known_consumer_event_ptr.zip(self.known_consumer_value)
    }

    #[cfg(test)]
    pub(crate) fn in_flight_owner_ptrs_for_test(&self) -> (Vec<usize>, Vec<usize>) {
        let Some(metadata) = self.metadata.as_ref() else {
            return (Vec::new(), Vec::new());
        };
        let statuses = metadata
            .status_checks
            .iter()
            .map(|status| match status {
                DirectStatusCheck::Classic { buffer, .. }
                | DirectStatusCheck::Ht { buffer, .. }
                | DirectStatusCheck::Idwt(buffer)
                | DirectStatusCheck::Mct(buffer) => buffer.as_ptr() as usize,
            })
            .collect();
        let scratch = metadata
            .scratch_buffers
            .iter()
            .map(|owner| owner.buffer.buffer().as_ptr() as usize)
            .collect();
        (statuses, scratch)
    }

    pub(crate) fn wait(mut self) -> Result<(), Error> {
        self.finish()
    }

    fn finish(&mut self) -> Result<(), Error> {
        let Some(command_buffer) = self.command_buffer.take() else {
            return Ok(());
        };
        let completion = wait_for_completion_metal(&command_buffer);
        let metadata = self.metadata.take().ok_or(Error::MetalStateInvariant {
            state: "J2K Metal direct destination submission",
            reason: "committed command buffer lost its retained execution resources",
        })?;
        let DirectExecutionMetadata {
            retained_buffers,
            status_checks,
            scratch_buffers,
        } = metadata;
        let status_retirement = retire_direct_status_checks(
            &self.runtime,
            status_checks,
            if completion.is_ok() {
                DirectStatusRetirementMode::Validate
            } else {
                DirectStatusRetirementMode::RecycleWithoutRead
            },
        );
        drop(retained_buffers);
        let scratch_retirement = recycle_scratch_buffers(&self.runtime, scratch_buffers);
        completion.and(status_retirement).and(scratch_retirement)
    }
}

impl Drop for SubmittedDirectDestination {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

pub(in crate::compute::direct_grayscale_execute) fn commit_direct_destination(
    runtime: Arc<MetalRuntime>,
    command_buffer: CommandBuffer,
    metadata: DirectExecutionMetadata,
    consumer_ordering: DirectDestinationConsumerOrdering,
) -> Result<SubmittedDirectDestination, Error> {
    let mut consumer_waits = Vec::new();
    #[cfg(test)]
    let mut known_consumer_event_ptr = None;
    #[cfg(test)]
    let mut known_consumer_value = None;
    let completion_dependency = match consumer_ordering {
        DirectDestinationConsumerOrdering::Deferred => {
            #[cfg(test)]
            crate::compute::test_counters::record_direct_destination_event_allocation();
            let event = runtime.device.new_shared_event();
            command_buffer.encode_signal_event(&event, 1);
            #[cfg(test)]
            crate::compute::test_counters::record_direct_destination_event_signal();
            command_buffer.commit();
            Some(DirectDestinationCompletionDependency::Deferred(event))
        }
        DirectDestinationConsumerOrdering::HostCompletionOnly => {
            command_buffer.commit();
            None
        }
        DirectDestinationConsumerOrdering::Known {
            consumer_queue,
            timeline: _,
        } if consumer_queue.as_ptr() == runtime.queue.as_ptr() => {
            command_buffer.commit();
            None
        }
        DirectDestinationConsumerOrdering::Known {
            consumer_queue,
            timeline,
        } => {
            let producer_registry_id = runtime.device.registry_id();
            let consumer_registry_id = consumer_queue.device().registry_id();
            if producer_registry_id != consumer_registry_id {
                return Err(crate::error::metal_kernel_support_error(
                    "J2K Metal consumer queue belongs to a different device",
                    j2k_metal_support::MetalSupportError::MetalImageDeviceMismatch {
                        image_registry_id: producer_registry_id,
                        requested_registry_id: consumer_registry_id,
                    },
                ));
            }
            crate::batch_allocation::try_reserve_for_push(
                &mut consumer_waits,
                "J2K Metal known consumer queue completion wait",
            )?;
            let wait_command = new_command_buffer(&consumer_queue)?;
            let mut timeline = timeline.lock().map_err(|_| Error::MetalStatePoisoned {
                state: "J2K Metal consumer event timeline",
            })?;
            let value = timeline
                .next_value
                .checked_add(1)
                .ok_or(Error::MetalStateInvariant {
                    state: "J2K Metal consumer event timeline",
                    reason: "event timeline value overflowed",
                })?;
            let event = timeline
                .event
                .get_or_insert_with(|| {
                    #[cfg(test)]
                    crate::compute::test_counters::record_direct_destination_event_allocation();
                    runtime.device.new_event()
                })
                .clone();
            command_buffer.encode_signal_event(&event, value);
            #[cfg(test)]
            crate::compute::test_counters::record_direct_destination_event_signal();
            wait_command.encode_wait_for_event(&event, value);
            timeline.next_value = value;
            command_buffer.commit();
            wait_command.commit();
            #[cfg(test)]
            crate::compute::test_counters::record_direct_destination_event_wait();
            drop(timeline);
            consumer_waits.push(wait_command);
            #[cfg(test)]
            {
                known_consumer_event_ptr = Some(event.as_ptr() as usize);
                known_consumer_value = Some(value);
            }
            Some(DirectDestinationCompletionDependency::Known { _event: event })
        }
    };
    Ok(SubmittedDirectDestination {
        runtime,
        command_buffer: Some(command_buffer),
        metadata: Some(metadata),
        completion_dependency,
        consumer_waits,
        #[cfg(test)]
        known_consumer_event_ptr,
        #[cfg(test)]
        known_consumer_value,
    })
}
