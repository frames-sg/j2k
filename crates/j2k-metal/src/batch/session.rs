// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::{Arc, Mutex, MutexGuard};

use j2k_core::{BackendRequest, DeviceSubmission, PixelFormat};
use j2k_metal_support::{CompletionSlot, CompletionSlots};

use crate::{profile, Error, MetalBackendSession, MetalSession, Surface};

use super::execute::process_batch;
use super::heuristics::group_metal_requests;
use super::request::{batch_scheduler_invariant, BatchOp, QueuedRequest};

#[derive(Default)]
pub(crate) struct SessionState {
    pub(crate) submissions: u64,
    pub(super) queued: Vec<QueuedRequest>,
    pub(super) completed: CompletionSlots<Result<Surface, Error>>,
}

#[derive(Clone)]
pub(crate) struct SharedSession {
    state: Arc<Mutex<SessionState>>,
    backend: Option<MetalBackendSession>,
}

impl Default for SharedSession {
    fn default() -> Self {
        Self {
            state: Arc::new(Mutex::new(SessionState::default())),
            backend: None,
        }
    }
}

impl SharedSession {
    #[cfg(target_os = "macos")]
    pub(crate) fn with_backend_session(backend: MetalBackendSession) -> Self {
        Self {
            state: Arc::new(Mutex::new(SessionState::default())),
            backend: Some(backend),
        }
    }

    pub(crate) fn lock(&self) -> Result<MutexGuard<'_, SessionState>, Error> {
        self.state.lock().map_err(|_| Error::MetalStatePoisoned {
            state: "J2K Metal session",
        })
    }

    pub(crate) fn backend_session(&self) -> Option<&MetalBackendSession> {
        self.backend.as_ref()
    }
}

/// Pending surface decode submitted through [`MetalSession`](crate::MetalSession).
pub struct MetalSubmission {
    session: SharedSession,
    pub(super) slot: Option<usize>,
}

#[doc(hidden)]
impl DeviceSubmission for MetalSubmission {
    type Output = Surface;
    type Error = Error;

    fn wait(mut self) -> Result<Self::Output, Self::Error> {
        let slot = self.slot.take().ok_or(Error::MetalStateInvariant {
            state: "J2K Metal submission",
            reason: "submission slot was already consumed",
        })?;
        let mut session = self.session.lock()?;
        flush_if_needed(&mut session, self.session.backend_session());
        take_surface(&mut session, slot)
    }
}

impl Drop for MetalSubmission {
    fn drop(&mut self) {
        let Some(slot) = self.slot.take() else {
            return;
        };
        let Ok(mut session) = self.session.lock() else {
            return;
        };
        let SessionState {
            queued, completed, ..
        } = &mut *session;
        completed.cancel(queued, slot, |request| request.output_slot);
    }
}

pub(crate) fn queue_tile_request(
    session: &mut MetalSession,
    input: &[u8],
    fmt: PixelFormat,
    backend: BackendRequest,
    op: BatchOp,
) -> Result<MetalSubmission, Error> {
    queue_tile_request_shared(session, Arc::<[u8]>::from(input), fmt, backend, op)
}

pub(crate) fn queue_tile_request_shared(
    session: &mut MetalSession,
    input: Arc<[u8]>,
    fmt: PixelFormat,
    backend: BackendRequest,
    op: BatchOp,
) -> Result<MetalSubmission, Error> {
    queue_tile_request_shared_with_retained(session, input, fmt, backend, op, 0)
}

pub(crate) fn queue_tile_request_shared_with_retained(
    session: &mut MetalSession,
    input: Arc<[u8]>,
    fmt: PixelFormat,
    backend: BackendRequest,
    op: BatchOp,
    retained_submission_capacity: usize,
) -> Result<MetalSubmission, Error> {
    let mut state = session.shared.lock()?;
    reserve_queue_metadata(&mut state, retained_submission_capacity)?;
    let slot = state
        .completed
        .allocate()
        .ok_or_else(|| batch_scheduler_invariant("completion slot capacity was not reserved"))?;
    state
        .queued
        .push(QueuedRequest::new(input, fmt, backend, op, slot));
    drop(state);
    Ok(MetalSubmission {
        session: session.shared.clone(),
        slot: Some(slot),
    })
}

fn reserve_queue_metadata(
    state: &mut SessionState,
    retained_submission_capacity: usize,
) -> Result<(), Error> {
    use crate::batch_allocation::{BatchMetadataBudget, BatchMetadataRequest};

    let grows_queued = state.queued.len() == state.queued.capacity();
    let grows_completed =
        !state.completed.can_reuse() && state.completed.len() == state.completed.capacity();
    let next_capacity = |len: usize, capacity: usize| {
        len.checked_add(1)
            .and_then(|required| capacity.checked_mul(2).map(|doubled| doubled.max(required)))
            .ok_or(j2k_core::BatchInfrastructureError::AllocationTooLarge {
                what: "J2K Metal queued request state",
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            })
    };
    let queued_capacity = if grows_queued {
        next_capacity(state.queued.len(), state.queued.capacity())?
    } else {
        0
    };
    let completed_capacity = if grows_completed {
        next_capacity(state.completed.len(), state.completed.capacity())?
    } else {
        0
    };
    let mut budget = BatchMetadataBudget::new("J2K Metal queued request state");
    budget.account_capacity::<MetalSubmission>(retained_submission_capacity)?;
    budget.account_capacity::<QueuedRequest>(state.queued.capacity())?;
    budget
        .account_capacity::<CompletionSlot<Result<Surface, Error>>>(state.completed.capacity())?;
    // Old storage stays live until both replacements fit, including their
    // allocator-reported capacities. A refusal leaves the session untouched.
    budget.preflight(&[
        BatchMetadataRequest::of::<QueuedRequest>(queued_capacity),
        BatchMetadataRequest::of::<CompletionSlot<Result<Surface, Error>>>(completed_capacity),
    ])?;
    let queued = grows_queued
        .then(|| budget.try_vec::<QueuedRequest>(queued_capacity, "J2K Metal queued requests"))
        .transpose()?;
    let completed = grows_completed
        .then(|| {
            budget.try_vec::<CompletionSlot<Result<Surface, Error>>>(
                completed_capacity,
                "J2K Metal queued completion slots",
            )
        })
        .transpose()?;

    if let Some(completed) = completed {
        if !state.completed.replace_storage(completed) {
            return Err(batch_scheduler_invariant(
                "replacement completion capacity is too small",
            ));
        }
    }
    if let Some(mut queued) = queued {
        queued.append(&mut state.queued);
        state.queued = queued;
    }
    Ok(())
}

fn flush_if_needed(session: &mut SessionState, backend: Option<&MetalBackendSession>) {
    if session.queued.is_empty() {
        return;
    }

    let profile_enabled = profile::metal_profile_stages_enabled();
    let queued = std::mem::take(&mut session.queued);
    let request_count = queued.len();
    let mut slot_budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal grouping recovery slots");
    let mut output_slots =
        match slot_budget.try_vec(queued.len(), "J2K Metal grouping recovery output slots") {
            Ok(slots) => slots,
            Err(error) => {
                for request in queued {
                    session.completed[request.output_slot] = Some(Err(error.into()));
                }
                return;
            }
        };
    output_slots.extend(queued.iter().map(|request| request.output_slot));
    let group_started = profile::profile_now(profile_enabled);
    let batches = match group_metal_requests(queued) {
        Ok(batches) => batches,
        Err(error) => {
            for output_slot in output_slots {
                session.completed[output_slot] = Some(Err(error.into()));
            }
            return;
        }
    };
    drop(output_slots);
    if profile_enabled {
        profile::emit_metal_batch_profile_row(
            "decode",
            &profile::MetalBatchProfileRow {
                slice: "decode_batch",
                stage: "group",
                pipeline: "metal_cpu_hybrid",
                processor: "scheduler",
                route: "all",
                backend: profile::MetalBatchProfileValue::Mixed,
                fmt: profile::MetalBatchProfileValue::Mixed,
                request_count,
                output_count: batches.len(),
                elapsed_us: profile::elapsed_us(group_started),
                outcome: "grouped",
            },
        );
    }

    for batch in batches {
        process_batch(session, batch, backend);
    }
}

fn take_surface(session: &mut SessionState, slot: usize) -> Result<Surface, Error> {
    session
        .completed
        .release(slot)
        .flatten()
        .ok_or_else(|| Error::MetalKernel {
            message: format!("missing queued J2K Metal surface for slot {slot}"),
        })?
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn abandoned_metal_completion_releases_its_device_buffer() {
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }
        let device = j2k_metal_support::system_default_device().expect("required Metal device");
        let shared = SharedSession::default();
        let (submission, weak) = objc2::rc::autoreleasepool(|_| {
            let buffer = j2k_metal_support::checked_shared_buffer(&device, 64)
                .expect("Metal completion buffer");
            let weak = objc2::rc::Weak::from_retained(&buffer);
            let surface = Surface::from_metal_buffer(buffer, (8, 8), PixelFormat::Gray8)
                .expect("completed Metal surface");
            let mut state = shared.lock().expect("completion state");
            state
                .completed
                .try_reserve_slot("test Metal completion")
                .expect("reserve completion");
            let slot = state.completed.allocate().expect("active completion");
            state.completed[slot] = Some(Ok(surface));
            (
                MetalSubmission {
                    session: shared.clone(),
                    slot: Some(slot),
                },
                weak,
            )
        });
        assert!(weak.load().is_some());
        drop(submission);
        assert!(shared
            .lock()
            .expect("released Metal state")
            .completed
            .is_empty());
        assert!(
            weak.load().is_none(),
            "the abandoned completion was the device buffer's last owner"
        );
    }
}
