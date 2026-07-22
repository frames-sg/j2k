// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::{Arc, Mutex, MutexGuard};

use j2k_core::{BackendRequest, DeviceSubmission, PixelFormat};

use crate::{profile, Error, MetalBackendSession, MetalSession, Surface};

use super::execute::process_batch;
use super::heuristics::group_metal_requests;
use super::request::{batch_scheduler_invariant, BatchOp, QueuedRequest};

#[derive(Default)]
pub(crate) struct SessionState {
    pub(crate) submissions: u64,
    pub(super) queued: Vec<QueuedRequest>,
    pub(super) completed: Vec<Option<Result<Surface, Error>>>,
    pub(super) free_slots: Vec<usize>,
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
        if let Some(position) = session
            .queued
            .iter()
            .position(|request| request.output_slot == slot)
        {
            session.queued.remove(position);
        }
        if let Some(completed) = session.completed.get_mut(slot) {
            *completed = None;
        }
        let _release_result = release_surface_slot(&mut session, slot);
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
    let reuses_slot = !state.free_slots.is_empty();
    if !reuses_slot {
        let slot_capacity = state.completed.len().checked_add(1).ok_or_else(|| {
            Error::BatchInfrastructure(j2k_core::BatchInfrastructureError::AllocationTooLarge {
                what: "J2K Metal reusable completion slots",
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            })
        })?;
        crate::batch_allocation::try_reserve_for_push(
            &mut state.completed,
            "J2K Metal queued completion slots",
        )?;
        j2k_core::try_batch_reserve_to(
            &mut state.free_slots,
            slot_capacity,
            "J2K Metal reusable completion slots",
        )?;
    }
    crate::batch_allocation::try_reserve_for_push(&mut state.queued, "J2K Metal queued requests")?;
    let aggregate =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal queued request state");
    aggregate.preflight(&[
        crate::batch_allocation::BatchMetadataRequest::of::<MetalSubmission>(
            retained_submission_capacity,
        ),
        crate::batch_allocation::BatchMetadataRequest::of::<QueuedRequest>(state.queued.capacity()),
        crate::batch_allocation::BatchMetadataRequest::of::<Option<Result<Surface, Error>>>(
            state.completed.capacity(),
        ),
        crate::batch_allocation::BatchMetadataRequest::of::<usize>(state.free_slots.capacity()),
    ])?;
    let slot = if reuses_slot {
        state
            .free_slots
            .pop()
            .ok_or_else(|| batch_scheduler_invariant("reusable slot disappeared"))?
    } else {
        let slot = state.completed.len();
        state.completed.push(None);
        slot
    };
    state
        .queued
        .push(QueuedRequest::new(input, fmt, backend, op, slot));
    Ok(MetalSubmission {
        session: session.shared.clone(),
        slot: Some(slot),
    })
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
    let result = session
        .completed
        .get_mut(slot)
        .and_then(Option::take)
        .ok_or_else(|| Error::MetalKernel {
            message: format!("missing queued J2K Metal surface for slot {slot}"),
        });
    release_surface_slot(session, slot)?;
    result?
}

pub(super) fn release_surface_slot(session: &mut SessionState, slot: usize) -> Result<(), Error> {
    if session.free_slots.contains(&slot) {
        return Ok(());
    }
    if session.free_slots.len() >= session.free_slots.capacity() {
        return Err(Error::MetalStateInvariant {
            state: "J2K Metal batch free-slot ledger",
            reason: "slot creation did not retain capacity for its eventual release",
        });
    }
    session.free_slots.push(slot);
    Ok(())
}
