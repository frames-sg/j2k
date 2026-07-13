// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::{Arc, Mutex, MutexGuard};

#[cfg(target_os = "macos")]
use j2k_core::BackendKind;
use j2k_core::BackendRequest;
use j2k_jpeg::adapter::{
    JpegCachedPlan, JpegPlanCache, JpegPlanCacheDiagnostics, SharedJpegFastPacket, SharedJpegInput,
};
#[cfg(target_os = "macos")]
use j2k_jpeg::Decoder as CpuDecoder;
#[cfg(target_os = "macos")]
use j2k_metal_support::{MetalRuntimeSession, MetalSupportError};
#[cfg(target_os = "macos")]
use metal::Device;

#[cfg(target_os = "macos")]
use crate::compute;
use crate::{batch, plan_owner_ledger::PlanOwnerLedger, Error};

mod allocation;
mod completions;

pub(crate) use allocation::submission_capacity_bytes;
use allocation::{capacity_bytes, prepare_queue_growth, projected_push_capacity};

#[cfg(target_os = "macos")]
#[derive(Clone)]
/// Reusable Metal device session for decode and encode submissions.
pub struct MetalBackendSession {
    runtime_session: MetalRuntimeSession<compute::MetalRuntime, MetalSupportError>,
}

#[cfg(target_os = "macos")]
impl MetalBackendSession {
    /// Create a session bound to an existing Metal device.
    pub fn new(device: Device) -> Self {
        Self {
            runtime_session: MetalRuntimeSession::new(device),
        }
    }

    /// Create a session from the system default Metal device.
    pub fn system_default() -> Result<Self, Error> {
        MetalRuntimeSession::system_default()
            .map(|runtime_session| Self { runtime_session })
            .map_err(|error| compute::runtime_initialization_error(&error))
    }

    /// Metal device used by this session.
    pub fn device(&self) -> &metal::DeviceRef {
        self.runtime_session.device()
    }

    pub(crate) fn runtime_result(&self) -> &Result<compute::MetalRuntime, MetalSupportError> {
        self.runtime_session
            .get_or_init_runtime(|device| compute::MetalRuntime::new_with_device(device.clone()))
    }

    #[cfg(test)]
    pub(crate) fn runtime_initialized_for_test(&self) -> bool {
        self.runtime_session.runtime_initialized()
    }

    #[cfg(test)]
    pub(crate) fn runtime_ptr_for_test(&self) -> Option<*const compute::MetalRuntime> {
        self.runtime_session
            .runtime_result()
            .and_then(|runtime| runtime.as_ref().ok())
            .map(std::ptr::from_ref::<compute::MetalRuntime>)
    }
}

#[cfg(target_os = "macos")]
#[doc(hidden)]
impl j2k_core::AcceleratorSession for MetalBackendSession {
    fn backend_kind(&self) -> BackendKind {
        BackendKind::Metal
    }
}

#[cfg(target_os = "macos")]
impl core::fmt::Debug for MetalBackendSession {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetalBackendSession")
            .field("device", &self.runtime_session.device_handle().name())
            .field(
                "runtime_initialized",
                &self.runtime_session.runtime_initialized(),
            )
            .finish()
    }
}

#[cfg(not(target_os = "macos"))]
#[derive(Clone, Copy, Debug, Default)]
/// Placeholder Metal session for non-macOS builds.
pub struct MetalBackendSession {
    _private: (),
}

#[cfg(not(target_os = "macos"))]
impl MetalBackendSession {
    /// Return `Error::MetalUnavailable` on hosts without Metal support.
    pub fn system_default() -> Result<Self, Error> {
        Err(Error::MetalUnavailable)
    }
}

#[derive(Debug)]
pub(crate) struct ResolvedJpegPlan {
    pub(crate) input: SharedJpegInput,
    pub(crate) fast_packet: Option<SharedJpegFastPacket>,
    pub(crate) shape: batch::BatchShape,
}

impl ResolvedJpegPlan {
    fn from_cached(plan: &JpegCachedPlan) -> Self {
        let shape = batch::BatchShape::from_summary(plan.batch_summary(), plan.color_space());
        Self {
            input: plan.input().clone(),
            fast_packet: plan.fast_packet().cloned(),
            shape,
        }
    }

    fn uninspected(input: SharedJpegInput) -> Self {
        Self {
            input,
            fast_packet: None,
            shape: batch::BatchShape::unknown(),
        }
    }
}

#[derive(Default)]
pub(crate) struct SessionState {
    pub(crate) submissions: u64,
    pub(crate) queued: Vec<crate::batch::QueuedRequest>,
    pub(crate) completed: Vec<Option<Result<crate::Surface, crate::Error>>>,
    #[cfg(target_os = "macos")]
    pub(crate) backend_session: Option<MetalBackendSession>,
    jpeg_plans: JpegPlanCache,
    queued_plan_ledger: PlanOwnerLedger,
    retained_execution_metadata_bytes: usize,
    completed_host_bytes: usize,
    peak_collective_host_bytes: usize,
    #[cfg(test)]
    queue_growth_capacity_override: Option<(usize, usize)>,
}

impl SessionState {
    #[cfg(target_os = "macos")]
    pub(crate) fn with_backend_session(backend_session: MetalBackendSession) -> Self {
        Self {
            backend_session: Some(backend_session),
            ..Self::default()
        }
    }

    pub(crate) fn queue_request(
        &mut self,
        request: crate::batch::QueuedRequest,
    ) -> Result<usize, Error> {
        self.queue_request_with_retained_metadata(request, 0)
    }

    pub(crate) fn queue_request_with_retained(
        &mut self,
        request: crate::batch::QueuedRequest,
        retained_submission_capacity: usize,
    ) -> Result<usize, Error> {
        let submission_metadata_bytes = submission_capacity_bytes(retained_submission_capacity)?;
        self.queue_request_with_retained_metadata(request, submission_metadata_bytes)
    }

    pub(crate) fn queue_request_with_retained_metadata(
        &mut self,
        request: crate::batch::QueuedRequest,
        retained_metadata_bytes: usize,
    ) -> Result<usize, Error> {
        let execution_metadata_bytes = self
            .retained_execution_metadata_bytes
            .max(retained_metadata_bytes);
        let owner_admission = self.queued_plan_ledger.preflight(
            &self.queued,
            &request,
            self.jpeg_plans.diagnostics().retained_bytes,
        )?;
        let projected_completed_capacity = projected_push_capacity(
            self.completed.len(),
            self.completed.capacity(),
            "JPEG Metal queued completion capacity",
        )?;
        let projected_queued_capacity = projected_push_capacity(
            self.queued.len(),
            self.queued.capacity(),
            "JPEG Metal queued request capacity",
        )?;
        let mut growth = prepare_queue_growth(
            self,
            owner_admission.retained_bytes(),
            execution_metadata_bytes,
            projected_queued_capacity,
            projected_completed_capacity,
        )?;
        let final_queued_capacity = growth.queued_capacity(self.queued.capacity());
        let final_completed_capacity = growth.completed_capacity(self.completed.capacity());
        self.preflight_collective_queue_state(
            owner_admission.retained_bytes(),
            execution_metadata_bytes,
            final_queued_capacity,
            final_completed_capacity,
        )?;
        let collective_host_bytes = self.collective_queue_state_bytes(
            owner_admission.retained_bytes(),
            execution_metadata_bytes,
            final_queued_capacity,
            final_completed_capacity,
        )?;

        // All fallible work is complete. Moving elements into admitted
        // replacements and pushing one slot cannot allocate at these capacities.
        if let Some(mut queued) = growth.queued.take() {
            debug_assert!(queued.capacity() > self.queued.len());
            queued.append(&mut self.queued);
            self.queued = queued;
        }
        if let Some(mut completed) = growth.completed.take() {
            debug_assert!(completed.capacity() > self.completed.len());
            completed.append(&mut self.completed);
            self.completed = completed;
        }
        let slot = self.completed.len();
        self.completed.push(None);
        self.queued.push(request.with_output_slot(slot));
        self.queued_plan_ledger.commit(owner_admission);
        self.retained_execution_metadata_bytes = execution_metadata_bytes;
        self.peak_collective_host_bytes =
            self.peak_collective_host_bytes.max(collective_host_bytes);
        Ok(slot)
    }

    pub(crate) fn take_queued_requests(
        &mut self,
    ) -> Result<Vec<crate::batch::QueuedRequest>, Error> {
        let cache_retained_bytes = self.jpeg_plans.diagnostics().retained_bytes;
        let metadata_live_bytes = self
            .session_metadata_live_bytes()?
            .checked_add(self.retained_execution_metadata_bytes)
            .ok_or(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                "JPEG Metal executing metadata owner baseline overflow",
            ))?;
        self.queued_plan_ledger.reset();
        let mut queued = std::mem::take(&mut self.queued);
        crate::batch::stamp_execution_owner_baseline(
            &mut queued,
            cache_retained_bytes,
            metadata_live_bytes,
        );
        self.retained_execution_metadata_bytes = 0;
        Ok(queued)
    }

    pub(crate) fn complete_queued_with_error(&mut self, error: &Error) {
        self.queued_plan_ledger.reset();
        self.retained_execution_metadata_bytes = 0;
        for request in std::mem::take(&mut self.queued) {
            if let Some(slot) = self.completed.get_mut(request.output_slot) {
                *slot = Some(Err(error.clone()));
            }
        }
    }

    pub(crate) fn resolve_jpeg_plan(
        &mut self,
        input: &[u8],
        backend: BackendRequest,
    ) -> Result<ResolvedJpegPlan, Error> {
        self.resolve_jpeg_plan_with_external_live(input, backend, 0)
    }

    pub(crate) fn resolve_jpeg_plan_with_external_live(
        &mut self,
        input: &[u8],
        backend: BackendRequest,
        additional_external_live_bytes: usize,
    ) -> Result<ResolvedJpegPlan, Error> {
        let adapter_live_bytes =
            self.plan_operation_external_live_bytes(additional_external_live_bytes)?;
        if !uses_inspected_metal_plan(backend) {
            let all_external_live_bytes = adapter_live_bytes
                .checked_add(self.jpeg_plans.diagnostics().retained_bytes)
                .ok_or(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                    "JPEG Metal session plan owner baseline overflow",
                ))?;
            return SharedJpegInput::try_copy_from_slice_with_external_live(
                input,
                all_external_live_bytes,
            )
            .map(ResolvedJpegPlan::uninspected)
            .map_err(Error::from);
        }

        self.jpeg_plans
            .resolve_with_external_live(input, adapter_live_bytes)
            .map(|plan| ResolvedJpegPlan::from_cached(&plan))
            .map_err(Error::from)
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn resolve_jpeg_plan_with_decoder_and_external_live<'a>(
        &mut self,
        input: &'a [u8],
        additional_external_live_bytes: usize,
    ) -> Result<(ResolvedJpegPlan, CpuDecoder<'a>), Error> {
        let adapter_live_bytes =
            self.plan_operation_external_live_bytes(additional_external_live_bytes)?;
        self.jpeg_plans
            .resolve_with_decoder_and_external_live(input, adapter_live_bytes)
            .map(|(plan, decoder)| (ResolvedJpegPlan::from_cached(&plan), decoder))
            .map_err(Error::from)
    }

    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn resolve_shared_jpeg_plan(
        &mut self,
        input: SharedJpegInput,
        backend: BackendRequest,
    ) -> Result<ResolvedJpegPlan, Error> {
        self.resolve_shared_jpeg_plan_with_external_live(input, backend, 0)
    }

    pub(crate) fn resolve_shared_jpeg_plan_with_external_live(
        &mut self,
        input: SharedJpegInput,
        backend: BackendRequest,
        additional_external_live_bytes: usize,
    ) -> Result<ResolvedJpegPlan, Error> {
        let adapter_live_bytes =
            self.plan_operation_external_live_bytes(additional_external_live_bytes)?;
        if !uses_inspected_metal_plan(backend) {
            return Ok(ResolvedJpegPlan::uninspected(input));
        }

        self.jpeg_plans
            .resolve_shared_with_external_live(input, adapter_live_bytes)
            .map(|plan| ResolvedJpegPlan::from_cached(&plan))
            .map_err(Error::from)
    }

    pub(crate) fn resolve_arc_jpeg_plan_with_external_live(
        &mut self,
        input: Arc<[u8]>,
        backend: BackendRequest,
        additional_external_live_bytes: usize,
    ) -> Result<ResolvedJpegPlan, Error> {
        let adapter_live_bytes =
            self.plan_operation_external_live_bytes(additional_external_live_bytes)?;
        let all_external_live_bytes = adapter_live_bytes
            .checked_add(self.jpeg_plans.diagnostics().retained_bytes)
            .ok_or(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                "JPEG Metal shared-input owner baseline overflow",
            ))?;
        let input =
            SharedJpegInput::try_from_arc_with_external_live(input, all_external_live_bytes)?;
        self.resolve_shared_jpeg_plan_with_external_live(
            input,
            backend,
            additional_external_live_bytes,
        )
    }

    fn plan_operation_external_live_bytes(
        &self,
        additional_external_live_bytes: usize,
    ) -> Result<usize, Error> {
        let metadata_live_bytes = self.session_metadata_live_bytes()?;
        let all_additional = additional_external_live_bytes
            .checked_add(metadata_live_bytes)
            .ok_or(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                "JPEG Metal plan and metadata owner baseline overflow",
            ))?;
        let adapter_live_bytes = self
            .queued_plan_ledger
            .external_live_bytes(all_additional)?;
        let complete_live_bytes = adapter_live_bytes
            .checked_add(self.jpeg_plans.diagnostics().retained_bytes)
            .ok_or(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                "JPEG Metal complete plan-operation baseline overflow",
            ))?;
        if complete_live_bytes > j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES {
            return Err(j2k_jpeg::adapter::JpegPlanCacheError::Limit {
                what: "JPEG Metal plan operation owner graph",
                requested: complete_live_bytes,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            }
            .into());
        }
        Ok(adapter_live_bytes)
    }

    fn session_metadata_live_bytes(&self) -> Result<usize, Error> {
        let queued_bytes = capacity_bytes::<crate::batch::QueuedRequest>(
            self.queued.capacity(),
            "JPEG Metal queued request metadata",
        )?;
        let completed_bytes = capacity_bytes::<Option<Result<crate::Surface, crate::Error>>>(
            self.completed.capacity(),
            "JPEG Metal completion metadata",
        )?;
        queued_bytes
            .checked_add(completed_bytes)
            .and_then(|bytes| bytes.checked_add(self.completed_host_bytes))
            .ok_or_else(|| {
                j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                    "JPEG Metal session metadata owner baseline overflow",
                )
                .into()
            })
    }

    fn preflight_collective_queue_state(
        &self,
        queued_owner_bytes: usize,
        retained_execution_metadata_bytes: usize,
        queued_capacity: usize,
        completed_capacity: usize,
    ) -> Result<(), Error> {
        let owner_and_cache_bytes = queued_owner_bytes
            .checked_add(self.jpeg_plans.diagnostics().retained_bytes)
            .ok_or(j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                "JPEG Metal collective queue owner baseline overflow",
            ))?;
        let aggregate = crate::batch_allocation::BatchMetadataBudget::with_external_live(
            "JPEG Metal collective queued request state",
            owner_and_cache_bytes,
        );
        aggregate
            .preflight(&[
                crate::batch_allocation::BatchMetadataRequest::of::<u8>(
                    retained_execution_metadata_bytes,
                ),
                crate::batch_allocation::BatchMetadataRequest::of::<u8>(self.completed_host_bytes),
                crate::batch_allocation::BatchMetadataRequest::of::<crate::batch::QueuedRequest>(
                    queued_capacity,
                ),
                crate::batch_allocation::BatchMetadataRequest::of::<
                    Option<Result<crate::Surface, crate::Error>>,
                >(completed_capacity),
            ])
            .map_err(Error::from)
    }

    fn collective_queue_state_bytes(
        &self,
        queued_owner_bytes: usize,
        retained_execution_metadata_bytes: usize,
        queued_capacity: usize,
        completed_capacity: usize,
    ) -> Result<usize, Error> {
        let queued_metadata_bytes = capacity_bytes::<crate::batch::QueuedRequest>(
            queued_capacity,
            "JPEG Metal queued request metadata",
        )?;
        let completed_metadata_bytes = capacity_bytes::<
            Option<Result<crate::Surface, crate::Error>>,
        >(
            completed_capacity, "JPEG Metal completion metadata"
        )?;
        queued_owner_bytes
            .checked_add(self.jpeg_plans.diagnostics().retained_bytes)
            .and_then(|bytes| bytes.checked_add(retained_execution_metadata_bytes))
            .and_then(|bytes| bytes.checked_add(self.completed_host_bytes))
            .and_then(|bytes| bytes.checked_add(queued_metadata_bytes))
            .and_then(|bytes| bytes.checked_add(completed_metadata_bytes))
            .ok_or_else(|| {
                j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                    "JPEG Metal collective queue byte count overflow",
                )
                .into()
            })
    }

    pub(crate) const fn jpeg_plan_cache_diagnostics(&self) -> JpegPlanCacheDiagnostics {
        self.jpeg_plans.diagnostics()
    }

    pub(crate) fn peak_collective_host_bytes(&self) -> usize {
        self.peak_collective_host_bytes
            .max(self.jpeg_plans.diagnostics().peak_bytes)
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn backend_session(&mut self) -> Result<&MetalBackendSession, Error> {
        if self.backend_session.is_none() {
            self.backend_session = Some(MetalBackendSession::system_default()?);
        }
        self.backend_session.as_ref().ok_or_else(|| {
            j2k_jpeg::adapter::JpegPlanCacheError::Invariant(
                "JPEG Metal backend session is missing after initialization",
            )
            .into()
        })
    }
}

const fn uses_inspected_metal_plan(backend: BackendRequest) -> bool {
    matches!(backend, BackendRequest::Metal)
        || (cfg!(target_os = "macos") && matches!(backend, BackendRequest::Auto))
}

#[derive(Clone, Default)]
pub(crate) struct SharedSession(pub(crate) Arc<Mutex<SessionState>>);

impl SharedSession {
    pub(crate) fn lock(&self) -> Result<MutexGuard<'_, SessionState>, Error> {
        self.0.lock().map_err(|_| Error::MetalStatePoisoned {
            state: "JPEG Metal session",
        })
    }
}

#[derive(Default)]
/// Shared batching session used by `JpegTileBatch` and submit APIs.
pub struct MetalSession {
    pub(crate) shared: SharedSession,
}

impl MetalSession {
    /// Create a tile batching session that reuses an existing Metal backend session.
    #[cfg(target_os = "macos")]
    pub fn with_backend_session(backend_session: MetalBackendSession) -> Self {
        Self {
            shared: SharedSession(Arc::new(Mutex::new(SessionState::with_backend_session(
                backend_session,
            )))),
        }
    }

    /// Number of Metal or emulated submissions flushed through this session.
    pub fn submissions(&self) -> Result<u64, Error> {
        Ok(self.shared.lock()?.submissions)
    }

    /// Current JPEG input-plan cache retention and admission diagnostics.
    #[doc(hidden)]
    pub fn jpeg_plan_cache_diagnostics(&self) -> Result<JpegPlanCacheDiagnostics, Error> {
        Ok(self.shared.lock()?.jpeg_plan_cache_diagnostics())
    }

    /// Peak collectively retained JPEG cache, queue-owner, and queue-metadata bytes.
    #[doc(hidden)]
    pub fn peak_collective_host_bytes(&self) -> Result<usize, Error> {
        Ok(self.shared.lock()?.peak_collective_host_bytes())
    }
}

impl core::fmt::Debug for MetalSession {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetalSession")
            .field("submissions", &self.submissions())
            .field("jpeg_plan_cache", &self.jpeg_plan_cache_diagnostics())
            .field(
                "peak_collective_host_bytes",
                &self.peak_collective_host_bytes(),
            )
            .finish()
    }
}

#[cfg(test)]
#[path = "session/tests.rs"]
mod tests;
