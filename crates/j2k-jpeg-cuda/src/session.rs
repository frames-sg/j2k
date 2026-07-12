// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-runtime")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda-runtime")]
use j2k_cuda_runtime::{CudaContext, CudaDeviceBuffer};
#[cfg(feature = "cuda-runtime")]
use j2k_jpeg::adapter::JpegPlanCacheDiagnostics;
#[cfg(feature = "cuda-runtime")]
use j2k_jpeg::Decoder as CpuDecoder;

#[cfg(feature = "cuda-runtime")]
use crate::runtime::cuda_error;
#[cfg(feature = "cuda-runtime")]
use crate::Error;

#[cfg(feature = "cuda-runtime")]
mod host_ledger;
#[cfg(feature = "cuda-runtime")]
mod packet_cache;
#[cfg(feature = "cuda-runtime")]
mod runtime_state;
#[cfg(feature = "cuda-runtime")]
pub(crate) use host_ledger::HostOwnerLease;
#[cfg(feature = "cuda-runtime")]
pub(crate) use packet_cache::LeasedOwnedPacket;
#[cfg(feature = "cuda-runtime")]
use packet_cache::OwnedPacketPlanCache;
#[cfg(feature = "cuda-runtime")]
use runtime_state::SharedCudaRuntimeState;

#[cfg(feature = "cuda-runtime")]
#[doc(hidden)]
/// Clone-shared host-memory ownership diagnostics for CUDA JPEG operations.
///
/// The current fields are sampled while the session JPEG-host operation gate,
/// context pinned-upload gate, cache mutex, and host-ledger allocation gate are
/// held in their normal acquisition order. Peak fields are monotonic owner
/// high-water marks shared by every clone of the session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaJpegHostMemoryDiagnostics {
    /// Neutral JPEG plans retained by the session cache.
    pub cache_retained_bytes: usize,
    /// Packet, decoder, checkpoint, report, and pinned-pool leases currently active.
    pub active_owner_bytes: usize,
    /// Pinned upload staging retained by the session's one bound CUDA context.
    pub pinned_upload_retained_bytes: usize,
    /// Current cache, active-owner, and pinned-staging total.
    pub current_combined_bytes: usize,
    /// Highest active-owner lease total observed by the session.
    pub peak_active_owner_bytes: usize,
    /// Highest combined host-owner total observed by the session.
    pub peak_combined_bytes: usize,
}

#[derive(Clone, Default)]
/// Reusable CUDA JPEG decode session.
pub struct CudaSession {
    submissions: u64,
    #[cfg(feature = "cuda-runtime")]
    owned_packet_cache: Arc<OwnedPacketPlanCache>,
    #[cfg(feature = "cuda-runtime")]
    jpeg_host_operation_gate: Arc<Mutex<()>>,
    #[cfg(feature = "cuda-runtime")]
    runtime_state: Arc<SharedCudaRuntimeState>,
}

impl CudaSession {
    /// Number of decode submissions recorded through this session.
    pub fn submissions(&self) -> u64 {
        self.submissions
    }

    /// Number of neutral JPEG plans retained by the shared CUDA cache.
    ///
    /// This count includes an explicit unsupported plan even when it has no
    /// fast packet. If the cache mutex is poisoned, the method returns the
    /// last coherent atomic entry-count snapshot; use
    /// [`owned_cuda_packet_cache_diagnostics`](Self::owned_cuda_packet_cache_diagnostics)
    /// to observe the typed poison error.
    #[doc(hidden)]
    pub fn owned_cuda_packet_cache_len(&self) -> usize {
        #[cfg(feature = "cuda-runtime")]
        {
            self.owned_packet_cache.last_coherent_entries()
        }
        #[cfg(not(feature = "cuda-runtime"))]
        {
            0
        }
    }

    #[cfg(feature = "cuda-runtime")]
    /// Detailed retained-byte and admission diagnostics for the shared packet-plan cache.
    ///
    /// # Errors
    /// Returns [`Error::OwnedPacketCachePoisoned`] if a prior panic poisoned
    /// the synchronized cache state.
    #[doc(hidden)]
    pub fn owned_cuda_packet_cache_diagnostics(&self) -> Result<JpegPlanCacheDiagnostics, Error> {
        self.owned_packet_cache.diagnostics()
    }

    #[cfg(feature = "cuda-runtime")]
    /// Snapshot clone-shared CUDA JPEG host ownership and high-water marks.
    ///
    /// # Errors
    /// Returns a typed error if the operation gate, lazy runtime state, pinned
    /// staging pool, plan cache, or exact owner ledger is poisoned.
    #[doc(hidden)]
    pub fn owned_cuda_host_memory_diagnostics(
        &self,
    ) -> Result<CudaJpegHostMemoryDiagnostics, Error> {
        let operation_gate = self.jpeg_host_operation_gate();
        let _operation = operation_gate
            .lock()
            .map_err(|_| Error::JpegHostOperationPoisoned)?;
        let context = self.runtime_state.existing_context()?;
        let pinned_operation = context
            .as_ref()
            .map(j2k_cuda_runtime::CudaContext::begin_pinned_upload_operation)
            .transpose()
            .map_err(cuda_error)?;
        let pinned_upload_retained_bytes = pinned_operation
            .as_ref()
            .map(j2k_cuda_runtime::CudaPinnedUploadOperationGuard::diagnostics)
            .transpose()
            .map_err(cuda_error)?
            .map_or(0, |diagnostics| diagnostics.retained_bytes);
        let diagnostics = self
            .owned_packet_cache
            .host_memory_diagnostics(pinned_upload_retained_bytes)?;
        Ok(CudaJpegHostMemoryDiagnostics {
            cache_retained_bytes: diagnostics.cache_retained_bytes,
            active_owner_bytes: diagnostics.active_owner_bytes,
            pinned_upload_retained_bytes,
            current_combined_bytes: diagnostics.current_combined_bytes,
            peak_active_owner_bytes: diagnostics.peak_active_owner_bytes,
            peak_combined_bytes: diagnostics.peak_combined_bytes,
        })
    }

    #[cfg(feature = "cuda-runtime")]
    /// Whether a CUDA runtime context has been initialized successfully.
    pub fn is_runtime_initialized(&self) -> bool {
        self.runtime_state.is_initialized()
    }

    #[cfg(feature = "cuda-runtime")]
    /// Borrow or allocate a reusable CUDA output buffer for owned JPEG decode.
    ///
    /// Return buffers to the session with
    /// [`recycle_owned_cuda_output_buffer`](Self::recycle_owned_cuda_output_buffer).
    ///
    /// # Errors
    /// Returns a CUDA adapter error if the runtime is unavailable or the pool
    /// lock is poisoned.
    #[doc(hidden)]
    pub fn take_owned_cuda_output_buffer(
        &mut self,
        byte_len: usize,
    ) -> Result<CudaDeviceBuffer, Error> {
        let buffer = self
            .owned_output_pool()?
            .take(byte_len)
            .map_err(cuda_error)?;
        buffer.into_device_buffer().map_err(cuda_error)
    }

    #[cfg(feature = "cuda-runtime")]
    /// Return a CUDA output buffer to this session's owned JPEG decode pool.
    ///
    /// # Errors
    /// Returns a CUDA adapter error if the pool lock is poisoned.
    #[doc(hidden)]
    pub fn recycle_owned_cuda_output_buffer(
        &mut self,
        buffer: CudaDeviceBuffer,
    ) -> Result<(), Error> {
        self.owned_output_pool()?
            .recycle(buffer)
            .map_err(cuda_error)
    }

    #[cfg(feature = "cuda-runtime")]
    /// Number of reusable owned CUDA output buffers retained by this session.
    #[doc(hidden)]
    pub fn retained_owned_cuda_output_buffers(&self) -> Result<usize, Error> {
        self.runtime_state.retained_output_buffers()
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn resolve_owned_packet(
        &self,
        input: &[u8],
    ) -> Result<Option<LeasedOwnedPacket>, Error> {
        self.owned_packet_cache.resolve_packet(input)
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn resolve_owned_packet_from_decoder(
        &self,
        decoder: &CpuDecoder<'_>,
    ) -> Result<Option<LeasedOwnedPacket>, Error> {
        self.owned_packet_cache.resolve_packet_from_decoder(decoder)
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn jpeg_host_operation_gate(&self) -> Arc<Mutex<()>> {
        Arc::clone(&self.jpeg_host_operation_gate)
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn allocate_owned_host_owner<T>(
        &self,
        allocate: impl FnOnce(usize) -> Result<(T, usize), Error>,
    ) -> Result<(T, HostOwnerLease), Error> {
        self.owned_packet_cache.allocate_host_owner(allocate)
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn reserve_existing_host_owner(
        &self,
        bytes: usize,
    ) -> Result<HostOwnerLease, Error> {
        self.owned_packet_cache.reserve_existing_host_owner(bytes)
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn reserve_pinned_upload_retention(
        &self,
        operation: &j2k_cuda_runtime::CudaPinnedUploadOperationGuard<'_>,
    ) -> Result<HostOwnerLease, Error> {
        let retained_bytes = operation.diagnostics().map_err(cuda_error)?.retained_bytes;
        self.reserve_existing_host_owner(retained_bytes)
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn owned_host_live_bytes(&self) -> Result<usize, Error> {
        self.owned_packet_cache.host_live_bytes()
    }

    #[cfg(all(test, feature = "cuda-runtime"))]
    fn with_owned_packet_cache_limits(entry_limit: usize, host_byte_limit: usize) -> Self {
        Self {
            owned_packet_cache: Arc::new(OwnedPacketPlanCache::with_limits(
                entry_limit,
                host_byte_limit,
            )),
            ..Self::default()
        }
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn cuda_context(&self) -> Result<CudaContext, Error> {
        self.runtime_state.context()
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn bind_cuda_context(&self, context: &CudaContext) -> Result<CudaContext, Error> {
        self.runtime_state.bind_context(context)
    }

    #[cfg(feature = "cuda-runtime")]
    fn owned_output_pool(&self) -> Result<j2k_cuda_runtime::CudaBufferPool, Error> {
        self.runtime_state.owned_output_pool()
    }
}

impl j2k_core::DeviceSubmitSession for CudaSession {
    fn record_submit(&mut self) {
        self.submissions = self.submissions.saturating_add(1);
    }
}

#[doc(hidden)]
impl j2k_core::AcceleratorSession for CudaSession {
    fn backend_kind(&self) -> j2k_core::BackendKind {
        j2k_core::BackendKind::Cuda
    }

    fn execution_stats(&self) -> j2k_core::ExecutionStats {
        j2k_core::ExecutionStats {
            submissions: self.submissions,
            ..j2k_core::ExecutionStats::default()
        }
    }
}

impl std::fmt::Debug for CudaSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("CudaSession");
        debug.field("submissions", &self.submissions);
        debug.field(
            "owned_cuda_packet_cache_len",
            &self.owned_cuda_packet_cache_len(),
        );
        #[cfg(feature = "cuda-runtime")]
        debug.field("runtime_initialized", &self.is_runtime_initialized());
        #[cfg(feature = "cuda-runtime")]
        debug.field(
            "owned_cuda_packet_cache_diagnostics",
            &self.owned_packet_cache.try_diagnostics(),
        );
        #[cfg(feature = "cuda-runtime")]
        debug.field(
            "retained_owned_cuda_output_buffers",
            &self.runtime_state.try_retained_output_buffers(),
        );
        debug.finish_non_exhaustive()
    }
}

#[cfg(all(test, feature = "cuda-runtime"))]
mod tests;
