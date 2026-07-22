// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-runtime")]
use j2k_cuda_runtime::{
    CudaBufferPool, CudaClassicDecodeTableResources, CudaContext, CudaContextDiagnostics,
    CudaHtj2kDecodeTableResources, CudaHtj2kDecodeTables, CudaHtj2kEncodeResources,
};
#[cfg(feature = "cuda-runtime")]
use j2k_native::{ht_uvlc_table0, ht_uvlc_table1, ht_vlc_table0, ht_vlc_table1};
#[cfg(feature = "cuda-runtime")]
use std::num::NonZeroUsize;
#[cfg(all(test, feature = "cuda-runtime"))]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "cuda-runtime")]
use std::sync::Arc;

#[cfg(feature = "cuda-runtime")]
use crate::runtime::cuda_error;
#[cfg(feature = "cuda-runtime")]
use crate::Error;

#[cfg(all(test, feature = "cuda-runtime"))]
static HTJ2K_DECODE_TABLE_UPLOADS: AtomicUsize = AtomicUsize::new(0);
#[cfg(all(test, feature = "cuda-runtime"))]
static CLASSIC_DECODE_TABLE_UPLOADS: AtomicUsize = AtomicUsize::new(0);

/// Stable retention snapshot for one internal CUDA decode buffer pool.
#[cfg(feature = "cuda-runtime")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaDecodePoolSnapshot {
    /// Completed device buffers immediately available for reuse.
    pub cached_buffers: usize,
    /// Completed device-allocation bytes immediately available for reuse.
    pub cached_bytes: usize,
    /// Buffers held until queued work establishes completion.
    pub deferred_buffers: usize,
    /// Device-allocation bytes held until queued work establishes completion.
    pub deferred_bytes: usize,
    /// Active completion guards preventing deferred-buffer reuse.
    pub reuse_holds: usize,
    /// Highest completed allocation-byte total observed by this pool.
    pub peak_cached_bytes: usize,
    /// Highest deferred allocation-byte total observed by this pool.
    pub peak_deferred_bytes: usize,
}

#[cfg(feature = "cuda-runtime")]
impl CudaDecodePoolSnapshot {
    /// Device bytes currently retained either for reuse or pending completion.
    #[must_use]
    pub const fn retained_bytes(self) -> usize {
        self.cached_bytes.saturating_add(self.deferred_bytes)
    }

    /// Conservative upper bound obtained by adding the independent pool peaks.
    #[must_use]
    pub const fn peak_retained_bytes_upper_bound(self) -> usize {
        self.peak_cached_bytes
            .saturating_add(self.peak_deferred_bytes)
    }
}

#[cfg(feature = "cuda-runtime")]
impl From<j2k_cuda_runtime::CudaBufferPoolDiagnostics> for CudaDecodePoolSnapshot {
    fn from(value: j2k_cuda_runtime::CudaBufferPoolDiagnostics) -> Self {
        Self {
            cached_buffers: value.cached_buffers,
            cached_bytes: value.cached_bytes,
            deferred_buffers: value.deferred_buffers,
            deferred_bytes: value.deferred_bytes,
            reuse_holds: value.reuse_holds,
            peak_cached_bytes: value.peak_cached_bytes,
            peak_deferred_bytes: value.peak_deferred_bytes,
        }
    }
}

/// Diagnostics for the two private pools retained by one CUDA codec session.
#[cfg(feature = "cuda-runtime")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaDecodePoolDiagnostics {
    /// General single-image/component decode pool, when initialized.
    pub decode: Option<CudaDecodePoolSnapshot>,
    /// Best-fit dense batch decode pool, when initialized.
    pub batch_decode: Option<CudaDecodePoolSnapshot>,
}

/// Combined runtime work counters and retained decode-pool state for one session.
#[cfg(feature = "cuda-runtime")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaSessionDiagnostics {
    /// Context-level transfer, launch, allocation, and synchronization counters,
    /// when CUDA was initialized.
    pub runtime: Option<CudaContextDiagnostics>,
    /// Private decode-buffer pool state.
    pub pools: CudaDecodePoolDiagnostics,
}

#[cfg(feature = "cuda-runtime")]
impl CudaDecodePoolDiagnostics {
    /// Total device bytes currently retained by both initialized decode pools.
    #[must_use]
    pub fn retained_bytes(self) -> usize {
        self.decode
            .map_or(0, CudaDecodePoolSnapshot::retained_bytes)
            .saturating_add(
                self.batch_decode
                    .map_or(0, CudaDecodePoolSnapshot::retained_bytes),
            )
    }

    /// Conservative sum of the independent high-water bounds for both pools.
    #[must_use]
    pub fn peak_retained_bytes_upper_bound(self) -> usize {
        self.decode
            .map_or(0, CudaDecodePoolSnapshot::peak_retained_bytes_upper_bound)
            .saturating_add(
                self.batch_decode
                    .map_or(0, CudaDecodePoolSnapshot::peak_retained_bytes_upper_bound),
            )
    }
}

#[cfg(feature = "cuda-runtime")]
mod encode_resources;

#[cfg(feature = "cuda-runtime")]
use self::encode_resources::get_or_try_init_context_bound;

/// Mutable CUDA adapter session reused across submissions.
#[derive(Clone, Default)]
pub struct CudaSession {
    submissions: u64,
    #[cfg(feature = "cuda-runtime")]
    context: Option<CudaContext>,
    #[cfg(feature = "cuda-runtime")]
    htj2k_decode_tables: Option<CudaHtj2kDecodeTableResources>,
    #[cfg(feature = "cuda-runtime")]
    classic_decode_tables: Option<CudaClassicDecodeTableResources>,
    #[cfg(feature = "cuda-runtime")]
    htj2k_encode_resources: Option<Arc<CudaHtj2kEncodeResources>>,
    #[cfg(all(test, feature = "cuda-runtime"))]
    htj2k_encode_resource_uploads: usize,
    #[cfg(feature = "cuda-runtime")]
    decode_buffer_pool: Option<CudaBufferPool>,
    #[cfg(feature = "cuda-runtime")]
    decode_batch_buffer_pool: Option<CudaBufferPool>,
    #[cfg(feature = "cuda-runtime")]
    htj2k_decode_chunk_limits: Option<j2k_core::HtGpuJobChunkLimits>,
    #[cfg(all(test, feature = "cuda-runtime"))]
    last_htj2k_decode_chunk_count: usize,
}

impl CudaSession {
    /// Create a session bound to an existing CUDA context.
    #[cfg(feature = "cuda-runtime")]
    #[doc(hidden)]
    pub fn with_context(context: CudaContext) -> Self {
        Self {
            context: Some(context),
            ..Self::default()
        }
    }

    /// Number of submissions recorded by this session.
    pub fn submissions(&self) -> u64 {
        self.submissions
    }

    #[cfg(feature = "cuda-runtime")]
    /// True when a CUDA runtime context has been initialized.
    pub fn is_runtime_initialized(&self) -> bool {
        self.context.is_some()
    }

    /// Return the session-owned context for a framework-owned destination.
    ///
    /// The first call retains the requested device's primary context. Later
    /// calls reject a different device ordinal, so adapters cannot construct a
    /// destination view against a context other than the one this session owns.
    #[cfg(feature = "cuda-runtime")]
    #[doc(hidden)]
    pub fn context_for_device_interop(
        &mut self,
        device_ordinal: usize,
    ) -> Result<CudaContext, Error> {
        if let Some(context) = &self.context {
            if context.device_ordinal() != device_ordinal {
                return Err(Error::UnsupportedCudaRequest {
                    reason: "J2K CUDA interop device does not match the persistent session",
                });
            }
            return Ok(context.clone());
        }

        let context = CudaContext::retain_primary(device_ordinal).map_err(cuda_error)?;
        self.context = Some(context.clone());
        Ok(context)
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn cuda_context(&mut self) -> Result<CudaContext, Error> {
        if self.context.is_none() {
            self.context = Some(CudaContext::system_default().map_err(cuda_error)?);
        }
        self.context.clone().ok_or(Error::CudaUnavailable)
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn htj2k_decode_table_resources(
        &mut self,
    ) -> Result<CudaHtj2kDecodeTableResources, Error> {
        if let Some(tables) = &self.htj2k_decode_tables {
            return Ok(tables.clone());
        }

        let context = self.cuda_context()?;
        let tables = CudaHtj2kDecodeTables {
            vlc_table0: ht_vlc_table0(),
            vlc_table1: ht_vlc_table1(),
            uvlc_table0: ht_uvlc_table0(),
            uvlc_table1: ht_uvlc_table1(),
        };
        let resources = context
            .upload_htj2k_decode_table_resources(tables)
            .map_err(cuda_error)?;
        #[cfg(test)]
        HTJ2K_DECODE_TABLE_UPLOADS.fetch_add(1, Ordering::Relaxed);
        self.htj2k_decode_tables = Some(resources.clone());
        Ok(resources)
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn classic_decode_table_resources(
        &mut self,
    ) -> Result<CudaClassicDecodeTableResources, Error> {
        if let Some(tables) = &self.classic_decode_tables {
            return Ok(tables.clone());
        }
        let context = self.cuda_context()?;
        let tables = context
            .upload_classic_decode_table_resources()
            .map_err(cuda_error)?;
        #[cfg(test)]
        CLASSIC_DECODE_TABLE_UPLOADS.fetch_add(1, Ordering::Relaxed);
        self.classic_decode_tables = Some(tables.clone());
        Ok(tables)
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn htj2k_encode_resources(
        &mut self,
        requested_context: &CudaContext,
    ) -> Result<Arc<CudaHtj2kEncodeResources>, Error> {
        let (resources, initialized) = get_or_try_init_context_bound(
            &mut self.context,
            &mut self.htj2k_encode_resources,
            requested_context,
            CudaContext::is_same_context,
            || Error::UnsupportedCudaRequest {
                reason: "J2K CUDA encode tile belongs to a different context than the session",
            },
            |context| {
                context
                    .upload_htj2k_encode_resources(crate::encode::cuda_htj2k_encode_tables())
                    .map_err(cuda_error)
            },
        )?;
        #[cfg(test)]
        if initialized {
            self.htj2k_encode_resource_uploads =
                self.htj2k_encode_resource_uploads.saturating_add(1);
        }
        #[cfg(not(test))]
        let _ = initialized;
        Ok(resources)
    }

    #[cfg(all(test, feature = "cuda-runtime"))]
    pub(crate) fn htj2k_encode_resource_uploads_for_test(&self) -> usize {
        self.htj2k_encode_resource_uploads
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn decode_buffer_pool(&mut self) -> Result<CudaBufferPool, Error> {
        if let Some(pool) = &self.decode_buffer_pool {
            return Ok(pool.clone());
        }
        let context = self.cuda_context()?;
        let pool = context.buffer_pool();
        self.decode_buffer_pool = Some(pool.clone());
        Ok(pool)
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn decode_batch_buffer_pool(&mut self) -> Result<CudaBufferPool, Error> {
        if let Some(pool) = &self.decode_batch_buffer_pool {
            return Ok(pool.clone());
        }
        let context = self.cuda_context()?;
        let pool = context.best_fit_buffer_pool();
        self.decode_batch_buffer_pool = Some(pool.clone());
        Ok(pool)
    }

    /// Snapshot only the two device-buffer pools retained for decode work.
    ///
    /// Calling this method does not initialize a CUDA context or either pool.
    #[cfg(feature = "cuda-runtime")]
    pub fn decode_pool_diagnostics(&self) -> Result<CudaDecodePoolDiagnostics, Error> {
        Ok(CudaDecodePoolDiagnostics {
            decode: self
                .decode_buffer_pool
                .as_ref()
                .map(|pool| pool.diagnostics().map(CudaDecodePoolSnapshot::from))
                .transpose()
                .map_err(cuda_error)?,
            batch_decode: self
                .decode_batch_buffer_pool
                .as_ref()
                .map(|pool| pool.diagnostics().map(CudaDecodePoolSnapshot::from))
                .transpose()
                .map_err(cuda_error)?,
        })
    }

    /// Snapshot transfer/event counters and decode-pool retention without initializing CUDA.
    #[cfg(feature = "cuda-runtime")]
    pub fn diagnostics(&self) -> Result<CudaSessionDiagnostics, Error> {
        Ok(CudaSessionDiagnostics {
            runtime: self
                .context
                .as_ref()
                .map(CudaContext::diagnostics)
                .transpose()
                .map_err(cuda_error)?,
            pools: self.decode_pool_diagnostics()?,
        })
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn htj2k_decode_chunk_limits(&self) -> j2k_core::HtGpuJobChunkLimits {
        if let Some(limits) = self.htj2k_decode_chunk_limits {
            return limits;
        }
        let Some(max_jobs) = NonZeroUsize::new(65_536) else {
            return j2k_core::HtGpuJobChunkLimits::new(NonZeroUsize::MIN, 0, 0);
        };
        j2k_core::HtGpuJobChunkLimits::new(
            max_jobs,
            64 * 1024 * 1024,
            max_jobs
                .get()
                .saturating_mul(j2k_cuda_runtime::htj2k_cleanup_multi_descriptor_bytes()),
        )
    }

    #[cfg(all(test, feature = "cuda-runtime"))]
    pub(crate) fn set_htj2k_decode_chunk_limits_for_test(
        &mut self,
        limits: j2k_core::HtGpuJobChunkLimits,
    ) {
        self.htj2k_decode_chunk_limits = Some(limits);
    }

    #[cfg(all(test, feature = "cuda-runtime"))]
    pub(crate) fn record_htj2k_decode_chunk_count_for_test(&mut self, count: usize) {
        self.last_htj2k_decode_chunk_count = count;
    }

    #[cfg(all(test, feature = "cuda-runtime"))]
    pub(crate) fn last_htj2k_decode_chunk_count_for_test(&self) -> usize {
        self.last_htj2k_decode_chunk_count
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

#[cfg(all(test, feature = "cuda-runtime"))]
pub(crate) fn reset_htj2k_decode_table_uploads_for_test() {
    HTJ2K_DECODE_TABLE_UPLOADS.store(0, Ordering::Relaxed);
}

#[cfg(all(test, feature = "cuda-runtime"))]
pub(crate) fn htj2k_decode_table_uploads_for_test() -> usize {
    HTJ2K_DECODE_TABLE_UPLOADS.load(Ordering::Relaxed)
}

#[cfg(all(test, feature = "cuda-runtime"))]
pub(crate) fn reset_classic_decode_table_uploads_for_test() {
    CLASSIC_DECODE_TABLE_UPLOADS.store(0, Ordering::Relaxed);
}

#[cfg(all(test, feature = "cuda-runtime"))]
pub(crate) fn classic_decode_table_uploads_for_test() -> usize {
    CLASSIC_DECODE_TABLE_UPLOADS.load(Ordering::Relaxed)
}

impl std::fmt::Debug for CudaSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("CudaSession");
        debug.field("submissions", &self.submissions);
        #[cfg(feature = "cuda-runtime")]
        debug.field("runtime_initialized", &self.is_runtime_initialized());
        #[cfg(feature = "cuda-runtime")]
        debug.field(
            "htj2k_decode_tables_cached",
            &self.htj2k_decode_tables.is_some(),
        );
        #[cfg(feature = "cuda-runtime")]
        debug.field(
            "classic_decode_tables_cached",
            &self.classic_decode_tables.is_some(),
        );
        #[cfg(feature = "cuda-runtime")]
        debug.field(
            "htj2k_encode_resources_cached",
            &self.htj2k_encode_resources.is_some(),
        );
        #[cfg(feature = "cuda-runtime")]
        debug.field(
            "decode_buffer_pool_cached",
            &self.decode_buffer_pool.is_some(),
        );
        #[cfg(feature = "cuda-runtime")]
        debug.field(
            "decode_batch_buffer_pool_cached",
            &self.decode_batch_buffer_pool.is_some(),
        );
        debug.finish_non_exhaustive()
    }
}

#[cfg(all(test, feature = "cuda-runtime"))]
mod tests;
