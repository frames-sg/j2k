// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "cuda-runtime")]
use j2k_cuda_runtime::{
    CudaBufferPool, CudaContext, CudaHtj2kDecodeTableResources, CudaHtj2kDecodeTables,
};
#[cfg(feature = "cuda-runtime")]
use j2k_native::{ht_uvlc_table0, ht_uvlc_table1, ht_vlc_table0, ht_vlc_table1};
#[cfg(all(test, feature = "cuda-runtime"))]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "cuda-runtime")]
use crate::runtime::cuda_error;
#[cfg(feature = "cuda-runtime")]
use crate::Error;

#[cfg(all(test, feature = "cuda-runtime"))]
static HTJ2K_DECODE_TABLE_UPLOADS: AtomicUsize = AtomicUsize::new(0);

/// Mutable CUDA adapter session reused across submissions.
#[derive(Clone, Default)]
pub struct CudaSession {
    submissions: u64,
    #[cfg(feature = "cuda-runtime")]
    context: Option<CudaContext>,
    #[cfg(feature = "cuda-runtime")]
    htj2k_decode_tables: Option<CudaHtj2kDecodeTableResources>,
    #[cfg(feature = "cuda-runtime")]
    decode_buffer_pool: Option<CudaBufferPool>,
    #[cfg(feature = "cuda-runtime")]
    decode_batch_buffer_pool: Option<CudaBufferPool>,
}

impl CudaSession {
    /// Number of submissions recorded by this session.
    pub fn submissions(&self) -> u64 {
        self.submissions
    }

    #[cfg(feature = "cuda-runtime")]
    /// True when a CUDA runtime context has been initialized.
    pub fn is_runtime_initialized(&self) -> bool {
        self.context.is_some()
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
}

impl j2k_core::DeviceSubmitSession for CudaSession {
    fn record_submit(&mut self) {
        self.submissions = self.submissions.saturating_add(1);
    }
}

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
mod tests {
    use super::CudaSession;
    use crate::Error;

    fn cuda_required() -> bool {
        std::env::var_os("J2K_REQUIRE_CUDA_RUNTIME").is_some()
    }

    #[test]
    fn htj2k_decode_tables_are_uploaded_once_per_session() {
        crate::session::reset_htj2k_decode_table_uploads_for_test();
        let mut session = CudaSession::default();

        let first = session.htj2k_decode_table_resources();
        if matches!(
            first,
            Err(Error::CudaUnavailable | Error::CudaRuntime { .. })
        ) && !cuda_required()
        {
            return;
        }
        first.expect("first HTJ2K decode table upload");
        session
            .htj2k_decode_table_resources()
            .expect("cached HTJ2K decode tables");

        assert_eq!(crate::session::htj2k_decode_table_uploads_for_test(), 1);
    }

    #[test]
    fn cuda_session_reuses_one_decode_buffer_pool_when_required() {
        let mut session = CudaSession::default();

        let first = session.decode_buffer_pool();
        if matches!(
            first,
            Err(Error::CudaUnavailable | Error::CudaRuntime { .. })
        ) && !cuda_required()
        {
            return;
        }
        let first = first.expect("first decode buffer pool");
        let second = session
            .decode_buffer_pool()
            .expect("cached decode buffer pool");
        {
            let buffer = first.take(16).expect("pooled decode buffer");
            assert_eq!(buffer.byte_len(), 16);
        }

        assert!(second.cached_count().expect("shared pool cached count") >= 1);
    }
}
