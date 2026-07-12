// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{CudaJpegChunkedEntropyPlan, CudaJpegChunkedEntropyReport};
use crate::{
    context::CudaContext, error::CudaError, execution::CudaExecutionStats,
    memory::CudaPinnedUploadOperationGuard,
};

impl CudaContext {
    #[doc(hidden)]
    /// Run experimental 4:2:0 JPEG entropy self-sync diagnostics.
    pub fn diagnose_jpeg_420_entropy_self_sync(
        &self,
        plan: &CudaJpegChunkedEntropyPlan<'_>,
    ) -> Result<CudaJpegChunkedEntropyReport, CudaError> {
        self.diagnose_jpeg_420_entropy_self_sync_with_external_live(plan, 0)
    }

    #[doc(hidden)]
    /// Diagnose entropy while charging host owners retained by the adapter.
    pub fn diagnose_jpeg_420_entropy_self_sync_with_external_live(
        &self,
        plan: &CudaJpegChunkedEntropyPlan<'_>,
        external_live_bytes: usize,
    ) -> Result<CudaJpegChunkedEntropyReport, CudaError> {
        let pinned_upload = self.begin_pinned_upload_operation()?;
        self.diagnose_jpeg_420_entropy_self_sync_with_pinned_upload_operation(
            plan,
            external_live_bytes,
            &pinned_upload,
        )
    }

    #[doc(hidden)]
    /// Diagnose entropy inside an adapter-held pinned-upload transaction.
    ///
    /// `external_live_bytes` must exclude this context's page-locked staging;
    /// the runtime charges the exact post-checkout pool plus best-fit staging
    /// aggregate inside this transaction.
    pub fn diagnose_jpeg_420_entropy_self_sync_with_pinned_upload_operation(
        &self,
        plan: &CudaJpegChunkedEntropyPlan<'_>,
        external_live_bytes: usize,
        pinned_upload: &CudaPinnedUploadOperationGuard<'_>,
    ) -> Result<CudaJpegChunkedEntropyReport, CudaError> {
        pinned_upload.ensure_for_context(self)?;
        plan.config.validate()?;
        let subsequences = plan
            .config
            .subsequence_count_for_entropy_bytes(plan.entropy_bytes.len())?;
        if subsequences == 0 {
            return Ok(CudaJpegChunkedEntropyReport {
                config: plan.config,
                entropy_bytes: plan.entropy_bytes.len(),
                states: Vec::new(),
                overflows: Vec::new(),
                execution: CudaExecutionStats {
                    kernel_dispatches: 0,
                    copy_kernel_dispatches: 0,
                    decode_kernel_dispatches: 0,
                    hardware_decode: false,
                },
            });
        }

        #[cfg(not(feature = "cuda-oxide-jpeg-decode"))]
        {
            let _ = (subsequences, external_live_bytes, pinned_upload);
            Err(CudaError::InvalidArgument {
                message: "CUDA JPEG entropy diagnostic PTX was not built".to_string(),
            })
        }

        #[cfg(feature = "cuda-oxide-jpeg-decode")]
        {
            self.diagnose_jpeg_420_entropy_self_sync_nonempty(
                plan,
                subsequences,
                external_live_bytes,
                pinned_upload,
            )
        }
    }
}
