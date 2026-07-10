// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::{
    error::CudaError,
    execution::{CudaExecutionStats, CudaLaunchMode},
    htj2k_decode::{
        htj2k_decode_needs_zero_fill, CudaHtj2kCodeBlockJob, CudaHtj2kDecodeOutput,
        CudaHtj2kDecodeStageTimings, CudaQueuedHtj2kCleanup,
    },
    memory::pooled_device_buffer,
};

mod band_transfer;
mod compact;
mod device;
mod inner;
mod kernel_cache;
mod kernel_dispatch;
mod pinned_host;
#[cfg(test)]
mod test_kernels;

pub use self::compact::{CudaHtj2kCompactEncodedCodeBlock, CudaHtj2kCompactEncodedCodeBlocks};
#[cfg(test)]
pub(crate) use self::test_kernels::{CudaKernelModule, CudaKernelName};
pub(crate) use self::{
    band_transfer::cuda_idwt_trace_enabled,
    compact::HTJ2K_UVLC_ENCODE_TABLE_BYTES,
    inner::ContextInner,
    kernel_cache::{CompiledKernel, CompiledKernelKey},
    pinned_host::PinnedUploadStaging,
};

/// CUDA driver context shared by J2K CUDA adapter crates.
#[derive(Clone)]
pub struct CudaContext {
    pub(crate) inner: Arc<ContextInner>,
}

impl CudaContext {
    /// Dequantize HTJ2K cleanup outputs using the metadata buffer already held
    /// live by a queued cleanup launch.
    #[doc(hidden)]
    pub fn j2k_dequantize_queued_htj2k_cleanup_with_pool(
        &self,
        cleanup: &CudaQueuedHtj2kCleanup,
    ) -> Result<CudaExecutionStats, CudaError> {
        self.inner.set_current()?;
        if cleanup.status_count == 0 {
            return Ok(CudaExecutionStats::default());
        }
        let Some(jobs_buffer) = cleanup.resources.first() else {
            return Err(CudaError::InvalidArgument {
                message: "queued HTJ2K cleanup has no metadata buffer".to_string(),
            });
        };
        self.launch_j2k_dequantize_htj2k_cleanup_jobs_multi(
            pooled_device_buffer(jobs_buffer)?,
            cleanup.status_count,
            CudaLaunchMode::Sync,
        )?;
        Ok(CudaExecutionStats {
            kernel_dispatches: 1,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 1,
            hardware_decode: false,
        })
    }

    pub(crate) fn decode_empty_htj2k_codeblocks(
        &self,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
    ) -> Result<CudaHtj2kDecodeOutput, CudaError> {
        self.inner.set_current()?;
        let output_bytes = output_words
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaError::LengthTooLarge { len: output_words })?;
        let coefficients = self.allocate(output_bytes)?;
        if htj2k_decode_needs_zero_fill(jobs, output_words)? {
            self.memset_d32(&coefficients, 0, output_words)?;
        }
        Ok(CudaHtj2kDecodeOutput {
            coefficients,
            execution: CudaExecutionStats::default(),
            statuses: Vec::new(),
            stage_timings: CudaHtj2kDecodeStageTimings::default(),
        })
    }
}

impl std::fmt::Debug for CudaContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaContext").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod structure_tests {
    const CONTEXT: &str = include_str!("context.rs");
    const MODULES: &[(&str, &str, usize)] = &[
        (
            "context/band_transfer.rs",
            include_str!("context/band_transfer.rs"),
            75,
        ),
        (
            "context/compact.rs",
            include_str!("context/compact.rs"),
            150,
        ),
        ("context/device.rs", include_str!("context/device.rs"), 80),
        ("context/inner.rs", include_str!("context/inner.rs"), 100),
        (
            "context/kernel_cache.rs",
            include_str!("context/kernel_cache.rs"),
            250,
        ),
        (
            "context/kernel_dispatch.rs",
            include_str!("context/kernel_dispatch.rs"),
            425,
        ),
        (
            "context/pinned_host.rs",
            include_str!("context/pinned_host.rs"),
            75,
        ),
        (
            "context/test_kernels.rs",
            include_str!("context/test_kernels.rs"),
            180,
        ),
    ];

    #[test]
    fn cuda_context_uses_focused_real_modules() {
        let include_macro = ["include", "!("].concat();
        let wildcard_import = ["use super::", "*"].concat();
        assert!(
            CONTEXT.lines().count() < 200,
            "context.rs must remain a focused module shell"
        );
        for module in [
            "mod band_transfer;",
            "mod compact;",
            "mod device;",
            "mod inner;",
            "mod kernel_cache;",
            "mod kernel_dispatch;",
            "mod pinned_host;",
        ] {
            assert!(CONTEXT.contains(module), "context.rs must contain {module}");
        }
        assert!(!CONTEXT.contains(&include_macro));

        for (path, source, max_lines) in MODULES {
            assert!(
                source.lines().count() < *max_lines,
                "{path} must stay below its focused-module line-count ratchet"
            );
            assert!(
                !source.contains(&include_macro),
                "{path} must be a real module"
            );
            assert!(
                !source.contains(&wildcard_import),
                "{path} must use explicit imports"
            );
        }
    }
}
