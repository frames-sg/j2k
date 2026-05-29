// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "cuda-runtime")]
use signinum_cuda_runtime::CudaContext;

#[cfg(feature = "cuda-runtime")]
use crate::runtime::cuda_error;
#[cfg(feature = "cuda-runtime")]
use crate::Error;

/// Mutable CUDA adapter session reused across submissions.
#[derive(Clone, Default)]
pub struct CudaSession {
    submissions: u64,
    #[cfg(feature = "cuda-runtime")]
    context: Option<CudaContext>,
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

    pub(crate) fn record_submit(&mut self) {
        self.submissions = self.submissions.saturating_add(1);
    }

    #[cfg(feature = "cuda-runtime")]
    pub(crate) fn cuda_context(&mut self) -> Result<CudaContext, Error> {
        if self.context.is_none() {
            self.context = Some(CudaContext::system_default().map_err(cuda_error)?);
        }
        self.context.clone().ok_or(Error::CudaUnavailable)
    }
}

impl std::fmt::Debug for CudaSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("CudaSession");
        debug.field("submissions", &self.submissions);
        #[cfg(feature = "cuda-runtime")]
        debug.field("runtime_initialized", &self.is_runtime_initialized());
        debug.finish_non_exhaustive()
    }
}
