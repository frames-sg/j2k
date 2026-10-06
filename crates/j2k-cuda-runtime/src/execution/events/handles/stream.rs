// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{context::CudaContext, driver::CuStream};

/// CUDA stream RAII handle.
#[derive(Debug)]
pub(crate) struct CudaStream {
    pub(crate) context: CudaContext,
    pub(crate) stream: CuStream,
}

/// `CU_STREAM_DEFAULT`: the stream synchronizes with the legacy default stream.
#[cfg(test)]
pub(crate) const CU_STREAM_DEFAULT: u32 = 0;
/// `CU_STREAM_NON_BLOCKING`: work can overlap the legacy default stream.
const CU_STREAM_NON_BLOCKING: u32 = 1;

impl CudaStream {
    /// Stream whose transfers can overlap kernels already queued on the legacy
    /// default stream.
    pub(crate) fn nonblocking(context: &CudaContext) -> Result<Self, crate::CudaError> {
        Self::with_flags(context, CU_STREAM_NON_BLOCKING)
    }

    pub(crate) fn with_flags(context: &CudaContext, flags: u32) -> Result<Self, crate::CudaError> {
        let mut stream = std::ptr::null_mut();
        context.inner.with_current_stateful_operation(|| {
            // SAFETY: CUDA writes a new stream handle while the context
            // lifecycle gate is held. CudaStream destroys the handle.
            context.inner.driver.check("cuStreamCreate", unsafe {
                (context.inner.driver.cu_stream_create)(&raw mut stream, flags)
            })?;
            crate::context::validate_resource_handle(
                stream,
                "CUDA returned a null stream after successful creation",
            )
        })?;
        Ok(Self {
            context: context.clone(),
            stream,
        })
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            let destroy_result = self.context.inner.with_current_stateful_operation(|| {
                // SAFETY: stream was created by this context and the context
                // lifecycle gate is held during destruction.
                self.context
                    .inner
                    .driver
                    .check("cuStreamDestroy_v2", unsafe {
                        (self.context.inner.driver.cu_stream_destroy)(self.stream)
                    })
            });
            if destroy_result.is_err() {
                std::mem::forget(self.context.clone());
            }
        }
    }
}

// SAFETY: CUDA stream handles are driver-owned resources. The Rust handle owns
// destruction and does not expose mutable aliasing of Rust memory.
unsafe impl Send for CudaStream {}
