// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{context::CudaContext, driver::CuStream};

/// CUDA stream RAII handle.
#[derive(Debug)]
pub(crate) struct CudaStream {
    pub(crate) context: CudaContext,
    pub(crate) stream: CuStream,
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
