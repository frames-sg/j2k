// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use crate::driver::CuStream;
use crate::{context::CudaContext, driver::CuEvent, error::CudaError};

/// CUDA stream RAII handle.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct CudaStream {
    pub(crate) context: CudaContext,
    pub(crate) stream: CuStream,
}

#[cfg(test)]
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
#[cfg(test)]
unsafe impl Send for CudaStream {}

/// CUDA event RAII handle for timing and synchronization.
#[derive(Debug)]
pub(crate) struct CudaEvent {
    pub(crate) context: CudaContext,
    pub(crate) event: CuEvent,
}

impl CudaEvent {
    /// Record this event on a CUDA stream.
    #[cfg(test)]
    pub(crate) fn record(&self, stream: &CudaStream) -> Result<(), CudaError> {
        if !self.context.is_same_context(&stream.context) {
            return Err(CudaError::InvalidArgument {
                message: "CUDA event and stream must belong to the same context".to_string(),
            });
        }
        self.context.inner.with_current_resource_operation(|| {
            // SAFETY: event and stream are live handles from this context, and
            // the context lifecycle gate is held.
            self.context.inner.driver.check("cuEventRecord", unsafe {
                (self.context.inner.driver.cu_event_record)(self.event, stream.stream)
            })
        })
    }

    pub(crate) fn record_default_stream(&self) -> Result<(), CudaError> {
        self.context.inner.with_current_resource_operation(|| {
            // SAFETY: a null stream is this current context's default stream,
            // and the context lifecycle gate is held.
            self.context.inner.driver.check("cuEventRecord", unsafe {
                (self.context.inner.driver.cu_event_record)(self.event, std::ptr::null_mut())
            })
        })
    }

    /// Wait for this event to complete.
    pub(crate) fn synchronize(&self) -> Result<(), CudaError> {
        self.context.inner.with_current_resource_operation(|| {
            // SAFETY: event is a live CUDA event owned by this context, and the
            // context lifecycle gate is held.
            self.context
                .inner
                .driver
                .check("cuEventSynchronize", unsafe {
                    (self.context.inner.driver.cu_event_synchronize)(self.event)
                })
        })
    }

    /// Elapsed time in microseconds from `start` to `end`.
    pub(crate) fn elapsed_time_us(start: &Self, end: &Self) -> Result<f32, CudaError> {
        if !start.context.is_same_context(&end.context) {
            return Err(CudaError::InvalidArgument {
                message: "CUDA timing events must belong to the same context".to_string(),
            });
        }
        let mut millis = 0.0f32;
        end.context.inner.with_current_resource_operation(|| {
            // SAFETY: start and end are live recorded events from this
            // context, and the context lifecycle gate is held.
            let status = unsafe {
                (end.context.inner.driver.cu_event_elapsed_time)(
                    &raw mut millis,
                    start.event,
                    end.event,
                )
            };
            end.context.inner.driver.check("cuEventElapsedTime", status)
        })?;
        Ok(millis * 1000.0)
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if !self.event.is_null() {
            let destroy_result = self.context.inner.with_current_stateful_operation(|| {
                // SAFETY: event was created by this context and the context
                // lifecycle gate is held during destruction.
                self.context
                    .inner
                    .driver
                    .check("cuEventDestroy_v2", unsafe {
                        (self.context.inner.driver.cu_event_destroy)(self.event)
                    })
            });
            if destroy_result.is_err() {
                std::mem::forget(self.context.clone());
            }
        }
    }
}

// SAFETY: CUDA event handles are driver-owned resources. The Rust handle owns
// destruction and does not expose mutable aliasing of Rust memory.
unsafe impl Send for CudaEvent {}
