// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{build_flags::CUDA_ERROR_NOT_READY, driver::CuStream};
use crate::{context::CudaContext, driver::CuEvent, error::CudaError};

#[cfg(test)]
mod stream;
#[cfg(test)]
pub(crate) use stream::CudaStream;

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

    pub(crate) fn record_raw_stream(&self, stream: CuStream) -> Result<(), CudaError> {
        self.context.inner.with_current_resource_operation(|| {
            // SAFETY: the caller's guarded stream handle has been bound to
            // this retained primary context for the duration of interop.
            self.context.inner.driver.check("cuEventRecord", unsafe {
                (self.context.inner.driver.cu_event_record)(self.event, stream)
            })
        })
    }

    pub(crate) fn wait_on_default_stream(&self) -> Result<(), CudaError> {
        self.wait_on_raw_stream(std::ptr::null_mut())
    }

    pub(crate) fn wait_on_raw_stream(&self, stream: CuStream) -> Result<(), CudaError> {
        self.context.inner.with_current_resource_operation(|| {
            // SAFETY: the event belongs to the current retained primary
            // context; CUDA validates the guarded stream handle.
            self.context
                .inner
                .driver
                .check("cuStreamWaitEvent", unsafe {
                    (self.context.inner.driver.cu_stream_wait_event)(stream, self.event, 0)
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
        })?;
        self.context.record_event_host_synchronization();
        Ok(())
    }

    /// Query whether this event has completed without waiting on the host.
    pub(crate) fn is_complete(&self) -> Result<bool, CudaError> {
        self.context.inner.with_current_resource_operation(|| {
            // SAFETY: event is a live CUDA event owned by this context, and
            // the context lifecycle gate is held for the query.
            let status = unsafe { (self.context.inner.driver.cu_event_query)(self.event) };
            if status == CUDA_ERROR_NOT_READY {
                return Ok(false);
            }
            self.context
                .inner
                .driver
                .check("cuEventQuery", status)
                .map(|()| true)
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
            // CUDA stream waits capture the event's most recently recorded
            // generation when cuStreamWaitEvent is called. Re-recording this
            // handle later does not alter an already-enqueued wait, so the
            // handle can return to this context's cache without a host wait.
            let recycle_result = self
                .context
                .inner
                .event_pool
                .lock()
                .map_err(|_| ())
                .and_then(|mut events| events.recycle(self.event));
            if recycle_result.is_ok() {
                self.event = std::ptr::null_mut();
                return;
            }
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
