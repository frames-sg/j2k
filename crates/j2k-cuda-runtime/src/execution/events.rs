// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    build_flags::cuda_stage_timings_disabled,
    context::CudaContext,
    driver::CudaNvtxRange,
    error::{select_resource_release_error, CudaError},
};

mod handles;
mod interop;

pub(crate) use handles::CudaEvent;
#[cfg(test)]
use handles::CudaStream;

impl CudaContext {
    fn complete_default_stream_work<T>(
        &self,
        mut work: impl FnMut() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        self.inner.set_current()?;
        let output = match work() {
            Ok(output) => output,
            Err(error) => return self.synchronize_then_error(error),
        };
        self.synchronize()?;
        Ok(output)
    }

    /// Create a CUDA stream owned by this context.
    #[cfg(test)]
    pub(crate) fn create_stream(&self) -> Result<CudaStream, CudaError> {
        let mut stream = std::ptr::null_mut();
        self.inner.with_current_stateful_operation(|| {
            // SAFETY: CUDA writes a new stream handle while the context
            // lifecycle gate is held. CudaStream destroys the handle.
            self.inner.driver.check("cuStreamCreate", unsafe {
                (self.inner.driver.cu_stream_create)(&raw mut stream, 0)
            })?;
            crate::context::validate_resource_handle(
                stream,
                "CUDA returned a null stream after successful creation",
            )
        })?;
        Ok(CudaStream {
            context: self.clone(),
            stream,
        })
    }

    /// Create a CUDA timing event owned by this context.
    pub(crate) fn create_event(&self) -> Result<CudaEvent, CudaError> {
        if let Some(event) = self
            .inner
            .event_pool
            .lock()
            .map_err(|error| CudaError::StatePoisoned {
                message: error.to_string(),
            })?
            .take()
        {
            return Ok(CudaEvent {
                context: self.clone(),
                event,
            });
        }
        let mut event = std::ptr::null_mut();
        self.inner.with_current_stateful_operation(|| {
            // SAFETY: CUDA writes a new event handle while the context
            // lifecycle gate is held. CudaEvent destroys the handle.
            self.inner.driver.check("cuEventCreate", unsafe {
                (self.inner.driver.cu_event_create)(&raw mut event, 0)
            })?;
            crate::context::validate_resource_handle(
                event,
                "CUDA returned a null event after successful creation",
            )
        })?;
        match self.inner.event_pool.lock() {
            Ok(mut events) => events.record_driver_allocation(),
            Err(error) => {
                let primary = CudaError::StatePoisoned {
                    message: error.to_string(),
                };
                let release = self.inner.with_current_stateful_operation(|| {
                    // SAFETY: event was just created by this context and has
                    // not escaped. A poisoned diagnostics/cache lock must not
                    // leak the raw driver allocation.
                    self.inner.driver.check("cuEventDestroy_v2", unsafe {
                        (self.inner.driver.cu_event_destroy)(event)
                    })
                });
                return match release {
                    Ok(()) => Err(primary),
                    Err(release) => Err(select_resource_release_error(primary, release)),
                };
            }
        }
        Ok(CudaEvent {
            context: self.clone(),
            event,
        })
    }

    /// Time work submitted to the default CUDA stream and return elapsed microseconds.
    ///
    /// `FnMut` keeps the closure environment owned by this frame until error
    /// synchronization completes, so captured CUDA resources cannot be
    /// dropped while submitted work may still reference them. Resources
    /// created inside `work` must still be returned or protected by the
    /// asynchronous operation's own completion guard.
    pub(crate) fn time_default_stream_us<T>(
        &self,
        mut work: impl FnMut() -> Result<T, CudaError>,
    ) -> Result<(T, u128), CudaError> {
        if cuda_stage_timings_disabled() {
            // Disabling event collection must not weaken the helper's
            // completion contract. Safe callers may return or recycle CUDA
            // resources immediately after this method succeeds.
            return self
                .complete_default_stream_work(&mut work)
                .map(|output| (output, 0));
        }
        self.inner.set_current()?;
        let start = self.create_event()?;
        let end = self.create_event()?;
        start.record_default_stream()?;
        let output = match work() {
            Ok(output) => output,
            Err(error) => {
                // Timed closures may submit asynchronous default-stream work.
                // On a later host-side error, wait before dropping any device
                // buffers captured by the closure.
                return self.synchronize_then_error(error);
            }
        };
        // The gated event operations already either establish context-wide
        // completion or quarantine the context before returning an error.
        end.record_default_stream()?;
        end.synchronize()?;
        Ok((output, elapsed_event_us_ceil(&start, &end)?))
    }

    #[doc(hidden)]
    /// Run work inside an optional NVTX profiling range.
    ///
    /// The range is a no-op unless the crate is built with `cuda-profiling`
    /// and an NVTX runtime library can be loaded dynamically.
    pub fn with_nvtx_range<T>(
        &self,
        name: &str,
        work: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        let _range = CudaNvtxRange::push(name);
        work()
    }

    #[doc(hidden)]
    /// Time work submitted to the default CUDA stream inside an optional NVTX range.
    ///
    /// The NVTX range is a no-op unless the crate is built with
    /// `cuda-profiling` and an NVTX runtime library can be loaded dynamically.
    pub fn time_default_stream_named_us<T>(
        &self,
        name: &str,
        mut work: impl FnMut() -> Result<T, CudaError>,
    ) -> Result<(T, u128), CudaError> {
        self.with_nvtx_range(name, || self.time_default_stream_us(&mut work))
    }

    #[doc(hidden)]
    /// Optionally time work submitted to the default CUDA stream inside an NVTX range.
    pub fn time_default_stream_named_us_if<T>(
        &self,
        collect_stage_timings: bool,
        name: &str,
        mut work: impl FnMut() -> Result<T, CudaError>,
    ) -> Result<(T, u128), CudaError> {
        if collect_stage_timings {
            self.time_default_stream_named_us(name, &mut work)
        } else {
            self.with_nvtx_range(name, || self.complete_default_stream_work(&mut work))
                .map(|output| (output, 0))
        }
    }

    #[doc(hidden)]
    /// Submit default-stream work without establishing completion on success.
    ///
    /// Errors are synchronized before return so captured resources can be
    /// released safely. A successful return proves only submission.
    ///
    /// # Safety
    ///
    /// The caller must retain every resource reachable by submitted work and
    /// establish context completion before any resource is mutated, reused,
    /// or released. Prefer a typed `#[must_use]` queued guard.
    pub unsafe fn submit_default_stream_named<T>(
        &self,
        name: &str,
        mut work: impl FnMut() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        self.inner.set_current()?;
        self.with_nvtx_range(name, || match work() {
            Ok(output) => Ok(output),
            Err(error) => self.synchronize_then_error(error),
        })
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "rounded normalized samples are clamped to the complete u8 output range"
)]
pub(crate) fn elapsed_event_us_ceil(start: &CudaEvent, end: &CudaEvent) -> Result<u128, CudaError> {
    let elapsed = CudaEvent::elapsed_time_us(start, end)?;
    if elapsed <= 0.0 {
        return Ok(1);
    }
    Ok(elapsed.ceil() as u128)
}
