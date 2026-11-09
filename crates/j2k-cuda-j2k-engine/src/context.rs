// SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) use j2k_cuda_runtime::CudaContext;

use crate::{error::CudaError, J2kCudaEngine};
use std::ops::Range;

pub(crate) fn ensure_context_ownership(
    matches_context: impl IntoIterator<Item = bool>,
    mismatch_message: &'static str,
) -> Result<(), CudaError> {
    if matches_context.into_iter().all(|matches| matches) {
        Ok(())
    } else {
        Err(CudaError::InvalidArgument {
            message: mismatch_message.to_string(),
        })
    }
}

pub(crate) fn cuda_idwt_trace_enabled() -> bool {
    std::env::var_os("J2K_CUDA_IDWT_TRACE").is_some()
}

impl J2kCudaEngine<'_> {
    pub(crate) fn prepare_operation(&self) -> Result<(), CudaError> {
        self.context.prepare_operation()
    }

    pub(crate) fn upload(
        &self,
        bytes: &[u8],
    ) -> Result<j2k_cuda_runtime::CudaDeviceBuffer, CudaError> {
        self.context.upload(bytes)
    }

    pub(crate) fn upload_pinned(
        &self,
        bytes: &[u8],
    ) -> Result<j2k_cuda_runtime::CudaDeviceBuffer, CudaError> {
        self.context.upload_pinned(bytes)
    }

    pub(crate) fn upload_i32_pinned(
        &self,
        values: &[i32],
    ) -> Result<j2k_cuda_runtime::CudaDeviceBuffer, CudaError> {
        self.upload_pinned(crate::bytes::i32_slice_as_bytes(values))
    }

    pub(crate) fn allocate(
        &self,
        len: usize,
    ) -> Result<j2k_cuda_runtime::CudaDeviceBuffer, CudaError> {
        self.context.allocate(len)
    }

    pub(crate) fn memset_d32(
        &self,
        dst: &j2k_cuda_runtime::CudaDeviceBuffer,
        value: u32,
        words: usize,
    ) -> Result<(), CudaError> {
        self.context.memset_d32(dst, value, words)
    }

    pub(crate) fn memset_d32_async(
        &self,
        dst: &j2k_cuda_runtime::CudaDeviceBuffer,
        value: u32,
        words: usize,
    ) -> Result<(), CudaError> {
        self.context.memset_d32_async(dst, value, words)
    }

    pub(crate) fn synchronize(&self) -> Result<(), CudaError> {
        self.context.synchronize()
    }

    pub(crate) fn is_same_context(&self, other: &CudaContext) -> bool {
        self.context.is_same_context(other)
    }

    pub(crate) fn create_event(&self) -> Result<j2k_cuda_runtime::CudaEvent, CudaError> {
        self.context.create_event()
    }

    pub(crate) fn synchronize_then_error<T>(&self, error: CudaError) -> Result<T, CudaError> {
        self.context.synchronize_then_error(error)
    }

    /// Run `submit`, which queues kernels with [`Self::launch_kernel_async`],
    /// then wait once for the context, so callers may read, reuse or release
    /// every buffer those kernels touch. A failed submission also waits
    /// before returning, so buffers referenced by kernels it already queued
    /// stay allocated until those kernels finish.
    ///
    /// Kernels queued on the legacy default stream run in submission order,
    /// so a sequence of dependent launches needs only this one wait.
    pub(crate) fn complete_launches<T>(
        &self,
        submit: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        match submit() {
            Ok(output) => self.synchronize().map(|()| output),
            Err(error) => self.synchronize_then_error(error),
        }
    }

    pub(crate) fn copy_device_range_to_device_with_kernel(
        &self,
        src: &j2k_cuda_runtime::CudaDeviceBuffer,
        range: Range<usize>,
    ) -> Result<j2k_cuda_runtime::CudaDeviceBuffer, CudaError> {
        self.context
            .copy_device_range_to_device_with_kernel(src, range)
    }

    pub(crate) fn launch_kernel(
        &self,
        spec: j2k_cuda_runtime::CudaKernelSpec,
        geometry: j2k_cuda_runtime::CudaLaunchGeometry,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<(), CudaError> {
        // SAFETY: engine launch sites construct parameters for their matching
        // static kernel specification and retain all referenced allocations.
        unsafe { self.context.launch_compiled_kernel(spec, geometry, params) }
    }

    pub(crate) fn launch_kernel_async(
        &self,
        spec: j2k_cuda_runtime::CudaKernelSpec,
        geometry: j2k_cuda_runtime::CudaLaunchGeometry,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<(), CudaError> {
        // SAFETY: callers retain referenced allocations until an event or
        // queued owner establishes completion.
        unsafe {
            self.context
                .launch_compiled_kernel_async(spec, geometry, params)
        }
    }

    /// Queue `submit` on the default stream between two timing events,
    /// without waiting for it. The returned timing is read after a later
    /// blocking readback that follows the work in stream order. A failed
    /// submission waits before returning, as [`Self::complete_launches`] does.
    pub(crate) fn queue_timed_stage(
        &self,
        name: &str,
        submit: impl FnOnce() -> Result<(), CudaError>,
    ) -> Result<QueuedStageTiming, CudaError> {
        let queued = self.context.with_nvtx_range(name, || {
            let start = self.create_event()?;
            let end = self.create_event()?;
            start.record_default_stream()?;
            submit()?;
            end.record_default_stream()?;
            Ok(QueuedStageTiming { start, end })
        });
        match queued {
            Ok(timing) => Ok(timing),
            Err(error) => self.synchronize_then_error(error),
        }
    }

    pub(crate) fn time_default_stream_named_us<T>(
        &self,
        name: &str,
        work: impl FnMut() -> Result<T, CudaError>,
    ) -> Result<(T, u128), CudaError> {
        self.context.time_default_stream_named_us(name, work)
    }

    pub(crate) unsafe fn submit_default_stream_named<T>(
        &self,
        name: &str,
        work: impl FnMut() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        // SAFETY: the engine caller retains every resource reachable by work.
        unsafe { self.context.submit_default_stream_named(name, work) }
    }
}

/// Timing events around work queued by
/// [`J2kCudaEngine::queue_timed_stage`].
pub(crate) struct QueuedStageTiming {
    start: j2k_cuda_runtime::CudaEvent,
    end: j2k_cuda_runtime::CudaEvent,
}

impl QueuedStageTiming {
    /// Elapsed device time of the queued work, in whole microseconds. Call it
    /// only after a wait that follows the work on the default stream; before
    /// then the driver reports the end event as not ready, an error.
    pub(crate) fn elapsed_us(&self) -> Result<u128, CudaError> {
        j2k_cuda_runtime::elapsed_event_us_ceil(&self.start, &self.end)
    }
}
