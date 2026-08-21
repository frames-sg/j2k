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
