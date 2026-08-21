// SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) mod completion;
mod events;
mod memory_ops;
mod queued;

pub(crate) use completion::select_uncertain_completion_error;
pub use completion::CudaSynchronizationOutcome;
#[doc(hidden)]
pub use events::elapsed_event_us_ceil;
pub use events::CudaEvent;
pub use queued::{
    CudaExecutionStats, CudaKernelBatchOutput, CudaKernelContiguousBatchOutput, CudaKernelOutput,
    CudaPooledKernelOutput, CudaQueuedExecution,
};

#[cfg(test)]
use crate::context::{CudaKernelModule, CudaKernelName};
use crate::{
    context::CudaContext,
    driver::{CuDevicePtr, CuFunction},
    error::CudaError,
    kernels::{self, copy_u8_launch_geometry},
    memory::{CudaBufferPool, CudaDeviceBuffer, CudaPooledDeviceBuffer},
};
use std::{ffi::c_void, ops::Range};

/// Marker for values that can be passed by address to a CUDA kernel launch.
///
/// # Safety
///
/// Implementors must have a stable, CUDA-compatible by-value representation for
/// the duration of `cuLaunchKernel`.
pub unsafe trait CudaKernelParam {}

// SAFETY: `CuDevicePtr` is the raw CUDA device pointer value expected by kernels.
unsafe impl CudaKernelParam for CuDevicePtr {}
// SAFETY: CUDA kernels receive these scalar ABI types by value via parameter pointers.
unsafe impl CudaKernelParam for u32 {}
// SAFETY: CUDA kernels receive these scalar ABI types by value via parameter pointers.
unsafe impl CudaKernelParam for i32 {}
// SAFETY: CUDA kernels receive these scalar ABI types by value via parameter pointers.
unsafe impl CudaKernelParam for f32 {}

/// Return the address CUDA expects for one by-value kernel argument.
///
/// The returned pointer is valid only while `value` remains alive and is not
/// moved. [`CudaContext::launch_compiled_kernel`] consumes it synchronously.
pub fn cuda_kernel_param<T>(value: &mut T) -> *mut c_void
where
    T: CudaKernelParam,
{
    std::ptr::from_mut(value).cast::<c_void>()
}

impl CudaContext {
    /// Load/cache a validated static kernel and launch it synchronously.
    ///
    /// # Safety
    ///
    /// Every pointer in `params` must address a live value whose layout matches
    /// the corresponding CUDA kernel parameter. Device pointers embedded in
    /// those values must belong to this context. `spec` must describe matching
    /// PTX and entry-point ABI. The values must remain alive until this method
    /// returns.
    pub unsafe fn launch_compiled_kernel(
        &self,
        spec: crate::CudaKernelSpec,
        geometry: crate::CudaLaunchGeometry,
        params: &mut [*mut c_void],
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function_from_spec(spec)?;
        self.launch_kernel(function, geometry, params)
    }

    /// Load/cache a validated static kernel and submit it asynchronously.
    ///
    /// # Safety
    ///
    /// The parameter ABI and device-pointer requirements match the synchronous
    /// launch primitive. Callers must additionally retain every referenced
    /// allocation and establish context completion before reuse or release.
    #[doc(hidden)]
    pub unsafe fn launch_compiled_kernel_async(
        &self,
        spec: crate::CudaKernelSpec,
        geometry: crate::CudaLaunchGeometry,
        params: &mut [*mut c_void],
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function_from_spec(spec)?;
        self.launch_kernel_async(function, geometry, params)
    }

    /// Launch a compiled kernel asynchronously while retaining pooled resources.
    ///
    /// The returned guard establishes context completion before the resources
    /// can be reused. Call
    /// [`CudaQueuedExecution::finish_with_resources`](crate::CudaQueuedExecution::finish_with_resources)
    /// when an engine needs a deferred device-to-host readback after the kernel.
    ///
    /// # Safety
    ///
    /// Every pointer in `params` must address a live value whose layout matches
    /// the corresponding kernel parameter. Device pointers must belong to this
    /// context, and every allocation used by submitted work must outlive the
    /// returned guard. `resources` transfers ownership only for the listed
    /// pooled buffers; callers must separately retain all other owners.
    pub unsafe fn launch_compiled_kernel_queued(
        &self,
        spec: crate::CudaKernelSpec,
        geometry: crate::CudaLaunchGeometry,
        params: &mut [*mut c_void],
        pool: &CudaBufferPool,
        resources: Vec<CudaPooledDeviceBuffer>,
        execution: CudaExecutionStats,
    ) -> Result<CudaQueuedExecution, CudaError> {
        if !pool.is_owned_by(self) {
            return Err(CudaError::InvalidArgument {
                message: "queued CUDA resource pool must belong to the launch context".to_string(),
            });
        }
        let function = self.inner.kernel_function_from_spec(spec)?;
        let pool_reuse_guard = pool.defer_reuse()?;
        if let Err(error) = self.launch_kernel_async(function, geometry, params) {
            return pool_reuse_guard.synchronize_then_error(error);
        }
        Ok(CudaQueuedExecution {
            resources,
            execution,
            pool_reuse_guard: Some(pool_reuse_guard),
        })
    }

    #[doc(hidden)]
    /// Copy host bytes through a CUDA copy kernel and return device output.
    pub fn copy_with_kernel(&self, bytes: &[u8]) -> Result<CudaKernelOutput, CudaError> {
        let staging = self.upload(bytes)?;
        let output = self.copy_device_to_device_with_kernel(&staging)?;
        let copy_dispatches = usize::from(!bytes.is_empty());
        Ok(CudaKernelOutput {
            buffer: output,
            execution: CudaExecutionStats {
                kernel_dispatches: copy_dispatches,
                copy_kernel_dispatches: copy_dispatches,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
        })
    }

    #[cfg(all(test, feature = "cuda-oxide-copy-u8", j2k_cuda_oxide_copy_u8_built))]
    pub(crate) fn copy_with_cuda_oxide_kernel(
        &self,
        bytes: &[u8],
    ) -> Result<CudaKernelOutput, CudaError> {
        let staging = self.upload(bytes)?;
        let output = self.copy_device_to_device_with_cuda_oxide_kernel(&staging)?;
        let copy_dispatches = usize::from(!bytes.is_empty());
        Ok(CudaKernelOutput {
            buffer: output,
            execution: CudaExecutionStats {
                kernel_dispatches: copy_dispatches,
                copy_kernel_dispatches: copy_dispatches,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
        })
    }

    pub(crate) fn launch_kernel(
        &self,
        function: CuFunction,
        geometry: kernels::CudaLaunchGeometry,
        params: &mut [*mut c_void],
    ) -> Result<(), CudaError> {
        self.launch_kernel_async(function, geometry, params)?;
        // SAFETY: `function` was loaded from a live module in this context, and
        // the kernel was launched on the current context; synchronize waits for
        // completion before callers inspect outputs.
        self.synchronize()
    }

    pub(crate) fn launch_kernel_async(
        &self,
        function: CuFunction,
        geometry: kernels::CudaLaunchGeometry,
        params: &mut [*mut c_void],
    ) -> Result<(), CudaError> {
        if !geometry.is_valid() {
            return Err(CudaError::InvalidArgument {
                message: format!(
                    "CUDA launch geometry exceeds static limits: grid {:?}, block {:?}",
                    geometry.grid(),
                    geometry.block()
                ),
            });
        }
        let (grid_x, grid_y, grid_z) = geometry.grid();
        let (block_x, block_y, block_z) = geometry.block();
        self.inner.with_current_resource_operation(|| {
            // SAFETY: `function` was loaded from a live module in this context,
            // and `params` contains kernel argument pointers valid for the
            // launch call. The context lifetime gate is held and this context
            // is current for the calling thread.
            let launch_status = unsafe {
                (self.inner.driver.cu_launch_kernel)(
                    function,
                    grid_x,
                    grid_y,
                    grid_z,
                    block_x,
                    block_y,
                    block_z,
                    0,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            };
            self.inner.driver.check("cuLaunchKernel", launch_status)
        })?;
        self.record_kernel_launch();
        Ok(())
    }

    pub(crate) fn copy_device_to_device_with_kernel(
        &self,
        src: &CudaDeviceBuffer,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.copy_device_range_to_device_with_kernel(src, 0..src.byte_len())
    }

    #[doc(hidden)]
    pub fn copy_device_range_to_device_with_kernel(
        &self,
        src: &CudaDeviceBuffer,
        range: Range<usize>,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.copy_device_range_to_device_with_copy_u8_loader(src, range, |context| {
            context.inner.cuda_oxide_copy_u8_kernel_function()
        })
    }

    #[cfg(all(test, feature = "cuda-oxide-copy-u8", j2k_cuda_oxide_copy_u8_built))]
    pub(crate) fn copy_device_to_device_with_cuda_oxide_kernel(
        &self,
        src: &CudaDeviceBuffer,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.copy_device_range_to_device_with_copy_u8_loader(src, 0..src.byte_len(), |context| {
            context.inner.cuda_oxide_copy_u8_kernel_function()
        })
    }

    fn copy_device_range_to_device_with_copy_u8_loader(
        &self,
        src: &CudaDeviceBuffer,
        range: Range<usize>,
        load_function: impl FnOnce(&Self) -> Result<CuFunction, CudaError>,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        if !src.is_owned_by(self) {
            return Err(CudaError::InvalidArgument {
                message: "CUDA copy source must belong to the launch context".to_string(),
            });
        }
        if range.start > range.end {
            return Err(CudaError::InvalidArgument {
                message: "CUDA copy range start must not exceed its end".to_string(),
            });
        }
        if range.end > src.byte_len() {
            return Err(CudaError::OutputTooSmall {
                required: range.end,
                have: src.byte_len(),
            });
        }
        let byte_len = range.end - range.start;
        if byte_len == 0 {
            self.inner.set_current()?;
            return self.allocate(0);
        }
        let geometry =
            copy_u8_launch_geometry(byte_len).ok_or(CudaError::LengthTooLarge { len: byte_len })?;
        self.inner.set_current()?;
        let dst = self.allocate(byte_len)?;

        let source_offset = u64::try_from(range.start)
            .map_err(|_| CudaError::LengthTooLarge { len: range.start })?;
        let src_ptr = src
            .device_ptr()
            .checked_add(source_offset)
            .ok_or(CudaError::LengthTooLarge { len: range.end })?;
        let function = load_function(self)?;
        let mut dst_ptr = dst.device_ptr();
        let mut src_ptr = src_ptr;
        let mut len =
            u64::try_from(byte_len).map_err(|_| CudaError::LengthTooLarge { len: byte_len })?;
        let mut params = cuda_kernel_params!(dst_ptr, src_ptr, len);

        self.launch_kernel(function, geometry, &mut params)?;

        Ok(dst)
    }

    /// Synchronize all work submitted to this CUDA context.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.synchronize_for_resource_release().into_result()
    }

    /// Synchronize this context for queued-resource release.
    #[doc(hidden)]
    pub fn synchronize_for_resource_release(&self) -> CudaSynchronizationOutcome {
        let result = self.inner.with_current_completion_operation(|| {
            // SAFETY: the context lifetime gate is held and this CUDA context
            // is current for the calling thread.
            let status = unsafe { (self.inner.driver.cu_ctx_synchronize)() };
            self.inner.driver.check("cuCtxSynchronize", status)
        });
        match result {
            Ok(()) => {
                self.record_context_host_synchronization();
                CudaSynchronizationOutcome::Completed
            }
            Err(error) => {
                // The CUDA API may return both precondition failures and fatal
                // asynchronous errors here. Neither is sufficient evidence
                // that host-side resource release is safe.
                CudaSynchronizationOutcome::CompletionUncertain(error)
            }
        }
    }

    /// Synchronize before selecting `error`; if synchronization itself fails,
    /// select that completion failure instead.
    #[doc(hidden)]
    pub fn error_after_synchronize(&self, error: CudaError) -> CudaError {
        if self.inner.resource_lifetimes_poisoned() {
            // A synchronous operation may already have surfaced the driver
            // error that poisoned this context. Do not replace that primary
            // diagnostic with the generic follow-up poison sentinel.
            return select_uncertain_completion_error(error, None);
        }
        match self.synchronize() {
            Ok(()) => error,
            Err(completion_error) => {
                select_uncertain_completion_error(error, Some(completion_error))
            }
        }
    }

    /// Synchronize before returning `error`; if synchronization itself fails,
    /// return that completion failure instead.
    #[doc(hidden)]
    pub fn synchronize_then_error<T>(&self, error: CudaError) -> Result<T, CudaError> {
        Err(self.error_after_synchronize(error))
    }

    /// Preload a bundled CUDA kernel module and return its metadata handle.
    #[cfg(test)]
    pub(crate) fn preload_kernel_module(
        &self,
        kernel: CudaKernelName,
    ) -> Result<CudaKernelModule, CudaError> {
        let _ = self.inner.cuda_oxide_kernel_function(kernel.kernel())?;
        Ok(CudaKernelModule {
            entrypoint: kernel.entrypoint(),
        })
    }
}
