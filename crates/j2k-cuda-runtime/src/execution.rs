use crate::{
    build_flags::cuda_stage_timings_disabled,
    context::{CudaContext, CudaKernelModule, CudaKernelName},
    driver::{CuDevicePtr, CuEvent, CuFunction, CuStream, CudaNvtxRange},
    error::CudaError,
    kernels::{self, copy_u8_launch_geometry, CudaKernel},
    memory::{CudaDeviceBuffer, CudaDeviceBufferRange, CudaPooledDeviceBuffer},
};
use std::{ffi::c_void, os::raw::c_uint};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CudaLaunchMode {
    Sync,
    Async,
}

impl CudaLaunchMode {
    pub(crate) fn from_synchronize(synchronize: bool) -> Self {
        if synchronize {
            Self::Sync
        } else {
            Self::Async
        }
    }
}

/// Marker for values that can be passed by address to a CUDA kernel launch.
///
/// # Safety
///
/// Implementors must have a stable, CUDA-compatible by-value representation for
/// the duration of `cuLaunchKernel`.
pub(crate) unsafe trait CudaKernelParam {}

// SAFETY: `CuDevicePtr` is the raw CUDA device pointer value expected by kernels.
unsafe impl CudaKernelParam for CuDevicePtr {}
// SAFETY: CUDA kernels receive these scalar ABI types by value via parameter pointers.
unsafe impl CudaKernelParam for u32 {}
// SAFETY: CUDA kernels receive these scalar ABI types by value via parameter pointers.
unsafe impl CudaKernelParam for i32 {}
// SAFETY: CUDA kernels receive these scalar ABI types by value via parameter pointers.
unsafe impl CudaKernelParam for f32 {}

pub(crate) fn cuda_kernel_param<T>(value: &mut T) -> *mut c_void
where
    T: CudaKernelParam,
{
    std::ptr::from_mut(value).cast::<c_void>()
}

impl CudaContext {
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

    /// Copy host bytes through the opt-in cuda-oxide `CopyU8` kernel.
    #[cfg(feature = "cuda-oxide-copy-u8")]
    pub fn copy_with_cuda_oxide_kernel(&self, bytes: &[u8]) -> Result<CudaKernelOutput, CudaError> {
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

    pub(crate) fn launch_named_kernel<const N: usize>(
        &self,
        kernel: CudaKernel,
        geometry: kernels::CudaLaunchGeometry,
        params: &mut [*mut c_void; N],
        mode: CudaLaunchMode,
    ) -> Result<(), CudaError> {
        let function = self.inner.kernel_function(kernel)?;
        match mode {
            CudaLaunchMode::Sync => self.launch_kernel(function, geometry, params),
            CudaLaunchMode::Async => self.launch_kernel_async(function, geometry, params),
        }
    }

    pub(crate) fn launch_kernel_async(
        &self,
        function: CuFunction,
        geometry: kernels::CudaLaunchGeometry,
        params: &mut [*mut c_void],
    ) -> Result<(), CudaError> {
        // SAFETY: `function` was loaded from a live module in this context, and
        // `params` contains kernel argument pointers valid for the launch call.
        let launch_status = unsafe {
            (self.inner.driver.cu_launch_kernel)(
                function,
                geometry.grid.0,
                geometry.grid.1,
                geometry.grid.2,
                geometry.block.0,
                geometry.block.1,
                geometry.block.2,
                0,
                std::ptr::null_mut(),
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        self.inner.driver.check("cuLaunchKernel", launch_status)
    }

    /// Copy one device buffer to another through a CUDA kernel.
    pub fn copy_device_to_device_with_kernel(
        &self,
        src: &CudaDeviceBuffer,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.copy_device_ptr_to_device_with_kernel(src.device_ptr(), src.byte_len())
    }

    #[cfg(feature = "cuda-oxide-copy-u8")]
    pub(crate) fn copy_device_to_device_with_cuda_oxide_kernel(
        &self,
        src: &CudaDeviceBuffer,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.copy_device_ptr_to_device_with_cuda_oxide_kernel(src.device_ptr(), src.byte_len())
    }

    pub(crate) fn copy_device_ptr_to_device_with_kernel(
        &self,
        src_ptr: CuDevicePtr,
        byte_len: usize,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.copy_device_ptr_to_device_with_copy_u8_loader(src_ptr, byte_len, |context| {
            context.inner.kernel_function(CudaKernel::CopyU8)
        })
    }

    #[cfg(feature = "cuda-oxide-copy-u8")]
    pub(crate) fn copy_device_ptr_to_device_with_cuda_oxide_kernel(
        &self,
        src_ptr: CuDevicePtr,
        byte_len: usize,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.copy_device_ptr_to_device_with_copy_u8_loader(src_ptr, byte_len, |context| {
            context.inner.cuda_oxide_copy_u8_kernel_function()
        })
    }

    fn copy_device_ptr_to_device_with_copy_u8_loader(
        &self,
        src_ptr: CuDevicePtr,
        byte_len: usize,
        load_function: impl FnOnce(&Self) -> Result<CuFunction, CudaError>,
    ) -> Result<CudaDeviceBuffer, CudaError> {
        self.inner.set_current()?;
        let dst = self.allocate(byte_len)?;
        if byte_len == 0 {
            return Ok(dst);
        }

        let function = load_function(self)?;
        let mut dst_ptr = dst.device_ptr();
        let mut src_ptr = src_ptr;
        let mut len =
            u64::try_from(byte_len).map_err(|_| CudaError::LengthTooLarge { len: byte_len })?;
        let mut params = cuda_kernel_params!(dst_ptr, src_ptr, len);
        let geometry =
            copy_u8_launch_geometry(byte_len).ok_or(CudaError::LengthTooLarge { len: byte_len })?;

        self.launch_kernel(function, geometry, &mut params)?;

        Ok(dst)
    }

    pub(crate) fn memset_d32(
        &self,
        dst: &CudaDeviceBuffer,
        value: c_uint,
        words: usize,
    ) -> Result<(), CudaError> {
        self.inner.set_current()?;
        let required = words
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(CudaError::LengthTooLarge { len: words })?;
        if required > dst.byte_len() {
            return Err(CudaError::OutputTooSmall {
                required,
                have: dst.byte_len(),
            });
        }
        if words == 0 {
            return Ok(());
        }
        // SAFETY: `dst` is a live CUDA allocation in this context and `words`
        // was bounds-checked against the allocation byte length above.
        self.inner.driver.check("cuMemsetD32_v2", unsafe {
            (self.inner.driver.cu_memset_d32)(dst.device_ptr(), value, words)
        })
    }

    /// Create a CUDA stream owned by this context.
    pub fn create_stream(&self) -> Result<CudaStream, CudaError> {
        self.inner.set_current()?;
        let mut stream = std::ptr::null_mut();
        // SAFETY: CUDA writes a new stream handle, destroyed by CudaStream.
        self.inner.driver.check("cuStreamCreate", unsafe {
            (self.inner.driver.cu_stream_create)(&raw mut stream, 0)
        })?;
        Ok(CudaStream {
            context: self.clone(),
            stream,
        })
    }

    /// Create a CUDA timing event owned by this context.
    pub fn create_event(&self) -> Result<CudaEvent, CudaError> {
        self.inner.set_current()?;
        let mut event = std::ptr::null_mut();
        // SAFETY: CUDA writes a new event handle, destroyed by CudaEvent.
        self.inner.driver.check("cuEventCreate", unsafe {
            (self.inner.driver.cu_event_create)(&raw mut event, 0)
        })?;
        Ok(CudaEvent {
            context: self.clone(),
            event,
        })
    }

    /// Time work submitted to the default CUDA stream and return elapsed microseconds.
    pub fn time_default_stream_us<T>(
        &self,
        work: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<(T, u128), CudaError> {
        self.inner.set_current()?;
        if cuda_stage_timings_disabled() {
            return work().map(|output| (output, 0));
        }
        let start = self.create_event()?;
        let end = self.create_event()?;
        start.record_default_stream()?;
        let output = match work() {
            Ok(output) => output,
            Err(error) => {
                // Timed closures may submit asynchronous default-stream work.
                // On a later host-side error, wait before dropping any device
                // buffers captured by the closure.
                self.synchronize()?;
                return Err(error);
            }
        };
        end.record_default_stream()?;
        end.synchronize()?;
        Ok((output, elapsed_event_us_ceil(&start, &end)?))
    }

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

    /// Time work submitted to the default CUDA stream inside an optional NVTX range.
    ///
    /// The NVTX range is a no-op unless the crate is built with
    /// `cuda-profiling` and an NVTX runtime library can be loaded dynamically.
    pub fn time_default_stream_named_us<T>(
        &self,
        name: &str,
        work: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<(T, u128), CudaError> {
        self.with_nvtx_range(name, || self.time_default_stream_us(work))
    }

    /// Optionally time work submitted to the default CUDA stream inside an NVTX range.
    pub fn time_default_stream_named_us_if<T>(
        &self,
        collect_stage_timings: bool,
        name: &str,
        work: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<(T, u128), CudaError> {
        if collect_stage_timings {
            self.time_default_stream_named_us(name, work)
        } else {
            self.with_nvtx_range(name, || work().map(|output| (output, 0)))
        }
    }

    /// Synchronize all work submitted to this CUDA context.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.inner.set_current()?;
        // SAFETY: a CUDA context is current for this `CudaContext`.
        let status = unsafe { (self.inner.driver.cu_ctx_synchronize)() };
        self.inner.driver.check("cuCtxSynchronize", status)
    }

    /// Preload a bundled CUDA kernel module and return its metadata handle.
    pub fn preload_kernel_module(
        &self,
        kernel: CudaKernelName,
    ) -> Result<CudaKernelModule, CudaError> {
        let _ = self.inner.kernel_function(kernel.kernel())?;
        Ok(CudaKernelModule {
            kernel,
            entrypoint: kernel.entrypoint(),
        })
    }
}

/// CUDA stream RAII handle.
#[derive(Debug)]
pub struct CudaStream {
    pub(crate) context: CudaContext,
    pub(crate) stream: CuStream,
}

impl CudaStream {
    /// Synchronize all work submitted to this stream.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.context.inner.set_current()?;
        // SAFETY: stream is a live CUDA stream owned by this handle.
        self.context
            .inner
            .driver
            .check("cuStreamSynchronize", unsafe {
                (self.context.inner.driver.cu_stream_synchronize)(self.stream)
            })
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.stream.is_null() {
            let _ = self.context.inner.set_current();
            // SAFETY: stream was created by this context. Drop cannot surface
            // errors, so cleanup failures are ignored.
            let _ = unsafe { (self.context.inner.driver.cu_stream_destroy)(self.stream) };
        }
    }
}

// SAFETY: CUDA stream handles are driver-owned resources. The Rust handle owns
// destruction and does not expose mutable aliasing of Rust memory.
unsafe impl Send for CudaStream {}

/// CUDA event RAII handle for timing and synchronization.
#[derive(Debug)]
pub struct CudaEvent {
    pub(crate) context: CudaContext,
    pub(crate) event: CuEvent,
}

impl CudaEvent {
    /// Record this event on a CUDA stream.
    pub fn record(&self, stream: &CudaStream) -> Result<(), CudaError> {
        self.context.inner.set_current()?;
        // SAFETY: event and stream are live CUDA handles.
        self.context.inner.driver.check("cuEventRecord", unsafe {
            (self.context.inner.driver.cu_event_record)(self.event, stream.stream)
        })
    }

    pub(crate) fn record_default_stream(&self) -> Result<(), CudaError> {
        self.context.inner.set_current()?;
        // SAFETY: a null stream is CUDA's default stream for the current context.
        self.context.inner.driver.check("cuEventRecord", unsafe {
            (self.context.inner.driver.cu_event_record)(self.event, std::ptr::null_mut())
        })
    }

    /// Wait for this event to complete.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.context.inner.set_current()?;
        // SAFETY: event is a live CUDA event owned by this handle.
        self.context
            .inner
            .driver
            .check("cuEventSynchronize", unsafe {
                (self.context.inner.driver.cu_event_synchronize)(self.event)
            })
    }

    /// Elapsed time in microseconds from `start` to `end`.
    pub fn elapsed_time_us(start: &Self, end: &Self) -> Result<f32, CudaError> {
        end.context.inner.set_current()?;
        let mut millis = 0.0f32;
        // SAFETY: start and end are live CUDA events that have been recorded.
        let status = unsafe {
            (end.context.inner.driver.cu_event_elapsed_time)(
                &raw mut millis,
                start.event,
                end.event,
            )
        };
        end.context
            .inner
            .driver
            .check("cuEventElapsedTime", status)?;
        Ok(millis * 1000.0)
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn elapsed_event_us_ceil(start: &CudaEvent, end: &CudaEvent) -> Result<u128, CudaError> {
    let elapsed = CudaEvent::elapsed_time_us(start, end)?;
    if elapsed <= 0.0 {
        return Ok(1);
    }
    Ok(elapsed.ceil() as u128)
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if !self.event.is_null() {
            let _ = self.context.inner.set_current();
            // SAFETY: event was created by this context. Drop cannot surface
            // errors, so cleanup failures are ignored.
            let _ = unsafe { (self.context.inner.driver.cu_event_destroy)(self.event) };
        }
    }
}

// SAFETY: CUDA event handles are driver-owned resources. The Rust handle owns
// destruction and does not expose mutable aliasing of Rust memory.
unsafe impl Send for CudaEvent {}

/// Device buffer plus execution metadata.
#[derive(Debug)]
pub struct CudaKernelOutput {
    pub(crate) buffer: CudaDeviceBuffer,
    pub(crate) execution: CudaExecutionStats,
}

/// Multiple device buffers plus shared execution metadata from one batched kernel.
#[derive(Debug)]
pub struct CudaKernelBatchOutput {
    pub(crate) outputs: Vec<CudaDeviceBuffer>,
    pub(crate) execution: CudaExecutionStats,
}

/// One contiguous device buffer plus per-item ranges from one batched kernel.
#[derive(Debug)]
pub struct CudaKernelContiguousBatchOutput {
    pub(crate) output: CudaDeviceBuffer,
    pub(crate) ranges: Vec<CudaDeviceBufferRange>,
    pub(crate) execution: CudaExecutionStats,
}

/// Pooled device buffer plus execution metadata.
#[derive(Debug)]
pub struct CudaPooledKernelOutput {
    pub(crate) buffer: CudaPooledDeviceBuffer,
    pub(crate) execution: CudaExecutionStats,
}

/// Enqueued CUDA work plus pooled resources that must stay live until the
/// default stream is synchronized.
#[derive(Debug)]
pub struct CudaQueuedExecution {
    pub(crate) resources: Vec<CudaPooledDeviceBuffer>,
    pub(crate) execution: CudaExecutionStats,
}

impl CudaQueuedExecution {
    /// CUDA execution counters for the enqueued work.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// Number of pooled resource buffers held live for the queued work.
    pub fn resource_count(&self) -> usize {
        self.resources.len()
    }
}

impl CudaKernelOutput {
    /// Device buffer produced by the kernel.
    pub fn buffer(&self) -> &CudaDeviceBuffer {
        &self.buffer
    }

    /// CUDA execution counters for the kernel.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// Split output into device buffer and execution metadata.
    pub fn into_parts(self) -> (CudaDeviceBuffer, CudaExecutionStats) {
        (self.buffer, self.execution)
    }
}

impl CudaKernelBatchOutput {
    /// Device buffers produced by the batched kernel.
    pub fn outputs(&self) -> &[CudaDeviceBuffer] {
        &self.outputs
    }

    /// CUDA execution counters for the batched kernel.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// Split output into device buffers and execution metadata.
    pub fn into_parts(self) -> (Vec<CudaDeviceBuffer>, CudaExecutionStats) {
        (self.outputs, self.execution)
    }
}

impl CudaKernelContiguousBatchOutput {
    /// Contiguous device buffer produced by the batched kernel.
    pub fn output(&self) -> &CudaDeviceBuffer {
        &self.output
    }

    /// Per-item byte ranges inside the contiguous output buffer.
    pub fn ranges(&self) -> &[CudaDeviceBufferRange] {
        &self.ranges
    }

    /// CUDA execution counters for the batched kernel.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// Split output into the contiguous buffer, per-item ranges, and execution metadata.
    pub fn into_parts(
        self,
    ) -> (
        CudaDeviceBuffer,
        Vec<CudaDeviceBufferRange>,
        CudaExecutionStats,
    ) {
        (self.output, self.ranges, self.execution)
    }
}

impl CudaPooledKernelOutput {
    /// Device buffer produced by the kernel.
    pub fn buffer(&self) -> Option<&CudaDeviceBuffer> {
        self.buffer.as_device_buffer()
    }

    /// CUDA execution counters for the kernel.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// Split output into pooled device buffer and execution metadata.
    pub fn into_parts(self) -> (CudaPooledDeviceBuffer, CudaExecutionStats) {
        (self.buffer, self.execution)
    }
}

/// CUDA execution counters exposed for dispatch observability.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaExecutionStats {
    pub(crate) kernel_dispatches: usize,
    pub(crate) copy_kernel_dispatches: usize,
    pub(crate) decode_kernel_dispatches: usize,
    pub(crate) hardware_decode: bool,
}

impl CudaExecutionStats {
    /// Total kernel dispatch count.
    pub fn kernel_dispatches(self) -> usize {
        self.kernel_dispatches
    }

    /// Copy-kernel dispatch count.
    pub fn copy_kernel_dispatches(self) -> usize {
        self.copy_kernel_dispatches
    }

    /// Hardware decode dispatch count.
    pub fn decode_kernel_dispatches(self) -> usize {
        self.decode_kernel_dispatches
    }

    /// True when a hardware decode path was used.
    pub fn used_hardware_decode(self) -> bool {
        self.hardware_decode
    }
}
