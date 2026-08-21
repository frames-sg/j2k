// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    error::CudaError,
    memory::{
        CudaBufferPoolReuseGuard, CudaDeviceBuffer, CudaDeviceBufferRange, CudaPooledDeviceBuffer,
    },
};

#[doc(hidden)]
/// Device buffer plus execution metadata.
#[derive(Debug)]
pub struct CudaKernelOutput {
    #[doc(hidden)]
    pub buffer: CudaDeviceBuffer,
    #[doc(hidden)]
    pub execution: CudaExecutionStats,
}

#[doc(hidden)]
/// Multiple device buffers plus shared execution metadata from one batched kernel.
#[derive(Debug)]
pub struct CudaKernelBatchOutput {
    #[doc(hidden)]
    pub outputs: Vec<CudaDeviceBuffer>,
    #[doc(hidden)]
    pub execution: CudaExecutionStats,
}

#[doc(hidden)]
/// One contiguous device buffer plus per-item ranges from one batched kernel.
#[derive(Debug)]
pub struct CudaKernelContiguousBatchOutput {
    #[doc(hidden)]
    pub output: CudaDeviceBuffer,
    #[doc(hidden)]
    pub ranges: Vec<CudaDeviceBufferRange>,
    #[doc(hidden)]
    pub execution: CudaExecutionStats,
}

#[doc(hidden)]
/// Pooled device buffer plus execution metadata.
#[derive(Debug)]
pub struct CudaPooledKernelOutput {
    #[doc(hidden)]
    pub buffer: CudaPooledDeviceBuffer,
    #[doc(hidden)]
    pub execution: CudaExecutionStats,
}

/// Enqueued CUDA work plus pooled resources that must stay unavailable for
/// reuse until the default stream is synchronized. Dropping an unreleased
/// value synchronizes before pool reuse.
#[doc(hidden)]
#[derive(Debug)]
#[must_use = "queued CUDA work must be finished or retained until Drop synchronizes it"]
pub struct CudaQueuedExecution {
    #[doc(hidden)]
    pub resources: Vec<CudaPooledDeviceBuffer>,
    #[doc(hidden)]
    pub execution: CudaExecutionStats,
    #[doc(hidden)]
    pub pool_reuse_guard: Option<CudaBufferPoolReuseGuard>,
}

impl CudaQueuedExecution {
    /// Retain pooled resources and their reuse guard for externally owned
    /// engine work submitted to this runtime context.
    #[doc(hidden)]
    pub fn new(
        resources: Vec<CudaPooledDeviceBuffer>,
        execution: CudaExecutionStats,
        pool_reuse_guard: Option<CudaBufferPoolReuseGuard>,
    ) -> Self {
        Self {
            resources,
            execution,
            pool_reuse_guard,
        }
    }

    /// CUDA execution counters for the enqueued work.
    pub fn execution(&self) -> CudaExecutionStats {
        self.execution
    }

    /// Number of pooled resource buffers held live for the queued work.
    pub fn resource_count(&self) -> usize {
        self.resources.len()
    }

    /// Synchronize the queued work, release its pool hold, and surface any
    /// completion failure.
    pub fn finish(mut self) -> Result<CudaExecutionStats, CudaError> {
        let completion_result = self
            .pool_reuse_guard
            .take()
            .map_or(Ok(()), CudaBufferPoolReuseGuard::synchronize_and_release);
        self.resources.clear();
        completion_result?;
        Ok(self.execution)
    }

    /// Synchronize queued work and return its retained pooled resources.
    ///
    /// This is the completion path for engines that defer a device-to-host
    /// readback until a submitted kernel has finished. The resources are safe
    /// to inspect and remain unavailable for pool reuse while returned owners
    /// stay live.
    #[doc(hidden)]
    pub fn finish_with_resources(
        mut self,
    ) -> Result<(Vec<CudaPooledDeviceBuffer>, CudaExecutionStats), CudaError> {
        let completion_result = self
            .pool_reuse_guard
            .take()
            .map_or(Ok(()), CudaBufferPoolReuseGuard::synchronize_and_release);
        if let Err(error) = completion_result {
            self.resources.clear();
            return Err(error);
        }
        Ok((std::mem::take(&mut self.resources), self.execution))
    }

    /// Release the pool hold and return retained resources after externally
    /// established context completion.
    ///
    /// # Safety
    ///
    /// The owning CUDA context must have completed this queued work. Merely
    /// ordering dependent default-stream work is insufficient.
    #[doc(hidden)]
    pub unsafe fn finish_with_resources_after_completion(
        mut self,
    ) -> Result<(Vec<CudaPooledDeviceBuffer>, CudaExecutionStats), CudaError> {
        if let Some(guard) = self.pool_reuse_guard.take() {
            guard.release()?;
        }
        Ok((std::mem::take(&mut self.resources), self.execution))
    }

    /// Release deferred pool buffers after the owning context has completed
    /// this queued work.
    ///
    /// # Safety
    ///
    /// The owning CUDA context must have completed this queued work. Merely
    /// ordering a dependent kernel is insufficient because Rust owners could
    /// otherwise deallocate the pool before either kernel completes.
    #[doc(hidden)]
    pub unsafe fn release_pool_reuse_after_completion(&mut self) -> Result<(), CudaError> {
        self.resources.clear();
        if let Some(guard) = self.pool_reuse_guard.take() {
            guard.release()?;
        }
        Ok(())
    }
}

impl Drop for CudaQueuedExecution {
    fn drop(&mut self) {
        let Some(guard) = self.pool_reuse_guard.take() else {
            return;
        };

        // Keep resources owned while driver synchronization is attempted. Any
        // synchronization error leaves completion uncertain, so recycling puts
        // them behind the permanently retained pool hold.
        let outcome = guard.synchronize_pool_context();
        self.resources.clear();
        if outcome.completion_established() {
            let _ = guard.release();
        } else {
            guard.abandon();
        }
    }
}

impl CudaKernelOutput {
    /// Combine a context-owned device buffer with its execution counters.
    #[doc(hidden)]
    pub fn new(buffer: CudaDeviceBuffer, execution: CudaExecutionStats) -> Self {
        Self { buffer, execution }
    }

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
    /// Combine context-owned device buffers with their shared execution counters.
    #[doc(hidden)]
    pub fn new(outputs: Vec<CudaDeviceBuffer>, execution: CudaExecutionStats) -> Self {
        Self { outputs, execution }
    }

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
    /// Combine a contiguous output allocation, item ranges, and execution counters.
    #[doc(hidden)]
    pub fn new(
        output: CudaDeviceBuffer,
        ranges: Vec<CudaDeviceBufferRange>,
        execution: CudaExecutionStats,
    ) -> Self {
        Self {
            output,
            ranges,
            execution,
        }
    }

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
    /// Combine a pooled device buffer with its execution counters.
    #[doc(hidden)]
    pub fn new(buffer: CudaPooledDeviceBuffer, execution: CudaExecutionStats) -> Self {
        Self { buffer, execution }
    }

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
    #[doc(hidden)]
    pub kernel_dispatches: usize,
    #[doc(hidden)]
    pub copy_kernel_dispatches: usize,
    #[doc(hidden)]
    pub decode_kernel_dispatches: usize,
    #[doc(hidden)]
    pub hardware_decode: bool,
}

impl CudaExecutionStats {
    /// Construct execution counters recorded by an external codec engine.
    #[doc(hidden)]
    pub const fn new(
        kernel_dispatches: usize,
        copy_kernel_dispatches: usize,
        decode_kernel_dispatches: usize,
        hardware_decode: bool,
    ) -> Self {
        Self {
            kernel_dispatches,
            copy_kernel_dispatches,
            decode_kernel_dispatches,
            hardware_decode,
        }
    }

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
