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
    pub(crate) buffer: CudaDeviceBuffer,
    pub(crate) execution: CudaExecutionStats,
}

#[doc(hidden)]
/// Multiple device buffers plus shared execution metadata from one batched kernel.
#[derive(Debug)]
pub struct CudaKernelBatchOutput {
    pub(crate) outputs: Vec<CudaDeviceBuffer>,
    pub(crate) execution: CudaExecutionStats,
}

#[doc(hidden)]
/// One contiguous device buffer plus per-item ranges from one batched kernel.
#[derive(Debug)]
pub struct CudaKernelContiguousBatchOutput {
    pub(crate) output: CudaDeviceBuffer,
    pub(crate) ranges: Vec<CudaDeviceBufferRange>,
    pub(crate) execution: CudaExecutionStats,
}

#[doc(hidden)]
/// Pooled device buffer plus execution metadata.
#[derive(Debug)]
pub struct CudaPooledKernelOutput {
    pub(crate) buffer: CudaPooledDeviceBuffer,
    pub(crate) execution: CudaExecutionStats,
}

/// Enqueued CUDA work plus pooled resources that must stay unavailable for
/// reuse until the default stream is synchronized. Dropping an unreleased
/// value synchronizes before pool reuse.
#[doc(hidden)]
#[derive(Debug)]
#[must_use = "queued CUDA work must be finished or retained until Drop synchronizes it"]
pub struct CudaQueuedExecution {
    pub(crate) resources: Vec<CudaPooledDeviceBuffer>,
    pub(crate) execution: CudaExecutionStats,
    pub(crate) pool_reuse_guard: Option<CudaBufferPoolReuseGuard>,
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
