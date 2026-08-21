// SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) use j2k_cuda_runtime::elapsed_event_us_ceil;
pub(crate) use j2k_cuda_runtime::{
    cuda_kernel_param, CudaEvent, CudaExecutionStats, CudaKernelBatchOutput,
    CudaKernelContiguousBatchOutput, CudaKernelOutput, CudaPooledKernelOutput, CudaQueuedExecution,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CudaLaunchMode {
    Sync,
    Async,
}
