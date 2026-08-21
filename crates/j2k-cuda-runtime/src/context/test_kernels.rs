// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::kernels::CudaKernel;

/// Bundled CUDA kernel identifiers that can be preloaded by runtime internals.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CudaKernelName {
    CopyU8,
}

impl CudaKernelName {
    pub(crate) fn kernel(self) -> CudaKernel {
        match self {
            Self::CopyU8 => CudaKernel::CopyU8,
        }
    }

    pub(crate) fn entrypoint(self) -> &'static str {
        match self {
            Self::CopyU8 => "j2k_copy_u8",
        }
    }
}

/// Metadata for a preloaded CUDA kernel module entry point.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CudaKernelModule {
    pub(crate) entrypoint: &'static str,
}

impl CudaKernelModule {
    pub(crate) fn entrypoint(&self) -> &'static str {
        self.entrypoint
    }
}
