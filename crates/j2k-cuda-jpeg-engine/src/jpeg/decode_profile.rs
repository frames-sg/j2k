// SPDX-License-Identifier: MIT OR Apache-2.0

/// Profile-only timing attribution for the existing CUDA JPEG RGB8 decode route.
///
/// The values describe one decode submission. They do not select a different
/// kernel or change launch geometry.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[doc(hidden)]
pub struct CudaJpegDecodeStageTimings {
    pub(crate) resource_upload_us: u128,
    pub(crate) fused_decode_kernel_us: u128,
    pub(crate) conversion_us: u128,
    pub(crate) status_readback_us: u128,
    pub(crate) component_workspace_bytes: usize,
}

impl CudaJpegDecodeStageTimings {
    /// Host wall time spent allocating and uploading immutable decode resources.
    pub const fn resource_upload_us(self) -> u128 {
        self.resource_upload_us
    }

    /// CUDA event time spent in the existing entropy, IDCT, and plane-output kernel.
    pub const fn fused_decode_kernel_us(self) -> u128 {
        self.fused_decode_kernel_us
    }

    /// CUDA event time spent in the existing subsampled-plane conversion kernel.
    pub const fn conversion_us(self) -> u128 {
        self.conversion_us
    }

    /// Host wall time spent reading and validating kernel status records.
    pub const fn status_readback_us(self) -> u128 {
        self.status_readback_us
    }

    /// Existing component-plane workspace used by 4:2:0 or 4:2:2 conversion.
    pub const fn component_workspace_bytes(self) -> usize {
        self.component_workspace_bytes
    }
}

/// Runtime-owned CUDA JPEG RGB8 output plus profile-only stage attribution.
#[derive(Debug)]
#[doc(hidden)]
pub struct CudaJpegProfiledOutput {
    pub(crate) output: j2k_cuda_runtime::CudaKernelOutput,
    pub(crate) stage_timings: CudaJpegDecodeStageTimings,
}

impl CudaJpegProfiledOutput {
    /// Split the output into its normal kernel result and profile-only timings.
    pub fn into_parts(
        self,
    ) -> (
        j2k_cuda_runtime::CudaKernelOutput,
        CudaJpegDecodeStageTimings,
    ) {
        (self.output, self.stage_timings)
    }
}
