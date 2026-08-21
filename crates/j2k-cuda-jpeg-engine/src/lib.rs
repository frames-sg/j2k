// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA JPEG codec-engine boundary.
//!
//! The borrowed engine owns JPEG plans, validation, PTX packaging, and launch
//! orchestration while preserving the low-level
//! [`CudaContext`] as the public context identity.

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]
#![warn(unreachable_pub)]

macro_rules! cuda_kernel_params {
    ($($arg:ident),+ $(,)?) => {
        [$(cuda_kernel_param(&mut $arg)),+]
    };
}

mod allocation;
mod build_flags;
mod bytes;
mod error;
mod jpeg;
mod kernels;

mod execution {
    pub(crate) use j2k_cuda_runtime::{cuda_kernel_param, CudaExecutionStats, CudaKernelOutput};
}

mod memory {
    pub(crate) use j2k_cuda_runtime::{
        CudaDeviceBuffer, CudaPinnedUploadOperationGuard, CudaPinnedUploadStagingCheckout,
    };
}

pub use jpeg::{
    CudaJpegBaselineEncodeFormat, CudaJpegBaselineEncodeHuffmanTable, CudaJpegBaselineEncodeParams,
    CudaJpegBaselineEntropyEncodeBatchJob, CudaJpegBaselineEntropyEncodeJob,
    CudaJpegChunkedEntropyConfig, CudaJpegChunkedEntropyPlan, CudaJpegChunkedEntropyReport,
    CudaJpegDecodeStageTimings, CudaJpegEntropyCheckpoint, CudaJpegEntropyOverflowState,
    CudaJpegEntropySyncState, CudaJpegHuffmanTable, CudaJpegProfiledOutput, CudaJpegRgb8DecodePlan,
    CudaJpegRgb8Sampling,
};

use j2k_cuda_runtime::{
    CudaContext, CudaDeviceBuffer, CudaError, CudaExecutionStats, CudaKernelOutput,
    CudaPinnedUploadOperationGuard,
};
use kernels::{CudaKernel, CudaLaunchGeometry};

/// Borrowed JPEG codec operations over one low-level CUDA context.
#[derive(Clone, Copy)]
pub struct JpegCudaEngine<'context> {
    context: &'context CudaContext,
}

impl<'context> JpegCudaEngine<'context> {
    /// Bind JPEG operations to `context` without changing its ownership.
    #[must_use]
    pub const fn new(context: &'context CudaContext) -> Self {
        Self { context }
    }

    /// Return the borrowed low-level context.
    #[must_use]
    pub const fn context(self) -> &'context CudaContext {
        self.context
    }

    fn allocate(self, len: usize) -> Result<CudaDeviceBuffer, CudaError> {
        self.context.allocate(len)
    }

    fn upload(self, bytes: &[u8]) -> Result<CudaDeviceBuffer, CudaError> {
        self.context.upload(bytes)
    }

    fn memset_d8(
        self,
        output: &CudaDeviceBuffer,
        value: u8,
        bytes: usize,
    ) -> Result<(), CudaError> {
        self.context.memset_d8(output, value, bytes)
    }

    fn begin_pinned_upload_operation(
        self,
    ) -> Result<CudaPinnedUploadOperationGuard<'context>, CudaError> {
        self.context.begin_pinned_upload_operation()
    }

    fn launch_kernel(
        self,
        kernel: CudaKernel,
        geometry: CudaLaunchGeometry,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<(), CudaError> {
        let spec = kernel.spec()?;
        // SAFETY: this private boundary is called only by the JPEG launch
        // builders, which keep their typed ABI values and context-owned device
        // buffers alive through the synchronous launch.
        unsafe { self.context.launch_compiled_kernel(spec, geometry, params) }
    }

    /// Decode baseline JPEG RGB8 into a runtime-owned device buffer while
    /// charging adapter-retained host owners.
    pub fn decode_rgb8_owned_with_external_live(
        self,
        plan: &CudaJpegRgb8DecodePlan<'_>,
        external_live_bytes: usize,
    ) -> Result<CudaKernelOutput, CudaError> {
        self.decode_jpeg_rgb8_owned_with_external_live(plan, external_live_bytes)
    }

    /// Profile the existing baseline JPEG RGB8 route without changing its
    /// kernel selection or launch geometry.
    #[doc(hidden)]
    pub fn profile_decode_rgb8_owned_with_external_live(
        self,
        plan: &CudaJpegRgb8DecodePlan<'_>,
        external_live_bytes: usize,
    ) -> Result<CudaJpegProfiledOutput, CudaError> {
        self.profile_decode_jpeg_rgb8_owned_with_external_live(plan, external_live_bytes)
    }

    /// Decode baseline JPEG RGB8 into caller-owned context memory while
    /// charging adapter-retained host owners.
    pub fn decode_rgb8_owned_into_with_external_live(
        self,
        plan: &CudaJpegRgb8DecodePlan<'_>,
        output: &CudaDeviceBuffer,
        pitch_bytes: usize,
        external_live_bytes: usize,
    ) -> Result<CudaExecutionStats, CudaError> {
        self.decode_jpeg_rgb8_owned_into_with_external_live(
            plan,
            output,
            pitch_bytes,
            external_live_bytes,
        )
    }

    /// Encode one resident tile into baseline JPEG entropy bytes while
    /// charging adapter-retained host owners.
    pub fn encode_baseline_entropy_with_external_live(
        self,
        job: &CudaJpegBaselineEntropyEncodeJob<'_>,
        external_live_bytes: usize,
    ) -> Result<Vec<u8>, CudaError> {
        self.encode_jpeg_baseline_entropy_with_external_live(job, external_live_bytes)
    }

    /// Encode resident tiles into baseline JPEG entropy chunks while charging
    /// adapter-retained host owners.
    pub fn encode_baseline_entropy_batch_with_external_live(
        self,
        job: &CudaJpegBaselineEntropyEncodeBatchJob<'_>,
        external_live_bytes: usize,
    ) -> Result<Vec<Vec<u8>>, CudaError> {
        self.encode_jpeg_baseline_entropy_batch_with_external_live(job, external_live_bytes)
    }

    /// Run 4:2:0 entropy diagnostics inside an adapter-held pinned-upload
    /// transaction.
    pub fn diagnose_entropy_with_pinned_upload_operation(
        self,
        plan: &CudaJpegChunkedEntropyPlan<'_>,
        external_live_bytes: usize,
        pinned_upload: &CudaPinnedUploadOperationGuard<'_>,
    ) -> Result<CudaJpegChunkedEntropyReport, CudaError> {
        self.diagnose_jpeg_420_entropy_self_sync_with_pinned_upload_operation(
            plan,
            external_live_bytes,
            pinned_upload,
        )
    }

    /// Validate that a caller-owned JPEG output buffer belongs to the bound
    /// low-level context.
    pub fn validate_output_buffer(self, output: &CudaDeviceBuffer) -> Result<(), CudaError> {
        self.validate_jpeg_output_buffer_context(output)
    }
}

#[cfg(test)]
mod tests {
    use super::JpegCudaEngine;

    #[test]
    fn engine_constructor_preserves_the_low_level_context_type() {
        assert_eq!(
            std::mem::size_of::<JpegCudaEngine<'static>>(),
            std::mem::size_of::<&j2k_cuda_runtime::CudaContext>()
        );
    }
}
