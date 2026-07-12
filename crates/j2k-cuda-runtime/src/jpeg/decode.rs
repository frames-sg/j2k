// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-oxide-jpeg-decode")]
use super::{validate_jpeg_rgb8_plan, validate_jpeg_rgb8_plan_with_pitch, CudaJpegDecodeStatus};
use super::{validation::validate_jpeg_buffer_context, CudaJpegRgb8DecodePlan};
#[cfg(feature = "cuda-oxide-jpeg-decode")]
use crate::allocation::HostPhaseBudget;
use crate::{
    context::CudaContext,
    error::CudaError,
    execution::{CudaExecutionStats, CudaKernelOutput},
    memory::CudaDeviceBuffer,
};

impl CudaContext {
    /// Decode one baseline JPEG RGB8 image to device-resident RGB8 using J2K CUDA kernels.
    #[doc(hidden)]
    pub fn decode_jpeg_rgb8_owned(
        &self,
        plan: &CudaJpegRgb8DecodePlan<'_>,
    ) -> Result<CudaKernelOutput, CudaError> {
        self.decode_jpeg_rgb8_owned_with_external_live(plan, 0)
    }

    /// Decode while charging host owners retained by the calling adapter.
    #[doc(hidden)]
    pub fn decode_jpeg_rgb8_owned_with_external_live(
        &self,
        plan: &CudaJpegRgb8DecodePlan<'_>,
        external_live_bytes: usize,
    ) -> Result<CudaKernelOutput, CudaError> {
        #[cfg(not(feature = "cuda-oxide-jpeg-decode"))]
        {
            let _ = (plan, external_live_bytes);
            Err(CudaError::InvalidArgument {
                message: "CUDA JPEG RGB8 decode PTX was not built".to_string(),
            })
        }

        #[cfg(feature = "cuda-oxide-jpeg-decode")]
        {
            let validated = validate_jpeg_rgb8_plan(plan)?;
            let statuses = allocate_decode_statuses_with_cap(
                plan.entropy_checkpoints.len(),
                external_live_bytes,
                j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            )?;
            self.inner.set_current()?;
            let output = self.allocate(validated.output_len)?;
            let execution =
                self.decode_jpeg_rgb8_owned_validated(plan, &output, validated, statuses)?;
            Ok(CudaKernelOutput {
                buffer: output,
                execution,
            })
        }
    }

    /// Decode one baseline JPEG RGB8 image into caller-owned CUDA RGB8 memory.
    /// `output` must belong to this context.
    #[doc(hidden)]
    pub fn decode_jpeg_rgb8_owned_into(
        &self,
        plan: &CudaJpegRgb8DecodePlan<'_>,
        output: &CudaDeviceBuffer,
        pitch_bytes: usize,
    ) -> Result<CudaExecutionStats, CudaError> {
        self.decode_jpeg_rgb8_owned_into_with_external_live(plan, output, pitch_bytes, 0)
    }

    /// Decode into caller memory while charging adapter-retained host owners.
    #[doc(hidden)]
    pub fn decode_jpeg_rgb8_owned_into_with_external_live(
        &self,
        plan: &CudaJpegRgb8DecodePlan<'_>,
        output: &CudaDeviceBuffer,
        pitch_bytes: usize,
        external_live_bytes: usize,
    ) -> Result<CudaExecutionStats, CudaError> {
        validate_jpeg_buffer_context(self, [output])?;

        #[cfg(not(feature = "cuda-oxide-jpeg-decode"))]
        {
            let _ = (plan, output, pitch_bytes, external_live_bytes);
            Err(CudaError::InvalidArgument {
                message: "CUDA JPEG RGB8 decode PTX was not built".to_string(),
            })
        }

        #[cfg(feature = "cuda-oxide-jpeg-decode")]
        {
            let validated = validate_jpeg_rgb8_plan_with_pitch(plan, pitch_bytes)?;
            if output.byte_len() < validated.output_len {
                return Err(CudaError::OutputTooSmall {
                    required: validated.output_len,
                    have: output.byte_len(),
                });
            }
            let statuses = allocate_decode_statuses_with_cap(
                plan.entropy_checkpoints.len(),
                external_live_bytes,
                j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            )?;
            self.inner.set_current()?;
            self.decode_jpeg_rgb8_owned_validated(plan, output, validated, statuses)
        }
    }
}

#[cfg(feature = "cuda-oxide-jpeg-decode")]
fn allocate_decode_statuses_with_cap(
    count: usize,
    external_live_bytes: usize,
    cap: usize,
) -> Result<Vec<CudaJpegDecodeStatus>, CudaError> {
    let mut host_budget = HostPhaseBudget::with_cap("CUDA JPEG decode status workspace", cap);
    host_budget.account_bytes(external_live_bytes)?;
    host_budget.try_vec_filled(count, CudaJpegDecodeStatus::default())
}

#[cfg(all(test, feature = "cuda-oxide-jpeg-decode"))]
mod tests {
    use super::allocate_decode_statuses_with_cap;
    use crate::{jpeg::CudaJpegDecodeStatus, CudaError};

    #[test]
    fn status_workspace_external_live_boundary_is_exact() {
        let external = 7;
        let status_bytes = 2 * core::mem::size_of::<CudaJpegDecodeStatus>();
        let exact = external + status_bytes;
        assert_eq!(
            allocate_decode_statuses_with_cap(2, external, exact)
                .expect("exact status workspace")
                .len(),
            2
        );
        assert!(matches!(
            allocate_decode_statuses_with_cap(2, external, exact - 1),
            Err(CudaError::HostAllocationTooLarge { requested, cap, .. })
                if requested == exact && cap == exact - 1
        ));
    }
}
