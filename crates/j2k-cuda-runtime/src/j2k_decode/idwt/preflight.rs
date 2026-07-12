// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{context::CudaContext, error::CudaError, memory::CudaDeviceBuffer};

use super::{
    context_validation::idwt_inputs_belong_to_context,
    job_validation::{validate_idwt_job, ValidatedIdwtJob},
    launch_validation::validate_idwt_single_launch,
};
use crate::j2k_decode::types::CudaJ2kIdwtJob;

pub(super) fn validate_idwt_single_request(
    context: &CudaContext,
    bands: [&CudaDeviceBuffer; 4],
    job: CudaJ2kIdwtJob,
) -> Result<ValidatedIdwtJob, CudaError> {
    if !idwt_inputs_belong_to_context(context, bands) {
        return Err(CudaError::InvalidArgument {
            message: "IDWT buffers must belong to the launch context".to_string(),
        });
    }
    let validated = validate_idwt_job(bands, None, job)?;
    validate_idwt_single_launch(validated.width, validated.height)?;
    Ok(validated)
}

impl CudaContext {
    /// Validate one inverse DWT request and return its exact output byte length
    /// without allocating or touching the CUDA driver.
    #[doc(hidden)]
    pub fn j2k_inverse_dwt_single_output_bytes(
        &self,
        ll: &CudaDeviceBuffer,
        hl: &CudaDeviceBuffer,
        lh: &CudaDeviceBuffer,
        hh: &CudaDeviceBuffer,
        job: CudaJ2kIdwtJob,
    ) -> Result<usize, CudaError> {
        validate_idwt_single_request(self, [ll, hl, lh, hh], job)
            .map(|validated| validated.output_bytes)
    }
}
