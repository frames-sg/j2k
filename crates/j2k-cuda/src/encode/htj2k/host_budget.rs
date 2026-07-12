// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::allocation::HostPhaseBudget;
use crate::encode::stage_error::CudaStageResult;

use super::htj2k_allocation_error;
use super::types::CudaEncodedHtj2kResolution;

pub(super) fn account_encoded_resolution_owners(
    host_budget: &mut HostPhaseBudget,
    resolutions: &[CudaEncodedHtj2kResolution],
    resolution_capacity: usize,
) -> CudaStageResult<()> {
    host_budget
        .account_capacity::<CudaEncodedHtj2kResolution>(resolution_capacity)
        .map_err(htj2k_allocation_error)?;
    for resolution in resolutions {
        host_budget
            .account_vec(&resolution.subbands)
            .map_err(htj2k_allocation_error)?;
        for subband in &resolution.subbands {
            host_budget
                .account_vec(&subband.code_blocks)
                .map_err(htj2k_allocation_error)?;
            for block in &subband.code_blocks {
                host_budget
                    .account_vec(&block.data)
                    .map_err(htj2k_allocation_error)?;
            }
        }
    }
    Ok(())
}
