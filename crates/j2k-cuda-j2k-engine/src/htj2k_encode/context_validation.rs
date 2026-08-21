// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    context::{ensure_context_ownership, CudaContext},
    error::CudaError,
    memory::{CudaBufferPool, CudaDeviceBuffer},
};

use super::CudaHtj2kEncodeResources;

const HTJ2K_ENCODE_CONTEXT_MISMATCH: &str =
    "HTJ2K encode coefficients, resources, and pool must belong to the launch context";

fn validate_htj2k_encode_context_matches(
    matches_context: impl IntoIterator<Item = bool>,
) -> Result<(), CudaError> {
    ensure_context_ownership(matches_context, HTJ2K_ENCODE_CONTEXT_MISMATCH)
}

impl CudaHtj2kEncodeResources {
    fn is_owned_by(&self, context: &CudaContext) -> bool {
        self.vlc_table0.is_owned_by(context)
            && self.vlc_table1.is_owned_by(context)
            && self.uvlc_table.is_owned_by(context)
    }
}

pub(super) fn validate_htj2k_encode_context<'a>(
    context: &CudaContext,
    coefficient_buffers: impl IntoIterator<Item = &'a CudaDeviceBuffer>,
    resources: Option<&CudaHtj2kEncodeResources>,
    pool: Option<&CudaBufferPool>,
) -> Result<(), CudaError> {
    let coefficients_match = coefficient_buffers
        .into_iter()
        .all(|buffer| buffer.is_owned_by(context));
    let resources_match = resources.is_none_or(|resources| resources.is_owned_by(context));
    let pool_matches = pool.is_none_or(|pool| pool.is_owned_by(context));
    validate_htj2k_encode_context_matches([coefficients_match, resources_match, pool_matches])
}

#[cfg(test)]
mod tests;
