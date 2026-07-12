// SPDX-License-Identifier: MIT OR Apache-2.0

mod buffer_access;
mod cleanup_dequant;
mod component;
mod error;
mod idwt;
mod routing;
mod surface;

pub(super) use self::buffer_access::pooled_cuda_buffer;
pub(super) use self::cleanup_dequant::run_component_cleanup_dequant_batches;
#[cfg(test)]
pub(super) use self::cleanup_dequant::split_htj2k_subband_decode_dispatches;
#[cfg(test)]
pub(super) use self::cleanup_dequant::{
    htj2k_batched_cleanup_dequant_dispatches, htj2k_batched_cleanup_dispatches,
    htj2k_batched_dequant_dispatches,
};
#[cfg(test)]
pub(super) use self::component::cuda_code_block_job_from_plan_block;
pub(super) use self::component::{
    decode_cuda_component_subbands_with_resources, finish_cuda_component_decode,
};
pub(super) use self::idwt::{
    can_batch_color_idwt, run_color_component_idwt_batches, run_cuda_component_idwt_steps,
};
pub(super) use self::routing::{
    decode_batch_to_cuda_resident_surface_with_profile_control,
    decode_region_scaled_to_cuda_resident_surface_impl,
    decode_region_to_cuda_resident_surface_impl, decode_scaled_to_cuda_resident_surface_impl,
    decode_to_cuda_resident_surface_impl, decode_to_cuda_resident_surface_with_profile_impl,
};
