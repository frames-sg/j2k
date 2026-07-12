// SPDX-License-Identifier: MIT OR Apache-2.0

//! Transform-stage orchestration and representation handoffs.

mod accelerated_dwt;
mod component_samples;
mod dwt53_output;
mod mct;
mod reversible;

#[expect(
    unused_imports,
    reason = "preserve sibling-visible transform helpers while the implementation is split"
)]
pub(super) use accelerated_dwt::{
    convert_forward_dwt53_output, convert_forward_dwt97_output, validate_dwt53_level,
    validate_dwt97_level,
};
pub(super) use accelerated_dwt::{
    encode_forward_dwt, forward_dwt53_output_retained_bytes, validate_band_len, ForwardDwtRequest,
};
pub(super) use component_samples::{
    try_component_plane_to_f32_for_session, validate_component_sample_info,
    validate_deinterleaved_components,
};
#[cfg(test)]
pub(super) use dwt53_output::forward_dwt53_output_from_decomposition;
pub(super) use dwt53_output::try_forward_dwt53_output_from_decomposition;
pub(super) use mct::{try_encode_forward_ict, try_encode_forward_rct};
pub(super) use reversible::{
    adjust_component_step_sizes_for_guard_delta, adjust_reversible_step_sizes_for_guard_delta,
    reversible_guard_bits_for_marker_limit,
};
