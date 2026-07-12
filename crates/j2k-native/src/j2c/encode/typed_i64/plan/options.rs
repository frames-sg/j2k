// SPDX-License-Identifier: MIT OR Apache-2.0

//! Accounted high-bit encode-option ownership.

use crate::j2c::encode::allocation::{checked_add_bytes, checked_element_bytes};
use crate::j2c::encode::multitile::{
    encode_options_retained_bytes, try_clone_options_with_component_sampling,
};
use crate::j2c::encode::{EncodeOptions, NativeEncodePipelineResult, NativeEncodeSession};

pub(in crate::j2c::encode::typed_i64) fn try_high_bit_options(
    options: &EncodeOptions,
    component_sampling: &[(u8, u8)],
    num_levels: u8,
    retained_base_bytes: usize,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<EncodeOptions> {
    let requested = requested_options_bytes(options, component_sampling.len())?;
    session.checked_phase(
        checked_add_bytes(retained_base_bytes, requested, "typed i64 encode options")?,
        "typed i64 encode options",
    )?;
    let mut high_bit_options =
        try_clone_options_with_component_sampling(options, Some(component_sampling))?;
    high_bit_options.num_decomposition_levels = num_levels;
    high_bit_options.reversible = true;
    high_bit_options.use_mct = false;
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            encode_options_retained_bytes(&high_bit_options)?,
            "typed i64 encode options",
        )?,
        "typed i64 encode options",
    )?;
    Ok(high_bit_options)
}

fn requested_options_bytes(
    options: &EncodeOptions,
    sampling_count: usize,
) -> NativeEncodePipelineResult<usize> {
    let mut bytes = checked_element_bytes::<u64>(
        options.quality_layer_byte_targets.len(),
        "typed i64 quality targets",
    )?;
    bytes = checked_add_bytes(
        bytes,
        checked_element_bytes::<(u8, u8)>(sampling_count, "typed i64 component sampling")?,
        "typed i64 encode options",
    )?;
    bytes = checked_add_bytes(
        bytes,
        checked_element_bytes::<u8>(options.roi_component_shifts.len(), "typed i64 ROI shifts")?,
        "typed i64 encode options",
    )?;
    Ok(checked_add_bytes(
        bytes,
        checked_element_bytes::<(u8, u8)>(
            options.precinct_exponents.len(),
            "typed i64 precinct exponents",
        )?,
        "typed i64 encode options",
    )?)
}
