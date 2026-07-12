// SPDX-License-Identifier: MIT OR Apache-2.0

//! Retained-capacity accounting for reusable encode parameters.

use alloc::vec::Vec;

use super::add_capacity;
use crate::j2c::encode::{EncodeComponentSampleInfo, EncodeParams};
use crate::EncodeResult;

pub(in crate::j2c::encode) fn encode_params_retained_bytes(
    params: &EncodeParams,
) -> EncodeResult<usize> {
    let mut bytes = 0;
    bytes = add_capacity::<EncodeComponentSampleInfo>(
        bytes,
        params.component_sample_info.capacity(),
        "tile component metadata",
    )?;
    bytes = add_capacity::<Vec<(u16, u16)>>(
        bytes,
        params.component_quantization_step_sizes.capacity(),
        "tile component quantization vectors",
    )?;
    for steps in &params.component_quantization_step_sizes {
        bytes = add_capacity::<(u16, u16)>(bytes, steps.capacity(), "tile component quantization")?;
    }
    bytes = add_capacity::<(u8, u8)>(
        bytes,
        params.component_sampling.capacity(),
        "tile component sampling",
    )?;
    bytes = add_capacity::<u8>(
        bytes,
        params.roi_component_shifts.capacity(),
        "tile marker ROI shifts",
    )?;
    add_capacity::<(u8, u8)>(
        bytes,
        params.precinct_exponents.capacity(),
        "tile precinct exponents",
    )
}
