// SPDX-License-Identifier: MIT OR Apache-2.0

//! Requested and allocator-returned capacity accounting for multi-tile plans.

use alloc::vec::Vec;

use crate::j2c::encode::allocation::{checked_add_bytes, checked_element_bytes};
use crate::j2c::encode::{
    EncodeComponentSampleInfo, EncodeOptions, NativeEncodePipelineResult, QuantStepSize,
};

pub(super) fn requested_step_graph_bytes(
    num_levels: u8,
    component_count: usize,
) -> NativeEncodePipelineResult<usize> {
    let step_count = usize::from(num_levels) * 3 + 1;
    let one = checked_element_bytes::<QuantStepSize>(step_count, "multi-tile step sizes")?;
    let outer = checked_element_bytes::<Vec<QuantStepSize>>(
        component_count,
        "multi-tile component step owners",
    )?;
    Ok(checked_add_bytes(
        one,
        checked_add_bytes(
            outer,
            one.checked_mul(component_count)
                .ok_or(crate::EncodeError::ArithmeticOverflow {
                    what: "multi-tile component step sizes",
                })?,
            "multi-tile component step sizes",
        )?,
        "multi-tile step graph",
    )?)
}

pub(super) fn step_graph_retained_bytes(
    steps: &Vec<QuantStepSize>,
    component_steps: &Vec<Vec<QuantStepSize>>,
) -> NativeEncodePipelineResult<usize> {
    let mut bytes = checked_element_bytes::<QuantStepSize>(steps.capacity(), "multi-tile steps")?;
    bytes = checked_add_bytes(
        bytes,
        checked_element_bytes::<Vec<QuantStepSize>>(
            component_steps.capacity(),
            "multi-tile component step owners",
        )?,
        "multi-tile step graph",
    )?;
    for component in component_steps {
        bytes = checked_add_bytes(
            bytes,
            checked_element_bytes::<QuantStepSize>(
                component.capacity(),
                "multi-tile component steps",
            )?,
            "multi-tile component steps",
        )?;
    }
    Ok(bytes)
}

pub(super) fn requested_options_clone_bytes(
    options: &EncodeOptions,
) -> NativeEncodePipelineResult<usize> {
    let mut bytes = checked_element_bytes::<u64>(
        options.quality_layer_byte_targets.len(),
        "multi-tile quality targets",
    )?;
    if let Some(sampling) = &options.component_sampling {
        bytes = checked_add_bytes(
            bytes,
            checked_element_bytes::<(u8, u8)>(sampling.len(), "multi-tile component sampling")?,
            "multi-tile child options",
        )?;
    }
    bytes = checked_add_bytes(
        bytes,
        checked_element_bytes::<u8>(options.roi_component_shifts.len(), "multi-tile ROI shifts")?,
        "multi-tile child options",
    )?;
    Ok(checked_add_bytes(
        bytes,
        checked_element_bytes::<(u8, u8)>(
            options.precinct_exponents.len(),
            "multi-tile precinct exponents",
        )?,
        "multi-tile child options",
    )?)
}

pub(super) fn final_plan_requested_bytes(
    num_levels: u8,
    num_components: u16,
    component_metadata_count: usize,
    precinct_count: usize,
) -> NativeEncodePipelineResult<usize> {
    let component_count = component_metadata_count;
    let step_graph = requested_step_graph_bytes(num_levels, component_count)?;
    let step_count = usize::from(num_levels) * 3 + 1;
    let quant_bytes = checked_element_bytes::<(u16, u16)>(step_count, "multi-tile quantization")?;
    let component_quant_outer = checked_element_bytes::<Vec<(u16, u16)>>(
        component_count,
        "multi-tile component quantization owners",
    )?;
    let component_quant_inner =
        quant_bytes
            .checked_mul(component_count)
            .ok_or(crate::EncodeError::ArithmeticOverflow {
                what: "multi-tile component quantization",
            })?;
    let mut bytes = checked_add_bytes(step_graph, quant_bytes, "multi-tile final plan")?;
    bytes = checked_add_bytes(bytes, component_quant_outer, "multi-tile final plan")?;
    bytes = checked_add_bytes(bytes, component_quant_inner, "multi-tile final plan")?;
    bytes = checked_add_bytes(
        bytes,
        checked_element_bytes::<EncodeComponentSampleInfo>(
            component_metadata_count,
            "multi-tile component metadata",
        )?,
        "multi-tile final plan",
    )?;
    bytes = checked_add_bytes(
        bytes,
        checked_element_bytes::<(u8, u8)>(
            usize::from(num_components),
            "multi-tile component sampling",
        )?,
        "multi-tile final plan",
    )?;
    bytes = checked_add_bytes(
        bytes,
        checked_element_bytes::<u8>(usize::from(num_components), "multi-tile ROI shifts")?,
        "multi-tile final plan",
    )?;
    Ok(checked_add_bytes(
        bytes,
        checked_element_bytes::<(u8, u8)>(precinct_count, "multi-tile precinct exponents")?,
        "multi-tile final plan",
    )?)
}
