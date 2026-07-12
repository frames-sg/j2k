// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible checked copies for multi-tile plan ownership transitions.

use alloc::vec::Vec;

use crate::j2c::encode::allocation::{checked_element_bytes, host_allocation_failed};
use crate::j2c::encode::{
    EncodeOptions, EncodeRoiRegion, NativeEncodePipelineError, NativeEncodePipelineResult,
};

pub(super) fn try_component_sampling(
    options: &EncodeOptions,
    num_components: u16,
) -> NativeEncodePipelineResult<Vec<(u8, u8)>> {
    if let Some(sampling) = &options.component_sampling {
        if sampling.len() != usize::from(num_components) {
            return Err(NativeEncodePipelineError::invalid_input(
                "component sampling count does not match component count",
            ));
        }
        return try_copy_slice(sampling, "multi-tile component sampling");
    }
    let count = usize::from(num_components);
    let bytes = checked_element_bytes::<(u8, u8)>(count, "multi-tile component sampling")?;
    let mut sampling = Vec::new();
    sampling
        .try_reserve_exact(count)
        .map_err(|_| host_allocation_failed("multi-tile component sampling", bytes))?;
    sampling.resize(count, (1, 1));
    Ok(sampling)
}

pub(super) fn try_roi_shifts(
    options: &EncodeOptions,
    regions: &[EncodeRoiRegion],
    num_components: u16,
) -> NativeEncodePipelineResult<Vec<u8>> {
    let count = usize::from(num_components);
    let bytes = checked_element_bytes::<u8>(count, "multi-tile ROI shifts")?;
    let mut shifts = Vec::new();
    shifts
        .try_reserve_exact(count)
        .map_err(|_| host_allocation_failed("multi-tile ROI shifts", bytes))?;
    shifts.resize(count, 0);
    if !options.roi_component_shifts.is_empty() {
        if options.roi_component_shifts.len() != count {
            return Err(NativeEncodePipelineError::invalid_input(
                "ROI component shift count does not match component count",
            ));
        }
        shifts.copy_from_slice(&options.roi_component_shifts);
    }
    for region in regions {
        *shifts.get_mut(usize::from(region.component)).ok_or(
            NativeEncodePipelineError::invalid_input("ROI region component index out of range"),
        )? = region.shift;
    }
    Ok(shifts)
}

pub(in crate::j2c::encode) fn try_clone_options(
    options: &EncodeOptions,
) -> NativeEncodePipelineResult<EncodeOptions> {
    try_clone_options_with_component_sampling(options, options.component_sampling.as_deref())
}

pub(in crate::j2c::encode) fn try_clone_options_with_component_sampling(
    options: &EncodeOptions,
    component_sampling: Option<&[(u8, u8)]>,
) -> NativeEncodePipelineResult<EncodeOptions> {
    Ok(EncodeOptions {
        num_decomposition_levels: options.num_decomposition_levels,
        reversible: options.reversible,
        code_block_width_exp: options.code_block_width_exp,
        code_block_height_exp: options.code_block_height_exp,
        guard_bits: options.guard_bits,
        use_ht_block_coding: options.use_ht_block_coding,
        progression_order: options.progression_order,
        write_tlm: options.write_tlm,
        write_plt: options.write_plt,
        write_plm: options.write_plm,
        write_ppm: options.write_ppm,
        write_ppt: options.write_ppt,
        write_sop: options.write_sop,
        write_eph: options.write_eph,
        use_mct: options.use_mct,
        num_layers: options.num_layers,
        quality_layer_byte_targets: try_copy_slice(
            &options.quality_layer_byte_targets,
            "multi-tile quality targets",
        )?,
        validate_high_throughput_codestream: options.validate_high_throughput_codestream,
        irreversible_quantization_scale: options.irreversible_quantization_scale,
        irreversible_quantization_subband_scales: options.irreversible_quantization_subband_scales,
        component_sampling: component_sampling
            .map(|sampling| try_copy_slice(sampling, "multi-tile component sampling"))
            .transpose()?,
        roi_component_shifts: try_copy_slice(
            &options.roi_component_shifts,
            "multi-tile ROI shifts",
        )?,
        tile_size: options.tile_size,
        tile_part_packet_limit: options.tile_part_packet_limit,
        precinct_exponents: try_copy_slice(
            &options.precinct_exponents,
            "multi-tile precinct exponents",
        )?,
    })
}

pub(in crate::j2c::encode) fn try_copy_slice<T: Copy>(
    values: &[T],
    what: &'static str,
) -> NativeEncodePipelineResult<Vec<T>> {
    let bytes = checked_element_bytes::<T>(values.len(), what)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(values.len())
        .map_err(|_| host_allocation_failed(what, bytes))?;
    output.extend_from_slice(values);
    Ok(output)
}
