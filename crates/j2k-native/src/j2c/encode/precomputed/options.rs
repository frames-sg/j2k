// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible option ownership for direct precomputed coefficient encoders.

use super::allocation::ConstructionTracker;
use super::{EncodeOptions, NativeEncodePipelineError, NativeEncodePipelineResult};

#[derive(Clone, Copy)]
pub(super) struct PrecomputedOptionMode {
    pub(super) num_levels: u8,
    pub(super) reversible: bool,
    pub(super) use_ht_block_coding: bool,
    pub(super) use_mct: bool,
}

pub(super) fn validate_single_layer_packet_input(
    options: &EncodeOptions,
    unsupported_what: &'static str,
) -> NativeEncodePipelineResult<()> {
    if options.num_layers == 0 {
        return Err(NativeEncodePipelineError::invalid_input(
            "quality layer count must be non-zero",
        ));
    }
    if !options.quality_layer_byte_targets.is_empty()
        && options.quality_layer_byte_targets.len() != usize::from(options.num_layers)
    {
        return Err(NativeEncodePipelineError::invalid_input(
            "quality layer byte target count must match quality layer count",
        ));
    }
    if options.num_layers != 1 || !options.quality_layer_byte_targets.is_empty() {
        return Err(NativeEncodePipelineError::unsupported(unsupported_what));
    }
    Ok(())
}

pub(super) fn try_precomputed_options(
    options: &EncodeOptions,
    sampling: impl ExactSizeIterator<Item = (u8, u8)>,
    mode: PrecomputedOptionMode,
    tracker: &mut ConstructionTracker<'_, '_>,
) -> NativeEncodePipelineResult<EncodeOptions> {
    let sampling_count = sampling.len();
    let quality_layer_byte_targets = tracker.try_copy_slice(
        &options.quality_layer_byte_targets,
        "precomputed quality-layer targets",
    )?;
    let mut component_sampling =
        tracker.try_vec::<(u8, u8)>(sampling_count, "precomputed component sampling")?;
    component_sampling.extend(sampling);
    let roi_component_shifts =
        tracker.try_copy_slice(&options.roi_component_shifts, "precomputed ROI shifts")?;
    let precinct_exponents = tracker.try_copy_slice(
        &options.precinct_exponents,
        "precomputed precinct exponents",
    )?;
    Ok(EncodeOptions {
        num_decomposition_levels: mode.num_levels,
        reversible: mode.reversible,
        code_block_width_exp: options.code_block_width_exp,
        code_block_height_exp: options.code_block_height_exp,
        guard_bits: options.guard_bits,
        use_ht_block_coding: mode.use_ht_block_coding,
        progression_order: options.progression_order,
        write_tlm: options.write_tlm,
        write_plt: options.write_plt,
        write_plm: options.write_plm,
        write_ppm: options.write_ppm,
        write_ppt: options.write_ppt,
        write_sop: options.write_sop,
        write_eph: options.write_eph,
        use_mct: mode.use_mct,
        num_layers: options.num_layers,
        quality_layer_byte_targets,
        validate_high_throughput_codestream: false,
        irreversible_quantization_scale: options.irreversible_quantization_scale,
        irreversible_quantization_subband_scales: options.irreversible_quantization_subband_scales,
        component_sampling: Some(component_sampling),
        roi_component_shifts,
        tile_size: options.tile_size,
        tile_part_packet_limit: options.tile_part_packet_limit,
        precinct_exponents,
    })
}
