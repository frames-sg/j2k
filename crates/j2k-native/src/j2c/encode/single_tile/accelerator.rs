// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    deinterleave_to_f32, encode_forward_dwt, forward_mct, profile,
    public_packetization_progression_order, try_encode_forward_ict, try_encode_forward_rct,
    validate_component_sampling_dwt_geometry, validate_deinterleaved_components, BlockCodingMode,
    DwtDecomposition, EncodeComponentSampleInfo, EncodeOptions, EncodeRoiRegion,
    J2kDeinterleaveToF32Job, J2kEncodeStageAccelerator, J2kHtj2kTileEncodeJob, Vec,
};
use super::plan::SingleTilePlan;

pub(super) struct PreparedComponentTransforms {
    pub(super) decompositions: Vec<DwtDecomposition>,
    pub(super) deinterleave_us: u128,
    pub(super) mct_us: u128,
    pub(super) dwt_us: u128,
}

pub(super) fn try_encode_complete_ht_tile(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
    component_sample_info: &[EncodeComponentSampleInfo],
    roi_regions: &[EncodeRoiRegion],
    plan: &SingleTilePlan,
    profile_enabled: bool,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Option<(Vec<u8>, u128)>, &'static str> {
    let stage_start = profile::profile_now(profile_enabled);
    if plan.params.block_coding_mode == BlockCodingMode::HighThroughput
        && component_sample_info.is_empty()
        && plan.roi_component_shifts.iter().all(|shift| *shift == 0)
        && roi_regions.is_empty()
        && !(plan.params.write_plt
            || plan.params.write_plm
            || plan.params.write_sop
            || plan.params.write_eph
            || options.tile_part_packet_limit.is_some())
    {
        if let Some(tile_data) = accelerator.encode_htj2k_tile(J2kHtj2kTileEncodeJob {
            pixels,
            width,
            height,
            num_components,
            bit_depth,
            signed,
            num_decomposition_levels: plan.num_levels,
            reversible: options.reversible,
            use_mct: plan.use_mct,
            guard_bits: plan.guard_bits,
            code_block_width: plan.cb_width,
            code_block_height: plan.cb_height,
            progression_order: public_packetization_progression_order(options.progression_order),
            component_sampling: &plan.params.component_sampling,
            quantization_steps: &plan.quant_params,
        })? {
            return Ok(Some((tile_data, profile::elapsed_us(stage_start))));
        }
    }

    Ok(None)
}

pub(super) fn prepare_accelerated_components(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
    plan: &SingleTilePlan,
    profile_enabled: bool,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<PreparedComponentTransforms, &'static str> {
    let stage_start = profile::profile_now(profile_enabled);
    let mut components = match accelerator.encode_deinterleave(J2kDeinterleaveToF32Job {
        pixels,
        num_pixels: plan.num_pixels,
        num_components,
        bit_depth,
        signed,
    })? {
        Some(components) => {
            validate_deinterleaved_components(components, num_components, plan.num_pixels)?
        }
        None => deinterleave_to_f32(pixels, plan.num_pixels, num_components, bit_depth, signed),
    };
    let deinterleave_us = profile::elapsed_us(stage_start);

    let stage_start = profile::profile_now(profile_enabled);
    if plan.use_mct {
        if options.reversible {
            if !try_encode_forward_rct(&mut components, accelerator)? {
                forward_mct::forward_rct(&mut components);
            }
        } else if !try_encode_forward_ict(&mut components, accelerator)? {
            forward_mct::forward_ict(&mut components);
        }
    }
    let mct_us = profile::elapsed_us(stage_start);

    let stage_start = profile::profile_now(profile_enabled);
    let decompositions = components
        .iter()
        .map(|component| {
            encode_forward_dwt(
                component,
                width,
                height,
                plan.num_levels,
                options.reversible,
                accelerator,
            )
        })
        .collect::<Result<Vec<_>, &'static str>>()?;
    validate_component_sampling_dwt_geometry(
        &decompositions,
        width,
        height,
        &plan.params.component_sampling,
    )?;
    let dwt_us = profile::elapsed_us(stage_start);

    Ok(PreparedComponentTransforms {
        decompositions,
        deinterleave_us,
        mct_us,
        dwt_us,
    })
}
