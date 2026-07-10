// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    encode_multitile_impl, profile, BlockCodingMode, EncodeComponentSampleInfo, EncodeOptions,
    EncodeRoiRegion, J2kEncodeStageAccelerator, Vec,
};

mod accelerator;
mod finalize;
mod plan;
mod reversible_i64;
#[cfg(test)]
mod tests;
mod tile_encode;

use accelerator::{prepare_accelerated_components, try_encode_complete_ht_tile};
use finalize::{finalize_accelerated_codestream, finalize_staged_codestream};
use plan::{build_single_tile_plan, validate_encode_request, ValidatedEncodeRoute};
use reversible_i64::{
    encode_reversible_i64_single_tile_codestream, ReversibleI64SingleTileRequest,
};
use tile_encode::encode_tile_packets;

#[expect(
    clippy::too_many_arguments,
    reason = "this codec boundary keeps geometry, state buffers, and validated options explicit without allocation or indirection"
)]
#[expect(
    clippy::too_many_lines,
    reason = "the ordered JPEG 2000 state machine stays cohesive to preserve marker, packet, pass, and sample order"
)]
pub(super) fn encode_impl(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
    block_coding_mode: BlockCodingMode,
    roi_regions: &[EncodeRoiRegion],
    component_sample_info: &[EncodeComponentSampleInfo],
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    let validated = validate_encode_request(
        pixels.len(),
        width,
        height,
        num_components,
        bit_depth,
        options,
        block_coding_mode,
        component_sample_info,
    )?;
    let validated = match validated {
        ValidatedEncodeRoute::MultiTile {
            tile_width,
            tile_height,
        } => {
            return encode_multitile_impl(
                pixels,
                width,
                height,
                num_components,
                bit_depth,
                signed,
                options,
                block_coding_mode,
                roi_regions,
                component_sample_info,
                accelerator,
                tile_width,
                tile_height,
            );
        }
        ValidatedEncodeRoute::SingleTile(validated) => validated,
    };

    let profile_enabled = profile::profile_stages_enabled();
    let total_start = profile::profile_now(profile_enabled);
    let plan = build_single_tile_plan(
        validated,
        width,
        height,
        num_components,
        bit_depth,
        signed,
        options,
        block_coding_mode,
        roi_regions,
        component_sample_info,
    )?;

    if plan.high_bit_exact && options.reversible {
        return encode_reversible_i64_single_tile_codestream(ReversibleI64SingleTileRequest {
            pixels,
            width,
            height,
            num_pixels: plan.num_pixels,
            num_components,
            bit_depth,
            signed,
            options,
            params: &plan.params,
            quant_params: &plan.quant_params,
            step_sizes: &plan.step_sizes,
            roi_plans: &plan.roi_plans,
            use_mct: plan.use_mct,
            guard_bits: plan.guard_bits,
            num_levels: plan.num_levels,
            cb_width: plan.cb_width,
            cb_height: plan.cb_height,
            ht_target_coding_passes: plan.ht_target_coding_passes,
            accelerator,
        });
    }

    if let Some((tile_data, tile_body_us)) = try_encode_complete_ht_tile(
        pixels,
        width,
        height,
        num_components,
        bit_depth,
        signed,
        options,
        component_sample_info,
        roi_regions,
        &plan,
        profile_enabled,
        accelerator,
    )? {
        return Ok(finalize_accelerated_codestream(
            &plan,
            &tile_data,
            tile_body_us,
            profile_enabled,
            total_start,
        ));
    }

    let prepared = prepare_accelerated_components(
        pixels,
        width,
        height,
        num_components,
        bit_depth,
        signed,
        options,
        &plan,
        profile_enabled,
        accelerator,
    )?;
    let encoded = encode_tile_packets(
        width,
        height,
        num_components,
        bit_depth,
        options,
        component_sample_info,
        &plan,
        &prepared.decompositions,
        profile_enabled,
        accelerator,
    )?;
    finalize_staged_codestream(
        options,
        &plan,
        &prepared,
        &encoded,
        profile_enabled,
        total_start,
    )
}
