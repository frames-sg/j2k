// SPDX-License-Identifier: MIT OR Apache-2.0

//! Direct single-tile encoding from borrowed precomputed coefficient trees.

use super::super::{
    block_coding_mode, profile, EncodeComponentSampleInfo, EncodeOptions,
    J2kEncodeStageAccelerator, NativeEncodePipelineError, NativeEncodePipelineResult,
    NativeEncodeSession, PrecomputedHtj2k53Image, PrecomputedHtj2k97Image, Vec,
};
use super::finalize::finalize_precomputed_codestream;
use super::plan::{
    build_single_tile_plan, validate_non_pixel_single_tile_request, NonPixelSingleTileRequest,
};
use super::tile_encode::encode_tile_packets;

pub(in crate::j2c::encode) fn encode_precomputed_53_single_tile(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    component_sample_info: &[EncodeComponentSampleInfo],
    session: &NativeEncodeSession<'_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<Vec<u8>> {
    let num_components = u16::try_from(image.components.len()).map_err(|_| {
        NativeEncodePipelineError::unsupported("component count exceeds the JPEG 2000 Part 1 limit")
    })?;
    let coding_mode = block_coding_mode(options);
    let validated = validate_non_pixel_single_tile_request(&NonPixelSingleTileRequest {
        width: image.width,
        height: image.height,
        num_components,
        bit_depth: image.bit_depth,
        options,
        block_coding_mode: coding_mode,
        component_sample_info,
        multi_tile_error: "precomputed DWT encode requires a single whole-image tile",
        session,
    })?;
    let profile_enabled = profile::profile_stages_enabled();
    let total_start = profile::profile_now(profile_enabled);
    let plan = build_single_tile_plan(
        validated,
        image.width,
        image.height,
        num_components,
        image.bit_depth,
        image.signed,
        options,
        coding_mode,
        &[],
        component_sample_info,
        session,
    )?;
    let encoded = encode_tile_packets(
        image.width,
        image.height,
        num_components,
        image.bit_depth,
        options,
        component_sample_info,
        &plan,
        &image.components,
        0,
        profile_enabled,
        session,
        accelerator,
    )?;
    let final_plan = plan.into_codestream_final_plan();
    finalize_precomputed_codestream(
        options,
        &final_plan,
        &encoded,
        profile_enabled,
        total_start,
        session,
    )
}

pub(in crate::j2c::encode) fn encode_precomputed_97_single_tile(
    image: &PrecomputedHtj2k97Image,
    options: &EncodeOptions,
    session: &NativeEncodeSession<'_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<Vec<u8>> {
    let num_components = u16::try_from(image.components.len()).map_err(|_| {
        NativeEncodePipelineError::unsupported("component count exceeds the JPEG 2000 Part 1 limit")
    })?;
    let coding_mode = block_coding_mode(options);
    let validated = validate_non_pixel_single_tile_request(&NonPixelSingleTileRequest {
        width: image.width,
        height: image.height,
        num_components,
        bit_depth: image.bit_depth,
        options,
        block_coding_mode: coding_mode,
        component_sample_info: &[],
        multi_tile_error: "precomputed DWT encode requires a single whole-image tile",
        session,
    })?;
    let profile_enabled = profile::profile_stages_enabled();
    let total_start = profile::profile_now(profile_enabled);
    let plan = build_single_tile_plan(
        validated,
        image.width,
        image.height,
        num_components,
        image.bit_depth,
        image.signed,
        options,
        coding_mode,
        &[],
        &[],
        session,
    )?;
    let encoded = encode_tile_packets(
        image.width,
        image.height,
        num_components,
        image.bit_depth,
        options,
        &[],
        &plan,
        &image.components,
        0,
        profile_enabled,
        session,
        accelerator,
    )?;
    let final_plan = plan.into_codestream_final_plan();
    finalize_precomputed_codestream(
        options,
        &final_plan,
        &encoded,
        profile_enabled,
        total_start,
        session,
    )
}
