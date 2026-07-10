// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    codestream_write, profile, write_single_tile_packetized_codestream, EncodeOptions, Vec,
};
use super::accelerator::PreparedComponentTransforms;
use super::plan::SingleTilePlan;
use super::tile_encode::EncodedTilePackets;

pub(super) fn finalize_accelerated_codestream(
    plan: &SingleTilePlan,
    tile_data: &[u8],
    tile_body_us: u128,
    profile_enabled: bool,
    total_start: Option<profile::ProfileInstant>,
) -> Result<Vec<u8>, &'static str> {
    let stage_start = profile::profile_now(profile_enabled);
    let codestream =
        codestream_write::write_codestream(&plan.params, tile_data, &plan.quant_params)?;
    let codestream_us = profile::elapsed_us(stage_start);
    if profile_enabled {
        profile::emit_profile_row(
            "encode",
            "accelerated",
            &[
                ("tile_body_us", tile_body_us),
                ("codestream_us", codestream_us),
                ("total_us", profile::elapsed_us(total_start)),
            ],
        );
    }
    Ok(codestream)
}

pub(super) fn finalize_staged_codestream(
    options: &EncodeOptions,
    plan: &SingleTilePlan,
    prepared: &PreparedComponentTransforms,
    encoded: &EncodedTilePackets,
    profile_enabled: bool,
    total_start: Option<profile::ProfileInstant>,
) -> Result<Vec<u8>, &'static str> {
    let stage_start = profile::profile_now(profile_enabled);
    let codestream = write_single_tile_packetized_codestream(
        &plan.params,
        &encoded.packetized_tile,
        &plan.quant_params,
        options.tile_part_packet_limit,
    )?;
    let codestream_us = profile::elapsed_us(stage_start);

    if profile_enabled {
        profile::emit_profile_row(
            "encode",
            "cpu",
            &[
                ("deinterleave_us", prepared.deinterleave_us),
                ("mct_us", prepared.mct_us),
                ("dwt_us", prepared.dwt_us),
                ("subband_prepare_us", encoded.subband_prepare_us),
                ("block_encode_us", encoded.block_encode_us),
                ("packetize_us", encoded.packetize_us),
                ("codestream_us", codestream_us),
                ("total_us", profile::elapsed_us(total_start)),
            ],
        );
    }

    Ok(codestream)
}
