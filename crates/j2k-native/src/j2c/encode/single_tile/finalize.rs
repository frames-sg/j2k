// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    codestream_write, profile, write_single_tile_packetized_codestream_for_session, EncodeOptions,
    NativeEncodePipelineError, NativeEncodePipelineResult, NativeEncodeSession, Vec,
};
use super::ownership::codestream_final_plan_retained_bytes;
use super::plan::CodestreamFinalPlan;
use super::tile_encode::EncodedTilePackets;
use crate::j2c::encode::allocation::checked_add_bytes;

pub(super) fn finalize_accelerated_codestream(
    plan: &CodestreamFinalPlan,
    tile_data: &Vec<u8>,
    tile_body_us: u128,
    profile_enabled: bool,
    total_start: Option<profile::ProfileInstant>,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<Vec<u8>> {
    let retained_bytes = checked_add_bytes(
        codestream_final_plan_retained_bytes(plan)?,
        tile_data.capacity(),
        "accelerated codestream retained owners",
    )?;
    session.checked_phase(retained_bytes, "accelerated codestream retained owners")?;
    let stage_start = profile::profile_now(profile_enabled);
    let accounted = codestream_write::write_codestream_accounted_with_peak_check(
        &plan.params,
        tile_data,
        &plan.quant_params,
        |writer_peak_bytes| {
            session
                .checked_phase(
                    checked_add_bytes(
                        retained_bytes,
                        writer_peak_bytes,
                        "accelerated codestream writer peak",
                    )?,
                    "accelerated codestream writer peak",
                )
                .map(|_| ())
        },
    )?;
    if accounted.writer_peak_bytes != accounted.codestream.capacity() {
        return Err(NativeEncodePipelineError::internal_invariant(
            "accelerated codestream writer peak disagrees with output capacity",
        ));
    }
    let codestream = accounted.codestream;
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
    plan: &CodestreamFinalPlan,
    transform: TransformStageTimings,
    encoded: &EncodedTilePackets,
    profile_enabled: bool,
    total_start: Option<profile::ProfileInstant>,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<Vec<u8>> {
    finalize_staged_codestream_with_timings(
        options,
        plan,
        transform,
        encoded,
        profile_enabled,
        total_start,
        session,
    )
}

pub(super) fn finalize_precomputed_codestream(
    options: &EncodeOptions,
    plan: &CodestreamFinalPlan,
    encoded: &EncodedTilePackets,
    profile_enabled: bool,
    total_start: Option<profile::ProfileInstant>,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<Vec<u8>> {
    finalize_staged_codestream_with_timings(
        options,
        plan,
        TransformStageTimings::default(),
        encoded,
        profile_enabled,
        total_start,
        session,
    )
}

#[derive(Clone, Copy, Default)]
pub(super) struct TransformStageTimings {
    pub(super) deinterleave: u128,
    pub(super) mct: u128,
    pub(super) dwt: u128,
}

fn finalize_staged_codestream_with_timings(
    options: &EncodeOptions,
    plan: &CodestreamFinalPlan,
    transform: TransformStageTimings,
    encoded: &EncodedTilePackets,
    profile_enabled: bool,
    total_start: Option<profile::ProfileInstant>,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<Vec<u8>> {
    let stage_start = profile::profile_now(profile_enabled);
    let retained_phase_bytes = codestream_final_plan_retained_bytes(plan)?;
    let codestream = write_single_tile_packetized_codestream_for_session(
        &plan.params,
        &encoded.packetized_tile,
        &plan.quant_params,
        options.tile_part_packet_limit,
        retained_phase_bytes,
        session,
    )?;
    let codestream_us = profile::elapsed_us(stage_start);

    if profile_enabled {
        profile::emit_profile_row(
            "encode",
            "cpu",
            &[
                ("deinterleave_us", transform.deinterleave),
                ("mct_us", transform.mct),
                ("dwt_us", transform.dwt),
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
