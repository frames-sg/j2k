// SPDX-License-Identifier: MIT OR Apache-2.0

use super::multitile::{encode_multitile_impl, MultiTileEncodeRequest};
use super::{
    packet_encode, profile, write_single_tile_packetized_codestream_for_session, BlockCodingMode,
    EncodeComponentSampleInfo, EncodeOptions, EncodeRoiRegion, J2kEncodeStageAccelerator,
    NativeEncodePipelineResult, NativeEncodeSession, Vec,
};

mod accelerator;
mod coefficient_source;
mod finalize;
pub(in crate::j2c::encode) mod ownership;
mod plan;
mod precomputed;
mod resident;
mod reversible_i64;
#[cfg(test)]
mod tests;
mod tile_encode;

pub(super) use coefficient_source::{OwnedDwtComponent, PackedF32DwtComponent};

use accelerator::{
    prepare_accelerated_components, try_encode_complete_ht_tile, AcceleratedComponentRequest,
};
use finalize::{
    finalize_accelerated_codestream, finalize_staged_codestream, TransformStageTimings,
};
use ownership::{codestream_final_plan_retained_bytes, prepared_transforms_retained_bytes};
use plan::{
    build_single_tile_plan, validate_encode_request, CodestreamFinalPlan, ValidatedEncodeRoute,
    ValidatedSingleTileInput,
};
pub(super) use precomputed::{
    encode_precomputed_53_single_tile, encode_precomputed_97_single_tile,
};
pub(super) use resident::encode_resident_impl;
use reversible_i64::{encode_reversible_i64_single_tile_packets, ReversibleI64SingleTileRequest};
use tile_encode::{encode_tile_packets, EncodedTilePackets};

enum PreparedSingleTile {
    ReversibleI64 {
        packetized_tile: packet_encode::PacketizedTileData,
        final_plan: CodestreamFinalPlan,
    },
    Accelerated {
        packetized_tile: packet_encode::PacketizedTileData,
        final_plan: CodestreamFinalPlan,
        tile_body_us: u128,
    },
    Staged {
        encoded: EncodedTilePackets,
        final_plan: CodestreamFinalPlan,
        transform_timings: TransformStageTimings,
    },
}

impl PreparedSingleTile {
    fn into_packetized_tile(self) -> packet_encode::PacketizedTileData {
        match self {
            Self::ReversibleI64 {
                packetized_tile, ..
            }
            | Self::Accelerated {
                packetized_tile, ..
            } => packetized_tile,
            Self::Staged { encoded, .. } => encoded.packetized_tile,
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "this codec boundary keeps geometry, state buffers, and validated options explicit without allocation or indirection"
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
    session: &NativeEncodeSession<'_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<Vec<u8>> {
    let validated = validate_encode_request(
        pixels.len(),
        width,
        height,
        num_components,
        bit_depth,
        options,
        block_coding_mode,
        component_sample_info,
        session,
    )?;
    let validated = match validated {
        ValidatedEncodeRoute::MultiTile {
            tile_width,
            tile_height,
        } => {
            return encode_multitile_impl(
                &MultiTileEncodeRequest {
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
                    session,
                    tile_width,
                    tile_height,
                },
                accelerator,
            );
        }
        ValidatedEncodeRoute::SingleTile(validated) => validated,
    };

    let profile_enabled = profile::profile_stages_enabled();
    let total_start = profile::profile_now(profile_enabled);
    let prepared = prepare_validated_single_tile(
        validated,
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
        session,
        profile_enabled,
        accelerator,
    )?;
    finalize_prepared_single_tile(prepared, options, profile_enabled, total_start, session)
}

/// Encode one already-isolated image tile through Tier-2 without serializing a
/// temporary child codestream. Multi-tile orchestration consumes the returned
/// packet owners directly into the parent tile-part graph.
#[expect(
    clippy::too_many_arguments,
    reason = "this internal tile boundary keeps validated geometry and accelerator state explicit"
)]
pub(in crate::j2c::encode) fn encode_single_tile_packets_impl(
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
    session: &NativeEncodeSession<'_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<packet_encode::PacketizedTileData> {
    let validated = validate_encode_request(
        pixels.len(),
        width,
        height,
        num_components,
        bit_depth,
        options,
        block_coding_mode,
        component_sample_info,
        session,
    )?;
    let ValidatedEncodeRoute::SingleTile(validated) = validated else {
        return Err(crate::EncodeError::InternalInvariant {
            what: "isolated multi-tile child unexpectedly requested nested tiling",
        }
        .into());
    };
    let prepared = prepare_validated_single_tile(
        validated,
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
        session,
        profile::profile_stages_enabled(),
        accelerator,
    )?;
    Ok(prepared.into_packetized_tile())
}

#[expect(
    clippy::too_many_arguments,
    reason = "this internal tile boundary keeps validated geometry and accelerator state explicit"
)]
fn prepare_validated_single_tile(
    validated: ValidatedSingleTileInput,
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
    session: &NativeEncodeSession<'_>,
    profile_enabled: bool,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<PreparedSingleTile> {
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
        session,
    )?;

    if plan.high_bit_exact && options.reversible {
        let (packetized_tile, plan) =
            encode_reversible_i64_single_tile_packets(ReversibleI64SingleTileRequest {
                pixels,
                width,
                height,
                num_components,
                bit_depth,
                signed,
                options,
                plan,
                session,
                accelerator,
            })?;
        return Ok(PreparedSingleTile::ReversibleI64 {
            packetized_tile,
            final_plan: plan.into_codestream_final_plan(),
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
        session,
        accelerator,
    )? {
        return Ok(PreparedSingleTile::Accelerated {
            packetized_tile: packet_encode::PacketizedTileData {
                data: tile_data,
                packet_lengths: Vec::new(),
                packet_headers: Vec::new(),
            },
            final_plan: plan.into_codestream_final_plan(),
            tile_body_us,
        });
    }

    let prepared = prepare_accelerated_components(
        &AcceleratedComponentRequest {
            pixels,
            width,
            height,
            num_components,
            bit_depth,
            signed,
            options,
            plan: &plan,
            profile_enabled,
            session,
        },
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
        prepared_transforms_retained_bytes(&prepared)?,
        profile_enabled,
        session,
        accelerator,
    )?;
    let transform_timings = TransformStageTimings {
        deinterleave: prepared.deinterleave_us,
        mct: prepared.mct_us,
        dwt: prepared.dwt_us,
    };
    drop(prepared);
    Ok(PreparedSingleTile::Staged {
        encoded,
        final_plan: plan.into_codestream_final_plan(),
        transform_timings,
    })
}

fn finalize_prepared_single_tile(
    prepared: PreparedSingleTile,
    options: &EncodeOptions,
    profile_enabled: bool,
    total_start: Option<profile::ProfileInstant>,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<Vec<u8>> {
    match prepared {
        PreparedSingleTile::ReversibleI64 {
            packetized_tile,
            final_plan,
        } => {
            let final_plan_bytes = codestream_final_plan_retained_bytes(&final_plan)?;
            write_single_tile_packetized_codestream_for_session(
                &final_plan.params,
                &packetized_tile,
                &final_plan.quant_params,
                options.tile_part_packet_limit,
                final_plan_bytes,
                session,
            )
        }
        PreparedSingleTile::Accelerated {
            packetized_tile,
            final_plan,
            tile_body_us,
        } => finalize_accelerated_codestream(
            &final_plan,
            &packetized_tile.data,
            tile_body_us,
            profile_enabled,
            total_start,
            session,
        ),
        PreparedSingleTile::Staged {
            encoded,
            final_plan,
            transform_timings,
        } => finalize_staged_codestream(
            options,
            &final_plan,
            transform_timings,
            &encoded,
            profile_enabled,
            total_start,
            session,
        ),
    }
}
