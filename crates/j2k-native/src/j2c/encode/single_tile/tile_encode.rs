// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::allocation::checked_add_bytes;
use super::super::{
    encode_component_resolution_packets, packet_encode, EncodeComponentSampleInfo, EncodeOptions,
    J2kEncodeStageAccelerator, NativeEncodePipelineResult, NativeEncodeSession, TilePacketRequest,
};
use super::coefficient_source::DwtComponentSource;
use super::ownership::single_tile_plan_retained_bytes;
use super::plan::SingleTilePlan;

mod components;
mod dwt_band;
use components::{prepare_component_packets, ComponentPacketRequest};

pub(super) struct EncodedTilePackets {
    pub(super) packetized_tile: packet_encode::PacketizedTileData,
    pub(super) subband_prepare_us: u128,
    pub(super) block_encode_us: u128,
    pub(super) packetize_us: u128,
}

#[expect(
    clippy::too_many_arguments,
    reason = "this codec boundary keeps geometry, state buffers, and validated options explicit without allocation or indirection"
)]
pub(super) fn encode_tile_packets<S: DwtComponentSource>(
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    options: &EncodeOptions,
    component_sample_info: &[EncodeComponentSampleInfo],
    plan: &SingleTilePlan,
    decompositions: impl ExactSizeIterator<Item = S>,
    source_retained_bytes: usize,
    source_owner_bytes: usize,
    profile_enabled: bool,
    session: &NativeEncodeSession<'_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<EncodedTilePackets> {
    let retained_base_bytes = checked_add_bytes(
        single_tile_plan_retained_bytes(plan)?,
        source_retained_bytes,
        "retained transform and single-tile plan owners",
    )?;
    let (component_resolution_packets, subband_prepare_us, released_source_bytes) =
        prepare_component_packets(
            &ComponentPacketRequest {
                num_components,
                bit_depth,
                options,
                component_sample_info,
                plan,
                retained_base_bytes,
                profile_enabled,
                session,
            },
            decompositions,
            accelerator,
        )?;
    // Later phases no longer hold the sources the preparation dropped, nor
    // the vector that owned them.
    let released_source_bytes = checked_add_bytes(
        released_source_bytes,
        source_owner_bytes,
        "released DWT sources and their owner",
    )?;
    let retained_base_bytes = retained_base_bytes
        .checked_sub(released_source_bytes)
        .ok_or(crate::EncodeError::InternalInvariant {
            what: "released DWT sources exceed the retained baseline",
        })?;
    let encoded = encode_component_resolution_packets(
        component_resolution_packets,
        TilePacketRequest {
            tile_bounds: super::super::tile_grid::tile_bounds(width, height, plan.tile_origin)?,
            num_levels: plan.num_levels,
            params: &plan.params,
            options,
            retained_base_bytes,
            phase_owners: plan,
            retained_what: "retained transform and single-tile plan owners",
            profile_enabled,
            session,
            accelerator,
        },
    )?;
    Ok(EncodedTilePackets {
        packetized_tile: encoded.packetized_tile,
        subband_prepare_us,
        block_encode_us: encoded.block_encode_us,
        packetize_us: encoded.packetize_us,
    })
}
