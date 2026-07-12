// SPDX-License-Identifier: MIT OR Apache-2.0

//! Move-only precinct partitioning for prepared Tier-1 owners.

use alloc::vec::Vec;

use super::super::{
    NativeEncodePipelineError, NativeEncodePipelineResult, NativeEncodeSession,
    PreparedEncodeSubband, PreparedResolutionPacket,
};

mod geometry;
use geometry::{
    component_split_packet_count, precinct_exponents_for_resolution, resolution_precinct_grid,
};
mod distribution;
use distribution::distribute_owned_subband;
mod ownership;
use ownership::{try_destination_vec, try_push_planned, PrecinctSplitAccounting};

pub(in crate::j2c::encode) fn split_component_resolution_packets_by_precinct_for_session(
    component_resolution_packets: Vec<Vec<PreparedResolutionPacket>>,
    image_width: u32,
    image_height: u32,
    num_decomposition_levels: u8,
    precinct_exponents: &[(u8, u8)],
    session: &NativeEncodeSession<'_>,
    retained_phase_bytes: usize,
) -> NativeEncodePipelineResult<Vec<Vec<PreparedResolutionPacket>>> {
    let (packets, _peak_phase_bytes) = split_component_resolution_packets_by_precinct_accounted(
        component_resolution_packets,
        image_width,
        image_height,
        num_decomposition_levels,
        precinct_exponents,
        session,
        retained_phase_bytes,
    )?;
    Ok(packets)
}

fn split_component_resolution_packets_by_precinct_accounted(
    component_resolution_packets: Vec<Vec<PreparedResolutionPacket>>,
    image_width: u32,
    image_height: u32,
    num_decomposition_levels: u8,
    precinct_exponents: &[(u8, u8)],
    session: &NativeEncodeSession<'_>,
    retained_phase_bytes: usize,
) -> NativeEncodePipelineResult<(Vec<Vec<PreparedResolutionPacket>>, usize)> {
    let source_capacity = component_resolution_packets.capacity();
    let mut accounting = PrecinctSplitAccounting::try_from_source(
        session,
        retained_phase_bytes,
        &component_resolution_packets,
        source_capacity,
    )?;
    if precinct_exponents.is_empty() {
        return Ok((component_resolution_packets, accounting.peak_phase_bytes()));
    }

    let component_count = component_resolution_packets.len();
    let mut split_components = try_destination_vec(
        component_count,
        &mut accounting,
        "split component packet owners",
    )?;
    for component_packets in component_resolution_packets {
        let component_capacity = component_packets.capacity();
        let split_packet_count = component_split_packet_count(
            &component_packets,
            image_width,
            image_height,
            num_decomposition_levels,
            precinct_exponents,
        )?;
        let mut split_packets = try_destination_vec(
            split_packet_count,
            &mut accounting,
            "split resolution packet owners",
        )?;
        for packet in component_packets {
            split_prepared_resolution_packet_by_precinct(
                packet,
                image_width,
                image_height,
                num_decomposition_levels,
                precinct_exponents,
                &mut split_packets,
                &mut accounting,
            )?;
        }
        accounting.release_source_capacity::<PreparedResolutionPacket>(
            component_capacity,
            "source resolution packet owners",
        )?;
        try_push_planned(&mut split_components, split_packets)?;
    }
    accounting.release_source_capacity::<Vec<PreparedResolutionPacket>>(
        source_capacity,
        "source component packet owners",
    )?;
    let peak_phase_bytes = accounting.finish(&split_components, split_components.capacity())?;
    Ok((split_components, peak_phase_bytes))
}

fn split_prepared_resolution_packet_by_precinct(
    packet: PreparedResolutionPacket,
    image_width: u32,
    image_height: u32,
    num_decomposition_levels: u8,
    precinct_exponents: &[(u8, u8)],
    split_packets: &mut Vec<PreparedResolutionPacket>,
    accounting: &mut PrecinctSplitAccounting<'_, '_>,
) -> NativeEncodePipelineResult<()> {
    let PreparedResolutionPacket {
        component,
        resolution,
        precinct: _,
        subbands,
    } = packet;
    let (horizontal_exponent, vertical_exponent) =
        precinct_exponents_for_resolution(precinct_exponents, resolution)?;
    let grid = resolution_precinct_grid(
        image_width,
        image_height,
        num_decomposition_levels,
        resolution,
        horizontal_exponent,
        vertical_exponent,
    )?;
    let packet_start = split_packets.len();
    let subband_count = subbands.len();
    for precinct_row in 0..grid.rows {
        for precinct_column in 0..grid.columns {
            let precinct = u64::from(precinct_row)
                .checked_mul(u64::from(grid.columns))
                .and_then(|value| value.checked_add(u64::from(precinct_column)))
                .ok_or_else(|| {
                    NativeEncodePipelineError::arithmetic_overflow("precinct index overflow")
                })?;
            let split_subbands =
                try_destination_vec(subband_count, accounting, "split prepared subband owners")?;
            try_push_planned(
                split_packets,
                PreparedResolutionPacket {
                    component,
                    resolution,
                    precinct,
                    subbands: split_subbands,
                },
            )?;
        }
    }

    let source_subband_capacity = subbands.capacity();
    let packet_end = split_packets.len();
    let packet_slice = split_packets
        .get_mut(packet_start..packet_end)
        .ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant("precinct packet output range mismatch")
        })?;
    for subband in subbands {
        distribute_owned_subband(
            subband,
            resolution,
            horizontal_exponent,
            vertical_exponent,
            grid,
            packet_slice,
            accounting,
        )?;
    }
    accounting.release_source_capacity::<PreparedEncodeSubband>(
        source_subband_capacity,
        "source prepared subband owners",
    )?;
    Ok(())
}

#[cfg(test)]
mod tests;
