// SPDX-License-Identifier: MIT OR Apache-2.0

//! Packed exact-i64 transform and prepared-packet ownership transitions.

use alloc::vec::Vec;

use super::super::allocation::{checked_add_bytes, checked_element_bytes, host_allocation_failed};
use super::super::tier1_allocation::{prepared_packets_ownership, prepared_subbands_ownership};
use super::super::{
    roi_subband_scale, I64SubbandEncodeSettings, NativeEncodePipelineError,
    NativeEncodePipelineResult, NativeEncodeSession, PreparedEncodeSubband,
    PreparedResolutionPacket, QuantStepSize, SubBandType,
};
use crate::j2c::fdwt::{PackedDwtGeometry, PackedSubbandRect, PackedSubbandView};

mod subband;
use subband::{prepare_packed_subband_i64, PackedSubbandRequest};
mod transform;
use transform::try_transform_component;
mod typed;
pub(super) use typed::{prepare_typed_component_planes_i64_packets, TypedPlanePacketRequest};

#[derive(Clone, Copy)]
pub(in crate::j2c::encode) struct I64ComponentPrepareRequest<'a, 'input> {
    pub(in crate::j2c::encode) component: u16,
    pub(in crate::j2c::encode) width: u32,
    pub(in crate::j2c::encode) height: u32,
    pub(in crate::j2c::encode) num_levels: u8,
    pub(in crate::j2c::encode) step_sizes: &'a [QuantStepSize],
    pub(in crate::j2c::encode) subband_settings: I64SubbandEncodeSettings<'a>,
    pub(in crate::j2c::encode) retained_base_bytes: usize,
    pub(in crate::j2c::encode) session: &'a NativeEncodeSession<'input>,
}

#[derive(Clone, Copy)]
struct BandRequest<'a> {
    rect: PackedSubbandRect,
    step_size: &'a QuantStepSize,
    sub_band_type: SubBandType,
    roi_scale: u32,
}

pub(in crate::j2c::encode) fn prepare_i64_component_packets(
    samples: Vec<i64>,
    request: I64ComponentPrepareRequest<'_, '_>,
) -> NativeEncodePipelineResult<Vec<PreparedResolutionPacket>> {
    let transform = try_transform_component(samples, &request)?;
    let packets = try_prepare_resolution_packets(
        &transform.samples,
        transform.geometry,
        transform.retained_source_bytes,
        &request,
    )?;
    drop(transform.line_scratch);
    drop(transform.samples);
    request.session.checked_phase(
        checked_add_bytes(
            request.retained_base_bytes,
            prepared_packets_ownership(&packets, packets.capacity())?.total()?,
            "exact i64 prepared component",
        )?,
        "exact i64 prepared component",
    )?;
    Ok(packets)
}

fn try_prepare_resolution_packets(
    samples: &[i64],
    geometry: PackedDwtGeometry,
    retained_source_bytes: usize,
    request: &I64ComponentPrepareRequest<'_, '_>,
) -> NativeEncodePipelineResult<Vec<PreparedResolutionPacket>> {
    let packet_count = usize::from(request.num_levels) + 1;
    let requested_packets = checked_element_bytes::<PreparedResolutionPacket>(
        packet_count,
        "exact i64 resolution packet owners",
    )?;
    request.session.checked_phase(
        checked_add_bytes(
            retained_source_bytes,
            requested_packets,
            "exact i64 resolution packet owners",
        )?,
        "exact i64 resolution packet owners",
    )?;
    let mut packets = Vec::new();
    packets.try_reserve_exact(packet_count).map_err(|_| {
        host_allocation_failed("exact i64 resolution packet owners", requested_packets)
    })?;
    check_component_peak(request.session, retained_source_bytes, &packets, 0)?;

    let ll = [BandRequest {
        rect: geometry.ll()?,
        step_size: request.step_sizes.first().ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant("reversible quantization step missing")
        })?,
        sub_band_type: SubBandType::LowLow,
        roi_scale: roi_subband_scale(request.num_levels, None)
            .map_err(NativeEncodePipelineError::internal_invariant)?,
    }];
    push_resolution_packet(
        &mut packets,
        &ResolutionPacketRequest {
            coefficients: samples,
            component: request.component,
            resolution: 0,
            bands: &ll,
            base_settings: request.subband_settings,
            retained_source_bytes,
            session: request.session,
        },
    )?;

    for resolution in 0..request.num_levels {
        let level = geometry.level(resolution)?;
        let step_base = 1 + usize::from(resolution) * 3;
        let roi_scale = roi_subband_scale(request.num_levels, Some(usize::from(resolution)))
            .map_err(NativeEncodePipelineError::internal_invariant)?;
        let bands = [
            BandRequest {
                rect: level.hl,
                step_size: request.step_sizes.get(step_base).ok_or_else(|| {
                    NativeEncodePipelineError::internal_invariant(
                        "reversible quantization step missing",
                    )
                })?,
                sub_band_type: SubBandType::HighLow,
                roi_scale,
            },
            BandRequest {
                rect: level.lh,
                step_size: request.step_sizes.get(step_base + 1).ok_or_else(|| {
                    NativeEncodePipelineError::internal_invariant(
                        "reversible quantization step missing",
                    )
                })?,
                sub_band_type: SubBandType::LowHigh,
                roi_scale,
            },
            BandRequest {
                rect: level.hh,
                step_size: request.step_sizes.get(step_base + 2).ok_or_else(|| {
                    NativeEncodePipelineError::internal_invariant(
                        "reversible quantization step missing",
                    )
                })?,
                sub_band_type: SubBandType::HighHigh,
                roi_scale,
            },
        ];
        push_resolution_packet(
            &mut packets,
            &ResolutionPacketRequest {
                coefficients: samples,
                component: request.component,
                resolution: u32::from(resolution) + 1,
                bands: &bands,
                base_settings: request.subband_settings,
                retained_source_bytes,
                session: request.session,
            },
        )?;
    }
    Ok(packets)
}

struct ResolutionPacketRequest<'a, 'input> {
    coefficients: &'a [i64],
    component: u16,
    resolution: u32,
    bands: &'a [BandRequest<'a>],
    base_settings: I64SubbandEncodeSettings<'a>,
    retained_source_bytes: usize,
    session: &'a NativeEncodeSession<'input>,
}

fn push_resolution_packet(
    packets: &mut Vec<PreparedResolutionPacket>,
    request: &ResolutionPacketRequest<'_, '_>,
) -> NativeEncodePipelineResult<()> {
    let retained_packet_bytes = prepared_packets_ownership(packets, packets.capacity())?.total()?;
    let requested_subbands = checked_element_bytes::<PreparedEncodeSubband>(
        request.bands.len(),
        "exact i64 prepared subband owners",
    )?;
    request.session.checked_phase(
        checked_add_bytes(
            request.retained_source_bytes,
            checked_add_bytes(
                retained_packet_bytes,
                requested_subbands,
                "exact i64 prepared subband owners",
            )?,
            "exact i64 prepared packet",
        )?,
        "exact i64 prepared packet",
    )?;
    let mut subbands = Vec::new();
    subbands
        .try_reserve_exact(request.bands.len())
        .map_err(|_| {
            host_allocation_failed("exact i64 prepared subband owners", requested_subbands)
        })?;
    for band in request.bands {
        let retained_subband_bytes =
            prepared_subbands_ownership(&subbands, subbands.capacity())?.total()?;
        let subband_base = checked_add_bytes(
            request.retained_source_bytes,
            checked_add_bytes(
                retained_packet_bytes,
                retained_subband_bytes,
                "exact i64 prepared packet",
            )?,
            "exact i64 prepared packet",
        )?;
        let view = PackedSubbandView::try_new(request.coefficients, band.rect)?;
        let prepared = prepare_packed_subband_i64(&PackedSubbandRequest {
            view,
            step_size: band.step_size,
            sub_band_type: band.sub_band_type,
            settings: I64SubbandEncodeSettings {
                roi_scale: band.roi_scale,
                ..request.base_settings
            },
            retained_base_bytes: subband_base,
            session: request.session,
        })?;
        let prepared_bytes =
            prepared_subbands_ownership(core::slice::from_ref(&prepared), 0)?.total()?;
        request.session.checked_phase(
            checked_add_bytes(subband_base, prepared_bytes, "exact i64 prepared subband")?,
            "exact i64 prepared subband",
        )?;
        subbands.push(prepared);
    }
    let packet = PreparedResolutionPacket {
        component: request.component,
        resolution: request.resolution,
        precinct: 0,
        subbands,
    };
    let packet_bytes = prepared_packets_ownership(core::slice::from_ref(&packet), 0)?.total()?;
    check_component_peak(
        request.session,
        request.retained_source_bytes,
        packets,
        packet_bytes,
    )?;
    packets.push(packet);
    Ok(())
}

fn check_component_peak(
    session: &NativeEncodeSession<'_>,
    retained_source_bytes: usize,
    packets: &Vec<PreparedResolutionPacket>,
    additional_packet_bytes: usize,
) -> NativeEncodePipelineResult<()> {
    let packet_bytes = prepared_packets_ownership(packets, packets.capacity())?.total()?;
    session.checked_phase(
        checked_add_bytes(
            retained_source_bytes,
            checked_add_bytes(
                packet_bytes,
                additional_packet_bytes,
                "exact i64 prepared packets",
            )?,
            "exact i64 prepared packets",
        )?,
        "exact i64 prepared packets",
    )?;
    Ok(())
}

#[cfg(test)]
#[path = "prepare/tests.rs"]
mod tests;
