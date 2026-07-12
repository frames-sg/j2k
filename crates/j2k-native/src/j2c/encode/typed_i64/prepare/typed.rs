// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed component-plane conversion and packet-tree ownership.

use alloc::vec::Vec;

use super::{prepare_i64_component_packets, I64ComponentPrepareRequest};
use crate::j2c::encode::allocation::{
    checked_add_bytes, checked_element_bytes, host_allocation_failed,
};
use crate::j2c::encode::tier1_allocation::{
    prepared_packet_tree_ownership, prepared_packets_ownership,
};
use crate::j2c::encode::{
    raw_pixel_bytes_per_sample, read_le_sample_value, sign_extend_sample,
    EncodeTypedComponentPlane, I64SubbandEncodeSettings, NativeEncodePipelineError,
    NativeEncodePipelineResult, NativeEncodeSession, PreparedResolutionPacket, QuantStepSize,
};

#[derive(Clone, Copy)]
pub(in crate::j2c::encode::typed_i64) struct TypedPlanePacketRequest<'a, 'input> {
    pub(in crate::j2c::encode::typed_i64) component_dimensions: &'a [(u32, u32)],
    pub(in crate::j2c::encode::typed_i64) component_step_sizes: &'a [Vec<QuantStepSize>],
    pub(in crate::j2c::encode::typed_i64) num_levels: u8,
    pub(in crate::j2c::encode::typed_i64) subband_settings: I64SubbandEncodeSettings<'a>,
    pub(in crate::j2c::encode::typed_i64) retained_base_bytes: usize,
    pub(in crate::j2c::encode::typed_i64) session: &'a NativeEncodeSession<'input>,
}

pub(in crate::j2c::encode::typed_i64) fn prepare_typed_component_planes_i64_packets(
    planes: &[EncodeTypedComponentPlane<'_>],
    request: TypedPlanePacketRequest<'_, '_>,
) -> NativeEncodePipelineResult<Vec<Vec<PreparedResolutionPacket>>> {
    validate_component_counts(planes, request)?;
    let requested_outer = checked_element_bytes::<Vec<PreparedResolutionPacket>>(
        planes.len(),
        "typed i64 prepared component owners",
    )?;
    request.session.checked_phase(
        checked_add_bytes(
            request.retained_base_bytes,
            requested_outer,
            "typed i64 prepared component owners",
        )?,
        "typed i64 prepared component owners",
    )?;
    let mut component_packets = Vec::new();
    component_packets
        .try_reserve_exact(planes.len())
        .map_err(|_| {
            host_allocation_failed("typed i64 prepared component owners", requested_outer)
        })?;
    check_packet_tree(
        request.session,
        request.retained_base_bytes,
        &component_packets,
    )?;

    for (component_idx, (plane, &(width, height))) in
        planes.iter().zip(request.component_dimensions).enumerate()
    {
        let retained_tree_bytes =
            prepared_packet_tree_ownership(&component_packets, component_packets.capacity())?
                .total()?;
        let component_base = checked_add_bytes(
            request.retained_base_bytes,
            retained_tree_bytes,
            "typed i64 prepared components",
        )?;
        let samples = try_typed_component_plane_to_i64(
            plane,
            width,
            height,
            component_base,
            request.session,
        )?;
        let packets = prepare_i64_component_packets(
            samples,
            I64ComponentPrepareRequest {
                component: u16::try_from(component_idx).map_err(|_| {
                    NativeEncodePipelineError::internal_invariant(
                        "validated component index exceeds u16",
                    )
                })?,
                width,
                height,
                num_levels: request.num_levels,
                step_sizes: request
                    .component_step_sizes
                    .get(component_idx)
                    .ok_or_else(|| {
                        NativeEncodePipelineError::internal_invariant(
                            "component quantization step count mismatch",
                        )
                    })?,
                subband_settings: request.subband_settings,
                retained_base_bytes: component_base,
                session: request.session,
            },
        )?;
        let new_packet_bytes = prepared_packets_ownership(&packets, packets.capacity())?.total()?;
        request.session.checked_phase(
            checked_add_bytes(
                component_base,
                new_packet_bytes,
                "typed i64 prepared components",
            )?,
            "typed i64 prepared components",
        )?;
        component_packets.push(packets);
    }
    check_packet_tree(
        request.session,
        request.retained_base_bytes,
        &component_packets,
    )?;
    Ok(component_packets)
}

fn validate_component_counts(
    planes: &[EncodeTypedComponentPlane<'_>],
    request: TypedPlanePacketRequest<'_, '_>,
) -> NativeEncodePipelineResult<()> {
    if request.component_dimensions.len() != planes.len() {
        return Err(NativeEncodePipelineError::internal_invariant(
            "component dimensions count does not match component count",
        ));
    }
    if request.component_step_sizes.len() != planes.len() {
        return Err(NativeEncodePipelineError::internal_invariant(
            "component quantization step count mismatch",
        ));
    }
    Ok(())
}

fn check_packet_tree(
    session: &NativeEncodeSession<'_>,
    retained_base_bytes: usize,
    packets: &Vec<Vec<PreparedResolutionPacket>>,
) -> NativeEncodePipelineResult<()> {
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            prepared_packet_tree_ownership(packets, packets.capacity())?.total()?,
            "typed i64 prepared packet tree",
        )?,
        "typed i64 prepared packet tree",
    )?;
    Ok(())
}

fn try_typed_component_plane_to_i64(
    plane: &EncodeTypedComponentPlane<'_>,
    width: u32,
    height: u32,
    retained_base_bytes: usize,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<Vec<i64>> {
    let bytes_per_sample = raw_pixel_bytes_per_sample(plane.bit_depth)
        .map_err(NativeEncodePipelineError::invalid_input)?;
    let sample_count = usize::try_from(width)
        .map_err(|_| {
            NativeEncodePipelineError::arithmetic_overflow("component width exceeds usize")
        })?
        .checked_mul(usize::try_from(height).map_err(|_| {
            NativeEncodePipelineError::arithmetic_overflow("component height exceeds usize")
        })?)
        .ok_or_else(|| NativeEncodePipelineError::arithmetic_overflow("image dimensions"))?;
    let expected_len = sample_count
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| NativeEncodePipelineError::arithmetic_overflow("image byte length"))?;
    if plane.data.len() != expected_len {
        return Err(NativeEncodePipelineError::invalid_input(
            "component plane data length mismatch",
        ));
    }
    let requested = checked_element_bytes::<i64>(sample_count, "typed i64 component samples")?;
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            requested,
            "typed i64 component samples",
        )?,
        "typed i64 component samples",
    )?;
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(sample_count)
        .map_err(|_| host_allocation_failed("typed i64 component samples", requested))?;
    let unsigned_offset = if plane.signed {
        0
    } else {
        1_i64 << (u32::from(plane.bit_depth) - 1)
    };
    for sample in plane.data.chunks_exact(bytes_per_sample) {
        let raw = read_le_sample_value(sample, plane.bit_depth);
        samples.push(if plane.signed {
            sign_extend_sample(raw, plane.bit_depth)
        } else {
            i64::try_from(raw).map_err(|_| {
                NativeEncodePipelineError::arithmetic_overflow("typed component sample exceeds i64")
            })? - unsigned_offset
        });
    }
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            checked_element_bytes::<i64>(samples.capacity(), "typed i64 component samples")?,
            "typed i64 component samples",
        )?,
        "typed i64 component samples",
    )?;
    Ok(samples)
}
