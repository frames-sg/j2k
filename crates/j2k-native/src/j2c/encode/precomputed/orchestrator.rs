// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared typed orchestration for non-compact precomputed 9/7 packet plans.

use alloc::vec::Vec;

use super::allocation::{add_capacity, encode_params_retained_bytes, ConstructionTracker};
use super::options::validate_single_layer_packet_input;
use super::{
    encode_prepared_resolution_packets_for_session,
    ordered_prepared_resolution_packets_for_session, packet_descriptors_for_order_for_session,
    packet_encode, packetization_requires_scalar,
    packetize_resolution_packets_with_options_for_session, quantize,
    split_component_resolution_packets_by_precinct_for_session,
    validate_irreversible_quantization_profile, validate_precinct_exponents_for_options,
    write_single_tile_packetized_codestream_for_session, BlockCodingMode, EncodeOptions,
    EncodeParams, J2kEncodeStageAccelerator, J2kPacketizationPacketDescriptor,
    NativeEncodePipelineError, NativeEncodePipelineResult, NativeEncodeSession,
    PreparedResolutionPacket, QuantStepSize,
};
use crate::j2c::encode::allocation::{checked_add_bytes, checked_element_bytes};
use crate::j2c::encode::tier1_allocation::prepared_packets_ownership;
use crate::EncodeResult;

const MAX_QUANTIZATION_GUARD_BITS: u8 = 7;

pub(in crate::j2c::encode) struct Prepared97Metadata {
    pub(in crate::j2c::encode) params: EncodeParams,
    pub(in crate::j2c::encode) quant_params: Vec<(u16, u16)>,
    pub(in crate::j2c::encode) step_sizes: Vec<QuantStepSize>,
    pub(in crate::j2c::encode) tile_part_packet_limit: Option<u16>,
}

pub(in crate::j2c::encode) struct Prepared97PacketPlan {
    pub(in crate::j2c::encode) params: EncodeParams,
    pub(in crate::j2c::encode) quant_params: Vec<(u16, u16)>,
    pub(in crate::j2c::encode) packet_descriptors: Vec<J2kPacketizationPacketDescriptor>,
    pub(in crate::j2c::encode) prepared_packets: Vec<PreparedResolutionPacket>,
    pub(in crate::j2c::encode) tile_part_packet_limit: Option<u16>,
}

struct QuantizationOwners {
    step_sizes: Vec<QuantStepSize>,
    marker_pairs: Vec<(u16, u16)>,
}

impl Prepared97PacketPlan {
    pub(in crate::j2c::encode) fn packet_count(&self) -> usize {
        self.prepared_packets.len()
    }

    pub(in crate::j2c::encode) fn metadata_retained_bytes(&self) -> EncodeResult<usize> {
        let bytes = encode_params_retained_bytes(&self.params)?;
        let bytes = add_capacity::<(u16, u16)>(
            bytes,
            self.quant_params.capacity(),
            "precomputed 9/7 quantization parameters",
        )?;
        add_capacity::<J2kPacketizationPacketDescriptor>(
            bytes,
            self.packet_descriptors.capacity(),
            "precomputed 9/7 packet descriptors",
        )
    }

    pub(in crate::j2c::encode) fn retained_bytes(&self) -> EncodeResult<usize> {
        checked_add_bytes(
            self.metadata_retained_bytes()?,
            prepared_packets_ownership(&self.prepared_packets, self.prepared_packets.capacity())?
                .total()?,
            "prepared precomputed 9/7 packet plan",
        )
    }
}

pub(in crate::j2c::encode) fn validate_legacy_packet_options(
    options: &EncodeOptions,
) -> NativeEncodePipelineResult<()> {
    validate_irreversible_quantization_profile(options)
        .map_err(NativeEncodePipelineError::invalid_input)?;
    if options.guard_bits > MAX_QUANTIZATION_GUARD_BITS {
        return Err(NativeEncodePipelineError::invalid_input(
            "guard bits exceed the JPEG 2000 quantization marker field",
        ));
    }
    validate_single_layer_packet_input(
        options,
        "precomputed 9/7 packet input supports one quality layer",
    )?;
    if options.tile_size.is_some() {
        return Err(NativeEncodePipelineError::unsupported(
            "precomputed 9/7 packet input does not support explicit tile sizes",
        ));
    }
    if !options.roi_component_shifts.is_empty() {
        return Err(NativeEncodePipelineError::unsupported(
            "precomputed 9/7 packet input does not support ROI shifts",
        ));
    }
    if options.component_sampling.is_some() {
        return Err(NativeEncodePipelineError::invalid_input(
            "precomputed 9/7 sampling comes from the coefficient image",
        ));
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "the validated coefficient-image boundary keeps JPEG 2000 geometry explicit"
)]
pub(in crate::j2c::encode) fn try_metadata(
    width: u32,
    height: u32,
    bit_depth: u8,
    signed: bool,
    sampling: impl ExactSizeIterator<Item = (u8, u8)>,
    num_levels: u8,
    options: &EncodeOptions,
    tracker: &mut ConstructionTracker<'_, '_>,
) -> NativeEncodePipelineResult<Prepared97Metadata> {
    validate_legacy_packet_options(options)?;
    validate_precinct_exponents_for_options(options, num_levels)
        .map_err(NativeEncodePipelineError::invalid_input)?;
    let sampling_count = sampling.len();
    let num_components = u16::try_from(sampling_count).map_err(|_| {
        NativeEncodePipelineError::unsupported("component count exceeds the JPEG 2000 Part 1 limit")
    })?;
    let guard_bits = options.guard_bits.max(2);
    let QuantizationOwners {
        step_sizes,
        marker_pairs: quant_params,
    } = try_quantization_owners(bit_depth, num_levels, guard_bits, options, tracker)?;
    let mut component_sampling =
        tracker.try_vec::<(u8, u8)>(sampling_count, "precomputed 9/7 component sampling")?;
    component_sampling.extend(sampling);
    let mut roi_component_shifts =
        tracker.try_vec::<u8>(sampling_count, "precomputed 9/7 ROI shifts")?;
    roi_component_shifts.resize(sampling_count, 0);
    let precinct_exponents = tracker.try_copy_slice(
        &options.precinct_exponents,
        "precomputed 9/7 precinct exponents",
    )?;

    Ok(Prepared97Metadata {
        params: EncodeParams {
            width,
            height,
            tile_width: width,
            tile_height: height,
            num_components,
            bit_depth,
            signed,
            component_sample_info: Vec::new(),
            component_quantization_step_sizes: Vec::new(),
            num_decomposition_levels: num_levels,
            reversible: false,
            code_block_width_exp: options.code_block_width_exp,
            code_block_height_exp: options.code_block_height_exp,
            num_layers: 1,
            use_mct: false,
            guard_bits,
            block_coding_mode: BlockCodingMode::HighThroughput,
            progression_order: options.progression_order,
            write_tlm: options.write_tlm,
            write_plt: options.write_plt,
            write_plm: options.write_plm,
            write_ppm: options.write_ppm,
            write_ppt: options.write_ppt,
            write_sop: options.write_sop,
            write_eph: options.write_eph,
            terminate_coding_passes: false,
            component_sampling,
            roi_component_shifts,
            precinct_exponents,
        },
        quant_params,
        step_sizes,
        tile_part_packet_limit: options.tile_part_packet_limit,
    })
}

fn try_quantization_owners(
    bit_depth: u8,
    num_levels: u8,
    guard_bits: u8,
    options: &EncodeOptions,
    tracker: &mut ConstructionTracker<'_, '_>,
) -> NativeEncodePipelineResult<QuantizationOwners> {
    let step_count = usize::from(num_levels)
        .checked_mul(3)
        .and_then(|count| count.checked_add(1))
        .ok_or_else(|| {
            NativeEncodePipelineError::arithmetic_overflow("precomputed 9/7 step-size count")
        })?;
    let mut step_sizes =
        tracker.try_vec::<QuantStepSize>(step_count, "precomputed 9/7 step sizes")?;
    quantize::append_step_sizes_with_irreversible_profile(
        &mut step_sizes,
        bit_depth,
        num_levels,
        false,
        guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
    );
    let mut quant_params = tracker
        .try_vec::<(u16, u16)>(step_sizes.len(), "precomputed 9/7 quantization parameters")?;
    quant_params.extend(step_sizes.iter().map(|step| (step.exponent, step.mantissa)));
    Ok(QuantizationOwners {
        step_sizes,
        marker_pairs: quant_params,
    })
}

pub(in crate::j2c::encode) fn finish_plan(
    metadata: Prepared97Metadata,
    component_packets: Vec<Vec<PreparedResolutionPacket>>,
    options: &EncodeOptions,
    session: &NativeEncodeSession<'_>,
    retained_base_bytes: usize,
) -> NativeEncodePipelineResult<Prepared97PacketPlan> {
    let Prepared97Metadata {
        params,
        quant_params,
        step_sizes,
        tile_part_packet_limit,
    } = metadata;
    drop(step_sizes);
    let metadata_bytes = plan_metadata_bytes(&params, &quant_params, 0)?;
    let phase_base = checked_add_bytes(
        retained_base_bytes,
        metadata_bytes,
        "precomputed 9/7 plan metadata",
    )?;
    let component_packets = split_component_resolution_packets_by_precinct_for_session(
        component_packets,
        params.width,
        params.height,
        params.num_decomposition_levels,
        &params.precinct_exponents,
        session,
        phase_base,
    )?;
    let prepared_packets = ordered_prepared_resolution_packets_for_session(
        component_packets,
        options,
        session,
        phase_base,
    )?;
    let packet_descriptors = packet_descriptors_for_order_for_session(
        &prepared_packets,
        prepared_packets.capacity(),
        1,
        params.progression_order,
        session,
        phase_base,
    )?;
    Ok(Prepared97PacketPlan {
        params,
        quant_params,
        packet_descriptors,
        prepared_packets,
        tile_part_packet_limit,
    })
}

pub(in crate::j2c::encode) fn encode_plan(
    plan: Prepared97PacketPlan,
    session: &NativeEncodeSession<'_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<Vec<u8>> {
    let Prepared97PacketPlan {
        params,
        quant_params,
        packet_descriptors,
        prepared_packets,
        tile_part_packet_limit,
    } = plan;
    let metadata_without_descriptors = plan_metadata_bytes(&params, &quant_params, 0)?;
    let descriptor_bytes = checked_element_bytes::<J2kPacketizationPacketDescriptor>(
        packet_descriptors.capacity(),
        "precomputed 9/7 packet descriptors",
    )?;
    let resolution_packets = encode_prepared_resolution_packets_for_session(
        prepared_packets,
        session,
        checked_add_bytes(
            metadata_without_descriptors,
            descriptor_bytes,
            "precomputed 9/7 Tier-1 metadata",
        )?,
        accelerator,
    )?;
    let packetized = {
        let packet_owners = (&params, &quant_params);
        let packet_session = session.checked_child_session(
            &packet_owners,
            metadata_without_descriptors,
            "precomputed 9/7 packet-plan metadata",
        )?;
        packetize_resolution_packets_with_options_for_session(
            &resolution_packets,
            resolution_packets.capacity(),
            &packet_descriptors,
            packet_descriptors.capacity(),
            1,
            params.num_components,
            params.progression_order,
            packet_encode::PacketMarkerOptions {
                write_sop: params.write_sop,
                write_eph: params.write_eph,
                separate_packet_headers: params.write_ppm || params.write_ppt,
            },
            true,
            packetization_requires_scalar(&params, tile_part_packet_limit),
            &packet_session,
            accelerator,
        )?
    };
    drop(resolution_packets);
    drop(packet_descriptors);
    write_single_tile_packetized_codestream_for_session(
        &params,
        &packetized,
        &quant_params,
        tile_part_packet_limit,
        metadata_without_descriptors,
        session,
    )
}

pub(in crate::j2c::encode) fn plan_metadata_bytes(
    params: &EncodeParams,
    quant_params: &Vec<(u16, u16)>,
    descriptor_capacity: usize,
) -> EncodeResult<usize> {
    let bytes = encode_params_retained_bytes(params)?;
    let bytes = add_capacity::<(u16, u16)>(
        bytes,
        quant_params.capacity(),
        "precomputed 9/7 quantization parameters",
    )?;
    add_capacity::<J2kPacketizationPacketDescriptor>(
        bytes,
        descriptor_capacity,
        "precomputed 9/7 packet descriptors",
    )
}
