// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible, phase-checked compact 9/7 packet-plan construction.

use super::{
    compact_payload_slice, preencoded_compact_97_level_count, quantize,
    validate_precinct_exponents_for_options, validate_preencoded_compact_htj2k97_image,
    BlockCodingMode, Compact97PacketPlan, EncodeOptions, EncodeParams, EncodeProgressionOrder,
    NativeEncodePipelineError, NativeEncodePipelineResult, NativeEncodeSession,
    PreencodedHtj2k97CompactImage, PreencodedHtj2k97CompactResolution, PreparedCompactCodeBlock,
    PreparedCompactResolutionPacket, PreparedCompactSubband, QuantStepSize, CONSTRUCTION,
};
use crate::j2c::encode::allocation::{
    checked_add_bytes, checked_element_bytes, host_allocation_failed,
};
use crate::{
    EncodeError, EncodeResult, J2kPacketizationBlockCodingMode, J2kPacketizationCodeBlock,
    J2kPacketizationPacketDescriptor, J2kPacketizationResolution, J2kPacketizationSubband,
};
use alloc::vec::Vec;

pub(super) fn try_build_plan<'a>(
    image: &'a PreencodedHtj2k97CompactImage,
    options: &EncodeOptions,
    retained_input_bytes: usize,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<Compact97PacketPlan<'a>> {
    let num_components = u16::try_from(image.components.len()).map_err(|_| {
        NativeEncodePipelineError::unsupported("component count exceeds the JPEG 2000 Part 1 limit")
    })?;
    let num_levels = preencoded_compact_97_level_count(&image.components)
        .map_err(NativeEncodePipelineError::invalid_input)?;
    validate_precinct_exponents_for_options(options, num_levels)
        .map_err(NativeEncodePipelineError::invalid_input)?;
    let guard_bits = options.guard_bits.max(2);
    let mut allocations = ConstructionTracker::new(session);

    let step_count = usize::from(num_levels)
        .checked_mul(3)
        .and_then(|count| count.checked_add(1))
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "compact 9/7 quantization step count",
        })?;
    let mut step_sizes = allocations
        .try_vec::<QuantStepSize>(step_count, "compact 9/7 construction quantization steps")?;
    quantize::append_step_sizes_with_irreversible_profile(
        &mut step_sizes,
        image.bit_depth,
        num_levels,
        false,
        guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
    );
    if step_sizes.len() != step_count {
        return Err(EncodeError::InternalInvariant {
            what: "compact 9/7 quantization step count changed during construction",
        }
        .into());
    }
    validate_preencoded_compact_htj2k97_image(image, guard_bits, &step_sizes)
        .map_err(NativeEncodePipelineError::invalid_input)?;

    let prepared_packets = try_prepared_packets(&mut allocations, image, options)?;
    let packet_count = prepared_packets.len();

    let mut packet_descriptors = allocations.try_vec::<J2kPacketizationPacketDescriptor>(
        packet_count,
        "compact 9/7 construction packet descriptors",
    )?;
    for (packet_index, packet) in prepared_packets.iter().enumerate() {
        let packet_index = u32::try_from(packet_index).map_err(|_| {
            NativeEncodePipelineError::arithmetic_overflow("packet descriptor index exceeds u32")
        })?;
        try_push(
            &mut packet_descriptors,
            J2kPacketizationPacketDescriptor {
                packet_index,
                state_index: packet_index,
                layer: 0,
                resolution: packet.resolution,
                component: packet.component,
                precinct: packet.precinct,
            },
            "compact 9/7 packet descriptor capacity changed during construction",
        )?;
    }

    let (params, quant_params) = try_plan_metadata(
        &mut allocations,
        image,
        options,
        num_components,
        num_levels,
        guard_bits,
        &step_sizes,
    )?;

    // The tracker deliberately keeps the step-size owner in the construction
    // peak through every retained plan allocation. It is released here because
    // only the compact marker pairs survive into packetization.
    drop(step_sizes);
    Ok(Compact97PacketPlan {
        params,
        quant_params,
        prepared_packets,
        packet_descriptors,
        retained_input_bytes,
    })
}

fn try_prepared_packets<'a>(
    allocations: &mut ConstructionTracker<'_, '_>,
    image: &'a PreencodedHtj2k97CompactImage,
    options: &EncodeOptions,
) -> NativeEncodePipelineResult<Vec<PreparedCompactResolutionPacket<'a>>> {
    let resolution_count = image
        .components
        .first()
        .map_or(0, |component| component.resolutions.len());
    if image
        .components
        .iter()
        .any(|component| component.resolutions.len() != resolution_count)
    {
        return Err(NativeEncodePipelineError::internal_invariant(
            "component packet resolution count mismatch",
        ));
    }
    let packet_count = image.components.len().checked_mul(resolution_count).ok_or(
        EncodeError::ArithmeticOverflow {
            what: "compact 9/7 prepared packet count",
        },
    )?;
    let mut packets = allocations.try_vec::<PreparedCompactResolutionPacket<'a>>(
        packet_count,
        "compact 9/7 construction prepared packets",
    )?;
    match options.progression_order {
        EncodeProgressionOrder::Lrcp
        | EncodeProgressionOrder::Rlcp
        | EncodeProgressionOrder::Rpcl => {
            for resolution_idx in 0..resolution_count {
                for (component_idx, component) in image.components.iter().enumerate() {
                    append_packet(
                        allocations,
                        &mut packets,
                        image,
                        component_idx,
                        resolution_idx,
                        &component.resolutions[resolution_idx],
                    )?;
                }
            }
        }
        EncodeProgressionOrder::Pcrl | EncodeProgressionOrder::Cprl => {
            for (component_idx, component) in image.components.iter().enumerate() {
                for (resolution_idx, resolution) in component.resolutions.iter().enumerate() {
                    append_packet(
                        allocations,
                        &mut packets,
                        image,
                        component_idx,
                        resolution_idx,
                        resolution,
                    )?;
                }
            }
        }
    }
    Ok(packets)
}

fn try_plan_metadata(
    allocations: &mut ConstructionTracker<'_, '_>,
    image: &PreencodedHtj2k97CompactImage,
    options: &EncodeOptions,
    num_components: u16,
    num_levels: u8,
    guard_bits: u8,
    step_sizes: &[QuantStepSize],
) -> NativeEncodePipelineResult<(EncodeParams, Vec<(u16, u16)>)> {
    let mut quant_params = allocations.try_vec::<(u16, u16)>(
        step_sizes.len(),
        "compact 9/7 construction quantization parameters",
    )?;
    for step in step_sizes {
        try_push(
            &mut quant_params,
            (step.exponent, step.mantissa),
            "compact 9/7 quantization parameter capacity changed during construction",
        )?;
    }
    let mut component_sampling = allocations.try_vec::<(u8, u8)>(
        image.components.len(),
        "compact 9/7 construction component sampling",
    )?;
    for component in &image.components {
        try_push(
            &mut component_sampling,
            (component.x_rsiz, component.y_rsiz),
            "compact 9/7 sampling capacity changed during construction",
        )?;
    }
    let mut roi_component_shifts = allocations.try_vec::<u8>(
        image.components.len(),
        "compact 9/7 construction ROI shifts",
    )?;
    for _ in &image.components {
        try_push(
            &mut roi_component_shifts,
            0,
            "compact 9/7 ROI capacity changed during construction",
        )?;
    }
    let mut precinct_exponents = allocations.try_vec::<(u8, u8)>(
        options.precinct_exponents.len(),
        "compact 9/7 construction precinct exponents",
    )?;
    for &exponents in &options.precinct_exponents {
        try_push(
            &mut precinct_exponents,
            exponents,
            "compact 9/7 precinct capacity changed during construction",
        )?;
    }

    let params = EncodeParams {
        width: image.width,
        height: image.height,
        tile_width: image.width,
        tile_height: image.height,
        num_components,
        bit_depth: image.bit_depth,
        signed: image.signed,
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
        write_plt: false,
        write_plm: false,
        write_ppm: false,
        write_ppt: false,
        write_sop: false,
        write_eph: false,
        terminate_coding_passes: false,
        component_sampling,
        roi_component_shifts,
        precinct_exponents,
    };
    Ok((params, quant_params))
}

fn append_packet<'a>(
    allocations: &mut ConstructionTracker<'_, '_>,
    packets: &mut Vec<PreparedCompactResolutionPacket<'a>>,
    image: &'a PreencodedHtj2k97CompactImage,
    component_idx: usize,
    resolution_idx: usize,
    resolution: &'a PreencodedHtj2k97CompactResolution,
) -> NativeEncodePipelineResult<()> {
    let mut subbands = allocations.try_vec::<PreparedCompactSubband<'a>>(
        resolution.subbands.len(),
        "compact 9/7 construction prepared subbands",
    )?;
    for subband in &resolution.subbands {
        let mut code_blocks = allocations.try_vec::<PreparedCompactCodeBlock<'a>>(
            subband.code_blocks.len(),
            "compact 9/7 construction prepared code blocks",
        )?;
        for block in &subband.code_blocks {
            try_push(
                &mut code_blocks,
                PreparedCompactCodeBlock {
                    data: compact_payload_slice(&image.payload, &block.payload_range)
                        .map_err(NativeEncodePipelineError::internal_invariant)?,
                    cleanup_length: block.cleanup_length,
                    refinement_length: block.refinement_length,
                    num_coding_passes: block.num_coding_passes,
                    num_zero_bitplanes: block.num_zero_bitplanes,
                },
                "compact 9/7 code-block capacity changed during construction",
            )?;
        }
        try_push(
            &mut subbands,
            PreparedCompactSubband {
                code_blocks,
                num_cbs_x: subband.num_cbs_x,
                num_cbs_y: subband.num_cbs_y,
            },
            "compact 9/7 subband capacity changed during construction",
        )?;
    }
    try_push(
        packets,
        PreparedCompactResolutionPacket {
            component: u16::try_from(component_idx).map_err(|_| {
                NativeEncodePipelineError::arithmetic_overflow("component index exceeds u16")
            })?,
            resolution: u32::try_from(resolution_idx).map_err(|_| {
                NativeEncodePipelineError::arithmetic_overflow("resolution index exceeds u32")
            })?,
            precinct: 0,
            subbands,
        },
        "compact 9/7 packet capacity changed during construction",
    )?;
    Ok(())
}

pub(super) fn try_public_packet_metadata<'a>(
    packets: &[PreparedCompactResolutionPacket<'a>],
    session: &NativeEncodeSession<'_>,
    retained_plan_bytes: usize,
) -> NativeEncodePipelineResult<Vec<J2kPacketizationResolution<'a>>> {
    let mut allocations = ConstructionTracker::with_live_bytes(session, retained_plan_bytes);
    let mut resolutions = allocations.try_vec::<J2kPacketizationResolution<'a>>(
        packets.len(),
        "compact 9/7 accelerator resolution metadata",
    )?;
    for packet in packets {
        let mut subbands = allocations.try_vec::<J2kPacketizationSubband<'a>>(
            packet.subbands.len(),
            "compact 9/7 accelerator subband metadata",
        )?;
        for subband in &packet.subbands {
            let mut code_blocks = allocations.try_vec::<J2kPacketizationCodeBlock<'a>>(
                subband.code_blocks.len(),
                "compact 9/7 accelerator code-block metadata",
            )?;
            for block in &subband.code_blocks {
                try_push(
                    &mut code_blocks,
                    J2kPacketizationCodeBlock {
                        data: block.data,
                        ht_cleanup_length: block.cleanup_length,
                        ht_refinement_length: block.refinement_length,
                        num_coding_passes: block.num_coding_passes,
                        num_zero_bitplanes: block.num_zero_bitplanes,
                        previously_included: false,
                        l_block: 3,
                        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
                    },
                    "compact 9/7 accelerator code-block capacity changed during construction",
                )?;
            }
            try_push(
                &mut subbands,
                J2kPacketizationSubband {
                    code_blocks,
                    num_cbs_x: subband.num_cbs_x,
                    num_cbs_y: subband.num_cbs_y,
                },
                "compact 9/7 accelerator subband capacity changed during construction",
            )?;
        }
        try_push(
            &mut resolutions,
            J2kPacketizationResolution { subbands },
            "compact 9/7 accelerator resolution capacity changed during construction",
        )?;
    }
    Ok(resolutions)
}

struct ConstructionTracker<'session, 'input> {
    session: &'session NativeEncodeSession<'input>,
    live_bytes: usize,
}

impl<'session, 'input> ConstructionTracker<'session, 'input> {
    const fn new(session: &'session NativeEncodeSession<'input>) -> Self {
        Self::with_live_bytes(session, 0)
    }

    const fn with_live_bytes(
        session: &'session NativeEncodeSession<'input>,
        live_bytes: usize,
    ) -> Self {
        Self {
            session,
            live_bytes,
        }
    }

    fn try_vec<T>(&mut self, count: usize, what: &'static str) -> EncodeResult<Vec<T>> {
        let requested_bytes = checked_element_bytes::<T>(count, what)?;
        let requested_live = checked_add_bytes(self.live_bytes, requested_bytes, CONSTRUCTION)?;
        self.session.checked_phase(requested_live, CONSTRUCTION)?;

        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| host_allocation_failed(what, requested_bytes))?;
        let actual_bytes = checked_element_bytes::<T>(values.capacity(), what)?;
        let actual_live = checked_add_bytes(self.live_bytes, actual_bytes, CONSTRUCTION)?;
        self.session.checked_phase(actual_live, CONSTRUCTION)?;
        self.live_bytes = actual_live;
        Ok(values)
    }
}

fn try_push<T>(values: &mut Vec<T>, value: T, what: &'static str) -> EncodeResult<()> {
    if values.len() == values.capacity() {
        return Err(EncodeError::InternalInvariant { what });
    }
    values.push(value);
    Ok(())
}
