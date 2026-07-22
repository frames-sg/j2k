// SPDX-License-Identifier: MIT OR Apache-2.0

//! Retained-owner orchestration for compact preencoded 9/7 HTJ2K images.

use super::options::validate_single_layer_packet_input;
use super::{
    codestream_write, compact_payload_slice, count_compact_code_blocks, packet_encode,
    preencoded_compact_97_level_count, quantize, validate_irreversible_quantization_profile,
    validate_precinct_exponents_for_options, validate_preencoded_compact_htj2k97_image,
    BlockCodingMode, EncodeComponentSampleInfo, EncodeOptions, EncodeParams,
    EncodeProgressionOrder, J2kEncodeStageAccelerator, J2kPacketizationEncodeJob,
    NativeEncodePhase, NativeEncodePipelineError, NativeEncodePipelineResult,
    NativeEncodeRetainedInput, NativeEncodeSession, PreencodedHtj2k97CompactCodeBlock,
    PreencodedHtj2k97CompactComponent, PreencodedHtj2k97CompactImage,
    PreencodedHtj2k97CompactResolution, PreencodedHtj2k97CompactSubband, PreparedCompactCodeBlock,
    PreparedCompactResolutionPacket, PreparedCompactSubband, QuantStepSize, Vec,
    MAX_J2K_SPEC_COMPONENTS,
};
use crate::j2c::encode::allocation::{checked_add_bytes, checked_element_bytes};
use crate::{
    EncodeError, EncodeResult, J2kPacketizationPacketDescriptor, J2kPacketizationResolution,
};

const PACKET_OWNERS: &str = "retained compact preencoded 9/7 packetization owners";
const CONSTRUCTION: &str = "retained compact preencoded 9/7 construction owners";
const ACCELERATOR_OUTPUT: &str = "compact preencoded 9/7 accelerator packet output";
const SCALAR_OUTPUT: &str = "compact preencoded 9/7 scalar packet output";
const FINAL_HIGH_WATER: &str = "compact preencoded 9/7 final codestream high-water";
const MAX_QUANTIZATION_GUARD_BITS: u8 = 7;

mod construction;

pub(super) fn encode_preencoded_htj2k_97_compact_owned_with_accelerator(
    image: PreencodedHtj2k97CompactImage,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
    cap: usize,
) -> NativeEncodePipelineResult<Vec<u8>> {
    validate_compact_request(&image, options)?;
    let retained_input_bytes = compact_image_retained_bytes(&image)?;
    let packetized = {
        let retained_input =
            NativeEncodeRetainedInput::from_owner_bytes(&image, retained_input_bytes);
        let session = NativeEncodeSession::try_with_lowered_cap(retained_input, cap)?;
        Compact97PacketPlan::try_new(&image, options, retained_input_bytes, &session)?
            .packetize(&session, accelerator)?
    };

    // Packet metadata only borrows the compact image. Once Tier-2 owns its
    // output, release the potentially large payload before allocating the
    // final codestream and start a new exact live-owner phase.
    drop(image);
    let final_session =
        NativeEncodeSession::try_with_lowered_cap(NativeEncodeRetainedInput::none(), cap)?;
    finalize_compact_codestream(&packetized, &final_session)
}

struct Compact97PacketPlan<'a> {
    params: EncodeParams,
    quant_params: Vec<(u16, u16)>,
    prepared_packets: Vec<PreparedCompactResolutionPacket<'a>>,
    packet_descriptors: Vec<J2kPacketizationPacketDescriptor>,
    retained_input_bytes: usize,
}

struct Compact97Packetized {
    params: EncodeParams,
    quant_params: Vec<(u16, u16)>,
    tile_data: Vec<u8>,
}

impl<'a> Compact97PacketPlan<'a> {
    fn try_new(
        image: &'a PreencodedHtj2k97CompactImage,
        options: &EncodeOptions,
        retained_input_bytes: usize,
        session: &NativeEncodeSession<'_>,
    ) -> NativeEncodePipelineResult<Self> {
        construction::try_build_plan(image, options, retained_input_bytes, session)
    }

    fn packetize(
        self,
        session: &NativeEncodeSession<'_>,
        accelerator: &mut impl J2kEncodeStageAccelerator,
    ) -> NativeEncodePipelineResult<Compact97Packetized> {
        let plan_owner_bytes = self.plan_owner_retained_bytes()?;
        let retained_plan_bytes = checked_add_bytes(
            plan_owner_bytes,
            checked_element_bytes::<J2kPacketizationPacketDescriptor>(
                self.packet_descriptors.capacity(),
                PACKET_OWNERS,
            )?,
            PACKET_OWNERS,
        )?;
        session.checked_phase(retained_plan_bytes, PACKET_OWNERS)?;
        let packetization_resolutions = construction::try_public_packet_metadata(
            &self.prepared_packets,
            session,
            retained_plan_bytes,
        )?;
        let packet_phase_bytes = self.packet_phase_retained_bytes(
            plan_owner_bytes,
            &packetization_resolutions,
            packetization_resolutions.capacity(),
        )?;
        let phase = session.checked_phase(packet_phase_bytes, PACKET_OWNERS)?;
        let scalar_additional = self.scalar_additional_retained_bytes(
            plan_owner_bytes,
            &packetization_resolutions,
            packetization_resolutions.capacity(),
        )?;
        let job = J2kPacketizationEncodeJob {
            resolution_count: u32::try_from(packetization_resolutions.len()).map_err(|_| {
                NativeEncodePipelineError::arithmetic_overflow(
                    "packetization resolution count exceeds u32",
                )
            })?,
            num_layers: 1,
            num_components: self.params.num_components,
            code_block_count: count_compact_code_blocks(&self.prepared_packets)
                .map_err(NativeEncodePipelineError::arithmetic_overflow)?,
            progression_order: self.params.progression_order.packetization_order(),
            packet_descriptors: &self.packet_descriptors,
            resolutions: &packetization_resolutions,
        };
        let tile_data = packetize_compact_job(
            &job,
            &phase,
            packet_phase_bytes,
            scalar_additional,
            session,
            accelerator,
        )?;

        drop(packetization_resolutions);
        let Self {
            params,
            quant_params,
            prepared_packets: _,
            packet_descriptors: _,
            retained_input_bytes: _,
        } = self;
        Ok(Compact97Packetized {
            params,
            quant_params,
            tile_data,
        })
    }

    fn plan_owner_retained_bytes(&self) -> EncodeResult<usize> {
        let bytes = encode_params_retained_bytes(&self.params)?;
        let bytes = add_capacity::<(u16, u16)>(
            bytes,
            self.quant_params.capacity(),
            "compact 9/7 quantization parameters",
        )?;
        prepared_compact_retained_bytes(
            bytes,
            &self.prepared_packets,
            self.prepared_packets.capacity(),
        )
    }

    fn packet_phase_retained_bytes(
        &self,
        plan_owner_bytes: usize,
        resolutions: &[J2kPacketizationResolution<'_>],
        resolution_capacity: usize,
    ) -> EncodeResult<usize> {
        let bytes = checked_add_bytes(
            plan_owner_bytes,
            checked_element_bytes::<J2kPacketizationPacketDescriptor>(
                self.packet_descriptors.capacity(),
                PACKET_OWNERS,
            )?,
            PACKET_OWNERS,
        )?;
        checked_add_bytes(
            bytes,
            packet_encode::packet_metadata_retained_bytes(resolutions, resolution_capacity, 0)?,
            PACKET_OWNERS,
        )
    }

    fn scalar_additional_retained_bytes(
        &self,
        plan_owner_bytes: usize,
        resolutions: &[J2kPacketizationResolution<'_>],
        resolution_capacity: usize,
    ) -> EncodeResult<usize> {
        let mut bytes = checked_add_bytes(
            self.retained_input_bytes,
            plan_owner_bytes,
            "compact 9/7 scalar retained owners",
        )?;
        let descriptor_excess = self
            .packet_descriptors
            .capacity()
            .checked_sub(self.packet_descriptors.len())
            .ok_or(EncodeError::InternalInvariant {
                what: "compact packet descriptor length exceeds capacity",
            })?;
        bytes = add_capacity::<J2kPacketizationPacketDescriptor>(
            bytes,
            descriptor_excess,
            "compact 9/7 scalar packet descriptor excess capacity",
        )?;
        let resolution_excess = resolution_capacity.checked_sub(resolutions.len()).ok_or(
            EncodeError::InternalInvariant {
                what: "compact packet resolution length exceeds capacity",
            },
        )?;
        add_capacity::<J2kPacketizationResolution<'_>>(
            bytes,
            resolution_excess,
            "compact 9/7 scalar packet resolution excess capacity",
        )
    }
}

fn validate_compact_request(
    image: &PreencodedHtj2k97CompactImage,
    options: &EncodeOptions,
) -> NativeEncodePipelineResult<()> {
    if image.width == 0 || image.height == 0 {
        return Err(NativeEncodePipelineError::invalid_input(
            "invalid dimensions",
        ));
    }
    if image.components.is_empty() {
        return Err(NativeEncodePipelineError::invalid_input(
            "component set must be non-empty",
        ));
    }
    if image.components.len() > usize::from(MAX_J2K_SPEC_COMPONENTS) {
        return Err(NativeEncodePipelineError::unsupported(
            "component count exceeds the JPEG 2000 Part 1 limit",
        ));
    }
    if image.bit_depth == 0 {
        return Err(NativeEncodePipelineError::invalid_input(
            "bit depth must be non-zero",
        ));
    }
    if image.bit_depth > 16 {
        return Err(NativeEncodePipelineError::unsupported(
            "compact preencoded HTJ2K bit depth exceeds 16 bits",
        ));
    }
    if options.guard_bits > MAX_QUANTIZATION_GUARD_BITS {
        return Err(NativeEncodePipelineError::invalid_input(
            "guard bits exceed the JPEG 2000 quantization marker field",
        ));
    }
    validate_single_layer_packet_input(
        options,
        "compact preencoded HTJ2K encode supports one quality layer",
    )?;
    if options.write_ppm && options.write_ppt {
        return Err(NativeEncodePipelineError::invalid_input(
            "PPM and PPT packet header markers are mutually exclusive",
        ));
    }
    if matches!(options.tile_part_packet_limit, Some(0)) {
        return Err(NativeEncodePipelineError::invalid_input(
            "tile-part packet limit must be non-zero",
        ));
    }
    if options.write_plt
        || options.write_plm
        || options.write_ppm
        || options.write_ppt
        || options.write_sop
        || options.write_eph
        || options.tile_part_packet_limit.is_some()
    {
        return Err(NativeEncodePipelineError::unsupported(
            "compact preencoded HTJ2K encode does not support packet marker or tile-part options",
        ));
    }
    if options.tile_size.is_some() {
        return Err(NativeEncodePipelineError::unsupported(
            "compact preencoded HTJ2K encode does not support explicit tile sizes",
        ));
    }
    if !options.roi_component_shifts.is_empty() {
        return Err(NativeEncodePipelineError::unsupported(
            "compact preencoded HTJ2K encode does not support ROI shifts",
        ));
    }
    if options.component_sampling.is_some() {
        return Err(NativeEncodePipelineError::invalid_input(
            "compact preencoded HTJ2K sampling comes from the compact image",
        ));
    }
    validate_irreversible_quantization_profile(options)
        .map_err(NativeEncodePipelineError::invalid_input)?;
    if image
        .components
        .iter()
        .any(|component| component.x_rsiz == 0 || component.y_rsiz == 0)
    {
        return Err(NativeEncodePipelineError::invalid_input(
            "component sampling factors must be non-zero",
        ));
    }
    Ok(())
}

fn try_compact_packetization_accelerator(
    job: J2kPacketizationEncodeJob<'_>,
    phase: &NativeEncodePhase<'_, '_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<Option<Vec<u8>>> {
    let Some(output) =
        accelerator
            .encode_packetization(job)
            .map_err(|source| EncodeError::Accelerator {
                operation: "compact preencoded 9/7 packetization",
                source,
            })?
    else {
        return Ok(None);
    };
    phase.reconcile_accelerator_vec(&output, ACCELERATOR_OUTPUT)?;
    Ok(Some(output))
}

fn packetize_compact_job(
    job: &J2kPacketizationEncodeJob<'_>,
    phase: &NativeEncodePhase<'_, '_>,
    packet_phase_bytes: usize,
    scalar_additional: usize,
    session: &NativeEncodeSession<'_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<Vec<u8>> {
    if let Some(output) = try_compact_packetization_accelerator(*job, phase, accelerator)? {
        return Ok(output);
    }
    let output = packet_encode::form_borrowed_packetization_scalar(*job, scalar_additional)?;
    let with_output = checked_add_bytes(packet_phase_bytes, output.capacity(), SCALAR_OUTPUT)?;
    session.checked_phase(with_output, SCALAR_OUTPUT)?;
    Ok(output)
}

fn finalize_compact_codestream(
    packetized: &Compact97Packetized,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<Vec<u8>> {
    let accounted = codestream_write::write_codestream_accounted_with_peak_check(
        &packetized.params,
        &packetized.tile_data,
        &packetized.quant_params,
        |writer_peak_bytes| {
            reconcile_compact_final_codestream(session, packetized, writer_peak_bytes)
        },
    )?;
    // The writer already runs this check before reserving and again with the
    // allocator-returned capacity. Keep the reported peak part of the final
    // handoff contract as a defense against future writer changes.
    reconcile_compact_final_codestream(session, packetized, accounted.writer_peak_bytes)?;
    Ok(accounted.codestream)
}

fn reconcile_compact_final_codestream(
    session: &NativeEncodeSession<'_>,
    packetized: &Compact97Packetized,
    writer_peak_bytes: usize,
) -> EncodeResult<()> {
    let owner_bytes = compact_final_owner_retained_bytes(packetized)?;
    let with_output = checked_add_bytes(owner_bytes, writer_peak_bytes, FINAL_HIGH_WATER)?;
    session.checked_phase(with_output, FINAL_HIGH_WATER)?;
    Ok(())
}

fn compact_final_owner_retained_bytes(packetized: &Compact97Packetized) -> EncodeResult<usize> {
    let bytes = encode_params_retained_bytes(&packetized.params)?;
    let bytes = add_capacity::<(u16, u16)>(
        bytes,
        packetized.quant_params.capacity(),
        "compact 9/7 final quantization parameters",
    )?;
    add_capacity::<u8>(
        bytes,
        packetized.tile_data.capacity(),
        "compact 9/7 retained packet output",
    )
}

fn compact_image_retained_bytes(image: &PreencodedHtj2k97CompactImage) -> EncodeResult<usize> {
    let mut bytes = add_capacity::<u8>(0, image.payload.capacity(), "compact 9/7 payload")?;
    bytes = add_capacity::<PreencodedHtj2k97CompactComponent>(
        bytes,
        image.components.capacity(),
        "compact 9/7 components",
    )?;
    for component in &image.components {
        bytes = add_capacity::<PreencodedHtj2k97CompactResolution>(
            bytes,
            component.resolutions.capacity(),
            "compact 9/7 resolutions",
        )?;
        for resolution in &component.resolutions {
            bytes = add_capacity::<PreencodedHtj2k97CompactSubband>(
                bytes,
                resolution.subbands.capacity(),
                "compact 9/7 subbands",
            )?;
            for subband in &resolution.subbands {
                bytes = add_capacity::<PreencodedHtj2k97CompactCodeBlock>(
                    bytes,
                    subband.code_blocks.capacity(),
                    "compact 9/7 code-block metadata",
                )?;
            }
        }
    }
    Ok(bytes)
}

fn prepared_compact_retained_bytes(
    mut bytes: usize,
    packets: &[PreparedCompactResolutionPacket<'_>],
    packet_capacity: usize,
) -> EncodeResult<usize> {
    bytes = add_capacity::<PreparedCompactResolutionPacket<'_>>(
        bytes,
        packet_capacity,
        "prepared compact 9/7 resolutions",
    )?;
    for packet in packets {
        bytes = add_capacity::<PreparedCompactSubband<'_>>(
            bytes,
            packet.subbands.capacity(),
            "prepared compact 9/7 subbands",
        )?;
        for subband in &packet.subbands {
            bytes = add_capacity::<PreparedCompactCodeBlock<'_>>(
                bytes,
                subband.code_blocks.capacity(),
                "prepared compact 9/7 code blocks",
            )?;
        }
    }
    Ok(bytes)
}

fn encode_params_retained_bytes(params: &EncodeParams) -> EncodeResult<usize> {
    let mut bytes = add_capacity::<EncodeComponentSampleInfo>(
        0,
        params.component_sample_info.capacity(),
        "compact 9/7 component sample metadata",
    )?;
    bytes = add_capacity::<Vec<(u16, u16)>>(
        bytes,
        params.component_quantization_step_sizes.capacity(),
        "compact 9/7 component quantization owners",
    )?;
    for steps in &params.component_quantization_step_sizes {
        bytes = add_capacity::<(u16, u16)>(
            bytes,
            steps.capacity(),
            "compact 9/7 component quantization values",
        )?;
    }
    bytes = add_capacity::<(u8, u8)>(
        bytes,
        params.component_sampling.capacity(),
        "compact 9/7 component sampling",
    )?;
    bytes = add_capacity::<u8>(
        bytes,
        params.roi_component_shifts.capacity(),
        "compact 9/7 ROI shifts",
    )?;
    add_capacity::<(u8, u8)>(
        bytes,
        params.precinct_exponents.capacity(),
        "compact 9/7 precinct exponents",
    )
}

fn add_capacity<T>(bytes: usize, capacity: usize, what: &'static str) -> EncodeResult<usize> {
    checked_add_bytes(bytes, checked_element_bytes::<T>(capacity, what)?, what)
}

#[cfg(test)]
mod accelerator_tests;
#[cfg(test)]
#[path = "compact97/tests.rs"]
mod tests;
