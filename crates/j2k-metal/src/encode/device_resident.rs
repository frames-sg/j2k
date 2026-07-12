// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    compute, host_outcome_from_buffer_outcome, lossless_device_coefficient_count,
    lossless_device_encode_plan, lossless_sample_shape,
    packet_descriptors_for_lossless_device_order,
    resident_packetization_resolutions_from_lossless_device_plan,
    validate_lossless_roundtrip_on_metal_region_with_session,
    validate_lossless_roundtrip_on_metal_tile_with_session, validate_metal_encode_tile,
    validate_padded_contiguous_metal_encode_tile, Duration, EncodeBackendPreference, Instant,
    J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions, MetalEncodeInputStaging,
    MetalEncodedJ2k, MetalLosslessBufferEncodeOutcome, MetalLosslessEncodeOutcome,
    MetalLosslessEncodeResidency, MetalLosslessEncodeTile, RESIDENT_CLASSIC_CODE_BLOCK_EDGE,
};

#[cfg(target_os = "macos")]
pub(super) fn try_encode_lossless_tile_device_resident_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
) -> Result<Option<MetalLosslessEncodeOutcome>, crate::Error> {
    let Some(outcome) = try_encode_lossless_tile_device_resident_to_metal_buffer_with_report(
        tile, options, session, staging,
    )?
    else {
        return Ok(None);
    };
    host_outcome_from_buffer_outcome(outcome).map(Some)
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "device-resident route keeps validation, fallback, and report accounting atomic"
)]
fn try_encode_lossless_tile_device_resident_to_metal_buffer_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
) -> Result<Option<MetalLosslessBufferEncodeOutcome>, crate::Error> {
    if options.backend == EncodeBackendPreference::CpuOnly {
        return Ok(None);
    }
    let (components, bit_depth) = lossless_sample_shape(tile.format)?;
    let bytes_per_pixel = tile.format.bytes_per_pixel();
    let bytes_per_sample =
        u8::try_from(tile.format.bytes_per_sample()).map_err(|_| crate::Error::MetalKernel {
            message: "J2K Metal resident encode bytes per sample exceeds u8".to_string(),
        })?;
    if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous) {
        validate_padded_contiguous_metal_encode_tile(tile, bytes_per_pixel)?;
    }
    let Some(mut plan) = lossless_device_encode_plan(
        tile.output_width,
        tile.output_height,
        components,
        bit_depth,
        options,
        RESIDENT_CLASSIC_CODE_BLOCK_EDGE,
        RESIDENT_CLASSIC_CODE_BLOCK_EDGE,
    )?
    else {
        return Ok(None);
    };

    let encode_started = Instant::now();
    let coefficient_count = lossless_device_coefficient_count(&plan.code_blocks)?;
    let code_block_count = plan.code_blocks.len();
    let resolution_count = plan.resolutions.len();
    let packetization_resolutions =
        resident_packetization_resolutions_from_lossless_device_plan(&plan)?;
    let packet_descriptors = packet_descriptors_for_lossless_device_order(
        resolution_count,
        plan.components,
        plan.progression_order,
    )?;
    let code_blocks = plan.take_code_blocks();
    let prepared = compute::prepare_lossless_device_code_blocks(
        session,
        compute::J2kLosslessDevicePrepareJob {
            input: tile.buffer,
            input_byte_offset: tile.byte_offset,
            input_width: tile.width,
            input_height: tile.height,
            input_pitch_bytes: tile.pitch_bytes,
            output_width: tile.output_width,
            output_height: tile.output_height,
            component_count: components,
            bytes_per_sample,
            bit_depth,
            num_decomposition_levels: plan.num_decomposition_levels,
            coefficient_count,
        },
        code_blocks,
    )?;
    let packetization_job = compute::J2kResidentPacketizationEncodeJob {
        resolution_count: u32::try_from(resolution_count).map_err(|_| {
            crate::Error::MetalKernel {
                message: "J2K Metal resident encode resolution count exceeds u32".to_string(),
            }
        })?,
        num_layers: 1,
        component_count: plan.components,
        code_block_count: u32::try_from(code_block_count).map_err(|_| {
            crate::Error::MetalKernel {
                message: "J2K Metal resident encode code-block count exceeds u32".to_string(),
            }
        })?,
        packet_descriptors: &packet_descriptors,
        resolutions: &packetization_resolutions,
    };
    let assembly_job = compute::J2kLosslessCodestreamAssemblyJob {
        width: tile.output_width,
        height: tile.output_height,
        component_count: plan.components,
        bit_depth: plan.bit_depth,
        signed: false,
        num_decomposition_levels: plan.num_decomposition_levels,
        use_mct: plan.use_mct,
        guard_bits: plan.guard_bits,
        code_block_width_exp: plan.code_block_width_exp,
        code_block_height_exp: plan.code_block_height_exp,
        progression_order: plan.progression_order,
        write_tlm: plan.write_tlm,
        block_coding_mode: match plan.block_coding_mode {
            J2kBlockCodingMode::Classic => compute::J2kLosslessCodestreamBlockCodingMode::Classic,
            J2kBlockCodingMode::HighThroughput => {
                compute::J2kLosslessCodestreamBlockCodingMode::HighThroughput
            }
        },
    };
    let codestream = match plan.block_coding_mode {
        J2kBlockCodingMode::Classic => {
            let resident_tier1 =
                compute::encode_classic_tier1_prepared_device_code_blocks_resident(
                    session, prepared,
                )?;
            compute::encode_lossless_codestream_buffer_from_resident_tier1(
                session,
                &resident_tier1,
                packetization_job,
                assembly_job,
            )?
        }
        J2kBlockCodingMode::HighThroughput => {
            let resident_tier1 =
                compute::encode_ht_prepared_device_code_blocks_resident(session, prepared)?;
            compute::encode_lossless_codestream_buffer_from_resident_tier1(
                session,
                &resident_tier1,
                packetization_job,
                assembly_job,
            )?
        }
    };
    let encode_duration = encode_started.elapsed();

    let codestream_end = codestream
        .byte_offset
        .checked_add(codestream.byte_len)
        .ok_or_else(|| crate::Error::MetalKernel {
            message: "J2K Metal codestream byte range overflows usize".to_string(),
        })?;
    let encoded = MetalEncodedJ2k::from_completed_buffer(
        codestream.buffer,
        codestream.byte_offset..codestream_end,
        codestream.capacity,
        (tile.output_width, tile.output_height),
        components,
        bit_depth,
        false,
    )?;

    let validation_duration = if options.validation == J2kEncodeValidation::CpuRoundTrip {
        let validation_started = Instant::now();
        if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous) {
            validate_lossless_roundtrip_on_metal_tile_with_session(
                tile,
                &encoded.codestream_bytes()?,
                session,
            )?;
        } else {
            validate_lossless_roundtrip_on_metal_region_with_session(
                tile,
                tile.output_width,
                tile.output_height,
                bytes_per_pixel,
                &encoded.codestream_bytes()?,
                session,
            )?;
        }
        validation_started.elapsed()
    } else {
        Duration::ZERO
    };

    Ok(Some(MetalLosslessBufferEncodeOutcome {
        encoded,
        input_copy_used: false,
        resident: MetalLosslessEncodeResidency {
            coefficient_prep_used: true,
            packetization_used: true,
            codestream_assembly_used: true,
        },
        input_copy_duration: Duration::ZERO,
        encode_duration,
        gpu_duration: codestream.gpu_duration,
        validation_duration,
    }))
}

#[cfg(target_os = "macos")]
pub(super) fn encode_lossless_tile_to_metal_buffer_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
) -> Result<MetalLosslessBufferEncodeOutcome, crate::Error> {
    validate_metal_encode_tile(tile)?;
    lossless_sample_shape(tile.format)?;
    if options.backend == EncodeBackendPreference::CpuOnly {
        return Err(crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal buffer output encode requires a device backend",
        });
    }
    let bytes_per_pixel = tile.format.bytes_per_pixel();
    if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous) {
        validate_padded_contiguous_metal_encode_tile(tile, bytes_per_pixel)?;
    }
    if let Some(outcome) = try_encode_lossless_tile_device_resident_to_metal_buffer_with_report(
        tile, options, session, staging,
    )? {
        return Ok(outcome);
    }
    Err(crate::Error::UnsupportedMetalRequest {
        reason: "J2K Metal buffer output encode requires classic padded contiguous Gray/RGB lossless input with at most one DWT level",
    })
}
