// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    compute, host_output_encode_options, lossless_sample_shape,
    should_try_auto_resident_lossless_host_encode, should_try_resident_lossless_host_encode,
    try_encode_lossless_tile_device_resident_with_report,
    validate_lossless_roundtrip_on_metal_with_session, validate_metal_encode_tile,
    validate_padded_contiguous_metal_encode_tile, BackendKind, Duration, EncodeBackendPreference,
    Instant, J2kEncodeValidation, J2kLosslessEncodeOptions, J2kLosslessSamples,
    MetalEncodeInputStaging, MetalEncodeStageAccelerator, MetalLosslessEncodeOutcome,
    MetalLosslessEncodeResidency, MetalLosslessEncodeTile,
};
#[cfg(target_os = "macos")]
use crate::error::metal_kernel_support_error;

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "host fallback keeps staging, acceleration, and dispatch reporting atomic"
)]
pub(super) fn encode_lossless_tile_with_report(
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
    staging: MetalEncodeInputStaging,
    accelerator: &mut MetalEncodeStageAccelerator,
) -> Result<MetalLosslessEncodeOutcome, crate::Error> {
    validate_metal_encode_tile(tile)?;
    let (components, bit_depth) = lossless_sample_shape(tile.format)?;
    let bytes_per_pixel = tile.format.bytes_per_pixel();
    if should_try_resident_lossless_host_encode(options)
        || should_try_auto_resident_lossless_host_encode(tile, options, staging, 1)
    {
        if let Some(outcome) =
            try_encode_lossless_tile_device_resident_with_report(tile, options, session, staging)?
        {
            return Ok(outcome);
        }
    }
    if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous)
        && options.backend == EncodeBackendPreference::RequireDevice
    {
        return Err(crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal resident encode requires classic padded contiguous Gray/RGB lossless input with at most one DWT level",
        });
    }
    let mut input_copy_used = false;
    let mut input_copy_duration = Duration::ZERO;
    let mut staged_buffer = None;
    let mut source_byte_offset = tile.byte_offset;
    if matches!(staging, MetalEncodeInputStaging::AlreadyPaddedContiguous) {
        validate_padded_contiguous_metal_encode_tile(tile, bytes_per_pixel)?;
        if !compute::buffer_is_cpu_visible(tile.buffer) {
            let copy_started = Instant::now();
            staged_buffer = Some(compute::copy_interleaved_padded_to_shared_buffer(
                compute::PaddedInterleavedCopy {
                    src_buffer: tile.buffer,
                    src_byte_offset: tile.byte_offset,
                    src_width: tile.width,
                    src_height: tile.height,
                    src_pitch_bytes: tile.pitch_bytes,
                    dst_width: tile.output_width,
                    dst_height: tile.output_height,
                    bytes_per_pixel,
                    session,
                },
            )?);
            input_copy_duration = copy_started.elapsed();
            input_copy_used = true;
            source_byte_offset = 0;
        }
    } else {
        let copy_started = Instant::now();
        staged_buffer = Some(compute::copy_interleaved_padded_to_shared_buffer(
            compute::PaddedInterleavedCopy {
                src_buffer: tile.buffer,
                src_byte_offset: tile.byte_offset,
                src_width: tile.width,
                src_height: tile.height,
                src_pitch_bytes: tile.pitch_bytes,
                dst_width: tile.output_width,
                dst_height: tile.output_height,
                bytes_per_pixel,
                session,
            },
        )?);
        input_copy_duration = copy_started.elapsed();
        input_copy_used = true;
        source_byte_offset = 0;
    }
    let buffer = staged_buffer.as_ref().unwrap_or(tile.buffer);
    let len = (tile.output_width as usize)
        .checked_mul(tile.output_height as usize)
        .and_then(|samples| samples.checked_mul(bytes_per_pixel))
        .ok_or_else(|| crate::Error::MetalKernel {
            message: "J2K Metal encode input byte length overflow".to_string(),
        })?;
    // SAFETY: Any staging copy has completed before this host readback, and
    // this route does not submit another Metal writer for the selected range.
    let data = match unsafe {
        j2k_metal_support::checked_buffer_read_vec::<u8>(buffer, source_byte_offset, len)
    } {
        Ok(data) => data,
        Err(j2k_metal_support::MetalSupportError::BufferContentsUnavailable) => {
            return Err(crate::Error::UnsupportedMetalRequest {
                reason: "J2K Metal encode input buffer is not host-visible",
            });
        }
        Err(error) => {
            return Err(metal_kernel_support_error(
                format!("J2K Metal encode input buffer view invalid: {error}"),
                error,
            ));
        }
    };
    let samples = J2kLosslessSamples::new(
        &data,
        tile.output_width,
        tile.output_height,
        u16::from(components),
        bit_depth,
        false,
    )
    .map_err(crate::Error::Decode)?;

    let encode_options = host_output_encode_options(options);
    let encode_started = Instant::now();
    let encoded = j2k::encode_j2k_lossless_with_accelerator(
        samples,
        &encode_options,
        BackendKind::Metal,
        accelerator,
    )
    .map_err(crate::Error::Decode)?;
    let encode_duration = encode_started.elapsed();
    let validation_duration = if options.validation == J2kEncodeValidation::CpuRoundTrip {
        let validation_started = Instant::now();
        validate_lossless_roundtrip_on_metal_with_session(samples, &encoded.codestream, session)?;
        validation_started.elapsed()
    } else {
        Duration::ZERO
    };
    Ok(MetalLosslessEncodeOutcome {
        encoded,
        input_copy_used,
        resident: MetalLosslessEncodeResidency {
            coefficient_prep_used: false,
            packetization_used: false,
            codestream_assembly_used: false,
        },
        input_copy_duration,
        encode_duration,
        gpu_duration: None,
        validation_duration,
        host_readback_duration: Duration::ZERO,
    })
}
