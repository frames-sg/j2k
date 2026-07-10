// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    estimate_resident_lossless_encode_peak_bytes, lossless_device_coefficient_count,
    lossless_device_encode_plan, lossless_sample_shape,
    packet_descriptors_for_lossless_device_order,
    resident_packetization_resolutions_from_lossless_device_plan, validate_metal_encode_tile,
    validate_padded_contiguous_metal_encode_tile, EncodeBackendPreference,
    J2kLosslessEncodeOptions, MetalEncodeInputStaging, MetalLosslessEncodeTile,
    OwnedMetalLosslessEncodeTile, PlannedResidentLosslessBufferEncode,
    ResidentLosslessBufferEncodeMetadata, RESIDENT_CLASSIC_CODE_BLOCK_EDGE,
};

#[cfg(target_os = "macos")]
pub(super) fn plan_resident_lossless_buffer_encode(
    index: usize,
    tile: MetalLosslessEncodeTile<'_>,
    options: J2kLosslessEncodeOptions,
    staging: MetalEncodeInputStaging,
) -> Result<Option<PlannedResidentLosslessBufferEncode>, crate::Error> {
    validate_metal_encode_tile(tile)?;
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
    let Some(plan) = lossless_device_encode_plan(
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
    let coefficient_count = lossless_device_coefficient_count(&plan.code_blocks)?;
    let packetization_resolutions =
        resident_packetization_resolutions_from_lossless_device_plan(&plan)?;
    let packet_descriptors = packet_descriptors_for_lossless_device_order(
        plan.resolutions.len(),
        plan.components,
        plan.progression_order,
    )?;
    let metadata = ResidentLosslessBufferEncodeMetadata {
        tile: OwnedMetalLosslessEncodeTile::from_tile(tile),
        components,
        bit_depth,
        bytes_per_pixel,
        plan,
        packet_descriptors,
        packetization_resolutions,
    };
    let estimated_peak_bytes =
        estimate_resident_lossless_encode_peak_bytes(&metadata, coefficient_count, staging);
    Ok(Some(PlannedResidentLosslessBufferEncode {
        index,
        metadata,
        coefficient_count,
        bytes_per_sample,
        estimated_peak_bytes,
        #[cfg(test)]
        failure_injection_index: super::test_resident_encode_failure_index(),
    }))
}
