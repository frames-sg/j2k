// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use j2k_core::Unsupported;
use j2k_native::{
    EncodeComponentPlane as NativeEncodeComponentPlane, EncodeOptions, EncodeProgressionOrder,
    EncodeTypedComponentPlane as NativeEncodeTypedComponentPlane,
};

use super::allocation::{try_collect_exact, try_vec};
#[cfg(test)]
use super::contracts::EncodeBackendPreference;
use super::contracts::{
    J2kBlockCodingMode, J2kLosslessEncodeOptions, J2kLossyEncodeOptions, J2kMarkerSegment,
    J2kProgressionOrder, ReversibleTransform,
};
use super::geometry::{
    j2k_lossless_decomposition_levels_for_resident_geometry,
    j2k_lossy_decomposition_levels_for_options,
};
use super::lossy::{lossy_quality_layer_byte_targets, lossy_quality_layer_count};
use super::samples::{
    raw_pixel_bytes_per_sample, J2kLosslessComponentSamples, J2kLosslessSamples,
    J2kLosslessTypedComponentSamples, J2kLossySamples, J2kRoiRegion,
};
use crate::{J2kEncodeStageAccelerator, J2kError, J2kResidentEncodeInput};

pub(super) fn encode_cpu(
    samples: J2kLosslessSamples<'_>,
    options: J2kLosslessEncodeOptions,
) -> Result<Vec<u8>, J2kError> {
    let options = native_lossless_options(samples, options);
    j2k_native::encode(
        samples.data,
        samples.width,
        samples.height,
        samples.components,
        samples.bit_depth,
        samples.signed,
        &options,
    )
    .map_err(|source| {
        J2kError::from_native_encode_error_with_context(
            source,
            "native JPEG 2000 lossless encode failed",
        )
    })
}

pub(super) fn encode_cpu_with_roi_regions(
    samples: J2kLosslessSamples<'_>,
    options: J2kLosslessEncodeOptions,
    roi_regions: &[J2kRoiRegion],
) -> Result<Vec<u8>, J2kError> {
    let options = native_lossless_options(samples, options);
    let native_roi_regions =
        super::roi::native_roi_regions_for_lossless_samples(samples, roi_regions)?;
    j2k_native::encode_with_roi_regions(
        samples.data,
        samples.width,
        samples.height,
        samples.components,
        samples.bit_depth,
        samples.signed,
        &options,
        &native_roi_regions,
    )
    .map_err(|source| {
        J2kError::from_native_encode_error_with_context(
            source,
            "native JPEG 2000 lossless ROI encode failed",
        )
    })
}

pub(super) fn encode_cpu_components(
    samples: J2kLosslessComponentSamples<'_>,
    options: J2kLosslessEncodeOptions,
) -> Result<Vec<u8>, J2kError> {
    let native_options = native_lossless_component_options(samples, options);
    let planes = try_collect_exact(
        samples
            .planes
            .iter()
            .map(|plane| NativeEncodeComponentPlane {
                data: plane.data,
                x_rsiz: plane.x_rsiz,
                y_rsiz: plane.y_rsiz,
            }),
        "native component-plane descriptors",
    )?;
    j2k_native::encode_component_planes_53(
        &planes,
        samples.width,
        samples.height,
        samples.bit_depth,
        samples.signed,
        &native_options,
    )
    .map_err(|source| {
        J2kError::from_native_encode_error_with_context(
            source,
            "native JPEG 2000 lossless component-plane encode failed",
        )
    })
}

pub(super) fn interleave_component_planes(
    samples: J2kLosslessComponentSamples<'_>,
) -> Result<Vec<u8>, J2kError> {
    let bytes_per_sample = raw_pixel_bytes_per_sample(samples.bit_depth);
    let pixel_count = (samples.width as usize)
        .checked_mul(samples.height as usize)
        .ok_or(J2kError::DimensionOverflow {
            width: samples.width,
            height: samples.height,
        })?;
    let capacity = pixel_count
        .checked_mul(samples.planes.len())
        .and_then(|sample_count| sample_count.checked_mul(bytes_per_sample))
        .ok_or(J2kError::DimensionOverflow {
            width: samples.width,
            height: samples.height,
        })?;
    let mut interleaved = try_vec(capacity, "high-bit interleaved component samples")?;
    for sample_idx in 0..pixel_count {
        let start =
            sample_idx
                .checked_mul(bytes_per_sample)
                .ok_or(J2kError::DimensionOverflow {
                    width: samples.width,
                    height: samples.height,
                })?;
        let end = start
            .checked_add(bytes_per_sample)
            .ok_or(J2kError::DimensionOverflow {
                width: samples.width,
                height: samples.height,
            })?;
        for plane in samples.planes {
            let sample = plane
                .data
                .get(start..end)
                .ok_or(J2kError::InternalInvariant {
                    what: "validated component plane is shorter than its interleave geometry",
                })?;
            interleaved.extend_from_slice(sample);
        }
    }
    if interleaved.len() != capacity {
        return Err(J2kError::InternalInvariant {
            what: "high-bit component interleave length mismatch",
        });
    }
    Ok(interleaved)
}

pub(super) fn encode_cpu_typed_components(
    samples: J2kLosslessTypedComponentSamples<'_>,
    options: J2kLosslessEncodeOptions,
) -> Result<Vec<u8>, J2kError> {
    let native_options = native_lossless_typed_component_options(samples, options);
    let planes = try_collect_exact(
        samples
            .planes
            .iter()
            .map(|plane| NativeEncodeTypedComponentPlane {
                data: plane.data,
                x_rsiz: plane.x_rsiz,
                y_rsiz: plane.y_rsiz,
                bit_depth: plane.bit_depth,
                signed: plane.signed,
            }),
        "native typed component-plane descriptors",
    )?;
    j2k_native::encode_typed_component_planes_53(
        &planes,
        samples.width,
        samples.height,
        &native_options,
    )
    .map_err(|source| {
        J2kError::from_native_encode_error_with_context(
            source,
            "native JPEG 2000 lossless typed component-plane encode failed",
        )
    })
}

pub(super) fn native_lossless_options(
    samples: J2kLosslessSamples<'_>,
    options: J2kLosslessEncodeOptions,
) -> EncodeOptions {
    native_lossless_options_for_geometry(samples.width, samples.height, samples.components, options)
}

pub(super) fn native_lossless_resident_options(
    input: J2kResidentEncodeInput,
    options: J2kLosslessEncodeOptions,
) -> EncodeOptions {
    native_lossless_options_for_geometry(
        input.width(),
        input.height(),
        input.num_components(),
        options,
    )
}

fn native_lossless_options_for_geometry(
    width: u32,
    height: u32,
    components: u16,
    options: J2kLosslessEncodeOptions,
) -> EncodeOptions {
    let progression_order = native_progression_order(options.progression);
    EncodeOptions {
        reversible: true,
        num_decomposition_levels: j2k_lossless_decomposition_levels_for_resident_geometry(
            width, height, options,
        ),
        use_ht_block_coding: options.block_coding_mode == J2kBlockCodingMode::HighThroughput,
        progression_order,
        write_tlm: options.write_tlm || options.progression == J2kProgressionOrder::Rpcl,
        write_plt: options.write_plt,
        write_plm: options.write_plm,
        write_ppm: options.write_ppm,
        write_ppt: options.write_ppt,
        write_sop: options.write_sop,
        write_eph: options.write_eph,
        use_mct: options.reversible_transform == ReversibleTransform::Rct53
            && matches!(components, 3 | 4),
        tile_size: options.tile_size,
        tile_part_packet_limit: options.tile_part_packet_limit,
        num_layers: options.quality_layers,
        validate_high_throughput_codestream: false,
        ..EncodeOptions::default()
    }
}

pub(super) fn encode_resident_with_native_accelerator(
    input: J2kResidentEncodeInput,
    options: J2kLosslessEncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, J2kError> {
    let native_options = native_lossless_resident_options(input, options);
    j2k_native::encode_resident_htj2k_with_accelerator(input, &native_options, accelerator)
        .map_err(map_native_resident_encode_error)
}

pub(super) fn map_native_resident_encode_error(
    err: j2k_native::ResidentHtj2kEncodeError,
) -> J2kError {
    use j2k_native::ResidentHtj2kEncodeError;

    match err {
        ResidentHtj2kEncodeError::InvalidInput(what) => {
            J2kError::from_native_encode_error_with_context(
                j2k_native::EncodeError::InvalidInput { what },
                "native JPEG 2000 resident lossless encode failed",
            )
        }
        ResidentHtj2kEncodeError::Unsupported(reason) => {
            J2kError::Unsupported(Unsupported { what: reason })
        }
        ResidentHtj2kEncodeError::Declined => J2kError::Unsupported(Unsupported {
            what: "resident HTJ2K tile accelerator declined encode",
        }),
        ResidentHtj2kEncodeError::Accelerator(source) => {
            J2kError::from_native_encode_error_with_context(
                j2k_native::EncodeError::Accelerator {
                    operation: "resident HTJ2K tile encode",
                    source,
                },
                "native JPEG 2000 resident lossless encode failed",
            )
        }
        ResidentHtj2kEncodeError::Resource(source) | ResidentHtj2kEncodeError::Backend(source) => {
            J2kError::from_native_encode_error_with_context(
                source,
                "native JPEG 2000 resident lossless encode failed",
            )
        }
        _ => J2kError::NativeResidentEncode {
            context: "native JPEG 2000 resident lossless encode failed",
            source: crate::NativeBackendError::resident_encode(err),
        },
    }
}

pub(super) fn native_lossless_component_options(
    samples: J2kLosslessComponentSamples<'_>,
    options: J2kLosslessEncodeOptions,
) -> EncodeOptions {
    let interleaved_shape = J2kLosslessSamples {
        data: &[],
        width: samples.width,
        height: samples.height,
        components: samples.components(),
        bit_depth: samples.bit_depth,
        signed: samples.signed,
    };
    let mut native = native_lossless_options(interleaved_shape, options);
    native.use_mct = false;
    native
}

pub(super) fn native_lossless_typed_component_options(
    samples: J2kLosslessTypedComponentSamples<'_>,
    options: J2kLosslessEncodeOptions,
) -> EncodeOptions {
    let interleaved_shape = J2kLosslessSamples {
        data: &[],
        width: samples.width,
        height: samples.height,
        components: samples.components(),
        bit_depth: samples.max_bit_depth(),
        signed: samples.all_components_signed(),
    };
    let mut native = native_lossless_options(interleaved_shape, options);
    native.use_mct = false;
    native
}

pub(super) fn native_lossy_options(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    quantization_scale: f32,
) -> Result<EncodeOptions, J2kError> {
    let num_layers = lossy_quality_layer_count(options);
    Ok(EncodeOptions {
        reversible: false,
        num_decomposition_levels: j2k_lossy_decomposition_levels_for_options(samples, options),
        use_ht_block_coding: options.block_coding_mode == J2kBlockCodingMode::HighThroughput,
        progression_order: native_progression_order(options.progression),
        write_tlm: options.marker_segments.contains(&J2kMarkerSegment::Tlm),
        write_plt: options.marker_segments.contains(&J2kMarkerSegment::Plt),
        write_plm: options.marker_segments.contains(&J2kMarkerSegment::Plm),
        write_ppm: options.marker_segments.contains(&J2kMarkerSegment::Ppm),
        write_ppt: options.marker_segments.contains(&J2kMarkerSegment::Ppt),
        write_sop: options.marker_segments.contains(&J2kMarkerSegment::Sop),
        write_eph: options.marker_segments.contains(&J2kMarkerSegment::Eph),
        use_mct: matches!(samples.components, 3 | 4),
        num_layers,
        quality_layer_byte_targets: lossy_quality_layer_byte_targets(samples, options)?,
        tile_size: options.tile_size,
        tile_part_packet_limit: options.tile_part_packet_limit,
        precinct_exponents: try_collect_exact(
            options.precinct_exponents.iter().copied(),
            "lossy precinct exponent options",
        )?,
        validate_high_throughput_codestream: false,
        irreversible_quantization_scale: quantization_scale,
        ..EncodeOptions::default()
    })
}

pub(crate) fn native_progression_order(progression: J2kProgressionOrder) -> EncodeProgressionOrder {
    match progression {
        J2kProgressionOrder::Lrcp => EncodeProgressionOrder::Lrcp,
        J2kProgressionOrder::Rlcp => EncodeProgressionOrder::Rlcp,
        J2kProgressionOrder::Rpcl => EncodeProgressionOrder::Rpcl,
        J2kProgressionOrder::Pcrl => EncodeProgressionOrder::Pcrl,
        J2kProgressionOrder::Cprl => EncodeProgressionOrder::Cprl,
    }
}

#[cfg(test)]
mod tests;
