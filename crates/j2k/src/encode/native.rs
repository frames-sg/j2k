// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{format, vec::Vec};

use j2k_core::Unsupported;
use j2k_native::{
    EncodeComponentPlane as NativeEncodeComponentPlane, EncodeOptions, EncodeProgressionOrder,
    EncodeRoiRegion as NativeEncodeRoiRegion,
    EncodeTypedComponentPlane as NativeEncodeTypedComponentPlane,
};

use super::contracts::{
    EncodeBackendPreference, J2kBlockCodingMode, J2kLosslessEncodeOptions, J2kLossyEncodeOptions,
    J2kMarkerSegment, J2kProgressionOrder, ReversibleTransform,
    MAX_CLASSIC_REVERSIBLE_MARKER_BITPLANES, MAX_HTJ2K_ENCODE_BITPLANES,
    MAX_RAW_PIXEL_ENCODE_BIT_DEPTH,
};
use super::lossy::{lossy_quality_layer_byte_targets, lossy_quality_layer_count};
use super::samples::{
    raw_pixel_bytes_per_sample, J2kLosslessComponentSamples, J2kLosslessSamples,
    J2kLosslessTypedComponentSamples, J2kLossySamples, J2kRoiRegion,
};
use super::{
    j2k_lossless_decomposition_levels_for_options, j2k_lossy_decomposition_levels_for_options,
};
use crate::J2kError;

pub(super) fn validate_lossless_high_bit_options(
    samples: J2kLosslessSamples<'_>,
    options: &J2kLosslessEncodeOptions,
) -> Result<(), J2kError> {
    if samples.bit_depth <= MAX_RAW_PIXEL_ENCODE_BIT_DEPTH {
        return Ok(());
    }
    let decomposition_levels = j2k_lossless_decomposition_levels_for_options(samples, *options);
    let reversible_gain = if decomposition_levels == 0 { 0 } else { 2 };
    let coded_bitplanes = u16::from(samples.bit_depth) + reversible_gain;
    if options.block_coding_mode == J2kBlockCodingMode::HighThroughput && decomposition_levels > 0 {
        return Err(J2kError::Unsupported(Unsupported {
            what: "HTJ2K high-bit lossless encode with DWT remains blocked by the current HT integer coefficient path",
        }));
    }
    if options.block_coding_mode == J2kBlockCodingMode::HighThroughput
        && coded_bitplanes > MAX_HTJ2K_ENCODE_BITPLANES
    {
        return Err(J2kError::Unsupported(Unsupported {
            what: "HTJ2K high-bit lossless encode exceeds the current HT block bitplane limit",
        }));
    }
    if options.block_coding_mode == J2kBlockCodingMode::Classic
        && coded_bitplanes > MAX_CLASSIC_REVERSIBLE_MARKER_BITPLANES
    {
        return Err(J2kError::Unsupported(Unsupported {
            what: "25-38 bit classic lossless encode exceeds the current no-quantization guard/exponent signaling limit",
        }));
    }
    if !matches!(
        options.block_coding_mode,
        J2kBlockCodingMode::Classic | J2kBlockCodingMode::HighThroughput
    ) {
        return Err(J2kError::Unsupported(Unsupported {
            what: "25-38 bit lossless encode currently requires classic J2K or HTJ2K block coding",
        }));
    }
    if options.backend == EncodeBackendPreference::RequireDevice {
        return Err(J2kError::Unsupported(Unsupported {
            what: "25-38 bit lossless encode currently uses the CPU reversible path only",
        }));
    }
    Ok(())
}

pub(super) fn validate_lossy_high_bit_options(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
) -> Result<(), J2kError> {
    if samples.bit_depth <= MAX_RAW_PIXEL_ENCODE_BIT_DEPTH {
        return Ok(());
    }
    if options.block_coding_mode == J2kBlockCodingMode::HighThroughput {
        return Err(J2kError::Unsupported(Unsupported {
            what: "HTJ2K high-bit lossy encode remains blocked by the current HT integer coefficient path",
        }));
    }
    if options.backend == EncodeBackendPreference::RequireDevice {
        return Err(J2kError::Unsupported(Unsupported {
            what: "25-38 bit lossy encode currently uses the CPU irreversible path only",
        }));
    }
    Ok(())
}

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
    .map_err(|err| J2kError::backend(format!("JPEG 2000 lossless encode failed: {err}")))
}

pub(super) fn encode_cpu_with_roi_regions(
    samples: J2kLosslessSamples<'_>,
    options: J2kLosslessEncodeOptions,
    roi_regions: &[J2kRoiRegion],
) -> Result<Vec<u8>, J2kError> {
    let options = native_lossless_options(samples, options);
    let native_roi_regions = native_roi_regions_for_lossless_samples(samples, roi_regions)?;
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
    .map_err(map_native_lossless_roi_encode_error)
}

pub(super) fn map_native_lossless_roi_encode_error(err: &'static str) -> J2kError {
    match err {
        "ROI maxshift exceeds supported coded bitplane count" => {
            J2kError::Unsupported(Unsupported { what: err })
        }
        _ => J2kError::backend(format!("JPEG 2000 lossless ROI encode failed: {err}")),
    }
}

pub(super) fn native_roi_regions_for_lossless_samples(
    samples: J2kLosslessSamples<'_>,
    roi_regions: &[J2kRoiRegion],
) -> Result<Vec<NativeEncodeRoiRegion>, J2kError> {
    native_roi_regions_for_samples(
        samples.width,
        samples.height,
        samples.components,
        roi_regions,
    )
}

pub(super) fn native_roi_regions_for_samples(
    width: u32,
    height: u32,
    components: u16,
    roi_regions: &[J2kRoiRegion],
) -> Result<Vec<NativeEncodeRoiRegion>, J2kError> {
    roi_regions
        .iter()
        .map(|region| {
            if region.component >= components {
                return Err(J2kError::InvalidSamples {
                    what: "ROI region component index out of range".to_string(),
                });
            }
            if region.width == 0 || region.height == 0 {
                return Err(J2kError::InvalidSamples {
                    what: "ROI region dimensions must be non-zero".to_string(),
                });
            }
            if region.shift == 0 {
                return Err(J2kError::InvalidSamples {
                    what: "ROI region maxshift must be non-zero".to_string(),
                });
            }
            let x1 =
                region
                    .x
                    .checked_add(region.width)
                    .ok_or_else(|| J2kError::InvalidSamples {
                        what: "ROI region bounds overflow".to_string(),
                    })?;
            let y1 =
                region
                    .y
                    .checked_add(region.height)
                    .ok_or_else(|| J2kError::InvalidSamples {
                        what: "ROI region bounds overflow".to_string(),
                    })?;
            if region.x >= width || region.y >= height || x1 > width || y1 > height {
                return Err(J2kError::InvalidSamples {
                    what: "ROI region must be inside image bounds".to_string(),
                });
            }
            Ok(NativeEncodeRoiRegion {
                component: region.component,
                x: region.x,
                y: region.y,
                width: region.width,
                height: region.height,
                shift: region.shift,
            })
        })
        .collect()
}

pub(super) fn encode_cpu_components(
    samples: J2kLosslessComponentSamples<'_>,
    options: J2kLosslessEncodeOptions,
) -> Result<Vec<u8>, J2kError> {
    let native_options = native_lossless_component_options(samples, options);
    let planes = samples
        .planes
        .iter()
        .map(|plane| NativeEncodeComponentPlane {
            data: plane.data,
            x_rsiz: plane.x_rsiz,
            y_rsiz: plane.y_rsiz,
        })
        .collect::<Vec<_>>();
    j2k_native::encode_component_planes_53(
        &planes,
        samples.width,
        samples.height,
        samples.bit_depth,
        samples.signed,
        &native_options,
    )
    .map_err(|err| {
        J2kError::backend(format!(
            "JPEG 2000 lossless component-plane encode failed: {err}"
        ))
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
    let mut interleaved = Vec::with_capacity(capacity);
    for sample_idx in 0..pixel_count {
        let start =
            sample_idx
                .checked_mul(bytes_per_sample)
                .ok_or(J2kError::DimensionOverflow {
                    width: samples.width,
                    height: samples.height,
                })?;
        let end = start + bytes_per_sample;
        for plane in samples.planes {
            interleaved.extend_from_slice(&plane.data[start..end]);
        }
    }
    Ok(interleaved)
}

pub(super) fn encode_cpu_typed_components(
    samples: J2kLosslessTypedComponentSamples<'_>,
    options: J2kLosslessEncodeOptions,
) -> Result<Vec<u8>, J2kError> {
    let native_options = native_lossless_typed_component_options(samples, options);
    let planes = samples
        .planes
        .iter()
        .map(|plane| NativeEncodeTypedComponentPlane {
            data: plane.data,
            x_rsiz: plane.x_rsiz,
            y_rsiz: plane.y_rsiz,
            bit_depth: plane.bit_depth,
            signed: plane.signed,
        })
        .collect::<Vec<_>>();
    j2k_native::encode_typed_component_planes_53(
        &planes,
        samples.width,
        samples.height,
        &native_options,
    )
    .map_err(|err| {
        J2kError::backend(format!(
            "JPEG 2000 lossless typed component-plane encode failed: {err}"
        ))
    })
}

pub(super) fn native_lossless_options(
    samples: J2kLosslessSamples<'_>,
    options: J2kLosslessEncodeOptions,
) -> EncodeOptions {
    let progression_order = native_progression_order(options.progression);
    EncodeOptions {
        reversible: true,
        num_decomposition_levels: j2k_lossless_decomposition_levels_for_options(samples, options),
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
            && matches!(samples.components, 3 | 4),
        tile_size: options.tile_size,
        tile_part_packet_limit: options.tile_part_packet_limit,
        num_layers: options.quality_layers,
        validate_high_throughput_codestream: false,
        ..EncodeOptions::default()
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
        precinct_exponents: options.precinct_exponents.clone(),
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
