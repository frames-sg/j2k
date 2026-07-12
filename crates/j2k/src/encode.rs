// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_codec_math::dwt::max_decomposition_levels;
use j2k_core::{BackendKind, Unsupported};
#[cfg(test)]
use j2k_native::{DecodeSettings, Image};

use crate::{
    J2kError, {J2kEncodeDispatchReport, J2kEncodeStageAccelerator},
};

mod allocation;
mod contracts;
use self::contracts::MAX_RAW_PIXEL_ENCODE_BIT_DEPTH;
pub use self::contracts::{
    EncodeBackendPreference, EncodedJ2k, EncodedLossyJ2k, J2kBlockCodingMode, J2kEncodeValidation,
    J2kLosslessEncodeOptions, J2kLossyEncodeOptions, J2kLossyEncodeReport, J2kMarkerSegment,
    J2kProgressionOrder, J2kQualityLayer, J2kRateTarget, ReversibleTransform,
};
mod samples;
pub use self::samples::{
    J2kLosslessComponentPlane, J2kLosslessComponentSamples, J2kLosslessSamples,
    J2kLosslessTypedComponentPlane, J2kLosslessTypedComponentSamples, J2kLossySamples,
    J2kRoiRegion,
};
mod native;
use self::allocation::try_collect_exact;
#[cfg(test)]
use self::native::native_lossless_options;
pub(crate) use self::native::native_progression_order;
use self::native::{
    encode_cpu, encode_cpu_components, encode_cpu_typed_components, encode_cpu_with_roi_regions,
    interleave_component_planes, native_roi_regions_for_samples,
    validate_lossless_high_bit_options, validate_lossy_high_bit_options,
};
mod routing;
use self::routing::{
    encode_lossy_with_native_accelerator, encode_with_native_accelerator, required_encode_stages,
    required_lossy_encode_stages, resolve_accelerated_encode_backend, resolve_encode_backend,
};
mod lossy;
use self::lossy::{
    effective_lossy_target, encode_cpu_lossy, encode_cpu_lossy_with_roi_regions,
    encode_lossy_targeted, lossy_report, validate_lossy_options,
};
mod resident;
mod validation;
#[doc(hidden)]
pub use self::resident::encode_j2k_lossless_resident_with_accelerator;
use self::validation::{
    validate_lossless_component_roundtrip, validate_lossless_high_bit_component_roundtrip,
    validate_lossless_roundtrip, validate_lossless_typed_component_roundtrip,
};

/// Encode interleaved samples into a raw JPEG 2000 lossless codestream.
pub fn encode_j2k_lossless(
    samples: J2kLosslessSamples<'_>,
    options: &J2kLosslessEncodeOptions,
) -> Result<EncodedJ2k, J2kError> {
    validate_lossless_high_bit_options(samples, options)?;
    let backend = resolve_encode_backend(options.backend)?;
    let codestream = encode_cpu(samples, *options)?;
    validate_lossless_roundtrip(samples, &codestream, options.validation)?;
    Ok(EncodedJ2k {
        codestream,
        backend,
        dispatch_report: J2kEncodeDispatchReport::default(),
        width: samples.width,
        height: samples.height,
        components: samples.components,
        bit_depth: samples.bit_depth,
        signed: samples.signed,
    })
}

/// Encode interleaved samples into a raw lossless JPEG 2000 codestream with
/// rectangular ROI maxshift.
///
/// ROI encode currently uses the native CPU encoder. The produced codestream
/// is validated with the same policy as [`encode_j2k_lossless`].
pub fn encode_j2k_lossless_with_roi_regions(
    samples: J2kLosslessSamples<'_>,
    options: &J2kLosslessEncodeOptions,
    roi_regions: &[J2kRoiRegion],
) -> Result<EncodedJ2k, J2kError> {
    validate_lossless_high_bit_options(samples, options)?;
    let backend = resolve_encode_backend(options.backend)?;
    let codestream = encode_cpu_with_roi_regions(samples, *options, roi_regions)?;
    validate_lossless_roundtrip(samples, &codestream, options.validation)?;
    Ok(EncodedJ2k {
        codestream,
        backend,
        dispatch_report: J2kEncodeDispatchReport::default(),
        width: samples.width,
        height: samples.height,
        components: samples.components,
        bit_depth: samples.bit_depth,
        signed: samples.signed,
    })
}

/// Encode component-plane samples into a raw JPEG 2000 lossless codestream.
///
/// This is the lossless encode entry point for images whose component grids
/// cannot be represented as one interleaved full-resolution sample stream, such
/// as codestreams with component sampling. Components are encoded without a
/// reversible color transform.
pub fn encode_j2k_lossless_components(
    samples: J2kLosslessComponentSamples<'_>,
    options: &J2kLosslessEncodeOptions,
) -> Result<EncodedJ2k, J2kError> {
    if samples.bit_depth > MAX_RAW_PIXEL_ENCODE_BIT_DEPTH {
        return encode_j2k_lossless_components_high_bit(samples, options);
    }
    let backend = resolve_encode_backend(options.backend)?;
    let codestream = encode_cpu_components(samples, *options)?;
    validate_lossless_component_roundtrip(samples, &codestream, options.validation)?;
    Ok(EncodedJ2k {
        codestream,
        backend,
        dispatch_report: J2kEncodeDispatchReport::default(),
        width: samples.width,
        height: samples.height,
        components: samples.components(),
        bit_depth: samples.bit_depth,
        signed: samples.signed,
    })
}

fn encode_j2k_lossless_components_high_bit(
    samples: J2kLosslessComponentSamples<'_>,
    options: &J2kLosslessEncodeOptions,
) -> Result<EncodedJ2k, J2kError> {
    if samples
        .planes
        .iter()
        .any(|plane| plane.x_rsiz != 1 || plane.y_rsiz != 1)
    {
        return encode_j2k_lossless_sampled_components_high_bit(samples, options);
    }

    let interleaved = interleave_component_planes(samples)?;
    let raw_samples = J2kLosslessSamples::new(
        &interleaved,
        samples.width,
        samples.height,
        samples.components(),
        samples.bit_depth,
        samples.signed,
    )?;
    let raw_options = (*options)
        .with_reversible_transform(ReversibleTransform::None53)
        .with_validation(J2kEncodeValidation::External);
    let encoded = encode_j2k_lossless(raw_samples, &raw_options)?;
    validate_lossless_high_bit_component_roundtrip(
        samples,
        &encoded.codestream,
        options.validation,
    )?;
    Ok(encoded)
}

fn encode_j2k_lossless_sampled_components_high_bit(
    samples: J2kLosslessComponentSamples<'_>,
    options: &J2kLosslessEncodeOptions,
) -> Result<EncodedJ2k, J2kError> {
    let typed_planes = try_collect_exact(
        samples
            .planes
            .iter()
            .map(|plane| J2kLosslessTypedComponentPlane {
                data: plane.data,
                x_rsiz: plane.x_rsiz,
                y_rsiz: plane.y_rsiz,
                bit_depth: samples.bit_depth,
                signed: samples.signed,
            }),
        "high-bit typed component descriptors",
    )?;
    let typed_samples =
        J2kLosslessTypedComponentSamples::new(&typed_planes, samples.width, samples.height)?;
    encode_j2k_lossless_typed_components(typed_samples, options)
}

/// Encode typed component-plane samples into a raw JPEG 2000 lossless
/// codestream.
///
/// This is the lossless encode entry point for codestreams whose components
/// have different precision or signedness. Components are encoded without a
/// reversible color transform.
pub fn encode_j2k_lossless_typed_components(
    samples: J2kLosslessTypedComponentSamples<'_>,
    options: &J2kLosslessEncodeOptions,
) -> Result<EncodedJ2k, J2kError> {
    let backend = resolve_encode_backend(options.backend)?;
    let codestream = encode_cpu_typed_components(samples, *options)?;
    validate_lossless_typed_component_roundtrip(samples, &codestream, options.validation)?;
    Ok(EncodedJ2k {
        codestream,
        backend,
        dispatch_report: J2kEncodeDispatchReport::default(),
        width: samples.width,
        height: samples.height,
        components: samples.components(),
        bit_depth: samples.max_bit_depth(),
        signed: samples.all_components_signed(),
    })
}

/// Encode interleaved samples with an optional device encode-stage accelerator.
///
/// Accelerators return CPU fallback by reporting no dispatch. `Auto` accepts
/// that fallback; `RequireDevice` requires at least one dispatch. Any
/// accelerator error or codestream validation error is returned to the caller.
pub fn encode_j2k_lossless_with_accelerator(
    samples: J2kLosslessSamples<'_>,
    options: &J2kLosslessEncodeOptions,
    accelerated_backend: BackendKind,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<EncodedJ2k, J2kError> {
    validate_lossless_high_bit_options(samples, options)?;
    if samples.bit_depth > MAX_RAW_PIXEL_ENCODE_BIT_DEPTH {
        return Err(J2kError::Unsupported(Unsupported {
            what: "25-38 bit lossless encode currently uses the CPU classic reversible path only",
        }));
    }
    if options.backend == EncodeBackendPreference::CpuOnly {
        return encode_j2k_lossless(samples, options);
    }

    let before = accelerator.dispatch_report();
    let required_stages = required_encode_stages(samples, *options, accelerated_backend);
    let codestream = encode_with_native_accelerator(samples, *options, accelerator)?;
    let dispatch = accelerator.dispatch_report().saturating_delta(before);
    validate_lossless_roundtrip(samples, &codestream, options.validation)?;

    let backend = resolve_accelerated_encode_backend(
        options.backend,
        accelerated_backend,
        dispatch,
        required_stages,
    )?;
    Ok(EncodedJ2k {
        codestream,
        backend,
        dispatch_report: dispatch,
        width: samples.width,
        height: samples.height,
        components: samples.components,
        bit_depth: samples.bit_depth,
        signed: samples.signed,
    })
}

/// Encode interleaved samples into a raw JPEG 2000 lossy codestream.
pub fn encode_j2k_lossy(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
) -> Result<EncodedLossyJ2k, J2kError> {
    validate_lossy_options(options)?;
    validate_lossy_high_bit_options(samples, options)?;
    let target = effective_lossy_target(options)?;
    let attempt = encode_lossy_targeted(samples, options, target, |scale| {
        encode_cpu_lossy(samples, options, scale)
    })?;
    let report = lossy_report(samples, options, target, &attempt)?;
    Ok(EncodedLossyJ2k {
        codestream: attempt.codestream,
        backend: resolve_encode_backend(options.backend)?,
        dispatch_report: J2kEncodeDispatchReport::default(),
        width: samples.width,
        height: samples.height,
        components: samples.components,
        bit_depth: samples.bit_depth,
        signed: samples.signed,
        report,
    })
}

/// Encode interleaved samples into a raw lossy JPEG 2000 codestream with
/// rectangular ROI maxshift.
///
/// ROI encode currently uses the native CPU encoder and preserves the normal
/// lossy rate/PSNR reporting behavior.
pub fn encode_j2k_lossy_with_roi_regions(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    roi_regions: &[J2kRoiRegion],
) -> Result<EncodedLossyJ2k, J2kError> {
    validate_lossy_options(options)?;
    validate_lossy_high_bit_options(samples, options)?;
    let native_roi_regions = native_roi_regions_for_samples(
        samples.width,
        samples.height,
        samples.components,
        roi_regions,
    )?;
    let target = effective_lossy_target(options)?;
    let attempt = encode_lossy_targeted(samples, options, target, |scale| {
        encode_cpu_lossy_with_roi_regions(samples, options, scale, &native_roi_regions)
    })?;
    let report = lossy_report(samples, options, target, &attempt)?;
    Ok(EncodedLossyJ2k {
        codestream: attempt.codestream,
        backend: resolve_encode_backend(options.backend)?,
        dispatch_report: J2kEncodeDispatchReport::default(),
        width: samples.width,
        height: samples.height,
        components: samples.components,
        bit_depth: samples.bit_depth,
        signed: samples.signed,
        report,
    })
}

/// Encode interleaved lossy samples with an optional device encode-stage accelerator.
pub fn encode_j2k_lossy_with_accelerator(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    accelerated_backend: BackendKind,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<EncodedLossyJ2k, J2kError> {
    if options.backend == EncodeBackendPreference::CpuOnly {
        return encode_j2k_lossy(samples, options);
    }

    validate_lossy_options(options)?;
    validate_lossy_high_bit_options(samples, options)?;
    let target = effective_lossy_target(options)?;
    let required_stages = required_lossy_encode_stages(samples, options, accelerated_backend);
    let mut final_dispatch = J2kEncodeDispatchReport::default();
    let attempt = encode_lossy_targeted(samples, options, target, |scale| {
        let before = accelerator.dispatch_report();
        let result = encode_lossy_with_native_accelerator(samples, options, scale, accelerator);
        final_dispatch = accelerator.dispatch_report().saturating_delta(before);
        result
    })?;
    let backend = resolve_accelerated_encode_backend(
        options.backend,
        accelerated_backend,
        final_dispatch,
        required_stages,
    )?;
    let report = lossy_report(samples, options, target, &attempt)?;
    Ok(EncodedLossyJ2k {
        codestream: attempt.codestream,
        backend,
        dispatch_report: final_dispatch,
        width: samples.width,
        height: samples.height,
        components: samples.components,
        bit_depth: samples.bit_depth,
        signed: samples.signed,
        report,
    })
}

const MIN_LOSSLESS_DWT_DIMENSION: u32 = 64;

/// Return the default lossless decomposition level policy used by the facade.
pub fn j2k_lossless_decomposition_levels(samples: J2kLosslessSamples<'_>) -> u8 {
    j2k_lossless_decomposition_levels_for_progression(samples, J2kProgressionOrder::Lrcp)
}

/// Return the default lossless decomposition level policy for a progression.
pub fn j2k_lossless_decomposition_levels_for_progression(
    samples: J2kLosslessSamples<'_>,
    progression: J2kProgressionOrder,
) -> u8 {
    j2k_lossless_decomposition_levels_for_geometry(samples.width, samples.height, progression)
}

fn j2k_lossless_decomposition_levels_for_geometry(
    width: u32,
    height: u32,
    progression: J2kProgressionOrder,
) -> u8 {
    if matches!(
        progression,
        J2kProgressionOrder::Rpcl | J2kProgressionOrder::Pcrl | J2kProgressionOrder::Cprl
    ) {
        return j2k_rpcl_lossless_decomposition_levels(width, height);
    }

    if width.min(height) < MIN_LOSSLESS_DWT_DIMENSION {
        return 0;
    }

    1
}

fn j2k_lossy_decomposition_levels_for_options(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
) -> u8 {
    let levels = if matches!(
        options.progression,
        J2kProgressionOrder::Rpcl | J2kProgressionOrder::Pcrl | J2kProgressionOrder::Cprl
    ) {
        j2k_lossy_position_progression_decomposition_levels(samples)
    } else {
        u8::from(samples.width.min(samples.height) >= MIN_LOSSLESS_DWT_DIMENSION)
    };
    options.max_decomposition_levels.map_or(levels, |max| {
        levels
            .min(max)
            .min(max_decomposition_levels(samples.width, samples.height))
    })
}

fn j2k_lossy_position_progression_decomposition_levels(samples: J2kLossySamples<'_>) -> u8 {
    j2k_rpcl_lossless_decomposition_levels(samples.width, samples.height)
}

/// Return the effective lossless decomposition level policy for encode options.
pub fn j2k_lossless_decomposition_levels_for_options(
    samples: J2kLosslessSamples<'_>,
    options: J2kLosslessEncodeOptions,
) -> u8 {
    j2k_lossless_decomposition_levels_for_resident_geometry(samples.width, samples.height, options)
}

pub(super) fn j2k_lossless_decomposition_levels_for_resident_geometry(
    width: u32,
    height: u32,
    options: J2kLosslessEncodeOptions,
) -> u8 {
    let levels = j2k_lossless_decomposition_levels_for_geometry(width, height, options.progression);
    options
        .max_decomposition_levels
        .map_or(levels, |requested| {
            if width.min(height) < MIN_LOSSLESS_DWT_DIMENSION {
                return 0;
            }
            requested.min(max_decomposition_levels(width, height))
        })
}

fn j2k_rpcl_lossless_decomposition_levels(width: u32, height: u32) -> u8 {
    let mut levels = 0u8;
    let mut current_width = width;
    let mut current_height = height;
    let max_levels = max_decomposition_levels(width, height);

    while current_width.min(current_height) > MIN_LOSSLESS_DWT_DIMENSION && levels < max_levels {
        current_width = current_width.div_ceil(2);
        current_height = current_height.div_ceil(2);
        levels += 1;
    }

    levels
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::{
        encode_j2k_lossless, j2k_lossless_decomposition_levels_for_options,
        native_lossless_options, DecodeSettings, EncodeBackendPreference, Image,
        J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions, J2kLosslessSamples,
        J2kProgressionOrder, ReversibleTransform,
    };

    fn cod_mct(codestream: &[u8]) -> u8 {
        let cod_offset = codestream
            .windows(2)
            .position(|window| window == [0xFF, 0x52])
            .expect("COD marker");
        codestream[cod_offset + 8]
    }

    #[test]
    fn lossless_encode_can_disable_component_transform() {
        let pixels: Vec<u8> = (0..4 * 4 * 3)
            .map(|value| u8::try_from((value * 17) & 0xFF).expect("masked fixture byte"))
            .collect();
        let samples = J2kLosslessSamples::new(&pixels, 4, 4, 3, 8, false).unwrap();
        let encoded = encode_j2k_lossless(
            samples,
            &J2kLosslessEncodeOptions {
                block_coding_mode: J2kBlockCodingMode::Classic,
                progression: J2kProgressionOrder::Lrcp,
                max_decomposition_levels: Some(0),
                reversible_transform: ReversibleTransform::None53,
                validation: J2kEncodeValidation::CpuRoundTrip,
                ..J2kLosslessEncodeOptions::default()
            },
        )
        .unwrap();

        assert_eq!(cod_mct(&encoded.codestream), 0);
    }

    #[test]
    fn explicit_decomposition_levels_override_default_lrcp_policy() {
        let pixels = vec![0; 128 * 128];
        let samples = J2kLosslessSamples::new(&pixels, 128, 128, 1, 8, false).unwrap();

        let levels = j2k_lossless_decomposition_levels_for_options(
            samples,
            J2kLosslessEncodeOptions {
                block_coding_mode: J2kBlockCodingMode::Classic,
                progression: J2kProgressionOrder::Lrcp,
                max_decomposition_levels: Some(5),
                ..J2kLosslessEncodeOptions::default()
            },
        );

        assert_eq!(levels, 5);
    }

    #[test]
    fn facade_native_options_skip_internal_ht_validation_for_external_validation() {
        let pixels = vec![0; 64 * 64];
        let samples = J2kLosslessSamples::new(&pixels, 64, 64, 1, 8, false).unwrap();

        let external = native_lossless_options(
            samples,
            J2kLosslessEncodeOptions {
                block_coding_mode: J2kBlockCodingMode::HighThroughput,
                validation: J2kEncodeValidation::External,
                ..J2kLosslessEncodeOptions::default()
            },
        );
        let roundtrip = native_lossless_options(
            samples,
            J2kLosslessEncodeOptions {
                block_coding_mode: J2kBlockCodingMode::HighThroughput,
                validation: J2kEncodeValidation::CpuRoundTrip,
                ..J2kLosslessEncodeOptions::default()
            },
        );

        assert!(!external.validate_high_throughput_codestream);
        assert!(!roundtrip.validate_high_throughput_codestream);
    }

    #[test]
    fn lossless_facade_roundtrips_four_component_via_public_api() {
        let width: u32 = 32;
        let height: u32 = 24;
        let components: u16 = 4;

        // Deterministic 4-component (RGBA/CMYK) 8-bit input, distinct per plane.
        let mut pixels = Vec::with_capacity((width * height * u32::from(components)) as usize);
        for y in 0..height {
            for x in 0..width {
                for c in 0..u32::from(components) {
                    let value = (x.wrapping_mul(7) ^ y.wrapping_mul(13)).wrapping_add(c * 41);
                    pixels.push((value & 0xFF) as u8);
                }
            }
        }

        // MUST go through the real public constructor.
        let samples = J2kLosslessSamples::new(&pixels, width, height, components, 8, false)
            .expect("4-component samples must be accepted by the public constructor");

        // Encode via the public CPU lossless entry.
        let encoded = encode_j2k_lossless(
            samples,
            &J2kLosslessEncodeOptions {
                backend: EncodeBackendPreference::CpuOnly,
                validation: J2kEncodeValidation::CpuRoundTrip,
                ..J2kLosslessEncodeOptions::default()
            },
        )
        .expect("4-component CPU lossless encode must succeed");

        assert_eq!(encoded.components, components);

        // Decode the bytes with the native decoder and assert an exact round-trip.
        let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
            .expect("native decode of 4-component codestream must construct")
            .decode_native()
            .expect("native decode of 4-component codestream must succeed");

        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        assert_eq!(decoded.num_components, components);
        assert_eq!(decoded.bit_depth, 8);
        assert_eq!(
            decoded.data, pixels,
            "4-component pixels must round-trip exactly"
        );

        // 2-component is accepted and handled as independent channels without MCT.
        let two_component = vec![0u8; (width * height * 2) as usize];
        let two_component = J2kLosslessSamples::new(&two_component, width, height, 2, 8, false)
            .expect("2-component samples must be accepted by the public constructor");
        assert_eq!(two_component.components, 2);
    }
}
