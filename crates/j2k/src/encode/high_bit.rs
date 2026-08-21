// SPDX-License-Identifier: MIT OR Apache-2.0

//! High-bit encode validation and component adaptation.

use j2k_core::Unsupported;

use super::allocation::try_collect_exact;
use super::contracts::{
    EncodeBackendPreference, J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions,
    J2kLossyEncodeOptions, ReversibleTransform, MAX_CLASSIC_REVERSIBLE_MARKER_BITPLANES,
    MAX_HTJ2K_ENCODE_BITPLANES, MAX_RAW_PIXEL_ENCODE_BIT_DEPTH,
};
use super::cpu::interleave_component_planes;
use super::geometry::j2k_lossless_decomposition_levels_for_options;
use super::samples::{
    J2kLosslessComponentPlane, J2kLosslessComponentSamples, J2kLosslessSamples,
    J2kLosslessTypedComponentPlane, J2kLosslessTypedComponentSamples, J2kLossySamples,
};
use super::validation::validate_lossless_high_bit_component_roundtrip;
use crate::{EncodedJ2k, J2kError};

pub(super) fn encode_components(
    samples: J2kLosslessComponentSamples<'_>,
    options: &J2kLosslessEncodeOptions,
) -> Result<EncodedJ2k, J2kError> {
    if samples
        .planes
        .iter()
        .any(|plane| plane.x_rsiz != 1 || plane.y_rsiz != 1)
    {
        return encode_sampled_components(samples, options);
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
    let encoded = super::lossless::encode(raw_samples, &raw_options)?;
    validate_lossless_high_bit_component_roundtrip(
        samples,
        &encoded.codestream,
        options.validation,
    )?;
    Ok(encoded)
}

fn encode_sampled_components(
    samples: J2kLosslessComponentSamples<'_>,
    options: &J2kLosslessEncodeOptions,
) -> Result<EncodedJ2k, J2kError> {
    let typed_planes = try_collect_exact(
        samples
            .planes
            .iter()
            .map(
                |plane: &J2kLosslessComponentPlane<'_>| J2kLosslessTypedComponentPlane {
                    data: plane.data,
                    x_rsiz: plane.x_rsiz,
                    y_rsiz: plane.y_rsiz,
                    bit_depth: samples.bit_depth,
                    signed: samples.signed,
                },
            ),
        "high-bit typed component descriptors",
    )?;
    let typed_samples =
        J2kLosslessTypedComponentSamples::new(&typed_planes, samples.width, samples.height)?;
    super::lossless::encode_typed_components(typed_samples, options)
}

pub(super) fn validate_lossless_options(
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

pub(super) fn validate_lossy_options(
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
