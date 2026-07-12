// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    bail, checked_decode_byte_len3, internal_j2k_code_block_style, internal_j2k_sub_band_type, j2c,
    EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock, HtCleanupEncodeDistribution, J2kCodeBlockSegment,
    J2kCodeBlockStyle, J2kForwardDwt53Level, J2kForwardDwt53Output, J2kForwardDwt97Level,
    J2kForwardDwt97Output, J2kPacketizationEncodeJob, J2kSubBandType, J2kTier1TokenSegment, Result,
    ValidationError, Vec, MAX_DEINTERLEAVE_REFERENCE_BIT_DEPTH,
};
use crate::{DecodingError, EncodeError, EncodeResult};

/// Adapter scalar classic J2K encoder helper for backend experimentation.
#[doc(hidden)]
pub fn encode_j2k_code_block_scalar_with_style(
    coefficients: &[i32],
    width: u32,
    height: u32,
    sub_band_type: J2kSubBandType,
    total_bitplanes: u8,
    style: J2kCodeBlockStyle,
) -> EncodeResult<EncodedJ2kCodeBlock> {
    let encoded = j2c::bitplane_encode::try_encode_code_block_segments_with_style(
        coefficients,
        width,
        height,
        internal_j2k_sub_band_type(sub_band_type),
        total_bitplanes,
        &internal_j2k_code_block_style(style),
    )?;
    try_encoded_j2k_code_block_from_internal(encoded)
}

/// Adapter scalar Classic Tier-1 compact token packer for backend experimentation.
///
/// The token format matches the Metal Classic Tier-1 token-emitter contract:
/// arithmetic segments are 6-bit `(context_label, bit)` MQ tokens, while raw
/// bypass segments are one bit per raw bypass event.
#[doc(hidden)]
pub fn pack_j2k_code_block_scalar_from_tier1_tokens(
    token_bytes: &[u8],
    token_segments: &[J2kTier1TokenSegment],
    number_of_coding_passes: u8,
    missing_bit_planes: u8,
) -> EncodeResult<EncodedJ2kCodeBlock> {
    let encoded = j2c::bitplane_encode::try_pack_classic_selective_bypass_tier1_tokens(
        token_bytes,
        token_segments,
        number_of_coding_passes,
        missing_bit_planes,
    )?;
    try_encoded_j2k_code_block_from_internal(encoded)
}

fn try_encoded_j2k_code_block_from_internal(
    encoded: j2c::bitplane_encode::EncodedCodeBlockWithSegments,
) -> EncodeResult<EncodedJ2kCodeBlock> {
    let mut segments = j2c::encode::allocation::try_untracked_vec(
        encoded.segments.len(),
        "public classic Tier-1 segment metadata",
    )?;
    for segment in encoded.segments {
        segments.push(J2kCodeBlockSegment {
            data_offset: segment.data_offset,
            data_length: segment.data_length,
            start_coding_pass: segment.start_coding_pass,
            end_coding_pass: segment.end_coding_pass,
            use_arithmetic: segment.use_arithmetic,
        });
    }

    Ok(EncodedJ2kCodeBlock {
        data: encoded.data,
        segments,
        number_of_coding_passes: encoded.num_coding_passes,
        missing_bit_planes: encoded.num_zero_bitplanes,
    })
}

/// Adapter scalar HTJ2K cleanup-only encoder helper for backend experimentation.
#[doc(hidden)]
pub fn encode_ht_code_block_scalar(
    coefficients: &[i32],
    width: u32,
    height: u32,
    total_bitplanes: u8,
) -> EncodeResult<EncodedHtJ2kCodeBlock> {
    let encoded =
        j2c::ht_block_encode::try_encode_code_block(coefficients, width, height, total_bitplanes)?;
    Ok(EncodedHtJ2kCodeBlock {
        data: encoded.data,
        cleanup_length: encoded.ht_cleanup_length,
        refinement_length: encoded.ht_refinement_length,
        num_coding_passes: encoded.num_coding_passes,
        num_zero_bitplanes: encoded.num_zero_bitplanes,
    })
}

/// Adapter scalar HTJ2K encoder helper with an explicit coding-pass request.
#[doc(hidden)]
pub fn encode_ht_code_block_scalar_with_passes(
    coefficients: &[i32],
    width: u32,
    height: u32,
    total_bitplanes: u8,
    target_coding_passes: u8,
) -> EncodeResult<EncodedHtJ2kCodeBlock> {
    let encoded = j2c::ht_block_encode::try_encode_code_block_with_passes(
        coefficients,
        width,
        height,
        total_bitplanes,
        target_coding_passes,
    )?;
    Ok(EncodedHtJ2kCodeBlock {
        data: encoded.data,
        cleanup_length: encoded.ht_cleanup_length,
        refinement_length: encoded.ht_refinement_length,
        num_coding_passes: encoded.num_coding_passes,
        num_zero_bitplanes: encoded.num_zero_bitplanes,
    })
}

/// Adapter HTJ2K cleanup-encode distribution helper for benchmark tuning.
#[doc(hidden)]
pub fn collect_ht_cleanup_encode_distribution(
    coefficients: &[i32],
    width: u32,
    height: u32,
    total_bitplanes: u8,
) -> EncodeResult<HtCleanupEncodeDistribution> {
    j2c::ht_block_encode::collect_encode_distribution(coefficients, width, height, total_bitplanes)
}

/// Adapter scalar forward 5/3 DWT reference for CUDA stage parity.
///
/// Runs the native CPU reversible 5/3 forward DWT on `samples` and returns
/// the decomposed subbands packed into the public `J2kForwardDwt53Output`
/// type.  The returned layout matches what the encoder feeds to Tier-1.
///
/// # Errors
///
/// Returns a typed error for invalid caller geometry, arithmetic overflow, or
/// a host allocation failure while materializing the decomposed reference.
#[doc(hidden)]
pub fn forward_dwt53_reference(
    samples: &[f32],
    width: u32,
    height: u32,
    num_levels: u8,
) -> EncodeResult<J2kForwardDwt53Output> {
    let decomposition = j2c::fdwt::try_forward_dwt(samples, width, height, num_levels, true)?;
    try_public_dwt_output::<Dwt53OutputAdapter>(decomposition)
}

/// Adapter scalar forward 9/7 DWT reference for Metal/CUDA stage parity.
///
/// Runs the native CPU irreversible 9/7 forward DWT on `samples` and returns
/// the decomposed subbands packed into the public `J2kForwardDwt97Output`
/// type. The returned layout matches what the encoder feeds to Tier-1.
///
/// # Errors
///
/// Returns a typed error for invalid caller geometry, arithmetic overflow, or
/// a host allocation failure while materializing the decomposed reference.
#[doc(hidden)]
pub fn forward_dwt97_reference(
    samples: &[f32],
    width: u32,
    height: u32,
    num_levels: u8,
) -> EncodeResult<J2kForwardDwt97Output> {
    let decomposition = j2c::fdwt::try_forward_dwt(samples, width, height, num_levels, false)?;
    try_public_dwt_output::<Dwt97OutputAdapter>(decomposition)
}

trait PublicDwtOutputAdapter {
    type Level;
    type Output;

    fn level(level: j2c::fdwt::DwtLevel) -> Self::Level;
    fn output(
        ll: Vec<f32>,
        ll_width: u32,
        ll_height: u32,
        levels: Vec<Self::Level>,
    ) -> Self::Output;
}

struct Dwt53OutputAdapter;

impl PublicDwtOutputAdapter for Dwt53OutputAdapter {
    type Level = J2kForwardDwt53Level;
    type Output = J2kForwardDwt53Output;

    fn level(level: j2c::fdwt::DwtLevel) -> Self::Level {
        J2kForwardDwt53Level {
            hl: level.hl,
            lh: level.lh,
            hh: level.hh,
            width: level.low_width + level.high_width,
            height: level.low_height + level.high_height,
            low_width: level.low_width,
            low_height: level.low_height,
            high_width: level.high_width,
            high_height: level.high_height,
        }
    }

    fn output(
        ll: Vec<f32>,
        ll_width: u32,
        ll_height: u32,
        levels: Vec<Self::Level>,
    ) -> Self::Output {
        J2kForwardDwt53Output {
            ll,
            ll_width,
            ll_height,
            levels,
        }
    }
}

struct Dwt97OutputAdapter;

impl PublicDwtOutputAdapter for Dwt97OutputAdapter {
    type Level = J2kForwardDwt97Level;
    type Output = J2kForwardDwt97Output;

    fn level(level: j2c::fdwt::DwtLevel) -> Self::Level {
        J2kForwardDwt97Level {
            hl: level.hl,
            lh: level.lh,
            hh: level.hh,
            width: level.low_width + level.high_width,
            height: level.low_height + level.high_height,
            low_width: level.low_width,
            low_height: level.low_height,
            high_width: level.high_width,
            high_height: level.high_height,
        }
    }

    fn output(
        ll: Vec<f32>,
        ll_width: u32,
        ll_height: u32,
        levels: Vec<Self::Level>,
    ) -> Self::Output {
        J2kForwardDwt97Output {
            ll,
            ll_width,
            ll_height,
            levels,
        }
    }
}

fn try_public_dwt_output<A: PublicDwtOutputAdapter>(
    decomposition: j2c::fdwt::DwtDecomposition,
) -> EncodeResult<A::Output> {
    let level_count = decomposition.levels.len();
    let requested_bytes = level_count
        .checked_mul(core::mem::size_of::<A::Level>())
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "scalar DWT public level owner bytes",
        })?;
    let mut levels = Vec::new();
    levels
        .try_reserve_exact(level_count)
        .map_err(|_| EncodeError::HostAllocationFailed {
            what: "scalar DWT public level owners",
            bytes: requested_bytes,
        })?;
    levels.extend(decomposition.levels.into_iter().map(A::level));
    Ok(A::output(
        decomposition.ll,
        decomposition.ll_width,
        decomposition.ll_height,
        levels,
    ))
}

/// Adapter scalar forward RCT reference for CUDA stage parity.
///
/// Applies the native CPU forward Reversible Color Transform to three
/// component planes supplied as owned `Vec<f32>` arrays.  The transform is
/// applied in place and the mutated planes are returned, so callers do not
/// need to pass a mutable slice.
#[doc(hidden)]
#[must_use]
pub fn forward_rct_reference(mut planes: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    j2c::forward_mct::forward_rct(&mut planes);
    planes
}

/// Adapter scalar forward ICT reference for Metal/CUDA stage parity.
///
/// Applies the native CPU forward Irreversible Color Transform to three
/// component planes supplied as owned `Vec<f32>` arrays. The transform is
/// applied in place and the mutated planes are returned.
#[doc(hidden)]
#[must_use]
pub fn forward_ict_reference(mut planes: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    j2c::forward_mct::forward_ict(&mut planes);
    planes
}

/// Adapter scalar sub-band quantization reference for backend stage parity.
#[doc(hidden)]
#[must_use]
pub fn quantize_subband_reference(
    coefficients: &[f32],
    step_exponent: u16,
    step_mantissa: u16,
    range_bits: u8,
    reversible: bool,
) -> Vec<i32> {
    let step = j2c::quantize::QuantStepSize {
        exponent: step_exponent,
        mantissa: step_mantissa,
    };
    j2c::quantize::quantize_subband(coefficients, &step, range_bits, reversible)
}

/// Adapter scalar reversible sub-band quantization reference for CUDA stage parity.
///
/// Quantizes `coefficients` using the reversible (lossless) integer path of
/// the native CPU quantizer.  `step_exponent` and `step_mantissa` encode the
/// JPEG 2000 `QuantStepSize` for the sub-band; `range_bits` is the nominal
/// bit depth for the sub-band.  When `reversible` is `true` the step-size
/// parameters are ignored and each coefficient is rounded to the nearest
/// integer.
#[doc(hidden)]
#[must_use]
pub fn quantize_reversible_reference(
    coefficients: &[f32],
    step_exponent: u16,
    step_mantissa: u16,
    range_bits: u8,
    reversible: bool,
) -> Vec<i32> {
    quantize_subband_reference(
        coefficients,
        step_exponent,
        step_mantissa,
        range_bits,
        reversible,
    )
}

fn checked_deinterleave_reference_bytes_per_sample(bit_depth: u8) -> Result<usize> {
    if bit_depth == 0 || bit_depth > MAX_DEINTERLEAVE_REFERENCE_BIT_DEPTH {
        bail!(ValidationError::InvalidComponentMetadata);
    }
    Ok(usize::from(bit_depth).div_ceil(8).max(1))
}

/// Checked adapter scalar pixel deinterleave/level-shift reference for backend
/// stage parity.
///
/// Converts interleaved pixel bytes to per-component f32 planes with the
/// same level-shift logic as the native CPU encode path.  The result is one
/// `Vec<f32>` per component, each of length `num_pixels`.
///
/// The input byte slice must exactly contain
/// `num_pixels * num_components * bytes_per_sample(bit_depth)` bytes.
#[doc(hidden)]
pub fn try_deinterleave_reference(
    pixels: &[u8],
    num_pixels: usize,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
) -> Result<Vec<Vec<f32>>> {
    if num_components == 0 {
        bail!(ValidationError::InvalidComponentMetadata);
    }
    let bytes_per_sample = checked_deinterleave_reference_bytes_per_sample(bit_depth)?;
    let expected_len =
        checked_decode_byte_len3(num_pixels, usize::from(num_components), bytes_per_sample)?;
    if pixels.len() != expected_len {
        bail!(ValidationError::InvalidComponentMetadata);
    }
    j2c::encode::try_deinterleave_to_f32(pixels, num_pixels, num_components, bit_depth, signed)
        .map_err(|error| map_deinterleave_encode_error(&error))
}

fn map_deinterleave_encode_error(error: &EncodeError) -> crate::DecodeError {
    match error {
        EncodeError::InvalidInput { .. } | EncodeError::Unsupported { .. } => {
            ValidationError::InvalidComponentMetadata.into()
        }
        EncodeError::ArithmeticOverflow { .. } | EncodeError::AllocationTooLarge { .. } => {
            ValidationError::ImageTooLarge.into()
        }
        EncodeError::HostAllocationFailed { .. } => DecodingError::HostAllocationFailed.into(),
        EncodeError::Accelerator { .. }
        | EncodeError::CodestreamValidation { .. }
        | EncodeError::InternalInvariant { .. } => {
            DecodingError::CodeBlockDecodeFailureWithContext(
                "scalar deinterleave returned an unexpected encode-stage error",
            )
            .into()
        }
    }
}

/// Adapter scalar Tier-2 packetization helper for backend experimentation.
#[doc(hidden)]
pub fn encode_j2k_packetization_scalar(
    job: J2kPacketizationEncodeJob<'_>,
) -> crate::EncodeResult<Vec<u8>> {
    j2c::packet_encode::form_borrowed_packetization_scalar(job, 0)
}
