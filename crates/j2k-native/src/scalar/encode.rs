// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    bail, checked_decode_byte_len3, internal_j2k_code_block_style, internal_j2k_sub_band_type, j2c,
    EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock, HtCleanupEncodeDistribution, J2kCodeBlockSegment,
    J2kCodeBlockStyle, J2kForwardDwt53Level, J2kForwardDwt53Output, J2kForwardDwt97Level,
    J2kForwardDwt97Output, J2kPacketizationBlockCodingMode, J2kPacketizationEncodeJob,
    J2kSubBandType, J2kTier1TokenSegment, Result, ValidationError, Vec,
    MAX_DEINTERLEAVE_REFERENCE_BIT_DEPTH,
};

/// Adapter scalar classic J2K encoder helper for backend experimentation.
#[doc(hidden)]
pub fn encode_j2k_code_block_scalar_with_style(
    coefficients: &[i32],
    width: u32,
    height: u32,
    sub_band_type: J2kSubBandType,
    total_bitplanes: u8,
    style: J2kCodeBlockStyle,
) -> core::result::Result<EncodedJ2kCodeBlock, &'static str> {
    let encoded = j2c::bitplane_encode::encode_code_block_segments_with_style(
        coefficients,
        width,
        height,
        internal_j2k_sub_band_type(sub_band_type),
        total_bitplanes,
        &internal_j2k_code_block_style(style),
    );
    Ok(encoded_j2k_code_block_from_internal(encoded))
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
) -> core::result::Result<EncodedJ2kCodeBlock, &'static str> {
    let internal_segments = token_segments
        .iter()
        .map(|segment| j2c::bitplane_encode::ClassicTier1TokenSegment {
            token_bit_offset: segment.token_bit_offset,
            token_bit_count: segment.token_bit_count,
            start_coding_pass: segment.start_coding_pass,
            end_coding_pass: segment.end_coding_pass,
            use_arithmetic: segment.use_arithmetic,
        })
        .collect::<Vec<_>>();
    let encoded = j2c::bitplane_encode::pack_classic_selective_bypass_tier1_tokens(
        token_bytes,
        &internal_segments,
        number_of_coding_passes,
        missing_bit_planes,
    )?;
    Ok(encoded_j2k_code_block_from_internal(encoded))
}

fn encoded_j2k_code_block_from_internal(
    encoded: j2c::bitplane_encode::EncodedCodeBlockWithSegments,
) -> EncodedJ2kCodeBlock {
    let segments = encoded
        .segments
        .into_iter()
        .map(|segment| J2kCodeBlockSegment {
            data_offset: segment.data_offset,
            data_length: segment.data_length,
            start_coding_pass: segment.start_coding_pass,
            end_coding_pass: segment.end_coding_pass,
            use_arithmetic: segment.use_arithmetic,
        })
        .collect();

    EncodedJ2kCodeBlock {
        data: encoded.data,
        segments,
        number_of_coding_passes: encoded.num_coding_passes,
        missing_bit_planes: encoded.num_zero_bitplanes,
    }
}

/// Adapter scalar HTJ2K cleanup-only encoder helper for backend experimentation.
#[doc(hidden)]
pub fn encode_ht_code_block_scalar(
    coefficients: &[i32],
    width: u32,
    height: u32,
    total_bitplanes: u8,
) -> core::result::Result<EncodedHtJ2kCodeBlock, &'static str> {
    let encoded =
        j2c::ht_block_encode::encode_code_block(coefficients, width, height, total_bitplanes)?;
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
) -> core::result::Result<EncodedHtJ2kCodeBlock, &'static str> {
    let encoded = j2c::ht_block_encode::encode_code_block_with_passes(
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
) -> core::result::Result<HtCleanupEncodeDistribution, &'static str> {
    j2c::ht_block_encode::collect_encode_distribution(coefficients, width, height, total_bitplanes)
}

/// Adapter scalar forward 5/3 DWT reference for CUDA stage parity.
///
/// Runs the native CPU reversible 5/3 forward DWT on `samples` and returns
/// the decomposed subbands packed into the public `J2kForwardDwt53Output`
/// type.  The returned layout matches what the encoder feeds to Tier-1.
#[doc(hidden)]
pub fn forward_dwt53_reference(
    samples: &[f32],
    width: u32,
    height: u32,
    num_levels: u8,
) -> J2kForwardDwt53Output {
    let decomp = j2c::fdwt::forward_dwt(samples, width, height, num_levels, true);
    let levels = decomp
        .levels
        .into_iter()
        .map(|lvl| J2kForwardDwt53Level {
            hl: lvl.hl,
            lh: lvl.lh,
            hh: lvl.hh,
            width: lvl.low_width + lvl.high_width,
            height: lvl.low_height + lvl.high_height,
            low_width: lvl.low_width,
            low_height: lvl.low_height,
            high_width: lvl.high_width,
            high_height: lvl.high_height,
        })
        .collect();
    J2kForwardDwt53Output {
        ll: decomp.ll,
        ll_width: decomp.ll_width,
        ll_height: decomp.ll_height,
        levels,
    }
}

/// Adapter scalar forward 9/7 DWT reference for Metal/CUDA stage parity.
///
/// Runs the native CPU irreversible 9/7 forward DWT on `samples` and returns
/// the decomposed subbands packed into the public `J2kForwardDwt97Output`
/// type. The returned layout matches what the encoder feeds to Tier-1.
#[doc(hidden)]
pub fn forward_dwt97_reference(
    samples: &[f32],
    width: u32,
    height: u32,
    num_levels: u8,
) -> J2kForwardDwt97Output {
    let decomp = j2c::fdwt::forward_dwt(samples, width, height, num_levels, false);
    let levels = decomp
        .levels
        .into_iter()
        .map(|lvl| J2kForwardDwt97Level {
            hl: lvl.hl,
            lh: lvl.lh,
            hh: lvl.hh,
            width: lvl.low_width + lvl.high_width,
            height: lvl.low_height + lvl.high_height,
            low_width: lvl.low_width,
            low_height: lvl.low_height,
            high_width: lvl.high_width,
            high_height: lvl.high_height,
        })
        .collect();
    J2kForwardDwt97Output {
        ll: decomp.ll,
        ll_width: decomp.ll_width,
        ll_height: decomp.ll_height,
        levels,
    }
}

/// Adapter scalar forward RCT reference for CUDA stage parity.
///
/// Applies the native CPU forward Reversible Color Transform to three
/// component planes supplied as owned `Vec<f32>` arrays.  The transform is
/// applied in place and the mutated planes are returned, so callers do not
/// need to pass a mutable slice.
#[doc(hidden)]
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
pub fn forward_ict_reference(mut planes: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    j2c::forward_mct::forward_ict(&mut planes);
    planes
}

/// Adapter scalar sub-band quantization reference for backend stage parity.
#[doc(hidden)]
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
    Ok(j2c::encode::deinterleave_to_f32(
        pixels,
        num_pixels,
        num_components,
        bit_depth,
        signed,
    ))
}

/// Adapter scalar pixel deinterleave/level-shift reference for backend stage
/// parity.
///
/// This compatibility wrapper panics on invalid geometry. Prefer
/// [`try_deinterleave_reference`] in new code.
#[doc(hidden)]
pub fn deinterleave_reference(
    pixels: &[u8],
    num_pixels: usize,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
) -> Vec<Vec<f32>> {
    try_deinterleave_reference(pixels, num_pixels, num_components, bit_depth, signed)
        .expect("deinterleave_reference requires valid interleaved pixel geometry")
}

/// Adapter scalar Tier-2 packetization helper for backend experimentation.
#[doc(hidden)]
pub fn encode_j2k_packetization_scalar(
    job: J2kPacketizationEncodeJob<'_>,
) -> core::result::Result<Vec<u8>, &'static str> {
    let mut resolutions = job
        .resolutions
        .iter()
        .map(|resolution| j2c::packet_encode::ResolutionPacket {
            subbands: resolution
                .subbands
                .iter()
                .map(|subband| j2c::packet_encode::SubbandPrecinct {
                    code_blocks: subband
                        .code_blocks
                        .iter()
                        .map(|code_block| j2c::packet_encode::CodeBlockPacketData {
                            data: code_block.data.to_vec(),
                            ht_cleanup_length: code_block.ht_cleanup_length,
                            ht_refinement_length: code_block.ht_refinement_length,
                            num_coding_passes: code_block.num_coding_passes,
                            classic_segment_lengths: Vec::new(),
                            num_zero_bitplanes: code_block.num_zero_bitplanes,
                            previously_included: code_block.previously_included,
                            l_block: code_block.l_block,
                            block_coding_mode: match code_block.block_coding_mode {
                                J2kPacketizationBlockCodingMode::Classic => {
                                    j2c::codestream_write::BlockCodingMode::Classic
                                }
                                J2kPacketizationBlockCodingMode::HighThroughput => {
                                    j2c::codestream_write::BlockCodingMode::HighThroughput
                                }
                            },
                        })
                        .collect(),
                    num_cbs_x: subband.num_cbs_x,
                    num_cbs_y: subband.num_cbs_y,
                })
                .collect(),
        })
        .collect::<Vec<_>>();

    let descriptors = job
        .packet_descriptors
        .iter()
        .map(|descriptor| j2c::packet_encode::PacketDescriptor {
            packet_index: descriptor.packet_index,
            state_index: descriptor.state_index,
            layer: descriptor.layer,
            resolution: descriptor.resolution,
            component: descriptor.component,
            precinct: descriptor.precinct,
        })
        .collect::<Vec<_>>();

    j2c::packet_encode::validate_ht_segment_lengths(&resolutions)?;

    if descriptors.is_empty() {
        Ok(j2c::packet_encode::form_tile_bitstream_for_progression(
            &mut resolutions,
            job.num_layers,
            job.num_components,
            job.progression_order,
        ))
    } else {
        j2c::packet_encode::form_tile_bitstream_with_descriptors(&mut resolutions, &descriptors)
    }
}
