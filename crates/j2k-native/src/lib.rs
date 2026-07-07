/*!
Internal pure-Rust JPEG 2000 codec engine for `j2k`.

This module tree was imported from the `dicom-toolkit-jpeg2000` 0.5.0 crate and
adapted in-repo so `j2k` no longer depends on an external production decoder crate.

`dicom-toolkit-jpeg2000` is the JPEG 2000 engine used by `dicom-toolkit-rs`.
It is a maintained fork of the original `hayro-jpeg2000` project with
DICOM-focused extensions, including native-bit-depth decode for 8/12/16-bit
images and pure-Rust JPEG 2000 encoding.

The crate can decode raw JPEG 2000 codestreams (`.j2c`) and still-image JP2/JPH
wrappers. It implements the JPEG 2000 core coding system (ISO/IEC 15444-1) and
HTJ2K block coding (ISO/IEC 15444-15) through the support boundary documented in
`docs/public-support.md`. The remaining declared gaps are tracked there.

The crate offers both a high-level 8-bit decode path for general image use and
a native-bit-depth decode path for integrations such as DICOM, plus encoder APIs
for emitting raw JPEG 2000 and HTJ2K codestreams.

# Example
```rust,no_run
use j2k_native::{DecodeSettings, Image};

let data = std::fs::read("image.jp2").unwrap();
let image = Image::new(&data, &DecodeSettings::default()).unwrap();

println!(
    "{}x{} image in {:?} with alpha={}",
    image.width(),
    image.height(),
    image.color_space(),
    image.has_alpha(),
);

let bitmap = image.decode().unwrap();
```

If you want to see a more comprehensive example, please take a look
at the example in [GitHub](https://github.com/knopkem/dicom-toolkit-rs/blob/main/crates/dicom-toolkit-jpeg2000/examples/png.rs),
which shows the main steps needed to convert a JPEG 2000 image into PNG.

# Testing
The decoder has been tested against 20.000+ images scraped from random PDFs
on the internet and also passes a large part of the `OpenJPEG` test suite. So you
can expect the crate to perform decently in terms of decoding correctness.

# Performance
A decent amount of effort has already been put into optimizing this crate
(both raw throughput and memory allocations), with remaining optimization work planned.

Overall, you should expect this crate to have worse performance than `OpenJPEG`,
but the difference gap should not be too large.

# Safety
By default, the crate has the `simd` feature enabled, which uses the
[`fearless_simd`](https://github.com/linebender/fearless_simd) crate to accelerate
important parts of the pipeline. If you want to eliminate any usage of unsafe
in this crate as well as its dependencies, you can simply disable this
feature, at the cost of worse decoding performance. Unsafe code is forbidden
via a crate-level attribute.

The crate is `no_std` compatible but requires an allocator to be available.
*/

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![forbid(missing_docs)]
#![allow(clippy::too_many_arguments)]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::error::{bail, err};
use crate::j2c::{ComponentData, Header};
use crate::jp2::cdef::ChannelAssociation;
#[cfg(test)]
use crate::jp2::colr::CieLab;
use crate::jp2::colr::EnumeratedColorspace;
use crate::jp2::{DecodedImage, ImageBoxes};

macro_rules! define_ht_code_block_job {
    (
        $(#[$meta:meta])*
        pub struct $name:ident $(<$lt:lifetime>)? {
            $($prefix:tt)*
        }
    ) => {
        $(#[$meta])*
        pub struct $name $(<$lt>)? {
            $($prefix)*
            /// Cleanup segment length in bytes.
            pub cleanup_length: u32,
            /// Refinement segment length in bytes.
            pub refinement_length: u32,
            /// Code-block width in samples.
            pub width: u32,
            /// Code-block height in samples.
            pub height: u32,
            /// Output row stride, in samples, for the target sub-band storage.
            pub output_stride: usize,
            /// Missing most-significant bit planes for this code block.
            pub missing_bit_planes: u8,
            /// Number of coding passes present for this code block.
            pub number_of_coding_passes: u8,
            /// Total coded bitplanes for the parent sub-band.
            pub num_bitplanes: u8,
            /// Region-of-interest maxshift value from RGN marker metadata.
            pub roi_shift: u8,
            /// Whether vertically causal context was enabled.
            pub stripe_causal: bool,
            /// Whether strict decode validation is enabled for the parent image.
            pub strict: bool,
            /// Dequantization step to apply to decoded coefficients.
            pub dequantization_step: f32,
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __j2k_component_plane_metadata_accessors {
    () => {
        /// Width and height of this decoded plane in output samples.
        #[must_use]
        pub fn dimensions(&self) -> (u32, u32) {
            self.dimensions
        }

        /// Horizontal and vertical SIZ sampling factors (`XRsiz`, `YRsiz`).
        #[must_use]
        pub fn sampling(&self) -> (u8, u8) {
            self.sampling
        }

        /// Bit depth of this component plane.
        #[must_use]
        pub fn bit_depth(&self) -> u8 {
            self.bit_depth
        }

        /// Whether this component plane stores signed sample values.
        #[must_use]
        pub fn signed(&self) -> bool {
            self.signed
        }
    };
}

mod backend;
mod color;
mod error;
mod ht_adapter;
mod inspect;
#[macro_use]
pub(crate) mod log;
mod direct_cpu;
mod direct_plan;
mod direct_roi;
pub(crate) mod math;
#[doc(hidden)]
pub mod packet_math;
pub(crate) mod profile;
mod roi;
pub(crate) mod writer;

#[cfg(test)]
use crate::math::{dispatch, Level};
#[cfg(test)]
use color::cielab_to_rgb;
pub(crate) use color::{
    convert_color_space, interleave_and_convert, interleave_and_convert_region,
    native_component_plane_dimensions, resolve_alpha_and_color_space, resolve_palette_indices,
    validate_channel_definition, validate_interleaved_output_buffer,
};
pub use color::{
    Bitmap, ColorSpace, ComponentPlane, DecodedComponents, DecodedNativeComponents,
    NativeComponentPlane, RawBitmap,
};
#[doc(hidden)]
pub use direct_cpu::{
    execute_direct_color_plan_rgb8_into, execute_direct_color_plan_rgba8_into, J2kDirectCpuScratch,
};
#[doc(hidden)]
pub use direct_plan::{
    HtOwnedCodeBlockBatchJob, HtOwnedSubBandPlan, J2kDirectBandId, J2kDirectColorPlan,
    J2kDirectGrayscalePlan, J2kDirectGrayscaleStep, J2kDirectIdwtStep, J2kDirectStoreStep,
    J2kOwnedCodeBlockBatchJob, J2kOwnedSubBandPlan,
};
#[doc(hidden)]
pub use direct_roi::{
    idwt_required_input_window_for_rects, idwt_required_input_windows, idwt_required_output_margin,
    J2kIdwtRequiredInputWindows, J2kRequiredBandRegion,
};
pub use inspect::{
    inspect_j2k_codestream_header, looks_like_j2k_codestream, J2kCodestreamComponentHeader,
    J2kCodestreamHeaderError, J2kCodestreamHeaderMetadata,
};
#[doc(hidden)]
pub use jp2::{
    extract_jp2_codestream_payload, inspect_jp2_container, Jp2ChannelAssociation,
    Jp2ChannelDefinition, Jp2ChannelType, Jp2ColorSpec, Jp2ComponentMapping,
    Jp2ComponentMappingType, Jp2ComponentMetadata, Jp2Container, Jp2FileKind, Jp2FileMetadata,
    Jp2ImageHeaderMetadata, Jp2PaletteColumn, Jp2PaletteMetadata,
};
#[doc(hidden)]
pub use roi::idwt_band_index;
pub(crate) use roi::{
    add_roi_shift_to_bitplanes, apply_roi_maxshift_inverse_i32, apply_roi_maxshift_inverse_i64,
    validate_roi,
};

pub use error::{
    ColorError, DecodeError, DecodingError, DirectPlanUnsupportedReason, FormatError, MarkerError,
    Result, TileError, ValidationError,
};
pub use j2c::encode::{
    encode, encode_component_planes_53, encode_htj2k, encode_precomputed_htj2k_53,
    encode_precomputed_htj2k_53_with_accelerator, encode_precomputed_htj2k_53_with_mct,
    encode_precomputed_htj2k_53_with_mct_and_accelerator, encode_precomputed_htj2k_97,
    encode_precomputed_htj2k_97_batch_with_accelerator,
    encode_precomputed_htj2k_97_with_accelerator, encode_precomputed_j2k_53,
    encode_precomputed_j2k_53_with_accelerator, encode_precomputed_j2k_53_with_mct,
    encode_precomputed_j2k_53_with_mct_and_accelerator, encode_preencoded_htj2k_97,
    encode_preencoded_htj2k_97_compact_owned_with_accelerator,
    encode_preencoded_htj2k_97_owned_with_accelerator, encode_preencoded_htj2k_97_with_accelerator,
    encode_prequantized_htj2k_97, encode_prequantized_htj2k_97_with_accelerator,
    encode_typed_component_planes_53, encode_with_accelerator,
    encode_with_accelerator_and_roi_regions, encode_with_roi_regions,
    irreversible_quantization_step_for_subband, EncodeComponentPlane, EncodeOptions,
    EncodeProgressionOrder, EncodeRoiRegion, EncodeTypedComponentPlane,
};
pub use j2c::{CpuDecodeParallelism, DecoderContext, Reversible53CoefficientImage};
#[doc(hidden)]
pub use j2k_types::{
    sort_packet_descriptors_for_progression, CpuOnlyJ2kEncodeStageAccelerator,
    EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock, IrreversibleQuantizationStep,
    IrreversibleQuantizationSubbandScales, J2kCodeBlockSegment, J2kCodeBlockStyle,
    J2kDeinterleaveToF32Job, J2kEncodeDispatchReport, J2kEncodeStageAccelerator,
    J2kForwardDwt53Job, J2kForwardDwt53Level, J2kForwardDwt53Output, J2kForwardDwt97Job,
    J2kForwardDwt97Level, J2kForwardDwt97Output, J2kForwardIctJob, J2kForwardRctJob,
    J2kHtCodeBlockEncodeJob, J2kHtSubbandEncodeJob, J2kHtj2kTileEncodeJob,
    J2kPacketizationBlockCodingMode, J2kPacketizationCodeBlock, J2kPacketizationEncodeJob,
    J2kPacketizationPacketDescriptor, J2kPacketizationProgressionOrder, J2kPacketizationResolution,
    J2kPacketizationSubband, J2kQuantizeSubbandJob, J2kSubBandType, J2kTier1CodeBlockEncodeJob,
    PrecomputedHtj2k53Component, PrecomputedHtj2k53Image, PrecomputedHtj2k97Component,
    PrecomputedHtj2k97Image, PreencodedHtj2k97CodeBlock, PreencodedHtj2k97CompactCodeBlock,
    PreencodedHtj2k97CompactComponent, PreencodedHtj2k97CompactImage,
    PreencodedHtj2k97CompactResolution, PreencodedHtj2k97CompactSubband,
    PreencodedHtj2k97Component, PreencodedHtj2k97Image, PreencodedHtj2k97Resolution,
    PreencodedHtj2k97Subband, PrequantizedHtj2k97CodeBlock, PrequantizedHtj2k97Component,
    PrequantizedHtj2k97Image, PrequantizedHtj2k97Resolution, PrequantizedHtj2k97Subband,
};

mod j2c;
mod jp2;
pub(crate) mod reader;
#[doc(hidden)]
pub use j2c::ht_encode_tables::HtUvlcTableEntry;

const MAX_CLASSIC_DECODE_BITPLANES: u8 = j2c::MAX_BITPLANE_COUNT;
const MAX_DEINTERLEAVE_REFERENCE_BIT_DEPTH: u8 = 38;
pub(crate) const MAX_J2K_SPEC_COMPONENTS: u16 = 16_384;
pub(crate) const MAX_J2K_IMAGE_DIMENSION: u32 = 60_000;
pub(crate) const MAX_J2K_TILE_COUNT: u64 = u16::MAX as u64 + 1;
pub(crate) const DEFAULT_MAX_DECODE_BYTES: usize = 512 * 1024 * 1024;

#[inline]
pub(crate) fn checked_decode_usize_product2(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or(ValidationError::ImageTooLarge.into())
}

#[inline]
fn checked_decode_byte_cap(len: usize) -> Result<usize> {
    if len > DEFAULT_MAX_DECODE_BYTES {
        bail!(ValidationError::ImageTooLarge);
    }
    Ok(len)
}

#[inline]
pub(crate) fn checked_decode_byte_len2(left: usize, right: usize) -> Result<usize> {
    checked_decode_byte_cap(checked_decode_usize_product2(left, right)?)
}

#[inline]
pub(crate) fn checked_decode_byte_len3(first: usize, second: usize, third: usize) -> Result<usize> {
    let partial = checked_decode_usize_product2(first, second)?;
    checked_decode_byte_cap(checked_decode_usize_product2(partial, third)?)
}

#[inline]
pub(crate) fn checked_decode_byte_len4(
    first: usize,
    second: usize,
    third: usize,
    fourth: usize,
) -> Result<usize> {
    let partial = checked_decode_usize_product2(first, second)?;
    let partial = checked_decode_usize_product2(partial, third)?;
    checked_decode_byte_cap(checked_decode_usize_product2(partial, fourth)?)
}

#[inline]
pub(crate) fn checked_decode_sample_count(width: u32, height: u32) -> Result<usize> {
    #[cfg(target_pointer_width = "64")]
    {
        Ok((u64::from(width) * u64::from(height)) as usize)
    }

    #[cfg(not(target_pointer_width = "64"))]
    {
        checked_decode_usize_product2(width as usize, height as usize)
    }
}

#[inline]
fn native_bytes_per_sample(bit_depth: u8) -> Result<usize> {
    if bit_depth == 0 || bit_depth > 63 {
        bail!(ValidationError::ImageTooLarge);
    }
    Ok(usize::from(bit_depth).div_ceil(8).max(1))
}

#[doc(hidden)]
pub use backend::{
    HtCleanupEncodeDistribution, HtCodeBlockBatchJob, HtCodeBlockDecodeJob,
    HtCodeBlockDecodePhaseLimit, HtCodeBlockDecoder, HtSubBandDecodeJob, J2kCodeBlockBatchJob,
    J2kCodeBlockDecodeJob, J2kIdwtBand, J2kInverseMctJob, J2kRect, J2kSingleDecompositionIdwtJob,
    J2kStoreComponentJob, J2kSubBandDecodeJob, J2kTier1TokenSegment, J2kWaveletTransform,
};
#[doc(hidden)]
pub use ht_adapter::{
    decode_ht_sigprop_benchmark_state, ht_uvlc_encode_table, ht_uvlc_encode_table_bytes,
    ht_uvlc_table0, ht_uvlc_table1, ht_vlc_encode_table0, ht_vlc_encode_table1, ht_vlc_table0,
    ht_vlc_table1, prepare_ht_sigprop_benchmark_state, HtSigPropBenchmarkState,
};

fn internal_j2k_sub_band_type(sub_band_type: J2kSubBandType) -> j2c::build::SubBandType {
    match sub_band_type {
        J2kSubBandType::LowLow => j2c::build::SubBandType::LowLow,
        J2kSubBandType::HighLow => j2c::build::SubBandType::HighLow,
        J2kSubBandType::LowHigh => j2c::build::SubBandType::LowHigh,
        J2kSubBandType::HighHigh => j2c::build::SubBandType::HighHigh,
    }
}

fn internal_j2k_code_block_style(style: J2kCodeBlockStyle) -> j2c::codestream::CodeBlockStyle {
    j2c::codestream::CodeBlockStyle {
        selective_arithmetic_coding_bypass: style.selective_arithmetic_coding_bypass,
        reset_context_probabilities: style.reset_context_probabilities,
        termination_on_each_pass: style.termination_on_each_pass,
        vertically_causal_context: style.vertically_causal_context,
        segmentation_symbols: style.segmentation_symbols,
        high_throughput_block_coding: false,
    }
}

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

/// Adapter scalar classic J2K decoder helper for backend experimentation.
#[doc(hidden)]
pub fn decode_j2k_code_block_scalar(
    job: J2kCodeBlockDecodeJob<'_>,
    output: &mut [f32],
) -> Result<()> {
    let mut workspace = J2kCodeBlockDecodeWorkspace::default();
    decode_j2k_code_block_scalar_with_workspace(job, output, &mut workspace)
}

/// Reusable scratch for scalar classic J2K code-block decoding.
#[derive(Default)]
#[doc(hidden)]
pub struct J2kCodeBlockDecodeWorkspace {
    bit_plane_decode_context: j2c::bitplane::BitPlaneDecodeContext,
}

/// Adapter scalar classic J2K decoder helper that reuses caller-provided scratch.
#[doc(hidden)]
pub fn decode_j2k_code_block_scalar_with_workspace(
    job: J2kCodeBlockDecodeJob<'_>,
    output: &mut [f32],
    workspace: &mut J2kCodeBlockDecodeWorkspace,
) -> Result<()> {
    let layout =
        checked_code_block_output_layout(job.width, job.height, job.output_stride, output.len())?;
    let style = internal_j2k_code_block_style(job.style);
    let sub_band_type = internal_j2k_sub_band_type(job.sub_band_type);
    let coded_bitplanes = add_roi_shift_to_bitplanes(
        job.total_bitplanes,
        job.roi_shift,
        MAX_CLASSIC_DECODE_BITPLANES,
    )?;

    j2c::bitplane::decode_code_block_segments_validated(
        job.data,
        job.segments,
        job.width,
        job.height,
        job.missing_bit_planes,
        job.number_of_coding_passes,
        coded_bitplanes,
        sub_band_type,
        &style,
        job.strict,
        &mut workspace.bit_plane_decode_context,
    )?;

    write_j2k_code_block_output(&workspace.bit_plane_decode_context, job, layout, output);

    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct CodeBlockOutputLayout {
    stride: usize,
    len: usize,
}

fn checked_code_block_output_layout(
    width: u32,
    height: u32,
    output_stride: usize,
    output_len: usize,
) -> Result<CodeBlockOutputLayout> {
    let stride = usize::try_from(width).map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
    let height = height as usize;
    let required_len = if height == 0 {
        0
    } else {
        output_stride
            .checked_mul(height - 1)
            .and_then(|prefix| prefix.checked_add(stride))
            .ok_or(DecodingError::CodeBlockDecodeFailure)?
    };
    if output_len < required_len {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    let len = stride
        .checked_mul(height)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;

    Ok(CodeBlockOutputLayout { stride, len })
}

fn write_j2k_code_block_output(
    decode_context: &j2c::bitplane::BitPlaneDecodeContext,
    job: J2kCodeBlockDecodeJob<'_>,
    layout: CodeBlockOutputLayout,
    output: &mut [f32],
) {
    for (row_idx, coeff_row) in decode_context
        .coefficient_rows()
        .enumerate()
        .take(job.height as usize)
    {
        let row_start = row_idx * job.output_stride;
        let output_row = &mut output[row_start..row_start + layout.stride];
        for (coefficient, sample) in coeff_row.iter().zip(output_row.iter_mut()) {
            let coefficient = apply_roi_maxshift_inverse_i64(coefficient.get_i64(), job.roi_shift);
            *sample = coefficient as f32 * job.dequantization_step;
        }
    }
}

/// Adapter scalar classic J2K pass timings for backend experimentation.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[doc(hidden)]
pub struct J2kCodeBlockDecodeProfile {
    /// Significance propagation pass elapsed time in microseconds.
    pub sigprop_us: u128,
    /// Magnitude refinement pass elapsed time in microseconds.
    pub magref_us: u128,
    /// Cleanup pass elapsed time in microseconds.
    pub cleanup_us: u128,
    /// Raw bypass pass elapsed time in microseconds.
    pub bypass_us: u128,
    /// Coefficient output conversion elapsed time in microseconds.
    pub output_convert_us: u128,
}

impl J2kCodeBlockDecodeProfile {
    /// Create an empty profile accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn add_native_stats(&mut self, stats: j2c::bitplane::J2kBlockDecodeStats) {
        self.sigprop_us += stats.sigprop_us;
        self.magref_us += stats.magref_us;
        self.cleanup_us += stats.cleanup_us;
        self.bypass_us += stats.bypass_us;
    }
}

/// Adapter scalar classic J2K decoder helper that records pass timings.
#[doc(hidden)]
pub fn decode_j2k_code_block_scalar_profiled(
    job: J2kCodeBlockDecodeJob<'_>,
    output: &mut [f32],
    profile: &mut J2kCodeBlockDecodeProfile,
) -> Result<()> {
    let mut workspace = J2kCodeBlockDecodeWorkspace::default();
    decode_j2k_code_block_scalar_with_workspace_profiled(job, output, &mut workspace, profile)
}

/// Adapter scalar classic J2K decoder helper that records pass timings and reuses scratch.
#[doc(hidden)]
pub fn decode_j2k_code_block_scalar_with_workspace_profiled(
    job: J2kCodeBlockDecodeJob<'_>,
    output: &mut [f32],
    workspace: &mut J2kCodeBlockDecodeWorkspace,
    profile: &mut J2kCodeBlockDecodeProfile,
) -> Result<()> {
    let layout =
        checked_code_block_output_layout(job.width, job.height, job.output_stride, output.len())?;
    let style = internal_j2k_code_block_style(job.style);
    let sub_band_type = internal_j2k_sub_band_type(job.sub_band_type);
    let coded_bitplanes = add_roi_shift_to_bitplanes(
        job.total_bitplanes,
        job.roi_shift,
        MAX_CLASSIC_DECODE_BITPLANES,
    )?;
    let mut stats = j2c::bitplane::J2kBlockDecodeStats::default();

    j2c::bitplane::decode_code_block_segments_validated_profiled(
        job.data,
        job.segments,
        job.width,
        job.height,
        job.missing_bit_planes,
        job.number_of_coding_passes,
        coded_bitplanes,
        sub_band_type,
        &style,
        job.strict,
        &mut workspace.bit_plane_decode_context,
        &mut stats,
        true,
    )?;
    profile.add_native_stats(stats);

    let output_convert_started = profile::profile_now(true);
    write_j2k_code_block_output(&workspace.bit_plane_decode_context, job, layout, output);
    profile.output_convert_us += profile::elapsed_us(output_convert_started);

    Ok(())
}

/// Adapter scalar classic J2K batched decoder helper for backend experimentation.
#[doc(hidden)]
pub fn decode_j2k_sub_band_scalar(job: J2kSubBandDecodeJob<'_>, output: &mut [f32]) -> Result<()> {
    let required_len = if job.height == 0 {
        0
    } else {
        usize::try_from(job.width)
            .ok()
            .and_then(|width| width.checked_mul(job.height as usize))
            .ok_or(DecodingError::CodeBlockDecodeFailure)?
    };
    if output.len() < required_len {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }

    let sub_band_width =
        usize::try_from(job.width).map_err(|_| DecodingError::CodeBlockDecodeFailure)?;

    for batch_job in job.jobs {
        let code_block = batch_job.code_block;
        if code_block.output_stride != sub_band_width {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }
        if batch_job
            .output_x
            .checked_add(code_block.width)
            .is_none_or(|x1| x1 > job.width)
            || batch_job
                .output_y
                .checked_add(code_block.height)
                .is_none_or(|y1| y1 > job.height)
        {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }

        let base_idx = usize::try_from(batch_job.output_y)
            .ok()
            .and_then(|y| y.checked_mul(sub_band_width))
            .and_then(|row| row.checked_add(batch_job.output_x as usize))
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        let block_output_len = if code_block.height == 0 {
            0
        } else {
            code_block
                .output_stride
                .checked_mul(code_block.height as usize - 1)
                .and_then(|prefix| prefix.checked_add(code_block.width as usize))
                .ok_or(DecodingError::CodeBlockDecodeFailure)?
        };
        let end_idx = base_idx
            .checked_add(block_output_len)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        if end_idx > output.len() {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }

        decode_j2k_code_block_scalar(code_block, &mut output[base_idx..end_idx])?;
    }

    Ok(())
}

/// Adapter scalar HTJ2K decoder helper for backend experimentation.
#[doc(hidden)]
pub fn decode_ht_code_block_scalar(
    job: HtCodeBlockDecodeJob<'_>,
    output: &mut [f32],
) -> Result<()> {
    decode_ht_code_block_scalar_for_phase::<{ j2c::ht_block_decode::PHASE_LIMIT_MAGREF }>(
        job, output,
    )
}

/// Adapter scalar HTJ2K decoder helper that stops after the selected phase.
#[doc(hidden)]
pub fn decode_ht_code_block_scalar_until_phase(
    job: HtCodeBlockDecodeJob<'_>,
    output: &mut [f32],
    phase_limit: HtCodeBlockDecodePhaseLimit,
) -> Result<()> {
    match phase_limit {
        HtCodeBlockDecodePhaseLimit::Cleanup => decode_ht_code_block_scalar_for_phase::<
            { j2c::ht_block_decode::PHASE_LIMIT_CLEANUP },
        >(job, output),
        HtCodeBlockDecodePhaseLimit::SignificancePropagation => {
            decode_ht_code_block_scalar_for_phase::<{ j2c::ht_block_decode::PHASE_LIMIT_SIGPROP }>(
                job, output,
            )
        }
        HtCodeBlockDecodePhaseLimit::MagnitudeRefinement => {
            decode_ht_code_block_scalar_for_phase::<{ j2c::ht_block_decode::PHASE_LIMIT_MAGREF }>(
                job, output,
            )
        }
    }
}

/// Adapter reusable scalar HTJ2K decode workspace for backend experimentation.
#[derive(Default)]
#[doc(hidden)]
pub struct HtCodeBlockDecodeWorkspace {
    coefficients: Vec<u32>,
    scratch: j2c::ht_block_decode::HtBlockDecodeScratch,
}

impl HtCodeBlockDecodeWorkspace {
    /// Current coefficient buffer capacity retained by this workspace.
    pub fn coefficient_capacity(&self) -> usize {
        self.coefficients.capacity()
    }
}

/// Adapter scalar HTJ2K phase timings for backend experimentation.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[doc(hidden)]
pub struct HtCodeBlockDecodeProfile {
    /// Number of decoded HT code blocks.
    pub blocks: u128,
    /// Number of decoded HT code blocks with refinement data.
    pub refinement_blocks: u128,
    /// Total cleanup segment bytes consumed by decoded HT code blocks.
    pub cleanup_bytes: u128,
    /// Total refinement segment bytes consumed by decoded HT code blocks.
    pub refinement_bytes: u128,
    /// Cleanup phase elapsed time in microseconds.
    pub cleanup_us: u128,
    /// Magnitude/sign phase elapsed time in microseconds.
    pub mag_sgn_us: u128,
    /// Sigma build phase elapsed time in microseconds.
    pub sigma_us: u128,
    /// Significance propagation phase elapsed time in microseconds.
    pub sigprop_us: u128,
    /// Magnitude refinement phase elapsed time in microseconds.
    pub magref_us: u128,
}

impl HtCodeBlockDecodeProfile {
    /// Create an empty profile accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn add_native_stats(&mut self, stats: j2c::ht_block_decode::HtBlockDecodeStats) {
        self.blocks += stats.blocks;
        self.refinement_blocks += stats.refinement_blocks;
        self.cleanup_bytes += stats.cleanup_bytes;
        self.refinement_bytes += stats.refinement_bytes;
        self.cleanup_us += stats.ht_cleanup_us;
        self.mag_sgn_us += stats.ht_mag_sgn_us;
        self.sigma_us += stats.ht_sigma_us;
        self.sigprop_us += stats.ht_sigprop_us;
        self.magref_us += stats.ht_magref_us;
    }
}

/// Adapter scalar HTJ2K decoder helper that reuses caller-owned scratch buffers.
#[doc(hidden)]
pub fn decode_ht_code_block_scalar_with_workspace(
    job: HtCodeBlockDecodeJob<'_>,
    output: &mut [f32],
    workspace: &mut HtCodeBlockDecodeWorkspace,
) -> Result<()> {
    decode_ht_code_block_scalar_for_phase_with_workspace::<
        { j2c::ht_block_decode::PHASE_LIMIT_MAGREF },
    >(job, output, workspace)
}

/// Adapter scalar HTJ2K decoder helper that reuses scratch and records phase timings.
#[doc(hidden)]
pub fn decode_ht_code_block_scalar_with_workspace_profiled(
    job: HtCodeBlockDecodeJob<'_>,
    output: &mut [f32],
    workspace: &mut HtCodeBlockDecodeWorkspace,
    profile: &mut HtCodeBlockDecodeProfile,
) -> Result<()> {
    decode_ht_code_block_scalar_for_phase_with_workspace_profiled::<
        { j2c::ht_block_decode::PHASE_LIMIT_MAGREF },
    >(job, output, workspace, profile)
}

fn decode_ht_code_block_scalar_for_phase<const PHASE_LIMIT: u8>(
    job: HtCodeBlockDecodeJob<'_>,
    output: &mut [f32],
) -> Result<()> {
    let mut workspace = HtCodeBlockDecodeWorkspace::default();
    decode_ht_code_block_scalar_for_phase_with_workspace::<PHASE_LIMIT>(job, output, &mut workspace)
}

fn decode_ht_code_block_scalar_for_phase_with_workspace<const PHASE_LIMIT: u8>(
    job: HtCodeBlockDecodeJob<'_>,
    output: &mut [f32],
    workspace: &mut HtCodeBlockDecodeWorkspace,
) -> Result<()> {
    let layout =
        checked_code_block_output_layout(job.width, job.height, job.output_stride, output.len())?;
    let segments = j2c::ht_block_decode::HtCodeBlockSegments::from_combined_payload(
        job.data,
        job.cleanup_length,
        job.refinement_length,
    )?;
    let coded_bitplanes = add_roi_shift_to_bitplanes(job.num_bitplanes, job.roi_shift, 31)?;
    workspace.coefficients.clear();
    workspace.coefficients.resize(layout.len, 0);
    j2c::ht_block_decode::decode_segments_validated_with_scratch_for_phase::<PHASE_LIMIT>(
        &segments,
        job.missing_bit_planes,
        coded_bitplanes,
        job.number_of_coding_passes,
        job.stripe_causal,
        job.strict,
        &mut workspace.coefficients,
        job.width,
        job.height,
        job.width,
        &mut workspace.scratch,
        None,
        false,
    )?;

    write_ht_code_block_output(
        &workspace.coefficients,
        job,
        layout,
        coded_bitplanes,
        output,
    );

    Ok(())
}

fn decode_ht_code_block_scalar_for_phase_with_workspace_profiled<const PHASE_LIMIT: u8>(
    job: HtCodeBlockDecodeJob<'_>,
    output: &mut [f32],
    workspace: &mut HtCodeBlockDecodeWorkspace,
    profile: &mut HtCodeBlockDecodeProfile,
) -> Result<()> {
    let layout =
        checked_code_block_output_layout(job.width, job.height, job.output_stride, output.len())?;
    let segments = j2c::ht_block_decode::HtCodeBlockSegments::from_combined_payload(
        job.data,
        job.cleanup_length,
        job.refinement_length,
    )?;
    let coded_bitplanes = add_roi_shift_to_bitplanes(job.num_bitplanes, job.roi_shift, 31)?;
    workspace.coefficients.clear();
    workspace.coefficients.resize(layout.len, 0);
    let mut stats = j2c::ht_block_decode::HtBlockDecodeStats::default();
    j2c::ht_block_decode::decode_segments_validated_with_scratch_for_phase::<PHASE_LIMIT>(
        &segments,
        job.missing_bit_planes,
        coded_bitplanes,
        job.number_of_coding_passes,
        job.stripe_causal,
        job.strict,
        &mut workspace.coefficients,
        job.width,
        job.height,
        job.width,
        &mut workspace.scratch,
        Some(&mut stats),
        true,
    )?;
    profile.add_native_stats(stats);

    write_ht_code_block_output(
        &workspace.coefficients,
        job,
        layout,
        coded_bitplanes,
        output,
    );

    Ok(())
}

fn write_ht_code_block_output(
    coefficients: &[u32],
    job: HtCodeBlockDecodeJob<'_>,
    layout: CodeBlockOutputLayout,
    coded_bitplanes: u8,
    output: &mut [f32],
) {
    for (row_idx, coeff_row) in coefficients
        .chunks_exact(layout.stride)
        .enumerate()
        .take(job.height as usize)
    {
        let row_start = row_idx * job.output_stride;
        let output_row = &mut output[row_start..row_start + layout.stride];
        for (coefficient, sample) in coeff_row.iter().copied().zip(output_row.iter_mut()) {
            let coefficient =
                j2c::ht_block_decode::coefficient_to_i32(coefficient, coded_bitplanes);
            let coefficient = apply_roi_maxshift_inverse_i32(coefficient, job.roi_shift);
            *sample = coefficient as f32 * job.dequantization_step;
        }
    }
}

/// JP2 signature box: 00 00 00 0C 6A 50 20 20
pub(crate) const JP2_MAGIC: &[u8] = b"\x00\x00\x00\x0C\x6A\x50\x20\x20";
/// Codestream signature: FF 4F FF 51 (SOC + SIZ markers)
pub(crate) const CODESTREAM_MAGIC: &[u8] = b"\xFF\x4F\xFF\x51";

/// Settings to apply during decoding.
#[derive(Debug, Copy, Clone)]
pub struct DecodeSettings {
    /// Whether palette indices should be resolved.
    ///
    /// JPEG2000 images can be stored in two different ways. First, by storing
    /// RGB values (depending on the color space) for each pixel. Secondly, by
    /// only storing a single index for each channel, and then resolving the
    /// actual color using the index.
    ///
    /// If you disable this option, in case you have an image with palette
    /// indices, they will not be resolved, but instead a grayscale image
    /// will be returned, with each pixel value corresponding to the palette
    /// index of the location.
    pub resolve_palette_indices: bool,
    /// Whether strict mode should be enabled when decoding.
    ///
    /// The default is lenient for compatibility with older releases. Lenient
    /// mode may tolerate malformed optional container metadata that strict mode
    /// rejects. Use [`DecodeSettings::strict`] for fail-closed validation of
    /// public or adversarial inputs.
    pub strict: bool,
    /// A hint for the target resolution that the image should be decoded at.
    pub target_resolution: Option<(u32, u32)>,
}

impl DecodeSettings {
    /// Compatibility decode settings.
    ///
    /// Lenient mode keeps the historical behavior of accepting recoverable
    /// optional metadata problems where possible.
    #[must_use]
    pub const fn lenient() -> Self {
        Self {
            resolve_palette_indices: true,
            strict: false,
            target_resolution: None,
        }
    }

    /// Strict decode settings for fail-closed validation.
    #[must_use]
    pub const fn strict() -> Self {
        Self {
            resolve_palette_indices: true,
            strict: true,
            target_resolution: None,
        }
    }

    /// Whether the settings permit lenient tolerance of malformed optional
    /// metadata.
    #[must_use]
    pub const fn lenient_tolerance_enabled(&self) -> bool {
        !self.strict
    }
}

impl Default for DecodeSettings {
    fn default() -> Self {
        Self::lenient()
    }
}

/// A JPEG2000 image or codestream.
pub struct Image<'a> {
    /// The tile-part payload used by the legacy JPEG 2000 decoder.
    pub(crate) codestream: &'a [u8],
    /// The header of the J2C codestream.
    pub(crate) header: Header<'a>,
    /// The JP2 boxes of the image. In the case of a raw codestream, we
    /// will synthesize the necessary boxes.
    pub(crate) boxes: ImageBoxes,
    /// Settings that should be applied during decoding.
    pub(crate) settings: DecodeSettings,
    /// Whether the image has an alpha channel.
    pub(crate) has_alpha: bool,
    /// The color space of the image.
    pub(crate) color_space: ColorSpace,
}

impl<'a> Image<'a> {
    /// Try to create a new JPEG2000 image from the given data.
    pub fn new(data: &'a [u8], settings: &DecodeSettings) -> Result<Self> {
        if data.starts_with(JP2_MAGIC) {
            jp2::parse(data, *settings)
        } else if data.starts_with(CODESTREAM_MAGIC) {
            j2c::parse(data, settings)
        } else {
            err!(FormatError::InvalidSignature)
        }
    }

    /// Whether the image has an alpha channel.
    pub fn has_alpha(&self) -> bool {
        self.has_alpha
    }

    /// The color space of the image.
    pub fn color_space(&self) -> &ColorSpace {
        &self.color_space
    }

    /// The width of the image.
    pub fn width(&self) -> u32 {
        self.header.size_data.image_width()
    }

    /// The height of the image.
    pub fn height(&self) -> u32 {
        self.header.size_data.image_height()
    }

    /// The original bit depth of the image. You usually don't need to do anything
    /// with this parameter, it just exists for informational purposes.
    pub fn original_bit_depth(&self) -> u8 {
        // Note that this only works if all components have the same precision.
        self.header.component_infos[0].size_info.precision
    }

    /// Whether decode finishes with additional host-side component mutation or reordering.
    #[doc(hidden)]
    pub fn supports_direct_device_plane_reuse(&self) -> bool {
        if self.settings.resolve_palette_indices && self.boxes.palette.is_some() {
            return false;
        }
        if self.boxes.channel_definition.is_some() {
            return false;
        }
        !matches!(
            self.boxes
                .color_specification
                .as_ref()
                .map(|spec| &spec.color_space),
            Some(jp2::colr::ColorSpace::Enumerated(
                EnumeratedColorspace::Sycc | EnumeratedColorspace::CieLab(_)
            ))
        )
    }

    /// Decode the image and return its decoded result as a `Vec<u8>`, with each
    /// channel interleaved.
    pub fn decode(&self) -> Result<Vec<u8>> {
        let bitmap = self.decode_with_context(&mut DecoderContext::default())?;
        Ok(bitmap.data)
    }

    /// Decode the image and return its decoded result using a caller-provided
    /// decoder context so allocations can be reused across repeated decodes.
    pub fn decode_with_context(&self, decoder_context: &mut DecoderContext<'a>) -> Result<Bitmap> {
        let buffer_size = checked_decode_byte_len3(
            self.width() as usize,
            self.height() as usize,
            self.color_space.num_channels() as usize + if self.has_alpha { 1 } else { 0 },
        )?;
        let mut buf = vec![0; buffer_size];
        self.decode_into(&mut buf, decoder_context)?;

        Ok(Bitmap {
            color_space: self.color_space.clone(),
            data: buf,
            has_alpha: self.has_alpha,
            width: self.width(),
            height: self.height(),
            original_bit_depth: self.original_bit_depth(),
        })
    }

    /// Decode the image into borrowed component planes using a caller-provided
    /// decoder context so allocations can be reused across repeated decodes.
    pub fn decode_components_with_context<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
    ) -> Result<DecodedComponents<'ctx>> {
        self.validate_component_plane_precision()?;
        let decoded_image = self.prepare_decoded_image(decoder_context)?;
        Ok(self.borrow_component_planes(
            decoded_image.decoded_components.as_slice(),
            (self.width(), self.height()),
        ))
    }

    /// Decode the image into owned native-bit-depth component planes.
    ///
    /// Unlike [`Self::decode_native`], this preserves per-component bit depth
    /// and signedness metadata and does not require all components to share a
    /// single packed interleaved representation.
    pub fn decode_native_components(&self) -> Result<DecodedNativeComponents> {
        let mut decoder_context = DecoderContext::default();
        self.decode_native_components_with_context(&mut decoder_context)
    }

    /// Decode the image into owned native-bit-depth component planes using a
    /// caller-provided decoder context.
    pub fn decode_native_components_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<DecodedNativeComponents> {
        let decoded_image = self.prepare_decoded_image(decoder_context)?;
        self.pack_native_component_planes(
            decoded_image.decoded_components,
            (self.width(), self.height()),
        )
    }

    /// Build a adapter grayscale direct device plan without materializing host component planes.
    #[doc(hidden)]
    pub fn build_direct_grayscale_plan_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<J2kDirectGrayscalePlan> {
        if !matches!(self.color_space, ColorSpace::Gray) || self.has_alpha {
            bail!(DecodingError::DirectPlanUnsupported(
                DirectPlanUnsupportedReason::GrayscaleImageWithoutAlpha
            ));
        }

        j2c::build_direct_grayscale_plan(self.codestream, &self.header, decoder_context)
    }

    /// Build a adapter grayscale direct device plan for an output-space region.
    #[doc(hidden)]
    pub fn build_direct_grayscale_plan_region_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
        output_region: (u32, u32, u32, u32),
    ) -> Result<J2kDirectGrayscalePlan> {
        if !matches!(self.color_space, ColorSpace::Gray) || self.has_alpha {
            bail!(DecodingError::DirectPlanUnsupported(
                DirectPlanUnsupportedReason::GrayscaleImageWithoutAlpha
            ));
        }

        decoder_context.set_output_region(Some(output_region));
        let result =
            j2c::build_direct_grayscale_plan(self.codestream, &self.header, decoder_context);
        decoder_context.set_output_region(None);
        result
    }

    /// Build a adapter RGB direct device plan without materializing host component planes.
    #[doc(hidden)]
    pub fn build_direct_color_plan_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<J2kDirectColorPlan> {
        if !matches!(self.color_space, ColorSpace::RGB) || self.has_alpha {
            bail!(DecodingError::DirectPlanUnsupported(
                DirectPlanUnsupportedReason::ColorRgbImageWithoutAlpha
            ));
        }

        j2c::build_direct_color_plan(self.codestream, &self.header, decoder_context)
    }

    /// Build a adapter RGB direct device plan for an output-space region.
    #[doc(hidden)]
    pub fn build_direct_color_plan_region_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
        output_region: (u32, u32, u32, u32),
    ) -> Result<J2kDirectColorPlan> {
        if !matches!(self.color_space, ColorSpace::RGB) || self.has_alpha {
            bail!(DecodingError::DirectPlanUnsupported(
                DirectPlanUnsupportedReason::ColorRgbImageWithoutAlpha
            ));
        }

        decoder_context.set_output_region(Some(output_region));
        let result = j2c::build_direct_color_plan(self.codestream, &self.header, decoder_context);
        decoder_context.set_output_region(None);
        result
    }

    /// Decode borrowed component planes while delegating HTJ2K code-block decode.
    #[doc(hidden)]
    pub fn decode_components_with_ht_decoder<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
        ht_decoder: &mut dyn HtCodeBlockDecoder,
    ) -> Result<DecodedComponents<'ctx>> {
        self.validate_component_plane_precision()?;
        let decoded_image =
            self.prepare_decoded_image_with_ht_decoder(decoder_context, ht_decoder)?;
        Ok(self.borrow_component_planes(
            decoded_image.decoded_components.as_slice(),
            (self.width(), self.height()),
        ))
    }

    /// Decode borrowed component planes for a requested region using a
    /// caller-provided decoder context.
    pub fn decode_region_components_with_context<'ctx>(
        &self,
        roi: (u32, u32, u32, u32),
        decoder_context: &'ctx mut DecoderContext<'a>,
    ) -> Result<DecodedComponents<'ctx>> {
        validate_roi((self.width(), self.height()), roi)?;
        self.validate_component_plane_precision()?;
        let (_x, _y, width, height) = roi;
        let decoded_image = self.prepare_decoded_image_with_region(decoder_context, Some(roi))?;
        Ok(self
            .borrow_component_planes(decoded_image.decoded_components.as_slice(), (width, height)))
    }

    /// Decode a source-coordinate region into owned native-bit-depth component
    /// planes using a caller-provided decoder context.
    pub fn decode_native_region_components_with_context(
        &self,
        roi: (u32, u32, u32, u32),
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<DecodedNativeComponents> {
        validate_roi((self.width(), self.height()), roi)?;
        if self.requires_exact_integer_decode() {
            return self.decode_native_region_components_via_full_decode(roi, decoder_context);
        }
        let (_x, _y, width, height) = roi;
        let decoded_image = self.prepare_decoded_image_with_region(decoder_context, Some(roi))?;
        self.pack_native_component_planes(decoded_image.decoded_components, (width, height))
    }

    /// Decode borrowed component planes for a requested region while
    /// delegating code-block/transform stages through the adapter backend hook.
    #[doc(hidden)]
    pub fn decode_region_components_with_ht_decoder<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
        roi: (u32, u32, u32, u32),
        ht_decoder: &mut dyn HtCodeBlockDecoder,
    ) -> Result<DecodedComponents<'ctx>> {
        validate_roi((self.width(), self.height()), roi)?;
        self.validate_component_plane_precision()?;
        let (_x, _y, width, height) = roi;
        let decoded_image = self.prepare_decoded_image_with_region_and_ht_decoder(
            decoder_context,
            Some(roi),
            Some(ht_decoder),
        )?;
        Ok(self
            .borrow_component_planes(decoded_image.decoded_components.as_slice(), (width, height)))
    }

    /// Decode a region of the image and return it as an 8-bit interleaved bitmap.
    pub fn decode_region(&self, roi: (u32, u32, u32, u32)) -> Result<Bitmap> {
        self.decode_region_with_context(roi, &mut DecoderContext::default())
    }

    /// Decode a region of the image and return it as an 8-bit interleaved bitmap
    /// using a caller-provided decoder context.
    pub fn decode_region_with_context(
        &self,
        roi: (u32, u32, u32, u32),
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<Bitmap> {
        validate_roi((self.width(), self.height()), roi)?;
        let mut decoded_image =
            self.prepare_decoded_image_with_region(decoder_context, Some(roi))?;
        let (_x, _y, width, height) = roi;
        let channels =
            self.color_space.num_channels() as usize + if self.has_alpha { 1 } else { 0 };
        let data_len = checked_decode_byte_len3(width as usize, height as usize, channels)?;
        let mut data = vec![0; data_len];
        interleave_and_convert_region(
            &mut decoded_image,
            width as usize,
            (0, 0, width, height),
            &mut data,
        );
        Ok(Bitmap {
            color_space: self.color_space.clone(),
            data,
            has_alpha: self.has_alpha,
            width,
            height,
            original_bit_depth: self.original_bit_depth(),
        })
    }

    /// Decode the image at native bit depth without scaling to 8-bit.
    ///
    /// For images with bit depth ≤ 8, returns pixel data as `Vec<u8>`.
    /// For images with bit depth > 8 (e.g., 12-bit or 16-bit), returns
    /// pixel data as little-endian `u16` values packed into `Vec<u8>`.
    ///
    /// This is essential for medical imaging (DICOM) where 12-bit and 16-bit
    /// images must preserve their full dynamic range.
    pub fn decode_native(&self) -> Result<RawBitmap> {
        let mut decoder_context = DecoderContext::default();
        self.decode_native_with_context(&mut decoder_context)
    }

    /// Extract reversible 5/3 wavelet coefficients for coefficient-domain
    /// classic JPEG 2000 to HTJ2K recoding.
    ///
    /// This decodes classic Tier-1 code-blocks into dequantized reversible
    /// wavelet coefficients, but does not run inverse DWT or color conversion.
    #[doc(hidden)]
    pub fn decode_reversible_53_coefficients(&self) -> Result<Reversible53CoefficientImage> {
        let mut decoder_context = DecoderContext::default();
        self.decode_reversible_53_coefficients_with_context(&mut decoder_context)
    }

    /// Extract reversible 5/3 wavelet coefficients using a caller-provided
    /// decoder context.
    #[doc(hidden)]
    pub fn decode_reversible_53_coefficients_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<Reversible53CoefficientImage> {
        j2c::recode::extract_reversible_53_coefficients(
            self.codestream,
            &self.header,
            decoder_context,
        )
    }

    /// Decode a region of the image at native bit depth.
    pub fn decode_native_region(&self, roi: (u32, u32, u32, u32)) -> Result<RawBitmap> {
        self.decode_native_region_with_context(roi, &mut DecoderContext::default())
    }

    /// Decode a source-coordinate region into owned native-bit-depth component
    /// planes.
    pub fn decode_native_region_components(
        &self,
        roi: (u32, u32, u32, u32),
    ) -> Result<DecodedNativeComponents> {
        self.decode_native_region_components_with_context(roi, &mut DecoderContext::default())
    }

    /// Decode the image at native bit depth using a caller-provided decoder
    /// context so allocations can be reused across repeated decodes.
    pub fn decode_native_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<RawBitmap> {
        let bit_depth = self.uniform_header_bit_depth()?;
        self.decode_with_output_region(decoder_context, None)?;

        let components = &decoder_context.tile_decode_context.channel_data;
        let num_components =
            u16::try_from(components.len()).map_err(|_| ValidationError::TooManyChannels)?;
        let width = self.width();
        let height = self.height();
        let pixel_count = checked_decode_sample_count(width, height)?;
        let component_signed = Self::component_signedness(components);
        let signed = component_signed.iter().all(|signed| *signed);

        let bytes_per_sample = native_bytes_per_sample(bit_depth)?;
        if bytes_per_sample == 1 {
            let capacity = checked_decode_byte_len2(pixel_count, usize::from(num_components))?;
            let mut data = Vec::with_capacity(capacity);
            for i in 0..pixel_count {
                for component in components.iter() {
                    Self::push_component_native_sample_bytes(&mut data, component, i, bit_depth);
                }
            }
            Ok(RawBitmap {
                data,
                width,
                height,
                bit_depth,
                signed,
                component_signed,
                num_components,
                bytes_per_sample: 1,
            })
        } else {
            let capacity = checked_decode_byte_len3(
                pixel_count,
                usize::from(num_components),
                bytes_per_sample,
            )?;
            let mut data = Vec::with_capacity(capacity);
            for i in 0..pixel_count {
                for component in components.iter() {
                    Self::push_component_native_sample_bytes(&mut data, component, i, bit_depth);
                }
            }
            Ok(RawBitmap {
                data,
                width,
                height,
                bit_depth,
                signed,
                component_signed,
                num_components,
                bytes_per_sample: u8::try_from(bytes_per_sample)
                    .map_err(|_| ValidationError::ImageTooLarge)?,
            })
        }
    }

    /// Decode a region of the image at native bit depth using a caller-provided
    /// decoder context.
    pub fn decode_native_region_with_context(
        &self,
        roi: (u32, u32, u32, u32),
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<RawBitmap> {
        validate_roi((self.width(), self.height()), roi)?;
        if self.requires_exact_integer_decode() {
            return self.decode_native_region_via_full_decode(roi, decoder_context);
        }
        let bit_depth = self.uniform_header_bit_depth()?;
        self.decode_with_output_region(decoder_context, Some(roi))?;

        let components = &decoder_context.tile_decode_context.channel_data;
        let num_components =
            u16::try_from(components.len()).map_err(|_| ValidationError::TooManyChannels)?;
        let bytes_per_sample = native_bytes_per_sample(bit_depth)?;
        let (_x, _y, width, height) = roi;
        let capacity = checked_decode_byte_len4(
            width as usize,
            height as usize,
            usize::from(num_components),
            bytes_per_sample,
        )?;
        let mut data = Vec::with_capacity(capacity);
        let component_signed = Self::component_signedness(components);
        let signed = component_signed.iter().all(|signed| *signed);

        for row in 0..height as usize {
            for col in 0..width as usize {
                let idx = row * width as usize + col;
                for component in components {
                    Self::push_component_native_sample_bytes(&mut data, component, idx, bit_depth);
                }
            }
        }

        Ok(RawBitmap {
            data,
            width,
            height,
            bit_depth,
            signed,
            component_signed,
            num_components,
            bytes_per_sample: u8::try_from(bytes_per_sample)
                .map_err(|_| ValidationError::ImageTooLarge)?,
        })
    }

    fn component_signedness(components: &[ComponentData]) -> Vec<bool> {
        components
            .iter()
            .map(|component| component.signed)
            .collect()
    }

    fn component_plane_sampling(&self, plane_count: usize) -> Vec<(u8, u8)> {
        if self.settings.resolve_palette_indices && self.boxes.palette.is_some() {
            return vec![(1, 1); plane_count];
        }

        let mut sampling = self
            .header
            .component_infos
            .iter()
            .take(plane_count)
            .map(|component| {
                (
                    component.size_info.horizontal_resolution,
                    component.size_info.vertical_resolution,
                )
            })
            .collect::<Vec<_>>();
        sampling.resize(plane_count, (1, 1));
        sampling
    }

    fn borrow_component_planes<'ctx>(
        &self,
        components: &'ctx [ComponentData],
        dimensions: (u32, u32),
    ) -> DecodedComponents<'ctx> {
        let sampling = self.component_plane_sampling(components.len());
        let planes = components
            .iter()
            .zip(sampling)
            .map(|(component, sampling)| ComponentPlane {
                samples: component.container.truncated(),
                dimensions,
                bit_depth: component.bit_depth,
                signed: component.signed,
                sampling,
            })
            .collect();

        DecodedComponents {
            dimensions,
            color_space: self.color_space.clone(),
            has_alpha: self.has_alpha,
            planes,
        }
    }

    fn uniform_header_bit_depth(&self) -> Result<u8> {
        let Some(first) = self.header.component_infos.first() else {
            bail!(DecodingError::CodeBlockDecodeFailure);
        };
        if self
            .header
            .component_infos
            .iter()
            .any(|component| component.size_info.precision != first.size_info.precision)
        {
            bail!(DecodingError::UnsupportedFeature(
                "decode_native requires uniform component bit depths; use decode_components for mixed-depth images"
            ));
        }
        if first.size_info.precision > 38 {
            bail!(DecodingError::UnsupportedFeature(
                "decode_native supports JPEG 2000 Part 1 component precision up to 38 bits"
            ));
        }
        Ok(first.size_info.precision)
    }

    fn validate_component_plane_precision(&self) -> Result<()> {
        if self
            .header
            .component_infos
            .iter()
            .any(|component| component.size_info.precision > 24)
        {
            bail!(DecodingError::UnsupportedFeature(
                "decode_components currently supports component planes up to 24 bits per component"
            ));
        }
        Ok(())
    }

    fn pack_native_component_planes(
        &self,
        components: &[ComponentData],
        dimensions: (u32, u32),
    ) -> Result<DecodedNativeComponents> {
        let sampling = self.component_plane_sampling(components.len());
        let mut planes = Vec::with_capacity(components.len());
        for (component, sampling) in components.iter().zip(sampling) {
            let bytes_per_sample = native_bytes_per_sample(component.bit_depth)?;
            let sample_count = component
                .integer_container
                .as_ref()
                .map_or(component.container.truncated().len(), Vec::len);
            let plane_dimensions =
                native_component_plane_dimensions(dimensions, sampling, sample_count)?;
            let capacity = checked_decode_byte_len2(sample_count, bytes_per_sample)?;
            let mut data = Vec::with_capacity(capacity);
            for idx in 0..sample_count {
                Self::push_component_native_sample_bytes(
                    &mut data,
                    component,
                    idx,
                    component.bit_depth,
                );
            }
            planes.push(NativeComponentPlane {
                data,
                dimensions: plane_dimensions,
                bit_depth: component.bit_depth,
                signed: component.signed,
                sampling,
                bytes_per_sample: u8::try_from(bytes_per_sample)
                    .map_err(|_| ValidationError::ImageTooLarge)?,
            });
        }

        Ok(DecodedNativeComponents {
            dimensions,
            color_space: self.color_space.clone(),
            has_alpha: self.has_alpha,
            planes,
        })
    }

    fn requires_exact_integer_decode(&self) -> bool {
        self.header
            .component_infos
            .iter()
            .any(|component| component.requires_exact_integer_decode())
    }

    fn decode_native_region_via_full_decode(
        &self,
        roi: (u32, u32, u32, u32),
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<RawBitmap> {
        let full = self.decode_native_with_context(decoder_context)?;
        let (x, y, width, height) = roi;
        let bytes_per_pixel = usize::from(full.num_components)
            .checked_mul(usize::from(full.bytes_per_sample))
            .ok_or(ValidationError::ImageTooLarge)?;
        let row_bytes = (width as usize)
            .checked_mul(bytes_per_pixel)
            .ok_or(ValidationError::ImageTooLarge)?;
        let capacity = checked_decode_byte_len3(height as usize, width as usize, bytes_per_pixel)?;
        let mut data = Vec::with_capacity(capacity);
        let full_width = full.width as usize;
        for row in y as usize..(y + height) as usize {
            let start = row
                .checked_mul(full_width)
                .and_then(|offset| offset.checked_add(x as usize))
                .and_then(|sample| sample.checked_mul(bytes_per_pixel))
                .ok_or(ValidationError::ImageTooLarge)?;
            data.extend_from_slice(&full.data[start..start + row_bytes]);
        }

        Ok(RawBitmap {
            data,
            width,
            height,
            bit_depth: full.bit_depth,
            signed: full.signed,
            component_signed: full.component_signed,
            num_components: full.num_components,
            bytes_per_sample: full.bytes_per_sample,
        })
    }

    fn decode_native_region_components_via_full_decode(
        &self,
        roi: (u32, u32, u32, u32),
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<DecodedNativeComponents> {
        let full = self.decode_native_components_with_context(decoder_context)?;
        let (x, y, width, height) = roi;
        let mut planes = Vec::with_capacity(full.planes.len());
        for plane in &full.planes {
            let bytes_per_sample = usize::from(plane.bytes_per_sample);
            let (crop_x, crop_y, crop_width, crop_height) = if plane.dimensions == full.dimensions {
                (x, y, width, height)
            } else {
                let x1 = x.checked_add(width).ok_or(ValidationError::ImageTooLarge)?;
                let y1 = y
                    .checked_add(height)
                    .ok_or(ValidationError::ImageTooLarge)?;
                let (x_rsiz, y_rsiz) = plane.sampling;
                let crop_x = x / u32::from(x_rsiz);
                let crop_y = y / u32::from(y_rsiz);
                let crop_x1 = x1.div_ceil(u32::from(x_rsiz)).min(plane.dimensions.0);
                let crop_y1 = y1.div_ceil(u32::from(y_rsiz)).min(plane.dimensions.1);
                (
                    crop_x,
                    crop_y,
                    crop_x1.saturating_sub(crop_x),
                    crop_y1.saturating_sub(crop_y),
                )
            };
            let row_bytes = (crop_width as usize)
                .checked_mul(bytes_per_sample)
                .ok_or(ValidationError::ImageTooLarge)?;
            let capacity = checked_decode_byte_len3(
                crop_height as usize,
                crop_width as usize,
                bytes_per_sample,
            )?;
            let mut data = Vec::with_capacity(capacity);
            let full_width = plane.dimensions.0 as usize;
            for row in crop_y as usize..(crop_y + crop_height) as usize {
                let start = row
                    .checked_mul(full_width)
                    .and_then(|offset| offset.checked_add(crop_x as usize))
                    .and_then(|sample| sample.checked_mul(bytes_per_sample))
                    .ok_or(ValidationError::ImageTooLarge)?;
                data.extend_from_slice(&plane.data[start..start + row_bytes]);
            }
            planes.push(NativeComponentPlane {
                data,
                dimensions: (crop_width, crop_height),
                bit_depth: plane.bit_depth,
                signed: plane.signed,
                sampling: plane.sampling,
                bytes_per_sample: plane.bytes_per_sample,
            });
        }

        Ok(DecodedNativeComponents {
            dimensions: (width, height),
            color_space: full.color_space,
            has_alpha: full.has_alpha,
            planes,
        })
    }

    fn push_component_native_sample_bytes(
        out: &mut Vec<u8>,
        component: &ComponentData,
        index: usize,
        bit_depth: u8,
    ) {
        if let Some(samples) = component.integer_container.as_ref() {
            Self::push_native_i64_sample_bytes(out, samples[index], bit_depth, component.signed);
        } else {
            Self::push_native_sample_bytes(
                out,
                component.container.truncated()[index],
                bit_depth,
                component.signed,
            );
        }
    }

    fn push_native_i64_sample_bytes(out: &mut Vec<u8>, sample: i64, bit_depth: u8, signed: bool) {
        if signed {
            let magnitude_bits = u32::from(bit_depth.saturating_sub(1));
            let min = -(1_i64 << magnitude_bits);
            let max = (1_i64 << magnitude_bits) - 1;
            let clamped = sample.clamp(min, max);
            if bit_depth <= 8 {
                out.push((clamped as i8) as u8);
            } else if bit_depth <= 16 {
                out.extend_from_slice(&(clamped as i16).to_le_bytes());
            } else {
                let bytes = clamped.to_le_bytes();
                let byte_count = native_bytes_per_sample(bit_depth).unwrap_or(8);
                out.extend_from_slice(&bytes[..byte_count]);
            }
        } else {
            let max = (1u64 << u32::from(bit_depth)) - 1;
            let clamped = if sample <= 0 {
                0
            } else {
                (sample as u64).min(max)
            };
            if bit_depth <= 8 {
                out.push(clamped as u8);
            } else if bit_depth <= 16 {
                out.extend_from_slice(&(clamped as u16).to_le_bytes());
            } else {
                let bytes = clamped.to_le_bytes();
                let byte_count = native_bytes_per_sample(bit_depth).unwrap_or(8);
                out.extend_from_slice(&bytes[..byte_count]);
            }
        }
    }

    fn push_native_sample_bytes(out: &mut Vec<u8>, sample: f32, bit_depth: u8, signed: bool) {
        let rounded = math::round_f32(sample);
        if signed {
            let magnitude_bits = u32::from(bit_depth.saturating_sub(1));
            let min = -(1_i64 << magnitude_bits);
            let max = (1_i64 << magnitude_bits) - 1;
            let rounded = f64::from(rounded);
            let clamped = if rounded.is_nan() {
                0
            } else if rounded <= min as f64 {
                min
            } else if rounded >= max as f64 {
                max
            } else {
                rounded as i64
            };
            if bit_depth <= 8 {
                out.push((clamped as i8) as u8);
            } else if bit_depth <= 16 {
                out.extend_from_slice(&(clamped as i16).to_le_bytes());
            } else {
                let bytes = clamped.to_le_bytes();
                let byte_count = native_bytes_per_sample(bit_depth).unwrap_or(8);
                out.extend_from_slice(&bytes[..byte_count]);
            }
        } else {
            let max = (1u64 << u32::from(bit_depth)) - 1;
            let rounded = f64::from(rounded);
            let clamped = if rounded.is_nan() || rounded <= 0.0 {
                0
            } else if rounded >= max as f64 {
                max
            } else {
                rounded as u64
            };
            if bit_depth <= 8 {
                out.push(clamped as u8);
            } else if bit_depth <= 16 {
                out.extend_from_slice(&(clamped as u16).to_le_bytes());
            } else {
                let bytes = clamped.to_le_bytes();
                let byte_count = native_bytes_per_sample(bit_depth).unwrap_or(8);
                out.extend_from_slice(&bytes[..byte_count]);
            }
        }
    }

    /// Decode the image into the given buffer.
    ///
    /// This method does the same as [`Image::decode`], but you can provide
    /// a custom buffer for the output, as well as a decoder context. Doing
    /// so allows the internal decode engine to reuse memory allocations, so
    /// this is especially recommended if you plan on converting multiple
    /// images in the same session.
    ///
    /// The buffer must have the correct size.
    pub fn decode_into(
        &self,
        buf: &mut [u8],
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<()> {
        let mut decoded_image = self.prepare_decoded_image(decoder_context)?;
        validate_interleaved_output_buffer(&decoded_image, buf)?;
        interleave_and_convert(&mut decoded_image, buf)?;

        Ok(())
    }

    fn prepare_decoded_image<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
    ) -> Result<DecodedImage<'ctx>> {
        self.prepare_decoded_image_with_region(decoder_context, None)
    }

    fn prepare_decoded_image_with_ht_decoder<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
        ht_decoder: &mut dyn HtCodeBlockDecoder,
    ) -> Result<DecodedImage<'ctx>> {
        self.prepare_decoded_image_with_region_and_ht_decoder(
            decoder_context,
            None,
            Some(ht_decoder),
        )
    }

    fn prepare_decoded_image_with_region<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
        output_region: Option<(u32, u32, u32, u32)>,
    ) -> Result<DecodedImage<'ctx>> {
        self.prepare_decoded_image_with_region_and_ht_decoder(decoder_context, output_region, None)
    }

    fn prepare_decoded_image_with_region_and_ht_decoder<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
        output_region: Option<(u32, u32, u32, u32)>,
        ht_decoder: Option<&mut dyn HtCodeBlockDecoder>,
    ) -> Result<DecodedImage<'ctx>> {
        let settings = &self.settings;
        self.decode_with_output_region_and_ht_decoder(decoder_context, output_region, ht_decoder)?;
        let mut decoded_image = DecodedImage {
            decoded_components: &mut decoder_context.tile_decode_context.channel_data,
            boxes: self.boxes.clone(),
        };

        if settings.resolve_palette_indices {
            let components = core::mem::take(decoded_image.decoded_components);
            *decoded_image.decoded_components =
                resolve_palette_indices(components, &decoded_image.boxes)?;
        }

        if let Some(cdef) = &decoded_image.boxes.channel_definition {
            validate_channel_definition(cdef, decoded_image.decoded_components.len())?;
            let mut components = decoded_image
                .decoded_components
                .iter()
                .cloned()
                .zip(
                    cdef.channel_definitions
                        .iter()
                        .map(|c| match c._association {
                            ChannelAssociation::WholeImage => u16::MAX,
                            ChannelAssociation::Colour(c) => c,
                            ChannelAssociation::Unspecified => u16::MAX,
                        }),
                )
                .collect::<Vec<_>>();
            components.sort_by_key(|component| component.1);
            *decoded_image.decoded_components = components.into_iter().map(|c| c.0).collect();
        }

        let bit_depth = decoded_image.decoded_components[0].bit_depth;
        convert_color_space(&mut decoded_image, bit_depth)?;
        Ok(decoded_image)
    }

    fn decode_with_output_region(
        &self,
        decoder_context: &mut DecoderContext<'a>,
        output_region: Option<(u32, u32, u32, u32)>,
    ) -> Result<()> {
        self.decode_with_output_region_and_ht_decoder(decoder_context, output_region, None)
    }

    fn decode_with_output_region_and_ht_decoder(
        &self,
        decoder_context: &mut DecoderContext<'a>,
        output_region: Option<(u32, u32, u32, u32)>,
        mut ht_decoder: Option<&mut dyn HtCodeBlockDecoder>,
    ) -> Result<()> {
        decoder_context.set_output_region(output_region);
        let decode_result = j2c::decode(
            self.codestream,
            &self.header,
            decoder_context,
            &mut ht_decoder,
        );
        decoder_context.set_output_region(None);
        decode_result
    }
}

#[cfg(test)]
mod tests;
