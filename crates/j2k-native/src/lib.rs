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

extern crate alloc;

#[cfg(test)]
use alloc::{vec, vec::Vec};

use crate::error::bail;
#[cfg(test)]
use crate::j2c::ComponentData;
#[cfg(test)]
use crate::jp2::colr::CieLab;

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
    ColorError, DecodeError, DecodeErrorClass, DecodingError, DirectPlanUnsupportedReason,
    FormatError, MarkerError, Result, TileError, ValidationError,
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
        usize::try_from(u64::from(width) * u64::from(height))
            .map_err(|_| ValidationError::ImageTooLarge.into())
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

mod scalar;
#[doc(hidden)]
pub use scalar::{
    collect_ht_cleanup_encode_distribution, decode_ht_code_block_scalar,
    decode_ht_code_block_scalar_until_phase, decode_ht_code_block_scalar_with_workspace,
    decode_ht_code_block_scalar_with_workspace_profiled, decode_j2k_code_block_scalar,
    decode_j2k_code_block_scalar_profiled, decode_j2k_code_block_scalar_with_workspace,
    decode_j2k_code_block_scalar_with_workspace_profiled, decode_j2k_sub_band_scalar,
    deinterleave_reference, encode_ht_code_block_scalar, encode_ht_code_block_scalar_with_passes,
    encode_j2k_code_block_scalar_with_style, encode_j2k_packetization_scalar,
    forward_dwt53_reference, forward_dwt97_reference, forward_ict_reference, forward_rct_reference,
    pack_j2k_code_block_scalar_from_tier1_tokens, quantize_reversible_reference,
    quantize_subband_reference, try_deinterleave_reference, HtCodeBlockDecodeProfile,
    HtCodeBlockDecodeWorkspace, J2kCodeBlockDecodeProfile, J2kCodeBlockDecodeWorkspace,
};

/// JP2 signature box: 00 00 00 0C 6A 50 20 20
pub(crate) const JP2_MAGIC: &[u8] = b"\x00\x00\x00\x0C\x6A\x50\x20\x20";
/// Codestream signature: FF 4F FF 51 (SOC + SIZ markers)
pub(crate) const CODESTREAM_MAGIC: &[u8] = b"\xFF\x4F\xFF\x51";

mod image;
pub use image::{DecodeSettings, Image};

#[cfg(test)]
mod tests;
