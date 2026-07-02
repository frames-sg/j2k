/*!
Internal pure-Rust JPEG 2000 codec engine for `j2k`.

This module tree was imported from the `dicom-toolkit-jpeg2000` 0.5.0 crate
and adapted in-repo so `j2k` no longer depends on an external
production decoder crate.

`dicom-toolkit-jpeg2000` is the JPEG 2000 engine used by `dicom-toolkit-rs`.
It is a maintained fork of the original `hayro-jpeg2000` project with
DICOM-focused extensions, including native-bit-depth decode for 8/12/16-bit
images and pure-Rust JPEG 2000 encoding.

The crate can decode raw JPEG 2000 codestreams (`.j2c`) and still-image JP2/JPH
wrappers. It implements the JPEG 2000 core coding system (ISO/IEC 15444-1) and
HTJ2K block coding (ISO/IEC 15444-15) through the support boundary documented in
`docs/public-support.md`. The remaining declared gaps are tracked there instead
of being hidden behind a broad "vast majority" claim.

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
(both in terms of raw performance but also memory allocations). However, there
are some more important optimizations that have not been implemented yet, so
there is definitely still room for improvement (and I am planning on implementing
them eventually).

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
use crate::jp2::cdef::{ChannelAssociation, ChannelType};
use crate::jp2::cmap::ComponentMappingType;
use crate::jp2::colr::{CieLab, EnumeratedColorspace};
use crate::jp2::icc::ICCMetadata;
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

macro_rules! impl_component_plane_metadata_accessors {
    () => {
        /// Width and height of this plane in output samples.
        pub fn dimensions(&self) -> (u32, u32) {
            self.dimensions
        }

        /// Horizontal and vertical SIZ sampling factors (`XRsiz`, `YRsiz`) for
        /// the source component represented by this plane.
        pub fn sampling(&self) -> (u8, u8) {
            self.sampling
        }

        /// Bit depth of this component plane.
        pub fn bit_depth(&self) -> u8 {
            self.bit_depth
        }

        /// Whether this component plane stores signed sample values.
        pub fn signed(&self) -> bool {
            self.signed
        }
    };
}

pub mod error;
mod inspect;
#[macro_use]
pub(crate) mod log;
mod direct_cpu;
mod direct_plan;
pub(crate) mod math;
#[doc(hidden)]
pub mod packet_math;
pub(crate) mod profile;
pub(crate) mod writer;

use crate::math::{dispatch, f32x8, Level, Simd, SIMD_WIDTH};
pub use direct_cpu::{
    execute_direct_color_plan_rgb8_into, execute_direct_color_plan_rgba8_into, J2kDirectCpuScratch,
};
pub use direct_plan::{
    HtOwnedCodeBlockBatchJob, HtOwnedSubBandPlan, J2kDirectBandId, J2kDirectColorPlan,
    J2kDirectGrayscalePlan, J2kDirectGrayscaleStep, J2kDirectIdwtStep, J2kDirectStoreStep,
    J2kOwnedCodeBlockBatchJob, J2kOwnedSubBandPlan,
};
pub use inspect::{
    inspect_j2k_codestream_header, looks_like_j2k_codestream, J2kCodestreamComponentHeader,
    J2kCodestreamHeaderError, J2kCodestreamHeaderMetadata,
};

/// Maps an output coordinate within an IDWT step to the source sub-band index.
///
/// `origin` is the global coordinate of the IDWT output rectangle,
/// `local_coord` is the coordinate within that output rectangle, and
/// `low_pass` selects the low-pass (`LL`/`LH`) or high-pass (`HL`/`HH`) band
/// along one axis. This helper is exposed so backend adapters can compute
/// required input windows with the same odd-origin rounding as the native IDWT.
#[must_use]
pub fn idwt_band_index(origin: u32, local_coord: u32, low_pass: bool) -> u32 {
    let global = u64::from(origin) + u64::from(local_coord);
    let origin = u64::from(origin);
    let index = if low_pass {
        global.div_ceil(2).saturating_sub(origin.div_ceil(2))
    } else {
        (global / 2).saturating_sub(origin / 2)
    };
    u32::try_from(index).unwrap_or(u32::MAX)
}

pub use error::{
    ColorError, DecodeError, DecodingError, FormatError, MarkerError, Result, TileError,
    ValidationError,
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
pub use j2k_types::{
    CpuOnlyJ2kEncodeStageAccelerator, EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock,
    IrreversibleQuantizationStep, IrreversibleQuantizationSubbandScales, J2kCodeBlockSegment,
    J2kCodeBlockStyle, J2kDeinterleaveToF32Job, J2kEncodeDispatchReport, J2kForwardDwt53Job,
    J2kForwardDwt53Level, J2kForwardDwt53Output, J2kForwardDwt97Job, J2kForwardDwt97Level,
    J2kForwardDwt97Output, J2kForwardIctJob, J2kForwardRctJob, J2kHtCodeBlockEncodeJob,
    J2kHtSubbandEncodeJob, J2kHtj2kTileEncodeJob, J2kPacketizationBlockCodingMode,
    J2kPacketizationCodeBlock, J2kPacketizationEncodeJob, J2kPacketizationPacketDescriptor,
    J2kPacketizationProgressionOrder, J2kPacketizationResolution, J2kPacketizationSubband,
    J2kQuantizeSubbandJob, J2kSubBandType, J2kTier1CodeBlockEncodeJob, PrecomputedHtj2k53Component,
    PrecomputedHtj2k53Image, PrecomputedHtj2k97Component, PrecomputedHtj2k97Image,
    PreencodedHtj2k97CodeBlock, PreencodedHtj2k97CompactCodeBlock,
    PreencodedHtj2k97CompactComponent, PreencodedHtj2k97CompactImage,
    PreencodedHtj2k97CompactResolution, PreencodedHtj2k97CompactSubband,
    PreencodedHtj2k97Component, PreencodedHtj2k97Image, PreencodedHtj2k97Resolution,
    PreencodedHtj2k97Subband, PrequantizedHtj2k97CodeBlock, PrequantizedHtj2k97Component,
    PrequantizedHtj2k97Image, PrequantizedHtj2k97Resolution, PrequantizedHtj2k97Subband,
};

mod j2c;
mod jp2;
pub(crate) mod reader;
pub use j2c::ht_encode_tables::HtUvlcTableEntry;

const MAX_CLASSIC_DECODE_BITPLANES: u8 = j2c::MAX_BITPLANE_COUNT;
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

define_ht_code_block_job! {
    /// Adapter HTJ2K code-block job description for backend experimentation.
    #[derive(Debug, Clone, Copy)]
    pub struct HtCodeBlockDecodeJob<'a> {
        /// Combined cleanup/refinement bytes for the code block.
        pub data: &'a [u8],
    }
}

/// Adapter HTJ2K scalar decode phase limit for backend experimentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HtCodeBlockDecodePhaseLimit {
    /// Stop after the cleanup pass has produced coefficient magnitudes/signs.
    Cleanup,
    /// Stop after the significance propagation refinement pass.
    SignificancePropagation,
    /// Decode through the magnitude refinement pass when present.
    MagnitudeRefinement,
}

/// Adapter HTJ2K batched code-block decode job for one sub-band.
#[derive(Debug, Clone, Copy)]
pub struct HtCodeBlockBatchJob<'a> {
    /// X offset within the target sub-band coefficient buffer.
    pub output_x: u32,
    /// Y offset within the target sub-band coefficient buffer.
    pub output_y: u32,
    /// The actual code-block decode parameters.
    pub code_block: HtCodeBlockDecodeJob<'a>,
}

/// Adapter HTJ2K batched sub-band decode request for backend experimentation.
#[derive(Debug, Clone, Copy)]
pub struct HtSubBandDecodeJob<'a> {
    /// Sub-band width in samples.
    pub width: u32,
    /// Sub-band height in samples.
    pub height: u32,
    /// Code blocks to decode into this sub-band.
    pub jobs: &'a [HtCodeBlockBatchJob<'a>],
}

/// Adapter Classic Tier-1 compact token segment for backend experimentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct J2kTier1TokenSegment {
    /// Bit offset of this segment within the compact token buffer.
    pub token_bit_offset: u32,
    /// Number of token bits in this segment.
    ///
    /// Arithmetic segments contain 6-bit MQ tokens. Raw bypass segments contain
    /// one bit per raw bypass event.
    pub token_bit_count: u32,
    /// First coding pass covered by this segment.
    pub start_coding_pass: u8,
    /// One-past-last coding pass covered by this segment.
    pub end_coding_pass: u8,
    /// Whether this segment should be packed through the MQ arithmetic path.
    pub use_arithmetic: bool,
}

/// Adapter classic J2K code-block job description for backend experimentation.
#[derive(Debug, Clone, Copy)]
pub struct J2kCodeBlockDecodeJob<'a> {
    /// Combined payload bytes for all coded segments in this code block.
    pub data: &'a [u8],
    /// Coded segments for the code block.
    pub segments: &'a [J2kCodeBlockSegment],
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
    pub total_bitplanes: u8,
    /// Region-of-interest maxshift value from RGN marker metadata.
    pub roi_shift: u8,
    /// The sub-band type containing this code block.
    pub sub_band_type: J2kSubBandType,
    /// The code-block style flags.
    pub style: J2kCodeBlockStyle,
    /// Whether strict decode validation is enabled for the parent image.
    pub strict: bool,
    /// Dequantization step to apply to decoded coefficients.
    pub dequantization_step: f32,
}

/// Adapter HTJ2K cleanup-encode shape counters for backend benchmarking.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HtCleanupEncodeDistribution {
    /// Total 2x2 cleanup quads visited.
    pub total_quads: u64,
    /// Quads encoded in the first cleanup row pair.
    pub initial_quads: u64,
    /// Quads encoded after the first cleanup row pair.
    pub non_initial_quads: u64,
    /// All-quad `rho` histogram, indexed by the low four `rho` bits.
    pub rho_counts: [u64; 16],
    /// First-row-pair `rho` histogram, indexed by the low four `rho` bits.
    pub initial_rho_counts: [u64; 16],
    /// Non-initial-row `rho` histogram, indexed by the low four `rho` bits.
    pub non_initial_rho_counts: [u64; 16],
    /// Non-initial-row `u_q` histogram.
    pub non_initial_u_q_counts: [u64; 32],
    /// Non-initial-row `e_qmax` histogram.
    pub non_initial_e_qmax_counts: [u64; 32],
    /// Non-initial-row `kappa` histogram.
    pub non_initial_kappa_counts: [u64; 32],
    /// Non-initial-row joint `rho`/`u_q` histogram.
    pub non_initial_rho_u_q_counts: [[u64; 32]; 16],
    /// Calls that emitted at least one magnitude/sign sample.
    pub mag_sign_calls: u64,
    /// Magnitude/sign call histogram, indexed by the low four `rho` bits.
    pub mag_sign_rho_counts: [u64; 16],
    /// Encoded magnitude/sign sample payload bit-count histogram.
    pub mag_sign_sample_bit_counts: [u64; 32],
    /// Number of individual magnitude/sign samples emitted.
    pub mag_sign_encoded_samples: u64,
}

/// Adapter JPEG 2000 encode-stage accelerator for backend experimentation.
pub trait J2kEncodeStageAccelerator {
    /// Report cumulative backend dispatches completed by this accelerator.
    fn dispatch_report(&self) -> J2kEncodeDispatchReport {
        J2kEncodeDispatchReport::default()
    }

    /// Optionally deinterleave interleaved pixel bytes into f32 component planes.
    ///
    /// Return `Ok(Some(components))` with one plane per component. Return
    /// `Ok(None)` to use the CPU fallback.
    fn encode_deinterleave(
        &mut self,
        _job: J2kDeinterleaveToF32Job<'_>,
    ) -> core::result::Result<Option<Vec<Vec<f32>>>, &'static str> {
        Ok(None)
    }

    /// Optionally apply forward RCT in place.
    ///
    /// Return `Ok(true)` after writing transformed planes. Return `Ok(false)`
    /// to use the CPU fallback.
    fn encode_forward_rct(
        &mut self,
        _job: J2kForwardRctJob<'_>,
    ) -> core::result::Result<bool, &'static str> {
        Ok(false)
    }

    /// Optionally apply forward ICT in place.
    ///
    /// Return `Ok(true)` after writing transformed planes. Return `Ok(false)`
    /// to use the CPU fallback.
    fn encode_forward_ict(
        &mut self,
        _job: J2kForwardIctJob<'_>,
    ) -> core::result::Result<bool, &'static str> {
        Ok(false)
    }

    /// Optionally run a forward reversible 5/3 DWT.
    ///
    /// Return `Ok(Some(output))` with all subbands populated. Return
    /// `Ok(None)` to use the CPU fallback.
    fn encode_forward_dwt53(
        &mut self,
        _job: J2kForwardDwt53Job<'_>,
    ) -> core::result::Result<Option<J2kForwardDwt53Output>, &'static str> {
        Ok(None)
    }

    /// Optionally run a forward irreversible 9/7 DWT.
    ///
    /// Return `Ok(Some(output))` with all subbands populated. Return
    /// `Ok(None)` to use the CPU fallback.
    fn encode_forward_dwt97(
        &mut self,
        _job: J2kForwardDwt97Job<'_>,
    ) -> core::result::Result<Option<J2kForwardDwt97Output>, &'static str> {
        Ok(None)
    }

    /// Optionally quantize one sub-band.
    ///
    /// Return `Ok(Some(coefficients))` with one quantized coefficient for each
    /// input coefficient. Return `Ok(None)` to use the CPU fallback.
    fn encode_quantize_subband(
        &mut self,
        _job: J2kQuantizeSubbandJob<'_>,
    ) -> core::result::Result<Option<Vec<i32>>, &'static str> {
        Ok(None)
    }

    /// Optionally encode one classic Tier-1 code-block.
    ///
    /// Return `Ok(Some(output))` with encoded bytes and pass metadata. Return
    /// `Ok(None)` to use the CPU fallback.
    fn encode_tier1_code_block(
        &mut self,
        _job: J2kTier1CodeBlockEncodeJob<'_>,
    ) -> core::result::Result<Option<EncodedJ2kCodeBlock>, &'static str> {
        Ok(None)
    }

    /// Optionally encode multiple classic Tier-1 code-blocks in one backend dispatch.
    ///
    /// Return `Ok(Some(outputs))` with one encoded output per input job. Return
    /// `Ok(None)` to use the per-block hook or CPU fallback.
    fn encode_tier1_code_blocks(
        &mut self,
        _jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
    ) -> core::result::Result<Option<Vec<EncodedJ2kCodeBlock>>, &'static str> {
        Ok(None)
    }

    /// Optionally encode one HTJ2K code-block.
    ///
    /// Return `Ok(Some(output))` with encoded bytes and pass metadata. Return
    /// `Ok(None)` to use the CPU fallback.
    fn encode_ht_code_block(
        &mut self,
        _job: J2kHtCodeBlockEncodeJob<'_>,
    ) -> core::result::Result<Option<EncodedHtJ2kCodeBlock>, &'static str> {
        Ok(None)
    }

    /// Optionally encode multiple HTJ2K code-blocks in one backend dispatch.
    ///
    /// Return `Ok(Some(outputs))` with one encoded output per input job. Return
    /// `Ok(None)` to use the per-block hook or CPU fallback.
    fn encode_ht_code_blocks(
        &mut self,
        _jobs: &[J2kHtCodeBlockEncodeJob<'_>],
    ) -> core::result::Result<Option<Vec<EncodedHtJ2kCodeBlock>>, &'static str> {
        Ok(None)
    }

    /// Optionally quantize and encode one HTJ2K cleanup/refinement sub-band.
    ///
    /// Return `Ok(Some(outputs))` with one encoded output per code block in
    /// raster code-block order. Return `Ok(None)` to use the separate
    /// quantization and code-block hooks or CPU fallback.
    fn encode_ht_subband(
        &mut self,
        _job: J2kHtSubbandEncodeJob<'_>,
    ) -> core::result::Result<Option<Vec<EncodedHtJ2kCodeBlock>>, &'static str> {
        Ok(None)
    }

    /// Optionally encode the complete HTJ2K tile packet body.
    ///
    /// Return `Ok(Some(bytes))` with the complete tile bitstream body. CPU
    /// marker/header writing remains outside this hook. Return `Ok(None)` to
    /// use the normal staged encode pipeline.
    fn encode_htj2k_tile(
        &mut self,
        _job: J2kHtj2kTileEncodeJob<'_>,
    ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
        Ok(None)
    }

    /// Return whether native CPU code-block fallback should use internal rayon parallelism.
    ///
    /// External accelerators keep serial per-block fallback so their hooks still
    /// observe every fallback block after a declined batch hook.
    fn prefer_parallel_cpu_code_block_fallback(&self) -> bool {
        false
    }

    /// Return whether whole-tile CPU-only batch encode may be parallelized by callers.
    ///
    /// This is narrower than [`Self::prefer_parallel_cpu_code_block_fallback`]:
    /// callers must only bypass the supplied accelerator when it is known to
    /// have no observable hooks.
    fn prefer_parallel_cpu_tile_encode(&self) -> bool {
        false
    }

    /// Optionally packetize prepared packet contributions.
    ///
    /// Return `Ok(Some(bytes))` with the complete tile bitstream. Return
    /// `Ok(None)` to use the CPU fallback.
    fn encode_packetization(
        &mut self,
        _job: J2kPacketizationEncodeJob<'_>,
    ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
        Ok(None)
    }
}

impl J2kEncodeStageAccelerator for CpuOnlyJ2kEncodeStageAccelerator {
    fn prefer_parallel_cpu_code_block_fallback(&self) -> bool {
        true
    }

    fn prefer_parallel_cpu_tile_encode(&self) -> bool {
        true
    }
}

/// Adapter classic J2K batched code-block decode job for one sub-band.
#[derive(Debug, Clone, Copy)]
pub struct J2kCodeBlockBatchJob<'a> {
    /// X offset within the target sub-band coefficient buffer.
    pub output_x: u32,
    /// Y offset within the target sub-band coefficient buffer.
    pub output_y: u32,
    /// The actual code-block decode parameters.
    pub code_block: J2kCodeBlockDecodeJob<'a>,
}

/// Adapter classic J2K batched sub-band decode request for backend experimentation.
#[derive(Debug, Clone, Copy)]
pub struct J2kSubBandDecodeJob<'a> {
    /// Sub-band width in samples.
    pub width: u32,
    /// Sub-band height in samples.
    pub height: u32,
    /// Code blocks to decode into this sub-band.
    pub jobs: &'a [J2kCodeBlockBatchJob<'a>],
}

/// Adapter integer rectangle for backend experimentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct J2kRect {
    /// Inclusive minimum x coordinate.
    pub x0: u32,
    /// Inclusive minimum y coordinate.
    pub y0: u32,
    /// Exclusive maximum x coordinate.
    pub x1: u32,
    /// Exclusive maximum y coordinate.
    pub y1: u32,
}

impl J2kRect {
    /// Rectangle width in samples.
    pub fn width(self) -> u32 {
        self.x1.saturating_sub(self.x0)
    }

    /// Rectangle height in samples.
    pub fn height(self) -> u32 {
        self.y1.saturating_sub(self.y0)
    }
}

/// Adapter wavelet transform selector for backend experimentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum J2kWaveletTransform {
    /// Reversible 5/3 transform.
    Reversible53,
    /// Irreversible 9/7 transform.
    Irreversible97,
}

/// Adapter single sub-band payload for backend experimentation.
#[derive(Debug, Clone, Copy)]
pub struct J2kIdwtBand<'a> {
    /// Rect covered by this band.
    pub rect: J2kRect,
    /// Band coefficients in row-major order.
    pub coefficients: &'a [f32],
}

/// Adapter single-decomposition IDWT job for backend experimentation.
#[derive(Debug, Clone, Copy)]
pub struct J2kSingleDecompositionIdwtJob<'a> {
    /// Output rect of the decomposition level.
    pub rect: J2kRect,
    /// Transform to apply.
    pub transform: J2kWaveletTransform,
    /// LL band input.
    pub ll: J2kIdwtBand<'a>,
    /// HL band input.
    pub hl: J2kIdwtBand<'a>,
    /// LH band input.
    pub lh: J2kIdwtBand<'a>,
    /// HH band input.
    pub hh: J2kIdwtBand<'a>,
}

/// Adapter inverse MCT job for backend experimentation.
#[derive(Debug)]
pub struct J2kInverseMctJob<'a> {
    /// Transform to apply.
    pub transform: J2kWaveletTransform,
    /// First component plane, updated in place.
    pub plane0: &'a mut [f32],
    /// Second component plane, updated in place.
    pub plane1: &'a mut [f32],
    /// Third component plane, updated in place.
    pub plane2: &'a mut [f32],
    /// Constant sign-shift addend applied to the first plane after inverse MCT.
    pub addend0: f32,
    /// Constant sign-shift addend applied to the second plane after inverse MCT.
    pub addend1: f32,
    /// Constant sign-shift addend applied to the third plane after inverse MCT.
    pub addend2: f32,
}

/// Adapter component-store job for backend experimentation.
#[derive(Debug)]
pub struct J2kStoreComponentJob<'a> {
    /// Source IDWT coefficients in row-major order.
    pub input: &'a [f32],
    /// Source row width.
    pub input_width: u32,
    /// Source x offset to begin copying from.
    pub source_x: u32,
    /// Source y offset to begin copying from.
    pub source_y: u32,
    /// Number of samples to copy per row.
    pub copy_width: u32,
    /// Number of rows to copy.
    pub copy_height: u32,
    /// Destination component plane in row-major order.
    pub output: &'a mut [f32],
    /// Destination row width.
    pub output_width: u32,
    /// Destination x offset to begin writing at.
    pub output_x: u32,
    /// Destination y offset to begin writing at.
    pub output_y: u32,
    /// Constant value added to every copied sample.
    pub addend: f32,
}

/// Adapter HTJ2K code-block decode hook for backend experimentation.
pub trait HtCodeBlockDecoder {
    /// Optionally decode a full classic J2K sub-band in one batch.
    ///
    /// Implementations should return `Ok(true)` if they handled the request and
    /// wrote the decoded coefficients into `output`. Returning `Ok(false)`
    /// falls back to per-code-block decode via `decode_j2k_code_block`.
    fn decode_j2k_sub_band(
        &mut self,
        _job: J2kSubBandDecodeJob<'_>,
        _output: &mut [f32],
    ) -> Result<bool> {
        Ok(false)
    }

    /// Optionally decode one classic J2K code block.
    ///
    /// Implementations should return `Ok(true)` if they handled the request
    /// and wrote the decoded coefficients into `output`. Returning `Ok(false)`
    /// falls back to the scalar bitplane decoder.
    fn decode_j2k_code_block(
        &mut self,
        _job: J2kCodeBlockDecodeJob<'_>,
        _output: &mut [f32],
    ) -> Result<bool> {
        Ok(false)
    }

    /// Optionally decode a full HTJ2K sub-band in one batch.
    ///
    /// Implementations should return `Ok(true)` if they handled the request and
    /// wrote the decoded coefficients into `output`. Returning `Ok(false)`
    /// falls back to per-code-block decode via `decode_code_block`.
    fn decode_sub_band(
        &mut self,
        _job: HtSubBandDecodeJob<'_>,
        _output: &mut [f32],
    ) -> Result<bool> {
        Ok(false)
    }

    /// Optionally decode one single-decomposition IDWT level on a backend.
    ///
    /// Implementations should return `Ok(true)` if they handled the request
    /// and wrote the transformed coefficients into `output`. Returning
    /// `Ok(false)` falls back to the scalar/SIMD CPU IDWT path.
    fn decode_single_decomposition_idwt(
        &mut self,
        _job: J2kSingleDecompositionIdwtJob<'_>,
        _output: &mut [f32],
    ) -> Result<bool> {
        Ok(false)
    }

    /// Optionally apply inverse MCT on a backend.
    ///
    /// Implementations should return `Ok(true)` if they handled the request
    /// and updated the component planes in place. Returning `Ok(false)` falls
    /// back to the scalar/SIMD CPU MCT path.
    fn decode_inverse_mct(&mut self, _job: J2kInverseMctJob<'_>) -> Result<bool> {
        Ok(false)
    }

    /// Optionally store one component plane on a backend.
    ///
    /// Implementations should return `Ok(true)` if they handled the request
    /// and updated the destination plane in place. Returning `Ok(false)` falls
    /// back to the CPU store path.
    fn decode_store_component(&mut self, _job: J2kStoreComponentJob<'_>) -> Result<bool> {
        Ok(false)
    }

    /// Decode one HTJ2K code block into `output`, writing `job.width` samples per row.
    fn decode_code_block(
        &mut self,
        job: HtCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> Result<()>;
}

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

pub(crate) fn add_roi_shift_to_bitplanes(
    bitplanes: u8,
    roi_shift: u8,
    max_bitplanes: u8,
) -> Result<u8> {
    let Some(coded_bitplanes) = bitplanes.checked_add(roi_shift) else {
        bail!(DecodingError::TooManyBitplanes);
    };
    if coded_bitplanes > max_bitplanes {
        bail!(DecodingError::TooManyBitplanes);
    }
    Ok(coded_bitplanes)
}

pub(crate) fn apply_roi_maxshift_inverse_i64(coefficient: i64, roi_shift: u8) -> i64 {
    if roi_shift == 0 || coefficient == 0 {
        return coefficient;
    }

    let magnitude = coefficient.unsigned_abs();
    let threshold = 1_u64.checked_shl(roi_shift as u32).unwrap_or(u64::MAX);
    if magnitude < threshold {
        return coefficient;
    }

    let shifted = magnitude >> roi_shift;
    let shifted = shifted.min(i64::MAX as u64) as i64;
    if coefficient < 0 {
        -shifted
    } else {
        shifted
    }
}

pub(crate) fn apply_roi_maxshift_inverse_i32(coefficient: i32, roi_shift: u8) -> i32 {
    apply_roi_maxshift_inverse_i64(i64::from(coefficient), roi_shift)
        .clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

/// Adapter scalar classic J2K encoder helper for backend experimentation.
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
pub fn forward_rct_reference(mut planes: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    j2c::forward_mct::forward_rct(&mut planes);
    planes
}

/// Adapter scalar forward ICT reference for Metal/CUDA stage parity.
///
/// Applies the native CPU forward Irreversible Color Transform to three
/// component planes supplied as owned `Vec<f32>` arrays. The transform is
/// applied in place and the mutated planes are returned.
pub fn forward_ict_reference(mut planes: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    j2c::forward_mct::forward_ict(&mut planes);
    planes
}

/// Adapter scalar sub-band quantization reference for backend stage parity.
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

/// Adapter scalar pixel deinterleave/level-shift reference for CUDA stage parity.
///
/// Converts interleaved pixel bytes to per-component f32 planes with the
/// same level-shift logic as the native CPU encode path.  The result is one
/// `Vec<f32>` per component, each of length `num_pixels`.
pub fn deinterleave_reference(
    pixels: &[u8],
    num_pixels: usize,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
) -> Vec<Vec<f32>> {
    j2c::encode::deinterleave_to_f32(pixels, num_pixels, num_components, bit_depth, signed)
}

/// Adapter scalar Tier-2 packetization helper for backend experimentation.
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
pub fn decode_j2k_code_block_scalar(
    job: J2kCodeBlockDecodeJob<'_>,
    output: &mut [f32],
) -> Result<()> {
    let mut workspace = J2kCodeBlockDecodeWorkspace::default();
    decode_j2k_code_block_scalar_with_workspace(job, output, &mut workspace)
}

/// Reusable scratch for scalar classic J2K code-block decoding.
#[derive(Default)]
pub struct J2kCodeBlockDecodeWorkspace {
    bit_plane_decode_context: j2c::bitplane::BitPlaneDecodeContext,
}

/// Adapter scalar classic J2K decoder helper that reuses caller-provided scratch.
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
pub fn decode_j2k_code_block_scalar_profiled(
    job: J2kCodeBlockDecodeJob<'_>,
    output: &mut [f32],
    profile: &mut J2kCodeBlockDecodeProfile,
) -> Result<()> {
    let mut workspace = J2kCodeBlockDecodeWorkspace::default();
    decode_j2k_code_block_scalar_with_workspace_profiled(job, output, &mut workspace, profile)
}

/// Adapter scalar classic J2K decoder helper that records pass timings and reuses scratch.
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
pub fn decode_ht_code_block_scalar(
    job: HtCodeBlockDecodeJob<'_>,
    output: &mut [f32],
) -> Result<()> {
    decode_ht_code_block_scalar_for_phase::<{ j2c::ht_block_decode::PHASE_LIMIT_MAGREF }>(
        job, output,
    )
}

/// Adapter scalar HTJ2K decoder helper that stops after the selected phase.
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

/// Adapter HTJ2K SigProp benchmark state for backend experimentation.
pub struct HtSigPropBenchmarkState(j2c::ht_block_decode::HtSigPropBenchmarkState);

impl HtSigPropBenchmarkState {
    /// Coefficient buffer length required by `decode_ht_sigprop_benchmark_state`.
    pub fn output_len(&self) -> usize {
        self.0.output_len()
    }
}

/// Adapter helper that precomputes cleanup-derived SigProp inputs for benchmarks.
pub fn prepare_ht_sigprop_benchmark_state(
    job: HtCodeBlockDecodeJob<'_>,
) -> Result<HtSigPropBenchmarkState> {
    let segments = j2c::ht_block_decode::HtCodeBlockSegments::from_combined_payload(
        job.data,
        job.cleanup_length,
        job.refinement_length,
    )?;
    let state = j2c::ht_block_decode::prepare_sigprop_benchmark_state(
        &segments,
        job.missing_bit_planes,
        job.num_bitplanes,
        job.number_of_coding_passes,
        job.stripe_causal,
        job.strict,
        job.width,
        job.height,
        job.width,
    )?;
    Ok(HtSigPropBenchmarkState(state))
}

/// Adapter helper that runs only the HTJ2K significance-propagation phase.
pub fn decode_ht_sigprop_benchmark_state(
    state: &mut HtSigPropBenchmarkState,
    output: &mut [u32],
) -> Result<()> {
    j2c::ht_block_decode::decode_sigprop_benchmark_state(&mut state.0, output)
}

/// Adapter HTJ2K VLC table 0 for backend experimentation.
pub fn ht_vlc_table0() -> &'static [u16; 1024] {
    &j2c::ht_tables::VLC_TABLE0
}

/// Adapter HTJ2K VLC table 1 for backend experimentation.
pub fn ht_vlc_table1() -> &'static [u16; 1024] {
    &j2c::ht_tables::VLC_TABLE1
}

/// Adapter HTJ2K UVLC table 0 for backend experimentation.
pub fn ht_uvlc_table0() -> &'static [u16; 320] {
    &j2c::ht_tables::UVLC_TABLE0
}

/// Adapter HTJ2K UVLC table 1 for backend experimentation.
pub fn ht_uvlc_table1() -> &'static [u16; 256] {
    &j2c::ht_tables::UVLC_TABLE1
}

/// Adapter HTJ2K cleanup encoder VLC table 0 for backend experimentation.
pub fn ht_vlc_encode_table0() -> &'static [u16; 2048] {
    &j2c::ht_encode_tables::HT_VLC_ENCODE_TABLE0
}

/// Adapter HTJ2K cleanup encoder VLC table 1 for backend experimentation.
pub fn ht_vlc_encode_table1() -> &'static [u16; 2048] {
    &j2c::ht_encode_tables::HT_VLC_ENCODE_TABLE1
}

/// Adapter HTJ2K cleanup encoder UVLC table for backend experimentation.
pub fn ht_uvlc_encode_table() -> &'static [HtUvlcTableEntry; 75] {
    &j2c::ht_encode_tables::HT_UVLC_ENCODE_TABLE
}

/// Adapter HTJ2K cleanup encoder UVLC table packed for byte-addressed backends.
pub fn ht_uvlc_encode_table_bytes() -> &'static [u8] {
    &j2c::ht_encode_tables::HT_UVLC_ENCODE_TABLE_BYTES
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
    /// It is recommended to leave this flag disabled, unless you have a
    /// specific reason not to.
    pub strict: bool,
    /// A hint for the target resolution that the image should be decoded at.
    pub target_resolution: Option<(u32, u32)>,
}

impl Default for DecodeSettings {
    fn default() -> Self {
        Self {
            resolve_palette_indices: true,
            strict: false,
            target_resolution: None,
        }
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
    pub fn build_direct_grayscale_plan_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<J2kDirectGrayscalePlan> {
        if !matches!(self.color_space, ColorSpace::Gray) || self.has_alpha {
            bail!(DecodingError::UnsupportedFeature(
                "direct grayscale plan only supports grayscale images without alpha"
            ));
        }

        j2c::build_direct_grayscale_plan(self.codestream, &self.header, decoder_context)
    }

    /// Build a adapter grayscale direct device plan for an output-space region.
    pub fn build_direct_grayscale_plan_region_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
        output_region: (u32, u32, u32, u32),
    ) -> Result<J2kDirectGrayscalePlan> {
        if !matches!(self.color_space, ColorSpace::Gray) || self.has_alpha {
            bail!(DecodingError::UnsupportedFeature(
                "direct grayscale plan only supports grayscale images without alpha"
            ));
        }

        decoder_context.set_output_region(Some(output_region));
        let result =
            j2c::build_direct_grayscale_plan(self.codestream, &self.header, decoder_context);
        decoder_context.set_output_region(None);
        result
    }

    /// Build a adapter RGB direct device plan without materializing host component planes.
    pub fn build_direct_color_plan_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<J2kDirectColorPlan> {
        if !matches!(self.color_space, ColorSpace::RGB) || self.has_alpha {
            bail!(DecodingError::UnsupportedFeature(
                "direct color plan only supports RGB images without alpha"
            ));
        }

        j2c::build_direct_color_plan(self.codestream, &self.header, decoder_context)
    }

    /// Build a adapter RGB direct device plan for an output-space region.
    pub fn build_direct_color_plan_region_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
        output_region: (u32, u32, u32, u32),
    ) -> Result<J2kDirectColorPlan> {
        if !matches!(self.color_space, ColorSpace::RGB) || self.has_alpha {
            bail!(DecodingError::UnsupportedFeature(
                "direct color plan only supports RGB images without alpha"
            ));
        }

        decoder_context.set_output_region(Some(output_region));
        let result = j2c::build_direct_color_plan(self.codestream, &self.header, decoder_context);
        decoder_context.set_output_region(None);
        result
    }

    /// Decode borrowed component planes while delegating HTJ2K code-block decode.
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
    pub fn decode_reversible_53_coefficients(&self) -> Result<Reversible53CoefficientImage> {
        let mut decoder_context = DecoderContext::default();
        self.decode_reversible_53_coefficients_with_context(&mut decoder_context)
    }

    /// Extract reversible 5/3 wavelet coefficients using a caller-provided
    /// decoder context.
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

fn validate_channel_definition(
    cdef: &jp2::cdef::ChannelDefinitionBox,
    component_count: usize,
) -> Result<()> {
    if cdef.channel_definitions.len() != component_count {
        bail!(ValidationError::InvalidChannelDefinition);
    }

    let mut seen_color_associations = vec![false; component_count];
    for definition in &cdef.channel_definitions {
        if let ChannelAssociation::Colour(association) = definition._association {
            let Some(index) = association.checked_sub(1).map(usize::from) else {
                bail!(ValidationError::InvalidChannelDefinition);
            };
            if index >= component_count || seen_color_associations[index] {
                bail!(ValidationError::InvalidChannelDefinition);
            }
            seen_color_associations[index] = true;
        }
    }

    Ok(())
}

pub(crate) fn resolve_alpha_and_color_space(
    boxes: &ImageBoxes,
    header: &Header<'_>,
    settings: &DecodeSettings,
) -> Result<(ColorSpace, bool)> {
    let mut num_components = header.component_infos.len();

    // Override number of components with what is actually in the palette box
    // in case we resolve them.
    if settings.resolve_palette_indices {
        if let Some(palette_box) = &boxes.palette {
            num_components = palette_box.columns.len();
        }
    }

    let mut has_alpha = false;

    if let Some(cdef) = &boxes.channel_definition {
        has_alpha = cdef.channel_definitions.iter().any(|definition| {
            matches!(
                definition.channel_type,
                ChannelType::Opacity | ChannelType::PremultipliedOpacity
            )
        });
    }

    let mut color_space = get_color_space(boxes, num_components)?;

    // If we didn't resolve palette indices, we need to assume grayscale image.
    if !settings.resolve_palette_indices && boxes.palette.is_some() {
        has_alpha = false;
        color_space = ColorSpace::Gray;
    }

    let actual_num_components = header.component_infos.len();

    // Validate the number of channels.
    if boxes.palette.is_none()
        && actual_num_components != usize::from(color_space.num_channels() + u16::from(has_alpha))
    {
        if !settings.strict
            && actual_num_components == usize::from(color_space.num_channels()) + 1
            && !has_alpha
        {
            // See OPENJPEG test case orb-blue10-lin-j2k. Assume that we have an
            // alpha channel in this case.
            has_alpha = true;
        } else {
            // Color space is invalid, attempt to repair.
            if actual_num_components == 1 || (actual_num_components == 2 && has_alpha) {
                color_space = ColorSpace::Gray;
            } else if actual_num_components == 3 {
                color_space = ColorSpace::RGB;
            } else if actual_num_components == 4 {
                if has_alpha {
                    color_space = ColorSpace::RGB;
                } else {
                    color_space = ColorSpace::CMYK;
                }
            } else {
                color_space = ColorSpace::Unknown {
                    num_channels: u16::try_from(actual_num_components)
                        .map_err(|_| ValidationError::TooManyChannels)?,
                };
            }
        }
    }

    Ok((color_space, has_alpha))
}

/// The color space of the image.
#[derive(Debug, Clone)]
pub enum ColorSpace {
    /// A grayscale image.
    Gray,
    /// An RGB image.
    RGB,
    /// A CMYK image.
    CMYK,
    /// An unknown color space.
    Unknown {
        /// The number of channels of the color space.
        num_channels: u16,
    },
    /// An image based on an ICC profile.
    Icc {
        /// The raw data of the ICC profile.
        profile: Vec<u8>,
        /// The number of channels used by the ICC profile.
        num_channels: u16,
    },
}

impl ColorSpace {
    /// Return the number of expected channels for the color space.
    pub fn num_channels(&self) -> u16 {
        match self {
            Self::Gray => 1,
            Self::RGB => 3,
            Self::CMYK => 4,
            Self::Unknown { num_channels } => *num_channels,
            Self::Icc {
                num_channels: num_components,
                ..
            } => *num_components,
        }
    }
}

/// A bitmap storing the decoded result of the image.
pub struct Bitmap {
    /// The color space of the image.
    pub color_space: ColorSpace,
    /// The raw pixel data of the image. The result will always be in
    /// 8-bit (in case the original image had a different bit-depth, this
    /// decode path scales it to 8-bit).
    ///
    /// The size is guaranteed to equal
    /// `width * height * (num_channels + (if has_alpha { 1 } else { 0 })`.
    /// Pixels are interleaved on a per-channel basis, the alpha channel always
    /// appearing as the last channel, if available.
    pub data: Vec<u8>,
    /// Whether the image has an alpha channel.
    pub has_alpha: bool,
    /// The width of the image.
    pub width: u32,
    /// The height of the image.
    pub height: u32,
    /// The original bit depth of the image. You usually don't need to do anything
    /// with this parameter, it just exists for informational purposes.
    pub original_bit_depth: u8,
}

/// Raw decoded pixel data at native bit depth (no 8-bit scaling).
///
/// For bit depths ≤ 8, `data` contains one byte per sample.
/// For bit depths > 8 (e.g., 12 or 16), `data` contains two bytes per sample
/// in little-endian byte order (`u16` LE).
///
/// Samples are interleaved: for a 3-component image, the layout is
/// `[R0, G0, B0, R1, G1, B1, ...]`.
pub struct RawBitmap {
    /// The raw pixel data at native bit depth.
    pub data: Vec<u8>,
    /// The width of the image in pixels.
    pub width: u32,
    /// The height of the image in pixels.
    pub height: u32,
    /// The original bit depth per sample (e.g., 8, 12, 16).
    pub bit_depth: u8,
    /// Whether every component in this packed bitmap is signed.
    ///
    /// Use [`Self::component_signed`] for per-component signedness when
    /// handling arbitrary JPEG 2000 component metadata.
    pub signed: bool,
    /// Per-component signedness in codestream/component order.
    pub component_signed: Vec<bool>,
    /// The number of components (e.g., 1 for grayscale, 3 for RGB).
    pub num_components: u16,
    /// Bytes per sample in the packed little-endian native representation.
    pub bytes_per_sample: u8,
}

/// One owned decoded component plane at native bit depth.
pub struct NativeComponentPlane {
    data: Vec<u8>,
    dimensions: (u32, u32),
    bit_depth: u8,
    signed: bool,
    sampling: (u8, u8),
    bytes_per_sample: u8,
}

impl NativeComponentPlane {
    /// Packed little-endian sample bytes for this component in row-major order.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    impl_component_plane_metadata_accessors!();

    /// Bytes used for each packed little-endian sample in [`Self::data`].
    pub fn bytes_per_sample(&self) -> u8 {
        self.bytes_per_sample
    }
}

/// Owned decoded native-bit-depth component planes for an image.
pub struct DecodedNativeComponents {
    dimensions: (u32, u32),
    color_space: ColorSpace,
    has_alpha: bool,
    planes: Vec<NativeComponentPlane>,
}

impl DecodedNativeComponents {
    /// Dimensions of the decoded image represented by these planes.
    pub fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    /// Color space after JPEG 2000 color conversion has been applied.
    pub fn color_space(&self) -> &ColorSpace {
        &self.color_space
    }

    /// Whether the decoded image has an alpha channel.
    pub fn has_alpha(&self) -> bool {
        self.has_alpha
    }

    /// Decoded component planes in display order.
    pub fn planes(&self) -> &[NativeComponentPlane] {
        &self.planes
    }
}

/// A borrowed decoded component plane.
pub struct ComponentPlane<'a> {
    samples: &'a [f32],
    dimensions: (u32, u32),
    bit_depth: u8,
    signed: bool,
    sampling: (u8, u8),
}

impl<'a> ComponentPlane<'a> {
    /// Component samples in row-major order.
    pub fn samples(&self) -> &'a [f32] {
        self.samples
    }

    impl_component_plane_metadata_accessors!();
}

/// Borrowed decoded component planes for an image.
pub struct DecodedComponents<'a> {
    dimensions: (u32, u32),
    color_space: ColorSpace,
    has_alpha: bool,
    planes: Vec<ComponentPlane<'a>>,
}

impl<'a> DecodedComponents<'a> {
    /// Dimensions of the decoded image represented by these planes.
    pub fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    /// Color space after JPEG 2000 color conversion has been applied.
    pub fn color_space(&self) -> &ColorSpace {
        &self.color_space
    }

    /// Whether the decoded image has an alpha channel.
    pub fn has_alpha(&self) -> bool {
        self.has_alpha
    }

    /// Borrowed decoded component planes in display order.
    pub fn planes(&self) -> &[ComponentPlane<'a>] {
        &self.planes
    }
}

fn validate_interleaved_output_buffer(image: &DecodedImage<'_>, buf: &[u8]) -> Result<()> {
    let required_len = interleaved_output_len(image)?;
    if buf.len() < required_len {
        bail!(DecodingError::OutputBufferTooSmall);
    }
    Ok(())
}

fn interleaved_output_len(image: &DecodedImage<'_>) -> Result<usize> {
    let Some(first) = image.decoded_components.first() else {
        bail!(DecodingError::CodeBlockDecodeFailure);
    };
    first
        .container
        .truncated()
        .len()
        .checked_mul(image.decoded_components.len())
        .ok_or(ValidationError::ImageTooLarge.into())
}

fn interleave_and_convert(image: &mut DecodedImage<'_>, buf: &mut [u8]) -> Result<()> {
    let components = &mut *image.decoded_components;
    let num_components = components.len();

    let mut all_same_bit_depth = Some(components[0].bit_depth);

    for component in components.iter().skip(1) {
        if Some(component.bit_depth) != all_same_bit_depth {
            all_same_bit_depth = None;
        }
    }

    let max_len = components[0].container.truncated().len();

    let mut output_iter = buf.iter_mut();

    if all_same_bit_depth == Some(8) && num_components <= 4 {
        // Fast path for the common case.
        match num_components {
            // Gray-scale.
            1 => {
                for (output, input) in output_iter.zip(
                    components[0]
                        .container
                        .iter()
                        .map(|v| math::round_f32(*v) as u8),
                ) {
                    *output = input;
                }
            }
            // Gray-scale with alpha.
            2 => {
                let c0 = &components[0];
                let c1 = &components[1];

                let c0 = &c0.container[..max_len];
                let c1 = &c1.container[..max_len];

                for i in 0..max_len {
                    *output_iter.next().unwrap() = math::round_f32(c0[i]) as u8;
                    *output_iter.next().unwrap() = math::round_f32(c1[i]) as u8;
                }
            }
            // RGB
            3 => {
                let c0 = &components[0];
                let c1 = &components[1];
                let c2 = &components[2];

                let c0 = &c0.container[..max_len];
                let c1 = &c1.container[..max_len];
                let c2 = &c2.container[..max_len];

                for i in 0..max_len {
                    *output_iter.next().unwrap() = math::round_f32(c0[i]) as u8;
                    *output_iter.next().unwrap() = math::round_f32(c1[i]) as u8;
                    *output_iter.next().unwrap() = math::round_f32(c2[i]) as u8;
                }
            }
            // RGBA or CMYK.
            4 => {
                let c0 = &components[0];
                let c1 = &components[1];
                let c2 = &components[2];
                let c3 = &components[3];

                let c0 = &c0.container[..max_len];
                let c1 = &c1.container[..max_len];
                let c2 = &c2.container[..max_len];
                let c3 = &c3.container[..max_len];

                for i in 0..max_len {
                    *output_iter.next().unwrap() = math::round_f32(c0[i]) as u8;
                    *output_iter.next().unwrap() = math::round_f32(c1[i]) as u8;
                    *output_iter.next().unwrap() = math::round_f32(c2[i]) as u8;
                    *output_iter.next().unwrap() = math::round_f32(c3[i]) as u8;
                }
            }
            _ => bail!(ValidationError::TooManyChannels),
        }
    } else {
        // Slow path that also requires us to scale to 8 bit.
        let mul_factor = ((1 << 8) - 1) as f32;

        for sample in 0..max_len {
            for channel in components.iter() {
                *output_iter.next().unwrap() = math::round_f32(
                    (channel.container[sample]
                        / ((1_u64 << u32::from(channel.bit_depth)) - 1) as f32)
                        * mul_factor,
                ) as u8;
            }
        }
    }

    Ok(())
}

fn interleave_and_convert_region(
    image: &mut DecodedImage<'_>,
    image_width: usize,
    roi: (u32, u32, u32, u32),
    buf: &mut [u8],
) {
    let components = &mut *image.decoded_components;
    let num_components = components.len();
    let (x, y, width, height) = roi;
    let mut output_iter = buf.iter_mut();

    let mut all_same_bit_depth = Some(components[0].bit_depth);
    for component in components.iter().skip(1) {
        if Some(component.bit_depth) != all_same_bit_depth {
            all_same_bit_depth = None;
        }
    }

    if all_same_bit_depth == Some(8) && num_components <= 4 {
        for row in y as usize..(y + height) as usize {
            let row_base = row * image_width;
            for col in x as usize..(x + width) as usize {
                let idx = row_base + col;
                for component in components.iter() {
                    *output_iter.next().unwrap() = math::round_f32(component.container[idx]) as u8;
                }
            }
        }
    } else {
        let mul_factor = ((1 << 8) - 1) as f32;
        for row in y as usize..(y + height) as usize {
            let row_base = row * image_width;
            for col in x as usize..(x + width) as usize {
                let idx = row_base + col;
                for component in components.iter() {
                    *output_iter.next().unwrap() = math::round_f32(
                        (component.container[idx]
                            / ((1_u64 << u32::from(component.bit_depth)) - 1) as f32)
                            * mul_factor,
                    ) as u8;
                }
            }
        }
    }
}

fn validate_roi(dims: (u32, u32), roi: (u32, u32, u32, u32)) -> Result<()> {
    let (image_width, image_height) = dims;
    let (x, y, width, height) = roi;
    let x_end = x
        .checked_add(width)
        .ok_or(ValidationError::InvalidDimensions)?;
    let y_end = y
        .checked_add(height)
        .ok_or(ValidationError::InvalidDimensions)?;
    if x_end > image_width || y_end > image_height {
        return Err(ValidationError::InvalidDimensions.into());
    }
    Ok(())
}

fn native_component_plane_dimensions(
    reference_dimensions: (u32, u32),
    sampling: (u8, u8),
    sample_count: usize,
) -> Result<(u32, u32)> {
    let reference_sample_count =
        checked_decode_sample_count(reference_dimensions.0, reference_dimensions.1)?;
    if sample_count == reference_sample_count {
        return Ok(reference_dimensions);
    }

    let (x_rsiz, y_rsiz) = sampling;
    if x_rsiz == 0 || y_rsiz == 0 {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    let sampled_dimensions = (
        reference_dimensions.0.div_ceil(u32::from(x_rsiz)),
        reference_dimensions.1.div_ceil(u32::from(y_rsiz)),
    );
    let sampled_sample_count =
        checked_decode_sample_count(sampled_dimensions.0, sampled_dimensions.1)?;
    if sample_count == sampled_sample_count {
        return Ok(sampled_dimensions);
    }

    bail!(DecodingError::CodeBlockDecodeFailure)
}

fn convert_color_space(image: &mut DecodedImage<'_>, bit_depth: u8) -> Result<()> {
    if let Some(jp2::colr::ColorSpace::Enumerated(e)) = &image
        .boxes
        .color_specification
        .as_ref()
        .map(|i| &i.color_space)
    {
        match e {
            EnumeratedColorspace::Sycc => {
                dispatch!(Level::new(), simd => {
                    sycc_to_rgb(simd, image.decoded_components, bit_depth)
                })?;
            }
            EnumeratedColorspace::CieLab(cielab) => {
                dispatch!(Level::new(), simd => {
                    cielab_to_rgb(simd, image.decoded_components, bit_depth, cielab)
                })?;
            }
            _ => {}
        }
    }

    Ok(())
}

fn get_color_space(boxes: &ImageBoxes, num_components: usize) -> Result<ColorSpace> {
    let cs = match boxes
        .color_specification
        .as_ref()
        .map(|c| &c.color_space)
        .unwrap_or(&jp2::colr::ColorSpace::Unknown)
    {
        jp2::colr::ColorSpace::Enumerated(e) => {
            match e {
                EnumeratedColorspace::Cmyk => ColorSpace::CMYK,
                EnumeratedColorspace::Srgb => ColorSpace::RGB,
                EnumeratedColorspace::RommRgb => {
                    // Use an ICC profile to process the RommRGB color space.
                    ColorSpace::Icc {
                        profile: include_bytes!("../assets/ProPhoto-v2-micro.icc").to_vec(),
                        num_channels: 3,
                    }
                }
                EnumeratedColorspace::EsRgb => ColorSpace::RGB,
                EnumeratedColorspace::Greyscale => ColorSpace::Gray,
                EnumeratedColorspace::Sycc => ColorSpace::RGB,
                EnumeratedColorspace::CieLab(_) => ColorSpace::Icc {
                    profile: include_bytes!("../assets/LAB.icc").to_vec(),
                    num_channels: 3,
                },
                _ => bail!(FormatError::Unsupported),
            }
        }
        jp2::colr::ColorSpace::Icc(icc) => {
            if let Some(metadata) = ICCMetadata::from_data(icc) {
                ColorSpace::Icc {
                    profile: icc.clone(),
                    num_channels: u16::from(metadata.color_space.num_components()),
                }
            } else {
                // See OPENJPEG test orb-blue10-lin-jp2.jp2. They seem to
                // assume RGB in this case (even though the image has 4
                // components with no opacity channel, they assume RGBA instead
                // of CMYK).
                ColorSpace::RGB
            }
        }
        jp2::colr::ColorSpace::Unknown => match num_components {
            1 => ColorSpace::Gray,
            3 => ColorSpace::RGB,
            4 => ColorSpace::CMYK,
            _ => ColorSpace::Unknown {
                num_channels: u16::try_from(num_components).unwrap_or(u16::MAX),
            },
        },
    };

    Ok(cs)
}

fn resolve_palette_indices(
    components: Vec<ComponentData>,
    boxes: &ImageBoxes,
) -> Result<Vec<ComponentData>> {
    let Some(palette) = boxes.palette.as_ref() else {
        // Nothing to resolve.
        return Ok(components);
    };

    let Some(mapping) = boxes.component_mapping.as_ref() else {
        bail!(ColorError::PaletteResolutionFailed);
    };
    if mapping.entries.is_empty() {
        bail!(ColorError::PaletteResolutionFailed);
    }

    let mut resolved = Vec::with_capacity(mapping.entries.len());

    for entry in &mapping.entries {
        let component_idx = entry.component_index as usize;
        let component = components
            .get(component_idx)
            .ok_or(ColorError::PaletteResolutionFailed)?;

        match entry.mapping_type {
            ComponentMappingType::Direct => resolved.push(component.clone()),
            ComponentMappingType::Palette { column } => {
                let column_idx = column as usize;
                let column_info = palette
                    .columns
                    .get(column_idx)
                    .ok_or(ColorError::PaletteResolutionFailed)?;

                let mut mapped =
                    Vec::with_capacity(component.container.truncated().len() + SIMD_WIDTH);

                for &sample in component.container.truncated() {
                    let index = math::round_f32(sample) as i64;
                    let raw = palette
                        .map(index as usize, column_idx)
                        .ok_or(ColorError::PaletteResolutionFailed)?;
                    let value = if column_info.signed {
                        sign_extend_palette_value(raw, column_info.bit_depth) as f32
                    } else {
                        raw as f32
                    };
                    mapped.push(value);
                }

                resolved.push(ComponentData {
                    container: math::SimdBuffer::new(mapped),
                    integer_container: None,
                    bit_depth: column_info.bit_depth,
                    signed: column_info.signed,
                });
            }
        }
    }

    Ok(resolved)
}

fn sign_extend_palette_value(raw: u64, bit_depth: u8) -> i64 {
    if bit_depth == 0 {
        return raw as i64;
    }
    if bit_depth >= 64 {
        return raw as i64;
    }

    let mask = (1_u64 << bit_depth) - 1;
    let value = raw & mask;
    let shift = 64 - u32::from(bit_depth);
    ((value << shift) as i64) >> shift
}

#[inline(always)]
fn cielab_to_rgb<S: Simd>(
    simd: S,
    components: &mut [ComponentData],
    bit_depth: u8,
    lab: &CieLab,
) -> Result<()> {
    let (head, _) = components
        .split_at_mut_checked(3)
        .ok_or(ColorError::LabConversionFailed)?;

    let [l, a, b] = head else {
        bail!(ColorError::LabConversionFailed);
    };

    let prec0 = l.bit_depth;
    let prec1 = a.bit_depth;
    let prec2 = b.bit_depth;

    // Prevent underflows/divisions by zero further below.
    if prec0 < 4 || prec1 < 4 || prec2 < 4 {
        bail!(ColorError::LabConversionFailed);
    }

    let rl = lab.rl.unwrap_or(100);
    let ra = lab.ra.unwrap_or(170);
    let rb = lab.ra.unwrap_or(200);
    let ol = lab.ol.unwrap_or(0);
    let default_oa = (1_u64 << u32::from(bit_depth - 1)).min(u64::from(u32::MAX)) as u32;
    let default_ob = ((1_u64 << u32::from(bit_depth - 2)) + (1_u64 << u32::from(bit_depth - 3)))
        .min(u64::from(u32::MAX)) as u32;
    let oa = lab.oa.unwrap_or(default_oa);
    let ob = lab.ob.unwrap_or(default_ob);

    // Copied from OpenJPEG.
    let min_l = -(rl as f32 * ol as f32) / ((1_u64 << u32::from(prec0)) - 1) as f32;
    let max_l = min_l + rl as f32;
    let min_a = -(ra as f32 * oa as f32) / ((1_u64 << u32::from(prec1)) - 1) as f32;
    let max_a = min_a + ra as f32;
    let min_b = -(rb as f32 * ob as f32) / ((1_u64 << u32::from(prec2)) - 1) as f32;
    let max_b = min_b + rb as f32;

    let bit_max = ((1_u64 << u32::from(bit_depth)) - 1).min(u64::from(u32::MAX)) as u32;

    // Note that we are not doing the actual conversion with the ICC profile yet,
    // just decoding the raw LAB values.
    // We leave applying the ICC profile to the user.
    let divisor_l = ((1_u64 << u32::from(prec0)) - 1) as f32;
    let divisor_a = ((1_u64 << u32::from(prec1)) - 1) as f32;
    let divisor_b = ((1_u64 << u32::from(prec2)) - 1) as f32;

    let scale_l_final = bit_max as f32 / 100.0;
    let scale_ab_final = bit_max as f32 / 255.0;

    let l_offset = min_l * scale_l_final;
    let l_scale = (max_l - min_l) / divisor_l * scale_l_final;
    let a_offset = (min_a + 128.0) * scale_ab_final;
    let a_scale = (max_a - min_a) / divisor_a * scale_ab_final;
    let b_offset = (min_b + 128.0) * scale_ab_final;
    let b_scale = (max_b - min_b) / divisor_b * scale_ab_final;

    let l_offset_v = f32x8::splat(simd, l_offset);
    let l_scale_v = f32x8::splat(simd, l_scale);
    let a_offset_v = f32x8::splat(simd, a_offset);
    let a_scale_v = f32x8::splat(simd, a_scale);
    let b_offset_v = f32x8::splat(simd, b_offset);
    let b_scale_v = f32x8::splat(simd, b_scale);

    // Note that we are not doing the actual conversion with the ICC profile yet,
    // just decoding the raw LAB values.
    // We leave applying the ICC profile to the user.
    for ((l_chunk, a_chunk), b_chunk) in l
        .container
        .chunks_exact_mut(SIMD_WIDTH)
        .zip(a.container.chunks_exact_mut(SIMD_WIDTH))
        .zip(b.container.chunks_exact_mut(SIMD_WIDTH))
    {
        let l_v = f32x8::from_slice(simd, l_chunk);
        let a_v = f32x8::from_slice(simd, a_chunk);
        let b_v = f32x8::from_slice(simd, b_chunk);

        l_v.mul_add(l_scale_v, l_offset_v).store(l_chunk);
        a_v.mul_add(a_scale_v, a_offset_v).store(a_chunk);
        b_v.mul_add(b_scale_v, b_offset_v).store(b_chunk);
    }

    Ok(())
}

#[inline(always)]
fn sycc_to_rgb<S: Simd>(simd: S, components: &mut [ComponentData], bit_depth: u8) -> Result<()> {
    let offset = (1_u64 << (u32::from(bit_depth) - 1)) as f32;
    let max_value = ((1_u64 << u32::from(bit_depth)) - 1) as f32;

    let (head, _) = components
        .split_at_mut_checked(3)
        .ok_or(ColorError::SyccConversionFailed)?;

    let [y, cb, cr] = head else {
        bail!(ColorError::SyccConversionFailed);
    };

    let offset_v = f32x8::splat(simd, offset);
    let max_v = f32x8::splat(simd, max_value);
    let zero_v = f32x8::splat(simd, 0.0);
    let cr_to_r = f32x8::splat(simd, 1.402);
    let cb_to_g = f32x8::splat(simd, -0.344136);
    let cr_to_g = f32x8::splat(simd, -0.714136);
    let cb_to_b = f32x8::splat(simd, 1.772);

    for ((y_chunk, cb_chunk), cr_chunk) in y
        .container
        .chunks_exact_mut(SIMD_WIDTH)
        .zip(cb.container.chunks_exact_mut(SIMD_WIDTH))
        .zip(cr.container.chunks_exact_mut(SIMD_WIDTH))
    {
        let y_v = f32x8::from_slice(simd, y_chunk);
        let cb_v = f32x8::from_slice(simd, cb_chunk) - offset_v;
        let cr_v = f32x8::from_slice(simd, cr_chunk) - offset_v;

        // r = y + 1.402 * cr
        let r = cr_v.mul_add(cr_to_r, y_v);
        // g = y - 0.344136 * cb - 0.714136 * cr
        let g = cr_v.mul_add(cr_to_g, cb_v.mul_add(cb_to_g, y_v));
        // b = y + 1.772 * cb
        let b = cb_v.mul_add(cb_to_b, y_v);

        r.min(max_v).max(zero_v).store(y_chunk);
        g.min(max_v).max(zero_v).store(cb_chunk);
        b.min(max_v).max(zero_v).store(cr_chunk);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ht_uvlc_encode_table_bytes_match_entry_packing_order() {
        let entries = ht_uvlc_encode_table();
        let bytes = ht_uvlc_encode_table_bytes();

        assert_eq!(bytes.len(), entries.len() * 6);
        for (index, entry) in entries.iter().enumerate() {
            let offset = index * 6;
            assert_eq!(
                &bytes[offset..offset + 6],
                &[
                    entry.pre,
                    entry.pre_len,
                    entry.suf,
                    entry.suf_len,
                    entry.ext,
                    entry.ext_len
                ],
            );
        }
    }

    #[test]
    fn roi_maxshift_inverse_preserves_background_and_unshifts_roi_coefficients() {
        assert_eq!(apply_roi_maxshift_inverse_i32(127, 7), 127);
        assert_eq!(apply_roi_maxshift_inverse_i32(-127, 7), -127);
        assert_eq!(apply_roi_maxshift_inverse_i32(128, 7), 1);
        assert_eq!(apply_roi_maxshift_inverse_i32(-128, 7), -1);
        assert_eq!(apply_roi_maxshift_inverse_i32(255, 7), 1);
        assert_eq!(apply_roi_maxshift_inverse_i32(-255, 7), -1);
        assert_eq!(apply_roi_maxshift_inverse_i32(256, 7), 2);
        assert_eq!(apply_roi_maxshift_inverse_i32(-256, 7), -2);
        assert_eq!(apply_roi_maxshift_inverse_i32(42, 0), 42);
        assert_eq!(apply_roi_maxshift_inverse_i64(1_i64 << 38, 7), 1_i64 << 31);
        assert_eq!(
            apply_roi_maxshift_inverse_i64(-(1_i64 << 38), 7),
            -(1_i64 << 31)
        );
    }

    #[test]
    fn classic_decode_adapter_accepts_legal_38_bit_roi_bitplane_count() {
        assert_eq!(
            add_roi_shift_to_bitplanes(38, 0, MAX_CLASSIC_DECODE_BITPLANES).unwrap(),
            38
        );
        assert_eq!(
            add_roi_shift_to_bitplanes(37, 1, MAX_CLASSIC_DECODE_BITPLANES).unwrap(),
            38
        );
    }

    #[test]
    fn classic_scalar_decode_applies_nonzero_roi_maxshift() {
        let roi_shift = 3;
        let total_bitplanes = 3;
        let style = J2kCodeBlockStyle {
            selective_arithmetic_coding_bypass: false,
            reset_context_probabilities: false,
            termination_on_each_pass: false,
            vertically_causal_context: false,
            segmentation_symbols: false,
        };
        let coded_coefficients = [0, 5, 1 << roi_shift, -(2 << roi_shift)];
        let encoded = encode_j2k_code_block_scalar_with_style(
            &coded_coefficients,
            2,
            2,
            J2kSubBandType::LowLow,
            total_bitplanes + roi_shift,
            style,
        )
        .expect("encode ROI-shifted code block");
        let job = J2kCodeBlockDecodeJob {
            data: &encoded.data,
            segments: &encoded.segments,
            width: 2,
            height: 2,
            output_stride: 2,
            missing_bit_planes: encoded.missing_bit_planes,
            number_of_coding_passes: encoded.number_of_coding_passes,
            total_bitplanes,
            roi_shift,
            sub_band_type: J2kSubBandType::LowLow,
            style,
            strict: true,
            dequantization_step: 1.0,
        };
        let mut output = [0.0; 4];

        decode_j2k_code_block_scalar(job, &mut output).expect("decode ROI-shifted code block");

        assert_eq!(output, [0.0, 5.0, 1.0, -2.0]);
    }

    #[test]
    fn classic_scalar_token_pack_matches_scalar_single_cleanup_block() {
        let style = J2kCodeBlockStyle {
            selective_arithmetic_coding_bypass: true,
            reset_context_probabilities: false,
            termination_on_each_pass: false,
            vertically_causal_context: false,
            segmentation_symbols: false,
        };
        let scalar =
            encode_j2k_code_block_scalar_with_style(&[1], 1, 1, J2kSubBandType::LowLow, 1, style)
                .expect("encode scalar");
        let token_bytes = pack_mq_test_tokens(&[(0, 1), (9, 0)]);
        let packed = pack_j2k_code_block_scalar_from_tier1_tokens(
            &token_bytes,
            &[J2kTier1TokenSegment {
                token_bit_offset: 0,
                token_bit_count: 12,
                start_coding_pass: 0,
                end_coding_pass: 1,
                use_arithmetic: true,
            }],
            scalar.number_of_coding_passes,
            scalar.missing_bit_planes,
        )
        .expect("pack tokens");

        assert_eq!(packed.data, scalar.data);
        assert_eq!(packed.segments, scalar.segments);
        assert_eq!(
            packed.number_of_coding_passes,
            scalar.number_of_coding_passes
        );
        assert_eq!(packed.missing_bit_planes, scalar.missing_bit_planes);
    }

    fn pack_mq_test_tokens(tokens: &[(u8, u8)]) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut current = 0u8;
        let mut bits = 0u8;
        for &(ctx, bit) in tokens {
            let value = (ctx & 0x1F) | ((bit & 1) << 5);
            for shift in (0..6).rev() {
                current = (current << 1) | ((value >> shift) & 1);
                bits += 1;
                if bits == 8 {
                    bytes.push(current);
                    current = 0;
                    bits = 0;
                }
            }
        }
        if bits != 0 {
            bytes.push(current << (8 - bits));
        }
        bytes
    }

    #[test]
    fn classic_scalar_profiled_decode_matches_unprofiled_decode() {
        let width = 64u32;
        let height = 64u32;
        let sample_count = width as usize * height as usize;
        let total_bitplanes = 12;
        let style = default_classic_test_style();
        let coefficients = (0..sample_count)
            .map(|idx| {
                let value = i32::try_from((idx * 37) % 4095).expect("sample value fits i32") - 2048;
                if idx % 17 == 0 {
                    0
                } else {
                    value
                }
            })
            .collect::<Vec<_>>();
        let encoded = encode_j2k_code_block_scalar_with_style(
            &coefficients,
            width,
            height,
            J2kSubBandType::LowLow,
            total_bitplanes,
            style,
        )
        .expect("encode classic block");
        let job = classic_low_low_decode_job(&encoded, width, height, total_bitplanes, style);
        let mut expected = vec![0.0_f32; sample_count];
        let mut actual = vec![0.0_f32; sample_count];
        let mut profile = J2kCodeBlockDecodeProfile::default();

        decode_j2k_code_block_scalar(job, &mut expected).expect("unprofiled classic decode");
        decode_j2k_code_block_scalar_profiled(job, &mut actual, &mut profile)
            .expect("profiled classic decode");

        assert_eq!(actual, expected);
        assert!(profile.cleanup_us > 0);
    }

    #[test]
    fn classic_scalar_workspace_reuse_matches_fresh_decode() {
        let total_bitplanes = 6;
        let style = default_classic_test_style();
        let mut workspace = J2kCodeBlockDecodeWorkspace::default();

        for (width, height, seed) in [(8, 8, 0x31), (4, 16, 0x47)] {
            let coefficients = (0..width * height)
                .map(|idx| {
                    let value = ((idx as i32 * seed) % 23) - 11;
                    if idx % 7 == 0 {
                        0
                    } else {
                        value
                    }
                })
                .collect::<Vec<_>>();
            let encoded = encode_j2k_code_block_scalar_with_style(
                &coefficients,
                width,
                height,
                J2kSubBandType::LowLow,
                total_bitplanes,
                style,
            )
            .expect("encode classic block");
            let job = classic_low_low_decode_job(&encoded, width, height, total_bitplanes, style);
            let mut fresh = vec![0.0_f32; width as usize * height as usize];
            let mut reused = vec![0.0_f32; width as usize * height as usize];

            decode_j2k_code_block_scalar(job, &mut fresh).expect("fresh classic decode");
            decode_j2k_code_block_scalar_with_workspace(job, &mut reused, &mut workspace)
                .expect("workspace classic decode");

            assert_eq!(reused, fresh);
        }
    }

    fn default_classic_test_style() -> J2kCodeBlockStyle {
        J2kCodeBlockStyle {
            selective_arithmetic_coding_bypass: false,
            reset_context_probabilities: false,
            termination_on_each_pass: false,
            vertically_causal_context: false,
            segmentation_symbols: false,
        }
    }

    fn classic_low_low_decode_job<'a>(
        encoded: &'a EncodedJ2kCodeBlock,
        width: u32,
        height: u32,
        total_bitplanes: u8,
        style: J2kCodeBlockStyle,
    ) -> J2kCodeBlockDecodeJob<'a> {
        J2kCodeBlockDecodeJob {
            data: &encoded.data,
            segments: &encoded.segments,
            width,
            height,
            output_stride: width as usize,
            missing_bit_planes: encoded.missing_bit_planes,
            number_of_coding_passes: encoded.number_of_coding_passes,
            total_bitplanes,
            roi_shift: 0,
            sub_band_type: J2kSubBandType::LowLow,
            style,
            strict: true,
            dequantization_step: 1.0,
        }
    }

    #[test]
    fn scalar_packetization_rejects_overflowing_ht_refinement_lengths_without_panic() {
        let payload = [0x12];
        let block = J2kPacketizationCodeBlock {
            data: &payload,
            ht_cleanup_length: u32::MAX,
            ht_refinement_length: 1,
            num_coding_passes: 3,
            num_zero_bitplanes: 2,
            previously_included: false,
            l_block: 3,
            block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
        };
        let subband = J2kPacketizationSubband {
            code_blocks: vec![block],
            num_cbs_x: 1,
            num_cbs_y: 1,
        };
        let resolution = J2kPacketizationResolution {
            subbands: vec![subband],
        };
        let resolutions = [resolution];
        let job = J2kPacketizationEncodeJob {
            resolution_count: 1,
            num_layers: 1,
            num_components: 1,
            code_block_count: 1,
            progression_order: J2kPacketizationProgressionOrder::Lrcp,
            packet_descriptors: &[],
            resolutions: &resolutions,
        };

        let err = encode_j2k_packetization_scalar(job)
            .expect_err("overflowing HT packetization segment lengths rejected");

        assert_eq!(err, "multi-pass HTJ2K packet contribution length overflow");
    }

    #[derive(Default)]
    struct DecodeWorkCounter {
        classic_code_blocks: usize,
        ht_code_blocks: usize,
        idwt_output_samples: usize,
    }

    impl DecodeWorkCounter {
        fn code_blocks(&self) -> usize {
            self.classic_code_blocks + self.ht_code_blocks
        }
    }

    struct FailingHtDecoder {
        called: bool,
    }

    impl HtCodeBlockDecoder for FailingHtDecoder {
        fn decode_code_block(
            &mut self,
            _job: HtCodeBlockDecodeJob<'_>,
            _output: &mut [f32],
        ) -> Result<()> {
            self.called = true;
            Err(DecodingError::CodeBlockDecodeFailure.into())
        }
    }

    struct FailingClassicDecoder {
        called: bool,
    }

    impl HtCodeBlockDecoder for FailingClassicDecoder {
        fn decode_code_block(
            &mut self,
            _job: HtCodeBlockDecodeJob<'_>,
            _output: &mut [f32],
        ) -> Result<()> {
            panic!("HT hook must not be used for classic J2K test")
        }

        fn decode_j2k_code_block(
            &mut self,
            _job: J2kCodeBlockDecodeJob<'_>,
            _output: &mut [f32],
        ) -> Result<bool> {
            self.called = true;
            Err(DecodingError::CodeBlockDecodeFailure.into())
        }
    }

    struct FailingClassicBatchDecoder {
        called: bool,
    }

    #[derive(Default)]
    struct CapturingHtDecoder {
        called: bool,
        blocks: usize,
        refinement_jobs: usize,
        max_coding_passes: u8,
    }

    impl HtCodeBlockDecoder for CapturingHtDecoder {
        fn decode_code_block(
            &mut self,
            job: HtCodeBlockDecodeJob<'_>,
            output: &mut [f32],
        ) -> Result<()> {
            self.called = true;
            self.blocks += 1;
            self.max_coding_passes = self.max_coding_passes.max(job.number_of_coding_passes);
            if job.refinement_length > 0 {
                self.refinement_jobs += 1;
                assert!(
                    job.number_of_coding_passes > 1,
                    "refinement bytes must correspond to refinement coding passes"
                );
            }

            decode_ht_code_block_scalar(job, output)
        }
    }

    #[derive(Clone)]
    struct CapturedHtDecodeJob {
        data: Vec<u8>,
        cleanup_length: u32,
        refinement_length: u32,
        width: u32,
        height: u32,
        output_stride: usize,
        missing_bit_planes: u8,
        number_of_coding_passes: u8,
        num_bitplanes: u8,
        roi_shift: u8,
        stripe_causal: bool,
        strict: bool,
        dequantization_step: f32,
    }

    impl CapturedHtDecodeJob {
        fn from_job(job: HtCodeBlockDecodeJob<'_>) -> Self {
            Self {
                data: job.data.to_vec(),
                cleanup_length: job.cleanup_length,
                refinement_length: job.refinement_length,
                width: job.width,
                height: job.height,
                output_stride: job.output_stride,
                missing_bit_planes: job.missing_bit_planes,
                number_of_coding_passes: job.number_of_coding_passes,
                num_bitplanes: job.num_bitplanes,
                roi_shift: job.roi_shift,
                stripe_causal: job.stripe_causal,
                strict: job.strict,
                dequantization_step: job.dequantization_step,
            }
        }

        fn borrowed(&self) -> HtCodeBlockDecodeJob<'_> {
            HtCodeBlockDecodeJob {
                data: &self.data,
                cleanup_length: self.cleanup_length,
                refinement_length: self.refinement_length,
                width: self.width,
                height: self.height,
                output_stride: self.output_stride,
                missing_bit_planes: self.missing_bit_planes,
                number_of_coding_passes: self.number_of_coding_passes,
                num_bitplanes: self.num_bitplanes,
                roi_shift: self.roi_shift,
                stripe_causal: self.stripe_causal,
                strict: self.strict,
                dequantization_step: self.dequantization_step,
            }
        }
    }

    #[derive(Default)]
    struct FirstHtJobDecoder {
        job: Option<CapturedHtDecodeJob>,
    }

    impl HtCodeBlockDecoder for FirstHtJobDecoder {
        fn decode_code_block(
            &mut self,
            job: HtCodeBlockDecodeJob<'_>,
            output: &mut [f32],
        ) -> Result<()> {
            if self.job.is_none() {
                self.job = Some(CapturedHtDecodeJob::from_job(job));
            }
            decode_ht_code_block_scalar(job, output)
        }
    }

    struct ZeroRefinementHtDecoder;

    impl HtCodeBlockDecoder for ZeroRefinementHtDecoder {
        fn decode_code_block(
            &mut self,
            job: HtCodeBlockDecodeJob<'_>,
            output: &mut [f32],
        ) -> Result<()> {
            let mut data = job.data.to_vec();
            let cleanup_len = job.cleanup_length as usize;
            let refinement_len = job.refinement_length as usize;
            data[cleanup_len..cleanup_len + refinement_len].fill(0);
            let zeroed = HtCodeBlockDecodeJob { data: &data, ..job };

            decode_ht_code_block_scalar(zeroed, output)
        }
    }

    #[derive(Default)]
    struct CleanupLimitedHtDecoder {
        blocks: usize,
        refinement_blocks: usize,
        cleanup_bytes: usize,
        refinement_bytes: usize,
    }

    impl HtCodeBlockDecoder for CleanupLimitedHtDecoder {
        fn decode_code_block(
            &mut self,
            job: HtCodeBlockDecodeJob<'_>,
            output: &mut [f32],
        ) -> Result<()> {
            self.blocks += 1;
            self.cleanup_bytes += job.cleanup_length as usize;
            if job.refinement_length > 0 {
                self.refinement_blocks += 1;
                self.refinement_bytes += job.refinement_length as usize;
            }

            decode_ht_code_block_scalar_until_phase(
                job,
                output,
                HtCodeBlockDecodePhaseLimit::Cleanup,
            )
        }
    }

    impl HtCodeBlockDecoder for FailingClassicBatchDecoder {
        fn decode_code_block(
            &mut self,
            _job: HtCodeBlockDecodeJob<'_>,
            _output: &mut [f32],
        ) -> Result<()> {
            panic!("HT hook must not be used for classic J2K batch test")
        }

        fn decode_j2k_code_block(
            &mut self,
            _job: J2kCodeBlockDecodeJob<'_>,
            _output: &mut [f32],
        ) -> Result<bool> {
            panic!(
                "per-block classic hook must not be used when the batch hook handles the sub-band"
            )
        }

        fn decode_j2k_sub_band(
            &mut self,
            _job: J2kSubBandDecodeJob<'_>,
            _output: &mut [f32],
        ) -> Result<bool> {
            self.called = true;
            Err(DecodingError::CodeBlockDecodeFailure.into())
        }
    }

    fn fixture() -> Vec<u8> {
        let pixels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        encode(&pixels, 2, 2, 3, 8, false, &options).expect("encode")
    }

    #[test]
    fn decode_into_rejects_short_output_buffer() {
        let bytes = fixture();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut output = vec![0; 11];

        let err = image
            .decode_into(&mut output, &mut context)
            .expect_err("short output buffer must be rejected");

        assert_eq!(
            err,
            DecodeError::Decoding(DecodingError::OutputBufferTooSmall)
        );
    }

    fn fixture_multi_block() -> Vec<u8> {
        let pixels: Vec<u8> = (0..64).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 0,
            code_block_width_exp: 0,
            code_block_height_exp: 0,
            ..EncodeOptions::default()
        };
        encode(&pixels, 8, 8, 1, 8, false, &options).expect("encode multi-block classic")
    }

    fn fixture_gray() -> Vec<u8> {
        let pixels: Vec<u8> = (0..16).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode classic gray8")
    }

    #[test]
    fn native_bytes_per_sample_tracks_high_bit_depths() {
        for (bit_depth, expected) in [
            (1_u8, 1_usize),
            (8, 1),
            (9, 2),
            (16, 2),
            (17, 3),
            (24, 3),
            (32, 4),
            (38, 5),
        ] {
            assert_eq!(native_bytes_per_sample(bit_depth).unwrap(), expected);
        }
    }

    #[test]
    fn native_sample_packing_writes_high_bit_unsigned_little_endian_bytes() {
        let mut out = Vec::new();
        Image::push_native_sample_bytes(&mut out, 0x12_34_56 as f32, 24, false);
        assert_eq!(out, [0x56, 0x34, 0x12]);

        out.clear();
        Image::push_native_sample_bytes(&mut out, f32::MAX, 24, false);
        assert_eq!(out, [0xff, 0xff, 0xff]);
    }

    #[test]
    fn native_sample_packing_writes_high_bit_signed_little_endian_bytes() {
        let mut out = Vec::new();
        Image::push_native_sample_bytes(&mut out, -1.0, 38, true);
        assert_eq!(out, [0xff, 0xff, 0xff, 0xff, 0xff]);

        out.clear();
        Image::push_native_sample_bytes(&mut out, -((1_i64 << 37) as f32), 38, true);
        assert_eq!(out, [0x00, 0x00, 0x00, 0x00, 0xe0]);
    }

    fn rewrite_siz_to_single_large_tile(codestream: &mut [u8], dimensions: u32) {
        let siz = codestream
            .windows(2)
            .position(|w| w == [0xFF, 0x51])
            .expect("SIZ marker");
        codestream[siz + 6..siz + 10].copy_from_slice(&dimensions.to_be_bytes());
        codestream[siz + 10..siz + 14].copy_from_slice(&dimensions.to_be_bytes());
        codestream[siz + 22..siz + 26].copy_from_slice(&dimensions.to_be_bytes());
        codestream[siz + 26..siz + 30].copy_from_slice(&dimensions.to_be_bytes());
    }

    fn rewrite_siz_tile_grid(codestream: &mut [u8], dimensions: (u32, u32), tile_size: (u32, u32)) {
        let siz = codestream
            .windows(2)
            .position(|w| w == [0xFF, 0x51])
            .expect("SIZ marker");
        codestream[siz + 6..siz + 10].copy_from_slice(&dimensions.0.to_be_bytes());
        codestream[siz + 10..siz + 14].copy_from_slice(&dimensions.1.to_be_bytes());
        codestream[siz + 22..siz + 26].copy_from_slice(&tile_size.0.to_be_bytes());
        codestream[siz + 26..siz + 30].copy_from_slice(&tile_size.1.to_be_bytes());
    }

    fn rewrite_siz_component_count(codestream: &mut Vec<u8>, component_count: u16) {
        let siz = codestream
            .windows(2)
            .position(|w| w == [0xFF, 0x51])
            .expect("SIZ marker");
        let old_component_count =
            u16::from_be_bytes([codestream[siz + 38], codestream[siz + 39]]) as usize;
        let component_start = siz + 40;
        let component_end = component_start + old_component_count * 3;
        let descriptor = codestream[component_start..component_start + 3].to_vec();
        let mut descriptors = Vec::with_capacity(usize::from(component_count) * 3);
        for _ in 0..component_count {
            descriptors.extend_from_slice(&descriptor);
        }

        let siz_len = 38_u16
            .checked_add(
                component_count
                    .checked_mul(3)
                    .expect("SIZ component bytes fit"),
            )
            .expect("SIZ length fits");
        codestream[siz + 2..siz + 4].copy_from_slice(&siz_len.to_be_bytes());
        codestream[siz + 38..siz + 40].copy_from_slice(&component_count.to_be_bytes());
        codestream.splice(component_start..component_end, descriptors);
    }

    #[test]
    fn inspect_rejects_component_count_above_j2k_spec_cap() {
        let mut bytes = fixture_gray();
        rewrite_siz_component_count(&mut bytes, MAX_J2K_SPEC_COMPONENTS + 1);

        let err = inspect_j2k_codestream_header(&bytes)
            .expect_err("SIZ component count above spec cap must be rejected");

        assert_eq!(
            err,
            J2kCodestreamHeaderError::InvalidSiz {
                what: "component count exceeds JPEG 2000 limit"
            }
        );
    }

    #[test]
    fn native_parse_accepts_spec_component_count_above_u8() {
        let mut bytes = fixture_gray();
        rewrite_siz_component_count(&mut bytes, MAX_J2K_SPEC_COMPONENTS + 1);

        let err = match Image::new(&bytes, &DecodeSettings::default()) {
            Err(err) => err,
            Ok(_) => panic!("component count above the JPEG 2000 spec cap must still reject"),
        };
        assert_eq!(
            err,
            DecodeError::Validation(ValidationError::TooManyChannels)
        );

        let mut bytes = fixture_gray();
        rewrite_siz_component_count(&mut bytes, 256);
        Image::new(&bytes, &DecodeSettings::default())
            .expect("component count above u8 should parse within the JPEG 2000 spec cap");
    }

    #[test]
    fn tile_parse_rejects_component_tile_structural_bomb_before_allocation() {
        let mut bytes = fixture_gray();
        rewrite_siz_component_count(&mut bytes, MAX_J2K_SPEC_COMPONENTS);
        rewrite_siz_tile_grid(&mut bytes, (256, 256), (1, 1));
        let parsed = j2c::parse_raw(&bytes, &DecodeSettings::default()).expect("raw header parses");
        let mut context = j2c::DecoderContext::default();
        let mut ht_decoder: Option<&mut dyn HtCodeBlockDecoder> = None;

        let err = j2c::decode(parsed.data, &parsed.header, &mut context, &mut ht_decoder)
            .expect_err("tile structural budget must reject before tile allocation");

        assert_eq!(err, DecodeError::Validation(ValidationError::ImageTooLarge));
    }

    #[test]
    fn owned_decode_rejects_large_siz_before_allocating_output() {
        let mut bytes = fixture_gray();
        rewrite_siz_to_single_large_tile(&mut bytes, 60_000);
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("large SIZ parses");

        let err = match image.decode() {
            Err(err) => err,
            Ok(_) => panic!("large owned decode must be capped"),
        };

        assert_eq!(err, DecodeError::Validation(ValidationError::ImageTooLarge));
    }

    #[test]
    fn decode_into_rejects_large_siz_before_allocating_component_storage() {
        let mut bytes = fixture_gray();
        rewrite_siz_to_single_large_tile(&mut bytes, 60_000);
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("large SIZ parses");
        let mut context = DecoderContext::default();
        let mut out = [];

        let err = match image.decode_into(&mut out, &mut context) {
            Err(err) => err,
            Ok(_) => panic!("component storage must be capped before allocation"),
        };

        assert_eq!(err, DecodeError::Validation(ValidationError::ImageTooLarge));
    }

    fn fixture_ht_gray() -> Vec<u8> {
        let pixels: Vec<u8> = (0..16).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        encode_htj2k(&pixels, 4, 4, 1, 8, false, &options).expect("encode ht gray8")
    }

    fn fixture_ht_multi_block() -> Vec<u8> {
        let pixels: Vec<u8> = (0..64).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 0,
            code_block_width_exp: 0,
            code_block_height_exp: 0,
            ..EncodeOptions::default()
        };
        encode_htj2k(&pixels, 8, 8, 1, 8, false, &options).expect("encode multi-block HT gray8")
    }

    fn fixture_ht_rgb_multi_block() -> Vec<u8> {
        let pixels = gradient_pixels(8, 8, 3);
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 0,
            code_block_width_exp: 0,
            code_block_height_exp: 0,
            ..EncodeOptions::default()
        };
        encode_htj2k(&pixels, 8, 8, 3, 8, false, &options).expect("encode multi-block HT RGB8")
    }

    fn direct_ht_job_count(plan: &J2kDirectGrayscalePlan) -> usize {
        plan.steps
            .iter()
            .map(|step| match step {
                J2kDirectGrayscaleStep::HtSubBand(sub_band) => sub_band.jobs.len(),
                _ => 0,
            })
            .sum()
    }

    fn direct_color_ht_job_count(plan: &J2kDirectColorPlan) -> usize {
        plan.component_plans.iter().map(direct_ht_job_count).sum()
    }

    fn fixture_openhtj2k_ht_refinement() -> &'static [u8] {
        include_bytes!("../fixtures/htj2k/openhtj2k_ds0_ht_12_b11.j2k")
    }

    fn fixture_openhtj2k_ht_refinement_pixels() -> &'static [u8] {
        include_bytes!("../fixtures/htj2k/openhtj2k_ds0_ht_12_b11.gray")
    }

    fn fixture_openhtj2k_ht_refinement_odd() -> &'static [u8] {
        include_bytes!("../fixtures/htj2k/openhtj2k_ds0_ht_09_b11.j2k")
    }

    fn fixture_openhtj2k_ht_refinement_odd_pixels() -> &'static [u8] {
        include_bytes!("../fixtures/htj2k/openhtj2k_ds0_ht_09_b11.gray")
    }

    fn gradient_pixels(width: u32, height: u32, components: u8) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * components as usize);
        for y in 0..height {
            for x in 0..width {
                for component in 0..components {
                    pixels.push(((x * 3 + y * 5 + u32::from(component) * 41) & 0xff) as u8);
                }
            }
        }
        pixels
    }

    fn roi_fixture(classic: bool, components: u8) -> Vec<u8> {
        let width = 64;
        let height = 64;
        let pixels = gradient_pixels(width, height, components);
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            code_block_width_exp: 0,
            code_block_height_exp: 0,
            ..EncodeOptions::default()
        };
        if classic {
            encode(
                &pixels,
                width,
                height,
                components.into(),
                8,
                false,
                &options,
            )
            .expect("encode ROI classic fixture")
        } else {
            encode_htj2k(
                &pixels,
                width,
                height,
                components.into(),
                8,
                false,
                &options,
            )
            .expect("encode ROI HT fixture")
        }
    }

    fn crop_interleaved(
        full: &[u8],
        full_width: u32,
        channels: usize,
        roi: (u32, u32, u32, u32),
    ) -> Vec<u8> {
        let (x, y, width, height) = roi;
        let mut out = Vec::with_capacity(width as usize * height as usize * channels);
        let row_bytes = full_width as usize * channels;
        let roi_row_bytes = width as usize * channels;
        for row in y as usize..(y + height) as usize {
            let start = row * row_bytes + x as usize * channels;
            out.extend_from_slice(&full[start..start + roi_row_bytes]);
        }
        out
    }

    fn count_decode_work(bytes: &[u8], roi: Option<(u32, u32, u32, u32)>) -> DecodeWorkCounter {
        let image = Image::new(bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        match roi {
            Some(roi) => {
                image
                    .decode_region_with_context(roi, &mut context)
                    .expect("region decode with counter");
            }
            None => {
                image
                    .decode_with_context(&mut context)
                    .expect("full decode with counter");
            }
        }
        let counters = context.tile_decode_context.debug_counters;
        DecodeWorkCounter {
            classic_code_blocks: counters.decoded_code_blocks,
            ht_code_blocks: 0,
            idwt_output_samples: counters.idwt_output_samples,
        }
    }

    #[test]
    fn roi_decode_matches_full_crop_for_classic_and_htj2k_gray_and_rgb() {
        let cases = [
            (true, 1_u8, true, false),
            (true, 3_u8, false, false),
            (false, 1_u8, true, false),
            (false, 3_u8, false, false),
        ];
        let rois = [
            (20, 18, 17, 19),
            (0, 0, 9, 11),
            (63, 63, 1, 1),
            (7, 5, 13, 9),
            (0, 0, 64, 64),
        ];

        for (classic, components, expect_gray, has_alpha) in cases {
            let bytes = roi_fixture(classic, components);
            let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
            let full = image.decode().expect("full decode");
            let channels = components as usize;
            for roi in rois {
                let region = image.decode_region(roi).expect("region decode");
                assert_eq!(matches!(region.color_space, ColorSpace::Gray), expect_gray);
                assert_eq!(region.has_alpha, has_alpha);
                assert_eq!(
                    region.data,
                    crop_interleaved(&full, 64, channels, roi),
                    "classic={classic} components={components} roi={roi:?}"
                );
            }
        }
    }

    #[test]
    fn roi_decode_prunes_code_blocks_and_idwt_work_for_classic_and_htj2k() {
        let roi = (48, 48, 16, 16);
        for classic in [true, false] {
            let bytes = {
                let pixels = gradient_pixels(128, 128, 1);
                let options = EncodeOptions {
                    reversible: true,
                    num_decomposition_levels: 3,
                    code_block_width_exp: 0,
                    code_block_height_exp: 0,
                    ..EncodeOptions::default()
                };
                if classic {
                    encode(&pixels, 128, 128, 1, 8, false, &options)
                        .expect("encode classic work fixture")
                } else {
                    encode_htj2k(&pixels, 128, 128, 1, 8, false, &options)
                        .expect("encode ht work fixture")
                }
            };
            let full = count_decode_work(&bytes, None);
            let region = count_decode_work(&bytes, Some(roi));

            assert!(
                region.code_blocks() > 0 && region.code_blocks() < full.code_blocks(),
                "ROI should decode fewer code-blocks for classic={classic}; full={}, region={}",
                full.code_blocks(),
                region.code_blocks()
            );
            assert!(
                region.idwt_output_samples > 0
                    && region.idwt_output_samples < full.idwt_output_samples,
                "ROI should produce fewer IDWT output samples for classic={classic}; full={}, region={}",
                full.idwt_output_samples,
                region.idwt_output_samples
            );
        }
    }

    #[test]
    fn region_decode_reuses_region_sized_component_storage() {
        let bytes = fixture();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();

        let bitmap = image
            .decode_region_with_context((1, 0, 1, 2), &mut context)
            .expect("region decode");

        assert_eq!((bitmap.width, bitmap.height), (1, 2));
        assert!(context
            .tile_decode_context
            .channel_data
            .iter()
            .all(|component| component.container.truncated().len() == 2));
    }

    #[test]
    fn native_region_decode_reuses_region_sized_component_storage() {
        let bytes = fixture();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();

        let bitmap = image
            .decode_native_region_with_context((1, 0, 1, 2), &mut context)
            .expect("native region decode");

        assert_eq!((bitmap.width, bitmap.height), (1, 2));
        assert!(context
            .tile_decode_context
            .channel_data
            .iter()
            .all(|component| component.container.truncated().len() == 2));
    }

    #[test]
    fn decoder_context_defaults_to_auto_cpu_parallelism() {
        let context = DecoderContext::default();

        assert_eq!(context.cpu_decode_parallelism(), CpuDecodeParallelism::Auto);
    }

    #[test]
    fn classic_j2k_auto_and_serial_cpu_parallelism_match_pixels() {
        let bytes = fixture_multi_block();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut auto_context = DecoderContext::default();
        let mut serial_context = DecoderContext::default();
        serial_context.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);

        let auto = image
            .decode_with_context(&mut auto_context)
            .expect("auto decode");
        let serial = image
            .decode_with_context(&mut serial_context)
            .expect("serial decode");

        assert_eq!(auto.data, serial.data);
    }

    #[test]
    fn htj2k_97_auto_and_serial_cpu_parallelism_match_pixels() {
        let width = 128_u32;
        let height = 128_u32;
        let pixels = (0..width * height)
            .map(|idx| ((idx * 17 + idx / width * 31) & 0xff) as u8)
            .collect::<Vec<_>>();
        let bytes = encode_htj2k(
            &pixels,
            width,
            height,
            1,
            8,
            false,
            &EncodeOptions {
                reversible: false,
                guard_bits: 2,
                num_decomposition_levels: 5,
                ..EncodeOptions::default()
            },
        )
        .expect("encode HTJ2K 9/7");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut auto_context = DecoderContext::default();
        let mut serial_context = DecoderContext::default();
        serial_context.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);

        let auto = image
            .decode_with_context(&mut auto_context)
            .expect("auto decode");
        let serial = image
            .decode_with_context(&mut serial_context)
            .expect("serial decode");

        assert_eq!(auto.data, serial.data);
    }

    #[test]
    fn serial_cpu_parallelism_disables_classic_sub_band_parallel_branch() {
        assert!(!j2c::should_decode_classic_sub_band_in_parallel(
            CpuDecodeParallelism::Serial,
            16
        ));
    }

    #[test]
    fn grayscale_direct_plan_is_built_without_materializing_channel_data() {
        let bytes = fixture_gray();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();

        let plan = image
            .build_direct_grayscale_plan_with_context(&mut context)
            .expect("build direct plan");

        assert_eq!(plan.dimensions, (4, 4));
        assert_eq!(plan.bit_depth, 8);
        assert!(
            !plan.steps.is_empty(),
            "direct plan must contain executable steps"
        );
        assert!(
            plan.steps.iter().any(|step| matches!(
                step,
                J2kDirectGrayscaleStep::ClassicSubBand(plan) if !plan.jobs.is_empty()
            )),
            "classic J2K direct plan must contain at least one non-empty classic sub-band job"
        );
        assert!(
            context.tile_decode_context.channel_data.is_empty(),
            "building a direct plan must not materialize host component planes"
        );
    }

    #[test]
    fn grayscale_direct_plan_honors_target_resolution() {
        let bytes = fixture_ht_gray();
        let image = Image::new(
            &bytes,
            &DecodeSettings {
                target_resolution: Some((2, 2)),
                ..DecodeSettings::default()
            },
        )
        .expect("scaled image");
        let mut context = DecoderContext::default();

        let plan = image
            .build_direct_grayscale_plan_with_context(&mut context)
            .expect("build scaled direct plan");

        assert_eq!(plan.dimensions, (2, 2));
        assert!(plan.steps.iter().any(|step| matches!(
            step,
            J2kDirectGrayscaleStep::HtSubBand(plan) if !plan.jobs.is_empty()
        )));
        assert!(plan.steps.iter().any(|step| matches!(
            step,
            J2kDirectGrayscaleStep::Store(store)
                if store.output_width == 2 && store.output_height == 2
        )));
        assert!(
            context.tile_decode_context.channel_data.is_empty(),
            "building a scaled direct plan must not materialize host component planes"
        );
    }

    #[test]
    fn odd_dimensions_honor_covering_target_resolution() {
        let pixels = gradient_pixels(9, 7, 1);
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        };
        let bytes =
            encode_htj2k(&pixels, 9, 7, 1, 8, false, &options).expect("encode odd HT gray8");

        for (target, expected) in [((5, 4), (5, 4)), ((3, 2), (3, 2))] {
            let image = Image::new(
                &bytes,
                &DecodeSettings {
                    target_resolution: Some(target),
                    ..DecodeSettings::default()
                },
            )
            .expect("scaled odd image");

            assert_eq!(
                (image.width(), image.height()),
                expected,
                "target {target:?}"
            );
        }
    }

    #[test]
    fn grayscale_direct_plan_region_prunes_unneeded_ht_code_blocks() {
        let bytes = fixture_ht_multi_block();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut full_context = DecoderContext::default();
        let mut roi_context = DecoderContext::default();

        let full = image
            .build_direct_grayscale_plan_with_context(&mut full_context)
            .expect("build full direct plan");
        let roi = image
            .build_direct_grayscale_plan_region_with_context(&mut roi_context, (0, 0, 2, 2))
            .expect("build ROI direct plan");

        let full_jobs = direct_ht_job_count(&full);
        let roi_jobs = direct_ht_job_count(&roi);
        assert!(full_jobs > 1, "fixture must expose multiple HT jobs");
        assert!(
            roi_jobs < full_jobs,
            "ROI direct plan must prune HT jobs before device preparation"
        );
    }

    #[test]
    fn color_direct_plan_region_prunes_unneeded_ht_code_blocks() {
        let bytes = fixture_ht_rgb_multi_block();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut full_context = DecoderContext::default();
        let mut roi_context = DecoderContext::default();

        let full = image
            .build_direct_color_plan_with_context(&mut full_context)
            .expect("build full RGB direct plan");
        let roi = image
            .build_direct_color_plan_region_with_context(&mut roi_context, (0, 0, 2, 2))
            .expect("build ROI RGB direct plan");

        let full_jobs = direct_color_ht_job_count(&full);
        let roi_jobs = direct_color_ht_job_count(&roi);
        assert!(full_jobs > 3, "fixture must expose multiple RGB HT jobs");
        assert!(
            roi_jobs < full_jobs,
            "RGB ROI direct plan must prune HT jobs before device preparation"
        );
    }

    #[test]
    fn color_direct_plan_honors_target_resolution() {
        for (name, bytes) in [
            ("classic", {
                let pixels = gradient_pixels(8, 8, 3);
                let options = EncodeOptions {
                    reversible: true,
                    num_decomposition_levels: 2,
                    ..EncodeOptions::default()
                };
                encode(&pixels, 8, 8, 3, 8, false, &options).expect("encode classic rgb8")
            }),
            ("htj2k", {
                let pixels = gradient_pixels(8, 8, 3);
                let options = EncodeOptions {
                    reversible: true,
                    num_decomposition_levels: 2,
                    ..EncodeOptions::default()
                };
                encode_htj2k(&pixels, 8, 8, 3, 8, false, &options).expect("encode ht rgb8")
            }),
        ] {
            let image = Image::new(
                &bytes,
                &DecodeSettings {
                    target_resolution: Some((4, 4)),
                    ..DecodeSettings::default()
                },
            )
            .expect("scaled RGB image");
            let mut context = DecoderContext::default();

            let plan = image
                .build_direct_color_plan_with_context(&mut context)
                .expect("build scaled direct color plan");

            assert_eq!(plan.dimensions, (4, 4), "{name}: output dimensions");
            assert_eq!(plan.component_plans.len(), 3, "{name}: component count");
            for component_plan in &plan.component_plans {
                assert_eq!(component_plan.dimensions, (4, 4), "{name}: component dims");
                assert!(component_plan.steps.iter().any(|step| matches!(
                    step,
                    J2kDirectGrayscaleStep::Store(store)
                        if store.output_width == 4 && store.output_height == 4
                )));
            }
            assert!(
                context.tile_decode_context.channel_data.is_empty(),
                "{name}: building a scaled color direct plan must not materialize host component planes"
            );
        }
    }

    #[test]
    fn direct_color_cpu_rgb8_executor_matches_scaled_region_decode() {
        for (name, bytes) in [
            ("classic", {
                let pixels = gradient_pixels(16, 16, 3);
                let options = EncodeOptions {
                    reversible: true,
                    num_decomposition_levels: 2,
                    ..EncodeOptions::default()
                };
                encode(&pixels, 16, 16, 3, 8, false, &options).expect("encode classic rgb8")
            }),
            ("htj2k", {
                let pixels = gradient_pixels(16, 16, 3);
                let options = EncodeOptions {
                    reversible: true,
                    num_decomposition_levels: 2,
                    ..EncodeOptions::default()
                };
                encode_htj2k(&pixels, 16, 16, 3, 8, false, &options).expect("encode ht rgb8")
            }),
        ] {
            let image = Image::new(
                &bytes,
                &DecodeSettings {
                    target_resolution: Some((4, 4)),
                    ..DecodeSettings::default()
                },
            )
            .expect("scaled RGB image");
            let mut expected_context = DecoderContext::default();
            let expected_full = image
                .decode_with_context(&mut expected_context)
                .expect("decode scaled reference");
            let output_region = J2kRect {
                x0: 1,
                y0: 1,
                x1: 3,
                y1: 3,
            };
            let mut direct_context = DecoderContext::default();
            let plan = image
                .build_direct_color_plan_region_with_context(
                    &mut direct_context,
                    (
                        output_region.x0,
                        output_region.y0,
                        output_region.width(),
                        output_region.height(),
                    ),
                )
                .expect("build direct RGB region plan");

            let stride = output_region.width() as usize * 3;
            let mut direct = vec![0_u8; stride * output_region.height() as usize];
            let mut scratch = J2kDirectCpuScratch::new();
            execute_direct_color_plan_rgb8_into(
                &plan,
                output_region,
                &mut scratch,
                &mut direct,
                stride,
            )
            .expect("execute direct RGB plan");

            let mut expected = Vec::with_capacity(direct.len());
            let full_stride = image.width() as usize * 3;
            for y in output_region.y0..output_region.y1 {
                let start = y as usize * full_stride + output_region.x0 as usize * 3;
                expected.extend_from_slice(&expected_full.data[start..start + stride]);
            }

            assert_eq!(direct, expected, "{name}: direct RGB output");

            let rgba_stride = output_region.width() as usize * 4;
            let mut direct_rgba = vec![0_u8; rgba_stride * output_region.height() as usize];
            execute_direct_color_plan_rgba8_into(
                &plan,
                output_region,
                &mut scratch,
                &mut direct_rgba,
                rgba_stride,
            )
            .expect("execute direct RGBA plan");

            let mut expected_rgba = Vec::with_capacity(direct_rgba.len());
            for rgb in expected.chunks_exact(3) {
                expected_rgba.extend_from_slice(rgb);
                expected_rgba.push(255);
            }
            assert_eq!(direct_rgba, expected_rgba, "{name}: direct RGBA output");
        }
    }

    #[test]
    fn htj2k_grayscale_direct_plan_contains_ht_sub_band_steps() {
        let bytes = fixture_ht_gray();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();

        let plan = image
            .build_direct_grayscale_plan_with_context(&mut context)
            .expect("build direct plan");

        assert!(
            plan.steps.iter().any(|step| matches!(
                step,
                J2kDirectGrayscaleStep::HtSubBand(plan) if !plan.jobs.is_empty()
            )),
            "HTJ2K direct plan must contain at least one non-empty HT sub-band decode step"
        );
    }

    #[test]
    fn ht_decoder_hook_is_used_for_htj2k_codeblocks() {
        let pixels: Vec<u8> = (0..16).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let bytes = encode_htj2k(&pixels, 4, 4, 1, 8, false, &options).expect("encode ht");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut hooked_context = DecoderContext::default();
        let mut hook = FailingHtDecoder { called: false };
        let error = match image.decode_components_with_ht_decoder(&mut hooked_context, &mut hook) {
            Ok(_) => panic!("hooked decode must use external HT decoder"),
            Err(error) => error,
        };

        assert!(hook.called, "HT decoder hook must be invoked");
        assert_eq!(
            error,
            DecodeError::Decoding(DecodingError::CodeBlockDecodeFailure)
        );
    }

    #[test]
    fn openhtj2k_conformance_fixture_exercises_refinement_passes() {
        for fixture in [
            (
                "ds0_ht_12_b11",
                fixture_openhtj2k_ht_refinement(),
                fixture_openhtj2k_ht_refinement_pixels(),
                (3, 5),
                8,
                2,
                4,
            ),
            (
                "ds0_ht_09_b11",
                fixture_openhtj2k_ht_refinement_odd(),
                fixture_openhtj2k_ht_refinement_odd_pixels(),
                (17, 37),
                14,
                14,
                629,
            ),
        ] {
            let (
                name,
                codestream,
                expected_pixels,
                dimensions,
                blocks,
                refinement_jobs,
                zero_diffs,
            ) = fixture;
            let image = Image::new(codestream, &DecodeSettings::default()).expect("image");
            let mut context = DecoderContext::default();
            let mut hook = CapturingHtDecoder::default();

            let components = image
                .decode_components_with_ht_decoder(&mut context, &mut hook)
                .expect("decode OpenHTJ2K HTJ2K fixture");

            assert!(
                hook.called,
                "{name}: HTJ2K fixture must use HT code-block decode"
            );
            assert!(
                hook.refinement_jobs > 0,
                "{name}: OpenHTJ2K fixture must contain non-empty refinement segments"
            );
            assert!(
                hook.max_coding_passes > 1,
                "{name}: OpenHTJ2K fixture must exercise more than the cleanup pass"
            );
            assert_eq!(hook.blocks, blocks, "{name}: HT code-block count");
            assert_eq!(
                hook.refinement_jobs, refinement_jobs,
                "{name}: refinement job count"
            );
            assert_eq!(hook.max_coding_passes, 3, "{name}: max HT coding passes");
            assert_eq!(components.dimensions(), dimensions, "{name}: dimensions");
            assert_eq!(components.planes().len(), 1, "{name}: component planes");

            let decoded: Vec<u8> = components.planes()[0]
                .samples()
                .iter()
                .map(|sample| sample.round().clamp(0.0, 255.0) as u8)
                .collect();
            assert_eq!(decoded, expected_pixels, "{name}: decoded pixels");

            let mut zero_context = DecoderContext::default();
            let mut zero_hook = ZeroRefinementHtDecoder;
            let zeroed_components = image
                .decode_components_with_ht_decoder(&mut zero_context, &mut zero_hook)
                .expect("decode OpenHTJ2K fixture with zeroed refinement bytes");
            let actual_zero_diffs = components.planes()[0]
                .samples()
                .iter()
                .zip(zeroed_components.planes()[0].samples())
                .filter(|(actual, zeroed)| (*actual - *zeroed).abs() > f32::EPSILON)
                .count();
            assert_eq!(
                actual_zero_diffs, zero_diffs,
                "{name}: zeroing refinement bytes must change decoded samples"
            );
        }
    }

    #[test]
    fn openhtj2k_refinement_phase_limited_decode_differs_and_records_ht_stats() {
        let image = Image::new(
            fixture_openhtj2k_ht_refinement_odd(),
            &DecodeSettings::default(),
        )
        .expect("image");
        let mut full_context = DecoderContext::default();

        let (full_samples, full_decoded) = {
            let full_components = image
                .decode_components_with_context(&mut full_context)
                .expect("full native decode of OpenHTJ2K refinement fixture");
            let full_samples = full_components.planes()[0].samples().to_vec();
            let full_decoded: Vec<u8> = full_samples
                .iter()
                .map(|sample| sample.round().clamp(0.0, 255.0) as u8)
                .collect();
            (full_samples, full_decoded)
        };
        assert_eq!(
            full_decoded,
            fixture_openhtj2k_ht_refinement_odd_pixels(),
            "full decode must match the checked-in OpenHTJ2K oracle"
        );

        let stats = full_context
            .tile_decode_context
            .debug_counters
            .ht_phase_stats;
        assert_eq!(stats.blocks, 14, "HT block count");
        assert_eq!(stats.refinement_blocks, 14, "HT refinement block count");
        assert!(stats.cleanup_bytes > 0, "cleanup byte total");
        assert!(stats.refinement_bytes > 0, "refinement byte total");

        let mut cleanup_context = DecoderContext::default();
        let mut cleanup_hook = CleanupLimitedHtDecoder::default();
        let cleanup_components = image
            .decode_components_with_ht_decoder(&mut cleanup_context, &mut cleanup_hook)
            .expect("cleanup-limited decode of OpenHTJ2K refinement fixture");
        let cleanup_decoded: Vec<u8> = cleanup_components.planes()[0]
            .samples()
            .iter()
            .map(|sample| sample.round().clamp(0.0, 255.0) as u8)
            .collect();
        let cleanup_sample_diffs = full_samples
            .iter()
            .zip(cleanup_components.planes()[0].samples())
            .filter(|(full, cleanup)| (*full - *cleanup).abs() > f32::EPSILON)
            .count();

        assert!(
            cleanup_sample_diffs > 0,
            "cleanup-limited decode must omit refinement effects"
        );
        assert_eq!(
            cleanup_decoded, full_decoded,
            "fixture refinement differences are below final u8 clamping"
        );
        assert_eq!(cleanup_hook.blocks, 14, "hook HT block count");
        assert_eq!(
            cleanup_hook.refinement_blocks, 14,
            "hook HT refinement block count"
        );
        assert!(cleanup_hook.cleanup_bytes > 0, "hook cleanup byte total");
        assert!(
            cleanup_hook.refinement_bytes > 0,
            "hook refinement byte total"
        );
    }

    #[test]
    fn scalar_htj2k_encoder_contract_is_cleanup_only() {
        let coefficients = (0..64)
            .map(|index| {
                let magnitude = (index % 7) + 1;
                if index % 2 == 0 {
                    magnitude
                } else {
                    -magnitude
                }
            })
            .collect::<Vec<_>>();

        let encoded =
            encode_ht_code_block_scalar(&coefficients, 8, 8, 8).expect("encode HT code block");

        assert_eq!(
            encoded.num_coding_passes, 1,
            "current scalar HTJ2K encoder emits only the cleanup pass"
        );
        assert_eq!(
            encoded.num_zero_bitplanes, 7,
            "current cleanup-only HTJ2K encoder includes one bitplane"
        );
        assert!(
            !encoded.data.is_empty(),
            "non-zero cleanup-only block must still produce payload bytes"
        );
    }

    #[test]
    fn scalar_htj2k_decode_workspace_matches_fresh_decode_and_reuses_capacity() {
        let image = Image::new(
            fixture_openhtj2k_ht_refinement_odd(),
            &DecodeSettings::default(),
        )
        .expect("image");
        let mut context = DecoderContext::default();
        let mut hook = FirstHtJobDecoder::default();
        image
            .decode_components_with_ht_decoder(&mut context, &mut hook)
            .expect("decode fixture while collecting HT jobs");
        let job = hook
            .job
            .as_ref()
            .expect("fixture must expose an HT decode job")
            .borrowed();
        let mut fresh = vec![0.0_f32; job.width as usize * job.height as usize];
        let mut reused = vec![0.0_f32; fresh.len()];
        let mut profiled = vec![0.0_f32; fresh.len()];
        let mut workspace = HtCodeBlockDecodeWorkspace::default();
        let mut profile = HtCodeBlockDecodeProfile::default();

        decode_ht_code_block_scalar(job, &mut fresh).expect("fresh HT decode");
        decode_ht_code_block_scalar_with_workspace(job, &mut reused, &mut workspace)
            .expect("workspace HT decode");
        let first_capacity = workspace.coefficient_capacity();
        decode_ht_code_block_scalar_with_workspace(job, &mut reused, &mut workspace)
            .expect("second workspace HT decode");
        decode_ht_code_block_scalar_with_workspace_profiled(
            job,
            &mut profiled,
            &mut workspace,
            &mut profile,
        )
        .expect("profiled workspace HT decode");

        assert_eq!(reused, fresh);
        assert_eq!(profiled, fresh);
        assert!(first_capacity >= fresh.len());
        assert_eq!(workspace.coefficient_capacity(), first_capacity);
        assert_eq!(profile.blocks, 1);
        assert!(profile.cleanup_bytes > 0);
    }

    #[test]
    fn classic_decoder_hook_is_used_for_j2k_codeblocks() {
        let bytes = fixture();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut hooked_context = DecoderContext::default();
        let mut hook = FailingClassicDecoder { called: false };
        let error = match image.decode_components_with_ht_decoder(&mut hooked_context, &mut hook) {
            Ok(_) => panic!("hooked decode must use external classic decoder"),
            Err(error) => error,
        };

        assert!(hook.called, "classic decoder hook must be invoked");
        assert_eq!(
            error,
            DecodeError::Decoding(DecodingError::CodeBlockDecodeFailure)
        );
    }

    #[test]
    fn classic_sub_band_decoder_hook_is_used_for_j2k_codeblocks() {
        let bytes = fixture_multi_block();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut hooked_context = DecoderContext::default();
        let mut hook = FailingClassicBatchDecoder { called: false };
        let error = match image.decode_components_with_ht_decoder(&mut hooked_context, &mut hook) {
            Ok(_) => panic!("hooked decode must use external classic batch decoder"),
            Err(error) => error,
        };

        assert!(hook.called, "classic sub-band decoder hook must be invoked");
        assert_eq!(
            error,
            DecodeError::Decoding(DecodingError::CodeBlockDecodeFailure)
        );
    }

    // -----------------------------------------------------------------------
    // Sanity tests for the four scalar-reference exports
    // -----------------------------------------------------------------------

    #[test]
    fn forward_dwt53_reference_matches_internal_path() {
        // 4×4 constant-ramp input; 1 decomposition level.
        let samples: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let out = forward_dwt53_reference(&samples, 4, 4, 1);

        // Internal path
        let internal = j2c::fdwt::forward_dwt(&samples, 4, 4, 1, true);

        assert_eq!(out.ll, internal.ll, "LL subband mismatch");
        assert_eq!(out.ll_width, internal.ll_width, "LL width mismatch");
        assert_eq!(out.ll_height, internal.ll_height, "LL height mismatch");
        assert_eq!(out.levels.len(), internal.levels.len(), "level count");
        for (pub_lvl, int_lvl) in out.levels.iter().zip(internal.levels.iter()) {
            assert_eq!(pub_lvl.hl, int_lvl.hl, "HL mismatch");
            assert_eq!(pub_lvl.lh, int_lvl.lh, "LH mismatch");
            assert_eq!(pub_lvl.hh, int_lvl.hh, "HH mismatch");
        }
    }

    #[test]
    fn forward_rct_reference_matches_internal_path() {
        // Single pixel: R=100, G=150, B=200
        let planes = vec![vec![100.0f32], vec![150.0f32], vec![200.0f32]];
        let result = forward_rct_reference(planes.clone());

        // Internal path
        let mut internal = planes;
        j2c::forward_mct::forward_rct(&mut internal);

        assert_eq!(result, internal, "RCT output mismatch");
        // Y = floor((100 + 300 + 200) / 4) = 150
        assert_eq!(result[0][0], 150.0, "Y component");
        assert_eq!(result[1][0], 50.0, "Cb component");
        assert_eq!(result[2][0], -50.0, "Cr component");
    }

    #[test]
    fn forward_ict_reference_matches_internal_path() {
        let planes = vec![vec![100.0f32], vec![150.0f32], vec![200.0f32]];
        let result = forward_ict_reference(planes.clone());

        let mut internal = planes;
        j2c::forward_mct::forward_ict(&mut internal);

        assert_eq!(result, internal, "ICT output mismatch");
    }

    #[test]
    fn forward_dwt97_reference_matches_internal_path() {
        let samples = (0..64)
            .map(|idx| {
                f32::from(u8::try_from((idx * 19 + idx / 3) & 0xff).expect("masked sample fits u8"))
                    - 128.0
            })
            .collect::<Vec<_>>();
        let result = forward_dwt97_reference(&samples, 8, 8, 2);
        let internal = j2c::fdwt::forward_dwt(&samples, 8, 8, 2, false);

        assert_eq!(result.ll, internal.ll, "DWT 9/7 LL mismatch");
        assert_eq!(result.ll_width, internal.ll_width);
        assert_eq!(result.ll_height, internal.ll_height);
        assert_eq!(result.levels.len(), internal.levels.len());
        for (actual, expected) in result.levels.iter().zip(internal.levels.iter()) {
            assert_eq!(actual.hl, expected.hl, "DWT 9/7 HL mismatch");
            assert_eq!(actual.lh, expected.lh, "DWT 9/7 LH mismatch");
            assert_eq!(actual.hh, expected.hh, "DWT 9/7 HH mismatch");
        }
    }

    #[test]
    fn quantize_reversible_reference_matches_internal_path() {
        let coefficients = vec![3.7f32, -8.2, 0.5, -0.5, 10.0];
        let exponent = 8u16;
        let mantissa = 0u16;
        let range_bits = 8u8;

        let result =
            quantize_reversible_reference(&coefficients, exponent, mantissa, range_bits, true);

        // Internal path
        let step = j2c::quantize::QuantStepSize { exponent, mantissa };
        let internal = j2c::quantize::quantize_subband(&coefficients, &step, range_bits, true);

        assert_eq!(result, internal, "quantize output mismatch");
        // reversible: round to nearest
        assert_eq!(result[0], 4, "3.7 rounds to 4");
        assert_eq!(result[1], -8, "-8.2 rounds to -8");
    }

    #[test]
    fn quantize_subband_reference_matches_irreversible_internal_path() {
        let coefficients = vec![3.7f32, -8.2, 0.5, -0.5, 10.0];
        let exponent = 8u16;
        let mantissa = 256u16;
        let range_bits = 8u8;

        let result =
            quantize_subband_reference(&coefficients, exponent, mantissa, range_bits, false);

        let step = j2c::quantize::QuantStepSize { exponent, mantissa };
        let internal = j2c::quantize::quantize_subband(&coefficients, &step, range_bits, false);

        assert_eq!(result, internal, "irreversible quantize output mismatch");
    }

    #[test]
    fn deinterleave_reference_matches_internal_path() {
        // 2-pixel RGB8 unsigned: [R0,G0,B0, R1,G1,B1]
        let pixels: Vec<u8> = vec![128, 64, 200, 10, 20, 30];
        let result = deinterleave_reference(&pixels, 2, 3, 8, false);

        let internal = j2c::encode::deinterleave_to_f32(&pixels, 2, 3, 8, false);

        assert_eq!(result, internal, "deinterleave output mismatch");
        assert_eq!(result.len(), 3, "three component planes");
        assert_eq!(result[0].len(), 2, "two pixels per plane");
        // unsigned 8-bit with level shift: val - 128
        assert!((result[0][0] - 0.0f32).abs() < 1e-6, "R0 level-shifted");
        assert!((result[1][0] - (-64.0f32)).abs() < 1e-6, "G0 level-shifted");
        assert!((result[2][0] - 72.0f32).abs() < 1e-6, "B0 level-shifted");
    }
}
