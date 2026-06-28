//! Top-level JPEG 2000 encode orchestration.
//!
//! Coordinates the full encoding pipeline:
//!   pixels → MCT → DWT → quantize → EBCOT T1 → T2 → codestream
//!
//! Supports both lossless (5-3 reversible) and lossy (9-7 irreversible) encoding.

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::ops::Range;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::bitplane_encode;
use super::build::SubBandType;
use super::codestream::CodeBlockStyle;
use super::codestream_write::{self, BlockCodingMode, EncodeComponentSampleInfo, EncodeParams};
use super::fdwt::{self, DwtDecomposition};
use super::forward_mct;
use super::ht_block_encode;
use super::packet_encode::{self, CodeBlockPacketData, ResolutionPacket, SubbandPrecinct};
pub use super::quantize::irreversible_quantization_step_for_subband;
use super::quantize::{self, QuantStepSize};
use crate::math::{floor_f32, log2_f32};
use crate::profile;
use crate::{
    CpuOnlyJ2kEncodeStageAccelerator, EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock,
    IrreversibleQuantizationSubbandScales, J2kDeinterleaveToF32Job, J2kEncodeStageAccelerator,
    J2kForwardDwt53Job, J2kForwardDwt53Level, J2kForwardDwt53Output, J2kForwardDwt97Job,
    J2kForwardDwt97Level, J2kForwardDwt97Output, J2kForwardIctJob, J2kForwardRctJob,
    J2kHtSubbandEncodeJob, J2kHtj2kTileEncodeJob, J2kPacketizationBlockCodingMode,
    J2kPacketizationCodeBlock, J2kPacketizationEncodeJob, J2kPacketizationPacketDescriptor,
    J2kPacketizationResolution, J2kPacketizationSubband, J2kQuantizeSubbandJob, J2kSubBandType,
    J2kTier1CodeBlockEncodeJob, PrecomputedHtj2k53Component, PrecomputedHtj2k53Image,
    PrecomputedHtj2k97Component, PrecomputedHtj2k97Image, PreencodedHtj2k97CodeBlock,
    PreencodedHtj2k97CompactCodeBlock, PreencodedHtj2k97CompactComponent,
    PreencodedHtj2k97CompactImage, PreencodedHtj2k97CompactResolution,
    PreencodedHtj2k97CompactSubband, PreencodedHtj2k97Component, PreencodedHtj2k97Image,
    PreencodedHtj2k97Resolution, PreencodedHtj2k97Subband, PrequantizedHtj2k97Component,
    PrequantizedHtj2k97Image, PrequantizedHtj2k97Resolution, PrequantizedHtj2k97Subband,
    MAX_J2K_SPEC_COMPONENTS,
};
use crate::{DecodeSettings, Image};

const HT_CPU_PARALLEL_FALLBACK_MIN_JOBS: usize = 4;
const MAX_RAW_PIXEL_ENCODE_BIT_DEPTH: u8 = 24;
const MAX_PART1_SAMPLE_BIT_DEPTH: u8 = 38;
const MAX_REVERSIBLE_NO_QUANT_EXPONENT: u16 = 31;
const MAX_REVERSIBLE_NO_QUANT_GUARD_BITS: u8 = 7;
const MAX_CLASSIC_REVERSIBLE_MARKER_BITPLANES: u16 =
    MAX_REVERSIBLE_NO_QUANT_GUARD_BITS as u16 + MAX_REVERSIBLE_NO_QUANT_EXPONENT - 1;
// Classic packet headers can signal at most 164 coding passes, i.e.
// 1 cleanup pass for the first bitplane plus 3 passes for each additional
// bitplane: 1 + 3 * (55 - 1) = 163.
const MAX_CLASSIC_ROI_CODED_BITPLANES: u8 = 55;
const MAX_HT_ROI_CODED_BITPLANES: u8 = 31;

/// Encoding options for JPEG 2000.
#[derive(Debug, Clone)]
pub struct EncodeOptions {
    /// Number of decomposition levels (default: 5).
    pub num_decomposition_levels: u8,
    /// Use reversible (lossless) transform (default: true).
    pub reversible: bool,
    /// Code-block width exponent minus 2 (default: 4, meaning 2^6=64).
    pub code_block_width_exp: u8,
    /// Code-block height exponent minus 2 (default: 4, meaning 2^6=64).
    pub code_block_height_exp: u8,
    /// Number of guard bits (default: 1 for reversible, 2 for irreversible).
    pub guard_bits: u8,
    /// Encode using HT block coding (HTJ2K / Part 15) instead of classic EBCOT.
    pub use_ht_block_coding: bool,
    /// Packet progression order to write in COD and use for packetization.
    pub progression_order: EncodeProgressionOrder,
    /// Write a TLM marker segment for the single tile-part.
    pub write_tlm: bool,
    /// Write PLT packet-length marker segments in the tile-part header.
    pub write_plt: bool,
    /// Write PLM packet-length marker segments in the main header.
    pub write_plm: bool,
    /// Write PPM packed packet-header marker segments in the main header.
    pub write_ppm: bool,
    /// Write PPT packed packet-header marker segments in tile-part headers.
    pub write_ppt: bool,
    /// Write SOP marker segments before packets.
    pub write_sop: bool,
    /// Write EPH markers after packet headers.
    pub write_eph: bool,
    /// Apply the JPEG 2000 multi-component color transform for 3+ component inputs.
    pub use_mct: bool,
    /// Number of cumulative quality layers to emit.
    pub num_layers: u8,
    /// Optional cumulative packet-body byte targets for each quality layer.
    pub quality_layer_byte_targets: Vec<u64>,
    /// Decode and verify HTJ2K codestreams inside the native encoder.
    pub validate_high_throughput_codestream: bool,
    /// Multiplier applied to irreversible 9/7 scalar quantization step sizes.
    ///
    /// `1.0` preserves the near-lossless default step sizes. Larger values
    /// produce smaller codestreams by coarsening quantization.
    pub irreversible_quantization_scale: f32,
    /// Per-subband multipliers applied on top of
    /// `irreversible_quantization_scale`.
    pub irreversible_quantization_subband_scales: IrreversibleQuantizationSubbandScales,
    /// Optional per-component SIZ sampling factors (`XRsiz`, `YRsiz`).
    ///
    /// `None` means every component is stored at the reference-grid
    /// resolution. This is experimental and primarily intended for precomputed
    /// coefficient encoders that preserve JPEG-native chroma subsampling.
    pub component_sampling: Option<Vec<(u8, u8)>>,
    /// Optional per-component whole-component ROI maxshift values.
    ///
    /// Non-zero entries emit RGN markers and encode every coefficient in that
    /// component with the requested maxshift. Rectangular ROI authoring is not
    /// represented by this field.
    pub roi_component_shifts: Vec<u8>,
    /// Optional tile width and height for multi-tile codestream output.
    pub tile_size: Option<(u32, u32)>,
    /// Optional maximum number of complete packets to place in each tile-part.
    pub tile_part_packet_limit: Option<u16>,
    /// Optional precinct exponents in COD order, one per resolution level.
    pub precinct_exponents: Vec<(u8, u8)>,
}

/// Borrowed component-plane samples for reversible 5/3 component-plane encode.
#[derive(Debug, Clone, Copy)]
pub struct EncodeComponentPlane<'a> {
    /// Row-major little-endian component samples at this component's own grid.
    pub data: &'a [u8],
    /// Horizontal SIZ sampling factor (`XRsiz`).
    pub x_rsiz: u8,
    /// Vertical SIZ sampling factor (`YRsiz`).
    pub y_rsiz: u8,
}

/// Borrowed component-plane samples with per-component precision metadata.
#[derive(Debug, Clone, Copy)]
pub struct EncodeTypedComponentPlane<'a> {
    /// Row-major little-endian component samples at this component's own grid.
    pub data: &'a [u8],
    /// Horizontal SIZ sampling factor (`XRsiz`).
    pub x_rsiz: u8,
    /// Vertical SIZ sampling factor (`YRsiz`).
    pub y_rsiz: u8,
    /// Significant bits per sample for this component.
    pub bit_depth: u8,
    /// Whether samples in this component are signed.
    pub signed: bool,
}

/// Rectangular region-of-interest request for JPEG 2000 maxshift encoding.
///
/// The rectangle is expressed in full-resolution reference-grid pixels. For
/// sampled components, the encoder maps the rectangle to that component's SIZ
/// grid before selecting wavelet coefficients. All regions for the same
/// component must use the same non-zero `shift`, because JPEG 2000 RGN stores
/// one maxshift value per component.
#[derive(Debug, Clone, Copy)]
pub struct EncodeRoiRegion {
    /// Component index to which the ROI applies.
    pub component: u16,
    /// Left edge in reference-grid pixels.
    pub x: u32,
    /// Top edge in reference-grid pixels.
    pub y: u32,
    /// Width in reference-grid pixels.
    pub width: u32,
    /// Height in reference-grid pixels.
    pub height: u32,
    /// Maxshift value to write in the component's RGN marker.
    pub shift: u8,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            num_decomposition_levels: 5,
            reversible: true,
            code_block_width_exp: 4,
            code_block_height_exp: 4,
            guard_bits: 1,
            use_ht_block_coding: false,
            progression_order: EncodeProgressionOrder::Lrcp,
            write_tlm: false,
            write_plt: false,
            write_plm: false,
            write_ppm: false,
            write_ppt: false,
            write_sop: false,
            write_eph: false,
            use_mct: true,
            num_layers: 1,
            quality_layer_byte_targets: Vec::new(),
            validate_high_throughput_codestream: true,
            irreversible_quantization_scale: 1.0,
            irreversible_quantization_subband_scales:
                IrreversibleQuantizationSubbandScales::default(),
            component_sampling: None,
            roi_component_shifts: Vec::new(),
            tile_size: None,
            tile_part_packet_limit: None,
            precinct_exponents: Vec::new(),
        }
    }
}

/// JPEG 2000 packet progression orders supported by the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum EncodeProgressionOrder {
    /// Layer-resolution-component-position progression.
    #[default]
    Lrcp,
    /// Resolution-layer-component-position progression.
    Rlcp,
    /// Resolution-position-component-layer progression.
    Rpcl,
    /// Position-component-resolution-layer progression.
    Pcrl,
    /// Component-position-resolution-layer progression.
    Cprl,
}

/// Encode pixel data into a JPEG 2000 codestream.
///
/// # Arguments
/// * `pixels` — Raw pixel data. For 8-bit: one byte per sample. For >8-bit: two bytes per sample (little-endian u16).
/// * `width` — Image width in pixels.
/// * `height` — Image height in pixels.
/// * `num_components` — Number of components (1 for grayscale, 3 for RGB).
/// * `bit_depth` — Bits per sample (e.g., 8, 12, 16).
/// * `signed` — Whether samples are signed.
/// * `options` — Encoding parameters.
///
/// # Returns
/// The encoded JPEG 2000 codestream bytes (`.j2c` format).
pub fn encode(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
) -> Result<Vec<u8>, &'static str> {
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_with_accelerator(
        pixels,
        width,
        height,
        num_components,
        bit_depth,
        signed,
        options,
        &mut accelerator,
    )
}

/// Encode pixel data into a JPEG 2000 codestream using optional encode-stage hooks.
///
/// Stage hooks may accelerate forward RCT, forward 5/3 DWT, Tier-1 code-block
/// encode, and packetization. Returning fallback from a hook preserves the CPU
/// baseline for that stage.
pub fn encode_with_accelerator(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    encode_with_accelerator_and_component_sample_info(
        pixels,
        width,
        height,
        num_components,
        bit_depth,
        signed,
        options,
        &[],
        accelerator,
    )
}

fn encode_with_accelerator_and_component_sample_info(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
    component_sample_info: &[EncodeComponentSampleInfo],
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    let block_coding_mode = block_coding_mode(options);
    let codestream = encode_impl(
        pixels,
        width,
        height,
        num_components,
        bit_depth,
        signed,
        options,
        block_coding_mode,
        &[],
        component_sample_info,
        accelerator,
    )?;

    if block_coding_mode == BlockCodingMode::HighThroughput
        && options.validate_high_throughput_codestream
    {
        validate_htj2k_codestream(
            &codestream,
            pixels,
            width,
            height,
            num_components,
            bit_depth,
            signed,
            options.reversible,
        )?;
    }

    Ok(codestream)
}

/// Encode pixel data into a JPEG 2000 codestream with rectangular ROI maxshift.
///
/// This uses the normal native encoder pipeline. Non-empty `roi_regions`
/// produce RGN markers and shift selected quantized coefficients before
/// code-block encoding.
pub fn encode_with_roi_regions(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
    roi_regions: &[EncodeRoiRegion],
) -> Result<Vec<u8>, &'static str> {
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_with_accelerator_and_roi_regions(
        pixels,
        width,
        height,
        num_components,
        bit_depth,
        signed,
        options,
        roi_regions,
        &mut accelerator,
    )
}

/// Encode pixel data with rectangular ROI maxshift and optional stage hooks.
pub fn encode_with_accelerator_and_roi_regions(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
    roi_regions: &[EncodeRoiRegion],
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    let block_coding_mode = block_coding_mode(options);
    let codestream = encode_impl(
        pixels,
        width,
        height,
        num_components,
        bit_depth,
        signed,
        options,
        block_coding_mode,
        roi_regions,
        &[],
        accelerator,
    )?;

    if block_coding_mode == BlockCodingMode::HighThroughput
        && options.validate_high_throughput_codestream
    {
        validate_htj2k_codestream(
            &codestream,
            pixels,
            width,
            height,
            num_components,
            bit_depth,
            signed,
            options.reversible,
        )?;
    }

    Ok(codestream)
}

/// Encode pixel data into an HTJ2K codestream.
///
/// Lossless HTJ2K output is self-validated before it is returned.
pub fn encode_htj2k(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
) -> Result<Vec<u8>, &'static str> {
    let mut options = options.clone();
    options.use_ht_block_coding = true;
    encode(
        pixels,
        width,
        height,
        num_components,
        bit_depth,
        signed,
        &options,
    )
}

/// Encode reversible 5/3 component planes into a classic J2K or HTJ2K
/// codestream.
///
/// Plane buffers are supplied at each component's own SIZ sampling grid. Set
/// [`EncodeOptions::use_ht_block_coding`] to select HTJ2K block coding; the
/// default writes classic Part 1 block coding.
pub fn encode_component_planes_53(
    planes: &[EncodeComponentPlane<'_>],
    width: u32,
    height: u32,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
) -> Result<Vec<u8>, &'static str> {
    let typed_planes = planes
        .iter()
        .map(|plane| EncodeTypedComponentPlane {
            data: plane.data,
            x_rsiz: plane.x_rsiz,
            y_rsiz: plane.y_rsiz,
            bit_depth,
            signed,
        })
        .collect::<Vec<_>>();
    encode_typed_component_planes_53(&typed_planes, width, height, options)
}

/// Encode reversible 5/3 typed component planes into a classic J2K or HTJ2K
/// codestream.
///
/// This is the component-plane entry point for JPEG 2000 codestreams whose
/// components have different precision or signedness. Plane buffers are
/// supplied at each component's own SIZ sampling grid. Components are encoded
/// without a reversible color transform.
pub fn encode_typed_component_planes_53(
    planes: &[EncodeTypedComponentPlane<'_>],
    width: u32,
    height: u32,
    options: &EncodeOptions,
) -> Result<Vec<u8>, &'static str> {
    if width == 0 || height == 0 {
        return Err("invalid dimensions");
    }
    if planes.is_empty() || planes.len() > usize::from(MAX_J2K_SPEC_COMPONENTS) {
        return Err("unsupported component count");
    }
    if planes
        .iter()
        .any(|plane| plane.x_rsiz == 0 || plane.y_rsiz == 0)
    {
        return Err("component sampling factors must be non-zero");
    }
    if planes.iter().any(|plane| plane.bit_depth == 0) {
        return Err("unsupported bit depth");
    }
    if planes
        .iter()
        .any(|plane| plane.bit_depth > MAX_PART1_SAMPLE_BIT_DEPTH)
    {
        return Err("unsupported bit depth");
    }
    if planes
        .iter()
        .any(|plane| plane.bit_depth > MAX_RAW_PIXEL_ENCODE_BIT_DEPTH)
    {
        return encode_typed_component_planes_53_i64(planes, width, height, options);
    }

    let max_levels = planes
        .iter()
        .map(|plane| {
            let component_width = width.div_ceil(u32::from(plane.x_rsiz));
            let component_height = height.div_ceil(u32::from(plane.y_rsiz));
            max_decomposition_levels(component_width, component_height)
        })
        .min()
        .unwrap_or(0);
    let num_levels = options.num_decomposition_levels.min(max_levels);
    let components = planes
        .iter()
        .map(|plane| {
            let component_width = width.div_ceil(u32::from(plane.x_rsiz));
            let component_height = height.div_ceil(u32::from(plane.y_rsiz));
            let samples = component_plane_to_f32(
                plane.data,
                component_width,
                component_height,
                plane.bit_depth,
                plane.signed,
            )?;
            let dwt = fdwt::forward_dwt(
                &samples,
                component_width,
                component_height,
                num_levels,
                true,
            );
            Ok(PrecomputedHtj2k53Component {
                x_rsiz: plane.x_rsiz,
                y_rsiz: plane.y_rsiz,
                dwt: forward_dwt53_output_from_decomposition(dwt),
            })
        })
        .collect::<Result<Vec<_>, &'static str>>()?;
    let max_bit_depth = planes
        .iter()
        .map(|plane| plane.bit_depth)
        .max()
        .ok_or("unsupported component count")?;
    let component_sample_info = planes
        .iter()
        .map(|plane| EncodeComponentSampleInfo {
            bit_depth: plane.bit_depth,
            signed: plane.signed,
        })
        .collect::<Vec<_>>();
    let image = PrecomputedHtj2k53Image {
        width,
        height,
        bit_depth: max_bit_depth,
        signed: planes.iter().all(|plane| plane.signed),
        components,
    };

    if options.use_ht_block_coding {
        let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
        encode_precomputed_53_with_component_sample_info_and_accelerator(
            &image,
            options,
            false,
            true,
            &component_sample_info,
            &mut accelerator,
        )
    } else {
        let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
        encode_precomputed_53_with_component_sample_info_and_accelerator(
            &image,
            options,
            false,
            false,
            &component_sample_info,
            &mut accelerator,
        )
    }
}

fn encode_typed_component_planes_53_i64(
    planes: &[EncodeTypedComponentPlane<'_>],
    width: u32,
    height: u32,
    options: &EncodeOptions,
) -> Result<Vec<u8>, &'static str> {
    if options.num_layers == 0 || options.num_layers > 32 {
        return Err("unsupported quality layer count");
    }
    if options.write_ppm && options.write_ppt {
        return Err("PPM and PPT packet header markers are mutually exclusive");
    }
    if matches!(options.tile_part_packet_limit, Some(0)) {
        return Err("tile-part packet limit must be non-zero");
    }
    if !options.quality_layer_byte_targets.is_empty()
        && options.quality_layer_byte_targets.len() != usize::from(options.num_layers)
    {
        return Err("quality layer byte target count must match quality layer count");
    }
    if let Some((tile_width, tile_height)) = options.tile_size {
        if tile_width == 0 || tile_height == 0 {
            return Err("invalid tile dimensions");
        }
    }

    let num_components = u16::try_from(planes.len()).map_err(|_| "unsupported component count")?;
    if let Some((tile_width, tile_height)) = options.tile_size {
        if tile_width < width || tile_height < height {
            return encode_typed_component_planes_53_i64_multitile(
                planes,
                width,
                height,
                options,
                tile_width,
                tile_height,
                num_components,
            );
        }
    }
    let max_bit_depth = planes
        .iter()
        .map(|plane| plane.bit_depth)
        .max()
        .ok_or("unsupported component count")?;
    let num_levels = planes
        .iter()
        .map(|plane| {
            let component_width = width.div_ceil(u32::from(plane.x_rsiz));
            let component_height = height.div_ceil(u32::from(plane.y_rsiz));
            max_decomposition_levels(component_width, component_height)
        })
        .min()
        .unwrap_or(0)
        .min(options.num_decomposition_levels);
    let requested_guard_bits = options.guard_bits;
    let guard_bits =
        reversible_guard_bits_for_marker_limit(max_bit_depth, num_levels, requested_guard_bits)?;
    let reversible_guard_delta = guard_bits.saturating_sub(requested_guard_bits);
    let mut step_sizes = quantize::compute_step_sizes_with_irreversible_profile(
        max_bit_depth,
        num_levels,
        true,
        guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
    );
    if reversible_guard_delta != 0 {
        adjust_reversible_step_sizes_for_guard_delta(&mut step_sizes, reversible_guard_delta)?;
    }
    let component_sample_info = planes
        .iter()
        .map(|plane| EncodeComponentSampleInfo {
            bit_depth: plane.bit_depth,
            signed: plane.signed,
        })
        .collect::<Vec<_>>();
    let mut component_step_sizes = component_step_sizes(
        &component_sample_info,
        num_levels,
        true,
        guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
    );
    if reversible_guard_delta != 0 {
        adjust_component_step_sizes_for_guard_delta(
            &mut component_step_sizes,
            reversible_guard_delta,
        )?;
    }
    if step_sizes.iter().any(|step| step.exponent > 31)
        || component_step_sizes
            .iter()
            .flatten()
            .any(|step| step.exponent > 31)
    {
        return Err("25-38 bit typed component-plane encode exceeds the current no-quantization guard/exponent signaling limit");
    }

    let quant_params = step_sizes
        .iter()
        .map(|step| (step.exponent, step.mantissa))
        .collect::<Vec<_>>();
    let component_quantization_step_sizes = component_step_sizes
        .iter()
        .map(|steps| {
            steps
                .iter()
                .map(|step| (step.exponent, step.mantissa))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let cb_width = 1u32 << (options.code_block_width_exp + 2);
    let cb_height = 1u32 << (options.code_block_height_exp + 2);
    let block_coding_mode = if options.use_ht_block_coding {
        BlockCodingMode::HighThroughput
    } else {
        BlockCodingMode::Classic
    };
    let component_sampling = planes
        .iter()
        .map(|plane| (plane.x_rsiz, plane.y_rsiz))
        .collect::<Vec<_>>();
    let mut high_bit_options = options.clone();
    high_bit_options.reversible = true;
    high_bit_options.use_mct = false;
    high_bit_options.component_sampling = Some(component_sampling.clone());
    let precinct_exponents = precinct_exponents_for_options(&high_bit_options, num_levels)?;
    let params = EncodeParams {
        width,
        height,
        tile_width: options
            .tile_size
            .map_or(width, |(tile_width, _)| tile_width),
        tile_height: options
            .tile_size
            .map_or(height, |(_, tile_height)| tile_height),
        num_components,
        bit_depth: max_bit_depth,
        signed: planes.iter().all(|plane| plane.signed),
        component_sample_info,
        component_quantization_step_sizes,
        num_decomposition_levels: num_levels,
        reversible: true,
        code_block_width_exp: options.code_block_width_exp,
        code_block_height_exp: options.code_block_height_exp,
        num_layers: options.num_layers,
        use_mct: false,
        guard_bits,
        block_coding_mode,
        progression_order: options.progression_order,
        write_tlm: options.write_tlm,
        write_plt: options.write_plt,
        write_plm: options.write_plm,
        write_ppm: options.write_ppm,
        write_ppt: options.write_ppt,
        write_sop: options.write_sop,
        write_eph: options.write_eph,
        terminate_coding_passes: block_coding_mode == BlockCodingMode::Classic
            && options.num_layers > 1,
        component_sampling,
        roi_component_shifts: vec![0; usize::from(num_components)],
        precinct_exponents,
    };

    let ht_target_coding_passes = ht_target_coding_passes_for_options(options);
    let mut component_resolution_packets = Vec::with_capacity(planes.len());
    for (component_idx, plane) in planes.iter().enumerate() {
        let component_width = width.div_ceil(u32::from(plane.x_rsiz));
        let component_height = height.div_ceil(u32::from(plane.y_rsiz));
        let samples = typed_component_plane_to_i64(plane, component_width, component_height)?;
        let decomp = fdwt::forward_dwt_i64(&samples, component_width, component_height, num_levels);
        let steps = component_step_sizes
            .get(component_idx)
            .ok_or("component quantization step count mismatch")?;
        let component = u16::try_from(component_idx).map_err(|_| "component index exceeds u16")?;
        let mut packets = Vec::with_capacity(num_levels as usize + 1);

        let ll_subband = prepare_subband_i64(
            &decomp.ll,
            decomp.ll_width,
            decomp.ll_height,
            steps
                .first()
                .ok_or("reversible quantization step missing")?,
            guard_bits,
            cb_width,
            cb_height,
            SubBandType::LowLow,
            0,
            &[],
            1,
            block_coding_mode,
            ht_target_coding_passes,
        )?;
        packets.push(PreparedResolutionPacket {
            component,
            resolution: 0,
            precinct: 0,
            subbands: vec![ll_subband],
        });

        for (level_idx, level) in decomp.levels.iter().enumerate() {
            let step_base = 1 + level_idx * 3;
            let hl_subband = prepare_subband_i64(
                &level.hl,
                level.high_width,
                level.low_height,
                steps
                    .get(step_base)
                    .ok_or("reversible quantization step missing")?,
                guard_bits,
                cb_width,
                cb_height,
                SubBandType::HighLow,
                0,
                &[],
                1,
                block_coding_mode,
                ht_target_coding_passes,
            )?;
            let lh_subband = prepare_subband_i64(
                &level.lh,
                level.low_width,
                level.high_height,
                steps
                    .get(step_base + 1)
                    .ok_or("reversible quantization step missing")?,
                guard_bits,
                cb_width,
                cb_height,
                SubBandType::LowHigh,
                0,
                &[],
                1,
                block_coding_mode,
                ht_target_coding_passes,
            )?;
            let hh_subband = prepare_subband_i64(
                &level.hh,
                level.high_width,
                level.high_height,
                steps
                    .get(step_base + 2)
                    .ok_or("reversible quantization step missing")?,
                guard_bits,
                cb_width,
                cb_height,
                SubBandType::HighHigh,
                0,
                &[],
                1,
                block_coding_mode,
                ht_target_coding_passes,
            )?;
            packets.push(PreparedResolutionPacket {
                component,
                resolution: u32::try_from(level_idx + 1)
                    .map_err(|_| "resolution index exceeds u32")?,
                precinct: 0,
                subbands: vec![hl_subband, lh_subband, hh_subband],
            });
        }
        component_resolution_packets.push(packets);
    }

    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_i64_component_resolution_packets(
        component_resolution_packets,
        width,
        height,
        num_components,
        num_levels,
        &params,
        &quant_params,
        &high_bit_options,
        &mut accelerator,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_typed_component_planes_53_i64_multitile(
    planes: &[EncodeTypedComponentPlane<'_>],
    width: u32,
    height: u32,
    options: &EncodeOptions,
    tile_width: u32,
    tile_height: u32,
    num_components: u16,
) -> Result<Vec<u8>, &'static str> {
    let num_x_tiles = width.div_ceil(tile_width);
    let num_y_tiles = height.div_ceil(tile_height);
    let num_tiles = num_x_tiles
        .checked_mul(num_y_tiles)
        .ok_or("tile count overflow")?;
    if num_tiles > u32::from(u16::MAX) + 1 {
        return Err("multi-tile encode supports at most 65536 tiles");
    }

    let num_levels = min_sampled_tile_component_decomposition_levels(
        planes,
        width,
        height,
        tile_width,
        tile_height,
    )?
    .min(options.num_decomposition_levels);
    let max_bit_depth = planes
        .iter()
        .map(|plane| plane.bit_depth)
        .max()
        .ok_or("unsupported component count")?;
    let requested_guard_bits = options.guard_bits;
    let guard_bits =
        reversible_guard_bits_for_marker_limit(max_bit_depth, num_levels, requested_guard_bits)?;
    let reversible_guard_delta = guard_bits.saturating_sub(requested_guard_bits);
    let mut step_sizes = quantize::compute_step_sizes_with_irreversible_profile(
        max_bit_depth,
        num_levels,
        true,
        guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
    );
    if reversible_guard_delta != 0 {
        adjust_reversible_step_sizes_for_guard_delta(&mut step_sizes, reversible_guard_delta)?;
    }
    let component_sample_info = planes
        .iter()
        .map(|plane| EncodeComponentSampleInfo {
            bit_depth: plane.bit_depth,
            signed: plane.signed,
        })
        .collect::<Vec<_>>();
    let mut component_step_sizes = component_step_sizes(
        &component_sample_info,
        num_levels,
        true,
        guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
    );
    if reversible_guard_delta != 0 {
        adjust_component_step_sizes_for_guard_delta(
            &mut component_step_sizes,
            reversible_guard_delta,
        )?;
    }
    if step_sizes.iter().any(|step| step.exponent > 31)
        || component_step_sizes
            .iter()
            .flatten()
            .any(|step| step.exponent > 31)
    {
        return Err("25-38 bit typed component-plane encode exceeds the current no-quantization guard/exponent signaling limit");
    }

    let quant_params = step_sizes
        .iter()
        .map(|step| (step.exponent, step.mantissa))
        .collect::<Vec<_>>();
    let component_quantization_step_sizes = component_step_sizes
        .iter()
        .map(|steps| {
            steps
                .iter()
                .map(|step| (step.exponent, step.mantissa))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let component_sampling = planes
        .iter()
        .map(|plane| (plane.x_rsiz, plane.y_rsiz))
        .collect::<Vec<_>>();
    let mut high_bit_options = options.clone();
    high_bit_options.num_decomposition_levels = num_levels;
    high_bit_options.reversible = true;
    high_bit_options.use_mct = false;
    high_bit_options.component_sampling = Some(component_sampling.clone());
    let precinct_exponents = precinct_exponents_for_options(&high_bit_options, num_levels)?;

    let mut child_options = high_bit_options.clone();
    child_options.tile_size = None;
    child_options.write_tlm = false;
    child_options.write_plt = false;
    child_options.write_plm = false;
    child_options.write_ppm = false;
    child_options.write_ppt = false;

    let block_coding_mode = if options.use_ht_block_coding {
        BlockCodingMode::HighThroughput
    } else {
        BlockCodingMode::Classic
    };
    let params = EncodeParams {
        width,
        height,
        tile_width,
        tile_height,
        num_components,
        bit_depth: max_bit_depth,
        signed: planes.iter().all(|plane| plane.signed),
        component_sample_info,
        component_quantization_step_sizes,
        num_decomposition_levels: num_levels,
        reversible: true,
        code_block_width_exp: options.code_block_width_exp,
        code_block_height_exp: options.code_block_height_exp,
        num_layers: options.num_layers,
        use_mct: false,
        guard_bits,
        block_coding_mode,
        progression_order: options.progression_order,
        write_tlm: options.write_tlm,
        write_plt: options.write_plt,
        write_plm: options.write_plm,
        write_ppm: options.write_ppm,
        write_ppt: options.write_ppt,
        write_sop: options.write_sop,
        write_eph: options.write_eph,
        terminate_coding_passes: block_coding_mode == BlockCodingMode::Classic
            && options.num_layers > 1,
        component_sampling,
        roi_component_shifts: vec![0; usize::from(num_components)],
        precinct_exponents,
    };

    let mut tile_bodies = Vec::with_capacity(num_tiles as usize);
    for tile_y in 0..num_y_tiles {
        for tile_x in 0..num_x_tiles {
            let tile_index = tile_y
                .checked_mul(num_x_tiles)
                .and_then(|base| base.checked_add(tile_x))
                .ok_or("tile index overflow")?;
            let tile_index = u16::try_from(tile_index).map_err(|_| "tile index exceeds u16")?;
            let x0 = tile_x * tile_width;
            let y0 = tile_y * tile_height;
            let actual_width = (width - x0).min(tile_width);
            let actual_height = (height - y0).min(tile_height);
            let tile_plane_data = planes
                .iter()
                .map(|plane| {
                    let x_rsiz = u32::from(plane.x_rsiz);
                    let y_rsiz = u32::from(plane.y_rsiz);
                    let component_image_width = width.div_ceil(x_rsiz);
                    let component_image_height = height.div_ceil(y_rsiz);
                    let (component_x0, component_tile_width) = sampled_tile_component_axis(
                        x0,
                        actual_width,
                        x_rsiz,
                        component_image_width,
                    )?;
                    let (component_y0, component_tile_height) = sampled_tile_component_axis(
                        y0,
                        actual_height,
                        y_rsiz,
                        component_image_height,
                    )?;
                    let data = extract_component_plane_tile(
                        plane.data,
                        component_image_width,
                        component_x0,
                        component_y0,
                        component_tile_width,
                        component_tile_height,
                        plane.bit_depth,
                    )?;
                    Ok((data, component_tile_width, component_tile_height))
                })
                .collect::<Result<Vec<_>, &'static str>>()?;
            let tile_planes = planes
                .iter()
                .zip(tile_plane_data.iter())
                .map(|(plane, (data, _, _))| EncodeTypedComponentPlane {
                    data,
                    x_rsiz: plane.x_rsiz,
                    y_rsiz: plane.y_rsiz,
                    bit_depth: plane.bit_depth,
                    signed: plane.signed,
                })
                .collect::<Vec<_>>();
            let component_dimensions = tile_planes
                .iter()
                .zip(tile_plane_data.iter())
                .map(|(_, (_, component_width, component_height))| {
                    (*component_width, *component_height)
                })
                .collect::<Vec<_>>();
            let component_resolution_packets = prepare_typed_component_planes_i64_packets(
                &tile_planes,
                &component_dimensions,
                &component_step_sizes,
                guard_bits,
                num_levels,
                1u32 << (options.code_block_width_exp + 2),
                1u32 << (options.code_block_height_exp + 2),
                block_coding_mode,
                ht_target_coding_passes_for_options(options),
            )?;
            let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
            let packetized_tile = packetize_i64_component_resolution_packets(
                component_resolution_packets,
                actual_width,
                actual_height,
                num_components,
                num_levels,
                &params,
                &child_options,
                &mut accelerator,
            )?;
            tile_bodies.extend(split_packetized_tile_into_tile_parts(
                tile_index,
                &packetized_tile.data,
                &packetized_tile.packet_lengths,
                &packetized_tile.packet_headers,
                options.tile_part_packet_limit,
            )?);
        }
    }

    let tile_packet_headers = tile_bodies
        .iter()
        .map(|tile| tile.packet_headers.as_slice())
        .collect::<Vec<_>>();
    validate_packet_header_marker_payloads(
        params.write_ppm,
        params.write_ppt,
        &tile_packet_headers,
    )?;
    let tile_parts = tile_bodies
        .iter()
        .map(|tile| codestream_write::TilePartData {
            tile_index: tile.tile_index,
            tile_part_index: tile.tile_part_index,
            num_tile_parts: tile.num_tile_parts,
            data: &tile.data,
            packet_lengths: &tile.packet_lengths,
            packet_headers: &tile.packet_headers,
        })
        .collect::<Vec<_>>();

    Ok(codestream_write::write_codestream_tiles(
        &params,
        &tile_parts,
        &quant_params,
    ))
}

fn min_sampled_tile_component_decomposition_levels(
    planes: &[EncodeTypedComponentPlane<'_>],
    width: u32,
    height: u32,
    tile_width: u32,
    tile_height: u32,
) -> Result<u8, &'static str> {
    let num_x_tiles = width.div_ceil(tile_width);
    let num_y_tiles = height.div_ceil(tile_height);
    let mut levels: Option<u8> = None;
    for tile_y in 0..num_y_tiles {
        for tile_x in 0..num_x_tiles {
            let x0 = tile_x * tile_width;
            let y0 = tile_y * tile_height;
            let actual_width = (width - x0).min(tile_width);
            let actual_height = (height - y0).min(tile_height);
            for plane in planes {
                let x_rsiz = u32::from(plane.x_rsiz);
                let y_rsiz = u32::from(plane.y_rsiz);
                let component_image_width = width.div_ceil(x_rsiz);
                let component_image_height = height.div_ceil(y_rsiz);
                let (_, component_tile_width) =
                    sampled_tile_component_axis(x0, actual_width, x_rsiz, component_image_width)?;
                let (_, component_tile_height) =
                    sampled_tile_component_axis(y0, actual_height, y_rsiz, component_image_height)?;
                let component_levels =
                    max_decomposition_levels(component_tile_width, component_tile_height);
                levels = Some(levels.map_or(component_levels, |min| min.min(component_levels)));
            }
        }
    }
    Ok(levels.unwrap_or(0))
}

fn sampled_tile_component_axis(
    tile_origin: u32,
    tile_extent: u32,
    sampling: u32,
    component_extent: u32,
) -> Result<(u32, u32), &'static str> {
    let tile_end = tile_origin
        .checked_add(tile_extent)
        .ok_or("tile component bounds overflow")?;
    let start = tile_origin.div_ceil(sampling).min(component_extent);
    let end = tile_end.div_ceil(sampling).min(component_extent);
    Ok((start, end.saturating_sub(start)))
}

#[allow(clippy::too_many_arguments)]
fn prepare_typed_component_planes_i64_packets(
    planes: &[EncodeTypedComponentPlane<'_>],
    component_dimensions: &[(u32, u32)],
    component_step_sizes: &[Vec<QuantStepSize>],
    guard_bits: u8,
    num_levels: u8,
    cb_width: u32,
    cb_height: u32,
    block_coding_mode: BlockCodingMode,
    ht_target_coding_passes: u8,
) -> Result<Vec<Vec<PreparedResolutionPacket>>, &'static str> {
    if component_dimensions.len() != planes.len() {
        return Err("component dimensions count does not match component count");
    }
    let mut component_resolution_packets = Vec::with_capacity(planes.len());
    for (component_idx, (plane, &(component_width, component_height))) in
        planes.iter().zip(component_dimensions).enumerate()
    {
        let samples = typed_component_plane_to_i64(plane, component_width, component_height)?;
        let decomp = fdwt::forward_dwt_i64(&samples, component_width, component_height, num_levels);
        let steps = component_step_sizes
            .get(component_idx)
            .ok_or("component quantization step count mismatch")?;
        let component = u16::try_from(component_idx).map_err(|_| "component index exceeds u16")?;
        let mut packets = Vec::with_capacity(num_levels as usize + 1);

        let ll_subband = prepare_subband_i64(
            &decomp.ll,
            decomp.ll_width,
            decomp.ll_height,
            steps
                .first()
                .ok_or("reversible quantization step missing")?,
            guard_bits,
            cb_width,
            cb_height,
            SubBandType::LowLow,
            0,
            &[],
            1,
            block_coding_mode,
            ht_target_coding_passes,
        )?;
        packets.push(PreparedResolutionPacket {
            component,
            resolution: 0,
            precinct: 0,
            subbands: vec![ll_subband],
        });

        for (level_idx, level) in decomp.levels.iter().enumerate() {
            let step_base = 1 + level_idx * 3;
            let hl_subband = prepare_subband_i64(
                &level.hl,
                level.high_width,
                level.low_height,
                steps
                    .get(step_base)
                    .ok_or("reversible quantization step missing")?,
                guard_bits,
                cb_width,
                cb_height,
                SubBandType::HighLow,
                0,
                &[],
                1,
                block_coding_mode,
                ht_target_coding_passes,
            )?;
            let lh_subband = prepare_subband_i64(
                &level.lh,
                level.low_width,
                level.high_height,
                steps
                    .get(step_base + 1)
                    .ok_or("reversible quantization step missing")?,
                guard_bits,
                cb_width,
                cb_height,
                SubBandType::LowHigh,
                0,
                &[],
                1,
                block_coding_mode,
                ht_target_coding_passes,
            )?;
            let hh_subband = prepare_subband_i64(
                &level.hh,
                level.high_width,
                level.high_height,
                steps
                    .get(step_base + 2)
                    .ok_or("reversible quantization step missing")?,
                guard_bits,
                cb_width,
                cb_height,
                SubBandType::HighHigh,
                0,
                &[],
                1,
                block_coding_mode,
                ht_target_coding_passes,
            )?;
            packets.push(PreparedResolutionPacket {
                component,
                resolution: u32::try_from(level_idx + 1)
                    .map_err(|_| "resolution index exceeds u32")?,
                precinct: 0,
                subbands: vec![hl_subband, lh_subband, hh_subband],
            });
        }
        component_resolution_packets.push(packets);
    }

    Ok(component_resolution_packets)
}

/// Encode precomputed reversible 5/3 wavelet coefficients into a classic
/// JPEG 2000 Part 1 codestream.
///
/// This mirrors [`encode_precomputed_htj2k_53`] while selecting classic EBCOT
/// block coding. It reuses the same quantization, packetization, and codestream
/// writer stages as the normal encoder and is primarily intended for fixtures
/// and coefficient-domain workflows that need JPEG-native component sampling.
pub fn encode_precomputed_j2k_53(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
) -> Result<Vec<u8>, &'static str> {
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_precomputed_j2k_53_with_mct_and_accelerator(image, options, false, &mut accelerator)
}

/// Encode precomputed reversible 5/3 wavelet coefficients into a classic
/// JPEG 2000 Part 1 codestream using optional block encode and packetization
/// hooks.
pub fn encode_precomputed_j2k_53_with_accelerator(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    encode_precomputed_j2k_53_with_mct_and_accelerator(image, options, false, accelerator)
}

/// Encode precomputed reversible 5/3 wavelet coefficients into a classic
/// JPEG 2000 Part 1 codestream while controlling the output COD
/// multi-component transform flag.
pub fn encode_precomputed_j2k_53_with_mct(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    use_mct: bool,
) -> Result<Vec<u8>, &'static str> {
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_precomputed_j2k_53_with_mct_and_accelerator(image, options, use_mct, &mut accelerator)
}

/// Encode precomputed reversible 5/3 wavelet coefficients into a classic
/// JPEG 2000 Part 1 codestream while controlling the output COD
/// multi-component transform flag and using optional encode stage hooks.
pub fn encode_precomputed_j2k_53_with_mct_and_accelerator(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    use_mct: bool,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    encode_precomputed_53_with_mct_and_accelerator(image, options, use_mct, false, accelerator)
}

/// Encode precomputed reversible 5/3 wavelet coefficients into an HTJ2K
/// codestream.
///
/// This experimental entry point reuses the existing quantization, HT block
/// coding, packetization, and codestream writer stages. It bypasses the
/// encoder's forward DWT stage by supplying precomputed DWT output through the
/// internal stage hook. Coefficients are expected in the same sample domain as
/// the native encoder's FDWT input: unsigned components are already level
/// shifted by subtracting `2^(bit_depth - 1)`.
pub fn encode_precomputed_htj2k_53(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
) -> Result<Vec<u8>, &'static str> {
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_precomputed_htj2k_53_with_mct_and_accelerator(image, options, false, &mut accelerator)
}

/// Encode precomputed reversible 5/3 wavelet coefficients into an HTJ2K
/// codestream using optional block encode and packetization hooks.
pub fn encode_precomputed_htj2k_53_with_accelerator(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    encode_precomputed_htj2k_53_with_mct_and_accelerator(image, options, false, accelerator)
}

/// Encode precomputed reversible 5/3 wavelet coefficients into an HTJ2K
/// codestream while controlling the output COD multi-component transform flag.
///
/// This is intended for coefficient-domain JPEG 2000 family recoding, where
/// source codestream components may already be reversible-color-transformed.
pub fn encode_precomputed_htj2k_53_with_mct(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    use_mct: bool,
) -> Result<Vec<u8>, &'static str> {
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_precomputed_htj2k_53_with_mct_and_accelerator(image, options, use_mct, &mut accelerator)
}

/// Encode precomputed reversible 5/3 wavelet coefficients while controlling
/// the output COD multi-component transform flag and using optional encode
/// stage hooks.
pub fn encode_precomputed_htj2k_53_with_mct_and_accelerator(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    use_mct: bool,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    encode_precomputed_53_with_mct_and_accelerator(image, options, use_mct, true, accelerator)
}

fn encode_precomputed_53_with_mct_and_accelerator(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    use_mct: bool,
    use_ht_block_coding: bool,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    encode_precomputed_53_with_component_sample_info_and_accelerator(
        image,
        options,
        use_mct,
        use_ht_block_coding,
        &[],
        accelerator,
    )
}

fn encode_precomputed_53_with_component_sample_info_and_accelerator(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    use_mct: bool,
    use_ht_block_coding: bool,
    component_sample_info: &[EncodeComponentSampleInfo],
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    if image.width == 0 || image.height == 0 {
        return Err("invalid dimensions");
    }
    if image.components.is_empty() || image.components.len() > usize::from(MAX_J2K_SPEC_COMPONENTS)
    {
        return Err("unsupported component count");
    }
    if image.bit_depth == 0 || image.bit_depth > MAX_RAW_PIXEL_ENCODE_BIT_DEPTH {
        return Err("unsupported bit depth");
    }
    validate_component_sample_info(component_sample_info, image.components.len())?;
    if image
        .components
        .iter()
        .any(|component| component.x_rsiz == 0 || component.y_rsiz == 0)
    {
        return Err("component sampling factors must be non-zero");
    }
    validate_precomputed_dwt_geometry(image)?;

    let num_components =
        u16::try_from(image.components.len()).map_err(|_| "unsupported component count")?;
    let num_levels = precomputed_level_count(&image.components)?;
    let mut precomputed_options = options.clone();
    precomputed_options.num_decomposition_levels = num_levels;
    precomputed_options.reversible = true;
    precomputed_options.use_ht_block_coding = use_ht_block_coding;
    precomputed_options.use_mct = use_mct;
    precomputed_options.validate_high_throughput_codestream = false;
    precomputed_options.component_sampling = Some(
        image
            .components
            .iter()
            .map(|component| (component.x_rsiz, component.y_rsiz))
            .collect(),
    );

    let dummy_pixels =
        zero_pixel_buffer(image.width, image.height, num_components, image.bit_depth)?;
    let mut precomputed_accelerator = PrecomputedDwtAccelerator {
        outputs: image
            .components
            .iter()
            .map(|component| component.dwt.clone())
            .collect(),
        encode_accelerator: accelerator,
    };

    encode_with_accelerator_and_component_sample_info(
        &dummy_pixels,
        image.width,
        image.height,
        num_components,
        image.bit_depth,
        image.signed,
        &precomputed_options,
        component_sample_info,
        &mut precomputed_accelerator,
    )
}

/// Encode precomputed irreversible 9/7 wavelet coefficients into an HTJ2K
/// codestream.
///
/// This experimental entry point is the lossy counterpart of
/// [`encode_precomputed_htj2k_53`]. It bypasses the encoder's forward 9/7 DWT
/// stage by supplying precomputed floating-point DWT output through the
/// internal stage hook. Coefficients are expected in the same sample domain as
/// the native irreversible FDWT input: unsigned components are already level
/// shifted by subtracting `2^(bit_depth - 1)`.
pub fn encode_precomputed_htj2k_97(
    image: &PrecomputedHtj2k97Image,
    options: &EncodeOptions,
) -> Result<Vec<u8>, &'static str> {
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_precomputed_htj2k_97_with_accelerator(image, options, &mut accelerator)
}

/// Encode precomputed irreversible 9/7 wavelet coefficients into an HTJ2K
/// codestream using optional block encode and packetization hooks.
pub fn encode_precomputed_htj2k_97_with_accelerator(
    image: &PrecomputedHtj2k97Image,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    if image.width == 0 || image.height == 0 {
        return Err("invalid dimensions");
    }
    if image.components.is_empty() || image.components.len() > usize::from(MAX_J2K_SPEC_COMPONENTS)
    {
        return Err("unsupported component count");
    }
    if image.bit_depth == 0 || image.bit_depth > 16 {
        return Err("unsupported bit depth");
    }
    validate_irreversible_quantization_profile(options)?;
    if image
        .components
        .iter()
        .any(|component| component.x_rsiz == 0 || component.y_rsiz == 0)
    {
        return Err("component sampling factors must be non-zero");
    }
    validate_precomputed_dwt97_geometry(image)?;

    let num_components =
        u16::try_from(image.components.len()).map_err(|_| "unsupported component count")?;
    let num_levels = precomputed_97_level_count(&image.components)?;
    let mut precomputed_options = options.clone();
    precomputed_options.num_decomposition_levels = num_levels;
    precomputed_options.reversible = false;
    precomputed_options.use_ht_block_coding = true;
    precomputed_options.use_mct = false;
    precomputed_options.validate_high_throughput_codestream = false;
    precomputed_options.component_sampling = Some(
        image
            .components
            .iter()
            .map(|component| (component.x_rsiz, component.y_rsiz))
            .collect(),
    );

    let dummy_pixels =
        zero_pixel_buffer(image.width, image.height, num_components, image.bit_depth)?;
    let mut precomputed_accelerator = PrecomputedDwt97Accelerator {
        outputs: image
            .components
            .iter()
            .map(|component| component.dwt.clone())
            .collect(),
        encode_accelerator: accelerator,
    };

    encode_with_accelerator(
        &dummy_pixels,
        image.width,
        image.height,
        num_components,
        image.bit_depth,
        image.signed,
        &precomputed_options,
        &mut precomputed_accelerator,
    )
}

/// Encode multiple precomputed irreversible 9/7 wavelet images while sharing
/// one HT code-block batch across all prepared tiles.
pub fn encode_precomputed_htj2k_97_batch_with_accelerator(
    images: &[PrecomputedHtj2k97Image],
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<Vec<u8>>, &'static str> {
    if images.is_empty() {
        return Ok(Vec::new());
    }
    if options.num_layers != 1 {
        return Err("batch precomputed 9/7 encode currently supports one quality layer");
    }

    let mut prepared_images = prepare_precomputed_htj2k97_images_for_batch(images, options)?;
    let mut all_packets = Vec::new();
    for prepared in &mut prepared_images {
        prepared.packet_count = prepared.prepared_packets.len();
        all_packets.append(&mut prepared.prepared_packets);
    }

    let mut encoded_packets =
        encode_prepared_resolution_packets(all_packets, accelerator)?.into_iter();
    let mut codestreams = Vec::with_capacity(prepared_images.len());
    for prepared in prepared_images {
        let mut resolution_packets = Vec::with_capacity(prepared.packet_count);
        for _ in 0..prepared.packet_count {
            resolution_packets.push(
                encoded_packets
                    .next()
                    .ok_or("encoded packet count mismatch")?,
            );
        }
        let scalar_packet_descriptors = scalar_packet_descriptors(&prepared.packet_descriptors);
        let packetized_tile =
            packet_encode::form_tile_bitstream_with_descriptors_lengths_and_markers(
                &mut resolution_packets,
                &scalar_packet_descriptors,
                packet_encode::PacketMarkerOptions {
                    write_sop: prepared.params.write_sop,
                    write_eph: prepared.params.write_eph,
                    separate_packet_headers: prepared.params.write_ppm || prepared.params.write_ppt,
                },
            )?;
        codestreams.push(write_single_tile_packetized_codestream(
            &prepared.params,
            &packetized_tile,
            &prepared.quant_params,
            options.tile_part_packet_limit,
        )?);
    }
    if encoded_packets.next().is_some() {
        return Err("encoded packet count mismatch");
    }

    Ok(codestreams)
}

#[cfg(feature = "parallel")]
fn prepare_precomputed_htj2k97_images_for_batch(
    images: &[PrecomputedHtj2k97Image],
    options: &EncodeOptions,
) -> Result<Vec<PreparedPrecomputedHtj2k97Image>, &'static str> {
    images
        .par_iter()
        .map(|image| prepare_precomputed_htj2k97_image_for_batch(image, options))
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn prepare_precomputed_htj2k97_images_for_batch(
    images: &[PrecomputedHtj2k97Image],
    options: &EncodeOptions,
) -> Result<Vec<PreparedPrecomputedHtj2k97Image>, &'static str> {
    images
        .iter()
        .map(|image| prepare_precomputed_htj2k97_image_for_batch(image, options))
        .collect()
}

/// Encode prequantized irreversible 9/7 code-block coefficients into an HTJ2K
/// codestream.
pub fn encode_prequantized_htj2k_97(
    image: &PrequantizedHtj2k97Image,
    options: &EncodeOptions,
) -> Result<Vec<u8>, &'static str> {
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_prequantized_htj2k_97_with_accelerator(image, options, &mut accelerator)
}

/// Encode prequantized irreversible 9/7 code-block coefficients into an HTJ2K
/// codestream using optional block encode and packetization hooks.
pub fn encode_prequantized_htj2k_97_with_accelerator(
    image: &PrequantizedHtj2k97Image,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    if image.width == 0 || image.height == 0 {
        return Err("invalid dimensions");
    }
    if image.components.is_empty() || image.components.len() > usize::from(MAX_J2K_SPEC_COMPONENTS)
    {
        return Err("unsupported component count");
    }
    if image.bit_depth == 0 || image.bit_depth > 16 {
        return Err("unsupported bit depth");
    }
    validate_irreversible_quantization_profile(options)?;
    if image
        .components
        .iter()
        .any(|component| component.x_rsiz == 0 || component.y_rsiz == 0)
    {
        return Err("component sampling factors must be non-zero");
    }

    let num_components =
        u16::try_from(image.components.len()).map_err(|_| "unsupported component count")?;
    let num_levels = prequantized_97_level_count(&image.components)?;
    let guard_bits = options.guard_bits.max(2);
    let step_sizes = quantize::compute_step_sizes_with_irreversible_profile(
        image.bit_depth,
        num_levels,
        false,
        guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
    );
    validate_prequantized_htj2k97_image(image, guard_bits, &step_sizes)?;

    let mut prequantized_options = options.clone();
    prequantized_options.num_decomposition_levels = num_levels;
    prequantized_options.reversible = false;
    prequantized_options.use_ht_block_coding = true;
    prequantized_options.use_mct = false;
    prequantized_options.validate_high_throughput_codestream = false;
    prequantized_options.component_sampling = Some(
        image
            .components
            .iter()
            .map(|component| (component.x_rsiz, component.y_rsiz))
            .collect(),
    );

    let component_resolution_packets = image
        .components
        .iter()
        .enumerate()
        .map(|(component_idx, component)| {
            prepared_resolution_packets_from_prequantized_component(component_idx, component)
        })
        .collect::<Result<Vec<_>, &'static str>>()?;
    let prepared_resolution_packets =
        ordered_prepared_resolution_packets(component_resolution_packets, &prequantized_options)?;
    let packet_descriptors = packet_descriptors_for_order(
        &prepared_resolution_packets,
        1,
        prequantized_options.progression_order,
    )?;
    let mut resolution_packets =
        encode_prepared_resolution_packets(prepared_resolution_packets, accelerator)?;
    let packetized_tile = packetize_resolution_packets_with_options(
        &mut resolution_packets,
        &packet_descriptors,
        1,
        num_components,
        prequantized_options.progression_order,
        packet_encode::PacketMarkerOptions {
            write_sop: prequantized_options.write_sop,
            write_eph: prequantized_options.write_eph,
            separate_packet_headers: prequantized_options.write_ppm
                || prequantized_options.write_ppt,
        },
        true,
        prequantized_options.write_plt
            || prequantized_options.write_plm
            || prequantized_options.write_ppm
            || prequantized_options.write_ppt
            || prequantized_options.write_sop
            || prequantized_options.write_eph
            || prequantized_options.tile_part_packet_limit.is_some(),
        accelerator,
    )?;

    let quant_params: Vec<(u16, u16)> = step_sizes
        .iter()
        .map(|s| (s.exponent, s.mantissa))
        .collect();
    let params = EncodeParams {
        width: image.width,
        height: image.height,
        tile_width: image.width,
        tile_height: image.height,
        num_components,
        bit_depth: image.bit_depth,
        signed: image.signed,
        component_sample_info: Vec::new(),
        component_quantization_step_sizes: Vec::new(),
        num_decomposition_levels: num_levels,
        reversible: false,
        code_block_width_exp: prequantized_options.code_block_width_exp,
        code_block_height_exp: prequantized_options.code_block_height_exp,
        num_layers: 1,
        use_mct: false,
        guard_bits,
        block_coding_mode: BlockCodingMode::HighThroughput,
        progression_order: prequantized_options.progression_order,
        write_tlm: prequantized_options.write_tlm,
        write_plt: prequantized_options.write_plt,
        write_plm: prequantized_options.write_plm,
        write_ppm: prequantized_options.write_ppm,
        write_ppt: prequantized_options.write_ppt,
        write_sop: prequantized_options.write_sop,
        write_eph: prequantized_options.write_eph,
        terminate_coding_passes: false,
        component_sampling: prequantized_options
            .component_sampling
            .clone()
            .ok_or("component sampling missing")?,
        roi_component_shifts: vec![0; usize::from(num_components)],
        precinct_exponents: precinct_exponents_for_options(&prequantized_options, num_levels)?,
    };

    write_single_tile_packetized_codestream(
        &params,
        &packetized_tile,
        &quant_params,
        prequantized_options.tile_part_packet_limit,
    )
}

/// Encode preencoded irreversible 9/7 HTJ2K code-block payloads into a
/// codestream.
pub fn encode_preencoded_htj2k_97(
    image: &PreencodedHtj2k97Image,
    options: &EncodeOptions,
) -> Result<Vec<u8>, &'static str> {
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_preencoded_htj2k_97_with_accelerator(image, options, &mut accelerator)
}

/// Encode preencoded irreversible 9/7 HTJ2K code-block payloads into a
/// codestream using optional packetization hooks.
pub fn encode_preencoded_htj2k_97_with_accelerator(
    image: &PreencodedHtj2k97Image,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    if image.width == 0 || image.height == 0 {
        return Err("invalid dimensions");
    }
    if image.components.is_empty() || image.components.len() > usize::from(MAX_J2K_SPEC_COMPONENTS)
    {
        return Err("unsupported component count");
    }
    if image.bit_depth == 0 || image.bit_depth > 16 {
        return Err("unsupported bit depth");
    }
    validate_irreversible_quantization_profile(options)?;
    if image
        .components
        .iter()
        .any(|component| component.x_rsiz == 0 || component.y_rsiz == 0)
    {
        return Err("component sampling factors must be non-zero");
    }

    let num_components =
        u16::try_from(image.components.len()).map_err(|_| "unsupported component count")?;
    let num_levels = preencoded_97_level_count(&image.components)?;
    let guard_bits = options.guard_bits.max(2);
    let step_sizes = quantize::compute_step_sizes_with_irreversible_profile(
        image.bit_depth,
        num_levels,
        false,
        guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
    );
    validate_preencoded_htj2k97_image(image, guard_bits, &step_sizes)?;

    let mut preencoded_options = options.clone();
    preencoded_options.num_decomposition_levels = num_levels;
    preencoded_options.reversible = false;
    preencoded_options.use_ht_block_coding = true;
    preencoded_options.use_mct = false;
    preencoded_options.validate_high_throughput_codestream = false;
    preencoded_options.component_sampling = Some(
        image
            .components
            .iter()
            .map(|component| (component.x_rsiz, component.y_rsiz))
            .collect(),
    );

    let component_resolution_packets = image
        .components
        .iter()
        .enumerate()
        .map(|(component_idx, component)| {
            prepared_resolution_packets_from_preencoded_component(component_idx, component)
        })
        .collect::<Result<Vec<_>, &'static str>>()?;
    let prepared_resolution_packets =
        ordered_prepared_resolution_packets(component_resolution_packets, &preencoded_options)?;
    let packet_descriptors = packet_descriptors_for_order(
        &prepared_resolution_packets,
        1,
        preencoded_options.progression_order,
    )?;
    let mut resolution_packets =
        encode_prepared_resolution_packets(prepared_resolution_packets, accelerator)?;
    let packetized_tile = packetize_resolution_packets_with_options(
        &mut resolution_packets,
        &packet_descriptors,
        1,
        num_components,
        preencoded_options.progression_order,
        packet_encode::PacketMarkerOptions {
            write_sop: preencoded_options.write_sop,
            write_eph: preencoded_options.write_eph,
            separate_packet_headers: preencoded_options.write_ppm || preencoded_options.write_ppt,
        },
        true,
        preencoded_options.write_plt
            || preencoded_options.write_plm
            || preencoded_options.write_ppm
            || preencoded_options.write_ppt
            || preencoded_options.write_sop
            || preencoded_options.write_eph
            || preencoded_options.tile_part_packet_limit.is_some(),
        accelerator,
    )?;

    let quant_params: Vec<(u16, u16)> = step_sizes
        .iter()
        .map(|s| (s.exponent, s.mantissa))
        .collect();
    let params = EncodeParams {
        width: image.width,
        height: image.height,
        tile_width: image.width,
        tile_height: image.height,
        num_components,
        bit_depth: image.bit_depth,
        signed: image.signed,
        component_sample_info: Vec::new(),
        component_quantization_step_sizes: Vec::new(),
        num_decomposition_levels: num_levels,
        reversible: false,
        code_block_width_exp: preencoded_options.code_block_width_exp,
        code_block_height_exp: preencoded_options.code_block_height_exp,
        num_layers: 1,
        use_mct: false,
        guard_bits,
        block_coding_mode: BlockCodingMode::HighThroughput,
        progression_order: preencoded_options.progression_order,
        write_tlm: preencoded_options.write_tlm,
        write_plt: preencoded_options.write_plt,
        write_plm: preencoded_options.write_plm,
        write_ppm: preencoded_options.write_ppm,
        write_ppt: preencoded_options.write_ppt,
        write_sop: preencoded_options.write_sop,
        write_eph: preencoded_options.write_eph,
        terminate_coding_passes: false,
        component_sampling: preencoded_options
            .component_sampling
            .clone()
            .ok_or("component sampling missing")?,
        roi_component_shifts: vec![0; usize::from(num_components)],
        precinct_exponents: precinct_exponents_for_options(&preencoded_options, num_levels)?,
    };

    write_single_tile_packetized_codestream(
        &params,
        &packetized_tile,
        &quant_params,
        preencoded_options.tile_part_packet_limit,
    )
}

/// Encode preencoded irreversible 9/7 HTJ2K code-block payloads into a
/// codestream, consuming the image so code-block payloads can move into packet
/// preparation without cloning.
pub fn encode_preencoded_htj2k_97_owned_with_accelerator(
    image: PreencodedHtj2k97Image,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    if image.width == 0 || image.height == 0 {
        return Err("invalid dimensions");
    }
    if image.components.is_empty() || image.components.len() > usize::from(MAX_J2K_SPEC_COMPONENTS)
    {
        return Err("unsupported component count");
    }
    if image.bit_depth == 0 || image.bit_depth > 16 {
        return Err("unsupported bit depth");
    }
    validate_irreversible_quantization_profile(options)?;
    if image
        .components
        .iter()
        .any(|component| component.x_rsiz == 0 || component.y_rsiz == 0)
    {
        return Err("component sampling factors must be non-zero");
    }

    let width = image.width;
    let height = image.height;
    let bit_depth = image.bit_depth;
    let signed = image.signed;
    let num_components =
        u16::try_from(image.components.len()).map_err(|_| "unsupported component count")?;
    let num_levels = preencoded_97_level_count(&image.components)?;
    let guard_bits = options.guard_bits.max(2);
    let step_sizes = quantize::compute_step_sizes_with_irreversible_profile(
        bit_depth,
        num_levels,
        false,
        guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
    );
    validate_preencoded_htj2k97_image(&image, guard_bits, &step_sizes)?;

    let component_sampling = image
        .components
        .iter()
        .map(|component| (component.x_rsiz, component.y_rsiz))
        .collect::<Vec<_>>();
    let mut preencoded_options = options.clone();
    preencoded_options.num_decomposition_levels = num_levels;
    preencoded_options.reversible = false;
    preencoded_options.use_ht_block_coding = true;
    preencoded_options.use_mct = false;
    preencoded_options.validate_high_throughput_codestream = false;
    preencoded_options.component_sampling = Some(component_sampling);

    let component_resolution_packets = image
        .components
        .into_iter()
        .enumerate()
        .map(|(component_idx, component)| {
            prepared_resolution_packets_from_preencoded_component_owned(component_idx, component)
        })
        .collect::<Result<Vec<_>, &'static str>>()?;
    let prepared_resolution_packets =
        ordered_prepared_resolution_packets(component_resolution_packets, &preencoded_options)?;
    let packet_descriptors = packet_descriptors_for_order(
        &prepared_resolution_packets,
        1,
        preencoded_options.progression_order,
    )?;
    let mut resolution_packets =
        encode_prepared_resolution_packets(prepared_resolution_packets, accelerator)?;
    let packetized_tile = packetize_resolution_packets_with_options(
        &mut resolution_packets,
        &packet_descriptors,
        1,
        num_components,
        preencoded_options.progression_order,
        packet_encode::PacketMarkerOptions {
            write_sop: preencoded_options.write_sop,
            write_eph: preencoded_options.write_eph,
            separate_packet_headers: preencoded_options.write_ppm || preencoded_options.write_ppt,
        },
        true,
        preencoded_options.write_plt
            || preencoded_options.write_plm
            || preencoded_options.write_ppm
            || preencoded_options.write_ppt
            || preencoded_options.write_sop
            || preencoded_options.write_eph
            || preencoded_options.tile_part_packet_limit.is_some(),
        accelerator,
    )?;

    let quant_params: Vec<(u16, u16)> = step_sizes
        .iter()
        .map(|s| (s.exponent, s.mantissa))
        .collect();
    let params = EncodeParams {
        width,
        height,
        tile_width: width,
        tile_height: height,
        num_components,
        bit_depth,
        signed,
        component_sample_info: Vec::new(),
        component_quantization_step_sizes: Vec::new(),
        num_decomposition_levels: num_levels,
        reversible: false,
        code_block_width_exp: preencoded_options.code_block_width_exp,
        code_block_height_exp: preencoded_options.code_block_height_exp,
        num_layers: 1,
        use_mct: false,
        guard_bits,
        block_coding_mode: BlockCodingMode::HighThroughput,
        progression_order: preencoded_options.progression_order,
        write_tlm: preencoded_options.write_tlm,
        write_plt: preencoded_options.write_plt,
        write_plm: preencoded_options.write_plm,
        write_ppm: preencoded_options.write_ppm,
        write_ppt: preencoded_options.write_ppt,
        write_sop: preencoded_options.write_sop,
        write_eph: preencoded_options.write_eph,
        terminate_coding_passes: false,
        component_sampling: preencoded_options
            .component_sampling
            .clone()
            .ok_or("component sampling missing")?,
        roi_component_shifts: vec![0; usize::from(num_components)],
        precinct_exponents: precinct_exponents_for_options(&preencoded_options, num_levels)?,
    };

    write_single_tile_packetized_codestream(
        &params,
        &packetized_tile,
        &quant_params,
        preencoded_options.tile_part_packet_limit,
    )
}

/// Encode compact preencoded irreversible 9/7 HTJ2K code-block payloads into a
/// codestream, borrowing code-block ranges from one image-level payload buffer
/// during packetization.
pub fn encode_preencoded_htj2k_97_compact_owned_with_accelerator(
    image: PreencodedHtj2k97CompactImage,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    if image.width == 0 || image.height == 0 {
        return Err("invalid dimensions");
    }
    if image.components.is_empty() || image.components.len() > usize::from(MAX_J2K_SPEC_COMPONENTS)
    {
        return Err("unsupported component count");
    }
    if image.bit_depth == 0 || image.bit_depth > 16 {
        return Err("unsupported bit depth");
    }
    if options.write_plt
        || options.write_plm
        || options.write_sop
        || options.write_eph
        || options.tile_part_packet_limit.is_some()
    {
        return Err(
            "compact preencoded HTJ2K encode does not support packet marker or tile-part options",
        );
    }
    validate_irreversible_quantization_profile(options)?;
    if image
        .components
        .iter()
        .any(|component| component.x_rsiz == 0 || component.y_rsiz == 0)
    {
        return Err("component sampling factors must be non-zero");
    }

    let width = image.width;
    let height = image.height;
    let bit_depth = image.bit_depth;
    let signed = image.signed;
    let num_components =
        u16::try_from(image.components.len()).map_err(|_| "unsupported component count")?;
    let num_levels = preencoded_compact_97_level_count(&image.components)?;
    let guard_bits = options.guard_bits.max(2);
    let step_sizes = quantize::compute_step_sizes_with_irreversible_profile(
        bit_depth,
        num_levels,
        false,
        guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
    );
    validate_preencoded_compact_htj2k97_image(&image, guard_bits, &step_sizes)?;

    let component_sampling = image
        .components
        .iter()
        .map(|component| (component.x_rsiz, component.y_rsiz))
        .collect::<Vec<_>>();
    let mut preencoded_options = options.clone();
    preencoded_options.num_decomposition_levels = num_levels;
    preencoded_options.reversible = false;
    preencoded_options.use_ht_block_coding = true;
    preencoded_options.use_mct = false;
    preencoded_options.validate_high_throughput_codestream = false;
    preencoded_options.component_sampling = Some(component_sampling);

    let PreencodedHtj2k97CompactImage {
        payload,
        components,
        ..
    } = image;
    let component_resolution_packets = components
        .iter()
        .enumerate()
        .map(|(component_idx, component)| {
            prepared_resolution_packets_from_preencoded_compact_component(
                component_idx,
                component,
                &payload,
            )
        })
        .collect::<Result<Vec<_>, &'static str>>()?;
    let prepared_resolution_packets = ordered_prepared_compact_resolution_packets(
        component_resolution_packets,
        &preencoded_options,
    )?;
    let packet_descriptors = packet_descriptors_for_compact_order(
        &prepared_resolution_packets,
        1,
        preencoded_options.progression_order,
    )?;
    let packetization_resolutions =
        public_packetization_resolutions_from_compact(&prepared_resolution_packets);
    let packetization_job = J2kPacketizationEncodeJob {
        resolution_count: packetization_resolutions.len() as u32,
        num_layers: 1,
        num_components,
        code_block_count: count_compact_code_blocks(&prepared_resolution_packets)?,
        progression_order: public_packetization_progression_order(
            preencoded_options.progression_order,
        ),
        packet_descriptors: &packet_descriptors,
        resolutions: &packetization_resolutions,
    };
    let tile_data = accelerator
        .encode_packetization(packetization_job)?
        .map_or_else(
            || crate::encode_j2k_packetization_scalar(packetization_job),
            Ok,
        )?;

    let quant_params: Vec<(u16, u16)> = step_sizes
        .iter()
        .map(|s| (s.exponent, s.mantissa))
        .collect();
    let params = EncodeParams {
        width,
        height,
        tile_width: width,
        tile_height: height,
        num_components,
        bit_depth,
        signed,
        component_sample_info: Vec::new(),
        component_quantization_step_sizes: Vec::new(),
        num_decomposition_levels: num_levels,
        reversible: false,
        code_block_width_exp: preencoded_options.code_block_width_exp,
        code_block_height_exp: preencoded_options.code_block_height_exp,
        num_layers: 1,
        use_mct: false,
        guard_bits,
        block_coding_mode: BlockCodingMode::HighThroughput,
        progression_order: preencoded_options.progression_order,
        write_tlm: preencoded_options.write_tlm,
        write_plt: preencoded_options.write_plt,
        write_plm: preencoded_options.write_plm,
        write_ppm: preencoded_options.write_ppm,
        write_ppt: preencoded_options.write_ppt,
        write_sop: preencoded_options.write_sop,
        write_eph: preencoded_options.write_eph,
        terminate_coding_passes: false,
        component_sampling: preencoded_options
            .component_sampling
            .clone()
            .ok_or("component sampling missing")?,
        roi_component_shifts: vec![0; usize::from(num_components)],
        precinct_exponents: precinct_exponents_for_options(&preencoded_options, num_levels)?,
    };

    Ok(codestream_write::write_codestream(
        &params,
        &tile_data,
        &quant_params,
    ))
}

fn validate_precomputed_dwt_geometry(image: &PrecomputedHtj2k53Image) -> Result<(), &'static str> {
    for component in &image.components {
        let component_width = image.width.div_ceil(u32::from(component.x_rsiz));
        let component_height = image.height.div_ceil(u32::from(component.y_rsiz));
        validate_precomputed_component_dwt_geometry(
            &component.dwt,
            component_width,
            component_height,
        )?;
    }

    Ok(())
}

fn validate_precomputed_dwt97_geometry(
    image: &PrecomputedHtj2k97Image,
) -> Result<(), &'static str> {
    for component in &image.components {
        let component_width = image.width.div_ceil(u32::from(component.x_rsiz));
        let component_height = image.height.div_ceil(u32::from(component.y_rsiz));
        validate_precomputed_component_dwt_geometry(
            &component.dwt,
            component_width,
            component_height,
        )?;
    }

    Ok(())
}

fn validate_precomputed_component_dwt_geometry(
    dwt: &impl PrecomputedDwtGeometryView,
    component_width: u32,
    component_height: u32,
) -> Result<(), &'static str> {
    if let Some(highest_level) = dwt.last_level_geometry() {
        if highest_level.width != component_width || highest_level.height != component_height {
            return Err("precomputed DWT component dimensions mismatch");
        }
    }

    let mut expected_width = component_width;
    let mut expected_height = component_height;
    for level_index in (0..dwt.level_count()).rev() {
        let level = dwt.level_geometry(level_index);
        let low_width = expected_width.div_ceil(2);
        let low_height = expected_height.div_ceil(2);
        let high_width = expected_width / 2;
        let high_height = expected_height / 2;

        if level.width != expected_width
            || level.height != expected_height
            || level.low_width != low_width
            || level.low_height != low_height
            || level.high_width != high_width
            || level.high_height != high_height
        {
            return Err("precomputed DWT recursive geometry mismatch");
        }
        validate_band_len(level.hl_len, high_width, low_height)?;
        validate_band_len(level.lh_len, low_width, high_height)?;
        validate_band_len(level.hh_len, high_width, high_height)?;

        expected_width = low_width;
        expected_height = low_height;
    }

    if dwt.ll_width() != expected_width || dwt.ll_height() != expected_height {
        return Err("precomputed DWT component dimensions mismatch");
    }
    validate_band_len(dwt.ll_len(), expected_width, expected_height)
}

#[derive(Debug, Clone, Copy)]
struct PrecomputedDwtLevelGeometry {
    width: u32,
    height: u32,
    low_width: u32,
    low_height: u32,
    high_width: u32,
    high_height: u32,
    hl_len: usize,
    lh_len: usize,
    hh_len: usize,
}

trait PrecomputedDwtGeometryView {
    fn ll_len(&self) -> usize;
    fn ll_width(&self) -> u32;
    fn ll_height(&self) -> u32;
    fn level_count(&self) -> usize;
    fn level_geometry(&self, index: usize) -> PrecomputedDwtLevelGeometry;

    fn last_level_geometry(&self) -> Option<PrecomputedDwtLevelGeometry> {
        self.level_count()
            .checked_sub(1)
            .map(|index| self.level_geometry(index))
    }
}

impl PrecomputedDwtGeometryView for J2kForwardDwt53Output {
    fn ll_len(&self) -> usize {
        self.ll.len()
    }

    fn ll_width(&self) -> u32 {
        self.ll_width
    }

    fn ll_height(&self) -> u32 {
        self.ll_height
    }

    fn level_count(&self) -> usize {
        self.levels.len()
    }

    fn level_geometry(&self, index: usize) -> PrecomputedDwtLevelGeometry {
        let level = &self.levels[index];
        PrecomputedDwtLevelGeometry {
            width: level.width,
            height: level.height,
            low_width: level.low_width,
            low_height: level.low_height,
            high_width: level.high_width,
            high_height: level.high_height,
            hl_len: level.hl.len(),
            lh_len: level.lh.len(),
            hh_len: level.hh.len(),
        }
    }
}

impl PrecomputedDwtGeometryView for J2kForwardDwt97Output {
    fn ll_len(&self) -> usize {
        self.ll.len()
    }

    fn ll_width(&self) -> u32 {
        self.ll_width
    }

    fn ll_height(&self) -> u32 {
        self.ll_height
    }

    fn level_count(&self) -> usize {
        self.levels.len()
    }

    fn level_geometry(&self, index: usize) -> PrecomputedDwtLevelGeometry {
        let level = &self.levels[index];
        PrecomputedDwtLevelGeometry {
            width: level.width,
            height: level.height,
            low_width: level.low_width,
            low_height: level.low_height,
            high_width: level.high_width,
            high_height: level.high_height,
            hl_len: level.hl.len(),
            lh_len: level.lh.len(),
            hh_len: level.hh.len(),
        }
    }
}

fn precomputed_level_count(components: &[PrecomputedHtj2k53Component]) -> Result<u8, &'static str> {
    let first = components
        .first()
        .ok_or("unsupported component count")?
        .dwt
        .levels
        .len();
    if components
        .iter()
        .any(|component| component.dwt.levels.len() != first)
    {
        return Err("precomputed components must use the same decomposition level count");
    }
    u8::try_from(first).map_err(|_| "decomposition level count exceeds u8")
}

fn precomputed_97_level_count(
    components: &[PrecomputedHtj2k97Component],
) -> Result<u8, &'static str> {
    let first = components
        .first()
        .ok_or("unsupported component count")?
        .dwt
        .levels
        .len();
    if components
        .iter()
        .any(|component| component.dwt.levels.len() != first)
    {
        return Err("precomputed components must use the same decomposition level count");
    }
    u8::try_from(first).map_err(|_| "decomposition level count exceeds u8")
}

fn prequantized_97_level_count(
    components: &[PrequantizedHtj2k97Component],
) -> Result<u8, &'static str> {
    let first = components
        .first()
        .ok_or("unsupported component count")?
        .resolutions
        .len()
        .checked_sub(1)
        .ok_or("prequantized components must contain at least one decomposition level")?;
    if components
        .iter()
        .any(|component| component.resolutions.len() != first + 1)
    {
        return Err("prequantized components must use the same decomposition level count");
    }
    u8::try_from(first).map_err(|_| "decomposition level count exceeds u8")
}

fn preencoded_97_level_count(
    components: &[PreencodedHtj2k97Component],
) -> Result<u8, &'static str> {
    let first = components
        .first()
        .ok_or("unsupported component count")?
        .resolutions
        .len()
        .checked_sub(1)
        .ok_or("preencoded components must contain at least one decomposition level")?;
    if components
        .iter()
        .any(|component| component.resolutions.len() != first + 1)
    {
        return Err("preencoded components must use the same decomposition level count");
    }
    u8::try_from(first).map_err(|_| "decomposition level count exceeds u8")
}

fn preencoded_compact_97_level_count(
    components: &[PreencodedHtj2k97CompactComponent],
) -> Result<u8, &'static str> {
    let first = components
        .first()
        .ok_or("unsupported component count")?
        .resolutions
        .len()
        .checked_sub(1)
        .ok_or("preencoded components must contain at least one decomposition level")?;
    if components
        .iter()
        .any(|component| component.resolutions.len() != first + 1)
    {
        return Err("preencoded components must use the same decomposition level count");
    }
    u8::try_from(first).map_err(|_| "decomposition level count exceeds u8")
}

fn validate_prequantized_htj2k97_image(
    image: &PrequantizedHtj2k97Image,
    guard_bits: u8,
    step_sizes: &[QuantStepSize],
) -> Result<(), &'static str> {
    for component in &image.components {
        if component.resolutions.is_empty() {
            return Err("prequantized components must contain at least one resolution");
        }
        validate_prequantized_resolution(
            &component.resolutions[0],
            &[J2kSubBandType::LowLow],
            guard_bits,
            &step_sizes[0..1],
        )?;
        for (level_index, resolution) in component.resolutions.iter().enumerate().skip(1) {
            let step_base = 1 + (level_index - 1) * 3;
            validate_prequantized_resolution(
                resolution,
                &[
                    J2kSubBandType::HighLow,
                    J2kSubBandType::LowHigh,
                    J2kSubBandType::HighHigh,
                ],
                guard_bits,
                &step_sizes[step_base..step_base + 3],
            )?;
        }
    }

    Ok(())
}

fn validate_preencoded_htj2k97_image(
    image: &PreencodedHtj2k97Image,
    guard_bits: u8,
    step_sizes: &[QuantStepSize],
) -> Result<(), &'static str> {
    for component in &image.components {
        if component.resolutions.is_empty() {
            return Err("preencoded components must contain at least one resolution");
        }
        validate_preencoded_resolution(
            &component.resolutions[0],
            &[J2kSubBandType::LowLow],
            guard_bits,
            &step_sizes[0..1],
        )?;
        for (level_index, resolution) in component.resolutions.iter().enumerate().skip(1) {
            let step_base = 1 + (level_index - 1) * 3;
            validate_preencoded_resolution(
                resolution,
                &[
                    J2kSubBandType::HighLow,
                    J2kSubBandType::LowHigh,
                    J2kSubBandType::HighHigh,
                ],
                guard_bits,
                &step_sizes[step_base..step_base + 3],
            )?;
        }
    }

    Ok(())
}

fn validate_preencoded_compact_htj2k97_image(
    image: &PreencodedHtj2k97CompactImage,
    guard_bits: u8,
    step_sizes: &[QuantStepSize],
) -> Result<(), &'static str> {
    for component in &image.components {
        if component.resolutions.is_empty() {
            return Err("preencoded components must contain at least one resolution");
        }
        validate_preencoded_compact_resolution(
            &component.resolutions[0],
            &[J2kSubBandType::LowLow],
            guard_bits,
            &step_sizes[0..1],
            image.payload.len(),
        )?;
        for (level_index, resolution) in component.resolutions.iter().enumerate().skip(1) {
            let step_base = 1 + (level_index - 1) * 3;
            validate_preencoded_compact_resolution(
                resolution,
                &[
                    J2kSubBandType::HighLow,
                    J2kSubBandType::LowHigh,
                    J2kSubBandType::HighHigh,
                ],
                guard_bits,
                &step_sizes[step_base..step_base + 3],
                image.payload.len(),
            )?;
        }
    }

    Ok(())
}

fn validate_prequantized_resolution(
    resolution: &PrequantizedHtj2k97Resolution,
    expected_subbands: &[J2kSubBandType],
    guard_bits: u8,
    step_sizes: &[QuantStepSize],
) -> Result<(), &'static str> {
    if resolution.subbands.len() != expected_subbands.len() {
        return Err("prequantized resolution subband count mismatch");
    }
    for ((subband, expected_subband), step_size) in resolution
        .subbands
        .iter()
        .zip(expected_subbands)
        .zip(step_sizes)
    {
        if subband.sub_band_type != *expected_subband {
            return Err("prequantized resolution subband order mismatch");
        }
        let expected_blocks = subband
            .num_cbs_x
            .checked_mul(subband.num_cbs_y)
            .ok_or("prequantized code-block count overflow")?;
        if expected_blocks == 0 {
            if subband.total_bitplanes != 0 || !subband.code_blocks.is_empty() {
                return Err("empty prequantized subbands must not contain code-block data");
            }
            continue;
        }
        debug_assert!(step_size.exponent <= u16::from(u8::MAX));
        let expected_total_bitplanes = guard_bits
            .saturating_add(step_size.exponent as u8)
            .saturating_sub(1);
        if subband.total_bitplanes != expected_total_bitplanes {
            return Err("prequantized subband bitplane count mismatch");
        }
        if usize::try_from(expected_blocks).map_err(|_| "prequantized code-block count overflow")?
            != subband.code_blocks.len()
        {
            return Err("prequantized code-block count mismatch");
        }
        for block in &subband.code_blocks {
            if block.width == 0 || block.height == 0 {
                return Err("prequantized code-block dimensions must be non-zero");
            }
            validate_band_len(block.coefficients.len(), block.width, block.height)?;
        }
    }

    Ok(())
}

fn validate_preencoded_resolution(
    resolution: &PreencodedHtj2k97Resolution,
    expected_subbands: &[J2kSubBandType],
    guard_bits: u8,
    step_sizes: &[QuantStepSize],
) -> Result<(), &'static str> {
    if resolution.subbands.len() != expected_subbands.len() {
        return Err("preencoded resolution subband count mismatch");
    }
    for ((subband, expected_subband), step_size) in resolution
        .subbands
        .iter()
        .zip(expected_subbands)
        .zip(step_sizes)
    {
        if subband.sub_band_type != *expected_subband {
            return Err("preencoded resolution subband order mismatch");
        }
        let expected_blocks = subband
            .num_cbs_x
            .checked_mul(subband.num_cbs_y)
            .ok_or("preencoded code-block count overflow")?;
        if expected_blocks == 0 {
            if subband.total_bitplanes != 0 || !subband.code_blocks.is_empty() {
                return Err("empty preencoded subbands must not contain code-block data");
            }
            continue;
        }
        debug_assert!(step_size.exponent <= u16::from(u8::MAX));
        let expected_total_bitplanes = guard_bits
            .saturating_add(step_size.exponent as u8)
            .saturating_sub(1);
        if subband.total_bitplanes != expected_total_bitplanes {
            return Err("preencoded subband bitplane count mismatch");
        }
        if usize::try_from(expected_blocks).map_err(|_| "preencoded code-block count overflow")?
            != subband.code_blocks.len()
        {
            return Err("preencoded code-block count mismatch");
        }
        for block in &subband.code_blocks {
            if block.width == 0 || block.height == 0 {
                return Err("preencoded code-block dimensions must be non-zero");
            }
            validate_preencoded_code_block_payload(&block.encoded, subband.total_bitplanes)?;
        }
    }

    Ok(())
}

fn validate_preencoded_compact_resolution(
    resolution: &PreencodedHtj2k97CompactResolution,
    expected_subbands: &[J2kSubBandType],
    guard_bits: u8,
    step_sizes: &[QuantStepSize],
    payload_len: usize,
) -> Result<(), &'static str> {
    if resolution.subbands.len() != expected_subbands.len() {
        return Err("preencoded resolution subband count mismatch");
    }
    for ((subband, expected_subband), step_size) in resolution
        .subbands
        .iter()
        .zip(expected_subbands)
        .zip(step_sizes)
    {
        if subband.sub_band_type != *expected_subband {
            return Err("preencoded resolution subband order mismatch");
        }
        let expected_blocks = subband
            .num_cbs_x
            .checked_mul(subband.num_cbs_y)
            .ok_or("preencoded code-block count overflow")?;
        if expected_blocks == 0 {
            if subband.total_bitplanes != 0 || !subband.code_blocks.is_empty() {
                return Err("empty preencoded subbands must not contain code-block data");
            }
            continue;
        }
        debug_assert!(step_size.exponent <= u16::from(u8::MAX));
        let expected_total_bitplanes = guard_bits
            .saturating_add(step_size.exponent as u8)
            .saturating_sub(1);
        if subband.total_bitplanes != expected_total_bitplanes {
            return Err("preencoded subband bitplane count mismatch");
        }
        if usize::try_from(expected_blocks).map_err(|_| "preencoded code-block count overflow")?
            != subband.code_blocks.len()
        {
            return Err("preencoded code-block count mismatch");
        }
        for block in &subband.code_blocks {
            if block.width == 0 || block.height == 0 {
                return Err("preencoded code-block dimensions must be non-zero");
            }
            validate_preencoded_compact_code_block_payload(
                block,
                payload_len,
                subband.total_bitplanes,
            )?;
        }
    }

    Ok(())
}

fn validate_preencoded_code_block_payload(
    block: &EncodedHtJ2kCodeBlock,
    total_bitplanes: u8,
) -> Result<(), &'static str> {
    let data_len = u32::try_from(block.data.len()).map_err(|_| "HTJ2K payload too large")?;
    if block.num_coding_passes == 0 {
        if data_len != 0 || block.cleanup_length != 0 || block.refinement_length != 0 {
            return Err("empty HTJ2K code-block payload metadata mismatch");
        }
        if block.num_zero_bitplanes != total_bitplanes {
            return Err("empty HTJ2K code-block zero-bitplane count mismatch");
        }
        return Ok(());
    }
    if block.num_coding_passes > 164 {
        return Err("HTJ2K code-block coding pass count out of range");
    }
    if block.num_zero_bitplanes >= total_bitplanes {
        return Err("HTJ2K code-block zero-bitplane count out of range");
    }
    let segment_len = block
        .cleanup_length
        .checked_add(block.refinement_length)
        .ok_or("HTJ2K payload segment length overflow")?;
    if segment_len != data_len {
        return Err("HTJ2K payload segment length mismatch");
    }
    Ok(())
}

fn validate_preencoded_compact_code_block_payload(
    block: &PreencodedHtj2k97CompactCodeBlock,
    payload_len: usize,
    total_bitplanes: u8,
) -> Result<(), &'static str> {
    if block.payload_range.start > block.payload_range.end || block.payload_range.end > payload_len
    {
        return Err("HTJ2K payload range out of bounds");
    }
    let data_len = u32::try_from(block.payload_range.end - block.payload_range.start)
        .map_err(|_| "HTJ2K payload too large")?;
    if block.num_coding_passes == 0 {
        if data_len != 0 || block.cleanup_length != 0 || block.refinement_length != 0 {
            return Err("empty HTJ2K code-block payload metadata mismatch");
        }
        if block.num_zero_bitplanes != total_bitplanes {
            return Err("empty HTJ2K code-block zero-bitplane count mismatch");
        }
        return Ok(());
    }
    if block.num_coding_passes > 164 {
        return Err("HTJ2K code-block coding pass count out of range");
    }
    if block.num_zero_bitplanes >= total_bitplanes {
        return Err("HTJ2K code-block zero-bitplane count out of range");
    }
    let segment_len = block
        .cleanup_length
        .checked_add(block.refinement_length)
        .ok_or("HTJ2K payload segment length overflow")?;
    if segment_len != data_len {
        return Err("HTJ2K payload segment length mismatch");
    }
    Ok(())
}

fn prepared_resolution_packets_from_prequantized_component(
    component_idx: usize,
    component: &PrequantizedHtj2k97Component,
) -> Result<Vec<PreparedResolutionPacket>, &'static str> {
    let component_idx = u16::try_from(component_idx).map_err(|_| "component index exceeds u16")?;
    component
        .resolutions
        .iter()
        .enumerate()
        .map(|(resolution_idx, resolution)| {
            Ok(PreparedResolutionPacket {
                component: component_idx,
                resolution: u32::try_from(resolution_idx)
                    .map_err(|_| "resolution index exceeds u32")?,
                precinct: 0,
                subbands: resolution
                    .subbands
                    .iter()
                    .map(prepared_subband_from_prequantized)
                    .collect::<Result<Vec<_>, &'static str>>()?,
            })
        })
        .collect()
}

fn prepared_subband_from_prequantized(
    subband: &PrequantizedHtj2k97Subband,
) -> Result<PreparedEncodeSubband, &'static str> {
    Ok(PreparedEncodeSubband {
        code_blocks: subband
            .code_blocks
            .iter()
            .map(|block| PreparedEncodeCodeBlock {
                coefficients: block.coefficients.iter().copied().map(i64::from).collect(),
                width: block.width,
                height: block.height,
            })
            .collect(),
        preencoded_ht_code_blocks: None,
        num_cbs_x: subband.num_cbs_x,
        num_cbs_y: subband.num_cbs_y,
        code_block_width: subband
            .code_blocks
            .iter()
            .map(|block| block.width)
            .max()
            .unwrap_or(0),
        code_block_height: subband
            .code_blocks
            .iter()
            .map(|block| block.height)
            .max()
            .unwrap_or(0),
        width: precomputed_subband_width(
            subband.num_cbs_x,
            subband.code_blocks.iter().map(|block| block.width),
        ),
        height: precomputed_subband_height(
            subband.num_cbs_x,
            subband.num_cbs_y,
            subband.code_blocks.iter().map(|block| block.height),
        ),
        sub_band_type: internal_sub_band_type(subband.sub_band_type),
        total_bitplanes: subband.total_bitplanes,
        block_coding_mode: BlockCodingMode::HighThroughput,
        ht_target_coding_passes: 1,
    })
}

fn precomputed_subband_width(width_in_blocks: u32, widths: impl Iterator<Item = u32>) -> u32 {
    if width_in_blocks == 0 {
        return 0;
    }

    widths.take(width_in_blocks as usize).sum()
}

fn precomputed_subband_height(
    width_in_blocks: u32,
    height_in_blocks: u32,
    heights: impl Iterator<Item = u32>,
) -> u32 {
    if width_in_blocks == 0 || height_in_blocks == 0 {
        return 0;
    }

    heights
        .step_by(width_in_blocks as usize)
        .take(height_in_blocks as usize)
        .sum()
}

fn prepared_resolution_packets_from_preencoded_component(
    component_idx: usize,
    component: &PreencodedHtj2k97Component,
) -> Result<Vec<PreparedResolutionPacket>, &'static str> {
    let component_idx = u16::try_from(component_idx).map_err(|_| "component index exceeds u16")?;
    component
        .resolutions
        .iter()
        .enumerate()
        .map(|(resolution_idx, resolution)| {
            Ok(PreparedResolutionPacket {
                component: component_idx,
                resolution: u32::try_from(resolution_idx)
                    .map_err(|_| "resolution index exceeds u32")?,
                precinct: 0,
                subbands: resolution
                    .subbands
                    .iter()
                    .map(prepared_subband_from_preencoded)
                    .collect::<Result<Vec<_>, &'static str>>()?,
            })
        })
        .collect()
}

fn prepared_resolution_packets_from_preencoded_component_owned(
    component_idx: usize,
    component: PreencodedHtj2k97Component,
) -> Result<Vec<PreparedResolutionPacket>, &'static str> {
    let component_idx = u16::try_from(component_idx).map_err(|_| "component index exceeds u16")?;
    component
        .resolutions
        .into_iter()
        .enumerate()
        .map(|(resolution_idx, resolution)| {
            Ok(PreparedResolutionPacket {
                component: component_idx,
                resolution: u32::try_from(resolution_idx)
                    .map_err(|_| "resolution index exceeds u32")?,
                precinct: 0,
                subbands: resolution
                    .subbands
                    .into_iter()
                    .map(prepared_subband_from_preencoded_owned)
                    .collect::<Result<Vec<_>, &'static str>>()?,
            })
        })
        .collect()
}

fn prepared_resolution_packets_from_preencoded_compact_component<'a>(
    component_idx: usize,
    component: &'a PreencodedHtj2k97CompactComponent,
    payload: &'a [u8],
) -> Result<Vec<PreparedCompactResolutionPacket<'a>>, &'static str> {
    let component_idx = u16::try_from(component_idx).map_err(|_| "component index exceeds u16")?;
    component
        .resolutions
        .iter()
        .enumerate()
        .map(|(resolution_idx, resolution)| {
            Ok(PreparedCompactResolutionPacket {
                component: component_idx,
                resolution: u32::try_from(resolution_idx)
                    .map_err(|_| "resolution index exceeds u32")?,
                precinct: 0,
                subbands: resolution
                    .subbands
                    .iter()
                    .map(|subband| prepared_subband_from_preencoded_compact(subband, payload))
                    .collect::<Result<Vec<_>, &'static str>>()?,
            })
        })
        .collect()
}

fn prepared_subband_from_preencoded(
    subband: &PreencodedHtj2k97Subband,
) -> Result<PreparedEncodeSubband, &'static str> {
    Ok(PreparedEncodeSubband {
        code_blocks: subband
            .code_blocks
            .iter()
            .map(|block| PreparedEncodeCodeBlock {
                coefficients: Vec::new(),
                width: block.width,
                height: block.height,
            })
            .collect(),
        preencoded_ht_code_blocks: Some(
            subband
                .code_blocks
                .iter()
                .map(|block| block.encoded.clone())
                .collect(),
        ),
        num_cbs_x: subband.num_cbs_x,
        num_cbs_y: subband.num_cbs_y,
        code_block_width: subband
            .code_blocks
            .iter()
            .map(|block| block.width)
            .max()
            .unwrap_or(0),
        code_block_height: subband
            .code_blocks
            .iter()
            .map(|block| block.height)
            .max()
            .unwrap_or(0),
        width: precomputed_subband_width(
            subband.num_cbs_x,
            subband.code_blocks.iter().map(|block| block.width),
        ),
        height: precomputed_subband_height(
            subband.num_cbs_x,
            subband.num_cbs_y,
            subband.code_blocks.iter().map(|block| block.height),
        ),
        sub_band_type: internal_sub_band_type(subband.sub_band_type),
        total_bitplanes: subband.total_bitplanes,
        block_coding_mode: BlockCodingMode::HighThroughput,
        ht_target_coding_passes: 1,
    })
}

fn prepared_subband_from_preencoded_owned(
    subband: PreencodedHtj2k97Subband,
) -> Result<PreparedEncodeSubband, &'static str> {
    let code_block_width = subband
        .code_blocks
        .iter()
        .map(|block| block.width)
        .max()
        .unwrap_or(0);
    let code_block_height = subband
        .code_blocks
        .iter()
        .map(|block| block.height)
        .max()
        .unwrap_or(0);
    let width = precomputed_subband_width(
        subband.num_cbs_x,
        subband.code_blocks.iter().map(|block| block.width),
    );
    let height = precomputed_subband_height(
        subband.num_cbs_x,
        subband.num_cbs_y,
        subband.code_blocks.iter().map(|block| block.height),
    );
    let code_blocks = subband
        .code_blocks
        .into_iter()
        .map(|block| {
            let PreencodedHtj2k97CodeBlock {
                width,
                height,
                encoded,
            } = block;
            (
                PreparedEncodeCodeBlock {
                    coefficients: Vec::new(),
                    width,
                    height,
                },
                encoded,
            )
        })
        .collect::<Vec<_>>();
    let (code_blocks, preencoded_ht_code_blocks): (Vec<_>, Vec<_>) =
        code_blocks.into_iter().unzip();

    Ok(PreparedEncodeSubband {
        code_blocks,
        preencoded_ht_code_blocks: Some(preencoded_ht_code_blocks),
        num_cbs_x: subband.num_cbs_x,
        num_cbs_y: subband.num_cbs_y,
        code_block_width,
        code_block_height,
        width,
        height,
        sub_band_type: internal_sub_band_type(subband.sub_band_type),
        total_bitplanes: subband.total_bitplanes,
        block_coding_mode: BlockCodingMode::HighThroughput,
        ht_target_coding_passes: 1,
    })
}

fn prepared_subband_from_preencoded_compact<'a>(
    subband: &'a PreencodedHtj2k97CompactSubband,
    payload: &'a [u8],
) -> Result<PreparedCompactSubband<'a>, &'static str> {
    let code_blocks = subband
        .code_blocks
        .iter()
        .map(|block| {
            Ok(PreparedCompactCodeBlock {
                data: compact_payload_slice(payload, &block.payload_range)?,
                cleanup_length: block.cleanup_length,
                refinement_length: block.refinement_length,
                num_coding_passes: block.num_coding_passes,
                num_zero_bitplanes: block.num_zero_bitplanes,
            })
        })
        .collect::<Result<Vec<_>, &'static str>>()?;

    Ok(PreparedCompactSubband {
        code_blocks,
        num_cbs_x: subband.num_cbs_x,
        num_cbs_y: subband.num_cbs_y,
    })
}

fn compact_payload_slice<'a>(
    payload: &'a [u8],
    range: &Range<usize>,
) -> Result<&'a [u8], &'static str> {
    if range.start > range.end || range.end > payload.len() {
        return Err("HTJ2K payload range out of bounds");
    }
    Ok(&payload[range.clone()])
}

fn zero_pixel_buffer(
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
) -> Result<Vec<u8>, &'static str> {
    let bytes_per_sample = raw_pixel_bytes_per_sample(bit_depth)?;
    let len = width as usize;
    let len = len
        .checked_mul(height as usize)
        .and_then(|value| value.checked_mul(usize::from(num_components)))
        .and_then(|value| value.checked_mul(bytes_per_sample))
        .ok_or("pixel buffer dimensions overflow")?;
    Ok(vec![0; len])
}

struct PrecomputedDwtAccelerator<'a, A: J2kEncodeStageAccelerator> {
    outputs: Vec<J2kForwardDwt53Output>,
    encode_accelerator: &'a mut A,
}

struct PrecomputedDwt97Accelerator<'a, A: J2kEncodeStageAccelerator> {
    outputs: Vec<J2kForwardDwt97Output>,
    encode_accelerator: &'a mut A,
}

impl<A: J2kEncodeStageAccelerator> J2kEncodeStageAccelerator for PrecomputedDwtAccelerator<'_, A> {
    fn dispatch_report(&self) -> crate::J2kEncodeDispatchReport {
        self.encode_accelerator.dispatch_report()
    }

    fn encode_forward_dwt53(
        &mut self,
        _job: J2kForwardDwt53Job<'_>,
    ) -> Result<Option<J2kForwardDwt53Output>, &'static str> {
        if self.outputs.is_empty() {
            return Err("precomputed DWT output exhausted");
        }

        Ok(Some(self.outputs.remove(0)))
    }

    fn encode_quantize_subband(
        &mut self,
        job: J2kQuantizeSubbandJob<'_>,
    ) -> Result<Option<Vec<i32>>, &'static str> {
        self.encode_accelerator.encode_quantize_subband(job)
    }

    fn encode_tier1_code_block(
        &mut self,
        job: J2kTier1CodeBlockEncodeJob<'_>,
    ) -> Result<Option<EncodedJ2kCodeBlock>, &'static str> {
        self.encode_accelerator.encode_tier1_code_block(job)
    }

    fn encode_tier1_code_blocks(
        &mut self,
        jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
    ) -> Result<Option<Vec<EncodedJ2kCodeBlock>>, &'static str> {
        self.encode_accelerator.encode_tier1_code_blocks(jobs)
    }

    fn encode_ht_code_block(
        &mut self,
        job: crate::J2kHtCodeBlockEncodeJob<'_>,
    ) -> Result<Option<EncodedHtJ2kCodeBlock>, &'static str> {
        self.encode_accelerator.encode_ht_code_block(job)
    }

    fn encode_ht_code_blocks(
        &mut self,
        jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
    ) -> Result<Option<Vec<EncodedHtJ2kCodeBlock>>, &'static str> {
        self.encode_accelerator.encode_ht_code_blocks(jobs)
    }

    fn prefer_parallel_cpu_code_block_fallback(&self) -> bool {
        self.encode_accelerator
            .prefer_parallel_cpu_code_block_fallback()
    }

    fn prefer_parallel_cpu_tile_encode(&self) -> bool {
        self.encode_accelerator.prefer_parallel_cpu_tile_encode()
    }

    fn encode_packetization(
        &mut self,
        job: J2kPacketizationEncodeJob<'_>,
    ) -> Result<Option<Vec<u8>>, &'static str> {
        self.encode_accelerator.encode_packetization(job)
    }
}

impl<A: J2kEncodeStageAccelerator> J2kEncodeStageAccelerator
    for PrecomputedDwt97Accelerator<'_, A>
{
    fn dispatch_report(&self) -> crate::J2kEncodeDispatchReport {
        self.encode_accelerator.dispatch_report()
    }

    fn encode_forward_dwt97(
        &mut self,
        _job: J2kForwardDwt97Job<'_>,
    ) -> Result<Option<J2kForwardDwt97Output>, &'static str> {
        if self.outputs.is_empty() {
            return Err("precomputed DWT output exhausted");
        }

        Ok(Some(self.outputs.remove(0)))
    }

    fn encode_quantize_subband(
        &mut self,
        job: J2kQuantizeSubbandJob<'_>,
    ) -> Result<Option<Vec<i32>>, &'static str> {
        self.encode_accelerator.encode_quantize_subband(job)
    }

    fn encode_tier1_code_block(
        &mut self,
        job: J2kTier1CodeBlockEncodeJob<'_>,
    ) -> Result<Option<EncodedJ2kCodeBlock>, &'static str> {
        self.encode_accelerator.encode_tier1_code_block(job)
    }

    fn encode_tier1_code_blocks(
        &mut self,
        jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
    ) -> Result<Option<Vec<EncodedJ2kCodeBlock>>, &'static str> {
        self.encode_accelerator.encode_tier1_code_blocks(jobs)
    }

    fn encode_ht_code_block(
        &mut self,
        job: crate::J2kHtCodeBlockEncodeJob<'_>,
    ) -> Result<Option<EncodedHtJ2kCodeBlock>, &'static str> {
        self.encode_accelerator.encode_ht_code_block(job)
    }

    fn encode_ht_code_blocks(
        &mut self,
        jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
    ) -> Result<Option<Vec<EncodedHtJ2kCodeBlock>>, &'static str> {
        self.encode_accelerator.encode_ht_code_blocks(jobs)
    }

    fn prefer_parallel_cpu_code_block_fallback(&self) -> bool {
        self.encode_accelerator
            .prefer_parallel_cpu_code_block_fallback()
    }

    fn prefer_parallel_cpu_tile_encode(&self) -> bool {
        self.encode_accelerator.prefer_parallel_cpu_tile_encode()
    }

    fn encode_packetization(
        &mut self,
        job: J2kPacketizationEncodeJob<'_>,
    ) -> Result<Option<Vec<u8>>, &'static str> {
        self.encode_accelerator.encode_packetization(job)
    }
}

fn block_coding_mode(options: &EncodeOptions) -> BlockCodingMode {
    if options.use_ht_block_coding {
        BlockCodingMode::HighThroughput
    } else {
        BlockCodingMode::Classic
    }
}

fn ht_target_coding_passes_for_options(options: &EncodeOptions) -> u8 {
    if options.use_ht_block_coding && options.num_layers > 1 {
        options.num_layers.min(3)
    } else {
        1
    }
}

fn validate_irreversible_quantization_scale(scale: f32) -> Result<(), &'static str> {
    if scale.is_finite() && scale > 0.0 {
        Ok(())
    } else {
        Err("irreversible quantization scale must be finite and greater than zero")
    }
}

fn validate_irreversible_quantization_profile(options: &EncodeOptions) -> Result<(), &'static str> {
    validate_irreversible_quantization_scale(options.irreversible_quantization_scale)?;
    if quantize::subband_scales_all_valid(options.irreversible_quantization_subband_scales) {
        Ok(())
    } else {
        Err("irreversible quantization subband scales must be finite and greater than zero")
    }
}

fn validate_htj2k_codestream(
    codestream: &[u8],
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    reversible: bool,
) -> Result<(), &'static str> {
    let image = Image::new(codestream, &DecodeSettings::default())
        .map_err(|_| "generated HTJ2K codestream failed self-validation")?;
    let decoded = image
        .decode_native()
        .map_err(|_| "generated HTJ2K codestream failed self-validation")?;

    if decoded.width != width
        || decoded.height != height
        || decoded.bit_depth != bit_depth
        || decoded.num_components != num_components
    {
        return Err("generated HTJ2K codestream failed self-validation");
    }

    if reversible && !native_samples_equal(pixels, &decoded.data, bit_depth, signed) {
        return Err("generated HTJ2K codestream did not roundtrip");
    }

    Ok(())
}

fn native_samples_equal(expected: &[u8], actual: &[u8], bit_depth: u8, signed: bool) -> bool {
    if expected.len() != actual.len() {
        return false;
    }

    let Ok(bytes_per_sample) = raw_pixel_bytes_per_sample(bit_depth) else {
        return false;
    };
    let sample_count = expected.len() / bytes_per_sample;
    (0..sample_count).all(|sample_index| {
        decode_native_sample(expected, sample_index, bit_depth, signed)
            == decode_native_sample(actual, sample_index, bit_depth, signed)
    })
}

fn decode_native_sample(bytes: &[u8], sample_index: usize, bit_depth: u8, signed: bool) -> i64 {
    let bytes_per_sample = raw_pixel_bytes_per_sample(bit_depth).unwrap_or(1);
    let byte_offset = sample_index * bytes_per_sample;
    let raw = read_le_sample_value(
        &bytes[byte_offset..byte_offset + bytes_per_sample],
        bit_depth,
    );

    if signed {
        sign_extend_sample(raw, bit_depth)
    } else {
        raw as i64
    }
}

fn raw_pixel_bytes_per_sample(bit_depth: u8) -> Result<usize, &'static str> {
    if bit_depth == 0 || bit_depth > MAX_PART1_SAMPLE_BIT_DEPTH {
        return Err("unsupported bit depth");
    }
    Ok(usize::from(bit_depth).div_ceil(8).max(1))
}

fn read_le_sample_value(bytes: &[u8], bit_depth: u8) -> u64 {
    let mut raw = 0_u64;
    for (shift, byte) in bytes.iter().enumerate() {
        raw |= u64::from(*byte) << (shift * 8);
    }
    let mask = (1_u64 << bit_depth) - 1;
    raw & mask
}

fn sign_extend_sample(raw: u64, bit_depth: u8) -> i64 {
    let shift = 64 - u32::from(bit_depth);
    ((raw << shift) as i64) >> shift
}

fn encode_impl(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
    block_coding_mode: BlockCodingMode,
    roi_regions: &[EncodeRoiRegion],
    component_sample_info: &[EncodeComponentSampleInfo],
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    if width == 0 || height == 0 {
        return Err("invalid dimensions");
    }
    if num_components == 0 || num_components > MAX_J2K_SPEC_COMPONENTS {
        return Err("unsupported component count");
    }
    if bit_depth == 0 || bit_depth > MAX_PART1_SAMPLE_BIT_DEPTH {
        return Err("unsupported bit depth");
    }
    if options.num_layers == 0 || options.num_layers > 32 {
        return Err("unsupported quality layer count");
    }
    if options.write_ppm && options.write_ppt {
        return Err("PPM and PPT packet header markers are mutually exclusive");
    }
    if matches!(options.tile_part_packet_limit, Some(0)) {
        return Err("tile-part packet limit must be non-zero");
    }
    if !options.quality_layer_byte_targets.is_empty()
        && options.quality_layer_byte_targets.len() != usize::from(options.num_layers)
    {
        return Err("quality layer byte target count must match quality layer count");
    }
    if !options.reversible {
        validate_irreversible_quantization_profile(options)?;
    }
    validate_component_sample_info(component_sample_info, usize::from(num_components))?;

    let num_pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or("image dimensions overflow")?;
    let bytes_per_sample = raw_pixel_bytes_per_sample(bit_depth)?;
    let expected_len = num_pixels
        .checked_mul(num_components as usize)
        .and_then(|len| len.checked_mul(bytes_per_sample))
        .ok_or("image dimensions overflow")?;
    if pixels.len() < expected_len {
        return Err("pixel data too short");
    }
    let component_sampling = component_sampling_for_options(options, num_components)?;
    let high_bit_exact = bit_depth > MAX_RAW_PIXEL_ENCODE_BIT_DEPTH;
    if high_bit_exact && options.reversible {
        validate_reversible_i64_encode_options(
            options,
            block_coding_mode,
            component_sample_info,
            &component_sampling,
        )?;
    }
    if let Some((tile_width, tile_height)) = options.tile_size {
        if tile_width == 0 || tile_height == 0 {
            return Err("invalid tile dimensions");
        }
        if component_sampling
            .iter()
            .any(|sampling| *sampling != (1, 1))
        {
            return Err("multi-tile encode with component sampling is not implemented");
        }
        if tile_width < width || tile_height < height {
            return encode_multitile_impl(
                pixels,
                width,
                height,
                num_components,
                bit_depth,
                signed,
                options,
                block_coding_mode,
                roi_regions,
                component_sample_info,
                accelerator,
                tile_width,
                tile_height,
            );
        }
    }

    let profile_enabled = profile::profile_stages_enabled();
    let total_start = profile::profile_now(profile_enabled);

    let use_mct = options.use_mct && matches!(num_components, 3 | 4);
    let num_levels = options.num_decomposition_levels.min(
        // Don't decompose more than the image supports
        max_decomposition_levels(width, height),
    );
    let requested_guard_bits = if options.reversible {
        if use_mct {
            options.guard_bits.max(2)
        } else {
            options.guard_bits
        }
    } else {
        options.guard_bits.max(2)
    };
    let guard_bits = if high_bit_exact && options.reversible {
        reversible_guard_bits_for_marker_limit(bit_depth, num_levels, requested_guard_bits)?
    } else {
        requested_guard_bits
    };
    let reversible_guard_delta = guard_bits.saturating_sub(requested_guard_bits);
    let mut step_sizes = quantize::compute_step_sizes_with_irreversible_profile(
        bit_depth,
        num_levels,
        options.reversible,
        guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
    );
    if options.reversible && reversible_guard_delta != 0 {
        adjust_reversible_step_sizes_for_guard_delta(&mut step_sizes, reversible_guard_delta)?;
    }
    let quant_params: Vec<(u16, u16)> = step_sizes
        .iter()
        .map(|s| (s.exponent, s.mantissa))
        .collect();
    let mut component_step_sizes = component_step_sizes(
        component_sample_info,
        num_levels,
        options.reversible,
        guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
    );
    if options.reversible && reversible_guard_delta != 0 {
        adjust_component_step_sizes_for_guard_delta(
            &mut component_step_sizes,
            reversible_guard_delta,
        )?;
    }
    let component_quantization_step_sizes = component_step_sizes
        .iter()
        .map(|steps| {
            steps
                .iter()
                .map(|step| (step.exponent, step.mantissa))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let cb_width = 1u32 << (options.code_block_width_exp + 2);
    let cb_height = 1u32 << (options.code_block_height_exp + 2);
    let ht_target_coding_passes = ht_target_coding_passes_for_options(options);
    let precinct_exponents = precinct_exponents_for_options(options, num_levels)?;
    let max_base_bitplanes =
        max_total_bitplanes_for_components(&step_sizes, &component_step_sizes, guard_bits)?;
    let roi_plans = roi_encode_plans_for_options(
        options,
        roi_regions,
        num_components,
        width,
        height,
        &component_sampling,
        max_base_bitplanes,
        block_coding_mode,
    )?;
    let roi_component_shifts: Vec<u8> = roi_plans.iter().map(|plan| plan.shift).collect();
    let params = EncodeParams {
        width,
        height,
        tile_width: options
            .tile_size
            .map_or(width, |(tile_width, _)| tile_width),
        tile_height: options
            .tile_size
            .map_or(height, |(_, tile_height)| tile_height),
        num_components,
        bit_depth,
        signed,
        component_sample_info: component_sample_info.to_vec(),
        component_quantization_step_sizes,
        num_decomposition_levels: num_levels,
        reversible: options.reversible,
        code_block_width_exp: options.code_block_width_exp,
        code_block_height_exp: options.code_block_height_exp,
        num_layers: options.num_layers,
        use_mct,
        guard_bits,
        block_coding_mode,
        progression_order: options.progression_order,
        write_tlm: options.write_tlm,
        write_plt: options.write_plt,
        write_plm: options.write_plm,
        write_ppm: options.write_ppm,
        write_ppt: options.write_ppt,
        write_sop: options.write_sop,
        write_eph: options.write_eph,
        terminate_coding_passes: block_coding_mode == BlockCodingMode::Classic
            && options.num_layers > 1,
        component_sampling,
        roi_component_shifts: roi_component_shifts.clone(),
        precinct_exponents,
    };

    if high_bit_exact && options.reversible {
        return encode_reversible_i64_single_tile_codestream(
            pixels,
            width,
            height,
            num_pixels,
            num_components,
            bit_depth,
            signed,
            options,
            &params,
            &quant_params,
            &step_sizes,
            &roi_plans,
            use_mct,
            guard_bits,
            num_levels,
            cb_width,
            cb_height,
            ht_target_coding_passes,
            accelerator,
        );
    }

    let stage_start = profile::profile_now(profile_enabled);
    if block_coding_mode == BlockCodingMode::HighThroughput
        && component_sample_info.is_empty()
        && roi_component_shifts.iter().all(|shift| *shift == 0)
        && roi_regions.is_empty()
        && !(params.write_plt
            || params.write_plm
            || params.write_sop
            || params.write_eph
            || options.tile_part_packet_limit.is_some())
    {
        if let Some(tile_data) = accelerator.encode_htj2k_tile(J2kHtj2kTileEncodeJob {
            pixels,
            width,
            height,
            num_components,
            bit_depth,
            signed,
            num_decomposition_levels: num_levels,
            reversible: options.reversible,
            use_mct,
            guard_bits,
            code_block_width: cb_width,
            code_block_height: cb_height,
            progression_order: public_packetization_progression_order(options.progression_order),
            component_sampling: &params.component_sampling,
            quantization_steps: &quant_params,
        })? {
            let tile_body_us = profile::elapsed_us(stage_start);
            let stage_start = profile::profile_now(profile_enabled);
            let codestream = codestream_write::write_codestream(&params, &tile_data, &quant_params);
            let codestream_us = profile::elapsed_us(stage_start);
            if profile_enabled {
                profile::emit_profile_row(
                    "encode",
                    "accelerated",
                    &[
                        ("tile_body_us", tile_body_us),
                        ("codestream_us", codestream_us),
                        ("total_us", profile::elapsed_us(total_start)),
                    ],
                );
            }
            return Ok(codestream);
        }
    }

    // Step 1: Convert pixel bytes to f32 component arrays
    let stage_start = profile::profile_now(profile_enabled);
    let mut components = match accelerator.encode_deinterleave(J2kDeinterleaveToF32Job {
        pixels,
        num_pixels,
        num_components,
        bit_depth,
        signed,
    })? {
        Some(components) => {
            validate_deinterleaved_components(components, num_components, num_pixels)?
        }
        None => deinterleave_to_f32(pixels, num_pixels, num_components, bit_depth, signed),
    };
    let deinterleave_us = profile::elapsed_us(stage_start);

    // Step 2: Apply forward MCT if RGB with 3+ components
    let stage_start = profile::profile_now(profile_enabled);
    if use_mct {
        if options.reversible {
            if !try_encode_forward_rct(&mut components, accelerator)? {
                forward_mct::forward_rct(&mut components);
            }
        } else if !try_encode_forward_ict(&mut components, accelerator)? {
            forward_mct::forward_ict(&mut components);
        }
    }
    let mct_us = profile::elapsed_us(stage_start);

    // Step 3: Apply forward DWT to each component
    let stage_start = profile::profile_now(profile_enabled);
    let decompositions: Vec<DwtDecomposition> = components
        .iter()
        .map(|comp| {
            encode_forward_dwt(
                comp,
                width,
                height,
                num_levels,
                options.reversible,
                accelerator,
            )
        })
        .collect::<Result<Vec<_>, &'static str>>()?;
    validate_component_sampling_dwt_geometry(
        &decompositions,
        width,
        height,
        &params.component_sampling,
    )?;
    let dwt_us = profile::elapsed_us(stage_start);

    // Step 5: Quantize and encode code-blocks for each component
    let mut component_resolution_packets: Vec<Vec<PreparedResolutionPacket>> =
        Vec::with_capacity(num_components as usize);

    let stage_start = profile::profile_now(profile_enabled);
    for (component_idx, decomp) in decompositions
        .iter()
        .take(num_components as usize)
        .enumerate()
    {
        let component = u16::try_from(component_idx).map_err(|_| "component index exceeds u16")?;
        let component_bit_depth = component_sample_info
            .get(component_idx)
            .map_or(bit_depth, |info| info.bit_depth);
        let component_steps = component_step_sizes
            .get(component_idx)
            .map_or(step_sizes.as_slice(), Vec::as_slice);
        let roi_shift = roi_component_shifts
            .get(component_idx)
            .copied()
            .unwrap_or(0);
        let roi_plan = roi_plans
            .get(component_idx)
            .ok_or("ROI plan count does not match component count")?;
        let mut packets = Vec::with_capacity(num_levels as usize + 1);

        // LL subband (resolution 0)
        let ll_roi_scale = roi_subband_scale(num_levels, None)?;
        let ll_subband = prepare_subband(
            &decomp.ll,
            decomp.ll_width,
            decomp.ll_height,
            &component_steps[0],
            component_bit_depth,
            guard_bits,
            options.reversible,
            block_coding_mode,
            cb_width,
            cb_height,
            SubBandType::LowLow,
            roi_shift,
            &roi_plan.regions,
            ll_roi_scale,
            ht_target_coding_passes,
            accelerator,
        )?;
        packets.push(PreparedResolutionPacket {
            component,
            resolution: 0,
            precinct: 0,
            subbands: vec![ll_subband],
        });

        // Higher resolution levels
        for (level_idx, level) in decomp.levels.iter().enumerate() {
            let step_base = 1 + level_idx * 3;
            let level_roi_scale = roi_subband_scale(num_levels, Some(level_idx))?;

            // HL subband
            let hl_subband = prepare_subband(
                &level.hl,
                level.high_width,
                level.low_height,
                &component_steps[step_base],
                component_bit_depth,
                guard_bits,
                options.reversible,
                block_coding_mode,
                cb_width,
                cb_height,
                SubBandType::HighLow,
                roi_shift,
                &roi_plan.regions,
                level_roi_scale,
                ht_target_coding_passes,
                accelerator,
            )?;

            // LH subband
            let lh_subband = prepare_subband(
                &level.lh,
                level.low_width,
                level.high_height,
                &component_steps[step_base + 1],
                component_bit_depth,
                guard_bits,
                options.reversible,
                block_coding_mode,
                cb_width,
                cb_height,
                SubBandType::LowHigh,
                roi_shift,
                &roi_plan.regions,
                level_roi_scale,
                ht_target_coding_passes,
                accelerator,
            )?;

            // HH subband
            let hh_subband = prepare_subband(
                &level.hh,
                level.high_width,
                level.high_height,
                &component_steps[step_base + 2],
                component_bit_depth,
                guard_bits,
                options.reversible,
                block_coding_mode,
                cb_width,
                cb_height,
                SubBandType::HighHigh,
                roi_shift,
                &roi_plan.regions,
                level_roi_scale,
                ht_target_coding_passes,
                accelerator,
            )?;

            packets.push(PreparedResolutionPacket {
                component,
                resolution: u32::try_from(level_idx + 1)
                    .map_err(|_| "resolution index exceeds u32")?,
                precinct: 0,
                subbands: vec![hl_subband, lh_subband, hh_subband],
            });
        }

        component_resolution_packets.push(packets);
    }
    let subband_prepare_us = profile::elapsed_us(stage_start);

    let component_resolution_packets = split_component_resolution_packets_by_precinct(
        component_resolution_packets,
        width,
        height,
        num_levels,
        &params.precinct_exponents,
    )?;
    let prepared_resolution_packets =
        ordered_prepared_resolution_packets(component_resolution_packets, options)?;
    let stage_start = profile::profile_now(profile_enabled);
    let (resolution_packets, packet_descriptors, allow_packetization_accelerator) =
        if options.num_layers > 1 {
            let (resolution_packets, packet_descriptors) =
                encode_prepared_resolution_packets_layered(
                    prepared_resolution_packets,
                    options.num_layers,
                    options.progression_order,
                    &options.quality_layer_byte_targets,
                    accelerator,
                )?;
            (resolution_packets, packet_descriptors, false)
        } else {
            let packet_descriptors = packet_descriptors_for_order(
                &prepared_resolution_packets,
                1,
                options.progression_order,
            )?;
            let resolution_packets =
                encode_prepared_resolution_packets(prepared_resolution_packets, accelerator)?;
            (resolution_packets, packet_descriptors, true)
        };
    let block_encode_us = profile::elapsed_us(stage_start);

    // Step 6: Form tile bitstream (T2)
    let stage_start = profile::profile_now(profile_enabled);
    let mut resolution_packets = resolution_packets;
    let packetized_tile = packetize_resolution_packets_with_options(
        &mut resolution_packets,
        &packet_descriptors,
        options.num_layers,
        num_components,
        options.progression_order,
        packet_encode::PacketMarkerOptions {
            write_sop: params.write_sop,
            write_eph: params.write_eph,
            separate_packet_headers: params.write_ppm || params.write_ppt,
        },
        allow_packetization_accelerator,
        packetization_requires_scalar(&params, options.tile_part_packet_limit),
        accelerator,
    )?;
    let packetize_us = profile::elapsed_us(stage_start);

    // Step 7: Write codestream
    let stage_start = profile::profile_now(profile_enabled);
    let codestream = write_single_tile_packetized_codestream(
        &params,
        &packetized_tile,
        &quant_params,
        options.tile_part_packet_limit,
    )?;
    let codestream_us = profile::elapsed_us(stage_start);

    if profile_enabled {
        profile::emit_profile_row(
            "encode",
            "cpu",
            &[
                ("deinterleave_us", deinterleave_us),
                ("mct_us", mct_us),
                ("dwt_us", dwt_us),
                ("subband_prepare_us", subband_prepare_us),
                ("block_encode_us", block_encode_us),
                ("packetize_us", packetize_us),
                ("codestream_us", codestream_us),
                ("total_us", profile::elapsed_us(total_start)),
            ],
        );
    }

    Ok(codestream)
}

fn validate_reversible_i64_encode_options(
    options: &EncodeOptions,
    block_coding_mode: BlockCodingMode,
    component_sample_info: &[EncodeComponentSampleInfo],
    component_sampling: &[(u8, u8)],
) -> Result<(), &'static str> {
    if !options.reversible {
        return Err("25-38 bit encode currently requires reversible 5/3 coding");
    }
    if !matches!(
        block_coding_mode,
        BlockCodingMode::Classic | BlockCodingMode::HighThroughput
    ) {
        return Err("25-38 bit encode requires classic J2K or HTJ2K block coding");
    }
    if !component_sample_info.is_empty() {
        return Err("25-38 bit encode currently requires uniform raw-pixel component metadata");
    }
    if component_sampling
        .iter()
        .any(|sampling| *sampling != (1, 1))
    {
        return Err("25-38 bit encode currently requires full-resolution components");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_reversible_i64_single_tile_codestream(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_pixels: usize,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
    params: &EncodeParams,
    quant_params: &[(u16, u16)],
    step_sizes: &[QuantStepSize],
    roi_plans: &[ComponentRoiEncodePlan],
    use_mct: bool,
    guard_bits: u8,
    num_levels: u8,
    cb_width: u32,
    cb_height: u32,
    ht_target_coding_passes: u8,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    let max_reversible_gain = if num_levels == 0 { 0 } else { 2 };
    if u16::from(bit_depth) + max_reversible_gain > MAX_CLASSIC_REVERSIBLE_MARKER_BITPLANES {
        return Err("25-38 bit reversible encode exceeds the current no-quantization guard/exponent signaling limit");
    }

    let mut components = deinterleave_to_i64(pixels, num_pixels, num_components, bit_depth, signed);
    if use_mct {
        forward_rct_i64(&mut components);
    }

    let decompositions = components
        .iter()
        .map(|component| fdwt::forward_dwt_i64(component, width, height, num_levels))
        .collect::<Vec<_>>();

    let mut component_resolution_packets = Vec::with_capacity(num_components as usize);
    for (component_idx, decomp) in decompositions
        .iter()
        .take(num_components as usize)
        .enumerate()
    {
        let component = u16::try_from(component_idx).map_err(|_| "component index exceeds u16")?;
        let roi_shift = params
            .roi_component_shifts
            .get(component_idx)
            .copied()
            .unwrap_or(0);
        let roi_plan = roi_plans
            .get(component_idx)
            .ok_or("ROI plan count does not match component count")?;
        let mut packets = Vec::with_capacity(num_levels as usize + 1);

        let ll_roi_scale = roi_subband_scale(num_levels, None)?;
        let ll_subband = prepare_subband_i64(
            &decomp.ll,
            decomp.ll_width,
            decomp.ll_height,
            step_sizes
                .first()
                .ok_or("reversible quantization step missing")?,
            guard_bits,
            cb_width,
            cb_height,
            SubBandType::LowLow,
            roi_shift,
            &roi_plan.regions,
            ll_roi_scale,
            params.block_coding_mode,
            ht_target_coding_passes,
        )?;
        packets.push(PreparedResolutionPacket {
            component,
            resolution: 0,
            precinct: 0,
            subbands: vec![ll_subband],
        });

        for (level_idx, level) in decomp.levels.iter().enumerate() {
            let step_base = 1 + level_idx * 3;
            let level_roi_scale = roi_subband_scale(num_levels, Some(level_idx))?;
            let hl_subband = prepare_subband_i64(
                &level.hl,
                level.high_width,
                level.low_height,
                step_sizes
                    .get(step_base)
                    .ok_or("reversible quantization step missing")?,
                guard_bits,
                cb_width,
                cb_height,
                SubBandType::HighLow,
                roi_shift,
                &roi_plan.regions,
                level_roi_scale,
                params.block_coding_mode,
                ht_target_coding_passes,
            )?;
            let lh_subband = prepare_subband_i64(
                &level.lh,
                level.low_width,
                level.high_height,
                step_sizes
                    .get(step_base + 1)
                    .ok_or("reversible quantization step missing")?,
                guard_bits,
                cb_width,
                cb_height,
                SubBandType::LowHigh,
                roi_shift,
                &roi_plan.regions,
                level_roi_scale,
                params.block_coding_mode,
                ht_target_coding_passes,
            )?;
            let hh_subband = prepare_subband_i64(
                &level.hh,
                level.high_width,
                level.high_height,
                step_sizes
                    .get(step_base + 2)
                    .ok_or("reversible quantization step missing")?,
                guard_bits,
                cb_width,
                cb_height,
                SubBandType::HighHigh,
                roi_shift,
                &roi_plan.regions,
                level_roi_scale,
                params.block_coding_mode,
                ht_target_coding_passes,
            )?;

            packets.push(PreparedResolutionPacket {
                component,
                resolution: u32::try_from(level_idx + 1)
                    .map_err(|_| "resolution index exceeds u32")?,
                precinct: 0,
                subbands: vec![hl_subband, lh_subband, hh_subband],
            });
        }
        component_resolution_packets.push(packets);
    }

    encode_i64_component_resolution_packets(
        component_resolution_packets,
        width,
        height,
        num_components,
        num_levels,
        params,
        quant_params,
        options,
        accelerator,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_i64_component_resolution_packets(
    component_resolution_packets: Vec<Vec<PreparedResolutionPacket>>,
    width: u32,
    height: u32,
    num_components: u16,
    num_levels: u8,
    params: &EncodeParams,
    quant_params: &[(u16, u16)],
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    let packetized_tile = packetize_i64_component_resolution_packets(
        component_resolution_packets,
        width,
        height,
        num_components,
        num_levels,
        params,
        options,
        accelerator,
    )?;

    write_single_tile_packetized_codestream(
        params,
        &packetized_tile,
        quant_params,
        options.tile_part_packet_limit,
    )
}

#[allow(clippy::too_many_arguments)]
fn packetize_i64_component_resolution_packets(
    component_resolution_packets: Vec<Vec<PreparedResolutionPacket>>,
    width: u32,
    height: u32,
    num_components: u16,
    num_levels: u8,
    params: &EncodeParams,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<packet_encode::PacketizedTileData, &'static str> {
    let component_resolution_packets = split_component_resolution_packets_by_precinct(
        component_resolution_packets,
        width,
        height,
        num_levels,
        &params.precinct_exponents,
    )?;
    let prepared_resolution_packets =
        ordered_prepared_resolution_packets(component_resolution_packets, options)?;
    let (resolution_packets, packet_descriptors, allow_packetization_accelerator) =
        if options.num_layers > 1 {
            let (resolution_packets, packet_descriptors) =
                encode_prepared_resolution_packets_layered(
                    prepared_resolution_packets,
                    options.num_layers,
                    options.progression_order,
                    &options.quality_layer_byte_targets,
                    accelerator,
                )?;
            (resolution_packets, packet_descriptors, false)
        } else {
            let packet_descriptors = packet_descriptors_for_order(
                &prepared_resolution_packets,
                1,
                options.progression_order,
            )?;
            let resolution_packets =
                encode_prepared_resolution_packets(prepared_resolution_packets, accelerator)?;
            (resolution_packets, packet_descriptors, true)
        };

    let mut resolution_packets = resolution_packets;
    let packetized_tile = packetize_resolution_packets_with_options(
        &mut resolution_packets,
        &packet_descriptors,
        options.num_layers,
        num_components,
        options.progression_order,
        packet_encode::PacketMarkerOptions {
            write_sop: params.write_sop,
            write_eph: params.write_eph,
            separate_packet_headers: params.write_ppm || params.write_ppt,
        },
        allow_packetization_accelerator,
        packetization_requires_scalar(params, options.tile_part_packet_limit),
        accelerator,
    )?;
    Ok(packetized_tile)
}

fn deinterleave_to_i64(
    pixels: &[u8],
    num_pixels: usize,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
) -> Vec<Vec<i64>> {
    let nc = num_components as usize;
    let mut components = vec![vec![0_i64; num_pixels]; nc];
    let unsigned_offset = if signed {
        0
    } else {
        1_i64 << (u32::from(bit_depth) - 1)
    };
    let bytes_per_sample = raw_pixel_bytes_per_sample(bit_depth).unwrap_or(5);
    for (i, pixel) in pixels
        .chunks_exact(nc * bytes_per_sample)
        .take(num_pixels)
        .enumerate()
    {
        for (component_idx, component) in components.iter_mut().enumerate().take(nc) {
            let offset = component_idx * bytes_per_sample;
            let raw = read_le_sample_value(&pixel[offset..offset + bytes_per_sample], bit_depth);
            component[i] = if signed {
                sign_extend_sample(raw, bit_depth)
            } else {
                i64::try_from(raw).unwrap_or(i64::MAX) - unsigned_offset
            };
        }
    }
    components
}

fn typed_component_plane_to_i64(
    plane: &EncodeTypedComponentPlane<'_>,
    width: u32,
    height: u32,
) -> Result<Vec<i64>, &'static str> {
    let bytes_per_sample = raw_pixel_bytes_per_sample(plane.bit_depth)?;
    let sample_count = (width as usize)
        .checked_mul(height as usize)
        .ok_or("image dimensions overflow")?;
    let expected_len = sample_count
        .checked_mul(bytes_per_sample)
        .ok_or("image dimensions overflow")?;
    if plane.data.len() != expected_len {
        return Err("component plane data length mismatch");
    }
    let unsigned_offset = if plane.signed {
        0
    } else {
        1_i64 << (u32::from(plane.bit_depth) - 1)
    };
    Ok(plane
        .data
        .chunks_exact(bytes_per_sample)
        .map(|sample| {
            let raw = read_le_sample_value(sample, plane.bit_depth);
            if plane.signed {
                sign_extend_sample(raw, plane.bit_depth)
            } else {
                i64::try_from(raw).unwrap_or(i64::MAX) - unsigned_offset
            }
        })
        .collect())
}

fn forward_rct_i64(components: &mut [Vec<i64>]) {
    debug_assert!(components.len() >= 3);
    let (r_components, rest) = components.split_at_mut(1);
    let (g_components, b_components) = rest.split_at_mut(1);
    let r_components = &mut r_components[0];
    let g_components = &mut g_components[0];
    let b_components = &mut b_components[0];

    for ((r, g), b) in r_components
        .iter_mut()
        .zip(g_components.iter_mut())
        .zip(b_components.iter_mut())
    {
        let r0 = *r;
        let g0 = *g;
        let b0 = *b;
        *r = (r0 + 2 * g0 + b0).div_euclid(4);
        *g = b0 - g0;
        *b = r0 - g0;
    }
}

struct EncodedTilePart {
    tile_index: u16,
    tile_part_index: u8,
    num_tile_parts: u8,
    data: Vec<u8>,
    packet_lengths: Vec<u32>,
    packet_headers: Vec<Vec<u8>>,
}

fn split_packetized_tile_into_tile_parts(
    tile_index: u16,
    data: &[u8],
    packet_lengths: &[u32],
    packet_headers: &[Vec<u8>],
    packet_limit: Option<u16>,
) -> Result<Vec<EncodedTilePart>, &'static str> {
    if !packet_headers.is_empty() && packet_headers.len() != packet_lengths.len() {
        return Err("packet header count does not match packet length count");
    }
    let Some(packet_limit) = packet_limit else {
        return Ok(vec![EncodedTilePart {
            tile_index,
            tile_part_index: 0,
            num_tile_parts: 1,
            data: data.to_vec(),
            packet_lengths: packet_lengths.to_vec(),
            packet_headers: packet_headers.to_vec(),
        }]);
    };
    if packet_limit == 0 {
        return Err("tile-part packet limit must be non-zero");
    }
    if packet_lengths.is_empty() {
        return Ok(vec![EncodedTilePart {
            tile_index,
            tile_part_index: 0,
            num_tile_parts: 1,
            data: data.to_vec(),
            packet_lengths: Vec::new(),
            packet_headers: Vec::new(),
        }]);
    }

    let expected_len = packet_lengths.iter().try_fold(0usize, |acc, &len| {
        acc.checked_add(usize::try_from(len).map_err(|_| "packet length exceeds usize")?)
            .ok_or("packet length sum overflow")
    })?;
    if expected_len != data.len() {
        return Err("packet lengths do not match tile data length");
    }

    let packet_limit = usize::from(packet_limit);
    let num_tile_parts = packet_lengths.len().div_ceil(packet_limit);
    if num_tile_parts > usize::from(u8::MAX) {
        return Err("tile-part packet limit would emit more than 255 tile-parts");
    }
    let num_tile_parts = u8::try_from(num_tile_parts).map_err(|_| "tile-part count exceeds u8")?;

    let mut parts = Vec::with_capacity(usize::from(num_tile_parts));
    let mut data_offset = 0usize;
    for (tile_part_index, packet_chunk) in packet_lengths.chunks(packet_limit).enumerate() {
        let chunk_len = packet_chunk.iter().try_fold(0usize, |acc, &len| {
            acc.checked_add(usize::try_from(len).map_err(|_| "packet length exceeds usize")?)
                .ok_or("packet length sum overflow")
        })?;
        let end = data_offset
            .checked_add(chunk_len)
            .ok_or("packet length sum overflow")?;
        let tile_part_index =
            u8::try_from(tile_part_index).map_err(|_| "tile-part index exceeds u8")?;
        parts.push(EncodedTilePart {
            tile_index,
            tile_part_index,
            num_tile_parts,
            data: data[data_offset..end].to_vec(),
            packet_lengths: packet_chunk.to_vec(),
            packet_headers: if packet_headers.is_empty() {
                Vec::new()
            } else {
                let packet_start = tile_part_index as usize * packet_limit;
                let packet_end = packet_start + packet_chunk.len();
                packet_headers[packet_start..packet_end].to_vec()
            },
        });
        data_offset = end;
    }
    Ok(parts)
}

fn write_single_tile_packetized_codestream(
    params: &EncodeParams,
    packetized_tile: &packet_encode::PacketizedTileData,
    quant_params: &[(u16, u16)],
    tile_part_packet_limit: Option<u16>,
) -> Result<Vec<u8>, &'static str> {
    validate_packet_header_marker_payload(params, packetized_tile)?;
    let tile_parts = split_packetized_tile_into_tile_parts(
        0,
        &packetized_tile.data,
        &packetized_tile.packet_lengths,
        &packetized_tile.packet_headers,
        tile_part_packet_limit,
    )?;
    let codestream_tile_parts = tile_parts
        .iter()
        .map(|part| codestream_write::TilePartData {
            tile_index: part.tile_index,
            tile_part_index: part.tile_part_index,
            num_tile_parts: part.num_tile_parts,
            data: &part.data,
            packet_lengths: &part.packet_lengths,
            packet_headers: &part.packet_headers,
        })
        .collect::<Vec<_>>();
    Ok(codestream_write::write_codestream_tiles(
        params,
        &codestream_tile_parts,
        quant_params,
    ))
}

fn validate_packet_header_marker_payload(
    params: &EncodeParams,
    packetized_tile: &packet_encode::PacketizedTileData,
) -> Result<(), &'static str> {
    if !params.write_ppm && !params.write_ppt {
        return Ok(());
    }
    if params.write_ppm && params.write_ppt {
        return Err("PPM and PPT packet header markers are mutually exclusive");
    }
    validate_packet_header_marker_payloads(
        params.write_ppm,
        params.write_ppt,
        &[&packetized_tile.packet_headers],
    )?;
    Ok(())
}

fn validate_packet_header_marker_payloads(
    write_ppm: bool,
    write_ppt: bool,
    tile_packet_headers: &[&[Vec<u8>]],
) -> Result<(), &'static str> {
    const PACKET_HEADER_MARKER_PAYLOAD_LIMIT: usize = u16::MAX as usize - 3;
    const PPM_PACKET_HEADER_LIMIT: usize = PACKET_HEADER_MARKER_PAYLOAD_LIMIT - 2;
    const MAX_PACKET_HEADER_MARKERS: usize = u8::MAX as usize + 1;

    if !write_ppm && !write_ppt {
        return Ok(());
    }
    if write_ppm && write_ppt {
        return Err("PPM and PPT packet header markers are mutually exclusive");
    }
    if tile_packet_headers.iter().any(|headers| headers.is_empty()) {
        return Err("PPM/PPT encode requires separated packet headers");
    }
    if write_ppm {
        let mut marker_count = 0usize;
        let mut payload_len = 0usize;
        for header in tile_packet_headers
            .iter()
            .flat_map(|headers| headers.iter())
        {
            if header.len() > PPM_PACKET_HEADER_LIMIT {
                return Err("PPM packet header exceeds marker payload limit");
            }
            let entry_len = 2usize
                .checked_add(header.len())
                .ok_or("PPM marker payload length overflow")?;
            if payload_len == 0 {
                marker_count = marker_count
                    .checked_add(1)
                    .ok_or("PPM marker count overflow")?;
            } else if payload_len
                .checked_add(entry_len)
                .is_none_or(|len| len > PACKET_HEADER_MARKER_PAYLOAD_LIMIT)
            {
                marker_count = marker_count
                    .checked_add(1)
                    .ok_or("PPM marker count overflow")?;
                payload_len = 0;
            }
            payload_len = payload_len
                .checked_add(entry_len)
                .ok_or("PPM marker payload length overflow")?;
            if marker_count > MAX_PACKET_HEADER_MARKERS {
                return Err("PPM packet headers require more than 256 marker segments");
            }
        }
    }
    if write_ppt {
        for headers in tile_packet_headers {
            let payload_len = headers.iter().try_fold(0usize, |acc, header| {
                acc.checked_add(header.len())
                    .ok_or("PPT marker payload length overflow")
            })?;
            let marker_count = payload_len.div_ceil(PACKET_HEADER_MARKER_PAYLOAD_LIMIT);
            if marker_count > MAX_PACKET_HEADER_MARKERS {
                return Err("PPT packet headers require more than 256 marker segments");
            }
        }
    }
    Ok(())
}
fn encode_multitile_impl(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
    block_coding_mode: BlockCodingMode,
    roi_regions: &[EncodeRoiRegion],
    component_sample_info: &[EncodeComponentSampleInfo],
    accelerator: &mut impl J2kEncodeStageAccelerator,
    tile_width: u32,
    tile_height: u32,
) -> Result<Vec<u8>, &'static str> {
    let num_x_tiles = width.div_ceil(tile_width);
    let num_y_tiles = height.div_ceil(tile_height);
    let num_tiles = num_x_tiles
        .checked_mul(num_y_tiles)
        .ok_or("tile count overflow")?;
    if num_tiles > u32::from(u16::MAX) + 1 {
        return Err("multi-tile encode supports at most 65536 tiles");
    }

    let min_tile_width = if width.is_multiple_of(tile_width) {
        tile_width
    } else {
        width % tile_width
    };
    let min_tile_height = if height.is_multiple_of(tile_height) {
        tile_height
    } else {
        height % tile_height
    };
    let num_levels = options
        .num_decomposition_levels
        .min(max_decomposition_levels(min_tile_width, min_tile_height));
    let use_mct = options.use_mct && matches!(num_components, 3 | 4);
    let requested_guard_bits = if options.reversible {
        if use_mct {
            options.guard_bits.max(2)
        } else {
            options.guard_bits
        }
    } else {
        options.guard_bits.max(2)
    };
    let high_bit_exact = bit_depth > MAX_RAW_PIXEL_ENCODE_BIT_DEPTH;
    let guard_bits = if high_bit_exact && options.reversible {
        reversible_guard_bits_for_marker_limit(bit_depth, num_levels, requested_guard_bits)?
    } else {
        requested_guard_bits
    };
    let reversible_guard_delta = guard_bits.saturating_sub(requested_guard_bits);
    let mut step_sizes = quantize::compute_step_sizes_with_irreversible_profile(
        bit_depth,
        num_levels,
        options.reversible,
        guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
    );
    if options.reversible && reversible_guard_delta != 0 {
        adjust_reversible_step_sizes_for_guard_delta(&mut step_sizes, reversible_guard_delta)?;
    }
    let quant_params: Vec<(u16, u16)> = step_sizes
        .iter()
        .map(|s| (s.exponent, s.mantissa))
        .collect();
    let mut component_step_sizes = component_step_sizes(
        component_sample_info,
        num_levels,
        options.reversible,
        guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
    );
    if options.reversible && reversible_guard_delta != 0 {
        adjust_component_step_sizes_for_guard_delta(
            &mut component_step_sizes,
            reversible_guard_delta,
        )?;
    }
    let component_quantization_step_sizes = component_step_sizes
        .iter()
        .map(|steps| {
            steps
                .iter()
                .map(|step| (step.exponent, step.mantissa))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut child_options = options.clone();
    child_options.num_decomposition_levels = num_levels;
    child_options.tile_size = None;
    child_options.write_tlm = false;
    child_options.write_plt = options.write_plt
        || options.write_plm
        || options.write_ppm
        || options.write_ppt
        || options.tile_part_packet_limit.is_some();
    child_options.write_plm = false;
    child_options.write_ppm = options.write_ppm || options.write_ppt;
    child_options.write_ppt = false;

    let mut tile_bodies = Vec::with_capacity(num_tiles as usize);
    for tile_y in 0..num_y_tiles {
        for tile_x in 0..num_x_tiles {
            let tile_index = tile_y
                .checked_mul(num_x_tiles)
                .and_then(|base| base.checked_add(tile_x))
                .ok_or("tile index overflow")?;
            let tile_index = u16::try_from(tile_index).map_err(|_| "tile index exceeds u16")?;
            let x0 = tile_x * tile_width;
            let y0 = tile_y * tile_height;
            let actual_width = (width - x0).min(tile_width);
            let actual_height = (height - y0).min(tile_height);
            let tile_pixels = extract_interleaved_tile(
                pixels,
                width,
                x0,
                y0,
                actual_width,
                actual_height,
                num_components,
                bit_depth,
            )?;
            let tile_roi_regions =
                roi_regions_for_tile(roi_regions, x0, y0, actual_width, actual_height)?;
            let tile_codestream = encode_impl(
                &tile_pixels,
                actual_width,
                actual_height,
                num_components,
                bit_depth,
                signed,
                &child_options,
                block_coding_mode,
                &tile_roi_regions,
                component_sample_info,
                accelerator,
            )?;
            let packet_lengths = if options.write_plt
                || options.write_plm
                || options.write_ppm
                || options.write_ppt
                || options.tile_part_packet_limit.is_some()
            {
                extract_single_tile_plt_packet_lengths(&tile_codestream)?
            } else {
                Vec::new()
            };
            let packet_headers = if options.write_ppm || options.write_ppt {
                extract_single_tile_ppm_packet_headers(&tile_codestream)?
            } else {
                Vec::new()
            };
            tile_bodies.extend(split_packetized_tile_into_tile_parts(
                tile_index,
                extract_single_tile_body(&tile_codestream)?,
                &packet_lengths,
                &packet_headers,
                options.tile_part_packet_limit,
            )?);
        }
    }

    let component_sampling = component_sampling_for_options(options, num_components)?;
    let roi_plans = roi_encode_plans_for_options(
        options,
        roi_regions,
        num_components,
        width,
        height,
        &component_sampling,
        max_total_bitplanes_for_components(&step_sizes, &component_step_sizes, guard_bits)?,
        block_coding_mode,
    )?;
    let precinct_exponents = precinct_exponents_for_options(options, num_levels)?;
    let params = EncodeParams {
        width,
        height,
        tile_width,
        tile_height,
        num_components,
        bit_depth,
        signed,
        component_sample_info: component_sample_info.to_vec(),
        component_quantization_step_sizes,
        num_decomposition_levels: num_levels,
        reversible: options.reversible,
        code_block_width_exp: options.code_block_width_exp,
        code_block_height_exp: options.code_block_height_exp,
        num_layers: options.num_layers,
        use_mct,
        guard_bits,
        block_coding_mode,
        progression_order: options.progression_order,
        write_tlm: options.write_tlm,
        write_plt: options.write_plt,
        write_plm: options.write_plm,
        write_ppm: options.write_ppm,
        write_ppt: options.write_ppt,
        write_sop: options.write_sop,
        write_eph: options.write_eph,
        terminate_coding_passes: block_coding_mode == BlockCodingMode::Classic
            && options.num_layers > 1,
        component_sampling,
        roi_component_shifts: roi_plans.iter().map(|plan| plan.shift).collect(),
        precinct_exponents,
    };
    let tile_packet_headers = tile_bodies
        .iter()
        .map(|tile| tile.packet_headers.as_slice())
        .collect::<Vec<_>>();
    validate_packet_header_marker_payloads(
        params.write_ppm,
        params.write_ppt,
        &tile_packet_headers,
    )?;
    let tile_parts = tile_bodies
        .iter()
        .map(|tile| codestream_write::TilePartData {
            tile_index: tile.tile_index,
            tile_part_index: tile.tile_part_index,
            num_tile_parts: tile.num_tile_parts,
            data: &tile.data,
            packet_lengths: &tile.packet_lengths,
            packet_headers: &tile.packet_headers,
        })
        .collect::<Vec<_>>();

    Ok(codestream_write::write_codestream_tiles(
        &params,
        &tile_parts,
        &quant_params,
    ))
}

fn extract_interleaved_tile(
    pixels: &[u8],
    image_width: u32,
    x0: u32,
    y0: u32,
    tile_width: u32,
    tile_height: u32,
    num_components: u16,
    bit_depth: u8,
) -> Result<Vec<u8>, &'static str> {
    let bytes_per_sample = raw_pixel_bytes_per_sample(bit_depth)?;
    let bytes_per_pixel = usize::from(num_components)
        .checked_mul(bytes_per_sample)
        .ok_or("pixel stride overflow")?;
    let row_bytes = usize::try_from(tile_width)
        .map_err(|_| "tile width exceeds usize")?
        .checked_mul(bytes_per_pixel)
        .ok_or("tile row byte count overflow")?;
    let out_len = row_bytes
        .checked_mul(usize::try_from(tile_height).map_err(|_| "tile height exceeds usize")?)
        .ok_or("tile byte count overflow")?;
    let mut tile = Vec::with_capacity(out_len);
    let image_row_bytes = usize::try_from(image_width)
        .map_err(|_| "image width exceeds usize")?
        .checked_mul(bytes_per_pixel)
        .ok_or("image row byte count overflow")?;
    let x_byte_offset = usize::try_from(x0)
        .map_err(|_| "tile x offset exceeds usize")?
        .checked_mul(bytes_per_pixel)
        .ok_or("tile x byte offset overflow")?;

    for y in y0..y0 + tile_height {
        let row_start = usize::try_from(y)
            .map_err(|_| "tile y offset exceeds usize")?
            .checked_mul(image_row_bytes)
            .and_then(|offset| offset.checked_add(x_byte_offset))
            .ok_or("tile row offset overflow")?;
        let row_end = row_start
            .checked_add(row_bytes)
            .ok_or("tile row range overflow")?;
        tile.extend_from_slice(
            pixels
                .get(row_start..row_end)
                .ok_or("tile row range outside source pixels")?,
        );
    }

    Ok(tile)
}

fn extract_component_plane_tile(
    data: &[u8],
    image_width: u32,
    x0: u32,
    y0: u32,
    tile_width: u32,
    tile_height: u32,
    bit_depth: u8,
) -> Result<Vec<u8>, &'static str> {
    let bytes_per_sample = raw_pixel_bytes_per_sample(bit_depth)?;
    let row_bytes = usize::try_from(tile_width)
        .map_err(|_| "tile width exceeds usize")?
        .checked_mul(bytes_per_sample)
        .ok_or("tile row byte count overflow")?;
    let out_len = row_bytes
        .checked_mul(usize::try_from(tile_height).map_err(|_| "tile height exceeds usize")?)
        .ok_or("tile byte count overflow")?;
    let mut tile = Vec::with_capacity(out_len);
    let image_row_bytes = usize::try_from(image_width)
        .map_err(|_| "image width exceeds usize")?
        .checked_mul(bytes_per_sample)
        .ok_or("image row byte count overflow")?;
    let x_byte_offset = usize::try_from(x0)
        .map_err(|_| "tile x offset exceeds usize")?
        .checked_mul(bytes_per_sample)
        .ok_or("tile x byte offset overflow")?;

    for y in y0..y0 + tile_height {
        let row_start = usize::try_from(y)
            .map_err(|_| "tile y offset exceeds usize")?
            .checked_mul(image_row_bytes)
            .and_then(|offset| offset.checked_add(x_byte_offset))
            .ok_or("tile row offset overflow")?;
        let row_end = row_start
            .checked_add(row_bytes)
            .ok_or("tile row range overflow")?;
        tile.extend_from_slice(
            data.get(row_start..row_end)
                .ok_or("component plane tile row range outside source data")?,
        );
    }

    Ok(tile)
}

fn roi_regions_for_tile(
    roi_regions: &[EncodeRoiRegion],
    tile_x: u32,
    tile_y: u32,
    tile_width: u32,
    tile_height: u32,
) -> Result<Vec<EncodeRoiRegion>, &'static str> {
    let tile_x1 = tile_x
        .checked_add(tile_width)
        .ok_or("tile ROI bounds overflow")?;
    let tile_y1 = tile_y
        .checked_add(tile_height)
        .ok_or("tile ROI bounds overflow")?;
    let mut clipped = Vec::new();

    for region in roi_regions {
        let region_x1 = region
            .x
            .checked_add(region.width)
            .ok_or("ROI region bounds overflow")?;
        let region_y1 = region
            .y
            .checked_add(region.height)
            .ok_or("ROI region bounds overflow")?;
        let x0 = region.x.max(tile_x);
        let y0 = region.y.max(tile_y);
        let x1 = region_x1.min(tile_x1);
        let y1 = region_y1.min(tile_y1);
        if x0 >= x1 || y0 >= y1 {
            continue;
        }
        clipped.push(EncodeRoiRegion {
            component: region.component,
            x: x0 - tile_x,
            y: y0 - tile_y,
            width: x1 - x0,
            height: y1 - y0,
            shift: region.shift,
        });
    }

    Ok(clipped)
}

fn extract_single_tile_body(codestream: &[u8]) -> Result<&[u8], &'static str> {
    let sod = codestream
        .windows(2)
        .position(|marker| marker == [0xFF, super::codestream::markers::SOD])
        .ok_or("encoded tile codestream missing SOD")?;
    let eoc = codestream
        .windows(2)
        .rposition(|marker| marker == [0xFF, super::codestream::markers::EOC])
        .ok_or("encoded tile codestream missing EOC")?;
    if eoc < sod + 2 {
        return Err("encoded tile codestream marker order invalid");
    }
    Ok(&codestream[sod + 2..eoc])
}

fn extract_single_tile_plt_packet_lengths(codestream: &[u8]) -> Result<Vec<u32>, &'static str> {
    let sod = codestream
        .windows(2)
        .position(|marker| marker == [0xFF, super::codestream::markers::SOD])
        .ok_or("encoded tile codestream missing SOD")?;
    let mut packet_lengths = Vec::new();
    let mut offset = 0usize;

    while offset + 4 <= sod {
        if codestream[offset] == 0xFF && codestream[offset + 1] == super::codestream::markers::PLT {
            let marker_len =
                u16::from_be_bytes([codestream[offset + 2], codestream[offset + 3]]) as usize;
            if marker_len < 3 {
                return Err("encoded tile codestream has invalid PLT length");
            }
            let marker_end = offset
                .checked_add(2)
                .and_then(|value| value.checked_add(marker_len))
                .ok_or("encoded tile codestream PLT length overflow")?;
            if marker_end > sod {
                return Err("encoded tile codestream PLT extends past SOD");
            }
            let length_bytes = codestream
                .get(offset + 5..marker_end)
                .ok_or("encoded tile codestream PLT payload out of range")?;
            packet_lengths.extend(
                super::codestream::decode_packet_lengths(length_bytes)
                    .ok_or("encoded tile codestream has invalid PLT packet lengths")?,
            );
            offset = marker_end;
        } else {
            offset += 1;
        }
    }

    if packet_lengths.is_empty() {
        return Err("encoded tile codestream missing PLT packet lengths");
    }

    Ok(packet_lengths)
}

fn extract_single_tile_ppm_packet_headers(codestream: &[u8]) -> Result<Vec<Vec<u8>>, &'static str> {
    let sot = codestream
        .windows(2)
        .position(|marker| marker == [0xFF, super::codestream::markers::SOT])
        .ok_or("encoded tile codestream missing SOT")?;
    let mut packet_headers = Vec::new();
    let mut offset = 0usize;

    while offset + 4 <= sot {
        if codestream[offset] == 0xFF && codestream[offset + 1] == super::codestream::markers::PPM {
            let marker_len =
                u16::from_be_bytes([codestream[offset + 2], codestream[offset + 3]]) as usize;
            if marker_len < 3 {
                return Err("encoded tile codestream has invalid PPM length");
            }
            let marker_end = offset
                .checked_add(2)
                .and_then(|value| value.checked_add(marker_len))
                .ok_or("encoded tile codestream PPM length overflow")?;
            if marker_end > sot {
                return Err("encoded tile codestream PPM extends past SOT");
            }
            let mut payload_offset = offset + 5;
            while payload_offset < marker_end {
                let header_len_end = payload_offset
                    .checked_add(2)
                    .ok_or("encoded tile codestream PPM payload overflow")?;
                let len_bytes = codestream
                    .get(payload_offset..header_len_end)
                    .ok_or("encoded tile codestream PPM packet length truncated")?;
                let header_len = u16::from_be_bytes([len_bytes[0], len_bytes[1]]) as usize;
                let header_start = header_len_end;
                let header_end = header_start
                    .checked_add(header_len)
                    .ok_or("encoded tile codestream PPM packet header overflow")?;
                let header = codestream
                    .get(header_start..header_end)
                    .ok_or("encoded tile codestream PPM packet header truncated")?;
                packet_headers.push(header.to_vec());
                payload_offset = header_end;
            }
            offset = marker_end;
        } else {
            offset += 1;
        }
    }

    if packet_headers.is_empty() {
        return Err("encoded tile codestream missing PPM packet headers");
    }

    Ok(packet_headers)
}

fn try_encode_forward_rct(
    components: &mut [Vec<f32>],
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<bool, &'static str> {
    debug_assert!(components.len() >= 3);
    let (plane0, rest) = components.split_at_mut(1);
    let (plane1, plane2) = rest.split_at_mut(1);
    accelerator.encode_forward_rct(J2kForwardRctJob {
        plane0: &mut plane0[0],
        plane1: &mut plane1[0],
        plane2: &mut plane2[0],
    })
}

fn try_encode_forward_ict(
    components: &mut [Vec<f32>],
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<bool, &'static str> {
    debug_assert!(components.len() >= 3);
    let (plane0, rest) = components.split_at_mut(1);
    let (plane1, plane2) = rest.split_at_mut(1);
    accelerator.encode_forward_ict(J2kForwardIctJob {
        plane0: &mut plane0[0],
        plane1: &mut plane1[0],
        plane2: &mut plane2[0],
    })
}

fn encode_forward_dwt(
    component: &[f32],
    width: u32,
    height: u32,
    num_levels: u8,
    reversible: bool,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<DwtDecomposition, &'static str> {
    if reversible {
        if let Some(output) = accelerator.encode_forward_dwt53(J2kForwardDwt53Job {
            samples: component,
            width,
            height,
            num_levels,
        })? {
            return convert_forward_dwt53_output(output);
        }
    } else if let Some(output) = accelerator.encode_forward_dwt97(J2kForwardDwt97Job {
        samples: component,
        width,
        height,
        num_levels,
    })? {
        return convert_forward_dwt97_output(output);
    }

    Ok(fdwt::forward_dwt(
        component, width, height, num_levels, reversible,
    ))
}

fn convert_forward_dwt53_output(
    output: J2kForwardDwt53Output,
) -> Result<DwtDecomposition, &'static str> {
    validate_band_len(output.ll.len(), output.ll_width, output.ll_height)?;
    let mut levels = Vec::with_capacity(output.levels.len());
    for level in output.levels {
        validate_dwt53_level(&level)?;
        levels.push(fdwt::DwtLevel {
            hl: level.hl,
            lh: level.lh,
            hh: level.hh,
            low_width: level.low_width,
            low_height: level.low_height,
            high_width: level.high_width,
            high_height: level.high_height,
        });
    }
    Ok(DwtDecomposition {
        ll: output.ll,
        ll_width: output.ll_width,
        ll_height: output.ll_height,
        levels,
    })
}

fn convert_forward_dwt97_output(
    output: J2kForwardDwt97Output,
) -> Result<DwtDecomposition, &'static str> {
    validate_band_len(output.ll.len(), output.ll_width, output.ll_height)?;
    let mut levels = Vec::with_capacity(output.levels.len());
    for level in output.levels {
        validate_dwt97_level(&level)?;
        levels.push(fdwt::DwtLevel {
            hl: level.hl,
            lh: level.lh,
            hh: level.hh,
            low_width: level.low_width,
            low_height: level.low_height,
            high_width: level.high_width,
            high_height: level.high_height,
        });
    }
    Ok(DwtDecomposition {
        ll: output.ll,
        ll_width: output.ll_width,
        ll_height: output.ll_height,
        levels,
    })
}

fn validate_dwt53_level(level: &J2kForwardDwt53Level) -> Result<(), &'static str> {
    validate_band_len(level.hl.len(), level.high_width, level.low_height)?;
    validate_band_len(level.lh.len(), level.low_width, level.high_height)?;
    validate_band_len(level.hh.len(), level.high_width, level.high_height)?;
    Ok(())
}

fn validate_dwt97_level(level: &J2kForwardDwt97Level) -> Result<(), &'static str> {
    validate_band_len(level.hl.len(), level.high_width, level.low_height)?;
    validate_band_len(level.lh.len(), level.low_width, level.high_height)?;
    validate_band_len(level.hh.len(), level.high_width, level.high_height)?;
    Ok(())
}

fn validate_band_len(actual: usize, width: u32, height: u32) -> Result<(), &'static str> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .ok_or("accelerated DWT output dimensions overflow")?;
    if actual != expected {
        return Err("accelerated DWT output length mismatch");
    }
    Ok(())
}

fn validate_deinterleaved_components(
    components: Vec<Vec<f32>>,
    num_components: u16,
    num_pixels: usize,
) -> Result<Vec<Vec<f32>>, &'static str> {
    if components.len() != usize::from(num_components) {
        return Err("accelerated deinterleave component count mismatch");
    }
    if components
        .iter()
        .any(|component| component.len() != num_pixels)
    {
        return Err("accelerated deinterleave component length mismatch");
    }
    Ok(components)
}

fn component_plane_to_f32(
    data: &[u8],
    width: u32,
    height: u32,
    bit_depth: u8,
    signed: bool,
) -> Result<Vec<f32>, &'static str> {
    let sample_count = (width as usize)
        .checked_mul(height as usize)
        .ok_or("image dimensions overflow")?;
    let bytes_per_sample = raw_pixel_bytes_per_sample(bit_depth)?;
    let expected_len = sample_count
        .checked_mul(bytes_per_sample)
        .ok_or("image dimensions overflow")?;
    if data.len() != expected_len {
        return Err("component plane data length mismatch");
    }

    let unsigned_offset = if signed {
        0.0
    } else {
        (1_u64 << (u32::from(bit_depth) - 1)) as f32
    };
    Ok(data
        .chunks_exact(bytes_per_sample)
        .map(|sample| {
            let raw = read_le_sample_value(sample, bit_depth);
            if signed {
                sign_extend_sample(raw, bit_depth) as f32
            } else {
                raw as f32 - unsigned_offset
            }
        })
        .collect())
}

fn forward_dwt53_output_from_decomposition(
    decomposition: DwtDecomposition,
) -> J2kForwardDwt53Output {
    J2kForwardDwt53Output {
        ll: decomposition.ll,
        ll_width: decomposition.ll_width,
        ll_height: decomposition.ll_height,
        levels: decomposition
            .levels
            .into_iter()
            .map(|level| {
                let width = level.low_width + level.high_width;
                let height = level.low_height + level.high_height;
                J2kForwardDwt53Level {
                    hl: level.hl,
                    lh: level.lh,
                    hh: level.hh,
                    width,
                    height,
                    low_width: level.low_width,
                    low_height: level.low_height,
                    high_width: level.high_width,
                    high_height: level.high_height,
                }
            })
            .collect(),
    }
}

fn validate_component_sampling_dwt_geometry(
    decompositions: &[DwtDecomposition],
    reference_width: u32,
    reference_height: u32,
    component_sampling: &[(u8, u8)],
) -> Result<(), &'static str> {
    if decompositions.len() != component_sampling.len() {
        return Err("component sampling count does not match component count");
    }
    for (decomposition, &(x_rsiz, y_rsiz)) in decompositions.iter().zip(component_sampling) {
        let expected_width = reference_width.div_ceil(u32::from(x_rsiz.max(1)));
        let expected_height = reference_height.div_ceil(u32::from(y_rsiz.max(1)));
        if dwt_decomposition_dimensions(decomposition) != (expected_width, expected_height) {
            return Err("component sampling requires component-sized DWT geometry");
        }
    }
    Ok(())
}

fn dwt_decomposition_dimensions(decomposition: &DwtDecomposition) -> (u32, u32) {
    decomposition
        .levels
        .last()
        .map_or((decomposition.ll_width, decomposition.ll_height), |level| {
            (
                level.low_width + level.high_width,
                level.low_height + level.high_height,
            )
        })
}

#[derive(Clone, Debug, Default)]
struct ComponentRoiEncodePlan {
    shift: u8,
    regions: Vec<ComponentRoiEncodeRegion>,
}

#[derive(Clone, Copy, Debug)]
struct ComponentRoiEncodeRegion {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

fn component_sampling_for_options(
    options: &EncodeOptions,
    num_components: u16,
) -> Result<Vec<(u8, u8)>, &'static str> {
    match &options.component_sampling {
        Some(component_sampling) => {
            if component_sampling.len() != usize::from(num_components) {
                return Err("component sampling count does not match component count");
            }
            if component_sampling
                .iter()
                .any(|&(x_rsiz, y_rsiz)| x_rsiz == 0 || y_rsiz == 0)
            {
                return Err("component sampling factors must be non-zero");
            }
            Ok(component_sampling.clone())
        }
        None => Ok(vec![(1, 1); usize::from(num_components)]),
    }
}

fn roi_encode_plans_for_options(
    options: &EncodeOptions,
    roi_regions: &[EncodeRoiRegion],
    num_components: u16,
    width: u32,
    height: u32,
    component_sampling: &[(u8, u8)],
    base_bitplanes: u8,
    block_coding_mode: BlockCodingMode,
) -> Result<Vec<ComponentRoiEncodePlan>, &'static str> {
    let whole_component_shifts = roi_component_shifts_for_options(
        options,
        num_components,
        base_bitplanes,
        block_coding_mode,
    )?;
    let mut plans = whole_component_shifts
        .iter()
        .map(|&shift| ComponentRoiEncodePlan {
            shift,
            regions: Vec::new(),
        })
        .collect::<Vec<_>>();

    for region in roi_regions {
        if region.component >= num_components {
            return Err("ROI region component index out of range");
        }
        if region.width == 0 || region.height == 0 {
            return Err("ROI region dimensions must be non-zero");
        }
        if region.shift == 0 {
            return Err("ROI region maxshift must be non-zero");
        }

        let x1 = region
            .x
            .checked_add(region.width)
            .ok_or("ROI region bounds overflow")?;
        let y1 = region
            .y
            .checked_add(region.height)
            .ok_or("ROI region bounds overflow")?;
        if region.x >= width || region.y >= height || x1 > width || y1 > height {
            return Err("ROI region must be inside image bounds");
        }

        let component_idx = usize::from(region.component);
        if whole_component_shifts[component_idx] != 0 {
            return Err("ROI region cannot be combined with whole-component ROI shift");
        }
        if region.shift < base_bitplanes {
            return Err("ROI region maxshift must cover background bitplanes");
        }
        validate_roi_shift(region.shift, base_bitplanes, block_coding_mode)?;

        let plan = &mut plans[component_idx];
        if plan.shift == 0 {
            plan.shift = region.shift;
        } else if plan.shift != region.shift {
            return Err("ROI regions for one component must use one maxshift");
        }

        let &(x_rsiz, y_rsiz) = component_sampling
            .get(component_idx)
            .ok_or("component sampling count does not match component count")?;
        let component_width = width.div_ceil(u32::from(x_rsiz));
        let component_height = height.div_ceil(u32::from(y_rsiz));
        let component_x0 = region.x / u32::from(x_rsiz);
        let component_y0 = region.y / u32::from(y_rsiz);
        let component_x1 = x1.div_ceil(u32::from(x_rsiz)).min(component_width);
        let component_y1 = y1.div_ceil(u32::from(y_rsiz)).min(component_height);
        if component_x0 >= component_x1 || component_y0 >= component_y1 {
            return Err("ROI region does not intersect component grid");
        }
        plan.regions.push(ComponentRoiEncodeRegion {
            x: component_x0,
            y: component_y0,
            width: component_x1 - component_x0,
            height: component_y1 - component_y0,
        });
    }

    Ok(plans)
}

fn roi_component_shifts_for_options(
    options: &EncodeOptions,
    num_components: u16,
    base_bitplanes: u8,
    block_coding_mode: BlockCodingMode,
) -> Result<Vec<u8>, &'static str> {
    if options.roi_component_shifts.is_empty() {
        return Ok(vec![0; usize::from(num_components)]);
    }
    if options.roi_component_shifts.len() != usize::from(num_components) {
        return Err("ROI component shift count does not match component count");
    }
    let max_bitplanes = max_roi_coded_bitplanes(block_coding_mode);
    for &shift in &options.roi_component_shifts {
        validate_roi_shift_for_max(shift, base_bitplanes, max_bitplanes)?;
    }
    Ok(options.roi_component_shifts.clone())
}

fn validate_roi_shift(
    shift: u8,
    base_bitplanes: u8,
    block_coding_mode: BlockCodingMode,
) -> Result<(), &'static str> {
    let max_bitplanes = max_roi_coded_bitplanes(block_coding_mode);
    validate_roi_shift_for_max(shift, base_bitplanes, max_bitplanes)
}

fn max_roi_coded_bitplanes(block_coding_mode: BlockCodingMode) -> u8 {
    match block_coding_mode {
        BlockCodingMode::Classic => MAX_CLASSIC_ROI_CODED_BITPLANES,
        BlockCodingMode::HighThroughput => MAX_HT_ROI_CODED_BITPLANES,
    }
}

fn validate_roi_shift_for_max(
    shift: u8,
    base_bitplanes: u8,
    max_bitplanes: u8,
) -> Result<(), &'static str> {
    if base_bitplanes
        .checked_add(shift)
        .is_none_or(|bitplanes| bitplanes > max_bitplanes)
    {
        return Err("ROI maxshift exceeds supported coded bitplane count");
    }
    Ok(())
}

fn roi_subband_scale(num_levels: u8, level_idx: Option<usize>) -> Result<u32, &'static str> {
    let shift = match level_idx {
        Some(level_idx) => usize::from(num_levels)
            .checked_sub(level_idx)
            .ok_or("ROI subband level exceeds decomposition level count")?,
        None => usize::from(num_levels),
    };
    if shift >= u32::BITS as usize {
        return Err("ROI subband scale exceeds supported coordinate range");
    }
    Ok(1_u32 << shift)
}

fn max_total_bitplanes(step_sizes: &[QuantStepSize], guard_bits: u8) -> Result<u8, &'static str> {
    step_sizes
        .iter()
        .map(|step_size| {
            debug_assert!(step_size.exponent <= u16::from(u8::MAX));
            guard_bits
                .checked_add(
                    u8::try_from(step_size.exponent)
                        .map_err(|_| "quantization exponent exceeds supported bitplane count")?,
                )
                .and_then(|value| value.checked_sub(1))
                .ok_or("quantization bitplane count underflows")
        })
        .max()
        .unwrap_or(Ok(0))
}

fn max_total_bitplanes_for_components(
    default_step_sizes: &[QuantStepSize],
    component_step_sizes: &[Vec<QuantStepSize>],
    guard_bits: u8,
) -> Result<u8, &'static str> {
    let default = max_total_bitplanes(default_step_sizes, guard_bits)?;
    component_step_sizes
        .iter()
        .try_fold(default, |max_bitplanes, step_sizes| {
            Ok(max_bitplanes.max(max_total_bitplanes(step_sizes, guard_bits)?))
        })
}

fn validate_component_sample_info(
    component_sample_info: &[EncodeComponentSampleInfo],
    num_components: usize,
) -> Result<(), &'static str> {
    if component_sample_info.is_empty() {
        return Ok(());
    }
    if component_sample_info.len() != num_components {
        return Err("component sample metadata count does not match component count");
    }
    if component_sample_info
        .iter()
        .any(|info| info.bit_depth == 0 || info.bit_depth > MAX_RAW_PIXEL_ENCODE_BIT_DEPTH)
    {
        return Err("unsupported bit depth");
    }
    Ok(())
}

fn component_step_sizes(
    component_sample_info: &[EncodeComponentSampleInfo],
    num_levels: u8,
    reversible: bool,
    guard_bits: u8,
    quantization_scale: f32,
    subband_scales: IrreversibleQuantizationSubbandScales,
) -> Vec<Vec<QuantStepSize>> {
    component_sample_info
        .iter()
        .map(|info| {
            quantize::compute_step_sizes_with_irreversible_profile(
                info.bit_depth,
                num_levels,
                reversible,
                guard_bits,
                quantization_scale,
                subband_scales,
            )
        })
        .collect()
}

fn reversible_guard_bits_for_marker_limit(
    bit_depth: u8,
    num_levels: u8,
    requested_guard_bits: u8,
) -> Result<u8, &'static str> {
    if requested_guard_bits > MAX_REVERSIBLE_NO_QUANT_GUARD_BITS {
        return Err("reversible guard bits exceed the Part 1 marker field");
    }
    let max_reversible_gain = if num_levels == 0 { 0 } else { 2 };
    let requested_bitplanes = u16::from(requested_guard_bits)
        .checked_add(u16::from(bit_depth))
        .and_then(|value| value.checked_add(max_reversible_gain))
        .and_then(|value| value.checked_sub(1))
        .ok_or("reversible no-quantization bitplane count underflows")?;
    if requested_bitplanes > MAX_CLASSIC_REVERSIBLE_MARKER_BITPLANES {
        return Err("25-38 bit reversible encode exceeds the current no-quantization guard/exponent signaling limit");
    }
    let min_guard_bits = requested_bitplanes.saturating_sub(MAX_REVERSIBLE_NO_QUANT_EXPONENT - 1);
    let guard_bits = requested_guard_bits
        .max(u8::try_from(min_guard_bits).map_err(|_| "reversible guard bits exceed u8")?);
    if guard_bits > MAX_REVERSIBLE_NO_QUANT_GUARD_BITS {
        return Err("reversible guard bits exceed the Part 1 marker field");
    }
    Ok(guard_bits)
}

fn adjust_component_step_sizes_for_guard_delta(
    component_step_sizes: &mut [Vec<QuantStepSize>],
    guard_delta: u8,
) -> Result<(), &'static str> {
    for step_sizes in component_step_sizes {
        adjust_reversible_step_sizes_for_guard_delta(step_sizes, guard_delta)?;
    }
    Ok(())
}

fn adjust_reversible_step_sizes_for_guard_delta(
    step_sizes: &mut [QuantStepSize],
    guard_delta: u8,
) -> Result<(), &'static str> {
    let guard_delta = u16::from(guard_delta);
    for step in step_sizes {
        step.exponent = step
            .exponent
            .checked_sub(guard_delta)
            .ok_or("reversible no-quantization exponent underflows guard-bit adjustment")?;
        if step.exponent > MAX_REVERSIBLE_NO_QUANT_EXPONENT {
            return Err("reversible no-quantization exponent exceeds the Part 1 marker field");
        }
    }
    Ok(())
}

fn precinct_exponents_for_options(
    options: &EncodeOptions,
    num_decomposition_levels: u8,
) -> Result<Vec<(u8, u8)>, &'static str> {
    if options.precinct_exponents.is_empty() {
        return Ok(Vec::new());
    }

    let expected = usize::from(num_decomposition_levels) + 1;
    if options.precinct_exponents.len() != expected {
        return Err("precinct exponent count must match resolution level count");
    }
    if options
        .precinct_exponents
        .iter()
        .any(|&(ppx, ppy)| ppx > 15 || ppy > 15)
    {
        return Err("precinct exponents must fit in COD marker nybbles");
    }
    let code_block_width_exp = options.code_block_width_exp + 2;
    let code_block_height_exp = options.code_block_height_exp + 2;
    for (resolution, &(ppx, ppy)) in options.precinct_exponents.iter().enumerate() {
        let min_ppx = if resolution == 0 {
            code_block_width_exp
        } else {
            code_block_width_exp + 1
        };
        let min_ppy = if resolution == 0 {
            code_block_height_exp
        } else {
            code_block_height_exp + 1
        };
        if ppx < min_ppx || ppy < min_ppy {
            return Err("precinct exponents must not reduce encoder code-block dimensions");
        }
    }
    Ok(options.precinct_exponents.clone())
}

fn count_code_blocks(resolution_packets: &[ResolutionPacket]) -> Result<u32, &'static str> {
    let count = resolution_packets
        .iter()
        .flat_map(|resolution| resolution.subbands.iter())
        .try_fold(0usize, |acc, subband| {
            acc.checked_add(subband.code_blocks.len())
                .ok_or("packetization code-block count overflow")
        })?;
    u32::try_from(count).map_err(|_| "packetization code-block count exceeds u32")
}

fn count_compact_code_blocks(
    resolution_packets: &[PreparedCompactResolutionPacket<'_>],
) -> Result<u32, &'static str> {
    let count = resolution_packets
        .iter()
        .flat_map(|resolution| resolution.subbands.iter())
        .try_fold(0usize, |acc, subband| {
            acc.checked_add(subband.code_blocks.len())
                .ok_or("packetization code-block count overflow")
        })?;
    u32::try_from(count).map_err(|_| "packetization code-block count exceeds u32")
}

fn split_component_resolution_packets_by_precinct(
    component_resolution_packets: Vec<Vec<PreparedResolutionPacket>>,
    width: u32,
    height: u32,
    num_decomposition_levels: u8,
    precinct_exponents: &[(u8, u8)],
) -> Result<Vec<Vec<PreparedResolutionPacket>>, &'static str> {
    if precinct_exponents.is_empty() {
        return Ok(component_resolution_packets);
    }

    component_resolution_packets
        .into_iter()
        .map(|component_packets| {
            let mut split_packets = Vec::new();
            for packet in component_packets {
                split_packets.extend(split_prepared_resolution_packet_by_precinct(
                    packet,
                    width,
                    height,
                    num_decomposition_levels,
                    precinct_exponents,
                )?);
            }
            Ok(split_packets)
        })
        .collect()
}

fn split_prepared_resolution_packet_by_precinct(
    packet: PreparedResolutionPacket,
    width: u32,
    height: u32,
    num_decomposition_levels: u8,
    precinct_exponents: &[(u8, u8)],
) -> Result<Vec<PreparedResolutionPacket>, &'static str> {
    let resolution =
        usize::try_from(packet.resolution).map_err(|_| "resolution index exceeds usize")?;
    let &(ppx, ppy) = precinct_exponents
        .get(resolution)
        .ok_or("missing precinct exponents for resolution")?;
    let (precincts_x, precincts_y) = resolution_precinct_grid(
        width,
        height,
        num_decomposition_levels,
        packet.resolution,
        ppx,
        ppy,
    )?;
    let packet_count = (precincts_x as usize)
        .checked_mul(precincts_y as usize)
        .ok_or("precinct packet count overflow")?;
    let component = packet.component;
    let resolution = packet.resolution;
    let subbands = packet.subbands;
    let mut packets = Vec::with_capacity(packet_count);

    for precinct_y in 0..precincts_y {
        for precinct_x in 0..precincts_x {
            let precinct = u64::from(precinct_y)
                .checked_mul(u64::from(precincts_x))
                .and_then(|value| value.checked_add(u64::from(precinct_x)))
                .ok_or("precinct index overflow")?;
            let split_subbands = subbands
                .iter()
                .map(|subband| {
                    split_prepared_subband_by_precinct(
                        subband, resolution, ppx, ppy, precinct_x, precinct_y,
                    )
                })
                .collect::<Result<Vec<_>, &'static str>>()?;
            packets.push(PreparedResolutionPacket {
                component,
                resolution,
                precinct,
                subbands: split_subbands,
            });
        }
    }

    Ok(packets)
}

fn resolution_precinct_grid(
    width: u32,
    height: u32,
    num_decomposition_levels: u8,
    resolution: u32,
    ppx: u8,
    ppy: u8,
) -> Result<(u32, u32), &'static str> {
    let resolution_shift = u32::from(num_decomposition_levels)
        .checked_sub(resolution)
        .ok_or("resolution exceeds decomposition level count")?;
    let resolution_scale = pow2_u32(resolution_shift)?;
    let resolution_width = width.div_ceil(resolution_scale);
    let resolution_height = height.div_ceil(resolution_scale);
    let precinct_width = pow2_u32(u32::from(ppx))?;
    let precinct_height = pow2_u32(u32::from(ppy))?;

    Ok((
        if resolution_width == 0 {
            0
        } else {
            resolution_width.div_ceil(precinct_width)
        },
        if resolution_height == 0 {
            0
        } else {
            resolution_height.div_ceil(precinct_height)
        },
    ))
}

fn split_prepared_subband_by_precinct(
    subband: &PreparedEncodeSubband,
    resolution: u32,
    ppx: u8,
    ppy: u8,
    precinct_x: u32,
    precinct_y: u32,
) -> Result<PreparedEncodeSubband, &'static str> {
    if subband.code_blocks.is_empty() || subband.width == 0 || subband.height == 0 {
        return Ok(empty_prepared_subband_precinct(subband));
    }

    let subband_ppx = if resolution > 0 {
        ppx.checked_sub(1)
            .ok_or("nonzero resolution precinct exponent underflow")?
    } else {
        ppx
    };
    let subband_ppy = if resolution > 0 {
        ppy.checked_sub(1)
            .ok_or("nonzero resolution precinct exponent underflow")?
    } else {
        ppy
    };
    let precinct_width = pow2_u32(u32::from(subband_ppx))?;
    let precinct_height = pow2_u32(u32::from(subband_ppy))?;
    let precinct_x0 = precinct_x
        .checked_mul(precinct_width)
        .ok_or("precinct x coordinate overflow")?;
    let precinct_y0 = precinct_y
        .checked_mul(precinct_height)
        .ok_or("precinct y coordinate overflow")?;
    let x0 = precinct_x0.min(subband.width);
    let y0 = precinct_y0.min(subband.height);
    let x1 = precinct_x0
        .checked_add(precinct_width)
        .ok_or("precinct x extent overflow")?
        .min(subband.width);
    let y1 = precinct_y0
        .checked_add(precinct_height)
        .ok_or("precinct y extent overflow")?
        .min(subband.height);

    if x0 >= x1 || y0 >= y1 {
        return Ok(empty_prepared_subband_precinct(subband));
    }

    let cb_width = subband.code_block_width;
    let cb_height = subband.code_block_height;
    if cb_width == 0 || cb_height == 0 {
        return Ok(empty_prepared_subband_precinct(subband));
    }

    let cb_x0 = (x0 / cb_width) * cb_width;
    let cb_y0 = (y0 / cb_height) * cb_height;
    let cb_x1 = x1.div_ceil(cb_width) * cb_width;
    let cb_y1 = y1.div_ceil(cb_height) * cb_height;
    let cbx_start = cb_x0 / cb_width;
    let cby_start = cb_y0 / cb_height;
    let cbx_end = cb_x1 / cb_width;
    let cby_end = cb_y1 / cb_height;
    let num_cbs_x = cbx_end.saturating_sub(cbx_start);
    let num_cbs_y = cby_end.saturating_sub(cby_start);
    let mut indices = Vec::with_capacity((num_cbs_x as usize).saturating_mul(num_cbs_y as usize));

    for cby in cby_start..cby_end {
        for cbx in cbx_start..cbx_end {
            let index = cby
                .checked_mul(subband.num_cbs_x)
                .and_then(|value| value.checked_add(cbx))
                .ok_or("precinct code-block index overflow")?;
            indices.push(usize::try_from(index).map_err(|_| "code-block index exceeds usize")?);
        }
    }

    let code_blocks = indices
        .iter()
        .map(|&idx| {
            subband
                .code_blocks
                .get(idx)
                .cloned()
                .ok_or("precinct code-block index out of range")
        })
        .collect::<Result<Vec<_>, &'static str>>()?;
    let preencoded_ht_code_blocks = subband
        .preencoded_ht_code_blocks
        .as_ref()
        .map(|blocks| {
            indices
                .iter()
                .map(|&idx| {
                    blocks
                        .get(idx)
                        .cloned()
                        .ok_or("precinct preencoded code-block index out of range")
                })
                .collect::<Result<Vec<_>, &'static str>>()
        })
        .transpose()?;

    Ok(PreparedEncodeSubband {
        code_blocks,
        preencoded_ht_code_blocks,
        num_cbs_x,
        num_cbs_y,
        code_block_width: cb_width,
        code_block_height: cb_height,
        width: x1 - x0,
        height: y1 - y0,
        sub_band_type: subband.sub_band_type,
        total_bitplanes: subband.total_bitplanes,
        block_coding_mode: subband.block_coding_mode,
        ht_target_coding_passes: subband.ht_target_coding_passes,
    })
}

fn empty_prepared_subband_precinct(subband: &PreparedEncodeSubband) -> PreparedEncodeSubband {
    PreparedEncodeSubband {
        code_blocks: Vec::new(),
        preencoded_ht_code_blocks: subband
            .preencoded_ht_code_blocks
            .as_ref()
            .map(|_| Vec::new()),
        num_cbs_x: 0,
        num_cbs_y: 0,
        code_block_width: subband.code_block_width,
        code_block_height: subband.code_block_height,
        width: 0,
        height: 0,
        sub_band_type: subband.sub_band_type,
        total_bitplanes: subband.total_bitplanes,
        block_coding_mode: subband.block_coding_mode,
        ht_target_coding_passes: subband.ht_target_coding_passes,
    }
}

fn pow2_u32(exponent: u32) -> Result<u32, &'static str> {
    1_u32
        .checked_shl(exponent)
        .ok_or("precinct exponent exceeds u32 shift width")
}

fn packet_descriptors_for_order(
    packets: &[PreparedResolutionPacket],
    num_layers: u8,
    progression_order: EncodeProgressionOrder,
) -> Result<Vec<J2kPacketizationPacketDescriptor>, &'static str> {
    if num_layers != 1 {
        return Err("encode currently prepares one packet contribution layer");
    }
    let mut descriptors = packets
        .iter()
        .enumerate()
        .map(|(packet_index, packet)| {
            Ok(J2kPacketizationPacketDescriptor {
                packet_index: u32::try_from(packet_index)
                    .map_err(|_| "packet descriptor index exceeds u32")?,
                state_index: u32::try_from(packet_index)
                    .map_err(|_| "packet descriptor state index exceeds u32")?,
                layer: 0,
                resolution: packet.resolution,
                component: packet.component,
                precinct: packet.precinct,
            })
        })
        .collect::<Result<Vec<_>, &'static str>>()?;
    sort_packet_descriptors_for_progression(&mut descriptors, progression_order);
    Ok(descriptors)
}

fn packet_descriptors_for_compact_order(
    packets: &[PreparedCompactResolutionPacket<'_>],
    num_layers: u8,
    progression_order: EncodeProgressionOrder,
) -> Result<Vec<J2kPacketizationPacketDescriptor>, &'static str> {
    if num_layers != 1 {
        return Err("encode currently prepares one packet contribution layer");
    }
    let mut descriptors = packets
        .iter()
        .enumerate()
        .map(|(packet_index, packet)| {
            Ok(J2kPacketizationPacketDescriptor {
                packet_index: u32::try_from(packet_index)
                    .map_err(|_| "packet descriptor index exceeds u32")?,
                state_index: u32::try_from(packet_index)
                    .map_err(|_| "packet descriptor state index exceeds u32")?,
                layer: 0,
                resolution: packet.resolution,
                component: packet.component,
                precinct: packet.precinct,
            })
        })
        .collect::<Result<Vec<_>, &'static str>>()?;
    sort_packet_descriptors_for_progression(&mut descriptors, progression_order);
    Ok(descriptors)
}

fn ordered_prepared_resolution_packets(
    component_resolution_packets: Vec<Vec<PreparedResolutionPacket>>,
    options: &EncodeOptions,
) -> Result<Vec<PreparedResolutionPacket>, &'static str> {
    match options.progression_order {
        EncodeProgressionOrder::Lrcp
        | EncodeProgressionOrder::Rlcp
        | EncodeProgressionOrder::Rpcl => {
            lrcp_ordered_prepared_resolution_packets(component_resolution_packets)
        }
        EncodeProgressionOrder::Pcrl | EncodeProgressionOrder::Cprl => {
            component_ordered_prepared_resolution_packets(component_resolution_packets)
        }
    }
}

fn ordered_prepared_compact_resolution_packets<'a>(
    component_resolution_packets: Vec<Vec<PreparedCompactResolutionPacket<'a>>>,
    options: &EncodeOptions,
) -> Result<Vec<PreparedCompactResolutionPacket<'a>>, &'static str> {
    match options.progression_order {
        EncodeProgressionOrder::Lrcp
        | EncodeProgressionOrder::Rlcp
        | EncodeProgressionOrder::Rpcl => {
            lrcp_ordered_prepared_compact_resolution_packets(component_resolution_packets)
        }
        EncodeProgressionOrder::Pcrl | EncodeProgressionOrder::Cprl => {
            component_ordered_prepared_compact_resolution_packets(component_resolution_packets)
        }
    }
}

fn lrcp_ordered_prepared_resolution_packets(
    component_resolution_packets: Vec<Vec<PreparedResolutionPacket>>,
) -> Result<Vec<PreparedResolutionPacket>, &'static str> {
    let resolution_count = component_resolution_packets
        .first()
        .map_or(0usize, alloc::vec::Vec::len);
    let mut component_iters: Vec<_> = component_resolution_packets
        .into_iter()
        .map(alloc::vec::Vec::into_iter)
        .collect();
    let mut resolution_packets =
        Vec::with_capacity(resolution_count.saturating_mul(component_iters.len()));

    for _resolution in 0..resolution_count {
        for component in &mut component_iters {
            resolution_packets.push(
                component
                    .next()
                    .ok_or("component packet resolution count mismatch")?,
            );
        }
    }

    if component_iters
        .iter_mut()
        .any(|component| component.next().is_some())
    {
        return Err("component packet resolution count mismatch");
    }

    Ok(resolution_packets)
}

fn lrcp_ordered_prepared_compact_resolution_packets<'a>(
    component_resolution_packets: Vec<Vec<PreparedCompactResolutionPacket<'a>>>,
) -> Result<Vec<PreparedCompactResolutionPacket<'a>>, &'static str> {
    let resolution_count = component_resolution_packets
        .first()
        .map_or(0usize, alloc::vec::Vec::len);
    let mut component_iters: Vec<_> = component_resolution_packets
        .into_iter()
        .map(alloc::vec::Vec::into_iter)
        .collect();
    let mut resolution_packets =
        Vec::with_capacity(resolution_count.saturating_mul(component_iters.len()));

    for _resolution in 0..resolution_count {
        for component in &mut component_iters {
            resolution_packets.push(
                component
                    .next()
                    .ok_or("component packet resolution count mismatch")?,
            );
        }
    }

    if component_iters
        .iter_mut()
        .any(|component| component.next().is_some())
    {
        return Err("component packet resolution count mismatch");
    }

    Ok(resolution_packets)
}

fn component_ordered_prepared_resolution_packets(
    component_resolution_packets: Vec<Vec<PreparedResolutionPacket>>,
) -> Result<Vec<PreparedResolutionPacket>, &'static str> {
    let resolution_count = component_resolution_packets
        .first()
        .map_or(0usize, alloc::vec::Vec::len);
    let mut resolution_packets =
        Vec::with_capacity(resolution_count.saturating_mul(component_resolution_packets.len()));

    for component in component_resolution_packets {
        if component.len() != resolution_count {
            return Err("component packet resolution count mismatch");
        }
        resolution_packets.extend(component);
    }

    Ok(resolution_packets)
}

fn component_ordered_prepared_compact_resolution_packets<'a>(
    component_resolution_packets: Vec<Vec<PreparedCompactResolutionPacket<'a>>>,
) -> Result<Vec<PreparedCompactResolutionPacket<'a>>, &'static str> {
    let resolution_count = component_resolution_packets
        .first()
        .map_or(0usize, alloc::vec::Vec::len);
    let mut resolution_packets =
        Vec::with_capacity(resolution_count.saturating_mul(component_resolution_packets.len()));

    for component in component_resolution_packets {
        if component.len() != resolution_count {
            return Err("component packet resolution count mismatch");
        }
        resolution_packets.extend(component);
    }

    Ok(resolution_packets)
}

fn public_packetization_progression_order(
    progression_order: EncodeProgressionOrder,
) -> crate::J2kPacketizationProgressionOrder {
    match progression_order {
        EncodeProgressionOrder::Lrcp => crate::J2kPacketizationProgressionOrder::Lrcp,
        EncodeProgressionOrder::Rlcp => crate::J2kPacketizationProgressionOrder::Rlcp,
        EncodeProgressionOrder::Rpcl => crate::J2kPacketizationProgressionOrder::Rpcl,
        EncodeProgressionOrder::Pcrl => crate::J2kPacketizationProgressionOrder::Pcrl,
        EncodeProgressionOrder::Cprl => crate::J2kPacketizationProgressionOrder::Cprl,
    }
}

fn scalar_packet_descriptors(
    descriptors: &[J2kPacketizationPacketDescriptor],
) -> Vec<packet_encode::PacketDescriptor> {
    descriptors
        .iter()
        .map(|descriptor| packet_encode::PacketDescriptor {
            packet_index: descriptor.packet_index,
            state_index: descriptor.state_index,
            layer: descriptor.layer,
            resolution: descriptor.resolution,
            component: descriptor.component,
            precinct: descriptor.precinct,
        })
        .collect()
}

fn public_packetization_resolutions(
    resolution_packets: &[ResolutionPacket],
) -> Vec<J2kPacketizationResolution<'_>> {
    resolution_packets
        .iter()
        .map(|resolution| J2kPacketizationResolution {
            subbands: resolution
                .subbands
                .iter()
                .map(|subband| J2kPacketizationSubband {
                    code_blocks: subband
                        .code_blocks
                        .iter()
                        .map(|code_block| J2kPacketizationCodeBlock {
                            data: &code_block.data,
                            ht_cleanup_length: code_block.ht_cleanup_length,
                            ht_refinement_length: code_block.ht_refinement_length,
                            num_coding_passes: code_block.num_coding_passes,
                            num_zero_bitplanes: code_block.num_zero_bitplanes,
                            previously_included: code_block.previously_included,
                            l_block: code_block.l_block,
                            block_coding_mode: public_packetization_block_coding_mode(
                                code_block.block_coding_mode,
                            ),
                        })
                        .collect(),
                    num_cbs_x: subband.num_cbs_x,
                    num_cbs_y: subband.num_cbs_y,
                })
                .collect(),
        })
        .collect()
}

fn packetize_resolution_packets_with_options(
    resolution_packets: &mut [ResolutionPacket],
    packet_descriptors: &[J2kPacketizationPacketDescriptor],
    num_layers: u8,
    num_components: u16,
    progression_order: EncodeProgressionOrder,
    marker_options: packet_encode::PacketMarkerOptions,
    allow_packetization_accelerator: bool,
    force_scalar_packetization: bool,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<packet_encode::PacketizedTileData, &'static str> {
    let packetization_resolutions = public_packetization_resolutions(resolution_packets);
    let packetization_job = J2kPacketizationEncodeJob {
        resolution_count: resolution_packets.len() as u32,
        num_layers,
        num_components,
        code_block_count: count_code_blocks(resolution_packets)?,
        progression_order: public_packetization_progression_order(progression_order),
        packet_descriptors,
        resolutions: &packetization_resolutions,
    };
    if allow_packetization_accelerator && !force_scalar_packetization {
        if let Some(data) = accelerator.encode_packetization(packetization_job)? {
            return Ok(packet_encode::PacketizedTileData {
                data,
                packet_lengths: Vec::new(),
                packet_headers: Vec::new(),
            });
        }
    }

    let scalar_packet_descriptors = scalar_packet_descriptors(packet_descriptors);
    packet_encode::form_tile_bitstream_with_descriptors_lengths_and_markers(
        resolution_packets,
        &scalar_packet_descriptors,
        marker_options,
    )
}

fn packetization_requires_scalar(
    params: &EncodeParams,
    tile_part_packet_limit: Option<u16>,
) -> bool {
    params.write_plt
        || params.write_plm
        || params.write_ppm
        || params.write_ppt
        || params.write_sop
        || params.write_eph
        || tile_part_packet_limit.is_some()
}

fn public_packetization_resolutions_from_compact<'a>(
    resolution_packets: &'a [PreparedCompactResolutionPacket<'a>],
) -> Vec<J2kPacketizationResolution<'a>> {
    resolution_packets
        .iter()
        .map(|resolution| J2kPacketizationResolution {
            subbands: resolution
                .subbands
                .iter()
                .map(|subband| J2kPacketizationSubband {
                    code_blocks: subband
                        .code_blocks
                        .iter()
                        .map(|code_block| J2kPacketizationCodeBlock {
                            data: code_block.data,
                            ht_cleanup_length: code_block.cleanup_length,
                            ht_refinement_length: code_block.refinement_length,
                            num_coding_passes: code_block.num_coding_passes,
                            num_zero_bitplanes: code_block.num_zero_bitplanes,
                            previously_included: false,
                            l_block: 3,
                            block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
                        })
                        .collect(),
                    num_cbs_x: subband.num_cbs_x,
                    num_cbs_y: subband.num_cbs_y,
                })
                .collect(),
        })
        .collect()
}

fn public_packetization_block_coding_mode(
    block_coding_mode: BlockCodingMode,
) -> J2kPacketizationBlockCodingMode {
    match block_coding_mode {
        BlockCodingMode::Classic => J2kPacketizationBlockCodingMode::Classic,
        BlockCodingMode::HighThroughput => J2kPacketizationBlockCodingMode::HighThroughput,
    }
}

#[derive(Clone)]
struct PreparedEncodeCodeBlock {
    coefficients: Vec<i64>,
    width: u32,
    height: u32,
}

#[derive(Clone)]
struct PreparedEncodeSubband {
    code_blocks: Vec<PreparedEncodeCodeBlock>,
    preencoded_ht_code_blocks: Option<Vec<EncodedHtJ2kCodeBlock>>,
    num_cbs_x: u32,
    num_cbs_y: u32,
    code_block_width: u32,
    code_block_height: u32,
    width: u32,
    height: u32,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
    block_coding_mode: BlockCodingMode,
    ht_target_coding_passes: u8,
}

struct PreparedResolutionPacket {
    component: u16,
    resolution: u32,
    precinct: u64,
    subbands: Vec<PreparedEncodeSubband>,
}

struct PreparedCompactCodeBlock<'a> {
    data: &'a [u8],
    cleanup_length: u32,
    refinement_length: u32,
    num_coding_passes: u8,
    num_zero_bitplanes: u8,
}

struct PreparedCompactSubband<'a> {
    code_blocks: Vec<PreparedCompactCodeBlock<'a>>,
    num_cbs_x: u32,
    num_cbs_y: u32,
}

struct PreparedCompactResolutionPacket<'a> {
    component: u16,
    resolution: u32,
    precinct: u64,
    subbands: Vec<PreparedCompactSubband<'a>>,
}

struct PreparedPrecomputedHtj2k97Image {
    params: EncodeParams,
    quant_params: Vec<(u16, u16)>,
    packet_descriptors: Vec<J2kPacketizationPacketDescriptor>,
    packet_count: usize,
    prepared_packets: Vec<PreparedResolutionPacket>,
}

fn prepare_precomputed_htj2k97_image_for_batch(
    image: &PrecomputedHtj2k97Image,
    options: &EncodeOptions,
) -> Result<PreparedPrecomputedHtj2k97Image, &'static str> {
    if image.width == 0 || image.height == 0 {
        return Err("invalid dimensions");
    }
    if image.components.is_empty() || image.components.len() > usize::from(MAX_J2K_SPEC_COMPONENTS)
    {
        return Err("unsupported component count");
    }
    if image.bit_depth == 0 || image.bit_depth > 16 {
        return Err("unsupported bit depth");
    }
    validate_irreversible_quantization_profile(options)?;
    if image
        .components
        .iter()
        .any(|component| component.x_rsiz == 0 || component.y_rsiz == 0)
    {
        return Err("component sampling factors must be non-zero");
    }
    validate_precomputed_dwt97_geometry(image)?;

    let num_components =
        u16::try_from(image.components.len()).map_err(|_| "unsupported component count")?;
    let num_levels = precomputed_97_level_count(&image.components)?;
    let guard_bits = options.guard_bits.max(2);
    let step_sizes = quantize::compute_step_sizes_with_irreversible_profile(
        image.bit_depth,
        num_levels,
        false,
        guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
    );
    let quant_params: Vec<(u16, u16)> = step_sizes
        .iter()
        .map(|s| (s.exponent, s.mantissa))
        .collect();
    let cb_width = 1u32 << (options.code_block_width_exp + 2);
    let cb_height = 1u32 << (options.code_block_height_exp + 2);
    let component_sampling = image
        .components
        .iter()
        .map(|component| (component.x_rsiz, component.y_rsiz))
        .collect::<Vec<_>>();
    let mut precomputed_options = options.clone();
    precomputed_options.num_decomposition_levels = num_levels;
    precomputed_options.reversible = false;
    precomputed_options.use_ht_block_coding = true;
    precomputed_options.use_mct = false;
    precomputed_options.validate_high_throughput_codestream = false;
    precomputed_options.component_sampling = Some(component_sampling.clone());
    let precinct_exponents = precinct_exponents_for_options(&precomputed_options, num_levels)?;
    let params = EncodeParams {
        width: image.width,
        height: image.height,
        tile_width: image.width,
        tile_height: image.height,
        num_components,
        bit_depth: image.bit_depth,
        signed: image.signed,
        component_sample_info: Vec::new(),
        component_quantization_step_sizes: Vec::new(),
        num_decomposition_levels: num_levels,
        reversible: false,
        code_block_width_exp: precomputed_options.code_block_width_exp,
        code_block_height_exp: precomputed_options.code_block_height_exp,
        num_layers: 1,
        use_mct: false,
        guard_bits,
        block_coding_mode: BlockCodingMode::HighThroughput,
        progression_order: precomputed_options.progression_order,
        write_tlm: precomputed_options.write_tlm,
        write_plt: precomputed_options.write_plt,
        write_plm: precomputed_options.write_plm,
        write_ppm: precomputed_options.write_ppm,
        write_ppt: precomputed_options.write_ppt,
        write_sop: precomputed_options.write_sop,
        write_eph: precomputed_options.write_eph,
        terminate_coding_passes: false,
        component_sampling,
        roi_component_shifts: vec![0; usize::from(num_components)],
        precinct_exponents,
    };

    let component_resolution_packets = image
        .components
        .iter()
        .enumerate()
        .map(|(component_idx, component)| {
            prepared_resolution_packets_from_precomputed_97_component(
                component_idx,
                component,
                &step_sizes,
                image.bit_depth,
                guard_bits,
                cb_width,
                cb_height,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let component_resolution_packets = split_component_resolution_packets_by_precinct(
        component_resolution_packets,
        image.width,
        image.height,
        num_levels,
        &params.precinct_exponents,
    )?;
    let prepared_packets =
        ordered_prepared_resolution_packets(component_resolution_packets, &precomputed_options)?;
    let packet_descriptors =
        packet_descriptors_for_order(&prepared_packets, 1, precomputed_options.progression_order)?;

    Ok(PreparedPrecomputedHtj2k97Image {
        params,
        quant_params,
        packet_descriptors,
        packet_count: 0,
        prepared_packets,
    })
}

fn prepared_resolution_packets_from_precomputed_97_component(
    component_idx: usize,
    component: &PrecomputedHtj2k97Component,
    step_sizes: &[QuantStepSize],
    bit_depth: u8,
    guard_bits: u8,
    cb_width: u32,
    cb_height: u32,
) -> Result<Vec<PreparedResolutionPacket>, &'static str> {
    let component_idx = u16::try_from(component_idx).map_err(|_| "component index exceeds u16")?;
    let mut packets = Vec::with_capacity(component.dwt.levels.len() + 1);
    packets.push(PreparedResolutionPacket {
        component: component_idx,
        resolution: 0,
        precinct: 0,
        subbands: vec![prepare_subband_cpu_quantized(
            &component.dwt.ll,
            component.dwt.ll_width,
            component.dwt.ll_height,
            step_sizes
                .first()
                .ok_or("irreversible quantization step missing")?,
            bit_depth,
            guard_bits,
            false,
            BlockCodingMode::HighThroughput,
            cb_width,
            cb_height,
            SubBandType::LowLow,
        )?],
    });

    for (level_idx, level) in component.dwt.levels.iter().enumerate() {
        let step_base = 1 + level_idx * 3;
        packets.push(PreparedResolutionPacket {
            component: component_idx,
            resolution: u32::try_from(level_idx + 1).map_err(|_| "resolution index exceeds u32")?,
            precinct: 0,
            subbands: vec![
                prepare_subband_cpu_quantized(
                    &level.hl,
                    level.high_width,
                    level.low_height,
                    step_sizes
                        .get(step_base)
                        .ok_or("irreversible quantization step missing")?,
                    bit_depth,
                    guard_bits,
                    false,
                    BlockCodingMode::HighThroughput,
                    cb_width,
                    cb_height,
                    SubBandType::HighLow,
                )?,
                prepare_subband_cpu_quantized(
                    &level.lh,
                    level.low_width,
                    level.high_height,
                    step_sizes
                        .get(step_base + 1)
                        .ok_or("irreversible quantization step missing")?,
                    bit_depth,
                    guard_bits,
                    false,
                    BlockCodingMode::HighThroughput,
                    cb_width,
                    cb_height,
                    SubBandType::LowHigh,
                )?,
                prepare_subband_cpu_quantized(
                    &level.hh,
                    level.high_width,
                    level.high_height,
                    step_sizes
                        .get(step_base + 2)
                        .ok_or("irreversible quantization step missing")?,
                    bit_depth,
                    guard_bits,
                    false,
                    BlockCodingMode::HighThroughput,
                    cb_width,
                    cb_height,
                    SubBandType::HighHigh,
                )?,
            ],
        });
    }

    Ok(packets)
}

fn copy_code_block_coefficients(
    quantized: &[i32],
    width: usize,
    x0: usize,
    y0: usize,
    cbw: usize,
    cbh: usize,
) -> Vec<i32> {
    let len = cbw * cbh;
    let start = y0 * width + x0;
    if cbw == width {
        return quantized[start..start + len].to_vec();
    }

    let mut coefficients = Vec::with_capacity(len);
    for y in 0..cbh {
        let row_start = (y0 + y) * width + x0;
        coefficients.extend_from_slice(&quantized[row_start..row_start + cbw]);
    }
    coefficients
}

fn copy_code_block_coefficients_i64(
    quantized: &[i64],
    width: usize,
    x0: usize,
    y0: usize,
    cbw: usize,
    cbh: usize,
) -> Vec<i64> {
    let len = cbw * cbh;
    let start = y0 * width + x0;
    if cbw == width {
        return quantized[start..start + len].to_vec();
    }

    let mut coefficients = Vec::with_capacity(len);
    for y in 0..cbh {
        let row_start = (y0 + y) * width + x0;
        coefficients.extend_from_slice(&quantized[row_start..row_start + cbw]);
    }
    coefficients
}

fn coefficients_fit_i32(coefficients: &[i64]) -> bool {
    coefficients
        .iter()
        .all(|&coefficient| i32::try_from(coefficient).is_ok())
}

fn downcast_i64_coefficients_to_i32(coefficients: &[i64]) -> Result<Vec<i32>, &'static str> {
    coefficients
        .iter()
        .map(|&coefficient| {
            i32::try_from(coefficient).map_err(|_| {
                "HTJ2K/accelerated code-block encode does not support i64 coefficients"
            })
        })
        .collect()
}

fn apply_roi_maxshift_encode(
    coefficients: &mut [i32],
    width: u32,
    height: u32,
    roi_shift: u8,
    roi_regions: &[ComponentRoiEncodeRegion],
    roi_scale: u32,
) -> Result<(), &'static str> {
    if roi_shift == 0 {
        return Ok(());
    }
    let expected_len = (width as usize)
        .checked_mul(height as usize)
        .ok_or("ROI subband dimensions overflow")?;
    if coefficients.len() != expected_len {
        return Err("ROI subband coefficient length mismatch");
    }
    if roi_regions.is_empty() {
        for coefficient in coefficients {
            shift_roi_coefficient(coefficient, roi_shift)?;
        }
        return Ok(());
    }

    let mut selected = vec![false; coefficients.len()];
    for region in roi_regions {
        let Some((x0, y0, x1, y1)) = roi_region_subband_window(*region, width, height, roi_scale)
        else {
            continue;
        };
        for y in y0..y1 {
            for x in x0..x1 {
                let idx = (y as usize)
                    .checked_mul(width as usize)
                    .and_then(|row| row.checked_add(x as usize))
                    .ok_or("ROI subband index overflow")?;
                if selected[idx] {
                    continue;
                }
                selected[idx] = true;
                shift_roi_coefficient(&mut coefficients[idx], roi_shift)?;
            }
        }
    }
    Ok(())
}

fn apply_roi_maxshift_encode_i64(
    coefficients: &mut [i64],
    width: u32,
    height: u32,
    roi_shift: u8,
    roi_regions: &[ComponentRoiEncodeRegion],
    roi_scale: u32,
) -> Result<(), &'static str> {
    if roi_shift == 0 {
        return Ok(());
    }
    let expected_len = (width as usize)
        .checked_mul(height as usize)
        .ok_or("ROI subband dimensions overflow")?;
    if coefficients.len() != expected_len {
        return Err("ROI subband coefficient length mismatch");
    }
    if roi_regions.is_empty() {
        for coefficient in coefficients {
            shift_roi_coefficient_i64(coefficient, roi_shift)?;
        }
        return Ok(());
    }

    let mut selected = vec![false; coefficients.len()];
    for region in roi_regions {
        let Some((x0, y0, x1, y1)) = roi_region_subband_window(*region, width, height, roi_scale)
        else {
            continue;
        };
        for y in y0..y1 {
            for x in x0..x1 {
                let idx = (y as usize)
                    .checked_mul(width as usize)
                    .and_then(|row| row.checked_add(x as usize))
                    .ok_or("ROI subband index overflow")?;
                if selected[idx] {
                    continue;
                }
                selected[idx] = true;
                shift_roi_coefficient_i64(&mut coefficients[idx], roi_shift)?;
            }
        }
    }
    Ok(())
}

fn shift_roi_coefficient(coefficient: &mut i32, roi_shift: u8) -> Result<(), &'static str> {
    *coefficient = coefficient
        .checked_shl(u32::from(roi_shift))
        .ok_or("ROI maxshift coefficient overflow")?;
    Ok(())
}

fn shift_roi_coefficient_i64(coefficient: &mut i64, roi_shift: u8) -> Result<(), &'static str> {
    let factor = 1_i64
        .checked_shl(u32::from(roi_shift))
        .ok_or("ROI maxshift coefficient overflow")?;
    *coefficient = coefficient
        .checked_mul(factor)
        .ok_or("ROI maxshift coefficient overflow")?;
    Ok(())
}

fn roi_region_subband_window(
    region: ComponentRoiEncodeRegion,
    width: u32,
    height: u32,
    roi_scale: u32,
) -> Option<(u32, u32, u32, u32)> {
    if width == 0 || height == 0 || roi_scale == 0 {
        return None;
    }
    let x1 = region.x.saturating_add(region.width);
    let y1 = region.y.saturating_add(region.height);
    let x0 = (region.x / roi_scale).min(width);
    let y0 = (region.y / roi_scale).min(height);
    let x1 = x1.div_ceil(roi_scale).min(width);
    let y1 = y1.div_ceil(roi_scale).min(height);
    if x0 >= x1 || y0 >= y1 {
        None
    } else {
        Some((x0, y0, x1, y1))
    }
}

fn prepare_subband(
    coefficients: &[f32],
    width: u32,
    height: u32,
    step_size: &QuantStepSize,
    bit_depth: u8,
    guard_bits: u8,
    reversible: bool,
    block_coding_mode: BlockCodingMode,
    cb_width: u32,
    cb_height: u32,
    sub_band_type: SubBandType,
    roi_shift: u8,
    roi_regions: &[ComponentRoiEncodeRegion],
    roi_scale: u32,
    ht_target_coding_passes: u8,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<PreparedEncodeSubband, &'static str> {
    if width == 0 || height == 0 {
        return Ok(PreparedEncodeSubband {
            code_blocks: Vec::new(),
            preencoded_ht_code_blocks: None,
            num_cbs_x: 0,
            num_cbs_y: 0,
            code_block_width: cb_width,
            code_block_height: cb_height,
            width,
            height,
            sub_band_type,
            total_bitplanes: 0,
            block_coding_mode,
            ht_target_coding_passes,
        });
    }

    let range_bits = subband_range_bits(bit_depth, sub_band_type);
    debug_assert!(step_size.exponent <= u16::from(u8::MAX));
    let base_total_bitplanes = guard_bits
        .saturating_add(step_size.exponent as u8)
        .saturating_sub(1);
    let total_bitplanes = base_total_bitplanes
        .checked_add(roi_shift)
        .ok_or("ROI maxshift exceeds supported coded bitplane count")?;
    let num_cbs_x = width.div_ceil(cb_width);
    let num_cbs_y = height.div_ceil(cb_height);

    if block_coding_mode == BlockCodingMode::HighThroughput
        && roi_shift == 0
        && ht_target_coding_passes == 1
    {
        if let Some(encoded) = accelerator.encode_ht_subband(J2kHtSubbandEncodeJob {
            coefficients,
            width,
            height,
            step_exponent: step_size.exponent,
            step_mantissa: step_size.mantissa,
            range_bits,
            reversible,
            code_block_width: cb_width,
            code_block_height: cb_height,
            total_bitplanes,
        })? {
            let expected_code_blocks = (num_cbs_x as usize)
                .checked_mul(num_cbs_y as usize)
                .ok_or("code-block count overflow")?;
            if encoded.len() != expected_code_blocks {
                return Err("accelerated HT subband code-block count mismatch");
            }
            return Ok(PreparedEncodeSubband {
                code_blocks: code_block_shapes(width, height, cb_width, cb_height)?,
                preencoded_ht_code_blocks: Some(encoded),
                num_cbs_x,
                num_cbs_y,
                code_block_width: cb_width,
                code_block_height: cb_height,
                width,
                height,
                sub_band_type,
                total_bitplanes,
                block_coding_mode,
                ht_target_coding_passes,
            });
        }
    }

    let mut quantized = match accelerator.encode_quantize_subband(J2kQuantizeSubbandJob {
        coefficients,
        step_exponent: step_size.exponent,
        step_mantissa: step_size.mantissa,
        range_bits,
        reversible,
    })? {
        Some(quantized) => {
            if quantized.len() != coefficients.len() {
                return Err("accelerated quantized subband length mismatch");
            }
            quantized
        }
        None => quantize::quantize_subband(coefficients, step_size, range_bits, reversible),
    };
    apply_roi_maxshift_encode(
        &mut quantized,
        width,
        height,
        roi_shift,
        roi_regions,
        roi_scale,
    )?;

    // Split into code-blocks
    let mut code_blocks = Vec::with_capacity((num_cbs_x * num_cbs_y) as usize);

    for cby in 0..num_cbs_y {
        for cbx in 0..num_cbs_x {
            let x0 = cbx * cb_width;
            let y0 = cby * cb_height;
            let x1 = (x0 + cb_width).min(width);
            let y1 = (y0 + cb_height).min(height);
            let cbw = x1 - x0;
            let cbh = y1 - y0;

            let cb_coeffs = copy_code_block_coefficients(
                &quantized,
                width as usize,
                x0 as usize,
                y0 as usize,
                cbw as usize,
                cbh as usize,
            )
            .into_iter()
            .map(i64::from)
            .collect();

            code_blocks.push(PreparedEncodeCodeBlock {
                coefficients: cb_coeffs,
                width: cbw,
                height: cbh,
            });
        }
    }

    Ok(PreparedEncodeSubband {
        code_blocks,
        preencoded_ht_code_blocks: None,
        num_cbs_x,
        num_cbs_y,
        code_block_width: cb_width,
        code_block_height: cb_height,
        width,
        height,
        sub_band_type,
        total_bitplanes,
        block_coding_mode,
        ht_target_coding_passes,
    })
}

#[allow(clippy::too_many_arguments)]
fn prepare_subband_i64(
    coefficients: &[i64],
    width: u32,
    height: u32,
    step_size: &QuantStepSize,
    guard_bits: u8,
    cb_width: u32,
    cb_height: u32,
    sub_band_type: SubBandType,
    roi_shift: u8,
    roi_regions: &[ComponentRoiEncodeRegion],
    roi_scale: u32,
    block_coding_mode: BlockCodingMode,
    ht_target_coding_passes: u8,
) -> Result<PreparedEncodeSubband, &'static str> {
    if width == 0 || height == 0 {
        return Ok(PreparedEncodeSubband {
            code_blocks: Vec::new(),
            preencoded_ht_code_blocks: None,
            num_cbs_x: 0,
            num_cbs_y: 0,
            code_block_width: cb_width,
            code_block_height: cb_height,
            width,
            height,
            sub_band_type,
            total_bitplanes: 0,
            block_coding_mode,
            ht_target_coding_passes,
        });
    }

    debug_assert!(step_size.exponent <= u16::from(u8::MAX));
    let base_total_bitplanes = guard_bits
        .saturating_add(step_size.exponent as u8)
        .saturating_sub(1);
    let total_bitplanes = base_total_bitplanes
        .checked_add(roi_shift)
        .ok_or("ROI maxshift exceeds supported coded bitplane count")?;
    let num_cbs_x = width.div_ceil(cb_width);
    let num_cbs_y = height.div_ceil(cb_height);
    let mut quantized = coefficients.to_vec();
    apply_roi_maxshift_encode_i64(
        &mut quantized,
        width,
        height,
        roi_shift,
        roi_regions,
        roi_scale,
    )?;

    let mut code_blocks = Vec::with_capacity((num_cbs_x * num_cbs_y) as usize);
    for cby in 0..num_cbs_y {
        for cbx in 0..num_cbs_x {
            let x0 = cbx * cb_width;
            let y0 = cby * cb_height;
            let x1 = (x0 + cb_width).min(width);
            let y1 = (y0 + cb_height).min(height);
            let cbw = x1 - x0;
            let cbh = y1 - y0;

            let cb_coeffs = copy_code_block_coefficients_i64(
                &quantized,
                width as usize,
                x0 as usize,
                y0 as usize,
                cbw as usize,
                cbh as usize,
            );

            code_blocks.push(PreparedEncodeCodeBlock {
                coefficients: cb_coeffs,
                width: cbw,
                height: cbh,
            });
        }
    }

    Ok(PreparedEncodeSubband {
        code_blocks,
        preencoded_ht_code_blocks: None,
        num_cbs_x,
        num_cbs_y,
        code_block_width: cb_width,
        code_block_height: cb_height,
        width,
        height,
        sub_band_type,
        total_bitplanes,
        block_coding_mode,
        ht_target_coding_passes,
    })
}

fn prepare_subband_cpu_quantized(
    coefficients: &[f32],
    width: u32,
    height: u32,
    step_size: &QuantStepSize,
    bit_depth: u8,
    guard_bits: u8,
    reversible: bool,
    block_coding_mode: BlockCodingMode,
    cb_width: u32,
    cb_height: u32,
    sub_band_type: SubBandType,
) -> Result<PreparedEncodeSubband, &'static str> {
    if width == 0 || height == 0 {
        return Ok(PreparedEncodeSubband {
            code_blocks: Vec::new(),
            preencoded_ht_code_blocks: None,
            num_cbs_x: 0,
            num_cbs_y: 0,
            code_block_width: cb_width,
            code_block_height: cb_height,
            width,
            height,
            sub_band_type,
            total_bitplanes: 0,
            block_coding_mode,
            ht_target_coding_passes: 1,
        });
    }

    let range_bits = subband_range_bits(bit_depth, sub_band_type);
    debug_assert!(step_size.exponent <= u16::from(u8::MAX));
    let total_bitplanes = guard_bits
        .saturating_add(step_size.exponent as u8)
        .saturating_sub(1);
    let num_cbs_x = width.div_ceil(cb_width);
    let num_cbs_y = height.div_ceil(cb_height);
    let quantized = quantize::quantize_subband(coefficients, step_size, range_bits, reversible);
    let mut code_blocks = Vec::with_capacity((num_cbs_x * num_cbs_y) as usize);

    for cby in 0..num_cbs_y {
        for cbx in 0..num_cbs_x {
            let x0 = cbx * cb_width;
            let y0 = cby * cb_height;
            let x1 = (x0 + cb_width).min(width);
            let y1 = (y0 + cb_height).min(height);
            let cbw = x1 - x0;
            let cbh = y1 - y0;
            let cb_coeffs = copy_code_block_coefficients(
                &quantized,
                width as usize,
                x0 as usize,
                y0 as usize,
                cbw as usize,
                cbh as usize,
            )
            .into_iter()
            .map(i64::from)
            .collect();

            code_blocks.push(PreparedEncodeCodeBlock {
                coefficients: cb_coeffs,
                width: cbw,
                height: cbh,
            });
        }
    }

    Ok(PreparedEncodeSubband {
        code_blocks,
        preencoded_ht_code_blocks: None,
        num_cbs_x,
        num_cbs_y,
        code_block_width: cb_width,
        code_block_height: cb_height,
        width,
        height,
        sub_band_type,
        total_bitplanes,
        block_coding_mode,
        ht_target_coding_passes: 1,
    })
}

fn code_block_shapes(
    width: u32,
    height: u32,
    cb_width: u32,
    cb_height: u32,
) -> Result<Vec<PreparedEncodeCodeBlock>, &'static str> {
    let num_cbs_x = width.div_ceil(cb_width);
    let num_cbs_y = height.div_ceil(cb_height);
    let count = (num_cbs_x as usize)
        .checked_mul(num_cbs_y as usize)
        .ok_or("code-block count overflow")?;
    let mut code_blocks = Vec::with_capacity(count);
    for cby in 0..num_cbs_y {
        for cbx in 0..num_cbs_x {
            let x0 = cbx * cb_width;
            let y0 = cby * cb_height;
            let x1 = (x0 + cb_width).min(width);
            let y1 = (y0 + cb_height).min(height);
            code_blocks.push(PreparedEncodeCodeBlock {
                coefficients: Vec::new(),
                width: x1 - x0,
                height: y1 - y0,
            });
        }
    }
    Ok(code_blocks)
}

fn subband_range_bits(bit_depth: u8, sub_band_type: SubBandType) -> u8 {
    let log_gain = match sub_band_type {
        SubBandType::LowLow => 0,
        SubBandType::LowHigh | SubBandType::HighLow => 1,
        SubBandType::HighHigh => 2,
    };

    bit_depth.saturating_add(log_gain)
}

fn encode_prepared_resolution_packets(
    prepared_packets: Vec<PreparedResolutionPacket>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<ResolutionPacket>, &'static str> {
    let subband_counts: Vec<_> = prepared_packets
        .iter()
        .map(|packet| packet.subbands.len())
        .collect();
    let prepared_subbands: Vec<_> = prepared_packets
        .into_iter()
        .flat_map(|packet| packet.subbands)
        .collect();
    let mut encoded_subbands =
        encode_prepared_subbands(prepared_subbands, accelerator)?.into_iter();

    subband_counts
        .into_iter()
        .map(|subband_count| {
            let mut subbands = Vec::with_capacity(subband_count);
            for _ in 0..subband_count {
                subbands.push(
                    encoded_subbands
                        .next()
                        .ok_or("encoded subband count mismatch")?,
                );
            }
            Ok(ResolutionPacket { subbands })
        })
        .collect()
}

fn encode_prepared_resolution_packets_layered(
    prepared_packets: Vec<PreparedResolutionPacket>,
    num_layers: u8,
    progression_order: EncodeProgressionOrder,
    quality_layer_byte_targets: &[u64],
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<(Vec<ResolutionPacket>, Vec<J2kPacketizationPacketDescriptor>), &'static str> {
    let layer_count = usize::from(num_layers);
    let mut layered_packets = Vec::with_capacity(prepared_packets.len());
    let mut classic_candidates = Vec::new();
    let mut classic_locations = Vec::new();
    let mut classic_block_index = 0usize;
    let mut ht_candidates = Vec::new();
    let mut ht_locations = Vec::new();
    let mut ht_block_index = 0usize;

    for prepared_packet in prepared_packets {
        let packet_idx = layered_packets.len();
        let mut layered_packet = LayeredPreparedPacket {
            component: prepared_packet.component,
            resolution: prepared_packet.resolution,
            precinct: prepared_packet.precinct,
            subbands: Vec::with_capacity(prepared_packet.subbands.len()),
        };

        for subband in prepared_packet.subbands {
            let subband_idx = layered_packet.subbands.len();
            let mut layered_subband = LayeredPreparedSubband {
                num_cbs_x: subband.num_cbs_x,
                num_cbs_y: subband.num_cbs_y,
                blocks: Vec::with_capacity(subband.code_blocks.len()),
            };

            match subband.block_coding_mode {
                BlockCodingMode::Classic => {
                    for block in subband.code_blocks {
                        let block_idx = layered_subband.blocks.len();
                        let encoded = bitplane_encode::encode_code_block_segments_with_style_i64(
                            &block.coefficients,
                            block.width,
                            block.height,
                            subband.sub_band_type,
                            subband.total_bitplanes,
                            &classic_multilayer_code_block_style(),
                        );
                        let segment_layers = if quality_layer_byte_targets.is_empty() {
                            classic_unbudgeted_segment_layers(&encoded, num_layers)?
                        } else {
                            for (segment_idx, segment) in encoded.segments.iter().enumerate() {
                                classic_candidates.push(ClassicSegmentAssignmentCandidate {
                                    block_index: classic_block_index,
                                    segment_index: segment_idx,
                                    rate: u64::from(segment.data_length),
                                    distortion_delta: segment.distortion_delta,
                                });
                                classic_locations.push(ClassicSegmentLocation {
                                    packet_idx,
                                    subband_idx,
                                    block_idx,
                                    segment_idx,
                                });
                            }
                            vec![layer_count.saturating_sub(1); encoded.segments.len()]
                        };
                        layered_subband.blocks.push(LayeredPreparedBlock::Classic {
                            encoded,
                            segment_layers,
                        });
                        classic_block_index = classic_block_index
                            .checked_add(1)
                            .ok_or("classic PCRD block index overflow")?;
                    }
                }
                BlockCodingMode::HighThroughput => {
                    let encoded_blocks =
                        encode_all_ht_code_blocks(core::slice::from_ref(&subband), accelerator)?;
                    let block_count = encoded_blocks.len();
                    for (block_idx, encoded) in encoded_blocks.into_iter().enumerate() {
                        let segment_layers = if quality_layer_byte_targets.is_empty() {
                            ht_unbudgeted_segment_layers(
                                &encoded,
                                num_layers,
                                block_idx,
                                block_count,
                            )?
                        } else {
                            let segment_count = ht_segment_count(&encoded);
                            let mut segment_layers = Vec::with_capacity(segment_count);
                            for segment_idx in 0..segment_count {
                                ht_candidates.push(HtSegmentAssignmentCandidate {
                                    block_index: ht_block_index,
                                    segment_index: segment_idx,
                                    rate: ht_segment_rate(&encoded, segment_idx)?,
                                });
                                ht_locations.push(HtSegmentLocation {
                                    packet_idx,
                                    subband_idx,
                                    block_idx: layered_subband.blocks.len(),
                                    segment_idx,
                                });
                                segment_layers.push(layer_count.saturating_sub(1));
                            }
                            segment_layers
                        };
                        layered_subband
                            .blocks
                            .push(LayeredPreparedBlock::HighThroughput {
                                encoded,
                                segment_layers,
                            });
                        ht_block_index = ht_block_index
                            .checked_add(1)
                            .ok_or("HTJ2K segment block index overflow")?;
                    }
                }
            }

            layered_packet.subbands.push(layered_subband);
        }

        layered_packets.push(layered_packet);
    }

    if !quality_layer_byte_targets.is_empty() {
        let assignments = assign_classic_segment_layers_by_slope(
            &classic_candidates,
            layer_count,
            quality_layer_byte_targets,
        )?;
        for (assignment_idx, layer) in assignments.into_iter().enumerate() {
            let location = classic_locations
                .get(assignment_idx)
                .ok_or("classic PCRD assignment location mismatch")?;
            let block = layered_packets
                .get_mut(location.packet_idx)
                .ok_or("classic PCRD packet index mismatch")?
                .subbands
                .get_mut(location.subband_idx)
                .ok_or("classic PCRD subband index mismatch")?
                .blocks
                .get_mut(location.block_idx)
                .ok_or("classic PCRD block index mismatch")?;
            let LayeredPreparedBlock::Classic { segment_layers, .. } = block else {
                return Err("classic PCRD assignment referenced HT block");
            };
            let segment_layer = segment_layers
                .get_mut(location.segment_idx)
                .ok_or("classic PCRD segment index mismatch")?;
            *segment_layer = layer;
        }
        enforce_classic_segment_layer_monotonicity(&mut layered_packets);
    }
    if !quality_layer_byte_targets.is_empty() {
        let assignments = assign_ht_segment_layers_by_budget(
            &ht_candidates,
            layer_count,
            quality_layer_byte_targets,
        )?;
        for (assignment_idx, layer) in assignments.into_iter().enumerate() {
            let location = ht_locations
                .get(assignment_idx)
                .ok_or("HTJ2K segment assignment location mismatch")?;
            let block = layered_packets
                .get_mut(location.packet_idx)
                .ok_or("HTJ2K packet index mismatch")?
                .subbands
                .get_mut(location.subband_idx)
                .ok_or("HTJ2K subband index mismatch")?
                .blocks
                .get_mut(location.block_idx)
                .ok_or("HTJ2K block index mismatch")?;
            let LayeredPreparedBlock::HighThroughput { segment_layers, .. } = block else {
                return Err("HTJ2K segment assignment referenced classic block");
            };
            let segment_layer = segment_layers
                .get_mut(location.segment_idx)
                .ok_or("HTJ2K segment index mismatch")?;
            *segment_layer = layer;
        }
        enforce_ht_segment_layer_monotonicity(&mut layered_packets);
    }

    let mut resolution_packets = Vec::with_capacity(layered_packets.len() * layer_count);
    let mut descriptors = Vec::with_capacity(layered_packets.len() * layer_count);
    for (state_index, layered_packet) in layered_packets.into_iter().enumerate() {
        let mut layer_packets: Vec<_> = (0..layer_count)
            .map(|_| ResolutionPacket {
                subbands: Vec::with_capacity(layered_packet.subbands.len()),
            })
            .collect();

        for subband in layered_packet.subbands {
            let mut layer_subbands: Vec<_> = (0..layer_count)
                .map(|_| SubbandPrecinct {
                    code_blocks: Vec::with_capacity(subband.blocks.len()),
                    num_cbs_x: subband.num_cbs_x,
                    num_cbs_y: subband.num_cbs_y,
                })
                .collect();

            for block in subband.blocks {
                let contributions = match block {
                    LayeredPreparedBlock::Classic {
                        encoded,
                        segment_layers,
                    } => classic_layer_contributions(encoded, num_layers, &segment_layers)?,
                    LayeredPreparedBlock::HighThroughput {
                        encoded,
                        segment_layers,
                    } => ht_layer_contributions(encoded, num_layers, &segment_layers)?,
                };
                for (layer_idx, contribution) in contributions.into_iter().enumerate() {
                    layer_subbands[layer_idx].code_blocks.push(contribution);
                }
            }

            for (layer_packet, layer_subband) in layer_packets.iter_mut().zip(layer_subbands) {
                layer_packet.subbands.push(layer_subband);
            }
        }

        let state_index =
            u32::try_from(state_index).map_err(|_| "packet descriptor state index exceeds u32")?;
        for (layer_idx, layer_packet) in layer_packets.into_iter().enumerate() {
            let packet_index = u32::try_from(resolution_packets.len())
                .map_err(|_| "packet descriptor index exceeds u32")?;
            resolution_packets.push(layer_packet);
            descriptors.push(J2kPacketizationPacketDescriptor {
                packet_index,
                state_index,
                layer: u8::try_from(layer_idx).map_err(|_| "quality layer index exceeds u8")?,
                resolution: layered_packet.resolution,
                component: layered_packet.component,
                precinct: layered_packet.precinct,
            });
        }
    }

    sort_packet_descriptors_for_progression(&mut descriptors, progression_order);

    Ok((resolution_packets, descriptors))
}

fn sort_packet_descriptors_for_progression(
    descriptors: &mut [J2kPacketizationPacketDescriptor],
    progression_order: EncodeProgressionOrder,
) {
    match progression_order {
        EncodeProgressionOrder::Lrcp => descriptors.sort_by_key(|descriptor| {
            (
                descriptor.layer,
                descriptor.resolution,
                descriptor.component,
                descriptor.precinct,
            )
        }),
        EncodeProgressionOrder::Rlcp => descriptors.sort_by_key(|descriptor| {
            (
                descriptor.resolution,
                descriptor.layer,
                descriptor.component,
                descriptor.precinct,
            )
        }),
        EncodeProgressionOrder::Rpcl => descriptors.sort_by_key(|descriptor| {
            (
                descriptor.resolution,
                descriptor.precinct,
                descriptor.component,
                descriptor.layer,
            )
        }),
        EncodeProgressionOrder::Pcrl => descriptors.sort_by_key(|descriptor| {
            (
                descriptor.precinct,
                descriptor.component,
                descriptor.resolution,
                descriptor.layer,
            )
        }),
        EncodeProgressionOrder::Cprl => descriptors.sort_by_key(|descriptor| {
            (
                descriptor.component,
                descriptor.precinct,
                descriptor.resolution,
                descriptor.layer,
            )
        }),
    }
}

fn classic_multilayer_code_block_style() -> CodeBlockStyle {
    CodeBlockStyle {
        termination_on_each_pass: true,
        ..CodeBlockStyle::default()
    }
}

struct LayeredPreparedPacket {
    component: u16,
    resolution: u32,
    precinct: u64,
    subbands: Vec<LayeredPreparedSubband>,
}

struct LayeredPreparedSubband {
    num_cbs_x: u32,
    num_cbs_y: u32,
    blocks: Vec<LayeredPreparedBlock>,
}

enum LayeredPreparedBlock {
    Classic {
        encoded: bitplane_encode::EncodedCodeBlockWithSegments,
        segment_layers: Vec<usize>,
    },
    HighThroughput {
        encoded: bitplane_encode::EncodedCodeBlock,
        segment_layers: Vec<usize>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ClassicSegmentAssignmentCandidate {
    block_index: usize,
    segment_index: usize,
    rate: u64,
    distortion_delta: f64,
}

#[derive(Debug, Clone, Copy)]
struct ClassicSegmentLocation {
    packet_idx: usize,
    subband_idx: usize,
    block_idx: usize,
    segment_idx: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HtSegmentAssignmentCandidate {
    block_index: usize,
    segment_index: usize,
    rate: u64,
}

#[derive(Debug, Clone, Copy)]
struct HtSegmentLocation {
    packet_idx: usize,
    subband_idx: usize,
    block_idx: usize,
    segment_idx: usize,
}

struct ClassicLayerBudgetAllocator {
    cumulative_targets: Vec<u64>,
    cumulative_used: Vec<u64>,
}

impl ClassicLayerBudgetAllocator {
    fn new(cumulative_targets: &[u64], layer_count: usize) -> Result<Self, &'static str> {
        if cumulative_targets.is_empty() {
            return Ok(Self {
                cumulative_targets: Vec::new(),
                cumulative_used: Vec::new(),
            });
        }
        if cumulative_targets.len() != layer_count {
            return Err("quality layer byte target count must match quality layer count");
        }
        if cumulative_targets.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err("quality layer byte targets must be cumulative and monotonic");
        }
        Ok(Self {
            cumulative_targets: cumulative_targets
                .iter()
                .map(|&target| target.saturating_add(classic_rate_target_tolerance(target)))
                .collect(),
            cumulative_used: vec![0; layer_count],
        })
    }

    fn is_budgeted(&self) -> bool {
        !self.cumulative_targets.is_empty()
    }

    fn assign_segment(
        &mut self,
        min_layer: usize,
        data_length: u64,
    ) -> Result<usize, &'static str> {
        if !self.is_budgeted() {
            return Ok(min_layer);
        }

        let rate = data_length;
        let last_layer = self
            .cumulative_targets
            .len()
            .checked_sub(1)
            .ok_or("quality layer target count underflow")?;
        for layer_idx in min_layer..last_layer {
            if self.layer_can_accept(layer_idx, rate)? {
                self.record_segment(layer_idx, rate)?;
                return Ok(layer_idx);
            }
        }
        self.record_segment(last_layer, rate)?;
        Ok(last_layer)
    }

    fn layer_can_accept(&self, layer_idx: usize, rate: u64) -> Result<bool, &'static str> {
        for cumulative_idx in layer_idx..self.cumulative_targets.len() {
            let used = self.cumulative_used[cumulative_idx]
                .checked_add(rate)
                .ok_or("quality layer byte budget overflow")?;
            if used > self.cumulative_targets[cumulative_idx] {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn record_segment(&mut self, layer_idx: usize, rate: u64) -> Result<(), &'static str> {
        for used in &mut self.cumulative_used[layer_idx..] {
            *used = used
                .checked_add(rate)
                .ok_or("quality layer byte budget overflow")?;
        }
        Ok(())
    }
}

fn classic_rate_target_tolerance(target: u64) -> u64 {
    (target / 100).max(512)
}

fn assign_classic_segment_layers_by_slope(
    candidates: &[ClassicSegmentAssignmentCandidate],
    layer_count: usize,
    cumulative_targets: &[u64],
) -> Result<Vec<usize>, &'static str> {
    let mut allocator = ClassicLayerBudgetAllocator::new(cumulative_targets, layer_count)?;
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let block_count = candidates
        .iter()
        .map(|candidate| candidate.block_index)
        .max()
        .and_then(|max| max.checked_add(1))
        .ok_or("classic PCRD block count overflow")?;
    let mut block_candidates = vec![Vec::new(); block_count];
    for (candidate_idx, candidate) in candidates.iter().enumerate() {
        block_candidates
            .get_mut(candidate.block_index)
            .ok_or("classic PCRD block index mismatch")?
            .push(candidate_idx);
    }
    for block in &mut block_candidates {
        block.sort_by_key(|&idx| candidates[idx].segment_index);
    }

    let mut block_min_layers = vec![0usize; block_count];
    let mut assignments = vec![layer_count.saturating_sub(1); candidates.len()];
    let mut next_block_segment = vec![0usize; block_count];
    let mut remaining = candidates.len();
    while remaining > 0 {
        let candidate_idx = block_candidates
            .iter()
            .enumerate()
            .filter_map(|(block_idx, block)| block.get(next_block_segment[block_idx]).copied())
            .min_by(|&left, &right| compare_classic_segment_candidates(candidates, left, right))
            .ok_or("classic PCRD candidate queue underflow")?;
        let candidate = candidates[candidate_idx];
        let min_layer = *block_min_layers
            .get(candidate.block_index)
            .ok_or("classic PCRD block index mismatch")?;
        let layer = allocator.assign_segment(min_layer, candidate.rate)?;
        assignments[candidate_idx] = layer;
        if let Some(block_layer) = block_min_layers.get_mut(candidate.block_index) {
            *block_layer = layer;
        }
        if let Some(next) = next_block_segment.get_mut(candidate.block_index) {
            *next = next
                .checked_add(1)
                .ok_or("classic PCRD segment index overflow")?;
        }
        remaining -= 1;
    }

    enforce_classic_assignment_monotonicity(candidates, &mut assignments);
    Ok(assignments)
}

fn compare_classic_segment_candidates(
    candidates: &[ClassicSegmentAssignmentCandidate],
    left: usize,
    right: usize,
) -> Ordering {
    let left_candidate = candidates[left];
    let right_candidate = candidates[right];
    pcrd_slope(right_candidate)
        .partial_cmp(&pcrd_slope(left_candidate))
        .unwrap_or(Ordering::Equal)
        .then_with(|| left_candidate.block_index.cmp(&right_candidate.block_index))
        .then_with(|| {
            left_candidate
                .segment_index
                .cmp(&right_candidate.segment_index)
        })
}

fn pcrd_slope(candidate: ClassicSegmentAssignmentCandidate) -> f64 {
    if candidate.rate == 0 {
        return f64::INFINITY;
    }
    candidate.distortion_delta / candidate.rate as f64
}

fn enforce_classic_assignment_monotonicity(
    candidates: &[ClassicSegmentAssignmentCandidate],
    assignments: &mut [usize],
) {
    let mut order: Vec<_> = (0..candidates.len()).collect();
    order.sort_by_key(|&idx| (candidates[idx].block_index, candidates[idx].segment_index));
    let mut current_block = None;
    let mut min_layer = 0usize;
    for idx in order {
        if current_block != Some(candidates[idx].block_index) {
            current_block = Some(candidates[idx].block_index);
            min_layer = 0;
        }
        if assignments[idx] < min_layer {
            assignments[idx] = min_layer;
        }
        min_layer = assignments[idx];
    }
}

fn enforce_classic_segment_layer_monotonicity(layered_packets: &mut [LayeredPreparedPacket]) {
    for packet in layered_packets {
        for subband in &mut packet.subbands {
            for block in &mut subband.blocks {
                if let LayeredPreparedBlock::Classic { segment_layers, .. } = block {
                    let mut min_layer = 0usize;
                    for layer in segment_layers {
                        if *layer < min_layer {
                            *layer = min_layer;
                        }
                        min_layer = *layer;
                    }
                }
            }
        }
    }
}

fn enforce_ht_segment_layer_monotonicity(layered_packets: &mut [LayeredPreparedPacket]) {
    for packet in layered_packets {
        for subband in &mut packet.subbands {
            for block in &mut subband.blocks {
                if let LayeredPreparedBlock::HighThroughput { segment_layers, .. } = block {
                    let mut min_layer = 0usize;
                    for layer in segment_layers {
                        if *layer < min_layer {
                            *layer = min_layer;
                        }
                        min_layer = *layer;
                    }
                }
            }
        }
    }
}

fn assign_ht_segment_layers_by_budget(
    candidates: &[HtSegmentAssignmentCandidate],
    layer_count: usize,
    cumulative_targets: &[u64],
) -> Result<Vec<usize>, &'static str> {
    let mut allocator = ClassicLayerBudgetAllocator::new(cumulative_targets, layer_count)?;
    let mut assignments = vec![layer_count.saturating_sub(1); candidates.len()];
    let mut candidate_order: Vec<_> = (0..candidates.len()).collect();
    candidate_order
        .sort_by_key(|&idx| (candidates[idx].block_index, candidates[idx].segment_index));
    let mut block_min_layers = vec![
        0usize;
        candidates
            .iter()
            .map(|c| c.block_index)
            .max()
            .map_or(0, |idx| idx + 1)
    ];

    for candidate_idx in candidate_order {
        let candidate = candidates
            .get(candidate_idx)
            .ok_or("HTJ2K segment candidate index mismatch")?;
        let min_layer = *block_min_layers
            .get(candidate.block_index)
            .ok_or("HTJ2K segment candidate block index mismatch")?;
        let layer = allocator.assign_segment(min_layer, candidate.rate)?;
        assignments[candidate_idx] = layer;
        if let Some(block_layer) = block_min_layers.get_mut(candidate.block_index) {
            *block_layer = layer;
        }
    }

    Ok(assignments)
}

fn ht_segment_count(encoded: &bitplane_encode::EncodedCodeBlock) -> usize {
    match encoded.num_coding_passes {
        0 => 0,
        1 => 1,
        _ => 2,
    }
}

fn ht_segment_rate(
    encoded: &bitplane_encode::EncodedCodeBlock,
    segment_idx: usize,
) -> Result<u64, &'static str> {
    match segment_idx {
        0 if encoded.num_coding_passes > 0 => Ok(u64::from(encoded.ht_cleanup_length)),
        1 if encoded.num_coding_passes > 1 => Ok(u64::from(encoded.ht_refinement_length)),
        _ => Err("HTJ2K segment index out of range"),
    }
}

fn ht_unbudgeted_segment_layers(
    encoded: &bitplane_encode::EncodedCodeBlock,
    num_layers: u8,
    block_idx: usize,
    block_count: usize,
) -> Result<Vec<usize>, &'static str> {
    let segment_count = ht_segment_count(encoded);
    if segment_count == 0 {
        return Ok(Vec::new());
    }
    let layer_count = usize::from(num_layers);
    if layer_count == 0 {
        return Err("HTJ2K layer allocation requires non-empty inputs");
    }
    if encoded.num_coding_passes == 1 {
        return Ok(vec![ht_target_layer(block_idx, block_count, layer_count)?]);
    }

    let mut segment_layers = Vec::with_capacity(segment_count);
    let mut min_layer = 0usize;
    for (_, end_pass) in [(0, 1), (1, encoded.num_coding_passes)] {
        let mut assigned = None;
        for layer_idx in min_layer..layer_count {
            let cumulative_passes = if layer_idx + 1 == layer_count {
                encoded.num_coding_passes
            } else {
                layer_pass_count(encoded.num_coding_passes, layer_idx + 1, num_layers)?
            };
            if end_pass <= cumulative_passes {
                assigned = Some(layer_idx);
                break;
            }
        }
        let assigned =
            assigned.ok_or("HTJ2K quality layer split must align to segment boundaries")?;
        segment_layers.push(assigned);
        min_layer = assigned;
    }
    Ok(segment_layers)
}

fn classic_unbudgeted_segment_layers(
    encoded: &bitplane_encode::EncodedCodeBlockWithSegments,
    num_layers: u8,
) -> Result<Vec<usize>, &'static str> {
    let mut segment_layers = Vec::with_capacity(encoded.segments.len());
    for segment in &encoded.segments {
        let mut assigned = None;
        for layer_idx in 0..usize::from(num_layers) {
            let previous_pass =
                previous_layer_pass_count(encoded.num_coding_passes, layer_idx, num_layers)?;
            let cumulative_passes = if layer_idx + 1 == usize::from(num_layers) {
                encoded.num_coding_passes
            } else {
                layer_pass_count(encoded.num_coding_passes, layer_idx + 1, num_layers)?
            };
            if segment.start_coding_pass >= previous_pass
                && segment.end_coding_pass <= cumulative_passes
            {
                assigned = Some(layer_idx);
                break;
            }
        }
        segment_layers.push(
            assigned.ok_or("classic quality layer split must align to terminated coding passes")?,
        );
    }
    Ok(segment_layers)
}

fn classic_layer_contributions(
    encoded: bitplane_encode::EncodedCodeBlockWithSegments,
    num_layers: u8,
    segment_layers: &[usize],
) -> Result<Vec<CodeBlockPacketData>, &'static str> {
    let layer_count = usize::from(num_layers);
    if segment_layers.len() != encoded.segments.len() {
        return Err("classic PCRD segment assignment count mismatch");
    }
    if segment_layers.iter().any(|&layer| layer >= layer_count) {
        return Err("classic PCRD segment layer exceeds layer count");
    }
    let mut contributions = Vec::with_capacity(layer_count);

    for layer_idx in 0..layer_count {
        let mut data = Vec::new();
        let mut classic_segment_lengths = Vec::new();
        let mut contribution_passes = 0u8;

        for (segment_idx, segment) in encoded.segments.iter().enumerate() {
            if segment_layers[segment_idx] != layer_idx {
                continue;
            }
            let start = usize::try_from(segment.data_offset)
                .map_err(|_| "classic code-block segment offset overflow")?;
            let len = usize::try_from(segment.data_length)
                .map_err(|_| "classic code-block segment length overflow")?;
            let end = start
                .checked_add(len)
                .ok_or("classic code-block segment range overflow")?;
            data.extend_from_slice(
                encoded
                    .data
                    .get(start..end)
                    .ok_or("classic code-block segment range invalid")?,
            );
            classic_segment_lengths.push(segment.data_length);
            contribution_passes = contribution_passes
                .checked_add(segment.end_coding_pass - segment.start_coding_pass)
                .ok_or("classic code-block contribution pass count overflow")?;
        }

        contributions.push(CodeBlockPacketData {
            data,
            ht_cleanup_length: 0,
            ht_refinement_length: 0,
            num_coding_passes: contribution_passes,
            classic_segment_lengths,
            num_zero_bitplanes: encoded.num_zero_bitplanes,
            previously_included: false,
            l_block: 3,
            block_coding_mode: BlockCodingMode::Classic,
        });
    }

    Ok(contributions)
}

fn layer_pass_count(
    num_coding_passes: u8,
    layer_count: usize,
    num_layers: u8,
) -> Result<u8, &'static str> {
    let numerator = u32::from(num_coding_passes)
        .checked_mul(u32::try_from(layer_count).map_err(|_| "layer index overflow")?)
        .ok_or("quality layer pass allocation overflow")?;
    numerator
        .div_ceil(u32::from(num_layers))
        .try_into()
        .map_err(|_| "quality layer pass allocation overflow")
}

fn previous_layer_pass_count(
    num_coding_passes: u8,
    layer_idx: usize,
    num_layers: u8,
) -> Result<u8, &'static str> {
    if layer_idx == 0 {
        Ok(0)
    } else {
        layer_pass_count(num_coding_passes, layer_idx, num_layers)
    }
}

fn ht_target_layer(
    block_idx: usize,
    block_count: usize,
    layer_count: usize,
) -> Result<usize, &'static str> {
    if block_count == 0 || layer_count == 0 {
        return Err("HTJ2K layer allocation requires non-empty inputs");
    }
    Ok(block_idx
        .checked_mul(layer_count)
        .ok_or("HTJ2K layer allocation overflow")?
        / block_count)
}

fn ht_layer_contributions(
    encoded: bitplane_encode::EncodedCodeBlock,
    num_layers: u8,
    segment_layers: &[usize],
) -> Result<Vec<CodeBlockPacketData>, &'static str> {
    let layer_count = usize::from(num_layers);
    if segment_layers.len() != ht_segment_count(&encoded) {
        return Err("HTJ2K segment assignment count mismatch");
    }
    if segment_layers.iter().any(|&layer| layer >= layer_count) {
        return Err("HTJ2K segment layer exceeds layer count");
    }
    if segment_layers
        .windows(2)
        .any(|layers| layers[1] < layers[0])
    {
        return Err("HTJ2K segment layers must be monotonic");
    }

    let cleanup_len = usize::try_from(encoded.ht_cleanup_length)
        .map_err(|_| "HTJ2K cleanup segment length overflow")?;
    let refinement_len = usize::try_from(encoded.ht_refinement_length)
        .map_err(|_| "HTJ2K refinement segment length overflow")?;
    let refinement_start = cleanup_len;
    let refinement_end = refinement_start
        .checked_add(refinement_len)
        .ok_or("HTJ2K refinement segment range overflow")?;
    if encoded.num_coding_passes > 0 && cleanup_len == 0 {
        return Err("HTJ2K cleanup segment is missing");
    }
    if encoded.num_coding_passes > 1 && refinement_len == 0 {
        return Err("HTJ2K refinement segment is missing");
    }
    if refinement_end > encoded.data.len() {
        return Err("HTJ2K segment range invalid");
    }

    let mut contributions = Vec::with_capacity(layer_count);
    for layer_idx in 0..layer_count {
        let mut data = Vec::new();
        let mut ht_cleanup_length = 0u32;
        let mut ht_refinement_length = 0u32;
        let mut num_coding_passes = 0u8;

        if segment_layers.first() == Some(&layer_idx) {
            data.extend_from_slice(
                encoded
                    .data
                    .get(..cleanup_len)
                    .ok_or("HTJ2K cleanup segment range invalid")?,
            );
            ht_cleanup_length = encoded.ht_cleanup_length;
            num_coding_passes = num_coding_passes
                .checked_add(1)
                .ok_or("HTJ2K packet contribution pass count overflow")?;
        }

        if encoded.num_coding_passes > 1 && segment_layers.get(1) == Some(&layer_idx) {
            data.extend_from_slice(
                encoded
                    .data
                    .get(refinement_start..refinement_end)
                    .ok_or("HTJ2K refinement segment range invalid")?,
            );
            ht_refinement_length = encoded.ht_refinement_length;
            num_coding_passes = num_coding_passes
                .checked_add(encoded.num_coding_passes - 1)
                .ok_or("HTJ2K packet contribution pass count overflow")?;
        }

        contributions.push(CodeBlockPacketData {
            data,
            ht_cleanup_length,
            ht_refinement_length,
            num_coding_passes,
            classic_segment_lengths: Vec::new(),
            num_zero_bitplanes: encoded.num_zero_bitplanes,
            previously_included: false,
            l_block: 3,
            block_coding_mode: BlockCodingMode::HighThroughput,
        });
    }

    Ok(contributions)
}

fn encode_prepared_subbands(
    prepared_subbands: Vec<PreparedEncodeSubband>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<SubbandPrecinct>, &'static str> {
    let block_coding_mode = prepared_subbands
        .iter()
        .find(|subband| !subband.code_blocks.is_empty())
        .map(|subband| subband.block_coding_mode);
    let encoded_blocks = match block_coding_mode {
        Some(BlockCodingMode::HighThroughput) => {
            encode_all_ht_code_blocks(&prepared_subbands, accelerator)?
        }
        Some(BlockCodingMode::Classic) => {
            encode_all_tier1_code_blocks(&prepared_subbands, accelerator)?
        }
        None => Vec::new(),
    };

    let mut encoded_iter = encoded_blocks.into_iter();
    let mut precincts = Vec::with_capacity(prepared_subbands.len());
    for subband in prepared_subbands {
        let mut code_blocks = Vec::with_capacity(subband.code_blocks.len());
        for _ in 0..subband.code_blocks.len() {
            let encoded = encoded_iter
                .next()
                .ok_or("encoded code-block count mismatch")?;
            code_blocks.push(CodeBlockPacketData {
                data: encoded.data,
                ht_cleanup_length: if subband.block_coding_mode == BlockCodingMode::HighThroughput {
                    encoded.ht_cleanup_length
                } else {
                    0
                },
                ht_refinement_length: if subband.block_coding_mode
                    == BlockCodingMode::HighThroughput
                {
                    encoded.ht_refinement_length
                } else {
                    0
                },
                num_coding_passes: encoded.num_coding_passes,
                classic_segment_lengths: Vec::new(),
                num_zero_bitplanes: encoded.num_zero_bitplanes,
                previously_included: false,
                l_block: 3,
                block_coding_mode: subband.block_coding_mode,
            });
        }
        precincts.push(SubbandPrecinct {
            code_blocks,
            num_cbs_x: subband.num_cbs_x,
            num_cbs_y: subband.num_cbs_y,
        });
    }
    if encoded_iter.next().is_some() {
        return Err("encoded code-block count mismatch");
    }

    Ok(precincts)
}

fn encode_all_ht_code_blocks(
    prepared_subbands: &[PreparedEncodeSubband],
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<bitplane_encode::EncodedCodeBlock>, &'static str> {
    if prepared_subbands.iter().all(|subband| {
        subband.code_blocks.is_empty() || subband.preencoded_ht_code_blocks.is_some()
    }) {
        let total_blocks = prepared_subbands
            .iter()
            .map(|subband| subband.code_blocks.len())
            .sum();
        let mut encoded = Vec::with_capacity(total_blocks);
        for subband in prepared_subbands {
            if let Some(blocks) = &subband.preencoded_ht_code_blocks {
                if blocks.len() != subband.code_blocks.len() {
                    return Err("preencoded HT subband code-block count mismatch");
                }
                encoded.extend(
                    blocks
                        .iter()
                        .cloned()
                        .map(ht_encoded_code_block_from_accelerator),
                );
            }
        }
        return Ok(encoded);
    }
    if prepared_subbands
        .iter()
        .any(|subband| subband.preencoded_ht_code_blocks.is_some())
    {
        return Err("mixed preencoded and quantized HT subbands are unsupported");
    }

    let job_coefficients = prepared_subbands
        .iter()
        .flat_map(|subband| subband.code_blocks.iter())
        .map(|block| downcast_i64_coefficients_to_i32(&block.coefficients))
        .collect::<Result<Vec<_>, _>>()?;
    let mut jobs = Vec::with_capacity(job_coefficients.len());
    let mut coefficient_idx = 0usize;
    for subband in prepared_subbands {
        for block in &subband.code_blocks {
            let coefficients = job_coefficients
                .get(coefficient_idx)
                .ok_or("HT coefficient storage count mismatch")?;
            jobs.push(crate::J2kHtCodeBlockEncodeJob {
                coefficients,
                width: block.width,
                height: block.height,
                total_bitplanes: subband.total_bitplanes,
                target_coding_passes: subband.ht_target_coding_passes,
            });
            coefficient_idx = coefficient_idx
                .checked_add(1)
                .ok_or("HT coefficient storage count overflow")?;
        }
    }

    if let Some(encoded) = accelerator.encode_ht_code_blocks(&jobs)? {
        if encoded.len() != jobs.len() {
            return Err("accelerated HT code-block batch length mismatch");
        }
        return Ok(encoded
            .into_iter()
            .map(ht_encoded_code_block_from_accelerator)
            .collect());
    }

    if accelerator.prefer_parallel_cpu_code_block_fallback() {
        if jobs.len() < HT_CPU_PARALLEL_FALLBACK_MIN_JOBS {
            return encode_all_ht_code_blocks_serial_cpu(&jobs);
        }
        return encode_all_ht_code_blocks_parallel(&jobs);
    }

    jobs.iter()
        .map(|job| {
            encode_ht_code_block(
                job.coefficients,
                job.width,
                job.height,
                job.total_bitplanes,
                job.target_coding_passes,
                accelerator,
            )
        })
        .collect()
}

fn encode_all_tier1_code_blocks(
    prepared_subbands: &[PreparedEncodeSubband],
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<bitplane_encode::EncodedCodeBlock>, &'static str> {
    let style = default_public_code_block_style();
    let can_use_i32_jobs = prepared_subbands
        .iter()
        .flat_map(|subband| &subband.code_blocks)
        .all(|block| coefficients_fit_i32(&block.coefficients));
    if !can_use_i32_jobs {
        let mut encoded = Vec::new();
        for subband in prepared_subbands {
            encoded.reserve(subband.code_blocks.len());
            for block in &subband.code_blocks {
                encoded.push(bitplane_encode::encode_code_block_i64(
                    &block.coefficients,
                    block.width,
                    block.height,
                    subband.sub_band_type,
                    subband.total_bitplanes,
                ));
            }
        }
        return Ok(encoded);
    }

    let job_coefficients = prepared_subbands
        .iter()
        .flat_map(|subband| subband.code_blocks.iter())
        .map(|block| downcast_i64_coefficients_to_i32(&block.coefficients))
        .collect::<Result<Vec<_>, _>>()?;
    let mut jobs = Vec::with_capacity(job_coefficients.len());
    let mut coefficient_idx = 0usize;
    for subband in prepared_subbands {
        let public_sub_band_type = public_sub_band_type(subband.sub_band_type);
        for block in &subband.code_blocks {
            let coefficients = job_coefficients
                .get(coefficient_idx)
                .ok_or("classic coefficient storage count mismatch")?;
            jobs.push(J2kTier1CodeBlockEncodeJob {
                coefficients,
                width: block.width,
                height: block.height,
                sub_band_type: public_sub_band_type,
                total_bitplanes: subband.total_bitplanes,
                style,
            });
            coefficient_idx = coefficient_idx
                .checked_add(1)
                .ok_or("classic coefficient storage count overflow")?;
        }
    }

    if let Some(encoded) = accelerator.encode_tier1_code_blocks(&jobs)? {
        if encoded.len() != jobs.len() {
            return Err("accelerated classic code-block batch length mismatch");
        }
        return Ok(encoded
            .into_iter()
            .map(encoded_code_block_from_accelerator)
            .collect());
    }

    if accelerator.prefer_parallel_cpu_code_block_fallback() {
        return encode_all_tier1_code_blocks_parallel(&jobs);
    }

    let mut encoded = Vec::with_capacity(jobs.len());
    for job in &jobs {
        encoded.push(encode_tier1_code_block(
            job.coefficients,
            job.width,
            job.height,
            internal_sub_band_type(job.sub_band_type),
            job.total_bitplanes,
            accelerator,
        )?);
    }
    Ok(encoded)
}

fn encode_all_ht_code_blocks_serial_cpu(
    jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
) -> Result<Vec<bitplane_encode::EncodedCodeBlock>, &'static str> {
    if jobs
        .iter()
        .any(|job| !(1..=3).contains(&job.target_coding_passes))
    {
        return Err("CPU HTJ2K code-block fallback supports at most three HT coding passes");
    }
    jobs.iter()
        .map(|job| {
            ht_block_encode::encode_code_block_with_passes(
                job.coefficients,
                job.width,
                job.height,
                job.total_bitplanes,
                job.target_coding_passes,
            )
        })
        .collect()
}

#[cfg(feature = "parallel")]
fn encode_all_ht_code_blocks_parallel(
    jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
) -> Result<Vec<bitplane_encode::EncodedCodeBlock>, &'static str> {
    if jobs
        .iter()
        .any(|job| !(1..=3).contains(&job.target_coding_passes))
    {
        return Err("CPU HTJ2K code-block fallback supports at most three HT coding passes");
    }
    jobs.par_iter()
        .map(|job| {
            ht_block_encode::encode_code_block_with_passes(
                job.coefficients,
                job.width,
                job.height,
                job.total_bitplanes,
                job.target_coding_passes,
            )
        })
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn encode_all_ht_code_blocks_parallel(
    jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
) -> Result<Vec<bitplane_encode::EncodedCodeBlock>, &'static str> {
    if jobs
        .iter()
        .any(|job| !(1..=3).contains(&job.target_coding_passes))
    {
        return Err("CPU HTJ2K code-block fallback supports at most three HT coding passes");
    }
    jobs.iter()
        .map(|job| {
            ht_block_encode::encode_code_block_with_passes(
                job.coefficients,
                job.width,
                job.height,
                job.total_bitplanes,
                job.target_coding_passes,
            )
        })
        .collect()
}

#[cfg(feature = "parallel")]
fn encode_all_tier1_code_blocks_parallel(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> Result<Vec<bitplane_encode::EncodedCodeBlock>, &'static str> {
    jobs.par_iter()
        .map(|job| {
            Ok(bitplane_encode::encode_code_block(
                job.coefficients,
                job.width,
                job.height,
                internal_sub_band_type(job.sub_band_type),
                job.total_bitplanes,
            ))
        })
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn encode_all_tier1_code_blocks_parallel(
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> Result<Vec<bitplane_encode::EncodedCodeBlock>, &'static str> {
    jobs.iter()
        .map(|job| {
            Ok(bitplane_encode::encode_code_block(
                job.coefficients,
                job.width,
                job.height,
                internal_sub_band_type(job.sub_band_type),
                job.total_bitplanes,
            ))
        })
        .collect()
}

fn encode_ht_code_block(
    coefficients: &[i32],
    width: u32,
    height: u32,
    total_bitplanes: u8,
    target_coding_passes: u8,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<bitplane_encode::EncodedCodeBlock, &'static str> {
    if let Some(encoded) = accelerator.encode_ht_code_block(crate::J2kHtCodeBlockEncodeJob {
        coefficients,
        width,
        height,
        total_bitplanes,
        target_coding_passes,
    })? {
        return Ok(ht_encoded_code_block_from_accelerator(encoded));
    }

    ht_block_encode::encode_code_block_with_passes(
        coefficients,
        width,
        height,
        total_bitplanes,
        target_coding_passes,
    )
}

fn ht_encoded_code_block_from_accelerator(
    encoded: crate::EncodedHtJ2kCodeBlock,
) -> bitplane_encode::EncodedCodeBlock {
    bitplane_encode::EncodedCodeBlock {
        data: encoded.data,
        num_coding_passes: encoded.num_coding_passes,
        num_zero_bitplanes: encoded.num_zero_bitplanes,
        ht_cleanup_length: encoded.cleanup_length,
        ht_refinement_length: encoded.refinement_length,
    }
}

fn encode_tier1_code_block(
    coefficients: &[i32],
    width: u32,
    height: u32,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<bitplane_encode::EncodedCodeBlock, &'static str> {
    if let Some(encoded) = accelerator.encode_tier1_code_block(J2kTier1CodeBlockEncodeJob {
        coefficients,
        width,
        height,
        sub_band_type: public_sub_band_type(sub_band_type),
        total_bitplanes,
        style: default_public_code_block_style(),
    })? {
        return Ok(encoded_code_block_from_accelerator(encoded));
    }

    Ok(bitplane_encode::encode_code_block(
        coefficients,
        width,
        height,
        sub_band_type,
        total_bitplanes,
    ))
}

fn encoded_code_block_from_accelerator(
    encoded: EncodedJ2kCodeBlock,
) -> bitplane_encode::EncodedCodeBlock {
    bitplane_encode::EncodedCodeBlock {
        data: encoded.data,
        num_coding_passes: encoded.number_of_coding_passes,
        num_zero_bitplanes: encoded.missing_bit_planes,
        ht_cleanup_length: 0,
        ht_refinement_length: 0,
    }
}

fn public_sub_band_type(sub_band_type: SubBandType) -> J2kSubBandType {
    match sub_band_type {
        SubBandType::LowLow => J2kSubBandType::LowLow,
        SubBandType::HighLow => J2kSubBandType::HighLow,
        SubBandType::LowHigh => J2kSubBandType::LowHigh,
        SubBandType::HighHigh => J2kSubBandType::HighHigh,
    }
}

fn internal_sub_band_type(sub_band_type: J2kSubBandType) -> SubBandType {
    match sub_band_type {
        J2kSubBandType::LowLow => SubBandType::LowLow,
        J2kSubBandType::HighLow => SubBandType::HighLow,
        J2kSubBandType::LowHigh => SubBandType::LowHigh,
        J2kSubBandType::HighHigh => SubBandType::HighHigh,
    }
}

fn default_public_code_block_style() -> crate::J2kCodeBlockStyle {
    crate::J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: false,
        reset_context_probabilities: false,
        termination_on_each_pass: false,
        vertically_causal_context: false,
        segmentation_symbols: false,
    }
}

/// Convert interleaved pixel bytes to per-component f32 arrays.
pub(crate) fn deinterleave_to_f32(
    pixels: &[u8],
    num_pixels: usize,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
) -> Vec<Vec<f32>> {
    if num_components == 3 && bit_depth == 8 && !signed {
        return deinterleave_rgb8_unsigned_to_f32(pixels, num_pixels);
    }

    let nc = num_components as usize;
    let mut components = vec![vec![0.0f32; num_pixels]; nc];
    let unsigned_offset = if signed {
        0.0
    } else {
        (1_u64 << (u32::from(bit_depth) - 1)) as f32
    };

    let bytes_per_sample = raw_pixel_bytes_per_sample(bit_depth).unwrap_or(2);
    for (i, pixel) in pixels
        .chunks_exact(nc * bytes_per_sample)
        .take(num_pixels)
        .enumerate()
    {
        for (c, component) in components.iter_mut().enumerate().take(nc) {
            let offset = c * bytes_per_sample;
            let raw = read_le_sample_value(&pixel[offset..offset + bytes_per_sample], bit_depth);
            component[i] = if signed {
                sign_extend_sample(raw, bit_depth) as f32
            } else {
                raw as f32 - unsigned_offset
            };
        }
    }

    components
}

fn deinterleave_rgb8_unsigned_to_f32(pixels: &[u8], num_pixels: usize) -> Vec<Vec<f32>> {
    let mut r = Vec::with_capacity(num_pixels);
    let mut g = Vec::with_capacity(num_pixels);
    let mut b = Vec::with_capacity(num_pixels);

    for pixel in pixels.chunks_exact(3).take(num_pixels) {
        r.push(f32::from(pixel[0]) - 128.0);
        g.push(f32::from(pixel[1]) - 128.0);
        b.push(f32::from(pixel[2]) - 128.0);
    }

    vec![r, g, b]
}

/// Calculate the maximum number of decomposition levels for given dimensions.
fn max_decomposition_levels(width: u32, height: u32) -> u8 {
    let min_dim = width.min(height);
    if min_dim <= 1 {
        return 0;
    }
    floor_f32(log2_f32(min_dim as f32)) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PrequantizedHtj2k97CodeBlock;

    fn test_preencoded_subband_payload(marker: u8) -> PreencodedHtj2k97Subband {
        PreencodedHtj2k97Subband {
            sub_band_type: J2kSubBandType::LowLow,
            num_cbs_x: 1,
            num_cbs_y: 1,
            total_bitplanes: 8,
            code_blocks: vec![PreencodedHtj2k97CodeBlock {
                width: 1,
                height: 1,
                encoded: crate::EncodedHtJ2kCodeBlock {
                    data: vec![marker; 8],
                    cleanup_length: 8,
                    refinement_length: 0,
                    num_coding_passes: 1,
                    num_zero_bitplanes: 0,
                },
            }],
        }
    }

    #[test]
    fn prepared_subband_from_owned_preencoded_moves_payload_without_clone() {
        let subband = test_preencoded_subband_payload(7);
        let original_ptr = subband.code_blocks[0].encoded.data.as_ptr() as usize;

        let prepared =
            prepared_subband_from_preencoded_owned(subband).expect("owned preencoded subband");
        let prepared_blocks = prepared
            .preencoded_ht_code_blocks
            .expect("preencoded payloads");

        assert_eq!(prepared_blocks[0].data.as_ptr() as usize, original_ptr);
        assert!(prepared.code_blocks[0].coefficients.is_empty());
    }

    #[test]
    fn compact_preencoded_packetization_borrows_payload_ranges() {
        #[derive(Default)]
        struct RecordingPacketizationAccelerator {
            payload_base: usize,
            observed_offsets: Vec<usize>,
            observed_lengths: Vec<usize>,
        }

        impl crate::J2kEncodeStageAccelerator for RecordingPacketizationAccelerator {
            fn encode_packetization(
                &mut self,
                job: crate::J2kPacketizationEncodeJob<'_>,
            ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
                for code_block in job
                    .resolutions
                    .iter()
                    .flat_map(|resolution| resolution.subbands.iter())
                    .flat_map(|subband| subband.code_blocks.iter())
                    .filter(|code_block| !code_block.data.is_empty())
                {
                    self.observed_offsets
                        .push((code_block.data.as_ptr() as usize) - self.payload_base);
                    self.observed_lengths.push(code_block.data.len());
                }
                Ok(Some(crate::encode_j2k_packetization_scalar(job)?))
            }
        }

        let (preencoded, options) = sample_preencoded_htj2k97_for_test();
        let expected = encode_preencoded_htj2k_97(&preencoded, &options).expect("owned preencoded");
        let mut payload = Vec::new();
        let mut expected_offsets = Vec::new();
        let mut expected_lengths = Vec::new();
        let components = preencoded
            .components
            .iter()
            .map(|component| PreencodedHtj2k97CompactComponent {
                x_rsiz: component.x_rsiz,
                y_rsiz: component.y_rsiz,
                resolutions: component
                    .resolutions
                    .iter()
                    .map(|resolution| PreencodedHtj2k97CompactResolution {
                        subbands: resolution
                            .subbands
                            .iter()
                            .map(|subband| PreencodedHtj2k97CompactSubband {
                                sub_band_type: subband.sub_band_type,
                                num_cbs_x: subband.num_cbs_x,
                                num_cbs_y: subband.num_cbs_y,
                                total_bitplanes: subband.total_bitplanes,
                                code_blocks: subband
                                    .code_blocks
                                    .iter()
                                    .map(|block| {
                                        let start = payload.len();
                                        payload.extend_from_slice(&block.encoded.data);
                                        let end = payload.len();
                                        if start != end {
                                            expected_offsets.push(start);
                                            expected_lengths.push(end - start);
                                        }
                                        PreencodedHtj2k97CompactCodeBlock {
                                            width: block.width,
                                            height: block.height,
                                            payload_range: start..end,
                                            cleanup_length: block.encoded.cleanup_length,
                                            refinement_length: block.encoded.refinement_length,
                                            num_coding_passes: block.encoded.num_coding_passes,
                                            num_zero_bitplanes: block.encoded.num_zero_bitplanes,
                                        }
                                    })
                                    .collect(),
                            })
                            .collect(),
                    })
                    .collect(),
            })
            .collect();
        let compact = PreencodedHtj2k97CompactImage {
            width: preencoded.width,
            height: preencoded.height,
            bit_depth: preencoded.bit_depth,
            signed: preencoded.signed,
            payload,
            components,
        };
        let mut accelerator = RecordingPacketizationAccelerator {
            payload_base: compact.payload.as_ptr() as usize,
            ..Default::default()
        };

        let actual = encode_preencoded_htj2k_97_compact_owned_with_accelerator(
            compact,
            &options,
            &mut accelerator,
        )
        .expect("compact preencoded");

        assert_eq!(actual, expected);
        assert_eq!(accelerator.observed_offsets, expected_offsets);
        assert_eq!(accelerator.observed_lengths, expected_lengths);
    }

    #[test]
    fn test_encode_8bit_gray() {
        let width = 8u32;
        let height = 8u32;
        let pixels: Vec<u8> = (0..64).collect();

        let result = encode(
            &pixels,
            width,
            height,
            1,
            8,
            false,
            &EncodeOptions {
                num_decomposition_levels: 2,
                ..Default::default()
            },
        );

        assert!(result.is_ok());
        let codestream = result.unwrap();
        // Verify SOC marker
        assert_eq!(codestream[0], 0xFF);
        assert_eq!(codestream[1], 0x4F);
        // Verify EOC marker
        let len = codestream.len();
        assert_eq!(codestream[len - 2], 0xFF);
        assert_eq!(codestream[len - 1], 0xD9);
    }

    #[test]
    fn test_encode_16bit_gray() {
        let width = 8u32;
        let height = 8u32;
        let mut pixels = Vec::with_capacity(128);
        for i in 0..64u16 {
            let val = i * 100;
            pixels.extend_from_slice(&val.to_le_bytes());
        }

        let result = encode(
            &pixels,
            width,
            height,
            1,
            16,
            false,
            &EncodeOptions {
                num_decomposition_levels: 2,
                ..Default::default()
            },
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_encode_rgb() {
        let width = 16u32;
        let height = 16u32;
        let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i & 0xFF) as u8).collect();

        let result = encode(
            &pixels,
            width,
            height,
            3,
            8,
            false,
            &EncodeOptions {
                num_decomposition_levels: 3,
                ..Default::default()
            },
        );

        assert!(result.is_ok(), "RGB encode failed: {:?}", result.err());
    }

    #[test]
    fn encode_with_accelerator_calls_lossless_stage_hooks() {
        #[derive(Default)]
        struct CountingAccelerator {
            forward_rct: usize,
            forward_dwt53: usize,
            tier1_code_blocks: usize,
            tier1_code_block_batches: usize,
            tier1_batched_jobs: usize,
            packetization: usize,
            packetization_resolution_count: u32,
            packetization_code_block_count: u32,
            packetization_saw_payload: bool,
        }

        impl crate::J2kEncodeStageAccelerator for CountingAccelerator {
            fn encode_forward_rct(
                &mut self,
                _job: crate::J2kForwardRctJob<'_>,
            ) -> core::result::Result<bool, &'static str> {
                self.forward_rct += 1;
                Ok(false)
            }

            fn encode_forward_dwt53(
                &mut self,
                _job: crate::J2kForwardDwt53Job<'_>,
            ) -> core::result::Result<Option<crate::J2kForwardDwt53Output>, &'static str>
            {
                self.forward_dwt53 += 1;
                Ok(None)
            }

            fn encode_tier1_code_block(
                &mut self,
                _job: crate::J2kTier1CodeBlockEncodeJob<'_>,
            ) -> core::result::Result<Option<crate::EncodedJ2kCodeBlock>, &'static str>
            {
                self.tier1_code_blocks += 1;
                Ok(None)
            }

            fn encode_tier1_code_blocks(
                &mut self,
                jobs: &[crate::J2kTier1CodeBlockEncodeJob<'_>],
            ) -> core::result::Result<Option<Vec<crate::EncodedJ2kCodeBlock>>, &'static str>
            {
                self.tier1_code_block_batches += 1;
                self.tier1_batched_jobs += jobs.len();
                Ok(None)
            }

            fn encode_packetization(
                &mut self,
                job: crate::J2kPacketizationEncodeJob<'_>,
            ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
                self.packetization += 1;
                self.packetization_resolution_count = job.resolution_count;
                self.packetization_code_block_count = job.code_block_count;
                self.packetization_saw_payload = job
                    .resolutions
                    .iter()
                    .flat_map(|resolution| resolution.subbands.iter())
                    .flat_map(|subband| subband.code_blocks.iter())
                    .any(|code_block| !code_block.data.is_empty());
                Ok(None)
            }
        }

        let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| (i & 0xFF) as u8).collect();
        let options = EncodeOptions {
            num_decomposition_levels: 1,
            reversible: true,
            ..EncodeOptions::default()
        };
        let mut accelerator = CountingAccelerator::default();

        let codestream =
            encode_with_accelerator(&pixels, 8, 8, 3, 8, false, &options, &mut accelerator)
                .expect("encode with accelerator hooks");

        assert!(codestream.starts_with(&[0xFF, 0x4F]));
        assert_eq!(accelerator.forward_rct, 1);
        assert_eq!(accelerator.forward_dwt53, 3);
        assert!(accelerator.tier1_code_block_batches > 0);
        assert_eq!(
            accelerator.tier1_code_blocks,
            accelerator.tier1_batched_jobs
        );
        assert_eq!(accelerator.packetization, 1);
        assert_eq!(accelerator.packetization_resolution_count, 6);
        assert_eq!(
            accelerator.packetization_code_block_count,
            u32::try_from(accelerator.tier1_code_blocks).expect("test code-block count fits u32")
        );
        assert!(accelerator.packetization_saw_payload);
    }

    #[test]
    fn cpu_only_accelerator_opts_into_parallel_block_fallback_only_for_native_cpu() {
        #[derive(Default)]
        struct ExternalAccelerator;

        impl crate::J2kEncodeStageAccelerator for ExternalAccelerator {}

        let cpu = crate::CpuOnlyJ2kEncodeStageAccelerator;
        let external = ExternalAccelerator;

        assert!(cpu.prefer_parallel_cpu_code_block_fallback());
        assert!(!external.prefer_parallel_cpu_code_block_fallback());
    }

    #[test]
    fn cpu_parallel_block_fallback_matches_serial_classic_and_htj2k_output() {
        #[derive(Default)]
        struct SerialCpuFallbackAccelerator;

        impl crate::J2kEncodeStageAccelerator for SerialCpuFallbackAccelerator {}

        let pixels = gradient_u8(96, 80);
        for use_ht_block_coding in [false, true] {
            let options = EncodeOptions {
                num_decomposition_levels: 1,
                code_block_width_exp: 2,
                code_block_height_exp: 2,
                use_ht_block_coding,
                ..EncodeOptions::default()
            };
            let parallel = encode(&pixels, 96, 80, 1, 8, false, &options)
                .expect("parallel CPU fallback encode");
            let mut serial_accelerator = SerialCpuFallbackAccelerator;
            let serial = encode_with_accelerator(
                &pixels,
                96,
                80,
                1,
                8,
                false,
                &options,
                &mut serial_accelerator,
            )
            .expect("serial CPU fallback encode");

            assert_eq!(parallel, serial);
        }
    }

    #[test]
    fn precomputed_htj2k53_offers_ht_code_blocks_to_encode_accelerator() {
        let image = sample_precomputed_htj2k53_image();
        let options = EncodeOptions {
            num_decomposition_levels: 1,
            reversible: true,
            guard_bits: 2,
            code_block_width_exp: 2,
            code_block_height_exp: 2,
            ..EncodeOptions::default()
        };
        let mut accelerator = CountingHtEncodeAccelerator::default();

        let encoded =
            encode_precomputed_htj2k_53_with_accelerator(&image, &options, &mut accelerator)
                .expect("precomputed 5/3 encode accepts encode accelerator");

        assert!(encoded.starts_with(&[0xff, 0x4f]));
        assert_eq!(accelerator.forward_dwt53, 0);
        assert_eq!(accelerator.forward_dwt97, 0);
        assert_eq!(accelerator.ht_batches, 1);
        assert!(accelerator.ht_jobs > 0);
        assert_eq!(accelerator.ht_single_blocks, accelerator.ht_jobs);
    }

    #[test]
    fn precomputed_htj2k97_offers_ht_code_blocks_to_encode_accelerator() {
        let image = sample_precomputed_htj2k97_image();
        let options = EncodeOptions {
            num_decomposition_levels: 1,
            reversible: false,
            guard_bits: 2,
            code_block_width_exp: 2,
            code_block_height_exp: 2,
            ..EncodeOptions::default()
        };
        let mut accelerator = CountingHtEncodeAccelerator::default();

        let encoded =
            encode_precomputed_htj2k_97_with_accelerator(&image, &options, &mut accelerator)
                .expect("precomputed 9/7 encode accepts encode accelerator");

        assert!(encoded.starts_with(&[0xff, 0x4f]));
        assert_eq!(accelerator.forward_dwt53, 0);
        assert_eq!(accelerator.forward_dwt97, 0);
        assert_eq!(accelerator.ht_batches, 1);
        assert!(accelerator.ht_jobs > 0);
        assert_eq!(accelerator.ht_single_blocks, accelerator.ht_jobs);
    }

    #[test]
    fn precomputed_dwt_geometry_validation_rejects_recursive_mismatch_for_both_filters() {
        let mut dwt53 = sample_precomputed_htj2k53_image();
        dwt53.components[0].dwt.levels[0].low_width += 1;
        assert_eq!(
            validate_precomputed_dwt_geometry(&dwt53),
            Err("precomputed DWT recursive geometry mismatch")
        );

        let mut dwt97 = sample_precomputed_htj2k97_image();
        dwt97.components[0].dwt.levels[0].low_width += 1;
        assert_eq!(
            validate_precomputed_dwt97_geometry(&dwt97),
            Err("precomputed DWT recursive geometry mismatch")
        );
    }

    #[test]
    fn prequantized_htj2k97_offers_ht_code_blocks_to_encode_accelerator() {
        let image = sample_precomputed_htj2k97_image();
        let options = EncodeOptions {
            num_decomposition_levels: 1,
            reversible: false,
            guard_bits: 2,
            code_block_width_exp: 2,
            code_block_height_exp: 2,
            ..EncodeOptions::default()
        };
        let prequantized = prequantized_htj2k97_image_from_precomputed_for_test(&image, &options)
            .expect("test prequantized image");
        let mut accelerator = CountingHtEncodeAccelerator::default();

        let encoded = encode_prequantized_htj2k_97_with_accelerator(
            &prequantized,
            &options,
            &mut accelerator,
        )
        .expect("prequantized 9/7 encode accepts encode accelerator");

        assert!(encoded.starts_with(&[0xff, 0x4f]));
        assert_eq!(accelerator.forward_dwt53, 0);
        assert_eq!(accelerator.forward_dwt97, 0);
        assert_eq!(accelerator.ht_batches, 1);
        assert!(accelerator.ht_jobs > 0);
        assert_eq!(accelerator.ht_single_blocks, accelerator.ht_jobs);
    }

    #[test]
    fn precomputed_htj2k97_batch_offers_all_ht_code_blocks_in_one_accelerator_call() {
        let image = sample_precomputed_htj2k97_image();
        let options = EncodeOptions {
            num_decomposition_levels: 1,
            reversible: false,
            guard_bits: 2,
            code_block_width_exp: 2,
            code_block_height_exp: 2,
            ..EncodeOptions::default()
        };
        let mut accelerator = CountingHtEncodeAccelerator::default();

        let encoded = encode_precomputed_htj2k_97_batch_with_accelerator(
            &[image.clone(), image],
            &options,
            &mut accelerator,
        )
        .expect("batch precomputed 9/7 encode accepts encode accelerator");

        assert_eq!(encoded.len(), 2);
        assert!(encoded
            .iter()
            .all(|codestream| codestream.starts_with(&[0xff, 0x4f])));
        assert_eq!(accelerator.forward_dwt53, 0);
        assert_eq!(accelerator.forward_dwt97, 0);
        assert_eq!(accelerator.ht_batches, 1);
        assert!(accelerator.ht_jobs > 0);
        assert_eq!(accelerator.ht_single_blocks, accelerator.ht_jobs);
    }

    #[derive(Default)]
    struct CountingHtEncodeAccelerator {
        forward_dwt53: usize,
        forward_dwt97: usize,
        ht_batches: usize,
        ht_jobs: usize,
        ht_single_blocks: usize,
    }

    impl crate::J2kEncodeStageAccelerator for CountingHtEncodeAccelerator {
        fn encode_forward_dwt53(
            &mut self,
            _job: crate::J2kForwardDwt53Job<'_>,
        ) -> core::result::Result<Option<crate::J2kForwardDwt53Output>, &'static str> {
            self.forward_dwt53 += 1;
            Ok(None)
        }

        fn encode_forward_dwt97(
            &mut self,
            _job: crate::J2kForwardDwt97Job<'_>,
        ) -> core::result::Result<Option<crate::J2kForwardDwt97Output>, &'static str> {
            self.forward_dwt97 += 1;
            Ok(None)
        }

        fn encode_ht_code_blocks(
            &mut self,
            jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
        ) -> core::result::Result<Option<Vec<crate::EncodedHtJ2kCodeBlock>>, &'static str> {
            self.ht_batches += 1;
            self.ht_jobs += jobs.len();
            Ok(None)
        }

        fn encode_ht_code_block(
            &mut self,
            _job: crate::J2kHtCodeBlockEncodeJob<'_>,
        ) -> core::result::Result<Option<crate::EncodedHtJ2kCodeBlock>, &'static str> {
            self.ht_single_blocks += 1;
            Ok(None)
        }
    }

    #[test]
    fn prepare_subband_uses_fused_ht_subband_without_host_quantized_codeblocks() {
        #[derive(Default)]
        struct FusedHtSubbandAccelerator {
            subband_calls: usize,
            quantize_calls: usize,
            ht_batch_calls: usize,
        }

        impl crate::J2kEncodeStageAccelerator for FusedHtSubbandAccelerator {
            fn encode_ht_subband(
                &mut self,
                job: crate::J2kHtSubbandEncodeJob<'_>,
            ) -> core::result::Result<Option<Vec<crate::EncodedHtJ2kCodeBlock>>, &'static str>
            {
                self.subband_calls += 1;
                let count = (job.width.div_ceil(job.code_block_width) as usize)
                    .checked_mul(job.height.div_ceil(job.code_block_height) as usize)
                    .ok_or("test code-block count overflow")?;
                Ok(Some(
                    (0..count)
                        .map(|idx| crate::EncodedHtJ2kCodeBlock {
                            data: vec![u8::try_from(idx).expect("test block index fits"), 0],
                            cleanup_length: 2,
                            refinement_length: 0,
                            num_coding_passes: 1,
                            num_zero_bitplanes: 0,
                        })
                        .collect(),
                ))
            }

            fn encode_quantize_subband(
                &mut self,
                _job: crate::J2kQuantizeSubbandJob<'_>,
            ) -> core::result::Result<Option<Vec<i32>>, &'static str> {
                self.quantize_calls += 1;
                Ok(None)
            }

            fn encode_ht_code_blocks(
                &mut self,
                _jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
            ) -> core::result::Result<Option<Vec<crate::EncodedHtJ2kCodeBlock>>, &'static str>
            {
                self.ht_batch_calls += 1;
                Ok(None)
            }
        }

        let coefficients = vec![0.0; 16];
        let mut accelerator = FusedHtSubbandAccelerator::default();
        let prepared = prepare_subband(
            &coefficients,
            4,
            4,
            &QuantStepSize {
                exponent: 8,
                mantissa: 0,
            },
            8,
            2,
            true,
            BlockCodingMode::HighThroughput,
            2,
            2,
            SubBandType::LowLow,
            0,
            &[],
            1,
            1,
            &mut accelerator,
        )
        .expect("fused HT subband prepare");

        assert_eq!(accelerator.subband_calls, 1);
        assert_eq!(accelerator.quantize_calls, 0);
        assert!(prepared.preencoded_ht_code_blocks.is_some());
        assert!(prepared
            .code_blocks
            .iter()
            .all(|block| block.coefficients.is_empty()));

        let precincts = encode_prepared_subbands(vec![prepared], &mut accelerator)
            .expect("preencoded HT subband packet data");

        assert_eq!(accelerator.ht_batch_calls, 0);
        assert_eq!(precincts[0].code_blocks.len(), 4);
        assert_eq!(precincts[0].code_blocks[2].data, vec![2, 0]);
    }

    #[test]
    fn ht_target_coding_passes_tracks_ht_quality_layers() {
        let mut options = EncodeOptions {
            use_ht_block_coding: true,
            reversible: false,
            num_layers: 1,
            ..EncodeOptions::default()
        };

        assert_eq!(ht_target_coding_passes_for_options(&options), 1);

        options.num_layers = 2;
        assert_eq!(ht_target_coding_passes_for_options(&options), 2);

        options.num_layers = 3;
        assert_eq!(ht_target_coding_passes_for_options(&options), 3);

        options.num_layers = 4;
        assert_eq!(ht_target_coding_passes_for_options(&options), 3);

        options.reversible = true;
        assert_eq!(ht_target_coding_passes_for_options(&options), 3);

        options.reversible = false;
        options.use_ht_block_coding = false;
        assert_eq!(ht_target_coding_passes_for_options(&options), 1);
    }

    #[test]
    fn packet_header_validation_allows_chunked_ppm_and_ppt_payloads() {
        const MARKER_PAYLOAD_LIMIT: usize = u16::MAX as usize - 3;
        let ppm_headers = vec![vec![0_u8; MARKER_PAYLOAD_LIMIT - 2], vec![1_u8; 1]];
        let ppt_headers = vec![vec![2_u8; MARKER_PAYLOAD_LIMIT + 1]];

        validate_packet_header_marker_payloads(true, false, &[&ppm_headers])
            .expect("chunked PPM payload should validate");
        validate_packet_header_marker_payloads(false, true, &[&ppt_headers])
            .expect("chunked PPT payload should validate");
    }

    #[test]
    fn ht_cpu_fallback_encodes_two_pass_sigprop_refinement() {
        let coefficients: Vec<i32> = (0usize..64 * 64)
            .map(|index| {
                let value = ((((index * 31) ^ (index / 3)) & 0x00ff) as i32 - 127) * 2;
                if index.is_multiple_of(11) {
                    0
                } else {
                    value
                }
            })
            .collect();
        let jobs = [crate::J2kHtCodeBlockEncodeJob {
            coefficients: &coefficients,
            width: 64,
            height: 64,
            total_bitplanes: 10,
            target_coding_passes: 2,
        }];

        let encoded = encode_all_ht_code_blocks_serial_cpu(&jobs).expect("two-pass CPU HT encode");

        assert_eq!(encoded.len(), 1);
        assert_eq!(encoded[0].num_coding_passes, 2);
        assert_eq!(encoded[0].ht_refinement_length, 48);
        assert_eq!(
            encoded[0].data.len(),
            encoded[0].ht_cleanup_length as usize + encoded[0].ht_refinement_length as usize
        );
        assert!(encoded[0].data[encoded[0].ht_cleanup_length as usize..]
            .iter()
            .all(|byte| *byte == 0));

        let segments = crate::j2c::ht_block_decode::HtCodeBlockSegments::from_combined_payload(
            &encoded[0].data,
            encoded[0].ht_cleanup_length,
            encoded[0].ht_refinement_length,
        )
        .expect("split HT segments");
        let mut decoded = vec![0u32; coefficients.len()];
        crate::j2c::ht_block_decode::decode_segments_validated(
            &segments,
            encoded[0].num_zero_bitplanes,
            10,
            encoded[0].num_coding_passes,
            false,
            true,
            &mut decoded,
            64,
            64,
            64,
        )
        .expect("decode two-pass HT block");
        let decoded_i32 = decoded
            .into_iter()
            .map(|value| crate::j2c::ht_block_decode::coefficient_to_i32(value, 10))
            .collect::<Vec<_>>();
        let max_abs_delta = decoded_i32
            .iter()
            .zip(&coefficients)
            .map(|(actual, expected)| actual.abs_diff(*expected))
            .max()
            .unwrap_or(0);

        assert!(
            max_abs_delta <= 1,
            "two-pass HT sigprop decode must stay within one coefficient LSB"
        );
    }

    #[test]
    fn ht_cpu_fallback_sigprop_refinement_encodes_new_significance_bits() {
        let mut coefficients = vec![0_i32; 8 * 8];
        for row in 0..8 {
            coefficients[row * 8] = 3;
            coefficients[row * 8 + 1] = 1;
            coefficients[row * 8 + 2] = -1;
        }
        let jobs = [crate::J2kHtCodeBlockEncodeJob {
            coefficients: &coefficients,
            width: 8,
            height: 8,
            total_bitplanes: 4,
            target_coding_passes: 2,
        }];

        let encoded = encode_all_ht_code_blocks_serial_cpu(&jobs).expect("two-pass CPU HT encode");

        assert_eq!(encoded[0].num_coding_passes, 2);
        assert!(encoded[0].ht_refinement_length > 0);
        assert!(
            encoded[0].data[encoded[0].ht_cleanup_length as usize..]
                .iter()
                .any(|byte| *byte != 0),
            "sigprop refinement should encode new significance/sign bits"
        );

        let segments = crate::j2c::ht_block_decode::HtCodeBlockSegments::from_combined_payload(
            &encoded[0].data,
            encoded[0].ht_cleanup_length,
            encoded[0].ht_refinement_length,
        )
        .expect("split HT segments");
        let mut decoded = vec![0u32; coefficients.len()];
        crate::j2c::ht_block_decode::decode_segments_validated(
            &segments,
            encoded[0].num_zero_bitplanes,
            4,
            encoded[0].num_coding_passes,
            false,
            true,
            &mut decoded,
            8,
            8,
            8,
        )
        .expect("decode two-pass HT block");
        let decoded_i32 = decoded
            .into_iter()
            .map(|value| crate::j2c::ht_block_decode::coefficient_to_i32(value, 4))
            .collect::<Vec<_>>();

        assert_eq!(decoded_i32, coefficients);
    }

    #[test]
    fn ht_cpu_fallback_encodes_three_pass_magref_refinement() {
        let mut coefficients = vec![0_i32; 8 * 8];
        for row in 0..8 {
            let base = row * 8;
            coefficients[base] = 2;
            coefficients[base + 1] = 3;
            coefficients[base + 2] = 1;
            coefficients[base + 3] = -1;
            coefficients[base + 4] = -2;
            coefficients[base + 5] = -3;
        }
        let jobs = [crate::J2kHtCodeBlockEncodeJob {
            coefficients: &coefficients,
            width: 8,
            height: 8,
            total_bitplanes: 4,
            target_coding_passes: 3,
        }];

        let encoded =
            encode_all_ht_code_blocks_serial_cpu(&jobs).expect("three-pass CPU HT encode");

        assert_eq!(encoded[0].num_coding_passes, 3);
        assert!(encoded[0].ht_refinement_length > 0);

        let segments = crate::j2c::ht_block_decode::HtCodeBlockSegments::from_combined_payload(
            &encoded[0].data,
            encoded[0].ht_cleanup_length,
            encoded[0].ht_refinement_length,
        )
        .expect("split HT segments");
        let mut decoded = vec![0u32; coefficients.len()];
        crate::j2c::ht_block_decode::decode_segments_validated(
            &segments,
            encoded[0].num_zero_bitplanes,
            4,
            encoded[0].num_coding_passes,
            false,
            true,
            &mut decoded,
            8,
            8,
            8,
        )
        .expect("decode three-pass HT block");
        let decoded_i32 = decoded
            .into_iter()
            .map(|value| crate::j2c::ht_block_decode::coefficient_to_i32(value, 4))
            .collect::<Vec<_>>();

        assert_eq!(decoded_i32, coefficients);
    }

    #[test]
    fn ht_cpu_fallback_rejects_unsupported_refinement_pass_count() {
        let coefficients = vec![1_i32; 64 * 64];
        let jobs = [crate::J2kHtCodeBlockEncodeJob {
            coefficients: &coefficients,
            width: 64,
            height: 64,
            total_bitplanes: 2,
            target_coding_passes: 4,
        }];

        let err = encode_all_ht_code_blocks_serial_cpu(&jobs)
            .expect_err("CPU HT encode must reject unsupported pass requests");

        assert!(err.contains("at most three HT coding passes"));
    }

    #[test]
    fn ht_cpu_parallel_fallback_threshold_matches_parallel_output() {
        assert_eq!(HT_CPU_PARALLEL_FALLBACK_MIN_JOBS, 4);

        let blocks: Vec<Vec<i32>> = (0..HT_CPU_PARALLEL_FALLBACK_MIN_JOBS)
            .map(|seed| {
                (0usize..64 * 64)
                    .map(|index| {
                        let value = (((index * 31) ^ (seed * 17)) & 0x01ff) as i32 - 255;
                        if (index + seed).is_multiple_of(11) {
                            0
                        } else {
                            value
                        }
                    })
                    .collect()
            })
            .collect();
        let jobs: Vec<_> = blocks
            .iter()
            .map(|coefficients| crate::J2kHtCodeBlockEncodeJob {
                coefficients,
                width: 64,
                height: 64,
                total_bitplanes: 10,
                target_coding_passes: 1,
            })
            .collect();

        let serial =
            encode_all_ht_code_blocks_serial_cpu(&jobs[..HT_CPU_PARALLEL_FALLBACK_MIN_JOBS - 1])
                .expect("serial tiny HT encode");
        let parallel =
            encode_all_ht_code_blocks_parallel(&jobs[..HT_CPU_PARALLEL_FALLBACK_MIN_JOBS])
                .expect("parallel HT encode");
        let serial_threshold =
            encode_all_ht_code_blocks_serial_cpu(&jobs[..HT_CPU_PARALLEL_FALLBACK_MIN_JOBS])
                .expect("serial threshold HT encode");

        assert_eq!(serial.len(), HT_CPU_PARALLEL_FALLBACK_MIN_JOBS - 1);
        assert_eq!(parallel.len(), HT_CPU_PARALLEL_FALLBACK_MIN_JOBS);
        assert_eq!(serial_threshold.len(), parallel.len());
        for (serial, parallel) in serial_threshold.iter().zip(&parallel) {
            assert_eq!(serial.data, parallel.data);
            assert_eq!(serial.num_coding_passes, parallel.num_coding_passes);
            assert_eq!(serial.num_zero_bitplanes, parallel.num_zero_bitplanes);
        }
    }

    #[test]
    fn code_block_extraction_copies_partial_edge_blocks_rowwise() {
        let quantized: Vec<i32> = (0..20).collect();

        let block = copy_code_block_coefficients(&quantized, 5, 3, 1, 2, 3);

        assert_eq!(block, vec![8, 9, 13, 14, 18, 19]);
    }

    #[test]
    fn test_encode_lossy() {
        let pixels: Vec<u8> = (0..64).collect();

        let result = encode(
            &pixels,
            8,
            8,
            1,
            8,
            false,
            &EncodeOptions {
                num_decomposition_levels: 2,
                reversible: false,
                guard_bits: 2,
                ..Default::default()
            },
        );

        assert!(result.is_ok());
    }

    #[test]
    fn prequantized_htj2k97_matches_precomputed_dwt97_codestream() {
        let image = sample_precomputed_htj2k97_image();
        let options = EncodeOptions {
            num_decomposition_levels: 1,
            reversible: false,
            guard_bits: 2,
            code_block_width_exp: 2,
            code_block_height_exp: 2,
            ..EncodeOptions::default()
        };

        let precomputed =
            encode_precomputed_htj2k_97(&image, &options).expect("precomputed DWT encode");
        let prequantized = prequantized_htj2k97_image_from_precomputed_for_test(&image, &options)
            .expect("test prequantized image");
        let direct =
            encode_prequantized_htj2k_97(&prequantized, &options).expect("prequantized encode");

        assert_eq!(direct, precomputed);
    }

    #[test]
    fn preencoded_htj2k97_matches_prequantized_codestream() {
        let image = sample_precomputed_htj2k97_image();
        let options = EncodeOptions {
            num_decomposition_levels: 1,
            reversible: false,
            guard_bits: 2,
            code_block_width_exp: 2,
            code_block_height_exp: 2,
            ..EncodeOptions::default()
        };
        let prequantized = prequantized_htj2k97_image_from_precomputed_for_test(&image, &options)
            .expect("test prequantized image");
        let expected =
            encode_prequantized_htj2k_97(&prequantized, &options).expect("prequantized encode");
        let preencoded = preencoded_htj2k97_image_from_prequantized_for_test(&prequantized)
            .expect("test preencoded image");
        let actual = encode_preencoded_htj2k_97(&preencoded, &options).expect("preencoded encode");

        assert_eq!(actual, expected);
    }

    #[test]
    fn preencoded_htj2k97_preserves_refinement_segments_in_packet_body() {
        let options = EncodeOptions {
            num_decomposition_levels: 0,
            reversible: false,
            guard_bits: 2,
            code_block_width_exp: 2,
            code_block_height_exp: 2,
            ..EncodeOptions::default()
        };
        let guard_bits = options.guard_bits.max(2);
        let step_sizes = quantize::compute_step_sizes_with_irreversible_profile(
            8,
            0,
            false,
            guard_bits,
            options.irreversible_quantization_scale,
            options.irreversible_quantization_subband_scales,
        );
        let total_bitplanes = guard_bits
            .saturating_add(step_sizes[0].exponent as u8)
            .saturating_sub(1);
        let payload = [0x12, 0x34, 0x56, 0x78];
        let image = PreencodedHtj2k97Image {
            width: 1,
            height: 1,
            bit_depth: 8,
            signed: false,
            components: vec![PreencodedHtj2k97Component {
                x_rsiz: 1,
                y_rsiz: 1,
                resolutions: vec![PreencodedHtj2k97Resolution {
                    subbands: vec![PreencodedHtj2k97Subband {
                        sub_band_type: J2kSubBandType::LowLow,
                        num_cbs_x: 1,
                        num_cbs_y: 1,
                        total_bitplanes,
                        code_blocks: vec![PreencodedHtj2k97CodeBlock {
                            width: 1,
                            height: 1,
                            encoded: EncodedHtJ2kCodeBlock {
                                data: payload.to_vec(),
                                cleanup_length: 2,
                                refinement_length: 2,
                                num_coding_passes: 3,
                                num_zero_bitplanes: 0,
                            },
                        }],
                    }],
                }],
            }],
        };

        let codestream =
            encode_preencoded_htj2k_97(&image, &options).expect("preencoded refinement encode");
        let eoc = codestream
            .windows(2)
            .rposition(|marker| marker == [0xff, crate::j2c::codestream::markers::EOC])
            .expect("EOC marker");

        assert_eq!(&codestream[eoc - payload.len()..eoc], payload);
    }

    #[test]
    fn preencoded_htj2k97_rejects_empty_block_with_wrong_zero_bitplanes() {
        let (mut image, options) = sample_preencoded_htj2k97_for_test();
        let block = &mut image.components[0].resolutions[0].subbands[0].code_blocks[0];
        block.encoded = EncodedHtJ2kCodeBlock {
            data: Vec::new(),
            cleanup_length: 0,
            refinement_length: 0,
            num_coding_passes: 0,
            num_zero_bitplanes: 0,
        };

        let error = encode_preencoded_htj2k_97(&image, &options)
            .expect_err("invalid all-zero block metadata must be rejected");

        assert_eq!(error, "empty HTJ2K code-block zero-bitplane count mismatch");
    }

    #[test]
    fn preencoded_htj2k97_rejects_coded_block_with_too_many_zero_bitplanes() {
        let (mut image, options) = sample_preencoded_htj2k97_for_test();
        let subband = &mut image.components[0].resolutions[0].subbands[0];
        subband.code_blocks[0].encoded.num_zero_bitplanes = subband.total_bitplanes;

        let error = encode_preencoded_htj2k_97(&image, &options)
            .expect_err("coded block with no coded bitplanes must be rejected");

        assert_eq!(error, "HTJ2K code-block zero-bitplane count out of range");
    }

    #[cfg(feature = "std")]
    #[test]
    fn preencoded_htj2k97_rejects_too_many_coding_passes_without_panic() {
        let (mut image, options) = sample_preencoded_htj2k97_for_test();
        image.components[0].resolutions[0].subbands[0].code_blocks[0]
            .encoded
            .num_coding_passes = 165;

        let result = std::panic::catch_unwind(|| encode_preencoded_htj2k_97(&image, &options));

        assert!(result.is_ok(), "invalid coding pass count must not panic");
        assert_eq!(
            result.expect("catch_unwind returned checked result"),
            Err("HTJ2K code-block coding pass count out of range")
        );
    }

    #[test]
    fn prequantized_htj2k97_accepts_empty_high_subbands() {
        let options = EncodeOptions {
            num_decomposition_levels: 1,
            reversible: false,
            guard_bits: 2,
            code_block_width_exp: 2,
            code_block_height_exp: 2,
            ..EncodeOptions::default()
        };
        let image = PrequantizedHtj2k97Image {
            width: 1,
            height: 1,
            bit_depth: 8,
            signed: false,
            components: vec![PrequantizedHtj2k97Component {
                x_rsiz: 1,
                y_rsiz: 1,
                resolutions: vec![
                    PrequantizedHtj2k97Resolution {
                        subbands: vec![PrequantizedHtj2k97Subband {
                            sub_band_type: J2kSubBandType::LowLow,
                            num_cbs_x: 1,
                            num_cbs_y: 1,
                            total_bitplanes: 11,
                            code_blocks: vec![PrequantizedHtj2k97CodeBlock {
                                coefficients: vec![0],
                                width: 1,
                                height: 1,
                            }],
                        }],
                    },
                    PrequantizedHtj2k97Resolution {
                        subbands: vec![
                            empty_prequantized_subband(J2kSubBandType::HighLow),
                            empty_prequantized_subband(J2kSubBandType::LowHigh),
                            empty_prequantized_subband(J2kSubBandType::HighHigh),
                        ],
                    },
                ],
            }],
        };

        let encoded =
            encode_prequantized_htj2k_97(&image, &options).expect("empty high subbands encode");

        assert!(encoded.starts_with(&[0xff, 0x4f]));
    }

    fn empty_prequantized_subband(sub_band_type: J2kSubBandType) -> PrequantizedHtj2k97Subband {
        PrequantizedHtj2k97Subband {
            sub_band_type,
            num_cbs_x: 0,
            num_cbs_y: 0,
            total_bitplanes: 0,
            code_blocks: Vec::new(),
        }
    }

    fn sample_precomputed_htj2k97_image() -> PrecomputedHtj2k97Image {
        let width = 17u32;
        let height = 13u32;
        let low_width = width.div_ceil(2);
        let low_height = height.div_ceil(2);
        let high_width = width / 2;
        let high_height = height / 2;

        PrecomputedHtj2k97Image {
            width,
            height,
            bit_depth: 8,
            signed: false,
            components: vec![PrecomputedHtj2k97Component {
                x_rsiz: 1,
                y_rsiz: 1,
                dwt: J2kForwardDwt97Output {
                    ll: sample_f32_coefficients(low_width * low_height, 0.25),
                    ll_width: low_width,
                    ll_height: low_height,
                    levels: vec![J2kForwardDwt97Level {
                        hl: sample_f32_coefficients(high_width * low_height, -0.75),
                        lh: sample_f32_coefficients(low_width * high_height, 1.25),
                        hh: sample_f32_coefficients(high_width * high_height, -1.5),
                        width,
                        height,
                        low_width,
                        low_height,
                        high_width,
                        high_height,
                    }],
                },
            }],
        }
    }

    fn sample_precomputed_htj2k53_image() -> PrecomputedHtj2k53Image {
        let width = 17u32;
        let height = 13u32;
        let low_width = width.div_ceil(2);
        let low_height = height.div_ceil(2);
        let high_width = width / 2;
        let high_height = height / 2;

        PrecomputedHtj2k53Image {
            width,
            height,
            bit_depth: 8,
            signed: false,
            components: vec![PrecomputedHtj2k53Component {
                x_rsiz: 1,
                y_rsiz: 1,
                dwt: J2kForwardDwt53Output {
                    ll: sample_f32_coefficients(low_width * low_height, 0.0),
                    ll_width: low_width,
                    ll_height: low_height,
                    levels: vec![J2kForwardDwt53Level {
                        hl: sample_f32_coefficients(high_width * low_height, -2.0),
                        lh: sample_f32_coefficients(low_width * high_height, 2.0),
                        hh: sample_f32_coefficients(high_width * high_height, -4.0),
                        width,
                        height,
                        low_width,
                        low_height,
                        high_width,
                        high_height,
                    }],
                },
            }],
        }
    }

    fn sample_f32_coefficients(len: u32, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|idx| ((idx % 17) as f32 - 8.0) * 0.5 + offset)
            .collect()
    }

    fn prequantized_htj2k97_image_from_precomputed_for_test(
        image: &PrecomputedHtj2k97Image,
        options: &EncodeOptions,
    ) -> Result<PrequantizedHtj2k97Image, &'static str> {
        let guard_bits = options.guard_bits.max(2);
        let step_sizes = quantize::compute_step_sizes_with_irreversible_profile(
            image.bit_depth,
            1,
            false,
            guard_bits,
            options.irreversible_quantization_scale,
            options.irreversible_quantization_subband_scales,
        );
        let cb_width = 1u32 << (options.code_block_width_exp + 2);
        let cb_height = 1u32 << (options.code_block_height_exp + 2);

        let components = image
            .components
            .iter()
            .map(|component| {
                let mut resolutions = Vec::with_capacity(component.dwt.levels.len() + 1);
                resolutions.push(PrequantizedHtj2k97Resolution {
                    subbands: vec![prequantized_subband_for_test(
                        &component.dwt.ll,
                        component.dwt.ll_width,
                        component.dwt.ll_height,
                        SubBandType::LowLow,
                        &step_sizes[0],
                        image.bit_depth,
                        guard_bits,
                        cb_width,
                        cb_height,
                    )?],
                });

                for (level_index, level) in component.dwt.levels.iter().enumerate() {
                    let step_base = 1 + level_index * 3;
                    resolutions.push(PrequantizedHtj2k97Resolution {
                        subbands: vec![
                            prequantized_subband_for_test(
                                &level.hl,
                                level.high_width,
                                level.low_height,
                                SubBandType::HighLow,
                                &step_sizes[step_base],
                                image.bit_depth,
                                guard_bits,
                                cb_width,
                                cb_height,
                            )?,
                            prequantized_subband_for_test(
                                &level.lh,
                                level.low_width,
                                level.high_height,
                                SubBandType::LowHigh,
                                &step_sizes[step_base + 1],
                                image.bit_depth,
                                guard_bits,
                                cb_width,
                                cb_height,
                            )?,
                            prequantized_subband_for_test(
                                &level.hh,
                                level.high_width,
                                level.high_height,
                                SubBandType::HighHigh,
                                &step_sizes[step_base + 2],
                                image.bit_depth,
                                guard_bits,
                                cb_width,
                                cb_height,
                            )?,
                        ],
                    });
                }

                Ok(PrequantizedHtj2k97Component {
                    x_rsiz: component.x_rsiz,
                    y_rsiz: component.y_rsiz,
                    resolutions,
                })
            })
            .collect::<Result<Vec<_>, &'static str>>()?;

        Ok(PrequantizedHtj2k97Image {
            width: image.width,
            height: image.height,
            bit_depth: image.bit_depth,
            signed: image.signed,
            components,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn prequantized_subband_for_test(
        coefficients: &[f32],
        width: u32,
        height: u32,
        sub_band_type: SubBandType,
        step_size: &QuantStepSize,
        bit_depth: u8,
        guard_bits: u8,
        cb_width: u32,
        cb_height: u32,
    ) -> Result<PrequantizedHtj2k97Subband, &'static str> {
        let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
        let prepared = prepare_subband(
            coefficients,
            width,
            height,
            step_size,
            bit_depth,
            guard_bits,
            false,
            BlockCodingMode::HighThroughput,
            cb_width,
            cb_height,
            sub_band_type,
            0,
            &[],
            1,
            1,
            &mut accelerator,
        )?;

        Ok(PrequantizedHtj2k97Subband {
            sub_band_type: public_sub_band_type(sub_band_type),
            num_cbs_x: prepared.num_cbs_x,
            num_cbs_y: prepared.num_cbs_y,
            total_bitplanes: prepared.total_bitplanes,
            code_blocks: prepared
                .code_blocks
                .into_iter()
                .map(|block| {
                    Ok(PrequantizedHtj2k97CodeBlock {
                        coefficients: downcast_i64_coefficients_to_i32(&block.coefficients)?,
                        width: block.width,
                        height: block.height,
                    })
                })
                .collect::<Result<Vec<_>, &'static str>>()?,
        })
    }

    fn preencoded_htj2k97_image_from_prequantized_for_test(
        image: &PrequantizedHtj2k97Image,
    ) -> Result<PreencodedHtj2k97Image, &'static str> {
        let components = image
            .components
            .iter()
            .map(|component| {
                Ok(PreencodedHtj2k97Component {
                    x_rsiz: component.x_rsiz,
                    y_rsiz: component.y_rsiz,
                    resolutions: component
                        .resolutions
                        .iter()
                        .map(|resolution| {
                            Ok(PreencodedHtj2k97Resolution {
                                subbands: resolution
                                    .subbands
                                    .iter()
                                    .map(preencoded_subband_from_prequantized_for_test)
                                    .collect::<Result<Vec<_>, &'static str>>()?,
                            })
                        })
                        .collect::<Result<Vec<_>, &'static str>>()?,
                })
            })
            .collect::<Result<Vec<_>, &'static str>>()?;

        Ok(PreencodedHtj2k97Image {
            width: image.width,
            height: image.height,
            bit_depth: image.bit_depth,
            signed: image.signed,
            components,
        })
    }

    fn sample_preencoded_htj2k97_for_test() -> (PreencodedHtj2k97Image, EncodeOptions) {
        let image = sample_precomputed_htj2k97_image();
        let options = EncodeOptions {
            num_decomposition_levels: 1,
            reversible: false,
            guard_bits: 2,
            code_block_width_exp: 2,
            code_block_height_exp: 2,
            ..EncodeOptions::default()
        };
        let prequantized = prequantized_htj2k97_image_from_precomputed_for_test(&image, &options)
            .expect("test prequantized image");
        let preencoded = preencoded_htj2k97_image_from_prequantized_for_test(&prequantized)
            .expect("test preencoded image");
        (preencoded, options)
    }

    fn preencoded_subband_from_prequantized_for_test(
        subband: &PrequantizedHtj2k97Subband,
    ) -> Result<PreencodedHtj2k97Subband, &'static str> {
        let code_blocks = subband
            .code_blocks
            .iter()
            .map(|block| {
                let encoded = ht_block_encode::encode_code_block(
                    &block.coefficients,
                    block.width,
                    block.height,
                    subband.total_bitplanes,
                )?;
                Ok(PreencodedHtj2k97CodeBlock {
                    width: block.width,
                    height: block.height,
                    encoded: EncodedHtJ2kCodeBlock {
                        data: encoded.data,
                        cleanup_length: encoded.ht_cleanup_length,
                        refinement_length: encoded.ht_refinement_length,
                        num_coding_passes: encoded.num_coding_passes,
                        num_zero_bitplanes: encoded.num_zero_bitplanes,
                    },
                })
            })
            .collect::<Result<Vec<_>, &'static str>>()?;

        Ok(PreencodedHtj2k97Subband {
            sub_band_type: subband.sub_band_type,
            num_cbs_x: subband.num_cbs_x,
            num_cbs_y: subband.num_cbs_y,
            total_bitplanes: subband.total_bitplanes,
            code_blocks,
        })
    }

    fn assert_htj2k_lossless_roundtrip(
        pixels: &[u8],
        width: u32,
        height: u32,
        bit_depth: u8,
        num_decomposition_levels: u8,
    ) {
        let codestream = encode_htj2k(
            pixels,
            width,
            height,
            1,
            bit_depth,
            false,
            &EncodeOptions {
                num_decomposition_levels,
                ..Default::default()
            },
        )
        .expect("HTJ2K encode");

        assert!(codestream.windows(2).any(|window| window == [0xFF, 0x50]));
        let cod_offset = codestream
            .windows(2)
            .position(|window| window == [0xFF, 0x52])
            .expect("COD marker");
        assert_eq!(codestream[cod_offset + 12], 0x40);

        let image = Image::new(
            &codestream,
            &DecodeSettings {
                resolve_palette_indices: true,
                strict: true,
                target_resolution: None,
            },
        )
        .expect("parse HT codestream");
        let decoded = image.decode_native().expect("decode HT codestream");

        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        assert_eq!(decoded.bit_depth, bit_depth);
        assert_eq!(decoded.data, pixels);
    }

    fn gradient_u8(width: u32, height: u32) -> Vec<u8> {
        let mut pixels = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x * 17 + y * 31) % 256) as u8);
            }
        }
        pixels
    }

    fn lossy_htj2k_roundtrip_u8(
        pixels: &[u8],
        width: u32,
        height: u32,
        num_decomposition_levels: u8,
    ) -> (Vec<u8>, usize) {
        let codestream = encode_htj2k(
            pixels,
            width,
            height,
            1,
            8,
            false,
            &EncodeOptions {
                num_decomposition_levels,
                reversible: false,
                guard_bits: 2,
                ..Default::default()
            },
        )
        .expect("lossy HT encode");

        assert!(codestream.windows(2).any(|window| window == [0xFF, 0x50]));

        let image = Image::new(
            &codestream,
            &DecodeSettings {
                resolve_palette_indices: true,
                strict: true,
                target_resolution: None,
            },
        )
        .expect("parse lossy HT codestream");
        let decoded = image.decode_native().expect("decode lossy HT codestream");

        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        assert_eq!(decoded.bit_depth, 8);

        (decoded.data, codestream.len())
    }

    fn max_abs_error(expected: &[u8], actual: &[u8]) -> u8 {
        expected
            .iter()
            .zip(actual)
            .map(|(&expected, &actual)| expected.abs_diff(actual))
            .max()
            .unwrap_or(0)
    }

    fn psnr_db(expected: &[u8], actual: &[u8]) -> f64 {
        let mse = expected
            .iter()
            .zip(actual)
            .map(|(&expected, &actual)| {
                let diff = f64::from(expected) - f64::from(actual);
                diff * diff
            })
            .sum::<f64>()
            / expected.len() as f64;

        if mse == 0.0 {
            f64::INFINITY
        } else {
            20.0 * 255.0f64.log10() - 10.0 * mse.log10()
        }
    }

    fn assert_not_flat_128(decoded: &[u8]) {
        assert!(
            decoded.iter().any(|&sample| sample != 128),
            "lossy decode collapsed to flat 128"
        );
    }

    #[test]
    fn test_encode_high_throughput_zero_image_roundtrip() {
        let width = 4u32;
        let height = 4u32;
        let sample = 2048u16.to_le_bytes();
        let mut pixels = Vec::with_capacity((width * height * 2) as usize);
        for _ in 0..(width * height) {
            pixels.extend_from_slice(&sample);
        }

        let codestream = encode(
            &pixels,
            width,
            height,
            1,
            12,
            false,
            &EncodeOptions {
                num_decomposition_levels: 2,
                use_ht_block_coding: true,
                ..Default::default()
            },
        )
        .expect("HT all-zero encode");

        assert!(codestream.windows(2).any(|window| window == [0xFF, 0x50]));
        let cod_offset = codestream
            .windows(2)
            .position(|window| window == [0xFF, 0x52])
            .expect("COD marker");
        assert_eq!(codestream[cod_offset + 12], 0x40);

        let image =
            Image::new(&codestream, &DecodeSettings::default()).expect("parse HT codestream");
        let decoded = image.decode_native().expect("decode HT codestream");

        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        assert_eq!(decoded.bit_depth, 12);
        assert_eq!(decoded.data, pixels);
    }

    #[test]
    fn test_encode_high_throughput_nonzero_roundtrip() {
        let width = 1u32;
        let height = 1u32;
        let pixels = 2049u16.to_le_bytes().to_vec();

        let codestream = encode_htj2k(
            &pixels,
            width,
            height,
            1,
            12,
            false,
            &EncodeOptions {
                num_decomposition_levels: 0,
                ..Default::default()
            },
        )
        .expect("HT non-zero encode");

        assert!(codestream.windows(2).any(|window| window == [0xFF, 0x50]));
        let image =
            Image::new(&codestream, &DecodeSettings::default()).expect("parse HT codestream");
        let decoded = image.decode_native().expect("decode HT codestream");

        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        assert_eq!(decoded.bit_depth, 12);
        assert_eq!(decoded.data, pixels);
    }

    #[test]
    fn test_encode_high_throughput_varied_12bit_roundtrip() {
        let mut pixels = Vec::with_capacity(32);
        for i in 0u16..16 {
            pixels.extend_from_slice(&((i * 257) & 0x0FFF).to_le_bytes());
        }

        let codestream = encode_htj2k(
            &pixels,
            4,
            4,
            1,
            12,
            false,
            &EncodeOptions {
                num_decomposition_levels: 1,
                ..Default::default()
            },
        )
        .expect("HT varied encode");

        let image =
            Image::new(&codestream, &DecodeSettings::default()).expect("parse HT codestream");
        let decoded = image.decode_native().expect("decode HT codestream");

        assert_eq!(decoded.width, 4);
        assert_eq!(decoded.height, 4);
        assert_eq!(decoded.bit_depth, 12);
        assert_eq!(decoded.data, pixels);
    }

    #[test]
    fn test_encode_high_throughput_gradient_8bit_roundtrip() {
        let pixels: Vec<u8> = (0..64).collect();

        let codestream = encode_htj2k(
            &pixels,
            8,
            8,
            1,
            8,
            false,
            &EncodeOptions {
                num_decomposition_levels: 3,
                ..Default::default()
            },
        )
        .expect("HT gradient encode");

        let image =
            Image::new(&codestream, &DecodeSettings::default()).expect("parse HT codestream");
        let decoded = image.decode_native().expect("decode HT codestream");

        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
        assert_eq!(decoded.bit_depth, 8);
        assert_eq!(decoded.data, pixels);
    }

    #[test]
    fn test_encode_high_throughput_varied_12bit_large_roundtrip() {
        let width = 16u32;
        let height = 8u32;
        let mut pixels = Vec::with_capacity((width * height * 2) as usize);
        for y in 0u16..height as u16 {
            for x in 0u16..width as u16 {
                let value = (x * 257 + y * 17) & 0x0FFF;
                pixels.extend_from_slice(&value.to_le_bytes());
            }
        }

        assert_htj2k_lossless_roundtrip(&pixels, width, height, 12, 4);
    }

    #[test]
    fn test_encode_high_throughput_ramp_16bit_roundtrip() {
        let width = 48u32;
        let height = 24u32;
        let mut pixels = Vec::with_capacity((width * height * 2) as usize);
        for y in 0u16..height as u16 {
            for x in 0u16..width as u16 {
                let value = x * 521 + y * 997;
                pixels.extend_from_slice(&value.to_le_bytes());
            }
        }

        assert_htj2k_lossless_roundtrip(&pixels, width, height, 16, 4);
    }

    #[test]
    fn test_encode_high_throughput_lossy_large_gradient_is_parseable() {
        let pixels = gradient_u8(128, 128);

        let (decoded, codestream_len) = lossy_htj2k_roundtrip_u8(&pixels, 128, 128, 5);

        assert!(codestream_len > 110);
        assert_not_flat_128(&decoded);
        assert!(
            psnr_db(&pixels, &decoded) >= 30.0,
            "psnr={} max_abs={}",
            psnr_db(&pixels, &decoded),
            max_abs_error(&pixels, &decoded)
        );
    }

    #[test]
    fn test_encode_high_throughput_lossy_constant_extremes_are_not_midgray() {
        for sample in [0u8, 255] {
            let pixels = vec![sample; 64 * 64];
            let (decoded, codestream_len) = lossy_htj2k_roundtrip_u8(&pixels, 64, 64, 4);

            assert!(codestream_len > 110);
            assert_not_flat_128(&decoded);
            assert!(
                max_abs_error(&pixels, &decoded) <= 2,
                "sample={sample} max_abs={} decoded_min={} decoded_max={}",
                max_abs_error(&pixels, &decoded),
                decoded.iter().min().unwrap(),
                decoded.iter().max().unwrap()
            );
        }
    }

    #[test]
    fn test_encode_invalid_dimensions() {
        let result = encode(&[], 0, 0, 1, 8, false, &EncodeOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_too_short() {
        let pixels = vec![0u8; 10]; // Way too short for 8x8
        let result = encode(&pixels, 8, 8, 1, 8, false, &EncodeOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_deinterleave_rgb() {
        let pixels = vec![
            10u8, 20, 30, // pixel 0: R=10, G=20, B=30
            40, 50, 60, // pixel 1: R=40, G=50, B=60
        ];
        let comps = deinterleave_to_f32(&pixels, 2, 3, 8, false);
        assert_eq!(comps[0], vec![-118.0, -88.0]); // R
        assert_eq!(comps[1], vec![-108.0, -78.0]); // G
        assert_eq!(comps[2], vec![-98.0, -68.0]); // B
    }

    #[test]
    fn deinterleave_rgb8_unsigned_fast_path_matches_generic_output() {
        let pixels = (0..96)
            .map(|value| ((value * 19 + value / 3) & 0xff) as u8)
            .collect::<Vec<_>>();

        let expected = deinterleave_to_f32(&pixels, 32, 3, 8, false);
        let actual = deinterleave_rgb8_unsigned_to_f32(&pixels, 32);

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_encode_decode_roundtrip_gray_8bit() {
        use crate::{DecodeSettings, Image};

        // Constant image: all pixels = 42 — simplest possible test
        let original: Vec<u8> = vec![42u8; 64]; // 8x8, all same value
        let encoded = encode(
            &original,
            8,
            8,
            1,
            8,
            false,
            &EncodeOptions {
                num_decomposition_levels: 0,
                reversible: true,
                ..Default::default()
            },
        )
        .expect("encode failed");

        let settings = DecodeSettings {
            resolve_palette_indices: false,
            strict: false,
            target_resolution: None,
        };
        let image = Image::new(&encoded, &settings).expect("parse failed");
        let decoded = image.decode_native().expect("decode failed");

        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
        assert_eq!(decoded.data, original, "round-trip mismatch");
    }

    #[test]
    fn test_encode_decode_roundtrip_gray_8bit_single_dwt_level() {
        use crate::{DecodeSettings, Image};

        let original: Vec<u8> = (0..64 * 64)
            .map(|value| ((value * 37 + value / 7) & 0xFF) as u8)
            .collect();
        let encoded = encode(
            &original,
            64,
            64,
            1,
            8,
            false,
            &EncodeOptions {
                num_decomposition_levels: 1,
                reversible: true,
                ..Default::default()
            },
        )
        .expect("encode failed");

        let image = Image::new(&encoded, &DecodeSettings::default()).expect("parse failed");
        let decoded = image.decode_native().expect("decode failed");

        assert_eq!(decoded.width, 64);
        assert_eq!(decoded.height, 64);
        assert_eq!(decoded.data, original, "round-trip mismatch");
    }

    /// Precondition gate: native encode_htj2k must produce byte-identical output
    /// across repeated invocations with the same input before CUDA parity can be
    /// asserted.  96x80 with 3 components and 5 decomposition levels exercises
    /// multi-codeblock subbands.
    #[cfg(feature = "std")]
    #[test]
    fn encode_htj2k_is_byte_deterministic() {
        const WIDTH: u32 = 96;
        const HEIGHT: u32 = 80;
        const NUM_COMPONENTS: u8 = 3;
        const BIT_DEPTH: u8 = 8;
        const REPETITIONS: usize = 8;

        // Deterministic pseudo-random pixel data: simple LCG-like sequence.
        let pixel_count = (WIDTH * HEIGHT) as usize * usize::from(NUM_COMPONENTS);
        let pixels: Vec<u8> = (0..pixel_count)
            .map(|i| {
                let v = i
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (v >> 56) as u8
            })
            .collect();

        let options = EncodeOptions {
            use_ht_block_coding: true,
            reversible: true,
            num_decomposition_levels: 5,
            validate_high_throughput_codestream: true,
            ..EncodeOptions::default()
        };

        let baseline = encode_htj2k(
            &pixels,
            WIDTH,
            HEIGHT,
            NUM_COMPONENTS.into(),
            BIT_DEPTH,
            false,
            &options,
        )
        .expect("encode_htj2k baseline failed");

        assert!(
            !baseline.is_empty(),
            "baseline codestream must not be empty"
        );

        for i in 0..REPETITIONS {
            let result = encode_htj2k(
                &pixels,
                WIDTH,
                HEIGHT,
                NUM_COMPONENTS.into(),
                BIT_DEPTH,
                false,
                &options,
            )
            .unwrap_or_else(|e| panic!("encode_htj2k repetition {i} failed: {e}"));
            assert_eq!(
                result,
                baseline,
                "encode_htj2k repetition {i} produced different bytes \
                 (len baseline={}, len result={})",
                baseline.len(),
                result.len()
            );
        }

        println!(
            "encode_htj2k_is_byte_deterministic: {} bytes, {} repetitions all identical",
            baseline.len(),
            REPETITIONS
        );
    }

    /// Precondition gate: prove native encode_htj2k round-trips 2-component
    /// 8-bit lossless images exactly with independent component channels.
    #[cfg(feature = "std")]
    #[test]
    fn native_htj2k_roundtrips_two_component_lossless() {
        const WIDTH: u32 = 32;
        const HEIGHT: u32 = 24;
        const NUM_COMPONENTS: u8 = 2;
        const BIT_DEPTH: u8 = 8;

        // Deterministic per-pixel pattern: each sample is a function of its
        // flat index so the two planes carry different, non-trivial data.
        let pixel_count = WIDTH as usize * HEIGHT as usize * usize::from(NUM_COMPONENTS);
        let pixels: Vec<u8> = (0..pixel_count)
            .map(|i| ((i.wrapping_mul(251).wrapping_add(i / 7)) & 0xFF) as u8)
            .collect();

        let codestream = encode_htj2k(
            &pixels,
            WIDTH,
            HEIGHT,
            NUM_COMPONENTS.into(),
            BIT_DEPTH,
            false,
            &EncodeOptions::default(),
        )
        .expect("native 2-component HTJ2K encode failed");

        let image = Image::new(
            &codestream,
            &DecodeSettings {
                resolve_palette_indices: true,
                strict: true,
                target_resolution: None,
            },
        )
        .expect("native 2-component HTJ2K parse failed");
        let decoded = image
            .decode_native()
            .expect("native 2-component HTJ2K decode failed");

        assert_eq!(decoded.width, WIDTH, "width mismatch");
        assert_eq!(decoded.height, HEIGHT, "height mismatch");
        assert_eq!(decoded.bit_depth, BIT_DEPTH, "bit_depth mismatch");
        assert_eq!(
            decoded.num_components,
            u16::from(NUM_COMPONENTS),
            "component count mismatch"
        );
        assert_eq!(
            decoded.data, pixels,
            "2-component HTJ2K lossless round-trip mismatch"
        );

        println!(
            "native_htj2k_roundtrips_two_component_lossless: {} bytes codestream, {} pixel bytes",
            codestream.len(),
            pixels.len()
        );
    }

    /// Precondition gate: prove native encode_htj2k round-trips 4-component
    /// (e.g. RGBA) 8-bit lossless images exactly.
    /// Required before a CUDA parity oracle can be established for this component count.
    #[cfg(feature = "std")]
    #[test]
    fn native_htj2k_roundtrips_four_component_lossless() {
        const WIDTH: u32 = 32;
        const HEIGHT: u32 = 24;
        const NUM_COMPONENTS: u8 = 4;
        const BIT_DEPTH: u8 = 8;

        // Deterministic per-sample pattern across all four planes.
        let pixel_count = WIDTH as usize * HEIGHT as usize * usize::from(NUM_COMPONENTS);
        let pixels: Vec<u8> = (0..pixel_count)
            .map(|i| ((i.wrapping_mul(197).wrapping_add(i / 13)) & 0xFF) as u8)
            .collect();

        let codestream = encode_htj2k(
            &pixels,
            WIDTH,
            HEIGHT,
            NUM_COMPONENTS.into(),
            BIT_DEPTH,
            false,
            &EncodeOptions::default(),
        )
        .expect("native 4-component HTJ2K encode failed");

        let image = Image::new(
            &codestream,
            &DecodeSettings {
                resolve_palette_indices: true,
                strict: true,
                target_resolution: None,
            },
        )
        .expect("native 4-component HTJ2K parse failed");
        let decoded = image
            .decode_native()
            .expect("native 4-component HTJ2K decode failed");

        assert_eq!(decoded.width, WIDTH, "width mismatch");
        assert_eq!(decoded.height, HEIGHT, "height mismatch");
        assert_eq!(decoded.bit_depth, BIT_DEPTH, "bit_depth mismatch");
        assert_eq!(
            decoded.num_components,
            u16::from(NUM_COMPONENTS),
            "component count mismatch"
        );
        assert_eq!(
            decoded.data, pixels,
            "4-component HTJ2K lossless round-trip mismatch"
        );

        println!(
            "native_htj2k_roundtrips_four_component_lossless: {} bytes codestream, {} pixel bytes",
            codestream.len(),
            pixels.len()
        );
    }

    #[test]
    fn classic_pcrd_assigns_limited_budget_by_distortion_slope() {
        let candidates = vec![
            ClassicSegmentAssignmentCandidate {
                block_index: 0,
                segment_index: 0,
                rate: 500,
                distortion_delta: 500.0,
            },
            ClassicSegmentAssignmentCandidate {
                block_index: 1,
                segment_index: 0,
                rate: 700,
                distortion_delta: 7_000.0,
            },
            ClassicSegmentAssignmentCandidate {
                block_index: 2,
                segment_index: 0,
                rate: 600,
                distortion_delta: 3_000.0,
            },
        ];

        let assignments = assign_classic_segment_layers_by_slope(&candidates, 2, &[256, 3_000])
            .expect("PCRD assignment");

        assert_eq!(
            assignments,
            vec![1, 0, 1],
            "the highest slope contribution should consume the constrained first-layer budget"
        );
    }

    #[test]
    fn classic_pcrd_allows_byte_target_tolerance_for_first_legal_truncation() {
        let candidates = vec![ClassicSegmentAssignmentCandidate {
            block_index: 0,
            segment_index: 0,
            rate: 300,
            distortion_delta: 1_000.0,
        }];

        let assignments = assign_classic_segment_layers_by_slope(&candidates, 2, &[256, 1_000])
            .expect("PCRD assignment");

        assert_eq!(assignments, vec![0]);
    }

    #[test]
    fn classic_pcrd_does_not_spend_budget_on_non_prefix_segments() {
        let candidates = vec![
            ClassicSegmentAssignmentCandidate {
                block_index: 0,
                segment_index: 0,
                rate: 1_000,
                distortion_delta: 1_000.0,
            },
            ClassicSegmentAssignmentCandidate {
                block_index: 0,
                segment_index: 1,
                rate: 500,
                distortion_delta: 10_000.0,
            },
            ClassicSegmentAssignmentCandidate {
                block_index: 1,
                segment_index: 0,
                rate: 300,
                distortion_delta: 600.0,
            },
        ];

        let assignments = assign_classic_segment_layers_by_slope(&candidates, 2, &[256, 2_000])
            .expect("PCRD assignment");

        assert_eq!(
            assignments,
            vec![1, 1, 0],
            "first-layer budget must go to the best legal prefix contribution"
        );
    }

    #[test]
    fn ht_layer_assignment_uses_segment_budget_before_block_index() {
        let candidates = vec![
            HtSegmentAssignmentCandidate {
                block_index: 0,
                segment_index: 0,
                rate: 900,
            },
            HtSegmentAssignmentCandidate {
                block_index: 1,
                segment_index: 0,
                rate: 200,
            },
            HtSegmentAssignmentCandidate {
                block_index: 2,
                segment_index: 0,
                rate: 200,
            },
        ];

        let assignments = assign_ht_segment_layers_by_budget(&candidates, 2, &[256, 2_000])
            .expect("HTJ2K segment assignment");

        assert_eq!(
            assignments,
            vec![1, 0, 0],
            "HTJ2K early layers should be filled by segment byte budget, not block index"
        );
    }

    #[test]
    fn ht_layer_assignment_keeps_refinement_after_cleanup() {
        let candidates = vec![
            HtSegmentAssignmentCandidate {
                block_index: 0,
                segment_index: 0,
                rate: 200,
            },
            HtSegmentAssignmentCandidate {
                block_index: 0,
                segment_index: 1,
                rate: 50,
            },
        ];

        let assignments = assign_ht_segment_layers_by_budget(&candidates, 2, &[256, 2_000])
            .expect("HTJ2K segment assignment");

        assert_eq!(
            assignments,
            vec![0, 0],
            "a refinement segment may share the cleanup layer but must not precede it"
        );
    }

    #[test]
    fn ht_layer_contributions_split_cleanup_and_refinement_across_layers() {
        let encoded = bitplane_encode::EncodedCodeBlock {
            data: vec![0x11, 0x22, 0x33, 0x44, 0x55],
            num_coding_passes: 3,
            num_zero_bitplanes: 2,
            ht_cleanup_length: 3,
            ht_refinement_length: 2,
        };

        let contributions = ht_layer_contributions(encoded, 2, &[0, 1]).expect("split HT layers");

        assert_eq!(contributions.len(), 2);
        assert_eq!(contributions[0].data, vec![0x11, 0x22, 0x33]);
        assert_eq!(contributions[0].ht_cleanup_length, 3);
        assert_eq!(contributions[0].ht_refinement_length, 0);
        assert_eq!(contributions[0].num_coding_passes, 1);
        assert_eq!(contributions[1].data, vec![0x44, 0x55]);
        assert_eq!(contributions[1].ht_cleanup_length, 0);
        assert_eq!(contributions[1].ht_refinement_length, 2);
        assert_eq!(contributions[1].num_coding_passes, 2);
    }

    #[test]
    fn htj2k_lossy_quality_layers_decode_split_refinement_layer() {
        let width = 32;
        let height = 32;
        let pixels = gradient_u8(width, height);
        let codestream = encode_htj2k(
            &pixels,
            width,
            height,
            1,
            8,
            false,
            &EncodeOptions {
                num_decomposition_levels: 0,
                reversible: false,
                guard_bits: 2,
                num_layers: 2,
                ..Default::default()
            },
        )
        .expect("HTJ2K layered encode");

        let image = Image::new(
            &codestream,
            &DecodeSettings {
                resolve_palette_indices: true,
                strict: true,
                target_resolution: None,
            },
        )
        .expect("parse layered HT codestream");
        let decoded = image.decode_native().expect("decode layered HT codestream");

        assert_eq!(decoded.width, width);
        assert_eq!(decoded.height, height);
        assert_eq!(decoded.bit_depth, 8);
        assert_not_flat_128(&decoded.data);
        assert!(
            psnr_db(&pixels, &decoded.data) >= 30.0,
            "psnr={} max_abs={}",
            psnr_db(&pixels, &decoded.data),
            max_abs_error(&pixels, &decoded.data)
        );
    }
}
