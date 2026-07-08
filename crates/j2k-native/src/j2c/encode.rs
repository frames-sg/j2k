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
#[doc(hidden)]
pub use super::quantize::irreversible_quantization_step_for_subband;
use super::quantize::{self, QuantStepSize};
use crate::profile;
pub(crate) use crate::J2kSubBandType;
use crate::{
    CpuOnlyJ2kEncodeStageAccelerator, EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock,
    IrreversibleQuantizationSubbandScales, J2kDeinterleaveToF32Job, J2kEncodeStageAccelerator,
    J2kForwardDwt53Job, J2kForwardDwt53Level, J2kForwardDwt53Output, J2kForwardDwt97Job,
    J2kForwardDwt97Level, J2kForwardDwt97Output, J2kForwardIctJob, J2kForwardRctJob,
    J2kHtSubbandEncodeJob, J2kHtj2kTileEncodeJob, J2kPacketizationBlockCodingMode,
    J2kPacketizationCodeBlock, J2kPacketizationEncodeJob, J2kPacketizationPacketDescriptor,
    J2kPacketizationResolution, J2kPacketizationSubband, J2kQuantizeSubbandJob,
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

mod api_helpers;
#[cfg(test)]
use self::api_helpers::deinterleave_rgb8_unsigned_to_f32;
pub(crate) use self::api_helpers::deinterleave_to_f32;
use self::api_helpers::{
    default_public_code_block_style, internal_sub_band_type, max_decomposition_levels,
    public_sub_band_type,
};
mod options;
use self::options::{precinct_exponents_for_options, validate_irreversible_quantization_profile};
pub use self::options::{
    EncodeComponentPlane, EncodeOptions, EncodeProgressionOrder, EncodeRoiRegion,
    EncodeTypedComponentPlane,
};
mod i64_packetize;
use self::i64_packetize::{
    encode_i64_component_resolution_packets, packetize_i64_component_resolution_packets,
    I64CodestreamPacketRequest, I64PacketizeRequest,
};
mod tile_parts;
use self::tile_parts::{
    split_packetized_tile_into_tile_parts, validate_packet_header_marker_payloads,
    write_single_tile_packetized_codestream,
};
mod precomputed;
pub use self::precomputed::*;
mod packet_plan;
use self::packet_plan::*;
mod rate_control;
use self::rate_control::*;
mod roi_plan;
use self::roi_plan::*;
mod samples;
use self::samples::{
    native_samples_equal, raw_pixel_bytes_per_sample, read_le_sample_value, sign_extend_sample,
};
mod single_tile;
use self::single_tile::encode_impl;
mod subband;
use self::subband::*;

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
#[doc(hidden)]
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
#[doc(hidden)]
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
        let subband_settings = I64SubbandEncodeSettings {
            guard_bits,
            cb_width,
            cb_height,
            roi_shift: 0,
            roi_regions: &[],
            roi_scale: 1,
            block_coding_mode,
            ht_target_coding_passes,
        };

        let ll_subband = prepare_subband_i64(
            &decomp.ll,
            decomp.ll_width,
            decomp.ll_height,
            steps
                .first()
                .ok_or("reversible quantization step missing")?,
            SubBandType::LowLow,
            subband_settings,
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
                SubBandType::HighLow,
                subband_settings,
            )?;
            let lh_subband = prepare_subband_i64(
                &level.lh,
                level.low_width,
                level.high_height,
                steps
                    .get(step_base + 1)
                    .ok_or("reversible quantization step missing")?,
                SubBandType::LowHigh,
                subband_settings,
            )?;
            let hh_subband = prepare_subband_i64(
                &level.hh,
                level.high_width,
                level.high_height,
                steps
                    .get(step_base + 2)
                    .ok_or("reversible quantization step missing")?,
                SubBandType::HighHigh,
                subband_settings,
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
        I64CodestreamPacketRequest {
            packetize: I64PacketizeRequest {
                width,
                height,
                num_components,
                num_levels,
                params: &params,
                options: &high_bit_options,
                accelerator: &mut accelerator,
            },
            quant_params: &quant_params,
        },
    )
}

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
                I64ComponentPlanePacketRequest {
                    component_dimensions: &component_dimensions,
                    component_step_sizes: &component_step_sizes,
                    num_levels,
                    subband_settings: I64SubbandEncodeSettings {
                        guard_bits,
                        cb_width: 1u32 << (options.code_block_width_exp + 2),
                        cb_height: 1u32 << (options.code_block_height_exp + 2),
                        roi_shift: 0,
                        roi_regions: &[],
                        roi_scale: 1,
                        block_coding_mode,
                        ht_target_coding_passes: ht_target_coding_passes_for_options(options),
                    },
                },
            )?;
            let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
            let packetized_tile = packetize_i64_component_resolution_packets(
                component_resolution_packets,
                I64PacketizeRequest {
                    width: actual_width,
                    height: actual_height,
                    num_components,
                    num_levels,
                    params: &params,
                    options: &child_options,
                    accelerator: &mut accelerator,
                },
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

struct I64ComponentPlanePacketRequest<'a> {
    component_dimensions: &'a [(u32, u32)],
    component_step_sizes: &'a [Vec<QuantStepSize>],
    num_levels: u8,
    subband_settings: I64SubbandEncodeSettings<'a>,
}

fn prepare_typed_component_planes_i64_packets(
    planes: &[EncodeTypedComponentPlane<'_>],
    request: I64ComponentPlanePacketRequest<'_>,
) -> Result<Vec<Vec<PreparedResolutionPacket>>, &'static str> {
    let I64ComponentPlanePacketRequest {
        component_dimensions,
        component_step_sizes,
        num_levels,
        subband_settings,
    } = request;
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
            SubBandType::LowLow,
            subband_settings,
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
                SubBandType::HighLow,
                subband_settings,
            )?;
            let lh_subband = prepare_subband_i64(
                &level.lh,
                level.low_width,
                level.high_height,
                steps
                    .get(step_base + 1)
                    .ok_or("reversible quantization step missing")?,
                SubBandType::LowHigh,
                subband_settings,
            )?;
            let hh_subband = prepare_subband_i64(
                &level.hh,
                level.high_width,
                level.high_height,
                steps
                    .get(step_base + 2)
                    .ok_or("reversible quantization step missing")?,
                SubBandType::HighHigh,
                subband_settings,
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

    crate::sort_packet_descriptors_for_progression(
        &mut descriptors,
        progression_order.packetization_order(),
    );

    Ok((resolution_packets, descriptors))
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

#[cfg(test)]
include!("encode_tests.rs");
