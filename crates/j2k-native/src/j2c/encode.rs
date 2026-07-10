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
mod exact;
use self::exact::{
    deinterleave_to_i64, forward_rct_i64, validate_htj2k_codestream,
    validate_reversible_i64_encode_options,
};
mod i64_packetize;
use self::i64_packetize::{
    encode_i64_component_resolution_packets, packetize_i64_component_resolution_packets,
    I64CodestreamPacketRequest, I64PacketizeRequest,
};
mod multitile;
use self::multitile::{encode_multitile_impl, extract_component_plane_tile};
mod tile_parts;
use self::tile_parts::{
    split_packetized_tile_into_tile_parts, validate_packet_header_marker_payloads,
    write_single_tile_packetized_codestream,
};
mod precomputed;
use self::precomputed::encode_precomputed_53_with_component_sample_info_and_accelerator;
#[cfg(test)]
use self::precomputed::prepared_subband_from_preencoded_owned_for_tests as prepared_subband_from_preencoded_owned;
pub use self::precomputed::{
    encode_precomputed_htj2k_53, encode_precomputed_htj2k_53_with_accelerator,
    encode_precomputed_htj2k_53_with_mct, encode_precomputed_htj2k_53_with_mct_and_accelerator,
    encode_precomputed_htj2k_97, encode_precomputed_htj2k_97_batch_with_accelerator,
    encode_precomputed_htj2k_97_with_accelerator, encode_precomputed_j2k_53,
    encode_precomputed_j2k_53_with_accelerator, encode_precomputed_j2k_53_with_mct,
    encode_precomputed_j2k_53_with_mct_and_accelerator, encode_preencoded_htj2k_97,
    encode_preencoded_htj2k_97_compact_owned_with_accelerator,
    encode_preencoded_htj2k_97_owned_with_accelerator, encode_preencoded_htj2k_97_with_accelerator,
    encode_prequantized_htj2k_97, encode_prequantized_htj2k_97_with_accelerator,
};
#[cfg(test)]
use self::precomputed::{validate_precomputed_dwt97_geometry, validate_precomputed_dwt_geometry};
mod precomputed_batch;
use self::precomputed_batch::{
    coefficients_fit_i32, copy_code_block_coefficients, copy_code_block_coefficients_i64,
    downcast_i64_coefficients_to_i32, prepare_precomputed_htj2k97_image_for_batch,
};
mod prepared_packets;
use self::prepared_packets::{
    encode_prepared_resolution_packets, encode_prepared_resolution_packets_layered,
};
mod packet_plan;
use self::packet_plan::{
    count_compact_code_blocks, ordered_prepared_compact_resolution_packets,
    ordered_prepared_resolution_packets, packet_descriptors_for_compact_order,
    packet_descriptors_for_order, packetization_requires_scalar,
    packetize_resolution_packets_with_options, public_packetization_progression_order,
    public_packetization_resolutions_from_compact, scalar_packet_descriptors,
    split_component_resolution_packets_by_precinct,
};
mod rate_control;
use self::rate_control::{
    assign_classic_segment_layers_by_slope, assign_ht_segment_layers_by_budget,
    classic_layer_contributions, classic_multilayer_code_block_style,
    classic_unbudgeted_segment_layers, enforce_classic_segment_layer_monotonicity,
    enforce_ht_segment_layer_monotonicity, ht_layer_contributions, ht_segment_count,
    ht_segment_rate, ht_unbudgeted_segment_layers, ClassicSegmentAssignmentCandidate,
    ClassicSegmentLocation, HtSegmentAssignmentCandidate, HtSegmentLocation, LayeredPreparedBlock,
    LayeredPreparedPacket, LayeredPreparedSubband,
};
mod roi_plan;
use self::roi_plan::{
    component_sampling_for_options, max_total_bitplanes_for_components,
    roi_encode_plans_for_options, roi_subband_scale, ComponentRoiEncodePlan,
    ComponentRoiEncodeRegion,
};
mod samples;
use self::samples::{
    native_samples_equal, raw_pixel_bytes_per_sample, read_le_sample_value, sign_extend_sample,
};
mod single_tile;
use self::single_tile::encode_impl;
mod subband;
use self::subband::{
    prepare_subband, prepare_subband_cpu_quantized, prepare_subband_i64, I64SubbandEncodeSettings,
};
mod tier1_driver;
use self::tier1_driver::{encode_all_ht_code_blocks, encode_prepared_subbands};
#[cfg(test)]
use self::tier1_driver::{
    encode_all_ht_code_blocks_parallel, encode_all_ht_code_blocks_serial_cpu,
};
mod transform;
use self::transform::{
    adjust_component_step_sizes_for_guard_delta, adjust_reversible_step_sizes_for_guard_delta,
    component_plane_to_f32, component_step_sizes, encode_forward_dwt,
    forward_dwt53_output_from_decomposition, reversible_guard_bits_for_marker_limit,
    try_encode_forward_ict, try_encode_forward_rct, validate_band_len,
    validate_component_sample_info, validate_component_sampling_dwt_geometry,
    validate_deinterleaved_components,
};
mod typed_i64;
use self::typed_i64::encode_typed_component_planes_53_i64;

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

#[cfg(test)]
include!("encode_tests.rs");
