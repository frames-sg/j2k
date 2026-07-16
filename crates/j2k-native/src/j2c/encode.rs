//! Top-level JPEG 2000 encode orchestration.
//!
//! Coordinates the full encoding pipeline:
//!   pixels → MCT → DWT → quantize → EBCOT T1 → T2 → codestream
//!
//! Supports both lossless (5-3 reversible) and lossy (9-7 irreversible) encoding.

use alloc::vec::Vec;
use core::cmp::Ordering;
use core::ops::Range;
use j2k_codec_math::dwt::max_decomposition_levels;

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
    J2kDeinterleaveToF32Job, J2kEncodeStageAccelerator, J2kForwardDwt53Job, J2kForwardDwt53Level,
    J2kForwardDwt53Output, J2kForwardDwt97Job, J2kForwardDwt97Level, J2kForwardDwt97Output,
    J2kForwardIctJob, J2kForwardRctJob, J2kHtSubbandEncodeJob, J2kHtj2kTileEncodeJob,
    J2kPacketizationBlockCodingMode, J2kPacketizationCodeBlock, J2kPacketizationEncodeJob,
    J2kPacketizationPacketDescriptor, J2kPacketizationResolution, J2kPacketizationSubband,
    J2kQuantizeSubbandJob, J2kResidentEncodeInput, J2kResidentHtj2kTileEncodeJob,
    J2kTier1CodeBlockEncodeJob, PrecomputedHtj2k53Component, PrecomputedHtj2k53Image,
    PrecomputedHtj2k97Component, PrecomputedHtj2k97Image, PreencodedHtj2k97CodeBlock,
    PreencodedHtj2k97CompactCodeBlock, PreencodedHtj2k97CompactComponent,
    PreencodedHtj2k97CompactImage, PreencodedHtj2k97CompactResolution,
    PreencodedHtj2k97CompactSubband, PreencodedHtj2k97Component, PreencodedHtj2k97Image,
    PreencodedHtj2k97Resolution, PreencodedHtj2k97Subband, PrequantizedHtj2k97Component,
    PrequantizedHtj2k97Image, PrequantizedHtj2k97Resolution, PrequantizedHtj2k97Subband,
    MAX_J2K_SPEC_COMPONENTS,
};

const HT_CPU_PARALLEL_FALLBACK_MIN_JOBS: usize = 4;
const MAX_RAW_PIXEL_ENCODE_BIT_DEPTH: u8 = 24;
const MAX_PART1_SAMPLE_BIT_DEPTH: u8 = j2k_types::MAX_JPEG2000_PART1_SAMPLE_BIT_DEPTH;
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
pub(crate) use self::api_helpers::try_deinterleave_to_f32;
use self::api_helpers::{
    default_public_code_block_style, internal_sub_band_type, public_sub_band_type,
};
#[cfg(test)]
pub(crate) use self::api_helpers::{deinterleave_rgb8_unsigned_to_f32, deinterleave_to_f32};
pub(crate) mod allocation;
mod code_block_metadata;
use self::allocation::{checked_add_bytes, checked_element_bytes, host_allocation_failed};
mod retained_api;
pub(crate) use self::retained_api::encode_with_accelerator_and_retained_input;
mod retained_input;
pub(crate) use self::retained_input::NativeEncodeRetainedInput;
use self::retained_input::{
    NativeEncodePhase, NativeEncodePipelineError, NativeEncodePipelineResult, NativeEncodeSession,
};
mod options;
use self::options::{
    validate_code_block_geometry, validate_irreversible_quantization_profile,
    validate_precinct_exponents_for_options, CodeBlockGeometry,
};
pub use self::options::{
    EncodeComponentPlane, EncodeOptions, EncodeProgressionOrder, EncodeRoiRegion,
    EncodeTypedComponentPlane,
};
mod resident_contract;
#[doc(hidden)]
pub use self::resident_contract::ResidentHtj2kEncodeError;
mod exact;
use self::exact::{
    forward_rct_i64, validate_htj2k_codestream, validate_reversible_i64_encode_options,
};
mod i64_packetize;
use self::i64_packetize::{packetize_i64_component_resolution_packets, I64PacketizeRequest};
mod multitile;
mod tile_parts;
use self::tile_parts::{
    validate_packet_header_marker_payloads, write_single_tile_packetized_codestream_for_session,
};
mod precomputed;
use self::precomputed::encode_precomputed_53_with_component_sample_info_for_session;
pub(in crate::j2c) use self::precomputed::encode_precomputed_htj2k_53_with_mct_and_retained_owner;
#[cfg(test)]
use self::precomputed::prepared_subband_from_preencoded_owned_for_tests as prepared_subband_from_preencoded_owned;
pub use self::precomputed::{
    encode_precomputed_htj2k_53, encode_precomputed_htj2k_53_with_accelerator,
    encode_precomputed_htj2k_53_with_accelerator_and_max_host_bytes,
    encode_precomputed_htj2k_53_with_mct, encode_precomputed_htj2k_53_with_mct_and_accelerator,
    encode_precomputed_htj2k_97, encode_precomputed_htj2k_97_batch_owned_with_accelerator,
    encode_precomputed_htj2k_97_batch_owned_with_accelerator_and_max_host_bytes,
    encode_precomputed_htj2k_97_batch_with_accelerator,
    encode_precomputed_htj2k_97_with_accelerator,
    encode_precomputed_htj2k_97_with_accelerator_and_max_host_bytes, encode_precomputed_j2k_53,
    encode_precomputed_j2k_53_with_accelerator, encode_precomputed_j2k_53_with_mct,
    encode_precomputed_j2k_53_with_mct_and_accelerator, encode_preencoded_htj2k_97,
    encode_preencoded_htj2k_97_compact_owned_with_accelerator,
    encode_preencoded_htj2k_97_compact_owned_with_accelerator_and_max_host_bytes,
    encode_preencoded_htj2k_97_owned_with_accelerator,
    encode_preencoded_htj2k_97_owned_with_accelerator_and_max_host_bytes,
    encode_preencoded_htj2k_97_with_accelerator, encode_prequantized_htj2k_97,
    encode_prequantized_htj2k_97_with_accelerator,
    encode_prequantized_htj2k_97_with_accelerator_and_max_host_bytes,
};
#[cfg(test)]
use self::precomputed::{validate_precomputed_dwt97_geometry, validate_precomputed_dwt_geometry};
mod precomputed_batch;
use self::precomputed_batch::prepare_precomputed_htj2k97_image_for_batch;
#[cfg(test)]
use self::precomputed_batch::{copy_code_block_coefficients, downcast_i64_coefficients_to_i32};
mod prepared_packets;
use self::prepared_packets::{
    encode_prepared_resolution_packets_for_session,
    encode_prepared_resolution_packets_layered_for_session,
};
mod packet_plan;
use self::packet_plan::{
    count_compact_code_blocks, ordered_prepared_resolution_packets_for_session,
    packet_descriptors_for_order_for_session, packetization_requires_scalar,
    packetize_resolution_packets_with_options_for_session,
    split_component_resolution_packets_by_precinct_for_session,
};
mod rate_control;
#[cfg(test)]
use self::rate_control::{
    assign_classic_segment_layers_by_slope, assign_ht_segment_layers_by_budget,
    ht_layer_contributions,
};
use self::rate_control::{
    assign_classic_segment_layers_by_slope_accounted, assign_ht_segment_layers_by_budget_accounted,
    classic_layer_contributions_accounted, classic_multilayer_code_block_style,
    classic_unbudgeted_segment_layers_accounted, enforce_classic_segment_layer_monotonicity,
    enforce_ht_segment_layer_monotonicity, ht_layer_contributions_accounted, ht_segment_count,
    ht_segment_rate, ht_unbudgeted_segment_layers_accounted, ClassicSegmentAssignmentCandidate,
    ClassicSegmentLocation, HtSegmentAssignmentCandidate, HtSegmentLocation, LayeredPreparedBlock,
    LayeredPreparedPacket, LayeredPreparedSubband,
};
mod roi_plan;
use self::roi_plan::{
    max_total_bitplanes_for_components, roi_subband_scale,
    validate_roi_encode_options_nonallocating, ComponentRoiEncodePlan, ComponentRoiEncodeRegion,
};
mod samples;
use self::samples::{
    native_samples_equal, raw_pixel_bytes_per_sample, read_le_sample_value, sign_extend_sample,
};
mod single_tile;
use self::single_tile::ownership::{cpu_dwt_transient_bytes, dwt_decompositions_retained_bytes};
use self::single_tile::{
    encode_impl, encode_precomputed_53_single_tile, encode_precomputed_97_single_tile,
    encode_resident_impl,
};
mod subband;
#[cfg(test)]
use self::subband::prepare_subband;
use self::subband::{
    prepare_subband_for_session, F32SubbandEncodeRequest, I64SubbandEncodeSettings,
};
mod tier1_allocation;
mod tier1_driver;
use self::tier1_driver::encode_prepared_subbands_for_session;
#[cfg(test)]
use self::tier1_driver::{
    encode_all_ht_code_blocks_parallel, encode_all_ht_code_blocks_serial_cpu,
    encode_prepared_subbands,
};
mod transform;
#[cfg(test)]
use self::transform::forward_dwt53_output_from_decomposition;
use self::transform::{
    adjust_component_step_sizes_for_guard_delta, adjust_reversible_step_sizes_for_guard_delta,
    encode_forward_dwt, forward_dwt53_output_retained_bytes,
    reversible_guard_bits_for_marker_limit, try_component_plane_to_f32_for_session,
    try_encode_forward_ict, try_encode_forward_rct, try_forward_dwt53_output_from_decomposition,
    validate_band_len, validate_component_sample_info, validate_deinterleaved_components,
    ForwardDwtRequest,
};
mod typed_i64;
use self::typed_i64::encode_typed_component_planes_53_i64;
mod typed_components;
use self::typed_components::encode_typed_component_planes_53_for_session;

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
///
/// # Errors
///
/// Returns an error when dimensions, sample data, component metadata, or
/// encoding options are invalid, or when a codec stage cannot encode them.
pub fn encode(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
) -> crate::EncodeResult<Vec<u8>> {
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
#[expect(
    clippy::too_many_arguments,
    reason = "this codec boundary keeps geometry, state buffers, and validated options explicit without allocation or indirection"
)]
pub fn encode_with_accelerator(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> crate::EncodeResult<Vec<u8>> {
    encode_with_accelerator_and_retained_input(
        pixels,
        width,
        height,
        num_components,
        bit_depth,
        signed,
        options,
        NativeEncodeRetainedInput::none(),
        accelerator,
    )
}

/// Encode a complete HTJ2K tile whose input pixels remain backend-resident.
///
/// This implementation-facing entry point reuses native request planning and
/// codestream finalization, but has no CPU fallback because no host samples are
/// present. A declined resident hook is returned as an explicit error.
#[doc(hidden)]
pub fn encode_resident_htj2k_with_accelerator(
    input: J2kResidentEncodeInput,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, ResidentHtj2kEncodeError> {
    encode_resident_impl(input, options, block_coding_mode(options), accelerator)
}

#[expect(
    clippy::too_many_arguments,
    reason = "this codec boundary keeps geometry, state buffers, and validated options explicit without allocation or indirection"
)]
fn encode_with_accelerator_and_component_sample_info_for_session(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
    component_sample_info: &[EncodeComponentSampleInfo],
    session: &NativeEncodeSession<'_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> crate::EncodeResult<Vec<u8>> {
    let block_coding_mode = block_coding_mode(options);
    encode_with_accelerator_and_mode_for_session(
        pixels,
        width,
        height,
        num_components,
        bit_depth,
        signed,
        options,
        component_sample_info,
        block_coding_mode,
        session,
        accelerator,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "this internal mode boundary keeps caller geometry and validated coding policy explicit"
)]
fn encode_with_accelerator_and_mode_for_session(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
    component_sample_info: &[EncodeComponentSampleInfo],
    block_coding_mode: BlockCodingMode,
    session: &NativeEncodeSession<'_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> crate::EncodeResult<Vec<u8>> {
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
        session,
        accelerator,
    )
    .map_err(NativeEncodePipelineError::into_encode_error)?;

    if block_coding_mode == BlockCodingMode::HighThroughput
        && options.validate_high_throughput_codestream
    {
        validate_htj2k_codestream(
            &codestream,
            codestream.capacity(),
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
///
/// # Errors
///
/// Returns an error for invalid image/sample metadata, invalid ROI regions or
/// options, or a failure in any codec stage.
#[expect(
    clippy::too_many_arguments,
    reason = "this codec boundary keeps geometry, state buffers, and validated options explicit without allocation or indirection"
)]
pub fn encode_with_roi_regions(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
    roi_regions: &[EncodeRoiRegion],
) -> crate::EncodeResult<Vec<u8>> {
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
#[expect(
    clippy::too_many_arguments,
    reason = "this codec boundary keeps geometry, state buffers, and validated options explicit without allocation or indirection"
)]
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
) -> crate::EncodeResult<Vec<u8>> {
    let session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())?;
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
        &session,
        accelerator,
    )
    .map_err(NativeEncodePipelineError::into_encode_error)?;

    if block_coding_mode == BlockCodingMode::HighThroughput
        && options.validate_high_throughput_codestream
    {
        validate_htj2k_codestream(
            &codestream,
            codestream.capacity(),
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
///
/// # Errors
///
/// Returns an error when the input or options are invalid, encoding fails, or
/// the requested output fails HTJ2K self-validation.
pub fn encode_htj2k(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
) -> crate::EncodeResult<Vec<u8>> {
    let session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())?;
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_with_accelerator_and_mode_for_session(
        pixels,
        width,
        height,
        num_components,
        bit_depth,
        signed,
        options,
        &[],
        BlockCodingMode::HighThroughput,
        &session,
        &mut accelerator,
    )
}

/// Encode reversible 5/3 component planes into a classic J2K or HTJ2K
/// codestream.
///
/// Plane buffers are supplied at each component's own SIZ sampling grid. Set
/// [`EncodeOptions::use_ht_block_coding`] to select HTJ2K block coding; the
/// default writes classic Part 1 block coding.
///
/// # Errors
///
/// Returns an error for invalid component geometry, sampling, sample buffers,
/// or options, or when a codec stage fails.
pub fn encode_component_planes_53(
    planes: &[EncodeComponentPlane<'_>],
    width: u32,
    height: u32,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
) -> crate::EncodeResult<Vec<u8>> {
    let session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())?;
    let requested_bytes = checked_element_bytes::<EncodeTypedComponentPlane<'_>>(
        planes.len(),
        "component-plane typed descriptor owners",
    )?;
    session.checked_phase(requested_bytes, "component-plane typed descriptor owners")?;
    let mut typed_planes = Vec::new();
    typed_planes.try_reserve_exact(planes.len()).map_err(|_| {
        host_allocation_failed("component-plane typed descriptor owners", requested_bytes)
    })?;
    typed_planes.extend(planes.iter().map(|plane| EncodeTypedComponentPlane {
        data: plane.data,
        x_rsiz: plane.x_rsiz,
        y_rsiz: plane.y_rsiz,
        bit_depth,
        signed,
    }));
    let actual_bytes = checked_element_bytes::<EncodeTypedComponentPlane<'_>>(
        typed_planes.capacity(),
        "component-plane typed descriptor owners",
    )?;
    let typed_session = session.checked_child_session(
        &typed_planes,
        actual_bytes,
        "component-plane typed descriptor owners",
    )?;
    encode_typed_component_planes_53_for_session(
        &typed_planes,
        width,
        height,
        options,
        &typed_session,
    )
    .map_err(NativeEncodePipelineError::into_encode_error)
}

/// Encode reversible 5/3 typed component planes into a classic J2K or HTJ2K
/// codestream.
///
/// This is the component-plane entry point for JPEG 2000 codestreams whose
/// components have different precision or signedness. Plane buffers are
/// supplied at each component's own SIZ sampling grid. Components are encoded
/// without a reversible color transform.
///
/// # Errors
///
/// Returns an error for invalid component count, dimensions, sampling,
/// precision, sample buffers, or options, or when a codec stage fails.
pub fn encode_typed_component_planes_53(
    planes: &[EncodeTypedComponentPlane<'_>],
    width: u32,
    height: u32,
    options: &EncodeOptions,
) -> crate::EncodeResult<Vec<u8>> {
    let session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())?;
    encode_typed_component_planes_53_for_session(planes, width, height, options, &session)
        .map_err(NativeEncodePipelineError::into_encode_error)
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

fn ht_target_coding_passes_for_options(
    options: &EncodeOptions,
    block_coding_mode: BlockCodingMode,
) -> u8 {
    if block_coding_mode == BlockCodingMode::HighThroughput && options.num_layers > 1 {
        options.num_layers.min(3)
    } else {
        1
    }
}

enum PreparedCodeBlockCoefficients {
    I32(Vec<i32>),
    I64(Vec<i64>),
    Empty,
}

#[cfg(test)]
impl PreparedCodeBlockCoefficients {
    fn is_empty(&self) -> bool {
        match self {
            Self::I32(values) => values.is_empty(),
            Self::I64(values) => values.is_empty(),
            Self::Empty => true,
        }
    }
}

struct PreparedEncodeCodeBlock {
    coefficients: PreparedCodeBlockCoefficients,
    width: u32,
    height: u32,
}

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

#[cfg(test)]
#[path = "encode_tests.rs"]
mod tests;
