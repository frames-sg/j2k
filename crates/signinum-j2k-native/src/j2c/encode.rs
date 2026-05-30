//! Top-level JPEG 2000 encode orchestration.
//!
//! Coordinates the full encoding pipeline:
//!   pixels → MCT → DWT → quantize → EBCOT T1 → T2 → codestream
//!
//! Supports both lossless (5-3 reversible) and lossy (9-7 irreversible) encoding.

use alloc::vec;
use alloc::vec::Vec;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::bitplane_encode;
use super::build::SubBandType;
use super::codestream_write::{self, BlockCodingMode, EncodeParams};
use super::fdwt::{self, DwtDecomposition};
use super::forward_mct;
use super::ht_block_encode;
use super::packet_encode::{self, CodeBlockPacketData, ResolutionPacket, SubbandPrecinct};
use super::quantize::{self, QuantStepSize};
use crate::math::{floor_f32, log2_f32};
use crate::profile;
use crate::{
    CpuOnlyJ2kEncodeStageAccelerator, EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock,
    J2kDeinterleaveToF32Job, J2kEncodeStageAccelerator, J2kForwardDwt53Job, J2kForwardDwt53Level,
    J2kForwardDwt53Output, J2kForwardDwt97Job, J2kForwardDwt97Level, J2kForwardDwt97Output,
    J2kForwardIctJob, J2kForwardRctJob, J2kHtSubbandEncodeJob, J2kHtj2kTileEncodeJob,
    J2kPacketizationBlockCodingMode, J2kPacketizationCodeBlock, J2kPacketizationEncodeJob,
    J2kPacketizationPacketDescriptor, J2kPacketizationResolution, J2kPacketizationSubband,
    J2kQuantizeSubbandJob, J2kSubBandType, J2kTier1CodeBlockEncodeJob,
};
use crate::{DecodeSettings, Image};

const HT_CPU_PARALLEL_FALLBACK_MIN_JOBS: usize = 4;

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
    /// Apply the JPEG 2000 multi-component color transform for 3+ component inputs.
    pub use_mct: bool,
    /// Decode and verify HTJ2K codestreams inside the native encoder.
    pub validate_high_throughput_codestream: bool,
    /// Multiplier applied to irreversible 9/7 scalar quantization step sizes.
    ///
    /// `1.0` preserves the near-lossless default step sizes. Larger values
    /// produce smaller codestreams by coarsening quantization.
    pub irreversible_quantization_scale: f32,
    /// Optional per-component SIZ sampling factors (`XRsiz`, `YRsiz`).
    ///
    /// `None` means every component is stored at the reference-grid
    /// resolution. This is experimental and primarily intended for precomputed
    /// coefficient encoders that preserve JPEG-native chroma subsampling.
    pub component_sampling: Option<Vec<(u8, u8)>>,
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
            use_mct: true,
            validate_high_throughput_codestream: true,
            irreversible_quantization_scale: 1.0,
            component_sampling: None,
        }
    }
}

/// JPEG 2000 packet progression orders supported by the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum EncodeProgressionOrder {
    /// Layer-resolution-component-position progression.
    #[default]
    Lrcp,
    /// Resolution-position-component-layer progression.
    Rpcl,
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
    num_components: u8,
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
    num_components: u8,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
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
    num_components: u8,
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

/// Precomputed reversible 5/3 wavelet coefficients for one component.
#[derive(Debug, Clone)]
pub struct PrecomputedHtj2k53Component {
    /// Horizontal SIZ sampling factor (`XRsiz`).
    pub x_rsiz: u8,
    /// Vertical SIZ sampling factor (`YRsiz`).
    pub y_rsiz: u8,
    /// Forward 5/3 DWT output, ordered as the encoder expects.
    pub dwt: J2kForwardDwt53Output,
}

/// Precomputed reversible 5/3 wavelet image.
#[derive(Debug, Clone)]
pub struct PrecomputedHtj2k53Image {
    /// Reference-grid image width.
    pub width: u32,
    /// Reference-grid image height.
    pub height: u32,
    /// Component precision in bits.
    pub bit_depth: u8,
    /// Whether component samples are signed.
    pub signed: bool,
    /// Components at their native resolution.
    pub components: Vec<PrecomputedHtj2k53Component>,
}

/// Precomputed irreversible 9/7 wavelet coefficients for one component.
#[derive(Debug, Clone)]
pub struct PrecomputedHtj2k97Component {
    /// Horizontal SIZ sampling factor (`XRsiz`).
    pub x_rsiz: u8,
    /// Vertical SIZ sampling factor (`YRsiz`).
    pub y_rsiz: u8,
    /// Forward 9/7 DWT output, ordered as the encoder expects.
    pub dwt: J2kForwardDwt97Output,
}

/// Precomputed irreversible 9/7 wavelet image.
#[derive(Debug, Clone)]
pub struct PrecomputedHtj2k97Image {
    /// Reference-grid image width.
    pub width: u32,
    /// Reference-grid image height.
    pub height: u32,
    /// Component precision in bits.
    pub bit_depth: u8,
    /// Whether component samples are signed.
    pub signed: bool,
    /// Components at their native resolution.
    pub components: Vec<PrecomputedHtj2k97Component>,
}

/// Prequantized irreversible 9/7 HTJ2K code-block image.
#[derive(Debug, Clone)]
pub struct PrequantizedHtj2k97Image {
    /// Reference-grid image width.
    pub width: u32,
    /// Reference-grid image height.
    pub height: u32,
    /// Component precision in bits.
    pub bit_depth: u8,
    /// Whether component samples are signed.
    pub signed: bool,
    /// Components at their native resolution.
    pub components: Vec<PrequantizedHtj2k97Component>,
}

/// Prequantized irreversible 9/7 HTJ2K component.
#[derive(Debug, Clone)]
pub struct PrequantizedHtj2k97Component {
    /// Horizontal SIZ sampling factor (`XRsiz`).
    pub x_rsiz: u8,
    /// Vertical SIZ sampling factor (`YRsiz`).
    pub y_rsiz: u8,
    /// Resolution packets for this component, ordered from lowest to highest.
    pub resolutions: Vec<PrequantizedHtj2k97Resolution>,
}

/// One component resolution's prequantized HTJ2K subbands.
#[derive(Debug, Clone)]
pub struct PrequantizedHtj2k97Resolution {
    /// Subbands in packet order: LL for resolution 0, then HL/LH/HH.
    pub subbands: Vec<PrequantizedHtj2k97Subband>,
}

/// One prequantized HTJ2K subband split into code-blocks.
#[derive(Debug, Clone)]
pub struct PrequantizedHtj2k97Subband {
    /// Subband kind.
    pub sub_band_type: J2kSubBandType,
    /// Number of code-blocks in the x direction.
    pub num_cbs_x: u32,
    /// Number of code-blocks in the y direction.
    pub num_cbs_y: u32,
    /// Total bitplanes declared for every code-block in this subband.
    pub total_bitplanes: u8,
    /// Code-block coefficients in row-major code-block order.
    pub code_blocks: Vec<PrequantizedHtj2k97CodeBlock>,
}

/// One prequantized HTJ2K code-block.
#[derive(Debug, Clone)]
pub struct PrequantizedHtj2k97CodeBlock {
    /// Quantized coefficients in row-major order.
    pub coefficients: Vec<i32>,
    /// Code-block width in coefficients.
    pub width: u32,
    /// Code-block height in coefficients.
    pub height: u32,
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
    if image.width == 0 || image.height == 0 {
        return Err("invalid dimensions");
    }
    if image.components.is_empty() || image.components.len() > 4 {
        return Err("unsupported component count");
    }
    if image.bit_depth == 0 || image.bit_depth > 16 {
        return Err("unsupported bit depth");
    }
    if image
        .components
        .iter()
        .any(|component| component.x_rsiz == 0 || component.y_rsiz == 0)
    {
        return Err("component sampling factors must be non-zero");
    }
    validate_precomputed_dwt_geometry(image)?;

    let num_components =
        u8::try_from(image.components.len()).map_err(|_| "unsupported component count")?;
    let num_levels = precomputed_level_count(&image.components)?;
    let mut precomputed_options = options.clone();
    precomputed_options.num_decomposition_levels = num_levels;
    precomputed_options.reversible = true;
    precomputed_options.use_ht_block_coding = true;
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
    if image.components.is_empty() || image.components.len() > 4 {
        return Err("unsupported component count");
    }
    if image.bit_depth == 0 || image.bit_depth > 16 {
        return Err("unsupported bit depth");
    }
    validate_irreversible_quantization_scale(options.irreversible_quantization_scale)?;
    if image
        .components
        .iter()
        .any(|component| component.x_rsiz == 0 || component.y_rsiz == 0)
    {
        return Err("component sampling factors must be non-zero");
    }
    validate_precomputed_dwt97_geometry(image)?;

    let num_components =
        u8::try_from(image.components.len()).map_err(|_| "unsupported component count")?;
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
    if image.components.is_empty() || image.components.len() > 4 {
        return Err("unsupported component count");
    }
    if image.bit_depth == 0 || image.bit_depth > 16 {
        return Err("unsupported bit depth");
    }
    validate_irreversible_quantization_scale(options.irreversible_quantization_scale)?;
    if image
        .components
        .iter()
        .any(|component| component.x_rsiz == 0 || component.y_rsiz == 0)
    {
        return Err("component sampling factors must be non-zero");
    }

    let num_components =
        u8::try_from(image.components.len()).map_err(|_| "unsupported component count")?;
    let num_levels = prequantized_97_level_count(&image.components)?;
    let guard_bits = options.guard_bits.max(2);
    let step_sizes = quantize::compute_step_sizes_with_irreversible_scale(
        image.bit_depth,
        num_levels,
        false,
        guard_bits,
        options.irreversible_quantization_scale,
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
        .map(prepared_resolution_packets_from_prequantized_component)
        .collect::<Result<Vec<_>, _>>()?;
    let prepared_resolution_packets =
        ordered_prepared_resolution_packets(component_resolution_packets, &prequantized_options)?;
    let mut resolution_packets =
        encode_prepared_resolution_packets(prepared_resolution_packets, accelerator)?;
    let packetization_resolutions = public_packetization_resolutions(&resolution_packets);
    let packet_descriptors =
        packet_descriptors_for_order(resolution_packets.len(), 1, num_components)?;
    let packetization_job = J2kPacketizationEncodeJob {
        resolution_count: resolution_packets.len() as u32,
        num_layers: 1,
        num_components,
        code_block_count: count_code_blocks(&resolution_packets)?,
        progression_order: public_packetization_progression_order(
            prequantized_options.progression_order,
        ),
        packet_descriptors: &packet_descriptors,
        resolutions: &packetization_resolutions,
    };
    let tile_data = accelerator
        .encode_packetization(packetization_job)?
        .unwrap_or_else(|| {
            packet_encode::form_tile_bitstream(&mut resolution_packets, 1, num_components)
        });

    let quant_params: Vec<(u16, u16)> = step_sizes
        .iter()
        .map(|s| (s.exponent, s.mantissa))
        .collect();
    let params = EncodeParams {
        width: image.width,
        height: image.height,
        num_components,
        bit_depth: image.bit_depth,
        signed: image.signed,
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
        component_sampling: prequantized_options
            .component_sampling
            .clone()
            .ok_or("component sampling missing")?,
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
        validate_precomputed_component_dwt(&component.dwt, component_width, component_height)?;
    }

    Ok(())
}

fn validate_precomputed_dwt97_geometry(
    image: &PrecomputedHtj2k97Image,
) -> Result<(), &'static str> {
    for component in &image.components {
        let component_width = image.width.div_ceil(u32::from(component.x_rsiz));
        let component_height = image.height.div_ceil(u32::from(component.y_rsiz));
        validate_precomputed_component_dwt97(&component.dwt, component_width, component_height)?;
    }

    Ok(())
}

fn validate_precomputed_component_dwt(
    dwt: &J2kForwardDwt53Output,
    component_width: u32,
    component_height: u32,
) -> Result<(), &'static str> {
    if dwt.levels.is_empty() {
        return Err("precomputed DWT must contain at least one decomposition level");
    }
    if let Some(highest_level) = dwt.levels.last() {
        if highest_level.width != component_width || highest_level.height != component_height {
            return Err("precomputed DWT component dimensions mismatch");
        }
    }

    let mut expected_width = component_width;
    let mut expected_height = component_height;
    for level in dwt.levels.iter().rev() {
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
        validate_band_len(level.hl.len(), high_width, low_height)?;
        validate_band_len(level.lh.len(), low_width, high_height)?;
        validate_band_len(level.hh.len(), high_width, high_height)?;

        expected_width = low_width;
        expected_height = low_height;
    }

    if dwt.ll_width != expected_width || dwt.ll_height != expected_height {
        return Err("precomputed DWT component dimensions mismatch");
    }
    validate_band_len(dwt.ll.len(), expected_width, expected_height)
}

fn validate_precomputed_component_dwt97(
    dwt: &J2kForwardDwt97Output,
    component_width: u32,
    component_height: u32,
) -> Result<(), &'static str> {
    if dwt.levels.is_empty() {
        return Err("precomputed DWT must contain at least one decomposition level");
    }
    if let Some(highest_level) = dwt.levels.last() {
        if highest_level.width != component_width || highest_level.height != component_height {
            return Err("precomputed DWT component dimensions mismatch");
        }
    }

    let mut expected_width = component_width;
    let mut expected_height = component_height;
    for level in dwt.levels.iter().rev() {
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
        validate_band_len(level.hl.len(), high_width, low_height)?;
        validate_band_len(level.lh.len(), low_width, high_height)?;
        validate_band_len(level.hh.len(), high_width, high_height)?;

        expected_width = low_width;
        expected_height = low_height;
    }

    if dwt.ll_width != expected_width || dwt.ll_height != expected_height {
        return Err("precomputed DWT component dimensions mismatch");
    }
    validate_band_len(dwt.ll.len(), expected_width, expected_height)
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

fn prepared_resolution_packets_from_prequantized_component(
    component: &PrequantizedHtj2k97Component,
) -> Result<Vec<PreparedResolutionPacket>, &'static str> {
    component
        .resolutions
        .iter()
        .map(|resolution| {
            Ok(PreparedResolutionPacket {
                subbands: resolution
                    .subbands
                    .iter()
                    .map(prepared_subband_from_prequantized)
                    .collect::<Result<Vec<_>, _>>()?,
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
                coefficients: block.coefficients.clone(),
                width: block.width,
                height: block.height,
            })
            .collect(),
        preencoded_ht_code_blocks: None,
        num_cbs_x: subband.num_cbs_x,
        num_cbs_y: subband.num_cbs_y,
        sub_band_type: internal_sub_band_type(subband.sub_band_type),
        total_bitplanes: subband.total_bitplanes,
        block_coding_mode: BlockCodingMode::HighThroughput,
    })
}

fn zero_pixel_buffer(
    width: u32,
    height: u32,
    num_components: u8,
    bit_depth: u8,
) -> Result<Vec<u8>, &'static str> {
    let bytes_per_sample = if bit_depth <= 8 { 1usize } else { 2usize };
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

fn validate_irreversible_quantization_scale(scale: f32) -> Result<(), &'static str> {
    if scale.is_finite() && scale > 0.0 {
        Ok(())
    } else {
        Err("irreversible quantization scale must be finite and greater than zero")
    }
}

fn validate_htj2k_codestream(
    codestream: &[u8],
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u8,
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

    let bytes_per_sample = if bit_depth <= 8 { 1 } else { 2 };
    let sample_count = expected.len() / bytes_per_sample;
    (0..sample_count).all(|sample_index| {
        decode_native_sample(expected, sample_index, bit_depth, signed)
            == decode_native_sample(actual, sample_index, bit_depth, signed)
    })
}

fn decode_native_sample(bytes: &[u8], sample_index: usize, bit_depth: u8, signed: bool) -> i32 {
    let byte_offset = sample_index * if bit_depth <= 8 { 1 } else { 2 };
    let mask = (1u32 << u32::from(bit_depth)) - 1;
    let raw = if bit_depth <= 8 {
        u32::from(bytes[byte_offset])
    } else {
        u32::from(u16::from_le_bytes([
            bytes[byte_offset],
            bytes[byte_offset + 1],
        ]))
    } & mask;

    if signed {
        let shift = 32 - u32::from(bit_depth);
        ((raw << shift) as i32) >> shift
    } else {
        raw as i32
    }
}

fn encode_impl(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u8,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
    block_coding_mode: BlockCodingMode,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    if width == 0 || height == 0 {
        return Err("invalid dimensions");
    }
    if num_components == 0 || num_components > 4 {
        return Err("unsupported component count");
    }
    if bit_depth == 0 || bit_depth > 16 {
        return Err("unsupported bit depth");
    }
    if !options.reversible {
        validate_irreversible_quantization_scale(options.irreversible_quantization_scale)?;
    }

    let num_pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or("image dimensions overflow")?;
    let bytes_per_sample = if bit_depth <= 8 { 1 } else { 2 };
    let expected_len = num_pixels
        .checked_mul(num_components as usize)
        .and_then(|len| len.checked_mul(bytes_per_sample))
        .ok_or("image dimensions overflow")?;
    if pixels.len() < expected_len {
        return Err("pixel data too short");
    }
    let component_sampling = component_sampling_for_options(options, num_components)?;

    let profile_enabled = profile::profile_stages_enabled();
    let total_start = profile::profile_now(profile_enabled);

    let use_mct = options.use_mct && num_components >= 3;
    let num_levels = options.num_decomposition_levels.min(
        // Don't decompose more than the image supports
        max_decomposition_levels(width, height),
    );
    let guard_bits = if options.reversible {
        if use_mct {
            options.guard_bits.max(2)
        } else {
            options.guard_bits
        }
    } else {
        options.guard_bits.max(2)
    };
    let step_sizes = quantize::compute_step_sizes_with_irreversible_scale(
        bit_depth,
        num_levels,
        options.reversible,
        guard_bits,
        options.irreversible_quantization_scale,
    );
    let quant_params: Vec<(u16, u16)> = step_sizes
        .iter()
        .map(|s| (s.exponent, s.mantissa))
        .collect();
    let cb_width = 1u32 << (options.code_block_width_exp + 2);
    let cb_height = 1u32 << (options.code_block_height_exp + 2);
    let params = EncodeParams {
        width,
        height,
        num_components,
        bit_depth,
        signed,
        num_decomposition_levels: num_levels,
        reversible: options.reversible,
        code_block_width_exp: options.code_block_width_exp,
        code_block_height_exp: options.code_block_height_exp,
        num_layers: 1,
        use_mct,
        guard_bits,
        block_coding_mode,
        progression_order: options.progression_order,
        write_tlm: options.write_tlm,
        component_sampling,
    };

    let stage_start = profile::profile_now(profile_enabled);
    if block_coding_mode == BlockCodingMode::HighThroughput {
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
        .collect::<Result<Vec<_>, _>>()?;
    let dwt_us = profile::elapsed_us(stage_start);

    // Step 5: Quantize and encode code-blocks for each component
    let mut component_resolution_packets: Vec<Vec<PreparedResolutionPacket>> =
        Vec::with_capacity(num_components as usize);

    let stage_start = profile::profile_now(profile_enabled);
    for decomp in decompositions.iter().take(num_components as usize) {
        let mut packets = Vec::with_capacity(num_levels as usize + 1);

        // LL subband (resolution 0)
        let ll_subband = prepare_subband(
            &decomp.ll,
            decomp.ll_width,
            decomp.ll_height,
            &step_sizes[0],
            bit_depth,
            guard_bits,
            options.reversible,
            block_coding_mode,
            cb_width,
            cb_height,
            SubBandType::LowLow,
            accelerator,
        )?;
        packets.push(PreparedResolutionPacket {
            subbands: vec![ll_subband],
        });

        // Higher resolution levels
        for (level_idx, level) in decomp.levels.iter().enumerate() {
            let step_base = 1 + level_idx * 3;

            // HL subband
            let hl_subband = prepare_subband(
                &level.hl,
                level.high_width,
                level.low_height,
                &step_sizes[step_base],
                bit_depth,
                guard_bits,
                options.reversible,
                block_coding_mode,
                cb_width,
                cb_height,
                SubBandType::HighLow,
                accelerator,
            )?;

            // LH subband
            let lh_subband = prepare_subband(
                &level.lh,
                level.low_width,
                level.high_height,
                &step_sizes[step_base + 1],
                bit_depth,
                guard_bits,
                options.reversible,
                block_coding_mode,
                cb_width,
                cb_height,
                SubBandType::LowHigh,
                accelerator,
            )?;

            // HH subband
            let hh_subband = prepare_subband(
                &level.hh,
                level.high_width,
                level.high_height,
                &step_sizes[step_base + 2],
                bit_depth,
                guard_bits,
                options.reversible,
                block_coding_mode,
                cb_width,
                cb_height,
                SubBandType::HighHigh,
                accelerator,
            )?;

            packets.push(PreparedResolutionPacket {
                subbands: vec![hl_subband, lh_subband, hh_subband],
            });
        }

        component_resolution_packets.push(packets);
    }
    let subband_prepare_us = profile::elapsed_us(stage_start);

    let prepared_resolution_packets =
        ordered_prepared_resolution_packets(component_resolution_packets, options)?;
    let stage_start = profile::profile_now(profile_enabled);
    let resolution_packets =
        encode_prepared_resolution_packets(prepared_resolution_packets, accelerator)?;
    let block_encode_us = profile::elapsed_us(stage_start);

    // Step 6: Form tile bitstream (T2)
    let stage_start = profile::profile_now(profile_enabled);
    let mut resolution_packets = resolution_packets;
    let packetization_resolutions = public_packetization_resolutions(&resolution_packets);
    let packet_descriptors =
        packet_descriptors_for_order(resolution_packets.len(), 1, num_components)?;
    let packetization_job = J2kPacketizationEncodeJob {
        resolution_count: resolution_packets.len() as u32,
        num_layers: 1,
        num_components,
        code_block_count: count_code_blocks(&resolution_packets)?,
        progression_order: public_packetization_progression_order(options.progression_order),
        packet_descriptors: &packet_descriptors,
        resolutions: &packetization_resolutions,
    };
    let tile_data = accelerator
        .encode_packetization(packetization_job)?
        .unwrap_or_else(|| {
            packet_encode::form_tile_bitstream(&mut resolution_packets, 1, num_components)
        });
    let packetize_us = profile::elapsed_us(stage_start);

    // Step 7: Write codestream
    let stage_start = profile::profile_now(profile_enabled);
    let codestream = codestream_write::write_codestream(&params, &tile_data, &quant_params);
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
    num_components: u8,
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

fn component_sampling_for_options(
    options: &EncodeOptions,
    num_components: u8,
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

fn packet_descriptors_for_order(
    packet_count: usize,
    num_layers: u8,
    num_components: u8,
) -> Result<Vec<J2kPacketizationPacketDescriptor>, &'static str> {
    if num_layers != 1 {
        return Err("encode currently prepares one packet contribution layer");
    }
    let component_count = usize::from(num_components).max(1);
    (0..packet_count)
        .map(|packet_index| {
            Ok(J2kPacketizationPacketDescriptor {
                packet_index: u32::try_from(packet_index)
                    .map_err(|_| "packet descriptor index exceeds u32")?,
                state_index: u32::try_from(packet_index)
                    .map_err(|_| "packet descriptor state index exceeds u32")?,
                layer: 0,
                resolution: u32::try_from(packet_index / component_count)
                    .map_err(|_| "packet descriptor resolution exceeds u32")?,
                component: u8::try_from(packet_index % component_count)
                    .map_err(|_| "packet descriptor component exceeds u8")?,
                precinct: 0,
            })
        })
        .collect()
}

fn ordered_prepared_resolution_packets(
    component_resolution_packets: Vec<Vec<PreparedResolutionPacket>>,
    options: &EncodeOptions,
) -> Result<Vec<PreparedResolutionPacket>, &'static str> {
    match options.progression_order {
        EncodeProgressionOrder::Lrcp | EncodeProgressionOrder::Rpcl => {
            lrcp_ordered_prepared_resolution_packets(component_resolution_packets)
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

fn public_packetization_progression_order(
    progression_order: EncodeProgressionOrder,
) -> crate::J2kPacketizationProgressionOrder {
    match progression_order {
        EncodeProgressionOrder::Lrcp => crate::J2kPacketizationProgressionOrder::Lrcp,
        EncodeProgressionOrder::Rpcl => crate::J2kPacketizationProgressionOrder::Rpcl,
    }
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

fn public_packetization_block_coding_mode(
    block_coding_mode: BlockCodingMode,
) -> J2kPacketizationBlockCodingMode {
    match block_coding_mode {
        BlockCodingMode::Classic => J2kPacketizationBlockCodingMode::Classic,
        BlockCodingMode::HighThroughput => J2kPacketizationBlockCodingMode::HighThroughput,
    }
}

struct PreparedEncodeCodeBlock {
    coefficients: Vec<i32>,
    width: u32,
    height: u32,
}

struct PreparedEncodeSubband {
    code_blocks: Vec<PreparedEncodeCodeBlock>,
    preencoded_ht_code_blocks: Option<Vec<EncodedHtJ2kCodeBlock>>,
    num_cbs_x: u32,
    num_cbs_y: u32,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
    block_coding_mode: BlockCodingMode,
}

struct PreparedResolutionPacket {
    subbands: Vec<PreparedEncodeSubband>,
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
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<PreparedEncodeSubband, &'static str> {
    if width == 0 || height == 0 {
        return Ok(PreparedEncodeSubband {
            code_blocks: Vec::new(),
            preencoded_ht_code_blocks: None,
            num_cbs_x: 0,
            num_cbs_y: 0,
            sub_band_type,
            total_bitplanes: 0,
            block_coding_mode,
        });
    }

    let range_bits = subband_range_bits(bit_depth, sub_band_type);
    debug_assert!(step_size.exponent <= u16::from(u8::MAX));
    let total_bitplanes = guard_bits
        .saturating_add(step_size.exponent as u8)
        .saturating_sub(1);
    let num_cbs_x = width.div_ceil(cb_width);
    let num_cbs_y = height.div_ceil(cb_height);

    if block_coding_mode == BlockCodingMode::HighThroughput {
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
                sub_band_type,
                total_bitplanes,
                block_coding_mode,
            });
        }
    }

    let quantized = match accelerator.encode_quantize_subband(J2kQuantizeSubbandJob {
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
        sub_band_type,
        total_bitplanes,
        block_coding_mode,
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

    let jobs: Vec<_> = prepared_subbands
        .iter()
        .flat_map(|subband| {
            subband
                .code_blocks
                .iter()
                .map(move |block| crate::J2kHtCodeBlockEncodeJob {
                    coefficients: &block.coefficients,
                    width: block.width,
                    height: block.height,
                    total_bitplanes: subband.total_bitplanes,
                    target_coding_passes: 1,
                })
        })
        .collect();

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
    let jobs: Vec<_> = prepared_subbands
        .iter()
        .flat_map(|subband| {
            let public_sub_band_type = public_sub_band_type(subband.sub_band_type);
            subband
                .code_blocks
                .iter()
                .map(move |block| J2kTier1CodeBlockEncodeJob {
                    coefficients: &block.coefficients,
                    width: block.width,
                    height: block.height,
                    sub_band_type: public_sub_band_type,
                    total_bitplanes: subband.total_bitplanes,
                    style,
                })
        })
        .collect();

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
    for subband in prepared_subbands {
        for block in &subband.code_blocks {
            encoded.push(encode_tier1_code_block(
                &block.coefficients,
                block.width,
                block.height,
                subband.sub_band_type,
                subband.total_bitplanes,
                accelerator,
            )?);
        }
    }
    Ok(encoded)
}

fn encode_all_ht_code_blocks_serial_cpu(
    jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
) -> Result<Vec<bitplane_encode::EncodedCodeBlock>, &'static str> {
    if jobs.iter().any(|job| job.target_coding_passes != 1) {
        return Err("CPU HTJ2K code-block fallback supports cleanup-only encode");
    }
    jobs.iter()
        .map(|job| {
            ht_block_encode::encode_code_block(
                job.coefficients,
                job.width,
                job.height,
                job.total_bitplanes,
            )
        })
        .collect()
}

#[cfg(feature = "parallel")]
fn encode_all_ht_code_blocks_parallel(
    jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
) -> Result<Vec<bitplane_encode::EncodedCodeBlock>, &'static str> {
    if jobs.iter().any(|job| job.target_coding_passes != 1) {
        return Err("CPU HTJ2K code-block fallback supports cleanup-only encode");
    }
    jobs.par_iter()
        .map(|job| {
            ht_block_encode::encode_code_block(
                job.coefficients,
                job.width,
                job.height,
                job.total_bitplanes,
            )
        })
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn encode_all_ht_code_blocks_parallel(
    jobs: &[crate::J2kHtCodeBlockEncodeJob<'_>],
) -> Result<Vec<bitplane_encode::EncodedCodeBlock>, &'static str> {
    if jobs.iter().any(|job| job.target_coding_passes != 1) {
        return Err("CPU HTJ2K code-block fallback supports cleanup-only encode");
    }
    jobs.iter()
        .map(|job| {
            ht_block_encode::encode_code_block(
                job.coefficients,
                job.width,
                job.height,
                job.total_bitplanes,
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
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<bitplane_encode::EncodedCodeBlock, &'static str> {
    if let Some(encoded) = accelerator.encode_ht_code_block(crate::J2kHtCodeBlockEncodeJob {
        coefficients,
        width,
        height,
        total_bitplanes,
        target_coding_passes: 1,
    })? {
        return Ok(ht_encoded_code_block_from_accelerator(encoded));
    }

    ht_block_encode::encode_code_block(coefficients, width, height, total_bitplanes)
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
    num_components: u8,
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
        (1u32 << (bit_depth as u32 - 1)) as f32
    };

    if bit_depth <= 8 {
        for (i, pixel) in pixels.chunks_exact(nc).take(num_pixels).enumerate() {
            for (c, component) in components.iter_mut().enumerate().take(nc) {
                let val = pixel[c];
                component[i] = if signed {
                    (val as i8) as f32
                } else {
                    val as f32 - unsigned_offset
                };
            }
        }
    } else {
        // 16-bit samples (little-endian)
        for (i, pixel) in pixels.chunks_exact(nc * 2).take(num_pixels).enumerate() {
            for (c, component) in components.iter_mut().enumerate().take(nc) {
                let offset = c * 2;
                let val = u16::from_le_bytes([pixel[offset], pixel[offset + 1]]);
                component[i] = if signed {
                    (val as i16) as f32
                } else {
                    val as f32 - unsigned_offset
                };
            }
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
        let step_sizes = quantize::compute_step_sizes_with_irreversible_scale(
            image.bit_depth,
            1,
            false,
            guard_bits,
            options.irreversible_quantization_scale,
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
                .map(|block| PrequantizedHtj2k97CodeBlock {
                    coefficients: block.coefficients,
                    width: block.width,
                    height: block.height,
                })
                .collect(),
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
                let v = i.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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

        let baseline = encode_htj2k(&pixels, WIDTH, HEIGHT, NUM_COMPONENTS, BIT_DEPTH, false, &options)
            .expect("encode_htj2k baseline failed");

        assert!(!baseline.is_empty(), "baseline codestream must not be empty");

        for i in 0..REPETITIONS {
            let result = encode_htj2k(&pixels, WIDTH, HEIGHT, NUM_COMPONENTS, BIT_DEPTH, false, &options)
                .unwrap_or_else(|e| panic!("encode_htj2k repetition {i} failed: {e}"));
            assert_eq!(
                result, baseline,
                "encode_htj2k repetition {i} produced different bytes \
                 (len baseline={}, len result={})",
                baseline.len(), result.len()
            );
        }

        println!(
            "encode_htj2k_is_byte_deterministic: {} bytes, {} repetitions all identical",
            baseline.len(),
            REPETITIONS
        );
    }

    /// Precondition gate: prove native encode_htj2k round-trips 2-component
    /// (e.g. Y+A or YCbCr-420-placeholder) 8-bit lossless images exactly.
    ///
    /// SCOPE FINDING: native's decoder raises `Validation(TooManyChannels)` when
    /// asked to decode a 2-component codestream it produced itself. Native cannot
    /// serve as a parity oracle for 2-component HTJ2K lossless; this component
    /// count is OUT OF SCOPE for CUDA parity verification.
    #[test]
    #[ignore = "native does not round-trip 2-component HTJ2K lossless; out of scope"]
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
            NUM_COMPONENTS,
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
        let decoded = image.decode_native().expect("native 2-component HTJ2K decode failed");

        assert_eq!(decoded.width, WIDTH, "width mismatch");
        assert_eq!(decoded.height, HEIGHT, "height mismatch");
        assert_eq!(decoded.bit_depth, BIT_DEPTH, "bit_depth mismatch");
        assert_eq!(decoded.num_components, NUM_COMPONENTS, "component count mismatch");
        assert_eq!(decoded.data, pixels, "2-component HTJ2K lossless round-trip mismatch");

        println!(
            "native_htj2k_roundtrips_two_component_lossless: {} bytes codestream, {} pixel bytes",
            codestream.len(),
            pixels.len()
        );
    }

    /// Precondition gate: prove native encode_htj2k round-trips 4-component
    /// (e.g. RGBA) 8-bit lossless images exactly.
    /// Required before a CUDA parity oracle can be established for this component count.
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
            NUM_COMPONENTS,
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
        let decoded = image.decode_native().expect("native 4-component HTJ2K decode failed");

        assert_eq!(decoded.width, WIDTH, "width mismatch");
        assert_eq!(decoded.height, HEIGHT, "height mismatch");
        assert_eq!(decoded.bit_depth, BIT_DEPTH, "bit_depth mismatch");
        assert_eq!(decoded.num_components, NUM_COMPONENTS, "component count mismatch");
        assert_eq!(decoded.data, pixels, "4-component HTJ2K lossless round-trip mismatch");

        println!(
            "native_htj2k_roundtrips_four_component_lossless: {} bytes codestream, {} pixel bytes",
            codestream.len(),
            pixels.len()
        );
    }
}
