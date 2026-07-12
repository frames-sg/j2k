// SPDX-License-Identifier: MIT OR Apache-2.0

//! JPEG 2000 encode option and request types.

use alloc::vec::Vec;

use super::super::quantize;
use crate::IrreversibleQuantizationSubbandScales;

/// Encoding options for JPEG 2000.
#[derive(Debug, Clone)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the public options expose independent JPEG 2000 coding and marker switches"
)]
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

impl EncodeProgressionOrder {
    pub(crate) const fn packetization_order(self) -> crate::J2kPacketizationProgressionOrder {
        match self {
            Self::Lrcp => crate::J2kPacketizationProgressionOrder::Lrcp,
            Self::Rlcp => crate::J2kPacketizationProgressionOrder::Rlcp,
            Self::Rpcl => crate::J2kPacketizationProgressionOrder::Rpcl,
            Self::Pcrl => crate::J2kPacketizationProgressionOrder::Pcrl,
            Self::Cprl => crate::J2kPacketizationProgressionOrder::Cprl,
        }
    }
}

fn validate_irreversible_quantization_scale(scale: f32) -> Result<(), &'static str> {
    if scale.is_finite() && scale > 0.0 {
        Ok(())
    } else {
        Err("irreversible quantization scale must be finite and greater than zero")
    }
}

pub(super) fn validate_irreversible_quantization_profile(
    options: &EncodeOptions,
) -> Result<(), &'static str> {
    validate_irreversible_quantization_scale(options.irreversible_quantization_scale)?;
    if quantize::subband_scales_all_valid(options.irreversible_quantization_subband_scales) {
        Ok(())
    } else {
        Err("irreversible quantization subband scales must be finite and greater than zero")
    }
}

/// Validated Part 1 code-block dimensions derived from COD's stored
/// exponent-minus-two fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CodeBlockGeometry {
    pub(super) width: u32,
    pub(super) height: u32,
}

/// Validate the JPEG 2000 Part 1 code-block exponent and area constraints
/// without allocating or shifting by an unchecked public value.
pub(super) fn validate_code_block_geometry(
    options: &EncodeOptions,
) -> Result<CodeBlockGeometry, &'static str> {
    const MAX_STORED_EXPONENT: u8 = 8;
    const MAX_COMBINED_ACTUAL_EXPONENT: u8 = 12;

    if options.code_block_width_exp > MAX_STORED_EXPONENT {
        return Err("code-block width exponent exceeds supported range");
    }
    if options.code_block_height_exp > MAX_STORED_EXPONENT {
        return Err("code-block height exponent exceeds supported range");
    }
    let width_exponent = options
        .code_block_width_exp
        .checked_add(2)
        .ok_or("code-block width exponent exceeds supported range")?;
    let height_exponent = options
        .code_block_height_exp
        .checked_add(2)
        .ok_or("code-block height exponent exceeds supported range")?;
    let combined_exponent = width_exponent
        .checked_add(height_exponent)
        .ok_or("code-block combined exponent exceeds supported range")?;
    if combined_exponent > MAX_COMBINED_ACTUAL_EXPONENT {
        return Err("code-block dimensions exceed JPEG 2000 Part 1 area limit");
    }
    let width = 1_u32
        .checked_shl(u32::from(width_exponent))
        .ok_or("code-block width exponent exceeds supported range")?;
    let height = 1_u32
        .checked_shl(u32::from(height_exponent))
        .ok_or("code-block height exponent exceeds supported range")?;
    Ok(CodeBlockGeometry { width, height })
}

pub(super) fn validate_precinct_exponents_for_options(
    options: &EncodeOptions,
    num_decomposition_levels: u8,
) -> Result<(), &'static str> {
    validate_code_block_geometry(options)?;
    if options.precinct_exponents.is_empty() {
        return Ok(());
    }

    let expected = usize::from(num_decomposition_levels) + 1;
    if options.precinct_exponents.len() != expected {
        return Err("precinct exponent count must match resolution level count");
    }
    if options
        .precinct_exponents
        .iter()
        .any(|&(horizontal_exponent, vertical_exponent)| {
            horizontal_exponent > 15 || vertical_exponent > 15
        })
    {
        return Err("precinct exponents must fit in COD marker nybbles");
    }
    let code_block_horizontal_exponent = options
        .code_block_width_exp
        .checked_add(2)
        .ok_or("code-block width exponent exceeds supported range")?;
    let code_block_vertical_exponent = options
        .code_block_height_exp
        .checked_add(2)
        .ok_or("code-block height exponent exceeds supported range")?;
    for (resolution, &(horizontal_exponent, vertical_exponent)) in
        options.precinct_exponents.iter().enumerate()
    {
        let minimum_horizontal_exponent = if resolution == 0 {
            code_block_horizontal_exponent
        } else {
            code_block_horizontal_exponent
                .checked_add(1)
                .ok_or("code-block width exponent exceeds supported range")?
        };
        let minimum_vertical_exponent = if resolution == 0 {
            code_block_vertical_exponent
        } else {
            code_block_vertical_exponent
                .checked_add(1)
                .ok_or("code-block height exponent exceeds supported range")?
        };
        if horizontal_exponent < minimum_horizontal_exponent
            || vertical_exponent < minimum_vertical_exponent
        {
            return Err("precinct exponents must not reduce encoder code-block dimensions");
        }
    }
    Ok(())
}
