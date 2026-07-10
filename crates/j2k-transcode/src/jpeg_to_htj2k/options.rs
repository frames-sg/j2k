// SPDX-License-Identifier: MIT OR Apache-2.0

use std::vec::Vec;

use j2k::IrreversibleQuantizationSubbandScales;
use j2k::J2kProgressionOrder;

/// Default irreversible quantization multiplier for JPEG direct 9/7 HTJ2K.
///
/// Empirically rate-match the explicit lossy comparison profile near the
/// external comparator output size on the bundled WSI tiles. Lower values
/// produce larger/higher-quality codestreams; `1.0` matches the native encoder
/// default but overshoots the external baseline size for this transcode path.
pub const JPEG_TO_HTJ2K_LOSSY_97_QUANTIZATION_SCALE: f32 = 1.9;

/// HTJ2K encode options used after JPEG coefficient-domain wavelet bands are produced.
#[derive(Debug, Clone, PartialEq)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "stable public options expose independent codec switches rather than an implicit mode matrix"
)]
pub struct JpegToHtj2kEncodeOptions {
    /// Number of wavelet decomposition levels.
    pub num_decomposition_levels: u8,
    /// Whether to emit reversible/lossless coding.
    pub reversible: bool,
    /// Code-block width exponent minus two.
    pub code_block_width_exp: u8,
    /// Code-block height exponent minus two.
    pub code_block_height_exp: u8,
    /// JPEG 2000 guard bits.
    pub guard_bits: u8,
    /// Whether to encode HTJ2K code blocks instead of classic EBCOT.
    pub use_ht_block_coding: bool,
    /// Packet progression order.
    pub progression_order: J2kProgressionOrder,
    /// Whether to write a TLM marker segment.
    pub write_tlm: bool,
    /// Whether to write PLT packet-length marker segments.
    pub write_plt: bool,
    /// Whether to write PLM packet-length marker segments.
    pub write_plm: bool,
    /// Whether to write PPM packed packet-header marker segments.
    pub write_ppm: bool,
    /// Whether to write PPT packed packet-header marker segments.
    pub write_ppt: bool,
    /// Whether to write SOP marker segments before packets.
    pub write_sop: bool,
    /// Whether to write EPH markers after packet headers.
    pub write_eph: bool,
    /// Whether to apply JPEG 2000 multi-component transform.
    pub use_mct: bool,
    /// Number of cumulative quality layers.
    pub num_layers: u8,
    /// Optional cumulative packet-body byte targets for each quality layer.
    pub quality_layer_byte_targets: Vec<u64>,
    /// Whether native HTJ2K validation is enabled after encode.
    pub validate_high_throughput_codestream: bool,
    /// Global irreversible 9/7 quantization scale.
    pub irreversible_quantization_scale: f32,
    /// Per-subband irreversible 9/7 quantization scales.
    pub irreversible_quantization_subband_scales: IrreversibleQuantizationSubbandScales,
    /// Optional per-component SIZ sampling factors (`XRsiz`, `YRsiz`).
    pub component_sampling: Option<Vec<(u8, u8)>>,
    /// Optional tile size for multi-tile codestreams.
    pub tile_size: Option<(u32, u32)>,
    /// Optional maximum number of complete packets to place in each tile-part.
    pub tile_part_packet_limit: Option<u16>,
    /// Optional precinct exponents in COD order.
    pub precinct_exponents: Vec<(u8, u8)>,
}

impl Default for JpegToHtj2kEncodeOptions {
    fn default() -> Self {
        Self {
            num_decomposition_levels: 5,
            reversible: true,
            code_block_width_exp: 4,
            code_block_height_exp: 4,
            guard_bits: 1,
            use_ht_block_coding: false,
            progression_order: J2kProgressionOrder::Lrcp,
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
            tile_size: None,
            tile_part_packet_limit: None,
            precinct_exponents: Vec::new(),
        }
    }
}

impl JpegToHtj2kEncodeOptions {
    pub(super) fn to_native(&self) -> j2k_native::EncodeOptions {
        j2k_native::EncodeOptions {
            num_decomposition_levels: self.num_decomposition_levels,
            reversible: self.reversible,
            code_block_width_exp: self.code_block_width_exp,
            code_block_height_exp: self.code_block_height_exp,
            guard_bits: self.guard_bits,
            use_ht_block_coding: self.use_ht_block_coding,
            progression_order: native_progression_order(self.progression_order),
            write_tlm: self.write_tlm,
            write_plt: self.write_plt,
            write_plm: self.write_plm,
            write_ppm: self.write_ppm,
            write_ppt: self.write_ppt,
            write_sop: self.write_sop,
            write_eph: self.write_eph,
            use_mct: self.use_mct,
            num_layers: self.num_layers,
            quality_layer_byte_targets: self.quality_layer_byte_targets.clone(),
            validate_high_throughput_codestream: self.validate_high_throughput_codestream,
            irreversible_quantization_scale: self.irreversible_quantization_scale,
            irreversible_quantization_subband_scales: self.irreversible_quantization_subband_scales,
            component_sampling: self.component_sampling.clone(),
            tile_size: self.tile_size,
            tile_part_packet_limit: self.tile_part_packet_limit,
            precinct_exponents: self.precinct_exponents.clone(),
            roi_component_shifts: Vec::new(),
        }
    }
}

/// Options for the experimental JPEG-to-HTJ2K path.
#[derive(Debug, Clone)]
pub struct JpegToHtj2kOptions {
    /// HTJ2K encode options used after wavelet bands are produced.
    pub encode_options: JpegToHtj2kEncodeOptions,
    /// Coefficient production path used for HTJ2K precomputed bands.
    pub coefficient_path: JpegToHtj2kCoefficientPath,
    /// Materialize the float IDCT-then-DWT oracle and report rounded
    /// coefficient differences. This is intended for validation and tests, not
    /// the production direct path.
    pub validate_against_float_reference: bool,
    /// Materialize j2k-jpeg scalar ISLOW samples and report reversible
    /// integer 5/3 coefficient differences against the rounded direct path.
    /// This is intended for validation and tests, not the production direct
    /// path.
    pub validate_against_integer_reference: bool,
}

impl Default for JpegToHtj2kOptions {
    fn default() -> Self {
        Self::lossless_53()
    }
}

impl JpegToHtj2kOptions {
    /// Options for the default reversible 5/3 HTJ2K coefficient path.
    #[must_use]
    pub fn lossless_53() -> Self {
        Self {
            encode_options: transcode_encode_options(true),
            coefficient_path: JpegToHtj2kCoefficientPath::IntegerDirect53,
            validate_against_float_reference: false,
            validate_against_integer_reference: false,
        }
    }

    /// Options for the irreversible 9/7 HTJ2K float-linear coefficient path.
    #[must_use]
    pub fn lossy_97() -> Self {
        let mut encode_options = transcode_encode_options(false);
        encode_options.irreversible_quantization_scale = JPEG_TO_HTJ2K_LOSSY_97_QUANTIZATION_SCALE;
        Self {
            encode_options,
            coefficient_path: JpegToHtj2kCoefficientPath::FloatDirectLinear97,
            validate_against_float_reference: false,
            validate_against_integer_reference: false,
        }
    }
}

fn transcode_encode_options(reversible: bool) -> JpegToHtj2kEncodeOptions {
    JpegToHtj2kEncodeOptions {
        num_decomposition_levels: 1,
        reversible,
        use_ht_block_coding: true,
        use_mct: false,
        validate_high_throughput_codestream: false,
        ..JpegToHtj2kEncodeOptions::default()
    }
}

/// Experimental production path used to generate HTJ2K wavelet coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegToHtj2kCoefficientPath {
    /// Exact reversible 5/3 coefficients relative to `j2k-jpeg` scalar
    /// ISLOW block decode semantics. The first 5/3 level is computed from DCT
    /// blocks without materializing a full spatial image plane; later levels
    /// recurse conventionally over the LL coefficient band.
    IntegerDirect53,
    /// Floating-point linear composition of IDCT and 5/3 analysis. This is the
    /// linear math oracle path and remains useful for validating the direct
    /// matrix composition, but it is not the integer reversible production
    /// default.
    FloatDirectLinear53,
    /// Floating-point linear composition of IDCT and irreversible 9/7
    /// analysis. This is a lossy experimental path and must be paired with an
    /// irreversible HTJ2K encode.
    FloatDirectLinear97,
}

fn native_progression_order(
    progression: J2kProgressionOrder,
) -> j2k_native::EncodeProgressionOrder {
    match progression {
        J2kProgressionOrder::Lrcp => j2k_native::EncodeProgressionOrder::Lrcp,
        J2kProgressionOrder::Rlcp => j2k_native::EncodeProgressionOrder::Rlcp,
        J2kProgressionOrder::Rpcl => j2k_native::EncodeProgressionOrder::Rpcl,
        J2kProgressionOrder::Pcrl => j2k_native::EncodeProgressionOrder::Pcrl,
        J2kProgressionOrder::Cprl => j2k_native::EncodeProgressionOrder::Cprl,
    }
}
