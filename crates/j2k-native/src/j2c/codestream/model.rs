// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::super::build::SubBandType;
use crate::error::{err, Result, ValidationError};

mod size;
pub(crate) use size::{ComponentSizeInfo, SizeData};

#[derive(Debug)]
pub(crate) struct Header<'a> {
    pub(crate) size_data: SizeData,
    pub(crate) global_coding_style: CodingStyleDefault,
    pub(crate) component_infos: Vec<ComponentInfo>,
    pub(crate) progression_changes: Vec<ProgressionChange>,
    pub(crate) plm_packet_lengths: Vec<u32>,
    pub(crate) ppm_packets: Vec<PpmPacket<'a>>,
    pub(crate) skipped_resolution_levels: u8,
    /// Whether strict mode is enabled for decoding.
    pub(crate) strict: bool,
}

#[derive(Debug)]
pub(crate) struct PpmMarkerData<'a> {
    pub(crate) sequence_idx: u8,
    pub(crate) data: &'a [u8],
}

#[derive(Debug, Clone)]
pub(crate) struct PpmPacket<'a> {
    pub(crate) data: &'a [u8],
    pub(crate) ends_tile_part: bool,
}

#[derive(Debug)]
pub(crate) struct PacketLengthMarker {
    pub(crate) sequence_idx: u8,
    pub(crate) packet_lengths: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RgnMarkerData {
    pub(crate) component_index: u16,
    pub(crate) style: u8,
    pub(crate) shift: u8,
}

#[derive(Debug)]
pub(crate) struct ComponentInfo {
    pub(crate) size_info: ComponentSizeInfo,
    pub(crate) coding_style: CodingStyleComponent,
    pub(crate) quantization_info: QuantizationInfo,
    pub(crate) roi_shift: u8,
}

impl ComponentInfo {
    pub(crate) fn exponent_mantissa(
        &self,
        sub_band_type: SubBandType,
        resolution: u8,
    ) -> Result<(u16, u16)> {
        let n_ll = self.coding_style.parameters.num_decomposition_levels;

        let sb_index = match sub_band_type {
            // LL only has a quantization entry at resolution 0; non-zero LL
            // lookups fall through to the missing-step error below.
            SubBandType::LowLow => u16::MAX,
            SubBandType::HighLow => 0,
            SubBandType::LowHigh => 1,
            SubBandType::HighHigh => 2,
        };

        let step_sizes = &self.quantization_info.step_sizes;
        match self.quantization_info.quantization_style {
            QuantizationStyle::NoQuantization | QuantizationStyle::ScalarExpounded => {
                let entry = if resolution == 0 {
                    step_sizes.first()
                } else {
                    step_sizes.get(1 + (resolution as usize - 1) * 3 + sb_index as usize)
                };

                Ok(entry
                    .map(|s| (s.exponent, s.mantissa))
                    .ok_or(ValidationError::MissingStepSize)?)
            }
            QuantizationStyle::ScalarDerived => {
                let (e_0, mantissa) = step_sizes
                    .first()
                    .map(|s| (s.exponent, s.mantissa))
                    .ok_or(ValidationError::MissingStepSize)?;
                let n_b = if resolution == 0 {
                    u16::from(n_ll)
                } else {
                    u16::from(n_ll) + 1 - u16::from(resolution)
                };

                let exponent = e_0
                    .checked_sub(u16::from(n_ll))
                    .and_then(|e| e.checked_add(n_b))
                    .ok_or(ValidationError::InvalidExponents)?;

                Ok((exponent, mantissa))
            }
        }
    }

    pub(crate) fn wavelet_transform(&self) -> WaveletTransform {
        self.coding_style.parameters.transformation
    }

    pub(crate) fn requires_exact_integer_decode(&self) -> bool {
        self.size_info.precision > 24
            && self.wavelet_transform() == WaveletTransform::Reversible53
            && self.quantization_info.quantization_style == QuantizationStyle::NoQuantization
    }

    pub(crate) fn num_resolution_levels(&self) -> u8 {
        self.coding_style.parameters.num_resolution_levels
    }

    pub(crate) fn num_decomposition_levels(&self) -> u8 {
        self.coding_style.parameters.num_decomposition_levels
    }

    pub(crate) fn code_block_style(&self) -> CodeBlockStyle {
        self.coding_style.parameters.code_block_style
    }
}

/// Progression order (Table A.16).
#[derive(Debug, Clone, Copy)]
pub(crate) enum ProgressionOrder {
    LayerResolutionComponentPosition,
    ResolutionLayerComponentPosition,
    ResolutionPositionComponentLayer,
    PositionComponentResolutionLayer,
    ComponentPositionResolutionLayer,
}

#[derive(Debug, Clone)]
pub(crate) struct ProgressionChange {
    pub(crate) resolution_start: u8,
    pub(crate) component_start: u16,
    pub(crate) layer_end: u8,
    pub(crate) resolution_end: u8,
    pub(crate) component_end: u16,
    pub(crate) progression_order: ProgressionOrder,
}

impl ProgressionOrder {
    pub(super) fn from_u8(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::LayerResolutionComponentPosition),
            1 => Ok(Self::ResolutionLayerComponentPosition),
            2 => Ok(Self::ResolutionPositionComponentLayer),
            3 => Ok(Self::PositionComponentResolutionLayer),
            4 => Ok(Self::ComponentPositionResolutionLayer),
            _ => err!(ValidationError::InvalidProgressionOrder),
        }
    }
}

/// Wavelet transformation type (Table A.20).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WaveletTransform {
    Irreversible97,
    Reversible53,
}

impl WaveletTransform {
    pub(super) fn from_u8(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::Irreversible97),
            1 => Ok(Self::Reversible53),
            _ => err!(ValidationError::InvalidTransformation),
        }
    }
}

impl From<WaveletTransform> for crate::J2kWaveletTransform {
    fn from(transform: WaveletTransform) -> Self {
        match transform {
            WaveletTransform::Reversible53 => Self::Reversible53,
            WaveletTransform::Irreversible97 => Self::Irreversible97,
        }
    }
}

/// Coding style flags (Table A.13).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CodingStyleFlags {
    pub(crate) raw: u8,
}

impl CodingStyleFlags {
    pub(super) fn from_u8(value: u8) -> Self {
        Self { raw: value }
    }

    #[expect(
        clippy::trivially_copy_pass_by_ref,
        reason = "the stable codec boundary borrows shared Copy metadata used across nested calls"
    )]
    pub(crate) fn has_precincts(&self) -> bool {
        (self.raw & 0x01) != 0
    }

    #[expect(
        clippy::trivially_copy_pass_by_ref,
        reason = "the stable codec boundary borrows shared Copy metadata used across nested calls"
    )]
    pub(crate) fn may_use_sop_markers(&self) -> bool {
        (self.raw & 0x02) != 0
    }

    #[expect(
        clippy::trivially_copy_pass_by_ref,
        reason = "the stable codec boundary borrows shared Copy metadata used across nested calls"
    )]
    pub(crate) fn uses_eph_marker(&self) -> bool {
        (self.raw & 0x04) != 0
    }
}

/// Code-block style flags (Table A.19).
#[derive(Debug, Clone, Copy, Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "each independent JPEG 2000 code-block style flag maps to a COD marker bit"
)]
pub(crate) struct CodeBlockStyle {
    pub(crate) selective_arithmetic_coding_bypass: bool,
    pub(crate) reset_context_probabilities: bool,
    pub(crate) termination_on_each_pass: bool,
    pub(crate) vertically_causal_context: bool,
    pub(crate) segmentation_symbols: bool,
    pub(crate) high_throughput_block_coding: bool,
    pub(crate) mixed_block_coding: bool,
}

impl CodeBlockStyle {
    pub(super) fn from_u8(value: u8) -> Self {
        Self {
            selective_arithmetic_coding_bypass: (value & 0x01) != 0,
            reset_context_probabilities: (value & 0x02) != 0,
            termination_on_each_pass: (value & 0x04) != 0,
            vertically_causal_context: (value & 0x08) != 0,
            // The predictable termination flag is only informative and
            // can therefore be ignored.
            segmentation_symbols: (value & 0x20) != 0,
            high_throughput_block_coding: (value & 0x40) != 0,
            mixed_block_coding: (value & 0x80) != 0,
        }
    }

    #[expect(
        clippy::trivially_copy_pass_by_ref,
        reason = "the stable codec boundary borrows shared Copy metadata used across nested calls"
    )]
    pub(crate) fn uses_high_throughput_block_coding(&self) -> bool {
        self.high_throughput_block_coding
    }

    pub(crate) fn allows_mixed_block_coding(self) -> bool {
        self.mixed_block_coding
    }
}

/// Quantization style (Table A.28).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QuantizationStyle {
    NoQuantization,
    ScalarDerived,
    ScalarExpounded,
}

impl QuantizationStyle {
    pub(super) fn from_u8(value: u8) -> Result<Self> {
        match value & 0x1F {
            0 => Ok(Self::NoQuantization),
            1 => Ok(Self::ScalarDerived),
            2 => Ok(Self::ScalarExpounded),
            _ => err!(ValidationError::InvalidQuantizationStyle),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct StepSize {
    pub(crate) mantissa: u16,
    pub(crate) exponent: u16,
}

/// Quantization properties, from the QCD and QCC markers (A.6.4 and A.6.5).
#[derive(Debug)]
pub(crate) struct QuantizationInfo {
    pub(crate) quantization_style: QuantizationStyle,
    pub(crate) guard_bits: u8,
    pub(crate) step_sizes: Vec<StepSize>,
}

/// Default values for coding style, from the COD marker (A.6.1).
#[derive(Debug)]
pub(crate) struct CodingStyleDefault {
    pub(crate) progression_order: ProgressionOrder,
    pub(crate) num_layers: u8,
    pub(crate) mct: bool,
    // This is the default used for all components, if not overridden by COC.
    pub(crate) component_parameters: CodingStyleComponent,
}

/// Values of coding style for each component, from the COC marker (A.6.2).
#[derive(Debug)]
pub(crate) struct CodingStyleComponent {
    pub(crate) flags: CodingStyleFlags,
    pub(crate) parameters: CodingStyleParameters,
}

/// Shared parameters between the COC and COD marker (A.6.1 and A.6.2).
#[derive(Debug)]
pub(crate) struct CodingStyleParameters {
    pub(crate) num_decomposition_levels: u8,
    pub(crate) num_resolution_levels: u8,
    pub(crate) code_block_width: u8,
    pub(crate) code_block_height: u8,
    pub(crate) code_block_style: CodeBlockStyle,
    pub(crate) transformation: WaveletTransform,
    pub(crate) precinct_exponents: Vec<(u8, u8)>,
}

crate::move_only::assert_move_only!(
    PpmMarkerData<'static>,
    PacketLengthMarker,
    ComponentInfo,
    QuantizationInfo,
    CodingStyleDefault,
    CodingStyleComponent,
    CodingStyleParameters,
);
