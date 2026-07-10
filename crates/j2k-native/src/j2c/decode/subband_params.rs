// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    bail, CodeBlock, ComponentInfo, DecodingError, J2kCodeBlockStyle, J2kSubBandType,
    QuantizationStyle, Result, SubBand, SubBandType, MAX_BITPLANE_COUNT,
};

pub(super) struct SubBandDecodeParameters {
    pub(super) dequantization_step: f32,
    pub(super) num_bitplanes: u8,
}

pub(super) fn sub_band_decode_parameters(
    sub_band: &SubBand,
    resolution: u8,
    component_info: &ComponentInfo,
) -> Result<SubBandDecodeParameters> {
    let dequantization_step = if component_info.quantization_info.quantization_style
        == QuantizationStyle::NoQuantization
    {
        1.0
    } else {
        let (exponent, mantissa) =
            component_info.exponent_mantissa(sub_band.sub_band_type, resolution)?;
        let log_gain = match sub_band.sub_band_type {
            SubBandType::LowLow => 0,
            SubBandType::LowHigh | SubBandType::HighLow => 1,
            SubBandType::HighHigh => 2,
        };
        let range_bits = u16::from(component_info.size_info.precision) + log_gain;
        crate::math::pow2i(i32::from(range_bits) - i32::from(exponent))
            * (1.0 + f32::from(mantissa) / 2048.0)
    };

    let (exponent, _) = component_info.exponent_mantissa(sub_band.sub_band_type, resolution)?;
    let num_bitplanes = u16::from(component_info.quantization_info.guard_bits)
        .checked_add(exponent)
        .and_then(|value| value.checked_sub(1))
        .ok_or(DecodingError::InvalidBitplaneCount)?;
    if num_bitplanes > u16::from(MAX_BITPLANE_COUNT) {
        bail!(DecodingError::TooManyBitplanes);
    }

    Ok(SubBandDecodeParameters {
        dequantization_step,
        num_bitplanes: u8::try_from(num_bitplanes).map_err(|_| DecodingError::TooManyBitplanes)?,
    })
}

pub(super) fn classic_decode_job_parameters(
    sub_band_type: SubBandType,
    component_info: &ComponentInfo,
) -> (J2kSubBandType, J2kCodeBlockStyle) {
    let sub_band_type = match sub_band_type {
        SubBandType::LowLow => J2kSubBandType::LowLow,
        SubBandType::HighLow => J2kSubBandType::HighLow,
        SubBandType::LowHigh => J2kSubBandType::LowHigh,
        SubBandType::HighHigh => J2kSubBandType::HighHigh,
    };
    let style = &component_info.coding_style.parameters.code_block_style;
    (
        sub_band_type,
        J2kCodeBlockStyle {
            selective_arithmetic_coding_bypass: style.selective_arithmetic_coding_bypass,
            reset_context_probabilities: style.reset_context_probabilities,
            termination_on_each_pass: style.termination_on_each_pass,
            vertically_causal_context: style.vertically_causal_context,
            segmentation_symbols: style.segmentation_symbols,
        },
    )
}

pub(super) fn ht_code_block_has_decodable_passes(
    code_block: &CodeBlock,
    coded_bitplanes: u8,
    strict: bool,
) -> Result<bool> {
    let actual_bitplanes = if strict {
        coded_bitplanes
            .checked_sub(code_block.missing_bit_planes)
            .ok_or(DecodingError::InvalidBitplaneCount)?
    } else {
        coded_bitplanes.saturating_sub(code_block.missing_bit_planes)
    };
    let max_coding_passes = if actual_bitplanes == 0 {
        0
    } else {
        1 + 3 * (actual_bitplanes - 1)
    };
    if code_block.number_of_coding_passes > max_coding_passes && strict {
        bail!(DecodingError::TooManyCodingPasses);
    }
    Ok(code_block.number_of_coding_passes != 0 && actual_bitplanes != 0)
}
