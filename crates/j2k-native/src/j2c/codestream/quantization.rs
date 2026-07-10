// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::{QuantizationInfo, QuantizationStyle, StepSize};
use crate::reader::BitReader;

/// QCD marker (A.6.4).
pub(crate) fn qcd_marker(reader: &mut BitReader<'_>) -> Option<QuantizationInfo> {
    // Length.
    let length = reader.read_u16()?;

    let sqcd_val = reader.read_byte()?;
    let quantization_style = QuantizationStyle::from_u8(sqcd_val & 0x1F).ok()?;
    let guard_bits = (sqcd_val >> 5) & 0x07;

    let remaining_bytes = length.checked_sub(3)? as usize;

    let mut parameters = quantization_parameters(reader, quantization_style, remaining_bytes)?;
    parameters.guard_bits = guard_bits;

    Some(parameters)
}

/// QCC marker (A.6.5).
pub(crate) fn qcc_marker(reader: &mut BitReader<'_>, csiz: u16) -> Option<(u16, QuantizationInfo)> {
    let length = reader.read_u16()?;

    let component_index = if csiz < 257 {
        u16::from(reader.read_byte()?)
    } else {
        reader.read_u16()?
    };

    let sqcc_val = reader.read_byte()?;
    let quantization_style = QuantizationStyle::from_u8(sqcc_val & 0x1F).ok()?;
    let guard_bits = (sqcc_val >> 5) & 0x07;

    let component_index_size = if csiz < 257 { 1 } else { 2 };
    let remaining_bytes = length
        .checked_sub(2)?
        .checked_sub(component_index_size)?
        .checked_sub(1)? as usize;

    let mut parameters = quantization_parameters(reader, quantization_style, remaining_bytes)?;
    parameters.guard_bits = guard_bits;

    Some((component_index, parameters))
}

fn quantization_parameters(
    reader: &mut BitReader<'_>,
    quantization_style: QuantizationStyle,
    remaining_bytes: usize,
) -> Option<QuantizationInfo> {
    let mut step_sizes = Vec::new();

    let irreversible = |val: u16| {
        let exponent = val >> 11;
        let mantissa = val & ((1 << 11) - 1);

        StepSize { mantissa, exponent }
    };

    match quantization_style {
        QuantizationStyle::NoQuantization => {
            // 8 bits per band (5 bits exponent, 3 bits reserved)
            for _ in 0..remaining_bytes {
                let value = u16::from(reader.read_byte()?);
                step_sizes.push(StepSize {
                    // Unused.
                    mantissa: 0,
                    exponent: (value >> 3),
                });
            }
        }
        QuantizationStyle::ScalarDerived => {
            let value = reader.read_u16()?;
            step_sizes.push(irreversible(value));
        }
        QuantizationStyle::ScalarExpounded => {
            let num_bands = remaining_bytes / 2;
            for _ in 0..num_bands {
                let value = reader.read_u16()?;

                step_sizes.push(irreversible(value));
            }
        }
    }

    Some(QuantizationInfo {
        quantization_style,
        guard_bits: 0,
        step_sizes,
    })
}
