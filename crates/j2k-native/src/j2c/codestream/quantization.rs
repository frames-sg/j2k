// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::{QuantizationInfo, QuantizationStyle, StepSize};
use crate::error::{MarkerError, Result};
use crate::reader::BitReader;
use crate::try_reserve_decode_elements;

/// QCD marker (A.6.4).
pub(crate) fn qcd_marker(reader: &mut BitReader<'_>) -> Result<QuantizationInfo> {
    // Length.
    let length = reader.read_u16().ok_or(MarkerError::ParseFailure("QCD"))?;

    let sqcd_val = reader.read_byte().ok_or(MarkerError::ParseFailure("QCD"))?;
    let quantization_style = QuantizationStyle::from_u8(sqcd_val & 0x1F)
        .map_err(|_| MarkerError::ParseFailure("QCD"))?;
    let guard_bits = (sqcd_val >> 5) & 0x07;

    let remaining_bytes = usize::from(
        length
            .checked_sub(3)
            .ok_or(MarkerError::ParseFailure("QCD"))?,
    );

    let mut parameters =
        quantization_parameters(reader, quantization_style, remaining_bytes, "QCD")?;
    parameters.guard_bits = guard_bits;

    Ok(parameters)
}

/// QCC marker (A.6.5).
pub(crate) fn qcc_marker(reader: &mut BitReader<'_>, csiz: u16) -> Result<(u16, QuantizationInfo)> {
    let length = reader.read_u16().ok_or(MarkerError::ParseFailure("QCC"))?;

    let component_index = if csiz < 257 {
        u16::from(reader.read_byte().ok_or(MarkerError::ParseFailure("QCC"))?)
    } else {
        reader.read_u16().ok_or(MarkerError::ParseFailure("QCC"))?
    };

    let sqcc_val = reader.read_byte().ok_or(MarkerError::ParseFailure("QCC"))?;
    let quantization_style = QuantizationStyle::from_u8(sqcc_val & 0x1F)
        .map_err(|_| MarkerError::ParseFailure("QCC"))?;
    let guard_bits = (sqcc_val >> 5) & 0x07;

    let component_index_size = if csiz < 257 { 1 } else { 2 };
    let remaining_bytes = length
        .checked_sub(2)
        .and_then(|remaining| remaining.checked_sub(component_index_size))
        .and_then(|remaining| remaining.checked_sub(1))
        .map(usize::from)
        .ok_or(MarkerError::ParseFailure("QCC"))?;

    let mut parameters =
        quantization_parameters(reader, quantization_style, remaining_bytes, "QCC")?;
    parameters.guard_bits = guard_bits;

    Ok((component_index, parameters))
}

fn quantization_parameters(
    reader: &mut BitReader<'_>,
    quantization_style: QuantizationStyle,
    remaining_bytes: usize,
    marker: &'static str,
) -> Result<QuantizationInfo> {
    let mut step_sizes = Vec::new();

    let irreversible = |val: u16| {
        let exponent = val >> 11;
        let mantissa = val & ((1 << 11) - 1);

        StepSize { mantissa, exponent }
    };

    let step_size_count = match quantization_style {
        QuantizationStyle::NoQuantization => remaining_bytes,
        QuantizationStyle::ScalarDerived => 1,
        QuantizationStyle::ScalarExpounded => remaining_bytes / 2,
    };
    try_reserve_decode_elements(&mut step_sizes, step_size_count)?;

    match quantization_style {
        QuantizationStyle::NoQuantization => {
            // 8 bits per band (5 bits exponent, 3 bits reserved)
            for _ in 0..remaining_bytes {
                let value = u16::from(
                    reader
                        .read_byte()
                        .ok_or(MarkerError::ParseFailure(marker))?,
                );
                step_sizes.push(StepSize {
                    // Unused.
                    mantissa: 0,
                    exponent: (value >> 3),
                });
            }
        }
        QuantizationStyle::ScalarDerived => {
            let value = reader.read_u16().ok_or(MarkerError::ParseFailure(marker))?;
            step_sizes.push(irreversible(value));
        }
        QuantizationStyle::ScalarExpounded => {
            let num_bands = remaining_bytes / 2;
            for _ in 0..num_bands {
                let value = reader.read_u16().ok_or(MarkerError::ParseFailure(marker))?;

                step_sizes.push(irreversible(value));
            }
        }
    }

    Ok(QuantizationInfo {
        quantization_style,
        guard_bits: 0,
        step_sizes,
    })
}
