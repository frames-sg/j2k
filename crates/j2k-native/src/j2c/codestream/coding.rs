// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::{
    CodeBlockStyle, CodingStyleComponent, CodingStyleDefault, CodingStyleFlags,
    CodingStyleParameters, ProgressionOrder, WaveletTransform,
};
use crate::error::{MarkerError, Result};
use crate::reader::BitReader;
use crate::try_reserve_decode_elements;

const MAX_LAYER_COUNT: u8 = 32;
const MAX_RESOLUTION_COUNT: u8 = 32;
const MAX_PRECINCT_EXPONENT: u8 = 31;

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the stable codec boundary borrows shared Copy metadata used across nested calls"
)]
fn coding_style_parameters(
    reader: &mut BitReader<'_>,
    coding_style: &CodingStyleFlags,
    marker: &'static str,
) -> Result<CodingStyleParameters> {
    let num_decomposition_levels = reader
        .read_byte()
        .ok_or(MarkerError::ParseFailure(marker))?;

    if num_decomposition_levels > MAX_RESOLUTION_COUNT {
        return Err(MarkerError::ParseFailure(marker).into());
    }

    let num_resolution_levels = num_decomposition_levels
        .checked_add(1)
        .ok_or(MarkerError::ParseFailure(marker))?;
    let code_block_width = reader
        .read_byte()
        .and_then(|value| value.checked_add(2))
        .ok_or(MarkerError::ParseFailure(marker))?;
    let code_block_height = reader
        .read_byte()
        .and_then(|value| value.checked_add(2))
        .ok_or(MarkerError::ParseFailure(marker))?;
    let code_block_style = CodeBlockStyle::from_u8(
        reader
            .read_byte()
            .ok_or(MarkerError::ParseFailure(marker))?,
    );
    let transformation = WaveletTransform::from_u8(
        reader
            .read_byte()
            .ok_or(MarkerError::ParseFailure(marker))?,
    )
    .map_err(|_| MarkerError::ParseFailure(marker))?;

    let mut precinct_exponents = Vec::new();
    try_reserve_decode_elements(&mut precinct_exponents, usize::from(num_resolution_levels))?;
    if coding_style.has_precincts() {
        // "Entropy coder with precincts defined below."
        for _ in 0..num_resolution_levels {
            // Table A.21.
            let precinct_size = reader
                .read_byte()
                .ok_or(MarkerError::ParseFailure(marker))?;
            let width_exp = precinct_size & 0xF;
            let height_exp = precinct_size >> 4;

            if width_exp > MAX_PRECINCT_EXPONENT || height_exp > MAX_PRECINCT_EXPONENT {
                return Err(MarkerError::ParseFailure(marker).into());
            }

            precinct_exponents.push((width_exp, height_exp));
        }
    } else {
        // "Entropy coder, precincts with PPx = 15 and PPy = 15"
        for _ in 0..num_resolution_levels {
            precinct_exponents.push((15, 15));
        }
    }

    Ok(CodingStyleParameters {
        num_decomposition_levels,
        num_resolution_levels,
        code_block_width,
        code_block_height,
        code_block_style,
        transformation,
        precinct_exponents,
    })
}

/// COD marker (A.6.1).
pub(crate) fn cod_marker(reader: &mut BitReader<'_>) -> Result<CodingStyleDefault> {
    // Length.
    let _ = reader.read_u16().ok_or(MarkerError::ParseFailure("COD"))?;

    let coding_style_flags =
        CodingStyleFlags::from_u8(reader.read_byte().ok_or(MarkerError::ParseFailure("COD"))?);
    let progression_order =
        ProgressionOrder::from_u8(reader.read_byte().ok_or(MarkerError::ParseFailure("COD"))?)
            .map_err(|_| MarkerError::ParseFailure("COD"))?;

    let num_layers = reader.read_u16().ok_or(MarkerError::ParseFailure("COD"))?;

    // We don't support more than 32-bit (and thus 32 layers).
    if num_layers == 0 || num_layers > u16::from(MAX_LAYER_COUNT) {
        return Err(MarkerError::ParseFailure("COD").into());
    }

    let mct = reader.read_byte().ok_or(MarkerError::ParseFailure("COD"))? == 1;

    let coding_style_parameters = coding_style_parameters(reader, &coding_style_flags, "COD")?;

    Ok(CodingStyleDefault {
        progression_order,
        num_layers: u8::try_from(num_layers).map_err(|_| MarkerError::ParseFailure("COD"))?,
        mct,
        component_parameters: CodingStyleComponent {
            flags: coding_style_flags,
            parameters: coding_style_parameters,
        },
    })
}

/// COC marker (A.6.2).
pub(crate) fn coc_marker(
    reader: &mut BitReader<'_>,
    csiz: u16,
) -> Result<(u16, CodingStyleComponent)> {
    // Length.
    let _ = reader.read_u16().ok_or(MarkerError::ParseFailure("COC"))?;

    let component_index = if csiz < 257 {
        u16::from(reader.read_byte().ok_or(MarkerError::ParseFailure("COC"))?)
    } else {
        reader.read_u16().ok_or(MarkerError::ParseFailure("COC"))?
    };
    let coding_style =
        CodingStyleFlags::from_u8(reader.read_byte().ok_or(MarkerError::ParseFailure("COC"))?);

    let parameters = coding_style_parameters(reader, &coding_style, "COC")?;

    let coc = CodingStyleComponent {
        flags: coding_style,
        parameters,
    };

    Ok((component_index, coc))
}
