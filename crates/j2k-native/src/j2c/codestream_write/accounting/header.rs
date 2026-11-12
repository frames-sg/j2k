// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact sizing for the allocation-free codestream main-header writer.

use super::super::{BlockCodingMode, EncodeParams};
use crate::j2c::codestream::component_index_bytes;
use crate::j2c::encode::allocation::{checked_add_bytes, checked_element_bytes};
use crate::{EncodeError, EncodeResult};

pub(super) fn main_header_prefix_len(
    params: &EncodeParams,
    quantization_step_sizes: &[(u16, u16)],
) -> EncodeResult<usize> {
    let mut bytes = 2usize;
    let siz_payload = checked_element_bytes::<[u8; 3]>(
        usize::from(params.num_components),
        "codestream SIZ component bytes",
    )?;
    bytes = checked_add_bytes(bytes, 40, "codestream SIZ marker bytes")?;
    bytes = checked_add_bytes(bytes, siz_payload, "codestream SIZ marker bytes")?;
    if params.block_coding_mode == BlockCodingMode::HighThroughput {
        bytes = checked_add_bytes(bytes, 10, "codestream CAP marker bytes")?;
    }
    u16::try_from(params.precinct_exponents.len())
        .ok()
        .and_then(|count| count.checked_add(12))
        .ok_or(EncodeError::InvalidInput {
            what: "codestream precinct exponent count exceeds COD marker capacity",
        })?;
    bytes = checked_add_bytes(bytes, 14, "codestream COD marker bytes")?;
    bytes = checked_add_bytes(
        bytes,
        params.precinct_exponents.len(),
        "codestream COD precinct bytes",
    )?;
    bytes = checked_add_bytes(
        bytes,
        quantization_marker_bytes(
            params.reversible,
            quantization_step_sizes.len(),
            0,
            "QCD marker length exceeds u16",
        )?,
        "codestream QCD marker bytes",
    )?;
    for steps in params
        .component_quantization_step_sizes
        .iter()
        .take(usize::from(params.num_components))
    {
        if !steps.is_empty() {
            let component_bytes = usize::from(component_index_bytes(params.num_components));
            bytes = checked_add_bytes(
                bytes,
                quantization_marker_bytes(
                    params.reversible,
                    steps.len(),
                    component_bytes,
                    "QCC marker length exceeds u16",
                )?,
                "codestream QCC marker bytes",
            )?;
        }
    }
    // Marker, Lrgn, Crgn, Srgn and SPrgn.
    let rgn_bytes = 6 + usize::from(component_index_bytes(params.num_components));
    for shift in params
        .roi_component_shifts
        .iter()
        .take(usize::from(params.num_components))
    {
        if *shift != 0 {
            bytes = checked_add_bytes(bytes, rgn_bytes, "codestream RGN marker bytes")?;
        }
    }
    Ok(bytes)
}

fn quantization_marker_bytes(
    reversible: bool,
    step_count: usize,
    component_bytes: usize,
    marker_length_error: &'static str,
) -> EncodeResult<usize> {
    let step_count_u16 = u16::try_from(step_count).map_err(|_| EncodeError::InvalidInput {
        what: "codestream quantization step count exceeds marker capacity",
    })?;
    let step_bytes = if reversible {
        usize::from(step_count_u16)
    } else {
        usize::from(
            step_count_u16
                .checked_mul(2)
                .ok_or(EncodeError::ArithmeticOverflow {
                    what: "codestream quantization step bytes",
                })?,
        )
    };
    let marker_bytes = 5usize
        .checked_add(component_bytes)
        .and_then(|bytes| bytes.checked_add(step_bytes))
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "codestream quantization marker bytes",
        })?;
    let marker_payload = marker_bytes
        .checked_sub(2)
        .ok_or(EncodeError::InternalInvariant {
            what: "codestream quantization marker length underflowed",
        })?;
    u16::try_from(marker_payload).map_err(|_| EncodeError::InvalidInput {
        what: marker_length_error,
    })?;
    Ok(marker_bytes)
}
