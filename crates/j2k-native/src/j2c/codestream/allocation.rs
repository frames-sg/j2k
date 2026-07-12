// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared fallible cloning for owned codestream metadata.

use alloc::vec::Vec;

use super::{CodingStyleComponent, CodingStyleParameters, QuantizationInfo};
use crate::error::Result;
use crate::try_reserve_decode_elements;

mod header;
pub(crate) use self::header::retained_header_bytes;

pub(crate) fn try_copy_vec<T: Copy>(source: &[T]) -> Result<Vec<T>> {
    let mut destination = Vec::new();
    try_reserve_decode_elements(&mut destination, source.len())?;
    destination.extend_from_slice(source);
    Ok(destination)
}

pub(crate) fn try_clone_coding_parameters(
    source: &CodingStyleParameters,
) -> Result<CodingStyleParameters> {
    Ok(CodingStyleParameters {
        num_decomposition_levels: source.num_decomposition_levels,
        num_resolution_levels: source.num_resolution_levels,
        code_block_width: source.code_block_width,
        code_block_height: source.code_block_height,
        code_block_style: source.code_block_style,
        transformation: source.transformation,
        precinct_exponents: try_copy_vec(&source.precinct_exponents)?,
    })
}

pub(crate) fn try_clone_coding_style(
    source: &CodingStyleComponent,
) -> Result<CodingStyleComponent> {
    Ok(CodingStyleComponent {
        flags: source.flags,
        parameters: try_clone_coding_parameters(&source.parameters)?,
    })
}

pub(crate) fn try_clone_quantization_info(source: &QuantizationInfo) -> Result<QuantizationInfo> {
    Ok(QuantizationInfo {
        quantization_style: source.quantization_style,
        guard_bits: source.guard_bits,
        step_sizes: try_copy_vec(&source.step_sizes)?,
    })
}
