// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{try_encode_code_block, try_encode_code_block_view, try_encode_code_block_with_passes};
use crate::j2c::bitplane_encode::EncodedCodeBlock;
use crate::j2c::coefficient_view::{legacy_coefficient_view_error, CoefficientBlockView};

pub(crate) fn encode_code_block(
    coefficients: &[i32],
    width: u32,
    height: u32,
    total_bitplanes: u8,
) -> Result<EncodedCodeBlock, &'static str> {
    try_encode_code_block(coefficients, width, height, total_bitplanes)
        .map_err(legacy_coefficient_view_error)
}

pub(crate) fn encode_code_block_with_passes(
    coefficients: &[i32],
    width: u32,
    height: u32,
    total_bitplanes: u8,
    target_coding_passes: u8,
) -> Result<EncodedCodeBlock, &'static str> {
    try_encode_code_block_with_passes(
        coefficients,
        width,
        height,
        total_bitplanes,
        target_coding_passes,
    )
    .map_err(legacy_coefficient_view_error)
}

pub(crate) fn encode_code_block_view(
    coefficients: CoefficientBlockView<'_, i32>,
    total_bitplanes: u8,
    target_coding_passes: u8,
) -> Result<EncodedCodeBlock, &'static str> {
    try_encode_code_block_view(coefficients, total_bitplanes, target_coding_passes)
        .map_err(legacy_coefficient_view_error)
}
