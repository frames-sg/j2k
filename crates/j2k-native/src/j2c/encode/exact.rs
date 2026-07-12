// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    allocation::checked_add_bytes, native_samples_equal, BlockCodingMode,
    EncodeComponentSampleInfo, EncodeOptions, Vec,
};
use crate::{DecodeError, DecodeSettings, DecoderContext, EncodeError, EncodeResult, Image};

#[cfg(test)]
mod tests;

#[expect(
    clippy::too_many_arguments,
    reason = "this codec boundary keeps geometry, state buffers, and validated options explicit without allocation or indirection"
)]
pub(super) fn validate_htj2k_codestream(
    codestream: &[u8],
    codestream_capacity: usize,
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    reversible: bool,
) -> EncodeResult<()> {
    let image = Image::new_with_retained_baseline(
        codestream,
        &DecodeSettings::default(),
        codestream_capacity,
    )
    .map_err(map_self_validation_decode_error)?;
    let retained_decode_baseline = checked_add_bytes(
        codestream_capacity,
        image
            .retained_allocation_bytes()
            .map_err(map_self_validation_decode_error)?,
        "HTJ2K self-validation retained output and metadata",
    )?;
    let mut decoder_context = DecoderContext::default();
    let decoded = image
        .decode_native_with_context_and_retained_baseline(
            &mut decoder_context,
            retained_decode_baseline,
        )
        .map_err(map_self_validation_decode_error)?;

    if decoded.width != width
        || decoded.height != height
        || decoded.bit_depth != bit_depth
        || decoded.num_components != num_components
    {
        return Err(EncodeError::CodestreamValidation {
            detail: "generated HTJ2K codestream failed self-validation",
        });
    }

    if reversible && !native_samples_equal(pixels, &decoded.data, bit_depth, signed) {
        return Err(EncodeError::CodestreamValidation {
            detail: "generated HTJ2K codestream did not roundtrip",
        });
    }

    Ok(())
}

fn map_self_validation_decode_error(error: DecodeError) -> EncodeError {
    match error {
        DecodeError::AllocationTooLarge {
            what,
            requested,
            cap,
        } => EncodeError::AllocationTooLarge {
            what,
            requested,
            cap,
        },
        DecodeError::HostAllocationFailed { what, bytes } => {
            EncodeError::HostAllocationFailed { what, bytes }
        }
        _ => EncodeError::CodestreamValidation {
            detail: "generated HTJ2K codestream failed self-validation",
        },
    }
}

pub(super) fn validate_reversible_i64_encode_options(
    options: &EncodeOptions,
    block_coding_mode: BlockCodingMode,
    component_sample_info: &[EncodeComponentSampleInfo],
    component_sampling: &[(u8, u8)],
) -> Result<(), &'static str> {
    if !options.reversible {
        return Err("25-38 bit encode currently requires reversible 5/3 coding");
    }
    if !matches!(
        block_coding_mode,
        BlockCodingMode::Classic | BlockCodingMode::HighThroughput
    ) {
        return Err("25-38 bit encode requires classic J2K or HTJ2K block coding");
    }
    if !component_sample_info.is_empty() {
        return Err("25-38 bit encode currently requires uniform raw-pixel component metadata");
    }
    if component_sampling
        .iter()
        .any(|sampling| *sampling != (1, 1))
    {
        return Err("25-38 bit encode currently requires full-resolution components");
    }
    Ok(())
}

pub(super) fn forward_rct_i64(components: &mut [Vec<i64>]) {
    debug_assert!(components.len() >= 3);
    let (r_components, rest) = components.split_at_mut(1);
    let (g_components, b_components) = rest.split_at_mut(1);
    let r_components = &mut r_components[0];
    let g_components = &mut g_components[0];
    let b_components = &mut b_components[0];

    for ((r, g), b) in r_components
        .iter_mut()
        .zip(g_components.iter_mut())
        .zip(b_components.iter_mut())
    {
        let r0 = *r;
        let g0 = *g;
        let b0 = *b;
        *r = (r0 + 2 * g0 + b0).div_euclid(4);
        *g = b0 - g0;
        *b = r0 - g0;
    }
}
