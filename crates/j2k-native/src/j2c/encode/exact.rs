// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    native_samples_equal, raw_pixel_bytes_per_sample, read_le_sample_value, sign_extend_sample,
    vec, BlockCodingMode, DecodeSettings, EncodeComponentSampleInfo, EncodeOptions, Image, Vec,
};

pub(super) fn validate_htj2k_codestream(
    codestream: &[u8],
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    reversible: bool,
) -> Result<(), &'static str> {
    let image = Image::new(codestream, &DecodeSettings::default())
        .map_err(|_| "generated HTJ2K codestream failed self-validation")?;
    let decoded = image
        .decode_native()
        .map_err(|_| "generated HTJ2K codestream failed self-validation")?;

    if decoded.width != width
        || decoded.height != height
        || decoded.bit_depth != bit_depth
        || decoded.num_components != num_components
    {
        return Err("generated HTJ2K codestream failed self-validation");
    }

    if reversible && !native_samples_equal(pixels, &decoded.data, bit_depth, signed) {
        return Err("generated HTJ2K codestream did not roundtrip");
    }

    Ok(())
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

pub(super) fn deinterleave_to_i64(
    pixels: &[u8],
    num_pixels: usize,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
) -> Vec<Vec<i64>> {
    let nc = num_components as usize;
    let mut components = vec![vec![0_i64; num_pixels]; nc];
    let unsigned_offset = if signed {
        0
    } else {
        1_i64 << (u32::from(bit_depth) - 1)
    };
    let bytes_per_sample = raw_pixel_bytes_per_sample(bit_depth).unwrap_or(5);
    for (i, pixel) in pixels
        .chunks_exact(nc * bytes_per_sample)
        .take(num_pixels)
        .enumerate()
    {
        for (component_idx, component) in components.iter_mut().enumerate().take(nc) {
            let offset = component_idx * bytes_per_sample;
            let raw = read_le_sample_value(&pixel[offset..offset + bytes_per_sample], bit_depth);
            component[i] = if signed {
                sign_extend_sample(raw, bit_depth)
            } else {
                i64::try_from(raw).unwrap_or(i64::MAX) - unsigned_offset
            };
        }
    }
    components
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
