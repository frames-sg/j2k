// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    encode_with_accelerator_and_component_sample_info, precomputed_level_count,
    validate_component_sample_info, validate_precomputed_dwt_geometry, zero_pixel_buffer,
    CpuOnlyJ2kEncodeStageAccelerator, EncodeComponentSampleInfo, EncodeOptions,
    J2kEncodeStageAccelerator, PrecomputedDwtAccelerator, PrecomputedHtj2k53Image, Vec,
    MAX_J2K_SPEC_COMPONENTS, MAX_RAW_PIXEL_ENCODE_BIT_DEPTH,
};

/// This mirrors [`encode_precomputed_htj2k_53`] while selecting classic EBCOT
/// block coding. It reuses the same quantization, packetization, and codestream
/// writer stages as the normal encoder and is primarily intended for fixtures
/// and coefficient-domain workflows that need JPEG-native component sampling.
#[doc(hidden)]
pub fn encode_precomputed_j2k_53(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
) -> Result<Vec<u8>, &'static str> {
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_precomputed_j2k_53_with_mct_and_accelerator(image, options, false, &mut accelerator)
}

/// Encode precomputed reversible 5/3 wavelet coefficients into a classic
/// JPEG 2000 Part 1 codestream using optional block encode and packetization
/// hooks.
#[doc(hidden)]
pub fn encode_precomputed_j2k_53_with_accelerator(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    encode_precomputed_j2k_53_with_mct_and_accelerator(image, options, false, accelerator)
}

/// Encode precomputed reversible 5/3 wavelet coefficients into a classic
/// JPEG 2000 Part 1 codestream while controlling the output COD
/// multi-component transform flag.
#[doc(hidden)]
pub fn encode_precomputed_j2k_53_with_mct(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    use_mct: bool,
) -> Result<Vec<u8>, &'static str> {
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_precomputed_j2k_53_with_mct_and_accelerator(image, options, use_mct, &mut accelerator)
}

/// Encode precomputed reversible 5/3 wavelet coefficients into a classic
/// JPEG 2000 Part 1 codestream while controlling the output COD
/// multi-component transform flag and using optional encode stage hooks.
#[doc(hidden)]
pub fn encode_precomputed_j2k_53_with_mct_and_accelerator(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    use_mct: bool,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    encode_precomputed_53_with_mct_and_accelerator(image, options, use_mct, false, accelerator)
}

/// Encode precomputed reversible 5/3 wavelet coefficients into an HTJ2K
/// codestream.
///
/// This experimental entry point reuses the existing quantization, HT block
/// coding, packetization, and codestream writer stages. It bypasses the
/// encoder's forward DWT stage by supplying precomputed DWT output through the
/// internal stage hook. Coefficients are expected in the same sample domain as
/// the native encoder's FDWT input: unsigned components are already level
/// shifted by subtracting `2^(bit_depth - 1)`.
#[doc(hidden)]
pub fn encode_precomputed_htj2k_53(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
) -> Result<Vec<u8>, &'static str> {
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_precomputed_htj2k_53_with_mct_and_accelerator(image, options, false, &mut accelerator)
}

/// Encode precomputed reversible 5/3 wavelet coefficients into an HTJ2K
/// codestream using optional block encode and packetization hooks.
#[doc(hidden)]
pub fn encode_precomputed_htj2k_53_with_accelerator(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    encode_precomputed_htj2k_53_with_mct_and_accelerator(image, options, false, accelerator)
}

/// Encode precomputed reversible 5/3 wavelet coefficients into an HTJ2K
/// codestream while controlling the output COD multi-component transform flag.
///
/// This is intended for coefficient-domain JPEG 2000 family recoding, where
/// source codestream components may already be reversible-color-transformed.
#[doc(hidden)]
pub fn encode_precomputed_htj2k_53_with_mct(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    use_mct: bool,
) -> Result<Vec<u8>, &'static str> {
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_precomputed_htj2k_53_with_mct_and_accelerator(image, options, use_mct, &mut accelerator)
}

/// Encode precomputed reversible 5/3 wavelet coefficients while controlling
/// the output COD multi-component transform flag and using optional encode
/// stage hooks.
#[doc(hidden)]
pub fn encode_precomputed_htj2k_53_with_mct_and_accelerator(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    use_mct: bool,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    encode_precomputed_53_with_mct_and_accelerator(image, options, use_mct, true, accelerator)
}

pub(super) fn encode_precomputed_53_with_mct_and_accelerator(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    use_mct: bool,
    use_ht_block_coding: bool,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    encode_precomputed_53_with_component_sample_info_and_accelerator(
        image,
        options,
        use_mct,
        use_ht_block_coding,
        &[],
        accelerator,
    )
}

pub(in crate::j2c::encode) fn encode_precomputed_53_with_component_sample_info_and_accelerator(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    use_mct: bool,
    use_ht_block_coding: bool,
    component_sample_info: &[EncodeComponentSampleInfo],
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, &'static str> {
    if image.width == 0 || image.height == 0 {
        return Err("invalid dimensions");
    }
    if image.components.is_empty() || image.components.len() > usize::from(MAX_J2K_SPEC_COMPONENTS)
    {
        return Err("unsupported component count");
    }
    if image.bit_depth == 0 || image.bit_depth > MAX_RAW_PIXEL_ENCODE_BIT_DEPTH {
        return Err("unsupported bit depth");
    }
    validate_component_sample_info(component_sample_info, image.components.len())?;
    if image
        .components
        .iter()
        .any(|component| component.x_rsiz == 0 || component.y_rsiz == 0)
    {
        return Err("component sampling factors must be non-zero");
    }
    validate_precomputed_dwt_geometry(image)?;

    let num_components =
        u16::try_from(image.components.len()).map_err(|_| "unsupported component count")?;
    let num_levels = precomputed_level_count(&image.components)?;
    let mut precomputed_options = options.clone();
    precomputed_options.num_decomposition_levels = num_levels;
    precomputed_options.reversible = true;
    precomputed_options.use_ht_block_coding = use_ht_block_coding;
    precomputed_options.use_mct = use_mct;
    precomputed_options.validate_high_throughput_codestream = false;
    precomputed_options.component_sampling = Some(
        image
            .components
            .iter()
            .map(|component| (component.x_rsiz, component.y_rsiz))
            .collect(),
    );

    let dummy_pixels =
        zero_pixel_buffer(image.width, image.height, num_components, image.bit_depth)?;
    let mut precomputed_accelerator = PrecomputedDwtAccelerator {
        outputs: image
            .components
            .iter()
            .map(|component| component.dwt.clone())
            .collect(),
        encode_accelerator: accelerator,
    };

    encode_with_accelerator_and_component_sample_info(
        &dummy_pixels,
        image.width,
        image.height,
        num_components,
        image.bit_depth,
        image.signed,
        &precomputed_options,
        component_sample_info,
        &mut precomputed_accelerator,
    )
}
