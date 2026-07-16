// SPDX-License-Identifier: MIT OR Apache-2.0

use super::allocation::{precomputed_53_image_retained_bytes, ConstructionTracker};
use super::{
    encode_precomputed_53_single_tile, precomputed_level_count, try_precomputed_options,
    validate_component_sample_info, validate_precomputed_dwt_geometry,
    CpuOnlyJ2kEncodeStageAccelerator, EncodeComponentSampleInfo, EncodeOptions,
    J2kEncodeStageAccelerator, NativeEncodePipelineError, NativeEncodePipelineResult,
    NativeEncodeRetainedInput, NativeEncodeSession, PrecomputedHtj2k53Image, PrecomputedOptionMode,
    PrecomputedStageAccelerator, Vec, MAX_J2K_SPEC_COMPONENTS, MAX_RAW_PIXEL_ENCODE_BIT_DEPTH,
};
use crate::j2c::encode::multitile::encode_options_retained_bytes;

/// This mirrors [`encode_precomputed_htj2k_53`] while selecting classic EBCOT
/// block coding. It reuses the same quantization, packetization, and codestream
/// writer stages as the normal encoder and is primarily intended for fixtures
/// and coefficient-domain workflows that need JPEG-native component sampling.
#[doc(hidden)]
pub fn encode_precomputed_j2k_53(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
) -> crate::EncodeResult<Vec<u8>> {
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
) -> crate::EncodeResult<Vec<u8>> {
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
) -> crate::EncodeResult<Vec<u8>> {
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
) -> crate::EncodeResult<Vec<u8>> {
    encode_precomputed_53_with_component_sample_info_and_accelerator_with_cap(
        image,
        options,
        use_mct,
        false,
        &[],
        accelerator,
        crate::DEFAULT_MAX_CODEC_BYTES,
    )
}

/// Encode precomputed reversible 5/3 wavelet coefficients into an HTJ2K
/// codestream.
///
/// This experimental entry point reuses the existing quantization, HT block
/// coding, packetization, and codestream writer stages. It bypasses sample
/// staging, color transform, and forward DWT by borrowing the supplied subbands
/// directly. Coefficients are expected in the same sample domain as the native
/// encoder's FDWT input: unsigned components are already level shifted by
/// subtracting `2^(bit_depth - 1)`.
#[doc(hidden)]
pub fn encode_precomputed_htj2k_53(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
) -> crate::EncodeResult<Vec<u8>> {
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
) -> crate::EncodeResult<Vec<u8>> {
    encode_precomputed_53_with_component_sample_info_and_accelerator_with_cap(
        image,
        options,
        false,
        true,
        &[],
        accelerator,
        crate::DEFAULT_MAX_CODEC_BYTES,
    )
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
) -> crate::EncodeResult<Vec<u8>> {
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
) -> crate::EncodeResult<Vec<u8>> {
    encode_precomputed_53_with_component_sample_info_and_accelerator_with_cap(
        image,
        options,
        use_mct,
        true,
        &[],
        accelerator,
        crate::DEFAULT_MAX_CODEC_BYTES,
    )
}

pub(in crate::j2c) fn encode_precomputed_htj2k_53_with_mct_and_retained_owner<O: ?Sized>(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    use_mct: bool,
    owner: &O,
    retained_bytes: usize,
) -> crate::EncodeResult<Vec<u8>> {
    let retained_input = NativeEncodeRetainedInput::from_owner_bytes(owner, retained_bytes);
    let session = NativeEncodeSession::try_new(retained_input)?;
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_precomputed_53_with_component_sample_info_for_session(
        image,
        options,
        use_mct,
        true,
        &[],
        &session,
        &mut accelerator,
    )
    .map_err(NativeEncodePipelineError::into_encode_error)
}

pub(super) fn encode_precomputed_53_with_component_sample_info_and_accelerator_with_cap(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    use_mct: bool,
    use_ht_block_coding: bool,
    component_sample_info: &[EncodeComponentSampleInfo],
    accelerator: &mut impl J2kEncodeStageAccelerator,
    max_host_bytes: usize,
) -> crate::EncodeResult<Vec<u8>> {
    let retained_bytes = precomputed_53_image_retained_bytes(image)?;
    let session = NativeEncodeSession::try_with_lowered_cap(
        NativeEncodeRetainedInput::from_owner_bytes(image, retained_bytes),
        max_host_bytes,
    )?;
    encode_precomputed_53_with_component_sample_info_for_session(
        image,
        options,
        use_mct,
        use_ht_block_coding,
        component_sample_info,
        &session,
        accelerator,
    )
    .map_err(NativeEncodePipelineError::into_encode_error)
}

pub(in crate::j2c::encode) fn encode_precomputed_53_with_component_sample_info_for_session(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    use_mct: bool,
    use_ht_block_coding: bool,
    component_sample_info: &[EncodeComponentSampleInfo],
    session: &NativeEncodeSession<'_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<Vec<u8>> {
    if image.width == 0 || image.height == 0 {
        return Err(NativeEncodePipelineError::invalid_input(
            "invalid dimensions",
        ));
    }
    if image.components.is_empty() {
        return Err(NativeEncodePipelineError::invalid_input(
            "component set must be non-empty",
        ));
    }
    if image.components.len() > usize::from(MAX_J2K_SPEC_COMPONENTS) {
        return Err(NativeEncodePipelineError::unsupported(
            "component count exceeds the JPEG 2000 Part 1 limit",
        ));
    }
    if image.bit_depth == 0 {
        return Err(NativeEncodePipelineError::invalid_input(
            "bit depth must be non-zero",
        ));
    }
    if image.bit_depth > MAX_RAW_PIXEL_ENCODE_BIT_DEPTH {
        return Err(NativeEncodePipelineError::unsupported(
            "precomputed 5/3 bit depth exceeds the native encode limit",
        ));
    }
    validate_component_sample_info(component_sample_info, image.components.len())
        .map_err(NativeEncodePipelineError::invalid_input)?;
    if image
        .components
        .iter()
        .any(|component| component.x_rsiz == 0 || component.y_rsiz == 0)
    {
        return Err(NativeEncodePipelineError::invalid_input(
            "component sampling factors must be non-zero",
        ));
    }
    validate_precomputed_dwt_geometry(image).map_err(NativeEncodePipelineError::invalid_input)?;

    let num_levels = precomputed_level_count(&image.components)
        .map_err(NativeEncodePipelineError::invalid_input)?;
    let mut tracker = ConstructionTracker::new(session, 0);
    let precomputed_options = try_precomputed_options(
        options,
        image
            .components
            .iter()
            .map(|component| (component.x_rsiz, component.y_rsiz)),
        PrecomputedOptionMode {
            num_levels,
            reversible: true,
            use_ht_block_coding,
            use_mct,
        },
        &mut tracker,
    );
    let precomputed_options = precomputed_options?;
    let options_bytes = encode_options_retained_bytes(&precomputed_options)?;
    let options_session = session.checked_child_session(
        &precomputed_options,
        options_bytes,
        "borrowed precomputed 5/3 options",
    )?;

    let mut stage_accelerator = PrecomputedStageAccelerator {
        encode_accelerator: accelerator,
    };
    encode_precomputed_53_single_tile(
        image,
        &precomputed_options,
        component_sample_info,
        &options_session,
        &mut stage_accelerator,
    )
}

#[cfg(test)]
#[path = "api53/tests.rs"]
mod tests;
