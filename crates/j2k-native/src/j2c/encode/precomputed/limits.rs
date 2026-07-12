// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lowered-cap adapters for cross-codec native encode composition.

use super::{
    allocation::{precomputed_97_image_retained_bytes, prequantized_97_image_retained_bytes},
    api53::encode_precomputed_53_with_component_sample_info_and_accelerator_with_cap,
    api97::{
        encode_precomputed_for_session, prepare_owned_preencoded_plan, prepare_prequantized_plan,
    },
    batch97::encode_precomputed_htj2k_97_batch_owned_with_accelerator_and_cap,
    EncodeOptions, J2kEncodeStageAccelerator, NativeEncodePipelineError, NativeEncodeRetainedInput,
    NativeEncodeSession, PrecomputedHtj2k53Image, PrecomputedHtj2k97Image,
    PreencodedHtj2k97CompactImage, PreencodedHtj2k97Image, PrequantizedHtj2k97Image, Vec,
};

/// Encode precomputed reversible 5/3 coefficients under a caller-selected
/// native host ceiling.
///
/// The ceiling is always clamped to the process-wide codec cap. Cross-codec
/// pipelines use this to reserve budget for owners that remain live outside
/// native encode without forging retained-owner tokens.
#[doc(hidden)]
pub fn encode_precomputed_htj2k_53_with_accelerator_and_max_host_bytes(
    image: &PrecomputedHtj2k53Image,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
    max_host_bytes: usize,
) -> crate::EncodeResult<Vec<u8>> {
    encode_precomputed_53_with_component_sample_info_and_accelerator_with_cap(
        image,
        options,
        false,
        true,
        &[],
        accelerator,
        max_host_bytes,
    )
}

/// Encode borrowed precomputed 9/7 coefficients under a lowered native host
/// ceiling. Values above the process-wide codec cap are clamped to that cap.
#[doc(hidden)]
pub fn encode_precomputed_htj2k_97_with_accelerator_and_max_host_bytes(
    image: &PrecomputedHtj2k97Image,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
    max_host_bytes: usize,
) -> crate::EncodeResult<Vec<u8>> {
    let retained_bytes = precomputed_97_image_retained_bytes(image)?;
    let retained_input = NativeEncodeRetainedInput::from_owner_bytes(image, retained_bytes);
    let session = NativeEncodeSession::try_with_lowered_cap(retained_input, max_host_bytes)?;
    encode_precomputed_for_session(image, options, &session, accelerator)
        .map_err(NativeEncodePipelineError::into_encode_error)
}

/// Encode an owned precomputed 9/7 batch under a lowered native host ceiling.
/// Values above the process-wide codec cap are clamped to that cap.
#[doc(hidden)]
pub fn encode_precomputed_htj2k_97_batch_owned_with_accelerator_and_max_host_bytes(
    images: Vec<PrecomputedHtj2k97Image>,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
    max_host_bytes: usize,
) -> crate::EncodeResult<Vec<Vec<u8>>> {
    encode_precomputed_htj2k_97_batch_owned_with_accelerator_and_cap(
        images,
        options,
        accelerator,
        max_host_bytes,
    )
}

/// Encode moved preencoded 9/7 payloads under a lowered native host ceiling.
/// Values above the process-wide codec cap are clamped to that cap.
#[doc(hidden)]
pub fn encode_preencoded_htj2k_97_owned_with_accelerator_and_max_host_bytes(
    image: PreencodedHtj2k97Image,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
    max_host_bytes: usize,
) -> crate::EncodeResult<Vec<u8>> {
    prepare_owned_preencoded_plan(image, options, max_host_bytes)
        .and_then(|plan| {
            let session = NativeEncodeSession::try_with_lowered_cap(
                NativeEncodeRetainedInput::none(),
                max_host_bytes,
            )?;
            super::orchestrator::encode_plan(plan, &session, accelerator)
        })
        .map_err(NativeEncodePipelineError::into_encode_error)
}

/// Encode moved compact preencoded 9/7 payloads under a lowered native host
/// ceiling. Values above the process-wide codec cap are clamped to that cap.
#[doc(hidden)]
pub fn encode_preencoded_htj2k_97_compact_owned_with_accelerator_and_max_host_bytes(
    image: PreencodedHtj2k97CompactImage,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
    max_host_bytes: usize,
) -> crate::EncodeResult<Vec<u8>> {
    super::compact97::encode_preencoded_htj2k_97_compact_owned_with_accelerator(
        image,
        options,
        accelerator,
        max_host_bytes,
    )
    .map_err(NativeEncodePipelineError::into_encode_error)
}

/// Encode borrowed prequantized 9/7 coefficients under a lowered native host
/// ceiling. Values above the process-wide codec cap are clamped to that cap.
#[doc(hidden)]
pub fn encode_prequantized_htj2k_97_with_accelerator_and_max_host_bytes(
    image: &PrequantizedHtj2k97Image,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
    max_host_bytes: usize,
) -> crate::EncodeResult<Vec<u8>> {
    let retained_bytes = prequantized_97_image_retained_bytes(image)?;
    let session = NativeEncodeSession::try_with_lowered_cap(
        NativeEncodeRetainedInput::from_owner_bytes(image, retained_bytes),
        max_host_bytes,
    )?;
    prepare_prequantized_plan(image, options, &session)
        .and_then(|plan| super::orchestrator::encode_plan(plan, &session, accelerator))
        .map_err(NativeEncodePipelineError::into_encode_error)
}

#[cfg(test)]
#[path = "limits/tests.rs"]
mod tests;
