// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public legacy 9/7 adapters over typed, phase-accounted orchestration.

#[cfg(test)]
use super::allocation::{
    precomputed_97_image_retained_bytes, prequantized_97_image_retained_bytes,
};
use super::allocation::{preencoded_97_image_retained_bytes, ConstructionTracker};
use super::orchestrator::{self, Prepared97Metadata};
use super::{
    encode_precomputed_97_single_tile, move_preencoded_payloads_into_skeleton,
    precomputed_97_level_count, preencoded_97_level_count, prequantized_97_level_count,
    try_precomputed_options, try_preencoded_owned_skeleton,
    try_prepared_packets_from_preencoded_component,
    try_prepared_packets_from_prequantized_component, validate_irreversible_quantization_profile,
    validate_precomputed_dwt97_geometry, validate_preencoded_htj2k97_image,
    validate_prequantized_htj2k97_image, CpuOnlyJ2kEncodeStageAccelerator, EncodeOptions,
    J2kEncodeStageAccelerator, NativeEncodePipelineError, NativeEncodePipelineResult,
    NativeEncodeRetainedInput, NativeEncodeSession, PrecomputedHtj2k97Image, PrecomputedOptionMode,
    PrecomputedStageAccelerator, PreencodedHtj2k97CompactImage, PreencodedHtj2k97Image,
    PreparedResolutionPacket, PrequantizedHtj2k97Image, Vec, MAX_J2K_SPEC_COMPONENTS,
};
use crate::j2c::encode::multitile::encode_options_retained_bytes;

/// Encode precomputed irreversible 9/7 wavelet coefficients into an HTJ2K
/// codestream.
///
/// Coefficients must use the native irreversible FDWT sample domain. Unsigned
/// components are level shifted by subtracting `2^(bit_depth - 1)` before the
/// transform. The encoder borrows every supplied LL/HL/LH/HH allocation for
/// the duration of the call.
#[doc(hidden)]
pub fn encode_precomputed_htj2k_97(
    image: &PrecomputedHtj2k97Image,
    options: &EncodeOptions,
) -> crate::EncodeResult<Vec<u8>> {
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_precomputed_htj2k_97_with_accelerator(image, options, &mut accelerator)
}

/// Encode precomputed irreversible 9/7 wavelet coefficients while borrowing
/// the supplied coefficient tree directly.
#[doc(hidden)]
pub fn encode_precomputed_htj2k_97_with_accelerator(
    image: &PrecomputedHtj2k97Image,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> crate::EncodeResult<Vec<u8>> {
    super::limits::encode_precomputed_htj2k_97_with_accelerator_and_max_host_bytes(
        image,
        options,
        accelerator,
        crate::DEFAULT_MAX_CODEC_BYTES,
    )
}

pub(super) fn encode_precomputed_for_session(
    image: &PrecomputedHtj2k97Image,
    options: &EncodeOptions,
    session: &NativeEncodeSession<'_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<Vec<u8>> {
    validate_common_image(
        image.width,
        image.height,
        image.bit_depth,
        image.components.len(),
        image
            .components
            .iter()
            .map(|component| (component.x_rsiz, component.y_rsiz)),
    )?;
    validate_irreversible_quantization_profile(options)
        .map_err(NativeEncodePipelineError::invalid_input)?;
    validate_precomputed_dwt97_geometry(image).map_err(NativeEncodePipelineError::invalid_input)?;
    let num_levels = precomputed_97_level_count(&image.components)
        .map_err(NativeEncodePipelineError::invalid_input)?;
    let mut tracker = ConstructionTracker::new(session, 0);
    let adjusted = try_precomputed_options(
        options,
        image
            .components
            .iter()
            .map(|component| (component.x_rsiz, component.y_rsiz)),
        PrecomputedOptionMode {
            num_levels,
            reversible: false,
            use_ht_block_coding: true,
            use_mct: false,
        },
        &mut tracker,
    )?;
    let options_bytes = encode_options_retained_bytes(&adjusted)?;
    let adjusted_session = session.checked_child_session(
        &adjusted,
        options_bytes,
        "borrowed precomputed 9/7 options",
    )?;
    let mut stage_accelerator = PrecomputedStageAccelerator {
        encode_accelerator: accelerator,
    };
    encode_precomputed_97_single_tile(image, &adjusted, &adjusted_session, &mut stage_accelerator)
}

/// Encode prequantized irreversible 9/7 code-block coefficients into HTJ2K.
#[doc(hidden)]
pub fn encode_prequantized_htj2k_97(
    image: &PrequantizedHtj2k97Image,
    options: &EncodeOptions,
) -> crate::EncodeResult<Vec<u8>> {
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_prequantized_htj2k_97_with_accelerator(image, options, &mut accelerator)
}

/// Encode prequantized irreversible 9/7 code-block coefficients with optional
/// Tier-1 and packetization acceleration.
#[doc(hidden)]
pub fn encode_prequantized_htj2k_97_with_accelerator(
    image: &PrequantizedHtj2k97Image,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> crate::EncodeResult<Vec<u8>> {
    super::limits::encode_prequantized_htj2k_97_with_accelerator_and_max_host_bytes(
        image,
        options,
        accelerator,
        crate::DEFAULT_MAX_CODEC_BYTES,
    )
}

pub(super) fn prepare_prequantized_plan(
    image: &PrequantizedHtj2k97Image,
    options: &EncodeOptions,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<orchestrator::Prepared97PacketPlan> {
    validate_common_image(
        image.width,
        image.height,
        image.bit_depth,
        image.components.len(),
        image
            .components
            .iter()
            .map(|component| (component.x_rsiz, component.y_rsiz)),
    )?;
    let num_levels = prequantized_97_level_count(&image.components)
        .map_err(NativeEncodePipelineError::invalid_input)?;
    let mut tracker = ConstructionTracker::new(session, 0);
    let metadata = try_packet_metadata(
        image.width,
        image.height,
        image.bit_depth,
        image.signed,
        image
            .components
            .iter()
            .map(|component| (component.x_rsiz, component.y_rsiz)),
        num_levels,
        options,
        &mut tracker,
    )?;
    validate_prequantized_htj2k97_image(image, metadata.params.guard_bits, &metadata.step_sizes)
        .map_err(NativeEncodePipelineError::invalid_input)?;
    let mut components = tracker.try_vec::<Vec<_>>(
        image.components.len(),
        "prequantized 9/7 prepared component owners",
    )?;
    for (component_idx, component) in image.components.iter().enumerate() {
        components.push(try_prepared_packets_from_prequantized_component(
            component_idx,
            component,
            &mut tracker,
        )?);
    }
    orchestrator::finish_plan(metadata, components, options, session, 0)
}

/// Encode preencoded irreversible 9/7 HTJ2K code-block payloads.
#[doc(hidden)]
pub fn encode_preencoded_htj2k_97(
    image: &PreencodedHtj2k97Image,
    options: &EncodeOptions,
) -> crate::EncodeResult<Vec<u8>> {
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_preencoded_htj2k_97_with_accelerator(image, options, &mut accelerator)
}

/// Encode borrowed preencoded 9/7 payloads. Payload copies are explicit,
/// fallible, and included with the borrowed source in one retained phase.
#[doc(hidden)]
pub fn encode_preencoded_htj2k_97_with_accelerator(
    image: &PreencodedHtj2k97Image,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> crate::EncodeResult<Vec<u8>> {
    let retained_bytes = preencoded_97_image_retained_bytes(image)?;
    let session = NativeEncodeSession::try_new(NativeEncodeRetainedInput::from_owner_bytes(
        image,
        retained_bytes,
    ))?;
    prepare_borrowed_preencoded_plan(image, options, &session)
        .and_then(|plan| orchestrator::encode_plan(plan, &session, accelerator))
        .map_err(NativeEncodePipelineError::into_encode_error)
}

fn prepare_borrowed_preencoded_plan(
    image: &PreencodedHtj2k97Image,
    options: &EncodeOptions,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<orchestrator::Prepared97PacketPlan> {
    validate_preencoded_request(image)?;
    let num_levels = preencoded_97_level_count(&image.components)
        .map_err(NativeEncodePipelineError::invalid_input)?;
    let mut tracker = ConstructionTracker::new(session, 0);
    let metadata = try_packet_metadata(
        image.width,
        image.height,
        image.bit_depth,
        image.signed,
        image
            .components
            .iter()
            .map(|component| (component.x_rsiz, component.y_rsiz)),
        num_levels,
        options,
        &mut tracker,
    )?;
    validate_preencoded_htj2k97_image(image, metadata.params.guard_bits, &metadata.step_sizes)
        .map_err(NativeEncodePipelineError::invalid_input)?;
    let mut components = tracker.try_vec::<Vec<_>>(
        image.components.len(),
        "preencoded 9/7 prepared component owners",
    )?;
    for (component_idx, component) in image.components.iter().enumerate() {
        components.push(try_prepared_packets_from_preencoded_component(
            component_idx,
            component,
            &mut tracker,
        )?);
    }
    orchestrator::finish_plan(metadata, components, options, session, 0)
}

/// Encode preencoded 9/7 payloads by moving every payload vector into packet
/// preparation without cloning it.
#[doc(hidden)]
pub fn encode_preencoded_htj2k_97_owned_with_accelerator(
    image: PreencodedHtj2k97Image,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> crate::EncodeResult<Vec<u8>> {
    super::limits::encode_preencoded_htj2k_97_owned_with_accelerator_and_max_host_bytes(
        image,
        options,
        accelerator,
        crate::DEFAULT_MAX_CODEC_BYTES,
    )
}

pub(super) fn prepare_owned_preencoded_plan(
    image: PreencodedHtj2k97Image,
    options: &EncodeOptions,
    max_host_bytes: usize,
) -> NativeEncodePipelineResult<orchestrator::Prepared97PacketPlan> {
    let OwnedPreencodedHandoff {
        metadata,
        mut components,
    } = prepare_owned_preencoded_handoff(&image, options, max_host_bytes)?;
    move_preencoded_payloads_into_skeleton(image, &mut components)
        .map_err(NativeEncodePipelineError::internal_invariant)?;
    let session = NativeEncodeSession::try_with_lowered_cap(
        NativeEncodeRetainedInput::none(),
        max_host_bytes,
    )?;
    orchestrator::finish_plan(metadata, components, options, &session, 0)
}

struct OwnedPreencodedHandoff {
    metadata: Prepared97Metadata,
    components: Vec<Vec<PreparedResolutionPacket>>,
}

fn prepare_owned_preencoded_handoff(
    image: &PreencodedHtj2k97Image,
    options: &EncodeOptions,
    max_host_bytes: usize,
) -> NativeEncodePipelineResult<OwnedPreencodedHandoff> {
    validate_preencoded_request(image)?;
    let num_levels = preencoded_97_level_count(&image.components)
        .map_err(NativeEncodePipelineError::invalid_input)?;
    let input_bytes = preencoded_97_image_retained_bytes(image)?;
    let input_session = NativeEncodeSession::try_with_lowered_cap(
        NativeEncodeRetainedInput::from_owner_bytes(image, input_bytes),
        max_host_bytes,
    )?;
    let mut tracker = ConstructionTracker::new(&input_session, 0);
    let metadata = try_packet_metadata(
        image.width,
        image.height,
        image.bit_depth,
        image.signed,
        image
            .components
            .iter()
            .map(|component| (component.x_rsiz, component.y_rsiz)),
        num_levels,
        options,
        &mut tracker,
    )?;
    validate_preencoded_htj2k97_image(image, metadata.params.guard_bits, &metadata.step_sizes)
        .map_err(NativeEncodePipelineError::invalid_input)?;
    let components = try_preencoded_owned_skeleton(image, &mut tracker)?;
    Ok(OwnedPreencodedHandoff {
        metadata,
        components,
    })
}

fn validate_preencoded_request(image: &PreencodedHtj2k97Image) -> NativeEncodePipelineResult<()> {
    validate_common_image(
        image.width,
        image.height,
        image.bit_depth,
        image.components.len(),
        image
            .components
            .iter()
            .map(|component| (component.x_rsiz, component.y_rsiz)),
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "the coefficient-image plan keeps validated geometry explicit"
)]
fn try_packet_metadata(
    width: u32,
    height: u32,
    bit_depth: u8,
    signed: bool,
    sampling: impl ExactSizeIterator<Item = (u8, u8)>,
    num_levels: u8,
    options: &EncodeOptions,
    tracker: &mut ConstructionTracker<'_, '_>,
) -> NativeEncodePipelineResult<Prepared97Metadata> {
    orchestrator::try_metadata(
        width, height, bit_depth, signed, sampling, num_levels, options, tracker,
    )
}

fn validate_common_image(
    width: u32,
    height: u32,
    bit_depth: u8,
    component_count: usize,
    mut sampling: impl Iterator<Item = (u8, u8)>,
) -> NativeEncodePipelineResult<()> {
    if width == 0 || height == 0 {
        return Err(NativeEncodePipelineError::invalid_input(
            "invalid dimensions",
        ));
    }
    if component_count == 0 {
        return Err(NativeEncodePipelineError::invalid_input(
            "component set must be non-empty",
        ));
    }
    if component_count > usize::from(MAX_J2K_SPEC_COMPONENTS) {
        return Err(NativeEncodePipelineError::unsupported(
            "component count exceeds the JPEG 2000 Part 1 limit",
        ));
    }
    if bit_depth == 0 {
        return Err(NativeEncodePipelineError::invalid_input(
            "bit depth must be non-zero",
        ));
    }
    if bit_depth > 16 {
        return Err(NativeEncodePipelineError::unsupported(
            "precomputed 9/7 bit depth exceeds 16 bits",
        ));
    }
    if sampling.any(|(x_rsiz, y_rsiz)| x_rsiz == 0 || y_rsiz == 0) {
        return Err(NativeEncodePipelineError::invalid_input(
            "component sampling factors must be non-zero",
        ));
    }
    Ok(())
}

/// Encode compact preencoded irreversible 9/7 HTJ2K payloads.
#[doc(hidden)]
pub fn encode_preencoded_htj2k_97_compact_owned_with_accelerator(
    image: PreencodedHtj2k97CompactImage,
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> crate::EncodeResult<Vec<u8>> {
    super::limits::encode_preencoded_htj2k_97_compact_owned_with_accelerator_and_max_host_bytes(
        image,
        options,
        accelerator,
        crate::DEFAULT_MAX_CODEC_BYTES,
    )
}

#[cfg(test)]
#[path = "api97/tests.rs"]
mod tests;
