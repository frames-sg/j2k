// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reversible 5/3 encoding for independently typed component planes.

use alloc::vec::Vec;

use j2k_codec_math::dwt::max_decomposition_levels;

use super::{
    checked_add_bytes, checked_element_bytes, cpu_dwt_transient_bytes,
    dwt_decompositions_retained_bytes,
    encode_precomputed_53_with_component_sample_info_for_session,
    encode_typed_component_planes_53_i64, fdwt, forward_dwt53_output_retained_bytes,
    host_allocation_failed, try_component_plane_to_f32_for_session,
    try_forward_dwt53_output_from_decomposition, validate_code_block_geometry,
    CpuOnlyJ2kEncodeStageAccelerator, EncodeComponentSampleInfo, EncodeOptions,
    EncodeTypedComponentPlane, J2kForwardDwt53Output, NativeEncodePipelineError,
    NativeEncodePipelineResult, NativeEncodeSession, PrecomputedHtj2k53Component,
    PrecomputedHtj2k53Image, MAX_J2K_SPEC_COMPONENTS, MAX_PART1_SAMPLE_BIT_DEPTH,
    MAX_RAW_PIXEL_ENCODE_BIT_DEPTH,
};

struct TypedComponentPlan {
    num_levels: u8,
    max_bit_depth: u8,
    sample_info: Vec<EncodeComponentSampleInfo>,
    sample_info_bytes: usize,
}

pub(super) fn encode_typed_component_planes_53_for_session(
    planes: &[EncodeTypedComponentPlane<'_>],
    width: u32,
    height: u32,
    options: &EncodeOptions,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<Vec<u8>> {
    validate_typed_component_request(planes, width, height, options)?;
    if planes
        .iter()
        .any(|plane| plane.bit_depth > MAX_RAW_PIXEL_ENCODE_BIT_DEPTH)
    {
        return encode_typed_component_planes_53_i64(planes, width, height, options, session);
    }

    let plan = try_typed_component_plan(planes, width, height, options, session)?;
    let components = try_prepare_components(
        planes,
        width,
        height,
        plan.num_levels,
        plan.sample_info_bytes,
        session,
    )?;
    encode_prepared_components(planes, width, height, options, &plan, components, session)
}

fn validate_typed_component_request(
    planes: &[EncodeTypedComponentPlane<'_>],
    width: u32,
    height: u32,
    options: &EncodeOptions,
) -> NativeEncodePipelineResult<()> {
    if width == 0 || height == 0 {
        return Err(NativeEncodePipelineError::invalid_input(
            "invalid dimensions",
        ));
    }
    if planes.is_empty() {
        return Err(NativeEncodePipelineError::invalid_input(
            "component planes must be non-empty",
        ));
    }
    if planes.len() > usize::from(MAX_J2K_SPEC_COMPONENTS) {
        return Err(NativeEncodePipelineError::unsupported(
            "component count exceeds the JPEG 2000 Part 1 limit",
        ));
    }
    if planes
        .iter()
        .any(|plane| plane.x_rsiz == 0 || plane.y_rsiz == 0)
    {
        return Err(NativeEncodePipelineError::invalid_input(
            "component sampling factors must be non-zero",
        ));
    }
    if planes.iter().any(|plane| plane.bit_depth == 0) {
        return Err(NativeEncodePipelineError::invalid_input(
            "component bit depth must be non-zero",
        ));
    }
    if planes
        .iter()
        .any(|plane| plane.bit_depth > MAX_PART1_SAMPLE_BIT_DEPTH)
    {
        return Err(NativeEncodePipelineError::unsupported(
            "component bit depth exceeds the JPEG 2000 Part 1 limit",
        ));
    }
    validate_code_block_geometry(options).map_err(NativeEncodePipelineError::invalid_input)?;
    Ok(())
}

fn try_typed_component_plan(
    planes: &[EncodeTypedComponentPlane<'_>],
    width: u32,
    height: u32,
    options: &EncodeOptions,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<TypedComponentPlan> {
    let max_levels = planes
        .iter()
        .map(|plane| {
            let component_width = width.div_ceil(u32::from(plane.x_rsiz));
            let component_height = height.div_ceil(u32::from(plane.y_rsiz));
            max_decomposition_levels(component_width, component_height)
        })
        .min()
        .unwrap_or(0);
    let max_bit_depth = planes
        .iter()
        .map(|plane| plane.bit_depth)
        .max()
        .ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant(
                "validated typed component set is unexpectedly empty",
            )
        })?;
    let requested_bytes = checked_element_bytes::<EncodeComponentSampleInfo>(
        planes.len(),
        "typed component sample metadata",
    )?;
    session.checked_phase(requested_bytes, "typed component sample metadata")?;
    let mut sample_info = Vec::new();
    sample_info
        .try_reserve_exact(planes.len())
        .map_err(|_| host_allocation_failed("typed component sample metadata", requested_bytes))?;
    sample_info.extend(planes.iter().map(|plane| EncodeComponentSampleInfo {
        bit_depth: plane.bit_depth,
        signed: plane.signed,
    }));
    let sample_info_bytes = checked_element_bytes::<EncodeComponentSampleInfo>(
        sample_info.capacity(),
        "typed component sample metadata",
    )?;
    session.checked_phase(sample_info_bytes, "typed component sample metadata")?;
    Ok(TypedComponentPlan {
        num_levels: options.num_decomposition_levels.min(max_levels),
        max_bit_depth,
        sample_info,
        sample_info_bytes,
    })
}

fn try_prepare_components(
    planes: &[EncodeTypedComponentPlane<'_>],
    width: u32,
    height: u32,
    num_levels: u8,
    sample_info_bytes: usize,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<Vec<PrecomputedHtj2k53Component>> {
    let requested_owner_bytes = checked_element_bytes::<PrecomputedHtj2k53Component>(
        planes.len(),
        "typed component DWT owners",
    )?;
    session.checked_phase(
        checked_add_bytes(
            sample_info_bytes,
            requested_owner_bytes,
            "typed component construction owners",
        )?,
        "typed component construction owners",
    )?;
    let mut components = Vec::new();
    components
        .try_reserve_exact(planes.len())
        .map_err(|_| host_allocation_failed("typed component DWT owners", requested_owner_bytes))?;
    let component_owner_bytes = checked_element_bytes::<PrecomputedHtj2k53Component>(
        components.capacity(),
        "typed component DWT owners",
    )?;
    session.checked_phase(
        checked_add_bytes(
            sample_info_bytes,
            component_owner_bytes,
            "typed component construction owners",
        )?,
        "typed component construction owners",
    )?;

    for plane in planes {
        let prior_component_bytes = precomputed_53_components_retained_bytes(&components)?;
        let retained_before_samples = checked_add_bytes(
            sample_info_bytes,
            prior_component_bytes,
            "typed component construction",
        )?;
        let dwt = try_prepare_component_dwt(
            plane,
            width,
            height,
            num_levels,
            retained_before_samples,
            session,
        )?;
        components.push(PrecomputedHtj2k53Component {
            x_rsiz: plane.x_rsiz,
            y_rsiz: plane.y_rsiz,
            dwt,
        });
        session.checked_phase(
            checked_add_bytes(
                sample_info_bytes,
                precomputed_53_components_retained_bytes(&components)?,
                "typed component retained DWT graph",
            )?,
            "typed component retained DWT graph",
        )?;
    }
    Ok(components)
}

fn try_prepare_component_dwt(
    plane: &EncodeTypedComponentPlane<'_>,
    image_width: u32,
    image_height: u32,
    num_levels: u8,
    retained_before_samples: usize,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<J2kForwardDwt53Output> {
    let component_width = image_width.div_ceil(u32::from(plane.x_rsiz));
    let component_height = image_height.div_ceil(u32::from(plane.y_rsiz));
    let samples = try_component_plane_to_f32_for_session(
        plane.data,
        component_width,
        component_height,
        plane.bit_depth,
        plane.signed,
        session,
        retained_before_samples,
    )?;
    let sample_bytes =
        checked_element_bytes::<f32>(samples.capacity(), "typed component floating-point samples")?;
    let transient_bytes = cpu_dwt_transient_bytes(samples.len(), num_levels)?;
    session.checked_phase(
        checked_add_bytes(
            retained_before_samples,
            checked_add_bytes(
                sample_bytes,
                transient_bytes,
                "typed component DWT transient",
            )?,
            "typed component DWT transient",
        )?,
        "typed component DWT transient",
    )?;
    let decomposition = fdwt::try_forward_dwt(
        &samples,
        component_width,
        component_height,
        num_levels,
        true,
    )?;
    let decomposition_bytes =
        dwt_decompositions_retained_bytes(core::slice::from_ref(&decomposition), 0)?;
    session.checked_phase(
        checked_add_bytes(
            retained_before_samples,
            checked_add_bytes(
                sample_bytes,
                decomposition_bytes,
                "typed component DWT output",
            )?,
            "typed component DWT output",
        )?,
        "typed component DWT output",
    )?;
    let dwt = try_forward_dwt53_output_from_decomposition(
        decomposition,
        session,
        checked_add_bytes(
            retained_before_samples,
            sample_bytes,
            "typed component DWT conversion baseline",
        )?,
    )?;
    let dwt_bytes = forward_dwt53_output_retained_bytes(&dwt)?;
    session.checked_phase(
        checked_add_bytes(
            retained_before_samples,
            checked_add_bytes(sample_bytes, dwt_bytes, "typed component DWT output")?,
            "typed component DWT output",
        )?,
        "typed component DWT output",
    )?;
    drop(samples);
    Ok(dwt)
}

fn encode_prepared_components(
    planes: &[EncodeTypedComponentPlane<'_>],
    width: u32,
    height: u32,
    options: &EncodeOptions,
    plan: &TypedComponentPlan,
    components: Vec<PrecomputedHtj2k53Component>,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<Vec<u8>> {
    let image = PrecomputedHtj2k53Image {
        width,
        height,
        bit_depth: plan.max_bit_depth,
        signed: planes.iter().all(|plane| plane.signed),
        components,
    };
    let retained_image_bytes = checked_add_bytes(
        plan.sample_info_bytes,
        precomputed_53_components_retained_bytes(&image.components)?,
        "typed component retained input",
    )?;
    let retained_owners = (&image, &plan.sample_info);
    let encode_session = session.checked_child_session(
        &retained_owners,
        retained_image_bytes,
        "typed component retained input",
    )?;
    let mut accelerator = CpuOnlyJ2kEncodeStageAccelerator;
    encode_precomputed_53_with_component_sample_info_for_session(
        &image,
        options,
        false,
        options.use_ht_block_coding,
        &plan.sample_info,
        &encode_session,
        &mut accelerator,
    )
}

fn precomputed_53_components_retained_bytes(
    components: &Vec<PrecomputedHtj2k53Component>,
) -> crate::EncodeResult<usize> {
    let mut bytes = checked_element_bytes::<PrecomputedHtj2k53Component>(
        components.capacity(),
        "typed component DWT owners",
    )?;
    for component in components {
        bytes = checked_add_bytes(
            bytes,
            forward_dwt53_output_retained_bytes(&component.dwt)?,
            "typed component retained DWT graph",
        )?;
    }
    Ok(bytes)
}
