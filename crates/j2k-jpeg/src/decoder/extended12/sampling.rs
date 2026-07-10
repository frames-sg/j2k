// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sampling-shape validation for extended and progressive precision paths.

use super::super::{
    Info, JpegError, LosslessColorSampling, PreparedDecodePlan, PreparedProgressivePlan, SofKind,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Extended12ColorSampling {
    S444,
    S422,
    S420,
}

pub(in crate::decoder) fn lossless_color_sampling(info: &Info) -> Option<LosslessColorSampling> {
    if info.sampling.len() != 3 {
        return None;
    }
    match (
        info.sampling.max_h,
        info.sampling.max_v,
        info.sampling.components(),
    ) {
        (1, 1, &[(1, 1), (1, 1), (1, 1)]) => Some(LosslessColorSampling::S444),
        (2, 1, &[(2, 1), (1, 1), (1, 1)])
            if matches!(info.bit_depth, 8 | 16) && info.dimensions.0.is_multiple_of(2) =>
        {
            Some(LosslessColorSampling::S422)
        }
        (2, 2, &[(2, 2), (1, 1), (1, 1)])
            if matches!(info.bit_depth, 8 | 16)
                && info.dimensions.0.is_multiple_of(2)
                && info.dimensions.1.is_multiple_of(2) =>
        {
            Some(LosslessColorSampling::S420)
        }
        _ => None,
    }
}

pub(super) fn validate_extended12_color444_plan(
    plan: &PreparedDecodePlan,
    sof: SofKind,
) -> Result<(), JpegError> {
    if plan.components.len() != 3 || plan.sampling.max_h != 1 || plan.sampling.max_v != 1 {
        return Err(JpegError::NotImplemented { sof });
    }
    let mut seen = [false; 3];
    for component in &plan.components {
        if component.h != 1 || component.v != 1 || component.output_index > 2 {
            return Err(JpegError::NotImplemented { sof });
        }
        if seen[component.output_index] {
            return Err(JpegError::NotImplemented { sof });
        }
        seen[component.output_index] = true;
    }
    if seen.iter().any(|&present| !present) {
        return Err(JpegError::NotImplemented { sof });
    }
    Ok(())
}

pub(super) fn validate_extended12_four_component444_plan(
    plan: &PreparedDecodePlan,
    sof: SofKind,
) -> Result<(), JpegError> {
    if extended12_four_component_sampling(plan, sof)? != Extended12ColorSampling::S444 {
        return Err(JpegError::NotImplemented { sof });
    }
    Ok(())
}

pub(super) fn extended12_color_sampling(
    plan: &PreparedDecodePlan,
    sof: SofKind,
) -> Result<Extended12ColorSampling, JpegError> {
    if plan.components.len() != 3 {
        return Err(JpegError::NotImplemented { sof });
    }
    let components = color_component_sampling_from_sequential(plan, sof)?;
    color_sampling_from_components(plan.sampling.max_h, plan.sampling.max_v, components, sof)
}

pub(super) fn extended12_four_component_sampling(
    plan: &PreparedDecodePlan,
    sof: SofKind,
) -> Result<Extended12ColorSampling, JpegError> {
    if plan.components.len() != 4 {
        return Err(JpegError::NotImplemented { sof });
    }
    let components = four_component_sampling_from_sequential(plan, sof)?;
    four_component_sampling_from_components(
        plan.sampling.max_h,
        plan.sampling.max_v,
        components,
        sof,
    )
}

pub(super) fn color_component_sampling_from_sequential(
    plan: &PreparedDecodePlan,
    sof: SofKind,
) -> Result<[(u8, u8); 3], JpegError> {
    let mut components = [(0u8, 0u8); 3];
    let mut seen = [false; 3];
    for component in &plan.components {
        if component.output_index > 2 || seen[component.output_index] {
            return Err(JpegError::NotImplemented { sof });
        }
        seen[component.output_index] = true;
        components[component.output_index] = (component.h, component.v);
    }
    if seen.iter().any(|&present| !present) {
        return Err(JpegError::NotImplemented { sof });
    }
    Ok(components)
}

pub(super) fn four_component_sampling_from_sequential(
    plan: &PreparedDecodePlan,
    sof: SofKind,
) -> Result<[(u8, u8); 4], JpegError> {
    let mut components = [(0u8, 0u8); 4];
    let mut seen = [false; 4];
    for component in &plan.components {
        if component.output_index > 3 || seen[component.output_index] {
            return Err(JpegError::NotImplemented { sof });
        }
        seen[component.output_index] = true;
        components[component.output_index] = (component.h, component.v);
    }
    if seen.iter().any(|&present| !present) {
        return Err(JpegError::NotImplemented { sof });
    }
    Ok(components)
}

pub(super) fn progressive_color_sampling(
    plan: &PreparedProgressivePlan,
    sof: SofKind,
) -> Result<Extended12ColorSampling, JpegError> {
    if plan.components.len() != 3 {
        return Err(JpegError::NotImplemented { sof });
    }
    let components = color_component_sampling_from_progressive(plan, sof)?;
    color_sampling_from_components(plan.sampling.max_h, plan.sampling.max_v, components, sof)
}

pub(super) fn progressive_four_component_sampling(
    plan: &PreparedProgressivePlan,
    sof: SofKind,
) -> Result<Extended12ColorSampling, JpegError> {
    if plan.components.len() != 4 {
        return Err(JpegError::NotImplemented { sof });
    }
    let components = four_component_sampling_from_progressive(plan, sof)?;
    four_component_sampling_from_components(
        plan.sampling.max_h,
        plan.sampling.max_v,
        components,
        sof,
    )
}

pub(super) fn color_component_sampling_from_progressive(
    plan: &PreparedProgressivePlan,
    sof: SofKind,
) -> Result<[(u8, u8); 3], JpegError> {
    let mut components = [(0u8, 0u8); 3];
    let mut seen = [false; 3];
    for component in &plan.components {
        if component.output_index > 2 || seen[component.output_index] {
            return Err(JpegError::NotImplemented { sof });
        }
        seen[component.output_index] = true;
        components[component.output_index] = (component.h, component.v);
    }
    if seen.iter().any(|&present| !present) {
        return Err(JpegError::NotImplemented { sof });
    }
    Ok(components)
}

pub(super) fn four_component_sampling_from_progressive(
    plan: &PreparedProgressivePlan,
    sof: SofKind,
) -> Result<[(u8, u8); 4], JpegError> {
    let mut components = [(0u8, 0u8); 4];
    let mut seen = [false; 4];
    for component in &plan.components {
        if component.output_index > 3 || seen[component.output_index] {
            return Err(JpegError::NotImplemented { sof });
        }
        seen[component.output_index] = true;
        components[component.output_index] = (component.h, component.v);
    }
    if seen.iter().any(|&present| !present) {
        return Err(JpegError::NotImplemented { sof });
    }
    Ok(components)
}

pub(super) fn color_sampling_from_components(
    max_h: u8,
    max_v: u8,
    components: [(u8, u8); 3],
    sof: SofKind,
) -> Result<Extended12ColorSampling, JpegError> {
    match (max_h, max_v, components) {
        (1, 1, [(1, 1), (1, 1), (1, 1)]) => Ok(Extended12ColorSampling::S444),
        (2, 1, [(2, 1), (1, 1), (1, 1)]) => Ok(Extended12ColorSampling::S422),
        (2, 2, [(2, 2), (1, 1), (1, 1)]) => Ok(Extended12ColorSampling::S420),
        _ => Err(JpegError::NotImplemented { sof }),
    }
}

pub(super) fn four_component_sampling_from_components(
    max_h: u8,
    max_v: u8,
    components: [(u8, u8); 4],
    sof: SofKind,
) -> Result<Extended12ColorSampling, JpegError> {
    match (max_h, max_v, components) {
        (1, 1, [(1, 1), (1, 1), (1, 1), (1, 1)]) => Ok(Extended12ColorSampling::S444),
        (2, 1, [(2, 1), (1, 1), (1, 1), (1, 1)]) => Ok(Extended12ColorSampling::S422),
        (2, 2, [(2, 2), (1, 1), (1, 1), (1, 1)]) => Ok(Extended12ColorSampling::S420),
        _ => Err(JpegError::NotImplemented { sof }),
    }
}

pub(super) fn progressive_color_component_indices(
    plan: &PreparedProgressivePlan,
) -> Result<[usize; 3], JpegError> {
    let mut indices = [usize::MAX; 3];
    for (component_index, component) in plan.components.iter().enumerate() {
        if component.output_index < 3 {
            if indices[component.output_index] != usize::MAX {
                return Err(JpegError::NotImplemented {
                    sof: SofKind::Progressive12,
                });
            }
            indices[component.output_index] = component_index;
        }
    }
    if indices.contains(&usize::MAX) {
        return Err(JpegError::NotImplemented {
            sof: SofKind::Progressive12,
        });
    }
    Ok(indices)
}
