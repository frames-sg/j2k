// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sampling-shape validation for extended and progressive precision paths.

use super::super::{
    Info, JpegError, LosslessColorSampling, PreparedDecodePlan, PreparedProgressivePlan, SofKind,
};
use crate::info::SamplingFactors;

/// Three-component layouts the full-size direct writers handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Extended12ColorSampling {
    S444,
    S422,
    S420,
}

/// How a 12-bit three- or four-component layout is rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Extended12Layout {
    /// 4:4:4, 4:2:2 or 4:2:0: the direct writers handle full-size decodes.
    Direct(Extended12ColorSampling),
    /// Any other layout whose factors divide the maximum factors, such as
    /// 4:4:0, 4:1:1, 4:1:0 or 1x4. Only the component-plane pipeline renders
    /// it, at every scale.
    Planes,
}

/// How lossless color reconstructs to pixels: 4:4:4, or any other layout
/// whose factors divide the maximum factors (replicated on output, as
/// libjpeg-turbo's lossless upsampler does). Any precision and size.
pub(in crate::decoder) fn lossless_color_sampling(info: &Info) -> Option<LosslessColorSampling> {
    if info.sampling.len() != 3 || !info.sampling.has_integral_ratios() {
        return None;
    }
    if info
        .sampling
        .components()
        .iter()
        .all(|&factors| factors == (1, 1))
    {
        Some(LosslessColorSampling::S444)
    } else {
        Some(LosslessColorSampling::Sampled)
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

pub(super) fn extended12_color_layout(
    plan: &PreparedDecodePlan,
    sof: SofKind,
) -> Result<Extended12Layout, JpegError> {
    if plan.components.len() != 3 {
        return Err(JpegError::NotImplemented { sof });
    }
    let components = color_component_sampling_from_sequential(plan, sof)?;
    color_layout_from_components(plan.sampling, components, sof)
}

pub(super) fn extended12_four_component_layout(
    plan: &PreparedDecodePlan,
    sof: SofKind,
) -> Result<Extended12Layout, JpegError> {
    if plan.components.len() != 4 {
        return Err(JpegError::NotImplemented { sof });
    }
    let components = four_component_sampling_from_sequential(plan, sof)?;
    four_component_layout_from_components(plan.sampling, components, sof)
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

pub(super) fn progressive_color_layout(
    plan: &PreparedProgressivePlan,
    sof: SofKind,
) -> Result<Extended12Layout, JpegError> {
    if plan.components.len() != 3 {
        return Err(JpegError::NotImplemented { sof });
    }
    let components = color_component_sampling_from_progressive(plan, sof)?;
    color_layout_from_components(plan.sampling, components, sof)
}

pub(super) fn progressive_four_component_layout(
    plan: &PreparedProgressivePlan,
    sof: SofKind,
) -> Result<Extended12Layout, JpegError> {
    if plan.components.len() != 4 {
        return Err(JpegError::NotImplemented { sof });
    }
    let components = four_component_sampling_from_progressive(plan, sof)?;
    four_component_layout_from_components(plan.sampling, components, sof)
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

fn color_layout_from_components(
    sampling: SamplingFactors,
    components: [(u8, u8); 3],
    sof: SofKind,
) -> Result<Extended12Layout, JpegError> {
    if let Ok(direct) =
        color_sampling_from_components(sampling.max_h, sampling.max_v, components, sof)
    {
        return Ok(Extended12Layout::Direct(direct));
    }
    if sampling.has_integral_ratios() {
        Ok(Extended12Layout::Planes)
    } else {
        Err(JpegError::NotImplemented { sof })
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

/// 4:4:4, 4:2:2 and 4:2:0 (subsampled first component) keep the direct
/// writers; any other whole-ratio layout, such as YCCK with Y and K at 2x2,
/// renders through component planes.
fn four_component_layout_from_components(
    sampling: SamplingFactors,
    components: [(u8, u8); 4],
    sof: SofKind,
) -> Result<Extended12Layout, JpegError> {
    if let Ok(direct) =
        four_component_sampling_from_components(sampling.max_h, sampling.max_v, components, sof)
    {
        return Ok(Extended12Layout::Direct(direct));
    }
    if sampling.has_integral_ratios() {
        Ok(Extended12Layout::Planes)
    } else {
        Err(JpegError::NotImplemented { sof })
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
