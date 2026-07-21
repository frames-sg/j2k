// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    bitplane, checked_area, component_band_count, component_plane_len, ht_block_decode,
    observe_max_dimensions, DecodingError, J2kDirectGrayscalePlan, J2kDirectGrayscaleStep,
    J2kReferencedClassicPlan, J2kReferencedHtj2kPlan, Result, ValidationError,
};

#[derive(Clone, Copy)]
pub(super) enum ReferencedPlanView<'a> {
    Htj2k(&'a J2kReferencedHtj2kPlan),
    Classic(&'a J2kReferencedClassicPlan),
}

impl<'a> ReferencedPlanView<'a> {
    pub(super) const fn component_count(self) -> usize {
        match self {
            Self::Htj2k(J2kReferencedHtj2kPlan::Grayscale { .. })
            | Self::Classic(J2kReferencedClassicPlan::Grayscale { .. }) => 1,
            Self::Htj2k(J2kReferencedHtj2kPlan::Color { .. })
            | Self::Classic(J2kReferencedClassicPlan::Color { .. }) => 3,
            Self::Htj2k(J2kReferencedHtj2kPlan::Rgba { .. })
            | Self::Classic(J2kReferencedClassicPlan::Rgba { .. }) => 4,
        }
    }

    pub(super) fn tile_count(self) -> usize {
        match self {
            Self::Htj2k(plan) => plan.tiles().len(),
            Self::Classic(plan) => plan.tiles().len(),
        }
    }

    pub(super) fn output_dimensions(self) -> (u32, u32) {
        let rect = match self {
            Self::Htj2k(plan) => plan.output_rect(),
            Self::Classic(plan) => plan.output_rect(),
        };
        (rect.width(), rect.height())
    }

    pub(super) fn component_plan(
        self,
        tile_index: usize,
        component_index: usize,
    ) -> Result<&'a J2kDirectGrayscalePlan> {
        match self {
            Self::Htj2k(plan) => match plan {
                J2kReferencedHtj2kPlan::Grayscale { .. } if component_index == 0 => plan
                    .tiles()
                    .get(tile_index)
                    .and_then(crate::J2kReferencedTilePlan::grayscale_geometry),
                J2kReferencedHtj2kPlan::Color { .. } => plan
                    .tiles()
                    .get(tile_index)
                    .and_then(crate::J2kReferencedTilePlan::color_geometry)
                    .and_then(|geometry| geometry.component_plans.get(component_index)),
                J2kReferencedHtj2kPlan::Rgba { .. } => plan
                    .tiles()
                    .get(tile_index)
                    .and_then(crate::J2kReferencedTilePlan::rgba_geometry)
                    .and_then(|geometry| geometry.component_plans.get(component_index)),
                J2kReferencedHtj2kPlan::Grayscale { .. } => None,
            },
            Self::Classic(plan) => match plan {
                J2kReferencedClassicPlan::Grayscale { .. } if component_index == 0 => plan
                    .tiles()
                    .get(tile_index)
                    .and_then(crate::J2kReferencedTilePlan::grayscale_geometry),
                J2kReferencedClassicPlan::Color { .. } => plan
                    .tiles()
                    .get(tile_index)
                    .and_then(crate::J2kReferencedTilePlan::color_geometry)
                    .and_then(|geometry| geometry.component_plans.get(component_index)),
                J2kReferencedClassicPlan::Rgba { .. } => plan
                    .tiles()
                    .get(tile_index)
                    .and_then(crate::J2kReferencedTilePlan::rgba_geometry)
                    .and_then(|geometry| geometry.component_plans.get(component_index)),
                J2kReferencedClassicPlan::Grayscale { .. } => None,
            },
        }
        .ok_or_else(|| DecodingError::CodeBlockDecodeFailure.into())
    }
}

pub(super) fn validate_referenced_shape(plan: ReferencedPlanView<'_>) -> Result<()> {
    if plan.tile_count() == 0 || plan.output_dimensions().0 == 0 || plan.output_dimensions().1 == 0
    {
        return Err(DecodingError::CodeBlockDecodeFailure.into());
    }
    for tile_index in 0..plan.tile_count() {
        for component_index in 0..plan.component_count() {
            let component = plan.component_plan(tile_index, component_index)?;
            if component.dimensions != plan.output_dimensions() {
                return Err(DecodingError::CodeBlockDecodeFailure.into());
            }
        }
    }
    Ok(())
}

pub(super) fn referenced_component_band_count(
    plan: ReferencedPlanView<'_>,
    component_index: usize,
) -> Result<usize> {
    let mut maximum = 0usize;
    for tile_index in 0..plan.tile_count() {
        maximum = maximum.max(component_band_count(
            plan.component_plan(tile_index, component_index)?,
        )?);
    }
    Ok(maximum)
}

pub(super) fn referenced_band_target(
    plan: ReferencedPlanView<'_>,
    component_index: usize,
    band_index: usize,
) -> Result<usize> {
    let mut maximum = None;
    for tile_index in 0..plan.tile_count() {
        let component = plan.component_plan(tile_index, component_index)?;
        for_each_staged_band_target(component, |current_index, target_len| {
            if current_index == band_index {
                maximum =
                    Some(maximum.map_or(target_len, |current: usize| current.max(target_len)));
            }
            Ok(())
        })?;
    }
    maximum.ok_or_else(|| DecodingError::CodeBlockDecodeFailure.into())
}

pub(super) fn referenced_component_plane_len(
    plan: ReferencedPlanView<'_>,
    component_index: usize,
) -> Result<usize> {
    let expected = checked_area(plan.output_dimensions().0, plan.output_dimensions().1)?;
    for tile_index in 0..plan.tile_count() {
        let current = component_plane_len(plan.component_plan(tile_index, component_index)?)?;
        if current != expected {
            return Err(DecodingError::CodeBlockDecodeFailure.into());
        }
    }
    Ok(expected)
}

fn for_each_staged_band_target(
    plan: &J2kDirectGrayscalePlan,
    mut visit: impl FnMut(usize, usize) -> Result<()>,
) -> Result<()> {
    let mut band_index = 0usize;
    for entropy_phase in [true, false] {
        for step in &plan.steps {
            let target_len = match (entropy_phase, step) {
                (true, J2kDirectGrayscaleStep::ClassicSubBand(sub_band)) => {
                    Some(checked_area(sub_band.width, sub_band.height)?)
                }
                (true, J2kDirectGrayscaleStep::HtSubBand(sub_band)) => {
                    Some(checked_area(sub_band.width, sub_band.height)?)
                }
                (false, J2kDirectGrayscaleStep::Idwt(step)) => {
                    Some(checked_area(step.rect.width(), step.rect.height())?)
                }
                _ => None,
            };
            if let Some(target_len) = target_len {
                visit(band_index, target_len)?;
                band_index = band_index
                    .checked_add(1)
                    .ok_or(ValidationError::ImageTooLarge)?;
            }
        }
    }
    Ok(())
}

pub(super) fn referenced_temporary_workspace_bytes(plan: ReferencedPlanView<'_>) -> Result<usize> {
    let classic = max_referenced_dimensions(plan, false).map_or(Ok(0), |(width, height)| {
        bitplane::classic_decode_workspace_bytes(width, height)
    })?;
    let ht = max_referenced_dimensions(plan, true).map_or(Ok(0), |(width, height)| {
        ht_block_decode::ht_decode_workspace_bytes(width, height)
    })?;
    Ok(classic.max(ht))
}

pub(super) fn max_referenced_payload_bytes(plan: &J2kReferencedHtj2kPlan) -> Result<usize> {
    plan.payloads().iter().try_fold(0usize, |maximum, payload| {
        payload
            .cleanup
            .length
            .checked_add(payload.refinement.map_or(0, |range| range.length))
            .map(|length| maximum.max(length))
            .ok_or(ValidationError::ImageTooLarge.into())
    })
}

pub(super) fn max_referenced_classic_payload_bytes(plan: &J2kReferencedClassicPlan) -> usize {
    plan.payloads()
        .iter()
        .map(|payload| payload.combined_length)
        .max()
        .unwrap_or(0)
}

pub(in super::super::super) fn max_referenced_classic_dimensions(
    plan: &J2kReferencedClassicPlan,
) -> Option<(u32, u32)> {
    max_referenced_dimensions(ReferencedPlanView::Classic(plan), false)
}

pub(in super::super::super) fn max_referenced_ht_dimensions(
    plan: &J2kReferencedHtj2kPlan,
) -> Option<(u32, u32)> {
    max_referenced_dimensions(ReferencedPlanView::Htj2k(plan), true)
}

fn max_referenced_dimensions(plan: ReferencedPlanView<'_>, ht: bool) -> Option<(u32, u32)> {
    let mut dimensions = None;
    for tile_index in 0..plan.tile_count() {
        for component_index in 0..plan.component_count() {
            let component = plan.component_plan(tile_index, component_index).ok()?;
            for step in &component.steps {
                match (ht, step) {
                    (true, J2kDirectGrayscaleStep::HtSubBand(sub_band)) => {
                        for job in &sub_band.jobs {
                            observe_max_dimensions(&mut dimensions, job.width, job.height);
                        }
                    }
                    (false, J2kDirectGrayscaleStep::ClassicSubBand(sub_band)) => {
                        for job in &sub_band.jobs {
                            observe_max_dimensions(&mut dimensions, job.width, job.height);
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    dimensions
}
