// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::{DecodingError, Result};
use crate::{
    J2kDirectGrayscalePlan, J2kReferencedClassicPlan, J2kReferencedHtj2kPlan, J2kWaveletTransform,
};

pub(super) const fn ht_component_count(plan: &J2kReferencedHtj2kPlan) -> usize {
    match plan {
        J2kReferencedHtj2kPlan::Grayscale { .. } => 1,
        J2kReferencedHtj2kPlan::Color { .. } => 3,
        J2kReferencedHtj2kPlan::Rgba { .. } => 4,
    }
}

pub(super) const fn classic_component_count(plan: &J2kReferencedClassicPlan) -> usize {
    match plan {
        J2kReferencedClassicPlan::Grayscale { .. } => 1,
        J2kReferencedClassicPlan::Color { .. } => 3,
        J2kReferencedClassicPlan::Rgba { .. } => 4,
    }
}

pub(super) fn ht_tile_components(
    plan: &J2kReferencedHtj2kPlan,
    tile_index: usize,
) -> Result<&[J2kDirectGrayscalePlan]> {
    let tile = plan
        .tiles()
        .get(tile_index)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    match plan {
        J2kReferencedHtj2kPlan::Grayscale { .. } => tile
            .grayscale_geometry()
            .map(core::slice::from_ref)
            .ok_or_else(|| DecodingError::CodeBlockDecodeFailure.into()),
        J2kReferencedHtj2kPlan::Color { .. } => tile
            .color_geometry()
            .map(|geometry| geometry.component_plans.as_slice())
            .filter(|components| components.len() == 3)
            .ok_or_else(|| DecodingError::CodeBlockDecodeFailure.into()),
        J2kReferencedHtj2kPlan::Rgba { .. } => tile
            .rgba_geometry()
            .map(|geometry| geometry.component_plans.as_slice())
            .filter(|components| components.len() == 4)
            .ok_or_else(|| DecodingError::CodeBlockDecodeFailure.into()),
    }
}

pub(super) fn classic_tile_components(
    plan: &J2kReferencedClassicPlan,
    tile_index: usize,
) -> Result<&[J2kDirectGrayscalePlan]> {
    let tile = plan
        .tiles()
        .get(tile_index)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    match plan {
        J2kReferencedClassicPlan::Grayscale { .. } => tile
            .grayscale_geometry()
            .map(core::slice::from_ref)
            .ok_or_else(|| DecodingError::CodeBlockDecodeFailure.into()),
        J2kReferencedClassicPlan::Color { .. } => tile
            .color_geometry()
            .map(|geometry| geometry.component_plans.as_slice())
            .filter(|components| components.len() == 3)
            .ok_or_else(|| DecodingError::CodeBlockDecodeFailure.into()),
        J2kReferencedClassicPlan::Rgba { .. } => tile
            .rgba_geometry()
            .map(|geometry| geometry.component_plans.as_slice())
            .filter(|components| components.len() == 4)
            .ok_or_else(|| DecodingError::CodeBlockDecodeFailure.into()),
    }
}

pub(super) fn ht_tile_color_transform(
    plan: &J2kReferencedHtj2kPlan,
    tile_index: usize,
) -> Result<Option<([u8; 3], bool, J2kWaveletTransform)>> {
    let tile = plan
        .tiles()
        .get(tile_index)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    match plan {
        J2kReferencedHtj2kPlan::Grayscale { .. } if tile.grayscale_geometry().is_some() => Ok(None),
        J2kReferencedHtj2kPlan::Color { .. } => tile
            .color_geometry()
            .map(|geometry| Some((geometry.bit_depths, geometry.mct, geometry.transform)))
            .ok_or_else(|| DecodingError::CodeBlockDecodeFailure.into()),
        J2kReferencedHtj2kPlan::Rgba { .. } => tile
            .rgba_geometry()
            .map(|geometry| {
                Some((
                    [
                        geometry.bit_depths[0],
                        geometry.bit_depths[1],
                        geometry.bit_depths[2],
                    ],
                    geometry.mct,
                    geometry.transform,
                ))
            })
            .ok_or_else(|| DecodingError::CodeBlockDecodeFailure.into()),
        J2kReferencedHtj2kPlan::Grayscale { .. } => {
            Err(DecodingError::CodeBlockDecodeFailure.into())
        }
    }
}

pub(super) fn classic_tile_color_transform(
    plan: &J2kReferencedClassicPlan,
    tile_index: usize,
) -> Result<Option<([u8; 3], bool, J2kWaveletTransform)>> {
    let tile = plan
        .tiles()
        .get(tile_index)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    match plan {
        J2kReferencedClassicPlan::Grayscale { .. } if tile.grayscale_geometry().is_some() => {
            Ok(None)
        }
        J2kReferencedClassicPlan::Color { .. } => tile
            .color_geometry()
            .map(|geometry| Some((geometry.bit_depths, geometry.mct, geometry.transform)))
            .ok_or_else(|| DecodingError::CodeBlockDecodeFailure.into()),
        J2kReferencedClassicPlan::Rgba { .. } => tile
            .rgba_geometry()
            .map(|geometry| {
                Some((
                    [
                        geometry.bit_depths[0],
                        geometry.bit_depths[1],
                        geometry.bit_depths[2],
                    ],
                    geometry.mct,
                    geometry.transform,
                ))
            })
            .ok_or_else(|| DecodingError::CodeBlockDecodeFailure.into()),
        J2kReferencedClassicPlan::Grayscale { .. } => {
            Err(DecodingError::CodeBlockDecodeFailure.into())
        }
    }
}
