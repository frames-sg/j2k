// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::{bail, DecodingError, Result};
use crate::J2kReferencedHtj2kPlan;

use super::super::{DirectComponentPlane, J2kDirectCpuScratch};

/// One decoded component plane borrowed from retained direct CPU scratch.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct J2kDirectDecodedPlane<'a> {
    pub(in crate::direct_cpu) dimensions: (u32, u32),
    pub(in crate::direct_cpu) bit_depth: u8,
    pub(in crate::direct_cpu) samples: &'a [f32],
}

impl<'a> J2kDirectDecodedPlane<'a> {
    /// Full reduced-resolution plane dimensions.
    #[must_use]
    pub const fn dimensions(self) -> (u32, u32) {
        self.dimensions
    }

    /// Declared component precision.
    #[must_use]
    pub const fn bit_depth(self) -> u8 {
        self.bit_depth
    }

    /// Row-major reconstructed samples.
    #[must_use]
    pub const fn samples(self) -> &'a [f32] {
        self.samples
    }
}

/// Gray, RGB, or RGBA decoded planes borrowed from retained direct CPU scratch.
#[doc(hidden)]
#[derive(Debug)]
pub struct J2kDirectDecodedComponents<'a> {
    pub(in crate::direct_cpu) dimensions: (u32, u32),
    pub(in crate::direct_cpu) planes: [Option<J2kDirectDecodedPlane<'a>>; 4],
    pub(in crate::direct_cpu) component_count: usize,
}

impl J2kDirectDecodedComponents<'_> {
    /// Full reduced-resolution plane dimensions.
    #[must_use]
    pub const fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    /// Number of decoded planes: one for Gray, three for RGB, or four for RGBA.
    #[must_use]
    pub const fn component_count(&self) -> usize {
        self.component_count
    }

    /// Return one decoded component plane.
    #[must_use]
    pub fn plane(&self, index: usize) -> Option<J2kDirectDecodedPlane<'_>> {
        self.planes.get(index).copied().flatten()
    }
}

pub(in crate::direct_cpu) fn decoded_components<'scratch>(
    plan: &J2kReferencedHtj2kPlan,
    scratch: &'scratch J2kDirectCpuScratch,
) -> Result<J2kDirectDecodedComponents<'scratch>> {
    let dimensions = (plan.output_rect().width(), plan.output_rect().height());
    let first = plan
        .tiles()
        .first()
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    match plan {
        J2kReferencedHtj2kPlan::Grayscale { .. } => {
            let geometry = first
                .grayscale_geometry()
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            for tile in plan.tiles() {
                let current = tile
                    .grayscale_geometry()
                    .ok_or(DecodingError::CodeBlockDecodeFailure)?;
                if current.dimensions != dimensions || current.bit_depth != geometry.bit_depth {
                    bail!(DecodingError::CodeBlockDecodeFailure);
                }
            }
            let plane = decoded_plane(
                scratch
                    .component_planes
                    .first()
                    .ok_or(DecodingError::CodeBlockDecodeFailure)?,
                dimensions,
                geometry.bit_depth,
            )?;
            Ok(J2kDirectDecodedComponents {
                dimensions,
                planes: [Some(plane), None, None, None],
                component_count: 1,
            })
        }
        J2kReferencedHtj2kPlan::Color { .. } => {
            let geometry = first
                .color_geometry()
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            for tile in plan.tiles() {
                let current = tile
                    .color_geometry()
                    .ok_or(DecodingError::CodeBlockDecodeFailure)?;
                if current.dimensions != dimensions
                    || current.bit_depths != geometry.bit_depths
                    || current.component_plans.len() != 3
                {
                    bail!(DecodingError::CodeBlockDecodeFailure);
                }
            }
            decoded_color_components(dimensions, &geometry.bit_depths, 3, scratch)
        }
        J2kReferencedHtj2kPlan::Rgba { .. } => {
            let geometry = first
                .rgba_geometry()
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            for tile in plan.tiles() {
                let current = tile
                    .rgba_geometry()
                    .ok_or(DecodingError::CodeBlockDecodeFailure)?;
                if current.dimensions != dimensions
                    || current.bit_depths != geometry.bit_depths
                    || current.component_plans.len() != 4
                {
                    bail!(DecodingError::CodeBlockDecodeFailure);
                }
            }
            decoded_color_components(dimensions, &geometry.bit_depths, 4, scratch)
        }
    }
}

pub(in crate::direct_cpu) fn decoded_color_components<'scratch>(
    dimensions: (u32, u32),
    bit_depths: &[u8],
    component_count: usize,
    scratch: &'scratch J2kDirectCpuScratch,
) -> Result<J2kDirectDecodedComponents<'scratch>> {
    if !matches!(component_count, 3 | 4)
        || bit_depths.len() != component_count
        || scratch.component_planes.len() < component_count
    {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    let mut planes = [None, None, None, None];
    for component in 0..component_count {
        planes[component] = Some(decoded_plane(
            &scratch.component_planes[component],
            dimensions,
            bit_depths[component],
        )?);
    }
    Ok(J2kDirectDecodedComponents {
        dimensions,
        planes,
        component_count,
    })
}

pub(in crate::direct_cpu) fn decoded_plane(
    plane: &DirectComponentPlane,
    dimensions: (u32, u32),
    bit_depth: u8,
) -> Result<J2kDirectDecodedPlane<'_>> {
    if (plane.width, plane.height) != dimensions
        || plane.samples.len() != super::super::checked_area(dimensions.0, dimensions.1)?
    {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    Ok(J2kDirectDecodedPlane {
        dimensions,
        bit_depth,
        samples: &plane.samples,
    })
}
