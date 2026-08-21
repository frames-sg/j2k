// SPDX-License-Identifier: MIT OR Apache-2.0

//! JPEG 2000 color metadata, conversion, packing, and decoded output values.

mod allocation;
#[cfg(test)]
mod boundary_tests;
mod cielab;
mod icc;
mod metadata;
mod output_planes;
mod packing;
mod palette;
mod sycc;
mod types;

#[cfg(test)]
pub(crate) use cielab::cielab_to_rgb;
pub(crate) use metadata::{resolve_alpha_and_color_space, validate_and_reorder_channels};
pub(crate) use output_planes::native_component_plane_dimensions;
pub use output_planes::{
    ComponentPlane, ComponentPlaneParts, DecodedComponents, DecodedComponentsParts,
    DecodedNativeComponents, DecodedNativeComponentsParts, NativeComponentPlane,
    NativeComponentPlaneParts,
};
pub(crate) use packing::{
    interleave_and_convert, interleave_and_convert_region, validate_interleaved_output_buffer,
};
pub(crate) use palette::resolve_palette_indices;
pub use types::{Bitmap, ColorSpace, RawBitmap};

use crate::jp2::colr::EnumeratedColorspace;
use crate::jp2::DecodedImage;
use crate::math::{dispatch, Level};
use crate::Result;

pub(crate) fn convert_color_space(image: &mut DecodedImage<'_, '_>, bit_depth: u8) -> Result<()> {
    if let Some(crate::jp2::colr::ColorSpace::Enumerated(enumerated)) = &image
        .boxes
        .primary_color_specification()
        .map(|information| &information.color_space)
    {
        match enumerated {
            EnumeratedColorspace::Sycc => {
                dispatch!(Level::new(), simd => {
                    sycc::sycc_to_rgb(simd, image.decoded_components, bit_depth)
                })?;
            }
            EnumeratedColorspace::CieLab(cielab) => {
                dispatch!(Level::new(), simd => {
                    cielab::cielab_to_rgb(simd, image.decoded_components, bit_depth, cielab)
                })?;
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
use crate::j2c::ComponentData;
#[cfg(test)]
use crate::jp2::ImageBoxes;
#[cfg(test)]
use alloc::vec::Vec;
