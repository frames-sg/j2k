// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::J2kError;
use j2k_core::{Colorspace, Info};

pub(crate) use j2k_native::{ColorSpace, DecodeSettings, DecodedComponents, Image, RawBitmap};

pub(crate) fn image(bytes: &[u8], settings: DecodeSettings) -> Result<Image<'_>, J2kError> {
    Image::new(bytes, &settings).map_err(J2kError::from_native_decode_error)
}

pub(crate) fn inspect_info(bytes: &[u8]) -> Result<Info, J2kError> {
    let image = image(bytes, DecodeSettings::default())?;
    Ok(inspect_info_from_image(&image))
}

pub(crate) fn inspect_info_from_image(image: &Image<'_>) -> Info {
    let components = image.color_space().num_channels() + u16::from(image.has_alpha());
    Info {
        dimensions: (image.width(), image.height()),
        components,
        colorspace: map_colorspace(image.color_space()),
        bit_depth: image.original_bit_depth(),
        tile_layout: None,
        coded_unit_layout: None,
        restart_interval: None,
        resolution_levels: 1,
    }
}

pub(crate) fn map_colorspace(color_space: &ColorSpace) -> Colorspace {
    match color_space {
        ColorSpace::Gray => Colorspace::SGray,
        ColorSpace::RGB => Colorspace::Rgb,
        ColorSpace::CMYK => Colorspace::Cmyk,
        ColorSpace::Unknown { .. } | ColorSpace::Icc { .. } => Colorspace::IccTagged,
    }
}
