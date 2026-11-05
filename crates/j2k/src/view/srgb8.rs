// SPDX-License-Identifier: MIT OR Apache-2.0

use super::J2kDecoder;
use crate::{
    decode::{decode_image_srgb8, RoutedSrgb8},
    J2kError, J2kSrgb8Image,
};

impl J2kDecoder<'_> {
    /// Decode the full image and normalize its colour data to 8-bit sRGB.
    ///
    /// JP2 palette mapping, component sampling, channel definitions, and
    /// enumerated sRGB-YCC conversion are applied before this output is
    /// produced. Restricted ICC input profiles are converted with the pinned
    /// colour-management implementation.
    ///
    /// # Memory
    /// A Gray, RGB or RGBA image without an ICC profile decodes on the route
    /// [`Self::decode_into`] takes for it, straight into the returned buffer:
    /// an image whose component planes would pass the decode budget streams
    /// its rows instead of holding the planes. An image converted from an ICC
    /// profile, or expanded from grey with alpha, decodes its whole component
    /// planes before it is converted.
    ///
    /// # Errors
    /// Returns [`J2kError`] when decoding fails, a primary ICC profile is
    /// malformed, the colour space is unsupported, or the bounded output
    /// allocation cannot be made. A routed decode that fails partway may
    /// have written some rows of an output it then drops.
    pub fn decode_srgb8(&mut self) -> Result<J2kSrgb8Image, J2kError> {
        self.ensure_image()?;
        let Some(image) = self.image.as_ref() else {
            return Err(J2kError::internal_backend("internal image cache missing"));
        };
        let Some(routed) = RoutedSrgb8::plan(image)? else {
            return decode_image_srgb8(image, &mut self.native_context);
        };
        let mut data = routed.allocate()?;
        let outcome = self.decode_into_cached(&mut data, routed.stride(), routed.format())?;
        routed.finish(data, outcome.warnings)
    }
}
