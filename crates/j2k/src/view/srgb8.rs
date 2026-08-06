// SPDX-License-Identifier: MIT OR Apache-2.0

use super::J2kDecoder;
use crate::{decode::decode_image_srgb8, J2kError, J2kSrgb8Image};

impl J2kDecoder<'_> {
    /// Decode the full image and normalize its colour data to 8-bit sRGB.
    ///
    /// JP2 palette mapping, component sampling, channel definitions, and
    /// enumerated sRGB-YCC conversion are applied before this output is
    /// produced. Restricted ICC input profiles are converted with the pinned
    /// colour-management implementation.
    ///
    /// # Errors
    /// Returns [`J2kError`] when decoding fails, a primary ICC profile is
    /// malformed, the colour space is unsupported, or the bounded output
    /// allocation cannot be made.
    pub fn decode_srgb8(&mut self) -> Result<J2kSrgb8Image, J2kError> {
        self.ensure_image()?;
        let (Some(image), native_context) = (self.image.as_ref(), &mut self.native_context) else {
            return Err(J2kError::internal_backend("internal image cache missing"));
        };
        decode_image_srgb8(image, native_context)
    }
}
