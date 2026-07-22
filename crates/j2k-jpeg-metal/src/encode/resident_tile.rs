// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_metal_support::ResidentMetalImage;

use super::JpegBaselineMetalEncodeTile;

impl<'a> JpegBaselineMetalEncodeTile<'a> {
    /// Describe a validated resident image for baseline JPEG encoding.
    ///
    /// The image allocation remains owned and logically immutable for the
    /// lifetime of this tile. The encode operation validates device identity
    /// before submitting any Metal work.
    #[must_use]
    pub fn from_resident(image: &'a ResidentMetalImage, output_dimensions: (u32, u32)) -> Self {
        let layout = image.layout();
        Self {
            // SAFETY: the resident abstraction exposes the handle only for
            // audited read-only backend binding, and this tile retains the
            // resident owner for its entire lifetime.
            buffer: unsafe { image.raw_buffer() },
            byte_offset: layout.byte_offset(),
            width: layout.dimensions().0,
            height: layout.dimensions().1,
            pitch_bytes: layout.pitch_bytes(),
            output_width: output_dimensions.0,
            output_height: output_dimensions.1,
            format: layout.pixel_format(),
        }
    }
}
