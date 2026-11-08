// SPDX-License-Identifier: MIT OR Apache-2.0

use super::J2kDecoder;
use crate::{
    decode::{decode_image_region_srgb8, decode_image_srgb8_with_control, RoutedSrgb8},
    J2kDecodeControl, J2kError, J2kSrgb8Image, Rect,
};

impl J2kDecoder<'_> {
    /// Decode the full image and normalize its colour data to 8-bit sRGB.
    ///
    /// JP2 palette mapping, component sampling, channel definitions, and
    /// enumerated sRGB-YCC conversion are applied before this output is
    /// produced. Restricted ICC input profiles are converted with the pinned
    /// colour-management implementation. Alpha is straight: premultiplied
    /// source colour is unassociated before colour conversion, with zero colour
    /// at zero opacity. Ordinary packed and native output retain source association.
    /// Premultiplied colour on one unsigned sampling grid is unassociated in
    /// source code values at the source precision, after palette mapping and
    /// before sYCC conversion (about the chroma midpoint) and quantization.
    /// Premultiplied CIE Lab input is unsupported, and premultiplied input with
    /// subsampled or signed components still requires eight-bit mapped colour
    /// and alpha without native colour conversion.
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
        self.decode_srgb8_with_control(&J2kDecodeControl::default())
    }

    /// Decode the full image to straight-alpha sRGB with operation-specific
    /// limits and cancellation, on the routes [`Self::decode_srgb8`] takes:
    /// a Gray, RGB or RGBA image without an ICC profile streams its rows
    /// into the result when its planes would pass the decode budget. The
    /// result buffer is codec-owned, so it counts toward the working limit.
    ///
    /// # Errors
    /// Returns the same colour/decode errors as [`Self::decode_srgb8`], or a
    /// control limit, cancellation or unsupported controlled-conversion error.
    pub fn decode_srgb8_with_control(
        &mut self,
        control: &J2kDecodeControl,
    ) -> Result<J2kSrgb8Image, J2kError> {
        let result = (|| {
            control.check_cancelled()?;
            self.ensure_image_at_reduction_with_control(0, control)?;
            let Some(image) = self.image.as_ref() else {
                return Err(J2kError::internal_backend("internal image cache missing"));
            };
            let Some(routed) = RoutedSrgb8::plan(image)? else {
                return control.with_native_context(&mut self.native_context, |context| {
                    decode_image_srgb8_with_control(image, context, control)
                });
            };
            let mut data = routed.allocate(
                self.native_context
                    .retained_allocation_bytes()
                    .map_err(|error| control.native_error(error))?,
                control,
            )?;
            let retained_capacity = data.capacity();
            let outcome = self.decode_into_cached_beside(
                &mut data,
                routed.stride(),
                routed.format(),
                retained_capacity,
                control,
            )?;
            routed.finish(data, outcome.warnings, control)
        })();
        self.trim_preparation_cache();
        result
    }

    /// Decode a source-coordinate region to tightly packed, straight-alpha
    /// eight-bit sRGB. Native reconstruction and colour conversion are bounded
    /// to the requested region and its wavelet/component support windows.
    /// Premultiplied input is unassociated as [`Self::decode_srgb8`] does.
    ///
    /// # Errors
    /// Rejects invalid regions, unsupported colour/alpha/precision combinations,
    /// malformed ICC profiles, allocation limits and invalid encoded data.
    pub fn decode_region_srgb8(&mut self, roi: Rect) -> Result<J2kSrgb8Image, J2kError> {
        self.decode_region_srgb8_at_reduction(roi, 0)
    }

    /// Decode a source-coordinate region at an exact native reduction, then
    /// normalize it to eight-bit sRGB with straight alpha. Reduction precedes
    /// colour normalization; this does not resample full-resolution sRGB output.
    ///
    /// # Errors
    /// Also rejects unavailable reductions and regions with no reduced samples.
    /// See [`Self::decode_region_srgb8`] for colour and decode errors.
    pub fn decode_region_srgb8_at_reduction(
        &mut self,
        roi: Rect,
        levels: u8,
    ) -> Result<J2kSrgb8Image, J2kError> {
        self.decode_region_srgb8_at_reduction_with_control(
            roi,
            levels,
            &J2kDecodeControl::default(),
        )
    }

    /// Decode an explicit sRGB region with operation-specific resource limits
    /// and cooperative cancellation. Colour conversion checks cancellation
    /// between bounded groups of pixels.
    ///
    /// ICC conversion supports output limits and cancellation. Under an
    /// explicit working-memory limit, the colour-management state of a
    /// restricted matrix/TRC or gray-TRC profile with an XYZ connection space
    /// is charged before it is created; other profiles are unsupported.
    ///
    /// # Errors
    /// Returns the region/decode errors above, or a control limit/cancellation error.
    pub fn decode_region_srgb8_at_reduction_with_control(
        &mut self,
        roi: Rect,
        levels: u8,
        control: &J2kDecodeControl,
    ) -> Result<J2kSrgb8Image, J2kError> {
        let result = (|| {
            control.check_cancelled()?;
            let plan = self
                .region_decode_plan_with_interpretation_and_control(None, roi, levels, control)?;
            control.check_output(plan.buffer_bytes)?;
            let image = self
                .image
                .as_ref()
                .ok_or_else(|| J2kError::internal_backend("internal image cache missing"))?;
            control.with_native_context(&mut self.native_context, |context| {
                decode_image_region_srgb8(image, context, plan.output_region, 0, control)
            })
        })();
        self.trim_preparation_cache();
        result
    }
}
