// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    backend_image_with_reduction, decode_image_region_into_with_native_context,
    decode_warnings_for_image, validate_buffer, validate_region, J2kDecoder,
};
use crate::{
    decode::{J2kDecodeOutcome, J2kDecodedNativeComponents},
    scratch::J2kScratchPool,
    view::component_handoff_image_bytes,
    J2kError,
};
use j2k_core::{PixelFormat, Rect, Unsupported};

const UNREPRESENTABLE_REDUCTION: &str = "requested reduction exceeds supported image geometry";
const INEXACT_REDUCTION: &str = "native backend did not honor the requested reduction level";

fn unsupported(what: &'static str) -> J2kError {
    J2kError::Unsupported(Unsupported { what })
}

fn reduction_denominator(levels: u8) -> Result<u32, J2kError> {
    1_u32
        .checked_shl(u32::from(levels))
        .ok_or_else(|| unsupported(UNREPRESENTABLE_REDUCTION))
}

fn scaled_covering_pow2(rect: Rect, denominator: u32) -> Rect {
    let x_end = rect.x.saturating_add(rect.w);
    let y_end = rect.y.saturating_add(rect.h);
    let x0 = rect.x / denominator;
    let y0 = rect.y / denominator;
    let x1 = x_end.div_ceil(denominator);
    let y1 = y_end.div_ceil(denominator);
    Rect {
        x: x0,
        y: y0,
        w: x1.saturating_sub(x0),
        h: y1.saturating_sub(y0),
    }
}

impl J2kDecoder<'_> {
    /// Decode owned native component planes after discarding an exact number
    /// of JPEG 2000 resolution levels.
    ///
    /// A reduction of zero delegates to [`Self::decode_native_components`].
    /// Each additional level halves both axes using the codestream's wavelet
    /// resolution ladder; this method does not resample a full-resolution
    /// output.
    ///
    /// # Errors
    /// Returns [`J2kError`] when the reduction cannot be represented, exceeds
    /// any component's available resolution ladder, is not honored exactly by
    /// the native backend, or decode validation fails.
    pub fn decode_native_components_at_reduction(
        &mut self,
        reduction_levels: u8,
    ) -> Result<J2kDecodedNativeComponents, J2kError> {
        if reduction_levels == 0 {
            return self.decode_native_components();
        }

        let denominator = reduction_denominator(reduction_levels)?;
        let expected_dims = (
            self.info.dimensions.0.div_ceil(denominator),
            self.info.dimensions.1.div_ceil(denominator),
        );
        let image = backend_image_with_reduction(self.bytes, self.settings, reduction_levels)?;
        if (image.width(), image.height()) != expected_dims {
            return Err(unsupported(INEXACT_REDUCTION));
        }

        let retained_image_bytes = component_handoff_image_bytes(&image)?;
        let mut native_context = self.scaled_decode_native_context();
        let decoded = image
            .decode_native_components_with_context(&mut native_context)
            .map_err(J2kError::from_native_decode_error)?;
        J2kDecodedNativeComponents::try_from_native(decoded, retained_image_bytes)
    }

    /// Decode a source-coordinate region after discarding an exact number of
    /// JPEG 2000 resolution levels.
    ///
    /// `levels` counts power-of-two halvings. Zero delegates to full-resolution
    /// region decode, three is equivalent to [`j2k_core::Downscale::Eighth`],
    /// and deeper values use the codestream's wavelet ladder. `roi` remains in
    /// full-resolution source coordinates.
    ///
    /// # Errors
    /// Returns [`J2kError`] when the reduction cannot be represented, exceeds
    /// any component's available resolution ladder, is not honored exactly by
    /// the native backend, or when region, buffer, format, or decode validation
    /// fails.
    pub fn decode_region_scaled_pow2_into(
        &mut self,
        pool: &mut J2kScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
        levels: u8,
    ) -> Result<J2kDecodeOutcome, J2kError> {
        if levels == 0 {
            return self.decode_region_into(pool, out, stride, fmt, roi);
        }

        let denominator = reduction_denominator(levels)?;
        validate_region(roi, self.info.dimensions)?;
        let scaled_roi = scaled_covering_pow2(roi, denominator);
        validate_buffer((scaled_roi.w, scaled_roi.h), out.len(), stride, fmt)?;

        let expected_dims = (
            self.info.dimensions.0.div_ceil(denominator),
            self.info.dimensions.1.div_ceil(denominator),
        );
        let image = backend_image_with_reduction(self.bytes, self.settings, levels)?;
        let image_dims = (image.width(), image.height());
        if image_dims != expected_dims {
            return Err(unsupported(INEXACT_REDUCTION));
        }
        validate_region(scaled_roi, image_dims)?;

        let warnings = decode_warnings_for_image(&image);
        let mut native_context = self.scaled_decode_native_context();
        decode_image_region_into_with_native_context(
            &image,
            &mut native_context,
            out,
            stride,
            fmt,
            scaled_roi,
        )?;
        Ok(j2k_core::DecodeOutcome::new(scaled_roi, warnings))
    }
}
