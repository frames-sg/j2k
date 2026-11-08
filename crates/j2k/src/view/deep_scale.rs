// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    decode_image_region_into_with_native_context, decode_warnings_for_image, validate_buffer,
    validate_region, J2kDecoder,
};
use crate::{
    adapter::device_plan::reference_grid_covering,
    backend::Image,
    decode::{
        decode_native_components_with_native_context, native_components_route, J2kDecodeOutcome,
        J2kDecodedNativeComponents, PlanesBudget,
    },
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

pub(super) fn reduction_denominator(levels: u8) -> Result<u32, J2kError> {
    1_u32
        .checked_shl(u32::from(levels))
        .ok_or_else(|| unsupported(UNREPRESENTABLE_REDUCTION))
}

pub(super) fn validate_reduced_image_dimensions(
    image: &Image<'_>,
    source_dimensions: (u32, u32),
    denominator: u32,
) -> Result<(), J2kError> {
    let expected = image_covering(
        image,
        source_dimensions,
        Rect::full(source_dimensions),
        denominator,
    );
    if (image.width(), image.height()) != (expected.w, expected.h) {
        return Err(unsupported(INEXACT_REDUCTION));
    }
    Ok(())
}

/// The samples of `image`'s reduction by `denominator` whose reference-grid
/// footprints overlap region `rect` of the full-resolution image of `size`,
/// in reduced image coordinates (see [`reference_grid_covering`]).
pub(super) fn image_covering(
    image: &Image<'_>,
    size: (u32, u32),
    rect: Rect,
    denominator: u32,
) -> Rect {
    reference_grid_covering(image.image_origin(), size, rect, denominator)
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

        let result = (|| {
            let denominator = reduction_denominator(reduction_levels)?;
            self.ensure_image_at_reduction(reduction_levels)?;
            self.native_context.release_reusable_allocations();
            let image = self.cached_image()?;
            validate_reduced_image_dimensions(image, self.info.dimensions, denominator)?;
            let planes_budget = match self.full_decode_planes_budget() {
                PlanesBudget::Default => j2k_native::DEFAULT_MAX_DECODE_BYTES,
                PlanesBudget::Fixed(bytes) => bytes,
            };
            let route = native_components_route(image, planes_budget);
            let mut native_context = self.fresh_native_context();
            let decoded = decode_native_components_with_native_context(
                image,
                &mut native_context,
                None,
                route,
            )?;
            J2kDecodedNativeComponents::try_from_native(
                decoded,
                component_handoff_image_bytes(image)?,
            )
        })();
        self.trim_preparation_cache();
        result
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
    ///
    /// A decode that fails partway may already have written some rows of
    /// `out`.
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

        let result = (|| {
            let denominator = reduction_denominator(levels)?;
            let dims = self.info.dimensions;
            validate_region(roi, dims)?;
            self.ensure_image_at_reduction(levels)?;
            self.native_context.release_reusable_allocations();
            let image = self.cached_image()?;
            let image_dims = (image.width(), image.height());
            validate_reduced_image_dimensions(image, dims, denominator)?;
            let scaled_roi = image_covering(image, dims, roi, denominator);
            validate_buffer((scaled_roi.w, scaled_roi.h), out.len(), stride, fmt)?;
            validate_region(scaled_roi, image_dims)?;
            let warnings = decode_warnings_for_image(image);
            let mut native_context = self.fresh_native_context();
            decode_image_region_into_with_native_context(
                image,
                &mut native_context,
                out,
                stride,
                fmt,
                scaled_roi,
            )?;
            Ok(j2k_core::DecodeOutcome::new(scaled_roi, warnings))
        })();
        self.trim_preparation_cache();
        result
    }
}
