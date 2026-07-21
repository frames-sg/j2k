// SPDX-License-Identifier: MIT OR Apache-2.0

//! Referenced HTJ2K plans whose compressed payloads remain in caller-owned input.

use crate::error::bail;
use crate::j2c;
use crate::{
    ColorSpace, DecoderContext, DecodingError, DirectPlanUnsupportedReason,
    J2kReferencedClassicPlan, J2kReferencedHtj2kPlan, Result,
};

use super::super::Image;

impl<'a> Image<'a> {
    /// Build owned execution geometry whose classic compressed payload fragments
    /// are ranges into this image's borrowed encoded input rather than copied buffers.
    #[doc(hidden)]
    pub fn build_referenced_classic_plan_region_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
        output_region: (u32, u32, u32, u32),
    ) -> Result<J2kReferencedClassicPlan> {
        validate_referenced_color_shape(&self.color_space, self.has_alpha)?;
        let retained_metadata_bytes = self.retained_metadata_bytes()?;
        decoder_context.set_output_region(Some(output_region));
        let result = if matches!(self.color_space, ColorSpace::Gray) {
            j2c::build_referenced_classic_grayscale_plan(
                self.codestream,
                self.encoded_input,
                &self.header,
                retained_metadata_bytes,
                decoder_context,
            )
        } else if self.has_alpha {
            j2c::build_referenced_classic_rgba_plan(
                self.codestream,
                self.encoded_input,
                &self.header,
                retained_metadata_bytes,
                decoder_context,
            )
        } else {
            j2c::build_referenced_classic_color_plan(
                self.codestream,
                self.encoded_input,
                &self.header,
                retained_metadata_bytes,
                decoder_context,
            )
        };
        decoder_context.set_output_region(None);
        result
    }

    /// Build owned execution geometry whose HT compressed payloads are ranges
    /// into this image's borrowed raw codestream rather than copied buffers.
    #[doc(hidden)]
    pub fn build_referenced_htj2k_plan_region_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
        output_region: (u32, u32, u32, u32),
    ) -> Result<J2kReferencedHtj2kPlan> {
        validate_referenced_color_shape(&self.color_space, self.has_alpha)?;
        let retained_metadata_bytes = self.retained_metadata_bytes()?;
        decoder_context.set_output_region(Some(output_region));
        let result = if matches!(self.color_space, ColorSpace::Gray) {
            j2c::build_referenced_htj2k_grayscale_plan(
                self.codestream,
                self.encoded_input,
                &self.header,
                retained_metadata_bytes,
                decoder_context,
            )
        } else if self.has_alpha {
            j2c::build_referenced_htj2k_rgba_plan(
                self.codestream,
                self.encoded_input,
                &self.header,
                retained_metadata_bytes,
                decoder_context,
            )
        } else {
            j2c::build_referenced_htj2k_color_plan(
                self.codestream,
                self.encoded_input,
                &self.header,
                retained_metadata_bytes,
                decoder_context,
            )
        };
        decoder_context.set_output_region(None);
        result
    }
}

fn validate_referenced_color_shape(color_space: &ColorSpace, has_alpha: bool) -> Result<()> {
    if matches!(
        (color_space, has_alpha),
        (ColorSpace::Gray, false) | (ColorSpace::RGB, _)
    ) {
        return Ok(());
    }
    let reason = if has_alpha {
        DirectPlanUnsupportedReason::RgbaRgbImageWithAlpha
    } else {
        DirectPlanUnsupportedReason::ColorRgbImageWithoutAlpha
    };
    bail!(DecodingError::DirectPlanUnsupported(reason));
}
