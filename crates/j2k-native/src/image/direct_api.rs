// SPDX-License-Identifier: MIT OR Apache-2.0

//! Direct device-plan construction without host component materialization.

use crate::error::bail;
use crate::j2c;
use crate::{
    ColorSpace, DecoderContext, DecodingError, DirectPlanUnsupportedReason, J2kDirectColorPlan,
    J2kDirectGrayscalePlan, Result,
};

use super::Image;

impl<'a> Image<'a> {
    /// Build an adapter grayscale direct device plan without materializing host component planes.
    #[doc(hidden)]
    pub fn build_direct_grayscale_plan_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<J2kDirectGrayscalePlan> {
        if !matches!(self.color_space, ColorSpace::Gray) || self.has_alpha {
            bail!(DecodingError::DirectPlanUnsupported(
                DirectPlanUnsupportedReason::GrayscaleImageWithoutAlpha
            ));
        }

        j2c::build_direct_grayscale_plan(
            self.codestream,
            &self.header,
            self.retained_metadata_bytes()?,
            decoder_context,
        )
    }

    /// Build an adapter grayscale direct device plan for an output-space region.
    #[doc(hidden)]
    pub fn build_direct_grayscale_plan_region_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
        output_region: (u32, u32, u32, u32),
    ) -> Result<J2kDirectGrayscalePlan> {
        if !matches!(self.color_space, ColorSpace::Gray) || self.has_alpha {
            bail!(DecodingError::DirectPlanUnsupported(
                DirectPlanUnsupportedReason::GrayscaleImageWithoutAlpha
            ));
        }

        let retained_metadata_bytes = self.retained_metadata_bytes()?;
        decoder_context.set_output_region(Some(output_region));
        let result = j2c::build_direct_grayscale_plan(
            self.codestream,
            &self.header,
            retained_metadata_bytes,
            decoder_context,
        );
        decoder_context.set_output_region(None);
        result
    }

    /// Build an adapter RGB direct device plan without materializing host component planes.
    #[doc(hidden)]
    pub fn build_direct_color_plan_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<J2kDirectColorPlan> {
        if !matches!(self.color_space, ColorSpace::RGB) || self.has_alpha {
            bail!(DecodingError::DirectPlanUnsupported(
                DirectPlanUnsupportedReason::ColorRgbImageWithoutAlpha
            ));
        }

        j2c::build_direct_color_plan(
            self.codestream,
            &self.header,
            self.retained_metadata_bytes()?,
            decoder_context,
        )
    }

    /// Build an adapter RGB direct device plan for an output-space region.
    #[doc(hidden)]
    pub fn build_direct_color_plan_region_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
        output_region: (u32, u32, u32, u32),
    ) -> Result<J2kDirectColorPlan> {
        if !matches!(self.color_space, ColorSpace::RGB) || self.has_alpha {
            bail!(DecodingError::DirectPlanUnsupported(
                DirectPlanUnsupportedReason::ColorRgbImageWithoutAlpha
            ));
        }

        let retained_metadata_bytes = self.retained_metadata_bytes()?;
        decoder_context.set_output_region(Some(output_region));
        let result = j2c::build_direct_color_plan(
            self.codestream,
            &self.header,
            retained_metadata_bytes,
            decoder_context,
        );
        decoder_context.set_output_region(None);
        result
    }
}
