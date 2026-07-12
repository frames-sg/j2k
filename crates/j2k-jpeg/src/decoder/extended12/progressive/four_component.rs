// SPDX-License-Identifier: MIT OR Apache-2.0

//! Progressive 12-bit CMYK and YCCK rendering.

use super::super::super::{
    decode_progressive_dct_blocks, scaled_rect_covering, try_clone_warnings, DecodeOutcome,
    Decoder, DownscaleFactor, JpegError, Rect,
};
use super::super::planes::render_progressive12_four_component_planes;
use super::super::sampling::{progressive_four_component_sampling, Extended12ColorSampling};
use super::super::writers::{
    write_extended12_four_component_planes_region, Extended12Output, Extended12WriteRegion,
};

impl Decoder<'_> {
    pub(super) fn decode_progressive12_four_component_region_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
        sampling: Extended12ColorSampling,
    ) -> Result<DecodeOutcome, JpegError> {
        let plan = self
            .progressive_plan
            .as_ref()
            .ok_or(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            })?;
        if progressive_four_component_sampling(plan, self.info.sof_kind)? != sampling {
            return Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            });
        }

        let output_rect = scaled_rect_covering(roi, downscale)?;
        let dct_blocks = decode_progressive_dct_blocks(plan, self.bytes, 0)?;
        let planes = render_progressive12_four_component_planes(plan, &dct_blocks)?;
        write_extended12_four_component_planes_region(
            out,
            stride,
            Extended12WriteRegion {
                output_rect,
                dimensions: self.info.dimensions,
                downscale,
                output: Extended12Output::Rgb16,
            },
            self.info.color_space,
            sampling,
            &planes,
        );

        Ok(DecodeOutcome {
            decoded: roi,
            warnings: try_clone_warnings(&self.warnings)?,
        })
    }
}
