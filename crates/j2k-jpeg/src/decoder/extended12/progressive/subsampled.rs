// SPDX-License-Identifier: MIT OR Apache-2.0

//! Progressive 12-bit 4:2:2 and 4:2:0 rendering.

use super::super::super::{
    decode_progressive_dct_blocks, scaled_rect_covering, try_clone_warnings, DecodeOutcome,
    Decoder, DownscaleFactor, JpegError, Rect,
};
use super::super::planes::render_progressive12_color_planes;
use super::super::sampling::{progressive_color_sampling, Extended12ColorSampling};
use super::super::writers::{
    write_extended12_color420_planes_region, write_extended12_color422_planes_region,
    Extended12Output, Extended12RgbProjection, Extended12WriteRegion,
};

impl Decoder<'_> {
    pub(super) fn decode_progressive12_color_subsampled_region_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
        sampling: Extended12ColorSampling,
        projection: Extended12RgbProjection,
    ) -> Result<DecodeOutcome, JpegError> {
        let plan = self
            .progressive_plan
            .as_ref()
            .ok_or(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            })?;
        debug_assert!(matches!(
            sampling,
            Extended12ColorSampling::S422 | Extended12ColorSampling::S420
        ));
        if progressive_color_sampling(plan, self.info.sof_kind)? != sampling {
            return Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            });
        }

        let output_rect = scaled_rect_covering(roi, downscale)?;
        let dct_blocks = decode_progressive_dct_blocks(plan, self.bytes, 0)?;
        let planes = render_progressive12_color_planes(plan, &dct_blocks)?;
        let write_region = Extended12WriteRegion {
            output_rect,
            dimensions: self.info.dimensions,
            downscale,
            output: Extended12Output::Rgb16,
        };
        match sampling {
            Extended12ColorSampling::S444 => unreachable!("4:4:4 path is handled directly"),
            Extended12ColorSampling::S422 => write_extended12_color422_planes_region(
                out,
                stride,
                write_region,
                projection,
                &planes,
            ),
            Extended12ColorSampling::S420 => write_extended12_color420_planes_region(
                out,
                stride,
                write_region,
                projection,
                &planes,
            ),
        }

        Ok(DecodeOutcome {
            decoded: roi,
            warnings: try_clone_warnings(&self.warnings)?,
        })
    }
}
