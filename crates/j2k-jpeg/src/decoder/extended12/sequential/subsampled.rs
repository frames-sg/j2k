// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sequential 12-bit 4:2:2 and 4:2:0 rendering.

use super::super::super::{
    merged_warnings, scaled_rect_covering, DecodeOutcome, Decoder, DownscaleFactor, JpegError, Rect,
};
use super::super::planes::decode_extended12_color_planes;
use super::super::sampling::{extended12_color_sampling, Extended12ColorSampling};
use super::super::writers::{
    write_extended12_color420_planes_region, write_extended12_color422_planes_region,
    Extended12Output, Extended12RgbProjection, Extended12WriteRegion,
};

impl Decoder<'_> {
    pub(super) fn decode_extended12_color_subsampled_region_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
        sampling: Extended12ColorSampling,
        projection: Extended12RgbProjection,
    ) -> Result<DecodeOutcome, JpegError> {
        debug_assert!(matches!(
            sampling,
            Extended12ColorSampling::S422 | Extended12ColorSampling::S420
        ));
        if extended12_color_sampling(&self.plan, self.info.sof_kind)? != sampling {
            return Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            });
        }

        let output_rect = scaled_rect_covering(roi, downscale)?;
        let scan_bytes = &self.bytes[self.plan.scan_offset..];
        let (planes, scan_warnings) =
            decode_extended12_color_planes(&self.plan, scan_bytes, self.info.sof_kind)?;
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
            warnings: merged_warnings(&self.warnings, scan_warnings),
        })
    }
}
