// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sequential 12-bit routing and grayscale rendering.

use super::super::{
    finish_scan, merged_warnings, scaled_rect_covering, BitReader, CoefficientBlock, ColorSpace,
    DecodeOutcome, Decoder, DownscaleFactor, JpegError, Rect, SofKind,
};
use super::sampling::{
    extended12_color_sampling, extended12_four_component_sampling, Extended12ColorSampling,
};
use super::state::{decode_extended12_block_pixels, Extended12RestartTracker};
use super::writers::{
    write_extended12_block_region, Extended12Output, Extended12RgbProjection, Extended12WriteRegion,
};

mod color444;
mod four_component;
mod subsampled;

impl Decoder<'_> {
    pub(in crate::decoder) fn decode_extended12_gray16_into(
        &self,
        out: &mut [u8],
        stride: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_extended12_region_into(
            out,
            stride,
            Rect::full(self.info.dimensions),
            DownscaleFactor::Full,
            Extended12Output::Gray16,
        )
    }

    pub(in crate::decoder) fn decode_extended12_gray16_region_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_extended12_region_into(
            out,
            stride,
            roi,
            DownscaleFactor::Full,
            Extended12Output::Gray16,
        )
    }

    pub(in crate::decoder) fn decode_extended12_gray16_region_scaled_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_extended12_region_into(out, stride, roi, downscale, Extended12Output::Gray16)
    }

    pub(in crate::decoder) fn decode_extended12_rgb16_into(
        &self,
        out: &mut [u8],
        stride: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_extended12_region_into(
            out,
            stride,
            Rect::full(self.info.dimensions),
            DownscaleFactor::Full,
            Extended12Output::Rgb16,
        )
    }

    pub(in crate::decoder) fn decode_extended12_rgb16_region_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_extended12_region_into(
            out,
            stride,
            roi,
            DownscaleFactor::Full,
            Extended12Output::Rgb16,
        )
    }

    pub(in crate::decoder) fn decode_extended12_rgb16_region_scaled_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_extended12_region_into(out, stride, roi, downscale, Extended12Output::Rgb16)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the 12-bit sequential pass keeps entropy state, IDCT, color conversion, and row emission in decode order"
    )]
    fn decode_extended12_region_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
        output: Extended12Output,
    ) -> Result<DecodeOutcome, JpegError> {
        if self.info.sof_kind != SofKind::Extended12 {
            return Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            });
        }
        if !roi.is_within(self.info.dimensions) {
            return Err(JpegError::RectOutOfBounds {
                rect: roi,
                width: self.info.dimensions.0,
                height: self.info.dimensions.1,
            });
        }
        if matches!(output, Extended12Output::Rgb16) {
            match self.info.color_space {
                ColorSpace::Rgb => {
                    let sampling = extended12_color_sampling(&self.plan, self.info.sof_kind)?;
                    return match sampling {
                        Extended12ColorSampling::S444 => self
                            .decode_extended12_color444_region_into(
                                out,
                                stride,
                                roi,
                                downscale,
                                Extended12RgbProjection::Identity,
                            ),
                        Extended12ColorSampling::S422 | Extended12ColorSampling::S420 => self
                            .decode_extended12_color_subsampled_region_into(
                                out,
                                stride,
                                roi,
                                downscale,
                                sampling,
                                Extended12RgbProjection::Identity,
                            ),
                    };
                }
                ColorSpace::YCbCr => {
                    let sampling = extended12_color_sampling(&self.plan, self.info.sof_kind)?;
                    return match sampling {
                        Extended12ColorSampling::S444 => self
                            .decode_extended12_color444_region_into(
                                out,
                                stride,
                                roi,
                                downscale,
                                Extended12RgbProjection::YCbCr,
                            ),
                        Extended12ColorSampling::S422 | Extended12ColorSampling::S420 => self
                            .decode_extended12_color_subsampled_region_into(
                                out,
                                stride,
                                roi,
                                downscale,
                                sampling,
                                Extended12RgbProjection::YCbCr,
                            ),
                    };
                }
                ColorSpace::Cmyk | ColorSpace::Ycck => {
                    let sampling =
                        extended12_four_component_sampling(&self.plan, self.info.sof_kind)?;
                    return match sampling {
                        Extended12ColorSampling::S444 => self
                            .decode_extended12_four_component444_region_into(
                                out, stride, roi, downscale,
                            ),
                        Extended12ColorSampling::S422 | Extended12ColorSampling::S420 => self
                            .decode_extended12_four_component_subsampled_region_into(
                                out, stride, roi, downscale, sampling,
                            ),
                    };
                }
                ColorSpace::Grayscale => {}
            }
        }
        if self.info.color_space != ColorSpace::Grayscale || self.plan.components.len() != 1 {
            return Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            });
        }

        let output_rect = scaled_rect_covering(roi, downscale)?;
        let scan_bytes = &self.bytes[self.plan.scan_offset..];
        let component = &self.plan.components[0];
        let (width, height) = self.info.dimensions;
        let mcu_cols = width.div_ceil(8);
        let mcu_rows = height.div_ceil(8);
        let mut br = BitReader::new(scan_bytes);
        let mut prev_dc = 0i32;
        let mut coeff = CoefficientBlock::default();
        let mut pixels = [0u16; 64];
        let total_mcus = mcu_cols * mcu_rows;
        let mut restart_tracker =
            Extended12RestartTracker::new(self.plan.restart_interval, total_mcus);
        let write_region = Extended12WriteRegion {
            output_rect,
            dimensions: (width, height),
            downscale,
            output,
        };

        for mcu_y in 0..mcu_rows {
            for mcu_x in 0..mcu_cols {
                let current_mcu = mcu_y * mcu_cols + mcu_x;
                if restart_tracker.begin_mcu(&mut br, current_mcu)? {
                    prev_dc = 0;
                }
                decode_extended12_block_pixels(
                    &mut br,
                    component,
                    &mut prev_dc,
                    &mut coeff,
                    &mut pixels,
                )?;
                write_extended12_block_region(
                    out,
                    stride,
                    write_region,
                    (mcu_x * 8, mcu_y * 8),
                    &pixels,
                );
                restart_tracker.finish_mcu();
            }
        }

        let scan_warnings = finish_scan(&mut br, true)?;
        Ok(DecodeOutcome {
            decoded: roi,
            warnings: merged_warnings(&self.warnings, scan_warnings),
        })
    }
}
