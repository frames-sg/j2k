// SPDX-License-Identifier: MIT OR Apache-2.0

//! Progressive 12-bit routing and grayscale rendering.

use super::super::{
    decode_progressive_dct_blocks, scaled_rect_covering, try_clone_warnings, ColorSpace,
    DecodeOutcome, Decoder, DownscaleFactor, JpegError, Rect, SofKind,
};
use super::planes::{dequantize_progressive12_block, ensure_progressive12_coefficient_capacities};
use super::sampling::{
    progressive_color_sampling, progressive_four_component_sampling, Extended12ColorSampling,
};
use super::writers::{
    write_extended12_block_region, Extended12Output, Extended12RgbProjection, Extended12WriteRegion,
};

mod color444;
mod four_component;
mod subsampled;

impl Decoder<'_> {
    pub(in crate::decoder) fn decode_progressive12_gray16_region_scaled_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_progressive12_region_into(out, stride, roi, downscale, Extended12Output::Gray16)
    }

    pub(in crate::decoder) fn decode_progressive12_rgb16_region_scaled_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_progressive12_region_into(out, stride, roi, downscale, Extended12Output::Rgb16)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the progressive 12-bit pass keeps scan state, coefficient reconstruction, and output routing in decode order"
    )]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "validated block indices are bounded by u32 JPEG image dimensions"
    )]
    fn decode_progressive12_region_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
        output: Extended12Output,
    ) -> Result<DecodeOutcome, JpegError> {
        let plan = self
            .progressive_plan
            .as_ref()
            .ok_or(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            })?;
        if self.info.sof_kind != SofKind::Progressive12
            || !matches!(
                self.info.color_space,
                ColorSpace::Grayscale
                    | ColorSpace::YCbCr
                    | ColorSpace::Rgb
                    | ColorSpace::Cmyk
                    | ColorSpace::Ycck
            )
        {
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
                    let sampling = progressive_color_sampling(plan, self.info.sof_kind)?;
                    return match sampling {
                        Extended12ColorSampling::S444 => self
                            .decode_progressive12_color444_region_into(
                                out,
                                stride,
                                roi,
                                downscale,
                                Extended12RgbProjection::Identity,
                            ),
                        Extended12ColorSampling::S422 | Extended12ColorSampling::S420 => self
                            .decode_progressive12_color_subsampled_region_into(
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
                    let sampling = progressive_color_sampling(plan, self.info.sof_kind)?;
                    return match sampling {
                        Extended12ColorSampling::S444 => self
                            .decode_progressive12_color444_region_into(
                                out,
                                stride,
                                roi,
                                downscale,
                                Extended12RgbProjection::YCbCr,
                            ),
                        Extended12ColorSampling::S422 | Extended12ColorSampling::S420 => self
                            .decode_progressive12_color_subsampled_region_into(
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
                    let sampling = progressive_four_component_sampling(plan, self.info.sof_kind)?;
                    return self.decode_progressive12_four_component_region_into(
                        out, stride, roi, downscale, sampling,
                    );
                }
                ColorSpace::Grayscale => {}
            }
        }
        if self.info.color_space != ColorSpace::Grayscale || plan.components.len() != 1 {
            return Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            });
        }

        let output_rect = scaled_rect_covering(roi, downscale)?;
        let dct_blocks = decode_progressive_dct_blocks(plan, self.bytes, 0)?;
        ensure_progressive12_coefficient_capacities(&dct_blocks, plan.scratch_bytes)?;
        let component = &plan.components[0];
        let component_coeffs = &dct_blocks.quantized[0];
        let (width, height) = self.info.dimensions;
        let mut dequant = [0i16; 64];
        let mut pixels = [0u16; 64];
        let write_region = Extended12WriteRegion {
            output_rect,
            dimensions: (width, height),
            downscale,
            output,
        };

        for block_y in 0..component.block_rows as usize {
            for block_x in 0..component.block_cols as usize {
                let block_index = block_y * component.block_cols as usize + block_x;
                dequantize_progressive12_block(
                    &component_coeffs[block_index],
                    &component.quant,
                    &mut dequant,
                );
                if dequant[1..].iter().all(|&coeff| coeff == 0) {
                    pixels.fill(crate::idct::idct_islow_12bit_dc_only_sample(dequant[0]));
                } else {
                    crate::idct::idct_islow_12bit(&dequant, &mut pixels);
                }
                write_extended12_block_region(
                    out,
                    stride,
                    write_region,
                    ((block_x as u32) * 8, (block_y as u32) * 8),
                    &pixels,
                );
            }
        }

        Ok(DecodeOutcome {
            decoded: roi,
            warnings: try_clone_warnings(&self.warnings)?,
        })
    }
}
