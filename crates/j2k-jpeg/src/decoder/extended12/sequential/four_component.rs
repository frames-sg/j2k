// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sequential 12-bit CMYK and YCCK rendering.

use super::super::super::{
    finish_scan, merged_warnings, scaled_rect_covering, BitReader, CoefficientBlock, DecodeOutcome,
    Decoder, DownscaleFactor, JpegError, Rect,
};
use super::super::planes::decode_extended12_four_component_planes;
use super::super::sampling::{
    extended12_four_component_sampling, validate_extended12_four_component444_plan,
    Extended12ColorSampling,
};
use super::super::state::{decode_extended12_block_pixels, Extended12RestartTracker};
use super::super::writers::{
    write_extended12_four_component_block_region, write_extended12_four_component_planes_region,
    Extended12Output, Extended12WriteRegion,
};

impl Decoder<'_> {
    pub(super) fn decode_extended12_four_component444_region_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
    ) -> Result<DecodeOutcome, JpegError> {
        validate_extended12_four_component444_plan(&self.plan, self.info.sof_kind)?;

        let output_rect = scaled_rect_covering(roi, downscale)?;
        let scan_bytes = &self.bytes[self.plan.scan_offset..];
        let (width, height) = self.info.dimensions;
        let mcu_cols = width.div_ceil(8);
        let mcu_rows = height.div_ceil(8);
        let mut br = BitReader::new(scan_bytes);
        let mut prev_dc = [0i32; 4];
        let mut coeffs: [CoefficientBlock; 4] =
            core::array::from_fn(|_| CoefficientBlock::default());
        let mut pixels = [[0u16; 64]; 4];
        let total_mcus = mcu_cols * mcu_rows;
        let mut restart_tracker =
            Extended12RestartTracker::new(self.plan.restart_interval, total_mcus);
        let write_region = Extended12WriteRegion {
            output_rect,
            dimensions: (width, height),
            downscale,
            output: Extended12Output::Rgb16,
        };

        for mcu_y in 0..mcu_rows {
            for mcu_x in 0..mcu_cols {
                let current_mcu = mcu_y * mcu_cols + mcu_x;
                if restart_tracker.begin_mcu(&mut br, current_mcu)? {
                    prev_dc.fill(0);
                }
                for component in &self.plan.components {
                    let output_index = component.output_index;
                    let component = self.plan.resolve_component(component)?;
                    decode_extended12_block_pixels(
                        &mut br,
                        component,
                        &mut prev_dc[output_index],
                        &mut coeffs[output_index],
                        &mut pixels[output_index],
                    )?;
                }
                write_extended12_four_component_block_region(
                    out,
                    stride,
                    write_region,
                    self.info.color_space,
                    (mcu_x * 8, mcu_y * 8),
                    &pixels,
                );
                restart_tracker.finish_mcu();
            }
        }

        let scan_warnings = finish_scan(&mut br, true)?;
        Ok(DecodeOutcome {
            decoded: roi,
            warnings: merged_warnings(&self.warnings, scan_warnings)?,
        })
    }

    pub(super) fn decode_extended12_four_component_subsampled_region_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
        sampling: Extended12ColorSampling,
    ) -> Result<DecodeOutcome, JpegError> {
        debug_assert!(matches!(
            sampling,
            Extended12ColorSampling::S422 | Extended12ColorSampling::S420
        ));
        if extended12_four_component_sampling(&self.plan, self.info.sof_kind)? != sampling {
            return Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            });
        }

        let output_rect = scaled_rect_covering(roi, downscale)?;
        let scan_bytes = &self.bytes[self.plan.scan_offset..];
        let (planes, scan_warnings) =
            decode_extended12_four_component_planes(&self.plan, scan_bytes, self.info.sof_kind)?;
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
            warnings: merged_warnings(&self.warnings, scan_warnings)?,
        })
    }
}
