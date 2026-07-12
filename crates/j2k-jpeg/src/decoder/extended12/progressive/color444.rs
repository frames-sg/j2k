// SPDX-License-Identifier: MIT OR Apache-2.0

//! Direct progressive 12-bit 4:4:4 rendering.

use super::super::super::{
    decode_progressive_dct_blocks, scaled_rect_covering, try_clone_warnings, DecodeOutcome,
    Decoder, DownscaleFactor, JpegError, Rect,
};
use super::super::planes::{
    dequantize_progressive12_block, ensure_progressive12_coefficient_capacities,
};
use super::super::sampling::progressive_color_component_indices;
use super::super::writers::{
    write_extended12_rgb_block_region, Extended12Output, Extended12RgbProjection,
    Extended12WriteRegion,
};

impl Decoder<'_> {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "validated block indices are bounded by u32 JPEG image dimensions"
    )]
    pub(super) fn decode_progressive12_color444_region_into(
        &self,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
        projection: Extended12RgbProjection,
    ) -> Result<DecodeOutcome, JpegError> {
        let plan = self
            .progressive_plan
            .as_ref()
            .ok_or(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            })?;
        if plan.components.len() != 3
            || plan.sampling.max_h != 1
            || plan.sampling.max_v != 1
            || plan
                .components
                .iter()
                .any(|component| component.h != 1 || component.v != 1 || component.output_index > 2)
        {
            return Err(JpegError::NotImplemented {
                sof: self.info.sof_kind,
            });
        }

        let output_rect = scaled_rect_covering(roi, downscale)?;
        let dct_blocks = decode_progressive_dct_blocks(plan, self.bytes, 0)?;
        ensure_progressive12_coefficient_capacities(&dct_blocks, plan.scratch_bytes)?;
        let (width, height) = self.info.dimensions;
        let component_indices = progressive_color_component_indices(plan)?;
        let block_cols = plan.components[component_indices[0]].block_cols as usize;
        let block_rows = plan.components[component_indices[0]].block_rows as usize;
        let mut dequant = [[0i16; 64]; 3];
        let mut pixels = [[0u16; 64]; 3];
        let write_region = Extended12WriteRegion {
            output_rect,
            dimensions: (width, height),
            downscale,
            output: Extended12Output::Rgb16,
        };

        for block_y in 0..block_rows {
            for block_x in 0..block_cols {
                for output_index in 0..3 {
                    let component_index = component_indices[output_index];
                    let component = &plan.components[component_index];
                    let component_coeffs = &dct_blocks.quantized[component_index];
                    let block_index = block_y * component.block_cols as usize + block_x;
                    dequantize_progressive12_block(
                        &component_coeffs[block_index],
                        &component.quant,
                        &mut dequant[output_index],
                    );
                    if dequant[output_index][1..].iter().all(|&coeff| coeff == 0) {
                        pixels[output_index].fill(crate::idct::idct_islow_12bit_dc_only_sample(
                            dequant[output_index][0],
                        ));
                    } else {
                        crate::idct::idct_islow_12bit(
                            &dequant[output_index],
                            &mut pixels[output_index],
                        );
                    }
                }
                write_extended12_rgb_block_region(
                    out,
                    stride,
                    write_region,
                    projection,
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
