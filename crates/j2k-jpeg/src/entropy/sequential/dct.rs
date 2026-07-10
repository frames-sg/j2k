// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sequential entropy decode into quantized and dequantized DCT block planes.

use super::restart::finish_scan;
use super::PreparedDecodePlan;
use crate::entropy::block::{
    decode_block_dequantized_into, decode_block_quantized_and_dequantized_with_activity,
    CoefficientBlock,
};
use crate::error::JpegError;
use crate::internal::bit_reader::BitReader;
use alloc::vec::Vec;

pub(crate) fn decode_scan_dct_blocks(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    retain_quantized_blocks: bool,
) -> Result<DecodedDctBlocks, JpegError> {
    let max_h = u32::from(plan.sampling.max_h);
    let max_v = u32::from(plan.sampling.max_v);
    let mcu_width_px = 8 * max_h;
    let mcu_height_px = 8 * max_v;
    let mcus_per_row = plan.dimensions.0.div_ceil(mcu_width_px);
    let mcu_rows = plan.dimensions.1.div_ceil(mcu_height_px);
    let component_count = plan.sampling.len();

    let mut block_cols_by_component = vec![0_u32; component_count];
    let mut block_rows_by_component = vec![0_u32; component_count];
    for component in &plan.components {
        block_cols_by_component[component.output_index] = mcus_per_row * u32::from(component.h);
        block_rows_by_component[component.output_index] = mcu_rows * u32::from(component.v);
    }

    let mut quantized_blocks = block_cols_by_component
        .iter()
        .zip(block_rows_by_component.iter())
        .map(|(&cols, &rows)| {
            if retain_quantized_blocks {
                vec![[0_i16; 64]; (cols * rows) as usize]
            } else {
                Vec::new()
            }
        })
        .collect::<Vec<_>>();
    let mut dequantized_blocks = block_cols_by_component
        .iter()
        .zip(block_rows_by_component.iter())
        .map(|(&cols, &rows)| vec![[0_i16; 64]; (cols * rows) as usize])
        .collect::<Vec<_>>();

    let mut br = BitReader::new(scan_bytes);
    let mut prev_dc = vec![0_i32; component_count];
    let mut quantized_coeff = CoefficientBlock::default();
    let mut dequantized_coeff = CoefficientBlock::default();
    let restart = plan.restart_interval.unwrap_or(0);
    let mut mcus_since_restart = 0_u32;
    let mut expected_rst = 0_u8;
    let total_mcus = mcu_rows * mcus_per_row;

    for mcu_y in 0..mcu_rows {
        for mcu_x in 0..mcus_per_row {
            let current_mcu = mcu_y * mcus_per_row + mcu_x;
            if restart > 0 && mcus_since_restart == u32::from(restart) {
                let _ = br.ensure_bits(1);
                let marker = br.take_marker().ok_or(JpegError::UnexpectedEoi {
                    mcu_at: current_mcu,
                    mcu_total: total_mcus,
                })?;
                let expected = 0xD0 | expected_rst;
                if marker != expected {
                    return Err(JpegError::RestartMismatch {
                        offset: br.position(),
                        expected: expected_rst,
                        found: marker,
                    });
                }
                expected_rst = (expected_rst + 1) & 0x07;
                br.reset_at_restart();
                prev_dc.fill(0);
                mcus_since_restart = 0;
            }

            for component in &plan.components {
                let plane_idx = component.output_index;
                let block_cols = block_cols_by_component[plane_idx];
                for vy in 0..u32::from(component.v) {
                    for vx in 0..u32::from(component.h) {
                        let block_x = mcu_x * u32::from(component.h) + vx;
                        let block_y = mcu_y * u32::from(component.v) + vy;
                        let block_idx = (block_y * block_cols + block_x) as usize;
                        if retain_quantized_blocks {
                            decode_block_quantized_and_dequantized_with_activity(
                                &mut br,
                                &component.dc_table,
                                &component.ac_table,
                                &mut prev_dc[plane_idx],
                                &component.quant,
                                &mut quantized_coeff,
                                &mut dequantized_coeff,
                            )?;
                            quantized_blocks[plane_idx][block_idx] =
                                *quantized_coeff.coefficients();
                        } else {
                            decode_block_dequantized_into(
                                &mut br,
                                &component.dc_table,
                                &component.ac_table,
                                &mut prev_dc[plane_idx],
                                &component.quant,
                                &mut dequantized_blocks[plane_idx][block_idx],
                            )?;
                        }
                        if retain_quantized_blocks {
                            dequantized_blocks[plane_idx][block_idx] =
                                *dequantized_coeff.coefficients();
                        }
                    }
                }
            }

            mcus_since_restart += 1;
        }
    }

    finish_scan(&mut br, true)?;
    Ok(DecodedDctBlocks {
        quantized: quantized_blocks,
        dequantized: dequantized_blocks,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecodedDctBlocks {
    pub(crate) quantized: Vec<Vec<[i16; 64]>>,
    pub(crate) dequantized: Vec<Vec<[i16; 64]>>,
}
