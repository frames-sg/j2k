// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sequential entropy traversal into preallocated DCT block storage.

use super::restart::{consume_restart_marker_if_due, finish_scan, McuPosition};
use super::PreparedDecodePlan;
use crate::entropy::block::{
    decode_block_dequantized_into, decode_block_quantized_and_dequantized_with_activity,
    CoefficientBlock,
};
use crate::error::JpegError;
use crate::internal::bit_reader::BitReader;

mod allocation;

use self::allocation::allocate_dct_decode_storage;
pub(crate) use self::allocation::{DecodedDctBlocks, SequentialDctLifecycleMetadata};

pub(crate) fn decode_scan_dct_blocks(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    retain_quantized_blocks: bool,
    lifecycle: SequentialDctLifecycleMetadata,
) -> Result<DecodedDctBlocks, JpegError> {
    let max_h = u32::from(plan.sampling.max_h);
    let max_v = u32::from(plan.sampling.max_v);
    let mcu_width_px = 8 * max_h;
    let mcu_height_px = 8 * max_v;
    let mcus_per_row = plan.dimensions.0.div_ceil(mcu_width_px);
    let mcu_rows = plan.dimensions.1.div_ceil(mcu_height_px);
    let mut storage = allocate_dct_decode_storage(
        plan,
        mcus_per_row,
        mcu_rows,
        retain_quantized_blocks,
        lifecycle,
    )?;

    let mut br = BitReader::new(scan_bytes);
    let mut quantized_coeff = CoefficientBlock::default();
    let mut dequantized_coeff = CoefficientBlock::default();
    let restart = plan.restart_interval.unwrap_or(0);
    let mut mcus_since_restart = 0_u32;
    let mut expected_rst = 0_u8;
    let total_mcus = mcu_rows * mcus_per_row;

    for mcu_y in 0..mcu_rows {
        for mcu_x in 0..mcus_per_row {
            let current_mcu = mcu_y * mcus_per_row + mcu_x;
            if consume_restart_marker_if_due(
                &mut br,
                restart,
                mcus_since_restart,
                &mut expected_rst,
                McuPosition {
                    current: current_mcu,
                    total: total_mcus,
                },
            )? {
                storage.prev_dc.fill(0);
                mcus_since_restart = 0;
            }

            for component in &plan.components {
                let plane_idx = component.output_index;
                let dc_table = plan.dc_table(component)?;
                let ac_table = plan.ac_table(component)?;
                let block_cols = storage.block_cols_by_component[plane_idx];
                for vy in 0..u32::from(component.v) {
                    for vx in 0..u32::from(component.h) {
                        let block_x = mcu_x * u32::from(component.h) + vx;
                        let block_y = mcu_y * u32::from(component.v) + vy;
                        let block_idx = (block_y * block_cols + block_x) as usize;
                        if retain_quantized_blocks {
                            decode_block_quantized_and_dequantized_with_activity(
                                &mut br,
                                dc_table,
                                ac_table,
                                &mut storage.prev_dc[plane_idx],
                                &component.quant,
                                &mut quantized_coeff,
                                &mut dequantized_coeff,
                            )?;
                            storage.quantized_blocks[plane_idx][block_idx] =
                                *quantized_coeff.coefficients();
                            storage.dequantized_blocks[plane_idx][block_idx] =
                                *dequantized_coeff.coefficients();
                        } else {
                            decode_block_dequantized_into(
                                &mut br,
                                dc_table,
                                ac_table,
                                &mut storage.prev_dc[plane_idx],
                                &component.quant,
                                &mut storage.dequantized_blocks[plane_idx][block_idx],
                            )?;
                        }
                    }
                }
            }

            mcus_since_restart += 1;
        }
    }

    finish_scan(&mut br, true)?;
    storage.finish(lifecycle)
}
