// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sequential entropy traversal into preallocated DCT block storage.

use super::layout::McuColumns;
use super::restart::{enter_scan_at_mcu, finish_scan, skip_mcu, RestartState};
use super::{HuffmanPrimaryWidth, PreparedDecodePlan};
use crate::entropy::block::{
    decode_block_dequantized_into, decode_block_quantized_and_dequantized_with_activity,
    CoefficientBlock,
};
use crate::entropy::huffman::{FULL_PRIMARY_ENTRIES, NARROW_PRIMARY_ENTRIES};
use crate::error::JpegError;
use crate::internal::bit_reader::BitReader;
use crate::internal::checkpoint::DeviceCheckpoint;

mod allocation;

use self::allocation::allocate_dct_decode_storage;
pub(crate) use self::allocation::{DecodedDctBlocks, SequentialDctLifecycleMetadata};

pub(crate) fn decode_scan_dct_blocks(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    retain_quantized_blocks: bool,
    lifecycle: SequentialDctLifecycleMetadata,
) -> Result<DecodedDctBlocks, JpegError> {
    match plan.huffman_primary_width {
        HuffmanPrimaryWidth::Narrow8 => {
            decode_scan_dct_blocks_with_entries::<NARROW_PRIMARY_ENTRIES>(
                plan,
                scan_bytes,
                retain_quantized_blocks,
                lifecycle,
            )
        }
        HuffmanPrimaryWidth::Full12 => decode_scan_dct_blocks_with_entries::<FULL_PRIMARY_ENTRIES>(
            plan,
            scan_bytes,
            retain_quantized_blocks,
            lifecycle,
        ),
    }
}

fn decode_scan_dct_blocks_with_entries<const N: usize>(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    retain_quantized_blocks: bool,
    lifecycle: SequentialDctLifecycleMetadata,
) -> Result<DecodedDctBlocks, JpegError> {
    let grid = sequential_mcu_grid(plan);
    let (mcus_per_row, mcu_rows) = grid;
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
    let mut restart = RestartState::new(plan.restart_interval.unwrap_or(0));
    let columns = McuColumns::whole_rows(grid);

    for mcu_y in 0..mcu_rows {
        for mcu_x in 0..mcus_per_row {
            if restart.begin_mcu(&mut br, columns.mcu_position(mcu_x, mcu_y))? {
                storage.prev_dc.fill(0);
            }

            for component in &plan.components {
                let plane_idx = component.output_index;
                let dc_table = plan.dc_table_with_entries::<N>(component)?;
                let ac_table = plan.ac_table_with_entries::<N>(component)?;
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

            restart.end_mcu();
        }
    }

    finish_scan(&mut br, true)?;
    storage.finish(lifecycle)
}

/// Full-size pixel size `(width, height)` of an MCU of a sequential
/// interleaved scan.
pub(crate) fn sequential_mcu_size(plan: &PreparedDecodePlan) -> (u32, u32) {
    (
        8 * u32::from(plan.sampling.max_h),
        8 * u32::from(plan.sampling.max_v),
    )
}

/// MCU grid `(mcus_per_row, mcu_rows)` of a sequential interleaved scan, at
/// full size or any downscale (both edges round up to whole MCUs).
pub(crate) fn sequential_mcu_grid(plan: &PreparedDecodePlan) -> (u32, u32) {
    let (mcu_width_px, mcu_height_px) = sequential_mcu_size(plan);
    (
        plan.dimensions.0.div_ceil(mcu_width_px),
        plan.dimensions.1.div_ceil(mcu_height_px),
    )
}

/// Decode a sequential scan straight into caller-owned dequantized blocks.
///
/// `blocks` holds every component's MCU-padded block grid back to back in
/// plane (`output_index`) order; each grid is `mcus_per_row * h` blocks wide
/// and `mcu_rows * v` blocks tall. Every block is overwritten, so the storage
/// need not be zeroed. Coefficients are dequantized exactly as the CPU decode
/// paths do before their IDCT.
pub(crate) fn decode_scan_dct_blocks_into(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    blocks: &mut [[i16; 64]],
) -> Result<alloc::vec::Vec<crate::Warning>, JpegError> {
    match plan.huffman_primary_width {
        HuffmanPrimaryWidth::Narrow8 => decode_scan_dct_blocks_into_with_entries::<
            NARROW_PRIMARY_ENTRIES,
        >(plan, scan_bytes, blocks),
        HuffmanPrimaryWidth::Full12 => decode_scan_dct_blocks_into_with_entries::<
            FULL_PRIMARY_ENTRIES,
        >(plan, scan_bytes, blocks),
    }
}

fn decode_scan_dct_blocks_into_with_entries<const N: usize>(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    blocks: &mut [[i16; 64]],
) -> Result<alloc::vec::Vec<crate::Warning>, JpegError> {
    let (mcus_per_row, mcu_rows) = sequential_mcu_grid(plan);
    let grid = DeviceBlockGrid::new(
        plan,
        (mcus_per_row, mcu_rows),
        blocks.len(),
        "sequential device DCT storage disagrees with the scan block grid",
    )?;

    let mut br = BitReader::new(scan_bytes);
    let mut prev_dc = [0_i32; DEVICE_MAX_COMPONENTS];
    let mut restart = RestartState::new(plan.restart_interval.unwrap_or(0));
    let columns = McuColumns::whole_rows((mcus_per_row, mcu_rows));

    for mcu_y in 0..mcu_rows {
        for mcu_x in 0..mcus_per_row {
            if restart.begin_mcu(&mut br, columns.mcu_position(mcu_x, mcu_y))? {
                prev_dc.fill(0);
            }
            grid.decode_mcu::<N>(plan, &mut br, &mut prev_dc, blocks, (mcu_x, mcu_y))?;
            restart.end_mcu();
        }
    }

    finish_scan(&mut br, true)
}

const DEVICE_MAX_COMPONENTS: usize = 4;

/// Layout of caller-owned device DCT storage: each plane's width in blocks
/// and the index of its first block.
struct DeviceBlockGrid {
    cols: [u32; DEVICE_MAX_COMPONENTS],
    bases: [usize; DEVICE_MAX_COMPONENTS],
}

impl DeviceBlockGrid {
    /// Grid of `mcu_cols` by `mcu_rows` MCUs in plane order, checked against
    /// the caller's `blocks_len`.
    fn new(
        plan: &PreparedDecodePlan,
        (mcu_cols, mcu_rows): (u32, u32),
        blocks_len: usize,
        mismatch: &'static str,
    ) -> Result<Self, JpegError> {
        let component_count = plan.sampling.len();
        if component_count > DEVICE_MAX_COMPONENTS || plan.components.len() != component_count {
            return Err(JpegError::InternalInvariant {
                reason: "sequential device DCT scan must cover every frame component",
            });
        }
        let mut cols = [0_u32; DEVICE_MAX_COMPONENTS];
        let mut bases = [0_usize; DEVICE_MAX_COMPONENTS];
        let mut total_blocks = 0_usize;
        for index in 0..component_count {
            let component = plan
                .components
                .iter()
                .find(|component| component.output_index == index)
                .ok_or(JpegError::InternalInvariant {
                    reason: "sequential device DCT components do not cover every plane",
                })?;
            let plane_cols = mcu_cols * u32::from(component.h);
            let plane_rows = mcu_rows * u32::from(component.v);
            cols[index] = plane_cols;
            bases[index] = total_blocks;
            total_blocks = (plane_cols as usize)
                .checked_mul(plane_rows as usize)
                .and_then(|count| total_blocks.checked_add(count))
                .ok_or(JpegError::InternalInvariant {
                    reason: "sequential device DCT block count overflowed",
                })?;
        }
        if blocks_len != total_blocks {
            return Err(JpegError::InternalInvariant { reason: mismatch });
        }
        Ok(Self { cols, bases })
    }

    /// Decode one interleaved MCU into grid MCU `(mcu_x, mcu_y)`.
    fn decode_mcu<const N: usize>(
        &self,
        plan: &PreparedDecodePlan,
        br: &mut BitReader<'_>,
        prev_dc: &mut [i32; DEVICE_MAX_COMPONENTS],
        blocks: &mut [[i16; 64]],
        (mcu_x, mcu_y): (u32, u32),
    ) -> Result<(), JpegError> {
        for component in &plan.components {
            let plane_idx = component.output_index;
            let dc_table = plan.dc_table_with_entries::<N>(component)?;
            let ac_table = plan.ac_table_with_entries::<N>(component)?;
            let cols = self.cols[plane_idx];
            for vy in 0..u32::from(component.v) {
                for vx in 0..u32::from(component.h) {
                    let block_x = mcu_x * u32::from(component.h) + vx;
                    let block_y = mcu_y * u32::from(component.v) + vy;
                    let block =
                        &mut blocks[self.bases[plane_idx] + (block_y * cols + block_x) as usize];
                    *block = [0; 64];
                    decode_block_dequantized_into(
                        br,
                        dc_table,
                        ac_table,
                        &mut prev_dc[plane_idx],
                        &component.quant,
                        block,
                    )?;
                }
            }
        }
        Ok(())
    }
}

/// MCU rectangle of a windowed sequential device decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct McuWindow {
    pub(crate) col: u32,
    pub(crate) cols: u32,
    pub(crate) row: u32,
    pub(crate) rows: u32,
}

/// Decode only the MCUs of `window` into caller-owned dequantized blocks.
///
/// As CPU region decode does, the scan is entered through
/// `scan_position_before_mcu` (at `checkpoint`, the decoder's resume point
/// for the window's first MCU row, or at the nearest restart marker, or at
/// the scan start), MCUs
/// left and right of the window are skipped entropy-only, and decoding stops
/// after the window's last MCU row without validating the rest of the scan.
/// `blocks` holds each component's `window.cols * h` by `window.rows * v`
/// block grid back to back in plane order.
pub(crate) fn decode_scan_dct_window_into(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    window: McuWindow,
    checkpoint: Option<&DeviceCheckpoint>,
    blocks: &mut [[i16; 64]],
) -> Result<(), JpegError> {
    match plan.huffman_primary_width {
        HuffmanPrimaryWidth::Narrow8 => decode_scan_dct_window_into_with_entries::<
            NARROW_PRIMARY_ENTRIES,
        >(plan, scan_bytes, window, checkpoint, blocks),
        HuffmanPrimaryWidth::Full12 => decode_scan_dct_window_into_with_entries::<
            FULL_PRIMARY_ENTRIES,
        >(plan, scan_bytes, window, checkpoint, blocks),
    }
}

fn decode_scan_dct_window_into_with_entries<const N: usize>(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    window: McuWindow,
    checkpoint: Option<&DeviceCheckpoint>,
    blocks: &mut [[i16; 64]],
) -> Result<(), JpegError> {
    let (mcus_per_row, mcu_rows) = sequential_mcu_grid(plan);
    if window.cols == 0
        || window.rows == 0
        || window.col + window.cols > mcus_per_row
        || window.row + window.rows > mcu_rows
    {
        return Err(JpegError::InternalInvariant {
            reason: "sequential device DCT window lies outside the scan",
        });
    }
    let grid = DeviceBlockGrid::new(
        plan,
        (window.cols, window.rows),
        blocks.len(),
        "sequential device DCT storage disagrees with the window block grid",
    )?;

    let columns = McuColumns::new((mcus_per_row, mcu_rows), window.col, window.cols);
    let wide_huffman = plan.prefers_wide_huffman(scan_bytes.len());
    let mut cursor = enter_scan_at_mcu::<N>(
        plan,
        scan_bytes,
        checkpoint,
        columns.mcu_position(0, window.row),
        wide_huffman,
    )?;

    for row in 0..window.rows {
        for mcu_x in 0..mcus_per_row {
            if cursor.restart.begin_mcu(
                &mut cursor.br,
                columns.mcu_position(mcu_x, window.row + row),
            )? {
                cursor.prev_dc = [0; 4];
            }
            if columns.contains(mcu_x) {
                grid.decode_mcu::<N>(
                    plan,
                    &mut cursor.br,
                    &mut cursor.prev_dc,
                    blocks,
                    (columns.local(mcu_x), row),
                )?;
            } else {
                skip_mcu::<N>(plan, &mut cursor.br, &mut cursor.prev_dc, wide_huffman)?;
            }
            cursor.restart.end_mcu();
        }
    }
    Ok(())
}
