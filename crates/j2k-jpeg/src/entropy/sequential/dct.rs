// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sequential entropy traversal into preallocated DCT block storage.

use super::layout::McuColumns;
use super::restart::{enter_scan_at_mcu, finish_scan, skip_mcu, RestartState};
use super::{HuffmanPrimaryWidth, PreparedDecodePlan};
use crate::entropy::block::{
    decode_block_dequantized_into, decode_block_quantized_and_dequantized_with_activity,
    decode_block_quantized_into, CoefficientBlock,
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
    retain_dequantized_blocks: bool,
    lifecycle: SequentialDctLifecycleMetadata,
) -> Result<DecodedDctBlocks, JpegError> {
    match plan.huffman_primary_width {
        HuffmanPrimaryWidth::Narrow8 => {
            decode_scan_dct_blocks_with_entries::<NARROW_PRIMARY_ENTRIES>(
                plan,
                scan_bytes,
                retain_quantized_blocks,
                retain_dequantized_blocks,
                lifecycle,
            )
        }
        HuffmanPrimaryWidth::Full12 => decode_scan_dct_blocks_with_entries::<FULL_PRIMARY_ENTRIES>(
            plan,
            scan_bytes,
            retain_quantized_blocks,
            retain_dequantized_blocks,
            lifecycle,
        ),
    }
}

fn decode_scan_dct_blocks_with_entries<const N: usize>(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    retain_quantized_blocks: bool,
    retain_dequantized_blocks: bool,
    lifecycle: SequentialDctLifecycleMetadata,
) -> Result<DecodedDctBlocks, JpegError> {
    let grid = sequential_mcu_grid(plan);
    let (mcus_per_row, mcu_rows) = grid;
    let mut storage = allocate_dct_decode_storage(
        plan,
        mcus_per_row,
        mcu_rows,
        retain_quantized_blocks,
        retain_dequantized_blocks,
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
                        if retain_quantized_blocks && retain_dequantized_blocks {
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
                        } else if retain_quantized_blocks {
                            let block = &mut storage.quantized_blocks[plane_idx][block_idx];
                            decode_block_quantized_into(
                                &mut br,
                                dc_table,
                                ac_table,
                                &mut storage.prev_dc[plane_idx],
                                block,
                            )?;
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

impl McuWindow {
    /// Whether the window is non-empty and inside an MCU grid of
    /// `(mcus_per_row, mcu_rows)`.
    ///
    /// The ends are added with overflow checks: device windows arrive through
    /// a public type with public fields, and release builds do not check
    /// arithmetic, so a wrapped end would otherwise pass the bounds test.
    pub(crate) fn lies_within(self, (mcus_per_row, mcu_rows): (u32, u32)) -> bool {
        self.cols != 0
            && self.rows != 0
            && self
                .col
                .checked_add(self.cols)
                .is_some_and(|end| end <= mcus_per_row)
            && self
                .row
                .checked_add(self.rows)
                .is_some_and(|end| end <= mcu_rows)
    }

    /// Whether MCU `(mcu_x, mcu_y)` is inside a window that [`Self::lies_within`]
    /// accepted, so its ends cannot overflow.
    fn contains(self, mcu_x: u32, mcu_y: u32) -> bool {
        (self.col..self.col + self.cols).contains(&mcu_x)
            && (self.row..self.row + self.rows).contains(&mcu_y)
    }
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

/// Decode several quantized MCU windows in one entropy traversal.
pub(crate) fn decode_scan_quantized_dct_windows_into(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    windows: &[McuWindow],
    blocks: &mut [Vec<Vec<[i16; 64]>>],
) -> Result<(), JpegError> {
    match plan.huffman_primary_width {
        HuffmanPrimaryWidth::Narrow8 => decode_scan_quantized_dct_windows_with_entries::<
            NARROW_PRIMARY_ENTRIES,
        >(plan, scan_bytes, windows, blocks),
        HuffmanPrimaryWidth::Full12 => decode_scan_quantized_dct_windows_with_entries::<
            FULL_PRIMARY_ENTRIES,
        >(plan, scan_bytes, windows, blocks),
    }
}

fn decode_scan_quantized_dct_windows_with_entries<const N: usize>(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    windows: &[McuWindow],
    blocks: &mut [Vec<Vec<[i16; 64]>>],
) -> Result<(), JpegError> {
    if windows.len() != blocks.len() || windows.is_empty() {
        return Err(JpegError::InternalInvariant {
            reason: "quantized DCT windows disagree with their output owners",
        });
    }
    let (mcus_per_row, mcu_rows) = sequential_mcu_grid(plan);
    for (window, output) in windows.iter().zip(blocks.iter()) {
        if !window.lies_within((mcus_per_row, mcu_rows)) || output.len() != plan.sampling.len() {
            return Err(JpegError::InternalInvariant {
                reason:
                    "quantized DCT window lies outside the scan or has the wrong component count",
            });
        }
        for component in &plan.components {
            let expected = (window.cols as usize)
                .checked_mul(window.rows as usize)
                .and_then(|count| count.checked_mul(usize::from(component.h)))
                .and_then(|count| count.checked_mul(usize::from(component.v)))
                .ok_or(JpegError::InternalInvariant {
                    reason: "quantized DCT component block count overflowed",
                })?;
            if output[component.output_index].len() != expected {
                return Err(JpegError::InternalInvariant {
                    reason: "quantized DCT component grid disagrees with its window",
                });
            }
        }
    }

    let first_row = windows.iter().map(|window| window.row).min().unwrap_or(0);
    let last_row = windows
        .iter()
        .map(|window| window.row + window.rows)
        .max()
        .unwrap_or(first_row);
    // Whole rows are traversed so every window's DC predictors stay exact;
    // MCUs outside every window are entropy-skipped.
    let columns = McuColumns::whole_rows((mcus_per_row, mcu_rows));
    let wide_huffman = plan.prefers_wide_huffman(scan_bytes.len());
    let mut cursor = enter_scan_at_mcu::<N>(
        plan,
        scan_bytes,
        None,
        columns.mcu_position(0, first_row),
        wide_huffman,
    )?;

    for mcu_y in first_row..last_row {
        for mcu_x in 0..mcus_per_row {
            if cursor
                .restart
                .begin_mcu(&mut cursor.br, columns.mcu_position(mcu_x, mcu_y))?
            {
                cursor.prev_dc = [0; 4];
            }
            let selected = windows.iter().any(|window| window.contains(mcu_x, mcu_y));
            if selected {
                for component in &plan.components {
                    let plane = component.output_index;
                    let dc_table = plan.dc_table_with_entries::<N>(component)?;
                    let ac_table = plan.ac_table_with_entries::<N>(component)?;
                    for vy in 0..u32::from(component.v) {
                        for vx in 0..u32::from(component.h) {
                            let mut block = [0; 64];
                            decode_block_quantized_into(
                                &mut cursor.br,
                                dc_table,
                                ac_table,
                                &mut cursor.prev_dc[plane],
                                &mut block,
                            )?;
                            for (window, output) in windows.iter().zip(blocks.iter_mut()) {
                                if !window.contains(mcu_x, mcu_y) {
                                    continue;
                                }
                                let cols = window.cols * u32::from(component.h);
                                let x = (mcu_x - window.col) * u32::from(component.h) + vx;
                                let y = (mcu_y - window.row) * u32::from(component.v) + vy;
                                output[plane][(y * cols + x) as usize] = block;
                            }
                        }
                    }
                }
            } else {
                skip_mcu::<N>(plan, &mut cursor.br, &mut cursor.prev_dc, wide_huffman)?;
            }
            cursor.restart.end_mcu();
        }
    }
    Ok(())
}

fn decode_scan_dct_window_into_with_entries<const N: usize>(
    plan: &PreparedDecodePlan,
    scan_bytes: &[u8],
    window: McuWindow,
    checkpoint: Option<&DeviceCheckpoint>,
    blocks: &mut [[i16; 64]],
) -> Result<(), JpegError> {
    let (mcus_per_row, mcu_rows) = sequential_mcu_grid(plan);
    if !window.lies_within((mcus_per_row, mcu_rows)) {
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
