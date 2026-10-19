// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sequential DCT entropy coding with Huffman tables built from the image's
//! own statistics, for Huffman-optimized baseline and 12-bit extended JPEG.
//!
//! Statistics are gathered exactly as libjpeg's `htest_one_block` gathers
//! them, so tables match `jpegtran -optimize` for the same coefficients.

use alloc::vec::Vec;

use crate::adapter::JpegBaselineHuffmanTable;
use crate::baseline_entropy::{encode_block, magnitude, BitWriter};
use crate::encoded_output::JPEG_BASELINE_MAX_ENTROPY_BYTES;
use crate::encoder::JpegEncodeError;
use crate::optimal_huffman::{optimal_huffman_spec, HuffmanFrequencies, HuffmanSpec};

/// One component's quantized coefficient blocks, natural order, row-major.
///
/// Only blocks inside the component (`real_cols` by `real_rows`) are read.
/// An interleaved scan's MCUs can extend past them; those padding blocks are
/// coded as libjpeg codes them: zero AC, and the DC of the MCU's previous
/// block, which costs the fewest bits.
pub(super) struct DctComponent<'a> {
    pub(super) h: u8,
    pub(super) v: u8,
    /// Blocks per stored row, at least `real_cols`.
    pub(super) block_cols: u32,
    /// Blocks covering the component: `ceil(component width / 8)`.
    pub(super) real_cols: u32,
    /// Block rows covering the component: `ceil(component height / 8)`.
    pub(super) real_rows: u32,
    pub(super) blocks: &'a [[i16; 64]],
    /// Huffman table set: 0 for luma, 1 for chroma.
    pub(super) table: usize,
}

/// One sequential scan over every component of a frame.
pub(super) struct DctScan<'a> {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) components: &'a [DctComponent<'a>],
    /// Restart interval in MCUs.
    pub(super) restart_interval: Option<u16>,
    /// Largest DC difference category the frame precision allows (11 at 8
    /// bits, 15 at 12 bits); AC categories may be one less.
    pub(super) max_dc_category: u8,
}

/// The table sets a scan uses, with their optimal tables.
pub(super) struct DctTables {
    pub(super) dc: [Option<HuffmanSpec>; 2],
    pub(super) ac: [Option<HuffmanSpec>; 2],
}

/// Entropy-coded scan data and the tables it was coded with.
pub(super) struct DctEntropy {
    pub(super) bytes: Vec<u8>,
    pub(super) tables: DctTables,
}

/// Gather statistics, build optimal tables, and code `scan`.
///
/// # Errors
/// Returns an invariant error for a coefficient outside the frame's
/// precision, or an allocation error.
pub(super) fn encode_dct_scan_optimized(scan: &DctScan<'_>) -> Result<DctEntropy, JpegEncodeError> {
    let mut statistics = Statistics {
        dc: [HuffmanFrequencies::new(), HuffmanFrequencies::new()],
        ac: [HuffmanFrequencies::new(), HuffmanFrequencies::new()],
        used: [false; 2],
        max_dc_category: scan.max_dc_category,
    };
    for_each_block(scan, &mut statistics)?;
    let tables = statistics.tables()?;
    let dc_codes = tables
        .dc
        .each_ref()
        .map(|spec| spec.as_ref().map(HuffmanSpec::table));
    let ac_codes = tables
        .ac
        .each_ref()
        .map(|spec| spec.as_ref().map(HuffmanSpec::table));
    let mut coder = Coder {
        writer: BitWriter::try_with_capacity(
            statistics.entropy_size_estimate(&dc_codes, &ac_codes, scan),
            JPEG_BASELINE_MAX_ENTROPY_BYTES,
        )?,
        dc: dc_codes,
        ac: ac_codes,
        restarts: 0,
    };
    for_each_block(scan, &mut coder)?;
    Ok(DctEntropy {
        bytes: coder.writer.into_bytes()?,
        tables,
    })
}

trait BlockSink {
    /// A restart interval ends before the next MCU.
    fn restart(&mut self) -> Result<(), JpegEncodeError>;
    fn block(
        &mut self,
        table: usize,
        block: &[i16; 64],
        prev_dc: &mut i32,
    ) -> Result<(), JpegEncodeError>;
}

/// Visit every block in scan order: the row-major block grid of a
/// one-component scan, or interleaved MCUs otherwise. DC prediction restarts
/// at zero with each restart interval.
fn for_each_block<S: BlockSink>(scan: &DctScan<'_>, sink: &mut S) -> Result<(), JpegEncodeError> {
    let max_h = scan.components.iter().map(|c| c.h).max().unwrap_or(1);
    let max_v = scan.components.iter().map(|c| c.v).max().unwrap_or(1);
    let interleaved = scan.components.len() > 1;
    let (mcu_cols, mcu_rows) = if interleaved {
        (
            scan.width.div_ceil(8 * u32::from(max_h)),
            scan.height.div_ceil(8 * u32::from(max_v)),
        )
    } else {
        let component = scan
            .components
            .first()
            .ok_or(JpegEncodeError::InternalInvariant {
                reason: "DCT scan has no components",
            })?;
        (
            (scan.width * u32::from(component.h))
                .div_ceil(u32::from(max_h))
                .div_ceil(8),
            (scan.height * u32::from(component.v))
                .div_ceil(u32::from(max_v))
                .div_ceil(8),
        )
    };
    let interval = scan.restart_interval.map_or(0, u32::from);
    let mut prev_dc = [0_i32; 3];
    let mut mcu = 0_u32;
    for mcu_y in 0..mcu_rows {
        for mcu_x in 0..mcu_cols {
            if interval != 0 && mcu != 0 && mcu.is_multiple_of(interval) {
                sink.restart()?;
                prev_dc = [0; 3];
            }
            mcu += 1;
            let mut last_dc = 0_i16;
            for (component, prev) in scan.components.iter().zip(&mut prev_dc) {
                let (h, v) = if interleaved {
                    (u32::from(component.h), u32::from(component.v))
                } else {
                    (1, 1)
                };
                for block_y in 0..v {
                    for block_x in 0..h {
                        let (col, row) = (mcu_x * h + block_x, mcu_y * v + block_y);
                        if col >= component.real_cols || row >= component.real_rows {
                            let mut padding = [0_i16; 64];
                            padding[0] = last_dc;
                            sink.block(component.table, &padding, prev)?;
                            continue;
                        }
                        let index = row as usize * component.block_cols as usize + col as usize;
                        let block = component.blocks.get(index).ok_or(
                            JpegEncodeError::InternalInvariant {
                                reason: "DCT scan visits a block its component does not store",
                            },
                        )?;
                        last_dc = block[0];
                        sink.block(component.table, block, prev)?;
                    }
                }
            }
        }
    }
    Ok(())
}

struct Statistics {
    dc: [HuffmanFrequencies; 2],
    ac: [HuffmanFrequencies; 2],
    used: [bool; 2],
    max_dc_category: u8,
}

impl Statistics {
    fn tables(&self) -> Result<DctTables, JpegEncodeError> {
        let build = |frequencies: &HuffmanFrequencies, used: bool| {
            used.then(|| optimal_huffman_spec(frequencies)).transpose()
        };
        Ok(DctTables {
            dc: [
                build(&self.dc[0], self.used[0])?,
                build(&self.dc[1], self.used[1])?,
            ],
            ac: [
                build(&self.ac[0], self.used[0])?,
                build(&self.ac[1], self.used[1])?,
            ],
        })
    }

    /// Entropy bytes the scan will take, with a margin for byte stuffing and
    /// restart markers; the writer grows past it if needed.
    fn entropy_size_estimate(
        &self,
        dc: &[Option<JpegBaselineHuffmanTable>; 2],
        ac: &[Option<JpegBaselineHuffmanTable>; 2],
        scan: &DctScan<'_>,
    ) -> usize {
        let bits =
            |frequencies: &HuffmanFrequencies, codes: &JpegBaselineHuffmanTable, ac: bool| {
                (0..=u8::MAX)
                    .map(|symbol| {
                        let value_bits = if ac { symbol & 15 } else { symbol };
                        frequencies
                            .count(symbol)
                            .saturating_mul(u64::from(codes.lens[usize::from(symbol)] + value_bits))
                    })
                    .fold(0_u64, u64::saturating_add)
            };
        let mut total = 0_u64;
        for table in 0..2 {
            if let (Some(dc_codes), Some(ac_codes)) = (&dc[table], &ac[table]) {
                total = total
                    .saturating_add(bits(&self.dc[table], dc_codes, false))
                    .saturating_add(bits(&self.ac[table], ac_codes, true));
            }
        }
        let bytes = usize::try_from(total.div_ceil(8)).unwrap_or(usize::MAX);
        let restarts = scan.restart_interval.map_or(0, |interval| {
            (bytes / usize::from(interval)).saturating_add(1)
        });
        bytes
            .saturating_add(bytes / 256 + restarts.saturating_mul(3) + 16)
            .min(JPEG_BASELINE_MAX_ENTROPY_BYTES)
    }
}

impl BlockSink for Statistics {
    fn restart(&mut self) -> Result<(), JpegEncodeError> {
        Ok(())
    }

    fn block(
        &mut self,
        table: usize,
        block: &[i16; 64],
        prev_dc: &mut i32,
    ) -> Result<(), JpegEncodeError> {
        let out_of_range = JpegEncodeError::InternalInvariant {
            reason: "DCT coefficient exceeds the frame precision",
        };
        self.used[table] = true;
        let dc = i32::from(block[0]);
        let (dc_category, _) = magnitude(dc - *prev_dc);
        *prev_dc = dc;
        if dc_category > self.max_dc_category {
            return Err(out_of_range);
        }
        self.dc[table].add(dc_category);
        let mut run = 0_u8;
        for &natural in &j2k_codec_math::jpeg::ZIGZAG[1..] {
            let coefficient = block[usize::from(natural)];
            if coefficient == 0 {
                run += 1;
                continue;
            }
            while run > 15 {
                self.ac[table].add(0xF0);
                run -= 16;
            }
            let (size, _) = magnitude(i32::from(coefficient));
            // AC coefficients have one category fewer than DC differences.
            if size >= self.max_dc_category {
                return Err(out_of_range);
            }
            self.ac[table].add((run << 4) | size);
            run = 0;
        }
        if run > 0 {
            self.ac[table].add(0x00);
        }
        Ok(())
    }
}

struct Coder {
    writer: BitWriter,
    dc: [Option<JpegBaselineHuffmanTable>; 2],
    ac: [Option<JpegBaselineHuffmanTable>; 2],
    restarts: u8,
}

impl BlockSink for Coder {
    fn restart(&mut self) -> Result<(), JpegEncodeError> {
        self.writer.write_restart_marker(0xD0 + self.restarts)?;
        self.restarts = (self.restarts + 1) % 8;
        Ok(())
    }

    fn block(
        &mut self,
        table: usize,
        block: &[i16; 64],
        prev_dc: &mut i32,
    ) -> Result<(), JpegEncodeError> {
        let (Some(dc), Some(ac)) = (&self.dc[table], &self.ac[table]) else {
            return Err(JpegEncodeError::InternalInvariant {
                reason: "DCT block uses a table set with no statistics",
            });
        };
        encode_block(&block.map(i32::from), prev_dc, dc, ac, &mut self.writer)
    }
}
