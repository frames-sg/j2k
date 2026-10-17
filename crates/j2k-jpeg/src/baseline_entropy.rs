// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared baseline JPEG entropy-writing primitives.

use alloc::vec::Vec;

use crate::adapter::{JpegBaselineHuffmanTable, JPEG_BASELINE_ZIGZAG};
use crate::baseline_encode_contract::JpegEncodeError;
use crate::encoded_output::CappedBytes;
use crate::standard_huffman::{
    STD_CHROMA_AC_CANONICAL, STD_CHROMA_DC_CANONICAL, STD_LUMA_AC_CANONICAL, STD_LUMA_DC_CANONICAL,
};

pub(crate) struct BitWriter {
    bytes: CappedBytes,
    reservoir: BitReservoir,
}

#[derive(Clone, Copy)]
struct BitReservoir {
    current: u64,
    used: u8,
}

#[repr(align(64))]
/// Complete standard DC code-and-amplitude payloads indexed by difference residue.
pub(crate) struct PackedDcHuffmanTable([u32; PACKED_DC_ENTRY_COUNT]);

#[repr(align(64))]
/// Complete standard AC code-and-amplitude payloads in coefficient-major order.
pub(crate) struct PackedAcHuffmanTable([u32; PACKED_AC_ENTRY_COUNT]);

pub(crate) struct PackedHuffmanTables {
    pub(crate) dc: [&'static PackedDcHuffmanTable; 2],
    pub(crate) ac: [&'static PackedAcHuffmanTable; 2],
}

const PACKED_DC_INDEX_MASK: u16 = 0x0fff;
const PACKED_DC_UNUSED_INDEX: usize = 0x0800;
const PACKED_DC_ENTRY_COUNT: usize = PACKED_DC_INDEX_MASK as usize + 1;
const PACKED_AC_INDEX_MASK: u16 = 0x07ff;
// The validated AC domain excludes -1024, whose low eleven bits are 0x400.
const PACKED_AC_ZRL_INDEX: u16 = 0x0400;
const PACKED_AC_INDEX_COUNT: usize = PACKED_AC_INDEX_MASK as usize + 1;
const PACKED_AC_ENTRY_COUNT: usize = PACKED_AC_INDEX_COUNT * 16;
const PACKED_AC_SLOT_MASK: usize = PACKED_AC_ENTRY_COUNT - 1;
/// Coefficient-residue bits stored above a token's four-bit zero run.
#[cfg(target_arch = "aarch64")]
pub(crate) const PACKED_AC_TOKEN_COEFFICIENT_MASK: u16 = PACKED_AC_INDEX_MASK;
/// Exact packed-table slot reserved for the JPEG ZRL control symbol.
#[cfg(target_arch = "aarch64")]
pub(crate) const PACKED_AC_TOKEN_ZRL: u16 = PACKED_AC_ZRL_INDEX << 4;

// A trusted token block begins with at most 63 reservoir bits, adds at most
// 22 DC bits and 63 26-bit AC tokens, and therefore completes at most 26
// words. Worst-case stuffing commits 416 bytes. One ignored speculative word
// after the final flush raises the physical high-water mark to 424 bytes.
#[cfg(target_arch = "aarch64")]
const TRUSTED_TOKEN_BLOCK_SPARE_BYTES: usize = 512;

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::manual_bit_width,
    reason = "the complete twelve-bit DC residue table bounds indices, differences, magnitudes, and categories"
)]
const fn packed_dc_entries(canonical: &JpegBaselineHuffmanTable) -> PackedDcHuffmanTable {
    let mut entries = [0u32; PACKED_DC_ENTRY_COUNT];
    let mut index = 0usize;
    while index < PACKED_DC_ENTRY_COUNT {
        // The low twelve bits uniquely identify every category-eleven DC
        // difference. Residue 0x800 is -2048, outside that JPEG domain.
        if index != PACKED_DC_UNUSED_INDEX {
            let difference = if index < PACKED_DC_UNUSED_INDEX {
                index as i32
            } else {
                index as i32 - PACKED_DC_ENTRY_COUNT as i32
            };
            let magnitude = if difference < 0 {
                (-difference) as u32
            } else {
                difference as u32
            };
            let size = (u32::BITS - magnitude.leading_zeros()) as u8;
            assert!(size <= 11);
            let category_mask = (1u32 << size) - 1;
            let value_bits = if difference < 0 {
                category_mask - magnitude
            } else {
                magnitude
            };
            let code_len = canonical.lens[size as usize];
            assert!(code_len != 0);
            let total_len = code_len + size;
            assert!(total_len <= 22);
            let combined = (canonical.codes[size as usize] as u32) << size | value_bits;
            assert!(combined < (1u32 << total_len));
            entries[index] = (combined << 5) | total_len as u32;
        }
        index += 1;
    }
    PackedDcHuffmanTable(entries)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::large_stack_arrays,
    clippy::manual_bit_width,
    reason = "const evaluation builds the fixed packed AC table from eleven-bit coefficient residues and ten-bit magnitudes"
)]
const fn packed_ac_entries(canonical: &JpegBaselineHuffmanTable) -> PackedAcHuffmanTable {
    let mut entries = [0u32; PACKED_AC_ENTRY_COUNT];

    // Normal bitmap-selected coefficients never use zero or -1024. Reserve
    // their zero-run entries for the only standalone AC control symbols.
    let eob_len = canonical.lens[0];
    assert!(eob_len != 0);
    entries[0] = ((canonical.codes[0] as u32) << 5) | eob_len as u32;

    let zrl_len = canonical.lens[0xf0];
    assert!(zrl_len != 0);
    entries[PACKED_AC_ZRL_INDEX as usize * 16] =
        ((canonical.codes[0xf0] as u32) << 5) | zrl_len as u32;

    let mut run = 0u8;
    while run < 16 {
        let mut index = 1u16;
        while index <= PACKED_AC_INDEX_MASK {
            if index != PACKED_AC_ZRL_INDEX {
                let coefficient = if index < PACKED_AC_ZRL_INDEX {
                    index as i32
                } else {
                    index as i32 - 2048
                };
                let magnitude = if coefficient < 0 {
                    (-coefficient) as u32
                } else {
                    coefficient as u32
                };
                let size = (u32::BITS - magnitude.leading_zeros()) as u8;
                assert!(size >= 1 && size <= 10);
                let category_mask = (1u32 << size) - 1;
                let value_bits = if coefficient < 0 {
                    category_mask - magnitude
                } else {
                    magnitude
                };
                let symbol = ((run << 4) | size) as usize;
                let code_len = canonical.lens[symbol];
                assert!(code_len != 0);
                let total_len = code_len + size;
                assert!(total_len <= 26);
                let combined = ((canonical.codes[symbol] as u32) << size) | value_bits;
                assert!(combined < (1u32 << total_len));
                entries[index as usize * 16 + run as usize] = (combined << 5) | total_len as u32;
            }
            index += 1;
        }
        run += 1;
    }
    PackedAcHuffmanTable(entries)
}

static LUMA_DC_ENTRIES: PackedDcHuffmanTable = packed_dc_entries(&STD_LUMA_DC_CANONICAL);
static CHROMA_DC_ENTRIES: PackedDcHuffmanTable = packed_dc_entries(&STD_CHROMA_DC_CANONICAL);
static LUMA_AC_ENTRIES: PackedAcHuffmanTable = packed_ac_entries(&STD_LUMA_AC_CANONICAL);
static CHROMA_AC_ENTRIES: PackedAcHuffmanTable = packed_ac_entries(&STD_CHROMA_AC_CANONICAL);
static PACKED_STANDARD_HUFFMAN_TABLES: PackedHuffmanTables = PackedHuffmanTables {
    dc: [&LUMA_DC_ENTRIES, &CHROMA_DC_ENTRIES],
    ac: [&LUMA_AC_ENTRIES, &CHROMA_AC_ENTRIES],
};

pub(crate) fn packed_standard_huffman_tables(
    dc: [&JpegBaselineHuffmanTable; 2],
    ac: [&JpegBaselineHuffmanTable; 2],
) -> &'static PackedHuffmanTables {
    debug_assert_eq!(dc[0].codes, STD_LUMA_DC_CANONICAL.codes);
    debug_assert_eq!(dc[0].lens, STD_LUMA_DC_CANONICAL.lens);
    debug_assert_eq!(dc[1].codes, STD_CHROMA_DC_CANONICAL.codes);
    debug_assert_eq!(dc[1].lens, STD_CHROMA_DC_CANONICAL.lens);
    debug_assert_eq!(ac[0].codes, STD_LUMA_AC_CANONICAL.codes);
    debug_assert_eq!(ac[0].lens, STD_LUMA_AC_CANONICAL.lens);
    debug_assert_eq!(ac[1].codes, STD_CHROMA_AC_CANONICAL.codes);
    debug_assert_eq!(ac[1].lens, STD_CHROMA_AC_CANONICAL.lens);
    &PACKED_STANDARD_HUFFMAN_TABLES
}

impl BitWriter {
    pub(crate) fn try_with_max_bytes(max_bytes: usize) -> Result<Self, JpegEncodeError> {
        Ok(Self {
            bytes: CappedBytes::try_with_capacity(max_bytes, max_bytes)?,
            reservoir: BitReservoir {
                current: 0,
                used: 0,
            },
        })
    }

    fn flush_partial_bytes(&mut self) -> Result<(), JpegEncodeError> {
        // Eight data bytes plus, at worst, eight JPEG stuffing bytes. Reserve
        // output once per word rather than once for every emitted byte.
        let mut bytes = [0u8; 16];
        let mut len = 0;
        while self.reservoir.used >= 8 {
            self.reservoir.used -= 8;
            let byte = ((self.reservoir.current >> self.reservoir.used) & 0xff) as u8;
            bytes[len] = byte;
            len += 1;
            if byte == 0xff {
                bytes[len] = 0;
                len += 1;
            }
        }
        self.reservoir.current &= (1u64 << self.reservoir.used) - 1;
        self.bytes.extend_from_slice(&bytes[..len])
    }

    fn align_with_ones(&mut self) -> Result<(), JpegEncodeError> {
        self.flush_partial_bytes()?;
        if self.reservoir.used == 0 {
            return Ok(());
        }
        let remaining = 8 - self.reservoir.used;
        self.reservoir.current <<= remaining;
        self.reservoir.current |= (1u64 << remaining) - 1;
        self.push_byte(self.reservoir.current.to_le_bytes()[0])?;
        self.reservoir.current = 0;
        self.reservoir.used = 0;
        Ok(())
    }

    pub(crate) fn into_bytes(mut self) -> Result<Vec<u8>, JpegEncodeError> {
        self.align_with_ones()?;
        Ok(self.bytes.into_vec())
    }

    pub(crate) fn capacity_bytes(&self) -> usize {
        self.bytes.capacity()
    }

    pub(crate) fn write_restart_marker(&mut self, marker: u8) -> Result<(), JpegEncodeError> {
        self.align_with_ones()?;
        self.bytes.push(0xFF)?;
        self.bytes.push(marker)
    }

    fn push_byte(&mut self, byte: u8) -> Result<(), JpegEncodeError> {
        self.bytes.push(byte)?;
        if byte == 0xFF {
            self.bytes.push(0x00)?;
        }
        Ok(())
    }
}

#[expect(
    clippy::inline_always,
    reason = "the measured entropy hot path keeps reservoir appends in callers so codes and lengths remain scalar"
)]
impl BitReservoir {
    #[inline(always)]
    fn write_bits(
        self,
        bytes: &mut CappedBytes,
        code: u32,
        len: u8,
    ) -> Result<Self, JpegEncodeError> {
        self.write_bits_inner::<false>(bytes, u64::from(code), len)
    }

    #[inline(always)]
    fn write_bits_trusted(
        self,
        bytes: &mut CappedBytes,
        code: u32,
        len: u8,
    ) -> Result<Self, JpegEncodeError> {
        self.write_bits_inner::<true>(bytes, u64::from(code), len)
    }

    #[inline(always)]
    fn write_bits_trusted_wide(
        self,
        bytes: &mut CappedBytes,
        code: u64,
        len: u8,
    ) -> Result<Self, JpegEncodeError> {
        self.write_bits_inner::<true>(bytes, code, len)
    }

    #[inline(always)]
    fn write_bits_inner<const TRUSTED: bool>(
        mut self,
        bytes: &mut CappedBytes,
        mut code: u64,
        len: u8,
    ) -> Result<Self, JpegEncodeError> {
        debug_assert!(len < 64);
        let total = self.used + len;
        if TRUSTED {
            debug_assert!(code < (1u64 << len));
        } else {
            code &= (1u64 << len) - 1;
        }
        if total < 64 {
            self.current = (self.current << len) | code;
            self.used = total;
            return Ok(self);
        }

        // Fill the reservoir exactly, emit one whole word, then retain the
        // remaining low bits. This writes fewer, wider chunks than draining a
        // partially filled reservoir before every crossing code.
        let overflow = total - 64;
        let prefix_len = len - overflow;
        self.current = (self.current << prefix_len) | (code >> overflow);
        flush_word(bytes, self.current)?;
        // Trusted codes already fit in `len`. Keeping their high prefix here
        // avoids another variable mask: it stays above `used`, shifts out on
        // the next full-word emission, and is ignored by partial-byte output.
        self.current = if TRUSTED {
            code
        } else {
            code & ((1u64 << overflow) - 1)
        };
        self.used = overflow;
        Ok(self)
    }

    /// Append one trusted token payload through a block-wide spare cursor.
    ///
    /// # Safety
    ///
    /// `cursor` must point at the current end of the prevalidated token-block
    /// region, whose complete physical high-water bound is 424 bytes.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    unsafe fn write_bits_trusted_speculative(
        mut self,
        mut cursor: *mut u8,
        code: u64,
        len: u8,
    ) -> (Self, *mut u8) {
        const BYTE_ONES: u64 = 0x0101_0101_0101_0101;
        const BYTE_HIGHS: u64 = 0x8080_8080_8080_8080;

        debug_assert!(len < 64);
        debug_assert!(code < (1u64 << len));
        let total = self.used + len;
        debug_assert!(total < 128);
        let flush = total >= 64;

        // Wrapping shifts keep the prospective word defined on no-flush
        // calls. When `used == 0`, `prefix_len == 64`; flushing is then
        // impossible because `len < 64`, so that word is deliberately ignored.
        let overflow = total.wrapping_sub(64);
        let prefix_len = len.wrapping_sub(overflow);
        let word = self.current.wrapping_shl(u32::from(prefix_len))
            | code.wrapping_shr(u32::from(overflow));
        let appended = self.current.wrapping_shl(u32::from(len)) | code;

        // SAFETY: the enclosing 512-byte gate includes the 424-byte maximum
        // speculative high-water mark. A no-flush store stays beyond the
        // logical vector length and is overwritten by the next actual flush.
        unsafe { cursor.cast::<u64>().write_unaligned(word.to_be()) };

        let inverted = !word;
        let has_ff = inverted.wrapping_sub(BYTE_ONES) & !inverted & BYTE_HIGHS != 0;
        if flush & has_ff {
            // SAFETY: the same block-wide proof includes the maximum stuffed
            // expansion, and this overwrites the preliminary unstuffed word.
            cursor = unsafe { flush_stuffed_word_cursor(cursor, word) };
        } else {
            // SAFETY: advancing by eight occurs only for an actual unstuffed
            // flush and remains inside the prevalidated region.
            cursor = unsafe { cursor.add(usize::from(flush) * 8) };
        }

        self.current = if flush { code } else { appended };
        self.used = total & 63;
        (self, cursor)
    }
}

#[inline(always)]
fn flush_word(bytes: &mut CappedBytes, word: u64) -> Result<(), JpegEncodeError> {
    const BYTE_ONES: u64 = 0x0101_0101_0101_0101;
    const BYTE_HIGHS: u64 = 0x8080_8080_8080_8080;

    let big_endian = word.to_be_bytes();
    let inverted = !word;
    let has_ff = inverted.wrapping_sub(BYTE_ONES) & !inverted & BYTE_HIGHS != 0;
    if !has_ff {
        return bytes.extend_from_slice(&big_endian);
    }

    flush_stuffed_word(bytes, word)
}

// Most entropy words contain no marker prefix. Keep the exceptional byte
// stuffing loop out of the block encoder's repeatedly inlined word flushes.
#[cold]
#[inline(never)]
fn flush_stuffed_word(bytes: &mut CappedBytes, word: u64) -> Result<(), JpegEncodeError> {
    let big_endian = word.to_be_bytes();
    let mut stuffed = [0u8; 16];
    let mut len = 0;
    for byte in big_endian {
        stuffed[len] = byte;
        len += 1;
        if byte == 0xff {
            stuffed[len] = 0;
            len += 1;
        }
    }
    bytes.extend_from_slice(&stuffed[..len])
}

#[cfg(target_arch = "aarch64")]
#[cold]
#[inline(never)]
unsafe fn flush_stuffed_word_cursor(mut cursor: *mut u8, word: u64) -> *mut u8 {
    for byte in word.to_be_bytes() {
        // SAFETY: the token-block proof includes each data byte and its
        // possible stuffing byte within the fixed spare region.
        unsafe { cursor.write(byte) };
        // SAFETY: the same bound keeps every intermediate cursor in range.
        cursor = unsafe { cursor.add(1) };
        if byte == 0xff {
            // SAFETY: every 0xff data byte reserves one stuffing byte.
            unsafe { cursor.write(0) };
            // SAFETY: the worst-case sixteen-byte expansion is prevalidated.
            cursor = unsafe { cursor.add(1) };
        }
    }
    cursor
}

pub(crate) fn encode_block(
    coeffs: &[i32; 64],
    prev_dc: &mut i32,
    dc_table: &JpegBaselineHuffmanTable,
    ac_table: &JpegBaselineHuffmanTable,
    writer: &mut BitWriter,
) -> Result<(), JpegEncodeError> {
    // Keep the bit reservoir in local scalar state throughout a block. Only
    // the byte sink escapes to the capped output helper when a word is full.
    let mut reservoir = writer.reservoir;
    let bytes = &mut writer.bytes;
    let diff = coeffs[0] - *prev_dc;
    *prev_dc = coeffs[0];
    let (dc_size, dc_bits) = magnitude(diff);
    reservoir = write_huffman_value(dc_table, dc_size, dc_bits, dc_size, reservoir, bytes)?;

    let mut zero_run = 0u8;
    for k in 1..64 {
        let coeff = coeffs[JPEG_BASELINE_ZIGZAG[k] as usize];
        if coeff == 0 {
            zero_run = zero_run.saturating_add(1);
            continue;
        }
        while zero_run >= 16 {
            reservoir = write_huffman_symbol(ac_table, 0xF0, reservoir, bytes)?;
            zero_run -= 16;
        }
        let (size, bits) = magnitude(coeff);
        let symbol = (zero_run << 4) | size;
        reservoir = write_huffman_value(ac_table, symbol, bits, size, reservoir, bytes)?;
        zero_run = 0;
    }
    if zero_run > 0 {
        reservoir = write_huffman_symbol(ac_table, 0, reservoir, bytes)?;
    }
    writer.reservoir = reservoir;
    Ok(())
}

#[inline(never)]
pub(crate) fn encode_block_zigzag(
    dc: i16,
    ac_coefficients: &[i16; 64],
    nonzero_ac: u64,
    prev_dc: &mut i32,
    dc_table: &PackedDcHuffmanTable,
    ac_table: &PackedAcHuffmanTable,
    writer: &mut BitWriter,
) -> Result<(), JpegEncodeError> {
    if nonzero_ac == u64::MAX - 1 {
        encode_dense_block_zigzag(dc, ac_coefficients, prev_dc, dc_table, ac_table, writer)
    } else {
        encode_sparse_block_zigzag(
            dc,
            ac_coefficients,
            nonzero_ac,
            prev_dc,
            dc_table,
            ac_table,
            writer,
        )
    }
}

// An odd AC state stores `token_count << 1 | 1`; an even state remains the
// raw nonzero bitmap. Each token is the exact u16 slot of the existing packed
// AC table, including slots zero (EOB) and 0x4000 (ZRL).
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub(crate) fn encode_token_block_zigzag(
    dc: i16,
    ac_tokens: &[i16; 64],
    ac_state: u64,
    prev_dc: &mut i32,
    dc_table: &PackedDcHuffmanTable,
    ac_table: &PackedAcHuffmanTable,
    writer: &mut BitWriter,
) -> Result<(), JpegEncodeError> {
    if writer
        .bytes
        .has_existing_spare::<TRUSTED_TOKEN_BLOCK_SPARE_BYTES>()
    {
        // SAFETY: the immediately preceding gate proves the complete physical
        // and logical spare region. The direct body creates and commits its
        // cursor without any intervening mutation of the byte sink.
        unsafe {
            encode_token_block_zigzag_direct(
                dc, ac_tokens, ac_state, prev_dc, dc_table, ac_table, writer,
            );
        }
        Ok(())
    } else {
        encode_token_block_zigzag_checked(
            dc, ac_tokens, ac_state, prev_dc, dc_table, ac_table, writer,
        )
    }
}

/// Encode one token block through a fixed, prevalidated output cursor.
///
/// # Safety
///
/// `writer.bytes` must have at least `TRUSTED_TOKEN_BLOCK_SPARE_BYTES` of
/// unchanged physical capacity and logical output budget.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
unsafe fn encode_token_block_zigzag_direct(
    dc: i16,
    ac_tokens: &[i16; 64],
    ac_state: u64,
    prev_dc: &mut i32,
    dc_table: &PackedDcHuffmanTable,
    ac_table: &PackedAcHuffmanTable,
    writer: &mut BitWriter,
) {
    // SAFETY: upheld by the caller's immediately preceding gate.
    let mut spare = unsafe {
        writer
            .bytes
            .existing_spare_cursor_unchecked::<TRUSTED_TOKEN_BLOCK_SPARE_BYTES>()
    };
    let mut cursor = spare.start();

    debug_assert_eq!(ac_state & 1, 1);
    let token_count = usize::try_from(ac_state >> 1)
        .expect("a baseline token count fits the platform index width");
    debug_assert!((1..=63).contains(&token_count));

    let mut reservoir = writer.reservoir;
    let dc = i32::from(dc);
    let diff = dc - *prev_dc;
    *prev_dc = dc;
    let (dc_code, dc_len) = packed_dc_difference(dc_table, diff);
    // SAFETY: every append shares this function's fixed block-wide cursor
    // contract and the packed DC entry is a trusted payload.
    (reservoir, cursor) =
        unsafe { reservoir.write_bits_trusted_speculative(cursor, u64::from(dc_code), dc_len) };

    let (quads, remainder) = ac_tokens[..token_count].as_chunks::<4>();
    for quad in quads {
        let (first_code, first_len) = packed_ac_token_entry(ac_table, quad[0]);
        let (second_code, second_len) = packed_ac_token_entry(ac_table, quad[1]);
        let (third_code, third_len) = packed_ac_token_entry(ac_table, quad[2]);
        let (fourth_code, fourth_len) = packed_ac_token_entry(ac_table, quad[3]);
        debug_assert!(first_len + second_len <= 52);
        debug_assert!(third_len + fourth_len <= 52);

        let pair01_code = (u64::from(first_code) << second_len) | u64::from(second_code);
        let pair01_len = first_len + second_len;
        let pair23_code = (u64::from(third_code) << fourth_len) | u64::from(fourth_code);
        let pair23_len = third_len + fourth_len;
        let combined_len = pair01_len + pair23_len;
        if combined_len < 64 {
            // SAFETY: the combined payload remains below 64 bits and shares
            // the fixed block-wide cursor contract.
            (reservoir, cursor) = unsafe {
                reservoir.write_bits_trusted_speculative(
                    cursor,
                    (pair01_code << pair23_len) | pair23_code,
                    combined_len,
                )
            };
        } else {
            // SAFETY: both trusted pairs share the fixed cursor contract.
            (reservoir, cursor) = unsafe {
                reservoir.write_bits_trusted_speculative(cursor, pair01_code, pair01_len)
            };
            // SAFETY: the second trusted pair shares the same fixed cursor
            // region, advanced only by the preceding bounded append.
            (reservoir, cursor) = unsafe {
                reservoir.write_bits_trusted_speculative(cursor, pair23_code, pair23_len)
            };
        }
    }

    let (pairs, remainder) = remainder.as_chunks::<2>();
    for pair in pairs {
        let (first_code, first_len) = packed_ac_token_entry(ac_table, pair[0]);
        let (second_code, second_len) = packed_ac_token_entry(ac_table, pair[1]);
        debug_assert!(first_len + second_len <= 52);
        // SAFETY: the trusted pair shares the fixed cursor contract.
        (reservoir, cursor) = unsafe {
            reservoir.write_bits_trusted_speculative(
                cursor,
                (u64::from(first_code) << second_len) | u64::from(second_code),
                first_len + second_len,
            )
        };
    }
    if let [last] = remainder {
        let (code, len) = packed_ac_token_entry(ac_table, *last);
        // SAFETY: the final trusted token shares the fixed cursor contract.
        (reservoir, cursor) =
            unsafe { reservoir.write_bits_trusted_speculative(cursor, u64::from(code), len) };
    }

    // SAFETY: actual flushes initialized every byte before `cursor` in one
    // contiguous prefix. Ignored speculative stores remain after `cursor`.
    unsafe { spare.commit(cursor) };
    writer.reservoir = reservoir;
}

#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn encode_token_block_zigzag_checked(
    dc: i16,
    ac_tokens: &[i16; 64],
    ac_state: u64,
    prev_dc: &mut i32,
    dc_table: &PackedDcHuffmanTable,
    ac_table: &PackedAcHuffmanTable,
    writer: &mut BitWriter,
) -> Result<(), JpegEncodeError> {
    debug_assert_eq!(ac_state & 1, 1);
    let token_count = usize::try_from(ac_state >> 1)
        .expect("a baseline token count fits the platform index width");
    debug_assert!((1..=63).contains(&token_count));

    let mut reservoir = writer.reservoir;
    let bytes = &mut writer.bytes;
    let dc = i32::from(dc);
    let diff = dc - *prev_dc;
    *prev_dc = dc;
    reservoir = write_packed_dc_difference(dc_table, diff, reservoir, bytes)?;

    let (quads, remainder) = ac_tokens[..token_count].as_chunks::<4>();
    for quad in quads {
        let (first_code, first_len) = packed_ac_token_entry(ac_table, quad[0]);
        let (second_code, second_len) = packed_ac_token_entry(ac_table, quad[1]);
        let (third_code, third_len) = packed_ac_token_entry(ac_table, quad[2]);
        let (fourth_code, fourth_len) = packed_ac_token_entry(ac_table, quad[3]);
        debug_assert!(first_len + second_len <= 52);
        debug_assert!(third_len + fourth_len <= 52);

        let pair01_code = (u64::from(first_code) << second_len) | u64::from(second_code);
        let pair01_len = first_len + second_len;
        let pair23_code = (u64::from(third_code) << fourth_len) | u64::from(fourth_code);
        let pair23_len = third_len + fourth_len;
        let combined_len = pair01_len + pair23_len;
        if combined_len < 64 {
            reservoir = reservoir.write_bits_trusted_wide(
                bytes,
                (pair01_code << pair23_len) | pair23_code,
                combined_len,
            )?;
        } else {
            reservoir = reservoir.write_bits_trusted_wide(bytes, pair01_code, pair01_len)?;
            reservoir = reservoir.write_bits_trusted_wide(bytes, pair23_code, pair23_len)?;
        }
    }

    let (pairs, remainder) = remainder.as_chunks::<2>();
    for pair in pairs {
        let (first_code, first_len) = packed_ac_token_entry(ac_table, pair[0]);
        let (second_code, second_len) = packed_ac_token_entry(ac_table, pair[1]);
        debug_assert!(first_len + second_len <= 52);
        reservoir = reservoir.write_bits_trusted_wide(
            bytes,
            (u64::from(first_code) << second_len) | u64::from(second_code),
            first_len + second_len,
        )?;
    }
    if let [last] = remainder {
        let (code, len) = packed_ac_token_entry(ac_table, *last);
        reservoir = reservoir.write_bits_trusted(bytes, code, len)?;
    }
    writer.reservoir = reservoir;
    Ok(())
}

#[inline(never)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "leading-zero counts from a u64 bitmap are at most 64 and fit u8"
)]
fn encode_sparse_block_zigzag(
    dc: i16,
    ac_coefficients: &[i16; 64],
    mut nonzero_ac: u64,
    prev_dc: &mut i32,
    dc_table: &PackedDcHuffmanTable,
    ac_table: &PackedAcHuffmanTable,
    writer: &mut BitWriter,
) -> Result<(), JpegEncodeError> {
    // This path is fed only by the built-in 8-bit baseline transform and its
    // standard Huffman tables. A level-shifted sample is in [-128, 127], so an
    // unquantized AC coefficient is bounded by
    // 127.5 * L1(basis_u) * L1(basis_v) / 4 <= 1020. Quantization cannot grow
    // that value, hence AC categories are at most 10. DC is in [-1024, 1016],
    // making the largest predictor difference 2040 (category 11). The
    // standard tables contain every resulting DC and run/AC symbol, allowing
    // one packed lookup without the missing-symbol checks retained by the
    // general transcode path.
    debug_assert_eq!(nonzero_ac & 1, 0);
    let mut reservoir = writer.reservoir;
    let bytes = &mut writer.bytes;

    let dc = i32::from(dc);
    let diff = dc - *prev_dc;
    *prev_dc = dc;
    reservoir = write_packed_dc_difference(dc_table, diff, reservoir, bytes)?;

    let mut coefficient_index = 1usize;
    while nonzero_ac != 0 {
        let mut zero_run = nonzero_ac.leading_zeros() as u8;
        coefficient_index += usize::from(zero_run);
        nonzero_ac <<= zero_run;
        while zero_run >= 16 {
            reservoir = write_packed_ac_control(ac_table, PACKED_AC_ZRL_INDEX, reservoir, bytes)?;
            zero_run -= 16;
        }

        let coefficient = ac_coefficients[coefficient_index];
        let (code, len) = packed_ac_huffman_value(ac_table, zero_run, coefficient);

        let remaining = nonzero_ac << 1;
        let after_current = coefficient_index + 1;
        if remaining != 0 {
            let next_zero_run = remaining.leading_zeros() as u8;
            if next_zero_run < 16 {
                let next_index = after_current + usize::from(next_zero_run);
                let next_coefficient = ac_coefficients[next_index];
                let (next_code, next_len) =
                    packed_ac_huffman_value(ac_table, next_zero_run, next_coefficient);
                debug_assert!(len + next_len <= 52);
                reservoir = reservoir.write_bits_trusted_wide(
                    bytes,
                    (u64::from(code) << next_len) | u64::from(next_code),
                    len + next_len,
                )?;
                coefficient_index = next_index + 1;
                nonzero_ac = (remaining << next_zero_run) << 1;
                continue;
            }
        }

        reservoir = reservoir.write_bits_trusted(bytes, code, len)?;
        coefficient_index = after_current;
        nonzero_ac = remaining;
    }
    if coefficient_index != 64 {
        reservoir = write_packed_ac_control(ac_table, 0, reservoir, bytes)?;
    }
    writer.reservoir = reservoir;
    Ok(())
}

// A completely occupied AC block has no zero runs or EOB. Keep its fixed
// pairs in a separate full-block function so the sparse function retains its
// original bitmap path.
#[inline(never)]
fn encode_dense_block_zigzag(
    dc: i16,
    ac_coefficients: &[i16; 64],
    prev_dc: &mut i32,
    dc_table: &PackedDcHuffmanTable,
    ac_table: &PackedAcHuffmanTable,
    writer: &mut BitWriter,
) -> Result<(), JpegEncodeError> {
    let mut reservoir = writer.reservoir;
    let bytes = &mut writer.bytes;

    let dc = i32::from(dc);
    let diff = dc - *prev_dc;
    *prev_dc = dc;
    reservoir = write_packed_dc_difference(dc_table, diff, reservoir, bytes)?;

    for pair in ac_coefficients[1..63].as_chunks::<2>().0 {
        let (first_code, first_len) = packed_ac_huffman_value(ac_table, 0, pair[0]);
        let (second_code, second_len) = packed_ac_huffman_value(ac_table, 0, pair[1]);
        reservoir = reservoir.write_bits_trusted_wide(
            bytes,
            (u64::from(first_code) << second_len) | u64::from(second_code),
            first_len + second_len,
        )?;
    }
    let (last_code, last_len) = packed_ac_huffman_value(ac_table, 0, ac_coefficients[63]);
    writer.reservoir = reservoir.write_bits_trusted(bytes, last_code, last_len)?;
    Ok(())
}

#[expect(
    clippy::inline_always,
    reason = "the measured entropy path folds fixed control-table lookup into its callers"
)]
#[inline(always)]
fn write_packed_ac_control(
    table: &PackedAcHuffmanTable,
    index: u16,
    reservoir: BitReservoir,
    bytes: &mut CappedBytes,
) -> Result<BitReservoir, JpegEncodeError> {
    debug_assert!(index == 0 || index == PACKED_AC_ZRL_INDEX);
    let (code, len) = packed_ac_entry(table, 0, index);
    reservoir.write_bits_trusted(bytes, code, len)
}

#[expect(
    clippy::inline_always,
    reason = "the measured entropy path folds packed DC lookup and reservoir append into its callers"
)]
#[inline(always)]
fn write_packed_dc_difference(
    table: &PackedDcHuffmanTable,
    difference: i32,
    reservoir: BitReservoir,
    bytes: &mut CappedBytes,
) -> Result<BitReservoir, JpegEncodeError> {
    let (combined, total_len) = packed_dc_difference(table, difference);
    reservoir.write_bits_trusted(bytes, combined, total_len)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::inline_always,
    reason = "the packed entry stores a payload length of at most 22 in its low five bits and lookup stays in the caller"
)]
#[inline(always)]
fn packed_dc_difference(table: &PackedDcHuffmanTable, difference: i32) -> (u32, u8) {
    debug_assert!((-2047..=2047).contains(&difference));
    // Reinterpret the signed difference before masking so negative values map
    // to their unique twelve-bit two's-complement residue without a hot check.
    let residue = u32::from_ne_bytes(difference.to_ne_bytes()) & u32::from(PACKED_DC_INDEX_MASK);
    let index = usize::try_from(residue).expect("a twelve-bit DC residue fits usize");
    let entry = table.0[index];
    let total_len = entry as u8 & 31;
    debug_assert_ne!(total_len, 0);
    debug_assert!(total_len <= 22);
    (entry >> 5, total_len)
}

#[expect(
    clippy::inline_always,
    reason = "coefficient residue lookup remains in the measured sparse and dense entropy loops"
)]
#[inline(always)]
fn packed_ac_huffman_value(table: &PackedAcHuffmanTable, run: u8, coefficient: i16) -> (u32, u8) {
    debug_assert_ne!(coefficient, 0);
    debug_assert!((-1023..=1023).contains(&coefficient));
    // Unsigned conversion is modulo 2^16; masking leaves the coefficient's
    // unique eleven-bit residue over the asserted nonzero domain.
    let index = u16::from_ne_bytes(coefficient.to_ne_bytes()) & PACKED_AC_INDEX_MASK;
    packed_ac_entry(table, run, index)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::inline_always,
    reason = "the packed entry stores a payload length of at most 26 in its low five bits and lookup stays in the caller"
)]
#[inline(always)]
fn packed_ac_entry(table: &PackedAcHuffmanTable, run: u8, index: u16) -> (u32, u8) {
    debug_assert!(run < 16);
    debug_assert!(index <= PACKED_AC_INDEX_MASK);
    let slot = ((usize::from(index) << 4) | usize::from(run)) & PACKED_AC_SLOT_MASK;
    let entry = table.0[slot];
    let total_len = entry as u8 & 31;
    debug_assert_ne!(total_len, 0);
    debug_assert!(total_len <= 26);
    (entry >> 5, total_len)
}

#[cfg(target_arch = "aarch64")]
#[expect(
    clippy::cast_possible_truncation,
    clippy::inline_always,
    reason = "trusted token slots contain payload lengths of at most 26 and lookup stays in the token loop"
)]
#[inline(always)]
fn packed_ac_token_entry(table: &PackedAcHuffmanTable, token: i16) -> (u32, u8) {
    debug_assert!(token >= 0);
    let raw_slot = u16::from_ne_bytes(token.to_ne_bytes());
    debug_assert!(usize::from(raw_slot) < PACKED_AC_ENTRY_COUNT);
    let slot = usize::from(raw_slot) & PACKED_AC_SLOT_MASK;
    let entry = table.0[slot];
    let total_len = entry as u8 & 31;
    debug_assert_ne!(total_len, 0);
    debug_assert!(total_len <= 26);
    (entry >> 5, total_len)
}

#[expect(
    clippy::inline_always,
    reason = "checked symbol lookup and reservoir append remain fused in the baseline entropy loop"
)]
#[inline(always)]
fn write_huffman_symbol(
    table: &JpegBaselineHuffmanTable,
    symbol: u8,
    reservoir: BitReservoir,
    bytes: &mut CappedBytes,
) -> Result<BitReservoir, JpegEncodeError> {
    let len = table.lens[symbol as usize];
    if len == 0 {
        return Err(JpegEncodeError::MissingHuffmanCode { symbol });
    }
    reservoir.write_bits(bytes, u32::from(table.codes[symbol as usize]), len)
}

#[expect(
    clippy::inline_always,
    reason = "checked value lookup and reservoir append remain fused in the baseline entropy loop"
)]
#[inline(always)]
fn write_huffman_value(
    table: &JpegBaselineHuffmanTable,
    symbol: u8,
    value_bits: u32,
    value_len: u8,
    reservoir: BitReservoir,
    bytes: &mut CappedBytes,
) -> Result<BitReservoir, JpegEncodeError> {
    let code_len = table.lens[symbol as usize];
    if code_len == 0 {
        return Err(JpegEncodeError::MissingHuffmanCode { symbol });
    }
    let total_len = code_len + value_len;
    if total_len <= 32 {
        let combined = (u32::from(table.codes[symbol as usize]) << value_len) | value_bits;
        reservoir.write_bits(bytes, combined, total_len)
    } else {
        let reservoir =
            reservoir.write_bits(bytes, u32::from(table.codes[symbol as usize]), code_len)?;
        reservoir.write_bits(bytes, value_bits, value_len)
    }
}

#[expect(
    clippy::manual_bit_width,
    reason = "the explicit leading-zero expression preserves the established JPEG magnitude-category calculation"
)]
pub(crate) fn magnitude(value: i32) -> (u8, u32) {
    if value == 0 {
        return (0, 0);
    }
    let magnitude = value.unsigned_abs();
    let size = u8::try_from(u32::BITS - magnitude.leading_zeros())
        .expect("an i32 magnitude category always fits in u8");
    let category_mask = u32::MAX >> (u32::BITS - u32::from(size));
    let bits = if value >= 0 {
        magnitude
    } else {
        category_mask - magnitude
    };
    (size, bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::baseline_encode_tables;
    use crate::encoder::{JpegBackend, JpegEncodeOptions, JpegSubsampling};

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "one matrix keeps checked, packed, and tokenized reservoir boundary cases together"
    )]
    fn paired_standard_blocks_match_checked_stream_across_wide_writes() {
        let tables = baseline_encode_tables(JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr444,
            restart_interval: None,
            backend: JpegBackend::Cpu,
        })
        .unwrap();
        let packed_tables = packed_standard_huffman_tables(
            [&tables.huff_dc_luma, &tables.huff_dc_chroma],
            [&tables.huff_ac_luma, &tables.huff_ac_chroma],
        );

        let mut blocks = Vec::new();
        let mut wide = [0i32; 64];
        wide[0] = 0;
        wide[JPEG_BASELINE_ZIGZAG[16] as usize] = 1020;
        wide[JPEG_BASELINE_ZIGZAG[32] as usize] = -1020;
        wide[JPEG_BASELINE_ZIGZAG[63] as usize] = 1020;
        blocks.push(wide);

        let mut alternating = [0i32; 64];
        alternating[0] = -1024;
        for zigzag in 1..64 {
            let magnitude = [1, 2, 7, 31, 127, 511, 1020][zigzag % 7];
            alternating[JPEG_BASELINE_ZIGZAG[zigzag] as usize] = if zigzag & 1 == 0 {
                magnitude
            } else {
                -magnitude
            };
        }
        blocks.push(alternating);

        let mut long_runs = [0i32; 64];
        long_runs[0] = 1016;
        long_runs[JPEG_BASELINE_ZIGZAG[17] as usize] = -1020;
        long_runs[JPEG_BASELINE_ZIGZAG[34] as usize] = 511;
        long_runs[JPEG_BASELINE_ZIGZAG[60] as usize] = -1;
        blocks.push(long_runs);
        blocks.push([0i32; 64]);

        let mut final_ac = [0i32; 64];
        final_ac[0] = -1024;
        final_ac[JPEG_BASELINE_ZIGZAG[63] as usize] = -1020;
        blocks.push(final_ac);

        let mut category_ten_edges = [0i32; 64];
        category_ten_edges[JPEG_BASELINE_ZIGZAG[1] as usize] = 1023;
        category_ten_edges[JPEG_BASELINE_ZIGZAG[2] as usize] = -1023;
        blocks.push(category_ten_edges);

        let dc_tables = [&tables.huff_dc_luma, &tables.huff_dc_chroma];
        let ac_tables = [&tables.huff_ac_luma, &tables.huff_ac_chroma];
        for table_index in 0..2 {
            for all_ones_seed in [false, true] {
                let mut checked = BitWriter::try_with_max_bytes(65_536).unwrap();
                let mut paired = BitWriter::try_with_max_bytes(65_536).unwrap();
                if all_ones_seed {
                    seed_all_ones_reservoir_at_61_bits(&mut checked);
                    seed_all_ones_reservoir_at_61_bits(&mut paired);
                } else {
                    seed_reservoir_at_61_bits(&mut checked);
                    seed_reservoir_at_61_bits(&mut paired);
                }
                let mut checked_dc = 0;
                let mut paired_dc = 0;
                for block in &blocks {
                    encode_block(
                        block,
                        &mut checked_dc,
                        dc_tables[table_index],
                        ac_tables[table_index],
                        &mut checked,
                    )
                    .unwrap();
                    let (dc, ac_coefficients, nonzero_ac) = raw_block(block);
                    encode_block_zigzag(
                        dc,
                        &ac_coefficients,
                        nonzero_ac,
                        &mut paired_dc,
                        packed_tables.dc[table_index],
                        packed_tables.ac[table_index],
                        &mut paired,
                    )
                    .unwrap();
                }

                // Exercise every category-eleven DC difference through the
                // checked and packed writers in one continued entropy stream.
                // Setting both predictors directly keeps the encoded DC at
                // zero while selecting the exact difference under test.
                let dc_only = [0i32; 64];
                let (dc, ac_coefficients, nonzero_ac) = raw_block(&dc_only);
                for difference in -2047..=2047 {
                    checked_dc = -difference;
                    paired_dc = -difference;
                    encode_block(
                        &dc_only,
                        &mut checked_dc,
                        dc_tables[table_index],
                        ac_tables[table_index],
                        &mut checked,
                    )
                    .unwrap();
                    encode_block_zigzag(
                        dc,
                        &ac_coefficients,
                        nonzero_ac,
                        &mut paired_dc,
                        packed_tables.dc[table_index],
                        packed_tables.ac[table_index],
                        &mut paired,
                    )
                    .unwrap();
                }

                assert_eq!(checked_dc, paired_dc);
                assert_eq!(checked.into_bytes().unwrap(), paired.into_bytes().unwrap());
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            use crate::encoder::prepare_entropy_ac_for_test;

            let mut probe = [0i16; 64];
            if prepare_entropy_ac_for_test(&mut probe, 0).is_none() {
                return;
            }

            let check_token_block = |block: &[i32; 64], capacity: usize| {
                let (dc, raw_coefficients, nonzero_ac) = raw_block(block);
                let mut token_coefficients = raw_coefficients;
                let token_state = prepare_entropy_ac_for_test(&mut token_coefficients, nonzero_ac)
                    .expect("NEON availability was established by the probe");
                assert_eq!(token_state & 1, 1, "eligible block must use tokens");

                for table_index in 0..2 {
                    let mut checked = BitWriter::try_with_max_bytes(capacity).unwrap();
                    let mut tokenized = BitWriter::try_with_max_bytes(capacity).unwrap();
                    seed_all_ones_reservoir_at_61_bits(&mut checked);
                    seed_all_ones_reservoir_at_61_bits(&mut tokenized);
                    assert_eq!(
                        tokenized
                            .bytes
                            .has_existing_spare::<TRUSTED_TOKEN_BLOCK_SPARE_BYTES>(),
                        capacity >= TRUSTED_TOKEN_BLOCK_SPARE_BYTES
                    );
                    let mut checked_dc = 0;
                    let mut tokenized_dc = 0;
                    encode_block(
                        block,
                        &mut checked_dc,
                        dc_tables[table_index],
                        ac_tables[table_index],
                        &mut checked,
                    )
                    .unwrap();
                    encode_token_block_zigzag(
                        dc,
                        &token_coefficients,
                        token_state,
                        &mut tokenized_dc,
                        packed_tables.dc[table_index],
                        packed_tables.ac[table_index],
                        &mut tokenized,
                    )
                    .unwrap();
                    assert_eq!(checked_dc, tokenized_dc);
                    assert_eq!(
                        checked.into_bytes().unwrap(),
                        tokenized.into_bytes().unwrap()
                    );
                }
            };

            // The 61-bit seed and size-zero DC code leave 63 used bits. The
            // first four category-ten tokens form two 52-bit pairs whose
            // 104-bit prefix cannot fit one append, so they take the two-pair
            // fallback. Later magnitude-one quads exercise the combined path.
            let mut guarded_four_tokens = [0i32; 64];
            for zigzag in 1..=40 {
                let magnitude = if zigzag <= 4 { 1020 } else { 1 };
                guarded_four_tokens[JPEG_BASELINE_ZIGZAG[zigzag] as usize] = if zigzag & 1 == 0 {
                    magnitude
                } else {
                    -magnitude
                };
            }
            check_token_block(&guarded_four_tokens, 1024);
            check_token_block(&guarded_four_tokens, 512);
            // One byte below the fixed window takes the unchanged checked
            // fallback while preserving the same predictor and entropy bytes.
            check_token_block(&guarded_four_tokens, 511);

            // Every valid row mask is exercised in every row. Row zero has
            // only seven AC lanes because its first lane is the separate DC.
            for affected_row in 0..8 {
                let mask_limit = if affected_row == 0 { 128 } else { 256 };
                for row_mask in 0..mask_limit {
                    let guard_zero = if affected_row == 7 { 1 } else { 63 };
                    let mut block = [0i32; 64];
                    block[0] = -127;
                    for zigzag in 1..64 {
                        let row = zigzag / 8;
                        let lane = zigzag % 8;
                        let selected = row != affected_row || row_mask & (0x80usize >> lane) != 0;
                        if selected && zigzag != guard_zero {
                            let magnitude = [1, 2, 7, 31, 127, 511, 1023][zigzag % 7];
                            block[JPEG_BASELINE_ZIGZAG[zigzag] as usize] = if zigzag & 1 == 0 {
                                magnitude
                            } else {
                                -magnitude
                            };
                        }
                    }
                    check_token_block(&block, 1024);
                }
            }

            // A complete sixteen-zero prefix followed by 47 nonzeros forces
            // ZRL. Stopping one coefficient earlier additionally forces EOB.
            for final_nonzero in [63usize, 62] {
                let mut block = [0i32; 64];
                block[0] = 127;
                for zigzag in 17..=final_nonzero {
                    block[JPEG_BASELINE_ZIGZAG[zigzag] as usize] =
                        if zigzag & 1 == 0 { 31 } else { -31 };
                }
                check_token_block(&block, 1024);
            }

            // Exactly 24 final nonzeros after 39 or 38 leading zeros require
            // two ZRLs. Ending at 62 additionally preserves a trailing EOB.
            for final_nonzero in [63usize, 62] {
                let mut block = [0i32; 64];
                block[0] = -127;
                for zigzag in final_nonzero - 23..=final_nonzero {
                    block[JPEG_BASELINE_ZIGZAG[zigzag] as usize] =
                        if zigzag & 1 == 0 { 1 } else { -1 };
                }
                check_token_block(&block, 1024);
            }
        }
    }

    fn seed_reservoir_at_61_bits(writer: &mut BitWriter) {
        let reservoir = writer
            .reservoir
            .write_bits(&mut writer.bytes, 0x01ab_cdef, 29)
            .unwrap();
        writer.reservoir = reservoir
            .write_bits(&mut writer.bytes, 0x89ab_cdef, 32)
            .unwrap();
        assert_eq!(writer.reservoir.used, 61);
    }

    fn seed_all_ones_reservoir_at_61_bits(writer: &mut BitWriter) {
        let reservoir = writer
            .reservoir
            .write_bits(&mut writer.bytes, u32::MAX, 32)
            .unwrap();
        writer.reservoir = reservoir
            .write_bits(&mut writer.bytes, 0x1fff_ffff, 29)
            .unwrap();
        assert_eq!(writer.reservoir.current, (1u64 << 61) - 1);
        assert_eq!(writer.reservoir.used, 61);
    }

    fn raw_block(block: &[i32; 64]) -> (i16, [i16; 64], u64) {
        let dc = i16::try_from(block[0]).unwrap();
        let mut ac_coefficients = [0i16; 64];
        let mut nonzero_ac = 0u64;
        for zigzag in 1..64 {
            let coefficient = i16::try_from(block[JPEG_BASELINE_ZIGZAG[zigzag] as usize]).unwrap();
            if coefficient != 0 {
                ac_coefficients[zigzag] = coefficient;
                nonzero_ac |= 1u64 << (64 - zigzag);
            }
        }
        (dc, ac_coefficients, nonzero_ac)
    }
}
