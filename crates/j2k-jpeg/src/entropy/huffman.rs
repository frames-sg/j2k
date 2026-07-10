// SPDX-License-Identifier: MIT OR Apache-2.0

//! Huffman decoder. Two layers:
//!
//! 1. **Fast lookup** — a 4096-entry table indexed by the next 12 bits of the
//!    stream. Each entry carries `(symbol, bit_length)` or `(_, 0)` if the
//!    code is longer than 12 bits.
//! 2. **Slow path** — per-length arrays (`min_code`, `max_code`, `val_offset`)
//!    implementing the T.81 §F.2.2.3 decode procedure for codes up to 16 bits.
//!
//! Built once from [`crate::parse::tables::RawHuffmanTable`]; read many times
//! by [`crate::entropy::block::decode_block`].

use crate::error::{HuffmanFailure, JpegError};
use crate::internal::bit_reader::BitReader;
use crate::parse::tables::{HuffmanValues, RawHuffmanTable};
use alloc::boxed::Box;

/// Number of fast-lookup entries. One per possible 12-bit peek value.
const FAST_BITS: u8 = 12;
const FAST_ENTRIES: usize = 1 << FAST_BITS;

const AC_FAST_KIND_SHIFT: u32 = 28;
pub(crate) const AC_FAST_KIND_MASK: u32 = 0xF << AC_FAST_KIND_SHIFT;
pub(crate) const AC_FAST_VALUE: u32 = 1 << AC_FAST_KIND_SHIFT;
pub(crate) const AC_FAST_EOB: u32 = 2 << AC_FAST_KIND_SHIFT;
pub(crate) const AC_FAST_ZRL: u32 = 3 << AC_FAST_KIND_SHIFT;
const AC_FAST_LEN_MASK: u32 = 0x0F;
const AC_FAST_RUN_MASK: u32 = 0xF0;
const AC_FAST_VALUE_SHIFT: u32 = 8;

const DC_FAST_LEN_MASK: u32 = 0x0F;
const DC_FAST_VALUE_SHIFT: u32 = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HuffmanTable {
    /// Fast path: `fast[peek8] = (symbol, bit_length)`. `bit_length == 0`
    /// means "code longer than 8 bits — use slow path".
    fast: [(u8, u8); FAST_ENTRIES],
    /// Slow path, indexed by code length `l` ∈ `1..=16`:
    /// - `min_code[l]`: smallest `l`-bit code; `i32::MAX` if no `l`-bit code.
    /// - `max_code[l]`: largest `l`-bit code; `-1` if no `l`-bit code.
    /// - `val_offset[l]`: index into `values` where `l`-bit symbols begin,
    ///   pre-adjusted by subtracting `min_code[l]` so
    ///   `symbol = values[code + val_offset[l]]`.
    min_code: [i32; 17],
    max_code: [i32; 17],
    val_offset: [i32; 17],
    values: HuffmanValues,
    fast_dc: Box<[u32; FAST_ENTRIES]>,
    fast_ac: Box<[u32; FAST_ENTRIES]>,
}

pub(crate) type CanonicalHuffmanDerivation = j2k_codec_math::jpeg::CanonicalHuffmanDerivation;

pub(crate) fn derive_canonical_huffman(
    raw: &RawHuffmanTable,
) -> Result<CanonicalHuffmanDerivation, JpegError> {
    j2k_codec_math::jpeg::derive_canonical_huffman(&raw.bits, raw.values.len()).map_err(|_| {
        JpegError::HuffmanDecode {
            mcu: 0,
            reason: HuffmanFailure::CodeOverflow,
        }
    })
}

impl HuffmanTable {
    /// Build the decode table from a raw `(bits, values)` pair parsed out of
    /// a DHT segment. Per T.81 §C.2 and Annex C.
    ///
    /// # Errors
    /// - `HuffmanDecode { CodeOverflow }` if `bits` is oversubscribed (Kraft
    ///   inequality violated — the table claims more codes of some length than
    ///   there is remaining code space).
    #[expect(
        clippy::cast_possible_truncation,
        reason = "canonical JPEG Huffman code lengths and symbol positions are bounded to 16 bits and 256 entries"
    )]
    pub(crate) fn from_raw(raw: &RawHuffmanTable) -> Result<Self, JpegError> {
        let canonical = derive_canonical_huffman(raw)?;
        let mut fast = [(0u8, 0u8); FAST_ENTRIES];
        let mut fast_dc = Box::new([0u32; FAST_ENTRIES]);
        let mut fast_ac = Box::new([0u32; FAST_ENTRIES]);

        let mut k = 0;
        for len_minus_1 in 0..FAST_BITS as usize {
            let len = (len_minus_1 + 1) as u8;
            let count = raw.bits[len_minus_1] as usize;
            for _ in 0..count {
                let c = canonical.huffcode[k];
                let fast_index_base = (c as usize) << (FAST_BITS - len);
                let fast_count = 1 << (FAST_BITS - len);
                for j in 0..fast_count {
                    fast[fast_index_base + j] = (raw.values.as_slice()[k], len);
                }
                k += 1;
            }
        }

        for (idx, &(sym, len)) in fast.iter().enumerate() {
            if len == 0 {
                continue;
            }
            if sym <= 15 {
                let total_len = len + sym;
                if total_len <= FAST_BITS {
                    let diff = if sym == 0 {
                        0
                    } else {
                        let mag_shift = FAST_BITS - total_len;
                        let mag_mask = (1u16 << sym) - 1;
                        let mag_bits = ((idx as u16) >> mag_shift) & mag_mask;
                        huff_extend(i32::from(mag_bits), sym)
                    };
                    if (i32::from(i16::MIN)..=i32::from(i16::MAX)).contains(&diff) {
                        fast_dc[idx] = pack_dc_value(total_len, diff as i16);
                    }
                }
            }

            let run = usize::from((sym >> 4) & 0x0F);
            let ssss = sym & 0x0F;
            if ssss == 0 {
                fast_ac[idx] = match run {
                    0 => pack_ac_eob(len),
                    15 => pack_ac_zrl(len),
                    _ => 0,
                };
                continue;
            }
            let total_len = len + ssss;
            if total_len > FAST_BITS {
                continue;
            }

            let mag_shift = FAST_BITS - total_len;
            let mag_mask = (1u16 << ssss) - 1;
            let mag_bits = ((idx as u16) >> mag_shift) & mag_mask;
            let value = huff_extend(i32::from(mag_bits), ssss);
            if !(i32::from(i16::MIN)..=i32::from(i16::MAX)).contains(&value) {
                continue;
            }
            fast_ac[idx] = pack_ac_value(total_len, run as u8, value as i16);
        }

        Ok(Self {
            fast,
            min_code: canonical.min_code,
            max_code: canonical.max_code,
            val_offset: canonical.val_offset,
            values: raw.values.clone(),
            fast_dc,
            fast_ac,
        })
    }

    /// Decode one symbol from the bit reader. Common case (code ≤ 8 bits) is
    /// a single array lookup; long codes fall through to a per-length scan.
    ///
    /// # Errors
    /// - `HuffmanDecode { TableExhausted }` if the stream ran out of bits.
    /// - `HuffmanDecode { CodeOverflow }` if no 1..=16-bit code matches.
    #[expect(
        clippy::inline_always,
        reason = "measured Huffman lookup hot path requires cross-helper inlining"
    )]
    #[inline(always)]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        reason = "the Huffman slow path converts only validated 16-bit codes and non-negative table offsets"
    )]
    pub(crate) fn decode(&self, br: &mut BitReader<'_>) -> Result<u8, JpegError> {
        br.ensure_bits_padded(FAST_BITS)?;
        let peek = br.peek_bits(FAST_BITS) as usize;
        let (sym, len) = self.fast[peek];
        if len != 0 {
            br.consume_bits(len);
            return Ok(sym);
        }
        // Slow path: compare against `max_code[l]` for l = 13..=16.
        br.ensure_bits_padded(16)?;
        let code16 = br.peek_bits(16) as i32;
        for len in (FAST_BITS as usize + 1)..=16 {
            let l = len as u8;
            let c = code16 >> (16 - l);
            if c <= self.max_code[len] {
                br.consume_bits(l);
                let idx = (c + self.val_offset[len]) as usize;
                return self.values.get(idx).ok_or(JpegError::HuffmanDecode {
                    mcu: 0,
                    reason: HuffmanFailure::InvalidSymbol,
                });
            }
        }
        Err(JpegError::HuffmanDecode {
            mcu: 0,
            reason: HuffmanFailure::CodeOverflow,
        })
    }

    #[expect(
        clippy::inline_always,
        reason = "measured Huffman lookup hot path requires cross-helper inlining"
    )]
    #[inline(always)]
    #[expect(
        clippy::cast_possible_wrap,
        reason = "packed DC values intentionally reinterpret a validated 16-bit two's-complement field"
    )]
    pub(crate) fn decode_fast_dc(&self, br: &mut BitReader<'_>) -> Result<i32, JpegError> {
        br.ensure_bits_padded(FAST_BITS)?;
        let peek = br.peek_bits(FAST_BITS) as usize;
        let packed = self.fast_dc[peek];
        if packed != 0 {
            br.consume_bits((packed & DC_FAST_LEN_MASK) as u8);
            return Ok(i32::from(
                ((packed >> DC_FAST_VALUE_SHIFT) & 0xFFFF) as u16 as i16,
            ));
        }

        let ssss = self.decode(br)?;
        if ssss > 15 {
            return Err(JpegError::HuffmanDecode {
                mcu: 0,
                reason: HuffmanFailure::InvalidSymbol,
            });
        }
        br.receive_extend(ssss)
    }

    #[expect(
        clippy::inline_always,
        reason = "measured Huffman lookup hot path requires cross-helper inlining"
    )]
    #[inline(always)]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "JPEG receive-extend values are bounded to the signed 16-bit packed AC field"
    )]
    pub(crate) fn decode_fast_ac(&self, br: &mut BitReader<'_>) -> Result<u32, JpegError> {
        br.ensure_bits_padded(FAST_BITS)?;
        let peek = br.peek_bits(FAST_BITS) as usize;
        let packed = self.fast_ac[peek];
        if packed != 0 {
            br.consume_bits((packed & AC_FAST_LEN_MASK) as u8);
            return Ok(packed);
        }

        let (sym, len) = self.fast[peek];
        let sym = if len != 0 {
            br.consume_bits(len);
            sym
        } else {
            self.decode(br)?
        };

        let run = sym >> 4;
        let ssss = sym & 0x0F;
        if ssss == 0 {
            return Ok(if run == 15 {
                pack_ac_zrl(0)
            } else {
                pack_ac_eob(0)
            });
        }

        let value = br.receive_extend(ssss)?;
        Ok(pack_ac_value(0, run, value as i16))
    }

    #[expect(
        clippy::inline_always,
        reason = "measured Huffman lookup hot path requires cross-helper inlining"
    )]
    #[inline(always)]
    pub(crate) fn skip_fast_ac(&self, br: &mut BitReader<'_>) -> Result<u32, JpegError> {
        br.ensure_bits_padded(FAST_BITS)?;
        let peek = br.peek_bits(FAST_BITS) as usize;
        let packed = self.fast_ac[peek];
        if packed != 0 {
            br.consume_bits((packed & AC_FAST_LEN_MASK) as u8);
            return Ok(packed);
        }

        let (sym, len) = self.fast[peek];
        let sym = if len != 0 {
            br.consume_bits(len);
            sym
        } else {
            self.decode(br)?
        };

        let run = sym >> 4;
        let ssss = sym & 0x0F;
        if ssss == 0 {
            return Ok(if run == 15 {
                pack_ac_zrl(0)
            } else {
                pack_ac_eob(0)
            });
        }

        br.ensure_bits(ssss)?;
        br.consume_bits(ssss);
        Ok(pack_ac_value(0, run, 0))
    }
}

#[expect(
    clippy::inline_always,
    reason = "measured Huffman lookup hot path requires cross-helper inlining"
)]
#[inline(always)]
pub(crate) fn ac_decoded_run(packed: u32) -> usize {
    ((packed & AC_FAST_RUN_MASK) >> 4) as usize
}

#[expect(
    clippy::inline_always,
    reason = "measured Huffman lookup hot path requires cross-helper inlining"
)]
#[inline(always)]
#[expect(
    clippy::cast_possible_wrap,
    reason = "packed AC values intentionally reinterpret a 16-bit two's-complement field"
)]
pub(crate) fn ac_decoded_value(packed: u32) -> i32 {
    i32::from(((packed >> AC_FAST_VALUE_SHIFT) & 0xFFFF) as u16 as i16)
}

#[inline]
#[expect(
    clippy::cast_sign_loss,
    reason = "signed AC coefficients are intentionally stored as a 16-bit two's-complement bit field"
)]
fn pack_ac_value(total_len: u8, run: u8, value: i16) -> u32 {
    AC_FAST_VALUE
        | ((u32::from(value as u16)) << AC_FAST_VALUE_SHIFT)
        | (u32::from(run) << 4)
        | u32::from(total_len)
}

#[inline]
fn pack_ac_eob(total_len: u8) -> u32 {
    AC_FAST_EOB | u32::from(total_len)
}

#[inline]
fn pack_ac_zrl(total_len: u8) -> u32 {
    AC_FAST_ZRL | (15 << 4) | u32::from(total_len)
}

#[inline]
#[expect(
    clippy::cast_sign_loss,
    reason = "signed DC coefficients are intentionally stored as a 16-bit two's-complement bit field"
)]
fn pack_dc_value(total_len: u8, value: i16) -> u32 {
    (u32::from(value as u16) << DC_FAST_VALUE_SHIFT) | u32::from(total_len)
}

fn huff_extend(v: i32, ssss: u8) -> i32 {
    let threshold = 1i32 << (ssss - 1);
    if v < threshold {
        v + ((-1i32) << ssss) + 1
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Standard JPEG luminance DC table from Annex K.3 — well-known fixture.
    /// `bits[0..16]` counts per length; `values` lists the symbols in order.
    fn luma_dc_raw() -> RawHuffmanTable {
        RawHuffmanTable {
            bits: [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        }
    }

    #[test]
    fn builds_fast_table_from_standard_luma_dc() {
        let table = HuffmanTable::from_raw(&luma_dc_raw()).unwrap();
        let (sym, len) = table.fast[0b0000_0000_0000];
        assert_eq!((sym, len), (0, 2));
        let (sym, len) = table.fast[0b0011_1111_1111];
        assert_eq!((sym, len), (0, 2));
        let (sym, len) = table.fast[0b0100_0000_0000];
        assert_eq!((sym, len), (1, 3));
    }

    #[test]
    fn widened_fast_table_covers_9_bit_luma_dc_code() {
        let table = HuffmanTable::from_raw(&luma_dc_raw()).unwrap();
        let idx = 0b1_1111_1110usize << usize::from(FAST_BITS - 9);
        let (sym, len) = table.fast.get(idx).copied().unwrap_or((0, 0));
        assert_eq!((sym, len), (11, 9));
    }

    #[test]
    fn rejects_oversubscribed_code_table() {
        let raw = RawHuffmanTable {
            bits: [1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0, 1, 2, 3, 4]),
        };
        let err = HuffmanTable::from_raw(&raw).unwrap_err();
        assert!(matches!(
            err,
            JpegError::HuffmanDecode {
                reason: HuffmanFailure::CodeOverflow,
                ..
            }
        ));
    }

    #[test]
    fn handles_empty_table_without_panic() {
        let raw = RawHuffmanTable {
            bits: [0; 16],
            values: HuffmanValues::default(),
        };
        let table = HuffmanTable::from_raw(&raw).unwrap();
        assert!(table.fast.iter().all(|&(_, len)| len == 0));
    }

    /// Exercises every standard JPEG luma DC code — Annex K.3.
    fn luma_dc_code_cases() -> &'static [(u32, u8, u8)] {
        &[
            (0b00, 2, 0),
            (0b010, 3, 1),
            (0b011, 3, 2),
            (0b100, 3, 3),
            (0b101, 3, 4),
            (0b110, 3, 5),
            (0b1110, 4, 6),
            (0b1_1110, 5, 7),
            (0b11_1110, 6, 8),
            (0b111_1110, 7, 9),
            (0b1111_1110, 8, 10),
            (0b1_1111_1110, 9, 11),
        ]
    }

    #[test]
    fn decodes_all_standard_luma_dc_codes() {
        let table = HuffmanTable::from_raw(&luma_dc_raw()).unwrap();
        for &(code, len, expected) in luma_dc_code_cases() {
            let shift = 32 - len;
            let aligned = code << shift;
            let bytes = aligned.to_be_bytes();
            let mut br = BitReader::new(&bytes);
            let sym = table.decode(&mut br).unwrap();
            assert_eq!(sym, expected, "code={code:b} len={len}");
        }
    }

    #[test]
    fn fast_dc_decodes_symbol_and_magnitude_in_one_lookup() {
        let table = HuffmanTable::from_raw(&luma_dc_raw()).unwrap();
        // Standard luma DC code `011` => category 2, followed by magnitude
        // bits `10` => diff +2. The fast DC path should consume all 5 bits.
        let bytes = [0b0111_0000u8, 0, 0, 0, 0, 0, 0, 0];
        let mut br = BitReader::new(&bytes);

        let diff = table.decode_fast_dc(&mut br).unwrap();

        assert_eq!(diff, 2);
        assert_eq!(br.snapshot().bits, 51);
        assert_eq!(br.peek_bits(3), 0);
    }

    #[test]
    fn decodes_single_bit_table_before_marker_padding() {
        let raw = RawHuffmanTable {
            bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            values: HuffmanValues::from_slice(&[0]),
        };
        let table = HuffmanTable::from_raw(&raw).unwrap();
        let mut br = BitReader::new(&[0x7f, 0xff, 0xc4]);

        let symbol = table.decode(&mut br).unwrap();

        assert_eq!(symbol, 0);
    }

    #[test]
    fn decodes_9_plus_bit_codes_via_slow_path() {
        let table = HuffmanTable::from_raw(&luma_dc_raw()).unwrap();
        // Code `111111110` (9 bits) → symbol 11. A literal 0xFF in a JPEG
        // entropy stream must be byte-stuffed as `FF 00` (T.81 §F.1.2.3) so
        // the BitReader does not mistake it for a marker prefix.
        let bytes = [0xFFu8, 0x00, 0b0100_0000];
        let mut br = BitReader::new(&bytes);
        let sym = table.decode(&mut br).unwrap();
        assert_eq!(sym, 11);
    }

    #[test]
    fn reports_huffman_failure_on_truncated_bit_stream() {
        let table = HuffmanTable::from_raw(&luma_dc_raw()).unwrap();
        let bytes = [];
        let mut br = BitReader::new(&bytes);
        let err = table.decode(&mut br).unwrap_err();
        assert!(matches!(
            err,
            JpegError::HuffmanDecode {
                reason: HuffmanFailure::TableExhausted,
                ..
            }
        ));
    }
}
