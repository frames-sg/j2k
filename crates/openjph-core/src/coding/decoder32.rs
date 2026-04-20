//! HTJ2K 32-bit block decoder — port of `ojph_block_decoder32.cpp`.
//!
//! Decodes one codeblock, processing the cleanup, significance-propagation,
//! and magnitude-refinement passes.

#![allow(clippy::identity_op, clippy::erasing_op)]

use crate::arch::{count_leading_zeros, population_count};
use crate::types::ojph_max;

use super::common::decoder_tables;

/// Emit a decoder warning (analogous to C++ OJPH_WARN).
macro_rules! ojph_warn {
    ($($arg:tt)*) => {
        eprintln!("OJPH WARN: {}", format!($($arg)*));
    };
}

// ============================================================================
// MEL decoder
// ============================================================================

/// MEL exponents (Table 2 in ITU T.814).
const MEL_EXP: [i32; 13] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5];

/// MEL state structure for reading and decoding the MEL bitstream.
///
/// A number of events is decoded from the MEL bitstream ahead of time and
/// stored in `runs` / `num_runs`.  Each run represents the number of zero
/// events before a one event.
struct DecMelSt<'a> {
    data: &'a [u8],
    pos: usize,
    tmp: u64,
    bits: i32,
    size: i32,
    unstuff: bool,
    k: i32,
    num_runs: i32,
    runs: u64,
}

impl<'a> DecMelSt<'a> {
    /// Read and unstuff 4 bytes from the MEL bitstream.
    fn mel_read(&mut self) {
        if self.bits > 32 {
            return;
        }

        let val: u32;
        if self.size > 4 {
            val = u32::from(self.data[self.pos])
                | (u32::from(self.data[self.pos + 1]) << 8)
                | (u32::from(self.data[self.pos + 2]) << 16)
                | (u32::from(self.data[self.pos + 3]) << 24);
            self.pos += 4;
            self.size -= 4;
        } else if self.size > 0 {
            let mut v = 0xFFFF_FFFFu32;
            let mut i = 0u32;
            while self.size > 1 {
                let b = u32::from(self.data[self.pos]);
                self.pos += 1;
                let m = !(0xFFu32 << i);
                v = (v & m) | (b << i);
                self.size -= 1;
                i += 8;
            }
            // last byte: overlap MEL and VLC
            let mut b = u32::from(self.data[self.pos]);
            self.pos += 1;
            b |= 0xF;
            let m = !(0xFFu32 << i);
            v = (v & m) | (b << i);
            self.size -= 1;
            val = v;
        } else {
            val = 0xFFFF_FFFFu32;
        }

        // Unstuff 4 bytes and accumulate in tmp
        let mut bits = (32 - self.unstuff as i32) as u32;

        let mut t = val & 0xFF;
        let mut unstuff = (val & 0xFF) == 0xFF;
        bits -= unstuff as u32;
        t <<= 8 - unstuff as u32;

        t |= (val >> 8) & 0xFF;
        unstuff = ((val >> 8) & 0xFF) == 0xFF;
        bits -= unstuff as u32;
        t <<= 8 - unstuff as u32;

        t |= (val >> 16) & 0xFF;
        unstuff = ((val >> 16) & 0xFF) == 0xFF;
        bits -= unstuff as u32;
        t <<= 8 - unstuff as u32;

        t |= (val >> 24) & 0xFF;
        self.unstuff = ((val >> 24) & 0xFF) == 0xFF;

        self.tmp |= (t as u64) << (64 - bits as i32 - self.bits);
        self.bits += bits as i32;
    }

    /// Decode unstuffed MEL segment bits to runs.
    fn mel_decode(&mut self) {
        if self.bits < 6 {
            self.mel_read();
        }

        while self.bits >= 6 && self.num_runs < 8 {
            let eval = MEL_EXP[self.k as usize];
            let run;
            if self.tmp & (1u64 << 63) != 0 {
                // 1 found
                run = (((1i32 << eval) - 1) << 1) as u64; // non-terminating
                self.k = (self.k + 1).min(12);
                self.tmp <<= 1;
                self.bits -= 1;
            } else {
                // 0 found
                let r = ((self.tmp >> (63 - eval)) as i32) & ((1 << eval) - 1);
                self.k = (self.k - 1).max(0);
                self.tmp <<= (eval + 1) as u32;
                self.bits -= eval + 1;
                run = ((r << 1) + 1) as u64; // terminating
            }
            let shift = self.num_runs * 7;
            self.runs &= !(0x3Fu64 << shift);
            self.runs |= run << shift;
            self.num_runs += 1;
        }
    }

    /// Initialise MEL structure and align read position.
    fn mel_init(data: &'a [u8], lcup: usize, scup: usize) -> Self {
        let start = lcup - scup;
        let mut mel = DecMelSt {
            data,
            pos: start,
            tmp: 0,
            bits: 0,
            size: (scup as i32) - 1,
            unstuff: false,
            k: 0,
            num_runs: 0,
            runs: 0,
        };

        // Align to 4-byte boundary (read 1..4 bytes)
        let num = 4 - (start & 0x3);
        for _ in 0..num {
            debug_assert!(!mel.unstuff || mel.data[mel.pos] <= 0x8F);
            let d: u64 = if mel.size > 0 {
                mel.data[mel.pos] as u64
            } else {
                0xFF
            };
            let d = if mel.size == 1 { d | 0xF } else { d };
            if mel.size > 0 {
                mel.pos += 1;
            }
            mel.size -= (mel.size > 0) as i32;
            let d_bits = 8 - mel.unstuff as i32;
            mel.tmp = (mel.tmp << d_bits as u32) | d;
            mel.bits += d_bits;
            mel.unstuff = (d & 0xFF) == 0xFF;
        }
        mel.tmp <<= (64 - mel.bits) as u32;
        mel
    }

    /// Retrieve one run from the MEL decoder.
    fn mel_get_run(&mut self) -> i32 {
        if self.num_runs == 0 {
            self.mel_decode();
        }
        let t = (self.runs & 0x7F) as i32;
        self.runs >>= 7;
        self.num_runs -= 1;
        t
    }
}

// ============================================================================
// Backward-growing reader (VLC / MRP)
// ============================================================================

/// State structure for reading and unstuffing a segment that grows backward
/// (VLC and MRP).
struct RevStruct<'a> {
    data: &'a [u8],
    /// Current read position (decrements).
    pos: usize,
    tmp: u64,
    bits: u32,
    size: i32,
    unstuff: bool,
}

impl<'a> RevStruct<'a> {
    // -- VLC helpers ---------------------------------------------------------

    /// Read and unstuff 4 bytes backward for VLC.
    fn rev_read(&mut self) {
        if self.bits > 32 {
            return;
        }
        let mut val: u32 = 0;
        if self.size > 3 {
            // read 4 bytes at [pos-3..=pos] in big-endian order
            val = (u32::from(self.data[self.pos]) << 24)
                | (u32::from(self.data[self.pos - 1]) << 16)
                | (u32::from(self.data[self.pos - 2]) << 8)
                | u32::from(self.data[self.pos - 3]);
            self.pos = self.pos.saturating_sub(4);
            self.size -= 4;
        } else if self.size > 0 {
            let mut i: i32 = 24;
            while self.size > 0 {
                let v = u32::from(self.data[self.pos]);
                if self.pos > 0 {
                    self.pos -= 1;
                }
                val |= v << i as u32;
                self.size -= 1;
                i -= 8;
            }
        }

        // Accumulate with unstuffing
        let mut tmp = val >> 24;
        let mut bits: u32 = 8 - if self.unstuff && ((val >> 24) & 0x7F) == 0x7F {
            1
        } else {
            0
        };
        let mut unstuff = (val >> 24) > 0x8F;

        tmp |= ((val >> 16) & 0xFF) << bits;
        bits += 8 - if unstuff && ((val >> 16) & 0x7F) == 0x7F {
            1
        } else {
            0
        };
        unstuff = ((val >> 16) & 0xFF) > 0x8F;

        tmp |= ((val >> 8) & 0xFF) << bits;
        bits += 8 - if unstuff && ((val >> 8) & 0x7F) == 0x7F {
            1
        } else {
            0
        };
        unstuff = ((val >> 8) & 0xFF) > 0x8F;

        tmp |= (val & 0xFF) << bits;
        bits += 8 - if unstuff && (val & 0x7F) == 0x7F {
            1
        } else {
            0
        };
        unstuff = (val & 0xFF) > 0x8F;

        self.tmp |= (tmp as u64) << self.bits;
        self.bits += bits;
        self.unstuff = unstuff;
    }

    /// Initialise VLC reader, skip first 12 bits (interface locator).
    fn rev_init(data: &'a [u8], lcup: usize, scup: usize) -> Self {
        let start_pos = lcup - 2;
        let mut vlc = RevStruct {
            data,
            pos: start_pos,
            tmp: 0,
            bits: 0,
            size: (scup as i32) - 2,
            unstuff: false,
        };

        // Read first (half) byte
        let d = u32::from(vlc.data[vlc.pos]);
        if vlc.pos > 0 {
            vlc.pos -= 1;
        }
        vlc.tmp = (d >> 4) as u64;
        vlc.bits = 4 - if (vlc.tmp & 7) == 7 { 1 } else { 0 };
        vlc.unstuff = (d | 0xF) > 0x8F;

        // Align to 4-byte boundary
        let num = 1 + (vlc.pos & 0x3);
        let tnum = if (num as i32) < vlc.size {
            num
        } else {
            vlc.size as usize
        };
        for _ in 0..tnum {
            let d = u64::from(vlc.data[vlc.pos]);
            if vlc.pos > 0 {
                vlc.pos -= 1;
            }
            let d_bits = 8 - if vlc.unstuff && (d & 0x7F) == 0x7F {
                1u32
            } else {
                0
            };
            vlc.tmp |= d << vlc.bits;
            vlc.bits += d_bits;
            vlc.unstuff = d > 0x8F;
        }
        vlc.size -= tnum as i32;
        vlc.rev_read();
        vlc
    }

    /// Fetch 32 bits from VLC (ensure ≥33 bits available).
    fn rev_fetch(&mut self) -> u32 {
        if self.bits < 32 {
            self.rev_read();
            if self.bits < 32 {
                self.rev_read();
            }
        }
        self.tmp as u32
    }

    /// Consume `num_bits` from VLC and return remaining bottom 32 bits.
    fn rev_advance(&mut self, num_bits: u32) -> u32 {
        debug_assert!(num_bits <= self.bits);
        self.tmp >>= num_bits;
        self.bits -= num_bits;
        self.tmp as u32
    }

    // -- MRP helpers ---------------------------------------------------------

    /// Read and unstuff 4 bytes backward for MRP (fills zeros when exhausted).
    fn rev_read_mrp(&mut self) {
        if self.bits > 32 {
            return;
        }
        let mut val: u32 = 0;
        if self.size > 3 {
            val = (u32::from(self.data[self.pos]) << 24)
                | (u32::from(self.data[self.pos - 1]) << 16)
                | (u32::from(self.data[self.pos - 2]) << 8)
                | u32::from(self.data[self.pos - 3]);
            self.pos = self.pos.saturating_sub(4);
            self.size -= 4;
        } else if self.size > 0 {
            let mut i: i32 = 24;
            while self.size > 0 {
                let v = u32::from(self.data[self.pos]);
                if self.pos > 0 {
                    self.pos -= 1;
                }
                val |= v << i as u32;
                self.size -= 1;
                i -= 8;
            }
        }

        let mut tmp = val >> 24;
        let mut bits: u32 = 8 - if self.unstuff && ((val >> 24) & 0x7F) == 0x7F {
            1
        } else {
            0
        };
        let mut unstuff = (val >> 24) > 0x8F;

        tmp |= ((val >> 16) & 0xFF) << bits;
        bits += 8 - if unstuff && ((val >> 16) & 0x7F) == 0x7F {
            1
        } else {
            0
        };
        unstuff = ((val >> 16) & 0xFF) > 0x8F;

        tmp |= ((val >> 8) & 0xFF) << bits;
        bits += 8 - if unstuff && ((val >> 8) & 0x7F) == 0x7F {
            1
        } else {
            0
        };
        unstuff = ((val >> 8) & 0xFF) > 0x8F;

        tmp |= (val & 0xFF) << bits;
        bits += 8 - if unstuff && (val & 0x7F) == 0x7F {
            1
        } else {
            0
        };
        unstuff = (val & 0xFF) > 0x8F;

        self.tmp |= (tmp as u64) << self.bits;
        self.bits += bits;
        self.unstuff = unstuff;
    }

    /// Initialise MRP reader.
    fn rev_init_mrp(data: &'a [u8], lcup: usize, len2: usize) -> Self {
        let start_pos = lcup + len2 - 1;
        let mut mrp = RevStruct {
            data,
            pos: start_pos,
            tmp: 0,
            bits: 0,
            size: len2 as i32,
            unstuff: true,
        };

        let num = 1 + (mrp.pos & 0x3);
        for _ in 0..num {
            let d: u64 = if mrp.size > 0 {
                let v = mrp.data[mrp.pos] as u64;
                if mrp.pos > 0 {
                    mrp.pos -= 1;
                }
                mrp.size -= 1;
                v
            } else {
                0
            };
            let d_bits = 8 - if mrp.unstuff && (d & 0x7F) == 0x7F {
                1u32
            } else {
                0
            };
            mrp.tmp |= d << mrp.bits;
            mrp.bits += d_bits;
            mrp.unstuff = d > 0x8F;
        }
        mrp.rev_read_mrp();
        mrp
    }

    /// Fetch 32 bits from MRP (ensure ≥33 bits available).
    fn rev_fetch_mrp(&mut self) -> u32 {
        if self.bits < 32 {
            self.rev_read_mrp();
            if self.bits < 32 {
                self.rev_read_mrp();
            }
        }
        self.tmp as u32
    }

    /// Consume `num_bits` from MRP and return remaining bottom 32 bits.
    fn rev_advance_mrp(&mut self, num_bits: u32) -> u32 {
        debug_assert!(num_bits <= self.bits);
        self.tmp >>= num_bits;
        self.bits -= num_bits;
        self.tmp as u32
    }
}

// ============================================================================
// Forward-growing reader (MagSgn / SPP)
// ============================================================================

/// State structure for reading and unstuffing of forward-growing bitstreams
/// (MagSgn and SPP).
struct FrwdStruct32<'a> {
    data: &'a [u8],
    pos: usize,
    tmp: u64,
    bits: u32,
    /// 0 or 1: whether the next bit needs unstuffing.
    unstuff: u32,
    size: i32,
}

impl<'a> FrwdStruct32<'a> {
    /// Read and unstuff 4 bytes forward.  `FEED` is the byte value used when
    /// the bitstream is exhausted (0xFF for MagSgn, 0x00 for SPP).
    fn frwd_read<const FEED: u8>(&mut self) {
        debug_assert!(self.bits <= 32);

        let val: u32;
        if self.size > 3 {
            val = u32::from(self.data[self.pos])
                | (u32::from(self.data[self.pos + 1]) << 8)
                | (u32::from(self.data[self.pos + 2]) << 16)
                | (u32::from(self.data[self.pos + 3]) << 24);
            self.pos += 4;
            self.size -= 4;
        } else if self.size > 0 {
            let mut v = if FEED != 0 { 0xFFFF_FFFFu32 } else { 0u32 };
            let mut i = 0u32;
            while self.size > 0 {
                let b = u32::from(self.data[self.pos]);
                self.pos += 1;
                let m = !(0xFFu32 << i);
                v = (v & m) | (b << i);
                self.size -= 1;
                i += 8;
            }
            val = v;
        } else {
            val = if FEED != 0 { 0xFFFF_FFFFu32 } else { 0u32 };
        }

        let mut bits = 8u32 - self.unstuff;
        let mut t = val & 0xFF;
        let mut unstuff = ((val & 0xFF) == 0xFF) as u32;

        t |= ((val >> 8) & 0xFF) << bits;
        bits += 8 - unstuff;
        unstuff = (((val >> 8) & 0xFF) == 0xFF) as u32;

        t |= ((val >> 16) & 0xFF) << bits;
        bits += 8 - unstuff;
        unstuff = (((val >> 16) & 0xFF) == 0xFF) as u32;

        t |= ((val >> 24) & 0xFF) << bits;
        bits += 8 - unstuff;
        self.unstuff = (((val >> 24) & 0xFF) == 0xFF) as u32;

        self.tmp |= (t as u64) << self.bits;
        self.bits += bits;
    }

    /// Initialise forward reader.
    fn frwd_init<const FEED: u8>(data: &'a [u8], size: i32) -> Self {
        let mut msp = FrwdStruct32 {
            data,
            pos: 0,
            tmp: 0,
            bits: 0,
            unstuff: 0,
            size,
        };

        // Align to 4-byte boundary (portable: always read up to 4 bytes)
        let num = 4usize;
        for _ in 0..num {
            let d: u64 = if msp.size > 0 {
                let v = msp.data[msp.pos] as u64;
                msp.pos += 1;
                msp.size -= 1;
                v
            } else {
                FEED as u64
            };
            msp.tmp |= d << msp.bits;
            msp.bits += 8 - msp.unstuff;
            msp.unstuff = ((d & 0xFF) == 0xFF) as u32;
        }
        msp.frwd_read::<FEED>();
        msp
    }

    /// Consume `num_bits` from the bitstream.
    fn frwd_advance(&mut self, num_bits: u32) {
        debug_assert!(num_bits <= self.bits);
        self.tmp >>= num_bits;
        self.bits -= num_bits;
    }

    /// Fetch 32 bits (ensure ≥32 bits available).
    fn frwd_fetch<const FEED: u8>(&mut self) -> u32 {
        if self.bits < 32 {
            self.frwd_read::<FEED>();
            if self.bits < 32 {
                self.frwd_read::<FEED>();
            }
        }
        self.tmp as u32
    }
}

// ============================================================================
// Main entry point
// ============================================================================

/// Decode one HTJ2K codeblock (32-bit path).
///
/// Returns `Ok(true)` on success, `Ok(false)` for non-fatal decode failures
/// (malformed data that can be silently skipped).
#[allow(unused_assignments)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_codeblock32(
    coded_data: &mut [u8],
    decoded_data: &mut [u32],
    missing_msbs: u32,
    mut num_passes: u32,
    lengths1: u32,
    lengths2: u32,
    width: u32,
    height: u32,
    stride: u32,
    stripe_causal: bool,
) -> crate::error::Result<bool> {
    if num_passes > 1 && lengths2 == 0 {
        ojph_warn!(
            "A malformed codeblock that has more than one coding pass, \
             but zero length for 2nd and potential 3rd pass."
        );
        num_passes = 1;
    }

    if num_passes > 3 {
        ojph_warn!(
            "We do not support more than 3 coding passes; \
             This codeblock has {} passes.",
            num_passes
        );
        return Ok(false);
    }

    if missing_msbs > 30 {
        ojph_warn!("32 bits are not enough to decode this codeblock.");
        return Ok(false);
    } else if missing_msbs == 30 {
        ojph_warn!("Not enough precision to decode the cleanup pass.");
        return Ok(false);
    } else if missing_msbs == 29 && num_passes > 1 {
        num_passes = 1;
        ojph_warn!(
            "Not enough precision to decode the SgnProp \
             nor MagRef passes; both will be skipped."
        );
    }

    let p = 30 - missing_msbs;

    if lengths1 < 2 {
        ojph_warn!("Wrong codeblock length.");
        return Ok(false);
    }

    let lcup = lengths1 as usize;
    let scup = ((coded_data[lcup - 1] as usize) << 4) + ((coded_data[lcup - 2] as usize) & 0xF);
    if scup < 2 || scup > lcup || scup > 4079 {
        return Ok(false);
    }

    let tables = decoder_tables();
    let vlc_tbl0 = &tables.vlc_tbl0;
    let vlc_tbl1 = &tables.vlc_tbl1;
    let uvlc_tbl0 = &tables.uvlc_tbl0;
    let uvlc_tbl1 = &tables.uvlc_tbl1;

    // Scratch buffer: 8 u16 entries per quad row, up to 513 rows.
    let sstr = ((width + 2) + 7) & !7; // stride in u16 entries, multiple of 8
    let scratch_len = (sstr as usize) * 513;
    let mut scratch = vec![0u16; scratch_len];

    let mmsbp2 = missing_msbs + 2;

    // ========================================================================
    // Step 1: Decode VLC and MEL segments
    // ========================================================================
    {
        let mut mel = DecMelSt::mel_init(coded_data, lcup, scup);
        let mut vlc = RevStruct::rev_init(coded_data, lcup, scup);

        let mut run = mel.mel_get_run();
        let mut c_q: u32 = 0;

        // --- Initial quad row -----------------------------------------------
        {
            let mut sp_off: usize = 0; // offset into scratch
            let mut x: u32 = 0;
            while x < width {
                // First quad
                let mut vlc_val = vlc.rev_fetch();

                let mut t0 = vlc_tbl0[(c_q + (vlc_val & 0x7F)) as usize];
                if c_q == 0 {
                    run -= 2;
                    t0 = if run == -1 { t0 } else { 0 };
                    if run < 0 {
                        run = mel.mel_get_run();
                    }
                }
                scratch[sp_off] = t0;
                x += 2;

                c_q = ((u32::from(t0) & 0x10) << 3) | ((u32::from(t0) & 0xE0) << 2);
                vlc_val = vlc.rev_advance(u32::from(t0) & 0x7);

                // Second quad
                let mut t1 = vlc_tbl0[(c_q + (vlc_val & 0x7F)) as usize];
                if c_q == 0 && x < width {
                    run -= 2;
                    t1 = if run == -1 { t1 } else { 0 };
                    if run < 0 {
                        run = mel.mel_get_run();
                    }
                }
                t1 = if x < width { t1 } else { 0 };
                scratch[sp_off + 2] = t1;
                x += 2;

                c_q = ((u32::from(t1) & 0x10) << 3) | ((u32::from(t1) & 0xE0) << 2);
                vlc_val = vlc.rev_advance(u32::from(t1) & 0x7);

                // Decode u (UVLC)
                let mut uvlc_mode = ((u32::from(t0) & 0x8) << 3) | ((u32::from(t1) & 0x8) << 4);
                if uvlc_mode == 0xC0 {
                    run -= 2;
                    uvlc_mode += if run == -1 { 0x40 } else { 0 };
                    if run < 0 {
                        run = mel.mel_get_run();
                    }
                }

                let mut uvlc_entry = u32::from(uvlc_tbl0[(uvlc_mode + (vlc_val & 0x3F)) as usize]);
                vlc_val = vlc.rev_advance(uvlc_entry & 0x7);
                uvlc_entry >>= 3;
                let len = uvlc_entry & 0xF;
                let tmp = vlc_val & ((1u32 << len) - 1);
                vlc_val = vlc.rev_advance(len);
                uvlc_entry >>= 4;
                let len0 = uvlc_entry & 0x7;
                uvlc_entry >>= 3;
                let u_q0 = 1 + (uvlc_entry & 7) + (tmp & !(0xFFu32 << len0));
                scratch[sp_off + 1] = u_q0 as u16; // kappa = 1
                let u_q1 = 1 + (uvlc_entry >> 3) + (tmp >> len0);
                scratch[sp_off + 3] = u_q1 as u16; // kappa = 1

                sp_off += 4;
            }
            // Sentinel
            scratch[sp_off] = 0;
            scratch[sp_off + 1] = 0;
        }

        // --- Non-initial quad rows ------------------------------------------
        let mut y: u32 = 2;
        while y < height {
            c_q = 0;
            let row_off = ((y >> 1) as usize) * (sstr as usize);
            let prev_row_off = row_off - (sstr as usize);
            let mut sp_x: usize = 0; // column offset within the row
            let mut x: u32 = 0;

            while x < width {
                // Context from row above
                c_q |= (u32::from(scratch[prev_row_off + sp_x]) & 0xA0) << 2;
                c_q |= (u32::from(scratch[prev_row_off + sp_x + 2]) & 0x20) << 4;

                // First quad
                let mut vlc_val = vlc.rev_fetch();
                let mut t0 = vlc_tbl1[(c_q + (vlc_val & 0x7F)) as usize];
                if c_q == 0 {
                    run -= 2;
                    t0 = if run == -1 { t0 } else { 0 };
                    if run < 0 {
                        run = mel.mel_get_run();
                    }
                }
                scratch[row_off + sp_x] = t0;
                x += 2;

                // Prepare context for next quad (eqn. 2)
                c_q = ((u32::from(t0) & 0x40) << 2) | ((u32::from(t0) & 0x80) << 1);
                c_q |= u32::from(scratch[prev_row_off + sp_x]) & 0x80;
                c_q |= (u32::from(scratch[prev_row_off + sp_x + 2]) & 0xA0) << 2;
                c_q |= (u32::from(scratch[prev_row_off + sp_x + 4]) & 0x20) << 4;

                vlc_val = vlc.rev_advance(u32::from(t0) & 0x7);

                // Second quad
                let mut t1 = vlc_tbl1[(c_q + (vlc_val & 0x7F)) as usize];
                if c_q == 0 && x < width {
                    run -= 2;
                    t1 = if run == -1 { t1 } else { 0 };
                    if run < 0 {
                        run = mel.mel_get_run();
                    }
                }
                t1 = if x < width { t1 } else { 0 };
                scratch[row_off + sp_x + 2] = t1;
                x += 2;

                c_q = ((u32::from(t1) & 0x40) << 2) | ((u32::from(t1) & 0x80) << 1);
                c_q |= u32::from(scratch[prev_row_off + sp_x + 2]) & 0x80;

                vlc_val = vlc.rev_advance(u32::from(t1) & 0x7);

                // Decode u (UVLC) — non-initial rows
                let uvlc_mode = ((u32::from(t0) & 0x8) << 3) | ((u32::from(t1) & 0x8) << 4);
                let mut uvlc_entry = u32::from(uvlc_tbl1[(uvlc_mode + (vlc_val & 0x3F)) as usize]);
                vlc_val = vlc.rev_advance(uvlc_entry & 0x7);
                uvlc_entry >>= 3;
                let len = uvlc_entry & 0xF;
                let tmp = vlc_val & ((1u32 << len) - 1);
                vlc_val = vlc.rev_advance(len);
                uvlc_entry >>= 4;
                let len0 = uvlc_entry & 0x7;
                uvlc_entry >>= 3;
                // No +1 for non-initial rows (kappa computed in step 2)
                let u_q0 = (uvlc_entry & 7) + (tmp & !(0xFFu32 << len0));
                scratch[row_off + sp_x + 1] = u_q0 as u16;
                let u_q1 = (uvlc_entry >> 3) + (tmp >> len0);
                scratch[row_off + sp_x + 3] = u_q1 as u16;

                sp_x += 4;
            }
            // Sentinel
            scratch[row_off + sp_x] = 0;
            scratch[row_off + sp_x + 1] = 0;

            y += 2;
        }
    }

    // ========================================================================
    // Step 2: Decode MagSgn
    // ========================================================================
    {
        const V_N_SIZE: usize = 512 + 4;
        let mut v_n_scratch = [0u32; V_N_SIZE];

        let mut magsgn = FrwdStruct32::frwd_init::<0xFF>(coded_data, (lcup - scup) as i32);

        // --- Initial row (y=0) ---------------------------------------------
        {
            let mut prev_v_n: u32 = 0;
            let mut dp_col: usize = 0; // column index in decoded_data
            let mut sp_off: usize = 0; // offset in scratch (pairs of 2)
            let mut vp: usize = 0; // index into v_n_scratch
            let mut x: u32 = 0;

            while x < width {
                let inf = u32::from(scratch[sp_off]);
                let u_q = u32::from(scratch[sp_off + 1]);
                if u_q > mmsbp2 {
                    return Ok(false);
                }

                // Sample bit 0
                let mut val: u32 = 0;
                let mut v_n: u32 = 0;
                if inf & (1 << 4) != 0 {
                    let ms_val = magsgn.frwd_fetch::<0xFF>();
                    let m_n = u_q - ((inf >> 12) & 1);
                    magsgn.frwd_advance(m_n);
                    val = ms_val << 31;
                    v_n = ms_val & ((1u32 << m_n) - 1);
                    v_n |= ((inf >> 8) & 1) << m_n;
                    v_n |= 1;
                    val |= (v_n + 2) << (p - 1);
                }
                decoded_data[0 * (stride as usize) + dp_col] = val;

                // Sample bit 1
                v_n = 0;
                val = 0;
                if inf & (1 << 5) != 0 {
                    let ms_val = magsgn.frwd_fetch::<0xFF>();
                    let m_n = u_q - ((inf >> 13) & 1);
                    magsgn.frwd_advance(m_n);
                    val = ms_val << 31;
                    v_n = ms_val & ((1u32 << m_n) - 1);
                    v_n |= ((inf >> 9) & 1) << m_n;
                    v_n |= 1;
                    val |= (v_n + 2) << (p - 1);
                }
                decoded_data[1 * (stride as usize) + dp_col] = val;
                v_n_scratch[vp] = prev_v_n | v_n;
                prev_v_n = 0;
                dp_col += 1;
                x += 1;
                if x >= width {
                    vp += 1;
                    break;
                }

                // Sample bit 2
                val = 0;
                if inf & (1 << 6) != 0 {
                    let ms_val = magsgn.frwd_fetch::<0xFF>();
                    let m_n = u_q - ((inf >> 14) & 1);
                    magsgn.frwd_advance(m_n);
                    val = ms_val << 31;
                    v_n = ms_val & ((1u32 << m_n) - 1);
                    v_n |= ((inf >> 10) & 1) << m_n;
                    v_n |= 1;
                    val |= (v_n + 2) << (p - 1);
                }
                decoded_data[0 * (stride as usize) + dp_col] = val;

                // Sample bit 3
                v_n = 0;
                val = 0;
                if inf & (1 << 7) != 0 {
                    let ms_val = magsgn.frwd_fetch::<0xFF>();
                    let m_n = u_q - ((inf >> 15) & 1);
                    magsgn.frwd_advance(m_n);
                    val = ms_val << 31;
                    v_n = ms_val & ((1u32 << m_n) - 1);
                    v_n |= ((inf >> 11) & 1) << m_n;
                    v_n |= 1;
                    val |= (v_n + 2) << (p - 1);
                }
                decoded_data[1 * (stride as usize) + dp_col] = val;
                prev_v_n = v_n;
                dp_col += 1;
                x += 1;

                sp_off += 2;
                vp += 1;
            }
            v_n_scratch[vp] = prev_v_n;
        }

        // --- Non-initial rows (y >= 2) --------------------------------------
        let mut y: u32 = 2;
        while y < height {
            let row_off = ((y >> 1) as usize) * (sstr as usize);
            let mut sp_off: usize = row_off;
            let mut vp: usize = 0;
            let dp_base = (y as usize) * (stride as usize);
            let mut dp_col: usize = 0;
            let mut prev_v_n: u32 = 0;
            let mut x: u32 = 0;

            while x < width {
                let inf = u32::from(scratch[sp_off]);
                let u_q = u32::from(scratch[sp_off + 1]);

                let gamma = {
                    let g = inf & 0xF0;
                    g & g.wrapping_sub(0x10)
                };
                let emax = v_n_scratch[vp] | v_n_scratch[vp + 1];
                let emax = 31 - count_leading_zeros(emax | 2);
                let kappa = if gamma != 0 { emax } else { 1 };
                let u_q_total = u_q + kappa;
                if u_q_total > mmsbp2 {
                    return Ok(false);
                }

                // Sample bit 0
                let mut val: u32 = 0;
                let mut v_n: u32 = 0;
                if inf & (1 << 4) != 0 {
                    let ms_val = magsgn.frwd_fetch::<0xFF>();
                    let m_n = u_q_total - ((inf >> 12) & 1);
                    magsgn.frwd_advance(m_n);
                    val = ms_val << 31;
                    v_n = ms_val & ((1u32 << m_n) - 1);
                    v_n |= ((inf >> 8) & 1) << m_n;
                    v_n |= 1;
                    val |= (v_n + 2) << (p - 1);
                }
                decoded_data[dp_base + dp_col] = val;

                // Sample bit 1
                v_n = 0;
                val = 0;
                if inf & (1 << 5) != 0 {
                    let ms_val = magsgn.frwd_fetch::<0xFF>();
                    let m_n = u_q_total - ((inf >> 13) & 1);
                    magsgn.frwd_advance(m_n);
                    val = ms_val << 31;
                    v_n = ms_val & ((1u32 << m_n) - 1);
                    v_n |= ((inf >> 9) & 1) << m_n;
                    v_n |= 1;
                    val |= (v_n + 2) << (p - 1);
                }
                decoded_data[dp_base + (stride as usize) + dp_col] = val;
                v_n_scratch[vp] = prev_v_n | v_n;
                prev_v_n = 0;
                dp_col += 1;
                x += 1;
                if x >= width {
                    vp += 1;
                    break;
                }

                // Sample bit 2
                val = 0;
                if inf & (1 << 6) != 0 {
                    let ms_val = magsgn.frwd_fetch::<0xFF>();
                    let m_n = u_q_total - ((inf >> 14) & 1);
                    magsgn.frwd_advance(m_n);
                    val = ms_val << 31;
                    v_n = ms_val & ((1u32 << m_n) - 1);
                    v_n |= ((inf >> 10) & 1) << m_n;
                    v_n |= 1;
                    val |= (v_n + 2) << (p - 1);
                }
                decoded_data[dp_base + dp_col] = val;

                // Sample bit 3
                v_n = 0;
                val = 0;
                if inf & (1 << 7) != 0 {
                    let ms_val = magsgn.frwd_fetch::<0xFF>();
                    let m_n = u_q_total - ((inf >> 15) & 1);
                    magsgn.frwd_advance(m_n);
                    val = ms_val << 31;
                    v_n = ms_val & ((1u32 << m_n) - 1);
                    v_n |= ((inf >> 11) & 1) << m_n;
                    v_n |= 1;
                    val |= (v_n + 2) << (p - 1);
                }
                decoded_data[dp_base + (stride as usize) + dp_col] = val;
                prev_v_n = v_n;
                dp_col += 1;
                x += 1;

                sp_off += 2;
                vp += 1;
            }
            v_n_scratch[vp] = prev_v_n;

            y += 2;
        }
    }

    // ========================================================================
    // Step 3: SPP + MRP (if num_passes > 1)
    // ========================================================================
    if num_passes > 1 {
        // Re-use scratch for sigma (column-organized significance).
        let mstr = {
            let m = (width + 3) >> 2; // 4 columns per u16
            ((m + 2) + 7) & !7 // multiple of 8
        } as usize;

        // sigma occupies the first part of scratch
        let _sigma_len = mstr * (((height + 3) >> 2) as usize + 2);
        // Re-arrange quad significance → column significance
        {
            let mut y: u32 = 0;
            while y < height {
                let sp_row = ((y >> 1) as usize) * (sstr as usize);
                let dp_row = ((y >> 2) as usize) * mstr;
                let mut col = 0usize;
                let mut x: u32 = 0;
                while x < width {
                    let s0 = u32::from(scratch[sp_row + col * 4]);
                    let s1 = u32::from(scratch[sp_row + col * 4 + 2]);
                    let mut t0 = ((s0 & 0x30) >> 4) | ((s0 & 0xC0) >> 2);
                    t0 |= ((s1 & 0x30) << 4) | ((s1 & 0xC0) << 6);

                    let sp_below = sp_row + (sstr as usize);
                    let s2 = if y + 2 < height {
                        u32::from(scratch[sp_below + col * 4])
                    } else {
                        0
                    };
                    let s3 = if y + 2 < height {
                        u32::from(scratch[sp_below + col * 4 + 2])
                    } else {
                        0
                    };
                    let mut t1 = ((s2 & 0x30) >> 2) | (s2 & 0xC0);
                    t1 |= ((s3 & 0x30) << 6) | ((s3 & 0xC0) << 8);

                    scratch[dp_row + col] = (t0 | t1) as u16;
                    col += 1;
                    x += 4;
                }
                scratch[dp_row + col] = 0; // extra entry on the right
                y += 4;
            }
            // Reset one row below the codeblock
            {
                let dp_row = ((y >> 2) as usize) * mstr;
                let mut col = 0usize;
                let mut x: u32 = 0;
                while x < width {
                    scratch[dp_row + col] = 0;
                    col += 1;
                    x += 4;
                }
                scratch[dp_row + col] = 0;
            }
        }

        // -- Significance Propagation Pass (SPP) -----------------------------
        {
            let mut prev_row_sig = vec![0u16; 256 + 8];

            let mut sigprop =
                FrwdStruct32::frwd_init::<0x00>(&coded_data[lengths1 as usize..], lengths2 as i32);

            let mut y: u32 = 0;
            while y < height {
                let mut pattern: u32 = 0xFFFF;
                if height - y < 4 {
                    pattern = 0x7777;
                    if height - y < 3 {
                        pattern = 0x3333;
                        if height - y < 2 {
                            pattern = 0x1111;
                        }
                    }
                }

                let mut prev: u32 = 0;
                let sigma_row = ((y >> 2) as usize) * mstr;
                let sigma_next = sigma_row + mstr;
                let dp_base = (y as usize) * (stride as usize);

                let mut prev_sig_idx: usize = 0;
                let mut cur_sig_idx: usize = 0;

                let mut x: u32 = 0;
                while x < width {
                    // Adjust pattern for right edge
                    let s = ojph_max((x as i32) + 4 - (width as i32), 0);
                    let cur_pattern = pattern >> (s * 4);

                    // Load prev_sig (32 bits = 2 consecutive u16)
                    let ps = u32::from(prev_row_sig[prev_sig_idx])
                        | (u32::from(prev_row_sig[prev_sig_idx + 1]) << 16);

                    // Load next sigma row (32 bits)
                    let ns = u32::from(scratch[sigma_next + cur_sig_idx])
                        | (u32::from(if cur_sig_idx + 1 < scratch.len() - sigma_next {
                            scratch[sigma_next + cur_sig_idx + 1]
                        } else {
                            0
                        }) << 16);

                    let mut u = (ps & 0x8888_8888) >> 3; // row on top
                    if !stripe_causal {
                        u |= (ns & 0x1111_1111) << 3; // row below
                    }

                    // Current sigma (32 bits)
                    let cs = u32::from(scratch[sigma_row + cur_sig_idx])
                        | (u32::from(if cur_sig_idx + 1 < scratch.len() - sigma_row {
                            scratch[sigma_row + cur_sig_idx + 1]
                        } else {
                            0
                        }) << 16);

                    // Vertical integration
                    let mut mbr = cs;
                    mbr |= (cs & 0x7777_7777) << 1; // above neighbors
                    mbr |= (cs & 0xEEEE_EEEE) >> 1; // below neighbors
                    mbr |= u;
                    // Horizontal integration
                    let t = mbr;
                    mbr |= t << 4;
                    mbr |= t >> 4;
                    mbr |= prev >> 12; // significance of previous group

                    mbr &= cur_pattern;
                    mbr &= !cs;

                    let mut new_sig = mbr;
                    if new_sig != 0 {
                        let mut cwd = sigprop.frwd_fetch::<0x00>();
                        let mut cnt: u32 = 0;
                        let mut col_mask: u32 = 0xF;
                        let inv_sig = !cs & cur_pattern;

                        let mut i: i32 = 0;
                        while i < 16 {
                            if (col_mask & new_sig) != 0 {
                                // Scan one column
                                let mut sample_mask = 0x1111u32 & col_mask;
                                if new_sig & sample_mask != 0 {
                                    new_sig &= !sample_mask;
                                    if cwd & 1 != 0 {
                                        let t = 0x33u32 << i;
                                        new_sig |= t & inv_sig;
                                    }
                                    cwd >>= 1;
                                    cnt += 1;
                                }

                                sample_mask <<= 1;
                                if new_sig & sample_mask != 0 {
                                    new_sig &= !sample_mask;
                                    if cwd & 1 != 0 {
                                        let t = 0x76u32 << i;
                                        new_sig |= t & inv_sig;
                                    }
                                    cwd >>= 1;
                                    cnt += 1;
                                }

                                sample_mask <<= 1;
                                if new_sig & sample_mask != 0 {
                                    new_sig &= !sample_mask;
                                    if cwd & 1 != 0 {
                                        let t = 0xECu32 << i;
                                        new_sig |= t & inv_sig;
                                    }
                                    cwd >>= 1;
                                    cnt += 1;
                                }

                                sample_mask <<= 1;
                                if new_sig & sample_mask != 0 {
                                    new_sig &= !sample_mask;
                                    if cwd & 1 != 0 {
                                        let t = 0xC8u32 << i;
                                        new_sig |= t & inv_sig;
                                    }
                                    cwd >>= 1;
                                    cnt += 1;
                                }
                            }
                            i += 4;
                            col_mask <<= 4;
                        }

                        if new_sig != 0 {
                            let val = 3u32 << (p - 2);
                            let mut col_mask2: u32 = 0xF;
                            for ci in 0..4u32 {
                                if (col_mask2 & new_sig) != 0 {
                                    let dp_col = (x + ci) as usize;

                                    let mut sample_mask = 0x1111u32 & col_mask2;
                                    if new_sig & sample_mask != 0 {
                                        decoded_data[dp_base + dp_col] = (cwd << 31) | val;
                                        cwd >>= 1;
                                        cnt += 1;
                                    }

                                    sample_mask += sample_mask;
                                    if new_sig & sample_mask != 0 {
                                        decoded_data[dp_base + (stride as usize) + dp_col] =
                                            (cwd << 31) | val;
                                        cwd >>= 1;
                                        cnt += 1;
                                    }

                                    sample_mask += sample_mask;
                                    if new_sig & sample_mask != 0 {
                                        decoded_data[dp_base + 2 * (stride as usize) + dp_col] =
                                            (cwd << 31) | val;
                                        cwd >>= 1;
                                        cnt += 1;
                                    }

                                    sample_mask += sample_mask;
                                    if new_sig & sample_mask != 0 {
                                        decoded_data[dp_base + 3 * (stride as usize) + dp_col] =
                                            (cwd << 31) | val;
                                        cwd >>= 1;
                                        cnt += 1;
                                    }
                                }
                                col_mask2 <<= 4;
                            }
                        }
                        sigprop.frwd_advance(cnt);
                    }

                    new_sig |= cs;
                    prev_row_sig[prev_sig_idx] = new_sig as u16;

                    // Vertical integration for new sig
                    let t = new_sig;
                    let mut ns2 = new_sig;
                    ns2 |= (t & 0x7777) << 1;
                    ns2 |= (t & 0xEEEE) >> 1;
                    prev = (ns2 | u) & 0xF000;

                    x += 4;
                    prev_sig_idx += 1;
                    cur_sig_idx += 1;
                }
                y += 4;
            }
        }

        // -- Magnitude Refinement Pass (MRP) ---------------------------------
        if num_passes > 2 {
            let mut magref =
                RevStruct::rev_init_mrp(coded_data, lengths1 as usize, lengths2 as usize);

            let mut y: u32 = 0;
            while y < height {
                let sigma_row = ((y >> 2) as usize) * mstr;
                let dp_base = (y as usize) * (stride as usize);
                let half = 1u32 << (p - 2);

                let mut i: u32 = 0;
                while i < width {
                    let sig_idx = sigma_row + ((i >> 2) as usize);
                    // Load 32 bits of sigma (two consecutive u16)
                    let sig = u32::from(scratch[sig_idx])
                        | (u32::from(if sig_idx + 1 < scratch.len() {
                            scratch[sig_idx + 1]
                        } else {
                            0
                        }) << 16);

                    let mut cwd = magref.rev_fetch_mrp();

                    if sig != 0 {
                        let mut col_mask: u32 = 0xF;
                        for j in 0..8u32 {
                            if sig & col_mask != 0 {
                                let col = i + j;
                                let mut sample_mask = 0x1111_1111u32 & col_mask;
                                let mut dp_off = dp_base + (col as usize);

                                for _ in 0..4u32 {
                                    if sig & sample_mask != 0 {
                                        let sym = cwd & 1;
                                        let sym = ((1 - sym) << (p - 1)) | half;
                                        decoded_data[dp_off] ^= sym;
                                        cwd >>= 1;
                                    }
                                    sample_mask += sample_mask;
                                    dp_off += stride as usize;
                                }
                            }
                            col_mask <<= 4;
                        }
                    }
                    magref.rev_advance_mrp(population_count(sig));

                    i += 8;
                }
                y += 4;
            }
        }
    }

    Ok(true)
}
