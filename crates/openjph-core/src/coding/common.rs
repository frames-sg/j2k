//! Common coding utilities shared by encoder and decoders.
//!
//! Port of `ojph_block_common.h/cpp` — lazily-initialised VLC / UVLC
//! decoder and encoder lookup tables.

#![allow(clippy::identity_op)]

use std::sync::OnceLock;

use super::tables::{VlcSrcEntry, VLC_SRC_TABLE0, VLC_SRC_TABLE1};
use crate::arch::population_count;

// ---------------------------------------------------------------------------
// Public data structures
// ---------------------------------------------------------------------------

/// Decoder lookup tables for VLC and UVLC decoding.
pub(crate) struct DecoderTables {
    /// VLC table for initial quad rows (1024 entries).
    pub vlc_tbl0: [u16; 1024],
    /// VLC table for non-initial quad rows (1024 entries).
    pub vlc_tbl1: [u16; 1024],
    /// UVLC table for initial rows (320 entries).
    pub uvlc_tbl0: [u16; 320],
    /// UVLC table for non-initial rows (256 entries).
    pub uvlc_tbl1: [u16; 256],
    /// UVLC bias for initial rows (320 entries).
    pub uvlc_bias: [u8; 320],
}

/// Encoder lookup tables for VLC and UVLC encoding.
pub(crate) struct EncoderTables {
    /// VLC encoding table for initial quad rows (2048 entries).
    pub vlc_tbl0: [u16; 2048],
    /// VLC encoding table for non-initial quad rows (2048 entries).
    pub vlc_tbl1: [u16; 2048],
    /// UVLC encoding table (75 entries).
    pub uvlc_tbl: [UvlcEncEntry; 75],
}

/// One entry of the UVLC encoder table.
#[derive(Clone, Copy, Default)]
pub(crate) struct UvlcEncEntry {
    pub pre: u8,
    pub pre_len: u8,
    pub suf: u8,
    pub suf_len: u8,
    #[allow(dead_code)]
    pub ext: u8,
    #[allow(dead_code)]
    pub ext_len: u8,
}

// ---------------------------------------------------------------------------
// Lazy accessors
// ---------------------------------------------------------------------------

/// Returns a reference to the lazily-initialised decoder tables.
pub(crate) fn decoder_tables() -> &'static DecoderTables {
    static TABLES: OnceLock<DecoderTables> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut t = DecoderTables {
            vlc_tbl0: [0u16; 1024],
            vlc_tbl1: [0u16; 1024],
            uvlc_tbl0: [0u16; 320],
            uvlc_tbl1: [0u16; 256],
            uvlc_bias: [0u8; 320],
        };
        vlc_init_dec_tables(&mut t);
        uvlc_init_dec_tables(&mut t);
        t
    })
}

/// Returns a reference to the lazily-initialised encoder tables.
pub(crate) fn encoder_tables() -> &'static EncoderTables {
    static TABLES: OnceLock<EncoderTables> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut t = EncoderTables {
            vlc_tbl0: [0u16; 2048],
            vlc_tbl1: [0u16; 2048],
            uvlc_tbl: [UvlcEncEntry::default(); 75],
        };
        vlc_init_enc_tables(&mut t);
        uvlc_init_enc_tables(&mut t);
        t
    })
}

// ---------------------------------------------------------------------------
// VLC decoder table initialisation
// ---------------------------------------------------------------------------

/// Populates `vlc_tbl0` and `vlc_tbl1` from the VLC source tables.
///
/// For each of 1024 indices (7-bit codeword | 3-bit context):
///   find the matching source entry and pack
///   `(rho << 4) | (u_off << 3) | (e_k << 12) | (e_1 << 8) | cwd_len`.
fn vlc_init_dec_tables(t: &mut DecoderTables) {
    vlc_init_one_dec_table(&mut t.vlc_tbl0, VLC_SRC_TABLE0);
    vlc_init_one_dec_table(&mut t.vlc_tbl1, VLC_SRC_TABLE1);
}

fn vlc_init_one_dec_table(tbl: &mut [u16; 1024], src: &[VlcSrcEntry]) {
    for i in 0..1024u32 {
        let cwd = i & 0x7F;
        let c_q = i >> 7;
        for entry in src {
            if entry.c_q as u32 == c_q && entry.cwd as u32 == (cwd & ((1u32 << entry.cwd_len) - 1))
            {
                tbl[i as usize] = ((entry.rho as u16) << 4)
                    | ((entry.u_off as u16) << 3)
                    | ((entry.e_k as u16) << 12)
                    | ((entry.e_1 as u16) << 8)
                    | (entry.cwd_len as u16);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// UVLC decoder table initialisation
// ---------------------------------------------------------------------------

/// Populates `uvlc_tbl0`, `uvlc_tbl1`, and `uvlc_bias`.
fn uvlc_init_dec_tables(t: &mut DecoderTables) {
    // Prefix decoder: index is the 3 LSBs of the VLC fragment.
    // Packed as: prefix_len(2) | suffix_len(3) | u_pfx(3).
    static DEC: [u8; 8] = [
        3 | (5 << 2) | (5 << 5), // 000
        1 | (0 << 2) | (1 << 5), // xx1
        2 | (0 << 2) | (2 << 5), // x10
        1 | (0 << 2) | (1 << 5), // xx1
        3 | (1 << 2) | (3 << 5), // 100
        1 | (0 << 2) | (1 << 5), // xx1
        2 | (0 << 2) | (2 << 5), // x10
        1 | (0 << 2) | (1 << 5), // xx1
    ];

    // --- uvlc_tbl0 (320 entries, modes 0-4) --------------------------------
    for i in 0u32..320 {
        let mode = i >> 6;
        let mut vlc = i & 0x3F;

        if mode == 0 {
            // Both u_off are 0
            t.uvlc_tbl0[i as usize] = 0;
            t.uvlc_bias[i as usize] = 0;
        } else if mode <= 2 {
            // One u_off set
            let d = DEC[(vlc & 0x7) as usize] as u32;

            let total_prefix = d & 0x3;
            let total_suffix = (d >> 2) & 0x7;
            let u0_suffix_len = if mode == 1 { total_suffix } else { 0 };
            let u0 = if mode == 1 { d >> 5 } else { 0 };
            let u1 = if mode == 1 { 0 } else { d >> 5 };

            t.uvlc_tbl0[i as usize] = (total_prefix
                | (total_suffix << 3)
                | (u0_suffix_len << 7)
                | (u0 << 10)
                | (u1 << 13)) as u16;
        } else if mode == 3 {
            // Both u_off are 1, MEL event = 0
            let d0 = DEC[(vlc & 0x7) as usize] as u32;
            vlc >>= d0 & 0x3;
            let d1 = DEC[(vlc & 0x7) as usize] as u32;

            let (total_prefix, u0_suffix_len, total_suffix, u0, u1);
            if (d0 & 0x3) == 3 {
                // First prefix codeword is "000" — special case
                total_prefix = (d0 & 0x3) + 1;
                u0_suffix_len = (d0 >> 2) & 0x7;
                total_suffix = u0_suffix_len;
                u0 = d0 >> 5;
                u1 = (vlc & 1) + 1;
                t.uvlc_bias[i as usize] = 4; // 0b00 for u0 and 0b01 for u1
            } else {
                total_prefix = (d0 & 0x3) + (d1 & 0x3);
                u0_suffix_len = (d0 >> 2) & 0x7;
                total_suffix = u0_suffix_len + ((d1 >> 2) & 0x7);
                u0 = d0 >> 5;
                u1 = d1 >> 5;
                t.uvlc_bias[i as usize] = 0;
            }

            t.uvlc_tbl0[i as usize] = (total_prefix
                | (total_suffix << 3)
                | (u0_suffix_len << 7)
                | (u0 << 10)
                | (u1 << 13)) as u16;
        } else {
            // mode == 4: Both u_off are 1, MEL event = 1
            let d0 = DEC[(vlc & 0x7) as usize] as u32;
            vlc >>= d0 & 0x3;
            let d1 = DEC[(vlc & 0x7) as usize] as u32;

            let total_prefix = (d0 & 0x3) + (d1 & 0x3);
            let u0_suffix_len = (d0 >> 2) & 0x7;
            let total_suffix = u0_suffix_len + ((d1 >> 2) & 0x7);
            let u0 = (d0 >> 5) + 2;
            let u1 = (d1 >> 5) + 2;

            t.uvlc_tbl0[i as usize] = (total_prefix
                | (total_suffix << 3)
                | (u0_suffix_len << 7)
                | (u0 << 10)
                | (u1 << 13)) as u16;
            t.uvlc_bias[i as usize] = 10; // 0b10 for u0 and 0b10 for u1
        }
    }

    // --- uvlc_tbl1 (256 entries, modes 0-3) --------------------------------
    for i in 0u32..256 {
        let mode = i >> 6;
        let mut vlc = i & 0x3F;

        if mode == 0 {
            t.uvlc_tbl1[i as usize] = 0;
        } else if mode <= 2 {
            let d = DEC[(vlc & 0x7) as usize] as u32;

            let total_prefix = d & 0x3;
            let total_suffix = (d >> 2) & 0x7;
            let u0_suffix_len = if mode == 1 { total_suffix } else { 0 };
            let u0 = if mode == 1 { d >> 5 } else { 0 };
            let u1 = if mode == 1 { 0 } else { d >> 5 };

            t.uvlc_tbl1[i as usize] = (total_prefix
                | (total_suffix << 3)
                | (u0_suffix_len << 7)
                | (u0 << 10)
                | (u1 << 13)) as u16;
        } else {
            // mode == 3: Both u_off are 1
            let d0 = DEC[(vlc & 0x7) as usize] as u32;
            vlc >>= d0 & 0x3;
            let d1 = DEC[(vlc & 0x7) as usize] as u32;

            let total_prefix = (d0 & 0x3) + (d1 & 0x3);
            let u0_suffix_len = (d0 >> 2) & 0x7;
            let total_suffix = u0_suffix_len + ((d1 >> 2) & 0x7);
            let u0 = d0 >> 5;
            let u1 = d1 >> 5;

            t.uvlc_tbl1[i as usize] = (total_prefix
                | (total_suffix << 3)
                | (u0_suffix_len << 7)
                | (u0 << 10)
                | (u1 << 13)) as u16;
        }
    }
}

// ---------------------------------------------------------------------------
// VLC encoder table initialisation
// ---------------------------------------------------------------------------

/// Populates `vlc_tbl0` and `vlc_tbl1` for encoding.
///
/// Index: `(c_q << 8) | (rho << 4) | emb`
/// Entry: `(cwd << 8) + (cwd_len << 4) + e_k`
fn vlc_init_enc_tables(t: &mut EncoderTables) {
    vlc_init_one_enc_table(&mut t.vlc_tbl0, VLC_SRC_TABLE0);
    vlc_init_one_enc_table(&mut t.vlc_tbl1, VLC_SRC_TABLE1);
}

fn vlc_init_one_enc_table(tbl: &mut [u16; 2048], src: &[VlcSrcEntry]) {
    // Pre-compute popcount for 4-bit patterns.
    let mut pattern_popcnt = [0i32; 16];
    for i in 0u32..16 {
        pattern_popcnt[i as usize] = population_count(i) as i32;
    }

    for i in 0..2048u32 {
        let c_q = i >> 8;
        let rho = (i >> 4) & 0xF;
        let emb = i & 0xF;

        if (emb & rho) != emb || (rho == 0 && c_q == 0) {
            tbl[i as usize] = 0;
            continue;
        }

        let mut best_entry: Option<&VlcSrcEntry> = None;

        if emb != 0 {
            // u_off = 1: find entry with matching EMB and highest popcount(e_k)
            let mut best_e_k = -1i32;
            for entry in src {
                if entry.c_q as u32 == c_q
                    && entry.rho as u32 == rho
                    && entry.u_off == 1
                    && (emb & entry.e_k as u32) == entry.e_1 as u32
                {
                    let ones = pattern_popcnt[entry.e_k as usize];
                    if ones >= best_e_k {
                        best_entry = Some(entry);
                        best_e_k = ones;
                    }
                }
            }
        } else {
            // u_off = 0: find first matching entry
            for entry in src {
                if entry.c_q as u32 == c_q && entry.rho as u32 == rho && entry.u_off == 0 {
                    best_entry = Some(entry);
                    break;
                }
            }
        }

        if let Some(e) = best_entry {
            tbl[i as usize] = ((e.cwd as u16) << 8) + ((e.cwd_len as u16) << 4) + (e.e_k as u16);
        }
    }
}

// ---------------------------------------------------------------------------
// UVLC encoder table initialisation
// ---------------------------------------------------------------------------

/// Populates the 75-entry UVLC encoder table.
fn uvlc_init_enc_tables(t: &mut EncoderTables) {
    let tbl = &mut t.uvlc_tbl;

    tbl[0] = UvlcEncEntry {
        pre: 0,
        pre_len: 0,
        suf: 0,
        suf_len: 0,
        ext: 0,
        ext_len: 0,
    };
    tbl[1] = UvlcEncEntry {
        pre: 1,
        pre_len: 1,
        suf: 0,
        suf_len: 0,
        ext: 0,
        ext_len: 0,
    };
    tbl[2] = UvlcEncEntry {
        pre: 2,
        pre_len: 2,
        suf: 0,
        suf_len: 0,
        ext: 0,
        ext_len: 0,
    };
    tbl[3] = UvlcEncEntry {
        pre: 4,
        pre_len: 3,
        suf: 0,
        suf_len: 1,
        ext: 0,
        ext_len: 0,
    };
    tbl[4] = UvlcEncEntry {
        pre: 4,
        pre_len: 3,
        suf: 1,
        suf_len: 1,
        ext: 0,
        ext_len: 0,
    };

    for (i, entry) in tbl[5..33].iter_mut().enumerate() {
        *entry = UvlcEncEntry {
            pre: 0,
            pre_len: 3,
            suf: i as u8,
            suf_len: 5,
            ext: 0,
            ext_len: 0,
        };
    }

    for (j, entry) in tbl[33..75].iter_mut().enumerate() {
        *entry = UvlcEncEntry {
            pre: 0,
            pre_len: 3,
            suf: (28 + j % 4) as u8,
            suf_len: 5,
            ext: (j / 4) as u8,
            ext_len: 4,
        };
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_tables_initialise() {
        let dt = decoder_tables();
        // Smoke test: not all zeros (modes 1+ produce nonzero entries)
        assert!(dt.vlc_tbl0.iter().any(|&v| v != 0));
        assert!(dt.vlc_tbl1.iter().any(|&v| v != 0));
        assert!(dt.uvlc_tbl0.iter().any(|&v| v != 0));
        assert!(dt.uvlc_tbl1.iter().any(|&v| v != 0));
    }

    #[test]
    fn encoder_tables_initialise() {
        let et = encoder_tables();
        assert!(et.vlc_tbl0.iter().any(|&v| v != 0));
        assert!(et.vlc_tbl1.iter().any(|&v| v != 0));
        // Entry 1 should have pre=1, pre_len=1
        assert_eq!(et.uvlc_tbl[1].pre, 1);
        assert_eq!(et.uvlc_tbl[1].pre_len, 1);
    }

    #[test]
    fn uvlc_enc_table_entry_33() {
        let et = encoder_tables();
        // i=33: pre=0, pre_len=3, suf=28+(33-33)%4=28, suf_len=5,
        //       ext=(33-33)/4=0, ext_len=4
        assert_eq!(et.uvlc_tbl[33].pre, 0);
        assert_eq!(et.uvlc_tbl[33].pre_len, 3);
        assert_eq!(et.uvlc_tbl[33].suf, 28);
        assert_eq!(et.uvlc_tbl[33].suf_len, 5);
        assert_eq!(et.uvlc_tbl[33].ext, 0);
        assert_eq!(et.uvlc_tbl[33].ext_len, 4);
    }

    #[test]
    fn vlc_dec_tbl0_known_entry() {
        // Index for c_q=0, cwd=0b0000110 (0x06, len 4):
        // First src entry: c_q=0, rho=1, u_off=0, e_k=0, e_1=0, cwd=0x06, cwd_len=4
        // i = (0 << 7) | 0x06 = 6
        // Expected: (1<<4)|(0<<3)|(0<<12)|(0<<8)|4 = 16+4 = 20
        let dt = decoder_tables();
        assert_eq!(dt.vlc_tbl0[6], 20);
    }
}
