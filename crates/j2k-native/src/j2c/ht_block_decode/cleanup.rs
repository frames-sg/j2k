// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::ht_tables::{UVLC_TABLE0, UVLC_TABLE1, VLC_TABLE0, VLC_TABLE1};
use super::readers::{MelDecoder, ReverseBitReader, VlcBits};

pub(super) fn cleanup_symbol_stride(width: u32) -> usize {
    ((width + 2 + 7) & !7) as usize
}

pub(super) fn cleanup_segment_suffix_length(coded_data: &[u8], lcup: usize) -> Option<usize> {
    if lcup < 2 || coded_data.len() < lcup {
        return None;
    }

    let scup = ((coded_data[lcup - 1] as usize) << 4) + usize::from(coded_data[lcup - 2] & 0x0F);
    if !(2..=lcup).contains(&scup) || scup > 4079 {
        return None;
    }

    Some(scup)
}

/// Non-initial U-VLC decoding of both quads in one lookup, indexed by the
/// two U-offset flags and the next ten VLC bits: `consumed | u0 << 8 |
/// u1 << 20`, or `WIDE_UVLC_FALLBACK` when the codewords are longer.
const UVLC_WIDE1: [u32; 4096] = uvlc_wide1();
const WIDE_UVLC_FALLBACK: u32 = 0x10;

#[expect(clippy::cast_possible_truncation, reason = "table indexes below 4096")]
const fn uvlc_wide1() -> [u32; 4096] {
    let mut table = [0; 4096];
    let mut index = 0;
    while index < 4096 {
        let (mode, bits) = ((index >> 10) as u32, (index & 0x3FF) as u32);
        let mut entry = UVLC_TABLE1[((mode << 6) + (bits & 0x3F)) as usize] as u32;
        let prefix = entry & 0x7;
        entry >>= 3;
        let suffix = entry & 0xF;
        table[index] = if prefix + suffix > 10 {
            WIDE_UVLC_FALLBACK
        } else {
            let tmp = (bits >> prefix) & ((1 << suffix) - 1);
            entry >>= 4;
            let len = entry & 0x7;
            entry >>= 3;
            let u0 = ((entry & 0x7) + (tmp & !(0xFF << len))) & 0xFFFF;
            let u1 = ((entry >> 3) + (tmp >> len)) & 0xFFFF;
            (prefix + suffix) | (u0 << 8) | (u1 << 20)
        };
        index += 1;
    }
    table
}

/// VLC decoding of the initial quad row ([`VLC_TABLE0`]) and of later rows
/// ([`VLC_TABLE1`]), laid out for the serial decode chain. Each entry is
/// `length | context << 7 | u_offset << 10 | symbol << 16`: the codeword
/// length alone in the low six bits, so a 64-bit shift by the entry consumes
/// the codeword; the bits 7-9 the symbol contributes to the context of the
/// quad after it; the U-VLC offset flag; and the original table symbol.
const VLC_DECODE0: [u32; 1024] = vlc_decode_table(&VLC_TABLE0, true);
const VLC_DECODE1: [u32; 1024] = vlc_decode_table(&VLC_TABLE1, false);

const fn vlc_decode_table(table: &[u16; 1024], initial_row: bool) -> [u32; 1024] {
    let mut decode = [0; 1024];
    let mut index = 0;
    while index < 1024 {
        let symbol = table[index] as u32;
        let context = if initial_row {
            ((symbol & 0x10) << 3) | ((symbol & 0xE0) << 2)
        } else {
            ((symbol & 0x40) << 2) | ((symbol & 0x80) << 1)
        };
        decode[index] = (symbol & 0x7) | context | ((symbol & 0x8) << 7) | (symbol << 16);
        index += 1;
    }
    decode
}

/// The [`VLC_DECODE0`]/[`VLC_DECODE1`] entry for context `c_q` and the next
/// VLC bits. `c_q` holds only bits 7-9, so the mask never changes the index;
/// it only drops the bounds check on the 1024-entry tables.
#[expect(clippy::inline_always, reason = "per-quad hot path")]
#[inline(always)]
fn vlc_entry(table: &[u32; 1024], c_q: u32, vlc_val: u32) -> u32 {
    table[((c_q | (vlc_val & 0x7F)) & 0x3FF) as usize]
}

/// The context bits a decoded VLC entry passes to the next quad.
#[expect(clippy::inline_always, reason = "per-quad hot path")]
#[inline(always)]
fn entry_context(entry: u32) -> u32 {
    entry & 0x380
}

/// The original VLC table symbol of a decoded entry.
#[expect(clippy::inline_always, reason = "per-quad hot path")]
#[inline(always)]
fn entry_symbol(entry: u32) -> u32 {
    entry >> 16
}

/// The U-VLC mode (`u_offset` flags in bits 6 and 7) of a quad pair.
#[expect(clippy::inline_always, reason = "per-pair hot path")]
#[inline(always)]
fn pair_uvlc_mode(entry0: u32, entry1: u32) -> u32 {
    ((entry0 >> 4) & 0x40) | ((entry1 >> 3) & 0x80)
}

/// Decodes one initial-row quad pair's symbols as `[t0, u0, t1, u1]`.
/// `second_quad` is whether its second quad lies inside the block; `c_q`
/// carries the context between pairs. `INLINE_REFILLS` inlines the MEL and
/// VLC refills, so a loop around it makes no call.
#[expect(
    clippy::cast_possible_truncation,
    clippy::inline_always,
    reason = "VLC table fields are defined as 16-bit values and this per-pair decoder is a hot path"
)]
#[inline(always)]
pub(super) fn decode_initial_pair_symbols<const INLINE_REFILLS: bool, V: VlcBits>(
    mel: &mut MelDecoder,
    vlc: &mut V,
    run: &mut i32,
    c_q: &mut u32,
    second_quad: bool,
) -> Option<[u32; 4]> {
    let mut vlc_val = vlc.top_up::<INLINE_REFILLS>();
    let mut e0 = vlc_entry(&VLC_DECODE0, *c_q, vlc_val);
    if *c_q == 0 {
        *run -= 2;
        e0 = if *run == -1 { e0 } else { 0 };
        if *run < 0 {
            *run = mel.get_run_with::<INLINE_REFILLS>()?;
        }
    }
    *c_q = entry_context(e0);
    vlc_val = vlc.advance_entry(e0);

    let mut e1 = vlc_entry(&VLC_DECODE0, *c_q, vlc_val);
    if *c_q == 0 && second_quad {
        *run -= 2;
        e1 = if *run == -1 { e1 } else { 0 };
        if *run < 0 {
            *run = mel.get_run_with::<INLINE_REFILLS>()?;
        }
    }
    if !second_quad {
        e1 = 0;
    }
    *c_q = entry_context(e1);
    vlc_val = vlc.advance_entry(e1);
    let (t0, t1) = (entry_symbol(e0), entry_symbol(e1));

    let mut uvlc_mode = pair_uvlc_mode(e0, e1);
    if uvlc_mode == 0xC0 {
        *run -= 2;
        if *run == -1 {
            uvlc_mode += 0x40;
        }
        if *run < 0 {
            *run = mel.get_run_with::<INLINE_REFILLS>()?;
        }
    }

    let mut uvlc_entry = u32::from(UVLC_TABLE0[(uvlc_mode + (vlc_val & 0x3F)) as usize]);
    vlc_val = vlc.advance(uvlc_entry & 0x7);
    uvlc_entry >>= 3;
    let mut len = uvlc_entry & 0xF;
    let tmp = vlc_val & ((1_u32 << len) - 1);
    let _ = vlc.advance(len);
    uvlc_entry >>= 4;
    len = uvlc_entry & 0x7;
    uvlc_entry >>= 3;
    let u0 = (1 + (uvlc_entry & 0x7) + (tmp & !(0xFF_u32 << len))) as u16;
    let u1 = (1 + (uvlc_entry >> 3) + (tmp >> len)) as u16;

    Some([t0, u32::from(u0), t1, u32::from(u1)])
}

/// Store one pair's `[t0, u0, t1, u1]` symbols at `scratch[row_offset..]`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::inline_always,
    reason = "the symbols are 16-bit table fields; per-pair hot path"
)]
#[inline(always)]
fn store_pair_symbols(scratch: &mut [u16], row_offset: usize, symbols: [u32; 4]) {
    let [t0, u0, t1, u1] = symbols;
    scratch[row_offset] = t0 as u16;
    scratch[row_offset + 1] = u0 as u16;
    scratch[row_offset + 2] = t1 as u16;
    scratch[row_offset + 3] = u1 as u16;
}

/// Decodes the initial-row quad pair whose first quad starts at column `x`,
/// stores its symbols at `scratch[row_offset..row_offset + 4]`, and returns
/// them as `[t0, u0, t1, u1]`. `c_q` carries the context between pairs.
#[expect(
    clippy::inline_always,
    clippy::too_many_arguments,
    reason = "this per-pair decoder is a hot path"
)]
#[inline(always)]
pub(super) fn decode_cleanup_first_row_pair<const INLINE_REFILLS: bool, V: VlcBits>(
    mel: &mut MelDecoder,
    vlc: &mut V,
    run: &mut i32,
    scratch: &mut [u16],
    row_offset: usize,
    c_q: &mut u32,
    x: u32,
    width: u32,
) -> Option<[u32; 4]> {
    let symbols =
        decode_initial_pair_symbols::<INLINE_REFILLS, _>(mel, vlc, run, c_q, x + 2 < width)?;
    store_pair_symbols(scratch, row_offset, symbols);
    Some(symbols)
}

/// Decodes the first (initial) cleanup quad row into `scratch`, advancing the
/// shared MEL/VLC state; the trailing sentinel pair is written after the row.
#[expect(clippy::inline_always, reason = "this per-row decoder is a hot path")]
#[inline(always)]
pub(super) fn decode_cleanup_symbols_first_row<V: VlcBits>(
    mel: &mut MelDecoder,
    vlc: &mut V,
    run: &mut i32,
    scratch: &mut [u16],
    width: u32,
) -> Option<()> {
    let mut c_q = 0u32;
    let mut row_offset = 0usize;
    let mut x = 0u32;

    while x < width {
        decode_cleanup_first_row_pair::<false, _>(
            mel, vlc, run, scratch, row_offset, &mut c_q, x, width,
        )?;
        x += 4;
        row_offset += 4;
    }
    scratch[row_offset] = 0;
    scratch[row_offset + 1] = 0;

    Some(())
}

/// As [`decode_initial_pair_symbols`] for a non-initial row. `above(k)` is
/// the previous quad row's symbol `k` (0, 2 or 4) entries after the one
/// above this pair's first quad.
#[expect(
    clippy::cast_possible_truncation,
    clippy::inline_always,
    reason = "VLC table fields are defined as 16-bit values and this per-pair decoder is a hot path"
)]
#[inline(always)]
pub(super) fn decode_pair_symbols<const INLINE_REFILLS: bool, V: VlcBits>(
    mel: &mut MelDecoder,
    vlc: &mut V,
    run: &mut i32,
    c_q: &mut u32,
    above: impl Fn(usize) -> u32,
    second_quad: bool,
) -> Option<[u32; 4]> {
    // The row above's part of each quad's context, off the decode chain.
    let above_first = ((above(0) & 0xA0) << 2) | ((above(2) & 0x20) << 4);
    let above_second = (above(0) & 0x80) | ((above(2) & 0xA0) << 2) | ((above(4) & 0x20) << 4);
    let above_next = above(2) & 0x80;
    *c_q |= above_first;

    let mut vlc_val = vlc.top_up::<INLINE_REFILLS>();
    let mut e0 = vlc_entry(&VLC_DECODE1, *c_q, vlc_val);
    if *c_q == 0 {
        *run -= 2;
        e0 = if *run == -1 { e0 } else { 0 };
        if *run < 0 {
            *run = mel.get_run_with::<INLINE_REFILLS>()?;
        }
    }

    *c_q = entry_context(e0) | above_second;
    vlc_val = vlc.advance_entry(e0);

    let mut e1 = vlc_entry(&VLC_DECODE1, *c_q, vlc_val);
    if *c_q == 0 && second_quad {
        *run -= 2;
        e1 = if *run == -1 { e1 } else { 0 };
        if *run < 0 {
            *run = mel.get_run_with::<INLINE_REFILLS>()?;
        }
    }
    if !second_quad {
        e1 = 0;
    }

    *c_q = entry_context(e1) | above_next;
    vlc_val = vlc.advance_entry(e1);
    let (t0, t1) = (entry_symbol(e0), entry_symbol(e1));

    // The `u_offset` flags land in index bits 10 and 11, so the index is
    // below 4096 and needs no bounds check.
    let wide = UVLC_WIDE1[((e0 & 0x400) | ((e1 & 0x400) << 1) | (vlc_val & 0x3FF)) as usize];
    if wide & WIDE_UVLC_FALLBACK == 0 {
        let _ = vlc.advance(wide & 0xF);
        return Some([t0, (wide >> 8) & 0xFFF, t1, wide >> 20]);
    }
    let uvlc_mode = pair_uvlc_mode(e0, e1);
    let mut uvlc_entry = u32::from(UVLC_TABLE1[((uvlc_mode + (vlc_val & 0x3F)) & 0xFF) as usize]);
    vlc_val = vlc.advance(uvlc_entry & 0x7);
    uvlc_entry >>= 3;
    let mut len = uvlc_entry & 0xF;
    let tmp = vlc_val & ((1_u32 << len) - 1);
    let _ = vlc.advance(len);
    uvlc_entry >>= 4;
    len = uvlc_entry & 0x7;
    uvlc_entry >>= 3;
    let u0 = ((uvlc_entry & 0x7) + (tmp & !(0xFF_u32 << len))) as u16;
    let u1 = ((uvlc_entry >> 3) + (tmp >> len)) as u16;

    Some([t0, u32::from(u0), t1, u32::from(u1)])
}

/// Decodes the non-initial-row quad pair whose first quad starts at column
/// `x`, reading the previous quad row's symbols `sstr` entries back, stores
/// its symbols at `scratch[row_offset..row_offset + 4]`, and returns them as
/// `[t0, u0, t1, u1]`. `c_q` carries the context between pairs.
#[expect(
    clippy::inline_always,
    clippy::too_many_arguments,
    reason = "this per-pair decoder is a hot path"
)]
#[inline(always)]
pub(super) fn decode_cleanup_row_pair<const INLINE_REFILLS: bool, V: VlcBits>(
    mel: &mut MelDecoder,
    vlc: &mut V,
    run: &mut i32,
    scratch: &mut [u16],
    row_offset: usize,
    sstr: usize,
    c_q: &mut u32,
    x: u32,
    width: u32,
) -> Option<[u32; 4]> {
    // `sstr >= 8`, so the row above never overlaps this pair's symbols.
    let prev = row_offset - sstr;
    let above = |k| u32::from(scratch[prev + k]);
    let symbols =
        decode_pair_symbols::<INLINE_REFILLS, _>(mel, vlc, run, c_q, above, x + 2 < width)?;
    store_pair_symbols(scratch, row_offset, symbols);
    Some(symbols)
}

/// Decodes one non-initial cleanup quad row (`y >= 2`, even) into `scratch`,
/// reading the previous quad row's context and advancing the shared MEL/VLC
/// state; the trailing sentinel pair is written after the row.
#[expect(clippy::inline_always, reason = "this per-row decoder is a hot path")]
#[inline(always)]
pub(super) fn decode_cleanup_symbols_row<V: VlcBits>(
    mel: &mut MelDecoder,
    vlc: &mut V,
    run: &mut i32,
    scratch: &mut [u16],
    width: u32,
    y: u32,
    sstr: usize,
) -> Option<()> {
    let mut x = 0u32;
    let mut c_q = 0u32;
    let mut row_offset = (y >> 1) as usize * sstr;

    while x < width {
        decode_cleanup_row_pair::<false, _>(
            mel, vlc, run, scratch, row_offset, sstr, &mut c_q, x, width,
        )?;
        x += 4;
        row_offset += 4;
    }

    scratch[row_offset] = 0;
    scratch[row_offset + 1] = 0;

    Some(())
}

#[inline(never)]
pub(super) fn decode_cleanup_symbols(
    coded_data: &[u8],
    lcup: usize,
    scup: usize,
    width: u32,
    height: u32,
    sstr: usize,
    scratch: &mut [u16],
) -> Option<()> {
    let quad_rows = height.div_ceil(2) as usize;
    if scratch.len() < sstr * (quad_rows + 1) {
        return None;
    }

    let mut mel = MelDecoder::new(coded_data, lcup, scup);
    let mut vlc = ReverseBitReader::new_vlc(coded_data, lcup, scup);
    let mut run = mel.get_run()?;

    decode_cleanup_symbols_first_row(&mut mel, &mut vlc, &mut run, scratch, width)?;

    for y in (2..height).step_by(2) {
        decode_cleanup_symbols_row(&mut mel, &mut vlc, &mut run, scratch, width, y, sstr)?;
    }

    Some(())
}
