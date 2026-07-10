// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::ht_tables::{UVLC_TABLE0, UVLC_TABLE1, VLC_TABLE0, VLC_TABLE1};
use super::readers::{MelDecoder, ReverseBitReader};

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

/// Decodes the first (initial) cleanup quad row into `scratch`, advancing the
/// shared MEL/VLC state; the trailing sentinel pair is written after the row.
#[expect(
    clippy::cast_possible_truncation,
    clippy::inline_always,
    reason = "VLC table fields are defined as 16-bit values and this per-row decoder is a hot path"
)]
#[inline(always)]
pub(super) fn decode_cleanup_symbols_first_row(
    mel: &mut MelDecoder,
    vlc: &mut ReverseBitReader,
    run: &mut i32,
    scratch: &mut [u16],
    width: u32,
) -> Option<()> {
    let mut c_q = 0u32;
    let mut row_offset = 0usize;
    let mut x = 0u32;

    while x < width {
        let mut vlc_val = vlc.fetch();
        let mut t0 = u32::from(VLC_TABLE0[(c_q + (vlc_val & 0x7F)) as usize]);
        if c_q == 0 {
            *run -= 2;
            t0 = if *run == -1 { t0 } else { 0 };
            if *run < 0 {
                *run = mel.get_run()?;
            }
        }
        scratch[row_offset] = t0 as u16;
        x += 2;
        c_q = ((t0 & 0x10) << 3) | ((t0 & 0xE0) << 2);
        vlc_val = vlc.advance(t0 & 0x7);

        let mut t1 = u32::from(VLC_TABLE0[(c_q + (vlc_val & 0x7F)) as usize]);
        if c_q == 0 && x < width {
            *run -= 2;
            t1 = if *run == -1 { t1 } else { 0 };
            if *run < 0 {
                *run = mel.get_run()?;
            }
        }
        if x >= width {
            t1 = 0;
        }
        scratch[row_offset + 2] = t1 as u16;
        x += 2;
        c_q = ((t1 & 0x10) << 3) | ((t1 & 0xE0) << 2);
        vlc_val = vlc.advance(t1 & 0x7);

        let mut uvlc_mode = ((t0 & 0x8) << 3) | ((t1 & 0x8) << 4);
        if uvlc_mode == 0xC0 {
            *run -= 2;
            if *run == -1 {
                uvlc_mode += 0x40;
            }
            if *run < 0 {
                *run = mel.get_run()?;
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
        scratch[row_offset + 1] = (1 + (uvlc_entry & 0x7) + (tmp & !(0xFF_u32 << len))) as u16;
        scratch[row_offset + 3] = (1 + (uvlc_entry >> 3) + (tmp >> len)) as u16;

        row_offset += 4;
    }
    scratch[row_offset] = 0;
    scratch[row_offset + 1] = 0;

    Some(())
}

/// Decodes one non-initial cleanup quad row (`y >= 2`, even) into `scratch`,
/// reading the previous quad row's context and advancing the shared MEL/VLC
/// state; the trailing sentinel pair is written after the row.
#[expect(
    clippy::cast_possible_truncation,
    clippy::inline_always,
    reason = "VLC table fields are defined as 16-bit values and this per-row decoder is a hot path"
)]
#[inline(always)]
pub(super) fn decode_cleanup_symbols_row(
    mel: &mut MelDecoder,
    vlc: &mut ReverseBitReader,
    run: &mut i32,
    scratch: &mut [u16],
    width: u32,
    y: u32,
    sstr: usize,
) -> Option<()> {
    let row_base = (y >> 1) as usize * sstr;
    let prev_base = row_base - sstr;
    let mut x = 0u32;
    let mut c_q = 0u32;
    let mut row_offset = row_base;

    while x < width {
        c_q |= (u32::from(scratch[prev_base + (row_offset - row_base)]) & 0xA0) << 2;
        c_q |= (u32::from(scratch[prev_base + (row_offset - row_base) + 2]) & 0x20) << 4;

        let mut vlc_val = vlc.fetch();
        let mut t0 = u32::from(VLC_TABLE1[(c_q + (vlc_val & 0x7F)) as usize]);
        if c_q == 0 {
            *run -= 2;
            t0 = if *run == -1 { t0 } else { 0 };
            if *run < 0 {
                *run = mel.get_run()?;
            }
        }
        scratch[row_offset] = t0 as u16;
        x += 2;

        c_q = ((t0 & 0x40) << 2) | ((t0 & 0x80) << 1);
        c_q |= u32::from(scratch[prev_base + (row_offset - row_base)]) & 0x80;
        c_q |= (u32::from(scratch[prev_base + (row_offset - row_base) + 2]) & 0xA0) << 2;
        c_q |= (u32::from(scratch[prev_base + (row_offset - row_base) + 4]) & 0x20) << 4;
        vlc_val = vlc.advance(t0 & 0x7);

        let mut t1 = u32::from(VLC_TABLE1[(c_q + (vlc_val & 0x7F)) as usize]);
        if c_q == 0 && x < width {
            *run -= 2;
            t1 = if *run == -1 { t1 } else { 0 };
            if *run < 0 {
                *run = mel.get_run()?;
            }
        }
        if x >= width {
            t1 = 0;
        }
        scratch[row_offset + 2] = t1 as u16;
        x += 2;

        c_q = ((t1 & 0x40) << 2) | ((t1 & 0x80) << 1);
        c_q |= u32::from(scratch[prev_base + (row_offset - row_base) + 2]) & 0x80;
        vlc_val = vlc.advance(t1 & 0x7);

        let uvlc_mode = ((t0 & 0x8) << 3) | ((t1 & 0x8) << 4);
        let mut uvlc_entry = u32::from(UVLC_TABLE1[(uvlc_mode + (vlc_val & 0x3F)) as usize]);
        vlc_val = vlc.advance(uvlc_entry & 0x7);
        uvlc_entry >>= 3;
        let mut len = uvlc_entry & 0xF;
        let tmp = vlc_val & ((1_u32 << len) - 1);
        let _ = vlc.advance(len);
        uvlc_entry >>= 4;
        len = uvlc_entry & 0x7;
        uvlc_entry >>= 3;
        scratch[row_offset + 1] = ((uvlc_entry & 0x7) + (tmp & !(0xFF_u32 << len))) as u16;
        scratch[row_offset + 3] = ((uvlc_entry >> 3) + (tmp >> len)) as u16;

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
