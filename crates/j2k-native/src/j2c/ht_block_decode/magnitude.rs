// SPDX-License-Identifier: MIT OR Apache-2.0

use super::cleanup::{decode_cleanup_symbols_first_row, decode_cleanup_symbols_row};
use super::readers::{ForwardBitReader, MelDecoder, ReverseBitReader};
use super::state::HtDecodeObserver;

fn sample_mask(bit: u32) -> u32 {
    1 << (4 + bit)
}

fn decode_mag_sgn_sample_with_vn(
    magsgn: &mut ForwardBitReader<0xFF>,
    inf: u32,
    bit: u32,
    uq: u32,
    p: u32,
) -> (u32, u32) {
    if (inf & sample_mask(bit)) == 0 {
        return (0, 0);
    }

    let ms_val = magsgn.fetch();
    let m_n = uq - ((inf >> (12 + bit)) & 1);
    magsgn.advance(m_n);

    let mut value = ms_val << 31;
    let mask = match m_n {
        0 => 0,
        32.. => u32::MAX,
        bits => (1_u32 << bits) - 1,
    };
    let mut v_n = ms_val & mask;
    if m_n < 32 {
        v_n |= ((inf >> (8 + bit)) & 1) << m_n;
    }
    v_n |= 1;
    if p == 0 {
        value |= ((u64::from(v_n) + 1) >> 1) as u32;
    } else {
        value |= (v_n + 2) << (p - 1);
    }
    (value, v_n)
}

#[inline(never)]
pub(super) fn decode_magnitude_sign_phase(
    coded_data: &[u8],
    lcup: usize,
    scup: usize,
    scratch: &[u16],
    decoded_data: &mut [u32],
    missing_msbs: u32,
    width: u32,
    height: u32,
    stride: u32,
    sstr: usize,
    v_n_scratch: &mut [u32],
) -> Option<()> {
    let v_n_width = width.div_ceil(2) as usize + 2;
    if v_n_scratch.len() < v_n_width {
        return None;
    }
    v_n_scratch[..v_n_width].fill(0);

    let mut magsgn = ForwardBitReader::<0xFF>::new(&coded_data[..lcup - scup]);

    decode_magnitude_sign_first_row_from_cleanup(
        &mut magsgn,
        scratch,
        decoded_data,
        v_n_scratch,
        missing_msbs,
        width,
        height,
        stride,
    )?;

    for y in (2..height).step_by(2) {
        decode_magnitude_sign_row_from_cleanup(
            &mut magsgn,
            scratch,
            decoded_data,
            v_n_scratch,
            missing_msbs,
            width,
            height,
            y,
            stride,
            sstr,
        )?;
    }

    Some(())
}

#[inline(always)]
fn decode_magnitude_sign_pair_from_cleanup(
    magsgn: &mut ForwardBitReader<0xFF>,
    decoded_data: &mut [u32],
    v_n_scratch: &mut [u32],
    inf: u32,
    uq: u32,
    mmsbp2: u32,
    p: u32,
    width: u32,
    stride: usize,
    second_row_present: bool,
    x: &mut u32,
    dp: &mut usize,
    vp: &mut usize,
    prev_v_n: &mut u32,
) -> Option<()> {
    if uq > mmsbp2 {
        return None;
    }

    let (val0, _) = decode_mag_sgn_sample_with_vn(magsgn, inf, 0, uq, p);
    decoded_data[*dp] = val0;

    let (val1, v_n1) = decode_mag_sgn_sample_with_vn(magsgn, inf, 1, uq, p);
    if second_row_present {
        decoded_data[*dp + stride] = val1;
    }
    v_n_scratch[*vp] = *prev_v_n | v_n1;
    *prev_v_n = 0;
    *dp += 1;
    *x += 1;

    if *x >= width {
        *vp += 1;
        return Some(());
    }

    let (val2, _) = decode_mag_sgn_sample_with_vn(magsgn, inf, 2, uq, p);
    decoded_data[*dp] = val2;

    let (val3, v_n3) = decode_mag_sgn_sample_with_vn(magsgn, inf, 3, uq, p);
    if second_row_present {
        decoded_data[*dp + stride] = val3;
    }
    *prev_v_n = v_n3;
    *dp += 1;
    *x += 1;
    *vp += 1;

    Some(())
}

#[inline(always)]
pub(super) fn decode_magnitude_sign_first_row_from_cleanup(
    magsgn: &mut ForwardBitReader<0xFF>,
    cleanup: &[u16],
    decoded_data: &mut [u32],
    v_n_scratch: &mut [u32],
    missing_msbs: u32,
    width: u32,
    height: u32,
    stride: u32,
) -> Option<()> {
    let p = 30 - missing_msbs;
    let mmsbp2 = missing_msbs + 2;
    let stride = stride as usize;
    let second_row_present = height > 1;
    let mut prev_v_n = 0u32;
    let mut x = 0u32;
    let mut sp = 0usize;
    let mut vp = 0usize;
    let mut dp = 0usize;

    while x < width {
        let inf = u32::from(cleanup[sp]);
        let uq = u32::from(cleanup[sp + 1]);
        decode_magnitude_sign_pair_from_cleanup(
            magsgn,
            decoded_data,
            v_n_scratch,
            inf,
            uq,
            mmsbp2,
            p,
            width,
            stride,
            second_row_present,
            &mut x,
            &mut dp,
            &mut vp,
            &mut prev_v_n,
        )?;
        sp += 2;
    }
    v_n_scratch[vp] = prev_v_n;

    Some(())
}

#[inline(always)]
pub(super) fn decode_magnitude_sign_row_from_cleanup(
    magsgn: &mut ForwardBitReader<0xFF>,
    cleanup: &[u16],
    decoded_data: &mut [u32],
    v_n_scratch: &mut [u32],
    missing_msbs: u32,
    width: u32,
    height: u32,
    y: u32,
    stride: u32,
    sstr: usize,
) -> Option<()> {
    let p = 30 - missing_msbs;
    let mmsbp2 = missing_msbs + 2;
    let row_base = (y >> 1) as usize * sstr;
    let stride = stride as usize;
    let mut sp = row_base;
    let mut vp = 0usize;
    let mut dp = y as usize * stride;
    let mut prev_v_n = 0u32;
    let mut x = 0u32;
    let second_row_present = y + 1 < height;

    while x < width {
        let inf = u32::from(cleanup[sp]);
        let u_q = u32::from(cleanup[sp + 1]);
        let mut gamma = inf & 0xF0;
        gamma &= gamma.wrapping_sub(0x10);
        let mut emax = v_n_scratch[vp] | v_n_scratch[vp + 1];
        emax = 31 - (emax | 2).leading_zeros();
        let kappa = if gamma != 0 { emax } else { 1 };
        let uq = u_q + kappa;

        decode_magnitude_sign_pair_from_cleanup(
            magsgn,
            decoded_data,
            v_n_scratch,
            inf,
            uq,
            mmsbp2,
            p,
            width,
            stride,
            second_row_present,
            &mut x,
            &mut dp,
            &mut vp,
            &mut prev_v_n,
        )?;
        sp += 2;
    }

    v_n_scratch[vp] = prev_v_n;

    Some(())
}

#[inline(never)]
pub(super) fn decode_cleanup_and_magnitude_sign_phase(
    coded_data: &[u8],
    lcup: usize,
    scup: usize,
    decoded_data: &mut [u32],
    missing_msbs: u32,
    width: u32,
    height: u32,
    stride: u32,
    sstr: usize,
    scratch: &mut [u16],
    v_n_scratch: &mut [u32],
    observer: &mut impl HtDecodeObserver,
) -> Option<()> {
    let quad_rows = height.div_ceil(2) as usize;
    if scratch.len() < sstr * (quad_rows + 1) {
        return None;
    }
    let v_n_width = width.div_ceil(2) as usize + 2;
    if v_n_scratch.len() < v_n_width {
        return None;
    }
    v_n_scratch[..v_n_width].fill(0);

    let mut mel = MelDecoder::new(coded_data, lcup, scup);
    let mut vlc = ReverseBitReader::new_vlc(coded_data, lcup, scup);
    let mut magsgn = ForwardBitReader::<0xFF>::new(&coded_data[..lcup - scup]);
    let mut run = mel.get_run()?;

    let phase_start = observer.phase_start();
    decode_cleanup_symbols_first_row(&mut mel, &mut vlc, &mut run, scratch, width)?;
    observer.add_cleanup_us(phase_start);

    let phase_start = observer.phase_start();
    decode_magnitude_sign_first_row_from_cleanup(
        &mut magsgn,
        scratch,
        decoded_data,
        v_n_scratch,
        missing_msbs,
        width,
        height,
        stride,
    )?;
    observer.add_mag_sgn_us(phase_start);

    for y in (2..height).step_by(2) {
        let phase_start = observer.phase_start();
        decode_cleanup_symbols_row(&mut mel, &mut vlc, &mut run, scratch, width, y, sstr)?;
        observer.add_cleanup_us(phase_start);

        let phase_start = observer.phase_start();
        decode_magnitude_sign_row_from_cleanup(
            &mut magsgn,
            scratch,
            decoded_data,
            v_n_scratch,
            missing_msbs,
            width,
            height,
            y,
            stride,
            sstr,
        )?;
        observer.add_mag_sgn_us(phase_start);
    }

    Some(())
}
