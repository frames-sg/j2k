// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::cleanup::{decode_cleanup_symbols_first_row, decode_cleanup_symbols_row};
use super::readers::{unstuff_magsgn, FlatBitReader, MelDecoder, ReverseBitReader};
use super::stats::HtDecodeObserver;
#[cfg(feature = "simd")]
use crate::math::{dispatch, Level};

#[cfg(feature = "simd")]
mod simd;

/// Whether the SIMD rows handle a block with this many missing MSBs.
#[cfg(feature = "simd")]
fn uses_simd_rows(missing_msbs: u32) -> bool {
    missing_msbs + 2 <= simd::MAX_SIMD_MMSBP2
}

#[cfg(feature = "simd")]
pub(super) fn supports_cleanup_to_f32(missing_msbs: u32, width: u32) -> bool {
    width != 0 && width.is_multiple_of(4) && uses_simd_rows(missing_msbs)
}

fn sample_mask(bit: u32) -> u32 {
    1 << (4 + bit)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::inline_always,
    reason = "31-bit decoded magnitudes; inlined so the `&mut` MagSgn reader stays in registers"
)]
#[inline(always)]
fn decode_mag_sgn_sample_with_vn(
    magsgn: &mut FlatBitReader<'_>,
    inf: u32,
    bit: u32,
    uq: u32,
    p: u32,
) -> (u32, u32) {
    // Branch-free: an insignificant sample consumes no bits and yields zero,
    // exactly as an early return would, without a data-dependent branch.
    let significant = 0_u32.wrapping_sub((inf & sample_mask(bit)) >> (4 + bit));
    let ms_val = magsgn.fetch();
    let m_n = (uq - ((inf >> (12 + bit)) & 1)) & significant;
    magsgn.advance(m_n);

    // `m_n` is at most 32 here; shifting in 64 bits covers 0 and 32 without
    // a case split, and the implicit one above bit 31 drops out as before.
    let mask = ((1_u64 << m_n) - 1) as u32;
    let e1 = (u64::from((inf >> (8 + bit)) & 1) << m_n) as u32;
    let v_n = ((ms_val & mask) | e1 | 1) & significant;
    let magnitude = if p == 0 {
        ((u64::from(v_n) + 1) >> 1) as u32
    } else {
        (v_n + 2) << (p - 1)
    };
    (((ms_val << 31) | magnitude) & significant, v_n)
}

#[inline(never)]
#[expect(
    clippy::too_many_arguments,
    reason = "the phase entry point keeps independent bitstream, geometry, and scratch inputs explicit"
)]
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
    magsgn_buffer: &mut Vec<u8>,
) -> Option<()> {
    let v_n_width = width.div_ceil(2) as usize + 2;
    if v_n_scratch.len() < v_n_width {
        return None;
    }
    v_n_scratch[..v_n_width].fill(0);
    let consumable = (width as usize)
        .checked_mul(height as usize)?
        .checked_mul(4)?;
    let flat = unstuff_magsgn(&coded_data[..lcup - scup], magsgn_buffer, consumable)?;
    let mut magsgn = FlatBitReader::new(flat)?;
    #[cfg(feature = "simd")]
    if uses_simd_rows(missing_msbs) {
        return dispatch!(Level::new(), simd => simd::decode_magnitude_sign_rows(
            simd,
            &mut magsgn,
            scratch,
            decoded_data,
            v_n_scratch,
            missing_msbs,
            width,
            height,
            stride,
            sstr,
        ));
    }
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

#[expect(
    clippy::inline_always,
    clippy::too_many_arguments,
    reason = "this stable per-quad hot helper keeps scan cursors explicit and must inline into both row walkers"
)]
#[inline(always)]
fn decode_magnitude_sign_pair_from_cleanup(
    magsgn: &mut FlatBitReader<'_>,
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

    if inf & 0xF0 == 0 {
        // No significant sample: no MagSgn bits, zero output and zero `v_n`,
        // exactly what the per-sample path produces for this quad.
        decoded_data[*dp] = 0;
        if second_row_present {
            decoded_data[*dp + stride] = 0;
        }
        v_n_scratch[*vp] = *prev_v_n;
        *prev_v_n = 0;
        *dp += 1;
        *x += 1;
        if *x < width {
            decoded_data[*dp] = 0;
            if second_row_present {
                decoded_data[*dp + stride] = 0;
            }
            *dp += 1;
            *x += 1;
        }
        *vp += 1;
        return Some(());
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

#[expect(
    clippy::inline_always,
    clippy::too_many_arguments,
    reason = "this stable first-row hot path mirrors the cleanup scan state without packaging transient cursors"
)]
#[inline(always)]
pub(super) fn decode_magnitude_sign_first_row_from_cleanup(
    magsgn: &mut FlatBitReader<'_>,
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

#[expect(
    clippy::inline_always,
    clippy::too_many_arguments,
    reason = "this stable row hot path mirrors the cleanup scan state without packaging transient cursors"
)]
#[inline(always)]
pub(super) fn decode_magnitude_sign_row_from_cleanup(
    magsgn: &mut FlatBitReader<'_>,
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
        emax = (emax | 2).ilog2();
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
#[expect(
    clippy::too_many_arguments,
    reason = "the fused cleanup/magnitude phase keeps bitstream, geometry, scratch, and observer state explicit"
)]
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
    magsgn_buffer: &mut Vec<u8>,
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
    let consumable = (width as usize)
        .checked_mul(height as usize)?
        .checked_mul(4)?;
    let flat = unstuff_magsgn(&coded_data[..lcup - scup], magsgn_buffer, consumable)?;
    let mut magsgn = FlatBitReader::new(flat)?;
    let mut run = mel.get_run()?;

    #[cfg(feature = "simd")]
    if uses_simd_rows(missing_msbs) {
        return dispatch!(Level::new(), simd => simd::decode_cleanup_and_magnitude_sign_rows(
            simd,
            &mut mel,
            &mut vlc,
            &mut run,
            &mut magsgn,
            scratch,
            decoded_data,
            v_n_scratch,
            missing_msbs,
            width,
            height,
            stride,
            sstr,
            observer,
        ));
    }

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

#[cfg(feature = "simd")]
#[inline(never)]
#[expect(
    clippy::too_many_arguments,
    reason = "the direct cleanup entry keeps bitstream, geometry, scratch, and output conversion explicit"
)]
pub(super) fn decode_cleanup_and_magnitude_sign_to_f32<const IRREVERSIBLE_MIDPOINT: bool>(
    coded_data: &[u8],
    lcup: usize,
    scup: usize,
    decoded_data: &mut [f32],
    missing_msbs: u32,
    coded_bitplanes: u8,
    dequantization_step: f32,
    width: u32,
    height: u32,
    stride: u32,
    sstr: usize,
    scratch: &mut [u16],
    v_n_scratch: &mut [u32],
    magsgn_buffer: &mut Vec<u8>,
) -> Option<()> {
    if !supports_cleanup_to_f32(missing_msbs, width) {
        return None;
    }
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
    let consumable = (width as usize)
        .checked_mul(height as usize)?
        .checked_mul(4)?;
    let flat = unstuff_magsgn(&coded_data[..lcup - scup], magsgn_buffer, consumable)?;
    let mut magsgn = FlatBitReader::new(flat)?;
    let mut run = mel.get_run()?;
    let output = simd::FinalOutput {
        coefficient_shift: 31 - u32::from(coded_bitplanes),
        midpoint_scale: crate::math::pow2i(-(31_i32 - i32::from(coded_bitplanes))),
        dequantization_step,
    };

    dispatch!(Level::new(), simd => simd::decode_cleanup_and_magnitude_sign_to_f32_rows::<IRREVERSIBLE_MIDPOINT, _>(
        simd,
        &mut mel,
        &mut vlc,
        &mut run,
        &mut magsgn,
        scratch,
        decoded_data,
        v_n_scratch,
        missing_msbs,
        width,
        height,
        stride,
        sstr,
        output,
    ))
}
