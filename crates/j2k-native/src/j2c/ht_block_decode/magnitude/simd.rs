// SPDX-License-Identifier: MIT OR Apache-2.0

//! `MagSgn` decoding of two quads at a time in 4-lane vectors.
//!
//! A quad's lanes follow its sample order: (x, y), (x, y + 1), (x + 1, y),
//! (x + 1, y + 1). The quad's bit lengths are prefix-summed, so all four
//! samples are gathered from one 16-byte window of the unstuffed segment
//! instead of one after another. Each lane computes exactly what
//! [`super::decode_mag_sgn_sample_with_vn`] computes for that sample.

use fearless_simd::{f32x4, i32x4, u32x4, u8x16, Bytes, Simd, SimdBase};

use super::super::cleanup::{
    decode_cleanup_first_row_pair, decode_cleanup_row_pair, decode_cleanup_symbols_first_row,
    decode_cleanup_symbols_row,
};
use super::super::readers::{FlatBitReader, MelDecoder, ReverseBitReader};
use super::super::stats::HtDecodeObserver;
use super::decode_magnitude_sign_pair_from_cleanup;

/// Largest per-sample bit budget (`missing_msbs + 2`) decoded here. A
/// quad's last lane then starts at most `7 + 3 * 25` bits into its window
/// (byte 10), and every lane needs at most `7 + 25` bits (four bytes), so a
/// 16-byte window always holds the whole quad. Larger budgets also imply
/// `p >= 7`, so the 31-bit `p == 0` reconstruction never occurs here.
pub(super) const MAX_SIMD_MMSBP2: u32 = 25;

/// Set-bit counts for the sixteen 4-bit values, packed into nibbles so the
/// `MagSgn` position chain does not wait on two scalar table loads per quad.
const NIBBLE_ONES: u64 = 0x4332_3221_3221_2110;

#[expect(clippy::inline_always, reason = "per-quad position update")]
#[inline(always)]
fn nibble_ones(value: u32) -> u32 {
    ((NIBBLE_ONES >> (value << 2)) & 0xF) as u32
}

#[derive(Clone, Copy)]
struct Lanes<S: Simd> {
    simd: S,
    zero: u32x4<S>,
    sig_shift: u32x4<S>,
    emb_k_shift: u32x4<S>,
    emb_1_shift: u32x4<S>,
    byte_spread: u32x4<S>,
    byte_offsets: u32x4<S>,
}

#[derive(Clone, Copy)]
pub(super) struct FinalOutput {
    pub(super) coefficient_shift: u32,
    pub(super) midpoint_scale: f32,
    pub(super) dequantization_step: f32,
}

impl<S: Simd> Lanes<S> {
    #[expect(
        clippy::inline_always,
        reason = "constant setup hoisted into row loops"
    )]
    #[inline(always)]
    fn new(simd: S) -> Self {
        Self {
            simd,
            zero: u32x4::splat(simd, 0),
            sig_shift: u32x4::from_slice(simd, &[4, 5, 6, 7]),
            emb_k_shift: u32x4::from_slice(simd, &[12, 13, 14, 15]),
            emb_1_shift: u32x4::from_slice(simd, &[8, 9, 10, 11]),
            byte_spread: u32x4::splat(simd, 0x0101_0101),
            byte_offsets: u32x4::splat(simd, 0x0302_0100),
        }
    }
}

/// Bits a quad consumes: `uq` per significant sample, less each of their
/// `e_k` bits. Matches the sum of the lanes' `m_n`.
#[expect(clippy::inline_always, reason = "per-quad position update")]
#[inline(always)]
fn quad_bits(inf: u32, uq: u32) -> usize {
    let rho = (inf >> 4) & 0xF;
    let emb_k = (inf >> 12) & rho;
    let significant = nibble_ones(rho);
    let refined = nibble_ones(emb_k);
    (uq * significant - refined) as usize
}

/// One quad's decoded samples and `v_n` values, starting at bit `position`.
#[expect(clippy::inline_always, reason = "per-quad hot path")]
#[inline(always)]
fn decode_quad<S: Simd>(
    lanes: &Lanes<S>,
    magsgn: &FlatBitReader<'_>,
    position: usize,
    inf: u32,
    uq: u32,
    p_shift: u32,
) -> (u32x4<S>, u32x4<S>) {
    let simd = lanes.simd;
    let inf_v = u32x4::splat(simd, inf);
    let significant = lanes.zero - ((inf_v >> lanes.sig_shift) & 1);
    let emb_k = (inf_v >> lanes.emb_k_shift) & 1;
    let emb_1 = (inf_v >> lanes.emb_1_shift) & 1;
    let m_n = (u32x4::splat(simd, uq) - emb_k) & significant;

    // Exclusive prefix sum: each lane's bit offset within the quad.
    let shifted = lanes.zero.slide::<3>(m_n);
    let pairs = shifted + lanes.zero.slide::<3>(shifted);
    let offsets = pairs + lanes.zero.slide::<2>(pairs);

    let (window, first_bit) = magsgn.window16(position);
    let bits = offsets + first_bit;
    let indices = (bits >> 3) * lanes.byte_spread + lanes.byte_offsets;
    let window = u8x16::from_slice(simd, window);
    let gathered: u32x4<S> = window.swizzle_dyn(indices.bitcast::<u8x16<S>>()).bitcast();
    let ms_val = gathered >> (bits & 7);

    // `1 << m_n` with `m_n <= 32`, split so no lane shifts by 32 or more.
    let half = m_n >> 1;
    let rest = m_n - half;
    let mask = !((!lanes.zero << half) << rest);
    let e1 = (emb_1 << half) << rest;
    let v_n = ((ms_val & mask) | e1 | 1) & significant;
    let magnitude = (v_n + 2) << p_shift;
    (((ms_val << 31) | magnitude) & significant, v_n)
}

/// Decode the two quads at columns `x..x + 4`, write both output rows, and
/// fold the bottom-row `v_n` values into the scratch as the scalar path does.
#[expect(
    clippy::inline_always,
    clippy::too_many_arguments,
    reason = "two-quad hot step shared by the first and later row walkers"
)]
#[inline(always)]
fn decode_two_quads<S: Simd>(
    lanes: &Lanes<S>,
    magsgn: &mut FlatBitReader<'_>,
    decoded_data: &mut [u32],
    v_n_scratch: &mut [u32],
    quads: [(u32, u32); 2],
    mmsbp2: u32,
    p_shift: u32,
    stride: usize,
    second_row_present: bool,
    dp: usize,
    vp: usize,
    carry: &mut u32,
) -> Option<()> {
    let [(inf0, uq0), (inf1, uq1)] = quads;
    if uq0 > mmsbp2 || uq1 > mmsbp2 {
        return None;
    }

    if (inf0 | inf1) & 0xF0 == 0 {
        store_empty_pair(
            decoded_data,
            v_n_scratch,
            stride,
            second_row_present,
            dp,
            vp,
            carry,
        );
        return Some(());
    }

    let first = magsgn.position();
    let second = first + quad_bits(inf0, uq0);
    let (samples0, v_n0) = decode_quad(lanes, magsgn, first, inf0, uq0, p_shift);
    let (samples1, v_n1) = decode_quad(lanes, magsgn, second, inf1, uq1, p_shift);
    magsgn.advance_to(second + quad_bits(inf1, uq1));

    samples0
        .unzip_low(samples1)
        .store_slice(&mut decoded_data[dp..dp + 4]);
    if second_row_present {
        samples0
            .unzip_high(samples1)
            .store_slice(&mut decoded_data[dp + stride..dp + stride + 4]);
    }
    let [c0, c1, c2, c3] = v_n0.unzip_high(v_n1).to_array();
    v_n_scratch[vp] = *carry | c0;
    v_n_scratch[vp + 1] = c1 | c2;
    *carry = c3;
    Some(())
}

#[expect(
    clippy::inline_always,
    reason = "per-pair final coefficient conversion"
)]
#[inline(always)]
fn final_samples<const IRREVERSIBLE_MIDPOINT: bool, S: Simd>(
    lanes: &Lanes<S>,
    samples: u32x4<S>,
    output: FinalOutput,
) -> f32x4<S> {
    let magnitude = samples & 0x7FFF_FFFF;
    let values = if IRREVERSIBLE_MIDPOINT {
        let magnitude = lanes.simd.cvt_f32_u32x4(magnitude) * output.midpoint_scale;
        let signed = magnitude.bitcast::<u32x4<S>>() | (samples & 0x8000_0000);
        signed.bitcast::<f32x4<S>>()
    } else {
        let magnitude = magnitude >> output.coefficient_shift;
        let sign = lanes.zero - (samples >> 31);
        let signed = ((magnitude ^ sign) - sign).bitcast::<i32x4<S>>();
        lanes.simd.cvt_f32_i32x4(signed)
    };
    values * output.dequantization_step
}

/// Decode a full-width pair directly into the final coefficient plane. This
/// is used only by whole-image cleanup-only decode, where no refinement pass
/// needs the intermediate sign-magnitude words.
#[expect(
    clippy::inline_always,
    clippy::too_many_arguments,
    reason = "two-quad direct-output hot step"
)]
#[inline(always)]
fn decode_two_quads_to_f32<const IRREVERSIBLE_MIDPOINT: bool, S: Simd>(
    lanes: &Lanes<S>,
    magsgn: &mut FlatBitReader<'_>,
    decoded_data: &mut [f32],
    v_n_scratch: &mut [u32],
    quads: [(u32, u32); 2],
    mmsbp2: u32,
    p_shift: u32,
    stride: usize,
    second_row_present: bool,
    dp: usize,
    vp: usize,
    carry: &mut u32,
    output: FinalOutput,
) -> Option<()> {
    let [(inf0, uq0), (inf1, uq1)] = quads;
    if uq0 > mmsbp2 || uq1 > mmsbp2 {
        return None;
    }

    if (inf0 | inf1) & 0xF0 == 0 {
        let zero = final_samples::<IRREVERSIBLE_MIDPOINT, _>(lanes, lanes.zero, output);
        zero.store_slice(&mut decoded_data[dp..dp + 4]);
        if second_row_present {
            zero.store_slice(&mut decoded_data[dp + stride..dp + stride + 4]);
        }
        v_n_scratch[vp] = *carry;
        v_n_scratch[vp + 1] = 0;
        *carry = 0;
        return Some(());
    }

    let first = magsgn.position();
    let second = first + quad_bits(inf0, uq0);
    let (samples0, v_n0) = decode_quad(lanes, magsgn, first, inf0, uq0, p_shift);
    let (samples1, v_n1) = decode_quad(lanes, magsgn, second, inf1, uq1, p_shift);
    magsgn.advance_to(second + quad_bits(inf1, uq1));

    final_samples::<IRREVERSIBLE_MIDPOINT, _>(lanes, samples0.unzip_low(samples1), output)
        .store_slice(&mut decoded_data[dp..dp + 4]);
    if second_row_present {
        final_samples::<IRREVERSIBLE_MIDPOINT, _>(lanes, samples0.unzip_high(samples1), output)
            .store_slice(&mut decoded_data[dp + stride..dp + stride + 4]);
    }
    let [c0, c1, c2, c3] = v_n0.unzip_high(v_n1).to_array();
    v_n_scratch[vp] = *carry | c0;
    v_n_scratch[vp + 1] = c1 | c2;
    *carry = c3;
    Some(())
}

/// Store a quad pair with no significant sample: zero output and zero `v_n`.
#[expect(clippy::inline_always, reason = "per-pair hot path")]
#[inline(always)]
fn store_empty_pair(
    decoded_data: &mut [u32],
    v_n_scratch: &mut [u32],
    stride: usize,
    second_row_present: bool,
    dp: usize,
    vp: usize,
    carry: &mut u32,
) {
    decoded_data[dp..dp + 4].fill(0);
    if second_row_present {
        decoded_data[dp + stride..dp + stride + 4].fill(0);
    }
    v_n_scratch[vp] = *carry;
    v_n_scratch[vp + 1] = 0;
    *carry = 0;
}

/// `kappa` for a non-initial quad: `E_max` of the four bottom-row samples
/// above it when the quad has two or more significant samples, else 1.
#[expect(clippy::inline_always, reason = "per-quad hot path")]
#[inline(always)]
fn kappa(inf: u32, above: u32) -> u32 {
    let mut gamma = inf & 0xF0;
    gamma &= gamma.wrapping_sub(0x10);
    if gamma != 0 {
        (above | 2).ilog2()
    } else {
        1
    }
}

/// Add `kappa` to a non-initial quad pair's `u` values. Reads the row above
/// from `v_n_scratch[vp..vp + 3]` before this pair overwrites it.
#[expect(clippy::inline_always, reason = "per-pair hot path")]
#[inline(always)]
fn with_kappa(symbols: [u32; 4], v_n_scratch: &[u32], vp: usize) -> [(u32, u32); 2] {
    let [t0, u0, t1, u1] = symbols;
    let above = [v_n_scratch[vp], v_n_scratch[vp + 1], v_n_scratch[vp + 2]];
    [
        (t0, u0 + kappa(t0, above[0] | above[1])),
        (t1, u1 + kappa(t1, above[1] | above[2])),
    ]
}

/// Cursor over one quad row's `MagSgn` outputs.
struct RowCursor {
    second_row_present: bool,
    dp: usize,
    vp: usize,
    carry: u32,
}

/// Decode the quad pair at columns `x..x + 4`: two vector quads when all four
/// columns are inside the block, else the scalar quads that are.
#[expect(
    clippy::inline_always,
    clippy::too_many_arguments,
    reason = "per-pair hot step shared by the split and fused row walkers"
)]
#[inline(always)]
fn decode_pair<S: Simd>(
    lanes: &Lanes<S>,
    magsgn: &mut FlatBitReader<'_>,
    decoded_data: &mut [u32],
    v_n_scratch: &mut [u32],
    quads: [(u32, u32); 2],
    missing_msbs: u32,
    width: u32,
    stride: usize,
    x: u32,
    cursor: &mut RowCursor,
) -> Option<()> {
    let p = 30 - missing_msbs;
    let mmsbp2 = missing_msbs + 2;
    if x + 4 <= width {
        decode_two_quads(
            lanes,
            magsgn,
            decoded_data,
            v_n_scratch,
            quads,
            mmsbp2,
            p - 1,
            stride,
            cursor.second_row_present,
            cursor.dp,
            cursor.vp,
            &mut cursor.carry,
        )?;
        cursor.dp += 4;
        cursor.vp += 2;
        return Some(());
    }
    let mut column = x;
    for (inf, uq) in quads {
        if column >= width {
            break;
        }
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
            cursor.second_row_present,
            &mut column,
            &mut cursor.dp,
            &mut cursor.vp,
            &mut cursor.carry,
        )?;
    }
    Some(())
}

#[expect(
    clippy::inline_always,
    clippy::too_many_arguments,
    reason = "per-pair direct-output hot step"
)]
#[inline(always)]
fn decode_pair_to_f32<const IRREVERSIBLE_MIDPOINT: bool, S: Simd>(
    lanes: &Lanes<S>,
    magsgn: &mut FlatBitReader<'_>,
    decoded_data: &mut [f32],
    v_n_scratch: &mut [u32],
    quads: [(u32, u32); 2],
    missing_msbs: u32,
    stride: usize,
    cursor: &mut RowCursor,
    output: FinalOutput,
) -> Option<()> {
    decode_two_quads_to_f32::<IRREVERSIBLE_MIDPOINT, _>(
        lanes,
        magsgn,
        decoded_data,
        v_n_scratch,
        quads,
        missing_msbs + 2,
        29 - missing_msbs,
        stride,
        cursor.second_row_present,
        cursor.dp,
        cursor.vp,
        &mut cursor.carry,
        output,
    )?;
    cursor.dp += 4;
    cursor.vp += 2;
    Some(())
}

/// Symbols of the quad pair at `sp` in the cleanup scratch.
#[expect(clippy::inline_always, reason = "per-pair hot path")]
#[inline(always)]
fn stored_symbols(cleanup: &[u16], sp: usize) -> [u32; 4] {
    [
        u32::from(cleanup[sp]),
        u32::from(cleanup[sp + 1]),
        u32::from(cleanup[sp + 2]),
        u32::from(cleanup[sp + 3]),
    ]
}

#[expect(
    clippy::inline_always,
    clippy::too_many_arguments,
    reason = "SIMD twin of the scalar MagSgn row walkers"
)]
#[inline(always)]
fn decode_stored_row<S: Simd>(
    lanes: &Lanes<S>,
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
    let stride = stride as usize;
    let mut cursor = RowCursor {
        second_row_present: y + 1 < height,
        dp: y as usize * stride,
        vp: 0,
        carry: 0,
    };
    let mut sp = (y >> 1) as usize * sstr;
    let mut x = 0u32;
    while x < width {
        let symbols = stored_symbols(cleanup, sp);
        let quads = if y == 0 {
            [(symbols[0], symbols[1]), (symbols[2], symbols[3])]
        } else {
            with_kappa(symbols, v_n_scratch, cursor.vp)
        };
        decode_pair(
            lanes,
            magsgn,
            decoded_data,
            v_n_scratch,
            quads,
            missing_msbs,
            width,
            stride,
            x,
            &mut cursor,
        )?;
        sp += 4;
        x += 4;
    }
    v_n_scratch[cursor.vp] = cursor.carry;
    Some(())
}

/// SIMD twin of the `MagSgn` row loop in [`super::decode_magnitude_sign_phase`].
#[expect(
    clippy::inline_always,
    clippy::too_many_arguments,
    reason = "inlined into the dispatched SIMD level"
)]
#[inline(always)]
pub(super) fn decode_magnitude_sign_rows<S: Simd>(
    simd: S,
    magsgn: &mut FlatBitReader<'_>,
    cleanup: &[u16],
    decoded_data: &mut [u32],
    v_n_scratch: &mut [u32],
    missing_msbs: u32,
    width: u32,
    height: u32,
    stride: u32,
    sstr: usize,
) -> Option<()> {
    let lanes = Lanes::new(simd);
    for y in (0..height).step_by(2) {
        decode_stored_row(
            &lanes,
            magsgn,
            cleanup,
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

/// Decode the cleanup symbols and `MagSgn` samples of quad row `y` together,
/// one quad pair at a time. The VLC chain of each pair is serial; the vector
/// `MagSgn` work of the pair before it is independent of it, so the core
/// overlaps the two.
#[expect(
    clippy::inline_always,
    clippy::too_many_arguments,
    reason = "fused row walker inlined into the dispatched SIMD level"
)]
#[inline(always)]
fn decode_fused_row<S: Simd>(
    lanes: &Lanes<S>,
    mel: &mut MelDecoder<'_>,
    vlc: &mut ReverseBitReader<'_>,
    run: &mut i32,
    magsgn: &mut FlatBitReader<'_>,
    scratch: &mut [u16],
    decoded_data: &mut [u32],
    v_n_scratch: &mut [u32],
    missing_msbs: u32,
    width: u32,
    height: u32,
    y: u32,
    stride: u32,
    sstr: usize,
) -> Option<()> {
    let stride = stride as usize;
    let mut cursor = RowCursor {
        second_row_present: y + 1 < height,
        dp: y as usize * stride,
        vp: 0,
        carry: 0,
    };
    let mut row_offset = (y >> 1) as usize * sstr;
    let mut c_q = 0u32;
    let mut x = 0u32;
    while x < width {
        let quads = if y == 0 {
            let [t0, u0, t1, u1] = decode_cleanup_first_row_pair(
                mel, vlc, run, scratch, row_offset, &mut c_q, x, width,
            )?;
            [(t0, u0), (t1, u1)]
        } else {
            let symbols = decode_cleanup_row_pair(
                mel, vlc, run, scratch, row_offset, sstr, &mut c_q, x, width,
            )?;
            with_kappa(symbols, v_n_scratch, cursor.vp)
        };
        decode_pair(
            lanes,
            magsgn,
            decoded_data,
            v_n_scratch,
            quads,
            missing_msbs,
            width,
            stride,
            x,
            &mut cursor,
        )?;
        row_offset += 4;
        x += 4;
    }
    scratch[row_offset] = 0;
    scratch[row_offset + 1] = 0;
    v_n_scratch[cursor.vp] = cursor.carry;
    Some(())
}

/// SIMD twin of the row loop in
/// [`super::decode_cleanup_and_magnitude_sign_phase`]. Rows are fused unless
/// the observer times the two phases separately.
#[expect(
    clippy::inline_always,
    clippy::too_many_arguments,
    reason = "inlined into the dispatched SIMD level"
)]
#[inline(always)]
pub(super) fn decode_cleanup_and_magnitude_sign_rows<S: Simd>(
    simd: S,
    mel: &mut MelDecoder<'_>,
    vlc: &mut ReverseBitReader<'_>,
    run: &mut i32,
    magsgn: &mut FlatBitReader<'_>,
    scratch: &mut [u16],
    decoded_data: &mut [u32],
    v_n_scratch: &mut [u32],
    missing_msbs: u32,
    width: u32,
    height: u32,
    stride: u32,
    sstr: usize,
    observer: &mut impl HtDecodeObserver,
) -> Option<()> {
    let lanes = Lanes::new(simd);
    let timed = observer.times_phases();
    for y in (0..height).step_by(2) {
        if !timed {
            decode_fused_row(
                &lanes,
                mel,
                vlc,
                run,
                magsgn,
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
            continue;
        }
        let phase_start = observer.phase_start();
        if y == 0 {
            decode_cleanup_symbols_first_row(mel, vlc, run, scratch, width)?;
        } else {
            decode_cleanup_symbols_row(mel, vlc, run, scratch, width, y, sstr)?;
        }
        observer.add_cleanup_us(phase_start);

        let phase_start = observer.phase_start();
        decode_stored_row(
            &lanes,
            magsgn,
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

#[expect(
    clippy::inline_always,
    clippy::too_many_arguments,
    reason = "direct-output fused row walker inlined into the dispatched SIMD level"
)]
#[inline(always)]
fn decode_fused_row_to_f32<const IRREVERSIBLE_MIDPOINT: bool, S: Simd>(
    lanes: &Lanes<S>,
    mel: &mut MelDecoder<'_>,
    vlc: &mut ReverseBitReader<'_>,
    run: &mut i32,
    magsgn: &mut FlatBitReader<'_>,
    scratch: &mut [u16],
    decoded_data: &mut [f32],
    v_n_scratch: &mut [u32],
    missing_msbs: u32,
    width: u32,
    height: u32,
    y: u32,
    stride: u32,
    sstr: usize,
    output: FinalOutput,
) -> Option<()> {
    let stride = stride as usize;
    let mut cursor = RowCursor {
        second_row_present: y + 1 < height,
        dp: y as usize * stride,
        vp: 0,
        carry: 0,
    };
    let mut row_offset = (y >> 1) as usize * sstr;
    let mut c_q = 0u32;
    let mut x = 0u32;
    while x < width {
        let quads = if y == 0 {
            let [t0, u0, t1, u1] = decode_cleanup_first_row_pair(
                mel, vlc, run, scratch, row_offset, &mut c_q, x, width,
            )?;
            [(t0, u0), (t1, u1)]
        } else {
            let symbols = decode_cleanup_row_pair(
                mel, vlc, run, scratch, row_offset, sstr, &mut c_q, x, width,
            )?;
            with_kappa(symbols, v_n_scratch, cursor.vp)
        };
        decode_pair_to_f32::<IRREVERSIBLE_MIDPOINT, _>(
            lanes,
            magsgn,
            decoded_data,
            v_n_scratch,
            quads,
            missing_msbs,
            stride,
            &mut cursor,
            output,
        )?;
        row_offset += 4;
        x += 4;
    }
    scratch[row_offset] = 0;
    scratch[row_offset + 1] = 0;
    v_n_scratch[cursor.vp] = cursor.carry;
    Some(())
}

/// Cleanup-only SIMD rows that publish final dequantized coefficients directly.
#[expect(
    clippy::inline_always,
    clippy::too_many_arguments,
    reason = "inlined into the dispatched SIMD level"
)]
#[inline(always)]
pub(super) fn decode_cleanup_and_magnitude_sign_to_f32_rows<
    const IRREVERSIBLE_MIDPOINT: bool,
    S: Simd,
>(
    simd: S,
    mel: &mut MelDecoder<'_>,
    vlc: &mut ReverseBitReader<'_>,
    run: &mut i32,
    magsgn: &mut FlatBitReader<'_>,
    scratch: &mut [u16],
    decoded_data: &mut [f32],
    v_n_scratch: &mut [u32],
    missing_msbs: u32,
    width: u32,
    height: u32,
    stride: u32,
    sstr: usize,
    output: FinalOutput,
) -> Option<()> {
    let lanes = Lanes::new(simd);
    for y in (0..height).step_by(2) {
        decode_fused_row_to_f32::<IRREVERSIBLE_MIDPOINT, _>(
            &lanes,
            mel,
            vlc,
            run,
            magsgn,
            scratch,
            decoded_data,
            v_n_scratch,
            missing_msbs,
            width,
            height,
            y,
            stride,
            sstr,
            output,
        )?;
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};

    use fearless_simd::{dispatch, Level};

    use super::super::super::cleanup::{
        cleanup_segment_suffix_length, cleanup_symbol_stride, decode_cleanup_symbols,
    };
    use super::super::super::readers::{
        flat_magsgn_capacity, unstuff_magsgn, FlatBitReader, MelDecoder, ReverseBitReader,
    };
    use super::super::super::stats::NoHtDecodeStats;
    use super::super::{
        decode_magnitude_sign_first_row_from_cleanup, decode_magnitude_sign_row_from_cleanup,
    };
    use super::{
        decode_cleanup_and_magnitude_sign_rows, decode_magnitude_sign_rows, MAX_SIMD_MMSBP2,
    };
    use crate::j2c::ht_block_encode::encode_code_block;

    struct Block<'a> {
        coded: &'a [u8],
        cleanup: &'a [u16],
        missing_msbs: u32,
        width: u32,
        height: u32,
        sstr: usize,
    }

    fn magsgn_reader<'b>(block: &Block<'_>, buffer: &'b mut Vec<u8>) -> Option<FlatBitReader<'b>> {
        let lcup = block.coded.len();
        let scup = cleanup_segment_suffix_length(block.coded, lcup)?;
        let samples = (block.width * block.height) as usize;
        let flat = unstuff_magsgn(&block.coded[..lcup - scup], buffer, samples * 4)?;
        FlatBitReader::new(flat)
    }

    fn scratch(block: &Block<'_>) -> (Vec<u32>, Vec<u32>, Vec<u8>) {
        let samples = (block.width * block.height) as usize;
        (
            vec![0xDEAD_BEEF; samples],
            vec![0; block.width.div_ceil(2) as usize + 2],
            Vec::with_capacity(flat_magsgn_capacity(samples).expect("MagSgn capacity")),
        )
    }

    fn scalar_rows(block: &Block<'_>) -> Option<Vec<u32>> {
        let (mut decoded, mut v_n, mut buffer) = scratch(block);
        let mut magsgn = magsgn_reader(block, &mut buffer)?;
        decode_magnitude_sign_first_row_from_cleanup(
            &mut magsgn,
            block.cleanup,
            &mut decoded,
            &mut v_n,
            block.missing_msbs,
            block.width,
            block.height,
            block.width,
        )?;
        for y in (2..block.height).step_by(2) {
            decode_magnitude_sign_row_from_cleanup(
                &mut magsgn,
                block.cleanup,
                &mut decoded,
                &mut v_n,
                block.missing_msbs,
                block.width,
                block.height,
                y,
                block.width,
                block.sstr,
            )?;
        }
        Some(decoded)
    }

    fn simd_rows(block: &Block<'_>, level: Level) -> Option<Vec<u32>> {
        let (mut decoded, mut v_n, mut buffer) = scratch(block);
        let mut magsgn = magsgn_reader(block, &mut buffer)?;
        dispatch!(level, simd => decode_magnitude_sign_rows(
            simd,
            &mut magsgn,
            block.cleanup,
            &mut decoded,
            &mut v_n,
            block.missing_msbs,
            block.width,
            block.height,
            block.width,
            block.sstr,
        ))?;
        Some(decoded)
    }

    /// Cleanup symbols and `MagSgn` samples decoded together, one quad pair at
    /// a time, as the fused cleanup-only phase does.
    fn fused_rows(
        coded: &[u8],
        missing_msbs: u32,
        width: u32,
        height: u32,
        level: Level,
    ) -> Option<Vec<u32>> {
        let lcup = coded.len();
        let scup = cleanup_segment_suffix_length(coded, lcup)?;
        let samples = (width * height) as usize;
        let sstr = cleanup_symbol_stride(width);
        let mut mel = MelDecoder::new(coded, lcup, scup);
        let mut vlc = ReverseBitReader::new_vlc(coded, lcup, scup);
        let mut buffer =
            Vec::with_capacity(flat_magsgn_capacity(samples).expect("MagSgn capacity"));
        let flat = unstuff_magsgn(&coded[..lcup - scup], &mut buffer, samples * 4)?;
        let mut magsgn = FlatBitReader::new(flat)?;
        let mut run = mel.get_run()?;
        let mut scratch = vec![0u16; sstr * (height.div_ceil(2) as usize + 1)];
        let mut decoded = vec![0xDEAD_BEEF; samples];
        let mut v_n = vec![0; width.div_ceil(2) as usize + 2];
        dispatch!(level, simd => decode_cleanup_and_magnitude_sign_rows(
            simd,
            &mut mel,
            &mut vlc,
            &mut run,
            &mut magsgn,
            &mut scratch,
            &mut decoded,
            &mut v_n,
            missing_msbs,
            width,
            height,
            width,
            sstr,
            &mut NoHtDecodeStats,
        ))?;
        Some(decoded)
    }

    /// Require the fused rows, and the SIMD rows over stored symbols, to agree
    /// with the scalar split rows on success, failure, and every output sample,
    /// on the native SIMD level and the portable fallback. Returns whether
    /// the block decoded.
    fn assert_rows_match(coded: &[u8], missing_msbs: u32, width: u32, height: u32) -> bool {
        assert!(missing_msbs + 2 <= MAX_SIMD_MMSBP2);
        let lcup = coded.len();
        let Some(scup) = cleanup_segment_suffix_length(coded, lcup) else {
            return false;
        };
        let sstr = cleanup_symbol_stride(width);
        let mut cleanup = vec![0u16; sstr * (height.div_ceil(2) as usize + 1)];
        if decode_cleanup_symbols(coded, lcup, scup, width, height, sstr, &mut cleanup).is_none() {
            for level in [Level::new(), Level::fallback()] {
                assert_eq!(
                    fused_rows(coded, missing_msbs, width, height, level),
                    None,
                    "{width}x{height} missing_msbs={missing_msbs} coded={coded:02x?}"
                );
            }
            return false;
        }
        let block = Block {
            coded,
            cleanup: &cleanup,
            missing_msbs,
            width,
            height,
            sstr,
        };
        let expected = scalar_rows(&block);
        for level in [Level::new(), Level::fallback()] {
            assert_eq!(
                simd_rows(&block, level),
                expected,
                "{width}x{height} missing_msbs={missing_msbs} coded={coded:02x?}"
            );
            assert_eq!(
                fused_rows(coded, missing_msbs, width, height, level),
                expected,
                "fused {width}x{height} missing_msbs={missing_msbs} coded={coded:02x?}"
            );
        }
        expected.is_some()
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        #[expect(clippy::cast_possible_truncation, reason = "bounded test draws")]
        fn below(&mut self, bound: u64) -> u32 {
            (self.next() % bound) as u32
        }
    }

    #[test]
    fn simd_rows_match_scalar_rows_for_encoded_blocks() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let mut decoded_blocks = 0;
        let sizes = [
            (1, 1),
            (2, 2),
            (3, 5),
            (4, 4),
            (5, 3),
            (7, 9),
            (8, 8),
            (12, 7),
            (16, 16),
            (17, 33),
            (32, 32),
            (33, 17),
            (64, 64),
            (63, 61),
            (64, 1),
            (1, 64),
        ];
        for (width, height) in sizes {
            for total_bitplanes in [1_u8, 2, 5, 8, 12, 16, 20, 23] {
                for density in [0, 1, 3, 8] {
                    let limit = 1_i64 << (total_bitplanes - 1);
                    let original: Vec<i32> = (0..width * height)
                        .map(|_| {
                            if density != 8 && rng.below(8) >= density {
                                return 0;
                            }
                            let magnitude = i64::from(rng.below(limit.unsigned_abs()));
                            let value = if rng.below(2) == 0 {
                                magnitude
                            } else {
                                -magnitude
                            };
                            i32::try_from(value).expect("test coefficient fits i32")
                        })
                        .collect();
                    let encoded = encode_code_block(&original, width, height, total_bitplanes)
                        .expect("encode HT block");
                    let missing_msbs = u32::from(encoded.num_zero_bitplanes);
                    if encoded.data.len() < 2 || missing_msbs + 2 > MAX_SIMD_MMSBP2 {
                        continue;
                    }
                    if assert_rows_match(&encoded.data, missing_msbs, width, height) {
                        decoded_blocks += 1;
                    }
                }
            }
        }
        assert!(decoded_blocks > 300, "{decoded_blocks}");
    }

    #[test]
    fn simd_rows_match_scalar_rows_for_arbitrary_bytes() {
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        let mut decoded_blocks = 0;
        for _ in 0..4000 {
            let width = 1 + rng.below(40);
            let height = 1 + rng.below(40);
            let missing_msbs = rng.below(u64::from(MAX_SIMD_MMSBP2 - 1));
            let len = 2 + rng.below(300) as usize;
            let mut coded: Vec<u8> = (0..len).map(|_| rng.next().to_be_bytes()[3]).collect();
            // Keep the cleanup suffix length valid so blocks reach the rows.
            let scup = 2 + rng.below(u64::try_from(len - 1).expect("len fits u64")) as usize;
            coded[len - 1] = u8::try_from(scup >> 4).expect("suffix high bits");
            coded[len - 2] = (coded[len - 2] & 0xF0) | u8::try_from(scup & 0xF).expect("nibble");
            if assert_rows_match(&coded, missing_msbs, width, height) {
                decoded_blocks += 1;
            }
        }
        assert!(decoded_blocks > 100, "{decoded_blocks}");
    }
}
