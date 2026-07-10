// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::emit::{encode_first_quad_pair, encode_non_initial_quad_pair};
use super::quad::{FirstQuadPairRequest, NonInitialQuadPairRequest, QuadMarkerRows, QuadPairState};
use super::writers::{terminate_mel_vlc, MagSgnEncoder, MelEncoder, VlcEncoder};

#[cfg(test)]
pub(super) fn convert_nonzero_to_aligned_sign_magnitude_and_max(
    coefficients: &[i32],
    k_max: u8,
) -> Option<(Vec<u32>, u32)> {
    let first_nonzero = coefficients
        .iter()
        .position(|&coefficient| coefficient != 0)?;
    let shift = u32::from(31_u8.saturating_sub(k_max));
    let mut aligned = Vec::with_capacity(coefficients.len());
    aligned.resize(first_nonzero, 0);
    let mut max_magnitude = 0u32;

    for &coefficient in &coefficients[first_nonzero..] {
        let magnitude = coefficient.unsigned_abs();
        max_magnitude = max_magnitude.max(magnitude);

        if magnitude == 0 {
            aligned.push(0);
        } else {
            let sign = if coefficient < 0 { 0x8000_0000 } else { 0 };
            aligned.push(sign | (magnitude << shift));
        }
    }

    Some((aligned, max_magnitude))
}

pub(super) fn max_nonzero_magnitude(coefficients: &[i32]) -> Option<u32> {
    let mut max_magnitude = 0u32;
    for &coefficient in coefficients {
        max_magnitude = max_magnitude.max(coefficient.unsigned_abs());
    }
    (max_magnitude != 0).then_some(max_magnitude)
}

pub(super) trait CleanupCoefficientSource {
    fn aligned_value(&self, index: usize) -> u32;
}

impl CleanupCoefficientSource for [u32] {
    #[expect(
        clippy::inline_always,
        reason = "coefficient loads are fused into the per-sample cleanup hot path"
    )]
    #[inline(always)]
    fn aligned_value(&self, index: usize) -> u32 {
        self[index]
    }
}

pub(super) struct I32CleanupCoefficients<'a> {
    pub(super) coefficients: &'a [i32],
    pub(super) shift: u32,
}

impl CleanupCoefficientSource for I32CleanupCoefficients<'_> {
    #[expect(
        clippy::inline_always,
        reason = "coefficient conversion is fused into the per-sample cleanup hot path"
    )]
    #[inline(always)]
    fn aligned_value(&self, index: usize) -> u32 {
        aligned_sign_magnitude(self.coefficients[index], self.shift)
    }
}

#[expect(
    clippy::inline_always,
    reason = "sign-magnitude conversion runs once per sample in the cleanup hot path"
)]
#[inline(always)]
fn aligned_sign_magnitude(coefficient: i32, shift: u32) -> u32 {
    let magnitude = coefficient.unsigned_abs();
    if magnitude == 0 {
        0
    } else {
        let sign = if coefficient < 0 { 0x8000_0000 } else { 0 };
        sign | (magnitude << shift)
    }
}

pub(super) fn encode_cleanup_segment_from_coefficients(
    coefficients: &[i32],
    missing_msbs: u8,
    width: usize,
    height: usize,
    total_bitplanes: u8,
) -> Result<Vec<u8>, &'static str> {
    let source = I32CleanupCoefficients {
        coefficients,
        shift: u32::from(31_u8.saturating_sub(total_bitplanes)),
    };
    encode_cleanup_segment_from_source(&source, missing_msbs, width, height)
}

#[cfg(test)]
pub(super) fn encode_cleanup_segment(
    coefficients: &[u32],
    missing_msbs: u8,
    width: usize,
    height: usize,
) -> Result<Vec<u8>, &'static str> {
    encode_cleanup_segment_from_source(coefficients, missing_msbs, width, height)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    reason = "locator fields are normatively bounded to 12 bits and this function preserves cleanup pass order"
)]
fn encode_cleanup_segment_from_source<S: CleanupCoefficientSource + ?Sized>(
    coefficients: &S,
    missing_msbs: u8,
    width: usize,
    height: usize,
) -> Result<Vec<u8>, &'static str> {
    let mut mel = MelEncoder::new();
    let mut vlc = VlcEncoder::new();
    let mut ms = MagSgnEncoder::new();

    let p = 30_u32.saturating_sub(u32::from(missing_msbs));
    let stride = width;

    let mut e_val = [0u8; 513];
    let mut cx_val = [0u8; 513];

    let mut e_qmax = [0i32; 2];
    let mut e_q = [0i32; 8];
    let mut rho = [0i32; 2];
    let mut c_q0 = 0usize;
    let mut s = [0u32; 8];
    let mut sp = 0usize;
    let mut x = 0usize;

    while x < width {
        encode_first_quad_pair(
            FirstQuadPairRequest {
                coefficients,
                stride,
                height,
                p,
                sp: &mut sp,
                x,
                markers: QuadMarkerRows {
                    e_val: &mut e_val,
                    cx_val: &mut cx_val,
                },
                state: QuadPairState {
                    c_q0: &mut c_q0,
                    rho: &mut rho,
                    e_q: &mut e_q,
                    e_qmax: &mut e_qmax,
                    s: &mut s,
                },
            },
            &mut mel,
            &mut vlc,
            &mut ms,
        )?;
        x += 4;
    }

    let e_val_sentinel = width.div_ceil(2) + 1;
    e_val[e_val_sentinel] = 0;

    let mut y = 2usize;
    while y < height {
        let mut lep = 0usize;
        let mut max_e = i32::from(e_val[lep].max(e_val[lep + 1])) - 1;
        e_val[lep] = 0;

        let mut lcxp = 0usize;
        c_q0 = usize::from(cx_val[lcxp]) + (usize::from(cx_val[lcxp + 1]) << 2);
        cx_val[lcxp] = 0;

        sp = y * stride;
        x = 0;
        while x < width {
            encode_non_initial_quad_pair(
                NonInitialQuadPairRequest {
                    coefficients,
                    stride,
                    width,
                    height,
                    y,
                    p,
                    sp: &mut sp,
                    x,
                    markers: QuadMarkerRows {
                        e_val: &mut e_val,
                        cx_val: &mut cx_val,
                    },
                    lep: &mut lep,
                    lcxp: &mut lcxp,
                    max_e: &mut max_e,
                    state: QuadPairState {
                        c_q0: &mut c_q0,
                        rho: &mut rho,
                        e_q: &mut e_q,
                        e_qmax: &mut e_qmax,
                        s: &mut s,
                    },
                },
                &mut mel,
                &mut vlc,
                &mut ms,
            )?;
            x += 4;
        }

        y += 2;
    }

    terminate_mel_vlc(&mut mel, &mut vlc)?;
    ms.terminate()?;

    let total_len = ms.pos + mel.pos + vlc.pos;
    if total_len < 2 {
        return Err("HTJ2K cleanup segment is too short");
    }

    let mut data = Vec::with_capacity(total_len);
    data.extend_from_slice(&ms.buffer[..ms.pos]);
    data.extend_from_slice(&mel.buffer[..mel.pos]);
    let vlc_start = vlc.buffer.len() - vlc.pos;
    data.extend_from_slice(&vlc.buffer[vlc_start..]);

    let locator_bytes = mel.pos + vlc.pos;
    let last = data.len() - 1;
    let prev = data.len() - 2;
    data[last] = (locator_bytes >> 4) as u8;
    data[prev] = (data[prev] & 0xF0) | ((locator_bytes as u8) & 0x0F);

    Ok(data)
}
