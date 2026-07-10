// SPDX-License-Identifier: MIT OR Apache-2.0

use core::convert::Infallible;

use super::super::ht_encode_tables::{HT_VLC_ENCODE_TABLE0, HT_VLC_ENCODE_TABLE1};
use super::cleanup::{max_nonzero_magnitude, CleanupCoefficientSource, I32CleanupCoefficients};
use super::facade::MAX_HT_BITPLANES;
use super::quad::{
    first_quad_pair, non_initial_quad_pair, FirstQuadPairRequest, InitialQuadRow,
    NonInitialQuadPairRequest, NonInitialQuadRow, QuadMarkerRows, QuadPairState, QuadSink,
};
use crate::HtCleanupEncodeDistribution;

#[expect(
    clippy::cast_sign_loss,
    clippy::inline_always,
    reason = "the clamped histogram bucket is nonnegative and this helper runs for every collected field"
)]
#[inline(always)]
fn increment_limited_count(counts: &mut [u64; 32], value: i32) {
    let index = value.clamp(0, 31) as usize;
    counts[index] += 1;
}

#[expect(clippy::cast_sign_loss, reason = "rho is a nonnegative four-bit mask")]
fn record_distribution_initial_quad(
    distribution: &mut HtCleanupEncodeDistribution,
    rho: i32,
    _e_qmax: i32,
    _u_q: i32,
) {
    let rho_index = (rho & 0xF) as usize;
    distribution.total_quads += 1;
    distribution.initial_quads += 1;
    distribution.rho_counts[rho_index] += 1;
    distribution.initial_rho_counts[rho_index] += 1;
}

#[expect(clippy::cast_sign_loss, reason = "clamped HT fields are valid indices")]
fn record_distribution_non_initial_quad(
    distribution: &mut HtCleanupEncodeDistribution,
    rho: i32,
    e_qmax: i32,
    kappa: i32,
    u_q: i32,
) {
    let rho_index = (rho & 0xF) as usize;
    let u_q_index = u_q.clamp(0, 31) as usize;
    distribution.total_quads += 1;
    distribution.non_initial_quads += 1;
    distribution.rho_counts[rho_index] += 1;
    distribution.non_initial_rho_counts[rho_index] += 1;
    increment_limited_count(&mut distribution.non_initial_u_q_counts, u_q);
    increment_limited_count(&mut distribution.non_initial_e_qmax_counts, e_qmax);
    increment_limited_count(&mut distribution.non_initial_kappa_counts, kappa);
    distribution.non_initial_rho_u_q_counts[rho_index][u_q_index] += 1;
}

#[expect(clippy::cast_sign_loss, reason = "bounded HT fields are nonnegative")]
fn record_distribution_mag_signs(
    distribution: &mut HtCleanupEncodeDistribution,
    rho: i32,
    u_q: i32,
    tuple: u16,
) {
    let rho_index = (rho & 0xF) as usize;
    let rho_bits = (rho as u32) & 0xF;
    if rho_bits == 0 {
        return;
    }

    let e_k = u32::from(tuple & 0xF);
    let u_q = u_q.max(0) as u32;

    distribution.mag_sign_calls += 1;
    distribution.mag_sign_rho_counts[rho_index] += 1;

    for bit in 0..4 {
        if (rho_bits & (1 << bit)) == 0 {
            continue;
        }
        let reduction = (e_k >> bit) & 1;
        let magnitude_bits = u_q.saturating_sub(reduction).min(31) as usize;
        distribution.mag_sign_sample_bit_counts[magnitude_bits] += 1;
        distribution.mag_sign_encoded_samples += 1;
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "u32 has at most 32 bitplanes"
)]
pub(crate) fn collect_encode_distribution(
    coefficients: &[i32],
    width: u32,
    height: u32,
    total_bitplanes: u8,
) -> Result<HtCleanupEncodeDistribution, &'static str> {
    if total_bitplanes == 0 || total_bitplanes > MAX_HT_BITPLANES {
        return Err("HTJ2K scalar encoder currently supports 1..=31 bitplanes");
    }

    let Some(max_magnitude) = max_nonzero_magnitude(coefficients) else {
        return Ok(HtCleanupEncodeDistribution::default());
    };

    let block_bitplanes = (u32::BITS - max_magnitude.leading_zeros()) as u8;
    if block_bitplanes > total_bitplanes {
        return Err("HTJ2K block magnitude exceeds configured bitplane count");
    }

    let source = I32CleanupCoefficients {
        coefficients,
        shift: u32::from(31_u8.saturating_sub(total_bitplanes)),
    };
    let mut distribution = HtCleanupEncodeDistribution::default();
    let missing_msbs = total_bitplanes.saturating_sub(1);
    collect_encode_distribution_from_source(
        &source,
        missing_msbs,
        width as usize,
        height as usize,
        &mut distribution,
    );
    Ok(distribution)
}

fn collect_encode_distribution_from_source<S: CleanupCoefficientSource + ?Sized>(
    coefficients: &S,
    missing_msbs: u8,
    width: usize,
    height: usize,
    distribution: &mut HtCleanupEncodeDistribution,
) {
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
        collect_first_quad_pair(
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
            distribution,
        );
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
            collect_non_initial_quad_pair(
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
                distribution,
            );
            x += 4;
        }

        y += 2;
    }
}

struct CollectQuadSink<'a> {
    distribution: &'a mut HtCleanupEncodeDistribution,
}

impl QuadSink for CollectQuadSink<'_> {
    type Error = Infallible;

    #[expect(clippy::inline_always, reason = "erase collector sink dispatch")]
    #[inline(always)]
    fn quad_initial(&mut self, row: InitialQuadRow<'_>) -> Result<i32, Self::Error> {
        Ok(collect_quad_initial_row(row, self.distribution))
    }

    #[expect(clippy::inline_always, reason = "erase collector sink dispatch")]
    #[inline(always)]
    fn quad_non_initial(&mut self, row: NonInitialQuadRow<'_>) -> Result<i32, Self::Error> {
        Ok(collect_quad_non_initial_row(row, self.distribution))
    }

    #[expect(clippy::inline_always, reason = "erase collector sink dispatch")]
    #[inline(always)]
    fn initial_uvlc_pair(&mut self, _u_q0: i32, _u_q1: i32) -> Result<(), Self::Error> {
        Ok(())
    }

    #[expect(clippy::inline_always, reason = "erase collector sink dispatch")]
    #[inline(always)]
    fn initial_uvlc_lone(&mut self, _u_q0: i32) -> Result<(), Self::Error> {
        Ok(())
    }

    #[expect(clippy::inline_always, reason = "erase collector sink dispatch")]
    #[inline(always)]
    fn non_initial_uvlc(&mut self, _u_q0: i32, _u_q1: i32) -> Result<(), Self::Error> {
        Ok(())
    }
}

#[expect(clippy::inline_always, reason = "fuse the collector quad walker")]
#[inline(always)]
fn collect_first_quad_pair<C: CleanupCoefficientSource + ?Sized>(
    request: FirstQuadPairRequest<'_, C>,
    distribution: &mut HtCleanupEncodeDistribution,
) {
    match first_quad_pair(request, &mut CollectQuadSink { distribution }) {
        Ok(()) => {}
        Err(err) => match err {},
    }
}

#[expect(clippy::inline_always, reason = "fuse the collector quad walker")]
#[inline(always)]
fn collect_non_initial_quad_pair<C: CleanupCoefficientSource + ?Sized>(
    request: NonInitialQuadPairRequest<'_, C>,
    distribution: &mut HtCleanupEncodeDistribution,
) {
    match non_initial_quad_pair(request, &mut CollectQuadSink { distribution }) {
        Ok(()) => {}
        Err(err) => match err {},
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::inline_always,
    reason = "packed quad fields are bounded by HT tables and the collector mirrors the encoder hot path"
)]
#[inline(always)]
fn collect_quad_initial_row(
    row: InitialQuadRow<'_>,
    distribution: &mut HtCleanupEncodeDistribution,
) -> i32 {
    let InitialQuadRow {
        offset,
        c_q,
        rho,
        e_qmax,
        e_q,
        lep,
        lcxp,
        e_val,
        cx_val,
        ..
    } = row;
    let u_q = e_qmax.max(1) - 1;
    let mut eps = 0u16;

    if u_q > 0 {
        eps |= u16::from(u8::from(e_q[offset] == e_qmax));
        eps |= u16::from(u8::from(e_q[offset + 1] == e_qmax)) << 1;
        eps |= u16::from(u8::from(e_q[offset + 2] == e_qmax)) << 2;
        eps |= u16::from(u8::from(e_q[offset + 3] == e_qmax)) << 3;
    }

    e_val[lep] = e_val[lep].max(e_q[offset + 1] as u8);
    e_val[lep + 1] = e_q[offset + 3] as u8;
    cx_val[lcxp] |= ((rho & 2) >> 1) as u8;
    cx_val[lcxp + 1] = ((rho & 8) >> 3) as u8;

    let tuple = HT_VLC_ENCODE_TABLE0[(c_q << 8) | ((rho as usize) << 4) | eps as usize];
    record_distribution_initial_quad(distribution, rho, e_qmax, u_q);
    record_distribution_mag_signs(distribution, rho, e_qmax.max(1), tuple);
    u_q
}

#[expect(
    clippy::cast_sign_loss,
    clippy::inline_always,
    clippy::needless_pass_by_value,
    reason = "the by-value row matches the sink contract and its bounded packed fields index HT tables"
)]
#[inline(always)]
fn collect_quad_non_initial_row(
    row: NonInitialQuadRow<'_>,
    distribution: &mut HtCleanupEncodeDistribution,
) -> i32 {
    let NonInitialQuadRow {
        offset,
        c_q,
        rho,
        e_qmax,
        max_e,
        e_q,
        ..
    } = row;
    let kappa = if (rho & (rho - 1)) != 0 {
        max_e.max(1)
    } else {
        1
    };
    let u_q = e_qmax.max(kappa) - kappa;
    let mut eps = 0u16;

    if u_q > 0 {
        eps |= u16::from(u8::from(e_q[offset] == e_qmax));
        eps |= u16::from(u8::from(e_q[offset + 1] == e_qmax)) << 1;
        eps |= u16::from(u8::from(e_q[offset + 2] == e_qmax)) << 2;
        eps |= u16::from(u8::from(e_q[offset + 3] == e_qmax)) << 3;
    }

    let tuple = HT_VLC_ENCODE_TABLE1[(c_q << 8) | ((rho as usize) << 4) | eps as usize];
    record_distribution_non_initial_quad(distribution, rho, e_qmax, kappa, u_q);
    record_distribution_mag_signs(distribution, rho, e_qmax.max(kappa), tuple);
    u_q
}
