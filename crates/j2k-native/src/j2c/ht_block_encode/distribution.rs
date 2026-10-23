// SPDX-License-Identifier: MIT OR Apache-2.0

use core::convert::Infallible;

use super::super::ht_encode_tables::{HT_VLC_ENCODE_TABLE0, HT_VLC_ENCODE_TABLE1};
use super::cleanup::{
    max_nonzero_magnitude_view, CleanupCoefficientSource, I32CleanupCoefficients,
};
use super::facade::MAX_HT_BITPLANES;
use super::quad::{quad_pair, CodedQuad, QuadSink, RowPair};
use crate::j2c::coefficient_view::{validate_tier1_code_block_geometry, CoefficientBlockView};
use crate::{EncodeError, EncodeResult, HtCleanupEncodeDistribution};

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
) -> EncodeResult<HtCleanupEncodeDistribution> {
    if total_bitplanes == 0 || total_bitplanes > MAX_HT_BITPLANES {
        return Err(EncodeError::InvalidInput {
            what: "HTJ2K scalar encoder currently supports 1..=31 bitplanes",
        });
    }

    let width = width as usize;
    let height = height as usize;
    validate_tier1_code_block_geometry(width, height)?;
    let coefficients_view = CoefficientBlockView::try_contiguous(coefficients, width, height)?;

    let Some(max_magnitude) = max_nonzero_magnitude_view(coefficients_view) else {
        return Ok(HtCleanupEncodeDistribution::default());
    };

    let block_bitplanes = max_magnitude.bit_width() as u8;
    if block_bitplanes > total_bitplanes {
        return Err(EncodeError::InvalidInput {
            what: "HTJ2K block magnitude exceeds configured bitplane count",
        });
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
        width,
        height,
        &mut distribution,
    )
    .map_err(|what| EncodeError::InternalInvariant { what })?;
    Ok(distribution)
}

fn collect_encode_distribution_from_source<S: CleanupCoefficientSource + ?Sized>(
    coefficients: &S,
    missing_msbs: u8,
    width: usize,
    height: usize,
    distribution: &mut HtCleanupEncodeDistribution,
) -> Result<(), &'static str> {
    let p = 30_u32.saturating_sub(u32::from(missing_msbs));
    let mut rows = RowPair::new();
    let mut sink = CollectQuadSink { distribution };
    for y in (0..height).step_by(2) {
        rows.fill(coefficients, y, width, height, p)?;
        for x in (0..width).step_by(4) {
            let collected = if y == 0 {
                quad_pair::<true, _>(&rows, width, x, &mut sink)
            } else {
                quad_pair::<false, _>(&rows, width, x, &mut sink)
            };
            match collected {
                Ok(()) => {}
                Err(err) => match err {},
            }
        }
    }
    Ok(())
}

struct CollectQuadSink<'a> {
    distribution: &'a mut HtCleanupEncodeDistribution,
}

impl QuadSink for CollectQuadSink<'_> {
    type Error = Infallible;

    #[expect(clippy::inline_always, reason = "erase collector sink dispatch")]
    #[inline(always)]
    fn quad<const INITIAL: bool>(&mut self, quad: CodedQuad) -> Result<(), Self::Error> {
        let table = if INITIAL {
            &HT_VLC_ENCODE_TABLE0
        } else {
            &HT_VLC_ENCODE_TABLE1
        };
        let tuple = table[usize::from(quad.index)];
        let (rho, e_qmax) = (i32::from(quad.rho()), i32::from(quad.e_qmax));
        let (u_q, bound) = (i32::from(quad.u_q), i32::from(quad.bound));
        if INITIAL {
            record_distribution_initial_quad(self.distribution, rho, e_qmax, u_q);
        } else {
            // `u_q` is `max(e_qmax, kappa) - kappa`.
            record_distribution_non_initial_quad(self.distribution, rho, e_qmax, bound - u_q, u_q);
        }
        record_distribution_mag_signs(self.distribution, rho, bound, tuple);
        Ok(())
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

#[cfg(test)]
mod tests;
