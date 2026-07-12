// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::emit::{encode_first_quad_pair, encode_non_initial_quad_pair};
use super::quad::{FirstQuadPairRequest, NonInitialQuadPairRequest, QuadMarkerRows, QuadPairState};
use super::writers::{terminate_mel_vlc, MagSgnEncoder, MelEncoder, VlcEncoder};
#[cfg(test)]
use crate::j2c::coefficient_view::legacy_coefficient_view_error;
use crate::j2c::coefficient_view::CoefficientBlockView;
use crate::j2c::encode::allocation::try_untracked_vec;
use crate::{EncodeError, EncodeResult};

mod source;

#[cfg(test)]
pub(super) use source::convert_nonzero_to_aligned_sign_magnitude_and_max;
use source::I32CleanupBlockView;
pub(super) use source::{
    max_nonzero_magnitude_view, CleanupCoefficientSource, I32CleanupCoefficients,
};

#[cfg(test)]
pub(super) fn encode_cleanup_segment_from_coefficients(
    coefficients: &[i32],
    missing_msbs: u8,
    width: usize,
    height: usize,
    total_bitplanes: u8,
) -> Result<Vec<u8>, &'static str> {
    let coefficients = CoefficientBlockView::try_contiguous(coefficients, width, height)
        .map_err(legacy_coefficient_view_error)?;
    try_encode_cleanup_segment_from_view(coefficients, missing_msbs, total_bitplanes)
        .map_err(legacy_coefficient_view_error)
}

pub(super) fn try_encode_cleanup_segment_from_view(
    coefficients: CoefficientBlockView<'_, i32>,
    missing_msbs: u8,
    total_bitplanes: u8,
) -> EncodeResult<Vec<u8>> {
    let source = I32CleanupBlockView::new(
        coefficients,
        u32::from(31_u8.saturating_sub(total_bitplanes)),
    );
    try_encode_cleanup_segment_from_source(
        &source,
        missing_msbs,
        coefficients.width(),
        coefficients.height(),
    )
}

#[cfg(test)]
pub(super) fn encode_cleanup_segment(
    coefficients: &[u32],
    missing_msbs: u8,
    width: usize,
    height: usize,
) -> Result<Vec<u8>, &'static str> {
    try_encode_cleanup_segment_from_source(coefficients, missing_msbs, width, height)
        .map_err(legacy_coefficient_view_error)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    reason = "locator fields are normatively bounded to 12 bits and this function preserves cleanup pass order"
)]
fn try_encode_cleanup_segment_from_source<S: CleanupCoefficientSource + ?Sized>(
    coefficients: &S,
    missing_msbs: u8,
    width: usize,
    height: usize,
) -> EncodeResult<Vec<u8>> {
    let mut mel = MelEncoder::try_new()?;
    let mut vlc = VlcEncoder::try_new()?;
    let mut ms = MagSgnEncoder::try_new()?;

    let p = 30_u32.saturating_sub(u32::from(missing_msbs));
    let stride = width;

    let mut e_val = [0u8; 514];
    let mut cx_val = [0u8; 514];

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
        )
        .map_err(ht_cleanup_invariant)?;
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
            )
            .map_err(ht_cleanup_invariant)?;
            x += 4;
        }

        y += 2;
    }

    terminate_mel_vlc(&mut mel, &mut vlc).map_err(ht_cleanup_invariant)?;
    ms.terminate().map_err(ht_cleanup_invariant)?;

    let total_len = ms.pos + mel.pos + vlc.pos;
    if total_len < 2 {
        return Err(EncodeError::InternalInvariant {
            what: "HTJ2K cleanup segment is too short",
        });
    }

    let mut data = try_untracked_vec(total_len, "HTJ2K cleanup segment")?;
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

fn ht_cleanup_invariant(what: &'static str) -> EncodeError {
    EncodeError::InternalInvariant { what }
}
