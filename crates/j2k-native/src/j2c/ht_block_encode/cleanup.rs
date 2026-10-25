// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::emit::{encode_first_quad_pair, encode_non_initial_quad_pair};
use super::quad::{
    FirstQuadPairRequest, NonInitialQuadPairRequest, QuadMarkerRows, QuadPairState, RowPair,
};
use super::workspace::HtEncodeWorkspace;
use super::writers::terminate_mel_vlc;
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

#[cfg(test)]
pub(super) fn try_encode_cleanup_segment_from_view(
    coefficients: CoefficientBlockView<'_, i32>,
    missing_msbs: u8,
    total_bitplanes: u8,
) -> EncodeResult<Vec<u8>> {
    let mut workspace = HtEncodeWorkspace::try_new()?;
    try_encode_cleanup_segment_from_view_in_workspace(
        coefficients,
        missing_msbs,
        total_bitplanes,
        &mut workspace,
    )
}

pub(super) fn try_encode_cleanup_segment_from_view_in_workspace(
    coefficients: CoefficientBlockView<'_, i32>,
    missing_msbs: u8,
    total_bitplanes: u8,
    workspace: &mut HtEncodeWorkspace,
) -> EncodeResult<Vec<u8>> {
    let shift = u32::from(31_u8.saturating_sub(total_bitplanes));
    // Gap-free blocks index samples directly; the strided view divides by
    // the width for every sample.
    if let Some(samples) = coefficients.contiguous() {
        return try_encode_cleanup_segment_from_source(
            &I32CleanupCoefficients {
                coefficients: samples,
                shift,
            },
            missing_msbs,
            coefficients.width(),
            coefficients.height(),
            workspace,
        );
    }
    let source = I32CleanupBlockView::new(coefficients, shift);
    try_encode_cleanup_segment_from_source(
        &source,
        missing_msbs,
        coefficients.width(),
        coefficients.height(),
        workspace,
    )
}

#[cfg(test)]
pub(super) fn encode_cleanup_segment(
    coefficients: &[u32],
    missing_msbs: u8,
    width: usize,
    height: usize,
) -> Result<Vec<u8>, &'static str> {
    let mut workspace = HtEncodeWorkspace::try_new().map_err(|_| "HTJ2K workspace allocation")?;
    try_encode_cleanup_segment_from_source(
        coefficients,
        missing_msbs,
        width,
        height,
        &mut workspace,
    )
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
    workspace: &mut HtEncodeWorkspace,
) -> EncodeResult<Vec<u8>> {
    workspace.reset_cleanup();
    let HtEncodeWorkspace {
        mel,
        vlc,
        mag_sgn: ms,
    } = workspace;

    let p = 30_u32.saturating_sub(u32::from(missing_msbs));
    let mut rows = RowPair::new();
    rows.fill(coefficients, 0, width, height, p)
        .map_err(ht_cleanup_invariant)?;

    let mut e_val = [0u8; 514];
    let mut cx_val = [0u8; 514];

    let mut e_qmax = [0i32; 2];
    let mut e_q = [0i32; 8];
    let mut rho = [0i32; 2];
    let mut c_q0 = 0usize;
    let mut s = [0u32; 8];
    let mut x = 0usize;

    while x < width {
        // A zero quad pair under zero context codes one MEL zero per quad: its
        // CxtVLC codewords, UVLC and MagSgn bits are empty, and the marker
        // rows it would write are still zero in the first row pair.
        if c_q0 == 0 && rows.pair_is_zero(x) {
            mel.encode(false).map_err(ht_cleanup_invariant)?;
            if x + 2 < width {
                mel.encode(false).map_err(ht_cleanup_invariant)?;
            }
            x += 4;
            continue;
        }
        encode_first_quad_pair(
            FirstQuadPairRequest {
                rows: &rows,
                width,
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
            mel,
            vlc,
            ms,
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

        rows.fill(coefficients, y, width, height, p)
            .map_err(ht_cleanup_invariant)?;
        x = 0;
        while x < width {
            // Below the first row pair the second quad's context also comes
            // from the markers above; the marker rows advance as for any
            // quad with rho = 0.
            if c_q0 == 0 && rows.pair_is_zero(x) && cx_val[lcxp + 1] | cx_val[lcxp + 2] == 0 {
                mel.encode(false).map_err(ht_cleanup_invariant)?;
                if x + 2 < width {
                    mel.encode(false).map_err(ht_cleanup_invariant)?;
                    e_val[lep + 1] = 0;
                    max_e = i32::from(e_val[lep + 2].max(e_val[lep + 3])) - 1;
                    e_val[lep + 2] = 0;
                    c_q0 = usize::from(cx_val[lcxp + 3]) << 2;
                    lep += 2;
                    lcxp += 2;
                } else {
                    max_e = i32::from(e_val[lep + 1].max(e_val[lep + 2])) - 1;
                    e_val[lep + 1] = 0;
                    lep += 1;
                    lcxp += 1;
                }
                x += 4;
                continue;
            }
            encode_non_initial_quad_pair(
                NonInitialQuadPairRequest {
                    rows: &rows,
                    width,
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
                mel,
                vlc,
                ms,
            )
            .map_err(ht_cleanup_invariant)?;
            x += 4;
        }

        y += 2;
    }

    terminate_mel_vlc(mel, vlc).map_err(ht_cleanup_invariant)?;
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
