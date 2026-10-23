// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::emit::encode_quad_pair;
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
    reason = "locator fields are normatively bounded to 12 bits"
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
        rows,
    } = workspace;

    let p = 30_u32.saturating_sub(u32::from(missing_msbs));
    for y in (0..height).step_by(2) {
        rows.fill(coefficients, y, width, height, p)
            .map_err(ht_cleanup_invariant)?;
        for x in (0..width).step_by(4) {
            if rows.pair_is_zero(x, width) {
                mel.encode(false).map_err(ht_cleanup_invariant)?;
                if x + 2 < width {
                    mel.encode(false).map_err(ht_cleanup_invariant)?;
                }
                continue;
            }
            if y == 0 {
                encode_quad_pair::<true>(rows, width, x, mel, vlc, ms)
            } else {
                encode_quad_pair::<false>(rows, width, x, mel, vlc, ms)
            }
            .map_err(ht_cleanup_invariant)?;
        }
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
