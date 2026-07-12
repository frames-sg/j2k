// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    ComponentWavelet, ComponentWavelet97, IntegerWavelet, J2kForwardDwt53Level,
    J2kForwardDwt53Output, J2kForwardDwt97Level, J2kForwardDwt97Output, JpegToHtj2kError,
};
use crate::allocation::try_vec_with_capacity;

#[expect(
    clippy::cast_possible_truncation,
    reason = "validated JPEG component geometry fits u32 and the public coefficient ABI stores f32"
)]
pub(super) fn j2k_dwt_from_wavelet(
    wavelet: &ComponentWavelet,
    width: usize,
    height: usize,
) -> Result<J2kForwardDwt53Output, JpegToHtj2kError> {
    let mut current_width = width;
    let mut current_height = height;
    let mut levels = try_vec_with_capacity(wavelet.levels.len())?;

    for level in &wavelet.levels {
        levels.push(J2kForwardDwt53Level {
            hl: f64_to_f32(&level.hl)?,
            lh: f64_to_f32(&level.lh)?,
            hh: f64_to_f32(&level.hh)?,
            width: current_width as u32,
            height: current_height as u32,
            low_width: level.low_width as u32,
            low_height: level.low_height as u32,
            high_width: level.high_width as u32,
            high_height: level.high_height as u32,
        });
        current_width = level.low_width;
        current_height = level.low_height;
    }
    levels.reverse();

    Ok(J2kForwardDwt53Output {
        ll: f64_to_f32(&wavelet.final_ll)?,
        ll_width: wavelet.final_ll_width as u32,
        ll_height: wavelet.final_ll_height as u32,
        levels,
    })
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "validated JPEG component geometry fits u32 and the public coefficient ABI stores f32"
)]
pub(super) fn j2k_dwt97_from_wavelet(
    wavelet: &ComponentWavelet97,
    width: usize,
    height: usize,
) -> Result<J2kForwardDwt97Output, JpegToHtj2kError> {
    let mut current_width = width;
    let mut current_height = height;
    let mut levels = try_vec_with_capacity(wavelet.levels.len())?;

    for level in &wavelet.levels {
        levels.push(J2kForwardDwt97Level {
            hl: f64_to_f32(&level.hl)?,
            lh: f64_to_f32(&level.lh)?,
            hh: f64_to_f32(&level.hh)?,
            width: current_width as u32,
            height: current_height as u32,
            low_width: level.low_width as u32,
            low_height: level.low_height as u32,
            high_width: level.high_width as u32,
            high_height: level.high_height as u32,
        });
        current_width = level.low_width;
        current_height = level.low_height;
    }
    levels.reverse();

    Ok(J2kForwardDwt97Output {
        ll: f64_to_f32(&wavelet.final_ll)?,
        ll_width: wavelet.final_ll_width as u32,
        ll_height: wavelet.final_ll_height as u32,
        levels,
    })
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "validated JPEG geometry fits u32 and the public coefficient ABI intentionally stores f32"
)]
pub(super) fn j2k_dwt_from_integer_wavelet(
    wavelet: &IntegerWavelet,
) -> Result<J2kForwardDwt53Output, JpegToHtj2kError> {
    let mut levels = try_vec_with_capacity(wavelet.levels.len())?;
    for level in &wavelet.levels {
        levels.push(J2kForwardDwt53Level {
            hl: i32_to_f32(&level.hl)?,
            lh: i32_to_f32(&level.lh)?,
            hh: i32_to_f32(&level.hh)?,
            width: level.width as u32,
            height: level.height as u32,
            low_width: level.low_width as u32,
            low_height: level.low_height as u32,
            high_width: level.high_width as u32,
            high_height: level.high_height as u32,
        });
    }
    levels.reverse();

    Ok(J2kForwardDwt53Output {
        ll: i32_to_f32(&wavelet.final_ll)?,
        ll_width: wavelet.final_ll_width as u32,
        ll_height: wavelet.final_ll_height as u32,
        levels,
    })
}

pub(super) fn rounded_wavelet_i32(
    wavelet: &ComponentWavelet,
) -> Result<Vec<i32>, JpegToHtj2kError> {
    let coefficient_count = wavelet_coefficient_count_53(wavelet)?;
    let mut output = try_vec_with_capacity(coefficient_count)?;
    append_rounded_i32(&wavelet.final_ll, &mut output)?;
    for level in wavelet.levels.iter().rev() {
        append_rounded_i32(&level.hl, &mut output)?;
        append_rounded_i32(&level.lh, &mut output)?;
        append_rounded_i32(&level.hh, &mut output)?;
    }
    Ok(output)
}

pub(super) fn rounded_wavelet97_i32(
    wavelet: &ComponentWavelet97,
) -> Result<Vec<i32>, JpegToHtj2kError> {
    let coefficient_count = wavelet_coefficient_count_97(wavelet)?;
    let mut output = try_vec_with_capacity(coefficient_count)?;
    append_rounded_i32(&wavelet.final_ll, &mut output)?;
    for level in wavelet.levels.iter().rev() {
        append_rounded_i32(&level.hl, &mut output)?;
        append_rounded_i32(&level.lh, &mut output)?;
        append_rounded_i32(&level.hh, &mut output)?;
    }
    Ok(output)
}

fn append_rounded_i32(values: &[f64], output: &mut Vec<i32>) -> Result<(), JpegToHtj2kError> {
    for &value in values {
        output.push(round_f64_to_i32(value)?);
    }
    Ok(())
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the finite rounded coefficient is explicitly checked against the complete i32 range"
)]
fn round_f64_to_i32(value: f64) -> Result<i32, JpegToHtj2kError> {
    let rounded = value.round();
    if !rounded.is_finite() {
        return Err(JpegToHtj2kError::Validation(
            "float reference coefficient is not finite",
        ));
    }
    if rounded < f64::from(i32::MIN) || rounded > f64::from(i32::MAX) {
        return Err(JpegToHtj2kError::Validation(
            "float reference coefficient exceeds i32 range",
        ));
    }
    Ok(rounded as i32)
}

fn wavelet_coefficient_count_53(wavelet: &ComponentWavelet) -> Result<usize, JpegToHtj2kError> {
    wavelet_coefficient_count(
        wavelet.final_ll.len(),
        wavelet
            .levels
            .iter()
            .map(|level| [level.hl.len(), level.lh.len(), level.hh.len()]),
    )
}

fn wavelet_coefficient_count_97(wavelet: &ComponentWavelet97) -> Result<usize, JpegToHtj2kError> {
    wavelet_coefficient_count(
        wavelet.final_ll.len(),
        wavelet
            .levels
            .iter()
            .map(|level| [level.hl.len(), level.lh.len(), level.hh.len()]),
    )
}

fn wavelet_coefficient_count(
    mut count: usize,
    levels: impl Iterator<Item = [usize; 3]>,
) -> Result<usize, JpegToHtj2kError> {
    for bands in levels {
        for band_len in bands {
            count = count.checked_add(band_len).ok_or_else(cap_overflow)?;
        }
    }
    Ok(count)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the public coefficient ABI intentionally stores validated f64 coefficients as f32"
)]
fn f64_to_f32(values: &[f64]) -> Result<Vec<f32>, JpegToHtj2kError> {
    let mut output = try_vec_with_capacity(values.len())?;
    output.extend(values.iter().map(|&value| value as f32));
    Ok(output)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "the public coefficient ABI intentionally stores reversible i32 coefficients as f32"
)]
fn i32_to_f32(values: &[i32]) -> Result<Vec<f32>, JpegToHtj2kError> {
    let mut output = try_vec_with_capacity(values.len())?;
    output.extend(values.iter().map(|&value| value as f32));
    Ok(output)
}

fn cap_overflow() -> JpegToHtj2kError {
    JpegToHtj2kError::MemoryCapExceeded {
        requested: usize::MAX,
        cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    }
}
