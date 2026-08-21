// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{error::CudaError, memory::CudaDeviceBuffer};

use super::super::{
    types::{CudaJ2kIdwtJob, CudaJ2kIdwtTarget, CudaJ2kRect},
    validation::checked_f32_words_byte_len,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ValidatedIdwtJob {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) output_bytes: usize,
}

impl ValidatedIdwtJob {
    pub(super) const fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }
}

pub(super) fn validate_idwt_job(
    bands: [&CudaDeviceBuffer; 4],
    output: Option<&CudaDeviceBuffer>,
    job: CudaJ2kIdwtJob,
) -> Result<ValidatedIdwtJob, CudaError> {
    validate_idwt_job_layout(
        bands.map(CudaDeviceBuffer::byte_len),
        output.map(CudaDeviceBuffer::byte_len),
        job,
    )
}

pub(in crate::j2k_decode) fn validate_idwt_target(
    target: &CudaJ2kIdwtTarget<'_>,
) -> Result<bool, CudaError> {
    validate_idwt_job(
        [target.ll, target.hl, target.lh, target.hh],
        Some(target.output),
        target.job,
    )
    .map(ValidatedIdwtJob::is_empty)
}

fn validate_idwt_job_layout(
    band_bytes: [usize; 4],
    output_bytes: Option<usize>,
    job: CudaJ2kIdwtJob,
) -> Result<ValidatedIdwtJob, CudaError> {
    let (width, height) = checked_rect_dimensions("output", job.rect)?;
    if width != 0 && height != 0 && (width == u32::MAX || height == u32::MAX) {
        return Err(CudaError::InvalidArgument {
            message: format!(
                "J2K IDWT output geometry {width}x{height} exceeds the CUDA u32 iteration ABI"
            ),
        });
    }
    let (_, required_output_bytes) = checked_f32_rect_layout("output", width, height)?;

    for (name, rect, low_x, low_y, available_bytes) in [
        ("LL", job.ll_rect, true, true, band_bytes[0]),
        ("HL", job.hl_rect, false, true, band_bytes[1]),
        ("LH", job.lh_rect, true, false, band_bytes[2]),
        ("HH", job.hh_rect, false, false, band_bytes[3]),
    ] {
        let (band_width, band_height) = checked_rect_dimensions(name, rect)?;
        let expected_width = idwt_band_extent(job.rect.x0, job.rect.x1, low_x);
        let expected_height = idwt_band_extent(job.rect.y0, job.rect.y1, low_y);
        if band_width != expected_width || band_height != expected_height {
            return Err(CudaError::InvalidArgument {
                message: format!(
                    "J2K IDWT {name} geometry {band_width}x{band_height} does not match the \
                     {width}x{height} output geometry; expected {expected_width}x{expected_height}"
                ),
            });
        }
        let (_, required_bytes) = checked_f32_rect_layout(name, band_width, band_height)?;
        if available_bytes < required_bytes {
            return Err(CudaError::InvalidArgument {
                message: format!(
                    "J2K IDWT {name} buffer is too small: required {required_bytes} bytes, \
                     have {available_bytes}"
                ),
            });
        }
    }

    if let Some(available) = output_bytes {
        if available < required_output_bytes {
            return Err(CudaError::OutputTooSmall {
                required: required_output_bytes,
                have: available,
            });
        }
    }

    Ok(ValidatedIdwtJob {
        width,
        height,
        output_bytes: required_output_bytes,
    })
}

fn checked_rect_dimensions(name: &str, rect: CudaJ2kRect) -> Result<(u32, u32), CudaError> {
    let Some(width) = rect.x1.checked_sub(rect.x0) else {
        return Err(CudaError::InvalidArgument {
            message: format!(
                "J2K IDWT {name} rectangle has inverted x bounds: {}..{}",
                rect.x0, rect.x1
            ),
        });
    };
    let Some(height) = rect.y1.checked_sub(rect.y0) else {
        return Err(CudaError::InvalidArgument {
            message: format!(
                "J2K IDWT {name} rectangle has inverted y bounds: {}..{}",
                rect.y0, rect.y1
            ),
        });
    };
    Ok((width, height))
}

fn idwt_band_extent(start: u32, end: u32, low_pass: bool) -> u32 {
    let half = |value: u32| {
        if low_pass {
            value / 2 + value % 2
        } else {
            value / 2
        }
    };
    half(end) - half(start)
}

fn checked_f32_rect_layout(
    name: &str,
    width: u32,
    height: u32,
) -> Result<(usize, usize), CudaError> {
    let words = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or_else(|| CudaError::InvalidArgument {
            message: format!("J2K IDWT {name} sample count overflows u64"),
        })?;
    if words.saturating_sub(1) > u64::from(u32::MAX) {
        return Err(CudaError::InvalidArgument {
            message: format!(
                "J2K IDWT {name} sample count {words} exceeds the CUDA u32 indexing ABI"
            ),
        });
    }
    let words = usize::try_from(words).map_err(|_| CudaError::InvalidArgument {
        message: format!("J2K IDWT {name} sample count cannot be represented by host usize"),
    })?;
    let bytes = checked_f32_words_byte_len(words)?;
    Ok((words, bytes))
}

#[cfg(test)]
mod tests;
