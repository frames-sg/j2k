// SPDX-License-Identifier: MIT OR Apache-2.0

use super::extended12::lossless_color_sampling;
use super::{JpegError, LosslessColorSampling, PreparedComponentPlan, DEFAULT_MAX_DECODE_BYTES};
use crate::info::{Info, SamplingFactors};

pub(super) fn compute_decode_scratch_bytes(
    (width, height): (u32, u32),
    sampling: SamplingFactors,
    cap: usize,
) -> Result<usize, JpegError> {
    let max_h = u32::from(sampling.max_h);
    let max_v = u32::from(sampling.max_v);
    let mcu_width = 8u32
        .checked_mul(max_h)
        .ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap,
        })?;
    let mcu_height = 8u32
        .checked_mul(max_v)
        .ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap,
        })?;
    let mcus_per_row = width.div_ceil(mcu_width);
    let _mcu_rows = height.div_ceil(mcu_height);

    let mut stripe_total = 0usize;
    for (h, v) in sampling.iter() {
        let cols = checked_usize_product(&[mcus_per_row as usize, usize::from(h), 8usize], cap)?;
        let rows = checked_usize_product(&[usize::from(v), 8usize], cap)?;
        let plane = cols.checked_mul(rows).ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap,
        })?;
        stripe_total = stripe_total
            .checked_add(plane)
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap,
            })?;
        if stripe_total > cap {
            return Err(JpegError::MemoryCapExceeded {
                requested: stripe_total,
                cap,
            });
        }
    }

    let stripe_buffers = checked_usize_product(&[stripe_total, 3], cap)?;
    let row_scratch = checked_usize_product(&[width as usize, 7], cap)?;
    let total = stripe_buffers
        .checked_add(row_scratch)
        .ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap,
        })?;
    if total > cap {
        return Err(JpegError::MemoryCapExceeded {
            requested: total,
            cap,
        });
    }

    Ok(total)
}

/// Checked size for a transient full-frame intermediate buffer, enforcing the
/// decode memory cap at the allocation site.
pub(super) fn checked_scratch_len(factors: &[usize]) -> Result<usize, JpegError> {
    let cap = DEFAULT_MAX_DECODE_BYTES;
    let len = checked_usize_product(factors, cap)?;
    if len > cap {
        return Err(JpegError::MemoryCapExceeded {
            requested: len,
            cap,
        });
    }
    Ok(len)
}

pub(super) fn compute_lossless_scratch_bytes(info: &Info, cap: usize) -> Result<usize, JpegError> {
    // Only the sampled-color lossless paths materialize full-frame component
    // planes; grayscale and 4:4:4 decode stream straight into caller buffers.
    // Region/scaled lossless intermediates are capped at their allocation
    // sites via checked_scratch_len.
    if !matches!(
        lossless_color_sampling(info),
        Some(LosslessColorSampling::S422 | LosslessColorSampling::S420)
    ) {
        return Ok(0);
    }
    let width = info.dimensions.0 as usize;
    let height = info.dimensions.1 as usize;
    let bytes_per_sample: usize = if info.bit_depth > 8 { 2 } else { 1 };
    let chroma_width = width.div_ceil(usize::from(info.sampling.max_h));
    let chroma_height = height.div_ceil(usize::from(info.sampling.max_v));
    let luma = checked_usize_product(&[width, height, bytes_per_sample], cap)?;
    let chroma = checked_usize_product(&[chroma_width, chroma_height, bytes_per_sample, 2], cap)?;
    let total = luma
        .checked_add(chroma)
        .ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap,
        })?;
    if total > cap {
        return Err(JpegError::MemoryCapExceeded {
            requested: total,
            cap,
        });
    }
    Ok(total)
}

pub(super) fn compute_extended12_planes_scratch_bytes(
    components: &[PreparedComponentPlan],
    (width, height): (u32, u32),
    sampling: SamplingFactors,
    cap: usize,
) -> Result<usize, JpegError> {
    let mcu_cols = width.div_ceil(u32::from(sampling.max_h) * 8) as usize;
    let mcu_rows = height.div_ceil(u32::from(sampling.max_v) * 8) as usize;
    let mut total = 0usize;
    for component in components {
        let stride = checked_usize_product(&[mcu_cols, usize::from(component.h), 8], cap)?;
        let rows = checked_usize_product(&[mcu_rows, usize::from(component.v), 8], cap)?;
        let plane = checked_usize_product(&[stride, rows, core::mem::size_of::<u16>()], cap)?;
        total = total
            .checked_add(plane)
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap,
            })?;
    }
    if total > cap {
        return Err(JpegError::MemoryCapExceeded {
            requested: total,
            cap,
        });
    }
    Ok(total)
}

pub(super) fn checked_usize_product(factors: &[usize], cap: usize) -> Result<usize, JpegError> {
    let mut value = 1usize;
    for factor in factors {
        value = value
            .checked_mul(*factor)
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap,
            })?;
    }
    Ok(value)
}
