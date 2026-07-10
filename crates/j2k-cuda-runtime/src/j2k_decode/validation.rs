// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    error::CudaError,
    memory::{checked_image_words, CudaDeviceBuffer},
};

use super::types::{CudaJ2kIdwtMultiKernelJob, CudaJ2kIdwtTarget, CudaJ2kRect};

pub(crate) fn active_dwt53_buffers<'a>(
    buffer_a: &'a CudaDeviceBuffer,
    buffer_b: &'a CudaDeviceBuffer,
    active_is_a: bool,
) -> (&'a CudaDeviceBuffer, &'a CudaDeviceBuffer) {
    if active_is_a {
        (buffer_a, buffer_b)
    } else {
        (buffer_b, buffer_a)
    }
}

pub(crate) fn j2k_idwt_multi_kernel_jobs(
    targets: &[CudaJ2kIdwtTarget<'_>],
) -> Result<Vec<CudaJ2kIdwtMultiKernelJob>, CudaError> {
    let mut kernel_jobs = Vec::with_capacity(targets.len());
    for target in targets {
        let width = target.job.rect.x1.saturating_sub(target.job.rect.x0);
        let height = target.job.rect.y1.saturating_sub(target.job.rect.y0);
        if width == 0 || height == 0 {
            continue;
        }
        super::ensure_idwt_buffer_len(target.output, target.job.rect)?;
        super::ensure_idwt_buffer_len(target.ll, target.job.ll_rect)?;
        super::ensure_idwt_buffer_len(target.hl, target.job.hl_rect)?;
        super::ensure_idwt_buffer_len(target.lh, target.job.lh_rect)?;
        super::ensure_idwt_buffer_len(target.hh, target.job.hh_rect)?;
        kernel_jobs.push(CudaJ2kIdwtMultiKernelJob {
            ll_ptr: target.ll.device_ptr(),
            hl_ptr: target.hl.device_ptr(),
            lh_ptr: target.lh.device_ptr(),
            hh_ptr: target.hh.device_ptr(),
            output_ptr: target.output.device_ptr(),
            job: target.job,
        });
    }
    Ok(kernel_jobs)
}

pub(crate) fn ensure_idwt_buffer_len(
    buffer: &CudaDeviceBuffer,
    rect: CudaJ2kRect,
) -> Result<(), CudaError> {
    let width = rect.x1.saturating_sub(rect.x0);
    let height = rect.y1.saturating_sub(rect.y0);
    let words = checked_image_words(width, height, 1)?;
    let bytes = super::checked_f32_words_byte_len(words)?;
    if bytes > buffer.byte_len() {
        return Err(CudaError::OutputTooSmall {
            required: bytes,
            have: buffer.byte_len(),
        });
    }
    Ok(())
}

pub(crate) fn checked_f32_words_byte_len(words: usize) -> Result<usize, CudaError> {
    words
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(CudaError::LengthTooLarge { len: words })
}

pub(crate) fn validate_store_rgb8_plane(
    plane: &CudaDeviceBuffer,
    input_width: u32,
    source_x: u32,
    source_y: u32,
    copy_width: u32,
    copy_height: u32,
) -> Result<(), CudaError> {
    if source_x
        .checked_add(copy_width)
        .is_none_or(|end_x| end_x > input_width)
    {
        return Err(CudaError::LengthTooLarge {
            len: plane.byte_len(),
        });
    }
    let last_sample = if copy_height == 0 {
        0
    } else {
        (source_y as usize)
            .checked_add(copy_height as usize - 1)
            .and_then(|row| row.checked_mul(input_width as usize))
            .and_then(|row| row.checked_add(source_x as usize))
            .and_then(|row| row.checked_add(copy_width as usize))
            .ok_or(CudaError::LengthTooLarge {
                len: plane.byte_len(),
            })?
    };
    let required_bytes =
        last_sample
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaError::LengthTooLarge {
                len: plane.byte_len(),
            })?;
    if required_bytes > plane.byte_len() {
        return Err(CudaError::LengthTooLarge {
            len: required_bytes,
        });
    }
    Ok(())
}
