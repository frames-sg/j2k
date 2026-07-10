// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{size_of, Error, J2kLosslessDevicePrepareJob};

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(in crate::compute) struct J2kLosslessPrepareSizes {
    pub(in crate::compute) plane_len: usize,
    pub(in crate::compute) plane_bytes: usize,
    pub(in crate::compute) coefficient_bytes: usize,
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn lossless_prepare_sizes(
    job: J2kLosslessDevicePrepareJob<'_>,
) -> Result<J2kLosslessPrepareSizes, Error> {
    if job.component_count != 1 && job.component_count != 3 {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal resident encode supports grayscale or RGB input",
        });
    }
    if job.bytes_per_sample != 1 && job.bytes_per_sample != 2 {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal resident encode supports 8-bit or 16-bit samples",
        });
    }
    let plane_len = (job.output_width as usize)
        .checked_mul(job.output_height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal resident encode plane size overflow".to_string(),
        })?;
    let plane_bytes =
        plane_len
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal resident encode plane byte size overflow".to_string(),
            })?;
    let coefficient_bytes = job
        .coefficient_count
        .max(1)
        .checked_mul(size_of::<i32>())
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal resident encode coefficient size overflow".to_string(),
        })?;
    Ok(J2kLosslessPrepareSizes {
        plane_len,
        plane_bytes,
        coefficient_bytes,
    })
}
