// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{context::CudaContext, error::CudaError, memory::CudaDeviceBuffer};

pub(super) fn validate_store_destination(
    output_width: u32,
    output_height: u32,
    output_x: u32,
    output_y: u32,
    copy_width: u32,
    copy_height: u32,
    channels: u32,
) -> Result<bool, CudaError> {
    if channels == 0 {
        return Err(CudaError::InvalidArgument {
            message: "J2K store destination requires at least one channel".to_string(),
        });
    }
    if copy_width.checked_mul(copy_height).is_none() {
        return Err(CudaError::InvalidArgument {
            message: format!(
                "J2K store copy geometry {copy_width}x{copy_height} exceeds the CUDA u32 kernel ABI"
            ),
        });
    }
    let end_x = output_x
        .checked_add(copy_width)
        .ok_or_else(|| CudaError::InvalidArgument {
            message: format!(
                "J2K store destination x extent overflows u32: {output_x} + {copy_width}"
            ),
        })?;
    let end_y = output_y
        .checked_add(copy_height)
        .ok_or_else(|| CudaError::InvalidArgument {
            message: format!(
                "J2K store destination y extent overflows u32: {output_y} + {copy_height}"
            ),
        })?;

    if end_x > output_width || end_y > output_height {
        return Err(CudaError::InvalidArgument {
            message: format!(
                "J2K store destination rectangle ({output_x}, {output_y})..({end_x}, {end_y}) \
                 exceeds output bounds {output_width}x{output_height}"
            ),
        });
    }

    if copy_width != 0 && copy_height != 0 {
        let last_pixel = u64::from(end_y - 1)
            .checked_mul(u64::from(output_width))
            .and_then(|row| row.checked_add(u64::from(end_x - 1)))
            .ok_or_else(|| CudaError::InvalidArgument {
                message: "J2K store destination pixel index overflows u64".to_string(),
            })?;
        let last_element = last_pixel
            .checked_mul(u64::from(channels))
            .and_then(|offset| offset.checked_add(u64::from(channels - 1)))
            .ok_or_else(|| CudaError::InvalidArgument {
                message: "J2K store destination element index overflows u64".to_string(),
            })?;
        if last_element > u64::from(u32::MAX) {
            return Err(CudaError::InvalidArgument {
                message: format!(
                    "J2K store destination element index {last_element} exceeds the CUDA u32 kernel ABI"
                ),
            });
        }
    }

    Ok(
        output_x == 0
            && output_y == 0
            && copy_width == output_width
            && copy_height == output_height,
    )
}

pub(super) fn zero_unwritten_store_output(
    context: &CudaContext,
    output: &CudaDeviceBuffer,
    output_bytes: usize,
    full_coverage: bool,
) -> Result<bool, CudaError> {
    if output_bytes == 0 || full_coverage {
        return Ok(false);
    }
    context.memset_d8(output, 0, output_bytes)?;
    Ok(true)
}
