// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{error::BufferError, pixel::PixelFormat};

/// Default cap for host-side codec-owned allocations.
pub const DEFAULT_MAX_HOST_ALLOCATION_BYTES: usize = 512 * 1024 * 1024;

/// Returns `len` if it is at or below `cap`.
pub fn ensure_allocation_within_cap(
    len: usize,
    cap: usize,
    what: &'static str,
) -> Result<usize, BufferError> {
    if len > cap {
        return Err(BufferError::AllocationTooLarge {
            requested: len,
            cap,
            what,
        });
    }
    Ok(len)
}

/// Returns the number of bytes required for a strided image output buffer.
///
/// The returned length covers the last written byte of the final row and does
/// not include trailing padding after that row.
pub fn strided_output_len(
    dimensions: (u32, u32),
    stride: usize,
    fmt: PixelFormat,
) -> Result<usize, BufferError> {
    if dimensions.0 == 0 || dimensions.1 == 0 {
        return Ok(0);
    }

    let row_bytes = row_bytes(dimensions.0, fmt)?;
    stride
        .checked_mul(dimensions.1 as usize - 1)
        .and_then(|prefix| prefix.checked_add(row_bytes))
        .ok_or(BufferError::SizeOverflow {
            what: "strided output size",
        })
}

/// Returns the strided output byte length, rejecting requests over `cap`.
pub fn strided_output_len_capped(
    dimensions: (u32, u32),
    stride: usize,
    fmt: PixelFormat,
    cap: usize,
    what: &'static str,
) -> Result<usize, BufferError> {
    let len = strided_output_len(dimensions, stride, fmt)?;
    ensure_allocation_within_cap(len, cap, what)
}

/// Validates that `out_len` and `stride` can hold an image output.
pub fn validate_strided_output_buffer(
    dimensions: (u32, u32),
    out_len: usize,
    stride: usize,
    fmt: PixelFormat,
) -> Result<(), BufferError> {
    if dimensions.0 == 0 || dimensions.1 == 0 {
        return Ok(());
    }

    let row_bytes = row_bytes(dimensions.0, fmt)?;
    if stride < row_bytes {
        return Err(BufferError::StrideTooSmall { row_bytes, stride });
    }
    let required = strided_output_len(dimensions, stride, fmt)?;
    if out_len < required {
        return Err(BufferError::OutputTooSmall {
            required,
            have: out_len,
        });
    }
    Ok(())
}

/// Copy tightly packed pixel rows into a caller-provided strided output buffer.
///
/// `src` must contain at least `width * height * fmt.bytes_per_pixel()` bytes.
/// The destination may have row padding, expressed by `stride`.
pub fn copy_tight_pixels_to_strided_output(
    src: &[u8],
    dimensions: (u32, u32),
    fmt: PixelFormat,
    out: &mut [u8],
    stride: usize,
) -> Result<(), BufferError> {
    if dimensions.0 == 0 || dimensions.1 == 0 {
        return Ok(());
    }

    let row_bytes = row_bytes(dimensions.0, fmt)?;
    let height = dimensions.1 as usize;
    let required_src = row_bytes
        .checked_mul(height)
        .ok_or(BufferError::SizeOverflow {
            what: "tight source size",
        })?;
    if src.len() < required_src {
        return Err(BufferError::InputTooSmall {
            required: required_src,
            have: src.len(),
        });
    }
    validate_strided_output_buffer(dimensions, out.len(), stride, fmt)?;

    for y in 0..dimensions.1 as usize {
        let src_row = &src[y * row_bytes..(y + 1) * row_bytes];
        let dst_start = y * stride;
        out[dst_start..dst_start + row_bytes].copy_from_slice(src_row);
    }

    Ok(())
}

fn row_bytes(width: u32, fmt: PixelFormat) -> Result<usize, BufferError> {
    (width as usize)
        .checked_mul(fmt.bytes_per_pixel())
        .ok_or(BufferError::SizeOverflow {
            what: "row byte count",
        })
}
