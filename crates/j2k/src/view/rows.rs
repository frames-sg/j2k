// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded row-decode planning, scratch ownership, and sink emission.

use super::{J2kDecoder, J2kRowDecodeOptions};
use crate::{decode::J2kDecodeOutcome, scratch::J2kScratchPool, J2kError};
use alloc::vec::Vec;
use j2k_core::{
    BufferError, DecodeRowsError, PixelFormat, Rect, RowSink, DEFAULT_MAX_HOST_ALLOCATION_BYTES,
};

impl J2kRowDecodeOptions {
    /// Create row decode options with the requested maximum stripe height.
    ///
    /// A zero value is normalized to one row.
    pub const fn new(max_rows_per_stripe: u32) -> Self {
        Self {
            max_rows_per_stripe,
            max_stripe_bytes: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        }
    }

    /// Create row decode options with explicit stripe height and byte caps.
    pub const fn new_with_max_stripe_bytes(
        max_rows_per_stripe: u32,
        max_stripe_bytes: usize,
    ) -> Self {
        Self {
            max_rows_per_stripe,
            max_stripe_bytes,
        }
    }

    /// Return a copy of these options with an explicit stripe byte cap.
    #[must_use]
    pub const fn with_max_stripe_bytes(mut self, max_stripe_bytes: usize) -> Self {
        self.max_stripe_bytes = max_stripe_bytes;
        self
    }

    /// Maximum number of decoded rows held per bounded row-decode stripe.
    pub const fn max_rows_per_stripe(self) -> u32 {
        if self.max_rows_per_stripe == 0 {
            1
        } else {
            self.max_rows_per_stripe
        }
    }

    /// Maximum number of packed bytes held per bounded row-decode stripe.
    pub const fn max_stripe_bytes(self) -> usize {
        self.max_stripe_bytes
    }
}

impl Default for J2kRowDecodeOptions {
    fn default() -> Self {
        Self::new(64)
    }
}

impl J2kDecoder<'_> {
    /// Decode rows into a `u8` row sink while bounding host output scratch to
    /// at most `options.max_rows_per_stripe()` rows.
    ///
    /// # Errors
    /// Returns a decode error for unsupported formats or malformed input, and
    /// forwards sink errors without converting them to successful decodes.
    pub fn decode_rows_u8_bounded<R: RowSink<u8>>(
        &mut self,
        sink: &mut R,
        options: J2kRowDecodeOptions,
    ) -> Result<J2kDecodeOutcome, DecodeRowsError<J2kError, R::Error>> {
        let fmt = row_format_u8(self.info()).map_err(DecodeRowsError::Decode)?;
        let row_bytes = row_bytes_for(self.info(), fmt).map_err(DecodeRowsError::Decode)?;
        let width = self.info.dimensions.0;
        let height = self.info.dimensions.1;
        let options = options.with_max_stripe_bytes(
            options
                .max_stripe_bytes()
                .min(DEFAULT_MAX_HOST_ALLOCATION_BYTES),
        );
        let (stripe_rows, max_stripe_len) = bounded_row_stripe_layout(row_bytes, height, options)
            .map_err(DecodeRowsError::Decode)?;
        let mut pool = J2kScratchPool::new();
        let mut y = 0_u32;
        while y < height {
            let rows = stripe_rows.min(height - y);
            let stripe_len = row_bytes.checked_mul(rows as usize).ok_or_else(|| {
                DecodeRowsError::Decode(J2kError::Buffer(BufferError::SizeOverflow {
                    what: "J2K bounded row decode stripe buffer",
                }))
            })?;
            let stripe = pool
                .packed_bytes(max_stripe_len)
                .map_err(|error| DecodeRowsError::Decode(J2kError::Buffer(error)))?;
            self.decode_region_into_cached(
                &mut stripe[..stripe_len],
                row_bytes,
                fmt,
                Rect {
                    x: 0,
                    y,
                    w: width,
                    h: rows,
                },
            )
            .map_err(DecodeRowsError::Decode)?;
            for row_index in 0..rows {
                let start = row_index as usize * row_bytes;
                sink.write_row(y + row_index, &stripe[start..start + row_bytes])
                    .map_err(DecodeRowsError::Sink)?;
            }
            y += rows;
        }
        Ok(j2k_core::DecodeOutcome::new(
            Rect::full(self.info.dimensions),
            Vec::new(),
        ))
    }

    /// Decode rows into a `u16` row sink while bounding host output scratch to
    /// at most `options.max_rows_per_stripe()` rows.
    ///
    /// # Errors
    /// Returns a decode error for unsupported formats or malformed input, and
    /// forwards sink errors without converting them to successful decodes.
    pub fn decode_rows_u16_bounded<R: RowSink<u16>>(
        &mut self,
        sink: &mut R,
        options: J2kRowDecodeOptions,
    ) -> Result<J2kDecodeOutcome, DecodeRowsError<J2kError, R::Error>> {
        let fmt = row_format_u16(self.info()).map_err(DecodeRowsError::Decode)?;
        let row_bytes = row_bytes_for(self.info(), fmt).map_err(DecodeRowsError::Decode)?;
        let samples_per_row = row_samples_for(self.info(), fmt).map_err(DecodeRowsError::Decode)?;
        let width = self.info.dimensions.0;
        let height = self.info.dimensions.1;
        let row_scratch_bytes = samples_per_row
            .checked_mul(core::mem::size_of::<u16>())
            .ok_or(DecodeRowsError::Decode(J2kError::Buffer(
                BufferError::SizeOverflow {
                    what: "J2K bounded row u16 scratch",
                },
            )))?;
        let packed_cap = DEFAULT_MAX_HOST_ALLOCATION_BYTES
            .checked_sub(row_scratch_bytes)
            .ok_or(DecodeRowsError::Decode(J2kError::Buffer(
                BufferError::AllocationTooLarge {
                    requested: row_scratch_bytes,
                    cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                    what: "J2K bounded row u16 scratch",
                },
            )))?;
        let options = options.with_max_stripe_bytes(options.max_stripe_bytes().min(packed_cap));
        let (stripe_rows, max_stripe_len) = bounded_row_stripe_layout(row_bytes, height, options)
            .map_err(DecodeRowsError::Decode)?;
        let mut pool = J2kScratchPool::new();
        let mut y = 0_u32;
        while y < height {
            let rows = stripe_rows.min(height - y);
            let stripe_len = row_bytes.checked_mul(rows as usize).ok_or_else(|| {
                DecodeRowsError::Decode(J2kError::Buffer(BufferError::SizeOverflow {
                    what: "J2K bounded row decode stripe buffer",
                }))
            })?;
            let (packed, row) = pool
                .packed_bytes_and_row_u16(max_stripe_len, samples_per_row)
                .map_err(|error| DecodeRowsError::Decode(J2kError::Buffer(error)))?;
            self.decode_region_into_cached(
                &mut packed[..stripe_len],
                row_bytes,
                fmt,
                Rect {
                    x: 0,
                    y,
                    w: width,
                    h: rows,
                },
            )
            .map_err(DecodeRowsError::Decode)?;
            for row_index in 0..rows {
                let start = row_index as usize * row_bytes;
                let packed_row = &packed[start..start + row_bytes];
                for (dst, src) in row.iter_mut().zip(packed_row.chunks_exact(2)) {
                    *dst = u16::from_le_bytes([src[0], src[1]]);
                }
                sink.write_row(y + row_index, row)
                    .map_err(DecodeRowsError::Sink)?;
            }
            y += rows;
        }
        Ok(j2k_core::DecodeOutcome::new(
            Rect::full(self.info.dimensions),
            Vec::new(),
        ))
    }
}

fn row_format_u8(info: &j2k_core::Info) -> Result<PixelFormat, J2kError> {
    match info.components {
        1 => Ok(PixelFormat::Gray8),
        3 => Ok(PixelFormat::Rgb8),
        4 => Ok(PixelFormat::Rgba8),
        _ => Err(j2k_core::Unsupported {
            what: "row decode only supports Gray/RGB/RGBA images in J2K-M2",
        }
        .into()),
    }
}

fn row_format_u16(info: &j2k_core::Info) -> Result<PixelFormat, J2kError> {
    match info.components {
        1 => Ok(PixelFormat::Gray16),
        3 => Ok(PixelFormat::Rgb16),
        4 => Ok(PixelFormat::Rgba16),
        _ => Err(j2k_core::Unsupported {
            what: "row decode only supports Gray/RGB/RGBA images in J2K-M2",
        }
        .into()),
    }
}

fn bounded_row_stripe_layout(
    row_bytes: usize,
    height: u32,
    options: J2kRowDecodeOptions,
) -> Result<(u32, usize), J2kError> {
    let cap = options.max_stripe_bytes();
    if row_bytes > cap {
        return Err(J2kError::Buffer(BufferError::AllocationTooLarge {
            requested: row_bytes,
            cap,
            what: "J2K bounded row decode stripe buffer",
        }));
    }

    let max_rows = options.max_rows_per_stripe();
    let image_rows = height.max(1);
    let rows_by_cap = cap.checked_div(row_bytes).map_or(max_rows, |capped_rows| {
        u32::try_from(capped_rows).unwrap_or(u32::MAX).max(1)
    });
    let stripe_rows = max_rows.min(image_rows).min(rows_by_cap);
    let max_stripe_len = row_bytes
        .checked_mul(stripe_rows as usize)
        .ok_or(J2kError::Buffer(BufferError::SizeOverflow {
            what: "J2K bounded row decode stripe buffer",
        }))?;

    Ok((stripe_rows, max_stripe_len))
}

fn row_bytes_for(info: &j2k_core::Info, fmt: PixelFormat) -> Result<usize, J2kError> {
    (info.dimensions.0 as usize)
        .checked_mul(fmt.bytes_per_pixel())
        .ok_or(J2kError::DimensionOverflow {
            width: info.dimensions.0,
            height: info.dimensions.1,
        })
}

fn row_samples_for(info: &j2k_core::Info, fmt: PixelFormat) -> Result<usize, J2kError> {
    (info.dimensions.0 as usize)
        .checked_mul(fmt.channels())
        .ok_or(J2kError::DimensionOverflow {
            width: info.dimensions.0,
            height: info.dimensions.1,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_row_stripe_layout_clamps_rows_to_byte_cap() {
        let options = J2kRowDecodeOptions::new_with_max_stripe_bytes(100, 1_024);

        let (rows, bytes) =
            bounded_row_stripe_layout(100, 50, options).expect("stripe layout should fit");

        assert_eq!(rows, 10);
        assert_eq!(bytes, 1_000);
    }

    #[test]
    fn bounded_row_stripe_layout_rejects_single_row_over_cap() {
        let options = J2kRowDecodeOptions::new_with_max_stripe_bytes(100, 99);

        let err =
            bounded_row_stripe_layout(100, 50, options).expect_err("single row should exceed cap");

        assert!(matches!(
            err,
            J2kError::Buffer(BufferError::AllocationTooLarge {
                requested: 100,
                cap: 99,
                what: "J2K bounded row decode stripe buffer",
            })
        ));
    }
}
