// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{EncodeError, EncodeResult};

const MAX_TIER1_CODE_BLOCK_AXIS: usize = 1024;
const MAX_TIER1_CODE_BLOCK_SAMPLES: usize = 4096;

pub(crate) fn validate_tier1_code_block_geometry(
    width: usize,
    height: usize,
) -> EncodeResult<usize> {
    if width == 0 || height == 0 {
        return Err(EncodeError::InvalidInput {
            what: "Tier-1 code-block dimensions must be non-zero",
        });
    }
    let samples = width
        .checked_mul(height)
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "Tier-1 code-block sample count",
        })?;
    if width > MAX_TIER1_CODE_BLOCK_AXIS
        || height > MAX_TIER1_CODE_BLOCK_AXIS
        || samples > MAX_TIER1_CODE_BLOCK_SAMPLES
    {
        return Err(EncodeError::InvalidInput {
            what: "Tier-1 code-block geometry exceeds JPEG 2000 limits",
        });
    }
    Ok(samples)
}

/// Validated borrowed coefficient rectangle with an explicit source row stride.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CoefficientBlockView<'a, T> {
    values: &'a [T],
    offset: usize,
    width: usize,
    height: usize,
    row_stride: usize,
}

impl<'a, T> CoefficientBlockView<'a, T> {
    pub(crate) fn try_new(
        values: &'a [T],
        offset: usize,
        width: usize,
        height: usize,
        row_stride: usize,
    ) -> EncodeResult<Self> {
        if (width == 0) != (height == 0) {
            return Err(EncodeError::InvalidInput {
                what: "coefficient block dimensions must both be zero or non-zero",
            });
        }
        if width == 0 {
            if offset > values.len() {
                return Err(EncodeError::InvalidInput {
                    what: "empty coefficient block offset exceeds storage",
                });
            }
            return Ok(Self {
                values,
                offset,
                width,
                height,
                row_stride,
            });
        }
        if row_stride < width {
            return Err(EncodeError::InvalidInput {
                what: "coefficient block row stride is shorter than its width",
            });
        }
        let end = height
            .checked_sub(1)
            .and_then(|rows| rows.checked_mul(row_stride))
            .and_then(|rows| rows.checked_add(offset))
            .and_then(|last_row| last_row.checked_add(width))
            .ok_or(EncodeError::ArithmeticOverflow {
                what: "strided coefficient block range",
            })?;
        if end > values.len() {
            return Err(EncodeError::InvalidInput {
                what: "coefficient block range exceeds storage",
            });
        }
        Ok(Self {
            values,
            offset,
            width,
            height,
            row_stride,
        })
    }

    pub(crate) fn try_contiguous(
        values: &'a [T],
        width: usize,
        height: usize,
    ) -> EncodeResult<Self> {
        let expected = width
            .checked_mul(height)
            .ok_or(EncodeError::ArithmeticOverflow {
                what: "contiguous coefficient block length",
            })?;
        if values.len() != expected {
            return Err(EncodeError::InvalidInput {
                what: "contiguous coefficient block length mismatch",
            });
        }
        Self::try_new(values, 0, width, height, width)
    }

    pub(crate) fn width(self) -> usize {
        self.width
    }

    pub(crate) fn height(self) -> usize {
        self.height
    }

    pub(crate) fn row(self, y: usize) -> Option<&'a [T]> {
        if y >= self.height {
            return None;
        }
        let start = self.offset + y * self.row_stride;
        Some(&self.values[start..start + self.width])
    }

    pub(crate) fn rows(self) -> impl ExactSizeIterator<Item = &'a [T]> {
        (0..self.height).map(move |y| {
            let start = self.offset + y * self.row_stride;
            &self.values[start..start + self.width]
        })
    }
}

impl<T: Copy> CoefficientBlockView<'_, T> {
    pub(crate) fn get(self, x: usize, y: usize) -> Option<T> {
        self.row(y).and_then(|row| row.get(x)).copied()
    }

    /// Load a coefficient using the validated row-major block index.
    ///
    /// Callers must pass an index below `width * height`, matching ordinary
    /// slice indexing used by contiguous coefficient sources.
    pub(crate) fn value_at_linear_index(self, index: usize) -> T {
        let y = index / self.width;
        let x = index % self.width;
        self.values[self.offset + y * self.row_stride + x]
    }
}

pub(crate) trait SignedCoefficient: Copy {
    fn unsigned_magnitude(self) -> u64;
    fn is_negative(self) -> bool;
}

impl SignedCoefficient for i32 {
    fn unsigned_magnitude(self) -> u64 {
        u64::from(self.unsigned_abs())
    }

    fn is_negative(self) -> bool {
        self < 0
    }
}

impl SignedCoefficient for i64 {
    fn unsigned_magnitude(self) -> u64 {
        self.unsigned_abs()
    }

    fn is_negative(self) -> bool {
        self < 0
    }
}

#[cfg(test)]
pub(crate) fn legacy_coefficient_view_error(error: EncodeError) -> &'static str {
    match error {
        EncodeError::InvalidInput { what }
        | EncodeError::Unsupported { what }
        | EncodeError::ArithmeticOverflow { what }
        | EncodeError::InternalInvariant { what } => what,
        EncodeError::AllocationTooLarge { .. } => "coefficient block exceeds the allocation cap",
        EncodeError::HostAllocationFailed { .. } => "coefficient block allocation failed",
        EncodeError::Accelerator { source, .. } => source.reason(),
        EncodeError::CodestreamValidation { detail } => detail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier1_geometry_accepts_boundary_and_rejects_invalid_shapes() {
        assert_eq!(
            validate_tier1_code_block_geometry(1024, 4).expect("maximum axis boundary is valid"),
            4096
        );
        for (width, height) in [(1024, 5), (1025, 1)] {
            assert_eq!(
                validate_tier1_code_block_geometry(width, height)
                    .expect_err("geometry exceeds the Tier-1 limits"),
                EncodeError::InvalidInput {
                    what: "Tier-1 code-block geometry exceeds JPEG 2000 limits",
                }
            );
        }
        assert_eq!(
            validate_tier1_code_block_geometry(0, 0).expect_err("empty Tier-1 geometry is invalid"),
            EncodeError::InvalidInput {
                what: "Tier-1 code-block dimensions must be non-zero",
            }
        );
    }

    #[test]
    fn empty_block_requires_both_axes_zero_and_a_valid_offset() {
        let values = [1_i32, 2, 3];
        let empty = CoefficientBlockView::try_new(&values, values.len(), 0, 0, 0)
            .expect("both-zero block at end of storage is valid");
        assert_eq!(empty.width(), 0);
        assert_eq!(empty.height(), 0);
        assert!(empty.row(0).is_none());

        for (width, height) in [(0, 1), (1, 0)] {
            assert_eq!(
                CoefficientBlockView::try_new(&values, 0, width, height, 1)
                    .expect_err("mismatched zero axes are rejected"),
                EncodeError::InvalidInput {
                    what: "coefficient block dimensions must both be zero or non-zero",
                }
            );
        }
        assert_eq!(
            CoefficientBlockView::try_new(&values, values.len() + 1, 0, 0, 0)
                .expect_err("empty block offset must remain in storage"),
            EncodeError::InvalidInput {
                what: "empty coefficient block offset exceeds storage",
            }
        );
    }

    #[test]
    fn contiguous_adapter_requires_exact_length() {
        let exact = [1_i32, 2, 3, 4, 5, 6];
        CoefficientBlockView::try_contiguous(&exact, 3, 2)
            .expect("exact contiguous storage is accepted");

        assert_eq!(
            CoefficientBlockView::try_contiguous(&exact[..5], 3, 2)
                .expect_err("one-short contiguous storage is rejected"),
            EncodeError::InvalidInput {
                what: "contiguous coefficient block length mismatch",
            }
        );
        let with_trailing = [1_i32, 2, 3, 4, 5, 6, 7];
        assert_eq!(
            CoefficientBlockView::try_contiguous(&with_trailing, 3, 2)
                .expect_err("one-over contiguous storage is rejected"),
            EncodeError::InvalidInput {
                what: "contiguous coefficient block length mismatch",
            }
        );
    }

    #[test]
    fn strided_view_checks_stride_and_last_row_extent() {
        let values = [0_i32; 12];
        assert_eq!(
            CoefficientBlockView::try_new(&values, 0, 4, 2, 3)
                .expect_err("short row stride is rejected"),
            EncodeError::InvalidInput {
                what: "coefficient block row stride is shorter than its width",
            }
        );
        assert_eq!(
            CoefficientBlockView::try_new(&values, 5, 4, 2, 4)
                .expect_err("last strided row must fit"),
            EncodeError::InvalidInput {
                what: "coefficient block range exceeds storage",
            }
        );
    }
}
