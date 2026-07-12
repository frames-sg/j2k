// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{forward_lift_53, forward_lift_53_i64, forward_lift_97};
use crate::{EncodeError, EncodeResult};

fn usize_from_u32(value: u32, what: &'static str) -> EncodeResult<usize> {
    usize::try_from(value).map_err(|_| EncodeError::ArithmeticOverflow { what })
}

fn u32_from_usize(value: usize, what: &'static str) -> EncodeResult<u32> {
    u32::try_from(value).map_err(|_| EncodeError::ArithmeticOverflow { what })
}

/// Geometry left in the top-left quadrant after an in-place forward DWT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PackedDwtShape {
    pub(crate) ll_width: u32,
    pub(crate) ll_height: u32,
    pub(crate) num_levels: u8,
}

/// One rectangular subband inside a packed quadrant coefficient plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PackedSubbandRect {
    pub(crate) offset: usize,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) row_stride: usize,
}

/// Validated borrowed view over a possibly strided packed subband or code block.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PackedSubbandView<'a, T> {
    coefficients: &'a [T],
    rect: PackedSubbandRect,
}

impl<'a, T> PackedSubbandView<'a, T> {
    pub(crate) fn try_new(coefficients: &'a [T], rect: PackedSubbandRect) -> EncodeResult<Self> {
        if rect.width == 0 || rect.height == 0 {
            if rect.offset > coefficients.len() {
                return Err(EncodeError::InternalInvariant {
                    what: "empty packed subband offset exceeds coefficient storage",
                });
            }
            return Ok(Self { coefficients, rect });
        }
        let width = usize_from_u32(rect.width, "packed subband width")?;
        let height = usize_from_u32(rect.height, "packed subband height")?;
        if rect.row_stride < width {
            return Err(EncodeError::InternalInvariant {
                what: "packed subband row stride is shorter than its width",
            });
        }
        let end = (height - 1)
            .checked_mul(rect.row_stride)
            .and_then(|rows| rows.checked_add(rect.offset))
            .and_then(|last_row| last_row.checked_add(width))
            .ok_or(EncodeError::ArithmeticOverflow {
                what: "packed subband borrowed range",
            })?;
        if end > coefficients.len() {
            return Err(EncodeError::InternalInvariant {
                what: "packed subband borrowed range exceeds coefficient storage",
            });
        }
        Ok(Self { coefficients, rect })
    }

    pub(crate) fn width(self) -> u32 {
        self.rect.width
    }

    pub(crate) fn height(self) -> u32 {
        self.rect.height
    }

    pub(crate) fn row(self, y: u32) -> Option<&'a [T]> {
        if y >= self.rect.height {
            return None;
        }
        let y = usize::try_from(y).ok()?;
        let width = usize::try_from(self.rect.width).ok()?;
        let row_offset = y.checked_mul(self.rect.row_stride)?;
        let start = self.rect.offset.checked_add(row_offset)?;
        let end = start.checked_add(width)?;
        self.coefficients.get(start..end)
    }
}

/// The three detail-subband rectangles at one resolution, ordered from the
/// lowest resolution to the highest resolution by [`PackedDwtGeometry::level`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PackedDwtLevelRects {
    pub(crate) hl: PackedSubbandRect,
    pub(crate) lh: PackedSubbandRect,
    pub(crate) hh: PackedSubbandRect,
    pub(crate) low_width: u32,
    pub(crate) low_height: u32,
    pub(crate) high_width: u32,
    pub(crate) high_height: u32,
}

/// Allocation-free geometry for a packed quadrant DWT plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PackedDwtGeometry {
    plane_width: u32,
    plane_height: u32,
    plane_len: usize,
    shape: PackedDwtShape,
}

impl PackedDwtGeometry {
    pub(crate) fn try_new(
        plane_width: u32,
        plane_height: u32,
        plane_len: usize,
        shape: PackedDwtShape,
    ) -> EncodeResult<Self> {
        validate_packed_dwt_plane(plane_len, plane_width, plane_height)?;
        let mut current_width = plane_width;
        let mut current_height = plane_height;
        for _ in 0..shape.num_levels {
            if current_width < 2 && current_height < 2 {
                return Err(EncodeError::InternalInvariant {
                    what: "packed forward DWT geometry has too many levels",
                });
            }
            current_width = current_width.div_ceil(2);
            current_height = current_height.div_ceil(2);
        }
        if current_width != shape.ll_width || current_height != shape.ll_height {
            return Err(EncodeError::InternalInvariant {
                what: "packed forward DWT LL geometry mismatch",
            });
        }
        Ok(Self {
            plane_width,
            plane_height,
            plane_len,
            shape,
        })
    }

    pub(crate) fn ll(self) -> EncodeResult<PackedSubbandRect> {
        self.rect(0, 0, self.shape.ll_width, self.shape.ll_height)
    }

    /// Return detail rectangles in decoder resolution order: index zero is the
    /// lowest-resolution HL/LH/HH triplet.
    pub(crate) fn level(self, resolution_index: u8) -> EncodeResult<PackedDwtLevelRects> {
        if resolution_index >= self.shape.num_levels {
            return Err(EncodeError::InvalidInput {
                what: "packed forward DWT resolution index is out of range",
            });
        }
        let forward_level = self.shape.num_levels - resolution_index - 1;
        let mut current_width = self.plane_width;
        let mut current_height = self.plane_height;
        for _ in 0..forward_level {
            current_width = current_width.div_ceil(2);
            current_height = current_height.div_ceil(2);
        }

        let low_width = current_width.div_ceil(2);
        let low_height = current_height.div_ceil(2);
        let high_width = current_width / 2;
        let high_height = current_height / 2;
        Ok(PackedDwtLevelRects {
            hl: self.rect(low_width, 0, high_width, low_height)?,
            lh: self.rect(0, low_height, low_width, high_height)?,
            hh: self.rect(low_width, low_height, high_width, high_height)?,
            low_width,
            low_height,
            high_width,
            high_height,
        })
    }

    pub(crate) fn num_levels(self) -> u8 {
        self.shape.num_levels
    }

    fn rect(self, x: u32, y: u32, width: u32, height: u32) -> EncodeResult<PackedSubbandRect> {
        let row_stride = usize_from_u32(self.plane_width, "packed forward DWT row stride")?;
        if width == 0 || height == 0 {
            return Ok(PackedSubbandRect {
                offset: 0,
                width,
                height,
                row_stride,
            });
        }
        let x_end = x
            .checked_add(width)
            .ok_or(EncodeError::ArithmeticOverflow {
                what: "packed forward DWT subband x extent",
            })?;
        let y_end = y
            .checked_add(height)
            .ok_or(EncodeError::ArithmeticOverflow {
                what: "packed forward DWT subband y extent",
            })?;
        if x_end > self.plane_width || y_end > self.plane_height {
            return Err(EncodeError::InternalInvariant {
                what: "packed forward DWT subband exceeds its coefficient plane",
            });
        }
        let x = usize_from_u32(x, "packed forward DWT subband x offset")?;
        let y = usize_from_u32(y, "packed forward DWT subband y offset")?;
        let x_end = usize_from_u32(x_end, "packed forward DWT subband x extent")?;
        let y_end = usize_from_u32(y_end, "packed forward DWT subband y extent")?;
        let offset = y
            .checked_mul(row_stride)
            .and_then(|row| row.checked_add(x))
            .ok_or(EncodeError::ArithmeticOverflow {
                what: "packed forward DWT subband offset",
            })?;
        let end = (y_end - 1)
            .checked_mul(row_stride)
            .and_then(|row| row.checked_add(x_end))
            .ok_or(EncodeError::ArithmeticOverflow {
                what: "packed forward DWT subband range",
            })?;
        if end > self.plane_len {
            return Err(EncodeError::InternalInvariant {
                what: "packed forward DWT subband range exceeds storage",
            });
        }
        Ok(PackedSubbandRect {
            offset,
            width,
            height,
            row_stride,
        })
    }
}

fn forward_dwt_packed_core<T: Copy>(
    coefficients: &mut [T],
    width: usize,
    height: usize,
    num_levels: u8,
    line_scratch: &mut [T],
    mut lift: impl FnMut(&mut [T]),
) -> EncodeResult<PackedDwtShape> {
    let mut current_width = width;
    let mut current_height = height;
    let mut actual_levels = 0_u8;

    for _ in 0..num_levels {
        if current_width < 2 && current_height < 2 {
            break;
        }

        // Analysis is vertical first and horizontal second because synthesis
        // applies the inverse operations in the opposite order.
        if current_height >= 2 {
            for x in 0..current_width {
                for y in 0..current_height {
                    line_scratch[y] = coefficients[y * width + x];
                }
                lift(&mut line_scratch[..current_height]);

                let low_height = current_height.div_ceil(2);
                for i in 0..low_height {
                    coefficients[i * width + x] = line_scratch[i * 2];
                }
                for i in 0..current_height / 2 {
                    coefficients[(low_height + i) * width + x] = line_scratch[i * 2 + 1];
                }
            }
        }

        if current_width >= 2 {
            for y in 0..current_height {
                let row_start = y * width;
                line_scratch[..current_width]
                    .copy_from_slice(&coefficients[row_start..row_start + current_width]);
                lift(&mut line_scratch[..current_width]);

                let low_width = current_width.div_ceil(2);
                for i in 0..low_width {
                    coefficients[row_start + i] = line_scratch[i * 2];
                }
                for i in 0..current_width / 2 {
                    coefficients[row_start + low_width + i] = line_scratch[i * 2 + 1];
                }
            }
        }

        current_width = current_width.div_ceil(2);
        current_height = current_height.div_ceil(2);
        actual_levels += 1;
    }

    Ok(PackedDwtShape {
        ll_width: u32_from_usize(current_width, "packed forward DWT LL width")?,
        ll_height: u32_from_usize(current_height, "packed forward DWT LL height")?,
        num_levels: actual_levels,
    })
}

/// Transform one f32 sample plane in place while retaining its packed quadrant
/// layout. The caller owns the coefficient plane and one reusable line buffer.
pub(crate) fn try_forward_dwt_packed_f32(
    coefficients: &mut [f32],
    width: u32,
    height: u32,
    num_levels: u8,
    reversible: bool,
    line_scratch: &mut [f32],
) -> EncodeResult<PackedDwtShape> {
    validate_packed_dwt_plane(coefficients.len(), width, height)?;
    validate_line_scratch(line_scratch.len(), width, height)?;
    forward_dwt_packed_core(
        coefficients,
        usize_from_u32(width, "packed forward DWT width")?,
        usize_from_u32(height, "packed forward DWT height")?,
        num_levels,
        line_scratch,
        |line| {
            if reversible {
                forward_lift_53(line);
            } else {
                forward_lift_97(line);
            }
        },
    )
}

/// Transform one exact i64 sample plane in place with the reversible 5/3 DWT.
pub(crate) fn try_forward_dwt_packed_i64(
    coefficients: &mut [i64],
    width: u32,
    height: u32,
    num_levels: u8,
    line_scratch: &mut [i64],
) -> EncodeResult<PackedDwtShape> {
    validate_packed_dwt_plane(coefficients.len(), width, height)?;
    validate_line_scratch(line_scratch.len(), width, height)?;
    forward_dwt_packed_core(
        coefficients,
        usize_from_u32(width, "packed forward DWT width")?,
        usize_from_u32(height, "packed forward DWT height")?,
        num_levels,
        line_scratch,
        forward_lift_53_i64,
    )
}

pub(super) fn validate_packed_dwt_plane(
    coefficient_count: usize,
    width: u32,
    height: u32,
) -> EncodeResult<()> {
    if width == 0 || height == 0 {
        return Err(EncodeError::InvalidInput {
            what: "packed forward DWT dimensions must be non-zero",
        });
    }
    let width = usize_from_u32(width, "packed forward DWT width")?;
    let height = usize_from_u32(height, "packed forward DWT height")?;
    let expected = width
        .checked_mul(height)
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "packed forward DWT coefficient count",
        })?;
    if coefficient_count != expected {
        return Err(EncodeError::InvalidInput {
            what: "packed forward DWT coefficient length mismatch",
        });
    }
    Ok(())
}

fn validate_line_scratch(scratch_count: usize, width: u32, height: u32) -> EncodeResult<()> {
    let required_scratch = usize_from_u32(width, "packed forward DWT scratch width")?
        .max(usize_from_u32(height, "packed forward DWT scratch height")?);
    if scratch_count < required_scratch {
        return Err(EncodeError::InvalidInput {
            what: "packed forward DWT line scratch is too short",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;
    use crate::j2c::fdwt::forward_dwt;

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn packed_shape_dimension_conversion_rejects_u32_overflow() {
        const WHAT: &str = "test packed DWT dimension";
        assert_eq!(
            u32_from_usize(usize::MAX, WHAT),
            Err(EncodeError::ArithmeticOverflow { what: WHAT })
        );
    }

    fn copied_view<T: Copy>(plane: &[T], rect: PackedSubbandRect) -> Vec<T> {
        let view = PackedSubbandView::try_new(plane, rect).expect("valid packed test view");
        let width = usize::try_from(view.width()).expect("test view width fits address space");
        let height = usize::try_from(view.height()).expect("test view height fits address space");
        let mut copied = Vec::with_capacity(
            width
                .checked_mul(height)
                .expect("test view area fits address space"),
        );
        for row in 0..view.height() {
            copied.extend_from_slice(view.row(row).expect("validated test row"));
        }
        copied
    }

    #[test]
    fn reversible_f32_packed_views_match_odd_multilevel_decomposition() {
        let original = (0..35)
            .map(|index| {
                f32::from(i16::try_from(index * 29 - 211).expect("test coefficient fits i16"))
            })
            .collect::<Vec<_>>();
        let expected = forward_dwt(&original, 7, 5, 3, true);
        let mut packed = original;
        let mut scratch = vec![0.0_f32; 7];
        let shape = try_forward_dwt_packed_f32(&mut packed, 7, 5, 3, true, &mut scratch)
            .expect("valid packed reversible transform");
        let geometry =
            PackedDwtGeometry::try_new(7, 5, packed.len(), shape).expect("valid geometry");

        assert_eq!(
            copied_view(&packed, geometry.ll().expect("LL rect")),
            expected.ll
        );
        for (resolution, expected_level) in expected.levels.iter().enumerate() {
            let rects = geometry
                .level(u8::try_from(resolution).expect("test resolution fits u8"))
                .expect("detail rects");
            assert_eq!(copied_view(&packed, rects.hl), expected_level.hl);
            assert_eq!(copied_view(&packed, rects.lh), expected_level.lh);
            assert_eq!(copied_view(&packed, rects.hh), expected_level.hh);
        }
    }

    #[test]
    fn irreversible_f32_packed_views_match_even_multilevel_decomposition() {
        let original = (0..64)
            .map(|index| {
                f32::from(i16::try_from(index * 17 - 301).expect("test coefficient fits i16"))
                    * 0.25
            })
            .collect::<Vec<_>>();
        let expected = forward_dwt(&original, 8, 8, 3, false);
        let mut packed = original;
        let mut scratch = vec![0.0_f32; 8];
        let shape = try_forward_dwt_packed_f32(&mut packed, 8, 8, 3, false, &mut scratch)
            .expect("valid packed irreversible transform");
        let geometry =
            PackedDwtGeometry::try_new(8, 8, packed.len(), shape).expect("valid geometry");

        assert_eq!(
            copied_view(&packed, geometry.ll().expect("LL rect")),
            expected.ll
        );
        for (resolution, expected_level) in expected.levels.iter().enumerate() {
            let rects = geometry
                .level(u8::try_from(resolution).expect("test resolution fits u8"))
                .expect("detail rects");
            assert_eq!(copied_view(&packed, rects.hl), expected_level.hl);
            assert_eq!(copied_view(&packed, rects.lh), expected_level.lh);
            assert_eq!(copied_view(&packed, rects.hh), expected_level.hh);
        }
    }

    #[test]
    fn exact_i64_packed_views_match_38_bit_reference_coefficients() {
        let original = (0_usize..25)
            .map(|index| {
                let magnitude = (1_i64 << 37)
                    - 1
                    - i64::try_from(index * 101).expect("test coefficient offset fits i64");
                if index.is_multiple_of(2) {
                    magnitude
                } else {
                    -magnitude
                }
            })
            .collect::<Vec<_>>();
        let mut packed = original;
        let mut scratch = vec![0_i64; 5];
        let shape = try_forward_dwt_packed_i64(&mut packed, 5, 5, 2, &mut scratch)
            .expect("valid exact packed transform");
        let geometry =
            PackedDwtGeometry::try_new(5, 5, packed.len(), shape).expect("valid geometry");

        assert_eq!(
            copied_view(&packed, geometry.ll().expect("LL rect")),
            [0; 4]
        );

        let lowest = geometry.level(0).expect("lowest detail rects");
        assert_eq!(copied_view(&packed, lowest.hl), [0; 2]);
        assert_eq!(copied_view(&packed, lowest.lh), [0; 2]);
        assert_eq!(copied_view(&packed, lowest.hh), [0]);

        let highest = geometry.level(1).expect("highest detail rects");
        assert_eq!(
            copied_view(&packed, highest.hl),
            [-1_010, -1_010, 0, 0, 1_010, 1_010]
        );
        assert_eq!(
            copied_view(&packed, highest.lh),
            [-202, 0, 202, -202, 0, 202]
        );
        assert_eq!(
            copied_view(&packed, highest.hh),
            [
                549_755_811_460,
                549_755_810_652,
                549_755_807_420,
                549_755_806_612
            ]
        );
    }

    #[test]
    fn odd_geometry_descriptors_use_original_plane_stride() {
        let mut coefficients = vec![0.0_f32; 15];
        let mut scratch = vec![0.0_f32; 5];
        let shape = try_forward_dwt_packed_f32(&mut coefficients, 5, 3, 2, true, &mut scratch)
            .expect("valid odd transform");
        let geometry =
            PackedDwtGeometry::try_new(5, 3, coefficients.len(), shape).expect("valid geometry");

        assert_eq!(
            geometry.ll().expect("LL rect"),
            PackedSubbandRect {
                offset: 0,
                width: 2,
                height: 1,
                row_stride: 5,
            }
        );
        let lowest = geometry.level(0).expect("lowest detail level");
        assert_eq!(
            (lowest.hl.offset, lowest.hl.width, lowest.hl.height),
            (2, 1, 1)
        );
        assert_eq!(
            (lowest.lh.offset, lowest.lh.width, lowest.lh.height),
            (5, 2, 1)
        );
        assert_eq!(
            (lowest.hh.offset, lowest.hh.width, lowest.hh.height),
            (7, 1, 1)
        );
        let highest = geometry.level(1).expect("highest detail level");
        assert_eq!(
            (highest.hl.offset, highest.hl.width, highest.hl.height),
            (3, 2, 2)
        );
        assert_eq!(
            (highest.lh.offset, highest.lh.width, highest.lh.height),
            (10, 3, 1)
        );
        assert_eq!(
            (highest.hh.offset, highest.hh.width, highest.hh.height),
            (13, 2, 1)
        );
    }

    #[test]
    fn packed_transform_rejects_zero_axes_and_wrong_storage_lengths() {
        let mut empty = Vec::<f32>::new();
        assert_eq!(
            try_forward_dwt_packed_f32(&mut empty, 0, 1, 1, true, &mut [0.0]),
            Err(EncodeError::InvalidInput {
                what: "packed forward DWT dimensions must be non-zero",
            })
        );
        assert_eq!(
            try_forward_dwt_packed_f32(&mut empty, 1, 0, 1, true, &mut [0.0]),
            Err(EncodeError::InvalidInput {
                what: "packed forward DWT dimensions must be non-zero",
            })
        );
        let mut one_short = vec![0.0_f32; 14];
        assert_eq!(
            try_forward_dwt_packed_f32(&mut one_short, 5, 3, 1, true, &mut [0.0; 5]),
            Err(EncodeError::InvalidInput {
                what: "packed forward DWT coefficient length mismatch",
            })
        );
        let mut one_over = vec![0_i64; 16];
        assert_eq!(
            try_forward_dwt_packed_i64(&mut one_over, 5, 3, 1, &mut [0; 5]),
            Err(EncodeError::InvalidInput {
                what: "packed forward DWT coefficient length mismatch",
            })
        );
    }

    #[test]
    fn packed_transform_accepts_exact_scratch_and_rejects_one_short() {
        let mut coefficients = vec![0.0_f32; 15];
        assert_eq!(
            try_forward_dwt_packed_f32(&mut coefficients, 5, 3, 1, true, &mut [0.0; 4]),
            Err(EncodeError::InvalidInput {
                what: "packed forward DWT line scratch is too short",
            })
        );
        let shape =
            try_forward_dwt_packed_f32(&mut coefficients, 5, 3, u8::MAX, true, &mut [0.0; 5])
                .expect("exact scratch length is accepted");
        assert_eq!(shape.num_levels, 3);
        assert_eq!((shape.ll_width, shape.ll_height), (1, 1));
    }
}
