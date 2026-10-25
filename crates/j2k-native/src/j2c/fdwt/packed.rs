// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{forward_lift_53_i64_phase, lifting};
use crate::math::{dispatch, Level};
use crate::{EncodeError, EncodeResult};
use j2k_types::encode_geometry::encode_dwt_level_dimensions_at_origin;

pub(super) fn usize_from_u32(value: u32, what: &'static str) -> EncodeResult<usize> {
    usize::try_from(value).map_err(|_| EncodeError::ArithmeticOverflow { what })
}

fn u32_from_usize(value: usize, what: &'static str) -> EncodeResult<u32> {
    u32::try_from(value).map_err(|_| EncodeError::ArithmeticOverflow { what })
}

/// Geometry left in the top-left quadrant after an in-place forward DWT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PackedDwtShape {
    pub(crate) origin: (u32, u32),
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
        let mut origin = shape.origin;
        for _ in 0..shape.num_levels {
            if current_width < 2 && current_height < 2 {
                return Err(EncodeError::InternalInvariant {
                    what: "packed forward DWT geometry has too many levels",
                });
            }
            let level =
                encode_dwt_level_dimensions_at_origin(current_width, current_height, origin);
            current_width = level.low_width;
            current_height = level.low_height;
            origin = (origin.0.div_ceil(2), origin.1.div_ceil(2));
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
        let mut origin = self.shape.origin;
        for _ in 0..forward_level {
            let level =
                encode_dwt_level_dimensions_at_origin(current_width, current_height, origin);
            current_width = level.low_width;
            current_height = level.low_height;
            origin = (origin.0.div_ceil(2), origin.1.div_ceil(2));
        }

        let level = encode_dwt_level_dimensions_at_origin(current_width, current_height, origin);
        let low_width = level.low_width;
        let low_height = level.low_height;
        let high_width = level.high_width;
        let high_height = level.high_height;
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

/// One level of the packed transform: the `width` x `height` region in the
/// top-left corner of a plane with row stride `stride`, its low-pass size, and
/// whether its first column and row sit at odd coordinates.
#[derive(Debug, Clone, Copy)]
pub(super) struct LevelRegion {
    pub(super) stride: usize,
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) low_width: usize,
    pub(super) low_height: usize,
    pub(super) odd_x: bool,
    pub(super) odd_y: bool,
}

fn forward_dwt_packed_core<T: Copy>(
    coefficients: &mut [T],
    width: usize,
    height: usize,
    num_levels: u8,
    line_scratch: &mut [T],
    origin: (u32, u32),
    mut transform_level: impl FnMut(&mut [T], &mut [T], LevelRegion) -> EncodeResult<()>,
) -> EncodeResult<PackedDwtShape> {
    let mut current_width = width;
    let mut current_height = height;
    let mut actual_levels = 0_u8;
    let mut level_origin = origin;

    for _ in 0..num_levels {
        if current_width < 2 && current_height < 2 {
            break;
        }
        let level = encode_dwt_level_dimensions_at_origin(
            u32_from_usize(current_width, "packed forward DWT level width")?,
            u32_from_usize(current_height, "packed forward DWT level height")?,
            level_origin,
        );
        let low_width = usize_from_u32(level.low_width, "packed forward DWT low width")?;
        let low_height = usize_from_u32(level.low_height, "packed forward DWT low height")?;
        transform_level(
            coefficients,
            line_scratch,
            LevelRegion {
                stride: width,
                width: current_width,
                height: current_height,
                low_width,
                low_height,
                odd_x: level_origin.0 & 1 != 0,
                odd_y: level_origin.1 & 1 != 0,
            },
        )?;

        current_width = low_width;
        current_height = low_height;
        level_origin = (level_origin.0.div_ceil(2), level_origin.1.div_ceil(2));
        actual_levels += 1;
    }

    Ok(PackedDwtShape {
        origin,
        ll_width: u32_from_usize(current_width, "packed forward DWT LL width")?,
        ll_height: u32_from_usize(current_height, "packed forward DWT LL height")?,
        num_levels: actual_levels,
    })
}

/// Lift every column, then every row, of one level as an interleaved line in
/// the line scratch and write the line back as its packed halves.
fn line_level<T: Copy>(
    coefficients: &mut [T],
    line_scratch: &mut [T],
    level: LevelRegion,
    lift: impl Fn(&mut [T], bool),
) {
    let LevelRegion {
        stride,
        width,
        height,
        low_width,
        low_height,
        odd_x,
        odd_y,
    } = level;
    let (x_phase, y_phase) = (usize::from(odd_x), usize::from(odd_y));

    // Analysis is vertical first and horizontal second because synthesis
    // applies the inverse operations in the opposite order.
    if height != 0 {
        for x in 0..width {
            for y in 0..height {
                line_scratch[y] = coefficients[y * stride + x];
            }
            lift(&mut line_scratch[..height], odd_y);

            for i in 0..low_height {
                coefficients[i * stride + x] = line_scratch[i * 2 + y_phase];
            }
            for i in 0..height - low_height {
                coefficients[(low_height + i) * stride + x] = line_scratch[i * 2 + 1 - y_phase];
            }
        }
    }

    if width != 0 {
        for y in 0..height {
            let row_start = y * stride;
            line_scratch[..width].copy_from_slice(&coefficients[row_start..row_start + width]);
            lift(&mut line_scratch[..width], odd_x);

            for i in 0..low_width {
                coefficients[row_start + i] = line_scratch[i * 2 + x_phase];
            }
            for i in 0..width - low_width {
                coefficients[row_start + low_width + i] = line_scratch[i * 2 + 1 - x_phase];
            }
        }
    }
}

/// Transform one f32 sample plane in place while retaining its packed quadrant
/// layout. The caller owns the coefficient plane and one reusable scratch of at
/// least `max(width, height)` samples; `packed_dwt_scratch_len` samples let the
/// vertical pass lift full-width column strips, and each further multiple of it
/// lets one more Rayon worker lift strips and rows. The output does not depend
/// on the scratch length.
///
/// With `spare`, a plane as long as `coefficients`, the first level is lifted
/// out of place into it in cache-sized bands, reading and writing each sample
/// about once; the transform is then left in `spare` and `coefficients` holds
/// scratch. This needs at least `packed_dwt_scratch_len` scratch samples.
#[expect(
    clippy::too_many_arguments,
    reason = "the plane geometry, scratch and spare plane are independent inputs"
)]
pub(crate) fn try_forward_dwt_packed_f32(
    coefficients: &mut [f32],
    width: u32,
    height: u32,
    num_levels: u8,
    reversible: bool,
    line_scratch: &mut [f32],
    mut spare: Option<&mut [f32]>,
    origin: (u32, u32),
) -> EncodeResult<PackedDwtShape> {
    validate_packed_dwt_plane(coefficients.len(), width, height)?;
    validate_line_scratch(line_scratch.len(), width, height)?;
    let workers = line_scratch.len() / lifting::packed_dwt_scratch_len(width, height)?;
    if spare
        .as_deref()
        .is_some_and(|spare| spare.len() != coefficients.len() || workers == 0)
    {
        return Err(EncodeError::InvalidInput {
            what: "out-of-place forward DWT needs a full-length spare plane and strip scratch",
        });
    }
    let mut first_level = true;
    let shape = forward_dwt_packed_core(
        coefficients,
        usize_from_u32(width, "packed forward DWT width")?,
        usize_from_u32(height, "packed forward DWT height")?,
        num_levels,
        line_scratch,
        origin,
        |coefficients, scratch, level| {
            let out_of_place = core::mem::take(&mut first_level);
            dispatch!(Level::new(), simd => match spare.as_deref_mut() {
                Some(spare) if out_of_place => lifting::transform_level_into(
                    simd, coefficients, spare, scratch, workers, level, reversible,
                ),
                Some(spare) => {
                    lifting::transform_level(simd, spare, scratch, workers, level, reversible)
                }
                None => lifting::transform_level(
                    simd, coefficients, scratch, workers, level, reversible,
                ),
            })
        },
    )?;
    if let (true, Some(spare)) = (first_level, spare) {
        spare.copy_from_slice(coefficients);
    }
    Ok(shape)
}

/// Transform one exact i64 sample plane in place with the reversible 5/3 DWT.
pub(crate) fn try_forward_dwt_packed_i64(
    coefficients: &mut [i64],
    width: u32,
    height: u32,
    num_levels: u8,
    line_scratch: &mut [i64],
    origin: (u32, u32),
) -> EncodeResult<PackedDwtShape> {
    validate_packed_dwt_plane(coefficients.len(), width, height)?;
    validate_line_scratch(line_scratch.len(), width, height)?;
    forward_dwt_packed_core(
        coefficients,
        usize_from_u32(width, "packed forward DWT width")?,
        usize_from_u32(height, "packed forward DWT height")?,
        num_levels,
        line_scratch,
        origin,
        |coefficients, scratch, level| {
            line_level(coefficients, scratch, level, forward_lift_53_i64_phase);
            Ok(())
        },
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
    use crate::j2c::fdwt::{
        forward_dwt, forward_lift_53_phase, forward_lift_97_phase, packed_dwt_scratch_len,
    };

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
        let shape =
            try_forward_dwt_packed_f32(&mut packed, 7, 5, 3, true, &mut scratch, None, (0, 0))
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
        let shape =
            try_forward_dwt_packed_f32(&mut packed, 8, 8, 3, false, &mut scratch, None, (0, 0))
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
        let shape = try_forward_dwt_packed_i64(&mut packed, 5, 5, 2, &mut scratch, (0, 0))
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
        let shape = try_forward_dwt_packed_f32(
            &mut coefficients,
            5,
            3,
            2,
            true,
            &mut scratch,
            None,
            (0, 0),
        )
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
            try_forward_dwt_packed_f32(&mut empty, 0, 1, 1, true, &mut [0.0], None, (0, 0)),
            Err(EncodeError::InvalidInput {
                what: "packed forward DWT dimensions must be non-zero",
            })
        );
        assert_eq!(
            try_forward_dwt_packed_f32(&mut empty, 1, 0, 1, true, &mut [0.0], None, (0, 0)),
            Err(EncodeError::InvalidInput {
                what: "packed forward DWT dimensions must be non-zero",
            })
        );
        let mut one_short = vec![0.0_f32; 14];
        assert_eq!(
            try_forward_dwt_packed_f32(&mut one_short, 5, 3, 1, true, &mut [0.0; 5], None, (0, 0)),
            Err(EncodeError::InvalidInput {
                what: "packed forward DWT coefficient length mismatch",
            })
        );
        let mut one_over = vec![0_i64; 16];
        assert_eq!(
            try_forward_dwt_packed_i64(&mut one_over, 5, 3, 1, &mut [0; 5], (0, 0)),
            Err(EncodeError::InvalidInput {
                what: "packed forward DWT coefficient length mismatch",
            })
        );
    }

    #[test]
    fn packed_transform_accepts_exact_scratch_and_rejects_one_short() {
        let mut coefficients = vec![0.0_f32; 15];
        assert_eq!(
            try_forward_dwt_packed_f32(
                &mut coefficients,
                5,
                3,
                1,
                true,
                &mut [0.0; 4],
                None,
                (0, 0)
            ),
            Err(EncodeError::InvalidInput {
                what: "packed forward DWT line scratch is too short",
            })
        );
        let mut spare = [0.0_f32; 15];
        assert_eq!(
            try_forward_dwt_packed_f32(
                &mut coefficients,
                5,
                3,
                1,
                true,
                &mut [0.0; 5],
                Some(&mut spare[..14]),
                (0, 0),
            ),
            Err(EncodeError::InvalidInput {
                what: "out-of-place forward DWT needs a full-length spare plane and strip scratch",
            })
        );
        let shape = try_forward_dwt_packed_f32(
            &mut coefficients,
            5,
            3,
            u8::MAX,
            true,
            &mut [0.0; 5],
            None,
            (0, 0),
        )
        .expect("exact scratch length is accepted");
        assert_eq!(shape.num_levels, 3);
        assert_eq!((shape.ll_width, shape.ll_height), (1, 1));
    }

    /// The column-at-a-time transform that the SIMD strips replaced.
    fn line_transform(
        plane: &mut [f32],
        (width, height): (usize, usize),
        num_levels: u8,
        origin: (u32, u32),
        reversible: bool,
    ) -> EncodeResult<PackedDwtShape> {
        let lift = if reversible {
            forward_lift_53_phase
        } else {
            forward_lift_97_phase
        };
        let mut line = vec![0.0; width.max(height)];
        forward_dwt_packed_core(
            plane,
            width,
            height,
            num_levels,
            &mut line,
            origin,
            |p, l, r| {
                line_level(p, l, r, lift);
                Ok(())
            },
        )
    }

    #[test]
    fn f32_packed_transform_matches_line_transform_bit_for_bit() {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            f32::from(u16::try_from(state >> 48).expect("16-bit sample")) - 32768.0
        };
        let geometries = [
            (1, 1),
            (1, 9),
            (9, 1),
            (2, 2),
            (3, 5),
            (8, 8),
            (31, 33),
            (33, 31),
            (64, 17),
            (65, 70),
            (100, 3),
            (2049, 5),
            (2112, 3),
            (300, 260),
            (517, 131),
        ];
        for ((width, height), origin, reversible, num_levels, extreme) in geometries
            .into_iter()
            .flat_map(|g| [(0, 0), (1, 1), (1, 0), (3, 6)].map(|o| (g, o)))
            .flat_map(|(g, o)| [true, false].map(|r| (g, o, r)))
            .flat_map(|(g, o, r)| [1, 2, u8::MAX].map(|n| (g, o, r, n)))
            .flat_map(|(g, o, r, n)| [false, true].map(|e| (g, o, r, n, e)))
        {
            // Extreme samples make the even-length 5/3 edge forms overflow.
            let original = (0..width * height)
                .map(|_| match (next(), extreme) {
                    (value, true) if value < -28672.0 => f32::MAX.copysign(value + 32000.0),
                    (value, _) if reversible => value,
                    (value, _) => value / 64.0,
                })
                .collect::<Vec<_>>();
            let mut expected = original.clone();
            let expected_shape = line_transform(
                &mut expected,
                (width, height),
                num_levels,
                origin,
                reversible,
            );
            let (w, h) = (
                u32::try_from(width).expect("width"),
                u32::try_from(height).expect("height"),
            );
            let strip_scratch = packed_dwt_scratch_len(w, h).expect("strip scratch");
            // Three strip scratches run three workers on large levels.
            for scratch_len in [
                width.max(height),
                width.max(5 * height),
                strip_scratch,
                3 * strip_scratch,
            ] {
                // Full strip scratch also runs the first level out of place.
                for out_of_place in [false, scratch_len >= strip_scratch] {
                    let mut actual = original.clone();
                    let mut spare = vec![0.0; original.len()];
                    let mut scratch = vec![0.0; scratch_len];
                    let shape = try_forward_dwt_packed_f32(
                        &mut actual,
                        w,
                        h,
                        num_levels,
                        reversible,
                        &mut scratch,
                        out_of_place.then_some(spare.as_mut_slice()),
                        origin,
                    );
                    let output = if out_of_place { &spare } else { &actual };
                    let context = format!(
                        "{width}x{height} origin {origin:?} levels {num_levels} \
                         reversible {reversible} extreme {extreme} scratch {scratch_len} \
                         out of place {out_of_place}"
                    );
                    assert_eq!(shape, expected_shape, "{context}");
                    let bits =
                        |plane: &[f32]| plane.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                    assert_eq!(bits(output), bits(&expected), "{context}");
                }
            }
        }
    }

    #[test]
    fn strip_scratch_covers_full_strips_and_rows() {
        for (width, height, expected) in [
            (10, 7, 70),
            (100, 7, 224),
            (100, 2, 100),
            (4096, 3, 4096),
            (4096, 100, 6400),
        ] {
            assert_eq!(packed_dwt_scratch_len(width, height), Ok(expected));
        }
    }
}
