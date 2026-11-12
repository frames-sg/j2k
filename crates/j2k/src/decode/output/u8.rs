// SPDX-License-Identifier: MIT OR Apache-2.0

//! Eight-bit direct-layout and component-plane conversion.

use crate::backend::{ColorSpace, DecodedComponents as NativeDecodedComponents};
use crate::decode::srgb8::is_srgb_equivalent_icc_profile;
use crate::J2kError;
use core::ops::Range;
use j2k_core::{PixelFormat, Unsupported};

static SRGB_ICC_AS_RGB: ColorSpace = ColorSpace::RGB;

/// Resolve only restricted RGB ICC profiles whose eight-bit samples can be
/// written as sRGB unchanged. Other profiles and native output stay explicit.
pub(crate) fn resolve_u8_color_space(color_space: &ColorSpace, fmt: PixelFormat) -> &ColorSpace {
    match (color_space, fmt) {
        (
            ColorSpace::Icc {
                profile,
                num_channels: 3,
            },
            PixelFormat::Rgb8 | PixelFormat::Rgba8,
        ) if is_srgb_equivalent_icc_profile(profile) => &SRGB_ICC_AS_RGB,
        _ => color_space,
    }
}

pub(in crate::decode) fn can_decode_u8_directly(
    color_space: &ColorSpace,
    has_alpha: bool,
    dims: (u32, u32),
    stride: usize,
    fmt: PixelFormat,
) -> bool {
    let width = dims.0 as usize;
    match (color_space, has_alpha, fmt) {
        (ColorSpace::RGB, false, PixelFormat::Rgb8) => stride == width * 3,
        (ColorSpace::RGB, true, PixelFormat::Rgba8) => stride == width * 4,
        (ColorSpace::Gray, false, PixelFormat::Gray8) => stride == width,
        _ => false,
    }
}

pub(in crate::decode) fn write_components_u8_output(
    components: &NativeDecodedComponents<'_>,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<(), J2kError> {
    write_planes_u8(
        resolve_u8_color_space(components.color_space(), fmt),
        components.has_alpha(),
        components.planes(),
        components.dimensions(),
        out,
        stride,
        fmt,
    )
}

/// Write `planes`, each `dims` in size, to `out` from its first byte.
pub(super) fn write_planes_u8(
    color_space: &ColorSpace,
    has_alpha: bool,
    planes: &[j2k_native::ComponentPlane<'_>],
    dims: (u32, u32),
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<(), J2kError> {
    let rows = 0..dims.1 as usize;
    write_plane_rows_u8(color_space, has_alpha, planes, dims, rows, out, stride, fmt)
}

/// Write rows `rows` of `planes`, each `dims` in size, to `out` from its
/// first byte, as [`write_planes_u8`] writes those rows of whole planes.
#[expect(
    clippy::too_many_arguments,
    reason = "the row-range writer keeps the planes' layout, the rows and the output layout explicit"
)]
pub(in crate::decode) fn write_plane_rows_u8(
    color_space: &ColorSpace,
    has_alpha: bool,
    planes: &[j2k_native::ComponentPlane<'_>],
    dims: (u32, u32),
    rows: Range<usize>,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<(), J2kError> {
    let expected_samples = component_sample_count(dims)?;
    let width = dims.0 as usize;
    if rows.start > rows.end || rows.end > dims.1 as usize {
        return Err(J2kError::internal_backend(
            "component rows outside their planes",
        ));
    }
    match (color_space, has_alpha, planes.len(), fmt) {
        (ColorSpace::Gray, false, 1, PixelFormat::Gray8) => {
            validate_component_planes(&planes[..1], expected_samples)?;
            write_component_rows_u8(&planes[0], out, stride, width, rows);
            Ok(())
        }
        (ColorSpace::RGB, false, 3, PixelFormat::Rgb8)
        | (ColorSpace::RGB, true, 4, PixelFormat::Rgb8) => {
            validate_component_planes(&planes[..3], expected_samples)?;
            write_rgb_component_rows_u8(planes, out, stride, width, rows);
            Ok(())
        }
        (ColorSpace::RGB, false, 3, PixelFormat::Rgba8) => {
            validate_component_planes(&planes[..3], expected_samples)?;
            write_rgba_component_rows_u8(planes, out, stride, width, rows, true);
            Ok(())
        }
        (ColorSpace::RGB, true, 4, PixelFormat::Rgba8) => {
            validate_component_planes(&planes[..4], expected_samples)?;
            write_rgba_component_rows_u8(planes, out, stride, width, rows, false);
            Ok(())
        }
        _ => Err(Unsupported {
            what: "backend color space cannot be mapped to requested 8-bit pixel format",
        }
        .into()),
    }
}

pub(super) fn component_sample_count(dims: (u32, u32)) -> Result<usize, J2kError> {
    (dims.0 as usize)
        .checked_mul(dims.1 as usize)
        .ok_or(J2kError::DimensionOverflow {
            width: dims.0,
            height: dims.1,
        })
}

pub(super) fn validate_component_planes(
    planes: &[j2k_native::ComponentPlane<'_>],
    expected_samples: usize,
) -> Result<(), J2kError> {
    for (index, plane) in planes.iter().enumerate() {
        let samples = plane.samples().len();
        if samples < expected_samples {
            return Err(J2kError::BackendComponentPlaneTooShort {
                component: index,
                samples,
                expected: expected_samples,
            });
        }
    }
    Ok(())
}

/// The first `row_bytes` of each of the first `height` rows of `out` at
/// `stride`. A validated buffer may end right after the last row's pixels
/// (see `j2k_core::strided_output_len`), so the last row may be shorter
/// than `stride`.
pub(super) fn output_rows(
    out: &mut [u8],
    stride: usize,
    row_bytes: usize,
    height: usize,
) -> impl Iterator<Item = &mut [u8]> {
    out.chunks_mut(stride)
        .take(height)
        .map(move |row| &mut row[..row_bytes])
}

/// Rows `rows` of `plane`, `width` samples each, written from the first row
/// of `out` on, at the CPU's SIMD level, as the conversion rounds every
/// sample (see [`j2k_native::at_simd_level`]).
fn write_component_rows_u8(
    plane: &j2k_native::ComponentPlane<'_>,
    out: &mut [u8],
    stride: usize,
    width: usize,
    rows: Range<usize>,
) {
    j2k_native::at_simd_level(
        #[inline(always)]
        || {
            for (index, y) in rows.enumerate() {
                let src = &plane.samples()[y * width..(y + 1) * width];
                let dst = &mut out[index * stride..index * stride + width];
                write_samples_as_u8(src, plane.bit_depth(), dst);
            }
        },
    );
}

fn write_rgb_component_rows_u8(
    planes: &[j2k_native::ComponentPlane<'_>],
    out: &mut [u8],
    stride: usize,
    width: usize,
    rows: Range<usize>,
) {
    if planes[..3].iter().all(|plane| plane.bit_depth() == 8) {
        // `sample_as_u8` at 8 bits is the RGB8 pack's rounding and saturation,
        // which j2k-native runs at the CPU's SIMD level.
        for (index, y) in rows.enumerate() {
            let row = y * width..(y + 1) * width;
            j2k_native::pack_rgb8(
                [0, 1, 2].map(|channel| &planes[channel].samples()[row.clone()]),
                &mut out[index * stride..index * stride + width * 3],
            );
        }
        return;
    }
    j2k_native::at_simd_level(
        #[inline(always)]
        || {
            for (index, y) in rows.enumerate() {
                let row = y * width;
                let dst = &mut out[index * stride..index * stride + width * 3];
                for x in 0..width {
                    let dst = &mut dst[x * 3..x * 3 + 3];
                    for channel in 0..3 {
                        dst[channel] = sample_as_u8(
                            planes[channel].samples()[row + x],
                            planes[channel].bit_depth(),
                        );
                    }
                }
            }
        },
    );
}

fn write_rgba_component_rows_u8(
    planes: &[j2k_native::ComponentPlane<'_>],
    out: &mut [u8],
    stride: usize,
    width: usize,
    rows: Range<usize>,
    synthesize_alpha: bool,
) {
    j2k_native::at_simd_level(
        #[inline(always)]
        || {
            for (index, y) in rows.enumerate() {
                let row = y * width;
                let dst = &mut out[index * stride..index * stride + width * 4];
                for x in 0..width {
                    let dst = &mut dst[x * 4..x * 4 + 4];
                    for channel in 0..3 {
                        dst[channel] = sample_as_u8(
                            planes[channel].samples()[row + x],
                            planes[channel].bit_depth(),
                        );
                    }
                    dst[3] = if synthesize_alpha {
                        u8::MAX
                    } else {
                        sample_as_u8(planes[3].samples()[row + x], planes[3].bit_depth())
                    };
                }
            }
        },
    );
}

#[expect(
    clippy::inline_always,
    reason = "inlined into the callers' SIMD dispatch (see `j2k_native::at_simd_level`)"
)]
#[inline(always)]
fn write_samples_as_u8(src: &[f32], bit_depth: u8, dst: &mut [u8]) {
    for (sample, dst) in src.iter().zip(dst.iter_mut()) {
        *dst = sample_as_u8(*sample, bit_depth);
    }
}

/// [`sample_as_u8`] for an 8-bit sample: rounded and clamped to a byte as
/// native packing does (see [`j2k_native::round_clamped_native`]).
#[expect(
    clippy::inline_always,
    reason = "inlined into the callers' SIMD dispatch"
)]
#[inline(always)]
pub(super) fn sample_as_u8_depth8(sample: f32) -> u8 {
    j2k_native::sample_as_u8_native(sample, 8)
}

/// One decoded sample as an 8-bit output sample, as native packing and the
/// direct CPU plans convert it (see [`j2k_native::sample_as_u8_native`]).
#[expect(
    clippy::inline_always,
    reason = "inlined into the callers' SIMD dispatch"
)]
#[inline(always)]
fn sample_as_u8(sample: f32, bit_depth: u8) -> u8 {
    j2k_native::sample_as_u8_native(sample, bit_depth)
}

pub(in crate::decode) fn write_u8_output(
    color_space: &ColorSpace,
    has_alpha: bool,
    dims: (u32, u32),
    decoded: &[u8],
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<(), J2kError> {
    let width = dims.0 as usize;
    let height = dims.1 as usize;
    match (resolve_u8_color_space(color_space, fmt), has_alpha, fmt) {
        (ColorSpace::RGB, false, PixelFormat::Rgb8) => {
            copy_rows_exact(decoded, out, stride, width * 3, height);
            Ok(())
        }
        (ColorSpace::RGB, true, PixelFormat::Rgb8) => {
            drop_alpha_u8(decoded, out, stride, width, height);
            Ok(())
        }
        (ColorSpace::RGB, false, PixelFormat::Rgba8) => {
            add_opaque_alpha_u8(decoded, out, stride, width, height);
            Ok(())
        }
        (ColorSpace::RGB, true, PixelFormat::Rgba8) => {
            copy_rows_exact(decoded, out, stride, width * 4, height);
            Ok(())
        }
        (ColorSpace::Gray, false, PixelFormat::Gray8) => {
            copy_rows_exact(decoded, out, stride, width, height);
            Ok(())
        }
        _ => Err(Unsupported {
            what: "backend color space cannot be mapped to requested 8-bit pixel format",
        }
        .into()),
    }
}

fn copy_rows_exact(src: &[u8], out: &mut [u8], stride: usize, row_bytes: usize, height: usize) {
    for (src_row, dst_row) in src
        .chunks_exact(row_bytes)
        .zip(output_rows(out, stride, row_bytes, height))
    {
        dst_row.copy_from_slice(src_row);
    }
}

fn add_opaque_alpha_u8(src: &[u8], out: &mut [u8], stride: usize, width: usize, height: usize) {
    let src_row_bytes = width * 3;
    let dst_row_bytes = width * 4;
    let dst_rows = output_rows(out, stride, dst_row_bytes, height);
    for (src_row, dst_row) in src.chunks_exact(src_row_bytes).zip(dst_rows) {
        for (rgb, rgba) in src_row
            .as_chunks::<3>()
            .0
            .iter()
            .zip(dst_row.as_chunks_mut::<4>().0)
        {
            rgba[..3].copy_from_slice(rgb);
            rgba[3] = u8::MAX;
        }
    }
}

/// Expand a packed RGB prefix into the same tight RGBA output allocation.
/// The last quarter's RGBA positions start beyond the remaining RGB bytes.
/// Convert that disjoint suffix, then repeat on the untouched prefix.
pub(in crate::decode) fn expand_rgb8_to_rgba8(out: &mut [u8]) {
    let mut pixels = out.len() / 4;
    while pixels >= 4 {
        let count = pixels / 4;
        let start = pixels - count;
        let (prefix, rgba) = out[..pixels * 4].split_at_mut(start * 4);
        let rgb = &prefix[start * 3..pixels * 3];
        add_opaque_alpha_u8(rgb, rgba, count * 4, count, 1);
        pixels = start;
    }
    for pixel in (0..pixels).rev() {
        let source = pixel * 3;
        let rgb = [out[source], out[source + 1], out[source + 2]];
        let target = pixel * 4;
        out[target..target + 3].copy_from_slice(&rgb);
        out[target + 3] = u8::MAX;
    }
}

fn drop_alpha_u8(src: &[u8], out: &mut [u8], stride: usize, width: usize, height: usize) {
    let src_row_bytes = width * 4;
    let dst_row_bytes = width * 3;
    let dst_rows = output_rows(out, stride, dst_row_bytes, height);
    for (src_row, dst_row) in src.chunks_exact(src_row_bytes).zip(dst_rows) {
        for (rgba, rgb) in src_row
            .as_chunks::<4>()
            .0
            .iter()
            .zip(dst_row.as_chunks_mut::<3>().0)
        {
            rgb.copy_from_slice(&rgba[..3]);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::ColorSpace;
    #[cfg(target_pointer_width = "32")]
    use crate::J2kError;
    use j2k_core::PixelFormat;

    use super::{can_decode_u8_directly, component_sample_count};

    #[test]
    fn direct_u8_decode_accepts_exact_rgb_and_gray_layouts() {
        assert!(can_decode_u8_directly(
            &ColorSpace::RGB,
            false,
            (128, 64),
            128 * 3,
            PixelFormat::Rgb8
        ));
        assert!(can_decode_u8_directly(
            &ColorSpace::Gray,
            false,
            (128, 64),
            128,
            PixelFormat::Gray8
        ));
    }

    #[test]
    fn direct_u8_decode_rejects_format_conversion_and_padded_stride() {
        assert!(!can_decode_u8_directly(
            &ColorSpace::RGB,
            false,
            (128, 64),
            128 * 4,
            PixelFormat::Rgba8
        ));
        assert!(!can_decode_u8_directly(
            &ColorSpace::RGB,
            true,
            (128, 64),
            128 * 3,
            PixelFormat::Rgb8
        ));
        assert!(!can_decode_u8_directly(
            &ColorSpace::Gray,
            false,
            (128, 64),
            160,
            PixelFormat::Gray8
        ));
    }

    #[test]
    fn component_sample_count_matches_image_dimensions() {
        assert_eq!(component_sample_count((16, 8)).expect("sample count"), 128);
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn component_sample_count_reports_dimension_overflow() {
        assert!(matches!(
            component_sample_count((u32::MAX, u32::MAX)),
            Err(J2kError::DimensionOverflow {
                width: u32::MAX,
                height: u32::MAX
            })
        ));
    }
}
