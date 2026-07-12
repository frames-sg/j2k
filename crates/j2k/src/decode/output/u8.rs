// SPDX-License-Identifier: MIT OR Apache-2.0

//! Eight-bit direct-layout and component-plane conversion.

use crate::backend::{ColorSpace, DecodedComponents as NativeDecodedComponents};
use crate::J2kError;
use j2k_core::{PixelFormat, Unsupported};

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
    let dims = components.dimensions();
    let expected_samples = component_sample_count(dims)?;
    let width = dims.0 as usize;
    let height = dims.1 as usize;
    let planes = components.planes();
    match (
        components.color_space(),
        components.has_alpha(),
        planes.len(),
        fmt,
    ) {
        (ColorSpace::Gray, false, 1, PixelFormat::Gray8) => {
            validate_component_planes(&planes[..1], expected_samples)?;
            write_component_rows_u8(&planes[0], out, stride, width, height);
            Ok(())
        }
        (ColorSpace::RGB, false, 3, PixelFormat::Rgb8)
        | (ColorSpace::RGB, true, 4, PixelFormat::Rgb8) => {
            validate_component_planes(&planes[..3], expected_samples)?;
            write_rgb_component_rows_u8(planes, out, stride, width, height);
            Ok(())
        }
        (ColorSpace::RGB, false, 3, PixelFormat::Rgba8) => {
            validate_component_planes(&planes[..3], expected_samples)?;
            write_rgba_component_rows_u8(planes, out, stride, width, height, true);
            Ok(())
        }
        (ColorSpace::RGB, true, 4, PixelFormat::Rgba8) => {
            validate_component_planes(&planes[..4], expected_samples)?;
            write_rgba_component_rows_u8(planes, out, stride, width, height, false);
            Ok(())
        }
        _ => Err(Unsupported {
            what: "backend color space cannot be mapped to requested 8-bit pixel format",
        }
        .into()),
    }
}

fn component_sample_count(dims: (u32, u32)) -> Result<usize, J2kError> {
    (dims.0 as usize)
        .checked_mul(dims.1 as usize)
        .ok_or(J2kError::DimensionOverflow {
            width: dims.0,
            height: dims.1,
        })
}

fn validate_component_planes(
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

fn write_component_rows_u8(
    plane: &j2k_native::ComponentPlane<'_>,
    out: &mut [u8],
    stride: usize,
    width: usize,
    height: usize,
) {
    for y in 0..height {
        let src = &plane.samples()[y * width..(y + 1) * width];
        let dst = &mut out[y * stride..y * stride + width];
        write_samples_as_u8(src, plane.bit_depth(), dst);
    }
}

fn write_rgb_component_rows_u8(
    planes: &[j2k_native::ComponentPlane<'_>],
    out: &mut [u8],
    stride: usize,
    width: usize,
    height: usize,
) {
    for y in 0..height {
        let row = y * width;
        let dst = &mut out[y * stride..y * stride + width * 3];
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
}

fn write_rgba_component_rows_u8(
    planes: &[j2k_native::ComponentPlane<'_>],
    out: &mut [u8],
    stride: usize,
    width: usize,
    height: usize,
    synthesize_alpha: bool,
) {
    for y in 0..height {
        let row = y * width;
        let dst = &mut out[y * stride..y * stride + width * 4];
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
}

fn write_samples_as_u8(src: &[f32], bit_depth: u8, dst: &mut [u8]) {
    for (sample, dst) in src.iter().zip(dst.iter_mut()) {
        *dst = sample_as_u8(*sample, bit_depth);
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "samples are rounded and clamped to the unsigned 8-bit output domain before conversion"
)]
fn sample_as_u8(sample: f32, bit_depth: u8) -> u8 {
    let rounded = sample.round();
    if bit_depth == 8 {
        return rounded.clamp(0.0, f32::from(u8::MAX)) as u8;
    }
    let max_value = if bit_depth >= 16 {
        f32::from(u16::MAX)
    } else {
        f32::from(((1_u16 << bit_depth) - 1).max(1))
    };
    ((rounded.clamp(0.0, max_value) / max_value) * f32::from(u8::MAX)).round() as u8
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
    match (color_space, has_alpha, fmt) {
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
        .zip(out.chunks_exact_mut(stride))
        .take(height)
    {
        dst_row[..row_bytes].copy_from_slice(src_row);
    }
}

fn add_opaque_alpha_u8(src: &[u8], out: &mut [u8], stride: usize, width: usize, height: usize) {
    let src_row_bytes = width * 3;
    let dst_row_bytes = width * 4;
    for (src_row, dst_row) in src
        .chunks_exact(src_row_bytes)
        .zip(out.chunks_exact_mut(stride))
        .take(height)
    {
        for (rgb, rgba) in src_row
            .chunks_exact(3)
            .zip(dst_row[..dst_row_bytes].chunks_exact_mut(4))
        {
            rgba[..3].copy_from_slice(rgb);
            rgba[3] = u8::MAX;
        }
    }
}

fn drop_alpha_u8(src: &[u8], out: &mut [u8], stride: usize, width: usize, height: usize) {
    let src_row_bytes = width * 4;
    let dst_row_bytes = width * 3;
    for (src_row, dst_row) in src
        .chunks_exact(src_row_bytes)
        .zip(out.chunks_exact_mut(stride))
        .take(height)
    {
        for (rgba, rgb) in src_row
            .chunks_exact(4)
            .zip(dst_row[..dst_row_bytes].chunks_exact_mut(3))
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
