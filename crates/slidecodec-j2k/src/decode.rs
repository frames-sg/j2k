// SPDX-License-Identifier: Apache-2.0

use crate::J2kError;
use alloc::{string::ToString, vec::Vec};
use dicom_toolkit_jpeg2000::{ColorSpace, DecodeSettings, Image, RawBitmap};
use slidecodec_core::{
    BufferError, Colorspace, DecodeOutcome, Info, NotImplemented, PixelFormat, Rect, Unsupported,
};
use core::convert::Infallible;

pub(crate) type J2kDecodeOutcome = DecodeOutcome<Infallible>;

pub(crate) fn decode_full_frame(
    bytes: &[u8],
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<J2kDecodeOutcome, J2kError> {
    validate_supported_format(fmt)?;
    let image = backend_image(bytes, DecodeSettings::default())?;
    let dims = (image.width(), image.height());
    validate_buffer(dims, out.len(), stride, fmt)?;

    match fmt {
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Gray8 => {
            let decoded = image.decode().map_err(|err| J2kError::Backend(err.to_string()))?;
            write_u8_output(
                image.color_space(),
                image.has_alpha(),
                dims,
                &decoded,
                out,
                stride,
                fmt,
            )?;
        }
        PixelFormat::Rgb16 | PixelFormat::Gray16 => {
            let raw = image
                .decode_native()
                .map_err(|err| J2kError::Backend(err.to_string()))?;
            write_u16_output(image.color_space(), image.has_alpha(), &raw, out, stride, fmt)?;
        }
        PixelFormat::Rgba16 => unreachable!("validated above"),
        _ => {
            return Err(Unsupported {
                what: "pixel format is not yet supported by slidecodec-j2k",
            }
            .into());
        }
    }

    Ok(DecodeOutcome {
        decoded: Rect::full(dims),
        warnings: Vec::new(),
    })
}

pub(crate) fn inspect_info_via_backend(bytes: &[u8]) -> Result<Info, J2kError> {
    let image = backend_image(bytes, DecodeSettings::default())?;
    let components = image.color_space().num_channels() + u8::from(image.has_alpha());
    Ok(Info {
        dimensions: (image.width(), image.height()),
        components,
        colorspace: map_backend_colorspace(image.color_space()),
        bit_depth: image.original_bit_depth(),
        tile_layout: None,
        resolution_levels: 1,
    })
}

pub(crate) fn decode_region_not_implemented() -> Result<J2kDecodeOutcome, J2kError> {
    Err(NotImplemented {
        what: "JPEG 2000 region decode lands in J2K-M2",
    }
    .into())
}

pub(crate) fn decode_scaled_not_implemented() -> Result<J2kDecodeOutcome, J2kError> {
    Err(NotImplemented {
        what: "JPEG 2000 scaled decode lands in J2K-M2",
    }
    .into())
}

fn backend_image(bytes: &[u8], settings: DecodeSettings) -> Result<Image<'_>, J2kError> {
    Image::new(bytes, &settings).map_err(|err| J2kError::Backend(err.to_string()))
}

fn map_backend_colorspace(color_space: &ColorSpace) -> Colorspace {
    match color_space {
        ColorSpace::Gray => Colorspace::SGray,
        ColorSpace::RGB => Colorspace::Rgb,
        ColorSpace::CMYK => Colorspace::Cmyk,
        ColorSpace::Unknown { .. } | ColorSpace::Icc { .. } => Colorspace::IccTagged,
    }
}

fn validate_supported_format(fmt: PixelFormat) -> Result<(), J2kError> {
    if matches!(fmt, PixelFormat::Rgba16) {
        return Err(Unsupported {
            what: "Rgba16 output is not supported by slidecodec-j2k M1",
        }
        .into());
    }
    Ok(())
}

fn validate_buffer(
    dims: (u32, u32),
    out_len: usize,
    stride: usize,
    fmt: PixelFormat,
) -> Result<(), J2kError> {
    let row_bytes = dims
        .0
        .checked_mul(fmt.bytes_per_pixel() as u32)
        .ok_or(J2kError::Backend("row byte count overflow".to_string()))?
        as usize;
    if stride < row_bytes {
        return Err(BufferError::StrideTooSmall { row_bytes, stride }.into());
    }
    let height = dims.1 as usize;
    let required = if height == 0 {
        0
    } else {
        stride
            .checked_mul(height - 1)
            .and_then(|prefix| prefix.checked_add(row_bytes))
            .ok_or(J2kError::Backend("output size overflow".to_string()))?
    };
    if out_len < required {
        return Err(BufferError::OutputTooSmall {
            required,
            have: out_len,
        }
        .into());
    }
    Ok(())
}

fn write_u8_output(
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

fn write_u16_output(
    color_space: &ColorSpace,
    has_alpha: bool,
    raw: &RawBitmap,
    out: &mut [u8],
    stride: usize,
    fmt: PixelFormat,
) -> Result<(), J2kError> {
    let width = raw.width as usize;
    let height = raw.height as usize;
    match (color_space, has_alpha, raw.num_components, fmt) {
        (ColorSpace::RGB, false, 3, PixelFormat::Rgb16) => {
            convert_or_copy_u16(
                &raw.data,
                raw.bytes_per_sample,
                raw.bit_depth,
                3,
                out,
                stride,
                (width, height),
            );
            Ok(())
        }
        (ColorSpace::Gray, false, 1, PixelFormat::Gray16) => {
            convert_or_copy_u16(
                &raw.data,
                raw.bytes_per_sample,
                raw.bit_depth,
                1,
                out,
                stride,
                (width, height),
            );
            Ok(())
        }
        _ => Err(Unsupported {
            what: "backend color space cannot be mapped to requested 16-bit pixel format",
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
        for (rgb, rgba) in src_row.chunks_exact(3).zip(dst_row[..dst_row_bytes].chunks_exact_mut(4)) {
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
        for (rgba, rgb) in src_row.chunks_exact(4).zip(dst_row[..dst_row_bytes].chunks_exact_mut(3)) {
            rgb.copy_from_slice(&rgba[..3]);
        }
    }
}

fn convert_or_copy_u16(
    src: &[u8],
    bytes_per_sample: u8,
    bit_depth: u8,
    channels: usize,
    out: &mut [u8],
    stride: usize,
    dims: (usize, usize),
) {
    let (width, height) = dims;
    let dst_row_bytes = width * channels * 2;
    let src_row_bytes = width * channels * usize::from(bytes_per_sample);
    let max_value = ((1_u32 << bit_depth.min(16)) - 1).max(1);
    for (src_row, dst_row) in src
        .chunks_exact(src_row_bytes)
        .zip(out.chunks_exact_mut(stride))
        .take(height)
    {
        let dst_row = &mut dst_row[..dst_row_bytes];
        if bytes_per_sample == 2 {
            dst_row.copy_from_slice(src_row);
            continue;
        }
        for (sample, dst_sample) in src_row.iter().zip(dst_row.chunks_exact_mut(2)) {
            let widened = (u32::from(*sample) * u32::from(u16::MAX) + (max_value / 2)) / max_value;
            dst_sample.copy_from_slice(&(widened as u16).to_le_bytes());
        }
    }
}
