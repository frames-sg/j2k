// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use crate::error::JpegError;
use crate::info::{DownscaleFactor, OutputFormat, Rect, SofKind};
use j2k_core::{Downscale, PixelFormat};

use super::DEFAULT_MAX_DECODE_BYTES;

pub(super) fn output_format_profile_name(fmt: OutputFormat) -> &'static str {
    match fmt {
        OutputFormat::Rgb8 | OutputFormat::Rgb8Scaled { .. } => "Rgb8",
        OutputFormat::Rgba8 { .. } | OutputFormat::Rgba8Scaled { .. } => "Rgba8",
        OutputFormat::Gray8 | OutputFormat::Gray8Scaled { .. } => "Gray8",
        OutputFormat::Gray16 | OutputFormat::Gray16Scaled { .. } => "Gray16",
        OutputFormat::Rgb16 | OutputFormat::Rgb16Scaled { .. } => "Rgb16",
        OutputFormat::Rgba16 { .. } | OutputFormat::Rgba16Scaled { .. } => "Rgba16",
    }
}

pub(super) fn downscale_profile_name(downscale: DownscaleFactor) -> &'static str {
    match downscale {
        DownscaleFactor::Full => "full",
        DownscaleFactor::Half => "half",
        DownscaleFactor::Quarter => "quarter",
        DownscaleFactor::Eighth => "eighth",
    }
}

pub(super) fn jpeg_downscale(scale: Downscale) -> DownscaleFactor {
    match scale {
        Downscale::None => DownscaleFactor::Full,
        Downscale::Half => DownscaleFactor::Half,
        Downscale::Quarter => DownscaleFactor::Quarter,
        Downscale::Eighth => DownscaleFactor::Eighth,
        _ => unreachable!("unsupported Downscale variant"),
    }
}

pub(super) fn output_format_from_parts(
    sof_kind: SofKind,
    fmt: PixelFormat,
    scale: Downscale,
) -> Result<OutputFormat, JpegError> {
    if matches!(sof_kind, SofKind::Extended12 | SofKind::Progressive12) {
        return match (sof_kind, fmt, scale) {
            (SofKind::Extended12, PixelFormat::Gray16, Downscale::None) => Ok(OutputFormat::Gray16),
            (SofKind::Extended12, PixelFormat::Gray16, scale) => Ok(OutputFormat::Gray16Scaled {
                factor: jpeg_downscale(scale),
            }),
            (SofKind::Extended12, PixelFormat::Rgb16, Downscale::None) => Ok(OutputFormat::Rgb16),
            (SofKind::Extended12, PixelFormat::Rgb16, scale) => Ok(OutputFormat::Rgb16Scaled {
                factor: jpeg_downscale(scale),
            }),
            (SofKind::Extended12, PixelFormat::Rgba16, Downscale::None) => {
                Ok(OutputFormat::Rgba16 { alpha: u16::MAX })
            }
            (SofKind::Extended12, PixelFormat::Rgba16, scale) => Ok(OutputFormat::Rgba16Scaled {
                alpha: u16::MAX,
                factor: jpeg_downscale(scale),
            }),
            (SofKind::Progressive12, PixelFormat::Gray16, Downscale::None) => {
                Ok(OutputFormat::Gray16)
            }
            (SofKind::Progressive12, PixelFormat::Gray16, scale) => {
                Ok(OutputFormat::Gray16Scaled {
                    factor: jpeg_downscale(scale),
                })
            }
            (SofKind::Progressive12, PixelFormat::Rgb16, Downscale::None) => {
                Ok(OutputFormat::Rgb16)
            }
            (SofKind::Progressive12, PixelFormat::Rgb16, scale) => Ok(OutputFormat::Rgb16Scaled {
                factor: jpeg_downscale(scale),
            }),
            (SofKind::Progressive12, PixelFormat::Rgba16, Downscale::None) => {
                Ok(OutputFormat::Rgba16 { alpha: u16::MAX })
            }
            (SofKind::Progressive12, PixelFormat::Rgba16, scale) => {
                Ok(OutputFormat::Rgba16Scaled {
                    alpha: u16::MAX,
                    factor: jpeg_downscale(scale),
                })
            }
            (_, PixelFormat::Rgb16 | PixelFormat::Rgba16 | PixelFormat::Gray16, _) => {
                Err(JpegError::NotImplemented { sof: sof_kind })
            }
            _ => Err(JpegError::UnsupportedBitDepth { depth: 12 }),
        };
    }
    if sof_kind == SofKind::Lossless {
        return match (fmt, scale) {
            (PixelFormat::Gray8, Downscale::None) => Ok(OutputFormat::Gray8),
            (PixelFormat::Gray8, scale) => Ok(OutputFormat::Gray8Scaled {
                factor: jpeg_downscale(scale),
            }),
            (PixelFormat::Gray16, Downscale::None) => Ok(OutputFormat::Gray16),
            (PixelFormat::Gray16, scale) => Ok(OutputFormat::Gray16Scaled {
                factor: jpeg_downscale(scale),
            }),
            (PixelFormat::Rgb8, Downscale::None) => Ok(OutputFormat::Rgb8),
            (PixelFormat::Rgb8, scale) => Ok(OutputFormat::Rgb8Scaled {
                factor: jpeg_downscale(scale),
            }),
            (PixelFormat::Rgba8, Downscale::None) => Ok(OutputFormat::Rgba8 { alpha: 255 }),
            (PixelFormat::Rgba8, scale) => Ok(OutputFormat::Rgba8Scaled {
                alpha: 255,
                factor: jpeg_downscale(scale),
            }),
            (PixelFormat::Rgb16, Downscale::None) => Ok(OutputFormat::Rgb16),
            (PixelFormat::Rgb16, scale) => Ok(OutputFormat::Rgb16Scaled {
                factor: jpeg_downscale(scale),
            }),
            (PixelFormat::Rgba16, Downscale::None) => Ok(OutputFormat::Rgba16 { alpha: u16::MAX }),
            (PixelFormat::Rgba16, scale) => Ok(OutputFormat::Rgba16Scaled {
                alpha: u16::MAX,
                factor: jpeg_downscale(scale),
            }),
            _ => Err(JpegError::NotImplemented { sof: sof_kind }),
        };
    }

    match (fmt, scale) {
        (PixelFormat::Rgb8, Downscale::None) => Ok(OutputFormat::Rgb8),
        (PixelFormat::Rgb8, scale) => Ok(OutputFormat::Rgb8Scaled {
            factor: jpeg_downscale(scale),
        }),
        (PixelFormat::Gray8, Downscale::None) => Ok(OutputFormat::Gray8),
        (PixelFormat::Gray8, scale) => Ok(OutputFormat::Gray8Scaled {
            factor: jpeg_downscale(scale),
        }),
        (PixelFormat::Rgba8, Downscale::None) => Ok(OutputFormat::Rgba8 { alpha: 255 }),
        (PixelFormat::Rgba8, scale) => Ok(OutputFormat::Rgba8Scaled {
            alpha: 255,
            factor: jpeg_downscale(scale),
        }),
        (PixelFormat::Rgb16 | PixelFormat::Rgba16 | PixelFormat::Gray16, _) => {
            Err(JpegError::UnsupportedBitDepth { depth: 16 })
        }
        _ => Err(JpegError::DownscaleUnsupported { sof: sof_kind }),
    }
}

#[allow(clippy::uninit_vec)]
pub(super) fn allocate_output_buffer(len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    // Safety: all owned-output entrypoints use tight row strides, and the
    // decode writers fully initialize every byte in the destination on success.
    // If decode returns an error, dropping a Vec<u8> with uninitialized bytes is
    // still sound because `u8` has no drop glue.
    unsafe {
        out.set_len(len);
    }
    out
}

pub(super) fn scaled_dimensions(dims: (u32, u32), factor: DownscaleFactor) -> (u32, u32) {
    let denom = factor.denominator();
    (dims.0.div_ceil(denom), dims.1.div_ceil(denom))
}

pub(super) fn scaled_rect_covering(rect: Rect, factor: DownscaleFactor) -> Result<Rect, JpegError> {
    let denom = factor.denominator();
    let x_end = rect
        .x
        .checked_add(rect.w)
        .ok_or(JpegError::RectOutOfBounds {
            rect,
            width: u32::MAX,
            height: u32::MAX,
        })?;
    let y_end = rect
        .y
        .checked_add(rect.h)
        .ok_or(JpegError::RectOutOfBounds {
            rect,
            width: u32::MAX,
            height: u32::MAX,
        })?;
    let x0 = rect.x / denom;
    let y0 = rect.y / denom;
    let x1 = x_end.div_ceil(denom);
    let y1 = y_end.div_ceil(denom);
    Ok(Rect {
        x: x0,
        y: y0,
        w: x1.saturating_sub(x0),
        h: y1.saturating_sub(y0),
    })
}

fn output_cap_error(requested: usize) -> JpegError {
    JpegError::MemoryCapExceeded {
        requested,
        cap: DEFAULT_MAX_DECODE_BYTES,
    }
}

#[inline]
pub(super) fn checked_output_geometry(
    width: u32,
    height: u32,
    bytes_per_pixel: usize,
) -> Result<(usize, usize), JpegError> {
    #[cfg(target_pointer_width = "64")]
    {
        // SOF parsing caps JPEG dimensions at 65_500, so these products cannot
        // overflow usize on 64-bit targets. Keep the hot path to one cap check.
        let stride = width as usize * bytes_per_pixel;
        let len = stride * height as usize;
        if len > DEFAULT_MAX_DECODE_BYTES {
            return Err(output_cap_error(len));
        }
        Ok((stride, len))
    }

    #[cfg(not(target_pointer_width = "64"))]
    {
        let stride = checked_output_product(width as usize, bytes_per_pixel)?;
        let len = checked_output_product(stride, height as usize)?;
        Ok((stride, len))
    }
}

#[cfg(not(target_pointer_width = "64"))]
#[inline]
fn checked_output_product(left: usize, right: usize) -> Result<usize, JpegError> {
    let len = left
        .checked_mul(right)
        .ok_or_else(|| output_cap_error(usize::MAX))?;
    if len > DEFAULT_MAX_DECODE_BYTES {
        return Err(output_cap_error(len));
    }
    Ok(len)
}
